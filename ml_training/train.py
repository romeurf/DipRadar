"""Walk-forward CV + champion training + calibrator + stacking ensemble.

Inclui também ``run_training()`` — orchestrator end-to-end usado por
``monthly_retrain.py`` (drop-in para o legacy ``train_v31.run_training``):

  1. Carregar parquet base (já com features e targets pré-computados)
  2. Adicionar alpha_60d_rank (cross-section por data) — target robusto
  3. (opcional) Neutralizar target por sector
  4. Walk-forward CV
  5. Seleccionar champion (IC máximo com PnL>0)
  6. Treinar champion full + isotonic calibrator OOF
  7. Empacotar DipModelsV3 + escrever ml_report.json
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from datetime import datetime
from typing import Optional, Any

import numpy as np
import pandas as pd

from ml_training.config import HORIZON_DAYS, N_FOLDS, PURGE_DAYS
from ml_training.cv import (
    build_walk_forward_folds,
    fold_metric_record,
    spearman_safe,
    temporal_weights,
    topk_pnl,
    winsorize,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _fit_model(model, X_tr, y_tr, sw_tr, X_val=None, y_val=None):
    """Ajusta modelo com suporte a early stopping (XGB/LGBM)."""
    has_es = hasattr(model, "early_stopping_rounds") and model.early_stopping_rounds

    if has_es and X_val is not None and y_val is not None:
        # XGBoost
        if hasattr(model, "get_booster"):
            try:
                model.fit(
                    X_tr, y_tr,
                    sample_weight=sw_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                return model
            except Exception:
                pass
        # LightGBM
        try:
            from lightgbm import early_stopping, log_evaluation
            model.fit(
                X_tr, y_tr,
                sample_weight=sw_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)],
            )
            return model
        except Exception:
            pass

    # Fallback: treino normal
    try:
        model.fit(X_tr, y_tr, sample_weight=sw_tr)
    except TypeError:
        model.fit(X_tr, y_tr)
    return model


def _split_val(X_tr: np.ndarray, y_tr: np.ndarray, sw_tr: np.ndarray, val_frac: float = 0.07):
    """Separa os últimos val_frac% do treino como validation set para early stopping.

    Usa os últimos rows (temporalmente mais recentes) para não criar leakage.
    """
    n_val = max(1, int(len(X_tr) * val_frac))
    X_t, X_v = X_tr[:-n_val], X_tr[-n_val:]
    y_t, y_v = y_tr[:-n_val], y_tr[-n_val:]
    sw_t = sw_tr[:-n_val]
    return X_t, y_t, sw_t, X_v, y_v


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward CV
# ─────────────────────────────────────────────────────────────────────────────

def run_walk_forward_cv(
    df_v31: pd.DataFrame,
    model_configs: dict[str, dict],
    n_folds: int,
    purge_days: int,
    min_train: int = 100,
    min_test: int = 20,
) -> tuple[dict[str, list[dict]], dict[str, np.ndarray], list[tuple]]:
    """Walk-forward CV em todos os modelos candidatos.

    Devolve:
      - results: ``{model_name: [fold_metric_records]}``
      - oof_pred: ``{model_name: np.ndarray (full size, NaN onde não testado)}``
      - fold_specs: lista (k, train_end, purge_end, test_end)
    """
    df_v31 = df_v31.sort_values("alert_date").reset_index(drop=True)
    df_v31["alert_date"] = pd.to_datetime(df_v31["alert_date"])
    max_date = df_v31["alert_date"].max()

    fold_specs = build_walk_forward_folds(df_v31, n_folds=n_folds, purge_days=purge_days)
    results: dict[str, list[dict]] = {name: [] for name in model_configs}
    oof_pred: dict[str, np.ndarray] = {
        name: np.full(len(df_v31), np.nan) for name in model_configs
    }

    for k, train_end, purge_end, test_end in fold_specs:
        tr_mask = df_v31["alert_date"] <= train_end
        te_mask = (df_v31["alert_date"] > purge_end) & (df_v31["alert_date"] <= test_end)
        df_tr = df_v31[tr_mask]
        df_te = df_v31[te_mask]
        if len(df_tr) < min_train or len(df_te) < min_test:
            log.info(f"Fold {k}: insuficiente (tr={len(df_tr)}, te={len(df_te)}) — saltar")
            continue

        y_alpha_tr = winsorize(df_tr["alpha_60d"].values)
        y_alpha_te = df_te["alpha_60d"].values
        y_down_tr  = winsorize(df_tr["max_drawdown_60d"].values)
        y_down_te  = df_te["max_drawdown_60d"].values

        sw_tr = temporal_weights(df_tr["alert_date"], max_date)

        for name, cfg in model_configs.items():
            feats = [f for f in cfg["feats"] if f in df_tr.columns]
            X_tr = df_tr[feats].fillna(0).values.astype(np.float32)
            X_te = df_te[feats].fillna(0).values.astype(np.float32)

            X_t, y_t, sw_t, X_v, y_v = _split_val(X_tr, y_alpha_tr, sw_tr)
            m_alpha = cfg["factory"]()
            m_alpha = _fit_model(m_alpha, X_t, y_t, sw_t, X_v, y_v)

            X_t_d, y_t_d, sw_t_d, X_v_d, y_v_d = _split_val(X_tr, y_down_tr, sw_tr)
            m_down = cfg["factory"]()
            m_down = _fit_model(m_down, X_t_d, y_t_d, sw_t_d, X_v_d, y_v_d)

            pred_alpha = m_alpha.predict(X_te)
            pred_down  = m_down.predict(X_te)

            oof_pred[name][df_te.index.values] = pred_alpha

            rho_alpha = spearman_safe(pred_alpha, y_alpha_te)
            rho_down  = spearman_safe(pred_down, y_down_te)
            pnl       = topk_pnl(pred_alpha, y_alpha_te)

            results[name].append(fold_metric_record(
                fold=k,
                n_test=len(df_te),
                rho_alpha=rho_alpha,
                rho_down=rho_down,
                pnl=pnl,
            ))
        log.info(f"Fold {k:2d} OK — n_train={len(df_tr)} n_test={len(df_te)}")

    log.info("Walk-forward CV concluído.")
    return results, oof_pred, fold_specs


# ─────────────────────────────────────────────────────────────────────────────
# Stacking Ensemble (Nível 2)
# ─────────────────────────────────────────────────────────────────────────────

def fit_stacking_ensemble(
    oof_pred: dict[str, np.ndarray],
    alpha_true: np.ndarray,
    base_model_names: list[str] | None = None,
) -> tuple["object", list[str], float]:
    """Treina um meta-learner Ridge sobre as OOF predictions dos modelos base.

    O stacking usa APENAS OOF predictions — dados que cada modelo base
    nunca viu durante o seu treino. Não há data leakage.

    Arquitectura:
      Nível 1: {XGB-ES-alpha, LGBM-ES-alpha, XGB-alpha, LGBM-alpha, RF-alpha, Ridge-alpha}
               → OOF predictions de cada fold
      Nível 2: Ridge(alpha=1.0) que aprende os pesos óptimos entre modelos

    Devolve
    -------
    (meta_model, names_used, ic_meta)
    """
    from ml_training.models import stack_meta_factory

    if base_model_names is None:
        priority = ["XGB-ES-alpha", "LGBM-ES-alpha", "XGB-alpha", "LGBM-alpha",
                    "RF-alpha", "Ridge-alpha"]
        base_model_names = [n for n in priority if n in oof_pred]

    if len(base_model_names) < 2:
        log.warning("Stacking requer >=2 modelos base. Stacking ignorado.")
        return None, [], float("nan")

    stacks = {n: oof_pred[n] for n in base_model_names}
    valid_mask = np.ones(len(alpha_true), dtype=bool)
    for arr in stacks.values():
        valid_mask &= np.isfinite(arr)
    valid_mask &= np.isfinite(alpha_true)

    n_valid = valid_mask.sum()
    if n_valid < 50:
        log.warning(f"Stacking: so {n_valid} amostras validas — stack ignorado")
        return None, [], float("nan")

    X_meta = np.column_stack([stacks[n][valid_mask] for n in base_model_names])
    y_meta = alpha_true[valid_mask]

    meta = stack_meta_factory()
    meta.fit(X_meta, y_meta)

    y_hat = meta.predict(X_meta)
    ic_meta = float(spearman_safe(y_hat, y_meta))

    log.info(
        f"Stacking: {len(base_model_names)} base models, "
        f"{n_valid} OOF samples, IC_meta={ic_meta:.4f}"
    )
    coef_str = ", ".join(
        f"{n}={c:.3f}" for n, c in zip(base_model_names, meta.coef_)
    )
    log.info(f"Stacking coefs: {coef_str}")

    return meta, base_model_names, ic_meta


# ─────────────────────────────────────────────────────────────────────────────
# Sumarização + Champion
# ─────────────────────────────────────────────────────────────────────────────

def summarize_results(results: dict[str, list[dict]]) -> pd.DataFrame:
    """Tabela resumo (médias e std) por modelo."""
    cols = ["model", "rho_alpha_mean", "rho_alpha_std",
            "rho_down_mean", "topk_pnl_mean", "topk_pnl_std", "n_folds"]
    rows = []
    for name, hist in results.items():
        if not hist:
            continue
        rho_alphas = np.array([h["rho_alpha"] for h in hist if math.isfinite(h["rho_alpha"])])
        rho_downs  = np.array([h["rho_down"]  for h in hist if math.isfinite(h["rho_down"])])
        pnls       = np.array([h["topk_pnl"]  for h in hist if math.isfinite(h["topk_pnl"])])
        rows.append({
            "model":          name,
            "rho_alpha_mean": rho_alphas.mean() if len(rho_alphas) else np.nan,
            "rho_alpha_std":  rho_alphas.std()  if len(rho_alphas) else np.nan,
            "rho_down_mean":  rho_downs.mean()  if len(rho_downs)  else np.nan,
            "topk_pnl_mean":  pnls.mean()       if len(pnls)       else np.nan,
            "topk_pnl_std":   pnls.std()        if len(pnls)       else np.nan,
            "n_folds":        len(hist),
        })
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    return df.sort_values("rho_alpha_mean", ascending=False)


def select_champion(summary: pd.DataFrame) -> tuple[str, pd.Series]:
    """Champion = melhor score composto (rho_alpha_mean - 0.5 * rho_alpha_std)
    entre modelos com topk_pnl_mean > 0.
    """
    if summary.empty:
        raise ValueError("summary vazio — sem modelos para escolher champion")

    df = summary.copy()
    df["champion_score"] = df["rho_alpha_mean"] - 0.5 * df["rho_alpha_std"]

    qualifiers = df[df["topk_pnl_mean"] > 0].sort_values(
        "champion_score", ascending=False
    )
    if len(qualifiers) == 0:
        log.warning("Nenhum modelo passou no criterio topk_pnl > 0 — fallback ao melhor score")
        qualifiers = df.sort_values("champion_score", ascending=False)

    champion_name = str(qualifiers.iloc[0]["model"])
    log.info(
        f"Champion selecionado: {champion_name} "
        f"(score={qualifiers.iloc[0]['champion_score']:.4f}, "
        f"rho_alpha_mean={qualifiers.iloc[0]['rho_alpha_mean']:.4f})"
    )
    return champion_name, qualifiers.iloc[0]


# ─────────────────────────────────────────────────────────────────────────────
# Calibrator
# ─────────────────────────────────────────────────────────────────────────────

class PlattCalibrator:
    """Wrapper picklable para Platt Scaling (logistic regression sobre score).

    Definido em módulo (não closure) para joblib/pickle funcionarem.
    """
    def __init__(self, scaler, lr):
        self.scaler = scaler
        self.lr = lr

    def predict(self, x):
        X = self.scaler.transform(np.asarray(x).reshape(-1, 1))
        return self.lr.predict_proba(X)[:, 1]


def fit_isotonic_calibrator(
    oof_pred_champion: np.ndarray,
    alpha_true: np.ndarray,
    alpha_threshold: float = 0.05,
) -> tuple[object, float, int]:
    """Platt Scaling (LR) se n_oof < 500, Isotónico caso contrário.

    Isotónico com amostras escassas tende a overfit — Platt é mais estável.
    Devolve (calibrator_model, brier_score, n_oof_samples).
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss
    from sklearn.preprocessing import StandardScaler

    mask   = np.isfinite(oof_pred_champion)
    y_oof  = oof_pred_champion[mask]
    y_bin  = (alpha_true[mask] > alpha_threshold).astype(int)
    n_oof  = int(mask.sum())

    if n_oof < 100:
        log.warning(f"OOF samples insuficientes: {n_oof}; calibrator pode ser instavel")

    if n_oof < 500:
        log.info(f"Calibrator: Platt Scaling (n_oof={n_oof} < 500)")
        scaler = StandardScaler()
        X_cal  = scaler.fit_transform(y_oof.reshape(-1, 1))
        lr     = LogisticRegression(C=1.0, max_iter=500)
        lr.fit(X_cal, y_bin)
        cal = PlattCalibrator(scaler, lr)
    else:
        log.info(f"Calibrator: Isotónico (n_oof={n_oof})")
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        cal.fit(y_oof, y_bin)

    prob_oof = cal.predict(y_oof)
    brier    = float(brier_score_loss(y_bin, prob_oof))
    return cal, brier, n_oof


# ─────────────────────────────────────────────────────────────────────────────
# Treino full do champion
# ─────────────────────────────────────────────────────────────────────────────

def train_full_champion(
    df_v31: pd.DataFrame,
    champion_cfg: dict,
) -> tuple[object, object, list[str], int]:
    """Treina (model_up, model_down) no dataset COMPLETO com early stopping.

    Devolve (champ_alpha, champ_down, feats_used, n_train).
    """
    df_v31   = df_v31.sort_values("alert_date").reset_index(drop=True)
    max_date = pd.to_datetime(df_v31["alert_date"]).max()
    feats    = [f for f in champion_cfg["feats"] if f in df_v31.columns]

    X_full        = df_v31[feats].fillna(0).values.astype(np.float32)
    y_alpha_full  = winsorize(df_v31["alpha_60d"].values)
    y_down_full   = winsorize(df_v31["max_drawdown_60d"].values)
    sw_full       = temporal_weights(df_v31["alert_date"], max_date)

    X_t, y_t_a, sw_t, X_v, y_v_a = _split_val(X_full, y_alpha_full, sw_full, val_frac=0.07)
    _, y_t_d, _, _, y_v_d         = _split_val(X_full, y_down_full,  sw_full, val_frac=0.07)

    champ_alpha = champion_cfg["factory"]()
    champ_alpha = _fit_model(champ_alpha, X_t, y_t_a, sw_t, X_v, y_v_a)

    champ_down  = champion_cfg["factory"]()
    champ_down  = _fit_model(champ_down,  X_t, y_t_d, sw_t, X_v, y_v_d)

    log.info(
        f"Champion treinado em {len(X_full)} amostras com {len(feats)} features"
    )
    return champ_alpha, champ_down, feats, len(X_full)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator end-to-end (drop-in para train_v31.run_training)
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_apply_rank_target(df: pd.DataFrame, *, enable: bool = True) -> pd.DataFrame:
    """Adiciona ``alpha_60d_rank`` cross-section por data."""
    if not enable:
        return df
    from ml_training.cv_robust import add_crosssection_rank
    return add_crosssection_rank(df, raw_col="alpha_60d", date_col="alert_date")


def _maybe_apply_sector_neutral(df: pd.DataFrame, *, enable: bool = True) -> pd.DataFrame:
    """Adiciona ``alpha_60d_neutral`` e ``alpha_60d_neutral_rank``.

    Subtrai mediana de sector por data antes de re-rankear. Útil para o
    modelo aprender "compra dip dentro do sector" em vez de "compra tech".
    """
    if not enable or "sector" not in df.columns:
        return df
    from ml_training.target_engineering import neutralize_target_by_sector
    return neutralize_target_by_sector(
        df,
        target_col="alpha_60d",
        sector_col="sector",
        date_col="alert_date",
    )


def run_training(
    input_parquet: "str | Path",
    output_bundle: "Optional[str | Path]" = None,
    output_report: "Optional[str | Path]" = None,
    *,
    n_folds: int = N_FOLDS,
    purge_days: int = PURGE_DAYS,
    horizon_days: int = HORIZON_DAYS,
    min_train: int = 100,
    min_test: int = 20,
    use_rank_target: bool = True,
    use_sector_neutral: bool = True,
    max_rows: Optional[int] = None,
    log_summary: bool = True,
) -> dict:
    """Pipeline completo de treino (drop-in para o legacy ``train_v31.run_training``).

    Parameters
    ----------
    input_parquet : Path
        Parquet com features + targets já computados. Esquema esperado:
        - Colunas FEATURE_COLUMNS (ml_features.py)
        - Targets: ``alpha_60d``, ``max_drawdown_60d``
        - Metadados: ``alert_date``, ``ticker``, ``sector``
    output_bundle : Path | None
        Onde escrever o ``dip_models.pkl``. ``None`` = não escreve.
    output_report : Path | None
        Onde escrever o ``ml_report.json``. ``None`` = não escreve.
    n_folds, purge_days, horizon_days : int
        Parâmetros do walk-forward CV.
    use_rank_target : bool
        Se True, adiciona ``alpha_60d_rank`` para diagnóstico (sempre treina
        em ``alpha_60d`` bruto para compatibilidade com o pipeline produção).
    use_sector_neutral : bool
        Se True, adiciona ``alpha_60d_neutral`` ao DataFrame (apenas
        diagnóstico — modelo continua a treinar em ``alpha_60d`` directo).
    max_rows : int | None
        Slicing para smoke tests.
    log_summary : bool
        Se True, loga o summary table com IC/PnL por modelo.

    Returns
    -------
    dict com chaves:
      - ``bundle``         : DipModelsV3
      - ``report``         : dict (conteúdo do ml_report.json)
      - ``summary``        : pd.DataFrame com IC/PnL por modelo
      - ``oof_pred``       : dict[name, np.ndarray]
      - ``champion_name``  : str
      - ``brier_oof``      : float
      - ``n_oof``          : int
    """
    from ml_training.bundle import DipModelsV3, build_report, save_bundle, save_report
    from ml_training.data import load_base_dataset
    from ml_training.models import build_feature_lists, build_model_configs

    input_parquet = Path(input_parquet)
    log.info(f"[run_training] Loading {input_parquet}")
    df = load_base_dataset(input_parquet)

    # Smoke test slicing
    if max_rows is not None and max_rows < len(df):
        df = df.head(max_rows).reset_index(drop=True)
        log.info(f"[run_training] max_rows={max_rows} → {len(df)} rows")

    # Filtrar linhas com alpha_60d resolvido
    if "alpha_60d" not in df.columns:
        raise KeyError("parquet sem coluna alpha_60d — targets não resolvidos")
    n_pre = len(df)
    df = df[df["alpha_60d"].notna()].reset_index(drop=True)
    log.info(f"[run_training] Alpha_60d resolvido: {len(df)}/{n_pre}")

    if "max_drawdown_60d" not in df.columns:
        # Para compat antiga: cria coluna com NaN (depois é winsorizada)
        log.warning("[run_training] coluna max_drawdown_60d ausente — usar zeros")
        df["max_drawdown_60d"] = 0.0

    # Robustness: target rank cross-section + sector-neutral (diagnóstico)
    df = _maybe_apply_rank_target(df, enable=use_rank_target)
    df = _maybe_apply_sector_neutral(df, enable=use_sector_neutral)

    # Verificar features
    feats_full, feats_baseline = build_feature_lists()
    missing = [c for c in feats_full if c not in df.columns]
    if missing:
        raise ValueError(
            f"FEATURE_COLUMNS em falta no parquet: {missing}. "
            f"Regenera o parquet via scripts/regenerate_training_base.py."
        )

    # Walk-forward CV
    log.info(
        f"[run_training] CV: {n_folds} folds | purge={purge_days}d | "
        f"min_train={min_train} min_test={min_test}"
    )
    model_configs = build_model_configs(feats_full, feats_baseline)
    results, oof_pred, fold_specs = run_walk_forward_cv(
        df_v31=df,
        model_configs=model_configs,
        n_folds=n_folds,
        purge_days=purge_days,
        min_train=min_train,
        min_test=min_test,
    )

    summary = summarize_results(results)
    if summary.empty:
        raise RuntimeError(
            f"Walk-forward CV vazio — todos os folds caíram abaixo de "
            f"min_train={min_train}/min_test={min_test}."
        )
    if log_summary:
        log.info("[run_training] Summary:\n" + summary.round(4).to_string(index=False))

    # Champion + ensemble (diagnóstico)
    champion_name, champion_row = select_champion(summary)
    try:
        from ml_training.cv_robust import build_champion_ensemble
        cv_df = _make_cv_df(results)
        ensemble_models, ensemble_weights = build_champion_ensemble(
            cv_df, n_folds=n_folds, ic_threshold=0.02, ic_min_floor=-0.05,
        )
        log.info(
            f"[run_training] Ensemble (diagnóstico): models={ensemble_models} | "
            f"weights={ {k: round(v,3) for k,v in ensemble_weights.items()} }"
        )
    except Exception as e:
        log.warning(f"[run_training] build_champion_ensemble falhou: {e}")
        ensemble_models, ensemble_weights = [], {}

    # Isotonic calibrator OOF
    iso, brier, n_oof = fit_isotonic_calibrator(
        oof_pred[champion_name],
        df["alpha_60d"].values,
        alpha_threshold=0.05,
    )
    log.info(
        f"[run_training] Calibrator: brier_oof={brier:.4f} | n_oof={n_oof}"
    )

    # Treino full champion
    champ_alpha, champ_down, feats_used, n_train = train_full_champion(
        df_v31=df,
        champion_cfg=model_configs[champion_name],
    )

    # Bundle
    bundle = DipModelsV3(
        model_up=champ_alpha,
        model_down=champ_down,
        feature_cols=feats_used,
        score_calibrator=iso,
        n_train_samples=int(n_train),
        train_date=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        champion_name=champion_name,
        schema_version=3,
        momentum_feats=[
            f for f in (
                "return_1m", "return_3m_pre", "return_6m_pre",
                "return_12m_pre", "beta_60d",
            ) if f in feats_used
        ],
        rho_mean=float(champion_row["rho_alpha_mean"]),
        rho_alpha=float(champion_row["rho_alpha_mean"]),
        rho_down=float(champion_row["rho_down_mean"]),
        topk_pnl=float(champion_row["topk_pnl_mean"]),
        fold_metrics=results[champion_name],
    )

    # Report
    win_rate_alpha = float((df["alpha_60d"] > 0.05).mean())
    report = build_report(
        bundle=bundle,
        summary_df=summary,
        brier_oof=brier,
        win_rate_alpha=win_rate_alpha,
        n_folds_used=len(fold_specs),
        purge_days=purge_days,
        horizon_days=horizon_days,
        new_features=[
            "return_6m_pre", "return_12m_pre", "sector_relative_6m",
            "vol_of_vol", "bb_width", "vix_percentile_1y",
            "spy_rsi_14", "yield_10y_change_5d",
        ],
    )
    # Robustness diagnostic fields (sem quebrar o schema legacy)
    report.setdefault("robustness", {})
    report["robustness"]["use_rank_target"]    = use_rank_target
    report["robustness"]["use_sector_neutral"] = use_sector_neutral
    report["robustness"]["ensemble_models"]    = ensemble_models
    report["robustness"]["ensemble_weights"]   = ensemble_weights

    # Persist
    if output_bundle is not None:
        save_bundle(bundle, Path(output_bundle))
    if output_report is not None:
        save_report(report, Path(output_report))

    return {
        "bundle":         bundle,
        "report":         report,
        "summary":        summary,
        "oof_pred":       oof_pred,
        "champion_name":  champion_name,
        "brier_oof":      brier,
        "n_oof":          n_oof,
        "ensemble_models":  ensemble_models,
        "ensemble_weights": ensemble_weights,
    }


def _make_cv_df(results: dict[str, list[dict]]) -> pd.DataFrame:
    """Converte ``results`` (dict model→list[fold record]) em DataFrame plano.

    Schema esperado por ``cv_robust.build_champion_ensemble``:
      ``model, fold, ic, topk_alpha, hit_rate``
    """
    rows = []
    for name, hist in results.items():
        for h in hist:
            rows.append({
                "model":      name,
                "fold":       h.get("fold"),
                "ic":         h.get("rho_alpha"),
                "topk_alpha": h.get("topk_pnl"),
                "hit_rate":   h.get("hit_rate", 0.5),
            })
    return pd.DataFrame(rows)
