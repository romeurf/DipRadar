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
    train_target: str = "alpha_60d",
) -> tuple[dict[str, list[dict]], dict[str, np.ndarray], list[tuple]]:
    """Walk-forward CV em todos os modelos candidatos.

    Parameters
    ----------
    train_target : str
        Coluna usada como Y para fit dos modelos. Use ``"alpha_60d_rank"``
        para treino sobre rank cross-section (mais estável). Avaliação
        (IC, top-K PnL) usa sempre ``alpha_60d`` bruto para
        comparabilidade entre experiências.

    Devolve:
      - results: ``{model_name: [fold_metric_records]}``
      - oof_pred: ``{model_name: np.ndarray (full size, NaN onde não testado)}``
      - fold_specs: lista (k, train_end, purge_end, test_end)
    """
    df_v31 = df_v31.sort_values("alert_date").reset_index(drop=True)
    df_v31["alert_date"] = pd.to_datetime(df_v31["alert_date"])
    max_date = df_v31["alert_date"].max()

    if train_target not in df_v31.columns:
        raise KeyError(
            f"train_target='{train_target}' não está no DataFrame "
            f"(colunas: {list(df_v31.columns)[:10]}...)"
        )

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

        # Training Y: winsorize qualquer target (rank [0,1] é no-op após clip).
        y_alpha_tr = winsorize(df_tr[train_target].values.astype(float))
        # Evaluation Y: usa o mesmo target que o treino para IC comparável.
        # Se o parquet não tiver a coluna (ex: alpha_90d em parquet antigo),
        # cai de volta para alpha_60d — avisa mas não falha.
        _eval_col = train_target if train_target in df_te.columns else "alpha_60d"
        y_alpha_te = df_te[_eval_col].values
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


class EnsembleRegressor:
    """Wrapper picklable que expõe `.predict()` como soma ponderada de N modelos.

    Drop-in para `model_up` / `model_down` em `DipModelsV3`. Cada base
    model é treinado no MESMO feature set (validado pelo orchestrator)
    com early stopping. As previsões são combinadas via weights >= 0
    que somam 1.0 (proporcionais ao IC de cada modelo em walk-forward CV).

    Definido em módulo (não closure) para joblib/pickle funcionarem.
    """

    def __init__(self, models: "list", weights: "list[float]", names: "list[str] | None" = None):
        if len(models) != len(weights):
            raise ValueError(f"models ({len(models)}) != weights ({len(weights)})")
        self.models = list(models)
        self.weights = np.asarray(weights, dtype=float)
        self.names = list(names) if names is not None else [f"m{i}" for i in range(len(models))]
        s = float(self.weights.sum())
        if s <= 0:
            self.weights = np.full(len(models), 1.0 / len(models))
        else:
            self.weights = self.weights / s

    def predict(self, X) -> np.ndarray:
        preds = np.zeros(len(X), dtype=float)
        for w, m in zip(self.weights, self.models):
            preds = preds + w * np.asarray(m.predict(X), dtype=float)
        return preds


# ─────────────────────────────────────────────────────────────────────────────
# Feature winsorization (training-time, não modifica o parquet)
# ─────────────────────────────────────────────────────────────────────────────

# Features prone a outliers extremos (yfinance retorna 5000%+ em casos como
# AXON.PA 2003-2004, ATO.PA 2024 reverse split, etc.). Clipamos a ±200%
# antes do treino para que linear models e GBMs não overfittem aos outliers.
# Aplicado apenas em runtime — o parquet mantém os valores brutos.
_FEATURE_WINSOR_CLIPS: dict[str, tuple[float, float]] = {
    "return_1m":      (-99.0,  200.0),
    "return_3m_pre":  (-99.0,  200.0),
    "return_6m_pre":  (-99.0,  200.0),
    "sector_relative":(-200.0, 200.0),
}


def _winsorize_outlier_features(df: pd.DataFrame, *, log_changes: bool = True) -> pd.DataFrame:
    """Clipa colunas com outliers extremos in-place no DataFrame.

    Não modifica o parquet em disco — apenas a cópia em memória para CV.
    """
    df = df.copy()
    for col, (lo, hi) in _FEATURE_WINSOR_CLIPS.items():
        if col not in df.columns:
            continue
        before = df[col].copy()
        df[col] = df[col].clip(lower=lo, upper=hi)
        n_clipped = int(((before != df[col]) & before.notna()).sum())
        if log_changes and n_clipped > 0:
            log.info(
                f"[winsor_features] {col}: clipped {n_clipped} rows to [{lo}, {hi}]"
            )
    return df


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
    *,
    train_target: str = "alpha_60d",
) -> tuple[object, object, list[str], int]:
    """Treina (model_up, model_down) no dataset COMPLETO com early stopping.

    Parameters
    ----------
    train_target : str
        Coluna usada como Y para fit do model_up. Default ``alpha_60d``.
        Use ``alpha_60d_rank`` para treino sobre rank cross-section.

    Devolve (champ_alpha, champ_down, feats_used, n_train).
    """
    df_v31   = df_v31.sort_values("alert_date").reset_index(drop=True)
    max_date = pd.to_datetime(df_v31["alert_date"]).max()
    feats    = [f for f in champion_cfg["feats"] if f in df_v31.columns]

    X_full        = df_v31[feats].fillna(0).values.astype(np.float32)
    if train_target == "alpha_60d":
        y_alpha_full  = winsorize(df_v31["alpha_60d"].values)
    else:
        y_alpha_full  = df_v31[train_target].values.astype(float)
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


def train_full_ensemble(
    df_v31: pd.DataFrame,
    model_configs: dict[str, dict],
    ensemble_names: list[str],
    ensemble_weights: dict[str, float],
    *,
    train_target: str = "alpha_60d",
) -> tuple["EnsembleRegressor", "EnsembleRegressor", list[str], int]:
    """Treina ensemble (model_up + model_down) no dataset COMPLETO.

    Cada modelo do ensemble usa o seu próprio feature set declarado em
    ``model_configs[name]["feats"]``. Para o wrapper EnsembleRegressor
    funcionar com uma única matriz X em inference time, todos os modelos
    do ensemble DEVEM partilhar o mesmo feature set. Esta função valida
    essa condição e levanta ValueError se houver divergência.

    Devolve (ensemble_alpha, ensemble_down, feats_used, n_train).
    """
    df_v31   = df_v31.sort_values("alert_date").reset_index(drop=True)
    max_date = pd.to_datetime(df_v31["alert_date"]).max()

    # Validar que os modelos partilham o mesmo feature set
    feat_sets = {
        name: tuple(f for f in model_configs[name]["feats"] if f in df_v31.columns)
        for name in ensemble_names
    }
    unique_sets = set(feat_sets.values())
    if len(unique_sets) > 1:
        raise ValueError(
            "Ensemble requer modelos com mesmo feature set. Encontrados: "
            + ", ".join(f"{n}={len(feat_sets[n])}" for n in ensemble_names)
        )
    feats = list(unique_sets.pop())

    X_full        = df_v31[feats].fillna(0).values.astype(np.float32)
    if train_target == "alpha_60d":
        y_alpha_full  = winsorize(df_v31["alpha_60d"].values)
    else:
        y_alpha_full  = df_v31[train_target].values.astype(float)
    y_down_full   = winsorize(df_v31["max_drawdown_60d"].values)
    sw_full       = temporal_weights(df_v31["alert_date"], max_date)

    X_t, y_t_a, sw_t, X_v, y_v_a = _split_val(X_full, y_alpha_full, sw_full, val_frac=0.07)
    _, y_t_d, _, _, y_v_d         = _split_val(X_full, y_down_full,  sw_full, val_frac=0.07)

    alpha_models: list = []
    down_models: list = []
    weight_list: list[float] = []
    for name in ensemble_names:
        cfg = model_configs[name]
        m_alpha = _fit_model(cfg["factory"](), X_t, y_t_a, sw_t, X_v, y_v_a)
        m_down  = _fit_model(cfg["factory"](), X_t, y_t_d, sw_t, X_v, y_v_d)
        alpha_models.append(m_alpha)
        down_models.append(m_down)
        weight_list.append(float(ensemble_weights.get(name, 0.0)))

    ensemble_alpha = EnsembleRegressor(alpha_models, weight_list, names=list(ensemble_names))
    ensemble_down  = EnsembleRegressor(down_models,  weight_list, names=list(ensemble_names))

    log.info(
        f"Ensemble treinado em {len(X_full)} amostras com {len(feats)} features | "
        f"members={ensemble_names} | weights={[round(w, 3) for w in ensemble_alpha.weights]}"
    )
    return ensemble_alpha, ensemble_down, feats, len(X_full)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator end-to-end (drop-in para train_v31.run_training)
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_apply_rank_target(
    df: pd.DataFrame, *, enable: bool = True, raw_col: str = "alpha_60d"
) -> pd.DataFrame:
    """Adiciona ``{raw_col}_rank`` cross-section por data."""
    if not enable:
        return df
    from ml_training.cv_robust import add_crosssection_rank
    return add_crosssection_rank(df, raw_col=raw_col, date_col="alert_date")


def _maybe_apply_sector_neutral(
    df: pd.DataFrame, *, enable: bool = True, target_col: str = "alpha_60d"
) -> pd.DataFrame:
    """Adiciona ``{target_col}_neutral`` e ``{target_col}_neutral_rank``.

    Subtrai mediana de sector por data antes de re-rankear. Útil para o
    modelo aprender "compra dip dentro do sector" em vez de "compra tech".
    """
    if not enable or "sector" not in df.columns:
        return df
    from ml_training.target_engineering import neutralize_target_by_sector
    return neutralize_target_by_sector(
        df,
        target_col=target_col,
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
    # PR #25 activou rank target training + ensemble champion como defaults
    # mas o run real (11/05/2026, 36929 rows × 10 folds) retornou
    # rho_alpha=-0.0130 (vs 0.0820 com PR #23 baseline). Causa raiz:
    #   1. rank target treina o modelo a prever rank WITHIN alert_date,
    #      mas avaliação Spearman é sobre o test fold inteiro (multi-data).
    #      Modelo perde a capacidade de distinguir entre datas.
    #      → Continua False (PR #27): activação requer alterar gating
    #         para rank-within-date evaluation também.
    #   2. ensemble juntava modelos com feature sets diferentes em CV mas
    #      filtrava após build_champion_ensemble, deixando subset sub-óptimo.
    #      OOF IC pooled vs single's per-fold mean IC — métricas
    #      não-comparáveis. → Fixed em PR #27 (eligibility filter
    #      pre-selecciona shared feature set; gating per-fold mean IC;
    #      top_n=4 truncation). Default True activado.
    use_rank_target_train: bool = False,
    use_ensemble_champion: bool = True,
    winsorize_features: bool = True,
    ensemble_ic_threshold: float = 0.04,
    ensemble_ic_min_floor: float = -0.05,
    ensemble_top_n: int = 4,
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
        Se True, computa ``alpha_60d_rank`` para uso pelo CV/full train
        (também útil como coluna de diagnóstico). Default True.
    use_sector_neutral : bool
        Se True, adiciona ``alpha_60d_neutral`` ao DataFrame (apenas
        diagnóstico — modelo nunca treina sobre este target).
    use_rank_target_train : bool
        Se True E ``use_rank_target=True``, modelos são treinados sobre
        ``alpha_60d_rank`` em vez do bruto. Evaluation continua em
        ``alpha_60d`` para comparabilidade. **Default False** (PR #25
        descobriu que rank-within-date training degrada IC cross-date
        global — flag mantido para experimentação futura, e.g. quando
        avaliação for também rank-within-date).
    use_ensemble_champion : bool
        Se True, constrói ensemble IC^2-weighted dos top_N modelos
        robustos (sharing feature set) e usa como champion final SE o
        seu IC per-fold mean for >= ao single best per-fold mean IC.
        **Default True** (PR #27 corrigiu os bugs do PR #25 que
        causaram regressão: 1. eligibility filter pré-selecciona apenas
        modelos partilhando feature set dominante; 2. gating usa IC
        per-fold mean comparable com single best; 3. top_n truncation
        evita diluição por modelos similares. Se ensemble não vencer,
        fallback automático para single).
    winsorize_features : bool
        Se True, clipa features prone a outliers extremos (return_1m,
        return_3m_pre, return_6m_pre, sector_relative) a ±200% antes do
        treino. Default True.
    ensemble_ic_threshold : float
        IC mínimo para entrar no ensemble (default 0.04).
    ensemble_ic_min_floor : float
        IC mínimo por fold (default -0.05; rejeita modelos com folds
        catastróficos).
    ensemble_top_n : int
        Número máximo de modelos no ensemble (default 4). Reduzir diluição
        de sinal por modelos similares com predições correlacionadas.
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
      - ``champion_kind``  : "single" ou "ensemble"
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
    # Selecção do target: alpha_90d (preferencial) com fallback automático para
    # alpha_60d se o parquet não tiver dados suficientes de 90d.
    # Quando o parquet for regenerado com --targets-only, o próximo retrain
    # usa automaticamente alpha_90d sem nenhuma intervenção manual.
    from ml_training.config import PRIMARY_TARGET, PRIMARY_TARGET_FALLBACK
    _n_90d = int(df[PRIMARY_TARGET].notna().sum()) if PRIMARY_TARGET in df.columns else 0
    if _n_90d >= min_train * 2:
        _active_target = PRIMARY_TARGET
        log.info(f"[run_training] Target: {_active_target} ({_n_90d} linhas resolvidas)")
    elif PRIMARY_TARGET_FALLBACK in df.columns and df[PRIMARY_TARGET_FALLBACK].notna().sum() >= min_train * 2:
        _active_target = PRIMARY_TARGET_FALLBACK
        log.warning(
            f"[run_training] {PRIMARY_TARGET} insuficiente ({_n_90d} linhas) — "
            f"fallback para {PRIMARY_TARGET_FALLBACK}. "
            f"Corre /admin_regen_parquet --targets-only para activar o horizonte de 90d."
        )
    else:
        raise KeyError(
            f"Parquet sem target resolvido ({PRIMARY_TARGET}: {_n_90d} linhas). "
            f"Corre /admin_regen_parquet --targets-only."
        )
    n_pre = len(df)
    df = df[df[_active_target].notna()].reset_index(drop=True)
    log.info(f"[run_training] Target '{_active_target}' resolvido: {len(df)}/{n_pre}")

    if "max_drawdown_60d" not in df.columns:
        # Para compat antiga: cria coluna com NaN (depois é winsorizada)
        log.warning("[run_training] coluna max_drawdown_60d ausente — usar zeros")
        df["max_drawdown_60d"] = 0.0

    # Robustness: target rank cross-section + sector-neutral
    df = _maybe_apply_rank_target(df, enable=use_rank_target, raw_col=_active_target)
    df = _maybe_apply_sector_neutral(df, enable=use_sector_neutral, target_col=_active_target)

    # Feature winsorization (clipa return_1m/3m/6m a ±200% para tratar
    # outliers tipo AXON.PA 5000% / ATO.PA 9549%)
    if winsorize_features:
        df = _winsorize_outlier_features(df)

    # Resolve training target (rank se ambos os flags activos + coluna existe)
    _rank_col = f"{_active_target}_rank"
    train_target_col = _active_target
    if use_rank_target_train and use_rank_target and _rank_col in df.columns:
        train_target_col = _rank_col
        log.info(f"[run_training] Training target: {train_target_col} (cross-section rank)")
    else:
        log.info(f"[run_training] Training target: {train_target_col} (raw alpha)")

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
        train_target=train_target_col,
    )

    summary = summarize_results(results)
    if summary.empty:
        raise RuntimeError(
            f"Walk-forward CV vazio — todos os folds caíram abaixo de "
            f"min_train={min_train}/min_test={min_test}."
        )
    if log_summary:
        log.info("[run_training] Summary:\n" + summary.round(4).to_string(index=False))

    # Single best champion (para fallback e comparação com ensemble)
    single_champion_name, single_champion_row = select_champion(summary)
    log.info(
        f"[run_training] Single best: {single_champion_name} "
        f"(rho_alpha={float(single_champion_row['rho_alpha_mean']):.4f})"
    )

    # Construção do ensemble (selecciona top modelos robustos)
    # Pré-filtramos os candidatos para apenas os que partilham o feature set
    # mais comum entre todos os modelos. Razão: o ensemble produção (matriz
    # X única) só funciona com modelos a partilharem features; se incluirmos
    # candidatos com feature sets diferentes, a OOF IC estimada não
    # corresponde ao ensemble que vai para produção (gating unsound).
    from collections import Counter
    all_feat_sets = {
        name: tuple(f for f in cfg["feats"] if f in df.columns)
        for name, cfg in model_configs.items()
    }
    dominant_feat_set = Counter(all_feat_sets.values()).most_common(1)[0][0]
    eligible_models = [
        name for name, fs in all_feat_sets.items() if fs == dominant_feat_set
    ]
    excluded_for_features = sorted(set(model_configs) - set(eligible_models))
    if excluded_for_features:
        log.info(
            f"[run_training] Ensemble eligibility filter: {len(eligible_models)} "
            f"models share dominant feature set ({len(dominant_feat_set)} features); "
            f"excluded (non-shared feats): {excluded_for_features}"
        )

    ensemble_names: list[str] = []
    ensemble_weights: dict[str, float] = {}
    try:
        from ml_training.cv_robust import build_champion_ensemble
        cv_df_all = _make_cv_df(results)
        cv_df = cv_df_all[cv_df_all["model"].isin(eligible_models)].copy()
        ensemble_names, ensemble_weights = build_champion_ensemble(
            cv_df,
            n_folds=n_folds,
            ic_threshold=ensemble_ic_threshold,
            ic_min_floor=ensemble_ic_min_floor,
            top_n=ensemble_top_n,
        )
        log.info(
            f"[run_training] Ensemble candidato: members={ensemble_names} | "
            f"weights={ {k: round(v,3) for k,v in ensemble_weights.items()} }"
        )
    except Exception as e:
        log.warning(f"[run_training] build_champion_ensemble falhou: {e}")

    # Defensive guard: confirmar que todos os ensemble_names partilham
    # feature set (deve ser sempre verdade após o eligibility filter).
    if use_ensemble_champion and ensemble_names:
        feat_sets = {name: all_feat_sets[name] for name in ensemble_names}
        if len(set(feat_sets.values())) > 1:
            log.error(
                "[run_training] BUG: ensemble candidates não partilham feature set "
                "após eligibility filter — fallback para single champion"
            )
            ensemble_names = []

    # Decidir champion final: ensemble vs single
    use_ensemble = (
        use_ensemble_champion
        and len(ensemble_names) >= 2  # 1 modelo não é ensemble
    )
    ensemble_oof: Optional[np.ndarray] = None
    ensemble_ic_oof: Optional[float] = None
    ensemble_ic_per_fold_mean: Optional[float] = None
    if use_ensemble:
        ensemble_oof = np.full(len(df), np.nan)
        valid_mask = np.zeros(len(df), dtype=bool)
        for name in ensemble_names:
            if name not in oof_pred:
                continue
            w = ensemble_weights.get(name, 0.0)
            arr = oof_pred[name]
            local_mask = np.isfinite(arr)
            if not valid_mask.any():
                valid_mask = local_mask.copy()
                ensemble_oof[local_mask] = 0.0
            else:
                valid_mask = valid_mask & local_mask
            ensemble_oof[valid_mask] += w * arr[valid_mask]
        # Mascarar onde não há predição válida para todos os membros
        ensemble_oof = np.where(valid_mask, ensemble_oof, np.nan)
        # IC OOF pooled (diagnóstico — não compara com per-fold mean)
        try:
            ensemble_ic_oof = float(
                spearman_safe(ensemble_oof[valid_mask], df["alpha_60d"].values[valid_mask])
            )
        except Exception:
            ensemble_ic_oof = None
        # IC per-fold mean — métrica COMPARÁVEL com single best (que usa
        # também per-fold mean). Reconstroi-se test masks por fold a partir
        # de fold_specs e calcula-se Spearman dentro de cada fold.
        per_fold_ics: list[float] = []
        y_alpha_all = df["alpha_60d"].values
        for k_idx, train_end, purge_end, test_end in fold_specs:
            te_mask = (df["alert_date"] > purge_end) & (df["alert_date"] <= test_end)
            fold_mask = te_mask.values & valid_mask
            if int(fold_mask.sum()) < 10:
                continue
            try:
                ic_fold = float(spearman_safe(
                    ensemble_oof[fold_mask], y_alpha_all[fold_mask]
                ))
                if np.isfinite(ic_fold):
                    per_fold_ics.append(ic_fold)
            except Exception:
                pass
        if per_fold_ics:
            ensemble_ic_per_fold_mean = float(np.mean(per_fold_ics))
        single_ic = float(single_champion_row["rho_alpha_mean"])
        log.info(
            f"[run_training] Ensemble IC (per-fold mean) = {ensemble_ic_per_fold_mean} "
            f"| pooled OOF = {ensemble_ic_oof} | single best (per-fold mean) = {single_ic:.4f}"
        )
        # Critério: ensemble vence se IC per-fold mean >= single's per-fold mean IC
        # (margem zero, comparação fair com a mesma métrica)
        if ensemble_ic_per_fold_mean is None or ensemble_ic_per_fold_mean < single_ic:
            log.info(
                "[run_training] Ensemble não bate single (per-fold mean) → champion = single"
            )
            use_ensemble = False

    champion_kind = "ensemble" if use_ensemble else "single"
    if use_ensemble:
        champion_name = "Ensemble(" + "+".join(ensemble_names) + ")"
        # OOF predictions do champion = ensemble OOF (para calibrator)
        champion_oof_pred = ensemble_oof
    else:
        champion_name = single_champion_name
        champion_oof_pred = oof_pred[single_champion_name]
    log.info(f"[run_training] Champion final: {champion_kind}={champion_name}")

    # Isotonic calibrator OOF (usa champion_oof_pred, seja single ou ensemble)
    from ml_training.config import CALIBRATOR_THRESHOLD
    iso, brier, n_oof = fit_isotonic_calibrator(
        champion_oof_pred,
        df[_active_target].values,
        alpha_threshold=CALIBRATOR_THRESHOLD,
    )
    log.info(
        f"[run_training] Calibrator: brier_oof={brier:.4f} | n_oof={n_oof}"
    )

    # Treino full do champion (single ou ensemble)
    if use_ensemble:
        champ_alpha, champ_down, feats_used, n_train = train_full_ensemble(
            df_v31=df,
            model_configs=model_configs,
            ensemble_names=ensemble_names,
            ensemble_weights=ensemble_weights,
            train_target=train_target_col,
        )
    else:
        champ_alpha, champ_down, feats_used, n_train = train_full_champion(
            df_v31=df,
            champion_cfg=model_configs[single_champion_name],
            train_target=train_target_col,
        )

    # Métricas para o report. Usamos sempre métrica per-fold mean para
    # comparabilidade entre runs e com floor gating em monthly_retrain.
    if use_ensemble and ensemble_ic_per_fold_mean is not None:
        rho_alpha_final = ensemble_ic_per_fold_mean
    else:
        rho_alpha_final = float(single_champion_row["rho_alpha_mean"])
    rho_down_final = float(single_champion_row["rho_down_mean"])
    topk_pnl_final = float(single_champion_row["topk_pnl_mean"])

    # win_rate: fracção do dataset com alpha > calibrator threshold
    win_rate_alpha = float((df[_active_target] > CALIBRATOR_THRESHOLD).mean())

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
                "beta_60d",
            ) if f in feats_used
        ],
        rho_mean=rho_alpha_final,
        rho_alpha=rho_alpha_final,
        rho_down=rho_down_final,
        topk_pnl=topk_pnl_final,
        fold_metrics=results.get(single_champion_name, []),
    )

    # Baseline de features para drift detection em produção.
    # health_monitor.check_feature_drift() compara distribuições live vs este baseline.
    _feature_stats: dict = {}
    for col in feats_used:
        if col in df.columns:
            try:
                vals = df[col].dropna().astype(float)
                if len(vals) > 5:
                    _feature_stats[col] = {
                        "mean": round(float(vals.mean()), 6),
                        "std":  round(float(vals.std()), 6),
                        "p5":   round(float(vals.quantile(0.05)), 6),
                        "p95":  round(float(vals.quantile(0.95)), 6),
                    }
            except Exception:
                pass

    # Report
    report = build_report(
        bundle=bundle,
        summary_df=summary,
        brier_oof=brier,
        win_rate_alpha=win_rate_alpha,
        n_folds_used=len(fold_specs),
        purge_days=purge_days,
        horizon_days=horizon_days,
        new_features=[
            "return_6m_pre", "vol_of_vol", "bb_width",
            "vix_percentile_1y", "spy_rsi_14",
        ],
        feature_stats=_feature_stats,
    )
    # Robustness diagnostic fields (sem quebrar o schema legacy)
    report.setdefault("robustness", {})
    report["robustness"]["use_rank_target"]        = use_rank_target
    report["robustness"]["use_rank_target_train"]  = use_rank_target_train and train_target_col == "alpha_60d_rank"
    report["robustness"]["use_sector_neutral"]     = use_sector_neutral
    report["robustness"]["train_target_column"]    = train_target_col
    report["robustness"]["winsorize_features"]     = winsorize_features
    report["robustness"]["champion_kind"]          = champion_kind
    report["robustness"]["ensemble_models"]        = ensemble_names
    report["robustness"]["ensemble_weights"]       = ensemble_weights
    report["robustness"]["ensemble_ic_oof"]        = ensemble_ic_oof
    report["robustness"]["ensemble_ic_per_fold_mean"] = ensemble_ic_per_fold_mean
    report["robustness"]["single_best_name"]       = single_champion_name
    report["robustness"]["single_best_rho_alpha"]  = float(single_champion_row["rho_alpha_mean"])

    # Persist
    if output_bundle is not None:
        save_bundle(bundle, Path(output_bundle))
    if output_report is not None:
        save_report(report, Path(output_report))

    return {
        "bundle":          bundle,
        "report":          report,
        "summary":         summary,
        "oof_pred":        oof_pred,
        "champion_name":   champion_name,
        "champion_kind":   champion_kind,
        "brier_oof":       brier,
        "n_oof":           n_oof,
        "ensemble_models":  ensemble_names,
        "ensemble_weights": ensemble_weights,
        "ensemble_ic_oof":  ensemble_ic_oof,
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
