"""Walk-forward CV + champion training + calibrator + stacking ensemble."""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from ml_training.config import HORIZON_DAYS
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

        class PlattCalibrator:
            def __init__(self, scaler, lr):
                self.scaler = scaler
                self.lr = lr
            def predict(self, x):
                import numpy as _np
                X = self.scaler.transform(_np.asarray(x).reshape(-1, 1))
                return self.lr.predict_proba(X)[:, 1]

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
