"""Walk-forward CV + champion training + calibrator — extraído do notebook (cells 23, 28-30)."""

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
# Walk-forward CV (cell 23)
# ─────────────────────────────────────────────────────────────────────────────

def run_walk_forward_cv(
    df_v31: pd.DataFrame,
    model_configs: dict[str, dict],
    n_folds: int,
    purge_days: int,
    min_train: int = 100,
    min_test: int = 20,
) -> tuple[dict[str, list[dict]], dict[str, np.ndarray], list[tuple]]:
    """Corre walk-forward CV em todos os modelos candidatos.

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
        y_down_tr = winsorize(df_tr["max_drawdown_60d"].values)
        y_down_te = df_te["max_drawdown_60d"].values

        sw_tr = temporal_weights(df_tr["alert_date"], max_date)

        for name, cfg in model_configs.items():
            feats = [f for f in cfg["feats"] if f in df_tr.columns]
            X_tr = df_tr[feats].fillna(0).values.astype(np.float32)
            X_te = df_te[feats].fillna(0).values.astype(np.float32)

            m_alpha = cfg["factory"]()
            try:
                m_alpha.fit(X_tr, y_alpha_tr, sample_weight=sw_tr)
            except TypeError:
                m_alpha.fit(X_tr, y_alpha_tr)

            m_down = cfg["factory"]()
            try:
                m_down.fit(X_tr, y_down_tr, sample_weight=sw_tr)
            except TypeError:
                m_down.fit(X_tr, y_down_tr)

            pred_alpha = m_alpha.predict(X_te)
            pred_down = m_down.predict(X_te)

            oof_pred[name][df_te.index.values] = pred_alpha

            rho_alpha = spearman_safe(pred_alpha, y_alpha_te)
            rho_down = spearman_safe(pred_down, y_down_te)
            pnl = topk_pnl(pred_alpha, y_alpha_te)

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
# Sumarização + Champion (cells 25, 28)
# ─────────────────────────────────────────────────────────────────────────────

def summarize_results(results: dict[str, list[dict]]) -> pd.DataFrame:
    """Tabela resumo (médias e std) por modelo. Replica cell 25.

    Devolve DataFrame vazio (com colunas certas) se nenhum modelo teve
    folds válidos — caller deve raise antes de tentar select_champion.
    """
    cols = ["model", "rho_alpha_mean", "rho_alpha_std",
            "rho_down_mean", "topk_pnl_mean", "topk_pnl_std", "n_folds"]
    rows = []
    for name, hist in results.items():
        if not hist:
            continue
        rho_alphas = np.array([h["rho_alpha"] for h in hist if math.isfinite(h["rho_alpha"])])
        rho_downs = np.array([h["rho_down"] for h in hist if math.isfinite(h["rho_down"])])
        pnls = np.array([h["topk_pnl"] for h in hist if math.isfinite(h["topk_pnl"])])
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
    """Champion = melhor rho_alpha_mean com topk_pnl_mean > 0 (cell 28).

    Fallback: se nenhum passa o filtro PnL, escolhe melhor rho mesmo assim.
    Devolve (champion_name, champion_row).
    """
    if summary.empty:
        raise ValueError("summary vazio — sem modelos para escolher champion")

    qualifiers = summary[summary["topk_pnl_mean"] > 0].sort_values(
        "rho_alpha_mean", ascending=False
    )
    if len(qualifiers) == 0:
        log.warning("Nenhum modelo passou no critério topk_pnl > 0 — fallback ao melhor rho")
        qualifiers = summary.sort_values("rho_alpha_mean", ascending=False)

    champion_name = str(qualifiers.iloc[0]["model"])
    return champion_name, qualifiers.iloc[0]


# ─────────────────────────────────────────────────────────────────────────────
# Calibrator isotónico em OOF (cell 29)
# ─────────────────────────────────────────────────────────────────────────────

def fit_isotonic_calibrator(
    oof_pred_champion: np.ndarray,
    alpha_true: np.ndarray,
    alpha_threshold: float = 0.05,
) -> tuple[object, float, int]:
    """Isotonic regression: pred_alpha → P(alpha > threshold).

    Usa apenas linhas com OOF prediction válida (não NaN).
    Devolve (iso_model, brier_score, n_oof_samples).
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss

    mask = np.isfinite(oof_pred_champion)
    y_oof = oof_pred_champion[mask]
    y_bin = (alpha_true[mask] > alpha_threshold).astype(int)

    if mask.sum() < 100:
        log.warning(f"OOF samples insuficientes: {mask.sum()}; calibrator pode ser instável")

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(y_oof, y_bin)
    prob_oof = iso.predict(y_oof)
    brier = float(brier_score_loss(y_bin, prob_oof))
    return iso, brier, int(mask.sum())


# ─────────────────────────────────────────────────────────────────────────────
# Treino full do champion (cell 30)
# ─────────────────────────────────────────────────────────────────────────────

def train_full_champion(
    df_v31: pd.DataFrame,
    champion_cfg: dict,
) -> tuple[object, object, list[str], int]:
    """Treina (model_up, model_down) no dataset COMPLETO.

    Devolve (champ_alpha, champ_down, feats_used, n_train).
    """
    df_v31 = df_v31.sort_values("alert_date").reset_index(drop=True)
    max_date = pd.to_datetime(df_v31["alert_date"]).max()
    feats = [f for f in champion_cfg["feats"] if f in df_v31.columns]

    X_full = df_v31[feats].fillna(0).values.astype(np.float32)
    y_alpha_full = winsorize(df_v31["alpha_60d"].values)
    y_down_full = winsorize(df_v31["max_drawdown_60d"].values)
    sw_full = temporal_weights(df_v31["alert_date"], max_date)

    champ_alpha = champion_cfg["factory"]()
    try:
        champ_alpha.fit(X_full, y_alpha_full, sample_weight=sw_full)
    except TypeError:
        champ_alpha.fit(X_full, y_alpha_full)

    champ_down = champion_cfg["factory"]()
    try:
        champ_down.fit(X_full, y_down_full, sample_weight=sw_full)
    except TypeError:
        champ_down.fit(X_full, y_down_full)

    log.info(
        f"Champion treinado em {len(X_full)} amostras com {len(feats)} features"
    )
    return champ_alpha, champ_down, feats, len(X_full)
