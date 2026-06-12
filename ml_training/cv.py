"""Helpers de CV e métricas — extraídos do notebook (cells 20, 21)."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ml_training.config import (
    HALF_LIFE_DAYS,
    N_FOLDS,
    PURGE_DAYS,
    TOPK_FRAC,
    WINSOR_PCT,
    WINSOR_ABS_LO,
    WINSOR_ABS_HI,
)

# Fraction of training set held out as early-stopping validation
# (taken from the most recent rows to respect temporal order)
_ES_VAL_FRAC: float = 0.07


# ─────────────────────────────────────────────────────────────────────────────
# Folds expanding-window com purga
# ─────────────────────────────────────────────────────────────────────────────

def build_walk_forward_folds(
    df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    purge_days: int = PURGE_DAYS,
    date_col: str = "alert_date",
) -> list[tuple[int, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Constrói folds (k, train_end, purge_end, test_end) expanding-window.

    Cada fold k cobre fracção (0.5 + k/(2*N_FOLDS)) do período total.
    Devolve apenas folds onde test_end > purge_end (folds inválidos são
    silenciosamente ignorados, igual ao notebook).
    """
    if df.empty:
        return []

    dates = pd.to_datetime(df[date_col])
    min_date = dates.min()
    max_date = dates.max()
    total_days = (max_date - min_date).days
    if total_days <= 0:
        return []

    folds: list[tuple[int, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for k in range(n_folds):
        train_frac = 0.50 + 0.50 * k / n_folds
        test_frac = 0.50 + 0.50 * (k + 1) / n_folds
        train_end = min_date + pd.Timedelta(days=int(total_days * train_frac))
        purge_end = train_end + pd.Timedelta(days=purge_days)
        test_end = min_date + pd.Timedelta(days=int(total_days * test_frac))
        if test_end <= purge_end:
            continue
        folds.append((k + 1, train_end, purge_end, test_end))
    return folds


# ─────────────────────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────────────────────

def winsorize(arr: np.ndarray, pct: float = WINSOR_PCT) -> np.ndarray:
    """Winsoriza nas caudas pct/(1-pct) + clip absoluto [-0.5, 2.0].

    Aplica primeiro o clip absoluto (remove outliers extremos como +355x)
    e depois o clip percentílico para suavizar as caudas restantes.
    Idempotente em arrays vazios.
    """
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return arr
    # 1. Clip absoluto — remove outliers extremos independentemente da distribuição
    arr = np.clip(arr, WINSOR_ABS_LO, WINSOR_ABS_HI)
    # 2. Clip percentílico — suaviza caudas restantes
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return arr
    lo = np.quantile(finite, pct)
    hi = np.quantile(finite, 1 - pct)
    return np.clip(arr, lo, hi)


def spearman_safe(pred: Iterable[float], true: Iterable[float]) -> float:
    """Spearman rho ignorando NaNs. NaN se < 5 pares finitos."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() < 5:
        return float("nan")
    rho, _ = spearmanr(pred[mask], true[mask])
    return float(rho)


def topk_pnl(
    pred_alpha: Iterable[float],
    true_alpha: Iterable[float],
    k: float = TOPK_FRAC,
) -> float:
    """Mean(true_alpha) sobre os top-K% por pred_alpha. NaN se < 5 finitos."""
    pred = np.asarray(pred_alpha, dtype=float)
    true = np.asarray(true_alpha, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(true)
    pred = pred[mask]
    true = true[mask]
    if len(pred) < 5:
        return float("nan")
    n_top = max(1, int(len(pred) * k))
    top = np.argsort(-pred)[:n_top]
    return float(true[top].mean())


# ─────────────────────────────────────────────────────────────────────────────
# Sample weights temporais (half-life decay)
# ─────────────────────────────────────────────────────────────────────────────

def temporal_weights(
    alert_dates: Iterable,
    max_date: pd.Timestamp,
    half_life_days: int = HALF_LIFE_DAYS,
) -> np.ndarray:
    """Decay exponencial: peso = 2^(-days_since/half_life_days)."""
    dt = pd.to_datetime(pd.Series(list(alert_dates)))
    days = (max_date - dt).dt.days.values.astype(float)
    return np.exp(-np.log(2) * days / half_life_days)


def apply_missed_alpha_boost(
    base_weights: np.ndarray,
    df: pd.DataFrame,
    boost_factor: float = 2.0,
    alpha_threshold: float = 0.15,
) -> np.ndarray:
    """Boost training weights for "missed alpha" rows — episodes the bot DID
    NOT alert on (source="simulated") but that recovered strongly
    (alpha_90d ≥ alpha_threshold).

    Rationale: the universe-wide builder captures every dip, but the bot's
    alert filter is what decides whether to fire. Boosting these rows during
    training pushes the model to learn the patterns that the filter has been
    over-restrictive on, so future scans recognize them.

    Parameters
    ----------
    base_weights : np.ndarray
        Output of temporal_weights or similar; gets multiplied in-place.
    df : pd.DataFrame
        Same row indexing as base_weights. Must contain `source` and
        `alpha_90d` columns. Missing columns → returns base_weights unchanged
        (graceful degradation for old parquets).
    boost_factor : float
        Multiplicative boost for missed-alpha rows. Default 2.0.
        - 1.0 = no boost (disable)
        - 2.0 = double weight on missed (recommended starting point)
        - 5.0 = aggressive, risk of overfitting / precision loss
    alpha_threshold : float
        Minimum alpha_90d to qualify as "high alpha missed". Default 0.15 (15%).

    Returns
    -------
    np.ndarray with same shape as base_weights, modified copy.
    """
    if base_weights is None or len(base_weights) == 0:
        return base_weights
    if "source" not in df.columns or "alpha_90d" not in df.columns:
        return base_weights
    if boost_factor == 1.0:
        return base_weights

    weights = np.asarray(base_weights, dtype=float).copy()
    if len(weights) != len(df):
        # Length mismatch — usually means the caller passed a different df
        # than the one weights were computed from. Skip boost to be safe.
        return weights

    try:
        is_missed = (
            (df["source"].astype(str) == "simulated") &
            (pd.to_numeric(df["alpha_90d"], errors="coerce") >= alpha_threshold)
        ).values
        n_boosted = int(is_missed.sum())
        if n_boosted > 0:
            weights[is_missed] *= float(boost_factor)
    except Exception:
        # Any failure: silently return original weights (don't break training)
        return base_weights

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Combinador de métricas por fold (mantém compat com o notebook)
# ─────────────────────────────────────────────────────────────────────────────

def fold_metric_record(
    fold: int,
    n_test: int,
    rho_alpha: float,
    rho_down: float,
    pnl: float,
) -> dict:
    """Constrói o dict de métricas usado no notebook (cell 23)."""
    if math.isfinite(rho_alpha) and math.isfinite(rho_down):
        rho_mean = (rho_alpha - rho_down) / 2
    else:
        rho_mean = rho_alpha
    return {
        "fold":      fold,
        "n_test":    n_test,
        "rho_alpha": rho_alpha,
        "rho_down":  rho_down,
        "rho_mean":  rho_mean,
        "topk_pnl":  pnl,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Early-stopping helpers
# ─────────────────────────────────────────────────────────────────────────────

def _supports_early_stopping(model: object) -> str:
    """Return 'xgb', 'lgbm', or '' depending on whether the model supports ES.

    Detection is done by inspecting the class name and the presence of
    early_stopping_rounds in the model's params, so it works without
    importing xgboost/lightgbm at module level.
    """
    cls_name = type(model).__name__
    # XGBRegressor stores early_stopping_rounds as a constructor kwarg
    if cls_name == "XGBRegressor":
        esr = getattr(model, "early_stopping_rounds", None)
        if esr is not None and esr > 0:
            return "xgb"
    # LGBMRegressor: early stopping is passed via callbacks at fit time,
    # but we signal intent by checking for a custom attribute or n_estimators > 500
    if cls_name == "LGBMRegressor":
        # lgbm_es_factory sets n_estimators=800; use that as the heuristic
        n_est = getattr(model, "n_estimators", 0)
        if n_est >= 800:
            return "lgbm"
    return ""


def _fit_with_early_stopping(
    model: object,
    es_type: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    w_tr: np.ndarray,
) -> None:
    """Fit model with early stopping using the provided validation split.

    Parameters
    ----------
    model   : fitted in place
    es_type : 'xgb' or 'lgbm'
    X_tr, y_tr, w_tr : training data + sample weights
    X_val, y_val     : validation set for early stopping signal
    """
    if es_type == "xgb":
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    elif es_type == "lgbm":
        try:
            from lightgbm import early_stopping as lgbm_early_stopping
            from lightgbm import log_evaluation as lgbm_log_evaluation
            callbacks = [
                lgbm_early_stopping(stopping_rounds=50, verbose=False),
                lgbm_log_evaluation(period=-1),
            ]
        except ImportError:
            # Older LightGBM API
            callbacks = None

        fit_kwargs: dict = {
            "X": X_tr,
            "y": y_tr,
            "sample_weight": w_tr,
            "eval_set": [(X_val, y_val)],
        }
        if callbacks is not None:
            fit_kwargs["callbacks"] = callbacks
        else:
            fit_kwargs["early_stopping_rounds"] = 50
            fit_kwargs["verbose"] = False

        model.fit(**fit_kwargs)
    else:
        raise ValueError(f"Unknown es_type: {es_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward CV — orquestrador principal
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_cv(
    df: pd.DataFrame,
    model_configs: dict,
    feature_cols_map: dict[str, list[str]],
    target_col: str = "alpha_90d",
    n_folds: int = N_FOLDS,
    purge_days: int = PURGE_DAYS,
    topk_frac: float = TOPK_FRAC,
    half_life_days: int = HALF_LIFE_DAYS,
    date_col: str = "alert_date",
) -> list[dict]:
    """Corre walk-forward CV para todos os modelos em model_configs.

    Para cada fold e cada modelo:
      1. Treina em train_set (com temporal weights)
         — modelos ES recebem os últimos 7% do treino como eval_set
      2. Avalia em test_set: IC (Spearman), topk_alpha, hit_rate
      3. Guarda registo com {fold, model, ic, topk_alpha, hit_rate, n_test}

    Parâmetros
    ----------
    df               : DataFrame com features + target_col + date_col
    model_configs    : {name: {'factory': callable, 'feats': list[str]}}
    feature_cols_map : {name: list[str]} — normalmente {n: cfg['feats'] for n, cfg in model_configs.items()}
    target_col       : coluna alvo (default 'alpha_90d')
    n_folds          : número de folds
    purge_days       : dias de purga entre train e test
    topk_frac        : fracção top-K para topk_alpha
    half_life_days   : half-life do decay temporal dos pesos
    date_col         : coluna de datas dos alertas

    Devolve
    -------
    Lista de dicts com chaves: fold, model, ic, topk_alpha, hit_rate, n_test
    """
    from sklearn.preprocessing import StandardScaler

    folds = build_walk_forward_folds(df, n_folds=n_folds, purge_days=purge_days, date_col=date_col)
    if not folds:
        raise ValueError("Nenhum fold válido gerado — verifica o tamanho do DataFrame.")

    results: list[dict] = []

    for fold_idx, (fold_k, train_end, purge_end, test_end) in enumerate(folds):
        dates = pd.to_datetime(df[date_col])
        train_mask = dates <= train_end
        test_mask  = (dates > purge_end) & (dates <= test_end)

        df_train = df[train_mask].copy()
        df_test  = df[test_mask].copy()

        if len(df_train) < 20 or len(df_test) < 5:
            print(f"  Fold {fold_k}: skip (train={len(df_train)}, test={len(df_test)})")
            continue

        y_train = df_train[target_col].values
        y_test  = df_test[target_col].values

        # Temporal sample weights (calculated on full train before ES split)
        w_train = temporal_weights(df_train[date_col], train_end, half_life_days)

        # ES validation split: last _ES_VAL_FRAC of training rows (temporal order)
        # Use at least 30 rows for val; fall back to no split if train is tiny.
        n_val = max(30, int(len(df_train) * _ES_VAL_FRAC))
        if n_val >= len(df_train) - 20:
            n_val = max(0, len(df_train) // 10)
        es_split_available = n_val >= 10

        print(f"  Fold {fold_k}/{n_folds}: train={len(df_train):,}  test={len(df_test):,}  "
              f"[{train_end.date()} → {test_end.date()}]  es_val={n_val if es_split_available else 'N/A'}")

        for name, cfg in model_configs.items():
            feats = feature_cols_map.get(name, cfg.get("feats", []))

            # Filtrar features presentes no df
            feats_ok = [f for f in feats if f in df.columns]
            if not feats_ok:
                continue

            X_train_full = df_train[feats_ok].values.astype(float)
            X_test  = df_test[feats_ok].values.astype(float)

            # Imputar NaN com mediana do treino (calculada em X_train_full)
            col_medians = np.nanmedian(X_train_full, axis=0)
            for col_i in range(X_train_full.shape[1]):
                nan_mask = ~np.isfinite(X_train_full[:, col_i])
                if nan_mask.any():
                    X_train_full[nan_mask, col_i] = col_medians[col_i]
                nan_mask_t = ~np.isfinite(X_test[:, col_i])
                if nan_mask_t.any():
                    X_test[nan_mask_t, col_i] = col_medians[col_i]

            # Scaler (Ridge precisa; tree-based é indiferente mas não prejudica)
            scaler = StandardScaler()
            X_train_full = scaler.fit_transform(X_train_full)
            X_test  = scaler.transform(X_test)

            try:
                model = cfg["factory"]()
                es_type = _supports_early_stopping(model)

                if es_type and es_split_available:
                    # Split the most recent n_val rows as ES validation
                    X_tr  = X_train_full[:-n_val]
                    y_tr  = y_train[:-n_val]
                    w_tr  = w_train[:-n_val]
                    X_val = X_train_full[-n_val:]
                    y_val = y_train[-n_val:]

                    _fit_with_early_stopping(model, es_type, X_tr, y_tr, X_val, y_val, w_tr)
                else:
                    # Standard fit: no early stopping
                    try:
                        model.fit(X_train_full, y_train, sample_weight=w_train)
                    except TypeError:
                        model.fit(X_train_full, y_train)

                pred = model.predict(X_test).astype(float)
            except Exception as e:
                print(f"    {name} fold {fold_k}: erro no fit/predict — {e}")
                continue

            ic       = spearman_safe(pred, y_test)
            tk_alpha = topk_pnl(pred, y_test, k=topk_frac)
            # hit_rate: fracção do top-K com alpha > 0
            pred_arr = np.asarray(pred, dtype=float)
            true_arr = np.asarray(y_test, dtype=float)
            mask_fin = np.isfinite(pred_arr) & np.isfinite(true_arr)
            if mask_fin.sum() >= 5:
                n_top = max(1, int(mask_fin.sum() * topk_frac))
                top_idx = np.argsort(-pred_arr[mask_fin])[:n_top]
                hit_rate = float((true_arr[mask_fin][top_idx] > 0).mean())
            else:
                hit_rate = float("nan")

            results.append({
                "fold":       fold_k,
                "model":      name,
                "ic":         ic,
                "topk_alpha": tk_alpha,
                "hit_rate":   hit_rate,
                "n_test":     int(len(df_test)),
            })

    return results
