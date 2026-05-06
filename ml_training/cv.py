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
# Walk-forward CV — orquestrador principal
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_cv(
    df: pd.DataFrame,
    model_configs: dict,
    feature_cols_map: dict[str, list[str]],
    target_col: str = "alpha_60d",
    n_folds: int = N_FOLDS,
    purge_days: int = PURGE_DAYS,
    topk_frac: float = TOPK_FRAC,
    half_life_days: int = HALF_LIFE_DAYS,
    date_col: str = "alert_date",
) -> list[dict]:
    """Corre walk-forward CV para todos os modelos em model_configs.

    Para cada fold e cada modelo:
      1. Treina em train_set (com temporal weights)
      2. Avalia em test_set: IC (Spearman), topk_alpha, hit_rate
      3. Guarda registo com {fold, model, ic, topk_alpha, hit_rate, n_test}

    Parâmetros
    ----------
    df               : DataFrame com features + target_col + date_col
    model_configs    : {name: {'factory': callable, 'feats': list[str]}}
    feature_cols_map : {name: list[str]} — normalmente {n: cfg['feats'] for n, cfg in model_configs.items()}
    target_col       : coluna alvo (default 'alpha_60d')
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

        # Temporal sample weights
        w_train = temporal_weights(df_train[date_col], train_end, half_life_days)

        print(f"  Fold {fold_k}/{n_folds}: train={len(df_train):,}  test={len(df_test):,}  "
              f"[{train_end.date()} → {test_end.date()}]")

        for name, cfg in model_configs.items():
            feats = feature_cols_map.get(name, cfg.get("feats", []))

            # Filtrar features presentes no df
            feats_ok = [f for f in feats if f in df.columns]
            if not feats_ok:
                continue

            X_train = df_train[feats_ok].values.astype(float)
            X_test  = df_test[feats_ok].values.astype(float)

            # Imputar NaN com mediana do treino
            col_medians = np.nanmedian(X_train, axis=0)
            for col_i in range(X_train.shape[1]):
                nan_mask = ~np.isfinite(X_train[:, col_i])
                if nan_mask.any():
                    X_train[nan_mask, col_i] = col_medians[col_i]
                nan_mask_t = ~np.isfinite(X_test[:, col_i])
                if nan_mask_t.any():
                    X_test[nan_mask_t, col_i] = col_medians[col_i]

            # Scaler (Ridge precisa; tree-based é indiferente mas não prejudica)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

            try:
                model = cfg["factory"]()
                # Passar sample_weight se o modelo suportar
                try:
                    model.fit(X_train, y_train, sample_weight=w_train)
                except TypeError:
                    model.fit(X_train, y_train)

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
