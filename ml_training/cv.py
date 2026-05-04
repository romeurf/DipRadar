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
    """Winsoriza nas caudas pct/(1-pct). Idempotente em arrays vazios."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return arr
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
