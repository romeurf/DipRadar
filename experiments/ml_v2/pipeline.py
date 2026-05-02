"""
experiments/ml_v2/pipeline.py

XGBoost regression pipeline for DipRadar v2.
Predicts max_return_60d (upside) and max_drawdown_60d (downside) separately,
then scores each alert as risk/reward = up / abs(down).

Key design decisions:
  - 2 separate models (up vs down have different distributions)
  - log1p transform on targets (stabilises heavy tails)
  - Temporal anti-leakage: features built only from data up to alert_date
  - Sequence-aware targets: drawdown measured BEFORE peak (buy-the-dip logic)
  - Clip extremes before transform to remove data errors
  - No accuracy metric — use Spearman + Top-K + profit sim (see evaluation.py)

Usage:
    from experiments.ml_v2.pipeline import train_v2, predict_v2, load_models_v2

    models = train_v2(df_alerts, df_prices)
    signals = predict_v2(X, models, entry_prices)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ml_features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# XGBoost hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

_PARAMS_UP: dict = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,      # slightly more flexible than original suggestion
    gamma=0.05,              # less restrictive
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

_PARAMS_DOWN: dict = dict(
    n_estimators=500,
    max_depth=4,             # slightly deeper (suggested tweak)
    learning_rate=0.025,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=6,      # conservative but not over-constrained
    gamma=0.2,
    reg_alpha=0.2,
    reg_lambda=1.5,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

# ─────────────────────────────────────────────────────────────────────────────
# New features added in v2 (momentum, mean-reversion, volatility context)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLUMNS_V2: list[str] = FEATURE_COLUMNS + [
    # Momentum
    "return_5d",
    "return_10d",
    "return_20d",
    # Mean reversion
    "zscore_20d",
    "distance_from_ma50",
    # Volatility context
    "volatility_20d",
    # Market regime (VIX already in FEATURE_COLUMNS; sp500_trend is new)
    "sp500_trend",
]


# ─────────────────────────────────────────────────────────────────────────────
# Target construction (anti-leakage, sequence-aware)
# ─────────────────────────────────────────────────────────────────────────────

def build_targets(
    alert_date: pd.Timestamp,
    entry_price: float,
    future_prices: pd.Series,
    horizon_days: int = 60,
) -> dict[str, float]:
    """
    Compute max_return_60d and max_drawdown_60d for a single alert.

    Sequence-aware logic (buy-the-dip order):
      1. Find the minimum price in the window  (deepest drawdown point)
      2. max_return is measured from that minimum onward  (recovery upside)
      3. max_drawdown is entry → minimum  (worst case before it recovers)

    This correctly models "stock drops further, THEN recovers" rather than
    measuring upside from a peak that happened before the drawdown.

    Anti-leakage:
      - future_prices must contain ONLY data AFTER alert_date.
      - Caller is responsible for slicing correctly.

    Parameters
    ----------
    alert_date    : pd.Timestamp  (used only for logging)
    entry_price   : float          price at alert time
    future_prices : pd.Series      daily close prices for the next horizon_days
                                   (must NOT include alert_date itself)
    horizon_days  : int            number of calendar days to look forward

    Returns
    -------
    dict with keys:
        max_return_60d    float  upside from deepest point (positive)
        max_drawdown_60d  float  drop from entry to deepest point (negative)
        min_idx           int    index of the deepest point in future_prices
    """
    if len(future_prices) == 0 or entry_price <= 0:
        logger.warning(f"build_targets: empty window or zero entry at {alert_date}")
        return {"max_return_60d": np.nan, "max_drawdown_60d": np.nan, "min_idx": -1}

    prices = future_prices.values.astype(float)

    # Step 1: find deepest point
    min_idx = int(np.argmin(prices))
    min_price = prices[min_idx]

    # Step 2: max return from the deepest point onward (sequence-aware)
    prices_after_min = prices[min_idx:]
    max_price_after_min = float(np.max(prices_after_min))

    max_return_60d = (max_price_after_min / min_price) - 1.0
    max_drawdown_60d = (min_price / entry_price) - 1.0

    # Step 3: clip extremes (removes data errors / delistings / splits)
    max_return_60d   = float(np.clip(max_return_60d,   -0.5,  1.0))
    max_drawdown_60d = float(np.clip(max_drawdown_60d, -0.5,  0.0))

    logger.debug(
        f"build_targets [{alert_date.date()}]: "
        f"entry={entry_price:.2f} min_idx={min_idx} "
        f"drawdown={max_drawdown_60d:.3f} upside={max_return_60d:.3f}"
    )

    return {
        "max_return_60d":   max_return_60d,
        "max_drawdown_60d": max_drawdown_60d,
        "min_idx":          min_idx,
    }


def build_v2_features(
    row: pd.Series,
    price_history: pd.DataFrame,
) -> dict[str, float]:
    """
    Compute the 7 additional v2 features from OHLCV price_history.

    price_history must contain ONLY data up to (and including) alert_date —
    no future data. Caller slices this correctly.

    Columns expected: Close (required), High, Low (for volatility).

    Returns a dict of the 7 new keys. Returns NaN-safe fallbacks if
    history is insufficient.
    """
    closes = price_history["Close"].dropna()

    def _ret(n: int) -> float:
        if len(closes) < n + 1:
            return 0.0
        return float((closes.iloc[-1] / closes.iloc[-n - 1]) - 1.0)

    return_5d  = _ret(5)
    return_10d = _ret(10)
    return_20d = _ret(20)

    # zscore_20d: how many std devs from 20d mean
    if len(closes) >= 20:
        mu   = float(closes.iloc[-20:].mean())
        sigma = float(closes.iloc[-20:].std())
        zscore_20d = float((closes.iloc[-1] - mu) / sigma) if sigma > 0 else 0.0
    else:
        zscore_20d = 0.0

    # distance_from_ma50: (price - MA50) / MA50
    if len(closes) >= 50:
        ma50 = float(closes.iloc[-50:].mean())
        distance_from_ma50 = float((closes.iloc[-1] - ma50) / ma50) if ma50 > 0 else 0.0
    else:
        distance_from_ma50 = 0.0

    # volatility_20d: std of daily returns over last 20 days
    if len(closes) >= 21:
        daily_rets = closes.iloc[-21:].pct_change().dropna()
        volatility_20d = float(daily_rets.std())
    else:
        volatility_20d = 0.02  # fallback: 2%

    # sp500_trend: 20d return of SPY (passed in as row metadata or 0)
    sp500_trend = float(row.get("sp500_return_20d", 0.0))

    return {
        "return_5d":           return_5d,
        "return_10d":          return_10d,
        "return_20d":          return_20d,
        "zscore_20d":          zscore_20d,
        "distance_from_ma50": distance_from_ma50,
        "volatility_20d":     volatility_20d,
        "sp500_trend":         sp500_trend,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Log transform / inverse
# ─────────────────────────────────────────────────────────────────────────────

def transform_targets(y_up: np.ndarray, y_down: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply log1p transform to stabilise heavy-tailed return distributions.

    upside  : y = log1p(max_return)          always >= 0 after clip
    downside: y = -log1p(abs(max_drawdown))  always <= 0 after clip
    """
    y_up_t   = np.log1p(np.clip(y_up, 0.0, None))
    y_down_t = -np.log1p(np.clip(np.abs(y_down), 0.0, None))
    return y_up_t, y_down_t


def inverse_transform(y_up_t: np.ndarray, y_down_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Inverse of transform_targets — back to original return space.
    """
    y_up   = np.expm1(np.clip(y_up_t, 0.0, None))
    y_down = -np.expm1(np.clip(-y_down_t, 0.0, None))
    return y_up, y_down


# ─────────────────────────────────────────────────────────────────────────────
# Model container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DipModelsV2:
    model_up:   XGBRegressor
    model_down: XGBRegressor
    feature_cols: list[str] = field(default_factory=lambda: FEATURE_COLUMNS_V2)
    n_train_samples: int = 0
    train_date: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_v2(
    X: pd.DataFrame,
    y_up: np.ndarray,
    y_down: np.ndarray,
    feature_cols: Optional[list[str]] = None,
) -> DipModelsV2:
    """
    Train two XGBoost regressors: one for upside, one for downside.

    Parameters
    ----------
    X         : pd.DataFrame  feature matrix (rows = alerts, cols = FEATURE_COLUMNS_V2)
    y_up      : np.ndarray    max_return_60d  (positive floats, already clipped)
    y_down    : np.ndarray    max_drawdown_60d (negative floats, already clipped)
    feature_cols : list[str]  override feature column list if needed

    Returns
    -------
    DipModelsV2 with both fitted models
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS_V2

    X_mat = X[feature_cols].values.astype(np.float32)

    # Apply log transform
    y_up_t, y_down_t = transform_targets(y_up, y_down)

    logger.info(f"train_v2: fitting upside model on {len(X_mat)} samples...")
    model_up = XGBRegressor(**_PARAMS_UP)
    model_up.fit(X_mat, y_up_t)

    logger.info("train_v2: fitting downside model...")
    model_down = XGBRegressor(**_PARAMS_DOWN)
    model_down.fit(X_mat, y_down_t)

    logger.info("train_v2: done.")

    from datetime import datetime
    return DipModelsV2(
        model_up=model_up,
        model_down=model_down,
        feature_cols=feature_cols,
        n_train_samples=len(X_mat),
        train_date=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_v2(
    X: pd.DataFrame,
    models: DipModelsV2,
    entry_prices: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Predict upside / downside / score / take-profit / stop-loss for each alert.

    Parameters
    ----------
    X             : pd.DataFrame   feature matrix (same columns as training)
    models        : DipModelsV2    fitted models
    entry_prices  : np.ndarray     current prices (used for TP/SL calculation)
                                   If None, TP/SL columns are omitted.

    Returns
    -------
    pd.DataFrame with columns:
        pred_up         predicted max_return_60d (original space)
        pred_down       predicted max_drawdown_60d (original space)
        score           risk/reward = pred_up / abs(pred_down)
        rejected        bool: True if fails minimum quality filters
        reject_reason   str: why rejected (or "" if not rejected)
        take_profit     entry * (1 + calibration_factor * pred_up)  [if entry_prices given]
        stop_loss       entry * (1 + 1.2 * pred_down)               [if entry_prices given]
    """
    X_mat = X[models.feature_cols].values.astype(np.float32)

    up_t   = models.model_up.predict(X_mat)
    down_t = models.model_down.predict(X_mat)

    pred_up, pred_down = inverse_transform(up_t, down_t)

    # Risk/reward score
    abs_down = np.abs(pred_down)
    score = np.where(abs_down > 0, pred_up / abs_down, 0.0)

    # Quality filters
    rejected     = (pred_up < 0.05) | (pred_down < -0.15)
    reject_up    = pred_up < 0.05
    reject_down  = pred_down < -0.15
    reject_reason = np.where(
        reject_up & reject_down, "low_upside+high_downside",
        np.where(reject_up, "low_upside",
        np.where(reject_down, "high_downside", ""))
    )

    result = pd.DataFrame({
        "pred_up":       pred_up,
        "pred_down":     pred_down,
        "score":         score,
        "rejected":      rejected,
        "reject_reason": reject_reason,
    }, index=X.index)

    if entry_prices is not None:
        ep = np.array(entry_prices, dtype=float)
        # tp_factor: 0.7 is conservative starting point;
        # calibrate empirically on validation set (see evaluation.py)
        result["take_profit"] = ep * (1.0 + 0.7 * pred_up)
        result["stop_loss"]   = ep * (1.0 + 1.2 * pred_down)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_models_v2(models: DipModelsV2, path: Path = MODELS_DIR) -> None:
    out = path / "dip_models_v2.pkl"
    with open(out, "wb") as f:
        pickle.dump(models, f)
    logger.info(f"save_models_v2: saved to {out}")


def load_models_v2(path: Path = MODELS_DIR) -> DipModelsV2:
    pkl = path / "dip_models_v2.pkl"
    with open(pkl, "rb") as f:
        models = pickle.load(f)
    logger.info(
        f"load_models_v2: loaded (trained {models.train_date}, "
        f"{models.n_train_samples} samples)"
    )
    return models
