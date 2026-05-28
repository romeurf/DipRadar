"""
scripts/train_core_model.py — Train the core_us ML model.

Reads core_training_base.parquet, trains a GradientBoostingRegressor to
predict alpha_1y (1-year excess return vs SPY at MA200w entry episodes).
Saves the model bundle to /data/core_us_model.pkl.

Usage:
    python scripts/train_core_model.py
    python scripts/train_core_model.py --parquet /path/to/core_training_base.parquet
    python scripts/train_core_model.py --dry-run   # evaluate only, do not save

Triggered via: /admin_train_core_model (bot command)
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import QuantileTransformer

# Add repo root to sys.path so we can import core_bundle.
_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))
from core_bundle import CoreModelBundle  # noqa: E402  pickle-compat shared dataclass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_core_model")

_REPO_ROOT   = Path(__file__).resolve().parent.parent
_DEFAULT_PARQUET = _REPO_ROOT / "core_training_base.parquet"
_DATA_DIR    = Path("/data") if Path("/data").exists() else _REPO_ROOT
_MODEL_PATH  = _DATA_DIR / "core_us_model.pkl"

# Full feature set — price/volume/sector (always available) + fundamentals (PIT via EDGAR)
# Fundamental columns will have NaN for pre-2009 episodes (before XBRL was mandatory);
# the model handles this via imputation (median fill).
CORE_FEATURE_COLS: list[str] = [
    # Price / MA signals
    "price_vs_ma200w",        # primary: how far below/above MA200 weekly
    "ma200d_trend_60d",       # secular trend: MA200d rising or falling
    "return_20d",             # dip depth (negative = deeper)
    "return_5d",              # recent momentum
    "dd_from_52w_high",       # distance from 52-week peak
    "rsi_14",                 # RSI 14-day
    # Volume / institutional flow
    "obv_trend_20d",          # OBV divergence (accumulation vs distribution)
    "up_down_vol_ratio_20d",  # up vs down volume last 20 days
    "high_vol_up_days_20d",   # block buy days (vol > 1.5x avg AND price up)
    "vol_spike_5d_vs_60d",    # recent volume spike ratio
    # Sector context
    "sector_etf_vs_ma200w",   # sector ETF vs its own MA200w
    "stock_vs_sector_spread", # stock dip vs sector dip (stock-specific signal)
    "sector_return_20d",      # sector momentum
    # User signal
    "annual_yield_pct",       # dividend yield (user-defined per ticker)
    # Fundamentals (PIT via SEC EDGAR XBRL — NaN for pre-2009, imputed with median)
    "revenue_growth",         # YoY revenue growth (compounder must grow)
    "gross_margin",           # gross margin ratio (pricing power / moat)
    "roe",                    # return on equity (capital allocation quality)
    "fcf_yield_fund",         # FCF yield (cash generation at entry price)
    # Relative valuation + dividend quality
    "pe_vs_sector_median",    # fwd PE / sector median: <1.0 = structural discount
    "dividend_growth_3y",     # 3y dividend CAGR: growing dividend = durable earnings
]

TARGET = "alpha_1y"  # 1-year excess return vs SPY

# CoreModelBundle is now defined in core_bundle.py (top-level module) so
# joblib.load can resolve __main__.CoreModelBundle regardless of which module
# is __main__ at load time (train_core_model in subprocess vs main.py in bot).


def _load_dataset(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    log.info(f"Loaded {len(df)} rows from {parquet_path}")

    # Drop rows without target
    before = len(df)
    df = df.dropna(subset=[TARGET])
    log.info(f"After dropping rows without {TARGET}: {len(df)} (dropped {before - len(df)})")

    # Drop rows where entries are from the last 12 months (future data unavailable)
    cutoff = pd.Timestamp.now() - pd.DateOffset(months=13)
    df["alert_date"] = pd.to_datetime(df["alert_date"])
    before = len(df)
    df = df[df["alert_date"] < cutoff]
    log.info(f"After excluding last-13-month entries (no outcome yet): {len(df)} (dropped {before - len(df)})")

    # Sort chronologically for time-series CV
    df = df.sort_values("alert_date").reset_index(drop=True)
    return df


def _build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (X, y, impute_medians).

    impute_medians: median per feature column, used at inference time to fill
    NaN for fundamental columns not available in live scan.
    """
    missing = [c for c in CORE_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in training data: {missing}")

    feat_df = df[CORE_FEATURE_COLS].copy()

    # Impute NaN with column median (fundamental columns are NaN for pre-XBRL episodes)
    impute_medians: dict[str, float] = {}
    for col in CORE_FEATURE_COLS:
        median_val = float(feat_df[col].median())
        if np.isnan(median_val):
            median_val = 0.0
        impute_medians[col] = median_val
        nan_count = feat_df[col].isna().sum()
        if nan_count > 0:
            log.info(f"  Imputing {nan_count} NaN in '{col}' with median={median_val:.4f}")
            feat_df[col] = feat_df[col].fillna(median_val)

    X = feat_df.values.astype(float)
    y = df[TARGET].values.astype(float)

    # Winsorise target at 1st/99th percentile
    p1, p99 = np.percentile(y, [1, 99])
    y = np.clip(y, p1, p99)
    log.info(
        f"Features: {X.shape[1]} cols, {X.shape[0]} rows. "
        f"Target {TARGET}: mean={y.mean():.3f} std={y.std():.3f} range=[{y.min():.3f}, {y.max():.3f}]"
    )
    return X, y, impute_medians


def _cross_validate(X: np.ndarray, y: np.ndarray, model_params: dict) -> tuple[float, float]:
    """Time-series cross-validation. Returns (mean_rmse, mean_r2)."""
    tscv = TimeSeriesSplit(n_splits=5)
    rmses, r2s = [], []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        m = GradientBoostingRegressor(**model_params, random_state=42)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        r2 = m.score(X_val, y_val)
        rmses.append(rmse)
        r2s.append(r2)
        log.info(f"  Fold {fold}: RMSE={rmse:.4f} R²={r2:.3f} (n_val={len(y_val)})")
    return float(np.mean(rmses)), float(np.mean(r2s))


def train(parquet_path: Path, dry_run: bool = False) -> CoreModelBundle:
    df = _load_dataset(parquet_path)

    if len(df) < 100:
        raise ValueError(
            f"Only {len(df)} labelled rows — need at least 100. "
            "Run build_core_dataset.py --force-rebuild to regenerate training data."
        )

    X, y, impute_medians = _build_features(df)

    model_params = {
        "n_estimators":    200,
        "max_depth":       3,
        "learning_rate":   0.05,
        "subsample":       0.8,
        "min_samples_leaf": max(5, len(X) // 50),
    }

    log.info(f"Cross-validating with TimeSeriesSplit(n_splits=5) on {len(X)} rows...")
    cv_rmse, cv_r2 = _cross_validate(X, y, model_params)
    log.info(f"CV result: RMSE={cv_rmse:.4f} R²={cv_r2:.3f}")
    if cv_r2 < 0.05:
        log.warning(
            f"CV R²={cv_r2:.3f} is below 0.05 — model predictions don't beat the mean. "
            "Bundle is still saved for diagnostics, but core_radar will fall back to "
            "rule-based scoring (see core_radar._MIN_CV_R2_FOR_USE)."
        )

    if dry_run:
        log.info("Dry run — model NOT saved.")
        scaler = QuantileTransformer(output_distribution="normal", random_state=42)
        scaler.fit(X)
        m = GradientBoostingRegressor(**model_params, random_state=42).fit(X, y)
        preds = m.predict(X)
        return CoreModelBundle(
            model=m, feature_cols=CORE_FEATURE_COLS, scaler=scaler,
            impute_medians=impute_medians,
            train_date=datetime.now().isoformat()[:10],
            n_train=len(X), cv_rmse=cv_rmse, cv_r2=cv_r2,
            alpha_p10=float(np.percentile(preds, 10)),
            alpha_p50=float(np.percentile(preds, 50)),
            alpha_p90=float(np.percentile(preds, 90)),
        )

    # Full training on all data
    log.info(f"Training final model on all {len(X)} rows...")
    scaler = QuantileTransformer(output_distribution="normal", random_state=42)
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingRegressor(**model_params, random_state=42)
    model.fit(X_scaled, y)

    preds = model.predict(X_scaled)
    log.info(f"Training RMSE: {np.sqrt(mean_squared_error(y, preds)):.4f}")

    importances = sorted(
        zip(CORE_FEATURE_COLS, model.feature_importances_),
        key=lambda x: -x[1],
    )
    log.info("Feature importances:")
    for feat, imp in importances:
        log.info(f"  {feat:35s} {imp:.4f}  {'█' * int(imp * 40)}")

    bundle = CoreModelBundle(
        model=model,
        feature_cols=CORE_FEATURE_COLS,
        scaler=scaler,
        impute_medians=impute_medians,
        train_date=datetime.now().isoformat()[:10],
        n_train=len(X),
        cv_rmse=cv_rmse,
        cv_r2=cv_r2,
        alpha_p10=float(np.percentile(preds, 10)),
        alpha_p50=float(np.percentile(preds, 50)),
        alpha_p90=float(np.percentile(preds, 90)),
    )

    joblib.dump(bundle, _MODEL_PATH)
    log.info(f"Model saved to {_MODEL_PATH}")
    log.info(
        f"Bundle: n_train={len(X)} CV_RMSE={cv_rmse:.4f} CV_R²={cv_r2:.3f} "
        f"alpha_p50={bundle.alpha_p50:.3f} alpha_p90={bundle.alpha_p90:.3f}"
    )
    return bundle


def main() -> None:
    p = argparse.ArgumentParser(description="Train core_us ML model")
    p.add_argument("--parquet", default=str(_DEFAULT_PARQUET), help="Path to core_training_base.parquet")
    p.add_argument("--dry-run", action="store_true", help="Evaluate only, do not save model")
    args = p.parse_args()

    bundle = train(Path(args.parquet), dry_run=args.dry_run)
    print(
        f"\n{'DRY RUN — ' if args.dry_run else ''}core_us model trained:\n"
        f"  n_train   = {bundle.n_train}\n"
        f"  CV RMSE   = {bundle.cv_rmse:.4f}\n"
        f"  CV R²     = {bundle.cv_r2:.3f}\n"
        f"  alpha p50 = {bundle.alpha_p50:.3f}  (median predicted 1y alpha)\n"
        f"  alpha p90 = {bundle.alpha_p90:.3f}  (top-decile predicted 1y alpha)\n"
        + ("" if args.dry_run else f"  saved to  = {_MODEL_PATH}")
    )


if __name__ == "__main__":
    main()
