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


from sklearn.linear_model import Ridge
from scipy.stats import spearmanr


def _purged_block_cv_indices(
    dates: pd.Series,
    n_splits: int = 5,
    embargo_days: int = 30,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged blocked CV for time-series.

    TimeSeriesSplit has two problems for our setting:
      1. Early folds train on tiny samples (~16% of data), giving unstable
         fold-level IC.
      2. There's no embargo between train and val, so a train row whose
         alert_date is May with a 1-year target can have its target
         realised on a date inside the val fold — leakage.

    BlockedCV partitions chronologically into `n_splits` contiguous folds.
    For each fold, that block is the test set; everything else is train,
    with a purge window of `embargo_days` removed around the fold boundary
    so 1-year-horizon labels don't bleed in.
    """
    dates = pd.to_datetime(dates).reset_index(drop=True)
    n = len(dates)
    if n_splits < 2 or n < 2 * n_splits:
        return [(np.arange(n // 2), np.arange(n // 2, n))]

    fold_size = n // n_splits
    embargo = pd.Timedelta(days=embargo_days)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        val_lo = k * fold_size
        val_hi = (k + 1) * fold_size if k < n_splits - 1 else n
        val_dates_lo = dates.iloc[val_lo]
        val_dates_hi = dates.iloc[val_hi - 1]

        val_idx = np.arange(val_lo, val_hi)
        # Train = everything outside the val block AND outside the embargo zone
        train_mask = np.ones(n, dtype=bool)
        train_mask[val_lo:val_hi] = False
        # Embargo: drop train rows whose alert_date falls within `embargo` of
        # the val block boundaries on either side. Forward-only purging would
        # suffice if val came AFTER train, but blocked CV puts val in the
        # middle for k>0; symmetric purge is simpler and equally correct.
        in_embargo = (
            (dates >= val_dates_lo - embargo) &
            (dates <= val_dates_hi + embargo)
        )
        train_mask[in_embargo.values] = False
        train_idx = np.where(train_mask)[0]
        if len(train_idx) < 100:
            log.warning(f"  fold {k+1}: only {len(train_idx)} train rows — skipping")
            continue
        folds.append((train_idx, val_idx))
    return folds


def _fit_score(
    model_factory,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    """Fit a fresh model on (X_tr,y_tr); predict on val; return (ic, rmse, r2, preds)."""
    m = model_factory()
    m.fit(X_tr, y_tr)
    preds = m.predict(X_val)
    rho, _ = spearmanr(preds, y_val)
    ic = float(rho) if rho is not None and not np.isnan(rho) else 0.0
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    r2 = float(m.score(X_val, y_val)) if hasattr(m, "score") else 0.0
    return ic, rmse, r2, preds


def _cross_validate_with_ic(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.Series,
    model_factory,
    label: str,
) -> tuple[float, float, float]:
    """Purged blocked CV, returning (mean_ic, mean_rmse, mean_r2)."""
    folds = _purged_block_cv_indices(dates, n_splits=5, embargo_days=30)
    ics, rmses, r2s = [], [], []
    for fold_i, (tr, va) in enumerate(folds, 1):
        ic, rmse, r2, _ = _fit_score(model_factory, X[tr], y[tr], X[va], y[va])
        ics.append(ic); rmses.append(rmse); r2s.append(r2)
        log.info(
            f"  [{label}] fold {fold_i}: IC={ic:+.4f} RMSE={rmse:.4f} R²={r2:+.3f} "
            f"(n_tr={len(tr)} n_va={len(va)})"
        )
    if not ics:
        return 0.0, 0.0, 0.0
    return float(np.mean(ics)), float(np.mean(rmses)), float(np.mean(r2s))


def train(parquet_path: Path, dry_run: bool = False) -> CoreModelBundle:
    df = _load_dataset(parquet_path)

    if len(df) < 100:
        raise ValueError(
            f"Only {len(df)} labelled rows — need at least 100. "
            "Run build_core_dataset.py --force-rebuild to regenerate training data."
        )

    X, y, impute_medians = _build_features(df)

    # Scale features once for both models — Ridge needs it, GBR is invariant.
    scaler = QuantileTransformer(output_distribution="normal", random_state=42)
    X_scaled = scaler.fit_transform(X)

    gbr_params = {
        "n_estimators":    200,
        "max_depth":       3,
        "learning_rate":   0.05,
        "subsample":       0.8,
        "min_samples_leaf": max(5, len(X) // 50),
    }

    def _gbr_factory():
        return GradientBoostingRegressor(**gbr_params, random_state=42)

    def _ridge_factory():
        return Ridge(alpha=10.0, random_state=42)

    log.info(f"Purged-blocked CV (5 folds, 30d embargo) on {len(X)} rows...")
    log.info("---- GBR ----")
    gbr_ic, gbr_rmse, gbr_r2 = _cross_validate_with_ic(
        X_scaled, y, df["alert_date"], _gbr_factory, "GBR")
    log.info(f"GBR CV mean: IC={gbr_ic:+.4f} RMSE={gbr_rmse:.4f} R²={gbr_r2:+.3f}")
    log.info("---- Ridge ----")
    ridge_ic, ridge_rmse, ridge_r2 = _cross_validate_with_ic(
        X_scaled, y, df["alert_date"], _ridge_factory, "Ridge")
    log.info(f"Ridge CV mean: IC={ridge_ic:+.4f} RMSE={ridge_rmse:.4f} R²={ridge_r2:+.3f}")

    # Champion = highest IC (Spearman). IC is more robust to alpha_1y's heavy
    # tails than R² and is what production sizing would actually use.
    if gbr_ic >= ridge_ic:
        champion_name = "GBR"
        final_factory = _gbr_factory
        cv_ic, cv_rmse, cv_r2 = gbr_ic, gbr_rmse, gbr_r2
    else:
        champion_name = "Ridge"
        final_factory = _ridge_factory
        cv_ic, cv_rmse, cv_r2 = ridge_ic, ridge_rmse, ridge_r2
    log.info(f"Champion: {champion_name} | CV IC={cv_ic:+.4f} RMSE={cv_rmse:.4f} R²={cv_r2:+.3f}")

    if cv_ic < 0.04:
        log.warning(
            f"Champion IC={cv_ic:.4f} below 0.04 — predictions barely better than random. "
            "Bundle still saved for diagnostics; core_radar will fall back to rule-based "
            "scoring (see core_radar._MIN_CV_IC_FOR_USE)."
        )

    if dry_run:
        log.info("Dry run — model NOT saved.")
        m = final_factory().fit(X_scaled, y)
        preds = m.predict(X_scaled)
        return CoreModelBundle(
            model=m, feature_cols=CORE_FEATURE_COLS, scaler=scaler,
            impute_medians=impute_medians,
            train_date=datetime.now().isoformat()[:10],
            n_train=len(X), cv_rmse=cv_rmse, cv_r2=cv_r2,
            alpha_p10=float(np.percentile(preds, 10)),
            alpha_p50=float(np.percentile(preds, 50)),
            alpha_p90=float(np.percentile(preds, 90)),
            cv_ic=cv_ic, champion_name=champion_name,
        )

    log.info(f"Training final {champion_name} on all {len(X)} rows...")
    model = final_factory()
    model.fit(X_scaled, y)
    preds = model.predict(X_scaled)
    log.info(f"Training RMSE: {np.sqrt(mean_squared_error(y, preds)):.4f}")

    if hasattr(model, "feature_importances_"):
        importances = sorted(
            zip(CORE_FEATURE_COLS, model.feature_importances_),
            key=lambda x: -x[1],
        )
        log.info("Feature importances:")
        for feat, imp in importances:
            log.info(f"  {feat:35s} {imp:.4f}  {'█' * int(imp * 40)}")
    elif hasattr(model, "coef_"):
        coefs = np.abs(model.coef_)
        total = coefs.sum() or 1.0
        rel = coefs / total
        ordered = sorted(zip(CORE_FEATURE_COLS, rel), key=lambda x: -x[1])
        log.info("Feature |coef| (normalised):")
        for feat, c in ordered:
            log.info(f"  {feat:35s} {c:.4f}  {'█' * int(c * 40)}")

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
        cv_ic=cv_ic,
        champion_name=champion_name,
    )

    joblib.dump(bundle, _MODEL_PATH)
    log.info(f"Model saved to {_MODEL_PATH}")
    log.info(
        f"Bundle: champion={champion_name} n_train={len(X)} "
        f"CV_IC={cv_ic:+.4f} CV_RMSE={cv_rmse:.4f} CV_R²={cv_r2:+.3f} "
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
        f"  champion  = {bundle.champion_name}\n"
        f"  n_train   = {bundle.n_train}\n"
        f"  CV IC     = {bundle.cv_ic:+.4f}  (Spearman; >0.04 = used; >0.06 = good)\n"
        f"  CV RMSE   = {bundle.cv_rmse:.4f}\n"
        f"  CV R²     = {bundle.cv_r2:+.3f}\n"
        f"  alpha p50 = {bundle.alpha_p50:.3f}  (median predicted 1y alpha)\n"
        f"  alpha p90 = {bundle.alpha_p90:.3f}  (top-decile predicted 1y alpha)\n"
        + ("" if args.dry_run else f"  saved to  = {_MODEL_PATH}")
    )


if __name__ == "__main__":
    main()
