"""calibrate_dip_alpha.py — Bucket calibration of DIP model predictions vs actuals.

The DIP /cash queue annualizes per-alert α as `pred_alpha_90d × (365/90)`.
When the model is over-optimistic (typical for ML trained on a mostly-bull
window), this surfaces inflated annualized α and makes every DIP look the
same. Calibration learns the historical mapping `pred → actual` and
applies it before annualization.

V1 design (this script):
  - Load ml_training_base.parquet (~28k samples with `alpha_90d` target).
  - Run the bundled DIP model on the parquet to get `pred_alpha_90d` per row.
  - Bucket predictions into N quantile bins (default 7).
  - For each bucket: mean of actual `alpha_90d`, sample count.
  - Persist as data/dip_alpha_calibration.json.
  - Note: in-sample (overfit) — V2 will refresh from mature alert_db
    samples to gradually replace synthetic with real observations.

/cash usage (separate change):
  pred = ml_result.pred_up
  bucket = lookup_bucket(pred, calibration)
  calibrated_alpha = bucket["mean_actual_alpha_90d"]
  annualized = calibrated_alpha × (365 / 90)
  → replaces the previous raw pred × (365/90)

Usage:
    python scripts/calibrate_dip_alpha.py
    python scripts/calibrate_dip_alpha.py --buckets 10
    python scripts/calibrate_dip_alpha.py --min-samples-per-bucket 50
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("calibrate_dip")


_OUTPUT_PATH = _ROOT / "data" / "dip_alpha_calibration.json"
_PARQUET_PATH = _ROOT / "ml_training_base.parquet"
_BUNDLE_PATH  = _ROOT / "ml_training" / "dip_models.pkl"

_MIN_R_SQUARED_ACCEPT = 0.10   # buckets accepted if alpha rises monotonically


def _ensure_data_dir() -> None:
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_parquet() -> "pd.DataFrame":
    import pandas as pd
    if not _PARQUET_PATH.exists():
        log.error(f"Parquet not found at {_PARQUET_PATH}")
        sys.exit(1)
    df = pd.read_parquet(_PARQUET_PATH)
    log.info(f"Loaded parquet: {df.shape[0]} rows × {df.shape[1]} columns")
    if "alpha_90d" not in df.columns:
        log.error("Parquet missing 'alpha_90d' target column — cannot calibrate")
        sys.exit(1)
    return df


def _load_bundle():
    """Load the DIP model bundle. Tries the canonical paths."""
    import joblib
    if not _BUNDLE_PATH.exists():
        log.error(f"DIP bundle not found at {_BUNDLE_PATH}")
        sys.exit(1)
    # Ensure DipModels class is on __main__ for unpickler
    try:
        from ml.ml_predictor import DipModels  # noqa: F401
        import sys as _sys
        if "__main__" in _sys.modules and not hasattr(_sys.modules["__main__"], "DipModels"):
            _sys.modules["__main__"].DipModels = DipModels
    except Exception:
        pass
    bundle = joblib.load(_BUNDLE_PATH)
    log.info(f"Loaded bundle: {type(bundle).__name__}")
    return bundle


def _predict_on_parquet(df, bundle):
    """Run the bundle's model_up on all parquet rows. Returns numpy array of
    raw predictions (before inverse-transform)."""
    from ml.ml_predictor import _inverse_transform_up

    feature_cols = getattr(bundle, "feature_cols", None) or list(df.columns)
    feature_cols = [c for c in feature_cols if c in df.columns]
    log.info(f"Using {len(feature_cols)} features for prediction")

    X = df[feature_cols].fillna(0.0).values  # crude impute — V2 uses bundle medians
    model_up = getattr(bundle, "model_up", None) or bundle["model_up"]

    log.info(f"Predicting on {len(X)} rows...")
    raw_preds = model_up.predict(X)
    preds = np.array([_inverse_transform_up(float(p)) for p in raw_preds])
    log.info(f"Predictions: mean={preds.mean():.4f} std={preds.std():.4f} "
             f"range=[{preds.min():.4f}, {preds.max():.4f}]")
    return preds


def _bucket_calibration(
    pred: np.ndarray,
    actual: np.ndarray,
    n_buckets: int = 7,
    min_samples_per_bucket: int = 30,
) -> dict:
    """Compute per-bucket mean(actual) sorted by pred.

    Uses quantile bin boundaries so each bucket has roughly equal sample count.
    Returns the calibration dict ready to persist.
    """
    mask = np.isfinite(pred) & np.isfinite(actual)
    pred   = pred[mask]
    actual = actual[mask]
    log.info(f"Calibrating on {len(pred)} valid (pred, actual) pairs")

    # Quantile bin edges
    quantiles = np.linspace(0.0, 1.0, n_buckets + 1)
    edges = np.quantile(pred, quantiles)
    # Ensure strictly increasing (collapse degenerate bins)
    edges = np.unique(edges)
    if len(edges) - 1 < 3:
        log.warning("Too few unique quantile bins — predictions may be degenerate")

    buckets = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        # Last bucket inclusive on the right
        if i == len(edges) - 2:
            in_bucket = (pred >= lo) & (pred <= hi)
        else:
            in_bucket = (pred >= lo) & (pred < hi)
        n = int(in_bucket.sum())
        if n < min_samples_per_bucket:
            log.warning(f"Bucket [{lo:.4f}, {hi:.4f}] has only {n} samples — including anyway")
        mean_actual = float(actual[in_bucket].mean()) if n > 0 else float("nan")
        median_actual = float(np.median(actual[in_bucket])) if n > 0 else float("nan")
        buckets.append({
            "pred_min":         round(float(lo), 6),
            "pred_max":         round(float(hi), 6),
            "mean_actual_alpha_90d":   round(mean_actual, 6),
            "median_actual_alpha_90d": round(median_actual, 6),
            "n_samples":        n,
            "n_real":           0,             # V2 will track real-alert count
            "n_synthetic":      n,             # V1 all-synthetic
        })

    # Check monotonicity (higher pred should map to higher actual)
    means = [b["mean_actual_alpha_90d"] for b in buckets if not np.isnan(b["mean_actual_alpha_90d"])]
    is_monotonic = all(means[i] <= means[i+1] + 0.01 for i in range(len(means)-1))
    log.info(f"Buckets monotonic: {is_monotonic}")

    # Compute pseudo-R² as overall variance explained by bucket means
    overall_mean = float(actual.mean())
    bucket_pred_by_sample = np.zeros_like(actual)
    for b in buckets:
        in_b = (pred >= b["pred_min"]) & (pred <= b["pred_max"])
        bucket_pred_by_sample[in_b] = b["mean_actual_alpha_90d"]
    ss_res = float(((actual - bucket_pred_by_sample) ** 2).sum())
    ss_tot = float(((actual - overall_mean) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

    return {
        "method":             "buckets_by_prediction",
        "buckets":            buckets,
        "r_squared":          round(r2, 4),
        "is_monotonic":       is_monotonic,
        "accepted":           r2 >= _MIN_R_SQUARED_ACCEPT and is_monotonic,
        "n_samples_total":    int(len(pred)),
        "n_real":             0,
        "n_synthetic":        int(len(pred)),
        "version":            "v1_parquet_in_sample",
        "calibrated_on":      date.today().isoformat(),
        "min_r_squared_accept": _MIN_R_SQUARED_ACCEPT,
        "parquet_source":     str(_PARQUET_PATH.name),
        "warning":            (
            "V1 calibration uses IN-SAMPLE predictions on the training parquet — "
            "model is overfit to these rows, so r_squared is optimistic. V2 will "
            "incorporate mature alerts from alert_db.csv with declining synthetic "
            "weight as real samples accumulate. Refresh weekly via "
            "scripts/refresh_dip_alpha_calibration.py (next iteration)."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--buckets", type=int, default=7,
                        help="Number of quantile buckets (default: 7)")
    parser.add_argument("--min-samples-per-bucket", type=int, default=30,
                        help="Warn if a bucket has fewer than this many samples")
    args = parser.parse_args()

    _ensure_data_dir()
    df     = _load_parquet()
    bundle = _load_bundle()

    pred   = _predict_on_parquet(df, bundle)
    actual = df["alpha_90d"].fillna(0.0).values

    calib = _bucket_calibration(
        pred, actual,
        n_buckets=args.buckets,
        min_samples_per_bucket=args.min_samples_per_bucket,
    )

    log.info("=" * 60)
    log.info(f"Calibration result: r²={calib['r_squared']:.4f}  "
             f"monotonic={calib['is_monotonic']}  accepted={calib['accepted']}")
    log.info("Buckets (pred range → mean actual alpha_90d):")
    for b in calib["buckets"]:
        log.info(f"  [{b['pred_min']:+.4f}, {b['pred_max']:+.4f}] → "
                 f"actual mean={b['mean_actual_alpha_90d']:+.4f} "
                 f"(median={b['median_actual_alpha_90d']:+.4f}) "
                 f"n={b['n_samples']}")
    log.info("=" * 60)

    with _OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)
    log.info(f"Persisted to {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
