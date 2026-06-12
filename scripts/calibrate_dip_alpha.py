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


# Output: prefer Railway volume /data/ (persists across deploys), fallback
# to /app/data/ for local dev. The /data/ vs /app/data split is the same
# pattern state.py uses for its JSON files.
_DATA_DIR     = Path("/data") if Path("/data").exists() else (_ROOT / "data")
_OUTPUT_PATH  = _DATA_DIR / "dip_alpha_calibration.json"
_PARQUET_PATH = _ROOT / "ml_training_base.parquet"
_BUNDLE_PATH  = _ROOT / "ml_training" / "dip_models.pkl"

_MIN_R_SQUARED_ACCEPT = 0.10   # one acceptance path: high explained variance
_MIN_SPREAD_FOR_ACCEPT = 0.04  # second acceptance path: monotonic AND ≥4pp spread
                               # between top and bottom bucket means


# Map target column to the TRAINING horizon in days (what the model actually
# learned to predict). The DISPLAY horizon (what /cash shows the user) is
# always 90d — user feedback: '60d pode ser pouco dar hold'. When the model
# trained on a shorter horizon, we extrapolate linearly: 90d_alpha = X_alpha
# × (90 / training_horizon). This assumes alpha grows linearly over the
# extrapolation window — fine for the typical 60→90d gap; less defensible
# for longer extrapolations.
_HORIZON_BY_COLUMN = {
    "alpha_90d":           90,
    "alpha_90d_rank":      90,
    "forward_return_90d":  90,
    "return_90d":          90,
    "alpha_60d":           60,
    "alpha_60d_rank":      60,
}
_DISPLAY_HORIZON_DAYS = 90


def _ensure_data_dir() -> None:
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


_TARGET_CANDIDATES = (
    "alpha_90d",            # canonical (build_dip_dataset.py current schema)
    "alpha_90d_rank",       # used internally by calibration.py
    "forward_return_90d",   # raw return without SPY subtraction
    "return_90d",           # alias
    "alpha_60d",            # fallback horizon
    "alpha_60d_rank",
)


def _load_parquet() -> tuple["pd.DataFrame", str]:
    """Load parquet + resolve which column is the target.

    The repo schema is meant to write alpha_90d but older parquets / renamed
    schemas might have a different column. Try a sequence of candidates and
    return the dataframe + the resolved target name. If none match, dump all
    columns so the user can see what the actual target is called.
    """
    import pandas as pd
    if not _PARQUET_PATH.exists():
        log.error(f"Parquet not found at {_PARQUET_PATH}")
        sys.exit(1)
    df = pd.read_parquet(_PARQUET_PATH)
    log.info(f"Loaded parquet: {df.shape[0]} rows × {df.shape[1]} columns")

    for candidate in _TARGET_CANDIDATES:
        if candidate in df.columns:
            valid = df[candidate].notna().sum()
            log.info(f"Target column resolved: '{candidate}' ({valid}/{len(df)} non-null)")
            if valid < 100:
                log.warning(f"Target '{candidate}' has only {valid} non-null values — trying next")
                continue
            return df, candidate

    # None matched — diagnostic dump
    log.error("Parquet has no recognized α target column. Tried: " + ", ".join(_TARGET_CANDIDATES))
    log.error("Full column list (sorted):")
    for c in sorted(df.columns.tolist()):
        log.error(f"  {c}")
    log.error("Pick the right column name and add it to _TARGET_CANDIDATES at the top of this script.")
    sys.exit(1)


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
    target_col: str = "alpha_90d",
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

    horizon_days = _HORIZON_BY_COLUMN.get(target_col, 90)
    # Linear extrapolation factor: training_horizon → display_horizon.
    # When training is 60d and display is 90d → factor 1.5.
    # When training already matches display → factor 1.0 (no-op).
    _scale = _DISPLAY_HORIZON_DAYS / float(horizon_days)
    log.info(f"Horizon scale: training={horizon_days}d → display={_DISPLAY_HORIZON_DAYS}d "
             f"(factor ×{_scale:.3f})")

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
        # Compute mean + percentiles at the training horizon, then scale
        # everything linearly to the display horizon. Percentiles are what
        # power the Kelly Criterion downside estimate — we look at the 5th
        # percentile of actual outcomes to get a realistic "bad case" tail
        # without resorting to a hand-tuned downside constant.
        if n > 0:
            samples_in    = actual[in_bucket]
            mean_train    = float(samples_in.mean())
            median_train  = float(np.median(samples_in))
            q05_train     = float(np.quantile(samples_in, 0.05))
            q25_train     = float(np.quantile(samples_in, 0.25))
            std_train     = float(samples_in.std())
        else:
            mean_train = median_train = q05_train = q25_train = std_train = float("nan")

        # Linear scale to display horizon
        mean_display   = mean_train  * _scale
        median_display = median_train * _scale
        q05_display    = q05_train * _scale
        q25_display    = q25_train * _scale
        std_display    = std_train * _scale

        # Downside estimate for Kelly: gap between mean and 5th percentile
        # in the DISPLAY horizon. abs() to handle negative-mean buckets.
        downside_q05 = abs(mean_display - q05_display) if not (np.isnan(mean_display) or np.isnan(q05_display)) else float("nan")

        buckets.append({
            "pred_min":                       round(float(lo), 6),
            "pred_max":                       round(float(hi), 6),
            # Display horizon (90d) — what /cash actually consumes
            "mean_actual_alpha_90d":          round(mean_display, 6),
            "median_actual_alpha_90d":        round(median_display, 6),
            "q05_alpha_90d":                  round(q05_display, 6),
            "q25_alpha_90d":                  round(q25_display, 6),
            "std_alpha_90d":                  round(std_display, 6),
            # Downside for Kelly Criterion: gap mean → 5th percentile (tail)
            "downside_q05_90d":               round(downside_q05, 6),
            # Training horizon (raw, for audit) — preserves what the model saw
            "mean_actual_alpha_training":     round(mean_train, 6),
            "median_actual_alpha_training":   round(median_train, 6),
            "q05_alpha_training":             round(q05_train, 6),
            "n_samples":                      n,
            "n_real":                         0,    # V2 will track real-alert count
            "n_synthetic":                    n,    # V1 all-synthetic
        })

    # Check monotonicity using the display (90d) values — same shape as
    # training values since extrapolation is linear, just different scale.
    means = [b["mean_actual_alpha_90d"] for b in buckets if not np.isnan(b["mean_actual_alpha_90d"])]
    is_monotonic = all(means[i] <= means[i+1] + 0.01 for i in range(len(means)-1))
    log.info(f"Buckets monotonic: {is_monotonic}")

    # Spread between top and bottom bucket means — a separate signal of
    # discrimination quality. With huge within-bucket variance (5k+ samples),
    # r² can be low even when the model genuinely discriminates between low-
    # and high-α regimes. Accepting on monotonic+spread covers this case.
    spread = max(means) - min(means) if len(means) >= 2 else 0.0
    log.info(f"Spread top-bottom buckets: {spread:.4f}")

    # Compute pseudo-R² as variance explained by bucket means. Use the
    # TRAINING-horizon means (raw values that match `actual`'s scale) — the
    # 90d-extrapolated values are just `× _scale` of the training values
    # and would give an artificially deflated r² if compared to 60d actuals.
    overall_mean = float(actual.mean())
    bucket_pred_by_sample = np.zeros_like(actual)
    for b in buckets:
        in_b = (pred >= b["pred_min"]) & (pred <= b["pred_max"])
        bucket_pred_by_sample[in_b] = b["mean_actual_alpha_training"]
    ss_res = float(((actual - bucket_pred_by_sample) ** 2).sum())
    ss_tot = float(((actual - overall_mean) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

    # Acceptance: either path qualifies
    # (a) classic: r² ≥ MIN_R_SQUARED + monotonic
    # (b) low-r² OK: monotonic + meaningful spread (≥ MIN_SPREAD)
    accepted_by_r2 = (r2 >= _MIN_R_SQUARED_ACCEPT) and is_monotonic
    accepted_by_spread = is_monotonic and (spread >= _MIN_SPREAD_FOR_ACCEPT)
    accepted = accepted_by_r2 or accepted_by_spread
    accept_reason = (
        "r²+monotonic" if accepted_by_r2
        else ("monotonic+spread" if accepted_by_spread else "rejected")
    )

    log.info(f"Training horizon: {horizon_days}d (column={target_col})")
    log.info(f"Display horizon (after extrap): {_DISPLAY_HORIZON_DAYS}d")

    return {
        "method":             "buckets_by_prediction",
        "buckets":            buckets,
        "r_squared":          round(r2, 4),
        "spread_top_bottom":  round(spread, 4),
        "is_monotonic":       is_monotonic,
        "accepted":           accepted,
        "accepted_by":        accept_reason,
        # Display horizon (what /cash uses for annualization) — always 90d
        # post user feedback '60d pode ser pouco dar hold'.
        "horizon_days":            _DISPLAY_HORIZON_DAYS,
        # Training horizon (what the model actually learned) — for audit
        "training_horizon_days":   horizon_days,
        "extrap_factor":           round(_scale, 4),
        "n_samples_total":    int(len(pred)),
        "n_real":             0,
        "n_synthetic":        int(len(pred)),
        "version":            "v1_parquet_in_sample",
        "calibrated_on":      date.today().isoformat(),
        "min_r_squared_accept": _MIN_R_SQUARED_ACCEPT,
        "min_spread_accept":  _MIN_SPREAD_FOR_ACCEPT,
        "parquet_source":     str(_PARQUET_PATH.name),
        "target_column":      target_col,
        # TODO marker for next DIP retrain — see notice below
        "retrain_target_recommendation": "alpha_90d",
        "warning":            (
            "V1 calibration uses IN-SAMPLE predictions on the training parquet — "
            "model is overfit to these rows, so r_squared is optimistic. V2 will "
            "incorporate mature alerts from alert_db.csv with declining synthetic "
            "weight as real samples accumulate. Refresh weekly via "
            "scripts/refresh_dip_alpha_calibration.py (next iteration)."
        ),
        "retrain_note":       (
            "If you retrain the DIP model: set DIP_PRIMARY_TARGET=alpha_90d in "
            "build_dip_dataset.py so the parquet has alpha_90d as the target. "
            "The current bundle was trained on alpha_60d, so this calibration "
            f"applies a linear extrapolation factor ×{_scale:.3f} to express α "
            f"at the {_DISPLAY_HORIZON_DAYS}d horizon. After retraining with "
            "alpha_90d, the extrap_factor will become 1.0 (no transformation) "
            "and predictions will match the user's actual hold horizon directly."
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
    df, target_col = _load_parquet()
    bundle = _load_bundle()

    pred   = _predict_on_parquet(df, bundle)
    actual = df[target_col].fillna(0.0).values
    log.info(f"Using '{target_col}' as α target")

    calib = _bucket_calibration(
        pred, actual,
        n_buckets=args.buckets,
        min_samples_per_bucket=args.min_samples_per_bucket,
        target_col=target_col,
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
