"""calibrate_core_alpha.py — 7-bucket calibration of CORE model α_1y predictions.

Replicates the DIP calibration pattern (scripts/calibrate_dip_alpha.py) for
the CORE model. Reads /data/core_oof_predictions.parquet written by
scripts/train_core_model.py during the most recent retrain (out-of-fold
predictions from purged blocked K-fold), buckets predictions by quantile,
and persists empirical bucket means + percentile downsides for Kelly
sizing in /cash.

OOF (out-of-fold) requirement: in-sample predictions inflate r² by 2-3x
and over-confidence in downside_q05. The trainer was modified on
2026-06-15 to persist OOF preds during the CV pass. If the file is
missing (legacy bundle), the script logs a warning and exits.

Persisted output: /data/core_alpha_calibration.json
Consumer:        bot_commands.py /cash → CORE candidates path

Usage:
    python scripts/calibrate_core_alpha.py
    python scripts/calibrate_core_alpha.py --buckets 10
    python scripts/calibrate_core_alpha.py --min-samples-per-bucket 50
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
log = logging.getLogger("calibrate_core")

# Output to Railway volume /data/ (persists across redeploys).
_DATA_DIR    = Path("/data") if Path("/data").exists() else (_ROOT / "data")
_OUTPUT_PATH = _DATA_DIR / "core_alpha_calibration.json"
_OOF_PATH    = _DATA_DIR / "core_oof_predictions.parquet"

# Acceptance gates: r² >= 0.10 OR (monotonic AND spread >= 4pp). Same logic
# as DIP. CORE has higher CV IC (0.119 vs DIP 0.063) so we expect higher
# r² but the gates are kept consistent for comparability.
_MIN_R_SQUARED_ACCEPT  = 0.10
_MIN_SPREAD_FOR_ACCEPT = 0.04

# CORE training target is alpha_1y (1-year excess return vs SPY). No
# horizon extrapolation: training horizon = display horizon = 365d.
_TRAINING_HORIZON_DAYS = 365


def _ensure_data_dir() -> None:
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_oof_predictions():
    """Return (df, target_col) where df has oof_pred + actual columns.

    The trainer persists alert_date, ticker, sector, actual, oof_pred —
    we drop NaN rows and return what's usable for calibration.
    """
    import pandas as pd
    if not _OOF_PATH.exists():
        log.error(
            f"[oof] {_OOF_PATH} not found. CORE bucket calibration requires "
            "out-of-fold predictions from the trainer. Run "
            "/admin_train_core_model to retrain — the updated trainer "
            "(commit 2026-06-15) persists OOF preds during the CV pass."
        )
        sys.exit(1)
    try:
        df = pd.read_parquet(_OOF_PATH)
    except Exception as e:
        log.error(f"[oof] read failed: {e}")
        sys.exit(1)
    if "oof_pred" not in df.columns or "actual" not in df.columns:
        log.error(
            f"[oof] {_OOF_PATH} missing required columns "
            f"(needs oof_pred + actual, got {list(df.columns)})"
        )
        sys.exit(1)
    df = df.dropna(subset=["oof_pred", "actual"]).reset_index(drop=True)
    if len(df) < 100:
        log.error(f"[oof] only {len(df)} usable rows — too few to calibrate")
        sys.exit(1)
    log.info(
        f"[oof] loaded {len(df):,} OOF predictions from {_OOF_PATH.name} "
        f"(mean_pred={df['oof_pred'].mean():+.4f}, "
        f"mean_actual={df['actual'].mean():+.4f})"
    )
    return df, "alpha_1y"


def _bucket_calibration(
    pred: np.ndarray,
    actual: np.ndarray,
    n_buckets: int = 7,
    min_samples_per_bucket: int = 30,
    target_col: str = "alpha_1y",
) -> dict:
    """Compute per-bucket calibration."""
    mask = np.isfinite(pred) & np.isfinite(actual)
    pred   = pred[mask]
    actual = actual[mask]
    log.info(f"Calibrating on {len(pred)} valid (pred, actual) pairs")

    quantiles = np.linspace(0.0, 1.0, n_buckets + 1)
    edges = np.quantile(pred, quantiles)
    edges = np.unique(edges)
    if len(edges) - 1 < 3:
        log.warning("Too few unique quantile bins — predictions may be degenerate")

    buckets = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == len(edges) - 2:
            in_bucket = (pred >= lo) & (pred <= hi)
        else:
            in_bucket = (pred >= lo) & (pred < hi)
        n = int(in_bucket.sum())
        if n < min_samples_per_bucket:
            log.warning(
                f"Bucket {i}: only {n} samples (< {min_samples_per_bucket}) — "
                f"skipping"
            )
            continue
        samples = actual[in_bucket]
        # Percentile envelope. q05 = "5% worst case in this bucket" — used
        # as downside in Kelly Full sizing.
        q05 = float(np.quantile(samples, 0.05))
        buckets.append({
            "pred_min":              float(lo),
            "pred_max":              float(hi),
            "n_samples":             n,
            f"mean_actual_{target_col}":   float(samples.mean()),
            f"median_actual_{target_col}": float(np.median(samples)),
            f"q05_actual_{target_col}":    q05,
            f"q25_actual_{target_col}":    float(np.quantile(samples, 0.25)),
            f"q75_actual_{target_col}":    float(np.quantile(samples, 0.75)),
            f"q95_actual_{target_col}":    float(np.quantile(samples, 0.95)),
            # downside_q05 is the absolute value of q05 (Kelly expects positive).
            # If q05 is positive (rare — would mean even worst-case is profitable),
            # use a minimum to avoid division blow-up.
            "downside_q05":          max(0.001, abs(q05)),
        })

    # Acceptance: r² of bucket index vs mean_actual; monotonic + spread.
    if len(buckets) >= 3:
        bucket_idx = np.arange(len(buckets), dtype=float)
        bucket_means = np.array([b[f"mean_actual_{target_col}"] for b in buckets])
        # r² via simple linear regression
        if len(bucket_means) > 1 and bucket_means.std() > 0:
            corr_matrix = np.corrcoef(bucket_idx, bucket_means)
            r_squared = float(corr_matrix[0, 1] ** 2) if not np.isnan(corr_matrix[0, 1]) else 0.0
        else:
            r_squared = 0.0
        monotonic = bool(np.all(np.diff(bucket_means) >= -1e-9))
        spread = float(bucket_means[-1] - bucket_means[0])
        accepted = (
            r_squared >= _MIN_R_SQUARED_ACCEPT
            or (monotonic and spread >= _MIN_SPREAD_FOR_ACCEPT)
        )
        log.info(f"Buckets monotonic: {monotonic}")
        log.info(f"Spread top-bottom buckets: {spread:.4f}")
    else:
        r_squared = 0.0
        monotonic = False
        spread = 0.0
        accepted = False

    log.info(f"Training horizon: {_TRAINING_HORIZON_DAYS}d (column={target_col})")
    log.info("=" * 60)
    log.info(
        f"Calibration result: r²={r_squared:.4f}  monotonic={monotonic}  "
        f"accepted={accepted}  oof_method=purged_blocked_kfold_OOF"
    )
    log.info("Buckets (pred range → mean actual alpha_1y):")
    for b in buckets:
        log.info(
            f"  [{b['pred_min']:+.4f}, {b['pred_max']:+.4f}] → "
            f"actual mean={b[f'mean_actual_{target_col}']:+.4f} "
            f"(median={b[f'median_actual_{target_col}']:+.4f}) "
            f"n={b['n_samples']}"
        )
    log.info("=" * 60)

    return {
        "buckets":              buckets,
        "n_total":              int(len(pred)),
        "n_buckets":            len(buckets),
        "r_squared":            r_squared,
        "is_monotonic":         monotonic,
        "spread_top_bottom":    spread,
        "accepted":             accepted,
        "training_horizon_days": _TRAINING_HORIZON_DAYS,
        "extrap_factor":        1.0,  # CORE: training horizon = display horizon
        "calibrated_on":        date.today().isoformat(),
        "oof_method":           "purged_blocked_kfold_OOF",
        "version":              "v2_oof",
        "target_col":           target_col,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--buckets", type=int, default=7,
                        help="Number of quantile buckets (default: 7)")
    parser.add_argument("--min-samples-per-bucket", type=int, default=30,
                        help="Warn if a bucket has fewer than this many samples")
    args = parser.parse_args()

    _ensure_data_dir()
    df, target_col = _load_oof_predictions()
    pred = df["oof_pred"].values.astype(float)
    actual = df["actual"].values.astype(float)
    log.info(f"Using '{target_col}' as α target | oof_method=purged_blocked_kfold_OOF")

    calib = _bucket_calibration(
        pred, actual,
        n_buckets=args.buckets,
        min_samples_per_bucket=args.min_samples_per_bucket,
        target_col=target_col,
    )

    with _OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)
    log.info(f"Persisted to {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
