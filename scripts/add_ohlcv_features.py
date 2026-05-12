#!/usr/bin/env python3
"""Backfill 4 raw-OHLCV features into an existing ml_training_base.parquet.

This script ONLY adds 4 new columns:
  • realized_vol_20d
  • mom_5d_slope
  • volume_zscore_20d
  • gap_pct_5d

It does **NOT** touch the existing columns. The point is to extend the
training parquet with the PR #30 features WITHOUT risking accidental
drift on the existing 49 columns (which would happen with a full
:mod:`scripts.regenerate_training_base` run that re-fetches yfinance).

Usage:
    python scripts/add_ohlcv_features.py
    python scripts/add_ohlcv_features.py --in custom.parquet --out custom_v2.parquet

Side effects:
  • Fetches price history per unique ticker via yfinance (with cache in
    ``/tmp/yf_cache_v2`` — re-uses cache from regenerate_training_base.py
    if available).
  • Writes the augmented parquet to ``--out`` (default: overwrite
    ``ml_training_base.parquet`` in-place; backup recommended).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from importlib import util as importlib_util
from pathlib import Path
from typing import Optional

import pandas as pd

# ────────────────────────────────────────────────────────────────────────────────
# Imports — repo path
# ────────────────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load the fetch helpers from the sibling script (no __init__.py in scripts/)
_REGEN_PATH = _REPO_ROOT / "scripts" / "regenerate_training_base.py"
_spec = importlib_util.spec_from_file_location("_regen_helpers", _REGEN_PATH)
_regen = importlib_util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_regen)

_fetch_batch = _regen._fetch_batch
_slice_history = _regen._slice_history

from ml_features import _FALLBACK, add_raw_ohlcv_features  # noqa: E402

# ────────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────────

NEW_COLS: list[str] = [
    "volume_zscore_20d",
    "close_in_range_20d",
    "up_days_pct_20d",
    "true_range_pct_20d",
]

DEFAULT_IN  = _REPO_ROOT / "ml_training_base.parquet"
DEFAULT_OUT = _REPO_ROOT / "ml_training_base.parquet"
DEFAULT_CACHE_DIR = Path("/tmp/yf_cache_v2")  # shared with regenerate_training_base

log = logging.getLogger("add_ohlcv_features")


# ────────────────────────────────────────────────────────────────────────────────
# Core
# ────────────────────────────────────────────────────────────────────────────────

def _compute_row(
    row: pd.Series,
    price_cache: dict[str, pd.DataFrame],
) -> dict[str, float]:
    """Compute the 4 raw-OHLCV features for one row.

    Returns dict with all 4 keys (falls back to neutral defaults if the
    slice is too short or the ticker is missing from cache).
    """
    fv: dict[str, float] = {k: _FALLBACK[k] for k in NEW_COLS}
    ticker = str(row["ticker"])
    ts = pd.Timestamp(row["alert_date"])
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    hist = _slice_history(price_cache.get(ticker), ts)
    if hist is None or hist.empty:
        return fv
    try:
        add_raw_ohlcv_features(fv, hist)
    except Exception as e:
        log.debug(f"  failed {ticker}@{ts.date()}: {e}")
    # Guard: ensure all 4 keys are floats
    return {k: float(fv.get(k, _FALLBACK[k])) for k in NEW_COLS}


def backfill(
    in_path: Path,
    out_path: Path,
    cache_dir: Path,
    start_date: str = "1995-01-01",
) -> None:
    """Pipeline principal."""
    if not in_path.exists():
        raise FileNotFoundError(f"input parquet não encontrado: {in_path}")

    log.info(f"[in] {in_path} ({in_path.stat().st_size/1e6:.1f} MB)")
    df = pd.read_parquet(in_path)
    df["alert_date"] = pd.to_datetime(df["alert_date"]).dt.tz_localize(None)
    log.info(f"[in] shape={df.shape}  unique tickers={df['ticker'].nunique()}")
    log.info(f"[in] date range: {df['alert_date'].min()} → {df['alert_date'].max()}")

    # Detectar se as colunas já existem (re-run safe)
    pre_existing = [c for c in NEW_COLS if c in df.columns]
    if pre_existing:
        log.warning(f"[in] colunas já existem (serão sobrepostas): {pre_existing}")

    # ── Fetch ticker histories ──────────────────────────────────────────────
    unique_tickers = sorted(df["ticker"].astype(str).unique().tolist())
    log.info(f"[fetch] {len(unique_tickers)} ticker histories...")
    price_cache = _fetch_batch(unique_tickers, cache_dir, start=start_date)
    log.info(f"[fetch] {len(price_cache)}/{len(unique_tickers)} tickers OK")

    missing = [t for t in unique_tickers if t not in price_cache]
    if missing:
        log.warning(
            f"[fetch] {len(missing)} tickers sem history (usarão fallbacks): "
            f"{missing[:10]}{' ...' if len(missing) > 10 else ''}"
        )

    # ── Compute features per row ────────────────────────────────────────────
    log.info("[compute] 4 raw-OHLCV features por linha...")
    t0 = time.time()
    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        results.append(_compute_row(row, price_cache))
        if (i + 1) % 2000 == 0 or (i + 1) == len(df):
            elapsed = time.time() - t0
            log.info(f"  [{i+1:>5}/{len(df)}] {elapsed:.0f}s elapsed")

    new_cols_df = pd.DataFrame(results, index=df.index)

    # ── Merge + save ────────────────────────────────────────────────────────
    for col in NEW_COLS:
        df[col] = new_cols_df[col].values

    log.info(f"[out] writing {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info(f"[out] done. shape={df.shape}  size={out_path.stat().st_size/1e6:.1f} MB")

    # Sanity stats
    log.info("[stats] new feature summary:")
    for col in NEW_COLS:
        s = df[col]
        log.info(
            f"  {col:24s} mean={s.mean():>8.4f}  std={s.std():>7.4f}  "
            f"min={s.min():>8.4f}  max={s.max():>8.4f}  "
            f"nan={s.isna().sum()}"
        )


# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in",  dest="in_path",  type=Path, default=DEFAULT_IN)
    p.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cache", dest="cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--start", dest="start_date", default="1995-01-01")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    backfill(args.in_path, args.out_path, args.cache_dir, args.start_date)
    return 0


if __name__ == "__main__":
    sys.exit(main())
