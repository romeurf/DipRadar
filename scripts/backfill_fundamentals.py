"""
backfill_fundamentals.py — Recompute ONLY fundamental features in the DIP parquet.

Problem this solves:
  The 211k-row parquet has all technicals (RSI, drawdown, drop_pct, etc.) but
  the fundamental columns are 100% NaN due to bugs that have now been fixed
  (lxml dep, yfinance schema renames, EDGAR YoY TTM not implemented). A full
  /admin_overnight rebuild would fix this but takes 4-18h and re-downloads
  yfinance OHLCV that's already in cache.

  This script:
    - Reads the existing parquet
    - For each row, recomputes fundamentals via the (now-fixed) PIT pipeline
    - Updates ONLY the fundamental columns
    - Preserves technicals, targets, and everything else untouched
    - Atomic write at end (writes .tmp then rename — original safe until done)

  Estimated time: 45-90 min for 211k rows
    - 1500 unique tickers × 1 EDGAR fetch each (cached after) = 10-20 min IO
    - 211k rows × per-row fundamental compute = 30-60 min CPU

Updated columns (only these — everything else preserved):
  EDGAR-sourced (PIT-correct, US tickers ≥2009):
    gross_margin, de_ratio, roe, quality_score, revenue_growth,
    earnings_growth, _eps_ttm, trailing_pe, fcf_yield

  yfinance-sourced (shallow history, recent rows only):
    earnings_beat_rate, analyst_rating

  Derived (computed from above):
    peg_ratio

Usage:
  python scripts/backfill_fundamentals.py
  python scripts/backfill_fundamentals.py --parquet /data/ml_training_base.parquet
  python scripts/backfill_fundamentals.py --tickers AAPL,MSFT  # subset for testing
  python scripts/backfill_fundamentals.py --since 2020-01-01  # rows after this date

Progress: logs every 1000 rows. Resume-safe: if you abort mid-run, the original
parquet is intact (atomic .tmp + rename only happens at end).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Lazy imports below to allow --help without heavy startup

log = logging.getLogger("backfill_fundamentals")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


# Columns the script will recompute. Everything else in the parquet is preserved.
#
# Order matters: tier 1 (fundamentals) runs first, tier 2 (Phase 5 EDGAR
# Form 4 / 8-K / earnings call) runs second, then derived features (peg_ratio)
# get computed last from the combined inputs.
_TIER1_FUNDAMENTAL_COLS = [
    # From _compute_pit_fundamentals_row (EDGAR + yfinance.quarterly tier merge)
    "gross_margin",
    "de_ratio",
    "roe",
    "quality_score",
    "fcf_yield",
    "revenue_growth",
    "earnings_growth",
    "_eps_ttm",
    "trailing_pe",
    "earnings_beat_rate",
    "analyst_rating",
]

_TIER2_PHASE5_COLS = [
    # From _compute_fase5_row (EDGAR Form 4 / 8-K / earnings calls)
    "insider_buy_recent",
    "insider_sell_recent",
    "insider_buy_amount_score",
    "insider_sell_amount_score",
    "recent_8k_score",
    "earnings_call_tone",
]

_DERIVED_COLS = [
    # Computed by add_derived_features from tier 1+2 + alert_date
    "peg_ratio",
]

_ALL_COLS_TO_REFRESH = _TIER1_FUNDAMENTAL_COLS + _TIER2_PHASE5_COLS + _DERIVED_COLS


def _process_row(row: pd.Series, cache_dir: Path) -> dict:
    """Compute updated fundamentals + Phase 5 features for one row.

    Two-tier recompute:
      Tier 1: _compute_pit_fundamentals_row → EDGAR TTM YoY (revenue_growth,
              earnings_growth, _eps_ttm), gross_margin, de_ratio, roe,
              earnings_beat_rate, analyst_rating
      Tier 2: _compute_fase5_row → EDGAR Form 4 insider activity (buys/sells/
              amounts), 8-K material event score, earnings call tone

    Then add_derived_features computes peg_ratio from the combined tier 1 inputs.

    Only returns columns where we got a fresh value; NaN/missing stays missing.

    Early-skip optimisation: alert_dates before 2003 have no realistic chance
    of EDGAR fundamentals (XBRL mandate is 2009) or Form 4 (electronic filing
    pre-2003 was inconsistent). Skipping these rows instantly avoids ~20k
    pointless HTTP attempts on the full parquet — net savings of hours.
    """
    from scripts.regenerate_training_base import (
        _compute_pit_fundamentals_row,
        _compute_fase5_row,
    )
    from utils.sec_guard import has_sec_filings
    from ml.ml_features import add_derived_features

    # Fast path: pre-2003 rows have no fundamentals or Form 4 to fetch.
    # Return NaN for all refresh cols (honest: we know we won't find data)
    # so the backfill writes NaN deliberately instead of trying every source
    # and timing out.
    try:
        _yr = pd.Timestamp(row["alert_date"]).year
    except Exception:
        _yr = 1900
    if _yr < 2003:
        return {c: float("nan") for c in _ALL_COLS_TO_REFRESH}

    # Tier 1: PIT fundamentals
    fund: dict = {}
    try:
        fund = _compute_pit_fundamentals_row(row, cache_dir) or {}
    except Exception as e:
        log.debug(f"[backfill] PIT fundamentals failed for {row.get('ticker', '?')}@{row.get('alert_date', '?')}: {e}")

    merged = dict(row)
    merged.update(fund)

    # Tier 2: Phase 5 EDGAR signals — but only for tickers with SEC coverage
    sector = str(row.get("sector", ""))
    country = ""  # not in parquet but has_sec_filings tolerates empty country for US tickers
    exchange = ""
    try:
        if has_sec_filings(str(row["ticker"]), country, exchange):
            # price_cache MUST be a dict-like object — _compute_fase5_row
            # does `price_cache.get(ticker)` on line 554. Passing None
            # raises AttributeError immediately, killing Phase 5 silently
            # (the outer try/except just logs at DEBUG and returns nothing
            # new — merged keeps row's old values, including any fake 0.0
            # from prior buggy runs). Empty dict {} is safe: price-based
            # signals (consecutive_red_days, ma_200d_ratio) stay NaN
            # because there's no history, but EDGAR signals (Form 4
            # insider, 8-K) still run.
            fase5 = _compute_fase5_row(pd.Series(merged), {}) or {}
            merged.update(fase5)
    except Exception as e:
        log.warning(f"[backfill] Phase 5 failed for {row.get('ticker', '?')}: {e}")

    # Derived: peg_ratio etc.
    try:
        add_derived_features(merged, alert_date=pd.Timestamp(row["alert_date"]))
    except Exception as e:
        log.debug(f"[backfill] add_derived_features failed for {row.get('ticker', '?')}: {e}")

    # Return ALL refresh columns including NaN. The previous version skipped
    # NaN writes to "preserve existing real data", but that logic preserved
    # FAKE zeros — insider_*_amount_score had 0.0 written by an earlier
    # buggy backfill when EDGAR rate-limited (function used to return 0.0
    # on parse failure; now correctly returns NaN). Skipping NaN now would
    # cement those false zeros forever.
    #
    # The fix: trust the source function. It returns NaN only when it
    # genuinely doesn't know; otherwise a real value (including 0.0 for
    # "confirmed no insider activity"). Always write what the function says.
    out = {}
    for col in _ALL_COLS_TO_REFRESH:
        if col in merged:
            out[col] = merged[col]  # may be NaN — that's now honest
    return out


def backfill(
    parquet_path: Path,
    cache_dir: Path,
    ticker_filter: Optional[set[str]] = None,
    since_date: Optional[pd.Timestamp] = None,
    log_every: int = 1000,
    max_rows: Optional[int] = None,
) -> dict:
    """Walk the parquet row-by-row, recompute fundamentals, atomic write.

    Returns stats dict with rows_processed, rows_updated, fields_updated, etc.
    """
    log.info(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    n_total = len(df)
    log.info(f"  Total rows: {n_total:,}")

    # Ensure all target columns exist (in case of older parquet without them)
    for col in _ALL_COLS_TO_REFRESH:
        if col not in df.columns:
            df[col] = np.nan

    # Apply optional filters
    mask = pd.Series([True] * n_total, index=df.index)
    if ticker_filter:
        mask &= df["ticker"].isin(ticker_filter)
        log.info(f"  Ticker filter: {len(ticker_filter)} tickers → {mask.sum():,} rows")
    if since_date is not None:
        mask &= pd.to_datetime(df["alert_date"]) >= since_date
        log.info(f"  Since filter ≥{since_date.date()} → {mask.sum():,} rows")

    process_idx = df.index[mask].tolist()
    if not process_idx:
        log.warning("Nothing to process after filters.")
        return {"rows_processed": 0, "rows_updated": 0}

    # Incremental session cap. The Form 4 XML cache is permanent so a
    # partial run still warms it for subsequent sessions. Selecting from
    # the front of process_idx (chronologically earliest after sort) gives
    # deterministic resume behaviour: the next session's --max-rows starts
    # where this one left off only if combined with `--since` shift, but
    # critically the cache is what matters most — XMLs once fetched don't
    # need to be fetched again.
    if max_rows is not None and max_rows > 0 and len(process_idx) > max_rows:
        log.info(
            f"  Max-rows cap: {max_rows:,} of {len(process_idx):,} rows "
            f"(remaining will need additional session)"
        )
        process_idx = process_idx[:max_rows]

    log.info(f"  Processing: {len(process_idx):,} rows")

    # NaN counts BEFORE backfill (for verification report at the end)
    nan_before: dict[str, int] = {}
    for col in _ALL_COLS_TO_REFRESH:
        nan_before[col] = int(df[col].isna().sum())

    t0 = time.time()
    rows_processed = 0
    rows_updated = 0
    field_counts: dict[str, int] = {c: 0 for c in _ALL_COLS_TO_REFRESH}

    # First-row diagnostic — confirms pipeline actually executes the work
    # we expect. Without this, a silent no-op (e.g. has_sec_filings
    # rejecting every ticker, or _compute_pit_fundamentals_row swallowing
    # an exception) is invisible from the final NaN report alone.
    _diag_emitted = False

    for i, idx in enumerate(process_idx):
        row = df.loc[idx]
        if not _diag_emitted:
            try:
                from utils.sec_guard import has_sec_filings as _has_sec
                _hs = _has_sec(str(row["ticker"]), "", "")
            except Exception as _e:
                _hs = f"ERR({type(_e).__name__})"
            new_values = _process_row(row, cache_dir)
            _real_keys = [k for k, v in new_values.items()
                          if v is not None and not (isinstance(v, float) and np.isnan(v))]
            _nan_keys  = [k for k, v in new_values.items()
                          if v is None or (isinstance(v, float) and np.isnan(v))]
            log.info(
                f"  [diag] first row: ticker={row.get('ticker', '?')} "
                f"alert_date={row.get('alert_date', '?')} "
                f"has_sec_filings={_hs} "
                f"new_values_returned={len(new_values)} "
                f"real={len(_real_keys)} nan={len(_nan_keys)}"
            )
            if _real_keys:
                log.info(f"  [diag]   real keys: {_real_keys}")
            if _nan_keys[:8]:
                log.info(f"  [diag]   nan keys (first 8): {_nan_keys[:8]}")
            _diag_emitted = True
        else:
            new_values = _process_row(row, cache_dir)
        if new_values:
            rows_updated += 1
            for col, val in new_values.items():
                df.at[idx, col] = val
                field_counts[col] += 1
        rows_processed += 1

        if (i + 1) % log_every == 0 or (i + 1) == len(process_idx):
            elapsed = time.time() - t0
            rate = rows_processed / elapsed if elapsed > 0 else 0
            eta_s = (len(process_idx) - rows_processed) / rate if rate > 0 else 0
            log.info(
                f"  [{i+1:,}/{len(process_idx):,} {(i+1)/len(process_idx)*100:.1f}%] "
                f"rate {rate:.1f} rows/s | "
                f"updated {rows_updated:,} | "
                f"ETA {eta_s/60:.0f}min"
            )

    elapsed = time.time() - t0
    log.info(f"  Compute done in {elapsed/60:.1f} min ({rows_processed:,} rows)")

    # Atomic write: temp file → rename
    tmp_path = parquet_path.with_suffix(".tmp.parquet")
    log.info(f"  Writing to temp: {tmp_path}")
    df.to_parquet(tmp_path, index=False)

    # Sanity check before swap
    tmp_size = tmp_path.stat().st_size
    orig_size = parquet_path.stat().st_size
    if tmp_size < orig_size * 0.5:
        raise RuntimeError(
            f"Sanity fail: new parquet ({tmp_size:,}B) is <50% of original "
            f"({orig_size:,}B). Aborting swap. Inspect {tmp_path}."
        )

    log.info(f"  Atomic rename: {tmp_path.name} → {parquet_path.name}")
    tmp_path.replace(parquet_path)

    # NaN counts AFTER backfill, computed on the saved dataframe
    log.info(f"✅ Backfill complete in {(time.time() - t0)/60:.1f} min")
    log.info(f"   Rows updated: {rows_updated:,}/{rows_processed:,}")
    log.info("")
    log.info("   NaN coverage report (BEFORE → AFTER, delta):")
    log.info("   " + "─" * 75)
    nan_after: dict[str, int] = {}
    for col in _ALL_COLS_TO_REFRESH:
        nan_after[col] = int(df[col].isna().sum())
        before = nan_before[col]
        after  = nan_after[col]
        delta  = before - after
        pct_real_before = (1 - before / n_total) * 100
        pct_real_after  = (1 - after  / n_total) * 100
        # Visual:
        #   ✅ real coverage improved (NaN→value)
        #   ➖ unchanged
        #   🔄 NaN count increased — backfill OVERWROTE previous "real"
        #      values with NaN. This is EXPECTED for insider_*_amount_score
        #      after the parse-failure fix (those zeros were fake; we're
        #      converting them back to honest NaN). For other columns, a
        #      negative delta means the source function couldn't recompute
        #      a value it previously had — investigate.
        if delta > 0:
            icon = "✅"
        elif delta == 0:
            icon = "➖"
        else:
            icon = "🔄"
        log.info(
            f"   {icon} {col:35s}  "
            f"NaN: {before:>7,} → {after:>7,}  ({delta:+,})  "
            f"real: {pct_real_before:5.1f}% → {pct_real_after:5.1f}%"
        )
    log.info("   " + "─" * 75)

    return {
        "rows_processed":  rows_processed,
        "rows_updated":    rows_updated,
        "fields_updated":  dict(field_counts),
        "nan_before":      nan_before,
        "nan_after":       nan_after,
        "elapsed_minutes": round((time.time() - t0) / 60, 1),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--parquet", default=None,
                   help="Path to parquet (default: /data/ml_training_base.parquet)")
    p.add_argument("--cache-dir", default=None,
                   help="EDGAR/yfinance cache dir (default: /data/fundamentals_cache)")
    p.add_argument("--tickers", help="Comma-separated ticker subset (default: all)")
    p.add_argument("--since", help="Only process rows with alert_date >= YYYY-MM-DD")
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--max-rows", type=int, default=None,
                   help="Cap rows processed this session (incremental backfill). "
                        "Form 4 XML cache is permanent so partial runs warm it for "
                        "subsequent sessions.")
    args = p.parse_args()

    data_dir = Path("/data") if Path("/data").exists() else Path("/tmp")
    parquet_path = Path(args.parquet) if args.parquet else (data_dir / "ml_training_base.parquet")
    cache_dir    = Path(args.cache_dir) if args.cache_dir else (data_dir / "fundamentals_cache")

    if not parquet_path.exists():
        log.error(f"Parquet not found: {parquet_path}")
        return 1

    ticker_filter = None
    if args.tickers:
        ticker_filter = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}

    since_date = None
    if args.since:
        try:
            since_date = pd.Timestamp(args.since)
        except Exception as e:
            log.error(f"Invalid --since date: {e}")
            return 1

    cache_dir.mkdir(parents=True, exist_ok=True)
    backfill(
        parquet_path=parquet_path,
        cache_dir=cache_dir,
        ticker_filter=ticker_filter,
        since_date=since_date,
        log_every=args.log_every,
        max_rows=args.max_rows,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
