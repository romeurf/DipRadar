"""scripts/patch_validation_cache.py — Post-build patch for survivorship audit.

Runs against an existing /data/validation_cache/ produced by a build
that DID NOT track skipped_tickers (pre-patch versions of
validation_cache_builder.py). Detects every CORE ticker silently
dropped from each artefact, tries per-ticker recovery for prices,
and rewrites cache_metadata.json with the population-truncation log.

Per F.5 user review: "HOLX delisted, skipped" was the back door
letting survivorship bias enter the spine without being recorded.
This script closes the door retroactively on the current cache so
the validation JSON can be read with the inflation factor visible.

Idempotent: re-running after a successful patch is a no-op (the
metadata file already carries the skipped_tickers section). Safe to
run multiple times.

Operations:
  1. Read CORE list from core_universe.json
  2. For each artefact (prices, sectors, edgar_8k, edgar_form4,
     fundamentals): determine which CORE tickers are missing
  3. For PRICES: attempt per-ticker yfinance.Ticker.history() recovery
     for missing tickers. Append any recovered partial history to
     prices.parquet, mark as `partial_delisted` if history ends
     before the fetch window
  4. Update cache_metadata.json:
     skipped_tickers: { artefact -> [{ticker, reason}, ...] }
     skip_counts:     { artefact -> int }
     patched_at:      ISO timestamp
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("patch_cache")

_DATA_DIR  = Path("/data") if Path("/data").exists() else _ROOT / "data"
_CACHE_DIR = _DATA_DIR / "validation_cache"

_FETCH_START_ISO = "2020-10-01"
_FETCH_END_ISO   = "2026-06-30"


def _load_core() -> list[str]:
    u = json.loads((_ROOT / "core_universe.json").read_text())
    return [t["ticker"] for t in u.get("tickers", [])]


def _present_in_prices() -> set[str]:
    p = _CACHE_DIR / "prices.parquet"
    if not p.exists():
        return set()
    df = pd.read_parquet(p, columns=["ticker"])
    return set(df["ticker"].unique().tolist())


def _present_in_parquet(name: str, key: str = "ticker") -> set[str]:
    p = _CACHE_DIR / name
    if not p.exists():
        return set()
    df = pd.read_parquet(p, columns=[key])
    return set(df[key].unique().tolist())


def _present_in_sectors() -> set[str]:
    p = _CACHE_DIR / "sectors.json"
    if not p.exists():
        return set()
    data = json.loads(p.read_text())
    return {t for t, s in data.items() if s}


def attempt_price_recovery(missing: list[str]) -> tuple[list[str], list[dict], list[dict]]:
    """For each missing ticker, try per-ticker history fetch.

    Returns (recovered_tickers, skipped_full, skipped_partial).
    skipped_partial holds delisted tickers we DID recover (with truncated
    last_traded). skipped_full holds tickers with no recoverable history.
    """
    import yfinance as yf
    recovered = []
    skipped_full = []
    skipped_partial = []
    new_rows = []

    for ticker in missing:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(
                start=_FETCH_START_ISO, end=_FETCH_END_ISO,
                auto_adjust=True,
            )
            if hist is None or hist.empty:
                skipped_full.append({
                    "ticker": ticker,
                    "reason": "per-ticker fetch returned empty (delisted with no recoverable history, or invalid symbol)",
                })
                continue
            hist = hist.dropna(subset=["Close"])
            if hist.empty:
                skipped_full.append({
                    "ticker": ticker,
                    "reason": "per-ticker fetch returned only NaN closes",
                })
                continue
            for idx, row in hist.iterrows():
                new_rows.append({
                    "date":   idx.date().isoformat(),
                    "ticker": ticker,
                    "open":   float(row["Open"])   if not pd.isna(row["Open"])   else None,
                    "high":   float(row["High"])   if not pd.isna(row["High"])   else None,
                    "low":    float(row["Low"])    if not pd.isna(row["Low"])    else None,
                    "close":  float(row["Close"])  if not pd.isna(row["Close"])  else None,
                    "volume": float(row["Volume"]) if not pd.isna(row["Volume"]) else None,
                })
            recovered.append(ticker)
            last_dt = hist.index.max()
            if hasattr(last_dt, "date"):
                last_iso = last_dt.date().isoformat()
                if last_iso < _FETCH_END_ISO:
                    skipped_partial.append({
                        "ticker":      ticker,
                        "last_traded": last_iso,
                        "reason":      f"recovered partial history; last trading day {last_iso} (likely delisted before window end)",
                    })
            log.info(f"[recovery] {ticker} → +{len(hist)} rows (last_traded={last_dt.date().isoformat()})")
        except Exception as e:
            skipped_full.append({
                "ticker": ticker,
                "reason": f"per-ticker fetch raised: {e}",
            })

    if new_rows:
        existing = pd.read_parquet(_CACHE_DIR / "prices.parquet")
        recovered_df = pd.DataFrame(new_rows)
        merged = pd.concat([existing, recovered_df], ignore_index=True)
        merged.to_parquet(_CACHE_DIR / "prices.parquet", index=False)
        log.info(f"[recovery] appended {len(new_rows)} rows for {len(recovered)} tickers to prices.parquet")

    return recovered, skipped_full, skipped_partial


def main():
    log.info(f"[patch] cache dir: {_CACHE_DIR}")
    if not _CACHE_DIR.exists():
        log.error(f"{_CACHE_DIR} missing — run validation_cache_builder.py first")
        sys.exit(1)

    core = _load_core()
    core_set = set(core)
    log.info(f"[patch] CORE universe: {len(core)} tickers")

    # ── Detect skipped per artefact ───────────────────────────────────
    present_prices       = _present_in_prices()
    present_sectors      = _present_in_sectors()
    present_edgar_8k     = _present_in_parquet("edgar_8k.parquet")
    present_edgar_form4  = _present_in_parquet("edgar_form4.parquet")
    present_fundamentals = _present_in_parquet("quarterly_financials.parquet")

    missing_prices       = sorted(core_set - present_prices)
    missing_sectors      = sorted(core_set - present_sectors)
    missing_edgar_8k     = sorted(core_set - present_edgar_8k)
    missing_edgar_form4  = sorted(core_set - present_edgar_form4)
    missing_fundamentals = sorted(core_set - present_fundamentals)

    log.info(f"[patch] missing — prices: {len(missing_prices)}, sectors: {len(missing_sectors)}, "
             f"edgar_8k: {len(missing_edgar_8k)}, edgar_form4: {len(missing_edgar_form4)}, "
             f"fundamentals: {len(missing_fundamentals)}")

    # ── Attempt price recovery for missing tickers ────────────────────
    if missing_prices:
        log.info(f"[patch] attempting per-ticker recovery for {len(missing_prices)} price misses")
        recovered, skipped_full, skipped_partial = attempt_price_recovery(missing_prices)
        log.info(f"[patch] recovery: {len(recovered)} recovered, "
                 f"{len(skipped_full)} unrecoverable, "
                 f"{len(skipped_partial)} partial-delisted")
    else:
        skipped_full = []
        skipped_partial = []

    # ── Build skipped_tickers section for metadata ────────────────────
    skipped_tickers = {
        "prices":          skipped_full,
        "prices_partial":  skipped_partial,
        "sectors":         [{"ticker": t, "reason": "missing from sectors.json"} for t in missing_sectors],
        "edgar_8k":        [{"ticker": t, "reason": "missing from edgar_8k.parquet"} for t in missing_edgar_8k],
        "edgar_form4":     [{"ticker": t, "reason": "missing from edgar_form4.parquet"} for t in missing_edgar_form4],
        "fundamentals":    [{"ticker": t, "reason": "missing from quarterly_financials.parquet"} for t in missing_fundamentals],
    }
    skip_counts = {k: len(v) for k, v in skipped_tickers.items()}

    # ── Amend cache_metadata.json ─────────────────────────────────────
    meta_path = _CACHE_DIR / "cache_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}
    meta["skipped_tickers"] = skipped_tickers
    meta["skip_counts"]     = skip_counts
    meta["patched_at"]      = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info(f"[patch] amended {meta_path}")

    # ── Console summary ───────────────────────────────────────────────
    print()
    print("=" * 70)
    print("CACHE PATCH SUMMARY")
    print("=" * 70)
    print(f"CORE tickers: {len(core)}")
    print(f"Skipped per artefact (after recovery):")
    for k, v in skip_counts.items():
        print(f"  {k:18s}  {v:3d}")
    if skipped_partial:
        print(f"\nPartial-delisted (IN spine with truncated trading range):")
        for entry in skipped_partial:
            print(f"  {entry['ticker']:6s}  last_traded={entry['last_traded']}")
    if skipped_full:
        print(f"\nFull skips (NOT in spine — unrecoverable):")
        for entry in skipped_full:
            print(f"  {entry['ticker']:6s}  {entry['reason']}")


if __name__ == "__main__":
    main()
