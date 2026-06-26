"""scripts/validation_cache_builder.py — Pre-fetch all data for the F.5 backtest.

The dislocation backtest must be reproducible byte-for-byte. The cache
is the input that guarantees reproducibility: once written, the spine
builder and backtest runner read from disk only, never from network.

Pre-fetched artifacts (written to /data/validation_cache/):
  prices.parquet           — 6y daily OHLCV for all CORE tickers + sector ETFs + SPY
  sectors.json             — ticker → yfinance sector string
  edgar_8k.parquet         — 8-K filings (date, items) per ticker
  edgar_form4.parquet      — Form 4 insider transactions (date, role, $)
  quarterly_financials.parquet  — quarterly statements per ticker (revenue, margins, etc.)
  cache_metadata.json      — fetched_at, ticker count, missing tickers, sources

Why batch yfinance.download over ThreadPool per-ticker (per F.5 user review):
  - ThreadPool corruption-by-rate-limit is silent: a few NaN rows in
    230k can pass review but bias the gate
  - yfinance.download supports multi-ticker single-request natively
  - SEC EDGAR has its own 10-req/sec ceiling; we throttle sequentially

Resume semantics: if any artifact already exists in the cache dir, the
builder skips that step. Re-run after a partial failure picks up where
it left off. Pass --force to wipe the cache and rebuild from scratch.

Runtime: ~30 min for the full sweep (prices + EDGAR + fundamentals).
Output size: ~25 MB total.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("validation_cache")

# Output directory — Railway volume in production, local data/ otherwise.
_DATA_DIR = Path("/data") if Path("/data").exists() else _ROOT / "data"
_CACHE_DIR = _DATA_DIR / "validation_cache"

# Window: 2020-Q4 to 2026-Q2. We pull 6 years to have a 6-month buffer
# at both ends — the spine needs forward returns for 2025-12 entries
# (uses 2026-06 prices) and the earliest entry (2021-01) needs prior
# 60d window (uses 2020-11 prices).
_FETCH_START_ISO = "2020-10-01"
_FETCH_END_ISO   = "2026-06-30"

# Sector ETFs (mirror data/sector_relative.SECTOR_TO_ETF values)
_SECTOR_ETFS = ["XLK", "XLV", "XLC", "XLF", "XLY", "XLP", "XLI", "XLE", "XLU", "XLRE", "XLB"]

# Benchmark for baseline returns
_BENCHMARK_TICKERS = ["SPY"]

# EDGAR throttle. SEC ceiling is 10/sec; we use 5/sec to leave generous
# margin for a one-shot build (the +2 min cost vs 0.15s is irrelevant
# next to the protection against a 429 that would force the build to
# restart in the middle).
_EDGAR_THROTTLE_S = 0.20


def _load_core_universe() -> list[str]:
    universe_path = _ROOT / "core_universe.json"
    with universe_path.open(encoding="utf-8") as f:
        data = json.load(f)
    tickers = [t["ticker"] for t in data.get("tickers", [])]
    log.info(f"[universe] loaded {len(tickers)} CORE tickers from {universe_path}")
    return tickers


def _all_tickers_to_fetch(core: list[str]) -> list[str]:
    """Union of CORE + sector ETFs + benchmarks (deduplicated, ordered)."""
    seen = set()
    out = []
    for group in (core, _SECTOR_ETFS, _BENCHMARK_TICKERS):
        for t in group:
            if t not in seen:
                out.append(t)
                seen.add(t)
    return out


# ── Step 1: prices ───────────────────────────────────────────────────

def fetch_prices(all_tickers: list[str], force: bool = False) -> Path:
    """Batch-download daily OHLCV for all tickers via yfinance.

    Single multi-ticker call. yfinance returns a long-form DataFrame
    with MultiIndex columns (level 0 = field, level 1 = ticker). We
    melt to long format (date, ticker, open, high, low, close, volume,
    adj_close) for cleaner parquet I/O downstream.
    """
    out_path = _CACHE_DIR / "prices.parquet"
    if out_path.exists() and not force:
        log.info(f"[prices] cache hit: {out_path} (use --force to rebuild)")
        return out_path

    import yfinance as yf
    log.info(
        f"[prices] downloading {len(all_tickers)} tickers "
        f"({_FETCH_START_ISO} → {_FETCH_END_ISO})"
    )
    df = yf.download(
        tickers=all_tickers,
        start=_FETCH_START_ISO,
        end=_FETCH_END_ISO,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError("yfinance.download returned empty DataFrame")

    # Convert MultiIndex columns (ticker, field) into a tidy long table.
    import pandas as pd
    rows = []
    for ticker in all_tickers:
        try:
            sub = df[ticker] if isinstance(df.columns, pd.MultiIndex) else df
            sub = sub.dropna(how="all")
            if sub.empty:
                log.warning(f"[prices] empty for {ticker} — skip")
                continue
            for idx, row in sub.iterrows():
                rows.append({
                    "date":   idx.date().isoformat(),
                    "ticker": ticker,
                    "open":   float(row.get("Open"))   if row.get("Open")   is not None else None,
                    "high":   float(row.get("High"))   if row.get("High")   is not None else None,
                    "low":    float(row.get("Low"))    if row.get("Low")    is not None else None,
                    "close":  float(row.get("Close"))  if row.get("Close")  is not None else None,
                    "volume": float(row.get("Volume")) if row.get("Volume") is not None else None,
                })
        except Exception as e:
            log.warning(f"[prices] {ticker}: {e}")

    out = pd.DataFrame(rows)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    log.info(f"[prices] persisted {len(out)} rows to {out_path}")
    return out_path


# ── Step 2: sectors ──────────────────────────────────────────────────

def fetch_sectors(core: list[str], force: bool = False) -> Path:
    """Per-ticker sector lookup via yfinance.Ticker.info.sector.

    Sequential to respect implicit rate-limits on .info (it's a heavier
    endpoint than .history). ~3 min for 230 tickers.
    """
    out_path = _CACHE_DIR / "sectors.json"
    if out_path.exists() and not force:
        log.info(f"[sectors] cache hit: {out_path}")
        return out_path

    import yfinance as yf
    sectors = {}
    log.info(f"[sectors] resolving {len(core)} ticker sectors")
    for i, t in enumerate(core, 1):
        try:
            info = yf.Ticker(t).info or {}
            sectors[t] = info.get("sector", "") or ""
        except Exception as e:
            log.warning(f"[sectors] {t}: {e}")
            sectors[t] = ""
        if i % 25 == 0:
            log.info(f"[sectors] {i}/{len(core)}")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sectors, indent=2))
    n_resolved = sum(1 for s in sectors.values() if s)
    log.info(f"[sectors] resolved {n_resolved}/{len(core)} sectors → {out_path}")
    return out_path


# ── Step 3: EDGAR (8-K + Form 4) ─────────────────────────────────────

def fetch_edgar(core: list[str], force: bool = False) -> tuple[Path, Path]:
    """Pre-fetch EDGAR submissions + parse Form 4 XMLs for direction.

    Reuses the existing infrastructure in `data.fundamental_signals`:
      - `_edgar_submissions` provides submissions JSON with 7-day TTL
      - `_parse_form4_xml` has a PERMANENT disk cache (Form 4s are
        immutable). The cache lives at /data/form4_cache/ and survives
        across runs — subsequent re-runs of this script are instant.

    Critical: Form 4 boolean alone would be directionally noisy because
    it includes both BUYS (insider confidence = positive signal) and
    SALES (red flag = negative signal). Per F.5 user review, we MUST
    extract the direction (is_sale, is_purchase, sale_amount_usd) so
    #7 in the spine penalizes SALES only — not a buy that would invert
    the signal in production.

    First-run runtime: ~30-60 min for Form 4 XML parses (bounded by
    EDGAR rate-limit + per-filing XML fetch). Re-runs: instant due to
    permanent cache.
    """
    out_8k    = _CACHE_DIR / "edgar_8k.parquet"
    out_form4 = _CACHE_DIR / "edgar_form4.parquet"
    if out_8k.exists() and out_form4.exists() and not force:
        log.info(f"[edgar] cache hit: {out_8k} + {out_form4}")
        return out_8k, out_form4

    from data.fundamental_signals import _edgar_submissions, _parse_form4_xml
    import pandas as pd

    rows_8k = []
    rows_form4 = []
    log.info(
        f"[edgar] fetching submissions + Form 4 parses for {len(core)} tickers "
        f"(first run: 30-60 min via permanent Form 4 cache)"
    )

    n_f4_parsed = 0
    n_f4_failed = 0

    for i, t in enumerate(core, 1):
        try:
            sub_data = _edgar_submissions(t) or {}
            # CIK is needed for Form 4 XML lookup; submissions JSON has it
            try:
                cik_int = int(sub_data.get("cik", 0) or 0)
            except Exception:
                cik_int = 0

            recent = sub_data.get("filings", {}).get("recent", {})
            forms        = recent.get("form", [])
            dates_filed  = recent.get("filingDate", [])
            items_list   = recent.get("items", [])
            primary_docs = recent.get("primaryDocument", [])
            accessions   = recent.get("accessionNumber", [])
            n = min(len(forms), len(dates_filed))
            for j in range(n):
                form = forms[j]
                filed = dates_filed[j][:10] if dates_filed[j] else ""
                if form in ("8-K", "8-K/A"):
                    items = items_list[j] if j < len(items_list) else ""
                    rows_8k.append({
                        "ticker":     t,
                        "filed":      filed,
                        "form":       form,
                        "items":      items if isinstance(items, str) else ", ".join(map(str, items)),
                        "accession":  accessions[j] if j < len(accessions) else "",
                    })
                elif form == "4" and cik_int > 0:
                    accession = accessions[j] if j < len(accessions) else ""
                    primary   = primary_docs[j] if j < len(primary_docs) else ""
                    if not accession:
                        continue
                    # Parse Form 4 XML to extract direction (S/D for SALE, A/P for BUY).
                    # Hits the permanent cache after first run.
                    parsed = _parse_form4_xml(accession, str(cik_int), primary) or {}
                    if not parsed:
                        n_f4_failed += 1
                        # Cache failure or fetch failure — record but flag unknown
                        rows_form4.append({
                            "ticker":          t,
                            "filed":           filed,
                            "accession":       accession,
                            "title":           "",
                            "is_sale":         False,
                            "is_purchase":     False,
                            "sale_amount_usd": 0.0,
                            "buy_amount_usd":  0.0,
                            "direction":       "unknown",
                        })
                        continue
                    n_f4_parsed += 1
                    is_sale  = bool(parsed.get("is_sale", False))
                    is_buy   = bool(parsed.get("is_purchase", False))
                    direction = (
                        "sale" if (is_sale and not is_buy) else
                        "buy"  if (is_buy and not is_sale) else
                        "both" if (is_sale and is_buy)     else
                        "neither"
                    )
                    rows_form4.append({
                        "ticker":          t,
                        "filed":           filed,
                        "accession":       accession,
                        "title":           parsed.get("title", "") or "",
                        "is_sale":         is_sale,
                        "is_purchase":     is_buy,
                        "sale_amount_usd": float(parsed.get("sale_amount_usd", 0) or 0),
                        "buy_amount_usd":  float(parsed.get("amount_usd", 0) or 0),
                        "direction":       direction,
                    })
            if i % 10 == 0:
                log.info(
                    f"[edgar] {i}/{len(core)}  "
                    f"(8-K={len(rows_8k)}, F4_parsed={n_f4_parsed}, F4_failed={n_f4_failed})"
                )
        except Exception as e:
            log.warning(f"[edgar] {t}: {e}")
        time.sleep(_EDGAR_THROTTLE_S)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_8k).to_parquet(out_8k, index=False)
    pd.DataFrame(rows_form4).to_parquet(out_form4, index=False)
    log.info(f"[edgar] 8-K rows: {len(rows_8k)} → {out_8k}")
    log.info(
        f"[edgar] Form 4 rows: {len(rows_form4)} "
        f"(parsed={n_f4_parsed}, failed={n_f4_failed}, "
        f"failure_rate={100*n_f4_failed/max(1,n_f4_parsed+n_f4_failed):.1f}%) → {out_form4}"
    )
    return out_8k, out_form4


# ── Step 4: quarterly fundamentals ───────────────────────────────────

def fetch_fundamentals(core: list[str], force: bool = False) -> Path:
    """Quarterly financial statements per ticker.

    yfinance returns ~5y of quarterly history via .quarterly_financials,
    .quarterly_balance_sheet, .quarterly_cashflow. Extract the metrics
    sub-score #6 (quality) needs: revenue, gross margin, ROE, FCF yield.

    Note: quarterly data is reported with a delay; "as-of" alignment in
    the spine uses the filing period_of_report, NOT the print date.
    Conservative: lag each quarter's data by 60 days to simulate the
    typical filing delay (10-Q within 40 days of quarter-end + buffer).
    """
    out_path = _CACHE_DIR / "quarterly_financials.parquet"
    if out_path.exists() and not force:
        log.info(f"[fundamentals] cache hit: {out_path}")
        return out_path

    import yfinance as yf
    import pandas as pd
    from datetime import timedelta

    rows = []
    log.info(f"[fundamentals] fetching quarterly financials for {len(core)} tickers")
    for i, t in enumerate(core, 1):
        try:
            tk = yf.Ticker(t)
            q_fin = tk.quarterly_financials
            q_bal = tk.quarterly_balance_sheet
            q_cf  = tk.quarterly_cashflow
            if q_fin is None or q_fin.empty:
                continue
            for period in q_fin.columns:
                try:
                    period_date = period.date() if hasattr(period, "date") else period
                    period_iso = period_date.isoformat() if hasattr(period_date, "isoformat") else str(period_date)
                    # Lag by 60d to simulate filing delay (PIT-safe for spine)
                    pit_date_iso = (period_date + timedelta(days=60)).isoformat() if hasattr(period_date, "isoformat") else period_iso

                    revenue       = _safe_get(q_fin, "Total Revenue", period)
                    gross_profit  = _safe_get(q_fin, "Gross Profit", period)
                    net_income    = _safe_get(q_fin, "Net Income", period)
                    equity        = _safe_get(q_bal, "Total Stockholder Equity", period) if q_bal is not None else None
                    fcf           = _safe_get(q_cf, "Free Cash Flow", period) if q_cf is not None else None

                    gross_margin = (gross_profit / revenue) if (revenue and gross_profit and revenue > 0) else None
                    roe          = (net_income / equity) if (equity and net_income and equity > 0) else None

                    rows.append({
                        "ticker":         t,
                        "period_end":     period_iso,
                        "pit_date":       pit_date_iso,
                        "revenue":        float(revenue) if revenue is not None else None,
                        "gross_profit":   float(gross_profit) if gross_profit is not None else None,
                        "net_income":     float(net_income) if net_income is not None else None,
                        "equity":         float(equity) if equity is not None else None,
                        "free_cash_flow": float(fcf) if fcf is not None else None,
                        "gross_margin":   float(gross_margin) if gross_margin is not None else None,
                        "roe":            float(roe) if roe is not None else None,
                    })
                except Exception as e:
                    log.debug(f"[fundamentals] {t} period {period}: {e}")
        except Exception as e:
            log.warning(f"[fundamentals] {t}: {e}")
        if i % 25 == 0:
            log.info(f"[fundamentals] {i}/{len(core)}  (rows so far: {len(rows)})")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    log.info(f"[fundamentals] persisted {len(rows)} rows to {out_path}")
    return out_path


def _safe_get(df, row_name, col):
    """Lookup a row by name in a yfinance financials DataFrame; None on missing."""
    try:
        if row_name in df.index:
            v = df.loc[row_name, col]
            return float(v) if v == v else None  # NaN guard
    except Exception:
        pass
    return None


# ── Metadata ─────────────────────────────────────────────────────────

def write_metadata(core: list[str], all_tickers: list[str]) -> Path:
    """Persist cache provenance for reproducibility audits."""
    meta_path = _CACHE_DIR / "cache_metadata.json"
    meta = {
        "built_at":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fetch_window":     {"start": _FETCH_START_ISO, "end": _FETCH_END_ISO},
        "n_core_tickers":   len(core),
        "n_total_tickers":  len(all_tickers),
        "sector_etfs":      _SECTOR_ETFS,
        "benchmarks":       _BENCHMARK_TICKERS,
        "core_universe_git_sha": _get_git_sha("core_universe.json"),
        "this_script_git_sha":   _get_git_sha(__file__),
        "sources": {
            "prices":       "yfinance.download (multi-ticker batch)",
            "sectors":      "yfinance.Ticker(t).info.sector",
            "edgar_8k":     "SEC EDGAR submissions endpoint via data.fundamental_signals._edgar_submissions",
            "edgar_form4":  "same",
            "fundamentals": "yfinance.Ticker(t).quarterly_financials/balance_sheet/cashflow",
        },
    }
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info(f"[meta] persisted cache metadata to {meta_path}")
    return meta_path


def _get_git_sha(path) -> str:
    """Return short git SHA of last commit touching `path`, or 'unknown'."""
    try:
        import subprocess
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(path)],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip()[:12] or "unknown"
    except Exception:
        return "unknown"


# ── Entrypoint ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true",
                        help="Wipe each cache artifact and rebuild from scratch.")
    parser.add_argument("--skip-prices", action="store_true")
    parser.add_argument("--skip-sectors", action="store_true")
    parser.add_argument("--skip-edgar", action="store_true")
    parser.add_argument("--skip-fundamentals", action="store_true")
    args = parser.parse_args()

    core = _load_core_universe()
    all_tickers = _all_tickers_to_fetch(core)

    log.info(f"[builder] CORE={len(core)}, total_fetch={len(all_tickers)}")
    log.info(f"[builder] cache dir: {_CACHE_DIR}")

    if not args.skip_prices:
        fetch_prices(all_tickers, force=args.force)
    if not args.skip_sectors:
        fetch_sectors(core, force=args.force)
    if not args.skip_edgar:
        fetch_edgar(core, force=args.force)
    if not args.skip_fundamentals:
        fetch_fundamentals(core, force=args.force)

    write_metadata(core, all_tickers)
    log.info("[builder] DONE")


if __name__ == "__main__":
    main()
