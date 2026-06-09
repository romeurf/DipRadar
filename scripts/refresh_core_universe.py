"""
scripts/refresh_core_universe.py — Self-updating CORE quality screen.

Scans the full US universe (~3,500 tickers) and identifies quality compounders
that pass a systematic screen. Merges with the existing core_universe.json,
respecting force_include / force_exclude overrides.

Quality screen criteria (any company passing ALL of these is included):
  - market_cap       >= MIN_MARKET_CAP  (default $5B — smaller than DIP's $2B floor
                                         but we still want established companies)
  - gross_margin     >= MIN_GROSS_MARGIN (default 35%)
  - revenue_growth   >= MIN_REV_GROWTH   (5y CAGR, default 5%)
  OR is a Dividend Aristocrat (25+ consecutive years of dividend increases)

Additionally, force_include overrides always stay in.
force_exclude overrides always stay out (even if they pass the screen).

Usage:
  python scripts/refresh_core_universe.py                  # full screen
  python scripts/refresh_core_universe.py --dry-run        # show what would change, don't write
  python scripts/refresh_core_universe.py --min-cap 10     # set market cap floor to $10B
  python scripts/refresh_core_universe.py --tickers AAPL,MSFT  # test specific tickers

Triggered via:
  /admin_refresh_core_universe  (Telegram bot command)
  Monthly cron (optional — add to scheduler in main.py)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("refresh_core_universe")

_UNIVERSE_FILE   = _REPO_ROOT / "core_universe.json"
_DEFAULT_MIN_CAP = 5_000_000_000   # $5B
_DEFAULT_GROSS_MARGIN = 35.0       # 35%
_DEFAULT_REV_GROWTH   =  5.0       # 5% 5y CAGR

# Known Dividend Aristocrats (S&P 500 members with 25+ consecutive years of dividend growth)
# Source: S&P Dividend Aristocrats Index. Updated 2026-05.
_DIVIDEND_ARISTOCRATS = {
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ADM", "APD", "ALB", "AMCR", "ATO",
    "ADP", "BDX", "BRO", "CAH", "CAT", "CVX", "CB", "CHD", "CINF", "CTAS",
    "CLX", "KO", "CL", "ED", "GPC", "GWW", "ETN", "FRT", "GD", "HON",
    "HRL", "ITW", "JKHY", "KVUE", "KMB", "LOW", "MCD", "MDT", "MKC",
    "NDSN", "NTRS", "O", "OTIS", "PBCT", "PNR", "PEP", "PG", "PH",
    "PPG", "PPL", "ROP", "SHW", "SWK", "SYY", "TROW", "TGT", "GIS",
    "UPS", "VFC", "WMT", "EXPD", "ESS", "WST", "WM", "RSG", "ECL",
    "EMR", "AMGN", "AXP", "BDX", "BLK", "DOV", "IBM", "ITW", "MCO",
    "SPGI", "TXN",
}


def _load_universe() -> dict:
    if _UNIVERSE_FILE.exists():
        return json.loads(_UNIVERSE_FILE.read_text(encoding="utf-8"))
    return {"tickers": [], "overrides": {"force_include": [], "force_exclude": []}}


def _fetch_quality_metrics(ticker: str) -> Optional[dict]:
    """Fetch fundamental quality metrics for a single ticker via yfinance."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}

        market_cap   = float(info.get("marketCap") or 0)
        gross_margin = float(info.get("grossMargins") or 0) * 100  # yfinance returns 0-1
        rev_growth   = float(info.get("revenueGrowth") or 0) * 100  # trailing 1y

        # 5y revenue CAGR approximation from financials
        try:
            fin = t.financials
            if fin is not None and not fin.empty and "Total Revenue" in fin.index:
                rev_row = fin.loc["Total Revenue"].dropna()
                if len(rev_row) >= 4:
                    oldest = float(rev_row.iloc[-1])
                    newest = float(rev_row.iloc[0])
                    n_years = len(rev_row) - 1
                    if oldest > 0 and newest > 0:
                        rev_growth_5y = ((newest / oldest) ** (1 / n_years) - 1) * 100
                    else:
                        rev_growth_5y = rev_growth
                else:
                    rev_growth_5y = rev_growth
            else:
                rev_growth_5y = rev_growth
        except Exception:
            rev_growth_5y = rev_growth

        # Dividend growth streak from dividends history
        div_streak = 0
        try:
            divs = t.dividends
            if divs is not None and not divs.empty:
                annual = divs.resample("Y").sum()
                streak = 0
                for i in range(len(annual) - 1, 0, -1):
                    if annual.iloc[i] > annual.iloc[i - 1]:
                        streak += 1
                    else:
                        break
                div_streak = streak
        except Exception:
            pass

        # Current dividend yield
        div_yield = float(info.get("dividendYield") or 0) * 100

        return {
            "ticker":         ticker,
            "market_cap":     market_cap,
            "gross_margin":   round(gross_margin, 2),
            "rev_growth_5y":  round(rev_growth_5y, 2),
            "div_streak":     div_streak,
            "div_yield":      round(div_yield, 2),
            "sector":         info.get("sector", ""),
            "name":           info.get("longName", ticker),
        }
    except Exception as e:
        log.debug(f"  {ticker}: metrics fetch failed — {e}")
        return None


def _passes_screen(
    m: dict,
    min_cap: float,
    min_gross_margin: float,
    min_rev_growth: float,
) -> tuple[bool, str]:
    """Return (passes, reason_string)."""
    ticker = m["ticker"]

    if m["market_cap"] < min_cap:
        return False, f"market_cap ${m['market_cap']/1e9:.1f}B < ${min_cap/1e9:.0f}B floor"

    # Dividend Aristocrat bypass — if 25+ years of consecutive increases, include regardless
    if ticker in _DIVIDEND_ARISTOCRATS or m["div_streak"] >= 25:
        return True, f"Dividend Aristocrat ({m['div_streak']} years)"

    # Quality screen
    if m["gross_margin"] >= min_gross_margin and m["rev_growth_5y"] >= min_rev_growth:
        return True, f"quality screen (GM={m['gross_margin']:.0f}%, rev_growth={m['rev_growth_5y']:.1f}%)"

    return False, (
        f"screen failed: GM={m['gross_margin']:.0f}% (need {min_gross_margin:.0f}%), "
        f"rev_growth={m['rev_growth_5y']:.1f}% (need {min_rev_growth:.0f}%)"
    )


def run_refresh(
    tickers_override: Optional[list[str]] = None,
    dry_run: bool = False,
    min_cap: float = _DEFAULT_MIN_CAP,
    min_gross_margin: float = _DEFAULT_GROSS_MARGIN,
    min_rev_growth: float = _DEFAULT_REV_GROWTH,
) -> dict:
    """
    Run the CORE quality screen and update core_universe.json.

    Returns summary dict:
      {added: [...], removed: [...], kept: int, skipped: int, errors: int, dry_run: bool}
    """
    data = _load_universe()
    existing = {t["ticker"]: t for t in data.get("tickers", [])}
    overrides = data.get("overrides", {})
    force_include = set(overrides.get("force_include", []))
    force_exclude = set(overrides.get("force_exclude", []))

    # Determine which tickers to screen
    if tickers_override:
        to_screen = list(tickers_override)
    else:
        # Full universe from universe.py (same ~3,500 as DIP/Momentum)
        try:
            from core.universe import get_ml_universe
            to_screen = get_ml_universe()
            log.info(f"Full universe: {len(to_screen)} tickers to screen")
        except Exception as e:
            log.warning(f"Could not load universe module ({e}) — using existing CORE list only")
            to_screen = list(existing.keys())

    n_total  = len(to_screen)
    passed   = {}   # ticker -> (metrics, reason)
    skipped  = 0
    errors   = 0

    for i, ticker in enumerate(to_screen):
        if i % 50 == 0:
            log.info(f"  [{i+1}/{n_total}] screening… (passed so far: {len(passed)})")

        if ticker in force_exclude:
            log.debug(f"  {ticker}: force_excluded")
            continue

        m = _fetch_quality_metrics(ticker)
        if m is None:
            errors += 1
            continue

        ok, reason = _passes_screen(m, min_cap, min_gross_margin, min_rev_growth)
        if ok:
            passed[ticker] = (m, reason)
            log.debug(f"  {ticker}: PASS — {reason}")
        else:
            log.debug(f"  {ticker}: FAIL — {reason}")
            skipped += 1

        time.sleep(0.05)  # gentle rate limiting

    # Add force_includes that weren't in the screen results
    for ticker in force_include:
        if ticker not in passed:
            if ticker in existing:
                passed[ticker] = (None, "force_include override")
            else:
                m = _fetch_quality_metrics(ticker)
                passed[ticker] = (m, "force_include override")

    # Compute diff
    added   = [t for t in passed if t not in existing]
    removed = [t for t in existing if t not in passed and t not in force_include]

    log.info(f"Screen complete: {len(passed)} pass, {skipped} fail, {errors} errors")
    log.info(f"Changes: +{len(added)} added, -{len(removed)} removed")

    if dry_run:
        log.info("[dry-run] Not writing — pass --no-dry-run to apply")
        return {
            "added":   added,
            "removed": removed,
            "kept":    len(passed) - len(added),
            "skipped": skipped,
            "errors":  errors,
            "dry_run": True,
        }

    # Build updated tickers list
    new_tickers = []
    for ticker, (m, reason) in passed.items():
        if ticker in existing:
            # Preserve existing entry but update yield if we have fresh data
            entry = dict(existing[ticker])
            if m and m.get("div_yield"):
                entry["annual_yield_pct"] = round(m["div_yield"], 2)
            new_tickers.append(entry)
        else:
            # New entry from screen
            div_yield = m["div_yield"] if m else 0.0
            name = m["name"] if m else ticker
            gm   = m["gross_margin"] if m else 0
            new_tickers.append({
                "ticker":           ticker,
                "annual_yield_pct": round(div_yield, 2),
                "note":             f"Auto-added by quality screen — {name} (GM {gm:.0f}%)",
            })

    new_tickers.sort(key=lambda x: x["ticker"])

    data["tickers"] = new_tickers
    data["_meta"]["generated"]   = __import__("datetime").date.today().isoformat()
    data["_meta"]["count"]       = len(new_tickers)
    data["_meta"]["source"]      = (
        f"Quality screen: gross_margin>={min_gross_margin:.0f}%, "
        f"rev_growth>={min_rev_growth:.0f}%, market_cap>=${min_cap/1e9:.0f}B. "
        f"Auto-refreshed {data['_meta']['generated']}."
    )

    _UNIVERSE_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info(f"Written {len(new_tickers)} tickers to {_UNIVERSE_FILE}")

    return {
        "added":   added,
        "removed": removed,
        "kept":    len(passed) - len(added),
        "skipped": skipped,
        "errors":  errors,
        "dry_run": False,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Refresh CORE quality universe")
    parser.add_argument("--dry-run",      action="store_true", help="Show changes without writing")
    parser.add_argument("--tickers",      help="Comma-separated list to screen (test mode)")
    parser.add_argument("--min-cap",      type=float, default=5.0, help="Min market cap in $B (default 5)")
    parser.add_argument("--min-gm",       type=float, default=35.0, help="Min gross margin %% (default 35)")
    parser.add_argument("--min-rev",      type=float, default=5.0,  help="Min 5y revenue growth %% (default 5)")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None

    result = run_refresh(
        tickers_override=tickers,
        dry_run=args.dry_run,
        min_cap=args.min_cap * 1e9,
        min_gross_margin=args.min_gm,
        min_rev_growth=args.min_rev,
    )

    print(f"\nResult:")
    print(f"  Added:   {len(result['added'])} — {result['added'][:20]}")
    print(f"  Removed: {len(result['removed'])} — {result['removed'][:20]}")
    print(f"  Kept:    {result['kept']}")
    print(f"  Skipped: {result['skipped']} (failed screen)")
    print(f"  Errors:  {result['errors']} (data fetch failed)")
    if result["dry_run"]:
        print("\n  [DRY RUN] — run without --dry-run to apply changes")
