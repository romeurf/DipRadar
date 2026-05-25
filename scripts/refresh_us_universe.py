"""
scripts/refresh_us_universe.py — Build us_universe_static.json from S&P 1500.

Sources (all Wikipedia, no login required):
  S&P 500       — ~500 large-cap US stocks
  S&P MidCap 400 — ~400 mid-cap US stocks
  S&P SmallCap 600 — ~600 small-cap US stocks

Together these form the S&P Composite 1500, the standard US total-market benchmark.
The $2B market cap floor is applied during bootstrap (config.MIN_MARKET_CAP_USD),
not here — small caps are included so the pipeline can apply its own filter.

Usage:
    python scripts/refresh_us_universe.py

Trigger manually or via /admin_refresh_us_universe bot command.
Never called in the hot path — the JSON is committed to the repo.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT   = Path(__file__).parent.parent
OUTPUT_FILE = REPO_ROOT / "us_universe_static.json"

_WIKIPEDIA_SOURCES = [
    {
        "name":    "S&P 500",
        "url":     "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "attr":    {"id": "constituents"},
        "col":     "Symbol",
    },
    {
        "name":    "S&P MidCap 400",
        "url":     "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        "attr":    None,
        "col":     "Symbol",
    },
    {
        "name":    "S&P SmallCap 600",
        "url":     "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
        "attr":    None,
        "col":     "Symbol",
    },
    {
        "name":    "Nasdaq 100",
        "url":     "https://en.wikipedia.org/wiki/Nasdaq-100",
        "attr":    None,
        "col":     "Ticker",
    },
]


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _scrape_wikipedia(source: dict) -> list[str]:
    """Fetch a Wikipedia page via requests (with User-Agent) and parse the ticker table."""
    import io
    import requests
    import pandas as pd
    try:
        resp = requests.get(source["url"], headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        kwargs: dict = {}
        if source["attr"]:
            kwargs["attrs"] = source["attr"]
        tables = pd.read_html(io.StringIO(resp.text), **kwargs)
        col = source["col"]
        for t in tables:
            if col in t.columns:
                tickers = t[col].dropna().astype(str).tolist()
                logger.info(f"  {source['name']}: {len(tickers)} tickers")
                return tickers
        raise ValueError(f"Column '{col}' not found in any table")
    except Exception as e:
        logger.warning(f"  {source['name']}: FAILED — {e}")
        return []


def _clean(tickers: list[str]) -> list[str]:
    """Normalise and deduplicate a raw ticker list."""
    seen: set[str] = set()
    result: list[str] = []
    for raw in tickers:
        t = raw.strip().upper().replace(".", "-")  # BRK.B → BRK-B (yfinance convention)
        if not t or t in ("-", "N/A", "CASH", "USD"):
            continue
        if len(t) > 7:
            continue
        # Allow letters, digits, hyphens — no dots (EU suffix), no slashes
        if not all(c.isalpha() or c.isdigit() or c == "-" for c in t):
            continue
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def main() -> None:
    logger.info("Building S&P Composite 1500 universe from Wikipedia...")
    raw_all: list[str] = []

    for source in _WIKIPEDIA_SOURCES:
        raw_all.extend(_scrape_wikipedia(source))

    clean = _clean(raw_all)
    logger.info(f"Total unique tickers after dedup: {len(clean)}")

    if len(clean) < 500:
        logger.error(
            f"Only {len(clean)} tickers — suspiciously low (Wikipedia blocked?). "
            "Aborting. us_universe_static.json NOT updated."
        )
        sys.exit(1)

    payload = {
        "source":    "S&P Composite 1500 (S&P 500 + S&P 400 + S&P 600) via Wikipedia",
        "generated": date.today().isoformat(),
        "note":      "$2B market cap floor is applied during bootstrap (config.MIN_MARKET_CAP_USD), not here.",
        "tickers":   clean,
    }

    OUTPUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"Written {len(clean)} tickers → {OUTPUT_FILE}")
    logger.info("Commit us_universe_static.json to the repository.")


if __name__ == "__main__":
    main()
