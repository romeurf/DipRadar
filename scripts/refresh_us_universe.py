"""
scripts/refresh_us_universe.py — Refresh us_universe_static.json from iShares ITOT holdings.

Downloads the iShares ITOT (S&P Total US Stock Market ETF) holdings CSV,
filters to US equity tickers (no dots, min $2B market cap), and writes
us_universe_static.json in the repo root.

Usage:
    python scripts/refresh_us_universe.py

Trigger manually or via /admin_refresh_us_universe bot command.
Never called in the hot path — the JSON is committed to the repo and loaded statically.
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

# iShares ITOT holdings CSV URL (S&P Total US Stock Market ETF)
ITOT_URL = "https://www.ishares.com/us/products/239724/ISHARES-CORE-SP-TOTAL-US-STOCK-MARKET-ETF/1467271812596.ajax?fileType=csv&fileName=ITOT_holdings&dataType=fund"

# Minimum market cap filter applied at bootstrap time (not here), but we log it
MIN_MKTCAP_NOTE = "$2B floor applied during bootstrap (config.MIN_MARKET_CAP_USD), not here"


def _fetch_itot_holdings() -> list[str]:
    """Download ITOT holdings CSV and extract ticker symbols."""
    import pandas as pd
    import requests

    logger.info(f"Downloading ITOT holdings from iShares...")
    try:
        resp = requests.get(ITOT_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        from io import StringIO
        # iShares CSV has a multi-line header; data starts after 'Ticker' column
        text = resp.text
        # Find the line containing "Ticker"
        lines = text.splitlines()
        header_idx = next((i for i, ln in enumerate(lines) if "Ticker" in ln), None)
        if header_idx is None:
            raise ValueError("Could not find 'Ticker' column in ITOT CSV")
        csv_body = "\n".join(lines[header_idx:])
        df = pd.read_csv(StringIO(csv_body))
        if "Ticker" not in df.columns:
            raise ValueError(f"Columns: {list(df.columns)}")
        tickers = df["Ticker"].dropna().astype(str).tolist()
        logger.info(f"ITOT: {len(tickers)} rows downloaded")
        return tickers
    except Exception as e:
        logger.error(f"ITOT download failed: {e}")
        return []


def _filter_us_equity(tickers: list[str]) -> list[str]:
    """Keep only clean US equity tickers (no dots, no cash/currency entries)."""
    result = []
    for t in tickers:
        t = t.strip().upper()
        if not t:
            continue
        if "." in t:
            continue   # EU/UK exchange suffix
        if len(t) > 6:
            continue   # unlikely to be a valid US ticker
        if t in ("-", "N/A", "CASH", "USD", ""):
            continue
        # Allow hyphens (BRK-B, BF-B) but reject other special characters
        if not all(c.isalpha() or c == "-" for c in t):
            continue
        result.append(t)
    # Deduplicate preserving order
    seen: set[str] = set()
    unique = []
    for t in result:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def main() -> None:
    tickers = _fetch_itot_holdings()

    if not tickers:
        logger.error("No tickers fetched — aborting. us_universe_static.json NOT updated.")
        sys.exit(1)

    clean = _filter_us_equity(tickers)
    logger.info(f"After filtering: {len(clean)} US equity tickers")

    if len(clean) < 500:
        logger.error(
            f"Only {len(clean)} tickers after filtering — suspiciously low. "
            f"Check the CSV format. Aborting."
        )
        sys.exit(1)

    payload = {
        "source": "iShares ITOT (S&P Total US Stock Market ETF)",
        "generated": date.today().isoformat(),
        "note": MIN_MKTCAP_NOTE,
        "tickers": clean,
    }

    OUTPUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"Written {len(clean)} tickers to {OUTPUT_FILE}")
    logger.info("Commit us_universe_static.json to the repository to make it available in production.")


if __name__ == "__main__":
    main()
