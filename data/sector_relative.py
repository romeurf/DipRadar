"""data/sector_relative.py — Stock return minus sector ETF return over a window.

Used by the dislocation scorer (sub-score #3, "drop attribution") to
classify whether a price drop is stock-specific (bad news → penalize)
or sector-wide (sentiment/macro → reward).

The signal: `sector_relative_return(ticker, sector, window_days)` returns
`ret_ticker - ret_sector_etf`. Negative numbers mean the ticker
underperformed its sector — that's the "broken alone" pattern the
dislocation scorer wants to penalize when caused by company-specific
news, but reward when it's the kind of forced selling that mean-reverts.

Sector ETF mapping mirrors `main.py:_SECTOR_ETF` (kept in sync; duplicated
here to avoid circular import — the new dislocation module imports this
helper from `data/`, and `main.py` imports from `data/` already).

Histograms cached in-process for the session: a single CORE universe
sweep would otherwise fetch the same XLK/XLV/etc series ~50× per scan.
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger("sector_relative")

# Mirror of main.py:_SECTOR_ETF. yfinance sector strings → SPDR sector ETF.
# Update both in sync if sector taxonomy changes.
SECTOR_TO_ETF: dict[str, str] = {
    "Technology":             "XLK",
    "Healthcare":             "XLV",
    "Communication Services": "XLC",
    "Financial Services":     "XLF",
    "Consumer Cyclical":      "XLY",
    "Consumer Defensive":     "XLP",
    "Industrials":            "XLI",
    "Energy":                 "XLE",
    "Utilities":              "XLU",
    "Real Estate":            "XLRE",
    "Basic Materials":        "XLB",
}

# In-process cache of ETF price histories. Single sweep fetches each ETF
# at most once per session.
_etf_hist_cache: dict[str, "object"] = {}


def _fetch_hist(ticker: str, period: str = "6mo"):
    """Fetch price history via yfinance. Returns pandas DataFrame or None."""
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if df is None or df.empty:
            return None
        return df
    except Exception as e:
        log.debug(f"[sector_relative] _fetch_hist({ticker}): {e}")
        return None


def sector_etf_for(sector: str) -> Optional[str]:
    """Return SPDR sector ETF ticker for a yfinance sector string, or None."""
    if not sector:
        return None
    return SECTOR_TO_ETF.get(sector)


def _return_over_window(hist, window_days: int) -> Optional[float]:
    """Return price return over the last `window_days` trading days.

    Returns None when history is too short or contains non-finite values.
    """
    if hist is None or hist.empty:
        return None
    close = hist["Close"]
    if len(close) < window_days + 1:
        return None
    try:
        latest = float(close.iloc[-1])
        prior = float(close.iloc[-window_days - 1])
    except (IndexError, ValueError):
        return None
    if prior <= 0:
        return None
    ret = latest / prior - 1.0
    if ret != ret:  # NaN guard
        return None
    return ret


def sector_relative_return(
    ticker: str,
    sector: str,
    window_days: int = 60,
    *,
    ticker_hist=None,
) -> Optional[float]:
    """Return `ret_ticker - ret_sector_etf` over `window_days`.

    Positive → ticker outperformed sector over the window.
    Negative → ticker underperformed sector (sentiment/forced selling
               candidate, or stock-specific breakdown — sub-score #3
               does the attribution layer on top).

    Returns None when:
      - sector unknown (not in SECTOR_TO_ETF)
      - either ticker or ETF history is missing / too short
      - prior close is non-positive

    Pass `ticker_hist` (already-fetched DataFrame) to avoid double-fetching
    when callers are already iterating with prices in hand.
    """
    etf = sector_etf_for(sector)
    if etf is None:
        return None

    if ticker_hist is None:
        ticker_hist = _fetch_hist(ticker)
    ticker_ret = _return_over_window(ticker_hist, window_days)
    if ticker_ret is None:
        return None

    etf_hist = _etf_hist_cache.get(etf)
    if etf_hist is None:
        etf_hist = _fetch_hist(etf)
        if etf_hist is not None:
            _etf_hist_cache[etf] = etf_hist
    etf_ret = _return_over_window(etf_hist, window_days)
    if etf_ret is None:
        return None

    return ticker_ret - etf_ret


def sector_relative_pair(
    ticker: str,
    sector: str,
    *,
    long_window: int = 60,
    short_window: int = 20,
    ticker_hist=None,
) -> tuple[Optional[float], Optional[float]]:
    """Convenience: return both windows in one call.

    Used by sub-score #3 which mixes 60d (primary) and 20d (tactical
    tiebreaker) per the plan:
      `0.7 * tanh((rel_60d) / 0.10) + 0.3 * tanh((rel_20d) / 0.05)`

    Single ticker history fetch shared between both windows. Sector ETF
    cache also shared via `sector_relative_return`.
    """
    if ticker_hist is None:
        ticker_hist = _fetch_hist(ticker)
    long_rel = sector_relative_return(
        ticker, sector, window_days=long_window, ticker_hist=ticker_hist,
    )
    short_rel = sector_relative_return(
        ticker, sector, window_days=short_window, ticker_hist=ticker_hist,
    )
    return long_rel, short_rel


def clear_cache() -> None:
    """Reset in-process ETF history cache. Called from daily scan boundary
    if needed (e.g. when a sector ETF's data goes stale intraday)."""
    _etf_hist_cache.clear()
