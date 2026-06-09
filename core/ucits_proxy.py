"""ucits_proxy.py — Derive UCITS metrics from their US proxy ETFs.

yfinance has no UCITS data (SEC0, SXR8, IS3N, etc. all return 404). The
sector_radar already maps UCITS ↔ US proxy for scoring; this module extends
the pattern to PnL estimation:

  - SEC0 (iShares MSCI EM Semis UCITS)  → tracked via SOXX (US semis)
  - SXR8 (iShares Core S&P 500 UCITS)   → tracked via SPY
  - IS3N (iShares Core MSCI EM IMI UCITS) → tracked via EEM

Caveats:
  - UCITS NAV in EUR vs proxy USD diverges with FX moves. The proxy %-change
    is a best-effort estimate; the real UCITS movement can deviate by
    the EUR/USD drift over the same window.
  - Some UCITS use representative sampling vs the proxy's full replication,
    introducing additional tracking error.

The proxy estimate is good enough for: tax friction math, rotation
decisions, monthly check display. Not good enough for: precise PnL
calculations for tax declaration purposes.
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)


def get_us_proxy(ucits_ticker: str) -> Optional[str]:
    """Return the US ETF ticker that proxies this UCITS, or None.

    Reverse lookup over USER_HOLDINGS_UCITS. Case-insensitive.
    """
    from core.etf_universe import USER_HOLDINGS_UCITS
    return USER_HOLDINGS_UCITS.get(ucits_ticker.upper())


def fetch_proxy_return_pct(
    ucits_ticker: str,
    lookback_days: int = 30,
) -> Optional[float]:
    """Return the US proxy's percent change over `lookback_days`.

    Returns None if no proxy mapping or data fetch fails. Output is a
    decimal (e.g., 0.05 for +5%). Used as a proxy for the UCITS's PnL
    direction over that window.
    """
    proxy = get_us_proxy(ucits_ticker)
    if not proxy:
        return None
    try:
        import yfinance as yf
        # Fetch slightly more than the lookback to absorb non-trading days
        hist = yf.Ticker(proxy).history(period=f"{max(lookback_days + 10, 60)}d", auto_adjust=True)
        if hist is None or len(hist) < 2:
            return None
        closes = hist["Close"]
        if len(closes) <= lookback_days:
            start_px = float(closes.iloc[0])
        else:
            start_px = float(closes.iloc[-(lookback_days + 1)])
        end_px = float(closes.iloc[-1])
        if start_px <= 0:
            return None
        return end_px / start_px - 1.0
    except Exception as e:
        log.debug(f"[ucits_proxy] {ucits_ticker} → {proxy} fetch failed: {e}")
        return None


def estimate_ucits_current_price(
    ucits_ticker: str,
    avg_price_eur: float,
    entry_date_iso: Optional[str] = None,
) -> Optional[float]:
    """Estimate the UCITS's current EUR price using its US proxy's % change.

    Args:
      ucits_ticker: e.g., "SEC0"
      avg_price_eur: user's cost basis per share in EUR
      entry_date_iso: ISO date of entry, e.g. "2026-05-15". If None or
        unparseable, uses a 30-day lookback as default.

    Returns: estimated current EUR price, or None if proxy data unavailable.
    """
    if avg_price_eur <= 0:
        return None
    proxy = get_us_proxy(ucits_ticker)
    if not proxy:
        return None
    try:
        from datetime import date, datetime
        import yfinance as yf

        # Determine lookback window. If entry_date provided, fetch since then.
        # Otherwise use last 30 days as a conservative recent direction.
        if entry_date_iso:
            try:
                entry_dt = datetime.fromisoformat(entry_date_iso).date()
                days_held = max(1, (date.today() - entry_dt).days)
            except ValueError:
                days_held = 30
        else:
            days_held = 30

        # Fetch history covering the holding period
        period_str = f"{max(days_held + 10, 60)}d"
        hist = yf.Ticker(proxy).history(period=period_str, auto_adjust=True)
        if hist is None or hist.empty:
            return None
        closes = hist["Close"]

        # Find the index closest to entry_date (or `days_held` ago)
        if entry_date_iso:
            try:
                import pandas as pd
                target_ts = pd.Timestamp(entry_date_iso)
                if hist.index.tz is not None:
                    hist_naive = closes.copy()
                    hist_naive.index = hist_naive.index.tz_localize(None)
                else:
                    hist_naive = closes
                before = hist_naive[hist_naive.index <= target_ts]
                if not before.empty:
                    entry_px = float(before.iloc[-1])
                else:
                    entry_px = float(hist_naive.iloc[0])
            except Exception:
                # Fallback: position by index
                if len(closes) > days_held:
                    entry_px = float(closes.iloc[-(days_held + 1)])
                else:
                    entry_px = float(closes.iloc[0])
        else:
            if len(closes) > days_held:
                entry_px = float(closes.iloc[-(days_held + 1)])
            else:
                entry_px = float(closes.iloc[0])

        current_px = float(closes.iloc[-1])
        if entry_px <= 0:
            return None
        proxy_pct_change = current_px / entry_px - 1.0
        # Apply to UCITS avg_price to estimate current
        return avg_price_eur * (1.0 + proxy_pct_change)
    except Exception as e:
        log.debug(f"[ucits_proxy] {ucits_ticker} estimate failed: {e}")
        return None
