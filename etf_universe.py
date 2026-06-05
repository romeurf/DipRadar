"""ETF universe + UCITS ↔ US mapping for sector_radar.

The radar uses US-listed ETFs (XLK, SPY, SOXX, ...) for data because they
have decades of clean OHLCV history and tight bid-ask. The user buys EU
UCITS equivalents (SXR8, SEC0, IS3N, ...) through a EU broker.

When an alert fires for a US ticker, the message includes the UCITS
equivalent so the user knows which to buy. If no UCITS equivalent is
mapped here, the US ticker is shown directly (user can buy that on
Trading 212 / IBKR Europe).

Categories:
- "sector"      — 11 SPDR Select Sector ETFs (US slice by sector)
- "broad"       — Total-market / index ETFs (SPY, QQQ, IWM)
- "international" — EM, Europe, world ex-US
- "thematic"    — Semis, biotech, etc. that aren't pure sector slices
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ETF:
    ticker: str            # US ticker — canonical for data fetch
    name: str
    category: str          # "sector" | "broad" | "international" | "thematic"
    sector: str | None = None     # GICS sector (only for sector ETFs)
    ucits_ticker: str | None = None
    ucits_name: str | None = None


# ─── SPDR Select Sector ETFs (11 sectors) ────────────────────────────────────
# These are the canonical sector slices used by the radar. Long history
# back to 1998 (most) — great for backtesting.
US_SECTOR_ETFS: list[ETF] = [
    ETF("XLK", "Technology Select Sector SPDR",
        "sector", sector="Technology",
        ucits_ticker="IUIT", ucits_name="iShares S&P 500 Information Technology UCITS"),
    ETF("XLF", "Financial Select Sector SPDR",
        "sector", sector="Financials",
        ucits_ticker="IUFS", ucits_name="iShares S&P 500 Financials UCITS"),
    ETF("XLV", "Health Care Select Sector SPDR",
        "sector", sector="Healthcare",
        ucits_ticker="IUHC", ucits_name="iShares S&P 500 Health Care UCITS"),
    ETF("XLY", "Consumer Discretionary Select Sector SPDR",
        "sector", sector="Consumer Cyclical",
        ucits_ticker="IUCD", ucits_name="iShares S&P 500 Consumer Discretionary UCITS"),
    ETF("XLP", "Consumer Staples Select Sector SPDR",
        "sector", sector="Consumer Defensive",
        ucits_ticker="IUCS", ucits_name="iShares S&P 500 Consumer Staples UCITS"),
    ETF("XLI", "Industrial Select Sector SPDR",
        "sector", sector="Industrials",
        ucits_ticker="IUIS", ucits_name="iShares S&P 500 Industrials UCITS"),
    ETF("XLE", "Energy Select Sector SPDR",
        "sector", sector="Energy",
        ucits_ticker="IUES", ucits_name="iShares S&P 500 Energy UCITS"),
    ETF("XLU", "Utilities Select Sector SPDR",
        "sector", sector="Utilities",
        ucits_ticker="IUUS", ucits_name="iShares S&P 500 Utilities UCITS"),
    ETF("XLRE", "Real Estate Select Sector SPDR",
        "sector", sector="Real Estate",
        ucits_ticker=None, ucits_name=None),  # No clean UCITS equivalent
    ETF("XLB", "Materials Select Sector SPDR",
        "sector", sector="Basic Materials",
        ucits_ticker=None, ucits_name=None),
    ETF("XLC", "Communication Services Select Sector SPDR",
        "sector", sector="Communication Services",
        ucits_ticker=None, ucits_name=None),
]


# ─── Broad-market / index ETFs ──────────────────────────────────────────────
BROAD_ETFS: list[ETF] = [
    ETF("SPY", "SPDR S&P 500", "broad",
        ucits_ticker="SXR8", ucits_name="iShares Core S&P 500 UCITS (Acc)"),
    ETF("QQQ", "Invesco QQQ Trust", "broad",
        ucits_ticker="EQQQ", ucits_name="Invesco EQQQ NASDAQ-100 UCITS"),
    ETF("IWM", "iShares Russell 2000 (small caps)", "broad",
        ucits_ticker=None, ucits_name=None),
]


# ─── International / EM ─────────────────────────────────────────────────────
INTERNATIONAL_ETFS: list[ETF] = [
    ETF("EEM", "iShares MSCI Emerging Markets", "international",
        ucits_ticker="IS3N", ucits_name="iShares Core MSCI EM IMI UCITS (Acc)"),
    ETF("VWO", "Vanguard FTSE Emerging Markets", "international",
        ucits_ticker="IS3N", ucits_name="iShares Core MSCI EM IMI UCITS (Acc)"),
    ETF("IEUR", "iShares Core MSCI Europe", "international",
        ucits_ticker="SXR7", ucits_name="iShares Core MSCI Europe UCITS (Acc)"),
]


# ─── Thematic (industries that aren't clean sector slices) ──────────────────
THEMATIC_ETFS: list[ETF] = [
    ETF("SOXX", "iShares Semiconductor", "thematic",
        ucits_ticker="SEC0", ucits_name="iShares MSCI EM Semiconductors UCITS"),
    ETF("SMH", "VanEck Semiconductor", "thematic",
        ucits_ticker="SEC0", ucits_name="iShares MSCI EM Semiconductors UCITS"),
    ETF("IBB", "iShares Biotechnology", "thematic",
        ucits_ticker=None, ucits_name=None),
]


# ─── User's current holdings (UCITS tickers as bought) ──────────────────────
# These are monitored for SELL signals. Each maps to a US proxy ETF for
# data (UCITS history is shorter and noisier; US has clean 20+ years).
USER_HOLDINGS_UCITS: dict[str, str] = {
    "SEC0": "SOXX",   # semis — track via SOXX
    "SXR8": "SPY",    # S&P 500 — track via SPY
    "IS3N": "EEM",    # EM — track via EEM
}


# ─── Lookup helpers ─────────────────────────────────────────────────────────

_ALL_ETFS: dict[str, ETF] = {
    e.ticker: e for e in US_SECTOR_ETFS + BROAD_ETFS + INTERNATIONAL_ETFS + THEMATIC_ETFS
}


def get_etf(ticker: str) -> ETF | None:
    """Lookup an ETF metadata by US ticker. Case-insensitive."""
    return _ALL_ETFS.get(ticker.upper())


def all_monitored_tickers() -> list[str]:
    """All US tickers the radar needs price data for."""
    return list(_ALL_ETFS.keys())


def sector_etf_tickers() -> list[str]:
    """Just the 11 SPDR sector ETFs."""
    return [e.ticker for e in US_SECTOR_ETFS]


def broad_etf_tickers() -> list[str]:
    """SPY, QQQ, IWM."""
    return [e.ticker for e in BROAD_ETFS]


def thematic_etf_tickers() -> list[str]:
    """SOXX, SMH, IBB, ..."""
    return [e.ticker for e in THEMATIC_ETFS]


def user_holding_proxies() -> dict[str, str]:
    """{UCITS_ticker: US_proxy_ticker} for tracking user's existing positions."""
    return dict(USER_HOLDINGS_UCITS)


def proxy_to_ucits(us_ticker: str) -> str | None:
    """Reverse lookup: US ticker → UCITS that the user owns mapped to it.

    Used in alerts: "SOXX is flashing SELL" → message references SEC0.
    """
    us = us_ticker.upper()
    for ucits, proxy in USER_HOLDINGS_UCITS.items():
        if proxy.upper() == us:
            return ucits
    return None


def display_ticker_pair(us_ticker: str) -> str:
    """Format as 'XLK (IUIT UCITS)' or 'XLK' if no UCITS mapping."""
    etf = get_etf(us_ticker)
    if etf and etf.ucits_ticker:
        return f"{etf.ticker} ({etf.ucits_ticker} UCITS)"
    return us_ticker
