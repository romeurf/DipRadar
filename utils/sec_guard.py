"""
sec_guard.py — Determines whether a ticker is expected to have SEC filings.

US-domiciled companies listed on NYSE/NASDAQ/AMEX file with the SEC.
Foreign companies listed on US exchanges (ADRs, cross-listings) generally
do not — examples: NVO (Novo Nordisk, DK), TSM (Taiwan Semi, TW),
ASML (Netherlands), BABA (Alibaba, CN).

Usage:
    from utils.sec_guard import has_sec_filings

    if has_sec_filings(ticker, country, exchange):
        insider_score = _fetch_form4(ticker)
    else:
        insider_score = _FALLBACK["insider_buy_recent"]  # 0.0, silent fallback
"""
from __future__ import annotations

from config import SEC_FILING_EXCHANGES

# yfinance returns "United States" (not "US" / "USA") for the country field.
# We need every accepted spelling here or the country check fails closed and
# silently disables EDGAR for every US ticker.
_US_COUNTRY_STRINGS: frozenset[str] = frozenset({
    "US", "USA",
    "UNITED STATES",
    "UNITED STATES OF AMERICA",
})

# yfinance exchange codes (NMS, NYQ, NGM, NCM, PCX, ...) differ from the formal
# names in config.SEC_FILING_EXCHANGES (NYSE / NASDAQ / AMEX). Without this
# mapping, every US ticker passed through yfinance would fail the gate. The
# canonical US-equity codes are also enumerated in universe._YFINANCE_US_CODES;
# keep the two lists in sync.
_YFINANCE_US_EXCHANGES: frozenset[str] = frozenset({
    "NMS",   # Nasdaq Market Select  (AAPL, MSFT, NVDA, GOOGL ...)
    "NYQ",   # NYSE                  (JPM, GS, BAC ...)
    "NGM",   # Nasdaq Global Market
    "NCM",   # Nasdaq Capital Market
    "PCX",   # NYSE Arca
    "BATS",
    "NYSEARCA",
    "NYSEAMERICAN",
})

_SEC_EXCHANGES_ALL: frozenset[str] = frozenset(
    {e.upper() for e in SEC_FILING_EXCHANGES} | _YFINANCE_US_EXCHANGES
)


def has_sec_filings(
    ticker: str,
    country: str | None = None,
    exchange: str | None = None,
) -> bool:
    """Return True if this ticker is expected to have SEC filings available.

    Decision logic:
    - Explicitly non-US country → no SEC filings (NVO=DK, ASML=NL, TSM=TW)
    - Known SEC exchange (formal name OR yfinance code) + US/unknown country
      → has filings
    - Unknown exchange + US/unknown country → assume yes (most permissive for
      US tickers that yfinance returned empty exchange for; downstream EDGAR
      calls fail gracefully if there really are no filings).
    - Unknown exchange + non-US country → no.
    """
    _country  = (country  or "").strip().upper()
    _exchange = (exchange or "").strip().upper()

    # Explicitly non-US country → no SEC filings regardless of exchange
    if _country and _country not in _US_COUNTRY_STRINGS:
        return False

    # Known SEC exchange (formal name or yfinance code) → has filings
    if _exchange in _SEC_EXCHANGES_ALL:
        return True

    # Country is US (or empty/unknown) but exchange wasn't recognised.
    # Allow EDGAR attempt — if the ticker actually doesn't have filings, the
    # downstream call returns None and the caller falls back to _FALLBACK.
    # This is much better than blocking EDGAR for every yfinance-empty US row.
    return _country in _US_COUNTRY_STRINGS or not _country
