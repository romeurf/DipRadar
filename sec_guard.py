"""
sec_guard.py — Determines whether a ticker is expected to have SEC filings.

US-domiciled companies listed on NYSE/NASDAQ/AMEX file with the SEC.
Foreign companies listed on US exchanges (ADRs, cross-listings) generally
do not — examples: NVO (Novo Nordisk, DK), TSM (Taiwan Semi, TW),
ASML (Netherlands), BABA (Alibaba, CN).

Usage:
    from sec_guard import has_sec_filings

    if has_sec_filings(ticker, country, exchange):
        insider_score = _fetch_form4(ticker)
    else:
        insider_score = _FALLBACK["insider_buy_recent"]  # 0.0, silent fallback
"""
from __future__ import annotations

from config import SEC_FILING_EXCHANGES


def has_sec_filings(
    ticker: str,
    country: str | None = None,
    exchange: str | None = None,
) -> bool:
    """Return True if this ticker is expected to have SEC filings available.

    Decision logic:
    - Explicitly non-US country → no SEC filings (NVO=DK, ASML=NL, TSM=TW)
    - Known SEC exchange (NYSE/NASDAQ/AMEX) + US or unknown country → has filings
    - Unknown exchange + unknown country → conservative: no SEC filings assumed
    """
    _country  = (country  or "").strip().upper()
    _exchange = (exchange or "").strip().upper()

    # Explicitly non-US country → no SEC filings regardless of exchange
    if _country and _country not in {"US", "USA"}:
        return False

    # Known SEC exchange + US (or unknown) country → has filings
    if _exchange in {e.upper() for e in SEC_FILING_EXCHANGES}:
        return True

    # Cannot determine — conservative: assume no SEC filings to avoid failed EDGAR calls
    return False
