"""
ppr_scraper.py — Fetch PPR fund NAVs from public web sources.

Currently supports investing.com: pt.investing.com/funds/<isin_lowercase>
The page exposes the NAV in a data-test="instrument-price-last" element,
visible without JavaScript or login.

PPR NAVs update once per business day after market close (~17h–22h Lisbon).
Recommended cadence: 1× per day after 22h Lisbon (cron in main.py).

Anti-bot considerations:
  - Realistic User-Agent + Accept-Language (Portuguese page expects pt-PT)
  - Single GET, no JS execution needed
  - On Cloudflare gate or page-structure change, returns None and the
    caller falls back to last cached NAV (or manual /ppr nav input)

The regex patterns target investing.com's React markup as of June 2026.
Pattern updates should go here when the site refactors (caller code
unchanged).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_INVESTING_URL_TEMPLATE = "https://pt.investing.com/funds/{isin_lower}"
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
_HEADERS = {
    "User-Agent":      _USER_AGENT,
    "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.8",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Investing.com markup (subject to refactors):
#   <span data-test="instrument-price-last">7,683</span>
#   <span data-test="instrument-price-change-percent">(-0,32%)</span>
#   <span data-test="trading-hours-status">Fechado</span>
_NAV_RE         = re.compile(r'data-test="instrument-price-last"[^>]*>\s*([\d.,]+)\s*<', re.I)
_CHANGE_PCT_RE  = re.compile(r'data-test="instrument-price-change-percent"[^>]*>\s*\(?([+\-]?[\d.,]+)%?\)?\s*<', re.I)
_STATUS_RE      = re.compile(r'data-test="trading-hours-status"[^>]*>\s*([^<]+)<', re.I)


def _parse_pt_number(s: str) -> Optional[float]:
    """Parse Portuguese-formatted numbers: '1.234,56' → 1234.56, '7,683' → 7.683."""
    if not s:
        return None
    s = s.strip().replace(" ", "")
    # If both '.' and ',' present: '.' is thousands separator, ',' is decimal
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    # Else: ',' alone is decimal
    elif "," in s:
        s = s.replace(",", ".")
    # Else: '.' alone is decimal (English-formatted page)
    try:
        return float(s)
    except ValueError:
        return None


def fetch_ppr_nav_investing(isin: str) -> Optional[dict]:
    """Try to scrape the NAV for `isin` from investing.com.

    Returns dict with keys:
      nav         (float, EUR)
      change_pct  (float, optional)
      status      (str like "Fechado" / "Aberto", optional)
      source      ("investing.com")
      url         (the fetched URL)
    Returns None on failure (network error, page structure changed, etc.).
    """
    url = _INVESTING_URL_TEMPLATE.format(isin_lower=isin.lower().strip())
    try:
        r = requests.get(url, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        html = r.text
    except requests.RequestException as e:
        logger.warning(f"[ppr_scraper] investing.com {isin} HTTP: {e}")
        return None
    except Exception as e:
        logger.warning(f"[ppr_scraper] investing.com {isin} unexpected: {e}")
        return None

    nav_match = _NAV_RE.search(html)
    if not nav_match:
        logger.warning(
            f"[ppr_scraper] {isin}: NAV regex did not match — page structure may have "
            f"changed. URL: {url}"
        )
        return None

    nav = _parse_pt_number(nav_match.group(1))
    if nav is None or nav <= 0:
        logger.warning(f"[ppr_scraper] {isin}: invalid NAV parse: '{nav_match.group(1)}'")
        return None

    out = {"nav": nav, "source": "investing.com", "url": url}

    cp_match = _CHANGE_PCT_RE.search(html)
    if cp_match:
        cp = _parse_pt_number(cp_match.group(1))
        if cp is not None:
            out["change_pct"] = cp

    st_match = _STATUS_RE.search(html)
    if st_match:
        out["status"] = st_match.group(1).strip()

    return out


def fetch_ppr_nav(isin: str) -> Optional[dict]:
    """Main entry. Currently tries investing.com only.

    Future: add fallbacks (Banco Invest monthly PDF parse, APFIPP
    calculator) when investing.com fails or rate-limits.
    """
    return fetch_ppr_nav_investing(isin)
