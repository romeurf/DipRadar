"""
ppr_scraper.py — Fetch PPR fund NAVs from public web sources.

Currently supports investing.com: pt.investing.com/funds/<isin_lowercase>
The page exposes the NAV in a data-test="instrument-price-last" element,
visible without JavaScript or login.

PPR NAVs update once per business day after market close (~17h–22h Lisbon).
Recommended cadence: 1× per day after 22h Lisbon (cron in main.py).

Anti-bot two-stage strategy:
  1. cloudscraper (if installed) — solves Cloudflare JS challenges
     transparently. Required for Railway datacenter IPs which plain
     `requests` gets 403'd on.
  2. requests fallback with rich browser headers — works for non-CF
     pages or weaker protections.

If both fail, returns None and the caller falls back to last cached NAV
(or manual /ppr nav input). PPR NAVs change once per day max, so a few
hours of staleness is acceptable.

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
# Rich headers that look like a real Chrome browser. Helps with weaker
# anti-bot setups; cloudscraper handles the Cloudflare challenges itself.
_HEADERS = {
    "User-Agent":                _USER_AGENT,
    "Accept":                    "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language":           "pt-PT,pt;q=0.9,en;q=0.8",
    "Accept-Encoding":           "gzip, deflate, br",
    "Cache-Control":             "no-cache",
    "Pragma":                    "no-cache",
    "Sec-Ch-Ua":                 '"Chromium";v="120", "Not_A Brand";v="8"',
    "Sec-Ch-Ua-Mobile":          "?0",
    "Sec-Ch-Ua-Platform":        '"Windows"',
    "Sec-Fetch-Dest":            "document",
    "Sec-Fetch-Mode":            "navigate",
    "Sec-Fetch-Site":            "none",
    "Sec-Fetch-User":            "?1",
    "Upgrade-Insecure-Requests": "1",
    "Referer":                   "https://pt.investing.com/",
}


def _http_get_with_cf_bypass(url: str, timeout: int = 25) -> Optional[str]:
    """Attempt to fetch a URL handling Cloudflare. Returns HTML text or None.

    Stage 1: cloudscraper (Cloudflare JS challenge solver). Lazy-imported
    so the bot still boots if the package isn't installed.
    Stage 2: plain requests with rich browser headers. Works for non-CF
    pages and some weaker WAFs.
    """
    # Stage 1: cloudscraper
    try:
        import cloudscraper
        scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        r = scraper.get(url, headers=_HEADERS, timeout=timeout)
        if r.status_code == 200 and r.text:
            return r.text
        logger.debug(f"[ppr_scraper] cloudscraper got status {r.status_code} for {url}")
    except ImportError:
        logger.info("[ppr_scraper] cloudscraper not installed — skipping CF bypass stage")
    except Exception as e:
        logger.warning(f"[ppr_scraper] cloudscraper failed: {e}")

    # Stage 2: plain requests with rich headers
    try:
        r = requests.get(url, headers=_HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.text
    except requests.RequestException as e:
        logger.warning(f"[ppr_scraper] plain requests failed for {url}: {e}")
        return None

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
    html = _http_get_with_cf_bypass(url)
    if not html:
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


def fetch_ppr_nav_banco_invest_pdf(isin: str) -> Optional[dict]:
    """Fetch NAV from Banco Invest's monthly PDF — fallback when investing.com fails.

    URL pattern: bancoinvest.pt/docs/default-source/ppr/<isin_lower>_mensal.pdf
    The PDF is a static file (no Cloudflare gate) so plain requests works.

    PDF structure (validated June 2026):
      Page 1 contains "Valor da UP: 7.4053€ | Abr-26:+6,3%" near the gestor
      commentary, with reference date like "30/Abr/2026" at the top.

    Updates monthly (typically 5–7 days after month-end). The returned NAV
    is end-of-previous-month, so it can be up to ~35 days stale during the
    transition window — but for a PPR (long-term hold, no daily trading
    decisions) this is sufficient.
    """
    url = f"https://www.bancoinvest.pt/docs/default-source/ppr/{isin.lower().strip()}_mensal.pdf"

    try:
        r = requests.get(url, headers=_HEADERS, timeout=30)
        r.raise_for_status()
        pdf_bytes = r.content
    except requests.RequestException as e:
        logger.warning(f"[ppr_scraper] PDF fetch {isin}: {e}")
        return None
    except Exception as e:
        logger.warning(f"[ppr_scraper] PDF fetch {isin} unexpected: {e}")
        return None

    try:
        import pypdf
        import io
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        logger.warning("[ppr_scraper] pypdf not installed — cannot parse PDF fallback")
        return None
    except Exception as e:
        logger.warning(f"[ppr_scraper] PDF parse {isin}: {e}")
        return None

    # Pattern: "Valor da UP: 7.4053€" — case-insensitive, tolerates extra whitespace
    nav_match = re.search(r'Valor\s+da\s+UP[:\s]+([\d.,]+)', text, re.IGNORECASE)
    if not nav_match:
        logger.warning(
            f"[ppr_scraper] PDF {isin}: 'Valor da UP' pattern not found "
            f"(PDF format may have changed)"
        )
        return None

    nav = _parse_pt_number(nav_match.group(1))
    if nav is None or nav <= 0:
        logger.warning(f"[ppr_scraper] PDF {isin}: invalid NAV '{nav_match.group(1)}'")
        return None

    out = {"nav": nav, "source": "bancoinvest.pt PDF (mensal)", "url": url}

    # Reference date — "30/Abr/2026" near the top
    date_match = re.search(r'(\d{1,2})/([A-Za-z]+)/(\d{4})', text)
    if date_match:
        out["report_date"] = date_match.group(0)

    # Monthly return — "Abr-26:+6,3%"
    month_match = re.search(r'[A-Za-z]+-\d{2}:\s*([+\-]?[\d.,]+)%', text)
    if month_match:
        ch = _parse_pt_number(month_match.group(1))
        if ch is not None:
            out["change_pct_month"] = ch

    return out


def fetch_ppr_nav(isin: str) -> Optional[dict]:
    """Main entry. Tries investing.com first (live daily), falls back to
    Banco Invest monthly PDF if investing.com is Cloudflare-blocked.

    On Railway IPs investing.com almost always returns 403; the PDF is the
    realistic primary source. On developer machines / Anthropic infra,
    investing.com works → it stays first for daily granularity when possible.
    """
    result = fetch_ppr_nav_investing(isin)
    if result and result.get("nav"):
        return result

    logger.info(f"[ppr_scraper] investing.com unavailable for {isin}, trying Banco Invest PDF")
    return fetch_ppr_nav_banco_invest_pdf(isin)
