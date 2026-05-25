"""
universe.py — US equity ticker universe for DipRadar.

Sources (US-only; European markets are Phase 2):
  - Static US universe : us_universe_static.json (iShares ITOT / DWCF holdings, ~3500 tickers)
                         Committed to repo — never fetched at runtime.
                         Run scripts/refresh_us_universe.py to update.
  - S&P 500            : Wikipedia (scraped dynamically, full fallback on failure)
  - Nasdaq 100         : Wikipedia (scraped dynamically, full fallback on failure)
  - User portfolio     : USER_PORTFOLIO — stocks only, bot-generated positions
  - User watchlist     : USER_WATCHLIST — stocks of interest for scanning

ETFs are excluded from the ML pipeline (no valid fundamentals in yfinance).
Use ETF_TICKERS for this exclusion in other modules:

  from universe import get_full_universe, ETF_TICKERS, get_ml_universe
  all_tickers = get_full_universe()   # includes all tickers
  ml_tickers  = get_ml_universe()    # excludes ETFs and non-US exchange tickers

Phase 2: European markets — not yet implemented.
See ModelID.DIP_EU, ModelID.CORE_EU, ModelID.MOMENTUM_EU in model_registry.py.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from config import US_EXCHANGES, MIN_MARKET_CAP_USD

# Disk cache — persists across Railway container restarts
_CACHE_DIR  = Path("/data") if Path("/data").exists() else Path("/tmp")
_CACHE_FILE = _CACHE_DIR / "ml_universe_cache.json"
_CACHE_DAYS = 7

# Static US universe JSON committed to the repository
_US_STATIC_FILE = Path(__file__).parent / "us_universe_static.json"

# ─────────────────────────────────────────────────────────────────────────
# ETFs — used as sector/benchmark proxies internally; excluded from ML.
# Only US-listed ETFs that could appear in the universe and need exclusion.
# ─────────────────────────────────────────────────────────────────────────
ETF_TICKERS: set[str] = {
    "SPY", "QQQ", "IVV", "VTI", "VOO",    # S&P 500 / total market
    "EEM", "IEMA",                          # EM — watchlist / internal use
    "GLD", "SLV",                           # commodities
    "TLT", "HYG", "LQD",                   # bonds
    # Sector ETFs (used internally for sector_relative; excluded from ML universe)
    "XLK", "XLV", "XLF", "XLE", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC", "XLB",
}

# Known delisted or acquired US tickers — removed from ML universe.
# Update when S&P 500 composition changes.
# Source: yfinance "possibly delisted" errors in Railway logs.
_KNOWN_DELISTED: set[str] = {
    # Acquired (M&A) — no longer exist as independent tickers
    "DFS",     # Discover Financial → acquired by Capital One (2024)
    "HES",     # Hess → acquired by Chevron (2024)
    "JNPR",    # Juniper Networks → acquired by HPE (2024)
    "K",       # Kellanova → acquired by Mars (2024)
    "MRO",     # Marathon Oil → acquired by ConocoPhillips (2024)
    "PARA",    # Paramount → merger/acquisition (2024)
    "WBA",     # Walgreens → going private (2024)
    "IPG",     # Interpublic → acquired by Omnicom (2024)
    "SGEN",    # Seagen → acquired by Pfizer (2023)
    "BCR",     # Bard → acquired by BD (2017)
    "WYDAY",   # Workday → obsolete ticker
    "WRK",     # WestRock → merger with Smurfit (2024)
    "CTLT",    # Catalent → acquired by Novo Holdings (2024)
    "DAY",     # Dayforce (formerly Ceridian) → ticker changed from CDAY
}

# ─────────────────────────────────────────────────────────────────────────
# User portfolio and watchlist — stocks only, bot-driven positions.
# ETFs removed: sector ETFs are used internally (not via user lists).
# Non-US exchange tickers (e.g. ALV.DE) are kept for price monitoring
# but filtered out of the ML pipeline by the dot-suffix rule in get_ml_universe().
# ─────────────────────────────────────────────────────────────────────────
USER_PORTFOLIO: list[str] = [
    "NVO", "ADBE", "UBER", "MSFT", "PINS",
    "ADP", "CRM", "VICI", "CRWD", "PLTR", "NOW", "DUOL",
]

USER_WATCHLIST: list[str] = [
    "O", "MDT", "ABBV", "LMT", "RTX",
    "PANW", "TSM", "AVGO", "ALV.DE",
]


# ─────────────────────────────────────────────────────────────────────────
# Disk cache helpers
# ─────────────────────────────────────────────────────────────────────────

def _save_universe_cache(tickers: list[str]) -> None:
    """Write ticker list to /data/ml_universe_cache.json."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"date": date.today().isoformat(), "tickers": tickers}
        _CACHE_FILE.write_text(json.dumps(payload), encoding="utf-8")
        logging.info(f"[universe] Disk cache updated: {len(tickers)} tickers → {_CACHE_FILE}")
    except Exception as e:
        logging.warning(f"[universe] Failed to write disk cache: {e}")


def _load_universe_cache() -> list[str]:
    """
    Load disk cache if it exists and is within _CACHE_DAYS.
    Returns [] if cache does not exist or has expired.
    If expired but non-empty, still returns it as an emergency fallback.
    """
    if not _CACHE_FILE.exists():
        return []
    try:
        payload = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        cached_date = date.fromisoformat(payload["date"])
        age = (date.today() - cached_date).days
        tickers = payload.get("tickers", [])
        if len(tickers) > 100:
            if age <= _CACHE_DAYS:
                logging.info(f"[universe] Disk cache valid ({age}d): {len(tickers)} tickers")
            else:
                # Expired but usable — better than empty
                logging.warning(
                    f"[universe] Disk cache expired ({age}d) but has {len(tickers)} tickers — "
                    f"using as emergency fallback until Wikipedia is available."
                )
            return tickers
        logging.info(f"[universe] Disk cache too small ({len(tickers)} tickers), rebuilding.")
        return []
    except Exception as e:
        logging.warning(f"[universe] Failed to read disk cache: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────
# Static US universe (committed to repo — never fetched at runtime)
# ─────────────────────────────────────────────────────────────────────────

def _load_us_static() -> list[str]:
    """
    Load the static US ticker list from us_universe_static.json.
    This file is committed to the repository and never fetched at runtime.
    Run scripts/refresh_us_universe.py to update it from iShares ITOT holdings.
    Returns [] if the file is missing (bootstrap still works via SP500/NDX fallbacks).
    """
    if not _US_STATIC_FILE.exists():
        logging.warning(
            f"[universe] us_universe_static.json not found at {_US_STATIC_FILE}. "
            f"Run scripts/refresh_us_universe.py to generate it."
        )
        return []
    try:
        data = json.loads(_US_STATIC_FILE.read_text(encoding="utf-8"))
        tickers = data.get("tickers", [])
        logging.info(
            f"[universe] Static US universe loaded: {len(tickers)} tickers "
            f"(source: {data.get('source', 'unknown')}, "
            f"generated: {data.get('generated', 'unknown')})"
        )
        return tickers
    except Exception as e:
        logging.warning(f"[universe] Failed to load us_universe_static.json: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────
# Dynamic fetch helpers
# ─────────────────────────────────────────────────────────────────────────

def _fetch_sp500() -> list[str]:
    """Scrape S&P 500 from Wikipedia. Falls back to static list on failure."""
    try:
        import pandas as pd
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url, attrs={"id": "constituents"})
        tickers = tables[0]["Symbol"].tolist()
        tickers = [t.replace(".", "-") for t in tickers]
        logging.info(f"[universe] S&P 500: {len(tickers)} tickers from Wikipedia")
        return tickers
    except Exception as e:
        logging.warning(f"[universe] S&P 500 Wikipedia failed: {e}. Using fallback.")
        return _SP500_FALLBACK


def _fetch_nasdaq100() -> list[str]:
    """Scrape Nasdaq 100 from Wikipedia. Falls back to static list on failure."""
    try:
        import pandas as pd
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url)
        for t in tables:
            if "Ticker" in t.columns:
                tickers = [str(x) for x in t["Ticker"].tolist()]
                logging.info(f"[universe] Nasdaq 100: {len(tickers)} tickers from Wikipedia")
                return tickers
        raise ValueError("Nasdaq 100 table not found")
    except Exception as e:
        logging.warning(f"[universe] Nasdaq 100 Wikipedia failed: {e}. Using fallback.")
        return _NASDAQ100_FALLBACK


# ─────────────────────────────────────────────────────────────────────────
# Universe filter for bootstrap scripts
# ─────────────────────────────────────────────────────────────────────────

def is_us_universe(ticker_info: dict) -> bool:
    """Return True if ticker_info describes a US-domiciled stock above the $2B market cap floor."""
    exchange   = str(ticker_info.get("exchange") or ticker_info.get("exchangeShortName") or "")
    market_cap = float(ticker_info.get("marketCap") or 0)
    return exchange in US_EXCHANGES and market_cap >= MIN_MARKET_CAP_USD


# In-memory cache — invalidated if result is empty
_universe_cache: list[str] = []


def get_full_universe() -> list[str]:
    """
    Complete ticker universe.
    For watchlist/price scanning. Do NOT use directly in ML pipeline.

    Priority order:
      1. In-memory cache (same process, non-empty)
      2. Disk cache (/data/ml_universe_cache.json, <= 7 days)
         Expired cache is still used as emergency fallback.
      3. Build from scratch: user lists + static JSON + Wikipedia scrape
    Saves to disk whenever built from scratch.
    """
    global _universe_cache

    if _universe_cache:
        return _universe_cache

    disk = _load_universe_cache()
    if disk:
        _universe_cache = disk
        return _universe_cache

    sp500     = _fetch_sp500()
    ndx100    = _fetch_nasdaq100()
    us_static = _load_us_static()

    raw = USER_PORTFOLIO + USER_WATCHLIST + us_static + sp500 + ndx100

    seen:   set[str]  = set()
    unique: list[str] = []
    for t in raw:
        t = t.strip().upper()
        if t and t not in seen:
            seen.add(t)
            unique.append(t)

    logging.info(f"[universe] Total universe: {len(unique)} tickers")

    if len(unique) > 100:
        _universe_cache = unique
        _save_universe_cache(unique)
    else:
        logging.error(
            f"[universe] Suspicious result: only {len(unique)} tickers — using static fallbacks."
        )
        _universe_cache = list(dict.fromkeys(
            USER_PORTFOLIO + USER_WATCHLIST + us_static + _SP500_FALLBACK + _NASDAQ100_FALLBACK
        ))
        _save_universe_cache(_universe_cache)
        logging.info(f"[universe] Static fallback: {len(_universe_cache)} tickers saved to cache")

    return _universe_cache


def get_ml_universe() -> list[str]:
    """
    Filtered ticker universe for the ML pipeline.
    Removes:
      - ETFs (no valid fundamentals in yfinance)
      - Known delisted tickers
      - Non-US exchange tickers (any ticker with '.' in it, e.g. ALV.DE, AZN.L, ASML.AS)

    Use in bootstrap scripts, train_model.py, and daily scan.
    The $2B market cap floor is applied later in bootstrap (where ticker_info is already fetched).
    """
    full = get_full_universe()
    filtered = [
        t for t in full
        if t not in ETF_TICKERS
        and t not in _KNOWN_DELISTED
        and "." not in t        # .DE / .PA / .L / .AS = EU/UK exchange → Phase 2
    ]
    excluded = len(full) - len(filtered)
    logging.info(f"[universe] ML universe: {len(filtered)} tickers ({excluded} excluded)")

    if len(filtered) < 50:
        raise RuntimeError(
            f"[universe] get_ml_universe() returned only {len(filtered)} tickers — "
            f"scraping failed and disk cache is missing. Check /data/ml_universe_cache.json."
        )

    return filtered


def is_etf(ticker: str) -> bool:
    """Return True if ticker is an ETF and should be excluded from ML."""
    return ticker.upper() in ETF_TICKERS


def get_universe_stats() -> dict:
    """Universe statistics for /admin or logging."""
    full = get_full_universe()
    ml   = get_ml_universe()
    return {
        "total":          len(full),
        "ml_eligible":    len(ml),
        "excluded":       len(full) - len(ml),
        "user_portfolio": len(USER_PORTFOLIO),
        "user_watchlist": len(USER_WATCHLIST),
    }


# ─────────────────────────────────────────────────────────────────────────
# STATIC FALLBACKS — activated if Wikipedia is inaccessible
# S&P 500: 503 constituents (Jan 2025)
# Nasdaq 100: 101 constituents (Jan 2025)
# ─────────────────────────────────────────────────────────────────────────

_SP500_FALLBACK: list[str] = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A",
    "APD","ABNB","AKAM","ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL",
    "GOOG","MO","AMZN","AMCR","AEE","AAL","AEP","AXP","AIG","AMT",
    "AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON","APA","AAPL",
    "AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ","T","ATO","ADSK",
    "AZO","AVB","AVY","AXON","BKR","BALL","BAC","BK","BBWI","BAX",
    "BDX","BRK-B","BBY","BIO","TECH","BIIB","BLK","BX","BA","BCR",
    "BWA","BSX","BMY","AVGO","BR","BRO","BF-B","BLDR","BG","CDNS",
    "CZR","CPT","CPB","COF","CAH","KMX","CCL","CARR","CTLT","CAT",
    "CBOE","CBRE","CDW","CE","COR","CNC","CNP","CF","CHRW","CRL",
    "SCHW","CHTR","CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO",
    "C","CFG","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CAG",
    "COP","ED","STZ","CEG","COO","CPRT","GLW","CPAY","CTVA","CSGP",
    "COST","CTRA","CRWD","CCI","CSX","CMI","CVS","DHR","DRI","DVA",
    "DAY","DE","DAL","XRAY","DVN","DXCM","FANG","DLR","DFS","DG",
    "DLTR","D","DPZ","DOV","DHI","DTE","DUK","DD","EMN","ETN",
    "EBAY","ECL","EIX","EW","EA","ELV","EMR","ENPH","ETR","EOG",
    "EPAM","EQT","EFX","EQIX","EQR","ESS","EL","ETSY","EG","EVRG",
    "ES","EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO","FAST",
    "FRT","FDX","FIS","FITB","FSLR","FE","FI","FMC","F","FTNT",
    "FTV","FOXA","FOX","BEN","FCX","GRMN","IT","GE","GEHC","GEV",
    "GEN","GNRC","GD","GIS","GM","GPC","GILD","GPN","GL","GDDY",
    "GS","HAL","HIG","HAS","HCA","DOC","HSIC","HSY","HES","HPE",
    "HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM",
    "HBAN","HII","IBM","IEX","IDXX","ITW","INCY","IR","PODD","INTC",
    "ICE","IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV","IRM",
    "JBHT","JBL","JKHY","J","JNJ","JCI","JPM","JNPR","K","KVUE",
    "KDP","KEY","KEYS","KMB","KIM","KMI","KKR","KLAC","KHC","KR",
    "LHX","LH","LRCX","LW","LVS","LDOS","LEN","LLY","LIN","LYV",
    "LKQ","LMT","L","LOW","LULU","LYB","MTB","MRO","MPC","MKTX",
    "MAR","MMC","MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT",
    "MRK","META","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA",
    "MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI",
    "MSCI","NDAQ","NTAP","NFLX","NEM","NWSA","NWS","NEE","NKE",
    "NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE","NVDA","NVR",
    "NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL","OTIS",
    "PCAR","PKG","PLTR","PANW","PARA","PH","PAYX","PAYC","PYPL",
    "PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG",
    "PPL","PFG","PG","PGR","PLD","PRU","PEG","PTC","PSA","PHM",
    "QRVO","PWR","QCOM","DGX","RL","RJF","RTX","O","REG","REGN",
    "RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI",
    "CRM","SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM",
    "SW","SNA","SOLV","SO","LUV","SWK","SBUX","STT","STLD","STE",
    "SYK","SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR",
    "TRGP","TGT","TEL","TDY","TFX","TER","TSLA","TXN","TPL","TXT",
    "TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL","TSN",
    "USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS",
    "VLO","VTR","VLTO","VRSN","VRSK","VZ","VRTX","VTRS","VICI",
    "V","VST","VMC","WRB","GWW","WAB","WBA","WMT","DIS","WBD",
    "WM","WAT","WEC","WFC","WELL","WST","WDC","WRK","WY","WHR",
    "WMB","WTW","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS",
]

_NASDAQ100_FALLBACK: list[str] = [
    "ADBE","ADP","ABNB","ALGN","GOOGL","GOOG","AMZN","AMD","AEP",
    "AMGN","ADI","ANSS","AAPL","AMAT","ASML","AZN","TEAM","ADSK",
    "BIDU","BIIB","BKNG","AVGO","CDNS","CDW","CHTR","CTAS","CSCO",
    "CCEP","CTSH","CMCSA","CEG","CPRT","CSGP","COST","CRWD","CSX",
    "DDOG","DXCM","FANG","DLTR","EBAY","EA","EXC","FAST","FTNT",
    "GILD","GFS","HON","IDXX","ILMN","INTC","INTU","ISRG","JD",
    "KDP","KLAC","KHC","LRCX","LIN","LULU","MAR","MRVL","MELI",
    "META","MCHP","MU","MSFT","MRNA","MDLZ","MDB","MNST","NFLX",
    "NVDA","NXPI","ORLY","ODFL","ON","PCAR","PANW","PAYX","PYPL",
    "PDD","PEP","QCOM","REGN","ROP","ROST","SBUX","SGEN","SIRI",
    "SNPS","TSLA","TXN","TMUS","TTWO","VRSK","VRTX","WBA","WBD",
    "WYDAY","XEL","ZM","ZS",
]
