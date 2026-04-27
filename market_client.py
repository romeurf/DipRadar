
""" 
DipRadar - Multi-region stock dip screener
US + Europe + UK + Asia via Yahoo Finance (free & unlimited)
Fundamentals via yfinance
"""
import time
import logging
import requests
import yfinance as yf
from typing import List, Dict, Optional

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

def screen_global_dips(
    min_drop_pct: float = 10.0, 
    min_market_cap: int = 2_000_000_000
) -> List[Dict]:
    """
    Screen largest daily losers across US, Europe, UK, and Asia.

    Returns deduplicated list with region tag.
    """
    regions = {
        "US": "day_losers",
        "Europe": "day_losers_europe", 
        "UK": "day_losers_gb",
        "Asia": "day_losers_asia"
    }

    all_losers = []
    seen_symbols = set()

    for region_name, screener_id in regions.items():
        try:
            url = f"https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds={screener_id}&count=100"
            r = requests.get(url, headers=_HEADERS, timeout=15)
            r.raise_for_status()

            quotes = (
                r.json()
                .get("finance", {})
                .get("result", [{}])[0]
                .get("quotes", [])
            )

            region_count = 0
            for q in quotes:
                symbol = q.get("symbol", "")
                if symbol in seen_symbols:
                    continue

                change_pct = q.get("regularMarketChangePercent", 0) or 0
                if change_pct > -min_drop_pct:
                    continue

                market_cap = q.get("marketCap", 0) or 0
                if market_cap < min_market_cap:
                    continue

                quote_type = q.get("quoteType", "")
                if quote_type in ["ETF", "MUTUALFUND"] or len(symbol) > 5:
                    continue

                all_losers.append({
                    "symbol": symbol,
                    "name": q.get("longName") or q.get("shortName") or symbol,
                    "price": q.get("regularMarketPrice"),
                    "change_pct": round(change_pct, 2),
                    "market_cap": market_cap,
                    "region": region_name
                })

                seen_symbols.add(symbol)
                region_count += 1

            logger.info(f"{region_name}: {region_count} candidates")

        except Exception as e:
            logging.warning(f"{region_name} failed: {e}")
            continue

    logger.info(f"Total global dips: {len(all_losers)}")
    return all_losers

def _yf_info(symbol: str) -> dict:
    """Get yfinance info with retry and backoff."""
    for attempt in range(4):
        wait = 10 + (20 * attempt)
        time.sleep(wait)
        try:
            inf = yf.Ticker(symbol).info
            if inf and len(inf) > 5:
                return inf
            logging.warning(f"{symbol}: empty response (attempt {attempt+1}/4)")
        except Exception as e:
            err_str = str(e)
            if any(x in err_str for x in ("429", "Too Many Requests", "Rate limit")):
                logging.warning(f"{symbol}: rate limit (wait {wait}s)")
            else:
                logging.error(f"{symbol}: {e}")
                break
    return {}

def get_fundamentals(symbol: str, region: str = "") -> dict:
    result = {"symbol": symbol, "region": region}
    inf = _yf_info(symbol)

    if not inf:
        logging.error(f"{symbol}: failed after all retries")
        return result

    mc = inf.get("marketCap", 0) or 0
    if 0 < mc < 1_000_000_000:
        result["skip"] = True
        logger.info(f"{symbol}: micro-cap ${mc/1e6:.0f}M — skip")
        return result

    price = inf.get("currentPrice") or inf.get("regularMarketPrice")

    # 52-week drawdown
    week52_high = inf.get("fiftyTwoWeekHigh")
    drawdown = None
    if week52_high and price and week52_high > 0:
        drawdown = round((price - week52_high) / week52_high * 100, 1)

    result.update({
        "name": inf.get("longName") or inf.get("shortName") or symbol,
        "sector": inf.get("sector", ""),
        "industry": inf.get("industry", ""),
        "price": price,
        "beta": inf.get("beta"),
        "market_cap": mc,
        "pe": inf.get("trailingPE") or inf.get("forwardPE"),
        "revenue_growth": inf.get("revenueGrowth"),
        "gross_margin": inf.get("grossMargins"),
        "ev_ebitda": inf.get("enterpriseToEbitda"),
        "roe": inf.get("returnOnEquity"),
        "debt_equity": inf.get("debtToEquity"),
        "dividend_yield": inf.get("dividendYield"),
        "payout_ratio": inf.get("payoutRatio"),
        "week52_high": week52_high,
        "drawdown_from_high": drawdown,
    })

    # FCF metrics
    fcf = inf.get("freeCashflow")
    shares = inf.get("sharesOutstanding")
    if fcf and mc > 0:
        result["fcf_yield"] = fcf / mc
    if fcf and shares and shares > 0:
        result["fcf_per_share"] = fcf / shares

    # Analyst targets
    target = inf.get("targetMeanPrice")
    if target and price and price > 0:
        result["analyst_upside"] = (target - price) / price * 100
        result["analyst_target"] = target

    return result

def get_52w_drawdown(symbol: str) -> Optional[float]:
    try:
        time.sleep(3)
        inf = yf.Ticker(symbol).info or {}
        high = inf.get("fiftyTwoWeekHigh")
        price = inf.get("currentPrice") or inf.get("regularMarketPrice")
        if high and price and high > 0:
            return round((price - high) / high * 100, 1)
    except Exception as e:
        logging.warning(f"52w drawdown {symbol}: {e}")
    return None

def get_news(symbol: str, limit: int = 3) -> List[Dict]:
    try:
        time.sleep(5)
        news = yf.Ticker(symbol).news or []
        return [{
            "title": (item.get("content") or {}).get("title") or item.get("title", ""),
            "url": ((item.get("content") or {}).get("canonicalUrl") or {}).get("url") or item.get("link", ""),
            "source": ((item.get("content") or {}).get("provider") or {}).get("displayName") or "",
        } for item in news[:limit]]
    except Exception as e:
        logging.error(f"News {symbol}: {e}")
        return []

def get_historical_pe(symbol: str) -> Optional[float]:
    try:
        time.sleep(5)
        pe = (yf.Ticker(symbol).info or {}).get("trailingPE")
        return round(pe, 1) if pe and 0 < pe < 300 else None
    except:
        return None

if __name__ == "__main__":
    # Test the new global screening
    candidates = screen_global_dips()
    print(f"Global dips found: {len(candidates)}")
    for c in candidates[:3]:
        print(f"{c['region']}: {c['symbol']} {c['change_pct']}%")
