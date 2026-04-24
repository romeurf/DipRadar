"""
Cliente para Financial Modeling Prep (FMP) API.
Free tier: 250 calls/day — https://financialmodelingprep.com/developer/docs
Regista em https://site.financialmodelingprep.com/register para API key gratuita.
"""
import os
import time
import logging
import requests
from functools import lru_cache

FMP_API_KEY = os.environ.get("FMP_API_KEY", "demo")
BASE = "https://financialmodelingprep.com/api"


def _get(endpoint: str, params: dict = None, version: int = 3) -> dict | list | None:
    url = f"{BASE}/v{version}/{endpoint}"
    p = {"apikey": FMP_API_KEY}
    if params:
        p.update(params)
    try:
        r = requests.get(url, params=p, timeout=15)
        r.raise_for_status()
        data = r.json()
        # FMP devolve {"Error Message": "..."} quando há problema
        if isinstance(data, dict) and "Error Message" in data:
            logging.warning(f"FMP erro em {endpoint}: {data['Error Message']}")
            return None
        return data
    except Exception as e:
        logging.error(f"FMP request falhou ({endpoint}): {e}")
        return None


def screen_big_drops(min_drop_pct: float = 10.0, min_market_cap: int = 500_000_000) -> list[dict]:
    """
    Usa o screener da FMP para encontrar acções que caíram X% hoje.
    min_market_cap: filtra penny stocks (default 500M)
    Devolve lista de {symbol, name, price, changesPercentage, sector, marketCap}
    """
    data = _get("stock_market/losers")
    if not data:
        return []

    results = []
    for item in data:
        change = item.get("changesPercentage", 0)
        # changesPercentage pode ser string "(-12.5%)" ou float
        if isinstance(change, str):
            change = float(change.strip("()%").replace(",", "."))

        if change <= -min_drop_pct:
            mc = item.get("marketCap", 0) or 0
            if mc >= min_market_cap:
                results.append({
                    "symbol": item.get("ticker") or item.get("symbol"),
                    "name": item.get("companyName") or item.get("name"),
                    "price": item.get("price"),
                    "change_pct": change,
                    "market_cap": mc,
                })

    return results


def get_fundamentals(symbol: str) -> dict:
    """
    Agrega métricas fundamentais de várias endpoints da FMP.
    Usa lru_cache implícito no dict para não repetir calls.
    """
    result = {"symbol": symbol}

    # 1. Profile (sector, P/E, target price, etc.)
    profile_data = _get(f"profile/{symbol}")
    if profile_data and len(profile_data) > 0:
        p = profile_data[0]
        result["sector"] = p.get("sector", "")
        result["industry"] = p.get("industry", "")
        result["name"] = p.get("companyName", symbol)
        result["pe"] = p.get("pe")
        result["market_cap"] = p.get("mktCap")
        result["price"] = p.get("price")
        result["description"] = p.get("description", "")[:200]

    # 2. Key metrics TTM
    time.sleep(0.2)
    km = _get(f"key-metrics-ttm/{symbol}")
    if km and len(km) > 0:
        k = km[0]
        result["fcf_yield"] = k.get("freeCashFlowYieldTTM")
        result["ev_ebitda"] = k.get("enterpriseValueOverEBITDATTM")
        result["revenue_per_share"] = k.get("revenuePerShareTTM")
        result["roe"] = k.get("roeTTM")
        result["debt_equity"] = k.get("debtToEquityTTM")
        result["dividend_yield"] = k.get("dividendYieldTTM") or k.get("dividendYieldPercentageTTM")
        result["payout_ratio"] = k.get("payoutRatioTTM")
        result["pb"] = k.get("pbRatioTTM")
        result["fcf_per_share"] = k.get("freeCashFlowPerShareTTM")

    # 3. Financial growth (revenue growth YoY)
    time.sleep(0.2)
    growth = _get(f"financial-growth/{symbol}", {"period": "annual", "limit": 1})
    if growth and len(growth) > 0:
        g = growth[0]
        result["revenue_growth"] = g.get("revenueGrowth")
        result["eps_growth"] = g.get("epsgrowth")
        result["fcf_growth"] = g.get("freeCashFlowGrowth")

    # 4. Income statement para gross margin
    time.sleep(0.2)
    income = _get(f"income-statement/{symbol}", {"period": "annual", "limit": 1})
    if income and len(income) > 0:
        i = income[0]
        revenue = i.get("revenue", 0)
        gross = i.get("grossProfit", 0)
        if revenue and revenue > 0:
            result["gross_margin"] = gross / revenue

    # 5. Analyst estimates / price target
    time.sleep(0.2)
    target = _get(f"price-target/{symbol}")
    if target and len(target) > 0:
        t = target[0]
        avg_target = t.get("targetConsensus") or t.get("targetMean")
        price = result.get("price")
        if avg_target and price and price > 0:
            result["analyst_upside"] = (avg_target - price) / price * 100
            result["analyst_target"] = avg_target

    return result


def get_news(symbol: str, limit: int = 3) -> list[dict]:
    """Notícias recentes via FMP."""
    data = _get("stock_news", {"tickers": symbol, "limit": limit})
    if not data:
        return []
    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "published": item.get("publishedDate", ""),
            "source": item.get("site", ""),
        }
        for item in data[:limit]
    ]


def get_historical_pe(symbol: str, years: int = 5) -> float | None:
    """P/E médio histórico usando key-metrics anuais."""
    data = _get(f"key-metrics/{symbol}", {"period": "annual", "limit": years})
    if not data:
        return None
    pe_values = [d["peRatio"] for d in data if d.get("peRatio") and d["peRatio"] > 0 and d["peRatio"] < 300]
    if not pe_values:
        return None
    return sum(pe_values) / len(pe_values)
