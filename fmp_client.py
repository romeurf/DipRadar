"""
Cliente para Financial Modeling Prep (FMP) API — endpoints actuais (2026).
Free tier: 250 calls/day
"""
import os
import time
import logging
import requests

FMP_API_KEY = os.environ.get("FMP_API_KEY", "demo")
BASE_V3 = "https://financialmodelingprep.com/api/v3"
BASE_V4 = "https://financialmodelingprep.com/api/v4"
BASE_STABLE = "https://financialmodelingprep.com/stable"


def _get(url: str, params: dict = None) -> dict | list | None:
    p = {"apikey": FMP_API_KEY}
    if params:
        p.update(params)
    try:
        r = requests.get(url, params=p, timeout=15)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and ("Error Message" in data or "message" in data):
            logging.warning(f"FMP erro: {data.get('Error Message') or data.get('message')}")
            return None
        return data
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP {e.response.status_code} em {url}")
        return None
    except Exception as e:
        logging.error(f"Request falhou: {e}")
        return None


def screen_big_drops(min_drop_pct: float = 10.0,
                     min_market_cap: int = 500_000_000) -> list[dict]:
    """
    Tenta 3 endpoints por ordem:
      1. /stable/biggest-losers  (endpoint actual)
      2. /v3/biggest-losers      (alias)
      3. /v3/stock-screener      (fallback)
    """
    def _parse(data):
        results = []
        for item in (data or []):
            symbol = item.get("symbol") or item.get("ticker")
            change = item.get("changesPercentage") or item.get("changePercentage") or item.get("change")
            if not symbol or change is None:
                continue
            if isinstance(change, str):
                change = change.replace("(","").replace(")","").replace("%","").strip()
                try: change = float(change)
                except: continue
            if change > -min_drop_pct:
                continue
            mc = item.get("marketCap") or item.get("market_cap") or 0
            if mc and mc < min_market_cap:
                continue
            results.append({
                "symbol": symbol,
                "name": item.get("name") or item.get("companyName") or symbol,
                "price": item.get("price"),
                "change_pct": change,
                "market_cap": mc,
            })
        return results

    for url in [f"{BASE_STABLE}/biggest-losers", f"{BASE_V3}/biggest-losers"]:
        data = _get(url)
        if data:
            r = _parse(data)
            if r:
                logging.info(f"{len(r)} losers via {url.split('/')[-1]}")
                return r
        time.sleep(0.3)

    # Fallback: screener
    data = _get(f"{BASE_V3}/stock-screener", {
        "marketCapMoreThan": min_market_cap,
        "isEtf": "false",
        "isActivelyTrading": "true",
        "limit": 500,
    })
    if data:
        r = _parse(data)
        logging.info(f"{len(r)} losers via screener")
        return r

    logging.warning("Todos os endpoints de losers falharam.")
    return []


def get_fundamentals(symbol: str) -> dict:
    result = {"symbol": symbol}

    data = _get(f"{BASE_V3}/profile/{symbol}")
    if data and len(data) > 0:
        p = data[0]
        result.update({
            "sector": p.get("sector",""), "industry": p.get("industry",""),
            "name": p.get("companyName", symbol), "pe": p.get("pe"),
            "market_cap": p.get("mktCap"), "price": p.get("price"),
            "beta": p.get("beta"),
        })
    time.sleep(0.25)

    data = _get(f"{BASE_V3}/key-metrics-ttm/{symbol}")
    if data and len(data) > 0:
        k = data[0]
        result.update({
            "fcf_yield": k.get("freeCashFlowYieldTTM"),
            "ev_ebitda": k.get("enterpriseValueOverEBITDATTM"),
            "roe": k.get("roeTTM"), "debt_equity": k.get("debtToEquityTTM"),
            "dividend_yield": k.get("dividendYieldPercentageTTM") or k.get("dividendYieldTTM"),
            "payout_ratio": k.get("payoutRatioTTM"), "pb": k.get("pbRatioTTM"),
            "fcf_per_share": k.get("freeCashFlowPerShareTTM"),
        })
    time.sleep(0.25)

    data = _get(f"{BASE_V3}/financial-growth/{symbol}", {"period":"annual","limit":1})
    if data and len(data) > 0:
        g = data[0]
        result.update({
            "revenue_growth": g.get("revenueGrowth"),
            "eps_growth": g.get("epsgrowth"),
        })
    time.sleep(0.25)

    data = _get(f"{BASE_V3}/income-statement/{symbol}", {"period":"annual","limit":1})
    if data and len(data) > 0:
        i = data[0]
        rev = i.get("revenue",0) or 0
        gross = i.get("grossProfit",0) or 0
        if rev > 0:
            result["gross_margin"] = gross / rev
    time.sleep(0.25)

    # Analyst target — tenta v4 depois v3
    for url, params in [
        (f"{BASE_V4}/price-target", {"symbol": symbol}),
        (f"{BASE_V3}/price-target/{symbol}", {}),
    ]:
        data = _get(url, params or None)
        if data and len(data) > 0:
            t = data[0]
            avg = t.get("targetConsensus") or t.get("priceTarget")
            price = result.get("price")
            if avg and price and price > 0:
                result["analyst_upside"] = (avg - price) / price * 100
                result["analyst_target"] = avg
            break

    return result


def get_news(symbol: str, limit: int = 3) -> list[dict]:
    data = _get(f"{BASE_V3}/stock_news", {"tickers": symbol, "limit": limit})
    if not data:
        return []
    return [{"title": i.get("title",""), "url": i.get("url",""),
             "source": i.get("site","")} for i in data[:limit]]


def get_historical_pe(symbol: str, years: int = 5) -> float | None:
    data = _get(f"{BASE_V3}/key-metrics/{symbol}", {"period":"annual","limit":years})
    if not data:
        return None
    vals = [d["peRatio"] for d in data if d.get("peRatio") and 0 < d["peRatio"] < 300]
    return round(sum(vals)/len(vals), 1) if vals else None
