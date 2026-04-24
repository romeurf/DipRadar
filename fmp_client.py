"""
Screening: FMP biggest-losers
Fundamentais: yfinance com rate limiting e pre-filtro de market cap
"""
import os
import time
import logging
import requests
import yfinance as yf

FMP_API_KEY = os.environ.get("FMP_API_KEY", "demo")
BASE_V3 = "https://financialmodelingprep.com/api/v3"
BASE_STABLE = "https://financialmodelingprep.com/stable"

_YF_SLEEP = 3.0


def _fmp_get(url: str, params: dict = None):
    p = {"apikey": FMP_API_KEY}
    if params:
        p.update(params)
    try:
        r = requests.get(url, params=p, timeout=15)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and ("Error Message" in data or "message" in data):
            logging.warning(f"FMP: {data.get('Error Message') or data.get('message')}")
            return None
        return data
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        logging.error(f"HTTP {code}: {url.split('/')[-1]}")
        return None
    except Exception as e:
        logging.error(f"FMP request: {e}")
        return None


def _batch_market_caps(symbols: list[str]) -> dict[str, float]:
    """
    Uma só chamada FMP para obter market cap de vários símbolos.
    Devolve {symbol: market_cap}.
    """
    if not symbols:
        return {}

    result = {}
    chunk_size = 50

    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        joined = ",".join(chunk)
        data = _fmp_get(f"{BASE_V3}/quote/{joined}")
        if data:
            for item in data:
                sym = item.get("symbol")
                mc = item.get("marketCap") or 0
                if sym:
                    result[sym] = mc
        time.sleep(0.3)

    return result


def screen_big_drops(min_drop_pct: float = 10.0,
                     min_market_cap: int = 2_000_000_000) -> list[dict]:
    """
    1) Vai buscar biggest losers da FMP
    2) Confirma market cap em batch para os que vieram sem cap
    3) Filtra por market cap e preço mínimo
    """

    raw = []
    for url in [f"{BASE_STABLE}/biggest-losers", f"{BASE_V3}/biggest-losers"]:
        data = _fmp_get(url)
        if data:
            for item in data:
                sym = item.get("symbol") or item.get("ticker")
                if not sym:
                    continue

                chg = item.get("changesPercentage") or item.get("changePercentage") or 0
                if isinstance(chg, str):
                    try:
                        chg = float(chg.replace("(", "").replace(")", "").replace("%", "").strip())
                    except Exception:
                        continue

                if chg > -min_drop_pct:
                    continue

                raw.append({
                    "symbol": sym,
                    "name": item.get("name") or item.get("companyName") or sym,
                    "price": item.get("price") or 0,
                    "change_pct": chg,
                    "market_cap": item.get("marketCap") or 0,
                })
            if raw:
                break
        time.sleep(0.3)

    if not raw:
        logging.warning("FMP losers: sem resultados")
        return []

    logging.info(f"FMP losers brutos: {len(raw)}")

    unknown_mc = [s["symbol"] for s in raw if not s["market_cap"]]
    if unknown_mc:
        logging.info(f"A verificar market caps em batch: {len(unknown_mc)} tickers")
        mc_map = _batch_market_caps(unknown_mc)
        for s in raw:
            if not s["market_cap"] and s["symbol"] in mc_map:
                s["market_cap"] = mc_map[s["symbol"]]

    qualified = [
        s for s in raw
        if s["market_cap"]
        and s["market_cap"] >= min_market_cap
        and (s["price"] or 0) >= 5
    ]

    logging.info(f"Após filtro market cap/preço: {len(qualified)} acções")
    return qualified


def get_fundamentals(symbol: str) -> dict:
    """
    yfinance com retry e backoff.
    Chamado apenas para os finalistas.
    """
    result = {"symbol": symbol}

    for attempt in range(3):
        try:
            if attempt > 0:
                wait = 15 * attempt
                logging.warning(f"  {symbol}: rate limited — a aguardar {wait}s")
                time.sleep(wait)
            else:
                time.sleep(_YF_SLEEP)

            t = yf.Ticker(symbol)
            inf = t.info or {}

            mc = inf.get("marketCap") or 0
            if mc > 0 and mc < 1_000_000_000:
                result["skip"] = True
                return result

            result.update({
                "name": inf.get("longName") or inf.get("shortName") or symbol,
                "sector": inf.get("sector", ""),
                "industry": inf.get("industry", ""),
                "price": inf.get("currentPrice") or inf.get("regularMarketPrice"),
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
            })

            fcf = inf.get("freeCashflow")
            shares = inf.get("sharesOutstanding")
            if fcf and mc > 0:
                result["fcf_yield"] = fcf / mc
            if fcf and shares and shares > 0:
                result["fcf_per_share"] = fcf / shares

            target = inf.get("targetMeanPrice")
            price = result.get("price")
            if target and price and price > 0:
                result["analyst_upside"] = (target - price) / price * 100
                result["analyst_target"] = target

            return result

        except Exception as e:
            if "Too Many Requests" in str(e) or "Rate limit" in str(e):
                continue
            logging.error(f"  {symbol}: {e}")
            break

    return result


def get_news(symbol: str, limit: int = 3) -> list[dict]:
    try:
        time.sleep(2)
        news = yf.Ticker(symbol).news or []
        out = []
        for item in news[:limit]:
            content = item.get("content") or {}
            out.append({
                "title": content.get("title") or item.get("title", ""),
                "url": (content.get("canonicalUrl") or {}).get("url") or item.get("link", ""),
                "source": (content.get("provider") or {}).get("displayName") or "",
            })
        return out
    except Exception as e:
        logging.error(f"News {symbol}: {e}")
        return []


def get_historical_pe(symbol: str, years: int = 5) -> float | None:
    try:
        time.sleep(2)
        pe = (yf.Ticker(symbol).info or {}).get("trailingPE")
        return round(pe, 1) if pe and 0 < pe < 300 else None
    except Exception:
        return None
