"""
Trigger: Yahoo Finance day_losers (gratuito, sem API key)
Fundamentais: yfinance (gratuito, sem API key)
Sem dependências pagas.
"""
import time
import logging
import requests
import yfinance as yf

# ── Headers para simular browser — evita bloqueio por bot detection ───────────
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

# Segundos entre chamadas yfinance
_YF_SLEEP = 4.0


# ── 1. LOSERS (Yahoo Finance) ─────────────────────────────────────────────────

def screen_big_drops(min_drop_pct: float = 10.0,
                     min_market_cap: int = 2_000_000_000) -> list[dict]:
    """
    Usa o screener predefinido do Yahoo Finance (day_losers).
    Gratuito, sem API key, funciona no Railway.
    """
    url = (
        "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
        "?formatted=false&scrIds=day_losers&count=100&start=0"
    )
    try:
        r = requests.get(url, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()

        quotes = (
            data.get("finance", {})
                .get("result", [{}])[0]
                .get("quotes", [])
        )
    except Exception as e:
        logging.error(f"Yahoo losers: {e}")
        return []

    results = []
    for q in quotes:
        chg = q.get("regularMarketChangePercent", 0) or 0
        if chg > -min_drop_pct:
            continue

        mc = q.get("marketCap") or 0
        if mc and mc < min_market_cap:
            continue

        # Saltar ETFs alavancados e warrants
        sym = q.get("symbol", "")
        qtype = q.get("quoteType", "")
        if qtype in ("ETF", "MUTUALFUND"):
            continue
        if len(sym) > 5:
            continue

        results.append({
            "symbol":      sym,
            "name":        q.get("longName") or q.get("shortName") or sym,
            "price":       q.get("regularMarketPrice"),
            "change_pct":  round(chg, 2),
            "market_cap":  mc,
        })

    logging.info(f"Yahoo day_losers: {len(results)} candidatos (queda >= {min_drop_pct}%, cap >= ${min_market_cap/1e9:.0f}B)")
    return results


# ── 2. FUNDAMENTAIS (yfinance) ────────────────────────────────────────────────

def get_fundamentals(symbol: str) -> dict:
    """Fundamentais via yfinance com backoff em caso de rate limit."""
    result = {"symbol": symbol}

    for attempt in range(3):
        try:
            wait = _YF_SLEEP if attempt == 0 else 20 * attempt
            time.sleep(wait)

            inf = yf.Ticker(symbol).info or {}

            mc = inf.get("marketCap") or 0
            # Verificar cap real — rejeitar micro-caps que escaparam ao filtro
            if mc > 0 and mc < 1_000_000_000:
                result["skip"] = True
                logging.info(f"  {symbol}: micro-cap ${mc/1e6:.0f}M — a saltar")
                return result

            result.update({
                "name":           inf.get("longName") or inf.get("shortName") or symbol,
                "sector":         inf.get("sector", ""),
                "industry":       inf.get("industry", ""),
                "price":          inf.get("currentPrice") or inf.get("regularMarketPrice"),
                "beta":           inf.get("beta"),
                "market_cap":     mc,
                "pe":             inf.get("trailingPE") or inf.get("forwardPE"),
                "revenue_growth": inf.get("revenueGrowth"),
                "gross_margin":   inf.get("grossMargins"),
                "ev_ebitda":      inf.get("enterpriseToEbitda"),
                "roe":            inf.get("returnOnEquity"),
                "debt_equity":    inf.get("debtToEquity"),
                "dividend_yield": inf.get("dividendYield"),
                "payout_ratio":   inf.get("payoutRatio"),
            })

            fcf    = inf.get("freeCashflow")
            shares = inf.get("sharesOutstanding")
            if fcf and mc > 0:
                result["fcf_yield"] = fcf / mc
            if fcf and shares and shares > 0:
                result["fcf_per_share"] = fcf / shares

            target = inf.get("targetMeanPrice")
            price  = result.get("price")
            if target and price and price > 0:
                result["analyst_upside"] = (target - price) / price * 100
                result["analyst_target"] = target

            return result

        except Exception as e:
            if "Too Many Requests" in str(e) or "429" in str(e):
                logging.warning(f"  {symbol}: rate limit (tentativa {attempt+1}/3)")
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
                "title":  content.get("title") or item.get("title", ""),
                "url":    (content.get("canonicalUrl") or {}).get("url") or item.get("link", ""),
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
    except:
        return None
