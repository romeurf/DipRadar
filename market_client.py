"""
Trigger: Yahoo Finance day_losers (gratuito, sem API key)
Fundamentais: yfinance
Catalisadores: Tavily Search API (TAVILY_API_KEY)
"""

import os
import time
import logging
import requests
import yfinance as yf
from datetime import datetime, date, timezone, timedelta

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# ISIN do PPR Invest Tendências Globais (usado só para logging/referência)
# O proxy de preço usado é o ETF ACWI (iShares MSCI ACWI) via yfinance
PPR_ISIN = "PTARMJHM0003"

_CATALYST_KEYWORDS = {
    "earnings":    "📅 Earnings próximo",
    "results":     "📅 Resultados próximos",
    "fda":         "💊 Decisão FDA",
    "approval":    "✅ Aprovação regulatória",
    "buyback":     "🔁 Recompra de acções",
    "repurchase":  "🔁 Recompra de acções",
    "dividend":    "💰 Dividendo especial",
    "merger":      "🤝 M&A",
    "acquisition": "🤝 Aquisição",
    "spin":        "🔀 Spin-off",
    "split":       "✂️ Stock split",
    "guidance":    "📈 Guidance update",
    "upgrade":     "⬆️ Upgrade analistas",
    "contract":    "📄 Contrato importante",
    "partnership": "🤝 Parceria estratégica",
}

# ── Feriados NYSE hardcoded (anos 2025-2027) ───────────────────────────────────
_NYSE_HOLIDAYS = {
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17),
    date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19),
    date(2025, 7, 4), date(2025, 9, 1), date(2025, 11, 27),
    date(2025, 12, 25),
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3),  date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3),  date(2026, 9, 7),  date(2026, 11, 26),
    date(2026, 12, 25),
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15),
    date(2027, 3, 26), date(2027, 5, 31), date(2027, 6, 18),
    date(2027, 7, 5),  date(2027, 9, 6),  date(2027, 11, 25),
    date(2027, 12, 24),
}

# ── Horas de mercado NYSE/NASDAQ ───────────────────────────────────────
_MARKET_OPEN_UTC  = 13
_MARKET_OPEN_MIN  = 30
_MARKET_CLOSE_UTC = 20
_MARKET_CLOSE_MIN = 0


def is_market_open() -> bool:
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    if now.date() in _NYSE_HOLIDAYS:
        return False
    open_dt  = now.replace(hour=_MARKET_OPEN_UTC,  minute=_MARKET_OPEN_MIN,  second=0, microsecond=0)
    close_dt = now.replace(hour=_MARKET_CLOSE_UTC, minute=_MARKET_CLOSE_MIN, second=0, microsecond=0)
    return open_dt <= now <= close_dt


def get_macro_context() -> dict:
    """VIX actual + SPY vs média móvel 20 dias."""
    result = {"vix": None, "spy_vs_20d": None, "spy_price": None}
    try:
        time.sleep(2)
        vix = yf.Ticker("^VIX").info or {}
        result["vix"] = round(float(vix.get("regularMarketPrice", 0)), 1) or None
    except Exception as e:
        logging.warning(f"VIX: {e}")
    try:
        time.sleep(2)
        hist = yf.Ticker("SPY").history(period="30d", interval="1d")["Close"].dropna()
        if len(hist) >= 20:
            price_now = float(hist.iloc[-1])
            ma20 = float(hist.iloc[-20:].mean())
            result["spy_price"] = round(price_now, 2)
            result["spy_vs_20d"] = round((price_now - ma20) / ma20 * 100, 2)
    except Exception as e:
        logging.warning(f"SPY MA20: {e}")
    return result


def screen_global_dips(
    min_drop_pct: float = 10.0,
    min_market_cap: int = 2_000_000_000,
) -> list[dict]:
    url = (
        "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
        "?formatted=false&scrIds=day_losers&count=100&start=0"
    )
    try:
        r = requests.get(url, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        quotes = (
            r.json()
            .get("finance", {})
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
        sym   = q.get("symbol", "")
        qtype = q.get("quoteType", "")
        if qtype in ("ETF", "MUTUALFUND") or len(sym) > 5:
            continue
        results.append({
            "symbol":     sym,
            "name":       q.get("longName") or q.get("shortName") or sym,
            "price":      q.get("regularMarketPrice"),
            "change_pct": round(chg, 2),
            "market_cap": mc,
        })

    logging.info(
        f"Yahoo day_losers: {len(results)} candidatos "
        f"(>={min_drop_pct}%, cap>=${min_market_cap/1e9:.0f}B)"
    )
    return results


def screen_structural_dips(
    min_drawdown_pct: float = 25.0,
    min_market_cap: int = 2_000_000_000,
    max_results: int = 60,
) -> list[dict]:
    """
    Varre stocks no mínimo de 52 semanas (dips graduais/estruturais).
    Não depende de queda diária — apanha o PINS a $14, o TTD a $23, etc.
    Usa o endpoint undervalued_large_caps do Yahoo + filtra pelo drawdown.
    Corre uma vez por semana (segunda-feira 8h45).
    """
    candidates = []
    screener_ids = ["undervalued_large_caps", "day_losers", "most_actives"]

    for scrId in screener_ids:
        url = (
            "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
            f"?formatted=false&scrIds={scrId}&count=100&start=0"
        )
        try:
            r = requests.get(url, headers=_HEADERS, timeout=15)
            r.raise_for_status()
            quotes = (
                r.json()
                .get("finance", {})
                .get("result", [{}])[0]
                .get("quotes", [])
            )
            for q in quotes:
                sym   = q.get("symbol", "")
                qtype = q.get("quoteType", "")
                mc    = q.get("marketCap") or 0
                if not sym or qtype in ("ETF", "MUTUALFUND") or len(sym) > 5:
                    continue
                if mc and mc < min_market_cap:
                    continue
                price  = q.get("regularMarketPrice") or 0
                high52 = q.get("fiftyTwoWeekHigh") or 0
                if high52 and price and high52 > 0:
                    drawdown = (price - high52) / high52 * 100
                    if drawdown <= -min_drawdown_pct:
                        candidates.append({
                            "symbol":      sym,
                            "name":        q.get("longName") or q.get("shortName") or sym,
                            "price":       price,
                            "change_pct":  q.get("regularMarketChangePercent") or 0,
                            "market_cap":  mc,
                            "drawdown_52w": round(drawdown, 1),
                        })
        except Exception as e:
            logging.warning(f"Structural dip screener {scrId}: {e}")

    # Deduplica e ordena por drawdown mais profundo
    seen = set()
    unique = []
    for c in sorted(candidates, key=lambda x: x["drawdown_52w"]):
        if c["symbol"] not in seen:
            seen.add(c["symbol"])
            unique.append(c)

    logging.info(f"Structural dip scan: {len(unique)} candidatos (drawdown ≥{min_drawdown_pct}%)")
    return unique[:max_results]


def screen_period_dips(
    min_market_cap: int = 2_000_000_000,
    week_threshold: float = 10.0,
    month_threshold: float = 15.0,
    max_results: int = 20,
) -> dict[str, list[dict]]:
    """
    Detecta dips acumulados na última semana (7d) e último mês (30d)
    usando yfinance.history() — 100% gratuito.
    """
    screener_ids = ["undervalued_large_caps", "day_losers", "most_actives"]
    candidates: dict[str, dict] = {}

    for scrId in screener_ids:
        url = (
            "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
            f"?formatted=false&scrIds={scrId}&count=100&start=0"
        )
        try:
            r = requests.get(url, headers=_HEADERS, timeout=15)
            r.raise_for_status()
            quotes = (
                r.json()
                .get("finance", {})
                .get("result", [{}])[0]
                .get("quotes", [])
            )
            for q in quotes:
                sym   = q.get("symbol", "")
                qtype = q.get("quoteType", "")
                mc    = q.get("marketCap") or 0
                if not sym or qtype in ("ETF", "MUTUALFUND") or len(sym) > 5:
                    continue
                if mc and mc < min_market_cap:
                    continue
                if sym not in candidates:
                    candidates[sym] = {
                        "name":       q.get("longName") or q.get("shortName") or sym,
                        "market_cap": mc,
                    }
        except Exception as e:
            logging.warning(f"Period dip screener {scrId}: {e}")

    logging.info(f"Period dip screener: {len(candidates)} tickers candidatos")

    weekly_dips:  list[dict] = []
    monthly_dips: list[dict] = []

    for sym, meta in candidates.items():
        try:
            time.sleep(1)
            hist = yf.Ticker(sym).history(period="35d", interval="1d")["Close"].dropna()
            if len(hist) < 6:
                continue

            price_now = float(hist.iloc[-1])

            if len(hist) >= 6:
                price_5d  = float(hist.iloc[-6])
                chg_week  = (price_now - price_5d) / price_5d * 100
                if chg_week <= -week_threshold:
                    weekly_dips.append({
                        "symbol":     sym,
                        "name":       meta["name"],
                        "price":      round(price_now, 2),
                        "change_pct": round(chg_week, 2),
                        "market_cap": meta["market_cap"],
                        "period":     "7d",
                    })

            if len(hist) >= 23:
                price_22d = float(hist.iloc[-23])
                chg_month = (price_now - price_22d) / price_22d * 100
                if chg_month <= -month_threshold:
                    monthly_dips.append({
                        "symbol":     sym,
                        "name":       meta["name"],
                        "price":      round(price_now, 2),
                        "change_pct": round(chg_month, 2),
                        "market_cap": meta["market_cap"],
                        "period":     "30d",
                    })
        except Exception as e:
            logging.warning(f"Period dip {sym}: {e}")

    weekly_dips.sort(key=lambda x: x["change_pct"])
    monthly_dips.sort(key=lambda x: x["change_pct"])

    weekly_syms  = {s["symbol"] for s in weekly_dips}
    monthly_dips = [s for s in monthly_dips if s["symbol"] not in weekly_syms]

    logging.info(
        f"Period dips: {len(weekly_dips)} weekly (≥{week_threshold}%) | "
        f"{len(monthly_dips)} monthly (≥{month_threshold}%)"
    )
    return {
        "weekly":  weekly_dips[:max_results],
        "monthly": monthly_dips[:max_results],
    }


def get_spy_change() -> float | None:
    try:
        time.sleep(2)
        inf = yf.Ticker("SPY").info or {}
        chg = inf.get("regularMarketChangePercent")
        if chg is not None:
            return round(float(chg), 2)
    except Exception as e:
        logging.warning(f"SPY change: {e}")
    return None


def get_usdeur() -> float:
    """
    Devolve a taxa de conversão USD → EUR (ex: 0.8550 significa 1 USD = €0.8550).

    O ticker EURUSD=X no Yahoo Finance representa EUR/USD
    (quantos USD vale 1 EUR), por exemplo 1.1705.
    Para converter USD → EUR é necessário inverter: 1 / 1.1705 ≈ 0.8544.

    Fallback: 0.92 (aproximação conservadora).
    """
    try:
        time.sleep(1)
        inf = yf.Ticker("EURUSD=X").info or {}
        # EURUSD=X = EUR/USD, i.e. 1 EUR = X USD  →  1 USD = 1/X EUR
        eur_usd = inf.get("regularMarketPrice") or inf.get("bid")
        if eur_usd and eur_usd > 0:
            usd_eur = round(1.0 / float(eur_usd), 6)
            logging.info(f"FX: EUR/USD={eur_usd:.4f} → USD/EUR={usd_eur:.4f}")
            return usd_eur
    except Exception as e:
        logging.warning(f"USD/EUR rate: {e}")
    return 0.92


def get_rsi(symbol: str, period: int = 14) -> float | None:
    try:
        time.sleep(3)
        hist = yf.Ticker(symbol).history(period="60d", interval="1d")["Close"]
        if hist is None or len(hist) < period + 1:
            return None
        delta    = hist.diff().dropna()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.rolling(period).mean().iloc[-1]
        avg_loss = loss.rolling(period).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        rs  = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return round(float(rsi), 1)
    except Exception as e:
        logging.warning(f"RSI {symbol}: {e}")
        return None


def get_historical_pe(symbol: str, years: int = 3) -> dict | None:
    try:
        time.sleep(5)
        ticker = yf.Ticker(symbol)
        inf    = ticker.info or {}
        eps    = inf.get("trailingEps")
        if not eps or eps <= 0:
            return None
        hist = ticker.history(period=f"{years}y", interval="1mo")["Close"]
        if hist is None or len(hist) < 6:
            return None
        pe_series = hist / eps
        pe_series = pe_series[(pe_series > 0) & (pe_series < 500)]
        if len(pe_series) < 4:
            return None
        return {
            "pe_hist_avg":    round(float(pe_series.mean()), 1),
            "pe_hist_median": round(float(pe_series.median()), 1),
            "pe_hist_min":    round(float(pe_series.min()), 1),
            "pe_hist_max":    round(float(pe_series.max()), 1),
        }
    except Exception as e:
        logging.warning(f"Historical PE {symbol}: {e}")
        return None


def _yf_info(symbol: str) -> dict:
    for attempt in range(4):
        wait = 10 + (20 * attempt)
        time.sleep(wait)
        try:
            inf = yf.Ticker(symbol).info
            if inf and len(inf) > 5:
                return inf
            logging.warning(f"  {symbol}: resposta vazia (tentativa {attempt+1}/4)")
        except Exception as e:
            err = str(e)
            if any(x in err for x in ("429", "Too Many Requests", "Rate limit", "401", "406")):
                logging.warning(f"  {symbol}: rate/auth error (tentativa {attempt+1}/4) — {wait}s")
            else:
                logging.error(f"  {symbol}: {e}")
                break
    return {}


def _normalize_dividend_yield(raw: float | None) -> float | None:
    """
    Normaliza o dividendYield devolvido pelo yfinance para decimal consistente.

    O yfinance devolve tipicamente valores decimais (0.0263 = 2.63%), mas
    alguns tickers devolvem o valor já em percentagem (2.63 em vez de 0.0263).
    Esta função garante que o resultado é SEMPRE decimal.

    Exemplos:
      0.0263  →  0.0263  (2.63% — já decimal)
      2.63    →  0.0263  (2.63% — convertido)
      0.0     →  None
      None    →  None

    Nota: valores > 0.50 (50%) são impossíveis como yield real num stock
    saudável, portanto são tratados como já estando em percentagem e
    divididos por 100. Limite superior de segurança: 50% (0.50 decimal).
    Tickers com yield genuinamente >50% são raros e geralmente sinais de
    distress que o score já penaliza de outras formas.
    """
    if raw is None or raw <= 0:
        return None
    # yfinance devolve > 0.50 → assume que já está em percentagem
    return float(raw) / 100 if raw > 0.50 else float(raw)


def _normalize_ratio(raw: float | None) -> float | None:
    """
    Normaliza ratios como payout_ratio, gross_margin, roe, revenue_growth
    que o yfinance devolve consistentemente como decimais.
    Não requer ajuste — apenas garante que None é devolvido para valores None.
    """
    if raw is None:
        return None
    return float(raw)


def get_fundamentals(symbol: str, region: str = "", min_market_cap: int = 2_000_000_000) -> dict:
    result = {"symbol": symbol}
    inf    = _yf_info(symbol)
    if not inf:
        logging.error(f"  {symbol}: falhou após todas as tentativas")
        return result

    mc = inf.get("marketCap") or 0
    if mc > 0 and mc < min_market_cap:
        result["skip"] = True
        logging.info(f"  {symbol}: cap ${mc/1e9:.1f}B < mínimo ${min_market_cap/1e9:.0f}B — a saltar")
        return result

    price       = inf.get("currentPrice") or inf.get("regularMarketPrice")
    week52_high = inf.get("fiftyTwoWeekHigh")
    drawdown_from_high = None
    if week52_high and price and week52_high > 0:
        drawdown_from_high = round((price - week52_high) / week52_high * 100, 1)

    result.update({
        "name":             inf.get("longName") or inf.get("shortName") or symbol,
        "sector":           inf.get("sector", ""),
        "industry":         inf.get("industry", ""),
        "price":            price,
        "beta":             inf.get("beta"),
        "market_cap":       mc,
        "pe":               inf.get("trailingPE") or inf.get("forwardPE"),
        "revenue_growth":   _normalize_ratio(inf.get("revenueGrowth")),
        "gross_margin":     _normalize_ratio(inf.get("grossMargins")),
        "ev_ebitda":        inf.get("enterpriseToEbitda"),
        "roe":              _normalize_ratio(inf.get("returnOnEquity")),
        "debt_equity":      inf.get("debtToEquity"),
        # dividend_yield normalizado para SEMPRE decimal (ex: 0.0263 = 2.63%)
        "dividend_yield":   _normalize_dividend_yield(inf.get("dividendYield")),
        "payout_ratio":     _normalize_ratio(inf.get("payoutRatio")),
        "week52_high":      week52_high,
        "drawdown_from_high": drawdown_from_high,
        "volume":           inf.get("volume") or inf.get("regularMarketVolume"),
        "average_volume":   inf.get("averageVolume") or inf.get("averageDailyVolume10Day"),
    })

    fcf    = inf.get("freeCashflow")
    shares = inf.get("sharesOutstanding")
    if fcf is not None and mc > 0:
        result["fcf_yield"] = fcf / mc
    if fcf and shares and shares > 0:
        result["fcf_per_share"] = fcf / shares

    target = inf.get("targetMeanPrice")
    if target and price and price > 0:
        result["analyst_upside"] = (target - price) / price * 100
        result["analyst_target"] = target

    return result


def get_earnings_days(symbol: str) -> int | None:
    """
    Devolve o número de dias até aos próximos earnings (0-45 dias)
    ou None se desconhecido / fora do intervalo.
    """
    try:
        time.sleep(3)
        cal = yf.Ticker(symbol).calendar
        if cal is None:
            return None
        if hasattr(cal, "to_dict"):
            cal = cal.to_dict()
        earnings_raw = None
        for key in ("Earnings Date", "earningsDate", "Earnings High"):
            val = cal.get(key)
            if val is not None:
                earnings_raw = val
                break
        if earnings_raw is None:
            return None
        if isinstance(earnings_raw, list):
            earnings_raw = earnings_raw[0]
        if isinstance(earnings_raw, (int, float)):
            dt = datetime.fromtimestamp(earnings_raw, tz=timezone.utc)
        elif hasattr(earnings_raw, "to_pydatetime"):
            dt = earnings_raw.to_pydatetime()
        else:
            return None
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        days = (dt - datetime.now(tz=timezone.utc)).days
        return days if 0 <= days <= 45 else None
    except Exception as e:
        logging.warning(f"Earnings days {symbol}: {e}")
        return None


def get_earnings_date(symbol: str) -> str | None:
    """Compat wrapper: devolve string dd/mm/yyyy ou None."""
    days = get_earnings_days(symbol)
    if days is None:
        return None
    dt = datetime.now(tz=timezone.utc) + timedelta(days=days)
    return dt.strftime("%d/%m/%Y")


def get_portfolio_snapshot(holdings, cashback_eur, ppr_shares, ppr_avg_cost, usd_eur):
    """
    Calcula o snapshot actual da carteira.

    PPR (ISIN {PPR_ISIN}):
      O fundo não tem ticker directo no Yahoo Finance.
      Usamos o ACWI (iShares MSCI ACWI ETF) como proxy de preço.
      O valor real do PPR = ppr_shares * acwi_price * usd_eur.
      O custo histórico = ppr_shares * ppr_avg_cost.
    """
    from portfolio import USD_TICKERS, EUR_TICKERS

    positions  = []
    total_eur  = 0.0
    total_cost = 0.0
    price_cache: dict[str, dict] = {}

    def _get_prices(symbol: str) -> dict:
        if symbol in price_cache:
            return price_cache[symbol]
        try:
            time.sleep(2)
            hist = yf.Ticker(symbol).history(period="35d", interval="1d")["Close"].dropna()
            if len(hist) < 2:
                return {}
            r = {
                "now":       float(hist.iloc[-1]),
                "yesterday": float(hist.iloc[-2]) if len(hist) >= 2 else None,
                "week_ago":  float(hist.iloc[-6]) if len(hist) >= 6 else None,
                "month_ago": float(hist.iloc[-22]) if len(hist) >= 22 else None,
            }
            price_cache[symbol] = r
            return r
        except Exception as e:
            logging.warning(f"Portfolio price {symbol}: {e}")
            return {}

    for symbol, shares, avg_cost in holdings:
        if not shares:
            continue
        prices = _get_prices(symbol)
        if not prices:
            continue

        fx        = usd_eur if symbol in USD_TICKERS else 1.0
        price_now = prices["now"] * fx
        value_eur = shares * price_now

        pnl_d = ((prices["now"] - prices["yesterday"]) * fx * shares
                 if prices.get("yesterday") else None)
        pnl_w = ((prices["now"] - prices["week_ago"]) * fx * shares
                 if prices.get("week_ago") else None)
        pnl_m = ((prices["now"] - prices["month_ago"]) * fx * shares
                 if prices.get("month_ago") else None)

        pnl_total = cost_eur = None
        if avg_cost:
            cost_eur  = shares * avg_cost
            pnl_total = value_eur - cost_eur
            total_cost += cost_eur

        positions.append({
            "symbol":    symbol,
            "shares":    shares,
            "price_eur": round(price_now, 4),
            "value_eur": round(value_eur, 2),
            "pnl_day":   round(pnl_d, 2) if pnl_d is not None else None,
            "pnl_week":  round(pnl_w, 2) if pnl_w is not None else None,
            "pnl_month": round(pnl_m, 2) if pnl_m is not None else None,
            "pnl_total": round(pnl_total, 2) if pnl_total is not None else None,
        })
        total_eur += value_eur

    # ── CashBack Pie ────────────────────────────────────────────────────────
    cashback_total = sum(cashback_eur.values())
    total_eur     += cashback_total
    pie_pnl_day    = 0.0
    for sym, val_eur in cashback_eur.items():
        p = _get_prices(sym)
        if p and p.get("yesterday") and p["yesterday"] > 0:
            pie_pnl_day += val_eur * (p["now"] - p["yesterday"]) / p["yesterday"]

    # ── PPR (ISIN PTARMJHM0003, proxy ACWI) ────────────────────────────────────
    # Valor real = shares * preço_actual_ACWI * usd_eur
    # Custo histórico = shares * avg_cost (em EUR, já convertido na compra)
    ppr_cost      = ppr_shares * ppr_avg_cost   # custo total histórico em EUR
    ppr_value_eur = ppr_cost                     # fallback se ACWI falhar

    ppr_pnl_day = ppr_pnl_week = ppr_pnl_month = None
    pp = _get_prices("ACWI")
    if pp and pp.get("now"):
        acwi_now       = pp["now"] * usd_eur
        ppr_value_eur  = ppr_shares * acwi_now   # ← valor real de mercado

        if pp.get("yesterday") and pp["yesterday"] > 0:
            acwi_yest     = pp["yesterday"] * usd_eur
            ppr_pnl_day   = ppr_shares * (acwi_now - acwi_yest)
        if pp.get("week_ago") and pp["week_ago"] > 0:
            acwi_w        = pp["week_ago"] * usd_eur
            ppr_pnl_week  = ppr_shares * (acwi_now - acwi_w)
        if pp.get("month_ago") and pp["month_ago"] > 0:
            acwi_m        = pp["month_ago"] * usd_eur
            ppr_pnl_month = ppr_shares * (acwi_now - acwi_m)

        logging.info(
            f"PPR ({PPR_ISIN}): {ppr_shares} UP × €{acwi_now:.4f}/UP "
            f"= €{ppr_value_eur:.2f} (custo €{ppr_cost:.2f})"
        )
    else:
        logging.warning(f"PPR ({PPR_ISIN}): ACWI sem dados, a usar custo histórico como fallback")

    total_eur  += ppr_value_eur
    total_cost += ppr_cost

    # ── Agregação P&L ─────────────────────────────────────────────────────
    agg_day   = sum(p["pnl_day"]   for p in positions if p["pnl_day"]   is not None) + pie_pnl_day
    agg_week  = sum(p["pnl_week"]  for p in positions if p["pnl_week"]  is not None)
    agg_month = sum(p["pnl_month"] for p in positions if p["pnl_month"] is not None)
    if ppr_pnl_day   is not None: agg_day   += ppr_pnl_day
    if ppr_pnl_week  is not None: agg_week  += ppr_pnl_week
    if ppr_pnl_month is not None: agg_month += ppr_pnl_month

    return {
        "total_eur":    round(total_eur, 2),
        "total_cost":   round(total_cost, 2),
        "pnl_day":      round(agg_day, 2),
        "pnl_week":     round(agg_week, 2),
        "pnl_month":    round(agg_month, 2),
        "pnl_total":    round(total_eur - total_cost, 2) if total_cost > 0 else None,
        "positions":    positions,
        "cashback_eur": cashback_total,
        "ppr_value":    round(ppr_value_eur, 2),
        "usd_eur":      usd_eur,
    }


def get_catalyst(symbol: str, company_name: str = "") -> dict:
    if not TAVILY_API_KEY:
        return {"found": False, "label": "⚠️ Tavily não configurado", "snippet": ""}
    name  = company_name or symbol
    query = f"{symbol} {name} catalyst earnings FDA buyback guidance 2026"
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": query,
                  "search_depth": "basic", "max_results": 5, "include_answer": True},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logging.warning(f"Tavily {symbol}: {e}")
        return {"found": False, "label": "⚠️ Sem catalisador identificado", "snippet": ""}

    texts = []
    if data.get("answer"):
        texts.append(data["answer"].lower())
    for r in data.get("results", []):
        texts.append((r.get("content") or "").lower())
        texts.append((r.get("title") or "").lower())
    full_text = " ".join(texts)

    for keyword, label in _CATALYST_KEYWORDS.items():
        if keyword in full_text:
            snippet = ""
            for r in data.get("results", []):
                content = r.get("content") or ""
                idx = content.lower().find(keyword)
                if idx != -1:
                    snippet = content[max(0, idx-40):min(len(content), idx+100)].strip()
                    break
            return {"found": True, "label": label, "snippet": snippet[:120]}

    return {"found": False, "label": "⚠️ Sem catalisador identificado", "snippet": ""}


def get_52w_drawdown(symbol: str) -> float | None:
    try:
        time.sleep(3)
        inf   = yf.Ticker(symbol).info or {}
        high  = inf.get("fiftyTwoWeekHigh")
        price = inf.get("currentPrice") or inf.get("regularMarketPrice")
        if high and price and high > 0:
            return round((price - high) / high * 100, 1)
    except Exception as e:
        logging.warning(f"52w drawdown {symbol}: {e}")
    return None


def get_news(symbol: str, limit: int = 3) -> list[dict]:
    try:
        time.sleep(5)
        news = yf.Ticker(symbol).news or []
        out  = []
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
