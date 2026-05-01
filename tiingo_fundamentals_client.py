"""
tiingo_fundamentals_client.py — Dados fundamentais point-in-time via Tiingo.

O Tiingo Fundamentals API devolve balanços trimestrais históricos reais,
ao contrário do yfinance.info que só devolve valores actuais.

Endpoints usados:
  GET /tiingo/fundamentals/{ticker}/statements
      → Balanços trimestrais (income statement, balance sheet, cash flow)
      → Requer plano Starter ($10/mês) ou superior

  GET /tiingo/fundamentals/{ticker}/daily
      → P/E, Market Cap, EV diários históricos
      → Requer plano Starter

Plano gratuito Tiingo:
  - Não inclui /fundamentals — só OHLCV.
  - Se a chave não tiver acesso, as funções retornam None graciosamente.

Variável de ambiente obrigatória:
  TIINGO_API_KEY=<chave em api.tiingo.com>

Uso no bootstrap_ml.py:
  from tiingo_fundamentals_client import get_fundamentals_at

  fund = get_fundamentals_at("AAPL", date(2020, 3, 15))
  # fund = {
  #   'fcf_yield':       0.045,
  #   'revenue_growth':  0.08,
  #   'gross_margin':    0.38,
  #   'de_ratio':        1.2,
  #   'pe_vs_fair':      -0.15,   # (pe_actual - pe_fair) / pe_fair
  #   'analyst_upside':  0.12,
  #   'quality_score':   3,
  # }

Fallback:
  Todos os campos retornam None se o ticker não tiver cobertura
  ou se o plano não incluir fundamentals. O bootstrap_ml usa
  os valores de fallback definidos em ml_features.py.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta
from functools import lru_cache
from typing import Any

import requests

log = logging.getLogger("tiingo_fund")

_BASE  = "https://api.tiingo.com/tiingo/fundamentals"
_TOKEN = os.getenv("TIINGO_API_KEY", "")
_SLEEP = 0.25   # 4 req/s — bem abaixo do rate limit do plano Starter

# Cache em memória para evitar pedidos repetidos ao mesmo ticker/trimestre
# O LRU limita a 512 entradas (~512 ticker-trimestres em memória)
_STATEMENT_CACHE: dict[str, list[dict]] = {}
_DAILY_CACHE:     dict[str, list[dict]] = {}


def _get(
    url: str,
    params: dict,
    retries: int = 3,
) -> list[dict] | dict | None:
    """
    GET com retry e tratamento de rate-limit.
    Retorna None se 403 (plano sem acesso), [] se 404 (ticker sem dados).
    """
    if not _TOKEN:
        log.warning("[tiingo_fund] TIINGO_API_KEY não definida.")
        return None

    params["token"] = _TOKEN

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=20)

            if resp.status_code == 403:
                log.warning(
                    "[tiingo_fund] 403 Forbidden — plano sem acesso a Fundamentals. "
                    "Upgrade para Starter em api.tiingo.com."
                )
                return None

            if resp.status_code == 404:
                log.debug(f"[tiingo_fund] 404 — ticker sem cobertura: {url}")
                return []

            if resp.status_code == 429:
                wait = 60 * attempt
                log.warning(f"[tiingo_fund] Rate limit — a aguardar {wait}s")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            time.sleep(_SLEEP)
            return resp.json()

        except requests.exceptions.Timeout:
            log.warning(f"[tiingo_fund] Timeout (tentativa {attempt}/{retries})")
            time.sleep(2 * attempt)
        except requests.exceptions.RequestException as e:
            log.warning(f"[tiingo_fund] Erro de rede (tentativa {attempt}/{retries}): {e}")
            time.sleep(2 * attempt)

    return None


def _fetch_statements(ticker: str) -> list[dict]:
    """
    Descarrega todos os balanços trimestrais históricos de um ticker.
    Resultado em cache para evitar pedidos repetidos.
    """
    if ticker in _STATEMENT_CACHE:
        return _STATEMENT_CACHE[ticker]

    url  = f"{_BASE}/{ticker}/statements"
    data = _get(url, {"startDate": "2000-01-01"})

    if not data or not isinstance(data, list):
        _STATEMENT_CACHE[ticker] = []
        return []

    _STATEMENT_CACHE[ticker] = data
    return data


def _fetch_daily(ticker: str) -> list[dict]:
    """
    Descarrega métricas diárias históricas (P/E, MarketCap, EV/EBITDA).
    Resultado em cache.
    """
    if ticker in _DAILY_CACHE:
        return _DAILY_CACHE[ticker]

    url  = f"{_BASE}/{ticker}/daily"
    data = _get(url, {"startDate": "2000-01-01"})

    if not data or not isinstance(data, list):
        _DAILY_CACHE[ticker] = []
        return []

    _DAILY_CACHE[ticker] = data
    return data


def _nearest_statement(
    statements: list[dict],
    target: date,
    max_lag_days: int = 95,
) -> dict | None:
    """
    Devolve o trimestre mais recente com data de publicação <= target.
    max_lag_days: ignora trimestres publicados há mais de X dias antes de target.
    """
    best: dict | None = None
    best_date: date | None = None

    for s in statements:
        raw = s.get("date") or s.get("quarter") or ""
        try:
            d = date.fromisoformat(str(raw)[:10])
        except ValueError:
            continue

        if d > target:
            continue
        if (target - d).days > max_lag_days:
            continue

        if best_date is None or d > best_date:
            best_date = d
            best = s

    return best


def _nearest_daily(
    daily: list[dict],
    target: date,
) -> dict | None:
    """
    Devolve a linha diária mais próxima de target (até ±5 dias).
    """
    for delta in range(0, 6):
        for sign in (0, 1, -1):
            check = (target + timedelta(days=delta * sign)).isoformat()[:10]
            for row in daily:
                raw = str(row.get("date", ""))[:10]
                if raw == check:
                    return row
    return None


def _safe(val: Any, default: float | None = None) -> float | None:
    """Converte para float, retorna default se NaN/None/vazio."""
    try:
        v = float(val)
        import math
        return default if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return default


def get_fundamentals_at(
    ticker: str,
    alert_date: date,
) -> dict[str, float | None]:
    """
    Retorna os fundamentais point-in-time para um ticker numa data específica.

    Usa o trimestre mais recente publicado ANTES de alert_date (evita look-ahead).
    Para métricas de mercado (P/E, analyst_upside) usa a linha diária mais próxima.

    Campos retornados (alinhados com ml_features.FEATURE_COLUMNS):
      fcf_yield       — Free Cash Flow Yield  (FCF / Market Cap)
      revenue_growth  — Crescimento YoY da receita (%)
      gross_margin    — Margem bruta (%)
      de_ratio        — Debt-to-Equity ratio
      pe_vs_fair      — (P/E actual - P/E mediana 5 anos) / P/E mediana 5 anos
      analyst_upside  — Upside implícito do price target dos analistas (%)
      quality_score   — Score composto 0-5 (gerado internamente)

    Retorna None em cada campo se dados indisponíveis.
    Nunca lança excepção — falha silenciosa com fallback para o bootstrap.
    """
    result: dict[str, float | None] = {
        "fcf_yield":      None,
        "revenue_growth": None,
        "gross_margin":   None,
        "de_ratio":       None,
        "pe_vs_fair":     None,
        "analyst_upside": None,
        "quality_score":  None,
    }

    try:
        statements = _fetch_statements(ticker)
        daily      = _fetch_daily(ticker)

        stmt = _nearest_statement(statements, alert_date)
        day  = _nearest_daily(daily, alert_date)

        if stmt:
            # ── Income Statement ─────────────────────────────────────
            revenue      = _safe(stmt.get("revenue"))
            revenue_prev = _safe(stmt.get("revenueYOY"))   # Tiingo inclui YoY
            gross_profit = _safe(stmt.get("grossProfit"))
            ebitda       = _safe(stmt.get("ebitda"))

            if revenue and revenue > 0:
                # Revenue growth: Tiingo pode devolver directamente como %
                rev_growth_raw = stmt.get("revenueGrowth") or stmt.get("revenueYOY")
                if rev_growth_raw is not None:
                    result["revenue_growth"] = _safe(rev_growth_raw)
                elif revenue_prev and revenue_prev != 0:
                    result["revenue_growth"] = round((revenue - revenue_prev) / abs(revenue_prev) * 100, 2)

                if gross_profit is not None:
                    result["gross_margin"] = round(gross_profit / revenue * 100, 2)

            # ── Balance Sheet ─────────────────────────────────────────
            total_debt   = _safe(stmt.get("totalDebt")) or _safe(stmt.get("longTermDebt"))
            equity       = _safe(stmt.get("totalEquity")) or _safe(stmt.get("stockholdersEquity"))
            fcf          = _safe(stmt.get("freeCashFlow")) or _safe(stmt.get("fcf"))
            market_cap   = _safe(stmt.get("marketCap"))

            if total_debt is not None and equity and equity != 0:
                result["de_ratio"] = round(total_debt / abs(equity), 2)

            if fcf is not None and market_cap and market_cap > 0:
                result["fcf_yield"] = round(fcf / market_cap * 100, 2)

        if day:
            # ── Métricas diárias ──────────────────────────────────────
            pe_actual   = _safe(day.get("peRatio")) or _safe(day.get("pe"))
            pe_hist_med = _safe(day.get("peRatioMedian5y")) or _safe(day.get("peMedian"))

            if pe_actual is not None and pe_hist_med and pe_hist_med != 0:
                result["pe_vs_fair"] = round((pe_actual - pe_hist_med) / abs(pe_hist_med), 4)

            # analyst_upside: (price_target - price) / price
            price_target = _safe(day.get("priceTarget")) or _safe(day.get("analystPriceTarget"))
            price        = _safe(day.get("close")) or _safe(day.get("adjClose"))
            if price_target is not None and price and price > 0:
                result["analyst_upside"] = round((price_target - price) / price * 100, 2)

        # ── Quality Score composto ─────────────────────────────────────
        score = 0
        if result["gross_margin"] is not None and result["gross_margin"] > 40:
            score += 1
        if result["revenue_growth"] is not None and result["revenue_growth"] > 5:
            score += 1
        if result["fcf_yield"] is not None and result["fcf_yield"] > 3:
            score += 1
        if result["de_ratio"] is not None and result["de_ratio"] < 1.0:
            score += 1
        if result["analyst_upside"] is not None and result["analyst_upside"] > 10:
            score += 1
        result["quality_score"] = score

    except Exception as e:
        log.debug(f"[tiingo_fund] {ticker} @ {alert_date}: {e}")

    return result


def clear_cache() -> None:
    """Limpa o cache em memória (útil entre sessões longas do bootstrap)."""
    _STATEMENT_CACHE.clear()
    _DAILY_CACHE.clear()
    log.info("[tiingo_fund] Cache limpo.")


def check_fundamentals_access(ticker: str = "AAPL") -> bool:
    """
    Verifica se a API key tem acesso ao endpoint de Fundamentals.
    Retorna True se acessível, False se plano gratuito ou key inválida.
    """
    if not _TOKEN:
        log.warning("[tiingo_fund] TIINGO_API_KEY não definida.")
        return False

    url  = f"{_BASE}/{ticker}/daily"
    data = _get(url, {"startDate": "2024-01-01"})

    if data is None:
        log.warning("[tiingo_fund] Plano sem acesso a Fundamentals (403).")
        return False

    if isinstance(data, list) and len(data) > 0:
        log.info(f"[tiingo_fund] Acesso a Fundamentals confirmado ({len(data)} dias de {ticker}).")
        return True

    log.warning(f"[tiingo_fund] Dados vazios para {ticker} — verifica a key.")
    return False
