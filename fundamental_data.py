"""
fundamental_data.py — Dados fundamentais históricos ponto-a-ponto.

OBJECTIVO PRINCIPAL: eliminar look-ahead bias no backfill_fund.
  yfinance.info devolve um snapshot ACTUAL — usar em treino com alertas
  históricos contamina o modelo com dados futuros.

ESTRATÉGIA (três camadas, por ordem de prioridade):
  1. yfinance quarterly statements (income, balance, cashflow)
     → interpolamos o quarter mais recente ANTES da alert_date
     → disponível para todos os tickers (US + EU + outros)
  2. SimFin free API (https://simfin.com)
     → base histórica limpa com point-in-time reporting
     → gratuito para uso não-comercial, ~2000 empresas US
     → requer SIMFIN_API_KEY no .env (string "free" para tier free)
  3. Fallback neutro — valores medianos do sector para não bloquear treino

CONTRATO DE SAÍDA (alinhar com ml_features.FEATURE_COLUMNS):
  fcf_yield        float  — FCF / Market Cap na data do alerta
  revenue_growth   float  — YoY revenue growth (último quarter disponível)
  gross_margin     float  — Gross Profit / Revenue
  de_ratio         float  — Total Debt / Equity (em %)
  pe_vs_fair       float  — trailing P/E / sector fair P/E
  analyst_upside   float  — proxy: EPS forward vs trailing growth
  quality_score    float  — score composto 0–1

USO:
  from fundamental_data import get_fundamentals_at_date
  feats = get_fundamentals_at_date("MSFT", date(2021, 3, 15), price=230.0)
  # → dict com os 7 campos acima
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta
from functools import lru_cache
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

# ── Sector fair P/E (mediana histórica por sector GICS) ──────────────────────
_SECTOR_FAIR_PE: dict[str, float] = {
    "Technology":             35.0,
    "Healthcare":             22.0,
    "Communication Services": 22.0,
    "Financial Services":     13.0,
    "Financials":             13.0,
    "Consumer Cyclical":      20.0,
    "Consumer Defensive":     22.0,
    "Industrials":            20.0,
    "Energy":                 12.0,
    "Utilities":              18.0,
    "Real Estate":            40.0,
    "Basic Materials":        14.0,
    "Materials":              14.0,
    "Unknown":                22.0,
}

# Medianas sectoriais como fallback quando não há dados reais
_SECTOR_MEDIANS: dict[str, dict[str, float]] = {
    "Technology":         {"fcf_yield": 0.040, "revenue_growth": 0.12, "gross_margin": 0.60, "de_ratio": 40.0},
    "Healthcare":         {"fcf_yield": 0.035, "revenue_growth": 0.07, "gross_margin": 0.55, "de_ratio": 50.0},
    "Financials":         {"fcf_yield": 0.050, "revenue_growth": 0.05, "gross_margin": 0.45, "de_ratio": 200.0},
    "Consumer Cyclical":  {"fcf_yield": 0.030, "revenue_growth": 0.06, "gross_margin": 0.35, "de_ratio": 80.0},
    "Consumer Defensive": {"fcf_yield": 0.040, "revenue_growth": 0.04, "gross_margin": 0.38, "de_ratio": 70.0},
    "Industrials":        {"fcf_yield": 0.035, "revenue_growth": 0.05, "gross_margin": 0.32, "de_ratio": 90.0},
    "Energy":             {"fcf_yield": 0.060, "revenue_growth": 0.03, "gross_margin": 0.25, "de_ratio": 70.0},
    "Utilities":          {"fcf_yield": 0.045, "revenue_growth": 0.03, "gross_margin": 0.30, "de_ratio": 150.0},
    "Real Estate":        {"fcf_yield": 0.040, "revenue_growth": 0.04, "gross_margin": 0.55, "de_ratio": 120.0},
    "Materials":          {"fcf_yield": 0.040, "revenue_growth": 0.04, "gross_margin": 0.28, "de_ratio": 60.0},
    "Unknown":            {"fcf_yield": 0.040, "revenue_growth": 0.05, "gross_margin": 0.35, "de_ratio": 80.0},
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val: Any, default: float = np.nan) -> float:
    try:
        v = float(val)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except (TypeError, ValueError):
        return default


def _last_before(series: pd.Series, cutoff: pd.Timestamp) -> float:
    """
    Devolve o último valor de uma Series (indexada por Timestamp) cujo
    índice é ESTRITAMENTE ANTERIOR a cutoff. Respeita reporting lag:
    resultados publicados com ~45 dias de atraso.
    """
    # Aplicar reporting lag de 45 dias (tempo entre fim do quarter e publicação)
    effective_cutoff = cutoff - pd.Timedelta(days=45)
    valid = series[series.index < effective_cutoff].dropna()
    if valid.empty:
        return np.nan
    return float(valid.iloc[-1])


def _annual_from_quarterly(quarterly: pd.Series, cutoff: pd.Timestamp) -> float:
    """
    Soma dos últimos 4 quarters TTM (Trailing Twelve Months) antes de cutoff.
    Usado para revenue TTM, FCF TTM, etc.
    """
    effective_cutoff = cutoff - pd.Timedelta(days=45)
    valid = quarterly[quarterly.index < effective_cutoff].dropna()
    if len(valid) < 2:
        return np.nan
    ttm = valid.iloc[-4:].sum() if len(valid) >= 4 else valid.sum()
    return float(ttm)


# ─────────────────────────────────────────────────────────────────────────────
# Camada 1 — yfinance quarterly statements (ponto-a-ponto)
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=512)
def _fetch_yf_statements(ticker: str) -> dict[str, pd.DataFrame]:
    """
    Faz download dos quarterly statements e devolve dict com:
      income_q, balance_q, cashflow_q
    Cached por ticker para evitar pedidos repetidos durante o backfill.
    """
    try:
        tk = yf.Ticker(ticker)
        income   = tk.quarterly_income_stmt
        balance  = tk.quarterly_balance_sheet
        cashflow = tk.quarterly_cashflow

        def _normalise(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return pd.DataFrame()
            # Colunas = datas dos quarters → transpor para ter datas no índice
            df = df.T.copy()
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            return df

        return {
            "income":   _normalise(income),
            "balance":  _normalise(balance),
            "cashflow": _normalise(cashflow),
        }
    except Exception as e:
        log.debug(f"[yf_statements] {ticker}: {e}")
        return {"income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame()}


def _get_series(df: pd.DataFrame, *keys: str) -> pd.Series:
    """Tenta vários nomes de linha (yfinance muda nomes entre versões)."""
    for k in keys:
        if k in df.columns:
            return df[k]
    return pd.Series(dtype=float)


def _fundamentals_from_yf(
    ticker: str,
    alert_date: date,
    price: float,
    sector: str = "Unknown",
) -> dict[str, float] | None:
    """
    Extrai fundamentais históricos ponto-a-ponto do yfinance.
    Devolve None se não há dados suficientes para esta data.
    """
    stmts = _fetch_yf_statements(ticker)
    inc   = stmts["income"]
    bal   = stmts["balance"]
    cf    = stmts["cashflow"]

    if inc.empty and bal.empty:
        return None

    cutoff = pd.Timestamp(alert_date)

    # ── Revenue TTM e crescimento YoY ─────────────────────────────────────
    rev_s = _get_series(inc, "Total Revenue", "Revenue", "TotalRevenue")
    rev_ttm = _annual_from_quarterly(rev_s, cutoff) if not rev_s.empty else np.nan

    # Revenue do ano anterior (4 quarters antes dos últimos 4)
    rev_growth = np.nan
    if not rev_s.empty:
        effective_cutoff = cutoff - pd.Timedelta(days=45)
        valid_rev = rev_s[rev_s.index < effective_cutoff].dropna()
        if len(valid_rev) >= 8:
            rev_ttm_curr = valid_rev.iloc[-4:].sum()
            rev_ttm_prev = valid_rev.iloc[-8:-4].sum()
            if rev_ttm_prev != 0:
                rev_growth = float((rev_ttm_curr - rev_ttm_prev) / abs(rev_ttm_prev))

    # ── Gross Margin ──────────────────────────────────────────────────────
    gp_s  = _get_series(inc, "Gross Profit", "GrossProfit")
    gm    = np.nan
    if not gp_s.empty and not rev_s.empty:
        gp_ttm = _annual_from_quarterly(gp_s, cutoff)
        if pd.notna(rev_ttm) and rev_ttm != 0 and pd.notna(gp_ttm):
            gm = float(gp_ttm / rev_ttm)

    # ── Free Cash Flow Yield ──────────────────────────────────────────────
    # FCF = Operating Cash Flow - CapEx
    ocf_s   = _get_series(cf, "Operating Cash Flow", "Cash From Operations",
                           "CashFlowFromContinuingOperatingActivities")
    capex_s = _get_series(cf, "Capital Expenditure", "Purchases Of Property Plant And Equipment",
                          "CapitalExpenditure", "FreeCashFlow")

    fcf_ttm = np.nan
    if not ocf_s.empty:
        ocf_ttm   = _annual_from_quarterly(ocf_s, cutoff)
        capex_ttm = _annual_from_quarterly(capex_s, cutoff) if not capex_s.empty else 0.0
        if pd.notna(ocf_ttm):
            # CapEx é normalmente negativo no yfinance — adicionar
            capex_val = capex_ttm if pd.notna(capex_ttm) else 0.0
            fcf_ttm = ocf_ttm + capex_val  # capex_val já é negativo

    # Market Cap na data do alerta = preço × shares outstanding
    shares_s = _get_series(bal, "Ordinary Shares Number", "Share Issued",
                            "CommonStock", "CommonStockSharesOutstanding")
    shares_at_date = _last_before(shares_s, cutoff) if not shares_s.empty else np.nan
    mcap_at_date   = price * shares_at_date if pd.notna(shares_at_date) and shares_at_date > 0 else np.nan

    fcf_yield = np.nan
    if pd.notna(fcf_ttm) and pd.notna(mcap_at_date) and mcap_at_date > 0:
        fcf_yield = float(fcf_ttm / mcap_at_date)

    # ── D/E ratio ─────────────────────────────────────────────────────────
    debt_s   = _get_series(bal, "Total Debt", "Long Term Debt And Capital Lease Obligation",
                            "LongTermDebt", "TotalDebt")
    equity_s = _get_series(bal, "Stockholders Equity", "Total Stockholders Equity",
                            "CommonStockEquity", "TotalEquityGrossMinorityInterest")
    debt_val   = _last_before(debt_s, cutoff)   if not debt_s.empty   else np.nan
    equity_val = _last_before(equity_s, cutoff) if not equity_s.empty else np.nan
    de_ratio   = np.nan
    if pd.notna(debt_val) and pd.notna(equity_val) and equity_val != 0:
        de_ratio = float((debt_val / abs(equity_val)) * 100)

    # ── EPS-based P/E e pe_vs_fair ────────────────────────────────────────
    eps_s = _get_series(inc, "Diluted EPS", "Basic EPS", "EPS")
    pe_vs_fair = np.nan
    if not eps_s.empty:
        eps_ttm_series = eps_s.copy()
        effective_cutoff = cutoff - pd.Timedelta(days=45)
        valid_eps = eps_ttm_series[eps_ttm_series.index < effective_cutoff].dropna()
        if len(valid_eps) >= 4:
            eps_ttm = float(valid_eps.iloc[-4:].sum())
            if eps_ttm > 0 and price > 0:
                pe_trailing = price / eps_ttm
                fair_pe     = _SECTOR_FAIR_PE.get(sector, 22.0)
                pe_vs_fair  = float(pe_trailing / fair_pe)

    # ── analyst_upside proxy: forward EPS growth ──────────────────────────
    # Sem dados de consensus reais (pagos), usamos crescimento EPS YoY
    analyst_upside = np.nan
    if not eps_s.empty:
        valid_eps = eps_s[eps_s.index < (cutoff - pd.Timedelta(days=45))].dropna()
        if len(valid_eps) >= 8:
            eps_curr_ttm = float(valid_eps.iloc[-4:].sum())
            eps_prev_ttm = float(valid_eps.iloc[-8:-4].sum())
            if eps_prev_ttm != 0:
                analyst_upside = float((eps_curr_ttm - eps_prev_ttm) / abs(eps_prev_ttm))
            analyst_upside = float(np.clip(analyst_upside, -1.0, 2.0)) if pd.notna(analyst_upside) else np.nan

    # ── quality_score composto ────────────────────────────────────────────
    score = 0.5
    if pd.notna(fcf_yield)   and fcf_yield > 0.05:   score += 0.15
    if pd.notna(gm)          and gm > 0.40:           score += 0.10
    if pd.notna(de_ratio)    and de_ratio < 60:       score += 0.10
    if pd.notna(rev_growth)  and rev_growth > 0.08:   score += 0.10
    if pd.notna(analyst_upside) and analyst_upside > 0.10: score += 0.05
    if pd.notna(pe_vs_fair)  and pe_vs_fair < 1.0:    score += 0.10
    quality_score = float(min(score, 1.0))

    # Verificar se temos dados suficientes — mínimo de 3 campos reais
    real_fields = sum(1 for v in [fcf_yield, rev_growth, gm, de_ratio, pe_vs_fair] if pd.notna(v))
    if real_fields < 2:
        return None  # sem dados suficientes para esta data

    return {
        "fcf_yield":      _safe_float(fcf_yield,     0.04),
        "revenue_growth": _safe_float(rev_growth,    0.05),
        "gross_margin":   _safe_float(gm,            0.35),
        "de_ratio":       _safe_float(de_ratio,      80.0),
        "pe_vs_fair":     _safe_float(pe_vs_fair,    1.0),
        "analyst_upside": _safe_float(analyst_upside, 0.10),
        "quality_score":  quality_score,
        "_source":        "yfinance_quarterly",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Camada 2 — SimFin free API (US tickers, point-in-time)
# ─────────────────────────────────────────────────────────────────────────────

_SIMFIN_BASE = "https://backend.simfin.com/api/v3"
_simfin_cache: dict[str, dict] = {}  # ticker → raw SimFin data


def _simfin_api_key() -> str | None:
    key = os.environ.get("SIMFIN_API_KEY", "")
    return key if key else None


def _fetch_simfin_statements(ticker: str) -> dict:
    """Faz download dos statements do SimFin para um ticker US."""
    if ticker in _simfin_cache:
        return _simfin_cache[ticker]

    key = _simfin_api_key()
    if not key:
        return {}

    try:
        import requests
        headers = {"Authorization": f"api-key {key}"}
        # income statement (quarterly)
        inc_url = f"{_SIMFIN_BASE}/companies/statements/compact?ticker={ticker}&statements=pl&period=quarterly&fyear=all"
        bal_url = f"{_SIMFIN_BASE}/companies/statements/compact?ticker={ticker}&statements=bs&period=quarterly&fyear=all"
        cf_url  = f"{_SIMFIN_BASE}/companies/statements/compact?ticker={ticker}&statements=cf&period=quarterly&fyear=all"

        def _get(url: str) -> list:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                return r.json().get("data", [])
            return []

        result = {
            "income":   _get(inc_url),
            "balance":  _get(bal_url),
            "cashflow": _get(cf_url),
        }
        _simfin_cache[ticker] = result
        time.sleep(0.5)  # rate limit gentil
        return result
    except Exception as e:
        log.debug(f"[SimFin] {ticker}: {e}")
        _simfin_cache[ticker] = {}
        return {}


def _simfin_series(rows: list, col_name: str, col_map: dict[str, int]) -> pd.Series:
    """Converte lista de rows SimFin numa pd.Series indexada por Timestamp."""
    if not rows or col_name not in col_map:
        return pd.Series(dtype=float)
    col_idx = col_map[col_name]
    data = {}
    for row in rows:
        try:
            pub_date = pd.Timestamp(row[col_map.get("Publish Date", 1)])
            val      = float(row[col_idx])
            data[pub_date] = val
        except (IndexError, ValueError, TypeError):
            continue
    return pd.Series(data).sort_index() if data else pd.Series(dtype=float)


def _fundamentals_from_simfin(
    ticker: str,
    alert_date: date,
    price: float,
    sector: str = "Unknown",
) -> dict[str, float] | None:
    """Extrai fundamentais históricos ponto-a-ponto do SimFin free API."""
    # SimFin só tem tickers US sem sufixo de exchange
    if "." in ticker:  # EU/UK tickers — não suportados
        return None

    data = _fetch_simfin_statements(ticker)
    if not data or not data.get("income"):
        return None

    try:
        cutoff = pd.Timestamp(alert_date)

        # ── Income statement ──────────────────────────────────────────
        inc_rows = data["income"]
        if not inc_rows:
            return None
        # Extrair headers do primeiro item
        headers_map = {h: i for i, h in enumerate(inc_rows[0])} if isinstance(inc_rows[0], list) else {}
        inc_data    = inc_rows[1:] if headers_map else inc_rows

        rev_s = _simfin_series(inc_data, "Revenue", headers_map)
        gp_s  = _simfin_series(inc_data, "Gross Profit", headers_map)
        eps_s = _simfin_series(inc_data, "EPS Diluted", headers_map)

        # ── Balance sheet ─────────────────────────────────────────────
        bal_rows    = data.get("balance", [])
        bal_headers = {h: i for i, h in enumerate(bal_rows[0])} if bal_rows and isinstance(bal_rows[0], list) else {}
        bal_data    = bal_rows[1:] if bal_headers else bal_rows

        debt_s   = _simfin_series(bal_data, "Total Debt", bal_headers)
        equity_s = _simfin_series(bal_data, "Total Equity", bal_headers)
        shares_s = _simfin_series(bal_data, "Shares (Diluted)", bal_headers)

        # ── Cash flow ─────────────────────────────────────────────────
        cf_rows    = data.get("cashflow", [])
        cf_headers = {h: i for i, h in enumerate(cf_rows[0])} if cf_rows and isinstance(cf_rows[0], list) else {}
        cf_data    = cf_rows[1:] if cf_headers else cf_rows

        ocf_s   = _simfin_series(cf_data, "Net Cash from Operations", cf_headers)
        capex_s = _simfin_series(cf_data, "Capital Expenditures", cf_headers)

        # ── Calcular métricas com reporting lag ───────────────────────
        rev_ttm  = _annual_from_quarterly(rev_s, cutoff)  if not rev_s.empty  else np.nan
        gp_ttm   = _annual_from_quarterly(gp_s, cutoff)   if not gp_s.empty   else np.nan
        ocf_ttm  = _annual_from_quarterly(ocf_s, cutoff)  if not ocf_s.empty  else np.nan
        capex_val = _annual_from_quarterly(capex_s, cutoff) if not capex_s.empty else 0.0

        gm        = float(gp_ttm / rev_ttm)   if pd.notna(rev_ttm) and rev_ttm != 0 and pd.notna(gp_ttm)   else np.nan
        fcf_ttm   = (ocf_ttm + (capex_val or 0)) if pd.notna(ocf_ttm) else np.nan
        shares_at = _last_before(shares_s, cutoff)  if not shares_s.empty  else np.nan
        mcap_at   = price * shares_at * 1e6 if pd.notna(shares_at) else np.nan  # SimFin shares em milhares
        fcf_yield = float(fcf_ttm / mcap_at) if pd.notna(fcf_ttm) and pd.notna(mcap_at) and mcap_at > 0 else np.nan

        debt_val   = _last_before(debt_s, cutoff)   if not debt_s.empty   else np.nan
        equity_val = _last_before(equity_s, cutoff) if not equity_s.empty else np.nan
        de_ratio   = float((debt_val / abs(equity_val)) * 100) if pd.notna(debt_val) and pd.notna(equity_val) and equity_val != 0 else np.nan

        # Revenue growth YoY
        rev_growth = np.nan
        if not rev_s.empty:
            eff = cutoff - pd.Timedelta(days=45)
            valid_rev = rev_s[rev_s.index < eff].dropna()
            if len(valid_rev) >= 8:
                curr = valid_rev.iloc[-4:].sum()
                prev = valid_rev.iloc[-8:-4].sum()
                rev_growth = float((curr - prev) / abs(prev)) if prev != 0 else np.nan

        # EPS-based pe_vs_fair
        pe_vs_fair = np.nan
        if not eps_s.empty:
            eff = cutoff - pd.Timedelta(days=45)
            valid_eps = eps_s[eps_s.index < eff].dropna()
            if len(valid_eps) >= 4:
                eps_ttm = float(valid_eps.iloc[-4:].sum())
                if eps_ttm > 0 and price > 0:
                    pe_trailing = price / eps_ttm
                    pe_vs_fair  = pe_trailing / _SECTOR_FAIR_PE.get(sector, 22.0)

        # analyst_upside como EPS growth proxy
        analyst_upside = np.nan
        if not eps_s.empty:
            eff = cutoff - pd.Timedelta(days=45)
            valid_eps = eps_s[eps_s.index < eff].dropna()
            if len(valid_eps) >= 8:
                curr_e = float(valid_eps.iloc[-4:].sum())
                prev_e = float(valid_eps.iloc[-8:-4].sum())
                analyst_upside = float(np.clip((curr_e - prev_e) / abs(prev_e), -1.0, 2.0)) if prev_e != 0 else np.nan

        # quality_score
        score = 0.5
        if pd.notna(fcf_yield)      and fcf_yield > 0.05:       score += 0.15
        if pd.notna(gm)             and gm > 0.40:              score += 0.10
        if pd.notna(de_ratio)       and de_ratio < 60:          score += 0.10
        if pd.notna(rev_growth)     and rev_growth > 0.08:      score += 0.10
        if pd.notna(analyst_upside) and analyst_upside > 0.10:  score += 0.05
        if pd.notna(pe_vs_fair)     and pe_vs_fair < 1.0:       score += 0.10
        quality_score = float(min(score, 1.0))

        real_fields = sum(1 for v in [fcf_yield, rev_growth, gm, de_ratio, pe_vs_fair] if pd.notna(v))
        if real_fields < 2:
            return None

        return {
            "fcf_yield":      _safe_float(fcf_yield,     0.04),
            "revenue_growth": _safe_float(rev_growth,    0.05),
            "gross_margin":   _safe_float(gm,            0.35),
            "de_ratio":       _safe_float(de_ratio,      80.0),
            "pe_vs_fair":     _safe_float(pe_vs_fair,    1.0),
            "analyst_upside": _safe_float(analyst_upside, 0.10),
            "quality_score":  quality_score,
            "_source":        "simfin_free",
        }
    except Exception as e:
        log.debug(f"[SimFin] {ticker} {alert_date}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Camada 3 — Fallback neutro (medianas sectoriais)
# ─────────────────────────────────────────────────────────────────────────────

def _fundamentals_fallback(sector: str = "Unknown") -> dict[str, float]:
    """Fallback: medianas sectoriais históricas. Não gera look-ahead bias."""
    med = _SECTOR_MEDIANS.get(sector, _SECTOR_MEDIANS["Unknown"])
    return {
        "fcf_yield":      med["fcf_yield"],
        "revenue_growth": med["revenue_growth"],
        "gross_margin":   med["gross_margin"],
        "de_ratio":       med["de_ratio"],
        "pe_vs_fair":     1.0,
        "analyst_upside": 0.10,
        "quality_score":  0.50,
        "_source":        "sector_median_fallback",
    }


# ─────────────────────────────────────────────────────────────────────────────
# API Pública
# ─────────────────────────────────────────────────────────────────────────────

def get_fundamentals_at_date(
    ticker: str,
    alert_date: date,
    price: float,
    sector: str = "Unknown",
) -> dict[str, float]:
    """
    Ponto de entrada principal para o backfill_fund.

    Tenta as camadas em sequência:
      1. yfinance quarterly (histórico ponto-a-ponto, sem look-ahead)
      2. SimFin free API (US tickers, se SIMFIN_API_KEY definida)
      3. Fallback medianas sectoriais

    GARANTIA: nenhum dado retornado é posterior a alert_date (+ 45d lag).

    Args:
        ticker:     símbolo (e.g. "MSFT", "SAP.DE")
        alert_date: data do alerta de dip
        price:      preço de fecho no dia do alerta (para calcular mcap)
        sector:     sector GICS (para pe_vs_fair e fallbacks)

    Returns:
        dict com 7 chaves: fcf_yield, revenue_growth, gross_margin,
        de_ratio, pe_vs_fair, analyst_upside, quality_score
        + "_source" para logging (não entra no modelo)
    """
    # Camada 1: yfinance quarterly
    result = _fundamentals_from_yf(ticker, alert_date, price, sector)
    if result is not None:
        log.debug(f"[fund] {ticker} {alert_date}: yfinance_quarterly OK")
        return result

    # Camada 2: SimFin (apenas US)
    if "." not in ticker:
        result = _fundamentals_from_simfin(ticker, alert_date, price, sector)
        if result is not None:
            log.debug(f"[fund] {ticker} {alert_date}: simfin_free OK")
            return result

    # Camada 3: fallback neutro
    log.debug(f"[fund] {ticker} {alert_date}: sector_median_fallback")
    return _fundamentals_fallback(sector)


# ── Cache clear (para testes) ─────────────────────────────────────────────────
def clear_cache() -> None:
    _fetch_yf_statements.cache_clear()
    _simfin_cache.clear()


# ── CLI de teste ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    tests = [
        ("MSFT",   date(2020, 3, 20), 150.0, "Technology"),
        ("AAPL",   date(2022, 6, 15), 130.0, "Technology"),
        ("JNJ",    date(2021, 9, 10), 165.0, "Healthcare"),
        ("SAP.DE", date(2022, 1, 15), 120.0, "Technology"),  # EU — só yfinance
    ]
    for tk, dt, px, sec in tests:
        print(f"\n{'='*55}")
        print(f"{tk} @ {dt} (price={px})")
        r = get_fundamentals_at_date(tk, dt, px, sec)
        print(json.dumps(r, indent=2))
