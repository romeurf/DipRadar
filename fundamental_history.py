"""
fundamental_history.py — Fundamentais point-in-time com três camadas.

O problema central do DipRadar era usar dados fundamentais de HOJE para
avaliar dips históricos de 2020/2022 — look-ahead bias puro. Este módulo
resolve isso ao devolver os fundamentais que eram conhecidos NA DATA do alerta.

Camadas (por ordem de prioridade):

  Tier 1 — Tiingo Fundamentals (melhor, $10/mês Starter)
    Usa tiingo_fundamentals_client.get_fundamentals_at(ticker, alert_date).
    Balanços trimestrais reais com data de filing. Cobre US + alguns internacionais.
    Requer TIINGO_API_KEY com plano Starter.

  Tier 2 — SEC EDGAR XBRL (gratuito, US tickers)
    API pública https://data.sec.gov sem autenticação.
    Balanços 10-Q/10-K com data de filing exacta (verdadeiramente PIT).
    Rate limit: 10 req/s; cache local em JSON por ticker.
    Cobre apenas empresas listadas na SEC (praticamente todos os tickers US).

  Tier 3 — yfinance quarterly (gratuito, todos os tickers)
    quarterly_financials / quarterly_balance_sheet.
    Approximate PIT: assume filing 45 dias após o fim do trimestre.
    Cobre tickers internacionais (EU, JP, etc.) que não têm EDGAR.

API pública:
  get_pit_fundamentals(ticker, alert_date, cache_dir) → dict
  prefetch_fundamentals(tickers, cache_dir) → int (n tickers OK)
  has_tiingo_access() → bool
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

log = logging.getLogger(__name__)

_DATA_DIR  = Path("/data") if Path("/data").exists() else Path("/tmp")
_CACHE_DIR = _DATA_DIR / "fundamental_cache"

# Semáforo de acesso Tiingo (evita verificar a cada chamada)
_TIINGO_CHECKED: Optional[bool] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sf(val: Any, default: Optional[float] = None) -> Optional[float]:
    """safe float — retorna None se NaN/None/infinito."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if not math.isfinite(f) else f
    except (TypeError, ValueError):
        return default


def _is_us_ticker(ticker: str) -> bool:
    """Heurística: ticker sem ponto ou com sufixo .US/.NYSE é provavelmente US."""
    t = ticker.upper()
    # Não-US tickers têm sufixo como .L (London), .PA (Paris), .DE (Frankfurt)
    if "." in t:
        suffix = t.rsplit(".", 1)[-1]
        if suffix not in ("US", "NYSE", "NASDAQ", "OQ"):
            return False
    return True


# ── Tier 1: Tiingo ────────────────────────────────────────────────────────────

def has_tiingo_access() -> bool:
    """Verifica se TIINGO_API_KEY tem acesso ao endpoint de Fundamentals."""
    global _TIINGO_CHECKED
    if _TIINGO_CHECKED is not None:
        return _TIINGO_CHECKED
    if not os.getenv("TIINGO_API_KEY"):
        _TIINGO_CHECKED = False
        return False
    try:
        from tiingo_fundamentals_client import check_fundamentals_access
        _TIINGO_CHECKED = check_fundamentals_access()
    except Exception:
        _TIINGO_CHECKED = False
    return _TIINGO_CHECKED


def _tiingo_get(ticker: str, alert_date: date) -> dict:
    try:
        from tiingo_fundamentals_client import get_fundamentals_at
        result = get_fundamentals_at(ticker, alert_date)
        # Considera sucesso se pelo menos 2 campos não-None
        n_valid = sum(1 for v in result.values() if v is not None)
        return result if n_valid >= 2 else {}
    except Exception as e:
        log.debug(f"[fund_hist] Tiingo falhou {ticker}@{alert_date}: {e}")
        return {}


# ── Tier 2: SEC EDGAR XBRL ───────────────────────────────────────────────────

_EDGAR_BASE     = "https://data.sec.gov"   # XBRL / submissions / facts
_EDGAR_WWW_BASE = "https://www.sec.gov"    # company_tickers.json lives here
_EDGAR_HEADERS = {"User-Agent": "DipRadar research@dipradar.io"}

# Mapeamento CIK: carregado uma vez e cached em memória
_CIK_MAP: dict[str, str] = {}   # ticker → CIK (zero-padded 10 digits)
_CIK_MAP_LOADED = False


def _load_cik_map(cache_dir: Path) -> None:
    global _CIK_MAP, _CIK_MAP_LOADED
    if _CIK_MAP_LOADED:
        return
    cache_file = cache_dir / "sec_cik_map.json"
    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 86400 * 7:
        try:
            _CIK_MAP = json.loads(cache_file.read_text(encoding="utf-8"))
            _CIK_MAP_LOADED = True
            return
        except Exception:
            pass
    try:
        import requests
        resp = requests.get(
            f"{_EDGAR_WWW_BASE}/files/company_tickers.json",
            headers=_EDGAR_HEADERS, timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()
        # raw = {index: {cik_str: "...", ticker: "...", title: "..."}, ...}
        _CIK_MAP = {
            v["ticker"].upper(): str(v["cik_str"]).zfill(10)
            for v in raw.values()
            if "ticker" in v and "cik_str" in v
        }
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(_CIK_MAP), encoding="utf-8")
        log.info(f"[edgar] CIK map: {len(_CIK_MAP)} tickers")
        _CIK_MAP_LOADED = True
    except Exception as e:
        log.warning(f"[edgar] Falha a carregar CIK map: {e}")
        _CIK_MAP_LOADED = True  # não tentar de novo nesta sessão


def _get_cik(ticker: str, cache_dir: Path) -> Optional[str]:
    _load_cik_map(cache_dir)
    return _CIK_MAP.get(ticker.upper().replace(".", ""))


def _edgar_facts(cik: str, cache_dir: Path) -> dict:
    """Descarrega e cacheia o JSON de company facts do EDGAR."""
    cache_file = cache_dir / f"edgar_{cik}.json"
    # Cache de 30 dias (fundamentais mudam trimestralmente)
    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 86400 * 30:
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        import requests
        url = f"{_EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
        resp = requests.get(url, headers=_EDGAR_HEADERS, timeout=60)
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        data = resp.json()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data), encoding="utf-8")
        time.sleep(0.12)  # ~8 req/s (limit é 10)
        return data
    except Exception as e:
        log.debug(f"[edgar] facts {cik}: {e}")
        return {}


def _edgar_pit_value(
    facts: dict,
    alert_date: date,
    concept_names: list[str],
    taxonomy: str = "us-gaap",
    unit: str = "USD",
    form_filter: tuple[str, ...] = ("10-Q", "10-K"),
) -> Optional[float]:
    """Extrai o valor mais recente de um conceito EDGAR com filed_date <= alert_date."""
    gaap = facts.get("facts", {}).get(taxonomy, {})
    for concept in concept_names:
        entries = gaap.get(concept, {}).get("units", {}).get(unit, [])
        best_val: Optional[float] = None
        best_filed = date(1900, 1, 1)
        for e in entries:
            form = e.get("form", "")
            if form not in form_filter:
                continue
            # frame exclusão: ignorar instantâneos anuais duplicados
            frame = e.get("frame", "")
            try:
                filed = date.fromisoformat(str(e.get("filed", ""))[:10])
            except ValueError:
                continue
            if filed > alert_date:
                continue
            if filed > best_filed:
                best_filed = filed
                best_val = _sf(e.get("val"))
        if best_val is not None:
            return best_val
    return None


def _edgar_get(ticker: str, alert_date: date, cache_dir: Path) -> dict:
    """Fundamentais PIT via SEC EDGAR para um ticker US."""
    cik = _get_cik(ticker, cache_dir)
    if not cik:
        return {}
    facts = _edgar_facts(cik, cache_dir)
    if not facts:
        return {}

    # Revenue (TTM = soma dos últimos 4 trimestres)
    rev = _edgar_pit_value(facts, alert_date,
                           ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                            "SalesRevenueNet", "RevenueFromContractWithCustomer"],
                           form_filter=("10-Q", "10-K", "10-Q/A", "10-K/A"))

    gross_profit = _edgar_pit_value(facts, alert_date, ["GrossProfit"],
                                    form_filter=("10-Q", "10-K", "10-Q/A", "10-K/A"))

    op_cf = _edgar_pit_value(facts, alert_date,
                             ["NetCashProvidedByUsedInOperatingActivities"],
                             form_filter=("10-Q", "10-K", "10-Q/A", "10-K/A"))

    capex = _edgar_pit_value(facts, alert_date,
                             ["PaymentsToAcquirePropertyPlantAndEquipment",
                              "PaymentsForCapitalImprovements"],
                             form_filter=("10-Q", "10-K", "10-Q/A", "10-K/A"))

    total_debt = _edgar_pit_value(facts, alert_date,
                                  ["LongTermDebt", "LongTermDebtAndCapitalLeaseObligations",
                                   "LongTermDebtNoncurrent"],
                                  form_filter=("10-Q", "10-K", "10-Q/A", "10-K/A"))

    equity = _edgar_pit_value(facts, alert_date,
                              ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
                              form_filter=("10-Q", "10-K", "10-Q/A", "10-K/A"))

    mkt_cap = _edgar_pit_value(facts, alert_date,
                               ["CommonStockSharesOutstanding"],
                               unit="shares",
                               form_filter=("10-Q", "10-K", "10-Q/A", "10-K/A"))

    result: dict[str, Optional[float]] = {
        "gross_margin":   None,
        "de_ratio":       None,
        "fcf_yield":      None,
        "revenue_growth": None,
        "quality_score":  None,
        "pe_vs_fair":     None,
        "analyst_upside": None,
    }

    if rev and rev > 0:
        if gross_profit is not None:
            result["gross_margin"] = round(gross_profit / rev, 4)

    if total_debt is not None and equity and equity > 0:
        result["de_ratio"] = round(total_debt / equity, 4)

    fcf: Optional[float] = None
    if op_cf is not None:
        fcf = op_cf - (capex or 0)

    # FCF yield precisa de market cap em $ — mkt_cap das shares × preço
    # Sem preço, pular (o preço virá do price_cache no build_dataset)
    # Guardamos o FCF raw para usar no build_dataset onde o preço é conhecido
    result["_fcf_raw"] = fcf

    # Quality score simples
    score = 0.0
    if result["gross_margin"] is not None and result["gross_margin"] > 0.35:
        score += 0.25
    if result["de_ratio"] is not None and result["de_ratio"] < 2.0:
        score += 0.25
    if fcf is not None and fcf > 0:
        score += 0.25
    result["quality_score"] = round(score, 3)

    # Filtrar campos None → não retornar dict vazio
    valid = {k: v for k, v in result.items() if v is not None}
    return valid if len(valid) >= 2 else {}


# ── Tier 3: yfinance quarterly ────────────────────────────────────────────────

def _yf_quarterly(ticker: str, alert_date: date, cache_dir: Path) -> dict:
    """Approximate PIT fundamentals via yfinance quarterly data.

    yfinance devolve datas do fim do período (não de filing). Assume-se
    que o report fica público 45 dias após o fim do período.
    """
    cache_file = cache_dir / f"yf_{ticker.replace('.','_').replace('^','')}_q.json"
    raw: dict = {}

    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 86400 * 14:
        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            raw = {}

    if not raw:
        try:
            import yfinance as yf
            tk = yf.Ticker(ticker)

            def _df_to_dict(df) -> dict:
                if df is None or df.empty:
                    return {}
                # df.columns são timestamps; converter para str
                return {str(c)[:10]: {str(k): float(v) if isinstance(v, (int, float)) else None
                                      for k, v in df[c].items()}
                        for c in df.columns}

            fin_dict = _df_to_dict(tk.quarterly_financials)
            bs_dict  = _df_to_dict(tk.quarterly_balance_sheet)
            cf_dict  = _df_to_dict(tk.quarterly_cashflow)

            raw = {"fin": fin_dict, "bs": bs_dict, "cf": cf_dict}
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(raw), encoding="utf-8")
            time.sleep(0.3)
        except Exception as e:
            log.debug(f"[yf_q] {ticker}: {e}")
            return {}

    def _pit_period(d_dict: dict) -> Optional[dict]:
        """Retorna o período mais recente com assumed filing <= alert_date."""
        best_key = None
        best_date = date(1900, 1, 1)
        cutoff = alert_date
        for k in d_dict:
            try:
                period_end = date.fromisoformat(k[:10])
                filing_assumed = period_end + timedelta(days=45)
                if filing_assumed <= cutoff and period_end > best_date:
                    best_date = period_end
                    best_key = k
            except ValueError:
                continue
        return d_dict.get(best_key) if best_key else None

    fin = _pit_period(raw.get("fin", {})) or {}
    bs  = _pit_period(raw.get("bs",  {})) or {}
    cf  = _pit_period(raw.get("cf",  {})) or {}

    def _get(d: dict, *keys) -> Optional[float]:
        for k in keys:
            v = d.get(k)
            if v is not None:
                f = _sf(v)
                if f is not None:
                    return f
        return None

    revenue      = _get(fin, "Total Revenue", "Revenue")
    gross_profit = _get(fin, "Gross Profit")
    op_cf        = _get(cf,  "Total Cash From Operations", "Operating Cash Flow")
    capex        = _get(cf,  "Capital Expenditures")
    total_debt   = _get(bs,  "Long Term Debt", "Long-term Debt")
    equity       = _get(bs,  "Total Stockholder Equity", "Stockholders Equity")

    result: dict = {}

    if revenue and revenue > 0 and gross_profit is not None:
        result["gross_margin"] = round(gross_profit / revenue, 4)

    if total_debt is not None and equity and equity > 0:
        result["de_ratio"] = round(total_debt / equity, 4)

    fcf = None
    if op_cf is not None:
        fcf = op_cf + (capex or 0)  # capex já é negativo no yfinance
    result["_fcf_raw"] = fcf

    score = 0.0
    if result.get("gross_margin", 0) > 0.35:
        score += 0.25
    if result.get("de_ratio", 99) < 2.0:
        score += 0.25
    if fcf is not None and fcf > 0:
        score += 0.25
    result["quality_score"] = round(score, 3)

    valid = {k: v for k, v in result.items() if v is not None}
    return valid if len(valid) >= 2 else {}


# ── API pública ───────────────────────────────────────────────────────────────

def get_pit_fundamentals(
    ticker: str,
    alert_date: date,
    cache_dir: Path = _CACHE_DIR,
) -> dict:
    """Retorna os fundamentais point-in-time para (ticker, alert_date).

    Tenta Tiingo → SEC EDGAR → yfinance quarterly, por esta ordem.
    Retorna dict vazio se nenhuma fonte tiver dados.
    Nunca lança excepção — falha silenciosa com fallback.

    Campos possíveis:
      gross_margin   : Margem bruta (ratio, ex: 0.38)
      de_ratio       : Dívida / Capital próprio
      fcf_yield      : FCF / Market Cap (só se o preço for passado)
      revenue_growth : Crescimento YoY de revenue
      quality_score  : Score composto [0, 1]
      pe_vs_fair     : (PE actual - PE fair sector) / PE fair sector
      analyst_upside : Upside implícito dos analistas
      _fcf_raw       : FCF em USD (para o caller calcular fcf_yield com o preço)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Tier 1: Tiingo
    if has_tiingo_access():
        result = _tiingo_get(ticker, alert_date)
        if result:
            return result

    # Tier 2: SEC EDGAR (US tickers)
    if _is_us_ticker(ticker):
        try:
            result = _edgar_get(ticker, alert_date, cache_dir)
            if result:
                return result
        except Exception as e:
            log.debug(f"[fund_hist] EDGAR falhou {ticker}: {e}")

    # Tier 3: yfinance quarterly (todos os tickers)
    try:
        result = _yf_quarterly(ticker, alert_date, cache_dir)
        if result:
            return result
    except Exception as e:
        log.debug(f"[fund_hist] yfinance quarterly falhou {ticker}: {e}")

    return {}


def prefetch_fundamentals(
    tickers: list[str],
    cache_dir: Path = _CACHE_DIR,
    max_workers: int = 4,
    sleep_between: float = 0.5,
) -> int:
    """Pre-fetcha fundamentais para uma lista de tickers em paralelo.

    Usa threading para acelerar EDGAR/yfinance sem exceder rate limits.
    Retorna o número de tickers com dados OK.
    """
    import threading
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Para EDGAR, carregar o CIK map primeiro (evita race condition)
    _load_cik_map(cache_dir)

    ok = 0
    lock = threading.Lock()

    def _fetch_one(t: str) -> None:
        nonlocal ok
        # Para prefetch usamos uma data arbitrária recente —
        # o que interessa é popular o cache de dados históricos.
        # O cache EDGAR (30 dias) e yfinance (14 dias) serve o build_dataset.
        try:
            result = get_pit_fundamentals(t, date.today(), cache_dir)
            with lock:
                if result:
                    ok += 1
        except Exception as e:
            log.debug(f"[prefetch] {t}: {e}")
        finally:
            time.sleep(sleep_between / max_workers)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_fetch_one, t) for t in tickers]
        for i, f in enumerate(futures):
            f.result()
            if (i + 1) % 50 == 0:
                log.info(f"[prefetch] {i+1}/{len(tickers)} — ok={ok}")

    log.info(f"[prefetch] concluído: {ok}/{len(tickers)} tickers com dados")
    return ok
