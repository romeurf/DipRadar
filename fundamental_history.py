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


# Only the XBRL concepts we actually query — strip everything else before caching.
# Raw EDGAR company facts JSON is 2–20 MB per ticker; keeping only these
# concepts reduces it to ~20–80 KB, preventing disk overflow on multi-ticker builds.
_NEEDED_CONCEPTS: set[str] = {
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomer",
    "SalesRevenueNet",
    "GrossProfit",
    "NetCashProvidedByUsedInOperatingActivities",
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsForCapitalImprovements",
    "LongTermDebt",
    "LongTermDebtAndCapitalLeaseObligations",
    "LongTermDebtNoncurrent",
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "NetIncomeLoss",
    "ProfitLoss",
    "CommonStockSharesOutstanding",
}


def _slim_facts(data: dict) -> dict:
    """Keep only needed XBRL concepts from a raw EDGAR company facts response."""
    gaap = data.get("facts", {}).get("us-gaap", {})
    slim_gaap = {k: v for k, v in gaap.items() if k in _NEEDED_CONCEPTS}
    return {
        "entityName": data.get("entityName", ""),
        "facts": {"us-gaap": slim_gaap},
    }


def _edgar_facts(cik: str, cache_dir: Path) -> dict:
    """Download and cache a slim EDGAR company facts dict for one CIK."""
    cache_file = cache_dir / f"edgar_{cik}.json"
    # 30-day cache (fundamentals change quarterly)
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
        data = _slim_facts(resp.json())   # strip to needed concepts before caching
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data), encoding="utf-8")
        time.sleep(0.12)  # ~8 req/s (limit is 10)
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

    net_income = _edgar_pit_value(facts, alert_date,
                                  ["NetIncomeLoss", "ProfitLoss"],
                                  form_filter=("10-Q", "10-K", "10-Q/A", "10-K/A"))

    mkt_cap = _edgar_pit_value(facts, alert_date,
                               ["CommonStockSharesOutstanding"],
                               unit="shares",
                               form_filter=("10-Q", "10-K", "10-Q/A", "10-K/A"))

    result: dict[str, Optional[float]] = {
        "gross_margin":   None,
        "de_ratio":       None,
        "roe":            None,
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

    if net_income is not None and equity and equity > 0:
        result["roe"] = round(net_income / abs(equity), 4)

    fcf: Optional[float] = None
    if op_cf is not None:
        fcf = op_cf - (capex or 0)

    # FCF yield precisa de market cap em $ — mkt_cap das shares × preço
    # Sem preço, pular (o preço virá do price_cache no build_dataset)
    # Guardamos o FCF raw para usar no build_dataset onde o preço é conhecido
    result["_fcf_raw"] = fcf

    # Simple quality score
    score = 0.0
    if result["gross_margin"] is not None and result["gross_margin"] > 0.35:
        score += 0.25
    if result["de_ratio"] is not None and result["de_ratio"] < 2.0:
        score += 0.25
    if fcf is not None and fcf > 0:
        score += 0.25
    if result["roe"] is not None and result["roe"] > 0.10:
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
        """Lookup by exact match first, then case-insensitive substring.

        yfinance has renamed financial line items multiple times — modern
        versions use long descriptive names like "Net Income From Continuing
        Operation Net Minority Interest" instead of the old "Net Income".
        Substring match handles that without us having to enumerate every
        variant. Order matters: exact match wins (more specific).
        """
        # Pass 1: exact match (fast path, also avoids ambiguity)
        for k in keys:
            v = d.get(k)
            if v is not None:
                f = _sf(v)
                if f is not None:
                    return f
        # Pass 2: case-insensitive substring match (handles renamed cols)
        d_lower = {str(k).lower(): k for k in d.keys()}
        for k in keys:
            k_low = k.lower()
            # Look for keys containing all words of the search term in order
            target_words = k_low.split()
            for key_lower, original_key in d_lower.items():
                # Match if all target words appear in the key, in any positions
                if all(w in key_lower for w in target_words):
                    v = d.get(original_key)
                    if v is not None:
                        f = _sf(v)
                        if f is not None:
                            return f
                    break  # found a partial match, don't try other variants
        return None

    revenue      = _get(fin, "Total Revenue", "Revenue")
    gross_profit = _get(fin, "Gross Profit")
    op_cf        = _get(cf,  "Total Cash From Operations", "Operating Cash Flow", "Operating Cash Flow")
    capex        = _get(cf,  "Capital Expenditures", "Capital Expenditure")
    total_debt   = _get(bs,  "Long Term Debt", "Long-term Debt", "Total Debt")
    equity       = _get(bs,  "Total Stockholder Equity", "Stockholders Equity",
                              "Common Stock Equity")

    result: dict = {}

    if revenue and revenue > 0 and gross_profit is not None:
        result["gross_margin"] = round(gross_profit / revenue, 4)

    if total_debt is not None and equity and equity > 0:
        result["de_ratio"] = round(total_debt / equity, 4)

    fcf = None
    if op_cf is not None:
        fcf = op_cf + (capex or 0)  # capex já é negativo no yfinance
    result["_fcf_raw"] = fcf

    # ── PIT trailing PE + earnings growth (needed for peg_ratio) ──
    # Sum the 4 quarters ending at-or-before this PIT cutoff to get TTM
    # Net Income, divide by shares outstanding to get TTM EPS, then PE.
    # Earnings growth = (TTM_NI_now / TTM_NI_year_ago) - 1.
    # Best-effort: silently skip if any required field is missing.
    try:
        # Sort all periods by date so we can walk window slices.
        all_periods = []
        for period_str in raw.get("fin", {}):
            try:
                period_end = date.fromisoformat(period_str[:10])
                all_periods.append((period_end, period_str))
            except ValueError:
                continue
        all_periods.sort()  # ascending date

        # Identify the most-recent period whose assumed filing <= alert_date.
        cutoff_idx = None
        for i, (period_end, _) in enumerate(all_periods):
            if period_end + timedelta(days=45) <= alert_date:
                cutoff_idx = i
        if cutoff_idx is not None and cutoff_idx >= 3:
            # TTM = sum of net income over the 4 periods ending at cutoff_idx.
            ttm_window = all_periods[cutoff_idx - 3:cutoff_idx + 1]
            ttm_ni = 0.0
            ok = True
            for _, key in ttm_window:
                ni_q = _get(
                    raw["fin"][key],
                    # Most specific first (current yfinance schema):
                    "Net Income From Continuing Operation Net Minority Interest",
                    "Net Income Common Stockholders",
                    "Net Income From Continuing Operations",
                    "Net Income",
                )
                if ni_q is None:
                    ok = False
                    break
                ttm_ni += ni_q
            if ok and cutoff_idx >= 7:
                # Prior-year TTM: the 4 periods ending one year before cutoff_idx.
                py_window = all_periods[cutoff_idx - 7:cutoff_idx - 3]
                py_ni = 0.0
                py_ok = True
                for _, key in py_window:
                    ni_q = _get(
                    raw["fin"][key],
                    # Most specific first (current yfinance schema):
                    "Net Income From Continuing Operation Net Minority Interest",
                    "Net Income Common Stockholders",
                    "Net Income From Continuing Operations",
                    "Net Income",
                )
                    if ni_q is None:
                        py_ok = False
                        break
                    py_ni += ni_q
                if py_ok and py_ni != 0:
                    result["earnings_growth"] = round((ttm_ni - py_ni) / abs(py_ni), 4)

            # trailing_pe = price / EPS_ttm; needs entry price (from caller) AND
            # shares outstanding. We pass eps_ttm separately so the caller can
            # finish the PE calc with the row's actual entry price.
            shares = _get(
                bs,
                # Most specific first (current yfinance schema variants):
                "Common Stock Shares Outstanding",
                "Ordinary Shares Number",
                "Share Issued",
                "Shares Outstanding",
                "Common Stock",  # fallback (substring match will hit any "Common Stock*")
            )
            if ok and shares and shares > 0 and ttm_ni > 0:
                result["_eps_ttm"] = ttm_ni / shares
    except Exception as _e:
        log.debug(f"[yf_q] {ticker}: PE/growth calc failed: {_e}")

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


# ── PIT analyst rating + earnings beat rate ──────────────────────────────────
# yfinance returns the FULL historical earnings/recommendations table for a
# ticker, not just a current snapshot. So we can implement PIT-correct lookups:
# slice the table by alert_date and take only entries that were already public
# at that moment. Cached per-ticker on disk to avoid 1500 fresh yfinance calls
# on every rebuild.

_PIT_EARNINGS_TTL_SECONDS = 7 * 24 * 3600


def _yf_pit_earnings_signals(
    ticker: str,
    alert_date: date,
    cache_dir: Path,
) -> dict:
    """Return PIT analyst_rating + earnings_beat_rate for (ticker, alert_date).

    earnings_beat_rate:
        fraction of the last 4 earnings releases strictly before alert_date
        that beat consensus EPS (surprise %% > 0). Range [0, 1]; NaN if fewer
        than 2 prior releases are available.

    analyst_rating:
        most recent analyst recommendation mean (1=Strong Buy … 5=Strong Sell)
        with grade date <= alert_date. NaN if none available.

    Both rely on yfinance's full-history tables for the ticker. Cached on disk
    per ticker; ~14-day TTL covers the live-scan re-use without re-fetching.
    Returns {} if neither signal is computable.
    """
    import json as _json
    import time as _time

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"yf_pit_earnings_{ticker.replace('.', '_').replace('^', '')}.json"

    raw: dict | None = None
    if cache_path.exists() and (_time.time() - cache_path.stat().st_mtime) < _PIT_EARNINGS_TTL_SECONDS:
        try:
            raw = _json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            raw = None

    if raw is None:
        try:
            import yfinance as yf
            tk = yf.Ticker(ticker)

            edates_payload: list[dict] = []
            try:
                edf = tk.earnings_dates
                if edf is not None and not edf.empty:
                    # Each row: index is the earnings date (with TZ); columns
                    # commonly: 'EPS Estimate', 'Reported EPS', 'Surprise(%)'
                    for ts, row in edf.iterrows():
                        try:
                            d = ts.date().isoformat() if hasattr(ts, "date") else str(ts)[:10]
                        except Exception:
                            d = str(ts)[:10]
                        edates_payload.append({
                            "date": d,
                            "estimate": _sf(row.get("EPS Estimate")),
                            "reported": _sf(row.get("Reported EPS")),
                            "surprise_pct": _sf(row.get("Surprise(%)")),
                        })
            except Exception as _e:
                log.debug(f"[yf_pit] {ticker}: earnings_dates failed: {_e}")

            recs_payload: list[dict] = []
            try:
                rdf = tk.recommendations
                if rdf is not None and not rdf.empty:
                    # Historical recommendations table; aim to get a "rating"
                    # mean per row. Some yfinance versions return a granular
                    # firm-by-firm log (Date/Firm/To Grade/From Grade/Action),
                    # others return a monthly count by bucket (period column +
                    # strongBuy/buy/hold/sell/strongSell columns).
                    cols = {c.lower(): c for c in rdf.columns}
                    if {"strongbuy", "buy", "hold", "sell", "strongsell"} <= set(cols):
                        # Monthly bucket form — period is RELATIVE: "0m"=current,
                        # "1m"=1 month ago, etc. (NOT an ISO date). yfinance only
                        # returns last 4 months. Convert to absolute date so the
                        # PIT filter below can compare against alert_date.
                        from datetime import date as _d
                        from datetime import timedelta as _td
                        _today = _d.today()
                        for _, row in rdf.iterrows():
                            try:
                                # period: "0m", "1m", "2m", "3m"
                                period_str = str(row.get(cols.get("period", ""), "0m")).strip()
                                if period_str.endswith("m") and period_str[:-1].lstrip("-").isdigit():
                                    months_ago = int(period_str[:-1])
                                    # ~30 days per month approximation
                                    abs_date = _today - _td(days=months_ago * 30)
                                    d = abs_date.isoformat()
                                else:
                                    d = str(period_str)[:10]
                                sb = float(row[cols["strongbuy"]] or 0)
                                b  = float(row[cols["buy"]] or 0)
                                h  = float(row[cols["hold"]] or 0)
                                s  = float(row[cols["sell"]] or 0)
                                ss = float(row[cols["strongsell"]] or 0)
                                total = sb + b + h + s + ss
                                if total > 0:
                                    mean = (1*sb + 2*b + 3*h + 4*s + 5*ss) / total
                                    recs_payload.append({"date": d, "rating_mean": round(mean, 3)})
                            except Exception:
                                continue
                    else:
                        # Granular form — collapse each Date to nearest 1-5 score
                        _GRADE_MAP = {
                            "STRONG BUY": 1.0, "BUY": 2.0, "OUTPERFORM": 2.0,
                            "OVERWEIGHT": 2.0, "ACCUMULATE": 2.0,
                            "HOLD": 3.0, "NEUTRAL": 3.0, "MARKET PERFORM": 3.0,
                            "EQUAL-WEIGHT": 3.0, "EQUAL WEIGHT": 3.0,
                            "UNDERPERFORM": 4.0, "UNDERWEIGHT": 4.0, "SELL": 4.0,
                            "STRONG SELL": 5.0,
                        }
                        for ts, row in rdf.iterrows():
                            try:
                                d = ts.date().isoformat() if hasattr(ts, "date") else str(ts)[:10]
                            except Exception:
                                d = str(ts)[:10]
                            grade_col = cols.get("to grade") or cols.get("tograde")
                            if not grade_col:
                                continue
                            g = str(row.get(grade_col, "")).strip().upper()
                            score = _GRADE_MAP.get(g)
                            if score is not None:
                                recs_payload.append({"date": d, "rating_mean": score})
            except Exception as _e:
                log.debug(f"[yf_pit] {ticker}: recommendations failed: {_e}")

            raw = {"earnings_dates": edates_payload, "recommendations": recs_payload}
            try:
                cache_path.write_text(_json.dumps(raw), encoding="utf-8")
            except Exception:
                pass
            time.sleep(0.3)  # gentle rate-limit
        except Exception as e:
            log.debug(f"[yf_pit] {ticker}: top-level fetch failed: {e}")
            raw = {"earnings_dates": [], "recommendations": []}

    result: dict = {}

    # ── earnings_beat_rate ──
    edates = raw.get("earnings_dates") or []
    prior = [e for e in edates if e.get("date") and date.fromisoformat(e["date"][:10]) < alert_date]
    prior.sort(key=lambda e: e["date"], reverse=True)
    last4 = prior[:4]
    valid_surprises = [e["surprise_pct"] for e in last4 if e.get("surprise_pct") is not None]
    if len(valid_surprises) >= 2:
        beats = sum(1 for s in valid_surprises if s > 0)
        result["earnings_beat_rate"] = round(beats / len(valid_surprises), 4)

    # ── analyst_rating ──
    recs = raw.get("recommendations") or []
    prior_recs = [r for r in recs if r.get("date") and r.get("rating_mean") is not None
                  and date.fromisoformat(r["date"][:10]) <= alert_date]
    if prior_recs:
        prior_recs.sort(key=lambda r: r["date"], reverse=True)
        result["analyst_rating"] = round(float(prior_recs[0]["rating_mean"]), 3)

    return result


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

    # MERGE strategy (not "first wins"): each tier has different strengths.
    # EDGAR is PIT-correct for US tickers and has gross_margin/de_ratio/roe.
    # yfinance fills gaps for _eps_ttm + earnings_growth (needs 8 quarters).
    # Tiingo (if user has access) has the deepest history.
    #
    # Priority order on key conflicts (most-correct → least):
    #   Tiingo > EDGAR > yfinance quarterly > yfinance.info
    # When tier N has a key, tier N+1's value for that key is IGNORED.
    merged: dict = {}

    def _merge_in(source: dict):
        """Add keys from source where not already present in merged."""
        if not source:
            return
        for k, v in source.items():
            if v is None:
                continue
            if k not in merged or merged[k] is None:
                merged[k] = v

    # Tier 1: Tiingo (highest priority)
    if has_tiingo_access():
        try:
            _merge_in(_tiingo_get(ticker, alert_date))
        except Exception as e:
            log.debug(f"[fund_hist] Tiingo falhou {ticker}: {e}")

    # Tier 2: SEC EDGAR (US tickers, gross_margin/de_ratio/roe/quality_score)
    if _is_us_ticker(ticker):
        try:
            _merge_in(_edgar_get(ticker, alert_date, cache_dir))
        except Exception as e:
            log.debug(f"[fund_hist] EDGAR falhou {ticker}: {e}")

    # Tier 3: yfinance quarterly (fills _eps_ttm + earnings_growth gaps,
    # PIT-correct from quarterly filings)
    try:
        _merge_in(_yf_quarterly(ticker, alert_date, cache_dir))
    except Exception as e:
        log.debug(f"[fund_hist] yfinance quarterly falhou {ticker}: {e}")

    # NO yfinance.info fallback — it returns CURRENT state, not PIT. Applying
    # today's revenueGrowth to a 2020 alert_date would leak forward-looking
    # information into training. NaN is the honest value when no PIT source
    # has the data. Tree models handle NaN natively; the linear model uses
    # column mean of NON-NULL rows (no zero inflation). The bot's live
    # /analisar flow uses a separate code path with yfinance.info because
    # there "current = PIT" for today's alert.

    return merged


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
