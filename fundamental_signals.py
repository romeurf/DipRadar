"""
fundamental_signals.py — Sinais de qualidade e sentimento para o modelo ML.

Filosofia: sem fallbacks silenciosos. Se um sinal não puder ser calculado, retorna
NaN (explicitamente "desconhecido") e loga o erro. O chamador decide o que fazer com
NaN — nunca inventamos um valor neutro que mascara o problema.

Camadas de dados (gratuitas):

  Camada 1 — Price history (já disponível):
    consecutive_red_days, ma_200d_ratio

  Camada 2 — yfinance info (sem API key):
    earnings_beat_rate, analyst_rating, short_interest_pct

  Camada 3 — SEC EDGAR Form 4 + 8-K + earnings calls (gratuito, sem key):
    insider_buy_recent, insider_sell_recent, classify_recent_8k,
    earnings_call_tone, insider_buy/sell_amount_score

Removidas em commit posterior (não eram consumidas pelos modelos):
  Camada 4 — Alpha Vantage earnings revision
  Camada 5 — FMP analyst revision trend

Valores de retorno:
  float        → calculado com sucesso
  float("nan") → não disponível (API em baixo, ticker sem cobertura, etc.)
                 NUNCA retornamos um valor "neutro" inventado.
"""

from __future__ import annotations

import logging
import math
import os
from datetime import date, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

NaN = float("nan")


# ── Camada 1: Price history ────────────────────────────────────────────────────

def consecutive_red_days(price_history: pd.DataFrame) -> float:
    """Conta dias consecutivos de queda antes da última barra.

    Raises se price_history é None/vazio — o chamador deve garantir dados válidos.
    """
    if price_history is None or price_history.empty or "Close" not in price_history.columns:
        raise ValueError("consecutive_red_days: price_history inválido ou sem coluna Close")
    closes = price_history["Close"].dropna()
    if len(closes) < 2:
        return NaN
    rets = closes.pct_change().dropna()
    count = 0
    for r in reversed(rets.values):
        if float(r) < 0:
            count += 1
        else:
            break
    return float(min(count, 30))


def ma_200d_ratio(price_history: pd.DataFrame) -> float:
    """Preço actual / MA200. < 1 = abaixo da média de longo prazo.

    Retorna NaN se menos de 20 observações (insufficiente para MA significativa).
    """
    if price_history is None or price_history.empty or "Close" not in price_history.columns:
        raise ValueError("ma_200d_ratio: price_history inválido ou sem coluna Close")
    closes = price_history["Close"].dropna()
    if len(closes) < 20:
        return NaN
    window = min(200, len(closes))
    ma = float(closes.tail(window).mean())
    current = float(closes.iloc[-1])
    if ma <= 0 or not math.isfinite(ma) or not math.isfinite(current):
        return NaN
    ratio = current / ma
    return round(float(np.clip(ratio, 0.3, 2.0)), 4)


# ── Camada 2: yfinance info ────────────────────────────────────────────────────

def yf_fundamental_signals(ticker: str) -> dict[str, float]:
    """Busca sinais fundamentais via yfinance.Ticker.info.

    Retorna dict com NaN nos campos que não puderem ser calculados.
    Propaga ImportError se yfinance não estiver instalado.
    """
    import yfinance as yf

    out: dict[str, float] = {
        "earnings_beat_rate":  NaN,
        "analyst_rating":      NaN,
        "short_interest_pct":  NaN,
    }

    tk = yf.Ticker(ticker)
    info = tk.info or {}

    # Short interest como % do float (yfinance retorna 0-1 ou 0-100 inconsistente)
    sif = info.get("shortPercentOfFloat")
    if sif is not None:
        v = float(sif)
        if math.isfinite(v):
            if 0 <= v <= 1:
                out["short_interest_pct"] = round(v, 4)
            elif 1 < v <= 100:
                out["short_interest_pct"] = round(v / 100.0, 4)
            # valores > 100 são erros de dados — deixamos NaN

    # Consenso analistas: 1=Strong Buy, 5=Strong Sell
    rm = info.get("recommendationMean")
    if rm is not None:
        v = float(rm)
        if math.isfinite(v) and 1.0 <= v <= 5.0:
            out["analyst_rating"] = round(v, 2)

    # Earnings beat rate: últimos 4 trimestres
    try:
        eq = tk.quarterly_earnings
        if eq is not None and not eq.empty:
            if "Earnings" in eq.columns and "Estimate" in eq.columns:
                q4 = eq.tail(4)
                valid = q4[["Earnings", "Estimate"]].dropna()
                if len(valid) >= 2:
                    beats = (valid["Earnings"] > valid["Estimate"]).sum()
                    out["earnings_beat_rate"] = round(float(beats / len(valid)), 2)
    except Exception as e:
        log.error(f"[yf_signals] earnings_beat_rate para {ticker}: {e}")

    return out


# ── Camada 3: SEC EDGAR Form 4 ────────────────────────────────────────────────

_EDGAR_HEADERS = {"User-Agent": "DipRadar research@dipradar.io"}
_CIK_MAP: dict[str, str] = {}
_CIK_LOADED = False


def _ensure_cik_map() -> None:
    global _CIK_MAP, _CIK_LOADED
    if _CIK_LOADED:
        return
    _CIK_LOADED = True  # marcar antes de tentar — evita retry storms
    try:
        import requests
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=_EDGAR_HEADERS, timeout=20,
        )
        r.raise_for_status()
        _CIK_MAP = {
            v["ticker"].upper(): str(v["cik_str"]).zfill(10)
            for v in r.json().values()
            if "ticker" in v and "cik_str" in v
        }
        log.info(f"[insider] CIK map carregado: {len(_CIK_MAP)} tickers SEC")
    except Exception as e:
        log.error(
            f"[insider] Falha a carregar CIK map do SEC EDGAR: {e}. "
            f"insider_buy_recent retornará NaN para esta sessão."
        )
        _CIK_MAP = {}  # vazio — insider_buy_recent retorna NaN para todos


_edgar_submissions_cache: dict[str, dict] = {}  # cache em memória por sessão
# Cap chosen to cover the full ~1500-ticker US universe in a single bulk
# build_dip_dataset run without LRU eviction churn.
_EDGAR_CACHE_MAX = 2000

# Disk cache for EDGAR submissions — survives across subprocess invocations.
# Without this, every `build_dip_dataset.py --force-rebuild` re-downloads all
# ~1500 submissions JSONs (>100 MB of HTTP). With it, the first run populates
# the cache and subsequent runs read straight from disk.
#
# Submissions index entries are append-only (immutable historical filings),
# so the only reason to refresh is to pick up new filings. A 7-day TTL is
# generous for live use and harmless for historical PIT lookups.
from pathlib import Path as _DiskPath
_EDGAR_DISK_CACHE_DIR = (
    _DiskPath("/data/edgar_submissions") if _DiskPath("/data").exists()
    else _DiskPath("/tmp/edgar_submissions")
)
_EDGAR_DISK_TTL_SECONDS = 7 * 24 * 3600   # 7 days


def _edgar_submissions(ticker: str) -> dict:
    """Return the EDGAR submissions JSON for a US ticker. Raises on failure.

    Three-tier cache: process-memory → disk (7-day TTL) → SEC HTTP. The disk
    tier matters most because each `build_dip_dataset.py` invocation is a
    fresh subprocess with an empty memory cache.
    """
    import requests
    import json as _json
    import os as _os
    import time as _time

    base = ticker.upper().split(".")[0]
    if "." in ticker:
        raise ValueError(f"{ticker} is not a US ticker (no EDGAR coverage)")

    # Tier 1: in-memory cache (fastest)
    if base in _edgar_submissions_cache:
        return _edgar_submissions_cache[base]

    # Tier 2: disk cache
    try:
        _EDGAR_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        disk_path = _EDGAR_DISK_CACHE_DIR / f"{base}.json"
        if disk_path.exists():
            age = _time.time() - disk_path.stat().st_mtime
            if age < _EDGAR_DISK_TTL_SECONDS:
                try:
                    data = _json.loads(disk_path.read_text(encoding="utf-8"))
                    _edgar_submissions_cache[base] = data
                    return data
                except Exception:
                    # Corrupt file — fall through and refetch
                    pass
    except Exception:
        # Disk unavailable; fall through to HTTP
        pass

    # Tier 3: HTTP fetch from SEC
    _ensure_cik_map()
    cik = _CIK_MAP.get(base)
    if not cik:
        raise ValueError(f"{base} not in EDGAR CIK map")
    r = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=_EDGAR_HEADERS, timeout=15,
    )
    if r.status_code == 404:
        raise ValueError(f"{base} not covered by EDGAR (404)")
    r.raise_for_status()
    data = r.json()

    # Promote to disk cache (best-effort)
    try:
        _EDGAR_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        disk_path = _EDGAR_DISK_CACHE_DIR / f"{base}.json"
        tmp = disk_path.with_suffix(".json.tmp")
        tmp.write_text(_json.dumps(data), encoding="utf-8")
        _os.replace(tmp, disk_path)
    except Exception:
        pass

    # Simple LRU on the memory cache
    if len(_edgar_submissions_cache) >= _EDGAR_CACHE_MAX:
        for k in list(_edgar_submissions_cache.keys())[:50]:
            del _edgar_submissions_cache[k]
    _edgar_submissions_cache[base] = data
    return data


def _filing_primary_url(cik_int: int, acc_clean: str, primary_doc: str = "") -> str:
    """Constrói URL do documento principal de uma filing EDGAR.

    Usa primaryDocument do submissions JSON se disponível.
    Fallback: tenta {acc_clean}.txt (antigo) e depois o índice JSON.
    """
    base = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}"
    if primary_doc:
        return f"{base}/{primary_doc}"
    return f"{base}/{acc_clean}.txt"


def _parse_form4_xml(accession_no: str, cik: str, primary_doc: str = "") -> dict:
    """Parse a Form 4 XML and return both purchase AND sale info.

    Returns a dict with:
      - amount_usd       : total purchase value (shares × price)         [legacy alias for buy]
      - is_purchase      : True if any acquisition (code A)              [legacy alias]
      - sale_amount_usd  : total sale value (shares × price)             [NEW]
      - is_sale          : True if any disposition (code D)              [NEW]
      - name             : insider name
      - title            : role (e.g. "Chief Executive Officer", "Director")

    Both sides are returned from the same filing so callers can score buys
    and sells independently — a CEO disposing of shares carries a very
    different signal from a CEO buying them, and many filings have both.
    """
    import requests
    import xml.etree.ElementTree as ET

    acc_clean = accession_no.replace("-", "")
    cik_int   = int(cik)
    doc_url   = _filing_primary_url(cik_int, acc_clean, primary_doc)

    try:
        r = requests.get(doc_url, headers=_EDGAR_HEADERS, timeout=10)
        if r.status_code != 200:
            return {}
        text  = r.text
        start = text.find("<ownershipDocument>")
        end   = text.find("</ownershipDocument>")
        if start == -1 or end == -1:
            return {}
        root = ET.fromstring(text[start:end + len("</ownershipDocument>")])
    except Exception:
        return {}

    # ── Insider identity ──────────────────────────────────────────────────
    name  = root.findtext(".//reportingOwner/reportingOwnerId/rptOwnerName", "").strip()
    title = root.findtext(
        ".//reportingOwner/reportingOwnerRelationship/officerTitle", ""
    ).strip()
    is_director = root.findtext(
        ".//reportingOwner/reportingOwnerRelationship/isDirector", "0"
    ).strip() == "1"
    if not title and is_director:
        title = "Director"

    # ── Transactions: split by acquired (A=buy) vs disposed (D=sell) ──────
    buy_amount  = 0.0
    sell_amount = 0.0
    has_buy     = False
    has_sell    = False
    for txn in root.findall(".//nonDerivativeTransaction"):
        code = txn.findtext(".//transactionAcquiredDisposedCode/value", "").strip()
        try:
            shares = float(txn.findtext(".//transactionShares/value", "0") or 0)
            price  = float(txn.findtext(".//transactionPricePerShare/value", "0") or 0)
            amt    = shares * price
        except (ValueError, TypeError):
            continue
        if code == "A":
            has_buy = True
            buy_amount += amt
        elif code == "D":
            has_sell = True
            sell_amount += amt

    return {
        # Buy side (legacy key names preserved)
        "amount_usd":      buy_amount,
        "is_purchase":     has_buy,
        # Sell side (new)
        "sale_amount_usd": sell_amount,
        "is_sale":         has_sell,
        # Identity
        "name":            name,
        "title":           title,
    }


def insider_buy_recent(ticker: str, lookback_days: int = 30, as_of_date: "date | None" = None) -> float:
    """1.0 se houve compra de insider (Form 4) nos últimos lookback_days, 0.0 se não.

    as_of_date: se fornecido, usa como referência temporal (PIT training).
                None = hoje (produção live).
    Retorna NaN se o ticker não tiver cobertura SEC (tickers não-US).
    """
    try:
        data   = _edgar_submissions(ticker)
        recent = data.get("filings", {}).get("recent", {})
        forms       = recent.get("form", [])
        dates_filed = recent.get("filingDate", [])
        as_of  = as_of_date or date.today()
        cutoff = as_of - timedelta(days=lookback_days)
        for form, filed_str in zip(forms, dates_filed):
            if form not in ("4", "4/A"):
                continue
            try:
                filed = date.fromisoformat(filed_str[:10])
                if cutoff <= filed <= as_of:
                    return 1.0
            except ValueError:
                continue
        return 0.0
    except Exception:
        return NaN


def insider_buy_amount_score(ticker: str, lookback_days: int = 60, as_of_date: "date | None" = None) -> tuple[float, float]:
    """Score normalizado + montante raw das compras de insiders.

    Devolve (score [0,1], amount_usd) para que a narrativa possa mostrar
    o valor real em vez de "muito significativa".

    score:
      0.0 = sem compras
      0.3 = compras pequenas (<$100k)
      0.7 = compras significativas ($100k-$1M)
      1.0 = compras muito grandes (>$1M)

    Usa Form 4 + parsing XML do SEC EDGAR (gratuito).
    """
    try:
        data   = _edgar_submissions(ticker)
        cik    = data.get("cik", "")
        if not cik:
            return NaN, 0.0
        cik_str = str(cik).zfill(10)
        recent  = data.get("filings", {}).get("recent", {})
        forms           = recent.get("form", [])
        dates_filed     = recent.get("filingDate", [])
        accessions      = recent.get("accessionNumber", [])
        primary_docs    = recent.get("primaryDocument", [""] * len(forms))  # nome real do ficheiro
        as_of  = as_of_date or date.today()
        cutoff = as_of - timedelta(days=lookback_days)

        total_purchase = 0.0
        top_name  = ""
        top_title = ""
        top_amount = 0.0

        for form, filed_str, acc, pdoc in zip(forms, dates_filed, accessions, primary_docs):
            if form not in ("4", "4/A"):
                continue
            try:
                filed = date.fromisoformat(filed_str[:10])
                if not (cutoff <= filed <= as_of):
                    continue
            except ValueError:
                continue
            parsed = _parse_form4_xml(acc, cik_str, primary_doc=str(pdoc or ""))
            if parsed.get("is_purchase"):
                amt = parsed.get("amount_usd", 0)
                total_purchase += amt
                # Guardar o insider com maior compra individual
                if amt > top_amount:
                    top_amount = amt
                    top_name   = parsed.get("name", "")
                    top_title  = parsed.get("title", "")

        if total_purchase <= 0:
            return 0.0, 0.0

        import math
        score = math.log10(max(total_purchase, 1000)) / math.log10(10_000_000)
        # Guardar nome e cargo na variável de módulo para uso na narrativa
        _last_insider_details[ticker.upper()] = {
            "name":   top_name,
            "title":  top_title,
            "amount": round(total_purchase, 0),
        }
        return round(float(min(score, 1.0)), 3), round(total_purchase, 0)
    except Exception as e:
        log.debug(f"[insider_amount] {ticker}: {e}")
        return NaN, 0.0


def insider_sell_recent(ticker: str, lookback_days: int = 30, as_of_date: "date | None" = None) -> float:
    """1.0 if any insider sale (Form 4 code D) in the last `lookback_days`, else 0.0.

    NaN if the ticker has no EDGAR coverage (non-US). 0.0 means EDGAR confirmed
    no sales in the window — semantically very different from NaN.
    """
    try:
        data   = _edgar_submissions(ticker)
        cik    = data.get("cik", "")
        if not cik:
            return NaN
        cik_str = str(cik).zfill(10)
        recent  = data.get("filings", {}).get("recent", {})
        forms        = recent.get("form", [])
        dates_filed  = recent.get("filingDate", [])
        accessions   = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [""] * len(forms))
        as_of  = as_of_date or date.today()
        cutoff = as_of - timedelta(days=lookback_days)
        for form, filed_str, acc, pdoc in zip(forms, dates_filed, accessions, primary_docs):
            if form not in ("4", "4/A"):
                continue
            try:
                filed = date.fromisoformat(filed_str[:10])
                if not (cutoff <= filed <= as_of):
                    continue
            except ValueError:
                continue
            parsed = _parse_form4_xml(acc, cik_str, primary_doc=str(pdoc or ""))
            if parsed.get("is_sale"):
                return 1.0
        return 0.0
    except Exception:
        return NaN


def insider_sell_amount_score(ticker: str, lookback_days: int = 60, as_of_date: "date | None" = None) -> tuple[float, float]:
    """Normalised sale-magnitude score + raw USD amount.

    Mirrors insider_buy_amount_score: same scoring scale
      0.0 = no sales
      0.3 = small sales (<$100k)
      0.7 = significant sales ($100k-$1M)
      1.0 = very large sales (>$1M)

    Returns (NaN, 0.0) if the ticker has no EDGAR coverage.
    Returns (0.0, 0.0) if EDGAR confirmed no sales — semantically different.
    """
    try:
        data   = _edgar_submissions(ticker)
        cik    = data.get("cik", "")
        if not cik:
            return NaN, 0.0
        cik_str = str(cik).zfill(10)
        recent  = data.get("filings", {}).get("recent", {})
        forms        = recent.get("form", [])
        dates_filed  = recent.get("filingDate", [])
        accessions   = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [""] * len(forms))
        as_of  = as_of_date or date.today()
        cutoff = as_of - timedelta(days=lookback_days)

        total_sale = 0.0
        top_name   = ""
        top_title  = ""
        top_amount = 0.0

        for form, filed_str, acc, pdoc in zip(forms, dates_filed, accessions, primary_docs):
            if form not in ("4", "4/A"):
                continue
            try:
                filed = date.fromisoformat(filed_str[:10])
                if not (cutoff <= filed <= as_of):
                    continue
            except ValueError:
                continue
            parsed = _parse_form4_xml(acc, cik_str, primary_doc=str(pdoc or ""))
            if parsed.get("is_sale"):
                amt = parsed.get("sale_amount_usd", 0)
                total_sale += amt
                if amt > top_amount:
                    top_amount = amt
                    top_name   = parsed.get("name", "")
                    top_title  = parsed.get("title", "")

        if total_sale <= 0:
            return 0.0, 0.0

        import math
        score = math.log10(max(total_sale, 1000)) / math.log10(10_000_000)
        # Store details for narrative use, in a separate cache so buy details aren't clobbered
        _last_insider_sell_details[ticker.upper()] = {
            "name":   top_name,
            "title":  top_title,
            "amount": round(total_sale, 0),
        }
        return round(float(min(score, 1.0)), 3), round(total_sale, 0)
    except Exception as e:
        log.debug(f"[insider_sell_amount] {ticker}: {e}")
        return NaN, 0.0


# Cache em memória dos detalhes do último insider e 8-K por ticker
_last_insider_details: dict[str, dict] = {}
_last_insider_sell_details: dict[str, dict] = {}
_last_8k_details: dict[str, dict] = {}


def get_insider_details(ticker: str) -> dict:
    """Devolve {name, title, amount} do insider com maior compra recente."""
    return _last_insider_details.get(ticker.upper(), {})


def get_insider_sell_details(ticker: str) -> dict:
    """Return {name, title, amount} of the largest recent insider SALE."""
    return _last_insider_sell_details.get(ticker.upper(), {})


def get_8k_details(ticker: str) -> dict:
    """Devolve {item, description, filed, score} do 8-K mais relevante recente."""
    return _last_8k_details.get(ticker.upper(), {})


# ── 8-K Classification ────────────────────────────────────────────────────────

# Mapeamento de item codes SEC → impacto para dip hunting
# Items completos: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=8-K
# Item codes 8-K: (score, descrição precisa para narrativa)
_8K_ITEMS: dict[str, tuple[float, str]] = {
    "1.01": (+0.2, "Material Definitive Agreement — contrato estratégico importante"),
    "1.02": (-0.2, "Termination of Material Agreement — rescisão de contrato relevante"),
    "1.05": (-0.5, "Material Cybersecurity Incident — incidente de cibersegurança grave"),
    "2.01": (+0.4, "Completion of Acquisition or Disposition — aquisição ou alienação completada"),
    "2.02": ( 0.0, "Results of Operations and Financial Condition — divulgação de resultados"),
    "2.04": (-0.8, "Triggering Events for Direct Financial Obligation — risco de default ou aceleração de dívida"),
    "2.06": (-0.3, "Material Impairment — imparidade de activos"),
    "3.01": (-0.6, "Notice of Delisting — risco de saída da bolsa"),
    "4.01": (-0.5, "Changes in Registrant's Certifying Accountant — mudança de auditor (sinal de problemas)"),
    "4.02": (-0.9, "Non-Reliance on Previously Issued Financial Statements — restatement de resultados anteriores"),
    "5.01": (+0.1, "Changes in Control of Registrant — mudança de controlo"),
    "5.02": (-0.3, "Departure of Directors or Certain Officers — saída de executivo ou director"),
    "5.03": (-0.1, "Amendments to AoI or Bylaws — alteração de estatutos"),
    "7.01": ( 0.0, "Regulation FD Disclosure — divulgação ao abrigo do Regulamento FD"),
    "8.01": ( 0.0, "Other Events — outros eventos"),
    "9.01": ( 0.0, "Financial Statements and Exhibits — demonstrações financeiras e anexos"),
}

# Manter o dict de scores para compatibilidade
_8K_ITEM_SCORES: dict[str, float] = {k: v[0] for k, v in _8K_ITEMS.items()}


def classify_recent_8k(ticker: str, lookback_days: int = 30, as_of_date: "date | None" = None) -> float:
    """Score de risco do 8-K mais recente: -1 (muito mau) a +1 (muito bom).

    Usa os item codes do 8-K para classificar sem parsear o texto completo:
      -0.9 → Restatement (4.02) — quase sempre estrutural, evitar
      -0.8 → Default/covenant breach (2.04) — muito negativo
      +0.4 → Aquisição completada (2.01) — frequentemente positivo
      0.0  → Earnings/outros (contexto necessário)

    Retorna NaN se sem cobertura EDGAR ou sem 8-K recente.
    """
    try:
        data   = _edgar_submissions(ticker)
        recent = data.get("filings", {}).get("recent", {})
        forms        = recent.get("form", [])
        dates_filed  = recent.get("filingDate", [])
        items_list   = recent.get("items", [])
        as_of  = as_of_date or date.today()
        cutoff = as_of - timedelta(days=lookback_days)

        # Encontrar o 8-K mais recente dentro do período
        for form, filed_str, items in zip(forms, dates_filed, items_list):
            if form not in ("8-K", "8-K/A"):
                continue
            try:
                filed = date.fromisoformat(filed_str[:10])
                if not (cutoff <= filed <= as_of):
                    continue
            except ValueError:
                continue

            # items é uma string como "2.02, 9.01" ou lista
            if isinstance(items, list):
                item_codes = [str(x).strip() for x in items]
            else:
                item_codes = [x.strip() for x in str(items).split(",")]

            scores = [_8K_ITEM_SCORES.get(code, 0.0) for code in item_codes if code]
            if not scores:
                return 0.0

            # O item mais extremo domina
            min_score = min(scores)
            max_score = max(scores)
            final = min_score if abs(min_score) > abs(max_score) else max_score
            dominant_code = (
                min(item_codes, key=lambda c: _8K_ITEM_SCORES.get(c, 0.0))
                if abs(min_score) >= abs(max_score)
                else max(item_codes, key=lambda c: _8K_ITEM_SCORES.get(c, 0.0))
            )
            # Guardar descrição exacta para a narrativa
            _last_8k_details[ticker.upper()] = {
                "item":        dominant_code,
                "description": _8K_ITEMS.get(dominant_code, (0, "Evento 8-K"))[1],
                "filed":       filed_str[:10],
                "score":       round(float(max(-1.0, min(1.0, final))), 2),
            }
            return round(float(max(-1.0, min(1.0, final))), 2)

        return NaN  # sem 8-K recente
    except Exception as e:
        log.debug(f"[8k_class] {ticker}: {e}")
        return NaN


# ── Short Interest Trend ──────────────────────────────────────────────────────

def short_interest_trend(ticker: str) -> float:
    """Variação do short interest no último mês (Δ fracção).

    Positivo = mais shorts este mês vs mês anterior.
      > +0.20 : shorts a aumentar muito — mais potencial de squeeze se recuperar
      ≈ 0     : estável
      < -0.20 : shorts a cobrir — smart money a fechar posições curtas

    Para dip hunting, short_interest_trend positivo + dip = setup de squeeze potencial.
    Usa yfinance info (sharesShort + sharesShortPriorMonth). Gratuito, sem API key.
    """
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        current = info.get("sharesShort")
        prior   = info.get("sharesShortPriorMonth")
        if current is None or prior is None or prior <= 0:
            return NaN
        trend = (float(current) - float(prior)) / float(prior)
        return round(float(max(-1.0, min(1.0, trend))), 4)
    except Exception as e:
        log.debug(f"[short_trend] {ticker}: {e}")
        return NaN


def earnings_call_tone(ticker: str, lookback_days: int = 90, as_of_date: "date | None" = None) -> float:
    """Score de sentimento de earnings calls [-1, +1] por análise de keywords.

    Usa transcrições 8-K de earnings (Item 2.02) como proxy quando disponível.
    Analisa palavras-chave positivas vs negativas no texto do filing.

    +1 = management confiante ("strong", "accelerating", "raised guidance")
     0 = neutro ou sem dados
    -1 = management pessimista ("challenging", "headwinds", "below expectations")

    Completamente gratuito via SEC EDGAR — sem API key adicional.
    """
    _POSITIVE = {
        "strong", "accelerating", "raised", "outperformed", "beat",
        "record", "growth", "confident", "robust", "exceeded",
        "improved", "momentum", "ahead", "positive", "strength",
    }
    _NEGATIVE = {
        "challenging", "headwinds", "uncertain", "cautious", "difficult",
        "below", "missed", "declined", "weak", "softness",
        "pressure", "slowdown", "reduced", "revised down", "disappointing",
    }
    try:
        data   = _edgar_submissions(ticker)
        cik    = str(data.get("cik", "")).zfill(10)
        recent = data.get("filings", {}).get("recent", {})
        forms           = recent.get("form", [])
        dates_filed     = recent.get("filingDate", [])
        accessions      = recent.get("accessionNumber", [])
        items_list      = recent.get("items", [])
        primary_docs    = recent.get("primaryDocument", [""] * len(forms))
        as_of  = as_of_date or date.today()
        cutoff = as_of - timedelta(days=lookback_days)

        import requests
        for form, filed_str, acc, items, pdoc in zip(forms, dates_filed, accessions, items_list, primary_docs):
            if form not in ("8-K", "8-K/A"):
                continue
            try:
                filed = date.fromisoformat(filed_str[:10])
                if not (cutoff <= filed <= as_of):
                    continue
            except ValueError:
                continue
            # Procurar earnings release (item 2.02)
            items_str = ",".join(items) if isinstance(items, list) else str(items)
            if "2.02" not in items_str:
                continue

            # Fetch documento usando primaryDocument (nome real, não .txt genérico)
            acc_clean = acc.replace("-", "")
            url = _filing_primary_url(int(cik), acc_clean, str(pdoc or ""))
            r = requests.get(url, headers=_EDGAR_HEADERS, timeout=15)
            if r.status_code != 200:
                continue
            text = r.text.lower()[:50_000]  # primeiros 50k chars são suficientes

            # Contar keywords (normalizado pelo tamanho do texto)
            words     = set(text.split())
            pos_count = sum(1 for kw in _POSITIVE if kw in text)
            neg_count = sum(1 for kw in _NEGATIVE if kw in text)
            total     = pos_count + neg_count
            if total == 0:
                return 0.0
            score = (pos_count - neg_count) / total
            return round(float(max(-1.0, min(1.0, score))), 3)

        return NaN  # sem earnings call recente
    except Exception as e:
        log.debug(f"[earnings_tone] {ticker}: {e}")
        return NaN


# Camadas 4 (Alpha Vantage) e 5 (FMP) foram removidas — os sinais que
# produziam (earnings_revision_av, analyst_revision_fmp) nunca foram
# consumidos por FEATURE_COLUMNS nem por nenhum scorer/modelo. Eram dead
# code que consumia quota de APIs gratuitas para gerar valores que iam
# directos para o vazio. analyst_rating + earnings_beat_rate (yfinance,
# Camada 2) cobrem o mesmo terreno conceptual e ESSES sim entram em
# features.


# ── Orquestrador ──────────────────────────────────────────────────────────────

def compute_fundamental_signals(
    ticker: str,
    price_history: Optional[pd.DataFrame] = None,
    alert_date: Optional[Any] = None,
    use_alphavantage: bool = False,   # DEPRECATED — ignored, kept for arg-compat
    use_fmp: bool = False,             # DEPRECATED — ignored, kept for arg-compat
) -> dict[str, float]:
    """Computa todos os sinais de qualidade e sentimento.

    Cada sinal que falha é registado com ERROR e retorna NaN.
    NaN significa "desconhecido" — nunca um valor neutro inventado.
    O chamador (build_features) decide como lidar com NaN.
    """
    result: dict[str, float] = {}

    # Camada 1: price history
    if price_history is not None and not price_history.empty:
        try:
            result["consecutive_red_days"] = consecutive_red_days(price_history)
        except Exception as e:
            log.error(f"[signals] consecutive_red_days {ticker}: {e}")
            result["consecutive_red_days"] = NaN

        try:
            result["ma_200d_ratio"] = ma_200d_ratio(price_history)
        except Exception as e:
            log.error(f"[signals] ma_200d_ratio {ticker}: {e}")
            result["ma_200d_ratio"] = NaN
    else:
        result["consecutive_red_days"] = NaN
        result["ma_200d_ratio"] = NaN

    # Camada 2: yfinance
    try:
        yf_out = yf_fundamental_signals(ticker)
        result.update(yf_out)
    except Exception as e:
        log.error(f"[signals] yf_fundamental_signals {ticker}: {e}")
        result.setdefault("earnings_beat_rate", NaN)
        result.setdefault("analyst_rating", NaN)
        result.setdefault("short_interest_pct", NaN)

    # Camada 3: SEC EDGAR (só para tickers US)
    _is_us = "." not in ticker.split("/")[-1]
    if _is_us:
        try:
            result["insider_buy_recent"] = insider_buy_recent(ticker)
        except Exception as e:
            log.error(f"[signals] insider_buy_recent {ticker}: {e}")
            result["insider_buy_recent"] = NaN

        try:
            _score, _amount = insider_buy_amount_score(ticker)
            result["insider_buy_amount_score"] = _score
            result["insider_buy_amount_usd"]   = _amount  # valor real em dólares
        except Exception as e:
            log.error(f"[signals] insider_buy_amount {ticker}: {e}")
            result["insider_buy_amount_score"] = NaN
            result["insider_buy_amount_usd"]   = 0.0

        try:
            result["insider_sell_recent"] = insider_sell_recent(ticker)
        except Exception as e:
            log.error(f"[signals] insider_sell_recent {ticker}: {e}")
            result["insider_sell_recent"] = NaN

        try:
            _sscore, _samount = insider_sell_amount_score(ticker)
            result["insider_sell_amount_score"] = _sscore
            result["insider_sell_amount_usd"]   = _samount
        except Exception as e:
            log.error(f"[signals] insider_sell_amount {ticker}: {e}")
            result["insider_sell_amount_score"] = NaN
            result["insider_sell_amount_usd"]   = 0.0

        try:
            result["recent_8k_score"] = classify_recent_8k(ticker)
        except Exception as e:
            log.error(f"[signals] 8k_class {ticker}: {e}")
            result["recent_8k_score"] = NaN
    else:
        result["insider_buy_recent"]        = NaN
        result["insider_buy_amount_score"]  = NaN
        result["insider_sell_recent"]       = NaN
        result["insider_sell_amount_score"] = NaN
        result["recent_8k_score"]           = NaN

    # Short interest trend (yfinance, US + alguns internacionais)
    try:
        result["short_interest_trend"] = short_interest_trend(ticker)
    except Exception as e:
        log.error(f"[signals] short_interest_trend {ticker}: {e}")
        result["short_interest_trend"] = NaN

    # Earnings call tone (SEC EDGAR 8-K, só US)
    if _is_us:
        try:
            result["earnings_call_tone"] = earnings_call_tone(ticker)
        except Exception as e:
            log.error(f"[signals] earnings_call_tone {ticker}: {e}")
            result["earnings_call_tone"] = NaN
    else:
        result["earnings_call_tone"] = NaN

    # Camadas 4 (Alpha Vantage) e 5 (FMP) removidas — ver nota acima da função.

    return result
