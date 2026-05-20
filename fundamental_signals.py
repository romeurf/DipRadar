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

  Camada 3 — SEC EDGAR Form 4 (gratuito, sem key):
    insider_buy_recent

  Camada 4 — Alpha Vantage (free tier, 25 req/day):
    env var: ALPHAVANTAGE_API_KEY (https://www.alphavantage.co/support/#api-key)

  Camada 5 — Financial Modeling Prep (free tier, 250 req/day):
    env var: FMP_API_KEY (https://financialmodelingprep.com/developer/docs)

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

_ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
_FMP_KEY          = os.getenv("FMP_API_KEY", "")

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


def _edgar_submissions(ticker: str) -> dict:
    """Retorna o JSON de submissions EDGAR para um ticker US. Raises se falhar.

    Cache em memória por sessão — evita 147k HTTP requests no regen ao carregar
    os dados de cada ticker apenas uma vez independentemente do número de linhas.
    """
    import requests
    base = ticker.upper().split(".")[0]
    if "." in ticker:
        raise ValueError(f"{ticker} não é um ticker US (sem cobertura EDGAR)")

    if base in _edgar_submissions_cache:
        return _edgar_submissions_cache[base]

    _ensure_cik_map()
    cik = _CIK_MAP.get(base)
    if not cik:
        raise ValueError(f"{base} sem CIK EDGAR")
    r = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=_EDGAR_HEADERS, timeout=15,
    )
    if r.status_code == 404:
        raise ValueError(f"{base} sem cobertura EDGAR (404)")
    r.raise_for_status()
    data = r.json()
    _edgar_submissions_cache[base] = data
    return data


def _parse_form4_xml(accession_no: str, cik: str) -> dict:
    """Parseia um Form 4 XML e devolve {amount_usd, is_purchase, name, title}.

    Extrai:
    - amount_usd : valor total das compras (shares × price)
    - is_purchase: True se houve aquisição (code A)
    - name        : nome do insider (ex: "John Smith")
    - title       : cargo (ex: "Chief Executive Officer", "Director")

    Usa xml.etree.ElementTree — sem dependências extra.
    """
    import requests
    import xml.etree.ElementTree as ET

    acc_clean = accession_no.replace("-", "")
    doc_url   = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}"
        f"/{acc_clean}/{acc_clean}.txt"
    )
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

    # ── Identidade do insider ─────────────────────────────────────────────────
    name  = root.findtext(".//reportingOwner/reportingOwnerId/rptOwnerName", "").strip()
    title = root.findtext(
        ".//reportingOwner/reportingOwnerRelationship/officerTitle", ""
    ).strip()
    is_director = root.findtext(
        ".//reportingOwner/reportingOwnerRelationship/isDirector", "0"
    ).strip() == "1"
    if not title and is_director:
        title = "Director"

    # ── Transacções ───────────────────────────────────────────────────────────
    total_amount = 0.0
    has_purchase = False
    for txn in root.findall(".//nonDerivativeTransaction"):
        code = txn.findtext(".//transactionAcquiredDisposedCode/value", "").strip()
        if code != "A":  # A = Acquired (compra); D = Disposed (venda)
            continue
        has_purchase = True
        try:
            shares = float(txn.findtext(".//transactionShares/value", "0") or 0)
            price  = float(txn.findtext(".//transactionPricePerShare/value", "0") or 0)
            total_amount += shares * price
        except (ValueError, TypeError):
            pass

    return {
        "amount_usd":  total_amount,
        "is_purchase": has_purchase,
        "name":        name,
        "title":       title,
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
        forms        = recent.get("form", [])
        dates_filed  = recent.get("filingDate", [])
        accessions   = recent.get("accessionNumber", [])
        as_of  = as_of_date or date.today()
        cutoff = as_of - timedelta(days=lookback_days)

        total_purchase = 0.0
        top_name  = ""
        top_title = ""
        top_amount = 0.0

        for form, filed_str, acc in zip(forms, dates_filed, accessions):
            if form not in ("4", "4/A"):
                continue
            try:
                filed = date.fromisoformat(filed_str[:10])
                if not (cutoff <= filed <= as_of):
                    continue
            except ValueError:
                continue
            parsed = _parse_form4_xml(acc, cik_str)
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


# Cache em memória dos detalhes do último insider e 8-K por ticker
_last_insider_details: dict[str, dict] = {}
_last_8k_details: dict[str, dict] = {}


def get_insider_details(ticker: str) -> dict:
    """Devolve {name, title, amount} do insider com maior compra recente."""
    return _last_insider_details.get(ticker.upper(), {})


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
        forms        = recent.get("form", [])
        dates_filed  = recent.get("filingDate", [])
        accessions   = recent.get("accessionNumber", [])
        items_list   = recent.get("items", [])
        as_of  = as_of_date or date.today()
        cutoff = as_of - timedelta(days=lookback_days)

        import requests
        for form, filed_str, acc, items in zip(forms, dates_filed, accessions, items_list):
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
            try:
                if date.fromisoformat(filed_str[:10]) < cutoff:
                    continue
            except ValueError:
                continue

            # Fetch documento
            acc_clean = acc.replace("-", "")
            url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}"
                f"/{acc_clean}/{acc_clean}.txt"
            )
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


# ── Camada 4: Alpha Vantage ────────────────────────────────────────────────────

def alphavantage_earnings_revision(ticker: str) -> float:
    """Tendência de revisões de EPS dos analistas via Alpha Vantage.

    Retorna valor normalizado [-1, +1]:
      +1 = apenas upgrades recentes
       0 = sem revisões ou neutro
      -1 = apenas downgrades recentes

    Raises ValueError se ALPHAVANTAGE_API_KEY não está definida.
    Propaga requests.HTTPError em falhas de API.
    """
    if not _ALPHAVANTAGE_KEY:
        raise ValueError(
            "ALPHAVANTAGE_API_KEY não definida. "
            "Cria a env var no Railway com a tua chave gratuita de alphavantage.co"
        )
    import requests
    # Earnings estimate consensus
    r = requests.get(
        "https://www.alphavantage.co/query",
        params={
            "function": "EARNINGS",
            "symbol": ticker,
            "apikey": _ALPHAVANTAGE_KEY,
        },
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()

    # Verificar rate limit
    if "Note" in data or "Information" in data:
        raise RuntimeError(
            f"Alpha Vantage rate limit atingido para {ticker}. "
            f"Free tier: 25 req/day. Mensagem: {data.get('Note') or data.get('Information')}"
        )

    # Parsear últimas 4 estimativas de EPS
    quarterly = data.get("quarterlyEarnings", [])[:4]
    if not quarterly:
        return NaN

    surprises = []
    for q in quarterly:
        est = q.get("estimatedEPS")
        actual = q.get("reportedEPS")
        if est is not None and actual is not None:
            try:
                e, a = float(est), float(actual)
                if abs(e) > 0.001:
                    surprises.append((a - e) / abs(e))
            except (ValueError, TypeError):
                pass

    if not surprises:
        return NaN

    # Tendência: surpresas recentes a melhorar ou piorar?
    avg = float(np.mean(surprises))
    return round(float(np.clip(avg, -2.0, 2.0)), 4)


# ── Camada 5: Financial Modeling Prep ─────────────────────────────────────────

def fmp_analyst_revision_trend(ticker: str, n_recent: int = 10) -> float:
    """Tendência de revisões de analistas via FMP API.

    Retorna [-1, +1]: +1 = só upgrades, -1 = só downgrades.
    Raises ValueError se FMP_API_KEY não está definida.
    """
    if not _FMP_KEY:
        raise ValueError(
            "FMP_API_KEY não definida. "
            "Cria a env var no Railway com a tua chave gratuita de financialmodelingprep.com"
        )
    import requests
    r = requests.get(
        f"https://financialmodelingprep.com/api/v3/analyst-stock-recommendations/{ticker}",
        params={"limit": n_recent, "apikey": _FMP_KEY},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()

    if not data or not isinstance(data, list):
        return NaN

    _UPGRADE   = {"buy", "outperform", "overweight", "strong buy", "accumulate"}
    _DOWNGRADE = {"sell", "underperform", "underweight", "strong sell", "reduce"}

    upgrades   = sum(1 for x in data if x.get("newGrade", "").lower() in _UPGRADE)
    downgrades = sum(1 for x in data if x.get("newGrade", "").lower() in _DOWNGRADE)
    total = len(data)

    if total == 0:
        return NaN

    trend = (upgrades - downgrades) / total
    return round(float(np.clip(trend, -1.0, 1.0)), 4)


# ── Orquestrador ──────────────────────────────────────────────────────────────

def compute_fundamental_signals(
    ticker: str,
    price_history: Optional[pd.DataFrame] = None,
    alert_date: Optional[Any] = None,
    use_alphavantage: bool = bool(_ALPHAVANTAGE_KEY),
    use_fmp: bool = bool(_FMP_KEY),
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
            result["recent_8k_score"] = classify_recent_8k(ticker)
        except Exception as e:
            log.error(f"[signals] 8k_class {ticker}: {e}")
            result["recent_8k_score"] = NaN
    else:
        result["insider_buy_recent"]      = NaN
        result["insider_buy_amount_score"] = NaN
        result["recent_8k_score"]          = NaN

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

    # Camada 4: Alpha Vantage
    if use_alphavantage:
        try:
            result["earnings_revision_av"] = alphavantage_earnings_revision(ticker)
        except Exception as e:
            log.error(f"[signals] alphavantage {ticker}: {e}")
            result["earnings_revision_av"] = NaN

    # Camada 5: FMP
    if use_fmp:
        try:
            result["analyst_revision_fmp"] = fmp_analyst_revision_trend(ticker)
        except Exception as e:
            log.error(f"[signals] fmp {ticker}: {e}")
            result["analyst_revision_fmp"] = NaN

    return result
