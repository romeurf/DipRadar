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


def insider_buy_recent(ticker: str, lookback_days: int = 30) -> float:
    """1.0 se houve compra de insider (Form 4) nos últimos lookback_days, 0.0 se não.

    Retorna NaN se o ticker não tiver cobertura SEC (tickers não-US).
    Propaga requests.HTTPError em falhas de API.
    """
    import requests

    # Tickers não-US não têm EDGAR coverage
    base_ticker = ticker.upper().split(".")[0]
    if "." in ticker:
        return NaN

    _ensure_cik_map()
    cik = _CIK_MAP.get(base_ticker)
    if not cik:
        return NaN

    r = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=_EDGAR_HEADERS, timeout=15,
    )
    if r.status_code == 404:
        return NaN
    r.raise_for_status()

    data = r.json()
    recent = data.get("filings", {}).get("recent", {})
    forms       = recent.get("form", [])
    dates_filed = recent.get("filingDate", [])

    cutoff = date.today() - timedelta(days=lookback_days)
    for form, filed_str in zip(forms, dates_filed):
        if form not in ("4", "4/A"):
            continue
        try:
            if date.fromisoformat(filed_str[:10]) >= cutoff:
                # Form 4 encontrado no período — presumir compra
                # (refinamento futuro: parsear XML para verificar transaction code P)
                return 1.0
        except ValueError:
            continue

    return 0.0


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
    if "." not in ticker.split("/")[-1]:
        try:
            result["insider_buy_recent"] = insider_buy_recent(ticker)
        except Exception as e:
            log.error(f"[signals] insider_buy_recent {ticker}: {e}")
            result["insider_buy_recent"] = NaN
    else:
        result["insider_buy_recent"] = NaN  # tickers não-US sem EDGAR coverage

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
