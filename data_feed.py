"""
data_feed.py — Módulo de ingestão de dados EOD para o DipRadar 2.0.

Fonte primária : Tiingo API (EOD limpos, sem rate limit agressivo)
Fallback       : yfinance (fundamentais e tickers não cobertos pela Tiingo)

Variáveis de ambiente necessárias (Railway):
  TIINGO_API_KEY   — chave da API Tiingo (gratuita em api.tiingo.com)

Uso rápido:
  from data_feed import get_eod_prices, get_bulk_eod
  df = get_eod_prices("MSFT", lookback_days=60)
  bulk = get_bulk_eod(["AAPL", "NVDA", "SAP.DE"], lookback_days=30)
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────────────────────────────────

TIINGO_API_KEY: str = os.getenv("TIINGO_API_KEY", "")
TIINGO_BASE    = "https://api.tiingo.com/tiingo/daily"
TIINGO_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Token {TIINGO_API_KEY}",
}

# Sufixos de exchange europeia que a Tiingo reconhece
# A Tiingo usa '-' em vez de '.' para separar sufixo: SAP.DE → SAP-DE
_TIINGO_EXCHANGE_MAP: dict[str, str] = {
    ".DE": "-DE",   # Xetra
    ".PA": "-PA",   # Euronext Paris
    ".AS": "-AS",   # Euronext Amsterdam
    ".MC": "-MC",   # BME Madrid
    ".MI": "-MI",   # Borsa Italiana
    ".L":  "-L",    # London Stock Exchange
    ".SW": "-SW",   # SIX Swiss
    ".ST": "-ST",   # Nasdaq Stockholm
    ".CO": "-CO",   # Nasdaq Copenhagen
    ".OL": "-OL",   # Oslo Bors
    ".HE": "-HE",   # Nasdaq Helsinki
    ".BR": "-BR",   # Euronext Brussels
    ".LS": "-LS",   # Euronext Lisboa
    ".VI": "-VI",   # Vienna
    ".WA": "-WA",   # Warsaw
    ".I":  "-I",    # Euronext Dublin
}

# Tickers europeus não cobertos pela Tiingo — vai directo ao yfinance
TIINGO_UNSUPPORTED: set[str] = set()


# ─────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────

def _to_tiingo_ticker(ticker: str) -> str:
    """Converte ticker yfinance para formato Tiingo. Ex: SAP.DE → SAP-DE."""
    t = ticker.upper()
    for yf_suffix, tiingo_suffix in _TIINGO_EXCHANGE_MAP.items():
        if t.endswith(yf_suffix):
            return t[: -len(yf_suffix)] + tiingo_suffix
    return t


def _date_range(lookback_days: int) -> tuple[str, str]:
    end   = date.today()
    start = end - timedelta(days=lookback_days)
    return start.isoformat(), end.isoformat()


def _parse_tiingo_response(data: list[dict], ticker: str) -> pd.DataFrame:
    """
    Normaliza resposta Tiingo para DataFrame padronizado.
    Colunas garantidas: date, open, high, low, close, volume, adjClose
    """
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)

    # Colunas obrigatórias — yfinance compatibility layer
    rename = {
        "adjClose":  "Adj Close",
        "adjOpen":   "Open",
        "adjHigh":   "High",
        "adjLow":    "Low",
        "adjVolume": "Volume",
        "close":     "Close",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["ticker"] = ticker

    # Garante que 'Adj Close' existe (usa 'Close' como fallback)
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    return df[["date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "ticker"]]


# ─────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────

def get_eod_prices(
    ticker: str,
    lookback_days: int = 60,
    force_yfinance: bool = False,
) -> pd.DataFrame:
    """
    Retorna DataFrame com preços EOD para um ticker.

    Fluxo:
      1. Tiingo API  (primário — EOD limpos)
      2. yfinance    (fallback automático se Tiingo falhar ou ticker não coberto)

    Returns:
      DataFrame com colunas: date, Open, High, Low, Close, Adj Close, Volume, ticker
      DataFrame vazio se ambas as fontes falharem.
    """
    if not force_yfinance and TIINGO_API_KEY and ticker not in TIINGO_UNSUPPORTED:
        df = _tiingo_fetch(ticker, lookback_days)
        if not df.empty:
            return df
        # Marca como não suportado para esta sessão
        TIINGO_UNSUPPORTED.add(ticker)
        log.warning(f"[data_feed] Tiingo sem dados para {ticker} — fallback yfinance")

    return _yfinance_fetch(ticker, lookback_days)


def get_bulk_eod(
    tickers: list[str],
    lookback_days: int = 60,
    delay_between: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """
    Fetch EOD em bulk para uma lista de tickers.
    Usa Tiingo com fallback automático por ticker.

    Args:
      tickers        : Lista de tickers (formato yfinance aceite)
      lookback_days  : Dias de histórico
      delay_between  : Pausa entre chamadas (segundos) para respeitar rate limit

    Returns:
      Dict {ticker: DataFrame} — tickers com falha têm DataFrame vazio.
    """
    results: dict[str, pd.DataFrame] = {}
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        try:
            df = get_eod_prices(ticker, lookback_days=lookback_days)
            results[ticker] = df
            if i % 50 == 0:
                log.info(f"[data_feed] bulk {i}/{total} tickers processados")
        except Exception as e:
            log.error(f"[data_feed] Erro a buscar {ticker}: {e}")
            results[ticker] = pd.DataFrame()

        time.sleep(delay_between)

    ok      = sum(1 for df in results.values() if not df.empty)
    failed  = total - ok
    log.info(f"[data_feed] Bulk EOD completo: {ok} OK, {failed} falhas de {total}")
    return results


def get_latest_price(ticker: str) -> Optional[float]:
    """
    Devolve o último preço de fecho (Adj Close) para um ticker.
    Usado pelo check_thesis_degradation() para updates rápidos de preço.
    """
    df = get_eod_prices(ticker, lookback_days=5)
    if df.empty:
        return None
    return float(df["Adj Close"].iloc[-1])


def is_tiingo_available() -> bool:
    """Verifica se a chave Tiingo está configurada e funcional."""
    if not TIINGO_API_KEY:
        return False
    try:
        url  = f"{TIINGO_BASE}/AAPL/prices"
        r    = requests.get(url, headers=TIINGO_HEADERS, params={"startDate": date.today().isoformat()}, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────
# Fontes internas
# ─────────────────────────────────────────────────────────────────────────

def _tiingo_fetch(ticker: str, lookback_days: int) -> pd.DataFrame:
    """Fetch directo à Tiingo REST API."""
    tiingo_ticker      = _to_tiingo_ticker(ticker)
    start_date, end_date = _date_range(lookback_days)

    url    = f"{TIINGO_BASE}/{tiingo_ticker}/prices"
    params = {
        "startDate": start_date,
        "endDate":   end_date,
        "resampleFreq": "daily",
    }

    try:
        r = requests.get(url, headers=TIINGO_HEADERS, params=params, timeout=10)
        if r.status_code == 404:
            return pd.DataFrame()  # ticker não coberto
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame()
        return _parse_tiingo_response(data, ticker)
    except requests.exceptions.RequestException as e:
        log.warning(f"[data_feed] Tiingo HTTP erro para {ticker}: {e}")
        return pd.DataFrame()
    except Exception as e:
        log.warning(f"[data_feed] Tiingo parse erro para {ticker}: {e}")
        return pd.DataFrame()


def _yfinance_fetch(ticker: str, lookback_days: int) -> pd.DataFrame:
    """Fallback: yfinance download."""
    try:
        import yfinance as yf
        start_date, end_date = _date_range(lookback_days)
        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            show_errors=False,
        )
        if raw.empty:
            return pd.DataFrame()

        # yfinance retorna MultiIndex se apenas 1 ticker — normalizar
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw.reset_index().rename(columns={"Date": "date", "index": "date"})
        raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)

        if "Adj Close" not in raw.columns and "Close" in raw.columns:
            raw["Adj Close"] = raw["Close"]

        raw["ticker"] = ticker
        cols = [c for c in ["date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "ticker"] if c in raw.columns]
        return raw[cols]

    except Exception as e:
        log.error(f"[data_feed] yfinance fallback falhou para {ticker}: {e}")
        return pd.DataFrame()
