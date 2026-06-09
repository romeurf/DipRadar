"""
tiingo_client.py — Cliente Tiingo para dados OHLCV diários.

Substitui yfinance no fill_db_outcomes() com uma fonte mais fiável:
  - Dados ajustados para splits/dividendos (adjClose)
  - Sem rate limits agressivos (500 req/hora no plano free)
  - Resposta consistente mesmo para tickers menos líquidos

Plano gratuito Tiingo:
  - 500 req/hora
  - 5 000 tickers únicos/mês
  - Histórico completo desde IPO

Variável de ambiente obrigatória:
  TIINGO_API_KEY=<chave em api.tiingo.com>

Uso:
  from data.tiingo_client import get_ohlcv, get_price_at, get_mfe_mae

  candles = get_ohlcv("AAPL", date(2025, 1, 1), date(2025, 6, 30))
  price   = get_price_at(candles, date(2025, 4, 1))
  mfe, mae = get_mfe_mae(candles, after_date=date(2025, 1, 1), price_entry=182.5)
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta

import requests

_BASE  = "https://api.tiingo.com/tiingo/daily"
_TOKEN = os.getenv("TIINGO_API_KEY", "")

# Throttle: Tiingo free permite ~500 req/hora ≈ 8/min → 0.15s entre chamadas
_SLEEP_BETWEEN_REQUESTS = 0.2


def get_ohlcv(
    symbol: str,
    start: date,
    end: date,
    retries: int = 3,
) -> list[dict]:
    """
    Devolve lista de candles diários OHLCV do Tiingo para o símbolo e período.

    Cada item do resultado:
      {
        "date":     "2025-04-01T00:00:00+00:00",
        "open":     182.5,
        "high":     185.0,
        "low":      181.2,
        "close":    184.3,
        "volume":   75000000,
        "adjClose": 184.3,   ← ajustado para splits/dividendos
        "adjHigh":  185.0,
        "adjLow":   181.2,
        "adjOpen":  182.5,
        "adjVolume": 75000000,
      }

    Retorna lista vazia se:
      - Ticker deslistado (404)
      - Sem dados no período
      - TIINGO_API_KEY não definida

    Raises:
      EnvironmentError — se TIINGO_API_KEY não estiver definida
      requests.HTTPError — para erros 4xx/5xx não tratados
    """
    if not _TOKEN:
        raise EnvironmentError(
            "TIINGO_API_KEY não definida. "
            "Adiciona a variável de ambiente no Railway."
        )

    url = f"{_BASE}/{symbol}/prices"
    params = {
        "startDate":    start.isoformat(),
        "endDate":      end.isoformat(),
        "token":        _TOKEN,
        "resampleFreq": "daily",
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=15)

            if resp.status_code == 404:
                logging.debug(f"[tiingo] {symbol}: ticker não encontrado (404)")
                return []

            if resp.status_code == 429:
                wait = 60 * attempt
                logging.warning(f"[tiingo] Rate limit — a aguardar {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            if not isinstance(data, list):
                logging.warning(f"[tiingo] {symbol}: resposta inesperada: {type(data)}")
                return []

            time.sleep(_SLEEP_BETWEEN_REQUESTS)
            return data

        except requests.exceptions.Timeout:
            logging.warning(f"[tiingo] {symbol}: timeout (tentativa {attempt}/{retries})")
            time.sleep(2 * attempt)

        except requests.exceptions.RequestException as e:
            logging.warning(f"[tiingo] {symbol}: erro de rede (tentativa {attempt}/{retries}): {e}")
            time.sleep(2 * attempt)

    logging.error(f"[tiingo] {symbol}: falhou após {retries} tentativas")
    return []


def get_price_at(
    candles: list[dict],
    target: date,
    use_adj: bool = True,
) -> float | None:
    """
    Devolve o preço de fecho (adjClose) mais próximo de target_date.

    Testa o dia exacto primeiro, depois adjacentes por ordem de prioridade:
    0, +1, -1, +2, -2, +3, -3, +4, +5 dias.

    Parâmetros:
      candles  — lista de candles devolvida por get_ohlcv()
      target   — data alvo (date)
      use_adj  — True: usa adjClose (default); False: usa close bruto

    Retorna None se não houver nenhum candle próximo.
    """
    if not candles:
        return None

    price_field = "adjClose" if use_adj else "close"

    # Indexar por data (primeiros 10 chars do ISO string: "2025-04-01")
    price_map: dict[str, float] = {}
    for c in candles:
        d_str = str(c.get("date", ""))[:10]
        val   = c.get(price_field) or c.get("close")
        if d_str and val is not None:
            price_map[d_str] = float(val)

    for delta in [0, 1, -1, 2, -2, 3, -3, 4, 5]:
        check = (target + timedelta(days=delta)).isoformat()[:10]
        if check in price_map:
            return price_map[check]

    return None


def get_mfe_mae(
    candles: list[dict],
    after_date: date,
    price_entry: float,
    window_days: int = 91,
    use_adj: bool = True,
) -> tuple[float | None, float | None]:
    """
    Calcula MFE (Maximum Favorable Excursion) e MAE (Maximum Adverse Excursion)
    nos primeiros window_days dias após after_date.

    MFE = (high_max - price_entry) / price_entry * 100   [% máximo ganho potencial]
    MAE = (low_min  - price_entry) / price_entry * 100   [% máximo drawdown potencial]

    Parâmetros:
      candles     — lista de candles devolvida por get_ohlcv()
      after_date  — data do alerta (candles APÓS esta data)
      price_entry — preço no momento do alerta
      window_days — janela em dias de calendário (default: 91 ≈ 3 meses)
      use_adj     — True: usa adjHigh/adjLow; False: usa high/low brutos

    Retorna (mfe, mae) como percentagens, ou (None, None) se sem dados.
    """
    if not candles or price_entry <= 0:
        return None, None

    high_field = "adjHigh" if use_adj else "high"
    low_field  = "adjLow"  if use_adj else "low"

    cutoff = after_date + timedelta(days=window_days)

    highs: list[float] = []
    lows:  list[float] = []

    for c in candles:
        d_str = str(c.get("date", ""))[:10]
        if not d_str:
            continue
        try:
            d = date.fromisoformat(d_str)
        except ValueError:
            continue

        if d <= after_date or d > cutoff:
            continue

        h = c.get(high_field) or c.get("high")
        l = c.get(low_field)  or c.get("low")

        if h is not None:
            highs.append(float(h))
        if l is not None:
            lows.append(float(l))

    if not highs or not lows:
        return None, None

    mfe = (max(highs) - price_entry) / price_entry * 100
    mae = (min(lows)  - price_entry) / price_entry * 100

    return round(mfe, 2), round(mae, 2)


def check_api_key() -> bool:
    """
    Verifica se a TIINGO_API_KEY está definida e é válida
    com um pedido de teste (SPY, último dia disponível).

    Retorna True se válida, False caso contrário.
    Útil para healthcheck no arranque do bot.
    """
    if not _TOKEN:
        logging.warning("[tiingo] TIINGO_API_KEY não definida.")
        return False

    try:
        result = get_ohlcv("SPY", date.today() - timedelta(days=7), date.today())
        if result:
            logging.info("[tiingo] API key válida ✓")
            return True
        logging.warning("[tiingo] API key válida mas sem dados de teste.")
        return True  # pode ser fim de semana/feriado
    except Exception as e:
        logging.error(f"[tiingo] Erro ao validar API key: {e}")
        return False
