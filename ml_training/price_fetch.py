"""yfinance fetch utilities — extraído do notebook (cell 10)."""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from ml_training.config import DEFAULT_ETF, HORIZON_DAYS, SECTOR_ETF

log = logging.getLogger(__name__)


def fetch_ohlcv_batch(
    tickers_list: list[str],
    start: str,
    end: str,
    batch_size: int = 40,
    progress_log: bool = True,
) -> dict[str, pd.DataFrame]:
    """Bulk yfinance.download em batches; devolve {ticker: DataFrame}.

    Match exacto do helper do notebook (cell 10):
      - ``auto_adjust=False, threads=True``
      - Fica com OHLCV (sem Adj Close)
      - Salta tickers com < 50 candles
    """
    import yfinance as yf  # import lazy

    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers_list), batch_size):
        batch = tickers_list[i:i + batch_size]
        try:
            data = yf.download(
                batch,
                start=start,
                end=end,
                progress=False,
                group_by="ticker",
                auto_adjust=False,
                threads=True,
            )
        except Exception as e:  # pragma: no cover (network)
            log.warning(f"  batch {i}-{i+batch_size}: erro {e!r}")
            continue
        for tk in batch:
            try:
                if len(batch) == 1:
                    sub = data[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                else:
                    sub = data[tk][["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                if len(sub) > 50:
                    out[tk] = sub
            except Exception:
                pass
        if progress_log:
            log.info(f"  fetched {min(i + batch_size, len(tickers_list))}/{len(tickers_list)}")
    return out


def fetch_caches_for_dataset(
    base_df: pd.DataFrame,
    horizon_days: int = HORIZON_DAYS,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Fetch SPY + sector ETFs + stocks usados em ``base_df``.

    Devolve (etf_cache, price_cache). Replica a lógica das cells 9 e 10.
    """
    if "ticker" not in base_df.columns:
        raise KeyError("base_df precisa de coluna 'ticker'")
    if "alert_date" not in base_df.columns:
        raise KeyError("base_df precisa de coluna 'alert_date'")
    if "sector" not in base_df.columns:
        raise KeyError("base_df precisa de coluna 'sector'")

    tickers = sorted(base_df["ticker"].dropna().unique().tolist())
    sectors_present = base_df["sector"].dropna().unique()
    etfs = sorted({DEFAULT_ETF} | {SECTOR_ETF.get(s, DEFAULT_ETF) for s in sectors_present})

    dates = pd.to_datetime(base_df["alert_date"])
    start = (dates.min() - pd.Timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    end = (dates.max() + pd.Timedelta(days=horizon_days + 7)).strftime("%Y-%m-%d")

    log.info(f"[fetch] A fetchar: {len(tickers)} stocks + {len(etfs)} ETFs (SPY incluído)")
    log.info(f"[fetch] Período: {start} → {end}")

    etf_cache = fetch_ohlcv_batch(etfs, start, end, batch_size=20)
    price_cache = fetch_ohlcv_batch(tickers, start, end, batch_size=40)
    log.info(f"[fetch] ETFs OK: {len(etf_cache)}/{len(etfs)} | Stocks OK: {len(price_cache)}/{len(tickers)}")
    return etf_cache, price_cache
