"""yfinance fetch utilities — extraído do notebook (cell 10).

Changelog:
  - load_etf_cache(): download de ETFs sector via yfinance (substituiu Tiingo)
    com cache local em parquet por ticker — re-runs instantâneos.
  - fetch_caches_for_dataset(): usa load_etf_cache() em vez de Tiingo.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from ml_training.config import DEFAULT_ETF, HORIZON_DAYS, SECTOR_ETF

log = logging.getLogger(__name__)

SECTOR_ETFS = [
    "SPY", "XLB", "XLC", "XLE", "XLF",
    "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
]


# ─────────────────────────────────────────────────────────────────────────────
# ETF cache (yfinance) — substitui Tiingo
# ─────────────────────────────────────────────────────────────────────────────

def load_etf_cache(
    etfs: list[str] | None = None,
    start: str = "2018-01-01",
    end: str | None = None,
    cache_dir: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Download de ETFs sector via yfinance com cache local em parquet.

    Parâmetros
    ----------
    etfs:
        Lista de tickers ETF. Por omissão usa ``SECTOR_ETFS``.
    start:
        Data de início (YYYY-MM-DD).
    end:
        Data de fim (YYYY-MM-DD). Por omissão hoje.
    cache_dir:
        Directório para guardar/ler parquets por ticker.
        Se ``None``, não faz cache em disco.

    Devolve
    -------
    dict[str, pd.DataFrame]
        Chave = ticker; valor = DataFrame com index DatetimeIndex e colunas
        Open, High, Low, Close, Volume (auto_adjust=True).
    """
    import yfinance as yf
    from datetime import date

    if etfs is None:
        etfs = SECTOR_ETFS
    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}
    to_download: list[str] = []

    # ── 1. verifica cache local ──────────────────────────────────────────────
    for etf in etfs:
        if cache_path is not None:
            f = cache_path / f"{etf}.parquet"
            if f.exists():
                df = pd.read_parquet(f)
                df.index = pd.to_datetime(df.index)
                results[etf] = df
                log.info(f"  📦 {etf}: cache local ✅ ({len(df)} linhas)")
                continue
        to_download.append(etf)

    # ── 2. download em batch único ───────────────────────────────────────────
    if to_download:
        log.info(f"  ⬇️  yfinance download: {to_download}")
        raw = yf.download(
            tickers=to_download,
            start=start,
            end=end,
            auto_adjust=True,   # adjClose incorporado em OHLC
            group_by="ticker",
            threads=True,
            progress=True,
        )

        for etf in to_download:
            try:
                df = raw[etf].copy() if len(to_download) > 1 else raw.copy()
                df = df.dropna(subset=["Close"])
                df.index = pd.to_datetime(df.index)

                if len(df) < 50:
                    log.warning(f"  ⚠️  {etf}: apenas {len(df)} candles — ignorado")
                    continue

                # guarda cache
                if cache_path is not None:
                    f = cache_path / f"{etf}.parquet"
                    df.to_parquet(f)
                    log.info(f"  ✅ {etf}: {len(df)} linhas → cache guardado")
                else:
                    log.info(f"  ✅ {etf}: {len(df)} linhas")

                results[etf] = df

            except (KeyError, Exception) as exc:
                log.warning(f"  ❌ {etf}: {exc!r}")

    log.info(
        f"[load_etf_cache] {len(results)}/{len(etfs)} ETFs carregados"
        + (f" | em falta: {[e for e in etfs if e not in results]}" if len(results) < len(etfs) else "")
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Stocks batch fetch (cell 10 original)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Fetch completo para dataset (cells 9+10)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_caches_for_dataset(
    base_df: pd.DataFrame,
    horizon_days: int = HORIZON_DAYS,
    etf_cache_dir: Path | str | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Fetch SPY + sector ETFs + stocks usados em ``base_df``.

    Devolve (etf_cache, price_cache). Replica a lógica das cells 9 e 10.
    ETFs são agora descarregados via yfinance (sem Tiingo).
    """
    for col in ("ticker", "alert_date", "sector"):
        if col not in base_df.columns:
            raise KeyError(f"base_df precisa de coluna '{col}'")

    tickers = sorted(base_df["ticker"].dropna().unique().tolist())
    sectors_present = base_df["sector"].dropna().unique()
    etfs = sorted({DEFAULT_ETF} | {SECTOR_ETF.get(s, DEFAULT_ETF) for s in sectors_present})

    dates = pd.to_datetime(base_df["alert_date"])
    start = (dates.min() - pd.Timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    end   = (dates.max() + pd.Timedelta(days=horizon_days + 7)).strftime("%Y-%m-%d")

    log.info(f"[fetch] A fetchar: {len(tickers)} stocks + {len(etfs)} ETFs (SPY incluído)")
    log.info(f"[fetch] Período: {start} → {end}")

    # ETFs via yfinance + cache local
    etf_cache = load_etf_cache(
        etfs=etfs,
        start=start,
        end=end,
        cache_dir=etf_cache_dir,
    )

    # Stocks via batch (sem cache — dataset muda frequentemente)
    price_cache = fetch_ohlcv_batch(tickers, start, end, batch_size=40)

    log.info(
        f"[fetch] ETFs OK: {len(etf_cache)}/{len(etfs)} | "
        f"Stocks OK: {len(price_cache)}/{len(tickers)}"
    )
    return etf_cache, price_cache
