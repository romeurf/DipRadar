"""regenerate_training_base.py — Recomputa ml_training_base.parquet com features novas.

Adiciona ao parquet existente as colunas:
  - return_6m_pre, return_12m_pre  — multi-window momentum
  - sector_relative_6m              — sector-relative momentum 6m
  - vol_of_vol                      — std of rolling vol (20d/5d)
  - bb_width                        — Bollinger band width (20d)
  - vix_percentile_1y               — VIX percentil 1 ano
  - spy_rsi_14                      — RSI 14 do SPY (macro)
  - yield_10y_change_5d             — variação do ^TNX em 5d

Não recomputa as 29 features existentes (que já estão correctas no parquet
actual) — só adiciona as 7 novas em colunas extra.

Não inclui:
  - fcf_yield (precisa de fundamentals quarterly — fora do scope)
  - short_interest_ratio, earnings_surprise_avg, earnings_distance_days
    (yfinance só fornece valor actual, não histórico — sem signal útil
     para alertas pré-2025; resolve-se quando houver fonte histórica)

Uso:
    python scripts/regenerate_training_base.py [--in PATH] [--out PATH] [--cache DIR]

Estratégia:
  1. Lê parquet existente (36k linhas)
  2. Bulk download yfinance: SPY, ^VIX, ^TNX + 12 sector ETFs + 678 tickers
     com cache local por ticker (resume em re-runs)
  3. Para cada linha: slice de price history até alert_date → compute
     features novas via funções já existentes em ml_features.py
  4. Escreve parquet novo com as 7 colunas adicionadas
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Garante que o package raiz do repo está no sys.path para importar
# `ml_features` (que está na raiz, não dentro de scripts/).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# yfinance é importado dentro de funções para permitir testes sem rede

log = logging.getLogger("regen")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_IN  = Path("ml_training_base.parquet")
DEFAULT_OUT = Path("ml_training_base.parquet")
DEFAULT_CACHE_DIR = Path("/tmp/yf_cache_v2")

MACRO_TICKERS: list[str] = ["SPY", "^VIX", "^TNX"]
SECTOR_ETFS: list[str] = [
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK",
    "XLP", "XLRE", "XLU", "XLV", "XLY",
]

SECTOR_ETF: dict[str, str] = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
    "Unknown": "SPY",
}

NEW_FEATURE_COLS: list[str] = [
    "return_6m_pre",
    "return_12m_pre",
    "sector_relative_6m",
    "vol_of_vol",
    "bb_width",
    "vix_percentile_1y",
    "spy_rsi_14",
    "yield_10y_change_5d",
]


# ─────────────────────────────────────────────────────────────────────────────
# yfinance helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_one_ticker(
    ticker: str,
    cache_dir: Path,
    start: str = "1995-01-01",
    end: Optional[str] = None,
    force_refresh: bool = False,
) -> Optional[pd.DataFrame]:
    """Download price history de um ticker, com cache local em parquet."""
    import yfinance as yf

    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = ticker.replace("^", "_idx_").replace("/", "_")
    cache_file = cache_dir / f"{safe_name}.parquet"

    if not force_refresh and cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            log.warning(f"  cache corrupto {ticker} ({e}) — re-download")

    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    for attempt in range(3):
        try:
            df = yf.Ticker(ticker).history(
                start=start,
                end=end,
                auto_adjust=True,
                raise_errors=False,
            )
            if df is not None and not df.empty:
                # Drop tz para uniformizar com slices
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df = df[["Open", "High", "Low", "Close", "Volume"]]
                df.to_parquet(cache_file)
                return df
            return None
        except Exception as e:
            if attempt == 2:
                log.warning(f"  {ticker}: yfinance falhou ({e})")
                return None
            time.sleep(0.5 * (attempt + 1))
    return None


def _fetch_batch(
    tickers: list[str],
    cache_dir: Path,
    start: str = "1995-01-01",
    batch_size: int = 50,
) -> dict[str, pd.DataFrame]:
    """Fetch ticker histories with batch logging and cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, pd.DataFrame] = {}

    n = len(tickers)
    t0 = time.time()
    fetched = 0
    cached = 0
    failed = 0

    for i, ticker in enumerate(tickers):
        was_cached = (cache_dir / f"{ticker.replace('^','_idx_').replace('/','_')}.parquet").exists()
        df = _fetch_one_ticker(ticker, cache_dir, start=start)
        if df is None or df.empty:
            failed += 1
            continue
        result[ticker] = df
        if was_cached:
            cached += 1
        else:
            fetched += 1

        if (i + 1) % batch_size == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n - i - 1) / rate if rate > 0 else 0
            log.info(
                f"  [{i+1:>4}/{n}] fetched={fetched} cached={cached} "
                f"failed={failed} ({elapsed:.0f}s elapsed, ~{eta:.0f}s ETA)"
            )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation
# ─────────────────────────────────────────────────────────────────────────────

def _slice_history(
    hist: Optional[pd.DataFrame],
    alert_date: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """Slice price history to dates <= alert_date."""
    if hist is None or hist.empty:
        return None
    out = hist[hist.index <= alert_date]
    return out if len(out) >= 5 else None


def _compute_row_features(
    row: pd.Series,
    price_cache: dict[str, pd.DataFrame],
    sector_etf_cache: dict[str, pd.DataFrame],
    macro_cache: dict[str, pd.DataFrame],
) -> dict[str, float]:
    """Compute the 7 new features for one alert row."""
    from ml_features import (
        _FALLBACK,
        add_momentum_features,
        add_regime_features,
    )

    ticker = str(row["ticker"])
    alert_date = pd.Timestamp(row["alert_date"]).tz_localize(None) if pd.Timestamp(row["alert_date"]).tz is not None else pd.Timestamp(row["alert_date"])
    sector = str(row.get("sector", "Unknown") or "Unknown")
    etf = SECTOR_ETF.get(sector, "SPY")

    hist = _slice_history(price_cache.get(ticker), alert_date)
    spy_slice = _slice_history(macro_cache.get("SPY"), alert_date)
    vix_slice = _slice_history(macro_cache.get("^VIX"), alert_date)
    tnx_slice = _slice_history(macro_cache.get("^TNX"), alert_date)
    sec_slice = _slice_history(sector_etf_cache.get(etf), alert_date)

    fv: dict[str, float] = {}

    # Momentum + bb_width + vol_of_vol
    if hist is not None:
        try:
            add_momentum_features(fv, hist, sec_slice, spy_slice)
        except Exception as e:
            log.debug(f"  momentum failed {ticker}@{alert_date.date()}: {e}")
    # Fallbacks para keys ausentes
    for k in ("return_6m_pre", "return_12m_pre", "sector_relative_6m", "vol_of_vol", "bb_width"):
        fv.setdefault(k, _FALLBACK.get(k, 0.0))

    # Regime
    try:
        add_regime_features(fv, spy_slice, tnx_slice, alert_date, vix_slice)
    except Exception as e:
        log.debug(f"  regime failed {ticker}@{alert_date.date()}: {e}")
    for k in ("vix_percentile_1y", "spy_rsi_14", "yield_10y_change_5d"):
        fv.setdefault(k, _FALLBACK.get(k, 0.0))

    # Devolver só as 8 colunas de interesse
    return {k: float(fv.get(k, _FALLBACK.get(k, 0.0))) for k in NEW_FEATURE_COLS}


def regenerate(
    in_path: Path,
    out_path: Path,
    cache_dir: Path,
    start_date: str = "1995-01-01",
) -> None:
    """Pipeline principal."""
    if not in_path.exists():
        raise FileNotFoundError(f"input parquet não encontrado: {in_path}")

    log.info(f"[in] {in_path} ({in_path.stat().st_size/1e6:.1f} MB)")
    df = pd.read_parquet(in_path)
    df["alert_date"] = pd.to_datetime(df["alert_date"]).dt.tz_localize(None)
    log.info(f"[in] shape={df.shape}  unique tickers={df['ticker'].nunique()}")
    log.info(f"[in] date range: {df['alert_date'].min()} → {df['alert_date'].max()}")

    # ── Fetch macro + ETFs ──────────────────────────────────────────────────
    log.info("[fetch] macro + sector ETFs...")
    macro_etf_cache = _fetch_batch(
        MACRO_TICKERS + SECTOR_ETFS,
        cache_dir,
        start=start_date,
    )
    macro_cache = {k: v for k, v in macro_etf_cache.items() if k in MACRO_TICKERS}
    sector_etf_cache = {k: v for k, v in macro_etf_cache.items() if k in SECTOR_ETFS}
    # SPY conta como sector_etf fallback
    if "SPY" in macro_cache:
        sector_etf_cache.setdefault("SPY", macro_cache["SPY"])
    log.info(f"[fetch] macro={len(macro_cache)} sector_etfs={len(sector_etf_cache)}")

    # ── Fetch ticker histories ──────────────────────────────────────────────
    unique_tickers = sorted(df["ticker"].astype(str).unique().tolist())
    log.info(f"[fetch] {len(unique_tickers)} ticker histories...")
    price_cache = _fetch_batch(unique_tickers, cache_dir, start=start_date)
    log.info(f"[fetch] {len(price_cache)}/{len(unique_tickers)} tickers OK")

    # Tickers sem history vão usar fallbacks (não fica null no parquet)
    missing = [t for t in unique_tickers if t not in price_cache]
    if missing:
        log.warning(f"[fetch] {len(missing)} tickers sem history: {missing[:10]}...")

    # ── Compute features per row ────────────────────────────────────────────
    log.info("[compute] features por linha...")
    t0 = time.time()
    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        results.append(_compute_row_features(row, price_cache, sector_etf_cache, macro_cache))
        if (i + 1) % 2000 == 0 or (i + 1) == len(df):
            elapsed = time.time() - t0
            log.info(f"  [{i+1:>5}/{len(df)}] {elapsed:.0f}s elapsed")

    new_cols_df = pd.DataFrame(results, index=df.index)

    # ── Merge + save ────────────────────────────────────────────────────────
    for col in NEW_FEATURE_COLS:
        df[col] = new_cols_df[col].values

    log.info(f"[out] writing {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info(f"[out] done. shape={df.shape}  size={out_path.stat().st_size/1e6:.1f} MB")

    # Sanity stats
    log.info("[stats] new feature summary:")
    for col in NEW_FEATURE_COLS:
        s = df[col]
        log.info(
            f"  {col:24s} mean={s.mean():>8.4f}  std={s.std():>7.4f}  "
            f"min={s.min():>8.4f}  max={s.max():>8.4f}  "
            f"nan={s.isna().sum()}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in",  dest="in_path",  type=Path, default=DEFAULT_IN)
    p.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cache", dest="cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--start", dest="start_date", default="1995-01-01")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    regenerate(args.in_path, args.out_path, args.cache_dir, args.start_date)
    return 0


if __name__ == "__main__":
    sys.exit(main())
