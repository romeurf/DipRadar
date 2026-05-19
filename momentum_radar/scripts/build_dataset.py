"""
build_momentum_dataset.py — Constrói dataset de treino para o Breakout ML.

Diferente do DipRadar (que usa alertas de dip como ponto de entrada),
o Breakout ML aprende a partir de dias em que um stock estava em momentum:
  - return_20d >= 10%  (breakout confirmado)
  - volume acima da média
  - RSI em zona de momentum (50-78)

Target: forward_return_30d — quanto o stock retornou nos 30 dias seguintes.
Horizonte mais curto que o DipRadar (90d) porque momentum fades faster.

Uso:
  python scripts/build_momentum_dataset.py
  python scripts/build_momentum_dataset.py --tickers AAPL,MSFT,NVDA  # teste rápido
  python scripts/build_momentum_dataset.py --start 2020-01-01

Output: /data/momentum_training.parquet
  Colunas: date, ticker, sector, [features], forward_return_30d, alpha_30d
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("momentum_dataset")

_DATA_DIR   = Path("/data") if Path("/data").exists() else Path("/tmp")
_CACHE_DIR  = _DATA_DIR / "price_cache"
_OUTPUT     = _DATA_DIR / "momentum_training.parquet"

# Critérios de entrada — dias que qualificam como "momentum"
_MIN_RETURN_20D   = 0.10    # +10% em 20 dias
_MIN_VOLUME_RATIO = 1.30    # 30% acima da média
_RSI_MIN          = 50.0
_RSI_MAX          = 78.0
_MIN_MARKET_CAP_B = 2.0     # liquidez mínima


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de preços
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_history(ticker: str, start: str = "2019-01-01") -> Optional[pd.DataFrame]:
    cache = _CACHE_DIR / f"{ticker.replace('^','_idx_').replace('/','_').replace('.','_')}.parquet"
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 86400 * 7:
        try:
            df = pd.read_parquet(cache)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception:
            pass
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
        if df is None or df.empty or len(df) < 30:
            return None
        idx = pd.DatetimeIndex(df.index)
        df.index = idx.tz_convert(None) if idx.tz is not None else idx
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        df.to_parquet(cache)
        return df
    except Exception as e:
        log.debug(f"  {ticker}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering para cada dia de momentum
# ─────────────────────────────────────────────────────────────────────────────

def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _build_momentum_features(hist: pd.DataFrame, idx: int) -> Optional[dict]:
    """Computa features de momentum para o dia hist.index[idx]."""
    if idx < 25:
        return None

    close  = hist["Close"]
    volume = hist["Volume"]
    high   = hist["High"]
    low    = hist["Low"]

    close_slice  = close.iloc[:idx+1]
    volume_slice = volume.iloc[:idx+1]

    # Retornos
    ret_20d = float(close.iloc[idx] / close.iloc[max(0, idx-20)] - 1) if idx >= 20 else None
    ret_5d  = float(close.iloc[idx] / close.iloc[max(0, idx-5)]  - 1) if idx >= 5  else None
    ret_60d = float(close.iloc[idx] / close.iloc[max(0, idx-60)] - 1) if idx >= 60 else None

    if ret_20d is None or ret_20d < _MIN_RETURN_20D:
        return None

    # Volume ratio
    vol_avg = float(volume_slice.iloc[-20:].mean()) if len(volume_slice) >= 20 else None
    vol_ratio = float(volume.iloc[idx] / vol_avg) if vol_avg and vol_avg > 0 else 1.0

    if vol_ratio < _MIN_VOLUME_RATIO:
        return None

    # RSI
    rsi_series = _compute_rsi(close_slice)
    rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

    if not (_RSI_MIN <= rsi <= _RSI_MAX):
        return None

    # Distância do máximo de 52 semanas
    high_52w = float(close_slice.iloc[-252:].max()) if len(close_slice) >= 20 else float(close_slice.max())
    pct_from_high = float(close.iloc[idx] / high_52w - 1)

    # ATR 14d
    h = high.iloc[max(0,idx-14):idx+1]
    l = low.iloc[max(0,idx-14):idx+1]
    c_prev = close.iloc[max(0,idx-15):idx].values
    if len(c_prev) > 0 and len(h) == len(l):
        tr_hl = h.values - l.values
        tr_hc = np.abs(h.values[1:] - c_prev[-len(h)+1:]) if len(c_prev) >= len(h)-1 else tr_hl[1:]
        tr_lc = np.abs(l.values[1:] - c_prev[-len(l)+1:]) if len(c_prev) >= len(l)-1 else tr_hl[1:]
        atr_pct = float(np.mean(tr_hl)) / float(close.iloc[idx]) if close.iloc[idx] > 0 else 0.02
    else:
        atr_pct = 0.02

    # Close in range 20d (fechamentos perto dos máximos = força)
    if len(high.iloc[max(0,idx-20):idx+1]) >= 5:
        h20 = high.iloc[max(0,idx-20):idx+1].values
        l20 = low.iloc[max(0,idx-20):idx+1].values
        c20 = close.iloc[max(0,idx-20):idx+1].values
        ranges = h20 - l20
        close_in_range = float(np.mean((c20 - l20) / np.where(ranges > 0, ranges, 1)))
    else:
        close_in_range = 0.5

    return {
        "return_20d":        round(ret_20d, 4),
        "return_5d":         round(ret_5d or 0, 4),
        "return_60d_pre":    round(ret_60d or 0, 4),
        "volume_ratio_20d":  round(vol_ratio, 3),
        "rsi_14":            round(rsi, 2),
        "pct_from_52w_high": round(pct_from_high, 4),
        "atr_pct":           round(atr_pct, 4),
        "close_in_range_20d":round(close_in_range, 4),
        "price":             round(float(close.iloc[idx]), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    tickers: list[str],
    spy_hist: pd.DataFrame,
    start: str = "2019-01-01",
    sample_every_n_days: int = 5,   # não amostrar cada dia — apenas 1 em 5 dias
) -> pd.DataFrame:
    """Para cada ticker, encontra dias com momentum e calcula forward_return_30d."""
    rows = []
    n_tickers = len(tickers)

    for i, ticker in enumerate(tickers):
        hist = _fetch_history(ticker, start=start)
        if hist is None or len(hist) < 60:
            continue

        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info or {}
            sector = info.get("sector", "Unknown") or "Unknown"
            mktcap = float(info.get("marketCap") or 0) / 1e9
            if mktcap > 0 and mktcap < _MIN_MARKET_CAP_B:
                continue
        except Exception:
            sector = "Unknown"

        close = hist["Close"]
        n = len(hist)

        # Amostrar cada N dias para evitar autocorrelação temporal extrema
        sample_indices = range(25, n - 35, sample_every_n_days)

        for idx in sample_indices:
            feats = _build_momentum_features(hist, idx)
            if feats is None:
                continue

            # Forward return 30d
            future_close = close.iloc[idx + 30] if idx + 30 < n else None
            if future_close is None:
                continue
            fwd_ret = float(future_close / close.iloc[idx] - 1)
            if not np.isfinite(fwd_ret) or abs(fwd_ret) > 2.0:
                continue

            # Alpha 30d vs SPY
            alert_date = hist.index[idx]
            spy_alpha  = None
            try:
                spy_at  = spy_hist[spy_hist.index <= alert_date]["Close"]
                spy_fwd = spy_hist[(spy_hist.index > alert_date) &
                                   (spy_hist.index <= alert_date + pd.Timedelta(days=35))]["Close"]
                if not spy_at.empty and not spy_fwd.empty:
                    spy_ret = float(spy_fwd.iloc[-1] / spy_at.iloc[-1] - 1)
                    spy_alpha = round(fwd_ret - spy_ret, 4)
            except Exception:
                pass

            rows.append({
                "ticker":            ticker,
                "date":              alert_date.date().isoformat(),
                "sector":            sector,
                **feats,
                "forward_return_30d": round(fwd_ret, 4),
                "alpha_30d":          spy_alpha,
            })

        if (i + 1) % 50 == 0 or (i + 1) == n_tickers:
            log.info(f"  [{i+1}/{n_tickers}] {ticker} | {len(rows)} amostras momentum até agora")
        time.sleep(0.05)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Constrói dataset de treino para Breakout ML")
    p.add_argument("--tickers", help="Lista de tickers separados por vírgula (teste rápido)")
    p.add_argument("--start",   default="2019-01-01", help="Data de início (YYYY-MM-DD)")
    p.add_argument("--out",     default=str(_OUTPUT),  help="Caminho de output do parquet")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
    )

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
        log.info(f"Modo teste: {len(tickers)} tickers")
    else:
        from universe import get_ml_universe
        tickers = get_ml_universe()
        log.info(f"Universo completo: {len(tickers)} tickers")

    log.info("A descarregar SPY como benchmark...")
    spy_hist = _fetch_history("SPY", start=args.start)
    if spy_hist is None:
        log.error("SPY não disponível — abortar")
        return 1

    out_path = Path(args.out)

    # Modo incremental: nunca apagar dados históricos.
    # Se o parquet já existe, só computar datas novas (após o último registo).
    existing_df = None
    incremental_start = args.start
    if out_path.exists() and not args.tickers:  # modo teste não é incremental
        try:
            existing_df = pd.read_parquet(out_path)
            if "date" in existing_df.columns and not existing_df.empty:
                last_date = existing_df["date"].max()
                # Recuar 35 dias para garantir forward_return_30d dos últimos registos
                cutoff = (pd.Timestamp(last_date) - pd.Timedelta(days=35)).strftime("%Y-%m-%d")
                incremental_start = cutoff
                log.info(f"Dataset existente: {len(existing_df)} amostras até {last_date}")
                log.info(f"Modo incremental: a computar desde {incremental_start}")
        except Exception as e:
            log.warning(f"Parquet existente ilegível ({e}) — a reconstruir de raiz")
            existing_df = None

    log.info(f"A construir dataset de momentum desde {incremental_start}...")
    new_df = build_dataset(tickers, spy_hist, start=incremental_start)

    if new_df.empty:
        if existing_df is not None and not existing_df.empty:
            log.info("Nenhum dado novo — dataset existente mantido sem alterações")
            return 0
        log.error("Dataset vazio — nenhum dia de momentum encontrado")
        return 1

    # Merge: dados existentes + novos, dedup por (ticker, date)
    if existing_df is not None and not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
        combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
        n_new = len(combined) - len(existing_df)
        log.info(f"Merge: {len(existing_df)} existentes + {len(new_df)} novos → {len(combined)} total ({n_new} adicionados)")
    else:
        combined = new_df
        log.info(f"Dataset novo: {len(combined)} amostras")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)

    log.info(f"Dataset: {len(combined)} amostras | {combined['ticker'].nunique()} tickers | {combined['sector'].nunique()} sectores")
    log.info(f"Forward return 30d: mean={combined['forward_return_30d'].mean():.2%} std={combined['forward_return_30d'].std():.2%}")
    if "alpha_30d" in combined.columns:
        valid_alpha = combined["alpha_30d"].dropna()
        log.info(f"Alpha 30d vs SPY: mean={valid_alpha.mean():.2%} | positivos: {(valid_alpha>0).mean():.0%}")
    log.info(f"Guardado em: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
