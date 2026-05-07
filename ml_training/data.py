"""Construção do dataset v3.2 — extraído do notebook."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml_training.config import DEFAULT_ETF, HORIZON_DAYS, SECTOR_ETF

log = logging.getLogger(__name__)

MACRO_TICKERS: list[str] = ["^VIX", "SPY", "^TNX", "^IRX", "HYG", "LQD", "IYT", "XLI"]


# ─────────────────────────────────────────────────────────────────────────────
# load_base_dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_base_dataset(parquet_path: Path) -> pd.DataFrame:
    if not Path(parquet_path).exists():
        raise FileNotFoundError(f"parquet base não encontrado: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if "alert_date" not in df.columns:
        raise KeyError(f"parquet sem coluna alert_date: {parquet_path}")
    df["alert_date"] = pd.to_datetime(df["alert_date"])
    df = df.sort_values("alert_date").reset_index(drop=True)
    if "symbol" in df.columns and "ticker" not in df.columns:
        df = df.rename(columns={"symbol": "ticker"})
    if "ticker" not in df.columns:
        raise KeyError(f"parquet sem coluna ticker/symbol: {parquet_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Helpers inline (antes em experiments.ml_v2.pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def _build_targets(
    alert_date: pd.Timestamp,
    entry_price: float,
    future_closes: pd.Series,
    horizon: int = HORIZON_DAYS,
) -> dict:
    """Calcula max_return_60d e max_drawdown_60d a partir do preço de entrada."""
    if entry_price <= 0 or len(future_closes) < 5:
        return {"max_return_60d": float("nan"), "max_drawdown_60d": float("nan")}
    fwd = future_closes[
        (future_closes.index > alert_date)
        & (future_closes.index <= alert_date + pd.Timedelta(days=horizon))
    ] if isinstance(future_closes.index, pd.DatetimeIndex) else future_closes
    if len(fwd) < 5:
        return {"max_return_60d": float("nan"), "max_drawdown_60d": float("nan")}
    return {
        "max_return_60d":   float(fwd.max() / entry_price - 1.0),
        "max_drawdown_60d": float(fwd.min() / entry_price - 1.0),
    }


def _build_v2_features(row: pd.Series, hist: pd.DataFrame) -> dict:
    """Features price-based que o parquet v1 não tem.
    Replica build_v2_features() que estava em experiments.ml_v2.pipeline.
    Serve de fallback para campos que porventura faltem no row.
    """
    from ml_features import _FALLBACK
    if hist.empty:
        return {}
    close = hist["Close"]
    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else last_close
    high_52w = float(hist["High"].iloc[-252:].max()) if len(hist) >= 5 else last_close

    drop_today   = (last_close / prev_close - 1.0) if prev_close > 0 else 0.0
    drawdown_52w = (last_close / high_52w - 1.0)   if high_52w   > 0 else 0.0

    # RSI-14
    delta = close.diff().dropna()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi_s = (100 - 100 / (1 + rs)).iloc[-1]
    rsi   = float(rsi_s) if pd.notna(rsi_s) else float(_FALLBACK["rsi_14"])

    return {
        "drop_pct_today": round(drop_today * 100, 4),
        "drawdown_52w":   round(drawdown_52w * 100, 4),
        "rsi_14":         rsi,
    }


# ─────────────────────────────────────────────────────────────────────────────
# compute_sector_alert_count_7d
# ─────────────────────────────────────────────────────────────────────────────

def compute_sector_alert_count_7d(
    df: pd.DataFrame,
) -> dict[tuple[str, pd.Timestamp], int]:
    if df.empty:
        return {}
    seq = df[["alert_date", "sector", "ticker"]].copy()
    seq["alert_date"] = pd.to_datetime(seq["alert_date"])
    seq = seq.sort_values(["sector", "alert_date"]).reset_index(drop=True)
    seq["sector_alert_count_7d"] = 0
    for _sec, sub in seq.groupby("sector", sort=False):
        dates_arr = sub["alert_date"].to_numpy()
        counts = np.zeros(len(sub), dtype=np.int32)
        for i in range(len(sub)):
            win_start = dates_arr[i] - np.timedelta64(7, "D")
            prior = dates_arr[:i]
            counts[i] = int(((prior >= win_start) & (prior < dates_arr[i])).sum())
        seq.loc[sub.index, "sector_alert_count_7d"] = counts
    lookup: dict[tuple[str, pd.Timestamp], int] = {}
    for _, r in seq.iterrows():
        lookup[(r["ticker"], pd.Timestamp(r["alert_date"]))] = int(r["sector_alert_count_7d"])
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# spy_close_return_forward / days_since_52w_high
# ─────────────────────────────────────────────────────────────────────────────

def spy_close_return_forward(
    spy_hist: Optional[pd.DataFrame],
    alert_date: pd.Timestamp,
    horizon: int = HORIZON_DAYS,
) -> float:
    """Retorno close-to-close do SPY no horizonte forward — consistente com close_60d do ticker."""
    if spy_hist is None:
        return float("nan")
    entry_slice = spy_hist[spy_hist.index <= alert_date]
    if len(entry_slice) == 0:
        return float("nan")
    spy_entry = float(entry_slice["Close"].iloc[-1])
    if spy_entry <= 0:
        return float("nan")
    fwd = spy_hist[
        (spy_hist.index > alert_date)
        & (spy_hist.index <= alert_date + pd.Timedelta(days=horizon))
    ]
    if len(fwd) < 5:
        return float("nan")
    return float(fwd["Close"].iloc[-1] / spy_entry - 1.0)


# Mantido por retrocompatibilidade — usar spy_close_return_forward em código novo
def spy_max_return_forward(
    spy_hist: Optional[pd.DataFrame],
    alert_date: pd.Timestamp,
    horizon: int = HORIZON_DAYS,
) -> float:
    """Deprecated: usava .max() em vez de close-to-close. Usar spy_close_return_forward."""
    return spy_close_return_forward(spy_hist, alert_date, horizon)


def days_since_52w_high(hist: pd.DataFrame, alert_date: pd.Timestamp) -> float:
    from ml_features import _FALLBACK
    window = hist[
        (hist.index <= alert_date)
        & (hist.index > alert_date - pd.Timedelta(days=365))
    ]
    if len(window) < 20:
        return float(_FALLBACK.get("days_since_52w_high", 60.0))
    high_idx = window["High"].idxmax()
    return float((alert_date - high_idx).days)


# ─────────────────────────────────────────────────────────────────────────────
# generate_historical_alerts  (Notebook Célula 3)
# ─────────────────────────────────────────────────────────────────────────────

def generate_historical_alerts(
    all_tickers: list[str],
    price_cache: dict[str, pd.DataFrame],
    sector_fn,
    dip_threshold: float = -0.05,
    min_history_days: int = 252,
    subsample_years: Optional[list[int]] = None,
    max_per_year: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Gera alertas históricos de dip a partir do price_cache.

    Para cada ticker e cada dia de troca com queda >= dip_threshold (e.g. -5%),
    cria uma linha de alerta com: ticker, alert_date, sector, drop_pct_today.

    Parâmetros
    ----------
    all_tickers       : lista de tickers a processar
    price_cache       : {ticker: OHLCV DataFrame} com index DatetimeIndex
    sector_fn         : callable(ticker) -> str  (get_ticker_sector do notebook)
    dip_threshold     : queda mínima para gerar alerta (default -5%)
    min_history_days  : mínimo de candles antes do alerta para ser válido
    subsample_years   : lista de anos a considerar (None = todos)
    max_per_year      : máximo de alertas por ano após subsample (None = sem limite)
    seed              : seed para reproducibilidade do subsample

    Devolve DataFrame com colunas: ticker, alert_date, sector, drop_pct_today
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    for ticker in all_tickers:
        ohlcv = price_cache.get(ticker)
        if ohlcv is None or len(ohlcv) < min_history_days + 1:
            continue
        close = ohlcv["Close"].dropna()
        if len(close) < min_history_days + 1:
            continue
        pct_change = close.pct_change()
        dip_dates = pct_change[pct_change <= dip_threshold].index
        sector = sector_fn(ticker)
        for dt in dip_dates:
            hist_before = ohlcv[ohlcv.index <= dt]
            if len(hist_before) < min_history_days:
                continue
            records.append({
                "ticker":         ticker,
                "alert_date":     pd.Timestamp(dt),
                "sector":         sector,
                "drop_pct_today": round(float(pct_change.loc[dt]) * 100, 4),
            })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["alert_date"] = pd.to_datetime(df["alert_date"])
    df["year"] = df["alert_date"].dt.year

    if subsample_years is not None:
        df = df[df["year"].isin(subsample_years)].copy()

    if max_per_year is not None and max_per_year > 0:
        parts = []
        for _yr, grp in df.groupby("year"):
            if len(grp) > max_per_year:
                parts.append(grp.sample(n=max_per_year, random_state=int(rng.integers(0, 9999))))
            else:
                parts.append(grp)
        df = pd.concat(parts, ignore_index=True)

    df = df.drop(columns=["year"]).sort_values("alert_date").reset_index(drop=True)
    log.info(
        f"[data] generate_historical_alerts: {len(df)} alertas | "
        f"tickers={df['ticker'].nunique()} | "
        f"período {df['alert_date'].min().date()} → {df['alert_date'].max().date()}"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# build_dataset_v31
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset_v31(
    base_df: pd.DataFrame,
    price_cache: dict[str, pd.DataFrame],
    etf_cache: dict[str, pd.DataFrame],
    feature_cols_v31: list[str],
    horizon_days: int = HORIZON_DAYS,
    macro_price_cache: Optional[dict[str, pd.DataFrame]] = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Constrói dataset v3.2 linha-a-linha."""
    from ml_features import (
        FEATURE_COLUMNS,
        _FALLBACK,
        add_derived_features,
        add_momentum_features,
        add_context_features,
    )
    from macro_data import get_macro_context_historical

    sector_count_lookup = compute_sector_alert_count_7d(base_df)
    spy_hist = etf_cache.get(DEFAULT_ETF)

    combined_macro_cache: dict[str, pd.DataFrame] = {}
    if macro_price_cache:
        combined_macro_cache.update(macro_price_cache)
    combined_macro_cache.update(etf_cache)

    rows_v31: list[dict] = []
    skipped = {"no_price": 0, "short_history": 0, "no_target": 0, "no_spy_target": 0}

    for _, row in base_df.iterrows():
        ticker     = row["ticker"]
        alert_date = pd.Timestamp(row["alert_date"])
        sector     = row.get("sector", "Unknown") or "Unknown"
        etf        = SECTOR_ETF.get(sector, DEFAULT_ETF)

        ohlcv = price_cache.get(ticker)
        if ohlcv is None:
            skipped["no_price"] += 1
            continue

        hist = ohlcv[ohlcv.index <= alert_date]
        if len(hist) < 25:
            skipped["short_history"] += 1
            continue

        # Stage 0: Macro point-in-time
        macro_ctx = get_macro_context_historical(
            as_of_date=alert_date,
            sector=sector,
            macro_price_cache=combined_macro_cache if combined_macro_cache else None,
        )

        # Feature vector com fallbacks
        fv: dict[str, float] = {}
        for c in FEATURE_COLUMNS:
            v = row.get(c) if c in row.index else None
            fv[c] = float(v) if (v is not None and pd.notna(v)) else _FALLBACK.get(c, 0.0)

        fv["macro_score"]        = float(macro_ctx["macro_score"])
        fv["vix"]                = float(macro_ctx["vix"])
        fv["spy_drawdown_5d"]    = float(macro_ctx["spy_drawdown_5d"])
        fv["sector_drawdown_5d"] = float(macro_ctx["sector_drawdown_5d"])

        # Features price-based (fallback para campos em falta no parquet)
        price_feats = _build_v2_features(row, hist)
        for k, v in price_feats.items():
            if not pd.notna(fv.get(k)) or fv.get(k) == _FALLBACK.get(k, 0.0):
                fv[k] = v

        add_derived_features(fv)

        sec_hist  = etf_cache.get(etf)
        sec_slice = sec_hist[sec_hist.index <= alert_date] if sec_hist is not None else None
        spy_slice = spy_hist[spy_hist.index <= alert_date] if spy_hist is not None else None
        add_momentum_features(fv, hist, sec_slice, spy_slice)

        add_context_features(
            fv,
            price_history=hist,
            sector_alert_count_7d=float(
                sector_count_lookup.get((ticker, alert_date), 0)
            ),
        )

        # Targets
        if "max_return_60d" in row.index and pd.notna(row.get("max_return_60d")):
            max_ret  = float(row["max_return_60d"])
            max_draw = float(row.get("max_drawdown_60d", 0.0))
        else:
            entry_price = float(row.get("price", 0.0))
            if entry_price <= 0:
                entry_price = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
            if entry_price <= 0:
                skipped["no_target"] += 1
                continue
            future_closes = ohlcv[
                (ohlcv.index > alert_date)
                & (ohlcv.index <= alert_date + pd.Timedelta(days=horizon_days))
            ]["Close"]
            tgt = _build_targets(alert_date, entry_price, future_closes, horizon_days)
            if math.isnan(tgt["max_return_60d"]):
                skipped["no_target"] += 1
                continue
            max_ret  = tgt["max_return_60d"]
            max_draw = tgt["max_drawdown_60d"]

        # close_60d: close-to-close do ticker
        future_close_slice = ohlcv[
            (ohlcv.index > alert_date)
            & (ohlcv.index <= alert_date + pd.Timedelta(days=horizon_days))
        ]["Close"]
        entry_px = float(hist["Close"].iloc[-1])
        if len(future_close_slice) >= 5 and entry_px > 0:
            close_60d = float(future_close_slice.iloc[-1] / entry_px - 1.0)
        else:
            close_60d = max_ret

        # spy_close_60d: close-to-close do SPY (consistente com close_60d)
        spy_close_60d = spy_close_return_forward(spy_hist, alert_date, horizon_days)
        if math.isnan(spy_close_60d):
            skipped["no_spy_target"] += 1
            continue

        alpha_60d = close_60d - spy_close_60d

        rec = {
            "ticker":     ticker,
            "alert_date": alert_date,
            "sector":     sector,
            **{c: fv[c] for c in feature_cols_v31 if c in fv},
            "max_return_60d":   max_ret,
            "max_drawdown_60d": max_draw,
            "close_60d":        close_60d,
            "spy_close_60d":    spy_close_60d,
            "alpha_60d":        alpha_60d,
        }
        rows_v31.append(rec)

    df_out = pd.DataFrame(rows_v31)
    log.info(f"[data] dataset v3.2: shape={df_out.shape} | skipped={skipped}")
    return df_out, skipped
