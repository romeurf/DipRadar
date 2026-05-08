from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml_training.config import DEFAULT_ETF, FEATURE_COLS, HORIZON_DAYS, SECTOR_ETF

log = logging.getLogger(__name__)

MACRO_TICKERS: list[str] = ["^VIX", "SPY", "^TNX", "^IRX", "HYG", "LQD", "IYT", "XLI"]


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


def _build_targets(alert_date: pd.Timestamp, entry_price: float, future_closes: pd.Series, horizon: int = HORIZON_DAYS) -> dict:
    if entry_price <= 0 or len(future_closes) < 5:
        return {
            "max_return_60d": float("nan"),
            "max_drawdown_60d": float("nan"),
            "max_drawdown_20d": float("nan"),
            "close_90d": float("nan"),
        }
    if isinstance(future_closes.index, pd.DatetimeIndex):
        fwd_60 = future_closes[(future_closes.index > alert_date) & (future_closes.index <= alert_date + pd.Timedelta(days=horizon))]
        fwd_20 = future_closes[(future_closes.index > alert_date) & (future_closes.index <= alert_date + pd.Timedelta(days=20))]
        fwd_90 = future_closes[(future_closes.index > alert_date) & (future_closes.index <= alert_date + pd.Timedelta(days=90))]
    else:
        fwd_60 = future_closes
        fwd_20 = future_closes.iloc[:20]
        fwd_90 = future_closes.iloc[:90]
    if len(fwd_60) < 5:
        return {
            "max_return_60d": float("nan"),
            "max_drawdown_60d": float("nan"),
            "max_drawdown_20d": float("nan"),
            "close_90d": float("nan"),
        }
    return {
        "max_return_60d": float(fwd_60.max() / entry_price - 1.0),
        "max_drawdown_60d": float(fwd_60.min() / entry_price - 1.0),
        "max_drawdown_20d": float(fwd_20.min() / entry_price - 1.0) if len(fwd_20) >= 3 else float("nan"),
        "close_90d": float(fwd_90.iloc[-1] / entry_price - 1.0) if len(fwd_90) >= 5 else float("nan"),
    }


def _build_v2_features(row: pd.Series, hist: pd.DataFrame) -> dict:
    from ml_features import _FALLBACK
    if hist.empty:
        return {}
    close = hist["Close"]
    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else last_close
    high_52w = float(hist["High"].iloc[-252:].max()) if len(hist) >= 5 else last_close
    drop_today = (last_close / prev_close - 1.0) if prev_close > 0 else 0.0
    drawdown_52w = (last_close / high_52w - 1.0) if high_52w > 0 else 0.0
    delta = close.diff().dropna()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_s = (100 - 100 / (1 + rs)).iloc[-1]
    rsi = float(rsi_s) if pd.notna(rsi_s) else float(_FALLBACK["rsi_14"])
    tr1 = hist["High"] - hist["Low"]
    tr2 = (hist["High"] - hist["Close"].shift(1)).abs()
    tr3 = (hist["Low"] - hist["Close"].shift(1)).abs()
    atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean().iloc[-1]
    atr_ratio = float(atr / last_close) if pd.notna(atr) and last_close > 0 else float(_FALLBACK["atr_ratio"])
    vol_avg20 = hist["Volume"].tail(20).mean() if "Volume" in hist.columns and len(hist) >= 20 else np.nan
    volume_spike = float(hist["Volume"].iloc[-1] / vol_avg20) if pd.notna(vol_avg20) and vol_avg20 > 0 and "Volume" in hist.columns else float(_FALLBACK["volume_spike"])
    return {
        "drop_pct_today": round(drop_today * 100, 4),
        "drawdown_52w": round(drawdown_52w * 100, 4),
        "rsi_14": rsi,
        "atr_ratio": atr_ratio,
        "volume_spike": volume_spike,
    }


def compute_sector_alert_count_7d(df: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], int]:
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


def spy_close_return_forward(spy_hist: Optional[pd.DataFrame], alert_date: pd.Timestamp, horizon: int = HORIZON_DAYS) -> float:
    if spy_hist is None:
        return float("nan")
    entry_slice = spy_hist[spy_hist.index <= alert_date]
    if len(entry_slice) == 0:
        return float("nan")
    spy_entry = float(entry_slice["Close"].iloc[-1])
    if spy_entry <= 0:
        return float("nan")
    fwd = spy_hist[(spy_hist.index > alert_date) & (spy_hist.index <= alert_date + pd.Timedelta(days=horizon))]
    if len(fwd) < 5:
        return float("nan")
    return float(fwd["Close"].iloc[-1] / spy_entry - 1.0)


def add_cross_sectional_rank(df: pd.DataFrame, target_col: str = "alpha_60d", rank_col: str = "alpha_60d_rank", window_days: int = 20) -> pd.DataFrame:
    df = df.copy()
    df[rank_col] = float("nan")
    valid_mask = df[target_col].notna() & np.isfinite(df[target_col].astype(float))
    if not valid_mask.any():
        return df
    dates = df.loc[valid_mask, "alert_date"].values
    targets = df.loc[valid_mask, target_col].values.astype(float)
    ranks = np.empty(len(targets), dtype=float)
    window_td = np.timedelta64(window_days, "D")
    for i in range(len(dates)):
        d = dates[i]
        mask = (dates >= d - window_td) & (dates <= d + window_td)
        peers = targets[mask]
        finite_peers = peers[np.isfinite(peers)]
        ranks[i] = 0.5 if len(finite_peers) < 2 else float(np.sum(finite_peers < targets[i])) / len(finite_peers)
    df.loc[valid_mask, rank_col] = np.clip(ranks, 0.0, 1.0)
    return df


def build_dataset_v31(base_df: pd.DataFrame, price_cache: dict[str, pd.DataFrame], etf_cache: dict[str, pd.DataFrame], horizon_days: int = HORIZON_DAYS, macro_price_cache: Optional[dict[str, pd.DataFrame]] = None) -> tuple[pd.DataFrame, dict[str, int]]:
    from ml_features import FEATURE_COLUMNS, _FALLBACK, add_derived_features, add_momentum_features, add_context_features, add_regime_features, add_short_earnings_features
    from macro_data import get_macro_context_historical
    if FEATURE_COLS != FEATURE_COLUMNS:
        missing = set(FEATURE_COLUMNS) - set(FEATURE_COLS)
        extra = set(FEATURE_COLS) - set(FEATURE_COLUMNS)
        raise ValueError(f"FEATURE_COLS (config) != FEATURE_COLUMNS (ml_features)!\n  em ml_features mas não no config: {missing}\n  no config mas não em ml_features: {extra}")
    sector_count_lookup = compute_sector_alert_count_7d(base_df)
    spy_hist = etf_cache.get(DEFAULT_ETF)
    combined_macro_cache: dict[str, pd.DataFrame] = {}
    if macro_price_cache:
        combined_macro_cache.update(macro_price_cache)
    combined_macro_cache.update(etf_cache)
    rows_v31: list[dict] = []
    skipped = {"no_price": 0, "short_history": 0, "no_target": 0, "no_spy_target": 0}
    for _, row in base_df.iterrows():
        ticker = row["ticker"]
        alert_date = pd.Timestamp(row["alert_date"])
        sector = row.get("sector", "Unknown") or "Unknown"
        etf = SECTOR_ETF.get(sector, DEFAULT_ETF)
        ohlcv = price_cache.get(ticker)
        if ohlcv is None:
            skipped["no_price"] += 1
            continue
        hist = ohlcv[ohlcv.index <= alert_date]
        if len(hist) < 25:
            skipped["short_history"] += 1
            continue
        macro_ctx = get_macro_context_historical(as_of_date=alert_date, sector=sector, macro_price_cache=combined_macro_cache if combined_macro_cache else None)
        fv: dict[str, float] = {}
        for c in FEATURE_COLUMNS:
            v = row.get(c) if c in row.index else None
            fv[c] = float(v) if (v is not None and pd.notna(v)) else _FALLBACK.get(c, 0.0)
        fv["macro_score"] = float(macro_ctx["macro_score"])
        fv["vix"] = float(macro_ctx["vix"])
        fv["spy_drawdown_5d"] = float(macro_ctx["spy_drawdown_5d"])
        fv["sector_drawdown_5d"] = float(macro_ctx["sector_drawdown_5d"])
        price_feats = _build_v2_features(row, hist)
        for k, v in price_feats.items():
            if not pd.notna(fv.get(k)) or fv.get(k) == _FALLBACK.get(k, 0.0):
                fv[k] = v
        add_derived_features(fv, alert_date=alert_date)
        sec_hist = etf_cache.get(etf)
        sec_slice = sec_hist[sec_hist.index <= alert_date] if sec_hist is not None else None
        spy_slice = spy_hist[spy_hist.index <= alert_date] if spy_hist is not None else None
        add_momentum_features(fv, hist, sec_slice, spy_slice)
        add_context_features(fv, price_history=hist, sector_alert_count_7d=float(sector_count_lookup.get((ticker, alert_date), 0)))
        ticker_info: Optional[dict] = None
        short_ratio = row.get("shortRatio") if "shortRatio" in row.index else None
        earnings_hist = row.get("earningsHistory") if "earningsHistory" in row.index else None
        if short_ratio is not None or earnings_hist is not None:
            ticker_info = {}
            if short_ratio is not None:
                ticker_info["shortRatio"] = short_ratio
            if earnings_hist is not None:
                ticker_info["earningsHistory"] = earnings_hist
        add_short_earnings_features(fv, ticker_info, alert_date=alert_date)
        tnx_hist = combined_macro_cache.get("^TNX")
        vix_hist = combined_macro_cache.get("^VIX")
        add_regime_features(fv, spy_slice, tnx_hist, alert_date, vix_hist)
        entry_price = float(row.get("price", 0.0))
        if entry_price <= 0:
            entry_price = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
        if entry_price <= 0:
            skipped["no_target"] += 1
            continue
        future_closes = ohlcv[(ohlcv.index > alert_date) & (ohlcv.index <= alert_date + pd.Timedelta(days=90))]["Close"]
        tgt = _build_targets(alert_date, entry_price, future_closes, horizon_days)
        if math.isnan(tgt["max_return_60d"]):
            skipped["no_target"] += 1
            continue
        max_ret = tgt["max_return_60d"]
        max_draw = tgt["max_drawdown_60d"]
        max_draw_20d = tgt["max_drawdown_20d"]
        close_90d = tgt["close_90d"]
        future_close_slice = ohlcv[(ohlcv.index > alert_date) & (ohlcv.index <= alert_date + pd.Timedelta(days=horizon_days))]["Close"]
        entry_px = float(hist["Close"].iloc[-1])
        if len(future_close_slice) >= 5 and entry_px > 0:
            close_60d = float(future_close_slice.iloc[-1] / entry_px - 1.0)
            if not np.isfinite(close_60d) or abs(close_60d) > 2.0:
                skipped["no_target"] += 1
                continue
        else:
            close_60d = max_ret
        spy_close_60d = spy_close_return_forward(spy_hist, alert_date, horizon_days)
        spy_close_90d = spy_close_return_forward(spy_hist, alert_date, 90)
        if not np.isfinite(spy_close_60d) or abs(spy_close_60d) > 2.0:
            skipped["no_spy_target"] += 1
            continue
        alpha_60d = math.log1p(close_60d) - math.log1p(spy_close_60d) if (close_60d > -1.0 and spy_close_60d > -1.0) else close_60d - spy_close_60d
        alpha_90d = math.log1p(close_90d) - math.log1p(spy_close_90d) if (np.isfinite(close_90d) and np.isfinite(spy_close_90d) and close_90d > -1.0 and spy_close_90d > -1.0) else float("nan")
        rec = {
            "ticker": ticker,
            "alert_date": alert_date,
            "sector": sector,
            **{c: fv[c] for c in FEATURE_COLS if c in fv},
            "max_return_60d": max_ret,
            "max_drawdown_60d": max_draw,
            "max_drawdown_20d": max_draw_20d,
            "close_60d": close_60d,
            "close_90d": close_90d,
            "spy_close_60d": spy_close_60d,
            "spy_close_90d": spy_close_90d,
            "alpha_60d": alpha_60d,
            "alpha_90d": alpha_90d,
            "label_upside_10_90d": float(close_90d >= 0.10) if np.isfinite(close_90d) else float("nan"),
            "label_downside_15_20d": float(max_draw_20d <= -0.15) if np.isfinite(max_draw_20d) else float("nan"),
        }
        rows_v31.append(rec)
    df_out = pd.DataFrame(rows_v31)
    if not df_out.empty:
        df_out = add_cross_sectional_rank(df_out, target_col="alpha_60d", rank_col="alpha_60d_rank")
        if "alpha_90d" in df_out.columns:
            df_out = add_cross_sectional_rank(df_out, target_col="alpha_90d", rank_col="alpha_90d_rank")
    log.info(f"[data] dataset: shape={df_out.shape} | skipped={skipped}")
    return df_out, skipped
