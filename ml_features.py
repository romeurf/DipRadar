"""
ml_features.py — Unified Feature Store for DipRadar ML Pipeline.

Builds the complete feature vector for a single stock at a given moment.
Used identically during training (with labels) and inference in production (labels=None).

Architecture: 4-stage pipeline
  Stage 0 — Macro:    macro regime, VIX, SPY/sector drawdown, FRED recession prob
  Stage 1 — Quality:  gross margin, D/E, P/E vs sector fair
                       analyst upside, ROIC — sourced from score.py hemispheres
  Stage 2 — Timing:   drop today, drawdown 52w, RSI, ATR ratio, volume spike
  Stage 3 — Engineered: non-linear interactions (rsi_oversold_strength, etc.)
  Stage 3b — Momentum (v4.0): multi-window momentum (1m, 3m, 6m, 12m),
                        sector-relative (3m and 6m), beta_60d, vol_of_vol
  Stage 3c — Dislocation: quality_dislocation, peg_implicit, relative_drop,
                        month_of_year
  Stage 3d — Context: sector_alert_count_7d, days_since_52w_high
  Stage 3e — Short/Earnings: short_interest_ratio, earnings_surprise_avg
  Stage 3f — Regime:  vix_percentile_1y, spy_rsi_14, yield_10y_change_5d

v4.0 changes (2026-05-08):
  - Target cross-sectional rank computed in data.py (alpha_60d_rank).
  - Momentum expanded: return_6m_pre, return_12m_pre added.
  - sector_relative_6m added (stock 6m minus sector 6m).
  - vol_of_vol added: 20-day rolling std of 5-day realised vol (last 60 bars).
  - All fallbacks updated.

NaN contract:
  Every feature has an explicit fallback. No raw NaN reaches the model.

Public API:
  build_features(ticker, fundamentals, price_history, sector) -> dict
  FEATURE_COLUMNS: list[str]   ordered feature names (model input columns)
  LABEL_COLUMNS:   list[str]   label names
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from macro_data import get_macro_context
from score import score_from_fundamentals, _safe_float

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────────
# Feature & label column order (the model sees columns in this exact order)
# ────────────────────────────────────────────────────────────────────────────────

FEATURE_COLUMNS: list[str] = [
    # ── Stage 0: Macro (4 features) ──────────────────────────────────
    "macro_score",          # int 0–4 (BEAR=0-1, NEUTRAL=2, BULL=3-4)
    "vix",                  # VIX level (e.g. 18.5)
    "spy_drawdown_5d",      # SPY % change last 5 trading days (negative=down)
    "sector_drawdown_5d",   # sector ETF % change last 5 trading days

    # ── Stage 1: Quality / Value (5 features) ───────────────────────
    "gross_margin",         # Gross margin ratio (e.g. 0.65)
    "de_ratio",             # Debt-to-Equity ratio (yfinance scale, e.g. 45)
    "pe_vs_fair",           # P/E actual / sector fair P/E (< 1 = cheap)
    "analyst_upside",       # consensus analyst upside (e.g. 0.25 = 25%)
    "quality_score",        # score.py quality hemisphere [0, 1]

    # ── Stage 2: Timing (5 features) ───────────────────────────────
    "drop_pct_today",       # % price drop that triggered the alert (e.g. -8.4)
    "drawdown_52w",         # % below 52-week high (e.g. -23.1)
    "rsi_14",               # RSI 14-period [0, 100]
    "atr_ratio",            # ATR(14) / current price — normalised volatility
    "volume_spike",         # today's volume / 20-day avg volume (e.g. 2.1)

    # ── Stage 3: Engineered / Non-linear interactions (5 features) ──
    "rsi_oversold_strength", # max(0, 40 - rsi_14)
    "vix_regime",            # 0=low (<15), 1=medium (15-25), 2=high (>25)
    "pe_attractive",         # max(0, 1 - pe_vs_fair)
    "drop_x_drawdown",       # drop_pct_today * drawdown_52w / 100
    "vol_x_drop",            # volume_spike * abs(drop_pct_today)

    # ── Stage 3b: Momentum (v4.0 — 8 features) ──────────────────────
    # Multi-window momentum + sector-relative (6m added) + vol-of-vol.
    # Literature: momentum at multiple horizons reduces single-window noise.
    "return_1m",            # % return 21 trading days before alert
    "return_3m_pre",        # % return 63 trading days before alert
    "return_6m_pre",        # % return 126 trading days before alert  [NEW v4.0]
    "return_12m_pre",       # % return 252 trading days before alert  [NEW v4.0]
    "sector_relative",      # return_3m_pre minus sector 3m return
    "sector_relative_6m",   # return_6m_pre minus sector 6m return   [NEW v4.0]
    "beta_60d",             # rolling beta vs SPY over 60 trading days
    "vol_of_vol",           # std of 5-day realised vol over last 60 bars  [NEW v4.0]

    # ── Stage 3c: Dislocation (4 features) ───────────────────────────
    "quality_dislocation",  # gross_margin * |drawdown_52w| / 100
    "peg_implicit",         # pe_vs_fair / (revenue_growth * 100), clip [0, 5]
    "relative_drop",        # drop_pct_today - sector_drawdown_5d
    "month_of_year",        # mês do alerta (1–12)

    # ── Stage 3d: Context (2 features) — v3.2 ────────────────────────
    "sector_alert_count_7d", # nº alertas no mesmo sector nos últimos 7 dias
    "days_since_52w_high",   # dias desde o máximo de 52 semanas

    # ── Stage 3e: Short Interest + Earnings Surprise (v3.3) ──────────
    "short_interest_ratio",   # dias para cobrir o short, clip [0, 30]
    "earnings_surprise_avg",  # média dos últimos 2 EPS surprises (%), clip [-50, 50]

    # ── Stage 3f: Regime (3 features) — v3.4 ────────────────────────
    "vix_percentile_1y",    # VIX rank nos últimos 252 sessões [0, 1]
    "spy_rsi_14",           # SPY RSI-14 na alert_date [0, 100]
    "yield_10y_change_5d",  # 5-day change in 10Y US Treasury yield (^TNX, %)
]

LABEL_COLUMNS: list[str] = [
    "label_win",            # int: 1 if recovery >=15% within 60d (Model A target)
    "label_further_drop",   # float: max additional % drop before recovery (Model B target)
]

# Total feature count
N_FEATURES = len(FEATURE_COLUMNS)  # 41


# ────────────────────────────────────────────────────────────────────────────────
# Sector fair P/E
# ────────────────────────────────────────────────────────────────────────────────

_SECTOR_FAIR_PE: dict[str, float] = {
    "Technology":             35.0,
    "Healthcare":             22.0,
    "Communication Services": 22.0,
    "Financial Services":     13.0,
    "Financials":             13.0,
    "Consumer Cyclical":      20.0,
    "Consumer Defensive":     22.0,
    "Industrials":            20.0,
    "Energy":                 12.0,
    "Utilities":              18.0,
    "Real Estate":            40.0,
    "Basic Materials":        14.0,
    "Materials":              14.0,
}

# Global median fallbacks (used when a feature cannot be computed)
_FALLBACK: dict[str, float] = {
    "macro_score":        2.0,
    "vix":               20.0,
    "spy_drawdown_5d":    0.0,
    "sector_drawdown_5d": 0.0,
    "fcf_yield":          0.04,
    "revenue_growth":     0.05,
    "gross_margin":       0.35,
    "de_ratio":          80.0,
    "pe_vs_fair":         1.0,
    "analyst_upside":     0.10,
    "quality_score":      0.50,
    "drop_pct_today":    -8.0,
    "drawdown_52w":      -15.0,
    "rsi_14":            40.0,
    "atr_ratio":          0.02,
    "volume_spike":       1.0,
    # Stage 3 engineered
    "rsi_oversold_strength": 0.0,
    "vix_regime":            1.0,
    "pe_attractive":         0.0,
    "drop_x_drawdown":       1.2,
    "vol_x_drop":            8.0,
    # Stage 3b momentum (v4.0)
    "return_1m":          0.0,
    "return_3m_pre":      0.0,
    "return_6m_pre":      0.0,
    "return_12m_pre":     0.0,
    "sector_relative":    0.0,
    "sector_relative_6m": 0.0,
    "beta_60d":           1.0,
    "vol_of_vol":         0.01,  # ~1% daily vol-of-vol as neutral baseline
    # Stage 3c dislocation
    "quality_dislocation": 0.08,
    "peg_implicit":        2.0,
    "relative_drop":       0.0,
    "month_of_year":       6.0,
    # Stage 3d context (v3.2)
    "sector_alert_count_7d": 0.0,
    "days_since_52w_high":  180.0,
    # Stage 3e short/earnings (v3.3)
    "short_interest_ratio":  3.5,
    "earnings_surprise_avg": 0.0,
    # Stage 3f regime (v3.4)
    "vix_percentile_1y":    0.5,
    "spy_rsi_14":           50.0,
    "yield_10y_change_5d":   0.0,
}


# ────────────────────────────────────────────────────────────────────────────────
# Timezone normalisation helper
# ────────────────────────────────────────────────────────────────────────────────

def _tz_normalize(ts: Any) -> pd.Timestamp:
    """Return a tz-naive UTC pd.Timestamp from any datetime-like input."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a tz-naive DatetimeIndex."""
    if df is None:
        return df
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df


# ────────────────────────────────────────────────────────────────────────────────
# Stage 3 — Engineered features
# ────────────────────────────────────────────────────────────────────────────────

def add_derived_features(features: dict, alert_date: Optional[Any] = None) -> dict:
    """
    Compute Stage-3 engineered features + Stage-3c dislocation features.
    Mutates and returns the same dict.
    """
    rsi    = float(features.get("rsi_14",         _FALLBACK["rsi_14"]))
    vix    = float(features.get("vix",            _FALLBACK["vix"]))
    pe_vf  = float(features.get("pe_vs_fair",     _FALLBACK["pe_vs_fair"]))
    drop   = float(features.get("drop_pct_today", _FALLBACK["drop_pct_today"]))
    dd52   = float(features.get("drawdown_52w",   _FALLBACK["drawdown_52w"]))
    volsp  = float(features.get("volume_spike",   _FALLBACK["volume_spike"]))

    # ── Stage 3: non-linear interactions ─────────────────────────────
    features["rsi_oversold_strength"] = round(max(0.0, 40.0 - rsi), 4)
    features["vix_regime"] = 0.0 if vix < 15.0 else (1.0 if vix < 25.0 else 2.0)
    features["pe_attractive"]   = round(max(0.0, 1.0 - pe_vf), 4)
    features["drop_x_drawdown"] = round(drop * dd52 / 100.0, 4)
    features["vol_x_drop"]      = round(volsp * abs(drop), 4)

    # ── Stage 3c: dislocation features ──────────────────────────────
    gm  = float(features.get("gross_margin",   _FALLBACK["gross_margin"]))
    fcf = float(features.get("fcf_yield",       _FALLBACK["fcf_yield"]))
    rg  = float(features.get("revenue_growth",  _FALLBACK["revenue_growth"]))

    features["quality_dislocation"] = (
        round(gm * abs(dd52) / 100.0, 4) if fcf >= 0 else 0.0
    )

    if rg > 0 and pe_vf > 0:
        features["peg_implicit"] = round(min(pe_vf / (rg * 100.0), 5.0), 4)
    else:
        features["peg_implicit"] = 3.0

    sec_dd = float(features.get("sector_drawdown_5d", _FALLBACK["sector_drawdown_5d"]))
    features["relative_drop"] = round(drop - sec_dd, 4)

    if alert_date is not None:
        try:
            features["month_of_year"] = float(pd.Timestamp(alert_date).month)
        except Exception:
            features["month_of_year"] = float(datetime.now().month)
    else:
        features["month_of_year"] = float(datetime.now().month)

    return features


# ────────────────────────────────────────────────────────────────────────────────
# Stage 3d — Context features (v3.2)
# ────────────────────────────────────────────────────────────────────────────────

def add_context_features(
    features: dict,
    price_history: Optional[pd.DataFrame] = None,
    sector_alert_count_7d: Optional[float] = None,
) -> dict:
    """Stage-3d context features (v3.2)."""
    if sector_alert_count_7d is not None:
        v = float(sector_alert_count_7d)
        features["sector_alert_count_7d"] = v if math.isfinite(v) else _FALLBACK["sector_alert_count_7d"]
    else:
        features["sector_alert_count_7d"] = _FALLBACK["sector_alert_count_7d"]

    if price_history is not None and "Close" in price_history.columns:
        try:
            closes = price_history["Close"].dropna()
            lookback = closes.tail(252)
            if len(lookback) >= 5:
                idx_max  = lookback.values.argmax()
                days_ago = len(lookback) - 1 - idx_max
                cal_days = round(days_ago * 1.4)
                features["days_since_52w_high"] = float(max(0, cal_days))
            else:
                features["days_since_52w_high"] = _FALLBACK["days_since_52w_high"]
        except Exception as e:
            logger.debug(f"add_context_features: days_since_52w_high failed: {e}")
            features["days_since_52w_high"] = _FALLBACK["days_since_52w_high"]
    else:
        features["days_since_52w_high"] = _FALLBACK["days_since_52w_high"]

    return features


# ────────────────────────────────────────────────────────────────────────────────
# Stage 3e — Short Interest + Earnings Surprise features (v3.3)
# ────────────────────────────────────────────────────────────────────────────────

def add_short_earnings_features(
    features: dict,
    ticker_info: Optional[dict] = None,
) -> dict:
    """Stage-3e: short_interest_ratio + earnings_surprise_avg."""
    info = ticker_info or {}

    sr = _safe_float(info.get("shortRatio"))
    if math.isfinite(sr) and sr >= 0:
        features["short_interest_ratio"] = float(min(sr, 30.0))
    else:
        features["short_interest_ratio"] = _FALLBACK["short_interest_ratio"]

    try:
        hist = info.get("earningsHistory", {})
        if isinstance(hist, dict):
            hist = hist.get("history", [])
        surprises = []
        for entry in (hist or [])[:4]:
            sp = _safe_float(entry.get("surprisePercent") if isinstance(entry, dict) else None)
            if math.isfinite(sp):
                surprises.append(sp * 100.0)
        if surprises:
            avg = float(np.mean(surprises[:2]))
            features["earnings_surprise_avg"] = float(np.clip(avg, -50.0, 50.0))
        else:
            features["earnings_surprise_avg"] = _FALLBACK["earnings_surprise_avg"]
    except Exception as e:
        logger.debug(f"add_short_earnings_features: earningsHistory failed: {e}")
        features["earnings_surprise_avg"] = _FALLBACK["earnings_surprise_avg"]

    return features


# ────────────────────────────────────────────────────────────────────────────────
# Stage 3f — Regime features (v3.4)
# ────────────────────────────────────────────────────────────────────────────────

def add_regime_features(
    features: dict,
    spy_history: Optional[pd.DataFrame],
    tnx_history: Optional[pd.DataFrame],
    alert_date: Any,
    vix_history: Optional[pd.DataFrame] = None,
) -> dict:
    """Stage-3f: point-in-time market regime signals (v3.4)."""
    alert_ts = _tz_normalize(alert_date)
    vix_hist = _normalize_index(vix_history)
    spy_hist = _normalize_index(spy_history)
    tnx_hist = _normalize_index(tnx_history)

    # vix_percentile_1y
    try:
        vix_val = float(features.get("vix", _FALLBACK["vix"]))
        pct = _FALLBACK["vix_percentile_1y"]
        if vix_hist is not None and "Close" in vix_hist.columns:
            vix_slice = vix_hist[vix_hist.index <= alert_ts]
            window = vix_slice["Close"].dropna().tail(252)
            if len(window) >= 20:
                arr  = window.values
                rank = float(np.sum(arr <= vix_val)) / len(arr)
                pct  = float(np.clip(rank, 0.0, 1.0))
        elif spy_hist is not None and "Close" in spy_hist.columns:
            rets = spy_hist[spy_hist.index <= alert_ts]["Close"].pct_change().dropna().tail(252)
            if len(rets) >= 20:
                rv_window = rets.rolling(5).std().dropna()
                if len(rv_window) >= 20:
                    cur_rv = float(rv_window.iloc[-1])
                    pct = float(np.clip(
                        np.sum(rv_window.values <= cur_rv) / len(rv_window), 0.0, 1.0
                    ))
        features["vix_percentile_1y"] = round(pct, 4)
    except Exception as e:
        logger.debug(f"add_regime_features: vix_percentile_1y failed: {e}")
        features["vix_percentile_1y"] = _FALLBACK["vix_percentile_1y"]

    # spy_rsi_14
    try:
        rsi_val = _FALLBACK["spy_rsi_14"]
        if spy_hist is not None and "Close" in spy_hist.columns:
            closes = spy_hist[spy_hist.index <= alert_ts]["Close"].dropna()
            if len(closes) >= 16:
                delta = closes.diff().dropna()
                gain  = delta.clip(lower=0).rolling(14).mean()
                loss  = (-delta.clip(upper=0)).rolling(14).mean()
                rs    = gain / loss.replace(0, np.nan)
                rsi_s = (100 - 100 / (1 + rs)).iloc[-1]
                if pd.notna(rsi_s):
                    rsi_val = float(np.clip(rsi_s, 0.0, 100.0))
        features["spy_rsi_14"] = round(rsi_val, 2)
    except Exception as e:
        logger.debug(f"add_regime_features: spy_rsi_14 failed: {e}")
        features["spy_rsi_14"] = _FALLBACK["spy_rsi_14"]

    # yield_10y_change_5d
    try:
        chg = _FALLBACK["yield_10y_change_5d"]
        if tnx_hist is not None and "Close" in tnx_hist.columns:
            tnx_slice = tnx_hist[tnx_hist.index <= alert_ts]["Close"].dropna()
            if len(tnx_slice) >= 6:
                chg = float(tnx_slice.iloc[-1] - tnx_slice.iloc[-6])
                if not math.isfinite(chg) or abs(chg) > 5.0:
                    chg = _FALLBACK["yield_10y_change_5d"]
        features["yield_10y_change_5d"] = round(chg, 4)
    except Exception as e:
        logger.debug(f"add_regime_features: yield_10y_change_5d failed: {e}")
        features["yield_10y_change_5d"] = _FALLBACK["yield_10y_change_5d"]

    return features


# ────────────────────────────────────────────────────────────────────────────────
# Stage 3b — Momentum features (v4.0 — multi-window + vol-of-vol)
# ────────────────────────────────────────────────────────────────────────────────

def _pct_return(prices: np.ndarray, lookback: int) -> float:
    """
    % return over the last `lookback` bars.
    Returns 0.0 if insufficient data or division by zero.
    Anti-leakage: uses prices[-1] as the last point BEFORE the alert.
    """
    if len(prices) < lookback + 1:
        return 0.0
    p_end   = float(prices[-1])
    p_start = float(prices[-lookback - 1])
    if p_start <= 0:
        return 0.0
    return round((p_end / p_start - 1.0) * 100.0, 4)


def _vol_of_vol(prices: np.ndarray, rv_window: int = 5, vov_window: int = 60) -> float:
    """
    Vol-of-vol: std of 5-day realised volatility over the last `vov_window` bars.
    Captures whether volatility itself is accelerating — a leading indicator of
    tail-risk and price instability that the single-window ATR misses.
    Returns value in the same scale as ATR ratio (daily %/price).
    """
    min_needed = rv_window + vov_window
    if len(prices) < min_needed:
        return _FALLBACK["vol_of_vol"]
    try:
        returns = np.diff(prices[-min_needed:]) / prices[-min_needed:-1]
        # rolling std of returns over rv_window
        rvs = np.array([
            float(np.std(returns[i:i + rv_window], ddof=1))
            for i in range(len(returns) - rv_window + 1)
        ])
        if len(rvs) < 10:
            return _FALLBACK["vol_of_vol"]
        vov = float(np.std(rvs[-vov_window:], ddof=1))
        return round(float(np.clip(vov, 0.0, 0.1)), 6)  # cap at 10%
    except Exception:
        return _FALLBACK["vol_of_vol"]


def add_momentum_features(
    features: dict,
    price_history: Optional[pd.DataFrame],
    sector_history: Optional[pd.DataFrame] = None,
    spy_history: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Stage-3b: pre-alert momentum features (v4.0).

    v4.0 additions:
      - return_6m_pre   (126-bar return)
      - return_12m_pre  (252-bar return)
      - sector_relative_6m (stock 6m minus sector 6m)
      - vol_of_vol      (std of 5-day realised vol over last 60 bars)

    All computed from price_history rows up to (and including) alert_date.
    The caller (data.py build_dataset_v31) slices to alert_date before
    calling — no leakage.
    """
    NEW_KEYS = [
        "return_1m", "return_3m_pre", "return_6m_pre", "return_12m_pre",
        "sector_relative", "sector_relative_6m", "beta_60d", "vol_of_vol",
    ]

    if price_history is None or price_history.empty:
        for k in NEW_KEYS:
            features.setdefault(k, _FALLBACK[k])
        return features

    closes = price_history["Close"].dropna().values

    features["return_1m"]     = _pct_return(closes, 21)
    features["return_3m_pre"] = _pct_return(closes, 63)
    features["return_6m_pre"] = _pct_return(closes, 126)
    features["return_12m_pre"] = _pct_return(closes, 252)

    # sector_relative: stock 3m return minus sector ETF 3m return
    if sector_history is not None and not sector_history.empty:
        sec_closes = sector_history["Close"].dropna().values
        sec_ret_3m = _pct_return(sec_closes, 63)
        sec_ret_6m = _pct_return(sec_closes, 126)
        features["sector_relative"]    = round(features["return_3m_pre"] - sec_ret_3m, 4)
        features["sector_relative_6m"] = round(features["return_6m_pre"] - sec_ret_6m, 4)
    else:
        features["sector_relative"]    = _FALLBACK["sector_relative"]
        features["sector_relative_6m"] = _FALLBACK["sector_relative_6m"]

    # beta_60d: rolling beta of stock vs SPY over 60 bars
    features["beta_60d"] = _FALLBACK["beta_60d"]
    if spy_history is not None and not spy_history.empty:
        try:
            spy_closes = spy_history["Close"].dropna().values
            n = 60
            if len(closes) >= n + 1 and len(spy_closes) >= n + 1:
                stock_rets = np.diff(closes[-n - 1:]) / closes[-n - 1:-1]
                spy_rets   = np.diff(spy_closes[-n - 1:]) / spy_closes[-n - 1:-1]
                min_len    = min(len(stock_rets), len(spy_rets))
                stock_rets = stock_rets[-min_len:]
                spy_rets   = spy_rets[-min_len:]
                cov = np.cov(stock_rets, spy_rets)
                var_spy = float(cov[1, 1])
                if var_spy > 1e-10:
                    features["beta_60d"] = round(float(cov[0, 1]) / var_spy, 4)
        except Exception as e:
            logger.debug(f"add_momentum_features: beta_60d failed: {e}")

    # vol_of_vol: std of 5-day realised vol over last 60 bars
    features["vol_of_vol"] = _vol_of_vol(closes)

    return features
