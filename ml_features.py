"""
ml_features.py — Unified Feature Store for DipRadar ML Pipeline.

Builds the complete feature vector for a single stock at a given moment.
Used identically during training (with labels) and inference in production (labels=None).

Architecture: 4-stage pipeline
  Stage 0 — Macro:    macro regime, VIX, SPY/sector drawdown, FRED recession prob
  Stage 1 — Quality:  gross margin, D/E, P/E vs sector fair,
                       analyst upside, ROIC, FCF yield — sourced from score.py hemispheres
  Stage 2 — Timing:   drop today, drawdown 52w, RSI, ATR ratio, volume spike, BB width
  Stage 3 — Engineered: non-linear interactions (rsi_oversold_strength, etc.)
  Stage 3b — Momentum: multi-window momentum (1m, 3m, 6m, 12m),
                        sector-relative (3m and 6m), beta_60d, vol_of_vol
  Stage 3c — Dislocation: quality_dislocation, peg_implicit, relative_drop,
                        month_of_year
  Stage 3d — Context: sector_alert_count_7d, days_since_52w_high
  Stage 3e — Short/Earnings: short_interest_ratio, earnings_surprise_avg, earnings_distance_days
  Stage 3f — Regime:  vix_percentile_1y, spy_rsi_14, yield_10y_change_5d

NaN contract:
  Every feature has an explicit fallback. No raw NaN reaches the model.
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

FEATURE_COLUMNS: list[str] = [
    "macro_score",
    "vix",
    "spy_drawdown_5d",
    "sector_drawdown_5d",
    "gross_margin",
    "de_ratio",
    "pe_vs_fair",
    "analyst_upside",
    "quality_score",
    "drop_pct_today",
    "drawdown_52w",
    "rsi_14",
    "atr_ratio",
    "volume_spike",
    "bb_width",
    "rsi_oversold_strength",
    "vix_regime",
    "pe_attractive",
    "drop_x_drawdown",
    "vol_x_drop",
    "return_1m",
    "return_3m_pre",
    "return_6m_pre",
    "sector_relative",
    "beta_60d",
    "vol_of_vol",
    "quality_dislocation",
    "peg_implicit",
    "relative_drop",
    "month_of_year",
    "sector_alert_count_7d",
    "days_since_52w_high",
    "vix_percentile_1y",
    "spy_rsi_14",
]
# Veja ml_training/config.py para nota sobre features removidas (4 constantes
# + 3 ruidosas com IC<0.01). _FALLBACK abaixo retém as 7 entradas para
# inference compat (cálculos que ainda lêem os keys herdados não rebentam).

LABEL_COLUMNS: list[str] = [
    "label_win",
    "label_further_drop",
    "label_upside_10_90d",
    "label_downside_15_20d",
]

N_FEATURES = len(FEATURE_COLUMNS)

_SECTOR_FAIR_PE: dict[str, float] = {
    "Technology": 35.0,
    "Healthcare": 22.0,
    "Communication Services": 22.0,
    "Financial Services": 13.0,
    "Financials": 13.0,
    "Consumer Cyclical": 20.0,
    "Consumer Defensive": 22.0,
    "Industrials": 20.0,
    "Energy": 12.0,
    "Utilities": 18.0,
    "Real Estate": 40.0,
    "Basic Materials": 14.0,
    "Materials": 14.0,
}

_FALLBACK: dict[str, float] = {
    "macro_score": 2.0,
    "vix": 20.0,
    "spy_drawdown_5d": 0.0,
    "sector_drawdown_5d": 0.0,
    "gross_margin": 0.35,
    "de_ratio": 80.0,
    "pe_vs_fair": 1.0,
    "analyst_upside": 0.10,
    "quality_score": 0.50,
    "fcf_yield": 0.04,
    "drop_pct_today": -8.0,
    "drawdown_52w": -15.0,
    "rsi_14": 40.0,
    "atr_ratio": 0.02,
    "volume_spike": 1.0,
    "bb_width": 0.12,
    "rsi_oversold_strength": 0.0,
    "vix_regime": 1.0,
    "pe_attractive": 0.0,
    "drop_x_drawdown": 1.2,
    "vol_x_drop": 8.0,
    "return_1m": 0.0,
    "return_3m_pre": 0.0,
    "return_6m_pre": 0.0,
    "return_12m_pre": 0.0,
    "sector_relative": 0.0,
    "sector_relative_6m": 0.0,
    "beta_60d": 1.0,
    "vol_of_vol": 0.01,
    "quality_dislocation": 0.08,
    "peg_implicit": 2.0,
    "relative_drop": 0.0,
    "month_of_year": 6.0,
    "sector_alert_count_7d": 0.0,
    "days_since_52w_high": 180.0,
    "short_interest_ratio": 3.5,
    "earnings_surprise_avg": 0.0,
    "earnings_distance_days": 45.0,
    "vix_percentile_1y": 0.5,
    "spy_rsi_14": 50.0,
    "yield_10y_change_5d": 0.0,
}

def _tz_normalize(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t

def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return df
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df

def add_derived_features(features: dict, alert_date: Optional[Any] = None) -> dict:
    rsi = float(features.get("rsi_14", _FALLBACK["rsi_14"]))
    vix = float(features.get("vix", _FALLBACK["vix"]))
    pe_vf = float(features.get("pe_vs_fair", _FALLBACK["pe_vs_fair"]))
    drop = float(features.get("drop_pct_today", _FALLBACK["drop_pct_today"]))
    dd52 = float(features.get("drawdown_52w", _FALLBACK["drawdown_52w"]))
    volsp = float(features.get("volume_spike", _FALLBACK["volume_spike"]))
    features["rsi_oversold_strength"] = round(max(0.0, 40.0 - rsi), 4)
    features["vix_regime"] = 0.0 if vix < 15.0 else (1.0 if vix < 25.0 else 2.0)
    features["pe_attractive"] = round(max(0.0, 1.0 - pe_vf), 4)
    features["drop_x_drawdown"] = round(drop * dd52 / 100.0, 4)
    features["vol_x_drop"] = round(volsp * abs(drop), 4)
    gm = float(features.get("gross_margin", _FALLBACK["gross_margin"]))
    fcf = float(features.get("fcf_yield", _FALLBACK["fcf_yield"]))
    revenue_growth = float(features.get("revenue_growth", 0.05))
    features["quality_dislocation"] = round(gm * abs(dd52) / 100.0, 4) if fcf >= 0 else 0.0
    if revenue_growth > 0 and pe_vf > 0:
        features["peg_implicit"] = round(min(pe_vf / (revenue_growth * 100.0), 5.0), 4)
    else:
        features["peg_implicit"] = 3.0
    sec_dd = float(features.get("sector_drawdown_5d", _FALLBACK["sector_drawdown_5d"]))
    features["relative_drop"] = round(drop - sec_dd, 4)
    try:
        features["month_of_year"] = float(pd.Timestamp(alert_date).month) if alert_date is not None else float(datetime.now().month)
    except Exception:
        features["month_of_year"] = float(datetime.now().month)
    return features


# ────────────────────────────────────────────────────────────────────────────────
# build_features — thin orchestrator usado pelo position_monitor para
# re-scoring diário das posições activas. Replica o contrato antigo:
#   build_features(ticker, fundamentals) → dict com todas as FEATURE_COLUMNS.
# Sem price_history/sector → as features técnicas e momentum caem para
# _FALLBACK determinístico (aceitável para vigilância diária).
# ────────────────────────────────────────────────────────────────────────────────
def build_features(
    ticker: str,
    fundamentals: dict,
    *,
    sector: str = "Unknown",
    macro_context: Optional[dict] = None,
    alert_date: Optional[Any] = None,
) -> dict:
    """Constrói um dict de features para inference (predict_dip / ml_score).

    Designed for **position_monitor** path — re-scoring diário das posições
    activas sem price_history nem sector ETF disponíveis. Para training
    histórico ou snapshot do universo, usar ``ml_training.data.build_dataset``
    ou ``universe_snapshot._build_snapshot_row`` (que constroem o dict
    manualmente e chamam apenas ``add_derived_features``).

    Parameters
    ----------
    ticker         : str   identificador (apenas para logs).
    fundamentals   : dict  output de ``_fetch_fundamentals_snapshot`` ou
                           equivalente. Chaves opcionais: ``pe``, ``debt_equity``,
                           ``revenue_growth``, ``gross_margin``, ``fcf_yield``,
                           ``analyst_upside``, ``price``, ``market_cap``.
    sector         : str   GICS sector (defaults "Unknown" → fair PE 22.0).
    macro_context  : dict  ``get_macro_context`` pre-calculado (evita API call
                           extra). ``None`` = chama ``get_macro_context``.
    alert_date     : Any   timestamp do alerta (usado para ``month_of_year``).
                           ``None`` = hoje.

    Returns
    -------
    dict com todas as FEATURE_COLUMNS preenchidas (valores fundamentais
    vindos do dict, técnicos/momentum/regime em fallback).
    """
    fund = fundamentals or {}

    # Stage 0 — Macro
    if macro_context is None:
        try:
            macro_context = get_macro_context(sector=sector)
        except Exception as e:
            logger.debug(f"build_features[{ticker}]: get_macro_context falhou: {e}")
            macro_context = {}
    macro_context = macro_context or {}

    features: dict = {col: _FALLBACK[col] for col in FEATURE_COLUMNS}

    # Macro overrides
    for k in ("macro_score", "vix", "spy_drawdown_5d", "sector_drawdown_5d"):
        if k in macro_context and macro_context[k] is not None:
            try:
                features[k] = float(macro_context[k])
            except (TypeError, ValueError):
                pass

    # Stage 1 — Quality / Value (do dict de fundamentais)
    def _set(col: str, val: Any) -> None:
        if val is None:
            return
        try:
            f = float(val)
        except (TypeError, ValueError):
            return
        if math.isfinite(f):
            features[col] = f

    _set("gross_margin",   fund.get("gross_margin"))
    _set("de_ratio",       fund.get("debt_equity") or fund.get("de_ratio"))
    _set("analyst_upside", fund.get("analyst_upside") or fund.get("upside"))
    _set("fcf_yield",      fund.get("fcf_yield"))

    # pe_vs_fair = pe / fair_pe_sector (ratio > 1 = overvalued)
    pe_raw = _safe_float(fund.get("pe"))
    fair_pe = _SECTOR_FAIR_PE.get(sector, 22.0)
    if not math.isnan(pe_raw) and pe_raw > 0 and fair_pe > 0:
        features["pe_vs_fair"] = round(pe_raw / fair_pe, 4)

    # Quality score via score.py (hemisphere A normalizado [0,1])
    try:
        sr = score_from_fundamentals(fund)
        if isinstance(sr, dict) and "quality_score" in sr:
            features["quality_score"] = float(sr["quality_score"])
        elif isinstance(sr, (int, float)):
            features["quality_score"] = float(sr)
    except Exception as e:
        logger.debug(f"build_features[{ticker}]: score_from_fundamentals falhou: {e}")

    # Stage 2 — Timing (do dict; sem price_history → fallback)
    _set("drop_pct_today", fund.get("drop_pct_today") or fund.get("change_pct"))
    _set("drawdown_52w",   fund.get("drawdown_52w"))
    _set("rsi_14",         fund.get("rsi_14") or fund.get("rsi"))
    _set("atr_ratio",      fund.get("atr_ratio"))
    _set("volume_spike",   fund.get("volume_spike"))

    # Stage 3 — Derived (interactions). Sobrescreve os 7 derivados a partir
    # do dict actual; o resto fica em fallback.
    add_derived_features(features, alert_date=alert_date)

    return features


def add_context_features(features: dict, price_history: Optional[pd.DataFrame] = None, sector_alert_count_7d: Optional[float] = None) -> dict:
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
                idx_max = lookback.values.argmax()
                days_ago = len(lookback) - 1 - idx_max
                cal_days = round(days_ago * 1.4)
                features["days_since_52w_high"] = float(max(0, cal_days))
            else:
                features["days_since_52w_high"] = _FALLBACK["days_since_52w_high"]
        except Exception:
            features["days_since_52w_high"] = _FALLBACK["days_since_52w_high"]
    else:
        features["days_since_52w_high"] = _FALLBACK["days_since_52w_high"]
    return features

def add_short_earnings_features(features: dict, ticker_info: Optional[dict] = None, alert_date: Optional[Any] = None) -> dict:
    info = ticker_info or {}
    sr = _safe_float(info.get("shortRatio"))
    features["short_interest_ratio"] = float(min(sr, 30.0)) if math.isfinite(sr) and sr >= 0 else _FALLBACK["short_interest_ratio"]
    try:
        hist = info.get("earningsHistory", {})
        if isinstance(hist, dict):
            hist = hist.get("history", [])
        surprises = []
        next_earnings_distance = _FALLBACK["earnings_distance_days"]
        now_ts = _tz_normalize(alert_date) if alert_date is not None else None
        for entry in (hist or [])[:8]:
            if isinstance(entry, dict):
                sp = _safe_float(entry.get("surprisePercent"))
                if math.isfinite(sp):
                    surprises.append(sp * 100.0)
                if now_ts is not None:
                    dt_raw = entry.get("quarter") or entry.get("date") or entry.get("earningsDate")
                    if dt_raw is not None:
                        try:
                            dt = _tz_normalize(dt_raw)
                            diff = (dt - now_ts).days
                            if diff >= 0:
                                next_earnings_distance = min(next_earnings_distance, float(diff))
                        except Exception:
                            pass
        features["earnings_surprise_avg"] = float(np.clip(np.mean(surprises[:2]), -50.0, 50.0)) if surprises else _FALLBACK["earnings_surprise_avg"]
        features["earnings_distance_days"] = float(np.clip(next_earnings_distance, 0.0, 180.0))
    except Exception:
        features["earnings_surprise_avg"] = _FALLBACK["earnings_surprise_avg"]
        features["earnings_distance_days"] = _FALLBACK["earnings_distance_days"]
    return features

def add_regime_features(features: dict, spy_history: Optional[pd.DataFrame], tnx_history: Optional[pd.DataFrame], alert_date: Any, vix_history: Optional[pd.DataFrame] = None) -> dict:
    alert_ts = _tz_normalize(alert_date)
    vix_hist = _normalize_index(vix_history)
    spy_hist = _normalize_index(spy_history)
    tnx_hist = _normalize_index(tnx_history)
    try:
        vix_val = float(features.get("vix", _FALLBACK["vix"]))
        pct = _FALLBACK["vix_percentile_1y"]
        if vix_hist is not None and "Close" in vix_hist.columns:
            vix_slice = vix_hist[vix_hist.index <= alert_ts]
            window = vix_slice["Close"].dropna().tail(252)
            if len(window) >= 20:
                arr = window.values
                pct = float(np.clip(np.sum(arr <= vix_val) / len(arr), 0.0, 1.0))
        elif spy_hist is not None and "Close" in spy_hist.columns:
            rets = spy_hist[spy_hist.index <= alert_ts]["Close"].pct_change().dropna().tail(252)
            if len(rets) >= 20:
                rv_window = rets.rolling(5).std().dropna()
                if len(rv_window) >= 20:
                    cur_rv = float(rv_window.iloc[-1])
                    pct = float(np.clip(np.sum(rv_window.values <= cur_rv) / len(rv_window), 0.0, 1.0))
        features["vix_percentile_1y"] = round(pct, 4)
    except Exception:
        features["vix_percentile_1y"] = _FALLBACK["vix_percentile_1y"]
    try:
        rsi_val = _FALLBACK["spy_rsi_14"]
        if spy_hist is not None and "Close" in spy_hist.columns:
            closes = spy_hist[spy_hist.index <= alert_ts]["Close"].dropna()
            if len(closes) >= 16:
                delta = closes.diff().dropna()
                gain = delta.clip(lower=0).rolling(14).mean()
                loss = (-delta.clip(upper=0)).rolling(14).mean()
                rs = gain / loss.replace(0, np.nan)
                rsi_s = (100 - 100 / (1 + rs)).iloc[-1]
                if pd.notna(rsi_s):
                    rsi_val = float(np.clip(rsi_s, 0.0, 100.0))
        features["spy_rsi_14"] = round(rsi_val, 2)
    except Exception:
        features["spy_rsi_14"] = _FALLBACK["spy_rsi_14"]
    try:
        chg = _FALLBACK["yield_10y_change_5d"]
        if tnx_hist is not None and "Close" in tnx_hist.columns:
            tnx_slice = tnx_hist[tnx_hist.index <= alert_ts]["Close"].dropna()
            if len(tnx_slice) >= 6:
                chg = float(tnx_slice.iloc[-1] - tnx_slice.iloc[-6])
                if not math.isfinite(chg) or abs(chg) > 5.0:
                    chg = _FALLBACK["yield_10y_change_5d"]
        features["yield_10y_change_5d"] = round(chg, 4)
    except Exception:
        features["yield_10y_change_5d"] = _FALLBACK["yield_10y_change_5d"]
    return features

def _pct_return(prices: np.ndarray, lookback: int) -> float:
    if len(prices) < lookback + 1:
        return 0.0
    p_end = float(prices[-1])
    p_start = float(prices[-lookback - 1])
    if p_start <= 0:
        return 0.0
    return round((p_end / p_start - 1.0) * 100.0, 4)

def _vol_of_vol(prices: np.ndarray, rv_window: int = 5, vov_window: int = 60) -> float:
    min_needed = rv_window + vov_window
    if len(prices) < min_needed:
        return _FALLBACK["vol_of_vol"]
    try:
        returns = np.diff(prices[-min_needed:]) / prices[-min_needed:-1]
        rvs = np.array([float(np.std(returns[i:i + rv_window], ddof=1)) for i in range(len(returns) - rv_window + 1)])
        if len(rvs) < 10:
            return _FALLBACK["vol_of_vol"]
        vov = float(np.std(rvs[-vov_window:], ddof=1))
        return round(float(np.clip(vov, 0.0, 0.1)), 6)
    except Exception:
        return _FALLBACK["vol_of_vol"]

def _bb_width(prices: np.ndarray, window: int = 20, n_std: float = 2.0) -> float:
    if len(prices) < window:
        return _FALLBACK["bb_width"]
    try:
        s = pd.Series(prices[-window:])
        ma = float(s.mean())
        sd = float(s.std(ddof=1))
        if ma <= 0 or not math.isfinite(ma) or not math.isfinite(sd):
            return _FALLBACK["bb_width"]
        upper = ma + n_std * sd
        lower = ma - n_std * sd
        width = (upper - lower) / ma
        return round(float(np.clip(width, 0.0, 2.0)), 6)
    except Exception:
        return _FALLBACK["bb_width"]

def add_momentum_features(features: dict, price_history: Optional[pd.DataFrame], sector_history: Optional[pd.DataFrame] = None, spy_history: Optional[pd.DataFrame] = None) -> dict:
    keys = ["return_1m", "return_3m_pre", "return_6m_pre", "return_12m_pre", "sector_relative", "sector_relative_6m", "beta_60d", "vol_of_vol", "bb_width"]
    if price_history is None or price_history.empty:
        for k in keys:
            features.setdefault(k, _FALLBACK[k])
        return features
    closes = price_history["Close"].dropna().values
    features["return_1m"] = _pct_return(closes, 21)
    features["return_3m_pre"] = _pct_return(closes, 63)
    features["return_6m_pre"] = _pct_return(closes, 126)
    features["return_12m_pre"] = _pct_return(closes, 252)
    features["bb_width"] = _bb_width(closes)
    if sector_history is not None and not sector_history.empty:
        sec_closes = sector_history["Close"].dropna().values
        sec_ret_3m = _pct_return(sec_closes, 63)
        sec_ret_6m = _pct_return(sec_closes, 126)
        features["sector_relative"] = round(features["return_3m_pre"] - sec_ret_3m, 4)
        features["sector_relative_6m"] = round(features["return_6m_pre"] - sec_ret_6m, 4)
    else:
        features["sector_relative"] = _FALLBACK["sector_relative"]
        features["sector_relative_6m"] = _FALLBACK["sector_relative_6m"]
    features["beta_60d"] = _FALLBACK["beta_60d"]
    if spy_history is not None and not spy_history.empty:
        try:
            spy_closes = spy_history["Close"].dropna().values
            n = 60
            if len(closes) >= n + 1 and len(spy_closes) >= n + 1:
                stock_rets = np.diff(closes[-n - 1:]) / closes[-n - 1:-1]
                spy_rets = np.diff(spy_closes[-n - 1:]) / spy_closes[-n - 1:-1]
                min_len = min(len(stock_rets), len(spy_rets))
                stock_rets = stock_rets[-min_len:]
                spy_rets = spy_rets[-min_len:]
                cov = np.cov(stock_rets, spy_rets)
                var_spy = float(cov[1, 1])
                if var_spy > 1e-10:
                    features["beta_60d"] = round(float(cov[0, 1]) / var_spy, 4)
        except Exception:
            pass
    features["vol_of_vol"] = _vol_of_vol(closes)
    return features
