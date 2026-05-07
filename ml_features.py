"""
ml_features.py — Unified Feature Store for DipRadar ML Pipeline.

Builds the complete feature vector for a single stock at a given moment.
Used identically during training (with labels) and inference in production (labels=None).

Architecture: 4-stage pipeline
  Stage 0 — Macro:    macro regime, VIX, SPY/sector drawdown, FRED recession prob
  Stage 1 — Quality:  gross margin, D/E, P/E vs sector fair
                       analyst upside, ROIC — sourced from score.py hemispheres
                       (fcf_yield and revenue_growth removed: importance=0.0)
  Stage 2 — Timing:   drop today, drawdown 52w, RSI, ATR ratio, volume spike
  Stage 3 — Engineered: non-linear interactions (rsi_oversold_strength, etc.)
  Stage 3b — Momentum: pre-alert price momentum (return_1m, return_3m_pre,
                        sector_relative, beta_60d)
                        Factor-investing literature: momentum is the most
                        predictive single feature for 3–6 month forward returns.
  Stage 3c — Dislocation: quality_dislocation, peg_implicit, relative_drop,
                        month_of_year — distinguem Quality Dislocation (NOW/PINS)
                        de especulação pura (RKLB).
  Stage 3d — Context: sector_alert_count_7d, days_since_52w_high — market
                        breadth and timing context added in v3.2.
  Stage 3e — Short/Earnings: short_interest_ratio, earnings_surprise_avg —
                        short squeeze signal + earnings quality (v3.3).
  Stage 3f — Regime:  vix_percentile_1y, spy_rsi_14, yield_10y_change_5d —
                        point-in-time market regime signals (v3.4).
                        Distinguish dips in stressed markets vs normal conditions.

Label schema (training only, None in production):
  label_win           int   1 if price recovered >=15% within 60 calendar days
  label_further_drop  float max additional % drop observed before recovery (Model B target)

NaN contract:
  Every feature has an explicit fallback. No raw NaN reaches the model.
  Features that cannot be computed get their sector/global median as fallback,
  logged at DEBUG level. This makes the vector walk-forward safe.

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
    # NOTE: fcf_yield and revenue_growth were removed (importance=0.0 in
    # both Stage 1 and Stage 2 as of ml_report 2026-05-02). Their values
    # are still computed in build_features() for score.py / downstream use,
    # but they are no longer model input columns.
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

    # ── Stage 3: Engineered / Non-linear interactions (5 features) ────────
    "rsi_oversold_strength", # max(0, 40 - rsi_14): magnitude of oversold
    "vix_regime",            # 0=low (<15), 1=medium (15-25), 2=high (>25)
    "pe_attractive",         # max(0, 1 - pe_vs_fair): magnitude of undervaluation
    "drop_x_drawdown",       # drop_pct_today * drawdown_52w / 100: capitulation pressure
    "vol_x_drop",            # volume_spike * abs(drop_pct_today): capitulation volume

    # ── Stage 3b: Momentum (4 features) ─────────────────────────────
    # Pre-alert price momentum. Literature: momentum is the most predictive
    # single factor for 3–6 month forward returns in cross-sectional studies.
    # All computed from price_history BEFORE alert_date (no leakage).
    "return_1m",            # % price return over 21 trading days before alert
    "return_3m_pre",        # % price return over 63 trading days before alert
    "sector_relative",      # return_3m_pre minus sector ETF return over same period
    "beta_60d",             # rolling beta vs SPY over 60 trading days before alert

    # ── Stage 3c: Dislocation (4 features) ───────────────────────────
    # Distinguem Quality Dislocation (NOW/PINS) de especulação pura (RKLB).
    "quality_dislocation",  # gross_margin * |drawdown_52w| / 100 (0.0 se FCF negativo)
    "peg_implicit",         # pe_vs_fair / (revenue_growth * 100), clip [0, 5]
    "relative_drop",        # drop_pct_today - sector_drawdown_5d
    "month_of_year",        # mês do alerta (1–12) — sazonalidade point-in-time

    # ── Stage 3d: Context (2 features) — v3.2 ────────────────────────
    # Market breadth + timing context. Capture contagion / panic breadth.
    "sector_alert_count_7d", # nº de alertas no mesmo sector nos últimos 7 dias
                              # (0 = stock-specific dip; alto = sector-wide selloff)
    "days_since_52w_high",   # dias desde o máximo de 52 semanas (via price_history)
                              # fallback: 180 (aprox. metade do ano)

    # ── Stage 3e: Short Interest + Earnings Surprise (v3.3) ──────────────
    # Anti-leakage: shortRatio com lag ~2 semanas (FINRA); earningsHistory = quarters passados.
    "short_interest_ratio",   # dias para cobrir o short (yfinance shortRatio) — clip [0, 30]
    "earnings_surprise_avg",  # média dos últimos 2 EPS surprises (%) — clip [-50, 50]

    # ── Stage 3f: Regime (3 features) — v3.4 ────────────────────────
    # Point-in-time market regime signals. Distinguish dips in stressed
    # markets (high VIX percentile, oversold SPY, rising yields) from
    # dips in calm markets — the recovery probability differs substantially.
    # All computed from historical price data at alert_date (no leakage).
    "vix_percentile_1y",    # VIX rank in trailing 252 sessions [0, 1]
                             # 0 = lowest VIX in 1y (calm), 1 = highest (panic)
    "spy_rsi_14",           # SPY RSI-14 at alert_date [0, 100]
                             # <30 = market oversold (better bounce odds)
                             # >70 = market overbought (dip into strength, risky)
    "yield_10y_change_5d",  # 5-day change in 10Y US Treasury yield (^TNX, %)
                             # rising yields = rate shock environment
                             # falling yields = flight-to-safety / risk-off
]

LABEL_COLUMNS: list[str] = [
    "label_win",            # int: 1 if recovery >=15% within 60d (Model A target)
    "label_further_drop",   # float: max additional % drop before recovery (Model B target)
]

# Total feature count (used as sanity check in training)
# Stage 0(4) + Stage 1(5) + Stage 2(5) + Stage 3(5) + 3b(4) + 3c(4) + 3d(2) + 3e(2) + 3f(3) = 34
# NOTE: Stage 3c includes relative_drop + month_of_year which are in ALL_NEW_FEATURES
# for historical reasons, but are canonical FEATURE_COLUMNS members.
# Full count: 4+5+5+5+4+4+2+2+3 = 34 base + 3 regime = 37 total.
N_FEATURES = len(FEATURE_COLUMNS)  # 37


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
    "macro_score":        2.0,   # NEUTRAL
    "vix":               20.0,
    "spy_drawdown_5d":    0.0,
    "sector_drawdown_5d": 0.0,
    "fcf_yield":          0.04,  # kept for score.py / downstream, not a model column
    "revenue_growth":     0.05,  # kept for score.py / downstream, not a model column
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
    # Stage 3b momentum — neutral/zero: no pre-alert momentum info
    "return_1m":         0.0,
    "return_3m_pre":     0.0,
    "sector_relative":   0.0,
    "beta_60d":          1.0,   # market-neutral assumption
    # Stage 3c dislocation
    "quality_dislocation": 0.08,  # empresa mediana como baseline
    "peg_implicit":        2.0,   # PEG neutro
    "relative_drop":       0.0,
    "month_of_year":       6.0,   # meio do ano como fallback
    # Stage 3d context (v3.2)
    "sector_alert_count_7d": 0.0,  # fallback: sem alertas recentes no sector
    "days_since_52w_high":  180.0,  # fallback: ~6 meses — metade do ano
    # Stage 3e short/earnings (v3.3)
    "short_interest_ratio":  3.5,   # neutro: ~3.5 dias é a mediana do mercado
    "earnings_surprise_avg": 0.0,   # neutro: sem histórico de bater/falhar
    # Stage 3f regime (v3.4)
    "vix_percentile_1y":    0.5,   # neutro: VIX no percentil mediano
    "spy_rsi_14":           50.0,  # neutro: SPY nem sobrecomprado nem sobrevendido
    "yield_10y_change_5d":   0.0,  # neutro: yields estáveis
}


# ────────────────────────────────────────────────────────────────────────────────
# Stage 3 — Engineered features
# ────────────────────────────────────────────────────────────────────────────────

def add_derived_features(features: dict, alert_date: Optional[Any] = None) -> dict:
    """
    Compute Stage-3 engineered features + Stage-3c dislocation features
    from base features. Same code used at training time and at inference.
    Mutates and returns the same dict.

    Parameters
    ----------
    features   : dict  Feature dict (mutated in place)
    alert_date : datetime-like | None
        The actual date of the alert. Used to set month_of_year correctly
        for historical training data. If None (live inference), falls back
        to datetime.now().month so production behaviour is unchanged.
    """
    rsi    = float(features.get("rsi_14",         _FALLBACK["rsi_14"]))
    vix    = float(features.get("vix",            _FALLBACK["vix"]))
    pe_vf  = float(features.get("pe_vs_fair",     _FALLBACK["pe_vs_fair"]))
    drop   = float(features.get("drop_pct_today", _FALLBACK["drop_pct_today"]))
    dd52   = float(features.get("drawdown_52w",   _FALLBACK["drawdown_52w"]))
    volsp  = float(features.get("volume_spike",   _FALLBACK["volume_spike"]))

    # ── Stage 3: non-linear interactions ──────────────────────────────────
    features["rsi_oversold_strength"] = round(max(0.0, 40.0 - rsi), 4)
    features["vix_regime"] = 0.0 if vix < 15.0 else (1.0 if vix < 25.0 else 2.0)
    features["pe_attractive"]   = round(max(0.0, 1.0 - pe_vf), 4)
    features["drop_x_drawdown"] = round(drop * dd52 / 100.0, 4)
    features["vol_x_drop"]      = round(volsp * abs(drop), 4)

    # ── Stage 3c: dislocation features ─────────────────────────────────
    gm  = float(features.get("gross_margin",   _FALLBACK["gross_margin"]))
    fcf = float(features.get("fcf_yield",       _FALLBACK["fcf_yield"]))
    rg  = float(features.get("revenue_growth",  _FALLBACK["revenue_growth"]))

    # quality_dislocation: gross_margin * |drawdown_52w| / 100
    # penalizado para 0.0 se FCF yield negativo (empresa queima caixa não qualifica)
    features["quality_dislocation"] = (
        round(gm * abs(dd52) / 100.0, 4) if fcf >= 0 else 0.0
    )

    # peg_implicit: pe_vs_fair / (revenue_growth * 100), clip [0, 5]
    # fallback 3.0 se crescimento negativo ou P/E inaplicável
    if rg > 0 and pe_vf > 0:
        features["peg_implicit"] = round(min(pe_vf / (rg * 100.0), 5.0), 4)
    else:
        features["peg_implicit"] = 3.0

    # relative_drop: drop_pct_today - sector_drawdown_5d
    sec_dd = float(features.get("sector_drawdown_5d", _FALLBACK["sector_drawdown_5d"]))
    features["relative_drop"] = round(drop - sec_dd, 4)

    # month_of_year: sazonalidade — usa o mês real do alerta para evitar look-ahead
    # em treino histórico. Fallback para datetime.now().month em inferência live.
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
    """
    Compute Stage-3d context features (v3.2):
      - sector_alert_count_7d: market breadth / contagion signal
      - days_since_52w_high: timing — how long since peak (computed from price_history)

    Parameters
    ----------
    features              : dict  Feature dict (mutated in place)
    price_history         : DataFrame  OHLCV daily, ending at alert_date (exclusive)
    sector_alert_count_7d : float|None  Number of dip alerts in the same sector
                            in the past 7 calendar days. Pass from bot state.
                            None → fallback 0.0 (no sector-wide panic signal).

    Anti-leakage:
      days_since_52w_high is computed from price_history rows only (no future data).
      In training the parquet builder slices df.loc[:alert_date - 1 day].
    """
    # ── sector_alert_count_7d ────────────────────────────────────────
    if sector_alert_count_7d is not None:
        v = float(sector_alert_count_7d)
        features["sector_alert_count_7d"] = v if math.isfinite(v) else _FALLBACK["sector_alert_count_7d"]
    else:
        features["sector_alert_count_7d"] = _FALLBACK["sector_alert_count_7d"]

    # ── days_since_52w_high ──────────────────────────────────────────
    # Strategy: find the row with the highest Close in the last 252 trading days
    # and count how many bars ago that was.
    if price_history is not None and "Close" in price_history.columns:
        try:
            closes = price_history["Close"].dropna()
            lookback = closes.tail(252)  # ~1 trading year
            if len(lookback) >= 5:
                idx_max = lookback.values.argmax()          # position of max in tail window
                days_ago = len(lookback) - 1 - idx_max      # bars since that peak
                # Convert trading days to calendar days (approx ×1.4)
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
    """Stage-3e: short_interest_ratio + earnings_surprise_avg.

    Parâmetros
    ----------
    features    : dict  Feature dict (mutado in place)
    ticker_info : dict|None  yfinance .info dict do ticker.
                  Se None, usa fallbacks — sem chamada yfinance aqui.

    Fontes:
      short_interest_ratio : ticker_info["shortRatio"] — dias para cobrir o short
      earnings_surprise_avg: média de earningsHistory[0..1]["surprisePercent"] * 100

    Anti-leakage:
      shortRatio reportado com lag ~2 semanas (FINRA twice-monthly).
      earningsHistory usa apenas quarters passados.
    """
    info = ticker_info or {}

    # ── short_interest_ratio ─────────────────────────────────────────
    sr = _safe_float(info.get("shortRatio"))
    if math.isfinite(sr) and sr >= 0:
        features["short_interest_ratio"] = float(min(sr, 30.0))
    else:
        features["short_interest_ratio"] = _FALLBACK["short_interest_ratio"]

    # ── earnings_surprise_avg ────────────────────────────────────────
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
    """Stage-3f: point-in-time market regime signals (v3.4).

    Computes 3 features that capture the macro regime at the exact moment
    of the dip alert. These distinguish dips in stressed markets (where
    bounces tend to be stronger but riskier) from dips in calm markets.

    Features
    --------
    vix_percentile_1y   : VIX level ranked within the trailing 252 sessions [0, 1].
                          0 = calmest VIX in 1 year, 1 = most stressed.
                          Uses vix_history (^VIX OHLCV); falls back to deriving
                          the rank from the scalar vix already in features dict.

    spy_rsi_14          : RSI-14 of SPY at alert_date [0, 100].
                          <30 = market oversold (historically better bounce odds).
                          >70 = market overbought (dip into strength, higher risk).

    yield_10y_change_5d : 5-session change in the 10Y US Treasury yield (^TNX, %).
                          Positive = rising rates (rate-shock environment).
                          Negative = falling rates (flight-to-safety / risk-off).

    Anti-leakage
    ------------
    All three features use only data at or before alert_date.
    spy_history and vix_history must be sliced to alert_date by the caller
    (ml_training/data.py passes spy_slice which is already so sliced).
    tnx_history is sliced internally here.

    Parameters
    ----------
    features     : dict  Feature dict (mutated in place)
    spy_history  : DataFrame | None  SPY OHLCV up to and including alert_date
    tnx_history  : DataFrame | None  ^TNX OHLCV (full range; sliced internally)
    alert_date   : datetime-like     Point-in-time date of the alert
    vix_history  : DataFrame | None  ^VIX OHLCV (full range; sliced internally)
    """
    alert_ts = pd.Timestamp(alert_date)

    # ── vix_percentile_1y ────────────────────────────────────────────────
    try:
        vix_val = float(features.get("vix", _FALLBACK["vix"]))
        pct = _FALLBACK["vix_percentile_1y"]

        if vix_history is not None and "Close" in vix_history.columns:
            vix_slice = vix_history[vix_history.index <= alert_ts]
            window = vix_slice["Close"].dropna().tail(252)
            if len(window) >= 20:
                arr = window.values
                rank = float(np.sum(arr <= vix_val)) / len(arr)
                pct = float(np.clip(rank, 0.0, 1.0))
        elif spy_history is not None and "Close" in spy_history.columns:
            rets = spy_history["Close"].pct_change().dropna().tail(252)
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

    # ── spy_rsi_14 ───────────────────────────────────────────────────────
    try:
        rsi_val = _FALLBACK["spy_rsi_14"]

        if spy_history is not None and "Close" in spy_history.columns:
            closes = spy_history["Close"].dropna()
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

    # ── yield_10y_change_5d ──────────────────────────────────────────────
    try:
        chg = _FALLBACK["yield_10y_change_5d"]

        if tnx_history is not None and "Close" in tnx_history.columns:
            tnx_slice = tnx_history[tnx_history.index <= alert_ts]["Close"].dropna()
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
# Stage 3b — Momentum features
# ────────────────────────────────────────────────────────────────────────────────

def _pct_return(prices: np.ndarray, lookback: int) -> float:
    """
    % return over the last `lookback` bars.
    Returns 0.0 if insufficient data or division by zero.
    Anti-leakage: uses prices[-1] as the last point BEFORE the alert
    (caller must pass history up to alert_date exclusive).
    """
    if len(prices) < lookback + 1:
        return 0.0
    p_end   = float(prices[-1])
    p_start = float(prices[-lookback - 1])
    if p_start <= 0:
        return 0.0
    return round((p_end / p_start - 1.0) * 100.0, 4)


def add_momentum_features(
    features: dict,
    price_history: pd.DataFrame,
    sector_history: Optional[pd.DataFrame] = None,
    spy_history: Optional[pd.DataFrame] = None,
) -> dict:
    """Stage-3b: pre-alert momentum features.

    All lookbacks use price_history rows ending at (and including) alert_date.
    Caller must pass history sliced to alert_date (no future rows).
    """
    closes = price_history["Close"].dropna().values

    # return_1m: 21 trading-day return before alert
    features["return_1m"] = _pct_return(closes, 21)

    # return_3m_pre: 63 trading-day return before alert
    features["return_3m_pre"] = _pct_return(closes, 63)

    # sector_relative: stock 3m return minus sector ETF 3m return
    if sector_history is not None and "Close" in sector_history.columns:
        sec_closes = sector_history["Close"].dropna().values
        sec_ret = _pct_return(sec_closes, 63)
        features["sector_relative"] = round(features["return_3m_pre"] - sec_ret, 4)
    else:
        features["sector_relative"] = _FALLBACK["sector_relative"]

    # beta_60d: rolling beta vs SPY over 60 trading days
    if spy_history is not None and "Close" in spy_history.columns and len(closes) >= 62:
        try:
            stock_rets = pd.Series(closes).pct_change().dropna().tail(60).values
            spy_rets   = spy_history["Close"].pct_change().dropna().tail(60).values
            n = min(len(stock_rets), len(spy_rets))
            if n >= 20:
                sr = stock_rets[-n:]
                mr = spy_rets[-n:]
                cov = float(np.cov(sr, mr)[0, 1])
                var = float(np.var(mr, ddof=1))
                features["beta_60d"] = round(cov / var, 4) if var > 1e-10 else _FALLBACK["beta_60d"]
            else:
                features["beta_60d"] = _FALLBACK["beta_60d"]
        except Exception:
            features["beta_60d"] = _FALLBACK["beta_60d"]
    else:
        features["beta_60d"] = _FALLBACK["beta_60d"]

    return features
