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
    # quality_dislocation e peg_implicit são versões compostas de features
    # existentes — tornam explícito para o modelo o que o score.py já computa.
    "quality_dislocation",  # gross_margin * |drawdown_52w| / 100 (0.0 se FCF negativo)
    "peg_implicit",         # pe_vs_fair / (revenue_growth * 100), clip [0, 5]
    "relative_drop",        # drop_pct_today - sector_drawdown_5d
    "month_of_year",        # mês do ano (1–12) — sazonalidade
]

LABEL_COLUMNS: list[str] = [
    "label_win",            # int: 1 if recovery >=15% within 60d (Model A target)
    "label_further_drop",   # float: max additional % drop before recovery (Model B target)
]

# Total feature count (used as sanity check in training)
N_FEATURES = len(FEATURE_COLUMNS)  # 27 (23 base + 4 dislocation)


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
}


# ────────────────────────────────────────────────────────────────────────────────
# Stage 3 — Engineered features
# ────────────────────────────────────────────────────────────────────────────────

def add_derived_features(features: dict) -> dict:
    """
    Compute Stage-3 engineered features + Stage-3c dislocation features
    from base features. Same code used at training time and at inference.
    Mutates and returns the same dict.
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

    # month_of_year: sazonalidade (1–12)
    features["month_of_year"] = float(datetime.now().month)

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
    p_start = float(prices[-(lookback + 1)])
    if p_start <= 0:
        return 0.0
    return round((p_end / p_start - 1.0) * 100.0, 4)


def add_momentum_features(
    features: dict,
    price_history: Optional[pd.DataFrame],
    sector_history: Optional[pd.DataFrame] = None,
    spy_history: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Compute 4 momentum features (Stage 3b) and add them to the feature dict.

    Parameters
    ----------
    features        : dict  Feature dict (mutated in place)
    price_history   : DataFrame  OHLCV for the stock, daily, ending at alert_date (exclusive)
    sector_history  : DataFrame  OHLCV for the sector ETF (same period) — for sector_relative
    spy_history     : DataFrame  OHLCV for SPY (same period) — for beta_60d

    Anti-leakage guarantee:
      price_history must contain only rows BEFORE the alert date.
      In training, the parquet builder slices df.loc[:alert_date - 1 day].
      In production, yfinance returns data up to yesterday by default.

    Returns the mutated dict.
    """
    # ── return_1m: 21 trading days ──────────────────────────────────
    if price_history is not None and "Close" in price_history.columns:
        try:
            closes = price_history["Close"].dropna().values
            features["return_1m"]    = _pct_return(closes, 21)
            features["return_3m_pre"] = _pct_return(closes, 63)
        except Exception as e:
            logger.debug(f"add_momentum_features: return computation failed: {e}")
            features["return_1m"]    = _FALLBACK["return_1m"]
            features["return_3m_pre"] = _FALLBACK["return_3m_pre"]
    else:
        features["return_1m"]    = _FALLBACK["return_1m"]
        features["return_3m_pre"] = _FALLBACK["return_3m_pre"]

    # ── sector_relative: stock return_3m_pre minus sector return_3m_pre ──
    if sector_history is not None and "Close" in sector_history.columns:
        try:
            sec_closes = sector_history["Close"].dropna().values
            sec_ret3m  = _pct_return(sec_closes, 63)
            features["sector_relative"] = round(
                features["return_3m_pre"] - sec_ret3m, 4
            )
        except Exception as e:
            logger.debug(f"add_momentum_features: sector_relative failed: {e}")
            features["sector_relative"] = _FALLBACK["sector_relative"]
    else:
        features["sector_relative"] = _FALLBACK["sector_relative"]

    # ── beta_60d: OLS beta of daily returns vs SPY over 60 trading days ──
    if (
        spy_history is not None
        and price_history is not None
        and "Close" in price_history.columns
        and "Close" in spy_history.columns
    ):
        try:
            stock_c = price_history["Close"].dropna()
            spy_c   = spy_history["Close"].dropna()

            # Align on common dates, take last 60 bars
            combined = pd.concat(
                [stock_c.rename("stock"), spy_c.rename("spy")], axis=1
            ).dropna().tail(61)

            if len(combined) >= 20:  # minimum for a stable beta
                stock_ret = combined["stock"].pct_change().dropna().values
                spy_ret   = combined["spy"].pct_change().dropna().values
                # OLS beta = cov(stock, spy) / var(spy)
                cov_matrix = np.cov(stock_ret, spy_ret)
                var_spy    = cov_matrix[1, 1]
                if var_spy > 0:
                    beta = round(float(cov_matrix[0, 1] / var_spy), 4)
                    # Clamp to [-3, 5] to avoid extreme leverage artefacts
                    features["beta_60d"] = float(np.clip(beta, -3.0, 5.0))
                else:
                    features["beta_60d"] = _FALLBACK["beta_60d"]
            else:
                features["beta_60d"] = _FALLBACK["beta_60d"]
        except Exception as e:
            logger.debug(f"add_momentum_features: beta_60d failed: {e}")
            features["beta_60d"] = _FALLBACK["beta_60d"]
    else:
        features["beta_60d"] = _FALLBACK["beta_60d"]

    return features


# ────────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────────────────

def _sf(val: Any, key: str) -> float:
    """
    Safe float with named fallback. Logs at DEBUG when fallback is used.
    Never returns NaN or Inf to the caller.
    """
    v = _safe_float(val)
    if math.isnan(v) or not math.isfinite(v):
        fallback = _FALLBACK[key]
        logger.debug(f"ml_features: '{key}' missing/invalid — using fallback {fallback}")
        return fallback
    return v


def _compute_atr_ratio(price_history: Optional[pd.DataFrame], current_price: float) -> float:
    """ATR(14) / current_price."""
    if price_history is None or len(price_history) < 15:
        return _FALLBACK["atr_ratio"]
    try:
        df = price_history.tail(20).copy()
        high  = df["High"].values
        low   = df["Low"].values
        close = df["Close"].values
        tr = np.maximum(
            high[1:]  - low[1:],
            np.maximum(
                np.abs(high[1:]  - close[:-1]),
                np.abs(low[1:]   - close[:-1]),
            )
        )
        atr14 = float(np.mean(tr[-14:]))
        if current_price <= 0:
            return _FALLBACK["atr_ratio"]
        return round(atr14 / current_price, 6)
    except Exception as e:
        logger.debug(f"ml_features: ATR computation failed: {e}")
        return _FALLBACK["atr_ratio"]


def _compute_volume_spike(price_history: Optional[pd.DataFrame], fund: dict) -> float:
    """Today's volume / 20-day average volume."""
    if price_history is not None and "Volume" in price_history.columns and len(price_history) >= 5:
        try:
            vols = price_history["Volume"].dropna().values
            if len(vols) >= 2:
                today_vol = float(vols[-1])
                avg_vol   = float(np.mean(vols[-21:-1])) if len(vols) > 21 else float(np.mean(vols[:-1]))
                if avg_vol > 0:
                    return round(today_vol / avg_vol, 4)
        except Exception as e:
            logger.debug(f"ml_features: volume_spike from history failed: {e}")

    vol     = _safe_float(fund.get("volume"))
    avg_vol = _safe_float(fund.get("average_volume"))
    if not math.isnan(vol) and not math.isnan(avg_vol) and avg_vol > 0:
        return round(vol / avg_vol, 4)
    return _FALLBACK["volume_spike"]


# ────────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────────

def build_features(
    ticker: str,
    fundamentals: dict,
    price_history: Optional[pd.DataFrame] = None,
    sector: str = "Unknown",
    drop_pct_today: Optional[float] = None,
    label_win: Optional[int] = None,
    label_further_drop: Optional[float] = None,
    macro_context: Optional[dict] = None,
    sector_history: Optional[pd.DataFrame] = None,
    spy_history: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Build the complete feature vector for one stock at one point in time.

    Parameters
    ----------
    ticker          : str       Stock ticker (used only for logging)
    fundamentals    : dict      Output of market_client.get_fundamentals()
    price_history   : DataFrame Optional OHLCV DataFrame (daily, OHLCV columns)
                                Required for ATR, volume_spike, and momentum features.
                                Must contain only rows BEFORE alert_date (no leakage).
    sector          : str       GICS sector string (e.g. "Technology")
    drop_pct_today  : float     % price change that triggered the dip alert (negative)
    label_win       : int|None  Training label only.
    label_further_drop: float|None  Training label only.
    macro_context   : dict|None Pre-computed macro dict.
    sector_history  : DataFrame Optional OHLCV for the sector ETF (for sector_relative).
    spy_history     : DataFrame Optional OHLCV for SPY (for beta_60d).

    Returns
    -------
    dict with keys = FEATURE_COLUMNS + LABEL_COLUMNS (+ fcf_yield, revenue_growth)
    """
    fund = fundamentals

    # ── Stage 0: Macro ────────────────────────────────────────────────────
    if macro_context is None:
        macro_context = get_macro_context(sector=sector)

    macro_score        = float(macro_context.get("macro_score",        _FALLBACK["macro_score"]))
    vix                = float(macro_context.get("vix",                _FALLBACK["vix"]))
    spy_drawdown_5d    = float(macro_context.get("spy_drawdown_5d",    _FALLBACK["spy_drawdown_5d"]))
    sector_drawdown_5d = float(macro_context.get("sector_drawdown_5d", _FALLBACK["sector_drawdown_5d"]))

    # ── Stage 1: Quality / Value ───────────────────────────────────────
    fcf_yield      = _sf(fund.get("fcf_yield"),      "fcf_yield")
    revenue_growth = _sf(fund.get("revenue_growth"), "revenue_growth")
    gross_margin   = _sf(fund.get("gross_margin"),   "gross_margin")
    de_ratio       = _sf(fund.get("debt_equity"),    "de_ratio")
    analyst_upside = _sf(fund.get("analyst_upside") or fund.get("upside"), "analyst_upside")

    pe_raw  = _safe_float(fund.get("pe"))
    fair_pe = _SECTOR_FAIR_PE.get(sector, 22.0)
    if not math.isnan(pe_raw) and pe_raw > 0 and fair_pe > 0:
        pe_vs_fair = round(pe_raw / fair_pe, 4)
    else:
        pe_vs_fair = _FALLBACK["pe_vs_fair"]
        logger.debug(f"ml_features [{ticker}]: pe_vs_fair fallback (pe={pe_raw}, fair={fair_pe})")

    try:
        score_result  = score_from_fundamentals(fund)
        quality_score = float(score_result["quality_score"])
        if not math.isfinite(quality_score):
            quality_score = _FALLBACK["quality_score"]
    except Exception as e:
        logger.warning(f"ml_features [{ticker}]: score_from_fundamentals failed: {e}")
        quality_score = _FALLBACK["quality_score"]

    # ── Stage 2: Timing ─────────────────────────────────────────────────
    if drop_pct_today is None:
        drop_pct_today = _sf(
            fund.get("change_pct") or fund.get("drop_pct") or fund.get("changePercent"),
            "drop_pct_today"
        )
    else:
        drop_pct_today = float(drop_pct_today)

    drawdown_52w = _sf(
        fund.get("drawdown_from_high") or fund.get("drawdown_52w"),
        "drawdown_52w"
    )
    rsi_14 = float(np.clip(_sf(fund.get("rsi") or fund.get("rsi_14"), "rsi_14"), 0.0, 100.0))

    current_price = _safe_float(fund.get("price") or fund.get("current_price"))
    if math.isnan(current_price) or current_price <= 0:
        if price_history is not None and len(price_history) > 0:
            try:
                current_price = float(price_history["Close"].dropna().iloc[-1])
            except Exception:
                current_price = 1.0
        else:
            current_price = 1.0

    atr_ratio    = _compute_atr_ratio(price_history, current_price)
    volume_spike = _compute_volume_spike(price_history, fund)

    # ── Assemble base vector ───────────────────────────────────────────────
    feature_vector = {
        # Stage 0
        "macro_score":          macro_score,
        "vix":                  vix,
        "spy_drawdown_5d":      spy_drawdown_5d,
        "sector_drawdown_5d":   sector_drawdown_5d,
        # Stage 1
        "gross_margin":         gross_margin,
        "de_ratio":             de_ratio,
        "pe_vs_fair":           pe_vs_fair,
        "analyst_upside":       analyst_upside,
        "quality_score":        quality_score,
        # Stage 2
        "drop_pct_today":       drop_pct_today,
        "drawdown_52w":         drawdown_52w,
        "rsi_14":               rsi_14,
        "atr_ratio":            atr_ratio,
        "volume_spike":         volume_spike,
        # Downstream-only (não entram no modelo mas necessários para Stage 3c)
        "fcf_yield":            fcf_yield,
        "revenue_growth":       revenue_growth,
    }

    # ── Stage 3: Engineered + 3c Dislocation features ───────────────────
    add_derived_features(feature_vector)

    # ── Stage 3b: Momentum features ────────────────────────────────────
    add_momentum_features(feature_vector, price_history, sector_history, spy_history)

    # Labels
    feature_vector["label_win"]          = label_win
    feature_vector["label_further_drop"] = label_further_drop

    # ── Sanity check ────────────────────────────────────────────────────
    for col in FEATURE_COLUMNS:
        val = feature_vector[col]
        if not math.isfinite(float(val)):
            logger.error(
                f"ml_features [{ticker}]: NaN/Inf in '{col}' after fallback — "
                f"forcing to global fallback {_FALLBACK[col]}"
            )
            feature_vector[col] = _FALLBACK[col]

    logger.debug(
        f"ml_features [{ticker}]: built {N_FEATURES} features — "
        f"macro={macro_context.get('regime')} "
        f"quality={quality_score:.3f} "
        f"rsi={rsi_14:.0f} "
        f"drop={drop_pct_today:.1f}% "
        f"ret1m={feature_vector['return_1m']:.1f}% "
        f"beta={feature_vector['beta_60d']:.2f} "
        f"qd={feature_vector['quality_dislocation']:.3f} "
        f"peg={feature_vector['peg_implicit']:.2f}"
    )

    return feature_vector


def build_feature_row(feature_vector: dict) -> list[float]:
    """
    Extract features in canonical FEATURE_COLUMNS order as a flat list.
    Used when feeding a single observation to a trained model.
    """
    return [float(feature_vector[col]) for col in FEATURE_COLUMNS]


def build_feature_df(rows: list[dict]) -> pd.DataFrame:
    """
    Convert a list of feature vectors into a DataFrame with the correct column order.
    """
    all_cols = FEATURE_COLUMNS + LABEL_COLUMNS
    return pd.DataFrame(rows, columns=all_cols)


# ────────────────────────────────────────────────────────────────────────────────
# CLI smoke test  (python ml_features.py)
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.DEBUG)

    print(f"FEATURE_COLUMNS ({N_FEATURES}): {FEATURE_COLUMNS}")
    print()

    mock_fund = {
        "price":          387.20,
        "fcf_yield":      0.062,
        "revenue_growth": 0.123,
        "gross_margin":   0.684,
        "debt_equity":    41.0,
        "pe":             24.1,
        "rsi":            28.0,
        "drawdown_from_high": -23.1,
        "analyst_upside": 0.25,
        "volume":         18_500_000,
        "average_volume": 9_000_000,
        "change_pct":     -8.4,
        "roic":           0.22,
        "fcf_margin":     0.14,
    }
    mock_macro = {
        "regime":              "NEUTRAL",
        "macro_score":         2,
        "vix":                 18.2,
        "spy_drawdown_5d":    -1.2,
        "sector_drawdown_5d": -2.8,
    }

    # Mock 90-day OHLCV history for momentum features
    np.random.seed(42)
    dates  = pd.date_range(end="2026-01-13", periods=90, freq="B")
    prices = 400.0 * np.cumprod(1 + np.random.normal(0, 0.01, 90))
    mock_history = pd.DataFrame({
        "Open":   prices * 0.99,
        "High":   prices * 1.01,
        "Low":    prices * 0.98,
        "Close":  prices,
        "Volume": np.random.randint(5_000_000, 20_000_000, 90),
    }, index=dates)

    spy_prices = 500.0 * np.cumprod(1 + np.random.normal(0, 0.008, 90))
    mock_spy = pd.DataFrame({
        "Close": spy_prices,
    }, index=dates)

    fv = build_features(
        ticker="MSFT",
        fundamentals=mock_fund,
        price_history=mock_history,
        sector="Technology",
        drop_pct_today=-8.4,
        macro_context=mock_macro,
        label_win=1,
        label_further_drop=-4.2,
        spy_history=mock_spy,
    )

    print("Feature vector:")
    for k, v in fv.items():
        print(f"  {k:25s}: {v}")

    print()
    row = build_feature_row(fv)
    print(f"Flat row ({len(row)} values): {row}")
    assert len(row) == N_FEATURES, f"Expected {N_FEATURES}, got {len(row)}"

    # Verifica dislocation features
    assert "quality_dislocation" in fv, "quality_dislocation ausente"
    assert "peg_implicit"        in fv, "peg_implicit ausente"
    assert "relative_drop"       in fv, "relative_drop ausente"
    assert "month_of_year"       in fv, "month_of_year ausente"
    print(f"\nAssert OK — {N_FEATURES} features (incl. 4 dislocation).")
    print(f"  quality_dislocation = {fv['quality_dislocation']:.4f}")
    print(f"  peg_implicit        = {fv['peg_implicit']:.4f}")
    print(f"  relative_drop       = {fv['relative_drop']:.4f}")
    print(f"  month_of_year       = {fv['month_of_year']}")
