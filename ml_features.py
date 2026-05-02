"""
ml_features.py — Unified Feature Store for DipRadar ML Pipeline.

Builds the complete feature vector for a single stock at a given moment.
Used identically during training (with labels) and inference in production (labels=None).

Architecture: 3-stage pipeline
  Stage 0 — Macro:   macro regime, VIX, SPY/sector drawdown, FRED recession prob
  Stage 1 — Quality: FCF yield, revenue growth, gross margin, D/E, P/E vs sector fair
                      analyst upside, ROIC — sourced from score.py hemispheres
  Stage 2 — Timing:  drop today, drawdown 52w, RSI, ATR ratio, volume spike

Label schema (training only, None in production):
  label_win           int   1 if price recovered ≥15% within 60 calendar days
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
from typing import Any, Optional

import numpy as np
import pandas as pd

from macro_data import get_macro_context
from score import score_from_fundamentals, _safe_float

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Feature & label column order (the model sees columns in this exact order)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLUMNS: list[str] = [
    # ── Stage 0: Macro (4 features) ──────────────────────────────────
    "macro_score",          # int 0–4 (BEAR=0-1, NEUTRAL=2, BULL=3-4)
    "vix",                  # VIX level (e.g. 18.5)
    "spy_drawdown_5d",      # SPY % change last 5 trading days (negative=down)
    "sector_drawdown_5d",   # sector ETF % change last 5 trading days

    # ── Stage 1: Quality / Value (7 features) ──────────────────────
    "fcf_yield",            # Free Cash Flow yield (e.g. 0.06 = 6%)
    "revenue_growth",       # YoY revenue growth (e.g. 0.12 = 12%)
    "gross_margin",         # Gross margin ratio (e.g. 0.65)
    "de_ratio",             # Debt-to-Equity ratio (yfinance scale, e.g. 45)
    "pe_vs_fair",           # P/E actual / sector fair P/E (< 1 = cheap)
    "analyst_upside",       # consensus analyst upside (e.g. 0.25 = 25%)
    "quality_score",        # score.py quality hemisphere [0, 1]

    # ── Stage 2: Timing (5 features) ─────────────────────────────
    "drop_pct_today",       # % price drop that triggered the alert (e.g. -8.4)
    "drawdown_52w",         # % below 52-week high (e.g. -23.1)
    "rsi_14",               # RSI 14-period [0, 100]
    "atr_ratio",            # ATR(14) / current price — normalised volatility
    "volume_spike",         # today's volume / 20-day avg volume (e.g. 2.1)

    # ── Stage 3: Engineered / Non-linear interactions (5 features) ────
    # Captures non-linearities and interactions that flat tree models miss.
    "rsi_oversold_strength", # max(0, 40 - rsi_14): magnitude of oversold (0 if not oversold)
    "vix_regime",            # 0=low (<15), 1=medium (15-25), 2=high (>25)
    "pe_attractive",         # max(0, 1 - pe_vs_fair): magnitude of undervaluation
    "drop_x_drawdown",       # drop_pct_today * drawdown_52w / 100: capitulation pressure
    "vol_x_drop",            # volume_spike * abs(drop_pct_today): capitulation volume
]

LABEL_COLUMNS: list[str] = [
    "label_win",            # int: 1 if recovery ≥15% within 60d (Model A target)
    "label_further_drop",   # float: max additional % drop before recovery (Model B target)
]

# Total feature count (used as sanity check in training)
N_FEATURES = len(FEATURE_COLUMNS)  # 16


# ─────────────────────────────────────────────────────────────────────────────
# Sector fair P/E (mirrors score.py _SECTOR_PE_MEAN for pe_vs_fair calculation)
# ─────────────────────────────────────────────────────────────────────────────

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
# Values are conservative: neutral, not optimistic
_FALLBACK: dict[str, float] = {
    "macro_score":        2.0,   # NEUTRAL
    "vix":               20.0,
    "spy_drawdown_5d":    0.0,
    "sector_drawdown_5d": 0.0,
    "fcf_yield":          0.04,  # 4% — global median
    "revenue_growth":     0.05,  # 5%
    "gross_margin":       0.35,
    "de_ratio":          80.0,
    "pe_vs_fair":         1.0,   # fairly valued
    "analyst_upside":     0.10,  # 10%
    "quality_score":      0.50,  # neutral
    "drop_pct_today":    -8.0,   # typical tier-1 dip
    "drawdown_52w":      -15.0,
    "rsi_14":            40.0,
    "atr_ratio":          0.02,
    "volume_spike":       1.0,   # no spike
    # Engineered fallbacks (derived from neutral defaults above)
    "rsi_oversold_strength": 0.0,   # max(0, 40 - 40) = 0
    "vix_regime":            1.0,   # medium (vix=20)
    "pe_attractive":         0.0,   # max(0, 1 - 1) = 0
    "drop_x_drawdown":       1.2,   # -8 * -15 / 100 = 1.2
    "vol_x_drop":            8.0,   # 1.0 * 8 = 8
}


# ─────────────────────────────────────────────────────────────────────────────
# Derived feature engineering — shared between training & inference
# ─────────────────────────────────────────────────────────────────────────────

def add_derived_features(features: dict) -> dict:
    """
    Compute the 5 engineered features (Stage 3) from base features.

    All 5 are deterministic functions of base features that are guaranteed
    to be present (after fallback). Same code is used at training time
    (over a parquet) and at inference (live feature dict) — guaranteed
    consistent.

    Mutates and returns the same dict for convenience.
    """
    rsi    = float(features.get("rsi_14",        _FALLBACK["rsi_14"]))
    vix    = float(features.get("vix",           _FALLBACK["vix"]))
    pe_vf  = float(features.get("pe_vs_fair",    _FALLBACK["pe_vs_fair"]))
    drop   = float(features.get("drop_pct_today", _FALLBACK["drop_pct_today"]))
    dd52   = float(features.get("drawdown_52w",  _FALLBACK["drawdown_52w"]))
    volsp  = float(features.get("volume_spike",  _FALLBACK["volume_spike"]))

    features["rsi_oversold_strength"] = round(max(0.0, 40.0 - rsi), 4)

    if vix < 15.0:
        vix_reg = 0.0
    elif vix < 25.0:
        vix_reg = 1.0
    else:
        vix_reg = 2.0
    features["vix_regime"] = vix_reg

    features["pe_attractive"]   = round(max(0.0, 1.0 - pe_vf), 4)
    features["drop_x_drawdown"] = round(drop * dd52 / 100.0, 4)
    features["vol_x_drop"]      = round(volsp * abs(drop), 4)

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    """
    ATR(14) / current_price.
    price_history must have columns: High, Low, Close (daily OHLC).
    Returns fallback if history is insufficient.
    """
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
    """
    Today's volume / 20-day average volume.
    Tries price_history first, falls back to fundamentals dict keys.
    """
    # Try from OHLCV history
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

    # Fallback: fundamentals dict
    vol     = _safe_float(fund.get("volume"))
    avg_vol = _safe_float(fund.get("average_volume"))
    if not math.isnan(vol) and not math.isnan(avg_vol) and avg_vol > 0:
        return round(vol / avg_vol, 4)

    return _FALLBACK["volume_spike"]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    ticker: str,
    fundamentals: dict,
    price_history: Optional[pd.DataFrame] = None,
    sector: str = "Unknown",
    drop_pct_today: Optional[float] = None,
    label_win: Optional[int] = None,
    label_further_drop: Optional[float] = None,
    macro_context: Optional[dict] = None,
) -> dict:
    """
    Build the complete feature vector for one stock at one point in time.

    Parameters
    ----------
    ticker          : str       Stock ticker (used only for logging)
    fundamentals    : dict      Output of market_client.get_fundamentals() or equivalent
    price_history   : DataFrame Optional OHLCV DataFrame (daily, columns: Open/High/Low/Close/Volume)
                                Required for ATR computation; volume_spike falls back to fund dict
    sector          : str       GICS sector string (e.g. "Technology")
    drop_pct_today  : float     % price change that triggered the dip alert (negative)
                                If None, read from fundamentals.get('change_pct')
    label_win       : int|None  1 if recovery ≥15% in 60d. None during production inference.
    label_further_drop: float|None  max additional drop before recovery. None in production.
    macro_context   : dict|None Pre-computed macro dict (avoids duplicate API call per ticker
                                when scanning a full universe). If None, fetched automatically.

    Returns
    -------
    dict with keys = FEATURE_COLUMNS + LABEL_COLUMNS
    All feature values are float. Labels are int/float/None.
    """
    fund = fundamentals  # alias for brevity

    # ── Stage 0: Macro ────────────────────────────────────────────────
    if macro_context is None:
        macro_context = get_macro_context(sector=sector)

    macro_score       = float(macro_context.get("macro_score",       _FALLBACK["macro_score"]))
    vix               = float(macro_context.get("vix",               _FALLBACK["vix"]))
    spy_drawdown_5d   = float(macro_context.get("spy_drawdown_5d",   _FALLBACK["spy_drawdown_5d"]))
    sector_drawdown_5d = float(macro_context.get("sector_drawdown_5d", _FALLBACK["sector_drawdown_5d"]))

    # ── Stage 1: Quality / Value ─────────────────────────────────────
    fcf_yield      = _sf(fund.get("fcf_yield"),      "fcf_yield")
    revenue_growth = _sf(fund.get("revenue_growth"), "revenue_growth")
    gross_margin   = _sf(fund.get("gross_margin"),   "gross_margin")
    de_ratio       = _sf(fund.get("debt_equity"),    "de_ratio")
    analyst_upside = _sf(fund.get("analyst_upside") or fund.get("upside"), "analyst_upside")

    # P/E vs fair: ratio < 1 = undervalued vs sector, > 1 = overvalued
    pe_raw   = _safe_float(fund.get("pe"))
    fair_pe  = _SECTOR_FAIR_PE.get(sector, 22.0)
    if not math.isnan(pe_raw) and pe_raw > 0 and fair_pe > 0:
        pe_vs_fair = round(pe_raw / fair_pe, 4)
    else:
        pe_vs_fair = _FALLBACK["pe_vs_fair"]
        logger.debug(f"ml_features [{ticker}]: pe_vs_fair fallback (pe={pe_raw}, fair={fair_pe})")

    # Quality score from score.py (hemisphere A, normalised [0,1])
    try:
        score_result = score_from_fundamentals(fund)
        quality_score = float(score_result["quality_score"])
        if not math.isfinite(quality_score):
            quality_score = _FALLBACK["quality_score"]
    except Exception as e:
        logger.warning(f"ml_features [{ticker}]: score_from_fundamentals failed: {e}")
        quality_score = _FALLBACK["quality_score"]

    # ── Stage 2: Timing ────────────────────────────────────────────────
    # drop_pct_today: prefer explicit argument, fallback to fund dict
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

    rsi_14 = _sf(fund.get("rsi") or fund.get("rsi_14"), "rsi_14")
    # Clamp RSI to valid range
    rsi_14 = float(np.clip(rsi_14, 0.0, 100.0))

    # Current price for ATR normalisation
    current_price = _safe_float(fund.get("price") or fund.get("current_price"))
    if math.isnan(current_price) or current_price <= 0:
        # Try to read from price_history last close
        if price_history is not None and len(price_history) > 0:
            try:
                current_price = float(price_history["Close"].dropna().iloc[-1])
            except Exception:
                current_price = 1.0
        else:
            current_price = 1.0

    atr_ratio    = _compute_atr_ratio(price_history, current_price)
    volume_spike = _compute_volume_spike(price_history, fund)

    # ── Assemble vector ─────────────────────────────────────────────────
    feature_vector = {
        # Stage 0
        "macro_score":          macro_score,
        "vix":                  vix,
        "spy_drawdown_5d":      spy_drawdown_5d,
        "sector_drawdown_5d":   sector_drawdown_5d,
        # Stage 1
        "fcf_yield":            fcf_yield,
        "revenue_growth":       revenue_growth,
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
    }
    # Stage 3 — engineered/derived features (shared with training pipeline)
    add_derived_features(feature_vector)

    # Labels (None in production, int/float during training)
    feature_vector["label_win"]          = label_win
    feature_vector["label_further_drop"] = label_further_drop

    # ── Sanity check ─────────────────────────────────────────────────
    # Verify no NaN/Inf leaked into features (labels can be None)
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
        f"drop={drop_pct_today:.1f}%"
    )

    return feature_vector


def build_feature_row(feature_vector: dict) -> list[float]:
    """
    Extract features in the canonical FEATURE_COLUMNS order as a flat list.
    Used when feeding a single observation to a trained model.

    Labels are excluded — this is the model input only.
    """
    return [float(feature_vector[col]) for col in FEATURE_COLUMNS]


def build_feature_df(rows: list[dict]) -> pd.DataFrame:
    """
    Convert a list of feature vectors (from build_features) into a DataFrame
    with the correct column order. Includes label columns if present.

    Used in training scripts to build the training matrix.
    """
    all_cols = FEATURE_COLUMNS + LABEL_COLUMNS
    return pd.DataFrame(rows, columns=all_cols)


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test  (python ml_features.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.DEBUG)

    print(f"FEATURE_COLUMNS ({N_FEATURES}): {FEATURE_COLUMNS}")
    print()

    # Mock fundamentals (as returned by market_client.get_fundamentals)
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

    # Mock macro (avoids live API call in test)
    mock_macro = {
        "regime":              "NEUTRAL",
        "macro_score":         2,
        "vix":                 18.2,
        "vix_pct_1m":          5.3,
        "spy_drawdown_5d":    -1.2,
        "sector_drawdown_5d": -2.8,
        "fred_recession_prob": 0.31,
        "sector_etf":          "XLK",
    }

    fv = build_features(
        ticker="MSFT",
        fundamentals=mock_fund,
        sector="Technology",
        drop_pct_today=-8.4,
        macro_context=mock_macro,
        label_win=1,
        label_further_drop=-4.2,
    )

    print("Feature vector:")
    for k, v in fv.items():
        print(f"  {k:25s}: {v}")

    print()
    row = build_feature_row(fv)
    print(f"Flat row ({len(row)} values): {row}")

    print()
    df = build_feature_df([fv, fv])
    print(f"DataFrame shape: {df.shape}")
    print(df.dtypes)
