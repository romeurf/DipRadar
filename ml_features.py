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
                        sector-relative (3m and 6m), beta_90d, vol_of_vol
  Stage 3c — Dislocation: quality_dislocation, peg_ratio, relative_drop,
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

# FEATURE_COLUMNS — the DIP model's feature schema.
#
# Each entry MUST be in FEATURE_SETS["dip_us"] below; the two are the single
# source of truth for what build_dip_dataset writes and ml_predictor reads.
# Momentum and Core models live in their own self-contained worlds with their
# own feature lists (momentum_radar/trainer.FEATURE_COLS,
# scripts/train_core_model.CORE_FEATURE_COLS) — they never touch
# FEATURE_COLUMNS or FEATURE_SETS.
#
# Audit policy: any entry here that doesn't appear in FEATURE_SETS["dip_us"]
# is dead code and should be removed; any entry there that isn't here is
# a bug.
FEATURE_COLUMNS: list[str] = [
    # Macro / regime
    "vix",
    "vix_regime",
    "vix_percentile_1y",
    "spy_rsi_14",
    "yield_10y_change_5d",
    # Drop / timing
    "drop_pct_today",
    "drawdown_52w",
    "rsi_14",
    "bb_width",
    "rsi_oversold_strength",
    "drop_x_drawdown",
    "vol_x_drop",
    # Pre-drop trend context
    "return_1m",
    "return_3m_pre",
    "sector_relative",
    # Volume / structure (capitulation + stop-hunt signals)
    "volume_zscore_20d",
    "close_in_range_20d",
    "true_range_pct_20d",
    "vwap_from_high_ratio",
    "volume_poc_ratio",
    "liquidity_sweep_score",
    "mfi_14",
    # Sector-relative oversold
    "rsi_sector_adjusted",
    "drawdown_sector_normalized",
    # Entry zone / valuation
    "consecutive_red_days",
    "ma_200d_ratio",
    "price_vs_52w_percentile",
    "peg_ratio",
    # SEC Filings features (Form 4 insider buys/sells/amounts, 8-K material
    # events, earnings call tone) REMOVED — pipeline produced too many false
    # zeros and the technical setup is the dominant signal. Columns stay in
    # the parquet schema for backward compat; trainer ignores them.
    # Earnings quality + sell-side view
    "earnings_beat_rate",
    "analyst_rating",
    # Dip maturity
    "days_below_ma200",
    "decline_deceleration",
    # Calendar / sector pressure
    "month_of_year",
    "sector_alert_count_7d",
]
# Dead features REMOVED 2026-05-30 (were computed but no model trained on them;
# the audit at the top of FEATURE_SETS explains the rationale):
#   return_6m_pre, beta_90d, vol_of_vol, up_days_pct_20d, short_interest_pct,
#   vix_sector_signal, days_to_next_earnings, sector_momentum_3m,
#   pe_vs_sector_median, dividend_growth_3y, volume_up_pct_20d
#   (the last 3 are still used by Core / Momentum trainers via their own lists;
#    they aren't DIP features so they don't belong in FEATURE_COLUMNS).

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

_SECTOR_MEDIAN_FWD_PE: dict[str, float] = {
    "Technology": 28.0,
    "Healthcare": 20.0,
    "Communication Services": 20.0,
    "Financial Services": 13.0,
    "Financials": 13.0,
    "Consumer Cyclical": 18.0,
    "Consumer Defensive": 20.0,
    "Industrials": 18.0,
    "Energy": 11.0,
    "Utilities": 17.0,
    "Real Estate": 35.0,
    "Basic Materials": 13.0,
    "Materials": 13.0,
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
    "beta_90d": 1.0,
    "vol_of_vol": 0.01,
    "peg_ratio": 2.0,
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
    # PR #30 (Phase B): raw OHLCV features (sliced point-in-time <= alert_date)
    "volume_zscore_20d": 0.0,    # neutral volume vs baseline
    "close_in_range_20d": 0.5,   # 0.5 = neutral (closing mid-range)
    "up_days_pct_20d": 0.5,      # 50% — random walk baseline
    "true_range_pct_20d": 0.02,  # 2% intraday — typical equity
    # Fase 5: sinais de qualidade e sentimento
    "consecutive_red_days": 3.0,      # 3 dias consecutivos — neutro
    "ma_200d_ratio": 0.90,            # 10% below MA200 — typical dip
    # SEC Filings fallbacks (insider, 8-K, earnings call tone) REMOVED.
    # Columns retained in some legacy parquets as NaN; trainer no longer
    # references them.
    "earnings_beat_rate": 0.50,        # 50% — neutro (random)
    "analyst_rating": 2.5,             # entre Buy e Hold — neutro
    "short_interest_pct": 0.05,        # 5% — typical short interest
    # Sector-conditioned features (v4.2)
    "rsi_sector_adjusted": 0.0,       # 0 = RSI na norma do sector (sem sinal de oversold)
    "vix_sector_signal": 0.0,         # 0 = VIX neutro ou sector tipo desconhecido
    "drawdown_sector_normalized": -1.0,  # -1 = mild drawdown (half of sector typical)
    # Market Structure (v4.3)
    "vwap_from_high_ratio": -0.12,   # ~12% below anchored VWAP since peak — typical dip
    "volume_poc_ratio": 0.0,         # neutro — sem info de volume profile
    "liquidity_sweep_score": 0.0,    # sem sweep recente — caso mais comum
    # Volume Flow
    "mfi_14": 50.0,                  # neutral — no directional pressure
    "volume_up_pct_20d": 0.50,       # 50% — random walk base rate
    "price_vs_52w_percentile": 0.5,  # mid-range — neutral when 52w range is unknown
    # Dip maturity
    "days_below_ma200": 30.0,        # neutral: been below MA200 for ~1 month
    "decline_deceleration": 1.0,     # neutral: same monthly rate (no decel/accel)
    # New features (tasks 3-6)
    "days_to_next_earnings": 45.0,   # 45d = typical mid-quarter distance
    "pe_vs_sector_median": 1.0,      # 1.0 = at sector median (no discount/premium)
    "dividend_growth_3y": 0.0,       # 0.0 = no dividend history (most common)
    "sector_momentum_3m": 0.0,       # 0.0 = flat sector (neutral)
}

def assert_feature_completeness(row: dict, context: str = "") -> None:
    """Raise ValueError if any FEATURE_COLUMNS key is missing from row."""
    missing = [col for col in FEATURE_COLUMNS if col not in row]
    if missing:
        raise ValueError(
            f"Feature schema violation [{context}]: "
            f"{len(missing)} columns missing: {missing}"
        )


FEATURE_SETS: dict[str, list[str]] = {
    # DIP — single source of truth: matches FEATURE_COLUMNS above verbatim.
    # If you add a feature here, add it to FEATURE_COLUMNS too (and vice versa).
    # The two-list redundancy is intentional: FEATURE_COLUMNS is the runtime
    # schema (used by row_base init / inference init / assert_feature_completeness);
    # FEATURE_SETS["dip_us"] is what get_feature_columns() returns to the trainer.
    "dip_us": list(FEATURE_COLUMNS),
    # CORE — mirrors scripts/train_core_model.CORE_FEATURE_COLS verbatim.
    # The Core trainer doesn't read this; this exists for /admin_features and
    # any external introspection. Keep in sync with the trainer.
    "core_us": [
        # Price / MA signals
        "price_vs_ma200w", "ma200d_trend_60d", "return_20d", "return_5d",
        "dd_from_52w_high", "rsi_14",
        # Volume / institutional flow
        "obv_trend_20d", "up_down_vol_ratio_20d",
        "high_vol_up_days_20d", "vol_spike_5d_vs_60d",
        # Sector context
        "sector_etf_vs_ma200w", "stock_vs_sector_spread", "sector_return_20d",
        # User signal
        "annual_yield_pct",
        # Fundamentals (PIT via SEC EDGAR XBRL)
        "revenue_growth", "gross_margin", "roe", "fcf_yield_fund",
        # Relative valuation + dividend quality
        "pe_vs_sector_median", "dividend_growth_3y",
    ],
    # The momentum model uses its own trainer at momentum_radar/trainer.FEATURE_COLS.
    # This entry mirrors that list verbatim so /admin_features and any external
    # introspection match what's actually trained.
    "momentum_us": [
        # Momentum trend
        "return_20d", "return_5d", "return_60d_pre",
        "volume_ratio_20d", "rsi_14",
        "pct_from_52w_high", "atr_pct", "close_in_range_20d",
        # Market structure
        "vwap_from_low_ratio", "volume_poc_ratio",
        # Volume flow
        "mfi_14", "volume_up_pct_20d",
        # SEC Filings (insider Form 4) removed — see DIP FEATURE_COLUMNS.
    ],
}


def get_feature_columns(model_id: str) -> list[str]:
    """Return the ordered feature list for a given model_id."""
    if model_id not in FEATURE_SETS:
        raise KeyError(f"Unknown model_id: '{model_id}'. Available: {list(FEATURE_SETS)}")
    return list(FEATURE_SETS[model_id])


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

# Sector → dynamics type mapping for sector-conditioned features.
# The model learns that VIX=35 means different things depending on the sector.
_SECTOR_TYPE: dict[str, str] = {
    "Technology":             "growth",
    "Healthcare":             "growth",
    "Communication Services": "growth",
    "Financial Services":     "cyclical",
    "Financials":             "cyclical",
    "Industrials":            "cyclical",
    "Real Estate":            "cyclical",
    "Energy":                 "commodity",
    "Basic Materials":        "commodity",
    "Consumer Defensive":     "defensive",
    "Utilities":              "defensive",
    "Consumer Cyclical":      "cyclical",   # cyclical by definition — reacts negatively to VIX
    "Unknown":                "cyclical",   # fallback neutro
}

# "Oversold" threshold varies by sector:
# Growth: RSI<35 is a strong signal (tech recovers quickly from panic selling)
# Commodity: RSI<30 required (move on longer cycles)
# Defensive: RSI<40 sufficient (naturally lower volatility)
_RSI_OVERSOLD_THRESHOLD: dict[str, float] = {
    "growth":    35.0,
    "cyclical":  32.0,
    "commodity": 28.0,
    "defensive": 40.0,
}


def add_derived_features(features: dict, alert_date: Optional[Any] = None) -> dict:
    """Compute derived features (interactions) + sector-conditioned features.

    Sector-conditioned features: the same signal (VIX, RSI, drawdown) carries
    different weights by sector type. Instead of separate sector models
    (impossible with ~46 rows/stock on average), we add sector-conditioned
    versions that the global model can learn to weight appropriately.

    Example: RSI<30 in tech → strong signal (temporary panic).
             RSI<30 in energy → weak signal (may reflect crude oil decline).
    """
    rsi = float(features.get("rsi_14", _FALLBACK["rsi_14"]))
    vix = float(features.get("vix", _FALLBACK["vix"]))
    pe_vf = float(features.get("pe_vs_fair", _FALLBACK["pe_vs_fair"]))
    drop = float(features.get("drop_pct_today", _FALLBACK["drop_pct_today"]))
    dd52 = float(features.get("drawdown_52w", _FALLBACK["drawdown_52w"]))
    volsp = float(features.get("volume_spike", _FALLBACK["volume_spike"]))

    # ── Features derivadas standard ───────────────────────────────────────────
    features["rsi_oversold_strength"] = round(max(0.0, 40.0 - rsi), 4)
    features["vix_regime"] = 0.0 if vix < 15.0 else (1.0 if vix < 25.0 else 2.0)
    features["pe_attractive"] = round(max(0.0, 1.0 - pe_vf), 4)
    features["drop_x_drawdown"] = round(drop * dd52 / 100.0, 4)
    features["vol_x_drop"] = round(volsp * abs(drop), 4)
    # Standard PEG: trailing_PE / EPS_growth% (Lynch: <1 undervalued, >2 expensive)
    # earnings_growth preferred; fall back to revenue_growth if not available.
    # If neither PE nor growth is available → NaN (not _FALLBACK 2.0). The model
    # treats NaN as "missing data" (native in LGBM, mean-imputed in Ridge via
    # the bundle's col_means at inference) instead of misreading the constant
    # as a real "neutral PEG" signal. Affects pre-2003 episodes and any ticker
    # without enough quarterly history.
    _tpe = features.get("trailing_pe", float("nan"))
    _eg  = features.get("earnings_growth", features.get("revenue_growth", float("nan")))
    _tpe = float(_tpe) if _tpe is not None else float("nan")
    _eg  = float(_eg)  if _eg  is not None else float("nan")
    if math.isfinite(_tpe) and _tpe > 0 and math.isfinite(_eg) and _eg > 0.005:
        features["peg_ratio"] = round(float(min(_tpe / (_eg * 100.0), 10.0)), 4)
    else:
        features["peg_ratio"] = float("nan")
    # pe_vs_sector_median: stock forward PE relative to sector median forward PE.
    # <1.0 = trading at a discount to sector peers (potential entry for Core).
    # Same NaN-propagation policy as peg_ratio.
    _fwd_pe = features.get("forward_pe")
    _fwd_pe = float(_fwd_pe) if _fwd_pe is not None else float("nan")
    sector_for_pe = str(features.get("sector", "Unknown") or "Unknown")
    _sec_med_pe = _SECTOR_MEDIAN_FWD_PE.get(sector_for_pe, 18.0)
    if math.isfinite(_fwd_pe) and _fwd_pe > 0 and _sec_med_pe > 0:
        features["pe_vs_sector_median"] = round(float(min(_fwd_pe / _sec_med_pe, 5.0)), 4)
    elif "pe_vs_sector_median" not in features:
        features["pe_vs_sector_median"] = float("nan")

    sec_dd = float(features.get("sector_drawdown_5d", _FALLBACK["sector_drawdown_5d"]))
    features["relative_drop"] = round(drop - sec_dd, 4)
    try:
        features["month_of_year"] = float(pd.Timestamp(alert_date).month) if alert_date is not None else float(datetime.now().month)
    except Exception:
        features["month_of_year"] = float(datetime.now().month)

    # ── Sector-conditioned features ───────────────────────────────────────────
    # The global model learns different weights for the same signals depending
    # on sector type. Not separate models, but the model learns that
    # VIX=35 in growth ≠ VIX=35 in commodity.
    sector   = str(features.get("sector", "Unknown") or "Unknown")
    sec_type = _SECTOR_TYPE.get(sector, "cyclical")
    rsi_thr  = _RSI_OVERSOLD_THRESHOLD.get(sec_type, 32.0)

    # RSI sector-adjusted: how oversold relative to sector threshold
    # growth: RSI<35 = "very oversold"; commodity: RSI<28 = "very oversold"
    features["rsi_sector_adjusted"] = round(max(0.0, rsi_thr - rsi), 4)

    # VIX × sector_type: high volatility is positive for defensives (flight to safety)
    # but negative for growth (risk-off → sell tech).
    _vix_factor = {"growth": -1.0, "cyclical": -0.5, "commodity": 0.0, "defensive": +0.5}
    features["vix_sector_signal"] = round(vix * _vix_factor.get(sec_type, -0.5) / 40.0, 4)

    # Drawdown normalised by sector: in energy, -30% drawdown is common;
    # in tech, -30% is extreme. Normalise by sector's typical drawdown.
    _dd_typical = {"growth": -25.0, "cyclical": -30.0, "commodity": -40.0, "defensive": -20.0}
    dd_norm = dd52 / abs(_dd_typical.get(sec_type, -30.0))
    features["drawdown_sector_normalized"] = round(float(max(-3.0, min(0.0, dd_norm))), 4)

    # decline_deceleration: recent monthly loss vs average monthly loss over 3m.
    # Formula: return_1m / (return_3m_pre / 3)
    # <1 = slowing down (good — e.g. NVO Mar/Apr: 3m=-30%, 1m=-5% → 0.5)
    # >1 = accelerating (bad — e.g. NVO Feb: 3m=-5%, 1m=-15% → 3.0)
    # <0 = recovering (best — 1m positive while 3m negative)
    # Neutral fallback=1.0 when 3m context is flat (no trend information).
    try:
        r1m  = float(features.get("return_1m",    _FALLBACK["return_1m"]))
        r3m  = float(features.get("return_3m_pre", _FALLBACK["return_3m_pre"]))
        avg_monthly_3m = r3m / 3.0
        if abs(avg_monthly_3m) >= 0.005:  # only meaningful if 3m had at least 0.5%/mo move
            ratio = r1m / avg_monthly_3m
            features["decline_deceleration"] = round(float(np.clip(ratio, -5.0, 5.0)), 4)
        # else: keep fallback 1.0 (flat 3m baseline — no decel signal)
    except Exception:
        pass

    return features


# ────────────────────────────────────────────────────────────────────────────────
# build_features — thin orchestrator used by position_monitor for daily
# re-scoring of active positions. Replicates the legacy contract:
#   build_features(ticker, fundamentals) → dict with all FEATURE_COLUMNS.
# Without price_history/sector → technical and momentum features fall back to
# deterministic _FALLBACK values (acceptable for daily position monitoring).
# ────────────────────────────────────────────────────────────────────────────────
def build_features(
    ticker: str,
    fundamentals: dict,
    *,
    sector: str = "Unknown",
    macro_context: Optional[dict] = None,
    alert_date: Optional[Any] = None,
    price_history: Optional[pd.DataFrame] = None,
) -> dict:
    """Build a feature dict for inference (predict_dip / ml_score).

    Designed for the **position_monitor** path — daily re-scoring of active
    positions without price_history or sector ETF available. For historical
    training or universe snapshots use ``ml_training.data.build_dataset`` or
    ``universe_snapshot._build_snapshot_row`` (which build the dict manually
    and call only ``add_derived_features``).

    Parameters
    ----------
    ticker         : str   ticker identifier (used in log messages only).
    fundamentals   : dict  output of ``_fetch_fundamentals_snapshot`` or
                           equivalent. Optional keys: ``pe``, ``debt_equity``,
                           ``revenue_growth``, ``gross_margin``, ``fcf_yield``,
                           ``analyst_upside``, ``price``, ``market_cap``.
    sector         : str   GICS sector (defaults "Unknown" → fair PE 22.0).
    macro_context  : dict  pre-computed ``get_macro_context`` output (avoids
                           an extra API call). ``None`` = calls get_macro_context.
    alert_date     : Any   alert timestamp (used for ``month_of_year``).
                           ``None`` = today.

    Returns
    -------
    dict with all FEATURE_COLUMNS populated (fundamental values from the dict;
    technical/momentum/regime features at fallback defaults).
    """
    fund = fundamentals or {}

    # Stage 0 — Macro
    if macro_context is None:
        try:
            macro_context = get_macro_context(sector=sector)
        except Exception as e:
            logger.debug(f"build_features[{ticker}]: get_macro_context failed: {e}")
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

    # Stage 1 — Quality / Value (from fundamentals dict)
    def _set(col: str, val: Any) -> None:
        if val is None:
            return
        try:
            f = float(val)
        except (TypeError, ValueError):
            return
        if math.isfinite(f):
            features[col] = f

    _set("gross_margin",    fund.get("gross_margin"))
    _set("de_ratio",        fund.get("debt_equity") or fund.get("de_ratio"))
    _set("analyst_upside",  fund.get("analyst_upside") or fund.get("upside"))
    _set("fcf_yield",       fund.get("fcf_yield"))
    # Inputs for standard PEG and pe_vs_sector_median (read by add_derived_features)
    _set("trailing_pe",     fund.get("trailingPE") or fund.get("pe"))
    _set("earnings_growth", fund.get("earningsGrowth") or fund.get("epsGrowth"))
    _set("forward_pe",      fund.get("forwardPE") or fund.get("forward_pe"))

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
        logger.debug(f"build_features[{ticker}]: score_from_fundamentals failed: {e}")

    # Stage 2 — Timing (from dict; without price_history → fallback)
    _set("drop_pct_today", fund.get("drop_pct_today") or fund.get("change_pct"))
    _set("drawdown_52w",   fund.get("drawdown_52w"))
    _set("rsi_14",         fund.get("rsi_14") or fund.get("rsi"))
    _set("atr_ratio",      fund.get("atr_ratio"))
    _set("volume_spike",   fund.get("volume_spike"))

    # Stage 3 — Derived (interactions). Overwrites derived features from the
    # current dict values; remainder stays at fallback.
    add_derived_features(features, alert_date=alert_date)

    # Stage 4 — Raw OHLCV + Market Structure. Opt-in: only runs when the
    # caller passes price_history; otherwise falls back to neutral defaults.
    if price_history is not None:
        add_raw_ohlcv_features(features, price_history)
        add_market_structure_features(features, price_history)
        add_volume_flow_features(features, price_history)

    # Stage 5 — Quality and sentiment signals.
    # NaN where unavailable — model treats NaN as "missing information",
    # not as a neutral invented value.
    try:
        from fundamental_signals import compute_fundamental_signals
        signals = compute_fundamental_signals(
            ticker,
            price_history=price_history,
            alert_date=alert_date,
        )
        for key, val in signals.items():
            if key not in FEATURE_COLUMNS:
                continue
            try:
                vf = float(val)
                if math.isfinite(vf):
                    features[key] = vf
                # If NaN: keep existing fallback in features[] rather than replacing with NaN
                # (model was trained on fallback values, not NaN)
            except (TypeError, ValueError):
                pass
    except Exception as e:
        logger.error(f"build_features[{ticker}]: fundamental_signals failed: {e}")

    # Stage 6 — Sector pressure context.
    # sector_alert_count_7d is computed post-hoc at training time (pandas
    # window across the parquet). At inference time we have to source the
    # same signal differently: count of bot alerts in the same sector over
    # the past 7 calendar days from alert_db. Without this, the model would
    # see the value at training (real variance) and 0 at inference (every
    # call) — classic train/inference skew. We compute it once here so the
    # call site doesn't have to repeat.
    try:
        from datetime import datetime as _dt_cls, timedelta as _td
        from pathlib import Path as _AdbP

        _data_dir = _AdbP("/data") if _AdbP("/data").exists() else _AdbP("/tmp")
        _adb_path = _data_dir / "alert_db.csv"
        if _adb_path.exists() and sector and sector != "Unknown":
            _ref_date = pd.Timestamp(alert_date) if alert_date is not None else pd.Timestamp(_dt_cls.utcnow())
            _ref_date = _ref_date.tz_localize(None) if getattr(_ref_date, "tz", None) is not None else _ref_date
            _window_start = _ref_date - pd.Timedelta(days=7)

            try:
                _adb = pd.read_csv(_adb_path, usecols=["sector", "date_iso"], engine="python", on_bad_lines="skip")
            except Exception:
                try:
                    _adb = pd.read_csv(_adb_path, engine="python", on_bad_lines="skip")
                    _adb = _adb[[c for c in ("sector", "date_iso") if c in _adb.columns]]
                except Exception:
                    _adb = None

            if _adb is not None and not _adb.empty and {"sector", "date_iso"} <= set(_adb.columns):
                _adb["date_iso"] = pd.to_datetime(_adb["date_iso"], errors="coerce")
                _mask = (
                    _adb["sector"].astype(str).str.casefold() == str(sector).casefold()
                ) & (
                    _adb["date_iso"] >= _window_start
                ) & (
                    _adb["date_iso"] <= _ref_date
                )
                features["sector_alert_count_7d"] = float(int(_mask.sum()))
            # else: leave at the _FALLBACK 0.0 set at dict init
    except Exception as _e:
        logger.debug(f"build_features[{ticker}]: sector_alert_count_7d failed: {_e}")

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

    # days_to_next_earnings: calendar days to next scheduled earnings release.
    # Uses earningsDate list from yfinance info (list of upcoming timestamps).
    # Capped at 90 — beyond that, proximity no longer affects entry risk.
    features.setdefault("days_to_next_earnings", _FALLBACK["days_to_next_earnings"])
    try:
        now_ne = _tz_normalize(alert_date) if alert_date is not None else pd.Timestamp.now().normalize()
        ed = info.get("earningsDate")
        if isinstance(ed, list) and ed:
            best_diff = None
            for ts in ed:
                try:
                    dt = (pd.Timestamp.fromtimestamp(int(ts)) if isinstance(ts, (int, float))
                          else pd.Timestamp(ts))
                    dt_norm = _tz_normalize(dt)
                    diff = (dt_norm - now_ne).days
                    if 0 <= diff and (best_diff is None or diff < best_diff):
                        best_diff = diff
                except Exception:
                    pass
            if best_diff is not None:
                features["days_to_next_earnings"] = float(min(best_diff, 90))
    except Exception:
        pass

    return features

def add_raw_ohlcv_features(features: dict, price_history: Optional[pd.DataFrame]) -> dict:
    """Compute 4 raw-OHLCV features from a point-in-time price slice.

    All features assume ``price_history`` is already sliced to dates
    ``<= alert_date`` (no look-ahead) and indexed by date in ascending order.
    Falls back to neutral defaults if the slice is too short.

    Features chosen for orthogonality to the existing top features
    (vol_of_vol, bb_width, drawdown_52w, return_1m/3m/6m):

      • ``volume_zscore_20d`` — z-score of today's volume against the prior
        20 sessions. Institutional flow / capitulation. Bounded to ±10.
      • ``close_in_range_20d`` — mean of ``(Close − Low) / (High − Low)``
        across the last 20 sessions. Captures *where* in the daily range
        the stock typically closes (>0.5 → strong closes; <0.5 → weak).
        Orthogonal to close-to-close vol.
      • ``up_days_pct_20d`` — fraction of the last 20 sessions with strictly
        positive daily return. Captures trend *persistence* (regular up-days
        vs occasional big rallies). Bounded [0, 1].
      • ``true_range_pct_20d`` — mean ATR-like intraday range as a percent
        of close. Captures intraday volatility (different from close-to-close).
    """
    out_keys = (
        "volume_zscore_20d", "close_in_range_20d", "up_days_pct_20d",
        "true_range_pct_20d", "price_vs_52w_percentile", "days_below_ma200",
    )
    for k in out_keys:
        features.setdefault(k, _FALLBACK[k])

    if price_history is None or price_history.empty:
        return features
    hist = _normalize_index(price_history)
    if "Close" not in hist.columns or len(hist) < 6:
        return features

    close = hist["Close"].astype(float)

    # volume_zscore_20d
    try:
        if "Volume" in hist.columns:
            vol = hist["Volume"].astype(float).tail(21)
            if len(vol) == 21:
                today_v = float(vol.iloc[-1])
                window = vol.iloc[:-1]
                mu = float(window.mean())
                sd = float(window.std())
                if sd > 0 and math.isfinite(today_v) and math.isfinite(mu):
                    z = (today_v - mu) / sd
                    if math.isfinite(z):
                        features["volume_zscore_20d"] = float(np.clip(z, -10.0, 10.0))
    except Exception:
        pass

    # close_in_range_20d: requer High e Low
    try:
        if "High" in hist.columns and "Low" in hist.columns:
            tail = hist.tail(20)
            high = tail["High"].astype(float)
            low = tail["Low"].astype(float)
            close_t = tail["Close"].astype(float)
            rng = high - low
            mask = (rng > 0) & np.isfinite(rng) & np.isfinite(close_t)
            if mask.sum() >= 5:
                cir = (close_t[mask] - low[mask]) / rng[mask]
                v = float(cir.mean())
                if math.isfinite(v):
                    features["close_in_range_20d"] = float(np.clip(v, 0.0, 1.0))
    except Exception:
        pass

    # up_days_pct_20d
    try:
        rets = close.pct_change().tail(20).dropna()
        if len(rets) >= 5:
            features["up_days_pct_20d"] = float((rets > 0).mean())
    except Exception:
        pass

    # true_range_pct_20d
    try:
        if "High" in hist.columns and "Low" in hist.columns:
            tail = hist.tail(21)  # need prev close → 21 rows for 20 TR values
            high = tail["High"].astype(float)
            low = tail["Low"].astype(float)
            close_t = tail["Close"].astype(float)
            prev_close = close_t.shift(1)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).iloc[1:]  # drop first NaN
            close_for_div = close_t.iloc[1:]
            mask = (close_for_div > 0) & np.isfinite(tr) & np.isfinite(close_for_div)
            if mask.sum() >= 5:
                tr_pct = (tr[mask] / close_for_div[mask])
                v = float(tr_pct.mean())
                if math.isfinite(v):
                    features["true_range_pct_20d"] = float(np.clip(v, 0.0, 1.0))
    except Exception:
        pass

    # price_vs_52w_percentile: where is price in the 52-week High/Low range?
    # 0.0 = at 52w low, 1.0 = at 52w high. Useful for entry zone assessment.
    try:
        if "High" in hist.columns and "Low" in hist.columns and len(hist) >= 10:
            high_52w = float(hist["High"].rolling(min(252, len(hist))).max().iloc[-1])
            low_52w  = float(hist["Low"].rolling(min(252, len(hist))).min().iloc[-1])
            rng = high_52w - low_52w
            if rng > 0 and math.isfinite(rng):
                price = float(hist["Close"].iloc[-1])
                features["price_vs_52w_percentile"] = round(
                    float(np.clip((price - low_52w) / rng, 0.0, 1.0)), 4
                )
    except Exception:
        pass

    # days_below_ma200: consecutive trading days with Close < 200d MA.
    # 0 = just crossed below (or above), 90+ = holders exhausted / washed out.
    # Distinguishes NVO-Feb (5 days below, risky entry) from NVO-Apr (90 days, better entry).
    # Uses min_periods=40 so partial history still yields a usable signal.
    try:
        if len(hist) >= 40:
            ma200 = close.rolling(min(200, len(hist)), min_periods=40).mean()
            # Count consecutive sessions below MA200 from the most recent bar backwards
            below = (close < ma200).values
            count = 0
            for i in range(len(below) - 1, -1, -1):
                if below[i] and math.isfinite(float(ma200.iloc[i])):
                    count += 1
                else:
                    break
            features["days_below_ma200"] = float(min(count, 365))
    except Exception:
        pass

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

def _anchored_vwap_ratio(hist: pd.DataFrame, use_high: bool = True) -> float:
    """VWAP ancorado no máximo (use_high=True) ou mínimo (False) de 52 semanas.

    Para dips: ancoramos no HIGH — mede o desconto face ao custo médio
    das instituições que compraram desde o topo (quase sempre negativo).
    Para momentum: ancoramos no LOW — mede quanto subiu acima do custo
    médio desde o fundo (positivo = momentum confirmado).
    """
    if hist is None or len(hist) < 10 or "Volume" not in hist.columns:
        return 0.0
    try:
        lookback = hist.iloc[-min(252, len(hist)):]
        anchor_idx = lookback["Close"].idxmax() if use_high else lookback["Close"].idxmin()
        sliced = hist[hist.index >= anchor_idx]
        if len(sliced) < 2:
            return 0.0
        tp  = (sliced["High"] + sliced["Low"] + sliced["Close"]) / 3
        vol = sliced["Volume"].clip(lower=0)
        if vol.sum() <= 0:
            return 0.0
        vwap  = float((tp * vol).sum() / vol.sum())
        price = float(hist["Close"].iloc[-1])
        return round((price - vwap) / vwap, 4) if vwap > 0 else 0.0
    except Exception:
        return 0.0


def _volume_poc_ratio(hist: pd.DataFrame, lookback: int = 90, n_bins: int = 24) -> float:
    """Rácio (preço − POC) / POC onde POC é o Point of Control do Volume Profile.

    O POC é o preço onde mais volume negociou nos últimos `lookback` dias.
    Negativo = preço abaixo do POC (overhead supply); positivo = acima.
    Computado via distribuição proporcional do volume diário pelos bins de preço.
    """
    if hist is None or len(hist) < 10 or "Volume" not in hist.columns:
        return 0.0
    try:
        recent    = hist.iloc[-min(lookback, len(hist)):]
        lows      = recent["Low"].values.astype(float)
        highs     = recent["High"].values.astype(float)
        vols      = recent["Volume"].values.astype(float)
        p_min, p_max = lows.min(), highs.max()
        if p_max <= p_min:
            return 0.0
        bins     = np.linspace(p_min, p_max, n_bins + 1)
        bin_vol  = np.zeros(n_bins)
        for i in range(len(recent)):
            d_range = highs[i] - lows[i]
            overlaps = np.clip(
                np.minimum(highs[i], bins[1:]) - np.maximum(lows[i], bins[:-1]),
                0, None,
            )
            if d_range > 0:
                bin_vol += vols[i] * overlaps / d_range
            else:
                b = int(np.clip(np.searchsorted(bins[1:], lows[i]), 0, n_bins - 1))
                bin_vol[b] += vols[i]
        poc_bin   = int(np.argmax(bin_vol))
        poc_price = float((bins[poc_bin] + bins[poc_bin + 1]) / 2)
        price     = float(hist["Close"].iloc[-1])
        return round((price - poc_price) / poc_price, 4) if poc_price > 0 else 0.0
    except Exception:
        return 0.0


def _liquidity_sweep_score(hist: pd.DataFrame, lookback: int = 20) -> float:
    """Deteta stop-hunts: barra que perfura o mínimo de `lookback` dias mas
    fecha acima dele (inversão — compradores absorveram a liquidez).

    Devolve recência normalizada: 1.0 = hoje, decai linearmente a 0 em 10 barras.
    Sinal positivo para dips — sweeps frequentemente marcam reversões.
    """
    if hist is None or len(hist) < lookback + 3:
        return 0.0
    try:
        lows   = hist["Low"].values.astype(float)
        closes = hist["Close"].values.astype(float)
        n      = len(hist)
        last_sweep = -1
        for i in range(lookback, n):
            support = lows[max(0, i - lookback):i].min()
            if lows[i] < support and closes[i] > support:
                last_sweep = i
        if last_sweep < 0:
            return 0.0
        return round(max(0.0, 1.0 - (n - 1 - last_sweep) / 10.0), 4)
    except Exception:
        return 0.0


def add_market_structure_features(features: dict, hist: pd.DataFrame) -> dict:
    """Anchored VWAP (desde 52w high), Volume Profile POC e Liquidity Sweep.

    Todas as features são PIT (hist já deve estar sliced até alert_date).
    Fallbacks aplicados quando hist é None ou demasiado curto.
    """
    _keys = ("vwap_from_high_ratio", "volume_poc_ratio", "liquidity_sweep_score")
    if hist is None or hist.empty or len(hist) < 10:
        for k in _keys:
            features.setdefault(k, _FALLBACK[k])
        return features
    features["vwap_from_high_ratio"] = _anchored_vwap_ratio(hist, use_high=True)
    features["volume_poc_ratio"]     = _volume_poc_ratio(hist, lookback=90)
    features["liquidity_sweep_score"]= _liquidity_sweep_score(hist, lookback=20)
    return features


def _compute_mfi(hist: pd.DataFrame, period: int = 14) -> float:
    """Money Flow Index — pressão compradora/vendedora ponderada por volume.

    MFI < 30: possível acumulação (útil em dips para confirmar fundo).
    MFI > 70: possível distribuição (útil em momentum para confirmar força).
    """
    if hist is None or len(hist) < period + 2:
        return 50.0
    if not all(c in hist.columns for c in ("High", "Low", "Close", "Volume")):
        return 50.0
    try:
        import numpy as _np
        typical = (hist["High"] + hist["Low"] + hist["Close"]) / 3.0
        flow    = typical * hist["Volume"]
        delta   = typical.diff()
        pos_flow = flow.where(delta > 0, 0.0)
        neg_flow = flow.where(delta <= 0, 0.0)
        pos_sum  = pos_flow.rolling(period).sum()
        neg_sum  = neg_flow.rolling(period).sum()
        mfr      = pos_sum / neg_sum.replace(0, float("nan"))
        mfi      = 100.0 - (100.0 / (1.0 + mfr))
        val      = float(mfi.iloc[-1])
        return round(val if _np.isfinite(val) else 50.0, 2)
    except Exception:
        return 50.0


def _compute_volume_up_pct(hist: pd.DataFrame, lookback: int = 20) -> float:
    """Fracção do volume total (últimos N dias) que ocorreu em dias de subida.

    >0.65: acumulação institucional (mais volume em dias de alta).
    <0.35: distribuição (mais volume em dias de queda).
    """
    if hist is None or len(hist) < 5 or "Volume" not in hist.columns:
        return 0.50
    try:
        recent    = hist.iloc[-lookback:]
        close_chg = recent["Close"].diff()
        up_vol    = float(recent["Volume"].where(close_chg > 0, 0.0).sum())
        total_vol = float(recent["Volume"].sum())
        return round(up_vol / total_vol if total_vol > 0 else 0.50, 4)
    except Exception:
        return 0.50


def add_volume_flow_features(features: dict, hist: pd.DataFrame) -> dict:
    """MFI 14d + buying pressure ratio 20d.

    PIT-safe: hist deve estar sliced até alert_date.
    Ambas as features capturam o mesmo fenómeno (acumulação institucional)
    por duas métricas ortogonais — MFI usa preço ponderado; volume_up_pct
    usa dias de subida vs descida brutos.
    """
    _keys = ("mfi_14", "volume_up_pct_20d")
    if hist is None or hist.empty or len(hist) < 5:
        for k in _keys:
            features.setdefault(k, _FALLBACK[k])
        return features
    features["mfi_14"]            = _compute_mfi(hist, period=14)
    features["volume_up_pct_20d"] = _compute_volume_up_pct(hist, lookback=20)
    return features


def add_dividend_features(
    features: dict,
    price_history: Optional[pd.DataFrame],
    alert_date: Optional[Any] = None,
) -> dict:
    """3-year dividend CAGR for Core model.

    Requires price_history fetched with actions=True (yfinance Dividends column).
    0.0 for non-dividend payers — model learns "not a dividend grower" is different
    from a growing dividend, without any invented value polluting the signal.
    """
    features.setdefault("dividend_growth_3y", _FALLBACK["dividend_growth_3y"])
    if price_history is None or "Dividends" not in price_history.columns:
        return features
    try:
        hist = _normalize_index(price_history)
        divs = hist["Dividends"].dropna()
        divs = divs[divs > 0]
        if divs.empty:
            features["dividend_growth_3y"] = 0.0
            return features

        ref = _tz_normalize(alert_date) if alert_date is not None else pd.Timestamp.now().normalize()
        cutoff_3y = ref - pd.DateOffset(years=3)
        cutoff_6y = ref - pd.DateOffset(years=6)

        recent = divs[(divs.index > cutoff_3y) & (divs.index <= ref)]
        prior  = divs[(divs.index > cutoff_6y) & (divs.index <= cutoff_3y)]

        recent_sum = float(recent.sum())
        prior_sum  = float(prior.sum())

        if prior_sum > 0 and recent_sum > 0:
            cagr = (recent_sum / prior_sum) ** (1.0 / 3.0) - 1.0
            features["dividend_growth_3y"] = round(float(np.clip(cagr, -0.5, 1.0)), 4)
        elif recent_sum > 0 and prior_sum == 0:
            features["dividend_growth_3y"] = 1.0   # initiated dividends in last 3y: treat as max growth
        else:
            features["dividend_growth_3y"] = 0.0
    except Exception:
        pass
    return features


def add_momentum_features(features: dict, price_history: Optional[pd.DataFrame], sector_history: Optional[pd.DataFrame] = None, spy_history: Optional[pd.DataFrame] = None) -> dict:
    keys = ["return_1m", "return_3m_pre", "return_6m_pre", "return_12m_pre", "sector_relative", "sector_relative_6m", "beta_90d", "vol_of_vol", "bb_width"]
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
        features["sector_momentum_3m"] = round(sec_ret_3m, 4)
    else:
        features["sector_relative"] = _FALLBACK["sector_relative"]
        features["sector_relative_6m"] = _FALLBACK["sector_relative_6m"]
        features.setdefault("sector_momentum_3m", _FALLBACK["sector_momentum_3m"])
    features["beta_90d"] = _FALLBACK["beta_90d"]
    if spy_history is not None and not spy_history.empty:
        try:
            spy_closes = spy_history["Close"].dropna().values
            n = 90
            if len(closes) >= n + 1 and len(spy_closes) >= n + 1:
                stock_rets = np.diff(closes[-n - 1:]) / closes[-n - 1:-1]
                spy_rets = np.diff(spy_closes[-n - 1:]) / spy_closes[-n - 1:-1]
                min_len = min(len(stock_rets), len(spy_rets))
                stock_rets = stock_rets[-min_len:]
                spy_rets = spy_rets[-min_len:]
                cov = np.cov(stock_rets, spy_rets)
                var_spy = float(cov[1, 1])
                if var_spy > 1e-10:
                    features["beta_90d"] = round(float(cov[0, 1]) / var_spy, 4)
        except Exception:
            pass
    features["vol_of_vol"] = _vol_of_vol(closes)
    return features
