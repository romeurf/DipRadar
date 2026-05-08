from __future__ import annotations
from typing import Optional

# ─── Alvo / cobertura ──────────────────────────────────────────────────────
HORIZON_DAYS: int   = 60
WINSOR_PCT:   float = 0.005
WINSOR_ABS_LO: float = -0.50
WINSOR_ABS_HI: float =  2.00
HALF_LIFE_DAYS: int = 548   # 18 meses — foca nos regimes recentes

# ─── Walk-forward CV ───────────────────────────────────────────────────
N_FOLDS:    int   = 10
PURGE_DAYS: int   = 90     # maior que horizonte (60d) + buffer — elimina leakage temporal
EMBARGO_DAYS: int = 20     # dias de embargo ANTES do test set (novo v4.0)
TOPK_FRAC:  float = 0.12   # top 12 % dos alertas avaliados no CV

# ─── Features (lista única, mesma ordem que FEATURE_COLUMNS em ml_features.py)
FEATURE_COLS: list[str] = [
    # Stage 0 — Macro
    "macro_score",
    "vix",
    "spy_drawdown_5d",
    "sector_drawdown_5d",
    # Stage 1 — Quality / Value
    "gross_margin",
    "de_ratio",
    "pe_vs_fair",
    "analyst_upside",
    "quality_score",
    # Stage 2 — Timing
    "drop_pct_today",
    "drawdown_52w",
    "rsi_14",
    "atr_ratio",
    "volume_spike",
    # Stage 3 — Engineered interactions
    "rsi_oversold_strength",
    "vix_regime",
    "pe_attractive",
    "drop_x_drawdown",
    "vol_x_drop",
    # Stage 3b — Momentum (v4.0 — multi-window)
    "return_1m",
    "return_3m_pre",
    "return_6m_pre",
    "return_12m_pre",
    "sector_relative",
    "sector_relative_6m",
    "beta_60d",
    "vol_of_vol",
    # Stage 3c — Dislocation
    "quality_dislocation",
    "peg_implicit",
    "relative_drop",
    "month_of_year",
    # Stage 3d — Context
    "sector_alert_count_7d",
    "days_since_52w_high",
    # Stage 3e — Short / Earnings
    "short_interest_ratio",
    "earnings_surprise_avg",
    # Stage 3f — Regime
    "vix_percentile_1y",
    "spy_rsi_14",
    "yield_10y_change_5d",
]

# ─── Subsample ─────────────────────────────────────────────────────────────
SUBSAMPLE_YEARS:    Optional[list[int]] = None
MAX_ALERTS_PER_YEAR: int = 2_000
SUBSAMPLE_SEED:      int = 42

# ─── Sector ETFs ───────────────────────────────────────────────────────────
SECTOR_ETF: dict[str, str] = {
    "Technology":             "XLK",
    "Financial Services":     "XLF",
    "Healthcare":             "XLV",
    "Consumer Cyclical":      "XLY",
    "Consumer Defensive":     "XLP",
    "Industrials":            "XLI",
    "Energy":                 "XLE",
    "Utilities":              "XLU",
    "Real Estate":            "XLRE",
    "Basic Materials":        "XLB",
    "Communication Services": "XLC",
    "Unknown":                "SPY",
}
DEFAULT_ETF: str = "SPY"
