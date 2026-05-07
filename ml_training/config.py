from __future__ import annotations
from typing import Optional

# Alvo / cobertura
HORIZON_DAYS: int = 60
WINSOR_PCT: float = 0.005
WINSOR_ABS_LO: float = -0.50
WINSOR_ABS_HI: float = 2.00
HALF_LIFE_DAYS: int = 548          # 18 meses — foca nos regimes recentes

# Walk-forward CV
N_FOLDS: int = 10
PURGE_DAYS: int = 60               # horizonte completo — elimina label overlap
TOPK_FRAC: float = 0.12            # top 12% → mais selectivo, mais alpha real

# Features v3 (momentum), v3.1 (dislocation), v3.2 (context), v3.3 (short/earnings)
MOMENTUM_FEATURES: list[str] = [
    "return_1m",
    "return_3m_pre",
    "sector_relative",
    "beta_60d",
]

NEW_FEATURES_V31: list[str] = [
    "relative_drop",
    "month_of_year",
    "vix_regime",
    "quality_dislocation",
    "peg_implicit",
]

NEW_FEATURES_V32: list[str] = [
    "sector_alert_count_7d",
    "days_since_52w_high",
]

# v3.3 — short interest + earnings surprise (dois sinais de alta convicção)
NEW_FEATURES_V33: list[str] = [
    "short_interest_ratio",   # dias para cobrir o short (yfinance shortRatio)
    "earnings_surprise_avg",  # média dos últimos 2 EPS surprises (%)
]

# NOTE: v3.4 regime features (vix_percentile_1y, spy_rsi_14,
# yield_10y_change_5d) are already the last 3 entries of FEATURE_COLUMNS
# in ml_features.py (Stage 3f). They are canonical base features —
# NOT add-ons — so they must NOT appear here in ALL_NEW_FEATURES.
# Adding them here would cause silent duplication when train.py builds:
#   FEATURE_COLS_V31 = FEATURE_COLUMNS + ALL_NEW_FEATURES
# The full 37-feature vector is: FEATURE_COLUMNS (37) + ALL_NEW_FEATURES (14)
# = 51 columns passed to the model (with regime features counted once).

ALL_NEW_FEATURES: list[str] = (
    NEW_FEATURES_V31
    + NEW_FEATURES_V32
    + NEW_FEATURES_V33
)

# Subsample: None = todos os anos disponíveis (2015-2026+).
# MAX_ALERTS_PER_YEAR limita anos de pico extremo (e.g. 2020, 2022)
# para não distorcer os pesos temporais — 5000/ano dá equilíbrio suficiente.
SUBSAMPLE_YEARS: Optional[list[int]] = None
MAX_ALERTS_PER_YEAR: int = 2_000
SUBSAMPLE_SEED: int = 42

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
