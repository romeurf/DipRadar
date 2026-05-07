"""Constantes do treino v3.2."""

from __future__ import annotations
from typing import Optional

# Alvo / cobertura
HORIZON_DAYS: int = 60
WINSOR_PCT: float = 0.005
WINSOR_ABS_LO: float = -0.50
WINSOR_ABS_HI: float = 2.00
HALF_LIFE_DAYS: int = 365 * 3

# Walk-forward CV
N_FOLDS: int = 10
PURGE_DAYS: int = 21
TOPK_FRAC: float = 0.20

# Features v3 (momentum), v3.1 (dislocation) e v3.2 (context)
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

ALL_NEW_FEATURES: list[str] = NEW_FEATURES_V31 + NEW_FEATURES_V32

# Subsample: None = todos os anos disponíveis (2015-2026+).
# MAX_ALERTS_PER_YEAR limita anos de pico extremo (e.g. 2020, 2022)
# para não distorcer os pesos temporais — 5000/ano dá equilíbrio suficiente.
SUBSAMPLE_YEARS: Optional[list[int]] = None
MAX_ALERTS_PER_YEAR: int = 5_000
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
