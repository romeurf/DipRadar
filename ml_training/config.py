"""Constantes do treino v3.1 — cópia 1:1 do notebook (cell 4)."""

from __future__ import annotations

# Alvo / cobertura
HORIZON_DAYS: int = 60
WINSOR_PCT: float = 0.01
HALF_LIFE_DAYS: int = 365 * 3   # sample weights — half-life 3 anos

# Walk-forward CV
N_FOLDS: int = 10
PURGE_DAYS: int = 21
TOPK_FRAC: float = 0.20

# Features v3 (momentum) e v3.1 (NEW). Os nomes vêm do código existente:
# - MOMENTUM_FEATURES está em ml_features.add_momentum_features
# - NEW_FEATURES_V31 são adicionadas no notebook (cell 12)
MOMENTUM_FEATURES: list[str] = [
    "return_1m",
    "return_3m_pre",
    "sector_relative",
    "beta_60d",
]

NEW_FEATURES_V31: list[str] = [
    "relative_drop",
    "sector_alert_count_7d",
    "days_since_52w_high",
    "month_of_year",
]

# Sector → ETF (cópia local de macro_data.SECTOR_ETF para isolar testes)
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
