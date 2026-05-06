"""Constantes do treino v3.2 — cópia 1:1 do notebook."""

from __future__ import annotations

# Alvo / cobertura
HORIZON_DAYS: int = 60
WINSOR_PCT: float = 0.005          # mais agressivo: 0.5% em vez de 1%
WINSOR_ABS_LO: float = -0.50       # clip absoluto inferior (alpha_60d)
WINSOR_ABS_HI: float = 2.00        # clip absoluto superior (alpha_60d)
HALF_LIFE_DAYS: int = 365 * 3   # sample weights — half-life 3 anos

# Walk-forward CV
N_FOLDS: int = 10
PURGE_DAYS: int = 21
TOPK_FRAC: float = 0.20

# Features v3 (momentum) e v3.2 (NEW). Os nomes vêm do código existente:
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
    "vix_regime",          # v3.2 — binário vix > 25 (regime de alta volatilidade)
]

# Subsample: anos com picos de alertas (crash COVID 2020, bear 2022)
# Limitado a MAX_ALERTS_PER_YEAR para não distorcer os pesos temporais
SUBSAMPLE_YEARS: list[int] = [2020, 2022]
MAX_ALERTS_PER_YEAR: int = 2_000
SUBSAMPLE_SEED: int = 42

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
