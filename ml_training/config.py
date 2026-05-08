from __future__ import annotations

from typing import Optional

HORIZON_DAYS: int = 60
WINSOR_PCT: float = 0.005
WINSOR_ABS_LO: float = -0.50
WINSOR_ABS_HI: float = 2.00
HALF_LIFE_DAYS: int = 548

N_FOLDS: int = 10
PURGE_DAYS: int = 90
EMBARGO_DAYS: int = 20
TOPK_FRAC: float = 0.12

FEATURE_COLS: list[str] = [
    "macro_score",
    "vix",
    "spy_drawdown_5d",
    "sector_drawdown_5d",
    "gross_margin",
    "de_ratio",
    "pe_vs_fair",
    "analyst_upside",
    "quality_score",
    "fcf_yield",
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
    "return_12m_pre",
    "sector_relative",
    "sector_relative_6m",
    "beta_60d",
    "vol_of_vol",
    "quality_dislocation",
    "peg_implicit",
    "relative_drop",
    "month_of_year",
    "sector_alert_count_7d",
    "days_since_52w_high",
    "short_interest_ratio",
    "earnings_surprise_avg",
    "earnings_distance_days",
    "vix_percentile_1y",
    "spy_rsi_14",
    "yield_10y_change_5d",
]

SUBSAMPLE_YEARS: Optional[list[int]] = None
MAX_ALERTS_PER_YEAR: int = 2_000
SUBSAMPLE_SEED: int = 42

SECTOR_ETF: dict[str, str] = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
    "Unknown": "SPY",
}
DEFAULT_ETF: str = "SPY"
