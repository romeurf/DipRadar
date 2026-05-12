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
    "vix",
    "drop_pct_today",
    "drawdown_52w",
    "rsi_14",
    "bb_width",
    "rsi_oversold_strength",
    "vix_regime",
    "drop_x_drawdown",
    "vol_x_drop",
    "return_1m",
    "return_3m_pre",
    "return_6m_pre",
    "sector_relative",
    "beta_60d",
    "vol_of_vol",
    "quality_dislocation",
    "month_of_year",
    "sector_alert_count_7d",
    "vix_percentile_1y",
    "spy_rsi_14",
    # PR #30 (Phase B): 4 raw-OHLCV features computadas point-in-time
    # (slice <= alert_date). Sincronizadas com FEATURE_COLUMNS em ml_features.py.
    "volume_zscore_20d",
    "close_in_range_20d",
    "up_days_pct_20d",
    "true_range_pct_20d",
]
# Features removidas vs PR #25 (Spearman IC profiling sobre o parquet completo
# 36929 rows confirmou 14 features inúteis):
#
#   • 9 constantes (std=0 → IC indefinida; bootstrap não tem fonte point-in-time):
#       gross_margin (=0.35), de_ratio (=80.0), pe_vs_fair (=1.0),
#       analyst_upside (=0.10), quality_score (=0.50), atr_ratio (=0.02),
#       volume_spike (=1.0), pe_attractive (=0.0), peg_implicit (=0.20)
#
#   • 5 noise (|Spearman IC| < 0.01 com alpha_60d):
#       days_since_52w_high (+0.0075), sector_drawdown_5d (-0.0072),
#       macro_score (+0.0046), spy_drawdown_5d (+0.0026), relative_drop (+0.0004)
#
# Total: 34 → 20 features. Smoke test (3000 rows × 3 folds) com este corte:
# single best IC 0.1744 → 0.1946 (+11%). Inference live em position_monitor
# continua a computar todas as features removidas (mantidas em _FALLBACK).
#
# Adicionalmente: month_of_year era constante=5.0 no parquet (bug bootstrap).
# Agora load_base_dataset força recompute via alert_date.dt.month (PR #28).
#
# Histórico de cortes anteriores (PR #25): fcf_yield, short_interest_ratio,
# earnings_surprise_avg, earnings_distance_days (4 constantes adicionadas no
# regen do PR #23) e yield_10y_change_5d, sector_relative_6m, return_12m_pre
# (3 ruidosas com IC ≈ 0).

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
