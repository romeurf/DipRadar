from __future__ import annotations

from typing import Optional

# ── Horizons ─────────────────────────────────────────────────────────────────
# Primary target: close_90d alpha (rotação de capital em ~90 dias).
# Blue-chips e stocks de qualidade podem sugerir hold estendido (6M+).
HORIZON_DAYS: int = 90                      # janela para close_90d / alpha_90d

# Colunas alvo no parquet — geradas em data.py
PRIMARY_TARGET: str = "alpha_90d"           # excesso de retorno sobre SPY em 90d
PRIMARY_TARGET_FALLBACK: str = "alpha_60d"  # fallback para parquets antigos sem alpha_90d
RISK_TARGET: str = "max_drawdown_60d"       # drawdown máximo nos primeiros 60d

# Threshold do calibrador: P(alpha_90d > CALIBRATOR_THRESHOLD)
CALIBRATOR_THRESHOLD: float = 0.07         # 7% de alpha — ligeiramente mais alto que 60d

WINSOR_PCT: float = 0.005
WINSOR_ABS_LO: float = -0.50
WINSOR_ABS_HI: float = 2.00
# Half-life do decay temporal dos pesos de treino.
# 730d (2 anos) → um alerta de 2020 pesa ~25% vs um alerta de hoje.
# 548d era demasiado agressivo — descartava crashes de 2022 e COVID.
# Aumento para 730d preserva crises históricas como sinal de treino válido.
# Formula: weight = 2^(-days_since / HALF_LIFE_DAYS)
HALF_LIFE_DAYS: int = 730

N_FOLDS: int = 10
PURGE_DAYS: int = 90
EMBARGO_DAYS: int = 20
TOPK_FRAC: float = 0.12

# ── Classificação de stocks (usada em ml_predictor ao fazer inference) ────────
# Stocks acima destes limites são tratados como acumulação de longo prazo,
# não como dip-and-flip. Podem ainda ser saídos se deterioração estrutural.
BLUE_CHIP_MARKET_CAP_B: float = 100.0    # market cap > $100B
BLUE_CHIP_QUALITY_SCORE: float = 0.65   # quality_score normalizado [0,1]
QUALITY_MARKET_CAP_B: float = 25.0      # $25B < cap <= $100B
QUALITY_SCORE_MIN: float = 0.55

# Alpha threshold para sugerir extensão de hold para ~6M em stocks de qualidade
EXTEND_HOLD_ALPHA_THRESHOLD: float = 0.08  # alpha_90d > 8% → considera segurar 6M

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
    # Fase 5: sinais de qualidade e sentimento
    "consecutive_red_days",
    "ma_200d_ratio",
    "insider_buy_recent",
    "earnings_beat_rate",
    "analyst_rating",
    "short_interest_pct",
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
