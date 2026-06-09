from __future__ import annotations

from typing import Optional

from ml.ml_features import get_feature_columns

# ── Horizons ─────────────────────────────────────────────────────────────────
# Primary target: close_90d alpha (rotação de capital em ~90 dias).
# Blue-chips e stocks de qualidade podem sugerir hold estendido (6M+).
HORIZON_DAYS: int = 90                      # janela para close_90d / alpha_90d

# Colunas alvo no parquet — geradas em data.py
#
# Target choice (PRIMARY_TARGET):
#   - "alpha_90d"           → excess return vs SPY (market-beta target).
#                             Predicts stock-specific + sector + market
#                             timing combined. Easier coverage (SPY exists
#                             for every row) but mixes signals.
#   - "alpha_vs_sector_90d" → excess return vs the matching sector ETF
#                             (XLK, XLF, XLV...). Stock-specific target —
#                             removes sector beta so the model learns
#                             relative-strength within sector. Cleaner for
#                             a stock-picking strategy.
#
# Switch via env var DIP_PRIMARY_TARGET. Default kept at alpha_90d for
# backward compatibility with existing trained bundles; flip to
# alpha_vs_sector_90d after rebuilding the parquet with both targets
# populated (build_dip_dataset.py writes both since this change).
import os as _os
PRIMARY_TARGET: str = _os.getenv("DIP_PRIMARY_TARGET", "alpha_90d")
RISK_TARGET: str = "max_drawdown_60d"       # drawdown máximo nos primeiros 60d

# Threshold do calibrador: P(alpha_90d > CALIBRATOR_THRESHOLD)
CALIBRATOR_THRESHOLD: float = 0.06         # 6% de alpha — alinhado com _SCORE_HIGH em ml_predictor

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

# DIP model feature list — single source of truth is FEATURE_SETS["dip_us"] in ml_features.py.
# Do NOT edit this list here; edit FEATURE_SETS in ml_features.py instead.
FEATURE_COLS: list[str] = get_feature_columns("dip_us")

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
