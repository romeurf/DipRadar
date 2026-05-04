"""ml_training — Pacote de treino v3.1 do DipRadar.

Refactor do notebook ``experiments/ml_v2/DipRadar_v3_Training.ipynb`` para
módulos Python testáveis. O entrypoint ``ml_training.train_v31.run_training``
orquestra o pipeline completo:

  1. Carregar dataset base (parquet com features v1/v2)
  2. Fetch yfinance (SPY + ETFs sectoriais + stocks)
  3. Construir dataset v3.1 (34 features + target alpha_60d)
  4. Walk-forward CV (10 folds expanding-window, purge 21d)
  5. Seleccionar champion (rho_alpha_mean máximo com PnL > 0)
  6. Treinar champion no dataset completo + calibrator isotónico
  7. Empacotar ``DipModelsV3`` + escrever ``ml_report_v3.json``

CLI:
    python -m ml_training.train_v31 --input ml_training_merged.parquet \\
        --output dip_models_v3.pkl --report ml_report_v3.json
"""

from ml_training.config import (
    HORIZON_DAYS,
    HALF_LIFE_DAYS,
    N_FOLDS,
    PURGE_DAYS,
    TOPK_FRAC,
    WINSOR_PCT,
    MOMENTUM_FEATURES,
    NEW_FEATURES_V31,
)

__all__ = [
    "HORIZON_DAYS",
    "HALF_LIFE_DAYS",
    "N_FOLDS",
    "PURGE_DAYS",
    "TOPK_FRAC",
    "WINSOR_PCT",
    "MOMENTUM_FEATURES",
    "NEW_FEATURES_V31",
]
