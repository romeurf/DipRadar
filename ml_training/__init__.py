"""
ml_training — módulos de treino, avaliação e calibração do DipRadar ML.

Módulos disponíveis:
  calibration        — DualHeadCalibrator (IsotonicRegression dual head)
  target_engineering — neutralize_target_by_sector, compute_realistic_target
  evaluation         — information_coefficient, ic_by_sector, ic_by_regime, full_fold_report
  alert_classifier   — classify_alert_type, add_alert_types, filter_target_types
  diagnostics        — dataset_health_check, check_survivorship_bias
  inference          — predict_with_confidence, final_entry_filter
"""

from .calibration        import DualHeadCalibrator
from .target_engineering import neutralize_target_by_sector, compute_realistic_target
from .evaluation         import (
    information_coefficient,
    ic_by_sector,
    ic_by_regime,
    full_fold_report,
)
from .alert_classifier   import (
    classify_alert_type,
    add_alert_types,
    filter_target_types,
    ALERT_TYPE_ENCODING,
    TARGET_ALERT_TYPES,
)
from .diagnostics        import (
    dataset_health_check,
    check_survivorship_bias,
    get_historical_sp500_constituents,
)
from .inference          import predict_with_confidence, final_entry_filter

__all__ = [
    "DualHeadCalibrator",
    "neutralize_target_by_sector",
    "compute_realistic_target",
    "information_coefficient",
    "ic_by_sector",
    "ic_by_regime",
    "full_fold_report",
    "classify_alert_type",
    "add_alert_types",
    "filter_target_types",
    "ALERT_TYPE_ENCODING",
    "TARGET_ALERT_TYPES",
    "dataset_health_check",
    "check_survivorship_bias",
    "get_historical_sp500_constituents",
    "predict_with_confidence",
    "final_entry_filter",
]
