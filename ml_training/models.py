"""Factories e configuração de modelos candidatos — extraído do notebook (cell 22)."""

from __future__ import annotations

from typing import Callable

# Imports lazy — os pacotes ML pesados só são importados se necessário.
# Isto permite testar `cv.py`/`config.py` sem ter xgboost/lightgbm instalados.


def xgb_factory() -> "object":
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist",
        verbosity=0,
    )


def lgbm_factory() -> "object":
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
    )


def rf_factory() -> "object":
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
    )


def ridge_factory() -> "object":
    from sklearn.linear_model import Ridge
    return Ridge(alpha=1.0, random_state=42)


# Mapping name → (factory_callable, feats_kind)
# feats_kind:
#   "v31"      → FEATURE_COLUMNS_V31 (v2 + momentum + 4 NEW)
#   "baseline" → FEATURE_COLUMNS_V2 + MOMENTUM_FEATURES (sem as NEW)
def build_model_configs(
    feature_cols_v31: list[str],
    feature_cols_baseline: list[str],
) -> dict[str, dict]:
    """Constrói o dicionário MODEL_CONFIGS exactamente como no notebook."""
    return {
        "XGB-alpha-v31":      {"factory": xgb_factory,   "feats": feature_cols_v31},
        "LGBM-alpha-v31":     {"factory": lgbm_factory,  "feats": feature_cols_v31},
        "RF-alpha-v31":       {"factory": rf_factory,    "feats": feature_cols_v31},
        "Ridge-alpha-v31":    {"factory": ridge_factory, "feats": feature_cols_v31},
        "XGB-alpha-baseline": {"factory": xgb_factory,   "feats": feature_cols_baseline},
    }


def build_feature_lists() -> tuple[list[str], list[str]]:
    """Devolve (FEATURE_COLUMNS_V31, FEATURE_COLUMNS_BASELINE).

    A definição replica cell 12 do notebook:
      v31 = unique(FEATURE_COLUMNS_V2 + MOMENTUM_FEATURES + NEW_FEATURES_V31)
      baseline = FEATURE_COLUMNS_V2 + MOMENTUM_FEATURES (sem as 4 NEW)
    """
    from experiments.ml_v2.pipeline import FEATURE_COLUMNS_V2
    from ml_training.config import MOMENTUM_FEATURES, NEW_FEATURES_V31

    seen: set[str] = set()
    v31: list[str] = []
    for c in list(FEATURE_COLUMNS_V2) + list(MOMENTUM_FEATURES) + list(NEW_FEATURES_V31):
        if c not in seen:
            seen.add(c)
            v31.append(c)

    baseline_seen: set[str] = set()
    baseline: list[str] = []
    for c in list(FEATURE_COLUMNS_V2) + list(MOMENTUM_FEATURES):
        if c not in baseline_seen:
            baseline_seen.add(c)
            baseline.append(c)

    return v31, baseline
