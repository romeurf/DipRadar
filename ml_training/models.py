"""Factories e configuração de modelos candidatos + stacking meta-learner."""

from __future__ import annotations

from typing import Callable

# Imports lazy — os pacotes ML pesados só são importados se necessário.


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


def xgb_es_factory() -> "object":
    """XGBoost com early stopping — evita overfit das últimas árvores.

    Early stopping usa os últimos 7% do dataset de treino como validation set.
    O número real de árvores é determinado por early_stopping_rounds=50.
    """
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.04,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist",
        verbosity=0,
        early_stopping_rounds=50,
        eval_metric="rmse",
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


def lgbm_es_factory() -> "object":
    """LightGBM com early stopping."""
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.04,
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
    return Ridge(alpha=10.0)


def stack_meta_factory() -> "object":
    """Ridge meta-learner para stacking ensemble (Nível 2).

    Aprende os pesos óptimos entre as OOF predictions dos modelos base.
    Ridge com alpha=1.0 porque o input (OOF preds) já está regularizado
    pelos modelos base.
    """
    from sklearn.linear_model import Ridge
    return Ridge(alpha=1.0, fit_intercept=True)


def build_feature_lists() -> tuple[list[str], list[str]]:
    """Devolve (FEATURE_COLS, FEATURE_COLS_BASELINE).

    FEATURE_COLS      = ml_features.FEATURE_COLUMNS (única source of truth).
    FEATURE_COLS_BASELINE = FEATURE_COLUMNS sem Stage-3c, 3d, 3e (controlo).
    """
    from ml_features import FEATURE_COLUMNS

    _EXCLUDE_BASELINE = {
        # Stage-3c dislocation
        "quality_dislocation",
        "peg_implicit",
        "relative_drop",
        "month_of_year",
        # Stage-3d context (v3.2)
        "sector_alert_count_7d",
        "days_since_52w_high",
        # Stage-3e short/earnings (v3.3)
        "short_interest_ratio",
        "earnings_surprise_avg",
    }

    full = list(FEATURE_COLUMNS)
    baseline = [c for c in FEATURE_COLUMNS if c not in _EXCLUDE_BASELINE]
    return full, baseline


def build_model_configs(
    feature_cols_v33: list[str],
    feature_cols_baseline: list[str],
) -> dict[str, dict]:
    """Constrói o dicionário MODEL_CONFIGS com candidatos base + stacking."""
    return {
        "XGB-alpha":          {"factory": xgb_factory,     "feats": feature_cols_v33},
        "XGB-ES-alpha":       {"factory": xgb_es_factory,  "feats": feature_cols_v33},
        "LGBM-alpha":         {"factory": lgbm_factory,    "feats": feature_cols_v33},
        "LGBM-ES-alpha":      {"factory": lgbm_es_factory, "feats": feature_cols_v33},
        "RF-alpha":           {"factory": rf_factory,      "feats": feature_cols_v33},
        "Ridge-alpha":        {"factory": ridge_factory,   "feats": feature_cols_v33},
        "XGB-alpha-baseline": {"factory": xgb_factory,     "feats": feature_cols_baseline},
    }
