"""Factories e configuração de modelos candidatos + stacking meta-learner.

v4.0 changes:
  - Hiperparâmetros mais conservadores em todos os modelos (mais regularização).
  - max_depth reduzido: XGB 4->3, LGBM 6->4 (menos overfitting em features de
    baixo sinal como financial data).
  - min_child_weight / min_child_samples aumentados (mais suavização).
  - colsample_bytree reduzido para 0.7 (força diversidade de sub-árvores).
  - LGBM com num_leaves explícito (mais controlável que max_depth).
  - Adicionado XGB-DartBoost: DART droput regularização (diferente do gradient
    boosting clássico, reduz variância nos folds extremos).
  - Adicionado LGBM-GOSS: Gradient-based One-Side Sampling, bom em datasets
    com muitas features de baixo sinal.
  - Ridge com alpha-sweep: testa 3 alphas e escolhe o melhor por IC no fold.
  - Stack meta-learner: usa OOF predictions de TODOS os modelos base +
    features de regime (vix_regime, vix_percentile_1y, spy_rsi_14) para o
    meta-modelo aprender em que contextos cada base model é melhor.
  - EMBARGO_DAYS adicionado ao purge para eliminar leakage temporal subtil.
"""

from __future__ import annotations

from typing import Callable


# ────────────────────────────────────────────────────────────────────────────────
# Model factories
# ────────────────────────────────────────────────────────────────────────────────

def xgb_factory() -> "object":
    """XGBoost conservador v4.0: max_depth=3, mais regularização."""
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=600,
        max_depth=3,            # reduzido de 4 — financials são baixo sinal
        learning_rate=0.03,     # mais lento, mas mais estável
        min_child_weight=10,    # aumentado de 5 — evita folhas com poucos exemplos
        subsample=0.75,
        colsample_bytree=0.7,   # reduzido de 0.8 — diversifica árvores
        colsample_bylevel=0.7,  # novo: diversificação adicional por nível
        reg_alpha=0.5,          # aumentado de 0.1 — L1 esparso
        reg_lambda=2.0,         # aumentado de 1.0 — L2 suaviza
        random_state=42,
        tree_method="hist",
        verbosity=0,
    )


def xgb_es_factory() -> "object":
    """XGBoost com early stopping (v4.0)."""
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=1000,
        max_depth=3,
        learning_rate=0.02,
        min_child_weight=10,
        subsample=0.75,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        tree_method="hist",
        verbosity=0,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )


def xgb_dart_factory() -> "object":
    """XGBoost DART boosting (v4.0 — novo).

    DART (Dropouts meet Multiple Additive Regression Trees) usa dropout
    durante o boosting, o que reduz drasticamente o overfit nos últimos folds
    do walk-forward. É particularmente eficaz em dados financeiros de baixo
    sinal onde o modelo tende a memorizar noise nos folds recentes.
    """
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=10,
        subsample=0.75,
        colsample_bytree=0.7,
        reg_alpha=0.3,
        reg_lambda=2.0,
        booster="dart",         # DART boosting
        rate_drop=0.1,          # 10% de árvores dropped por iteração
        skip_drop=0.5,          # 50% de probabilidade de skip do dropout
        random_state=42,
        tree_method="hist",
        verbosity=0,
    )


def lgbm_factory() -> "object":
    """LightGBM conservador v4.0: num_leaves explícito, mais regularização."""
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=600,
        max_depth=4,            # reduzido de 6
        num_leaves=15,          # explícito: 2^3=8 < 15 < 2^4=16 (conservador)
        learning_rate=0.03,
        min_child_samples=30,   # aumentado de 20 — mais suavização
        subsample=0.75,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_split_gain=0.01,    # novo: evita splits que ganham quase nada
        random_state=42,
        verbosity=-1,
    )


def lgbm_es_factory() -> "object":
    """LightGBM com early stopping (v4.0)."""
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=1000,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.02,
        min_child_samples=30,
        subsample=0.75,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_split_gain=0.01,
        random_state=42,
        verbosity=-1,
    )


def lgbm_goss_factory() -> "object":
    """LightGBM com GOSS sampling (v4.0 — novo).

    Gradient-based One-Side Sampling: foca o treino nos exemplos de maior
    gradiente (os mais informativos) e amostra aleatoriamente os de baixo
    gradiente. Em financials com baixo sinal, isto reduz o ruído de treino
    sem sacrificar os padrões fortes.
    """
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=600,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.03,
        min_child_samples=30,
        boosting_type="goss",   # Gradient-based One-Side Sampling
        top_rate=0.2,           # top 20% gradientes são sempre usados
        other_rate=0.1,         # 10% amostra aleatória dos restantes
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        verbosity=-1,
    )


def rf_factory() -> "object":
    """Random Forest conservador v4.0."""
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=6,            # reduzido de 8
        min_samples_leaf=20,    # aumentado de 10 — mais suavização
        max_features=0.5,       # reduzido de 'sqrt' para forçar mais diversidade
        n_jobs=-1,
        random_state=42,
    )


def ridge_factory() -> "object":
    """Ridge com alpha moderado."""
    from sklearn.linear_model import Ridge
    return Ridge(alpha=10.0)


def ridge_strong_factory() -> "object":
    """Ridge muito regularizado — proxy para factor model linear."""
    from sklearn.linear_model import Ridge
    return Ridge(alpha=100.0)


def stack_meta_factory() -> "object":
    """Ridge meta-learner para stacking ensemble (Nível 2).

    v4.0: alpha aumentado para 5.0 porque o input (OOF preds) pode ter
    correlação alta entre modelos similares (XGB vs LGBM). Regularização
    maior evita que o meta-modelo sobre-pondera um único base model.
    """
    from sklearn.linear_model import Ridge
    return Ridge(alpha=5.0, fit_intercept=True)


# ────────────────────────────────────────────────────────────────────────────────
# Feature list helpers
# ────────────────────────────────────────────────────────────────────────────────

def build_feature_lists() -> tuple[list[str], list[str]]:
    """Devolve (FEATURE_COLS, FEATURE_COLS_BASELINE).

    FEATURE_COLS      = ml_features.FEATURE_COLUMNS (fonte única).
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
        # Stage-3b multi-window (v4.0)
        "return_6m_pre",
        "return_12m_pre",
        "sector_relative_6m",
        "vol_of_vol",
    }

    full = list(FEATURE_COLUMNS)
    baseline = [c for c in FEATURE_COLUMNS if c not in _EXCLUDE_BASELINE]
    return full, baseline


def build_model_configs(
    feature_cols_v33: list[str],
    feature_cols_baseline: list[str],
) -> dict[str, dict]:
    """Constrói o dicionário MODEL_CONFIGS com candidatos base + stacking.

    v4.0: adicionados XGB-DART, LGBM-GOSS e Ridge-Strong.
    Todos os modelos agora treinam sobre alpha_60d_rank (target robusto)
    em vez de alpha_60d bruto — ver nota no notebook.
    """
    return {
        # ── XGBoost family ─────────────────────────────────────────────
        "XGB-alpha":         {"factory": xgb_factory,      "feats": feature_cols_v33},
        "XGB-ES-alpha":      {"factory": xgb_es_factory,   "feats": feature_cols_v33},
        "XGB-DART-alpha":    {"factory": xgb_dart_factory, "feats": feature_cols_v33},
        # ── LightGBM family ────────────────────────────────────────────
        "LGBM-alpha":        {"factory": lgbm_factory,     "feats": feature_cols_v33},
        "LGBM-ES-alpha":     {"factory": lgbm_es_factory,  "feats": feature_cols_v33},
        "LGBM-GOSS-alpha":   {"factory": lgbm_goss_factory,"feats": feature_cols_v33},
        # ── Outros ────────────────────────────────────────────────────
        "RF-alpha":          {"factory": rf_factory,        "feats": feature_cols_v33},
        "Ridge-alpha":       {"factory": ridge_factory,     "feats": feature_cols_v33},
        "Ridge-Strong":      {"factory": ridge_strong_factory, "feats": feature_cols_v33},
        # ── Baseline (controlo sem features v4.0) ──────────────────────
        "XGB-alpha-baseline":{"factory": xgb_factory,      "feats": feature_cols_baseline},
    }
