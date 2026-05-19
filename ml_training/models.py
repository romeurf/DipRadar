"""Factories e configuração de modelos candidatos + stacking meta-learner.

v4.1 changes:
  - ScaledRidge: Ridge com StandardScaler embutido. Ridge é extremamente sensível
    a escala (return_6m_pre [-99,200] vs vix_regime [0,1,2]). O wrapper garante
    que o scaler é fittado em X_train e guardado no bundle → inference correcta
    sem scaler separado. Este era o principal bug a suprimir o IC do Ridge.
  - lgbm_es_factory: adicionado sentinel m.early_stopping_rounds = 50 para que
    _fit_model em train.py detecte ES. Sem ele, LGBM-ES treinava 1000 estimadores
    na totalidade sem parar.
  - XGB-DART: DART boosting para reduzir overfit nos folds recentes.
  - LGBM-GOSS: Gradient-based One-Side Sampling para datasets de baixo sinal.
"""

from __future__ import annotations

import numpy as np
from typing import Callable


# ────────────────────────────────────────────────────────────────────────────────
# ScaledRidge — Ridge com StandardScaler integrado
# ────────────────────────────────────────────────────────────────────────────────

class ScaledRidge:
    """Ridge regression com StandardScaler embutido no objecto.

    Resolve o problema de Ridge ser extremamente sensível a escala quando as
    features têm ranges muito diferentes (ex: return_6m_pre ∈ [-99, 200] vs
    vix_regime ∈ {0, 1, 2}). Sem scaler, Ridge pesa return_6m_pre ~200x mais
    que vix_regime, ignorando efectivamente as features de pequena escala.

    O scaler é fittado em X_train durante .fit() e guardado internamente →
    produção não precisa de scaler separado no bundle. Interface 100% compatível
    com sklearn: fit(X, y, sample_weight=None) / predict(X).
    Picklável via joblib (apenas contém objectos sklearn standard + float).
    """

    def __init__(self, alpha: float = 10.0) -> None:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        self._alpha = alpha
        self._scaler = StandardScaler()
        self._ridge = Ridge(alpha=alpha)
        # _fitted guarda se já foi treinado (para mensagens de erro úteis)
        self._fitted = False

    def fit(self, X, y, sample_weight=None):
        X_sc = self._scaler.fit_transform(X)
        self._ridge.fit(X_sc, y, sample_weight=sample_weight)
        self._fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ScaledRidge.predict() chamado antes de .fit()")
        X_sc = self._scaler.transform(X)
        return self._ridge.predict(X_sc)

    @property
    def coef_(self) -> np.ndarray:
        """Coeficientes Ridge (para feature importance logging)."""
        return self._ridge.coef_

    @property
    def intercept_(self) -> float:
        return float(self._ridge.intercept_)

    def feature_importance_dict(self, feature_names: list[str]) -> dict[str, float]:
        """Devolve {feature: |coef| / sum(|coefs|)} para diagnóstico."""
        coefs = np.abs(self._ridge.coef_)
        total = coefs.sum()
        if total > 0:
            coefs = coefs / total
        return dict(zip(feature_names, coefs.tolist()))

    def __repr__(self) -> str:
        return f"ScaledRidge(alpha={self._alpha}, fitted={self._fitted})"


# ────────────────────────────────────────────────────────────────────────────────
# Model factories
# ────────────────────────────────────────────────────────────────────────────────

def xgb_factory() -> "object":
    """XGBoost conservador v4.0: max_depth=3, mais regularização."""
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=600,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=10,
        subsample=0.75,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
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
    """XGBoost DART boosting (v4.0).

    DART usa dropout durante o boosting, reduzindo overfit nos folds recentes.
    Especialmente eficaz em dados financeiros de baixo sinal.
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
        booster="dart",
        rate_drop=0.1,
        skip_drop=0.5,
        random_state=42,
        tree_method="hist",
        verbosity=0,
    )


def lgbm_factory() -> "object":
    """LightGBM conservador v4.0: num_leaves explícito, mais regularização."""
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=600,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.03,
        min_child_samples=30,
        subsample=0.75,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_split_gain=0.01,
        random_state=42,
        verbosity=-1,
    )


def lgbm_es_factory() -> "object":
    """LightGBM com early stopping (v4.1).

    BUG FIX v4.1: adicionado m.early_stopping_rounds = 50 como sentinel para
    _fit_model em train.py poder detectar ES via hasattr(). Sem este atributo,
    a detecção falhava e LGBM-ES treinava 1000 estimadores sem parar.
    """
    from lightgbm import LGBMRegressor
    m = LGBMRegressor(
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
    # Sentinel para _fit_model detectar ES (LightGBM não guarda este param
    # como atributo por defeito — XGBoost sim). Ver train.py::_fit_model().
    m.early_stopping_rounds = 50
    return m


def lgbm_goss_factory() -> "object":
    """LightGBM com GOSS sampling (v4.0).

    Gradient-based One-Side Sampling: foca o treino nos exemplos de maior
    gradiente, reduzindo o ruído em datasets financeiros de baixo sinal.
    """
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=600,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.03,
        min_child_samples=30,
        boosting_type="goss",
        top_rate=0.2,
        other_rate=0.1,
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
        max_depth=6,
        min_samples_leaf=20,
        max_features=0.5,
        n_jobs=-1,
        random_state=42,
    )


def ridge_factory() -> "object":
    """ScaledRidge com alpha moderado (v4.1).

    BUG FIX v4.1: substituído Ridge(alpha=10) por ScaledRidge(alpha=10).
    Ridge sem scaler pesava return_6m_pre (range [-99,200]) ~200x mais que
    vix_regime (range [0,2]), ignorando efectivamente features de pequena
    escala. ScaledRidge integra o StandardScaler no próprio objecto → bundle
    é auto-suficiente, sem scaler separado em inference.
    """
    return ScaledRidge(alpha=10.0)


def ridge_strong_factory() -> "object":
    """ScaledRidge muito regularizado (proxy de factor model linear, v4.1)."""
    return ScaledRidge(alpha=100.0)


def stack_meta_factory() -> "object":
    """Ridge meta-learner para stacking ensemble (Nível 2).

    alpha=5.0 para regularizar correlação alta entre OOF preds de modelos
    similares (XGB vs LGBM). Não usa ScaledRidge porque as OOF preds já
    estão numa escala comparável (predições do mesmo target).
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


def build_sector_model_configs(
    feature_cols: list[str],
) -> dict[str, dict]:
    """Configurações de modelos por sector individual (11 sectores GICS).

    Um ScaledRidge por sector. Ridge tem apenas n_features parâmetros → treina
    de forma estável com 150+ linhas. O sector mais pequeno no universo do bot
    (ex: Utilities, Real Estate) tem tipicamente 900-1800 alertas históricos,
    mais do que suficiente.

    Agrupar sectores em 4 blocos contaminaria modelos com dinâmicas opostas
    (ex: Consumer Cyclical vs Consumer Defensive, Real Estate vs Financials).
    Modelos individuais aprendem pesos completamente distintos por sector:
    - Technology:          momentum e FCF dominam, VIX negativo
    - Healthcare:          earnings beat rate e insider buying dominam
    - Financial Services:  macro regime e yield curve dominam
    - Consumer Cyclical:   RSI + drawdown dominam, sensível a VIX
    - Consumer Defensive:  dividend safety e estabilidade
    - Energy:              VIX neutro-positivo, short interest relevante
    - Basic Materials:     correlação com commodity cycle
    - Industrials:         macro regime, beta moderado
    - Real Estate (REITs): dividend yield, FCF/FFO dominam
    - Communication Svcs:  similar a Tech mas menos FCF-driven
    - Utilities:           dividend safety, inversamente correlado com yields

    O score final em inference é: 0.70 × global + 0.30 × sector model.
    Se o sector não tiver dados suficientes (< 150 rows), usa só o global.
    """
    from ml_training.config import SECTOR_ETF
    return {
        sector: {
            "factory": ridge_factory,
            "feats":   feature_cols,
        }
        for sector in SECTOR_ETF
        if sector != "Unknown"
    }


def build_model_configs(
    feature_cols_v33: list[str],
    feature_cols_baseline: list[str],
) -> dict[str, dict]:
    """Constrói o dicionário MODEL_CONFIGS com candidatos base + stacking.

    v4.1: Ridge e Ridge-Strong usam agora ScaledRidge (bug fix de escala).
    LGBM-ES tem sentinel early_stopping_rounds para detecção correcta em _fit_model.
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
