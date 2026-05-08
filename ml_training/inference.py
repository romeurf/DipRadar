"""
inference.py — Ensemble inference com confidence intervals.

Usa os N modelos do ensemble para calcular mean + std do score.
Std alto = ensemble em desacordo = não entrar.

Integração com ml_predictor.py:
  Este módulo é usado no notebook de treino/inferência batch.
  Para inferência single-alert em produção, usa ml_predictor.ml_score().

Uso:
  from ml_training.inference import predict_with_confidence, final_entry_filter

  result_df = predict_with_confidence(models_up, X_test, calibrator=calibrator)
  entries   = final_entry_filter(result_df.assign(**extra_cols))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ml_training.calibration import DualHeadCalibrator


def predict_with_confidence(
    models: list,
    X: pd.DataFrame,
    calibrator: "DualHeadCalibrator | None" = None,
    cv_threshold: float = 0.20,
) -> pd.DataFrame:
    """
    Usa os N modelos do ensemble para calcular mean + std do score.

    Parâmetros
    ----------
    models       : lista de modelos sklearn/xgb/lgbm com .predict(X)
    X            : feature matrix (pd.DataFrame)
    calibrator   : DualHeadCalibrator opcional — se fornecido, adiciona p_upside
    cv_threshold : coeficiente de variação máximo para high_confidence

    Retorna
    -------
    DataFrame com colunas:
      score_mean      — média das previsões do ensemble
      score_std       — desvio padrão (incerteza do ensemble)
      score_cv        — coeficiente de variação (std / |mean|)
      high_confidence — bool: True se cv < cv_threshold
      p_upside        — (opcional) probabilidade calibrada de upside
    """
    if not models:
        raise ValueError("Lista de modelos está vazia.")

    preds = np.column_stack([m.predict(X) for m in models])  # (n_samples, n_models)

    mean_score = preds.mean(axis=1)
    std_score  = preds.std(axis=1)
    cv_score   = std_score / (np.abs(mean_score) + 1e-6)

    result = pd.DataFrame(
        {
            "score_mean":      mean_score,
            "score_std":       std_score,
            "score_cv":        cv_score,
            "high_confidence": cv_score < cv_threshold,
        },
        index=X.index,
    )

    if calibrator is not None:
        try:
            result["p_upside"] = calibrator.upside_calibrator.predict(
                mean_score.astype(np.float64)
            )
        except Exception:
            result["p_upside"] = np.nan

    return result


def final_entry_filter(
    df: pd.DataFrame,
    upside_min: float = 0.65,
    downside_max: float = 0.30,
    cv_max: float = 0.20,
    earnings_flag_col: str = "earnings_near_flag",
) -> pd.Series:
    """
    Filtro final de entrada — os 4 critérios combinados.

    Critérios:
      1. p_upside > upside_min          — probabilidade calibrada de upside alta
      2. p_downside < downside_max      — risco de drawdown baixo
      3. high_confidence == True        — ensemble em acordo (cv baixo)
      4. earnings_near_flag == 0        — nunca entrar <14d de earnings

    Retorna boolean Series — True onde todos os critérios são cumpridos.
    """
    conditions = pd.Series(True, index=df.index)

    if "p_upside" in df.columns:
        conditions &= df["p_upside"] > upside_min

    if "p_downside" in df.columns:
        conditions &= df["p_downside"] < downside_max

    if "high_confidence" in df.columns:
        conditions &= df["high_confidence"] == True  # noqa: E712

    if earnings_flag_col in df.columns:
        conditions &= df[earnings_flag_col] == 0

    return conditions
