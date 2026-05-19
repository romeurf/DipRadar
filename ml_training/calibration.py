"""
calibration.py — DualHeadCalibrator para os heads upside + downside.

Usa IsotonicRegression treinada nas OOF predictions do walk-forward.
Sem data leakage: o calibrador só vê as previsões fora-de-amostra.

Integração com ml_predictor.py:
  O bundle pkl deve conter 'score_calibrator' (DualHeadCalibrator ou
  um wrapper compatível). ml_predictor._score_to_prob() já suporta
  o campo score_calibrator do bundle — esta classe expande esse suporte
  para o dual-head (upside + downside independentes).

Uso no notebook de treino (colab_bootstrap.ipynb):
  from ml_training.calibration import DualHeadCalibrator

  calibrator = DualHeadCalibrator()
  calibrator.fit(
      oof_upside_scores   = oof_preds['upside'],
      oof_downside_scores = oof_preds['downside'],
      y_upside_rank       = df.loc[oof_idx, 'alpha_90d_rank'],
      y_downside_rank     = df.loc[oof_idx, 'max_drawdown_20d_rank'],
  )

  # Guardar no bundle:
  bundle['calibrator'] = calibrator
  joblib.dump(bundle, 'dip_models.pkl')
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class DualHeadCalibrator:
    """
    Calibra os scores raw de upside e downside para probabilidades reais.

    upside_threshold  — percentil acima do qual consideramos 'bom trade'
                        (default: 0.70 → top 30% de alpha_90d_rank)
    downside_threshold — percentil abaixo do qual consideramos 'perigoso'
                        (default: 0.30 → pior 30% de max_drawdown_20d_rank)
    """

    def __init__(
        self,
        upside_threshold: float = 0.70,
        downside_threshold: float = 0.30,
    ) -> None:
        self.upside_threshold   = upside_threshold
        self.downside_threshold = downside_threshold
        self.upside_calibrator   = IsotonicRegression(out_of_bounds="clip")
        self.downside_calibrator = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    # ------------------------------------------------------------------
    # Treino
    # ------------------------------------------------------------------

    def fit(
        self,
        oof_upside_scores: np.ndarray,
        oof_downside_scores: np.ndarray,
        y_upside_rank: np.ndarray,
        y_downside_rank: np.ndarray,
    ) -> "DualHeadCalibrator":
        """
        Treina os dois calibradores nas OOF predictions.

        Parâmetros
        ----------
        oof_upside_scores   : scores raw do model_up (OOF, sem leakage)
        oof_downside_scores : scores raw do model_down (OOF, sem leakage)
        y_upside_rank       : coluna alpha_60d_rank [0, 1] — label para upside
        y_downside_rank     : coluna max_drawdown_20d_rank [0, 1] — label para downside
        """
        oof_up   = np.asarray(oof_upside_scores,   dtype=np.float64).ravel()
        oof_down = np.asarray(oof_downside_scores, dtype=np.float64).ravel()
        y_up     = np.asarray(y_upside_rank,       dtype=np.float64).ravel()
        y_down   = np.asarray(y_downside_rank,     dtype=np.float64).ravel()

        # Binariza: top quartil de upside = bom trade
        y_up_binary   = (y_up   >= self.upside_threshold).astype(int)
        # Binariza: pior quartil de drawdown = perigoso
        y_down_binary = (y_down <= self.downside_threshold).astype(int)

        self.upside_calibrator.fit(oof_up,   y_up_binary)
        self.downside_calibrator.fit(oof_down, y_down_binary)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Inferência
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        upside_scores: np.ndarray,
        downside_scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retorna (p_upside, p_downside) — arrays [0, 1].

        p_upside  = probabilidade de estar no top 30% de alpha_60d
        p_downside = probabilidade de ter drawdown grave (pior 30%)
        """
        if not self._fitted:
            raise RuntimeError("DualHeadCalibrator não foi treinado. Chama .fit() primeiro.")

        p_up   = self.upside_calibrator.predict(np.asarray(upside_scores,   dtype=np.float64).ravel())
        p_down = self.downside_calibrator.predict(np.asarray(downside_scores, dtype=np.float64).ravel())
        return np.clip(p_up, 0, 1), np.clip(p_down, 0, 1)

    def entry_signal(
        self,
        upside_scores: np.ndarray,
        downside_scores: np.ndarray,
        upside_min: float = 0.65,
        downside_max: float = 0.30,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filtro dual: entra só quando upside é alto E downside é baixo.

        Retorna
        -------
        signal   : bool array — True onde vale a pena entrar
        p_up     : probabilidade calibrada de upside
        p_down   : probabilidade calibrada de downside perigoso
        """
        p_up, p_down = self.predict_proba(upside_scores, downside_scores)
        signal = (p_up > upside_min) & (p_down < downside_max)
        return signal, p_up, p_down

    # ------------------------------------------------------------------
    # Compatibilidade com ml_predictor._score_to_prob(calibrator=...)
    # ------------------------------------------------------------------

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """
        Compat shim: ml_predictor chama calibrator.predict(score_array).
        Usa apenas o upside calibrator — o downside é usado em entry_signal.
        """
        arr = np.asarray(scores, dtype=np.float64).ravel()
        return np.clip(self.upside_calibrator.predict(arr), 0, 1)
