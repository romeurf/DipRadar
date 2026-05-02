"""
ml_ensemble.py — Pickle-safe wrappers para ensembles soft-voting calibrados.

Estas classes são usadas pelo `train_model.py` para treinar e pelo
`ml_predictor.py` (via joblib.load) para inferência. Estão num módulo
próprio para que joblib/pickle as encontre em qualquer processo que
faça apenas `import joblib; bundle = joblib.load(...)`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression


class PrefittedSoftVote:
    """
    Soft-voting classifier built from already-fitted base estimators.

    Implements predict_proba / predict and exposes classes_, so it works
    as a drop-in classifier in ml_predictor and is fully pickleable.
    """

    def __init__(self, estimators: list[tuple[str, Any]], weights: list[float]) -> None:
        if not estimators:
            raise ValueError("At least one estimator required.")
        if len(weights) != len(estimators):
            raise ValueError("estimators and weights must match")
        self.estimators_ = estimators
        w_sum = float(sum(weights))
        self.weights_ = [
            float(w) / w_sum if w_sum > 0 else 1.0 / len(weights) for w in weights
        ]
        first_clf = estimators[0][1]
        if hasattr(first_clf, "classes_"):
            self.classes_ = first_clf.classes_
        elif hasattr(first_clf, "named_steps"):
            self.classes_ = first_clf.named_steps["clf"].classes_
        else:
            self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Any = None) -> "PrefittedSoftVote":
        return self  # already fitted; no-op for sklearn compatibility

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = np.zeros((len(X), len(self.classes_)), dtype=np.float64)
        for (_, est), w in zip(self.estimators_, self.weights_):
            probas += w * np.asarray(est.predict_proba(X))
        s = probas.sum(axis=1, keepdims=True)
        probas = np.divide(probas, s, out=probas, where=s > 0)
        return probas

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def get_params(self, deep: bool = False) -> dict:
        return {}

    def set_params(self, **params: Any) -> "PrefittedSoftVote":
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True


class IsotonicCalibratedVote:
    """
    Wraps a prefitted soft-voting (or other) classifier with an isotonic-regression
    calibrator fit on a held-out calibration set.

    Implements predict_proba / predict, so it's a drop-in replacement compatible
    with ml_predictor.py and joblib pickling.
    """

    def __init__(self, base: Any, raw_cal_probas: np.ndarray, y_cal: np.ndarray) -> None:
        self.base = base
        self.classes_ = getattr(base, "classes_", np.array([0, 1]))
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(np.asarray(raw_cal_probas, dtype=float), np.asarray(y_cal, dtype=float))
        self.iso_ = iso

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw1 = np.asarray(self.base.predict_proba(X))[:, 1]
        cal1 = self.iso_.predict(raw1).clip(0.0, 1.0)
        cal0 = 1.0 - cal1
        return np.vstack([cal0, cal1]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def __sklearn_is_fitted__(self) -> bool:
        return True
