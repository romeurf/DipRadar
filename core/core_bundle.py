"""CoreModelBundle dataclass + __main__ registration for pickle compat.

Shared by:
  - scripts/train_core_model.py (writes the pickle)
  - core_radar.py               (reads the pickle)

Both joblib.dump and joblib.load resolve the class reference through
`__main__.CoreModelBundle`. When the trainer runs as a subprocess, __main__ is
train_core_model.py; when the bot loads the pickle, __main__ is main.py. We
register the canonical class on whatever __main__ happens to be so the
unpickler always finds it. Mirrors the pattern in ml_training/bundle.py.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class CoreModelBundle:
    """ML bundle for the core_us model. Saved as /data/core_us_model.pkl.

    cv_ic is the primary quality metric — Spearman rank correlation between
    out-of-fold predictions and the true target. It tolerates the heavy tails
    of alpha_1y much better than R² and is the industry-standard quant metric.
    champion_name tells you which model architecture won (GBR vs Ridge) across
    the purged-CV evaluation.
    """
    model:            Any         # GradientBoostingRegressor OR Ridge (after champion selection)
    feature_cols:     list
    scaler:           Any         # QuantileTransformer
    impute_medians:   dict        # median fill values for NaN features at inference
    train_date:       str
    n_train:          int
    cv_rmse:          float
    cv_r2:            float
    alpha_p10:        float
    alpha_p50:        float
    alpha_p90:        float
    # ── Added 2026-05-29 — IC + champion selection ──
    # cv_ic and champion_name default to neutral values for backward compatibility
    # with bundles trained before this field existed (joblib will assign defaults).
    cv_ic:            float = 0.0
    champion_name:    str = "GBR"


def _register_in_main() -> None:
    """Ensure ``__main__.CoreModelBundle`` resolves to this class."""
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "CoreModelBundle"):
        main_mod.CoreModelBundle = CoreModelBundle  # type: ignore[attr-defined]


_register_in_main()
