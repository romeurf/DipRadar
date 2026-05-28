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
    """ML bundle for the core_us model. Saved as /data/core_us_model.pkl."""
    model:            Any         # GradientBoostingRegressor
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


def _register_in_main() -> None:
    """Ensure ``__main__.CoreModelBundle`` resolves to this class."""
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "CoreModelBundle"):
        main_mod.CoreModelBundle = CoreModelBundle  # type: ignore[attr-defined]


_register_in_main()
