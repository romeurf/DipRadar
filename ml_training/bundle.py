"""DipModels dataclass + save/load + ml_report.json."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DipModels — canonical bundle dataclass
# ─────────────────────────────────────────────────────────────────────────────
# Pickled in production as ``dip_models.pkl``. Loaded at runtime by
# ``ml_predictor.py``, which registers a shim in ``__main__`` so the unpickler
# resolves the class reference regardless of which module is __main__.
#
# ``DipModelsV3`` is kept as a module-level alias so bundles pickled under the
# old name continue to load. Do not remove the alias until all on-disk bundles
# have been retrained under the new name.

@dataclass
class DipModels:
    """ML bundle with calibrator and full metrics. Saved as ``dip_models.pkl``."""
    model_up:         Any
    model_down:       Any
    feature_cols:     list
    score_calibrator: Any = None
    n_train_samples:  int = 0
    train_date:       str = ""
    champion_name:    str = ""
    schema_version:   int = 3
    momentum_feats:   list = field(default_factory=list)
    rho_mean:         Optional[float] = None
    rho_alpha:        Optional[float] = None
    rho_down:         Optional[float] = None
    topk_pnl:         Optional[float] = None
    fold_metrics:     list = field(default_factory=list)
    # Per-sector models (one ScaledRidge per GICS sector).
    # Each entry: {sector_name: {"model": <fitted ScaledRidge>, "n_train": int}}
    # Inference: use sector model if available; global as fallback.
    # Absent in old bundles → inference uses only the global model.
    sector_models:    dict = field(default_factory=dict)


# Backwards-compatible alias. Existing pickles reference `DipModelsV3` in
# `__main__`; mapping the name to the canonical class keeps them loadable.
DipModelsV3 = DipModels


def _register_in_main() -> None:
    """Ensure ``__main__.DipModels`` (and the legacy alias) resolve to this class."""
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return
    for name in ("DipModels", "DipModelsV3"):
        if not hasattr(main_mod, name):
            setattr(main_mod, name, DipModels)


_register_in_main()


# ─────────────────────────────────────────────────────────────────────────────
# Save/Load (cell 32)
# ─────────────────────────────────────────────────────────────────────────────

def save_bundle(bundle: DipModels, path: Path) -> Path:
    """Pickle via joblib (mais robusto para árvores grandes)."""
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    size_kb = path.stat().st_size / 1024
    log.info(f"[bundle] Escrito: {path} ({size_kb:.1f} KB)")
    return path


def load_bundle(path: Path) -> DipModels:
    """Round-trip joblib.load com sanity asserts.

    Raise se o objecto carregado não tiver ``model_up``/``model_down``.
    """
    import joblib
    obj = joblib.load(path)
    if not hasattr(obj, "model_up") or not hasattr(obj, "model_down"):
        raise ValueError(
            f"Bundle inválido em {path}: faltam model_up/model_down. "
            f"Tipo carregado: {type(obj).__name__}"
        )
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Report (cell 33)
# ─────────────────────────────────────────────────────────────────────────────

def build_report(
    bundle: DipModels,
    summary_df: "object",  # pd.DataFrame, evita import top-level
    brier_oof: float,
    win_rate_alpha: float,
    n_folds_used: int,
    purge_days: int,
    horizon_days: int,
    new_features: list[str],
    feature_stats: "dict | None" = None,
) -> dict:
    """Constrói o dict do ``ml_report.json`` (legacy ``ml_report_v3.json``).

    Replica cell 33 com tipos JSON-safe (floats nativos, listas de dicts).
    """
    metrics = {
        "rho_alpha_mean": float(bundle.rho_alpha) if bundle.rho_alpha is not None else None,
        "rho_down_mean":  float(bundle.rho_down)  if bundle.rho_down  is not None else None,
        "topk_pnl_mean":  float(bundle.topk_pnl)  if bundle.topk_pnl  is not None else None,
        "brier_oof":      float(brier_oof),
        "win_rate_alpha": float(win_rate_alpha),
    }

    # summary.round(4).to_dict(orient='records') — sem importar pandas no topo
    try:
        comparison = summary_df.round(4).to_dict(orient="records")
    except Exception:
        comparison = []

    return {
        "schema_version":   3,
        "trained_at":       bundle.train_date,
        "champion":         bundle.champion_name,
        "n_features":       len(bundle.feature_cols),
        "feature_cols":     list(bundle.feature_cols),
        "momentum_feats":   list(bundle.momentum_feats),
        "new_features_v31": list(new_features),
        "n_train":          int(bundle.n_train_samples),
        "horizon_days":     horizon_days,
        "cv": {
            "kind":       "walk_forward_purged_expanding",
            "n_folds":    int(n_folds_used),
            "purge_days": int(purge_days),
        },
        "metrics":     metrics,
        "comparison":  comparison,
        "target": {
            "name":         "alpha_90d",
            "formula":      "log1p(close_90d) - log1p(spy_close_90d)",
            "horizon_days": horizon_days,
        },
        # feature_stats: usado pelo health_monitor.check_feature_drift() para PSI.
        # Sem este campo, drift detection retorna {"skipped": True} graciosamente.
        "feature_stats": feature_stats or {},
    }


def save_report(report: dict, path: Path) -> Path:
    """JSON-safe write (atómico via .tmp)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2, default=str))
    tmp.replace(path)
    log.info(f"[report] Escrito: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de comparação (para diff entre treinos consecutivos)
# ─────────────────────────────────────────────────────────────────────────────

def metrics_from_report(report_path: Path) -> dict[str, Optional[float]]:
    """Lê ``ml_report.json`` (ou legacy ``ml_report_v3.json``) e devolve
    só o dict ``metrics``.

    Devolve dict com chaves vazias se o ficheiro não existe / é inválido.
    """
    empty: dict[str, Optional[float]] = {
        "rho_alpha_mean": None,
        "rho_down_mean":  None,
        "topk_pnl_mean":  None,
        "brier_oof":      None,
        "win_rate_alpha": None,
    }
    if not Path(report_path).exists():
        return empty
    try:
        data = json.loads(Path(report_path).read_text())
    except Exception as e:
        log.warning(f"[bundle] Report corrupt at {report_path}: {e}")
        return empty
    metrics = data.get("metrics") or {}
    out = empty.copy()
    for key in out:
        v = metrics.get(key)
        try:
            out[key] = float(v) if v is not None else None
        except (TypeError, ValueError):
            out[key] = None
    return out
