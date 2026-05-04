"""DipModelsV3 dataclass + save/load + ml_report_v3.json — cells 32, 33."""

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
# DipModelsV3 — replica EXACTA do notebook (cell 32)
# ─────────────────────────────────────────────────────────────────────────────
# Notebook define a classe em ``__main__`` antes de pickle.dump. Em produção,
# ``ml_predictor.py`` regista uma versão compatível em ``__main__.DipModelsV3``
# para que joblib.load resolva a referência.
#
# Esta versão tem TODOS os campos que o notebook produz (14), incluindo
# score_calibrator e métricas — ao contrário da versão truncada de 8 campos
# em ml_predictor.py. O ml_predictor consegue ler ambas porque usa
# ``_to_dict`` com aliases canónicos.

@dataclass
class DipModelsV3:
    """Bundle ML v3 com calibrator e métricas completas.

    Picklado em produção como ``dip_models_v3.pkl``. Carregado em runtime
    por ``ml_predictor.py`` (que regista um shim com ``__main__.DipModelsV3``).
    """
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


def _register_in_main() -> None:
    """Garante que ``__main__.DipModelsV3`` aponta para esta classe.

    Necessário para ``joblib.dump(bundle, ...)`` produzir um pickle que possa
    ser carregado em produção (onde ``__main__`` é ``main.py``, sem a classe).
    """
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "DipModelsV3"):
        main_mod.DipModelsV3 = DipModelsV3  # type: ignore[attr-defined]


_register_in_main()


# ─────────────────────────────────────────────────────────────────────────────
# Save/Load (cell 32)
# ─────────────────────────────────────────────────────────────────────────────

def save_bundle(bundle: DipModelsV3, path: Path) -> Path:
    """Pickle via joblib (mais robusto para árvores grandes)."""
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    size_kb = path.stat().st_size / 1024
    log.info(f"[bundle] Escrito: {path} ({size_kb:.1f} KB)")
    return path


def load_bundle(path: Path) -> DipModelsV3:
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
    bundle: DipModelsV3,
    summary_df: "object",  # pd.DataFrame, evita import top-level
    brier_oof: float,
    win_rate_alpha: float,
    n_folds_used: int,
    purge_days: int,
    horizon_days: int,
    new_features: list[str],
) -> dict:
    """Constrói o dict do ``ml_report_v3.json``.

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
            "name":         "alpha_60d",
            "formula":      "max_return_60d - spy_max_return_60d",
            "horizon_days": horizon_days,
        },
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
    """Lê ``ml_report_v3.json`` e devolve só o dict ``metrics``.

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
