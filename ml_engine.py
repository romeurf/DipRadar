"""
ml_engine.py — Shim de compatibilidade pós-refactor v3.

A versão antiga deste módulo (Tier A+B+C) foi removida no refactor para v3.
Os call-sites em `position_monitor.py` ainda referenciam:
    load_predictor, predict_dip, extract_shap_top3, format_shap_drivers

Para evitar `ModuleNotFoundError` no boot do bot, este shim adapta as
chamadas para o novo `ml_predictor.ml_score` (regressor dual v3) e devolve
defaults seguros para SHAP (que era opcional no fluxo antigo).

Comportamento:
    load_predictor()        → carrega o bundle v3 e devolve dict normalizado
    predict_dip(...)        → corre ml_score e devolve um Prediction com
                              win_prob + sell_target/hold_days padronizados
                              (não vêm do modelo v3 — são heurísticos default)
    extract_shap_top3(...)  → [] (SHAP não está integrado em v3)
    format_shap_drivers([]) → string vazia / placeholder

Quando o pipeline v3 incluir explicabilidade, este shim deve ser substituído
por chamadas directas ao módulo correspondente.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

import ml_predictor


# Defaults heurísticos (vinham do ml_engine antigo). Em v3 não temos
# previsão directa de target nem horizonte óptimo, por isso usamos:
#   sell_target = price × (1 + DEFAULT_TARGET_PCT)
#   hold_days   = DEFAULT_HOLD_DAYS
DEFAULT_TARGET_PCT = 0.15   # +15% recovery target
DEFAULT_HOLD_DAYS  = 60     # alinhado com o horizonte de treino do v3


@dataclass
class Prediction:
    """Compatível com o objecto que o ml_engine antigo devolvia."""
    win_prob:    float
    sell_target: float
    hold_days:   int
    label:       str       = ""
    pred_up:     float | None = None
    pred_down:   float | None = None


def load_predictor() -> dict | None:
    """Carrega (ou refresh) o bundle v3 e devolve o dict normalizado.

    Devolve None se o bundle não estiver disponível.
    """
    if not ml_predictor._load_bundle():
        return None
    return ml_predictor._bundle


def _coerce_features(feature_row: Any) -> dict:
    """Aceita dict, mapping, ou list (na ordem FEATURE_COLS) e devolve dict."""
    if feature_row is None:
        return {}
    if isinstance(feature_row, dict):
        return feature_row
    # list / tuple → mapeia para FEATURE_COLS por posição
    if isinstance(feature_row, (list, tuple)):
        cols = ml_predictor._FEATURE_COLS
        return {col: feature_row[i] for i, col in enumerate(cols)
                if i < len(feature_row)}
    # objecto com atributos
    try:
        return dict(feature_row)
    except Exception:
        return {}


def predict_dip(
    feature_row:   Any = None,
    current_price: float | None = None,
    ticker:        str | None = None,
    bundle:        Any = None,           # noqa: ARG001 — compat só
    **kwargs: Any,
) -> Prediction:
    """Inferência v3 com adapter para o contrato antigo do ml_engine.

    sell_target e hold_days são heurísticos (DEFAULT_TARGET_PCT, DEFAULT_HOLD_DAYS)
    porque o modelo v3 não os estima directamente. Quando houver módulo
    dedicado a estimar targets dinâmicos, substituir aqui.
    """
    features = _coerce_features(feature_row)
    result   = ml_predictor.ml_score(
        features,
        symbol=ticker,
        log_to_file=False,
    )

    if not result.model_ready:
        return Prediction(
            win_prob    = 0.0,
            sell_target = float(current_price or 0.0) * (1.0 + DEFAULT_TARGET_PCT),
            hold_days   = DEFAULT_HOLD_DAYS,
            label       = result.label or "NO_MODEL",
        )

    base_price  = float(current_price) if current_price else 0.0
    sell_target = base_price * (1.0 + DEFAULT_TARGET_PCT) if base_price > 0 else 0.0

    return Prediction(
        win_prob    = float(result.win_prob),
        sell_target = sell_target,
        hold_days   = DEFAULT_HOLD_DAYS,
        label       = result.label,
        pred_up     = result.pred_up,
        pred_down   = result.pred_down,
    )


def extract_shap_top3(
    bundle:    Any = None,                 # noqa: ARG001
    row_alert: Iterable[Any] | None = None,  # noqa: ARG001
    row_today: Iterable[Any] | None = None,  # noqa: ARG001
) -> list:
    """SHAP não está integrado no pipeline v3. Devolve lista vazia.

    Para reactivar drivers, integrar `shap.TreeExplainer(model_up)` no
    treino e guardar valores OOF no bundle.
    """
    logging.debug("[ml_engine.extract_shap_top3] SHAP não integrado em v3 — []")
    return []


def format_shap_drivers(drivers: list) -> str:
    """Formata drivers para mensagem Telegram. Placeholder se vazio."""
    if not drivers:
        return "_drivers indisponíveis (SHAP não integrado em v3)_"
    lines = []
    for d in drivers[:3]:
        if isinstance(d, dict):
            name = d.get("feature") or d.get("name") or "?"
            val  = d.get("contribution") or d.get("value") or 0.0
            try:
                lines.append(f"  • {name}: {float(val):+.3f}")
            except (TypeError, ValueError):
                lines.append(f"  • {name}: {val}")
        else:
            lines.append(f"  • {d}")
    return "\n".join(lines)
