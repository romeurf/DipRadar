"""
model_registry.py — Authoritative registry for all 6 DipRadar ML models.

Phase 1: dip_us, core_us, momentum_us (implemented)
Phase 2: dip_eu, core_eu, momentum_eu (stubbed — raise NotImplementedError)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from config import ModelID, IMPLEMENTED_MODELS
from ml.ml_features import get_feature_columns


@dataclass
class ModelConfig:
    model_id: str
    description_en: str
    description_pt: str
    target: str
    feature_columns: list[str]
    min_train_rows: int = 500
    implemented: bool = True


def _stub(model_id: str) -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        description_en=(
            f"'{model_id}' is Phase 2 (European markets — not yet implemented)."
        ),
        description_pt=(
            f"'{model_id}' é Fase 2 (mercados europeus — ainda não implementado)."
        ),
        target="alpha_90d",
        feature_columns=[],
        implemented=False,
    )


REGISTRY: dict[str, ModelConfig] = {
    ModelID.DIP_US: ModelConfig(
        model_id=ModelID.DIP_US,
        description_en=(
            "Dip Hunter — US equities. "
            "Identifies oversold quality stocks after sharp drops. "
            "Trigger: price drop >X% in a day/week. Horizon: 60–90d."
        ),
        description_pt=(
            "Dip Hunter — Ações americanas. "
            "Identifica ações de qualidade sobrevendidas após quedas acentuadas. "
            "Trigger: queda >X% num dia/semana. Horizonte: 60–90d."
        ),
        target="alpha_90d",
        feature_columns=get_feature_columns(ModelID.DIP_US),
    ),
    ModelID.CORE_US: ModelConfig(
        model_id=ModelID.CORE_US,
        description_en=(
            "Core — US equities. Fundamental quality screener. No price trigger. "
            "Always on. Answers: is this quality company at a structural discount right now? "
            "Hold forever; sell only on thesis deterioration. Horizon: 90–180d+."
        ),
        description_pt=(
            "Core — Ações americanas. Screener de qualidade fundamental. Sem trigger de preço. "
            "Sempre activo. Responde: esta empresa de qualidade está num desconto estrutural agora? "
            "Hold para sempre; vende apenas em deterioração de tese. Horizonte: 90–180d+."
        ),
        target="alpha_90d",
        feature_columns=get_feature_columns(ModelID.CORE_US),
    ),
    ModelID.MOMENTUM_US: ModelConfig(
        model_id=ModelID.MOMENTUM_US,
        description_en=(
            "Momentum — US equities. "
            "Trend continuation and regime-aware signals. "
            "Trigger: stock rising relative to sector. Horizon: 60–90d."
        ),
        description_pt=(
            "Momentum — Ações americanas. "
            "Continuação de tendência e sinais de regime. "
            "Trigger: ação a subir relativamente ao sector. Horizonte: 60–90d."
        ),
        target="alpha_90d",
        feature_columns=get_feature_columns(ModelID.MOMENTUM_US),
    ),
    ModelID.DIP_EU:      _stub(ModelID.DIP_EU),
    ModelID.CORE_EU:     _stub(ModelID.CORE_EU),
    ModelID.MOMENTUM_EU: _stub(ModelID.MOMENTUM_EU),
}


def get_model_config(model_id: str) -> ModelConfig:
    """Return ModelConfig for model_id, raising NotImplementedError for Phase 2 stubs."""
    cfg = REGISTRY.get(model_id)
    if cfg is None:
        raise KeyError(
            f"Unknown model_id: '{model_id}'. Valid: {list(REGISTRY)}"
        )
    if not cfg.implemented:
        raise NotImplementedError(
            cfg.description_en + "\n" + cfg.description_pt
        )
    return cfg


def list_implemented() -> list[str]:
    """Return model_ids for all currently implemented models."""
    return [mid for mid, cfg in REGISTRY.items() if cfg.implemented]
