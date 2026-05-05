"""
conflict_resolver.py — Fase 2: Cruzar análise fundamental vs. ML

Quando os dois sinais (Score V2 fundamental, label ML) divergem, o utilizador
recebe um veredicto cinzento e não sabe o que fazer. Este módulo arbitra:

  fund_score ≥ 65 + ml_bull                         →  CONSENSUS_BULL              (compra forte)
  fund_score < 50 + ml_bear                          →  CONSENSUS_BEAR              (rejeitar)
  fund_score < 55 + ml_bull + has_dislocation        →  TECHNICAL_WITH_DISLOCATION  (padrão NOW/PINS)
  fund_score < 55 + ml_bull + sem dislocation        →  TECHNICAL_ONLY              (especulação pura, ex: RKLB)
  fund_score ≥ 65 + ml_bear                          →  FUNDAMENTAL_ONLY            (DCA gradual)
  caso contrário                                     →  NEUTRAL                     (zona cinzenta)

ML labels considerados bull: WIN, WIN_STRONG, WIN_40

API pública:
  ConflictState (Enum)
  resolve_conflict(fund_score, ml_label, has_dislocation, is_preprofit) -> (state, message, emoji, sizing)
"""
from __future__ import annotations

from enum import Enum


class ConflictState(Enum):
    CONSENSUS_BULL             = "CONSENSUS_BULL"
    CONSENSUS_BEAR             = "CONSENSUS_BEAR"
    TECHNICAL_WITH_DISLOCATION = "TECHNICAL_WITH_DISLOCATION"
    TECHNICAL_ONLY             = "TECHNICAL_ONLY"
    FUNDAMENTAL_ONLY           = "FUNDAMENTAL_ONLY"
    NEUTRAL                    = "NEUTRAL"


# Mapeamento de estado → (verdict, emoji, sizing_pct_of_base)
_STATE_META: dict[ConflictState, tuple[str, str, str]] = {
    ConflictState.CONSENSUS_BULL:             ("COMPRAR",      "🟢", "100-150%"),
    ConflictState.CONSENSUS_BEAR:             ("EVITAR",       "🔴", "0%"),
    ConflictState.TECHNICAL_WITH_DISLOCATION: ("MONITORIZAR",  "🟡", "60-80%"),
    ConflictState.TECHNICAL_ONLY:             ("ESPECULAÇÃO",  "🟠", "20-30%"),
    ConflictState.FUNDAMENTAL_ONLY:           ("ACUMULAR",     "🔵", "70% DCA"),
    ConflictState.NEUTRAL:                    ("NEUTRO",       "⚪", "0%"),
}

_BULL_LABELS = frozenset({"WIN", "WIN_STRONG", "WIN_40"})


def resolve_conflict(
    fund_score: float,
    ml_label: str | None,
    has_dislocation: bool = False,
    is_preprofit: bool = False,
) -> tuple[ConflictState, str, str, str]:
    """
    Cruza Score V2 com label ML e devolve (estado, mensagem, emoji, sizing).

    fund_score      : 0-100 (saída de calculate_score['final_score'])
    ml_label        : WIN_STRONG / WIN / WIN_40 / WEAK / NO_WIN / NO_MODEL / None
    has_dislocation : True se quality_dislocation > 0.30
    is_preprofit    : True se fcf_yield < 0 e revenue_growth > 0.20
    """
    is_ml_bull = (ml_label in _BULL_LABELS) if ml_label else False

    # ── 1. CONSENSUS BULL ────────────────────────────────────────────────────
    if fund_score >= 65 and is_ml_bull:
        state = ConflictState.CONSENSUS_BULL
        msg = "Sinal forte: fundamentais sólidos aliados a momentum técnico."

    # ── 2. CONSENSUS BEAR ────────────────────────────────────────────────────
    elif fund_score < 50 and not is_ml_bull:
        state = ConflictState.CONSENSUS_BEAR
        if is_preprofit:
            msg = (
                "Empresa pré-lucro sem momentum técnico. "
                "Métricas tradicionais inaplicáveis — aguardar catalisador de crescimento."
            )
        else:
            msg = "Rejeitado: sem suporte técnico nem fundamental."

    # ── 3. TÉCNICO COM DISLOCATION (padrão NOW / PINS) ───────────────────────
    elif fund_score < 55 and is_ml_bull and has_dislocation:
        state = ConflictState.TECHNICAL_WITH_DISLOCATION
        msg = (
            "Padrão de Quality Dislocation detectado: empresa de qualidade com "
            "drawdown injustificado + momentum técnico. "
            "Entrar com 60-80% da posição base — aguardar confirmação fundamental."
        )

    # ── 4. TÉCNICO SEM DISLOCATION (especulação pura, ex: RKLB) ─────────────
    elif fund_score < 55 and is_ml_bull and not has_dislocation:
        state = ConflictState.TECHNICAL_ONLY
        if is_preprofit:
            msg = (
                "⚠️ Empresa pré-lucro com padrão técnico de entrada. "
                "Fundamentos não qualificam dislocation. "
                "Especulação pura — usar apenas 20-30% da posição base com stop apertado."
            )
        else:
            msg = (
                "Padrão técnico detectado MAS empresa sem qualidade fundamental. "
                "Especulação pura — usar apenas 20-30% da posição base com stop apertado."
            )

    # ── 5. FUNDAMENTAL SEM MOMENTUM (DCA gradual) ────────────────────────────
    elif fund_score >= 65 and not is_ml_bull:
        state = ConflictState.FUNDAMENTAL_ONLY
        msg = (
            "Empresa de qualidade sem padrão técnico de reversão iminente. "
            "Acumular lentamente via DCA — não entrar com posição completa agora."
        )

    # ── 6. NEUTRAL (zona cinzenta) ───────────────────────────────────────────
    else:
        state = ConflictState.NEUTRAL
        msg = "Sinal misto: score fundamental na zona cinzenta (55-65). Aguardar clareza."

    verdict, emoji, sizing = _STATE_META[state]
    return state, msg, emoji, sizing


def get_verdict_line(
    fund_score: float,
    ml_label: str | None,
    has_dislocation: bool = False,
    is_preprofit: bool = False,
) -> str:
    """Devolve linha formatada para alerta Telegram."""
    state, msg, emoji, sizing = resolve_conflict(
        fund_score, ml_label, has_dislocation, is_preprofit
    )
    _, verdict, _, _ = state, *_STATE_META[state]
    verdict_str = _STATE_META[state][0]
    return (
        f"{emoji} *{verdict_str}* | Sizing: {sizing}\n"
        f"_{msg}_"
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cases = [
        # (fund_score, ml_label, has_dislocation, is_preprofit, descrição)
        (80, "WIN_STRONG", False, False, "CONSENSUS_BULL esperado"),
        (75, "WIN",        True,  False, "CONSENSUS_BULL esperado"),
        (40, "NO_WIN",     False, False, "CONSENSUS_BEAR esperado"),
        (45, "WIN",        True,  False, "TECHNICAL_WITH_DISLOCATION esperado (NOW/PINS)"),
        (45, "WIN",        False, True,  "TECHNICAL_ONLY esperado (RKLB)"),
        (70, "NO_WIN",     False, False, "FUNDAMENTAL_ONLY esperado"),
        (60, "WIN",        False, False, "NEUTRAL esperado (zona cinzenta)"),
        (50, None,         False, False, "NEUTRAL esperado"),
    ]
    for s, lab, disloc, preprofit, desc in cases:
        state, msg, emoji, sizing = resolve_conflict(s, lab, disloc, preprofit)
        verdict = _STATE_META[state][0]
        print(f"\n  [{desc}]")
        print(f"  score={s} ml={str(lab):12} disloc={disloc} preprofit={preprofit}")
        print(f"  → {emoji} {state.value} | {verdict} | {sizing}")
        print(f"  {msg}")
