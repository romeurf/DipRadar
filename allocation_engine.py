"""
allocation_engine.py — Motor read-only de sugestão de alocação (Fase 1).

Dado um sinal (ticker + dip_score + ML metrics + macro regime + liquidez),
o motor sugere:

  • Categoria  : ETF_CORE | HOLD_FOREVER | APARTAMENTO | GROWTH | FLIP | PASS
  • Sizing     : amount_eur (clamped à liquidez disponível e a mínimos T212)
  • Confiança  : Alta | Média | Baixa
  • Exit rule  : NEVER | THESIS_BREAK | TARGET_+15% | TARGET_+20% | TIME_60D
  • Rationale  : razão humanamente legível

Este módulo é *puro* — nenhuma I/O, nenhuma chamada à yfinance, nenhuma
escrita em ficheiros. Recebe contexto pronto, decide. Isto torna o motor
testável em isolamento e seguro de invocar a partir de qualquer lado
(Telegram, scheduler mensal, scripts ad-hoc).

Para a versão "from-symbol-to-decision" (que faz o data fetching), ver
`main.allocate_ticker()`, que orquestra a leitura de fundamentals + ML +
macro e depois invoca `suggest_allocation()` aqui.

Filosofia de design
-------------------
1. Read-only: nunca executa ordens. Só sugere.
2. Conservador por defeito: regimes RED bloqueiam novas entradas em
   GROWTH/FLIP; ETF_CORE e HOLD_FOREVER continuam autorizados (lógica
   defensiva do design doc).
3. Mínimos T212-compatíveis: floor €20 para fractional shares.
4. Cap por nome: nunca propõe mais que `monthly_budget * MAX_SINGLE_NAME_PCT`
   nem mais que a liquidez disponível.

Ver `docs/allocation_engine_design.md` para o roadmap completo (Fases 1-4).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


# ── Orçamento mensal (configurável via env var Railway) ───────────────────────
# Define MONTHLY_BUDGET_EUR=<valor> no Railway para alterar sem tocar no código.
# Fallback: 1050.0 (valor histórico).
_MONTHLY_BUDGET_EUR: float = float(os.environ.get("MONTHLY_BUDGET_EUR", "1050.0"))


# ── Constantes ────────────────────────────────────────────────────────────────

# Categorias (strings estáveis para serialização / logs)
CAT_ETF_CORE     = "ETF_CORE"
CAT_HOLD_FOREVER = "HOLD_FOREVER"
CAT_APARTAMENTO  = "APARTAMENTO"
CAT_GROWTH       = "GROWTH"
CAT_FLIP         = "FLIP"
CAT_PASS         = "PASS"

# Pesos-alvo (fracção do orçamento mensal). Soma > 100% por design — alguns
# meses não acertam todas as categorias e o orçamento que sobra fica em cash.
_TARGET_PCT: dict[str, float] = {
    CAT_ETF_CORE:     0.45,   # core ETF (DCA)
    CAT_HOLD_FOREVER: 0.12,   # blue-chip raro
    CAT_APARTAMENTO:  0.10,   # dividend / value
    CAT_GROWTH:       0.06,   # active stock pick
    CAT_FLIP:         0.04,   # tactical
}

# Mínimo absoluto para uma sugestão fazer sentido (T212 fractional shares
# trabalham bem a partir de €20). Abaixo disto, é melhor acumular cash para
# o mês seguinte.
_MIN_TICKET_EUR    = 20.0
_MAX_FLIP_EUR      = 40.0     # cap absoluto para FLIP (alta rotação, single-name)
_MAX_GROWTH_EUR    = 80.0     # cap absoluto para GROWTH (single-name risk)

# Cap por nome em fracção do orçamento mensal (qualquer categoria não-ETF)
_MAX_SINGLE_NAME_PCT = 0.15

# Thresholds de label do ML (em pred_up = retorno previsto a 60d)
_ML_WIN_STRONG = 0.10   # > +10% expected → WIN_STRONG
_ML_WIN        = 0.05   # > +5%  expected → WIN

# Dip score thresholds (0-100)
_DIP_GROWTH_MIN  = 60
_DIP_PASS_MAX    = 40
_DIP_HOLD_FOREVER_MIN = 70   # mesma lógica que score.classify_dip_category


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class AllocationContext:
    """Tudo que o motor precisa para decidir, num único objecto."""
    ticker:              str
    dip_score:           float                = 0.0      # 0-100
    is_etf:              bool                 = False
    is_bluechip:         bool                 = False
    sector:              str                  = ""
    drawdown_52w:        float | None         = None     # negativo ou None
    dividend_yield:      float | None         = None     # 0.025 = 2.5%
    classify_category:   str | None           = None     # output de score.classify_dip_category
    pred_up:             float | None         = None     # MLResult.pred_up
    pred_down:           float | None         = None     # MLResult.pred_down (informativo)
    win_prob:            float | None         = None     # MLResult.win_prob (calibrado)
    ml_label:            str                  = "NO_MODEL"
    model_ready:         bool                 = False
    macro_regime_color:  str                  = "GREEN"  # GREEN | YELLOW | RED
    macro_multiplier:    float                = 1.0      # do macro_semaphore
    cash_available_eur:  float                = 0.0
    monthly_budget_eur:  float                = field(default_factory=lambda: _MONTHLY_BUDGET_EUR)


@dataclass
class AllocationDecision:
    """Resultado read-only do motor. Nada aqui executa nada."""
    ticker:        str
    category:      str
    amount_eur:    float
    confidence:    str            # "Alta" | "Média" | "Baixa"
    rationale:     str            # 1-2 linhas
    exit_rule:     str            # "NEVER" | "THESIS_BREAK" | "TARGET_+15%" | "TARGET_+20%" | "TIME_60D"
    target_price:  float | None   = None
    notes:         list[str]      = field(default_factory=list)
    raw_amount_eur: float         = 0.0  # antes de cap por liquidez (debug)


# ── Helpers internos ──────────────────────────────────────────────────────────

def _ml_signal(ctx: AllocationContext) -> str:
    """Mapeia ML state → sinal categorial usado pelas regras."""
    if not ctx.model_ready or ctx.pred_up is None:
        return "NONE"
    if ctx.pred_up >= _ML_WIN_STRONG:
        return "WIN_STRONG"
    if ctx.pred_up >= _ML_WIN:
        return "WIN"
    if ctx.pred_up > 0:
        return "WEAK"
    return "NO_WIN"


def _regime_multiplier(category: str, regime_color: str) -> float:
    """Modificador de sizing por regime macro.

    ETF_CORE, HOLD_FOREVER, APARTAMENTO continuam autorizados em RED
    (postura defensiva — quality dips são mais raros e devem ser
    aproveitados). GROWTH e FLIP são desactivados.
    """
    defensive_cats = (CAT_ETF_CORE, CAT_HOLD_FOREVER, CAT_APARTAMENTO)
    if regime_color == "RED":
        return 1.0 if category in defensive_cats else 0.0
    if regime_color == "YELLOW":
        return 1.0 if category in defensive_cats else 0.5
    return 1.0  # GREEN


def _ml_confidence_multiplier(ml_signal: str) -> float:
    """Multiplicador de sizing baseado na força do sinal ML."""
    return {
        "WIN_STRONG": 1.2,
        "WIN":        1.0,
        "WEAK":       0.6,
        "NO_WIN":     0.0,
        "NONE":       0.8,   # sem ML não bloqueia, mas reduz convicção
    }.get(ml_signal, 0.0)


def _drawdown_boost(drawdown_52w: float | None) -> float:
    """Boost para FLIP quando o drawdown 52w é severo."""
    if drawdown_52w is None:
        return 1.0
    dd = abs(float(drawdown_52w))
    if dd >= 0.50:
        return 1.3
    if dd >= 0.40:
        return 1.15
    return 1.0


def _confidence_label(ml_signal: str, dip_score: float, regime_color: str) -> str:
    """Label de confiança humano (Alta/Média/Baixa)."""
    if regime_color == "RED":
        return "Baixa"
    if ml_signal == "WIN_STRONG" and dip_score >= 70:
        return "Alta"
    if ml_signal == "WIN" and dip_score >= 60:
        return "Alta"
    if ml_signal in ("WIN", "WIN_STRONG"):
        return "Média"
    if dip_score >= 70:
        return "Média"
    return "Baixa"


def _exit_rule(category: str, ml_signal: str) -> tuple[str, float | None]:
    """Devolve (exit_rule, target_pct). target_pct=None significa sem target."""
    if category in (CAT_ETF_CORE, CAT_HOLD_FOREVER):
        return "NEVER", None
    if category == CAT_APARTAMENTO:
        return "THESIS_BREAK", None
    if category == CAT_GROWTH:
        target = 0.20 if ml_signal == "WIN_STRONG" else 0.15
        label  = "TARGET_+20%" if target == 0.20 else "TARGET_+15%"
        return label, target
    if category == CAT_FLIP:
        return "TIME_60D", 0.15
    return "NONE", None


# ── Decisão principal ─────────────────────────────────────────────────────────

def _classify(ctx: AllocationContext) -> tuple[str, str]:
    """Devolve (category, base_rationale). Não faz sizing."""

    # PASS rules (ordem importa)
    if not ctx.ticker:
        return CAT_PASS, "Ticker vazio."

    # ETF Core: tem precedência absoluta — qualquer ETF vai para core (DCA)
    if ctx.is_etf:
        return CAT_ETF_CORE, "ETF — alimentado via DCA mensal (anchor passivo)."

    # ML signal e dip score — as decisões abaixo dependem destes
    sig = _ml_signal(ctx)

    # PASS: dip score muito baixo OU regime RED+não-defensivo+ML morto
    if ctx.dip_score < _DIP_PASS_MAX:
        return CAT_PASS, f"Dip score {ctx.dip_score:.0f} < {_DIP_PASS_MAX} — sem convicção."

    if sig == "NO_WIN" and ctx.classify_category not in ("🏗️ Hold Forever", "🏠 Apartamento"):
        return CAT_PASS, "ML prevê retorno negativo a 60 d e tese não é defensiva."

    # HOLD_FOREVER: blue chip + score >= 70 (mesmo critério do score.classify_dip_category)
    if ctx.is_bluechip and ctx.dip_score >= _DIP_HOLD_FOREVER_MIN:
        return CAT_HOLD_FOREVER, (
            f"Blue chip com dip score {ctx.dip_score:.0f} ≥ {_DIP_HOLD_FOREVER_MIN}. "
            f"Acumular sem target de venda."
        )

    # APARTAMENTO: classify_dip_category já decidiu (dividendo + drawdown estrutural)
    if ctx.classify_category and "Apartamento" in ctx.classify_category:
        return CAT_APARTAMENTO, (
            f"Apartamento (dividend yield + drawdown estrutural). "
            f"Acumular em dips, sair só por deterioração de tese."
        )

    # FLIP: drawdown muito severo + ML ainda dá sinal positivo (mesmo que WEAK)
    if ctx.drawdown_52w is not None and abs(float(ctx.drawdown_52w)) >= 0.35 and sig in ("WIN_STRONG", "WIN", "WEAK"):
        return CAT_FLIP, (
            f"Dip severo (drawdown 52w {ctx.drawdown_52w * 100:.0f}%) com sinal ML "
            f"{sig}. Trade táctico de 60d."
        )

    # GROWTH: ML WIN/WIN_STRONG + dip score decente
    if sig in ("WIN_STRONG", "WIN") and ctx.dip_score >= _DIP_GROWTH_MIN:
        return CAT_GROWTH, (
            f"ML {sig} (pred_up {ctx.pred_up:+.1%}) + dip score "
            f"{ctx.dip_score:.0f}. Stock pick activo."
        )

    # Fallback: dip score alto mas ML ausente / fraco → APARTAMENTO defensivo
    # se há dividendo ou GROWTH com aviso senão PASS
    if ctx.dip_score >= 60 and ctx.dividend_yield and ctx.dividend_yield >= 0.02:
        return CAT_APARTAMENTO, (
            f"Dip score {ctx.dip_score:.0f} + dividend yield {ctx.dividend_yield * 100:.1f}% — "
            f"tratar como apartamento defensivo (sem ML claro)."
        )

    return CAT_PASS, (
        f"Dip score {ctx.dip_score:.0f}, ML {sig}. "
        f"Sem encaixe limpo em nenhuma categoria — esperar sinal melhor."
    )


def _size(ctx: AllocationContext, category: str) -> tuple[float, float, list[str]]:
    """Devolve (amount_eur, raw_amount, notes). Aplica caps e regime."""
    notes: list[str] = []

    if category == CAT_PASS:
        return 0.0, 0.0, notes

    base_pct = _TARGET_PCT.get(category, 0.0)
    raw      = ctx.monthly_budget_eur * base_pct
    sig      = _ml_signal(ctx)

    # Modificadores por categoria
    regime_mult = _regime_multiplier(category, ctx.macro_regime_color)
    if regime_mult < 1.0:
        notes.append(f"Regime {ctx.macro_regime_color} ×{regime_mult:.1f}")
    if regime_mult == 0.0:
        # Categoria desactivada pelo regime
        return 0.0, raw, notes

    amount = raw * regime_mult

    # ETF_CORE e HOLD_FOREVER ignoram modulação ML (são decisões de tese, não de sinal)
    if category in (CAT_GROWTH, CAT_FLIP):
        ml_mult = _ml_confidence_multiplier(sig)
        amount *= ml_mult
        if ml_mult != 1.0:
            notes.append(f"ML {sig} ×{ml_mult:.1f}")

    # FLIP boost por drawdown severo
    if category == CAT_FLIP:
        dd_mult = _drawdown_boost(ctx.drawdown_52w)
        amount *= dd_mult
        if dd_mult > 1.0:
            notes.append(f"Drawdown severo ×{dd_mult:.2f}")

    # Caps absolutos (single-name risk)
    if category == CAT_FLIP and amount > _MAX_FLIP_EUR:
        notes.append(f"Cap FLIP €{_MAX_FLIP_EUR:.0f}")
        amount = _MAX_FLIP_EUR
    if category == CAT_GROWTH and amount > _MAX_GROWTH_EUR:
        notes.append(f"Cap GROWTH €{_MAX_GROWTH_EUR:.0f}")
        amount = _MAX_GROWTH_EUR

    # Cap por nome (qualquer categoria não-ETF)
    if category != CAT_ETF_CORE:
        max_single = ctx.monthly_budget_eur * _MAX_SINGLE_NAME_PCT
        if amount > max_single:
            notes.append(f"Cap single-name {_MAX_SINGLE_NAME_PCT * 100:.0f}% (€{max_single:.0f})")
            amount = max_single

    # Cap por liquidez disponível
    if ctx.cash_available_eur > 0 and amount > ctx.cash_available_eur:
        notes.append(f"Cap liquidez €{ctx.cash_available_eur:.0f}")
        amount = ctx.cash_available_eur

    # Floor mínimo (T212 fractional)
    if 0 < amount < _MIN_TICKET_EUR:
        notes.append(f"Abaixo do mínimo €{_MIN_TICKET_EUR:.0f} → 0 (acumular cash)")
        amount = 0.0

    return round(amount, 0), raw, notes


def suggest_allocation(ctx: AllocationContext) -> AllocationDecision:
    """Função pública principal. Recebe contexto, devolve decisão.

    Sem I/O, sem efeitos secundários. Pode ser chamada em loops, testes,
    scripts ad-hoc.
    """
    category, base_rationale = _classify(ctx)
    amount, raw, notes       = _size(ctx, category)

    sig                      = _ml_signal(ctx)
    confidence               = _confidence_label(sig, ctx.dip_score, ctx.macro_regime_color)
    exit_rule, target_pct    = _exit_rule(category, sig)

    # Se sizing zero por liquidez/floor mas categoria é válida — mantem categoria
    # mas dá nota clara de "esperar próximo mês".
    if category != CAT_PASS and amount == 0.0:
        notes.append("Sizing zerado — recomenda acumular cash até próximo mês.")

    return AllocationDecision(
        ticker         = ctx.ticker,
        category       = category,
        amount_eur     = amount,
        raw_amount_eur = round(raw, 2),
        confidence     = confidence,
        rationale      = base_rationale,
        exit_rule      = exit_rule,
        target_price   = None,   # preenchido pelo caller se quiser (precisa do preço actual)
        notes          = notes,
    )


# ── Formatação Telegram ───────────────────────────────────────────────────────

_CATEGORY_EMOJI: dict[str, str] = {
    CAT_ETF_CORE:     "🟦",
    CAT_HOLD_FOREVER: "🏗️",
    CAT_APARTAMENTO:  "🏠",
    CAT_GROWTH:       "🚀",
    CAT_FLIP:         "🔄",
    CAT_PASS:         "⏸️",
}

_CATEGORY_LABEL: dict[str, str] = {
    CAT_ETF_CORE:     "ETF Core (DCA)",
    CAT_HOLD_FOREVER: "Hold Forever",
    CAT_APARTAMENTO:  "Apartamento",
    CAT_GROWTH:       "Growth",
    CAT_FLIP:         "Flip",
    CAT_PASS:         "Pass",
}


def format_allocation_telegram(
    decision: AllocationDecision,
    ctx: AllocationContext,
    current_price: float | None = None,
) -> str:
    """Formata uma decisão como mensagem Telegram (Markdown)."""
    emoji = _CATEGORY_EMOJI.get(decision.category, "•")
    label = _CATEGORY_LABEL.get(decision.category, decision.category)

    lines: list[str] = [
        f"{emoji} *{decision.ticker} → {label}*",
        f"_{decision.rationale}_",
        "",
    ]

    # Sizing
    if decision.amount_eur > 0:
        pct_of_budget = decision.amount_eur / ctx.monthly_budget_eur * 100
        lines.append(f"💶 *Sugestão*: €{decision.amount_eur:.0f} _({pct_of_budget:.0f}% do mensal de €{ctx.monthly_budget_eur:.0f})_")
        if current_price is not None and current_price > 0:
            shares = decision.amount_eur / current_price
            lines.append(f"   _≈ {shares:.4f} shares @ ${current_price:.2f}_")
    else:
        lines.append(f"💶 *Sugestão*: €0 — _não comprar agora_")

    # Exit rule
    exit_human = {
        "NEVER":         "Sem target de venda (acumular sempre)",
        "THESIS_BREAK":  "Sair só por deterioração de tese",
        "TARGET_+15%":   "Target +15% ou stop por tese",
        "TARGET_+20%":   "Target +20% ou stop por tese",
        "TIME_60D":      "Time-stop 60 dias OU target +15%",
        "NONE":          "—",
    }.get(decision.exit_rule, decision.exit_rule)
    lines.append(f"🎯 *Saída*: {exit_human}")

    # ML metrics
    if ctx.model_ready and ctx.pred_up is not None:
        ml_line = f"🤖 *ML*: pred_up {ctx.pred_up:+.1%}"
        if ctx.win_prob is not None:
            ml_line += f" | win_prob {ctx.win_prob:.0%}"
        if ctx.pred_down is not None:
            ml_line += f" | pred_down {ctx.pred_down:+.1%}"
        ml_line += f" ({ctx.ml_label})"
        lines.append(ml_line)

    # Confidence + regime
    regime_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(ctx.macro_regime_color, "⚪")
    lines.append(f"⚖️ *Confiança*: {decision.confidence} | Regime: {regime_emoji} {ctx.macro_regime_color}")

    # Notes (só se houver)
    if decision.notes:
        lines.append("")
        lines.append("_" + " · ".join(decision.notes) + "_")

    # Disclaimer read-only
    lines.append("")
    lines.append("_⚠️ Sugestão informativa. Nenhuma ordem foi executada._")

    return "\n".join(lines)
