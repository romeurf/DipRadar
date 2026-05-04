"""
allocation_engine.py — Motor read-only de sugestão de alocação (Fase 4).

Dado um sinal (ticker + V2 fund_score + ML metrics + macro regime + liquidez),
o motor sugere:

  • Categoria  : CORE | HIGH_CONVICTION | GROWTH | FLIP | PASS
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

Filosofia de design (Fase 4)
----------------------------
1. Read-only: nunca executa ordens. Só sugere.
2. **Sem listas hardcoded de tickers** — HIGH_CONVICTION é derivada de
   `is_bluechip` + `fund_score>=65` + `not is_preprofit`. A categoria
   sai/entra automaticamente conforme a saúde fundamental da empresa.
3. Conservador por defeito: regime RED zera GROWTH/FLIP; CORE e
   HIGH_CONVICTION continuam autorizados (postura defensiva).
4. Mínimos T212-compatíveis: floor €20 para fractional shares.
5. Sizing tier-based no fund_score (V2 puro, sem ML baked-in):
     fund≥85 + WIN_STRONG → 1.50x
     fund≥75              → 1.00x
     fund≥65              → 0.70x
     fund≥55              → 0.30x
     fund<55              → 0     (PASS)
6. Pre-profit cap ×0.5: empresas que queimam caixa nunca recebem alocação
   máxima, mesmo com sinal técnico forte.
7. Sectores premium (Technology, Energy): bonus ×1.20 — convicção temática
   na tese de IA + transição energética. Lista hardcoded (não env var).
8. Concentration cap: HIGH_CONVICTION até 30% portfolio, restantes 12%;
   slowdown ×0.5 quando posição actual >8% (para não saltar limites).

Backward compat: aliases CAT_ETF_CORE / CAT_HOLD_FOREVER / CAT_APARTAMENTO
mantidos como aliases dos novos nomes para não quebrar consumers externos.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


# ── Orçamento mensal (obrigatório via env var Railway) ──────────────────────
# Define MONTHLY_BUDGET_EUR=<valor> no Railway.
# Arranque falha imediatamente se a variável não estiver definida.
_raw_budget = os.environ.get("MONTHLY_BUDGET_EUR")
if _raw_budget is None:
    raise EnvironmentError(
        "MONTHLY_BUDGET_EUR não está definida. "
        "Adiciona esta variável de ambiente no Railway antes de arrancar."
    )
_MONTHLY_BUDGET_EUR: float = float(_raw_budget)


# ── Constantes ────────────────────────────────────────────────────────────────

# Categorias activas (4 + PASS)
CAT_CORE             = "CORE"             # ETFs (passive DCA)
CAT_HIGH_CONVICTION  = "HIGH_CONVICTION"  # bluechip saudável (derivado de fundamentals)
CAT_GROWTH           = "GROWTH"           # active stock pick (fund_score>=55)
CAT_FLIP             = "FLIP"             # CONFLICT_TECH (fund<55 + ml bull)
CAT_PASS             = "PASS"             # sem encaixe — esperar cash

# Aliases backward-compat (não usar em código novo)
CAT_ETF_CORE     = CAT_CORE
CAT_HOLD_FOREVER = CAT_HIGH_CONVICTION
CAT_APARTAMENTO  = CAT_HIGH_CONVICTION

# Pesos-alvo (fracção do orçamento mensal). Soma = 100%.
_TARGET_PCT: dict[str, float] = {
    CAT_CORE:            0.20,   # €210/mês (com 1050€ budget)
    CAT_HIGH_CONVICTION: 0.30,   # €315
    CAT_GROWTH:          0.40,   # €420
    CAT_FLIP:            0.10,   # €105
}

# Sectores "premium" (tese estratégica IA + transição energética).
# Hardcoded em código, não env var — para ser auditável em git.
HIGH_CONVICTION_SECTORS: frozenset[str] = frozenset({
    "Technology",
    "Energy",
})
_SECTOR_BONUS = 1.20

# Mínimo absoluto T212 (fractional shares trabalham bem a partir de €20)
_MIN_TICKET_EUR = 20.0

# Concentration caps (% do portfolio actual)
_CAP_PCT_HIGH_CONVICTION = 0.30
_CAP_PCT_DEFAULT         = 0.12
_SLOWDOWN_THRESHOLD      = 0.08   # >8% no default → slowdown ×0.5
_SLOWDOWN_MULT           = 0.5

# Pre-profit cap (FCF negativo)
_PREPROFIT_MULT = 0.5

# Score multiplier tiers (segue plano da Fase 4)
def _score_multiplier(fund_score: float, ml_label: str) -> float:
    """Multiplicador de sizing baseado no fund_score (V2 puro).

    Fase 4: tier agressivo no topo (1.50x para 85+ com WIN_STRONG),
    nada abaixo de 55 (cai em PASS).
    """
    if fund_score >= 85.0 and ml_label == "WIN_STRONG":
        return 1.50
    if fund_score >= 75.0:
        return 1.00
    if fund_score >= 65.0:
        return 0.70
    if fund_score >= 55.0:
        return 0.30
    return 0.0


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class AllocationContext:
    """Tudo que o motor precisa para decidir, num único objecto."""
    ticker:                str
    # Score V2 (preferir fund_only_score sem ML baked-in para sizing tiered)
    fund_score:            float                = 0.0      # 0-100
    is_preprofit:          bool                 = False    # FCF<0 → cap ×0.5
    # Categorização
    is_etf:                bool                 = False
    is_bluechip:           bool                 = False    # is_bluechip(fund) — derivado
    sector:                str                  = ""
    drawdown_52w:          float | None         = None     # negativo (-0.35) ou None
    dividend_yield:        float | None         = None     # 0.025 = 2.5%
    classify_category:     str | None           = None     # legado — output de classify_dip_category
    # ML
    pred_up:               float | None         = None
    pred_down:             float | None         = None
    win_prob:              float | None         = None
    ml_label:              str                  = "NO_MODEL"
    model_ready:           bool                 = False
    # Macro
    macro_regime_color:    str                  = "GREEN"  # GREEN | YELLOW | RED
    macro_multiplier:      float                = 1.0      # informativo
    # Sizing inputs
    cash_available_eur:    float                = 0.0
    monthly_budget_eur:    float                = field(default_factory=lambda: _MONTHLY_BUDGET_EUR)
    existing_position_pct: float                = 0.0      # 0.10 = 10% do portfolio actual
    # Backward-compat — aceita callers antigos que passam dip_score
    dip_score:             float                = 0.0


@dataclass
class AllocationDecision:
    """Resultado read-only do motor. Nada aqui executa nada."""
    ticker:        str
    category:      str
    amount_eur:    float
    confidence:    str
    rationale:     str
    exit_rule:     str
    target_price:  float | None   = None
    notes:         list[str]      = field(default_factory=list)
    raw_amount_eur: float         = 0.0


# ── Helpers internos ──────────────────────────────────────────────────────────

def _ml_label_or_derive(ctx: AllocationContext) -> str:
    """Devolve label coerente. Quando ml_label="NO_MODEL" tenta derivar
    via pred_up para callers antigos que só passam pred_up."""
    if ctx.ml_label and ctx.ml_label != "NO_MODEL":
        return ctx.ml_label
    if ctx.model_ready and ctx.pred_up is not None:
        if ctx.pred_up >= 0.10:
            return "WIN_STRONG"
        if ctx.pred_up >= 0.05:
            return "WIN"
        if ctx.pred_up > 0:
            return "WEAK"
        return "NO_WIN"
    return "NO_MODEL"


def _is_ml_bull(label: str) -> bool:
    return label in ("WIN", "WIN_STRONG", "WIN_40")


def _regime_multiplier(category: str, regime_color: str) -> float:
    """Modificador de sizing por regime macro.

    CORE e HIGH_CONVICTION continuam autorizados em RED (postura defensiva
    — quality dips são raros). GROWTH e FLIP são desactivados em RED e
    halved em YELLOW.
    """
    defensive = (CAT_CORE, CAT_HIGH_CONVICTION)
    if regime_color == "RED":
        return 1.0 if category in defensive else 0.0
    if regime_color == "YELLOW":
        return 1.0 if category in defensive else 0.5
    return 1.0  # GREEN


def _drawdown_boost(drawdown_52w: float | None) -> float:
    """Boost para FLIP quando o drawdown 52w é severo."""
    if drawdown_52w is None:
        return 1.0
    dd = abs(float(drawdown_52w))
    if dd >= 0.50:
        return 1.30
    if dd >= 0.40:
        return 1.15
    return 1.0


def _confidence_label(fund_score: float, ml_label: str, regime_color: str) -> str:
    """Label de confiança humano (Alta/Média/Baixa)."""
    if regime_color == "RED":
        return "Baixa"
    if fund_score >= 75 and ml_label == "WIN_STRONG":
        return "Alta"
    if fund_score >= 65 and _is_ml_bull(ml_label):
        return "Alta"
    if _is_ml_bull(ml_label) or fund_score >= 65:
        return "Média"
    return "Baixa"


def _exit_rule(category: str, ml_label: str) -> tuple[str, float | None]:
    """Devolve (exit_rule, target_pct). target_pct=None → sem target."""
    if category == CAT_CORE:
        return "NEVER", None
    if category == CAT_HIGH_CONVICTION:
        return "THESIS_BREAK", None
    if category == CAT_GROWTH:
        target = 0.20 if ml_label == "WIN_STRONG" else 0.15
        label  = "TARGET_+20%" if target == 0.20 else "TARGET_+15%"
        return label, target
    if category == CAT_FLIP:
        return "TIME_60D", 0.15
    return "NONE", None


# ── Decisão: classificação ────────────────────────────────────────────────────

def _classify(ctx: AllocationContext) -> tuple[str, str]:
    """Devolve (category, base_rationale). Não faz sizing.

    Categorização data-driven (sem listas hardcoded de tickers):
      is_etf                                                           → CORE
      is_bluechip + fund_score>=65 + not is_preprofit                  → HIGH_CONVICTION
      fund_score<55 + ml_bull                                          → FLIP
      fund_score>=55                                                   → GROWTH
      else                                                             → PASS
    """
    if not ctx.ticker:
        return CAT_PASS, "Ticker vazio."

    if ctx.is_etf:
        return CAT_CORE, "ETF — alimentado via DCA mensal (anchor passivo)."

    label = _ml_label_or_derive(ctx)

    if ctx.is_bluechip and ctx.fund_score >= 65 and not ctx.is_preprofit:
        return CAT_HIGH_CONVICTION, (
            f"Blue chip saudável (fund_score {ctx.fund_score:.0f}). "
            f"Acumular consistentemente; sair só por deterioração de tese."
        )

    if ctx.fund_score < 55 and _is_ml_bull(label):
        return CAT_FLIP, (
            f"CONFLICT_TECH: ML {label} mas fundamentos fracos "
            f"(fund_score {ctx.fund_score:.0f}). Trade táctico 60d, stop apertado."
        )

    if ctx.fund_score >= 55:
        return CAT_GROWTH, (
            f"Active stock pick (fund_score {ctx.fund_score:.0f}). "
            f"ML {label}."
        )

    return CAT_PASS, (
        f"Score {ctx.fund_score:.0f}/100 sem suporte fundamental nem técnico — "
        f"esperar sinal melhor."
    )


# ── Decisão: sizing ───────────────────────────────────────────────────────────

def _flip_ml_multiplier(label: str) -> float:
    """Modulação ML para FLIP. WIN_STRONG > WIN; WEAK reduz; NO_WIN zera."""
    return {
        "WIN_STRONG": 1.20,
        "WIN":        1.00,
        "WIN_40":     1.00,
        "WEAK":       0.50,
        "NO_WIN":     0.00,
        "NO_MODEL":   0.50,
    }.get(label, 0.50)


def _size(ctx: AllocationContext, category: str) -> tuple[float, float, list[str]]:
    """Devolve (amount_eur, raw_amount, notes). Aplica todos os caps.

    Sizing por categoria:
      • CORE          : DCA passivo (base, sem score_multiplier)
      • HIGH_CONVICTION / GROWTH : tiered score_multiplier (gate em fund<55)
      • FLIP          : base × ml_mult × drawdown_boost (categoria já encoda
                        a tese; score_multiplier não se aplica aqui)
    """
    notes: list[str] = []

    if category == CAT_PASS:
        return 0.0, 0.0, notes

    base_pct = _TARGET_PCT.get(category, 0.0)
    raw_amount = ctx.monthly_budget_eur * base_pct
    label = _ml_label_or_derive(ctx)

    # 1) Sizing inicial — depende da categoria
    if category == CAT_CORE:
        # CORE: DCA passivo, sem gate de fund_score (ETFs são tese estrutural)
        amount = raw_amount

    elif category == CAT_FLIP:
        # FLIP: ML modulation + drawdown boost (categoria já encoda fund<55)
        ml_mult = _flip_ml_multiplier(label)
        if ml_mult == 0.0:
            notes.append(f"ML {label} sem suporte → 0")
            return 0.0, raw_amount, notes
        amount = raw_amount * ml_mult
        if ml_mult != 1.0:
            notes.append(f"ML {label} ×{ml_mult:.2f}")
        dd_mult = _drawdown_boost(ctx.drawdown_52w)
        if dd_mult > 1.0:
            amount *= dd_mult
            notes.append(f"Drawdown severo ×{dd_mult:.2f}")

    else:
        # GROWTH / HIGH_CONVICTION: score_multiplier tiered (Fase 4)
        score_mult = _score_multiplier(ctx.fund_score, label)
        if score_mult == 0.0:
            notes.append(f"Score {ctx.fund_score:.0f} < 55 → 0")
            return 0.0, raw_amount, notes
        notes.append(f"Score {ctx.fund_score:.0f} ×{score_mult:.2f}")
        amount = raw_amount * score_mult

    # 2) Regime macro (RED zera growth/flip, YELLOW halves)
    regime_mult = _regime_multiplier(category, ctx.macro_regime_color)
    if regime_mult < 1.0:
        notes.append(f"Regime {ctx.macro_regime_color} ×{regime_mult:.1f}")
    if regime_mult == 0.0:
        return 0.0, raw_amount, notes
    amount *= regime_mult

    # 3) Pre-profit cap ×0.5 (empresa queima caixa) — não aplica a CORE
    if ctx.is_preprofit and category != CAT_CORE:
        amount *= _PREPROFIT_MULT
        notes.append(f"Pre-profit ×{_PREPROFIT_MULT:.1f}")

    # 4) Sectores premium ×1.20 — não aplica a CORE (ETF não tem sector)
    if ctx.sector in HIGH_CONVICTION_SECTORS and category != CAT_CORE:
        amount *= _SECTOR_BONUS
        notes.append(f"Sector premium ×{_SECTOR_BONUS:.2f}")

    # 5) Concentration cap (% do portfolio actual) — não aplica a CORE
    if category != CAT_CORE:
        cap_pct = _CAP_PCT_HIGH_CONVICTION if category == CAT_HIGH_CONVICTION else _CAP_PCT_DEFAULT
        if ctx.existing_position_pct >= cap_pct:
            notes.append(
                f"Cap {cap_pct*100:.0f}% atingido (actual {ctx.existing_position_pct*100:.0f}%) → 0"
            )
            return 0.0, raw_amount, notes
        if (
            ctx.existing_position_pct > _SLOWDOWN_THRESHOLD
            and category != CAT_HIGH_CONVICTION
        ):
            amount *= _SLOWDOWN_MULT
            notes.append(
                f"Slowdown perto do cap (actual {ctx.existing_position_pct*100:.0f}%) ×{_SLOWDOWN_MULT}"
            )

    # 6) Cap por liquidez disponível
    if ctx.cash_available_eur > 0 and amount > ctx.cash_available_eur:
        notes.append(f"Cap liquidez €{ctx.cash_available_eur:.0f}")
        amount = ctx.cash_available_eur

    # 7) Floor mínimo T212
    if 0 < amount < _MIN_TICKET_EUR:
        notes.append(f"Abaixo do mínimo €{_MIN_TICKET_EUR:.0f} → 0 (acumular cash)")
        amount = 0.0

    return round(amount, 0), raw_amount, notes


# ── Função pública ────────────────────────────────────────────────────────────

def suggest_allocation(ctx: AllocationContext) -> AllocationDecision:
    """Função pública principal. Recebe contexto, devolve decisão.

    Sem I/O, sem efeitos secundários. Pode ser chamada em loops, testes,
    scripts ad-hoc.
    """
    # Backward compat: se caller passou só dip_score (legado), usa-o como fund_score.
    if ctx.fund_score == 0.0 and ctx.dip_score > 0.0:
        ctx.fund_score = ctx.dip_score

    category, base_rationale = _classify(ctx)
    amount, raw, notes       = _size(ctx, category)

    label                    = _ml_label_or_derive(ctx)
    confidence               = _confidence_label(ctx.fund_score, label, ctx.macro_regime_color)
    exit_rule, _             = _exit_rule(category, label)

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
        target_price   = None,
        notes          = notes,
    )


# ── Formatação Telegram ───────────────────────────────────────────────────────

_CATEGORY_EMOJI: dict[str, str] = {
    CAT_CORE:             "🟦",
    CAT_HIGH_CONVICTION:  "🏛️",
    CAT_GROWTH:           "🚀",
    CAT_FLIP:             "🔄",
    CAT_PASS:             "⏸️",
}

_CATEGORY_LABEL: dict[str, str] = {
    CAT_CORE:             "Core (DCA)",
    CAT_HIGH_CONVICTION:  "High Conviction",
    CAT_GROWTH:           "Growth",
    CAT_FLIP:             "Flip",
    CAT_PASS:             "Pass",
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

    if decision.amount_eur > 0:
        pct_of_budget = decision.amount_eur / ctx.monthly_budget_eur * 100
        lines.append(
            f"💶 *Sugestão*: €{decision.amount_eur:.0f} "
            f"_({pct_of_budget:.0f}% do mensal de €{ctx.monthly_budget_eur:.0f})_"
        )
        if current_price is not None and current_price > 0:
            shares = decision.amount_eur / current_price
            lines.append(f"   _≈ {shares:.4f} shares @ ${current_price:.2f}_")
    else:
        lines.append(f"💶 *Sugestão*: €0 — _não comprar agora_")

    exit_human = {
        "NEVER":         "Sem target de venda (acumular sempre)",
        "THESIS_BREAK":  "Sair só por deterioração de tese",
        "TARGET_+15%":   "Target +15% ou stop por tese",
        "TARGET_+20%":   "Target +20% ou stop por tese",
        "TIME_60D":      "Time-stop 60 dias OU target +15%",
        "NONE":          "—",
    }.get(decision.exit_rule, decision.exit_rule)
    lines.append(f"🎯 *Saída*: {exit_human}")

    if ctx.model_ready and ctx.pred_up is not None:
        ml_line = f"🤖 *ML*: pred_up {ctx.pred_up:+.1%}"
        if ctx.win_prob is not None:
            ml_line += f" | win_prob {ctx.win_prob:.0%}"
        if ctx.pred_down is not None:
            ml_line += f" | pred_down {ctx.pred_down:+.1%}"
        ml_line += f" ({ctx.ml_label})"
        lines.append(ml_line)

    regime_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(ctx.macro_regime_color, "⚪")
    lines.append(f"⚖️ *Confiança*: {decision.confidence} | Regime: {regime_emoji} {ctx.macro_regime_color}")

    if decision.notes:
        lines.append("")
        lines.append("_" + " · ".join(decision.notes) + "_")

    lines.append("")
    lines.append("_⚠️ Sugestão informativa. Nenhuma ordem foi executada._")

    return "\n".join(lines)
