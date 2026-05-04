"""
test_allocation_engine.py — Smoke + branch-coverage do motor de alocação Fase 4.

Sem pytest dependency: corre directamente como `python -m tests.test_allocation_engine`
ou `python tests/test_allocation_engine.py`. Falha rápida (assert) e imprime
sumário no stdout.

Modelo Fase 4: 4 categorias activas (CORE / HIGH_CONVICTION / GROWTH / FLIP)
+ PASS. Sizing tier-based em fund_score (V2 puro). Pre-profit cap ×0.5,
sectores premium ×1.20, concentration cap 30%/12% com slowdown >8%.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Garante MONTHLY_BUDGET_EUR antes de importar (necessário a partir de PR #15)
os.environ.setdefault("MONTHLY_BUDGET_EUR", "1050")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from allocation_engine import (
    AllocationContext,
    AllocationDecision,
    CAT_CORE,
    CAT_HIGH_CONVICTION,
    CAT_GROWTH,
    CAT_FLIP,
    CAT_PASS,
    # backward-compat aliases
    CAT_ETF_CORE,
    CAT_HOLD_FOREVER,
    CAT_APARTAMENTO,
    suggest_allocation,
    format_allocation_telegram,
)


def _base_ctx(**overrides) -> AllocationContext:
    """Contexto neutro 'GREEN regime, growth-tier', útil para overrides nos testes.

    Default: fund_score=70 (entra em GROWTH), Tech sector (premium ×1.2),
    sem dividend, ML WIN, GREEN regime, €2000 cash, €1050 budget mensal,
    sem posição actual no portfolio.
    """
    defaults = dict(
        ticker="TEST",
        fund_score=70.0,
        is_preprofit=False,
        is_etf=False,
        is_bluechip=False,
        sector="Technology",
        drawdown_52w=-0.20,
        dividend_yield=0.0,
        classify_category=None,
        pred_up=0.07,
        pred_down=-0.10,
        win_prob=0.55,
        ml_label="WIN",
        model_ready=True,
        macro_regime_color="GREEN",
        macro_multiplier=1.0,
        cash_available_eur=2000.0,
        monthly_budget_eur=1050.0,
        existing_position_pct=0.0,
    )
    defaults.update(overrides)
    return AllocationContext(**defaults)


# ── Categorização (Fase 4 nova) ───────────────────────────────────────────────

def test_etf_core_priority():
    """ETF tem prioridade absoluta — independentemente de score/ML."""
    ctx = _base_ctx(
        ticker="VWCE", is_etf=True, fund_score=10.0,
        pred_up=-0.1, ml_label="NO_WIN",
    )
    d = suggest_allocation(ctx)
    assert d.category == CAT_CORE, f"Esperava CORE, got {d.category}"
    assert d.amount_eur > 0, "Core sempre tem allocation positiva"
    assert d.exit_rule == "NEVER"
    # Backward compat: alias deve apontar para o mesmo
    assert CAT_ETF_CORE == CAT_CORE
    print(f"  ✓ Core priority: {d.category}, €{d.amount_eur:.0f}, exit={d.exit_rule}")


def test_high_conviction_bluechip_healthy():
    """Bluechip + fund_score>=65 + not preprofit → HIGH_CONVICTION (sem lista hardcoded)."""
    ctx = _base_ctx(
        ticker="MSFT",
        is_bluechip=True,
        fund_score=72.0,
        is_preprofit=False,
    )
    d = suggest_allocation(ctx)
    assert d.category == CAT_HIGH_CONVICTION, f"Esperava HIGH_CONVICTION, got {d.category}"
    assert d.exit_rule == "THESIS_BREAK"
    assert d.amount_eur > 0
    # Backward compat: aliases mapeiam todos para HIGH_CONVICTION
    assert CAT_HOLD_FOREVER == CAT_HIGH_CONVICTION
    assert CAT_APARTAMENTO == CAT_HIGH_CONVICTION
    print(f"  ✓ High Conviction (bluechip saudável): {d.category}, €{d.amount_eur:.0f}")


def test_high_conviction_excludes_preprofit_bluechip():
    """Mesmo bluechip se is_preprofit=True não vai para HIGH_CONVICTION."""
    ctx = _base_ctx(
        ticker="TSLA",
        is_bluechip=True,
        fund_score=72.0,
        is_preprofit=True,  # FCF negativo desclassifica
    )
    d = suggest_allocation(ctx)
    assert d.category != CAT_HIGH_CONVICTION, \
        f"Bluechip preprofit deveria sair de HIGH_CONVICTION, got {d.category}"
    print(f"  ✓ Pre-profit bluechip excluído de HC: cai em {d.category}")


def test_high_conviction_excludes_low_score_bluechip():
    """Bluechip com fund_score<65 não vai para HIGH_CONVICTION."""
    ctx = _base_ctx(
        ticker="INTC",
        is_bluechip=True,
        fund_score=58.0,  # bluechip a degradar
        is_preprofit=False,
    )
    d = suggest_allocation(ctx)
    assert d.category != CAT_HIGH_CONVICTION
    assert d.category == CAT_GROWTH, f"Esperava GROWTH, got {d.category}"
    print(f"  ✓ Bluechip degradado sai de HC: {d.category}")


def test_growth_default_path():
    """fund_score>=55, não-bluechip → GROWTH."""
    ctx = _base_ctx(ticker="SHOP", fund_score=68.0, is_bluechip=False, ml_label="WIN")
    d = suggest_allocation(ctx)
    assert d.category == CAT_GROWTH, f"Esperava GROWTH, got {d.category}"
    print(f"  ✓ Growth (fund>=55): {d.category}, €{d.amount_eur:.0f}")


def test_growth_target_20_for_win_strong():
    """WIN_STRONG → target +20% (não +15%)."""
    ctx = _base_ctx(ticker="NU", fund_score=78.0, ml_label="WIN_STRONG", pred_up=0.13)
    d = suggest_allocation(ctx)
    assert d.category == CAT_GROWTH
    assert d.exit_rule == "TARGET_+20%", f"Esperava +20%, got {d.exit_rule}"
    print(f"  ✓ Growth WIN_STRONG → +20%: {d.exit_rule}")


def test_flip_conflict_tech():
    """fund_score<55 + ml_bull → FLIP (CONFLICT_TECH)."""
    ctx = _base_ctx(
        ticker="XPEV",
        fund_score=42.0,            # fundamentos fracos
        ml_label="WIN",             # mas ML bull
        drawdown_52w=-0.45,
        is_bluechip=False,
    )
    d = suggest_allocation(ctx)
    assert d.category == CAT_FLIP, f"Esperava FLIP, got {d.category}"
    assert d.exit_rule == "TIME_60D"
    print(f"  ✓ Flip (CONFLICT_TECH): {d.category}, €{d.amount_eur:.0f}")


def test_pass_low_score_no_ml_bull():
    """fund_score<55 + ML não-bull → PASS."""
    ctx = _base_ctx(fund_score=40.0, ml_label="NO_WIN", pred_up=-0.05)
    d = suggest_allocation(ctx)
    assert d.category == CAT_PASS, f"Esperava PASS, got {d.category}"
    assert d.amount_eur == 0.0
    print(f"  ✓ Pass (fund baixo, sem ML bull): {d.category}")


# ── Sizing tiered ─────────────────────────────────────────────────────────────

def test_score_multiplier_85_strong_max():
    """fund>=85 + WIN_STRONG → multiplicador 1.50x (máximo)."""
    base = _base_ctx(fund_score=85.0, ml_label="WIN_STRONG", pred_up=0.15, sector="Health")
    d = suggest_allocation(base)
    # base GROWTH = 1050 * 0.40 = 420; * 1.5 = 630; cash cap... mas €2000 disponível
    # Sem sector premium (Health), sem preprofit → 420 * 1.5 = 630
    assert d.amount_eur >= 600.0 and d.amount_eur <= 700.0, \
        f"Tier máximo deveria dar ~€630, got €{d.amount_eur}"
    print(f"  ✓ Score 85+WIN_STRONG → tier 1.5x: €{d.amount_eur:.0f}")


def test_score_multiplier_55_min_tier():
    """fund=55 → multiplicador 0.30x."""
    ctx = _base_ctx(fund_score=55.0, ml_label="WIN", sector="Health")
    d = suggest_allocation(ctx)
    # GROWTH base = 420; * 0.30 = 126
    assert d.amount_eur >= 100.0 and d.amount_eur <= 150.0, \
        f"Tier 0.3 deveria dar ~€126, got €{d.amount_eur}"
    print(f"  ✓ Score 55 → tier 0.3x: €{d.amount_eur:.0f}")


def test_sector_bonus_technology():
    """Technology sector → bonus ×1.20."""
    ctx_tech    = _base_ctx(fund_score=75.0, sector="Technology", ml_label="WIN")
    ctx_health  = _base_ctx(fund_score=75.0, sector="Healthcare", ml_label="WIN")
    d_tech   = suggest_allocation(ctx_tech)
    d_health = suggest_allocation(ctx_health)
    assert d_tech.amount_eur > d_health.amount_eur, \
        f"Technology devia ter bonus, mas €{d_tech.amount_eur} <= €{d_health.amount_eur}"
    # Razão deve ser ~1.20
    ratio = d_tech.amount_eur / d_health.amount_eur
    assert 1.15 <= ratio <= 1.25, f"Bonus esperado ~1.20, got ×{ratio:.2f}"
    print(f"  ✓ Sector premium Tech: €{d_health.amount_eur:.0f} → €{d_tech.amount_eur:.0f} (×{ratio:.2f})")


def test_preprofit_cap_half():
    """is_preprofit → ×0.5 sizing."""
    ctx_clean    = _base_ctx(fund_score=75.0, sector="Healthcare", is_preprofit=False)
    ctx_burnt    = _base_ctx(fund_score=75.0, sector="Healthcare", is_preprofit=True)
    d_clean  = suggest_allocation(ctx_clean)
    d_burnt  = suggest_allocation(ctx_burnt)
    ratio = d_burnt.amount_eur / d_clean.amount_eur
    assert 0.45 <= ratio <= 0.55, f"Pre-profit deveria ×0.5, got ×{ratio:.2f}"
    print(f"  ✓ Pre-profit cap ×0.5: €{d_clean.amount_eur:.0f} → €{d_burnt.amount_eur:.0f}")


# ── Concentration cap ─────────────────────────────────────────────────────────

def test_concentration_cap_default_blocks_at_12pct():
    """existing_position_pct >= 12% (default) → 0."""
    ctx = _base_ctx(fund_score=75.0, existing_position_pct=0.13)
    d = suggest_allocation(ctx)
    assert d.amount_eur == 0.0, f"Cap 12% deveria zerar, got €{d.amount_eur}"
    assert any("Cap 12" in n for n in d.notes)
    print(f"  ✓ Concentration cap 12% blocks: €{d.amount_eur:.0f}, notes={d.notes[-1]}")


def test_concentration_cap_high_conviction_30pct():
    """HIGH_CONVICTION cap é 30%, não 12%."""
    # 25% é demais para default, mas dentro do limite HC
    ctx = _base_ctx(
        ticker="MSFT",
        is_bluechip=True,
        fund_score=72.0,
        existing_position_pct=0.25,
    )
    d = suggest_allocation(ctx)
    assert d.category == CAT_HIGH_CONVICTION
    assert d.amount_eur > 0, f"HC com 25% deveria continuar a comprar, got €{d.amount_eur}"
    print(f"  ✓ HC cap 30%: posição 25% → €{d.amount_eur:.0f} (continua a acumular)")


def test_concentration_slowdown_above_8pct():
    """existing_position_pct > 8% (default) → slowdown ×0.5."""
    ctx_low  = _base_ctx(fund_score=75.0, existing_position_pct=0.05)
    ctx_high = _base_ctx(fund_score=75.0, existing_position_pct=0.10)
    d_low  = suggest_allocation(ctx_low)
    d_high = suggest_allocation(ctx_high)
    ratio = d_high.amount_eur / d_low.amount_eur
    assert 0.45 <= ratio <= 0.55, f"Slowdown ×0.5 esperado, got ×{ratio:.2f}"
    print(f"  ✓ Slowdown >8%: €{d_low.amount_eur:.0f} → €{d_high.amount_eur:.0f}")


# ── Macro regime ──────────────────────────────────────────────────────────────

def test_regime_red_blocks_growth():
    """Regime RED zera GROWTH mas mantém categoria."""
    ctx = _base_ctx(
        fund_score=78.0, ml_label="WIN_STRONG", pred_up=0.12,
        macro_regime_color="RED",
    )
    d = suggest_allocation(ctx)
    assert d.category == CAT_GROWTH
    assert d.amount_eur == 0.0, f"RED deveria zerar GROWTH, got €{d.amount_eur}"
    print(f"  ✓ Regime RED zera Growth: €{d.amount_eur:.0f}")


def test_regime_red_keeps_core():
    """Regime RED não bloqueia CORE."""
    ctx = _base_ctx(ticker="VWCE", is_etf=True, macro_regime_color="RED")
    d = suggest_allocation(ctx)
    assert d.category == CAT_CORE
    assert d.amount_eur > 0, "Core deveria continuar autorizado em RED"
    print(f"  ✓ Regime RED permite Core: €{d.amount_eur:.0f}")


def test_regime_red_keeps_high_conviction():
    """Regime RED não bloqueia HIGH_CONVICTION (postura defensiva preserva quality dips)."""
    ctx = _base_ctx(
        ticker="MSFT", is_bluechip=True, fund_score=72.0,
        macro_regime_color="RED",
    )
    d = suggest_allocation(ctx)
    assert d.category == CAT_HIGH_CONVICTION
    assert d.amount_eur > 0, "HC deveria continuar autorizado em RED"
    print(f"  ✓ Regime RED permite HC: €{d.amount_eur:.0f}")


def test_regime_yellow_halves_growth():
    """Regime YELLOW deve halvar GROWTH (×0.5)."""
    ctx_g = _base_ctx(fund_score=70.0, ml_label="WIN", macro_regime_color="GREEN")
    ctx_y = _base_ctx(fund_score=70.0, ml_label="WIN", macro_regime_color="YELLOW")
    d_g = suggest_allocation(ctx_g)
    d_y = suggest_allocation(ctx_y)
    assert d_g.amount_eur > d_y.amount_eur
    ratio = d_y.amount_eur / d_g.amount_eur
    assert 0.45 <= ratio <= 0.55, f"YELLOW deveria ×0.5, got ×{ratio:.2f}"
    print(f"  ✓ Regime YELLOW halves growth: €{d_g.amount_eur:.0f} → €{d_y.amount_eur:.0f}")


# ── Caps & floor ──────────────────────────────────────────────────────────────

def test_cash_cap():
    """Cash disponível inferior à sugestão deve cap a allocation."""
    ctx = _base_ctx(
        ticker="VWCE", is_etf=True,
        monthly_budget_eur=1050.0,
        cash_available_eur=100.0,
    )
    d = suggest_allocation(ctx)
    assert d.amount_eur <= 100.0, f"Devia cap a €100, got €{d.amount_eur}"
    print(f"  ✓ Cash cap: Core sugeria mais mas cap a €{d.amount_eur:.0f}")


def test_floor_minimum_ticket():
    """Sugestão abaixo de €20 deve ser zerada (acumular cash)."""
    # FLIP path: fund<55 + WIN, mas budget muito pequeno → abaixo do floor
    # base = 100 * 0.10 = 10; ml WIN ×1.0 = 10; sem drawdown boost; tech ×1.2 = 12 → < 20 → 0
    ctx = _base_ctx(
        ticker="XPEV", fund_score=42.0, ml_label="WIN", pred_up=0.07,
        drawdown_52w=-0.10,
        monthly_budget_eur=100.0,
    )
    d = suggest_allocation(ctx)
    assert d.amount_eur == 0.0, f"Esperava €0, got €{d.amount_eur}"
    assert any("mínimo" in n.lower() for n in d.notes), \
        f"Devia ter nota de floor, got {d.notes}"
    print(f"  ✓ Floor minimum: orçamento pequeno → €0")


# ── Backward compat (callers antigos com dip_score) ───────────────────────────

def test_backward_compat_dip_score_field():
    """Caller antigo passa só dip_score=70 — deve funcionar como fund_score=70."""
    ctx = AllocationContext(
        ticker="LEGACY",
        dip_score=70.0,            # campo legado
        # fund_score=0 (default)   # mas será preenchido a partir do dip_score
        ml_label="WIN",
        sector="Healthcare",
        macro_regime_color="GREEN",
        cash_available_eur=2000.0,
    )
    d = suggest_allocation(ctx)
    assert d.category == CAT_GROWTH, f"Esperava GROWTH, got {d.category}"
    assert d.amount_eur > 0
    print(f"  ✓ Backward compat (dip_score só): {d.category}, €{d.amount_eur:.0f}")


# ── Formatação Telegram ───────────────────────────────────────────────────────

def test_format_telegram_smoke():
    """Smoke test do formatador Telegram — não rebenta com input variado."""
    cases = [
        _base_ctx(ticker="VWCE", is_etf=True),
        _base_ctx(ticker="MSFT", is_bluechip=True, fund_score=72.0,
                  ml_label="WIN_STRONG", pred_up=0.12),
        _base_ctx(ticker="XPEV", fund_score=40.0, drawdown_52w=-0.45,
                  ml_label="WIN"),
        _base_ctx(ticker="UNKNOWN", fund_score=20.0, ml_label="NO_WIN"),
    ]
    for ctx in cases:
        d   = suggest_allocation(ctx)
        out = format_allocation_telegram(d, ctx, current_price=125.50)
        assert ctx.ticker in out
        assert "€" in out
        assert "Sugestão" in out
        assert "informativa" in out, "Disclaimer read-only deveria estar presente"
    print(f"  ✓ Telegram format smoke: {len(cases)} contextos, todos OK")


# ── Runner ────────────────────────────────────────────────────────────────────

def main() -> int:
    tests = [
        # Categorização
        test_etf_core_priority,
        test_high_conviction_bluechip_healthy,
        test_high_conviction_excludes_preprofit_bluechip,
        test_high_conviction_excludes_low_score_bluechip,
        test_growth_default_path,
        test_growth_target_20_for_win_strong,
        test_flip_conflict_tech,
        test_pass_low_score_no_ml_bull,
        # Sizing tiered
        test_score_multiplier_85_strong_max,
        test_score_multiplier_55_min_tier,
        test_sector_bonus_technology,
        test_preprofit_cap_half,
        # Concentration
        test_concentration_cap_default_blocks_at_12pct,
        test_concentration_cap_high_conviction_30pct,
        test_concentration_slowdown_above_8pct,
        # Regime
        test_regime_red_blocks_growth,
        test_regime_red_keeps_core,
        test_regime_red_keeps_high_conviction,
        test_regime_yellow_halves_growth,
        # Caps
        test_cash_cap,
        test_floor_minimum_ticket,
        # Backward compat
        test_backward_compat_dip_score_field,
        # Format
        test_format_telegram_smoke,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  ✗ {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: EXCEPTION {type(e).__name__}: {e}")
            failed += 1
    if failed == 0:
        print(f"\n✅ {len(tests)} testes passaram")
        return 0
    print(f"\n❌ {failed}/{len(tests)} testes falharam")
    return 1


if __name__ == "__main__":
    sys.exit(main())
