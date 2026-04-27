"""
Thresholds fundamentais por sector.
Cada sector tem os seus próprios critérios de "fundamentos intactos"
porque um P/E de 40 é barato em Software mas caro em Utilities.
"""

# ── Estrutura por sector ──────────────────────────────────────────────────────
# pe_max       : P/E máximo aceitável (above = expensive)
# pe_fair      : P/E "justo" para o sector (used for discount calc)
# fcf_yield_min: FCF yield mínimo para considerar atrativo
# ev_ebitda_max: EV/EBITDA máximo aceitável
# debt_equity_max: D/E máximo (None = não aplicar)
# revenue_growth_min: crescimento mínimo esperado (0 = não aplica)
# gross_margin_min: margem bruta mínima

SECTOR_CONFIG = {
    "Technology": {
        "label": "💻 Tecnologia",
        "pe_max": 55,
        "pe_fair": 35,
        "fcf_yield_min": 0.02,
        "ev_ebitda_max": 35,
        "debt_equity_max": 1.5,
        "revenue_growth_min": 0.08,
        "gross_margin_min": 0.50,
        "key_metrics": ["P/E vs histórico", "FCF yield", "Revenue growth", "Gross margin"],
        "red_flags": ["FCF negativo", "Revenue decline", "Margens em queda acelerada"],
    },
    "Healthcare": {
        "label": "🏥 Saúde",
        "pe_max": 35,
        "pe_fair": 22,
        "fcf_yield_min": 0.025,
        "ev_ebitda_max": 25,
        "debt_equity_max": 1.0,
        "revenue_growth_min": 0.05,
        "gross_margin_min": 0.40,
        "key_metrics": ["Pipeline R&D", "FCF yield", "Patent cliff risk", "Revenue growth"],
        "red_flags": ["FDA rejection", "Patent expiry sem pipeline", "FCF negativo"],
    },
    "Financial Services": {
        "label": "🏦 Financeiro",
        "pe_max": 20,
        "pe_fair": 13,
        "fcf_yield_min": 0.04,
        "ev_ebitda_max": None,
        "debt_equity_max": None,
        "revenue_growth_min": 0.03,
        "gross_margin_min": 0.0,
        "key_metrics": ["P/B ratio", "ROE", "Net interest margin", "Dividend yield"],
        "red_flags": ["NIM compression", "Credit losses a subir", "Capital ratio em queda"],
    },
    "Consumer Cyclical": {
        "label": "🛍️ Consumo Cíclico",
        "pe_max": 28,
        "pe_fair": 20,
        "fcf_yield_min": 0.03,
        "ev_ebitda_max": 18,
        "debt_equity_max": 2.0,
        "revenue_growth_min": 0.03,
        "gross_margin_min": 0.30,
        "key_metrics": ["Same-store sales", "Inventory turns", "FCF yield", "Brand strength"],
        "red_flags": ["SSS negativo", "Inventory build", "Margin compression por tarifas"],
    },
    "Consumer Defensive": {
        "label": "🛒 Consumo Defensivo",
        "pe_max": 28,
        "pe_fair": 22,
        "fcf_yield_min": 0.03,
        "ev_ebitda_max": 18,
        "debt_equity_max": 2.0,
        "revenue_growth_min": 0.02,
        "gross_margin_min": 0.30,
        "key_metrics": ["Dividend growth", "FCF yield", "Pricing power", "Market share"],
        "red_flags": ["Dividend cut", "Market share loss", "Input cost spike sem pricing power"],
    },
    "Industrials": {
        "label": "🏭 Industrial",
        "pe_max": 28,
        "pe_fair": 20,
        "fcf_yield_min": 0.03,
        "ev_ebitda_max": 18,
        "debt_equity_max": 1.5,
        "revenue_growth_min": 0.03,
        "gross_margin_min": 0.25,
        "key_metrics": ["Backlog growth", "FCF yield", "Operating leverage", "Order book"],
        "red_flags": ["Backlog decline", "Margin compression", "Capex out of control"],
    },
    "Real Estate": {
        "label": "🏢 Real Estate (REIT)",
        "pe_max": 60,
        "pe_fair": 40,
        "fcf_yield_min": 0.04,
        "ev_ebitda_max": 25,
        "debt_equity_max": 3.0,
        "revenue_growth_min": 0.02,
        "gross_margin_min": 0.50,
        "key_metrics": ["FFO yield", "Occupancy rate", "Dividend yield", "Lease escalations"],
        "red_flags": ["Occupancy em queda", "Dividend cut", "Interest rate spike sem hedge"],
    },
    "Energy": {
        "label": "⚡ Energia",
        "pe_max": 18,
        "pe_fair": 12,
        "fcf_yield_min": 0.05,
        "ev_ebitda_max": 10,
        "debt_equity_max": 1.0,
        "revenue_growth_min": 0.0,
        "gross_margin_min": 0.20,
        "key_metrics": ["FCF yield", "Breakeven oil price", "Dividend sustainability", "Reserves"],
        "red_flags": ["FCF negativo a $60 barril", "Dividend insustentável", "Debt spike"],
    },
    "Communication Services": {
        "label": "📡 Comunicação",
        "pe_max": 30,
        "pe_fair": 20,
        "fcf_yield_min": 0.03,
        "ev_ebitda_max": 20,
        "debt_equity_max": 2.0,
        "revenue_growth_min": 0.05,
        "gross_margin_min": 0.40,
        "key_metrics": ["Subscriber growth", "ARPU", "FCF yield", "Content moat"],
        "red_flags": ["Subscriber loss", "ARPU em queda", "Content spend out of control"],
    },
    "Utilities": {
        "label": "💡 Utilities",
        "pe_max": 25,
        "pe_fair": 18,
        "fcf_yield_min": 0.03,
        "ev_ebitda_max": 15,
        "debt_equity_max": 3.0,
        "revenue_growth_min": 0.01,
        "gross_margin_min": 0.30,
        "key_metrics": ["Dividend yield", "Rate base growth", "Regulatory moat", "Payout ratio"],
        "red_flags": ["Regulatory adverso", "Dividend insustentável", "Capex sem retorno"],
    },
    "Basic Materials": {
        "label": "🪨 Materiais",
        "pe_max": 20,
        "pe_fair": 14,
        "fcf_yield_min": 0.04,
        "ev_ebitda_max": 12,
        "debt_equity_max": 1.0,
        "revenue_growth_min": 0.0,
        "gross_margin_min": 0.20,
        "key_metrics": ["FCF yield", "Cost curve position", "Commodity exposure", "Balance sheet"],
        "red_flags": ["Commodity collapse", "Cost inflation", "Leverage too high"],
    },
}

# Fallback para sectores não mapeados
DEFAULT_CONFIG = {
    "label": "📊 Outro",
    "pe_max": 30,
    "pe_fair": 20,
    "fcf_yield_min": 0.03,
    "ev_ebitda_max": 20,
    "debt_equity_max": 2.0,
    "revenue_growth_min": 0.03,
    "gross_margin_min": 0.25,
    "key_metrics": ["P/E", "FCF yield", "Revenue growth"],
    "red_flags": ["FCF negativo", "Revenue decline"],
}


def get_sector_config(sector: str) -> dict:
    return SECTOR_CONFIG.get(sector, DEFAULT_CONFIG)


def score_fundamentals(metrics: dict, sector: str) -> tuple[str, str, list[str]]:
    """
    Devolve (verdict, emoji, reasons[]).
    verdict: COMPRAR / MONITORIZAR / EVITAR
    """
    cfg = get_sector_config(sector)
    bull_signals = []
    bear_signals = []

    pe = metrics.get("pe")
    pe_fair = cfg["pe_fair"]
    pe_max = cfg["pe_max"]

    if pe and pe > 0:
        if pe < pe_fair * 0.75:
            bull_signals.append(f"P/E {pe:.1f}x — 25%+ abaixo do justo ({pe_fair}x) para o sector")
        elif pe < pe_fair:
            bull_signals.append(f"P/E {pe:.1f}x abaixo do justo ({pe_fair}x)")
        elif pe > pe_max:
            bear_signals.append(f"P/E {pe:.1f}x acima do máximo aceitável ({pe_max}x)")

    fcf_yield = metrics.get("fcf_yield")
    if fcf_yield is not None:
        if fcf_yield < 0:
            bear_signals.append(f"FCF negativo ({fcf_yield*100:.1f}%) — empresa a consumir caixa")
        elif fcf_yield > cfg["fcf_yield_min"] * 2:
            bull_signals.append(f"FCF yield {fcf_yield*100:.1f}% — muito atrativo")
        elif fcf_yield > cfg["fcf_yield_min"]:
            bull_signals.append(f"FCF yield {fcf_yield*100:.1f}%")

    rev_growth = metrics.get("revenue_growth")
    if rev_growth is not None:
        min_growth = cfg["revenue_growth_min"]
        if rev_growth < 0:
            bear_signals.append(f"Revenue a CAIR {rev_growth*100:.1f}% — red flag estrutural")
        elif min_growth > 0 and rev_growth < min_growth * 0.5:
            bear_signals.append(f"Revenue growth {rev_growth*100:.1f}% — muito abaixo do mínimo para sector ({min_growth*100:.0f}%)")
        elif rev_growth > min_growth * 1.5:
            bull_signals.append(f"Revenue growth {rev_growth*100:.1f}% — forte")

    gross_margin = metrics.get("gross_margin")
    gm_min = cfg["gross_margin_min"]
    if gross_margin is not None and gm_min > 0:
        if gross_margin < gm_min:
            bear_signals.append(f"Margem bruta {gross_margin*100:.1f}% — abaixo do mínimo para sector ({gm_min*100:.0f}%)")

    analyst_upside = metrics.get("analyst_upside")
    if analyst_upside is not None:
        if analyst_upside > 30:
            bull_signals.append(f"Analistas vêem +{analyst_upside:.0f}% de upside")
        elif analyst_upside > 15:
            bull_signals.append(f"Analistas vêem +{analyst_upside:.0f}% de upside")
        elif analyst_upside < -5:
            bear_signals.append(f"Analistas vêem downside de {analyst_upside:.0f}%")

    ev_ebitda = metrics.get("ev_ebitda")
    max_ev = cfg.get("ev_ebitda_max")
    if ev_ebitda and max_ev and ev_ebitda > max_ev * 1.3:
        bear_signals.append(f"EV/EBITDA {ev_ebitda:.1f}x — caro vs sector")

    # Scoring
    n_bull = len(bull_signals)
    n_bear = len(bear_signals)

    if n_bear >= 2:
        verdict = "EVITAR"
        emoji = "🔴"
    elif n_bull >= 2 and n_bear == 0:
        verdict = "COMPRAR"
        emoji = "🟢"
    elif n_bull >= 1 and n_bear <= 1:
        verdict = "MONITORIZAR"
        emoji = "🟡"
    else:
        verdict = "MONITORIZAR"  # fix: dados insuficientes → MONITORIZAR, não EVITAR
        emoji = "🟡"

    reasons = bull_signals[:2] + bear_signals[:2]
    return verdict, emoji, reasons
