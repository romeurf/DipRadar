"""
DCF simplificado e métricas de avaliação.
"""


def dcf_intrinsic_value(
    fcf_per_share: float,
    growth_rate: float,
    wacc: float = 0.09,
    terminal_growth: float = 0.025,
    years: int = 5,
) -> float:
    """
    Valor intrínseco por acção via DCF a 5 anos + valor terminal.
    Conservador: WACC 9%, crescimento terminal 2.5%.
    """
    if fcf_per_share <= 0 or wacc <= terminal_growth:
        return 0.0

    pv = 0.0
    cf = fcf_per_share
    for i in range(1, years + 1):
        cf *= (1 + growth_rate)
        pv += cf / (1 + wacc) ** i

    terminal = cf * (1 + terminal_growth) / (wacc - terminal_growth)
    pv += terminal / (1 + wacc) ** years
    return pv


def margin_of_safety(intrinsic: float, current_price: float) -> float:
    """Margem de segurança em % (positivo = barato, negativo = caro)."""
    if not current_price or current_price <= 0:
        return 0.0
    return (intrinsic - current_price) / current_price * 100


def estimate_wacc(sector: str, beta: float | None = None) -> float:
    """
    WACC estimado por sector. Mais preciso se tiveres beta.
    Risk-free rate assumida: 4.5% (US 10Y treasury 2026).
    """
    risk_free = 0.045
    equity_premium = 0.055  # historical ERP

    default_betas = {
        "Technology": 1.3,
        "Healthcare": 0.9,
        "Financial Services": 1.1,
        "Consumer Cyclical": 1.2,
        "Consumer Defensive": 0.7,
        "Industrials": 1.1,
        "Real Estate": 1.0,
        "Energy": 1.1,
        "Communication Services": 1.1,
        "Utilities": 0.6,
        "Basic Materials": 1.1,
    }

    b = beta or default_betas.get(sector, 1.0)
    cost_of_equity = risk_free + b * equity_premium
    cost_of_debt_after_tax = 0.035
    equity_weight = 0.7
    debt_weight = 0.3

    wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt_after_tax
    return round(wacc, 4)


def format_valuation_block(
    fundamentals: dict,
    historical_pe: dict | None,
    sector: str,
) -> str:
    """
    Formata bloco de avaliação para mensagem Telegram.
    historical_pe: dict com pe_hist_avg, pe_hist_median, pe_hist_min, pe_hist_max

    Nota: dividend_yield está SEMPRE em decimal após normalização em market_client.py
    (ex: 0.0263 = 2.63%). Não é necessário qualquer heurística aqui.
    """
    lines = []

    price = fundamentals.get("price")
    pe = fundamentals.get("pe")
    fcf_yield = fundamentals.get("fcf_yield")
    fcf_ps = fundamentals.get("fcf_per_share")
    rev_growth = fundamentals.get("revenue_growth")
    analyst_upside = fundamentals.get("analyst_upside")
    analyst_target = fundamentals.get("analyst_target")
    ev_ebitda = fundamentals.get("ev_ebitda")
    dy = fundamentals.get("dividend_yield")   # sempre decimal: 0.0263 = 2.63%
    payout = fundamentals.get("payout_ratio")  # sempre decimal: 0.45 = 45%
    gross_margin = fundamentals.get("gross_margin")  # decimal
    roe = fundamentals.get("roe")              # decimal

    # P/E actual vs histórico real
    if pe:
        pe_str = f"{pe:.1f}x"
        if historical_pe:
            avg = historical_pe.get("pe_hist_median")  # mediana é mais robusta
            if avg:
                disc = (avg - pe) / avg * 100
                pe_str += f" vs hist. {avg:.1f}x ({disc:+.0f}%)"
                # Mostra range histórico também
                lo = historical_pe.get("pe_hist_min")
                hi = historical_pe.get("pe_hist_max")
                if lo and hi:
                    pe_str += f" \u2014 range 3a: {lo:.0f}\u2013{hi:.0f}x"
        lines.append(f"  \u2022 P/E: {pe_str}")

    # FCF
    if fcf_yield is not None:
        lines.append(f"  \u2022 FCF Yield: {fcf_yield*100:.1f}%")

    # DCF
    if fcf_ps and fcf_ps > 0 and rev_growth is not None and price:
        wacc = estimate_wacc(sector)
        growth = min(max(rev_growth, 0), 0.30)
        intrinsic = dcf_intrinsic_value(fcf_ps, growth, wacc)
        mos = margin_of_safety(intrinsic, price)
        lines.append(f"  \u2022 DCF intr\u00ednseco: ${intrinsic:.1f} (margem: {mos:+.0f}%)")

    if ev_ebitda:
        lines.append(f"  \u2022 EV/EBITDA: {ev_ebitda:.1f}x")

    if rev_growth is not None:
        lines.append(f"  \u2022 Revenue growth: {rev_growth*100:.1f}%")

    if gross_margin is not None:
        lines.append(f"  \u2022 Gross margin: {gross_margin*100:.1f}%")

    if roe is not None:
        lines.append(f"  \u2022 ROE: {roe*100:.1f}%")

    # dividend_yield já é sempre decimal (normalizado na fonte)
    if dy and dy > 0:
        payout_str = f" (payout {payout*100:.0f}%)" if payout and payout > 0 else ""
        lines.append(f"  \u2022 Dividendo: {dy*100:.2f}%{payout_str}")

    if analyst_target:
        lines.append(f"  \u2022 Target analistas: ${analyst_target:.1f} ({analyst_upside:+.0f}%)")

    return "\n".join(lines) if lines else "  \u2022 Dados insuficientes"
