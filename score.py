"""
Score quantitativo de qualidade do dip (0-10 pts).

Critérios:
  +2  FCF yield > 5%  |  +1 se > 3%
  +2  Revenue growth > 10%  |  +1 se > 5%
  +1  Gross margin > 40%
  +2  RSI < 30  |  +1 se < 40
  +1  D/E < 100
  +1  PE atual < 75% do pe_fair do sector
  +1  Drawdown 52w < -20%
  +1  Analyst upside > 25%

Total máximo: 10 pts (cap)
"""

from market_client import get_rsi
from sectors import get_sector_config


def calculate_dip_score(fundamentals: dict, symbol: str) -> tuple[float, str | None]:
    """
    Devolve (score, rsi_str).
    score é float para consistência com o resto do código.
    """
    score = 0

    rsi_val = fundamentals.get("rsi") or get_rsi(symbol)

    # ── FCF yield ────────────────────────────────────────────────────
    fcf_yield = fundamentals.get("fcf_yield", 0) or 0
    if fcf_yield > 0.05:
        score += 2
    elif fcf_yield > 0.03:
        score += 1

    # ── Revenue growth ──────────────────────────────────────────────
    rev_growth = fundamentals.get("revenue_growth", 0) or 0
    if rev_growth > 0.10:
        score += 2
    elif rev_growth > 0.05:
        score += 1

    # ── Gross margin ────────────────────────────────────────────────
    gross_margin = fundamentals.get("gross_margin", 0) or 0
    if gross_margin > 0.40:
        score += 1

    # ── RSI oversold ────────────────────────────────────────────────
    if rsi_val is not None:
        if rsi_val < 30:
            score += 2
        elif rsi_val < 40:
            score += 1

    # ── D/E baixo ───────────────────────────────────────────────────
    debt_equity = fundamentals.get("debt_equity", 999)
    if debt_equity is not None and debt_equity < 100:
        score += 1

    # ── PE muito abaixo do fair ──────────────────────────────────────
    pe = fundamentals.get("pe") or 0
    sector = fundamentals.get("sector", "")
    pe_fair = get_sector_config(sector).get("pe_fair", 22)
    if pe and pe > 0 and pe_fair and pe < pe_fair * 0.75:
        score += 1

    # ── Drawdown 52w significativo ───────────────────────────────────
    drawdown = fundamentals.get("drawdown_from_high") or 0
    if drawdown < -20:
        score += 1

    # ── Analyst upside forte ─────────────────────────────────────────
    # Consenso de analistas > 25% de upside é sinal de que a queda é exagerada
    analyst_upside = fundamentals.get("analyst_upside") or 0
    if analyst_upside > 25:
        score += 1

    score = min(score, 10)
    rsi_str = f"{rsi_val:.0f}" if rsi_val is not None else None
    return float(score), rsi_str
