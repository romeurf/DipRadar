"""
Score quantitativo de qualidade do dip (0-100 pts, cap 100).

Critérios e pesos:

  FCF (rei do score):
    +20  FCF yield > 5%
    +10  FCF yield > 3%
    -15  FCF negativo E revenue growth < 5%  (value trap real)
    - 5  FCF negativo MAS revenue growth > 10%  (capex de crescimento)
    -10  FCF negativo zona cinzenta (growth 5-10%)

  Crescimento:
    +15  Revenue growth > 10%
    + 5  Revenue growth > 5%

  Qualidade de negócio:
    +10  Gross margin > threshold do sector

  Técnico:
    +10  RSI < 30  |  +5 se < 40

  Catalisador:
    +15  Earnings ≤30 dias
    + 5  Earnings ≤60 dias

  Capitulação:
    +10  Volume spike > 1.5x média

  Consenso externo:
    +10  Analyst upside > 25%

  Insider buying:
    + 8  Compras de insiders nos últimos 90 dias

  Valuation / estrutura:
    +10  Drawdown 52w < -20%
    + 5  Market cap > $10B  (liquidez para re-rating)
    + 5  D/E < 100
    + 5  PE < 75% do pe_fair do sector

  Penalização sector rotation:
    -10  ETF sectorial cair ≥-2% no mesmo dia (dip arrastado pelo sector)

Máximo teórico: ~128 → cap 100
Badges: 🔥 ≥80  ·  ⭐ 55-79  ·  📊 <55

Equivalências antigas (escala 20):
  16/20  →  ~80/100
  11/20  →  ~55/100
  10/20  →  ~50/100  (MIN_DIP_SCORE default → 50)
"""

from datetime import datetime, timedelta
from market_client import get_rsi
from sectors import get_sector_config

_MARGIN_THRESHOLD = {
    "Technology":             0.40,
    "Healthcare":             0.35,
    "Communication Services": 0.35,
    "Real Estate":            0.20,
    "Industrials":            0.30,
    "Consumer Defensive":     0.30,
    "Consumer Cyclical":      0.30,
    "Financial Services":     0.25,
    "Energy":                 0.25,
    "Utilities":              0.20,
    "Basic Materials":        0.25,
}


def _get_insider_bought(symbol: str) -> bool:
    """True se houve compras de insiders nos últimos 90 dias."""
    try:
        import yfinance as yf
        transactions = yf.Ticker(symbol).insider_transactions
        if transactions is None or transactions.empty:
            return False
        cutoff = datetime.now() - timedelta(days=90)
        recent = transactions[
            (transactions.index >= cutoff) &
            (transactions["Shares"].fillna(0) > 0)
        ]
        return not recent.empty
    except Exception:
        return False


def calculate_dip_score(
    fundamentals: dict,
    symbol: str,
    earnings_days: int | None = None,
    sector_change: float | None = None,  # variação % do ETF sectorial no dia
) -> tuple[float, str | None]:
    """
    Devolve (score, rsi_str). Escala 0-100.
    earnings_days  : dias até próximos earnings (None = desconhecido).
    sector_change  : variação % do ETF sectorial (passar de get_sector_change() em main.py).
    """
    score = 0

    rsi_val    = fundamentals.get("rsi") or get_rsi(symbol)
    fcf_yield  = fundamentals.get("fcf_yield")
    rev_growth = fundamentals.get("revenue_growth") or 0
    sector     = fundamentals.get("sector", "")

    # ── FCF (critério principal) ──────────────────────────────────────────
    if fcf_yield is not None:
        if fcf_yield > 0.05:
            score += 20
        elif fcf_yield > 0.03:
            score += 10
        elif fcf_yield < 0:
            if rev_growth < 0.05:
                score -= 15
            elif rev_growth > 0.10:
                score -= 5
            else:
                score -= 10

    # ── Revenue growth ───────────────────────────────────────────────
    if rev_growth > 0.10:
        score += 15
    elif rev_growth > 0.05:
        score += 5

    # ── Gross margin (threshold por sector) ─────────────────────────────
    gross_margin     = fundamentals.get("gross_margin") or 0
    margin_threshold = _MARGIN_THRESHOLD.get(sector, 0.40)
    if gross_margin > margin_threshold:
        score += 10

    # ── RSI oversold ──────────────────────────────────────────────────
    if rsi_val is not None:
        if rsi_val < 30:
            score += 10
        elif rsi_val < 40:
            score += 5

    # ── Earnings próximos (catalisador concreto) ─────────────────────────
    if earnings_days is not None and earnings_days >= 0:
        if earnings_days <= 30:
            score += 15
        elif earnings_days <= 60:
            score += 5

    # ── Volume spike (capitulação real) ───────────────────────────────
    volume         = fundamentals.get("volume") or 0
    average_volume = fundamentals.get("average_volume") or 0
    if volume and average_volume and average_volume > 0 and volume > average_volume * 1.5:
        score += 10

    # ── Analyst upside forte ─────────────────────────────────────────
    analyst_upside = fundamentals.get("analyst_upside") or 0
    if analyst_upside > 25:
        score += 10

    # ── Insider buying ──────────────────────────────────────────────
    if _get_insider_bought(symbol):
        score += 8

    # ── Drawdown 52w significativo ─────────────────────────────────────
    drawdown = fundamentals.get("drawdown_from_high") or 0
    if drawdown < -20:
        score += 10

    # ── Market cap > $10B ─────────────────────────────────────────────
    mc = fundamentals.get("market_cap") or 0
    if mc >= 10_000_000_000:
        score += 5

    # ── D/E baixo ───────────────────────────────────────────────────
    debt_equity = fundamentals.get("debt_equity", 999)
    if debt_equity is not None and debt_equity < 100:
        score += 5

    # ── PE muito abaixo do fair ─────────────────────────────────────
    pe      = fundamentals.get("pe") or 0
    pe_fair = get_sector_config(sector).get("pe_fair", 22)
    if pe and pe > 0 and pe_fair and pe < pe_fair * 0.75:
        score += 5

    # ── Sector rotation penalty ──────────────────────────────────────
    if sector_change is not None and sector_change <= -2.0:
        score -= 10

    score = max(0, min(score, 100))
    rsi_str = f"{rsi_val:.0f}" if rsi_val is not None else None
    return float(score), rsi_str
