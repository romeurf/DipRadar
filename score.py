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

  *** Earnings: REMOVIDO do score ***
  Earnings NÃO pontuam — entrar antes de earnings é risco puro para flip.
  Os alertas de earnings continuam a ser mostrados como aviso separado.

  Relative Strength vs sector:
    +10  Stock caiu ≤50% do que o sector (correcção idiossincrática ou sobre-reacção)
    - 5  Stock caiu >3x o sector (fuga de capital estrutural)

  Dip within uptrend (SMA50):
    +10  Preço actual > SMA50 (dip dentro de tendência ascendente)

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

import time
import logging
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


def get_sma50(symbol: str) -> float | None:
    """Devolve SMA50 diária. None se não houver dados suficientes."""
    try:
        import yfinance as yf
        time.sleep(2)
        hist = yf.Ticker(symbol).history(period="80d", interval="1d")["Close"].dropna()
        if len(hist) >= 50:
            return float(hist.iloc[-50:].mean())
    except Exception as e:
        logging.debug(f"SMA50 {symbol}: {e}")
    return None


def get_relative_strength(symbol: str, sector_change: float | None) -> float | None:
    """
    Devolve a variação do stock no dia (%).
    A comparação com sector_change é feita externamente no calculate_dip_score.
    Retorna None se não disponível.
    """
    return None


# ── Classificação de categoria ────────────────────────────────────────────

def classify_dip_category(fundamentals: dict, dip_score: float, is_bluechip_flag: bool) -> str:
    """
    Classifica o dip em uma de 3 categorias estratégicas:

    🏛️ Hold Forever — Blue chip de qualidade máxima. Compounder inabalável.
        Critérios: is_bluechip=True AND score >= 70
        Estratégia: nunca vender, acumular em dips.
        Exemplo: MSFT, AAPL, GOOGL

    🏠 Apartamento — Ativo com drawdown estrutural + dividendo que paga
        para esperar a reprecificação. "Cobras a renda enquanto o imóvel valoriza."
        Critérios: dividend_yield >= 2% AND drawdown_52w <= -20% AND FCF não profundamente negativo
        Estratégia: YIELD + REPRECIFICAÇÃO — entrada faseada, stop em corte de dividendo.
        Exemplo: NVO em queda acentuada, pharma com pipeline strong + yield

    🔄 Rotação — Oportunidade táctica. Ineficiência de curto/médio prazo.
        Tudo o resto. Flip clássico com target e stop definidos.
        Estratégia: FLIP — entrada, target +X%, stop -15%.
        Exemplo: PINS, stock de crescimento em correção sem dividendo.

    Devolve string com emoji para usar directamente nas mensagens.
    """
    dividend_yield = fundamentals.get("dividend_yield") or 0
    drawdown       = fundamentals.get("drawdown_from_high") or 0
    fcf_yield      = fundamentals.get("fcf_yield")  # pode ser None

    # Hold Forever: blue chip confirmado com score alto
    if is_bluechip_flag and dip_score >= 70:
        return "🏛️ Hold Forever"

    # Apartamento: dividendo decente + queda estrutural + FCF não catastrófico
    fcf_ok = (fcf_yield is None) or (fcf_yield > -0.02)  # FCF ligeiramente negativo ainda ok
    if dividend_yield >= 0.02 and drawdown <= -20 and fcf_ok:
        return "🏠 Apartamento"

    # Rotação: tudo o resto
    return "🔄 Rotação"


def calculate_dip_score(
    fundamentals: dict,
    symbol: str,
    earnings_days: int | None = None,  # mantido na assinatura para retrocompatibilidade, mas NÃO pontua
    sector_change: float | None = None,
    stock_change_pct: float | None = None,  # variação % do stock no dia
) -> tuple[float, str | None]:
    """
    Devolve (score, rsi_str). Escala 0-100.
    earnings_days    : mantido para compatibilidade — NÃO afecta o score.
    sector_change    : variação % do ETF sectorial (para relative strength e penalização).
    stock_change_pct : variação % do stock no dia (para relative strength).
    """
    score = 0

    rsi_val    = fundamentals.get("rsi") or get_rsi(symbol)
    fcf_yield  = fundamentals.get("fcf_yield")
    rev_growth = fundamentals.get("revenue_growth") or 0
    sector     = fundamentals.get("sector", "")

    # FCF
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

    # Revenue growth
    if rev_growth > 0.10:
        score += 15
    elif rev_growth > 0.05:
        score += 5

    # Gross margin
    gross_margin     = fundamentals.get("gross_margin") or 0
    margin_threshold = _MARGIN_THRESHOLD.get(sector, 0.40)
    if gross_margin > margin_threshold:
        score += 10

    # RSI
    if rsi_val is not None:
        if rsi_val < 30:
            score += 10
        elif rsi_val < 40:
            score += 5

    # ── Earnings: REMOVIDO do score ──────────────────────────────────────
    # earnings_days não pontua — é mostrado como aviso independente no output.

    # Relative Strength vs sector
    if stock_change_pct is not None and sector_change is not None and sector_change != 0:
        ratio = stock_change_pct / sector_change  # ambos negativos → ratio positivo se semelhantes
        if ratio >= 0.5:  # stock caiu ≤50% do que o sector → sobre-reacção relativa
            score += 10
        elif ratio > 3.0:  # stock caiu >3x o sector → fuga estrutural
            score -= 5

    # Dip within uptrend — SMA50
    price = fundamentals.get("price") or 0
    if price > 0:
        sma50 = get_sma50(symbol)
        if sma50 is not None and price > sma50:
            score += 10

    # Volume spike (capitulação)
    volume         = fundamentals.get("volume") or 0
    average_volume = fundamentals.get("average_volume") or 0
    if volume and average_volume and average_volume > 0 and volume > average_volume * 1.5:
        score += 10

    # Analyst upside
    analyst_upside = fundamentals.get("analyst_upside") or 0
    if analyst_upside > 25:
        score += 10

    # Insider buying
    if _get_insider_bought(symbol):
        score += 8

    # Drawdown 52w
    drawdown = fundamentals.get("drawdown_from_high") or 0
    if drawdown < -20:
        score += 10

    # Market cap
    mc = fundamentals.get("market_cap") or 0
    if mc >= 10_000_000_000:
        score += 5

    # D/E
    debt_equity = fundamentals.get("debt_equity", 999)
    if debt_equity is not None and debt_equity < 100:
        score += 5

    # PE vs fair
    pe      = fundamentals.get("pe") or 0
    pe_fair = get_sector_config(sector).get("pe_fair", 22)
    if pe and pe > 0 and pe_fair and pe < pe_fair * 0.75:
        score += 5

    # Sector rotation penalty
    if sector_change is not None and sector_change <= -2.0:
        score -= 10

    score = max(0, min(score, 100))
    rsi_str = f"{rsi_val:.0f}" if rsi_val is not None else None
    return float(score), rsi_str


def build_score_breakdown(
    fundamentals: dict,
    symbol: str,
    earnings_days: int | None = None,
    sector_change: float | None = None,
    stock_change_pct: float | None = None,
) -> list[str]:
    """
    Devolve lista de linhas descritivas para cada critério do score.
    Cada linha começa com ✅ (positivo), ❌ (negativo) ou ⬜ (neutro/não aplicável).
    """
    lines: list[str] = []

    rsi_val    = fundamentals.get("rsi") or get_rsi(symbol)
    fcf_yield  = fundamentals.get("fcf_yield")
    rev_growth = fundamentals.get("revenue_growth") or 0
    sector     = fundamentals.get("sector", "")

    # FCF
    if fcf_yield is not None:
        if fcf_yield > 0.05:
            lines.append(f"✅ FCF yield {fcf_yield*100:.1f}% › 5% (+20)")
        elif fcf_yield > 0.03:
            lines.append(f"✅ FCF yield {fcf_yield*100:.1f}% › 3% (+10)")
        elif fcf_yield < 0:
            if rev_growth < 0.05:
                lines.append(f"❌ FCF negativo + crescimento fraco (-15)")
            elif rev_growth > 0.10:
                lines.append(f"⚠️ FCF negativo mas crescimento forte (-5)")
            else:
                lines.append(f"⚠️ FCF negativo zona cinzenta (-10)")
    else:
        lines.append("⬜ FCF yield — sem dados")

    # Revenue growth
    if rev_growth > 0.10:
        lines.append(f"✅ Revenue growth {rev_growth*100:.1f}% › 10% (+15)")
    elif rev_growth > 0.05:
        lines.append(f"✅ Revenue growth {rev_growth*100:.1f}% › 5% (+5)")
    else:
        lines.append(f"⬜ Revenue growth {rev_growth*100:.1f}% (sem pts)")

    # Gross margin
    gross_margin     = fundamentals.get("gross_margin") or 0
    margin_threshold = _MARGIN_THRESHOLD.get(sector, 0.40)
    if gross_margin > margin_threshold:
        lines.append(f"✅ Gross margin {gross_margin*100:.1f}% › {margin_threshold*100:.0f}% sector (+10)")
    else:
        lines.append(f"⬜ Gross margin {gross_margin*100:.1f}% (abaixo do threshold {margin_threshold*100:.0f}%)")

    # RSI
    if rsi_val is not None:
        if rsi_val < 30:
            lines.append(f"✅ RSI {rsi_val:.0f} — oversold forte (+10)")
        elif rsi_val < 40:
            lines.append(f"✅ RSI {rsi_val:.0f} — oversold moderado (+5)")
        else:
            lines.append(f"⬜ RSI {rsi_val:.0f} — neutro (sem pts)")
    else:
        lines.append("⬜ RSI — sem dados")

    # Earnings — aviso informativo apenas (não pontua)
    if earnings_days is not None and earnings_days >= 0:
        if earnings_days <= 7:
            lines.append(f"🔴 Earnings em {earnings_days}d — RISCO ALTO (não pontua, cuidado em flip)")
        elif earnings_days <= 21:
            lines.append(f"🟡 Earnings em {earnings_days}d — atenção (não pontua)")
        else:
            lines.append(f"📅 Earnings em {earnings_days}d (informativo)")
    else:
        lines.append("⬜ Earnings — data desconhecida")

    # Relative Strength
    if stock_change_pct is not None and sector_change is not None and sector_change != 0:
        ratio = stock_change_pct / sector_change
        if ratio >= 0.5:
            lines.append(f"✅ Relative strength: stock {stock_change_pct:.1f}% vs sector {sector_change:.1f}% — sobre-reacção (+10)")
        elif ratio > 3.0:
            lines.append(f"❌ Relative strength: stock {stock_change_pct:.1f}% vs sector {sector_change:.1f}% — fuga estrutural (-5)")
        else:
            lines.append(f"⬜ Relative strength: {ratio:.1f}x sector (sem pts)")
    else:
        lines.append("⬜ Relative strength — sector change indisponível")

    # SMA50 — dip within uptrend
    price = fundamentals.get("price") or 0
    if price > 0:
        sma50 = get_sma50(symbol)
        if sma50 is not None:
            if price > sma50:
                lines.append(f"✅ Preço ${price:.2f} > SMA50 ${sma50:.2f} — dip em tendência (+10)")
            else:
                lines.append(f"⬜ Preço ${price:.2f} < SMA50 ${sma50:.2f} — tendência quebrada (sem pts)")
        else:
            lines.append("⬜ SMA50 — sem dados suficientes")
    else:
        lines.append("⬜ SMA50 — preço indisponível")

    # Volume spike
    volume         = fundamentals.get("volume") or 0
    average_volume = fundamentals.get("average_volume") or 0
    if volume and average_volume and average_volume > 0:
        ratio = volume / average_volume
        if ratio > 1.5:
            lines.append(f"✅ Volume spike {ratio:.1f}x média — capitulação (+10)")
        else:
            lines.append(f"⬜ Volume {ratio:.1f}x média (sem pts)")
    else:
        lines.append("⬜ Volume — sem dados")

    # Analyst upside
    analyst_upside = fundamentals.get("analyst_upside") or 0
    if analyst_upside > 25:
        lines.append(f"✅ Analyst upside {analyst_upside:.1f}% › 25% (+10)")
    else:
        lines.append(f"⬜ Analyst upside {analyst_upside:.1f}% (sem pts)")

    # Insider buying
    if _get_insider_bought(symbol):
        lines.append("✅ Insider buying nos últimos 90d (+8)")
    else:
        lines.append("⬜ Sem insider buying recente")

    # Drawdown
    drawdown = fundamentals.get("drawdown_from_high") or 0
    if drawdown < -20:
        lines.append(f"✅ Drawdown 52w {drawdown:.1f}% — dip significativo (+10)")
    else:
        lines.append(f"⬜ Drawdown 52w {drawdown:.1f}% (sem pts)")

    # Market cap
    mc = fundamentals.get("market_cap") or 0
    if mc >= 10_000_000_000:
        lines.append(f"✅ Market cap ${mc/1e9:.0f}B › $10B — liquidez (+5)")
    else:
        lines.append(f"⬜ Market cap ${mc/1e9:.1f}B (sem pts)")

    # D/E
    debt_equity = fundamentals.get("debt_equity", 999)
    if debt_equity is not None and debt_equity < 100:
        lines.append(f"✅ D/E {debt_equity:.0f} ‹ 100 — balanço saudável (+5)")
    else:
        lines.append(f"⬜ D/E {debt_equity if debt_equity else 'N/D'} (sem pts)")

    # PE vs fair
    pe      = fundamentals.get("pe") or 0
    pe_fair = get_sector_config(sector).get("pe_fair", 22)
    if pe and pe > 0 and pe_fair and pe < pe_fair * 0.75:
        lines.append(f"✅ PE {pe:.1f} ‹ 75% do fair ({pe_fair*0.75:.1f}) — barato (+5)")
    elif pe and pe > 0:
        lines.append(f"⬜ PE {pe:.1f} (fair sector {pe_fair}) — sem desconto")
    else:
        lines.append("⬜ PE — sem dados")

    # Sector rotation penalty
    if sector_change is not None and sector_change <= -2.0:
        lines.append(f"❌ Sector rotation: ETF {sector_change:.1f}% — penalização (-10)")
    elif sector_change is not None:
        lines.append(f"⬜ Sector ETF {sector_change:.1f}% — sem penalização")

    return lines
