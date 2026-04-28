"""
macro_semaphore.py — Semáforo Macro do DipRadar 2.0

Avalia o regime de mercado com base em 4 indicadores:
  1. SPY (S&P 500) vs SMA50 e SMA200
  2. VIX (volatilidade)
  3. Credit Spreads (HYG / LQD ratio) vs SMA50 e SMA200
  4. Yield Curve (T10Y2Y via FRED)

Cada indicador contribui com uma pontuação:
  +2  Verde forte
  +1  Verde moderado
   0  Neutro
  -2  Vermelho

Agregação final:
  Score >= 4  →  🟢 Verde   — position_multiplier = 1.0
  Score 0-3   →  🟡 Amarelo — position_multiplier = 0.5
  Score <= -1 →  🔴 Vermelho — position_multiplier = 0.0

Uso:
  from macro_semaphore import get_macro_regime
  regime = get_macro_regime()
  # regime["color"], regime["position_multiplier"], regime["explanation"]

Variáveis de ambiente necessárias:
  FRED_API_KEY — chave da FRED (https://fred.stlouisfed.org/docs/api/api_key.html)
"""

import os
import logging
from datetime import datetime

# ── SPY e VIX ────────────────────────────────────────────────────────────────

def _fetch_history(symbol: str, period: str = "250d"):
    """Descarrega histórico via yfinance. Devolve Series de Close ou None."""
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period=period, interval="1d")["Close"].dropna()
        if len(hist) < 50:
            logging.warning(f"[semaphore] {symbol}: dados insuficientes ({len(hist)} dias)")
            return None
        return hist
    except Exception as e:
        logging.warning(f"[semaphore] Erro a descarregar {symbol}: {e}")
        return None


def _score_spy(hist) -> tuple[int, str]:
    """
    Verde  (+2): Preço > SMA50 E > SMA200 — bull market pleno
    Amarelo (+0): Preço < SMA50 MAS > SMA200 — correção em bull market
    Vermelho (-2): Preço < SMA200 — tendência de baixa
    """
    if hist is None or len(hist) < 200:
        return 0, "SPY: dados insuficientes"

    price  = float(hist.iloc[-1])
    sma50  = float(hist.iloc[-50:].mean())
    sma200 = float(hist.iloc[-200:].mean())

    if price > sma50 and price > sma200:
        return 2, f"SPY ${price:.2f} > SMA50 ${sma50:.2f} e SMA200 ${sma200:.2f} ✅ Bull market"
    elif price > sma200:
        return 0, f"SPY ${price:.2f} < SMA50 ${sma50:.2f} mas > SMA200 ${sma200:.2f} ⚠️ Correção"
    else:
        return -2, f"SPY ${price:.2f} < SMA200 ${sma200:.2f} 🚨 Tendência de baixa"


def _score_vix(hist) -> tuple[int, str]:
    """
    Verde  (+2): VIX < 20
    Amarelo (+0): VIX 20-30
    Vermelho (-2): VIX > 30
    """
    if hist is None:
        return 0, "VIX: dados insuficientes"

    vix = float(hist.iloc[-1])

    if vix < 20:
        return 2, f"VIX {vix:.1f} < 20 ✅ Volatilidade normal"
    elif vix <= 30:
        return 0, f"VIX {vix:.1f} entre 20-30 ⚠️ Stress moderado"
    else:
        return -2, f"VIX {vix:.1f} > 30 🚨 Pânico — volatilidade extrema"


def _score_credit(hist_hyg, hist_lqd) -> tuple[int, str]:
    """
    Calcula o rácio HYG/LQD e compara com as suas próprias SMA50 e SMA200.

    Verde  (+2): Rácio > SMA50 e SMA200 — apetite por risco elevado
    Amarelo (+0): Rácio < SMA50 mas > SMA200 — início de cautela
    Vermelho (-2): Rácio < SMA200 — fuga sistémica para segurança
    """
    if hist_hyg is None or hist_lqd is None:
        return 0, "Credit Spreads: dados insuficientes"

    # Alinhar os dois históricos pelo índice comum
    import pandas as pd
    ratio = (hist_hyg / hist_lqd).dropna()

    if len(ratio) < 200:
        return 0, f"Credit Spreads: histórico insuficiente ({len(ratio)} dias)"

    current = float(ratio.iloc[-1])
    sma50   = float(ratio.iloc[-50:].mean())
    sma200  = float(ratio.iloc[-200:].mean())

    if current > sma50 and current > sma200:
        return 2, f"HYG/LQD {current:.4f} > SMA50 {sma50:.4f} e SMA200 {sma200:.4f} ✅ Apetite por risco alto"
    elif current > sma200:
        return 0, f"HYG/LQD {current:.4f} < SMA50 {sma50:.4f} mas > SMA200 {sma200:.4f} ⚠️ Cautela crescente"
    else:
        return -2, f"HYG/LQD {current:.4f} < SMA200 {sma200:.4f} 🚨 Fuga sistémica para segurança"


def _score_yield_curve() -> tuple[int, str]:
    """
    Lê o spread T10Y2Y da FRED (10Y Treasury - 2Y Treasury).

    Verde  (+2): spread > 0 — curva normal
    Amarelo (+0): spread entre 0 e -0.5 — inversão ligeira
    Vermelho (-2): spread < -0.5 — inversão severa (sinal histórico de recessão)
    """
    fred_key = os.environ.get("FRED_API_KEY", "")
    if not fred_key:
        logging.warning("[semaphore] FRED_API_KEY não configurada — yield curve ignorada")
        return 0, "Yield Curve: FRED_API_KEY não configurada"

    try:
        from fredapi import Fred
        fred   = Fred(api_key=fred_key)
        series = fred.get_series("T10Y2Y", observation_start="2020-01-01")
        series = series.dropna()
        if series.empty:
            return 0, "Yield Curve: sem dados da FRED"

        spread = float(series.iloc[-1])
        date   = series.index[-1].strftime("%d/%m/%Y")

        if spread > 0:
            return 2, f"Yield Curve T10Y2Y {spread:+.2f}% ({date}) ✅ Curva normal"
        elif spread >= -0.5:
            return 0, f"Yield Curve T10Y2Y {spread:+.2f}% ({date}) ⚠️ Inversão ligeira"
        else:
            return -2, f"Yield Curve T10Y2Y {spread:+.2f}% ({date}) 🚨 Inversão severa"

    except Exception as e:
        logging.warning(f"[semaphore] Erro FRED yield curve: {e}")
        return 0, f"Yield Curve: erro na FRED ({e})"


# ── Função principal ──────────────────────────────────────────────────────────

def get_macro_regime() -> dict:
    """
    Avalia o regime macro atual e devolve um dicionário com:
      color               : "GREEN" | "YELLOW" | "RED"
      label               : "🟢 Verde" | "🟡 Amarelo" | "🔴 Vermelho"
      position_multiplier : 1.0 | 0.5 | 0.0
      total_score         : int (soma dos 4 indicadores)
      explanation         : str (resumo XAI para Telegram)
      indicators          : dict com estado detalhado de cada métrica
      timestamp           : str (hora do cálculo)

    Em caso de falha total, devolve regime AMARELO por defeito (cautela defensiva).
    """
    logging.info("[semaphore] A calcular regime macro...")

    # ── Descarregar dados ─────────────────────────────────────────────────
    hist_spy = _fetch_history("SPY",  period="250d")
    hist_vix = _fetch_history("^VIX", period="60d")
    hist_hyg = _fetch_history("HYG",  period="250d")
    hist_lqd = _fetch_history("LQD",  period="250d")

    # ── Calcular scores individuais ───────────────────────────────────────
    spy_score,    spy_note    = _score_spy(hist_spy)
    vix_score,    vix_note    = _score_vix(hist_vix)
    credit_score, credit_note = _score_credit(hist_hyg, hist_lqd)
    yc_score,     yc_note     = _score_yield_curve()

    total_score = spy_score + vix_score + credit_score + yc_score

    # ── Agregação final ───────────────────────────────────────────────────
    if total_score >= 4:
        color       = "GREEN"
        label       = "🟢 Verde"
        multiplier  = 1.0
        summary     = "Regime favorável. Entradas normais autorizadas."
    elif total_score >= 0:
        color       = "YELLOW"
        label       = "🟡 Amarelo"
        multiplier  = 0.5
        summary     = "Cautela. Reduzir sizing para 50% do normal."
    else:
        color       = "RED"
        label       = "🔴 Vermelho"
        multiplier  = 0.0
        summary     = "Risco sistémico. Novas entradas bloqueadas (exceto Hold Forever)."

    # ── Explanation XAI ───────────────────────────────────────────────────
    explanation = (
        f"*🌍 Semáforo Macro: {label}* (score {total_score:+d}/8)\n"
        f"_{summary}_\n\n"
        f"• {spy_note}\n"
        f"• {vix_note}\n"
        f"• {credit_note}\n"
        f"• {yc_note}"
    )

    regime = {
        "color":                color,
        "label":                label,
        "position_multiplier":  multiplier,
        "total_score":          total_score,
        "explanation":          explanation,
        "indicators": {
            "spy":          {"score": spy_score,    "note": spy_note},
            "vix":          {"score": vix_score,    "note": vix_note},
            "credit":       {"score": credit_score, "note": credit_note},
            "yield_curve":  {"score": yc_score,     "note": yc_note},
        },
        "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
    }

    logging.info(
        f"[semaphore] Regime: {label} | Score: {total_score:+d} | "
        f"SPY:{spy_score:+d} VIX:{vix_score:+d} Credit:{credit_score:+d} YC:{yc_score:+d}"
    )
    return regime
