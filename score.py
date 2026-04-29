"""
score.py — Motor Quantitativo Institucional (DipRadar 2.0)

SUBSTITUI as heurísticas de limites fixos pelo motor estatístico:
  Z-Scores + Sigmóide  →  normalização contínua e resistente a outliers
  Quality / Value / Timing  →  três hemisférios ponderados
  Confidence Penalty  →  penaliza scores com dados em falta
  ML Prob Multiplier  →  integra a probabilidade WIN do classificador

Equação final:
  base_score  = 0.50 * quality + 0.30 * value + 0.20 * timing
  final_score = base_score * ml_prob * confidence * 100

API pública (compatível com toda a base de código existente):
  calculate_score(features, ml_prob)           — motor puro
  score_from_fundamentals(fund, ml_prob)       — adaptador market_client
  format_score_v2_breakdown(result)            — bloco Telegram
  calculate_dip_score(fund, sym, ...)          — shim retro-compat
  build_score_breakdown(fund, sym, ...)        — shim retro-compat
  is_bluechip(fund)                            — mantido sem alterações
  classify_dip_category(fund, score, bc_flag)  — mantido sem alterações
  CATEGORY_HOLD_FOREVER / APARTAMENTO / ROTACAO
"""

from __future__ import annotations

import math
import logging
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# 0. Constantes de categoria — fonte da verdade única
# ---------------------------------------------------------------------------

CATEGORY_HOLD_FOREVER = "🏗️ Hold Forever"
CATEGORY_APARTAMENTO  = "🏠 Apartamento"
CATEGORY_ROTACAO      = "🔄 Rotação"


# ---------------------------------------------------------------------------
# 1. Thresholds de sector (partilhados por is_bluechip e classify_dip_category)
# ---------------------------------------------------------------------------

_MARGIN_THRESHOLD: dict[str, float] = {
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

_APARTAMENTO_YIELD_THRESHOLD: dict[str, float] = {
    "Technology":             0.025,
    "Communication Services": 0.030,
    "Healthcare":             0.020,
    "Consumer Defensive":     0.025,
    "Consumer Cyclical":      0.025,
    "Industrials":            0.020,
    "Financial Services":     0.030,
    "Energy":                 0.035,
    "Utilities":              0.030,
    "Real Estate":            0.035,
    "Basic Materials":        0.025,
}


# ---------------------------------------------------------------------------
# 2. Médias e desvios empíricos para Z-Scores
# ---------------------------------------------------------------------------

_DEFAULT_MEAN: dict[str, float] = {
    "roic":           0.12,
    "fcf_margin":     0.08,
    "revenue_growth": 0.07,
    "debt_equity":  100.0,
    "pe":            22.0,
    "fcf_yield":      0.04,
}

_DEFAULT_STD: dict[str, float] = {
    "roic":           0.08,
    "fcf_margin":     0.07,
    "revenue_growth": 0.12,
    "debt_equity":   80.0,
    "pe":            15.0,
    "fcf_yield":      0.03,
}

_SECTOR_PE_MEAN: dict[str, float] = {
    "Technology":             30.0,
    "Healthcare":             25.0,
    "Communication Services": 22.0,
    "Financial Services":     13.0,
    "Consumer Cyclical":      20.0,
    "Consumer Defensive":     20.0,
    "Industrials":            18.0,
    "Energy":                 12.0,
    "Utilities":              16.0,
    "Real Estate":            35.0,
    "Basic Materials":        14.0,
}

_SECTOR_PE_STD: dict[str, float] = {
    "Technology":             18.0,
    "Healthcare":             15.0,
    "Communication Services": 14.0,
    "Financial Services":      6.0,
    "Consumer Cyclical":      12.0,
    "Consumer Defensive":      8.0,
    "Industrials":            10.0,
    "Energy":                  7.0,
    "Utilities":               6.0,
    "Real Estate":            20.0,
    "Basic Materials":         8.0,
}

_SECTOR_ROIC_MEAN: dict[str, float] = {
    "Technology":             0.20,
    "Healthcare":             0.15,
    "Communication Services": 0.12,
    "Financial Services":     0.10,
    "Consumer Cyclical":      0.12,
    "Consumer Defensive":     0.18,
    "Industrials":            0.10,
    "Energy":                 0.08,
    "Utilities":              0.06,
    "Real Estate":            0.06,
    "Basic Materials":        0.09,
}


# ---------------------------------------------------------------------------
# 3. Função Sigmóide — o "esmagador de outliers"
# ---------------------------------------------------------------------------

def z_to_score(z: float | np.floating) -> float:
    """
    Converte um Z-Score num valor contínuo em [0, 1] via sigmóide.

      f(z) = 1 / (1 + exp(-z))

    z = 0  → 0.50 (neutro)    z = +2 → 0.88    z = -2 → 0.12

    Totalmente seguro contra NaN/Inf; devolve 0.5 se entrada inválida.
    """
    try:
        z = float(z)
        if not math.isfinite(z):
            return 0.5
        z = max(-10.0, min(10.0, z))
        return float(1.0 / (1.0 + np.exp(-z)))
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# 4. Utilitários internos
# ---------------------------------------------------------------------------

def _safe_float(v: Any, fallback: float = float("nan")) -> float:
    if v is None:
        return fallback
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


def _z(value: float, mean: float, std: float) -> float:
    if std <= 0:
        return 0.0
    return (value - mean) / std


# ---------------------------------------------------------------------------
# 5. Hemisfério A — Quality (50%)
# ---------------------------------------------------------------------------

def _compute_quality(
    features: dict,
    sector: str,
    missing: list[str],
    n_total: list[int],
) -> float:
    scores: list[float] = []

    roic_mean = _SECTOR_ROIC_MEAN.get(sector, _DEFAULT_MEAN["roic"])
    roic_std  = _DEFAULT_STD["roic"]

    # ROIC (com fallback para fcf_yield como proxy)
    n_total.append(1)
    roic = _safe_float(features.get("roic"))
    if math.isnan(roic):
        roic = _safe_float(features.get("fcf_yield"))
    if math.isnan(roic):
        missing.append("roic")
        scores.append(0.5)
    else:
        scores.append(z_to_score(_z(roic, roic_mean, roic_std)))

    # FCF Margin (com fallback para gross_margin)
    n_total.append(1)
    fcf_margin = _safe_float(features.get("fcf_margin"))
    if math.isnan(fcf_margin):
        fcf_margin = _safe_float(features.get("gross_margin") or features.get("fcf_yield"))
    if math.isnan(fcf_margin):
        missing.append("fcf_margin")
        scores.append(0.5)
    else:
        scores.append(z_to_score(_z(fcf_margin, _DEFAULT_MEAN["fcf_margin"], _DEFAULT_STD["fcf_margin"])))

    # Revenue Growth
    n_total.append(1)
    rev_growth = _safe_float(features.get("revenue_growth"))
    if math.isnan(rev_growth):
        missing.append("revenue_growth")
        scores.append(0.5)
    else:
        scores.append(z_to_score(_z(rev_growth, _DEFAULT_MEAN["revenue_growth"], _DEFAULT_STD["revenue_growth"])))

    # Debt/Equity — invertido: menor dívida = melhor
    n_total.append(1)
    de = _safe_float(features.get("debt_equity"))
    if math.isnan(de):
        missing.append("debt_equity")
        scores.append(0.5)
    else:
        scores.append(z_to_score(-_z(de, _DEFAULT_MEAN["debt_equity"], _DEFAULT_STD["debt_equity"])))

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 6. Hemisfério B — Value (30%)
# ---------------------------------------------------------------------------

def _compute_value(
    features: dict,
    sector: str,
    missing: list[str],
    n_total: list[int],
) -> float:
    scores: list[float] = []

    pe_mean = _SECTOR_PE_MEAN.get(sector, _DEFAULT_MEAN["pe"])
    pe_std  = _SECTOR_PE_STD.get(sector, _DEFAULT_STD["pe"])

    # P/E — invertido: PE baixo = barato = bom
    n_total.append(1)
    pe = _safe_float(features.get("pe"))
    if math.isnan(pe) or pe <= 0:
        missing.append("pe")
        scores.append(0.5)
    else:
        scores.append(z_to_score(-_z(pe, pe_mean, pe_std)))

    # FCF Yield — maior = melhor
    n_total.append(1)
    fcf_yield = _safe_float(features.get("fcf_yield"))
    if math.isnan(fcf_yield):
        missing.append("fcf_yield")
        scores.append(0.5)
    else:
        scores.append(z_to_score(_z(fcf_yield, _DEFAULT_MEAN["fcf_yield"], _DEFAULT_STD["fcf_yield"])))

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 7. Hemisfério C — Timing (20%)
# ---------------------------------------------------------------------------

def _compute_timing(
    features: dict,
    missing: list[str],
    n_total: list[int],
) -> float:
    scores: list[float] = []

    # RSI — direto, sem Z-score (já é adimensional [0, 100])
    n_total.append(1)
    rsi = _safe_float(features.get("rsi"))
    if math.isnan(rsi) or not (0 <= rsi <= 100):
        missing.append("rsi")
        scores.append(0.5)
    else:
        scores.append(1.0 - (rsi / 100.0))

    # Drawdown 52w (valor negativo, ex: -30 = queda de 30%)
    # Queda maior → z mais positivo → score mais alto (oportunidade)
    n_total.append(1)
    drawdown = _safe_float(features.get("drawdown_from_high"))
    if math.isnan(drawdown):
        missing.append("drawdown_from_high")
        scores.append(0.5)
    else:
        # Média empírica: -15%, std 15%; inverte sinal (queda grande = boa)
        z_dd = -_z(drawdown, -15.0, 15.0)
        scores.append(z_to_score(z_dd))

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 8. Value Trap Gate
# ---------------------------------------------------------------------------

def _is_value_trap(features: dict) -> bool:
    """
    True se revenue_growth < 0 E FCF é negativo.
    Negócio em dupla contracção — quality penalizada em 50%.
    """
    rev_growth = _safe_float(features.get("revenue_growth"), fallback=0.0)
    fcf        = _safe_float(
        features.get("fcf_margin") or features.get("fcf_yield"),
        fallback=0.0,
    )
    return bool(rev_growth < 0 and fcf < 0)


# ---------------------------------------------------------------------------
# 9. Motor principal
# ---------------------------------------------------------------------------

def calculate_score(
    features: dict,
    ml_prob: float | None = None,
) -> dict:
    """
    Motor quantitativo institucional.

    Parâmetros
    ----------
    features : dict
        Chaves esperadas: roic, fcf_margin, fcf_yield, revenue_growth,
        debt_equity, pe, rsi, drawdown_from_high, sector, gross_margin
    ml_prob : float | None
        Probabilidade WIN do classificador ML [0, 1]. None = 1.0.

    Retorno
    -------
    dict: final_score, quality_score, value_score, timing_score,
          confidence, is_value_trap, skip_recommended, missing_fields
    """
    sector: str = str(features.get("sector") or "")

    missing: list[str] = []
    n_total: list[int] = []

    try:
        quality = _compute_quality(features, sector, missing, n_total)
    except Exception as exc:
        logging.warning(f"[score] quality hemisphere error: {exc}")
        quality = 0.5

    try:
        value = _compute_value(features, sector, missing, n_total)
    except Exception as exc:
        logging.warning(f"[score] value hemisphere error: {exc}")
        value = 0.5

    try:
        timing = _compute_timing(features, missing, n_total)
    except Exception as exc:
        logging.warning(f"[score] timing hemisphere error: {exc}")
        timing = 0.5

    # Value Trap Gate
    vt = _is_value_trap(features)
    if vt:
        quality *= 0.5
        logging.debug("[score] value trap detected — quality halved")

    # Confidence
    total_attempted = len(n_total)
    n_missing       = len(missing)
    n_valid         = max(0, total_attempted - n_missing)
    confidence      = (n_valid / total_attempted) if total_attempted > 0 else 0.0
    skip            = confidence < 0.6

    if skip:
        logging.debug(
            f"[score] skip_recommended=True — confidence={confidence:.2f} "
            f"(missing: {missing})"
        )

    # ML weight
    ml_weight = 1.0 if ml_prob is None else float(np.clip(ml_prob, 0.0, 1.0))

    # Score final
    base_score  = (0.50 * quality) + (0.30 * value) + (0.20 * timing)
    raw_final   = base_score * ml_weight * confidence * 100.0
    final_score = float(np.clip(raw_final, 0.0, 100.0))

    return {
        "final_score":      round(final_score, 2),
        "quality_score":    round(quality,    4),
        "value_score":      round(value,      4),
        "timing_score":     round(timing,     4),
        "confidence":       round(confidence, 4),
        "is_value_trap":    vt,
        "skip_recommended": skip,
        "missing_fields":   missing,
    }


# ---------------------------------------------------------------------------
# 10. Bridge de compatibilidade — adaptador para o dicionário do market_client
# ---------------------------------------------------------------------------

def score_from_fundamentals(
    fundamentals: dict,
    ml_prob: float | None = None,
) -> dict:
    """
    Adapta o dicionário de get_fundamentals() ao motor calculate_score().
    Mapeamento:
      gross_margin  → fcf_margin proxy (quando fcf_margin não existe)
      todos os outros campos são passados directamente
    """
    features = {
        "roic":               fundamentals.get("roic"),
        "fcf_margin":         fundamentals.get("fcf_margin") or fundamentals.get("gross_margin"),
        "fcf_yield":          fundamentals.get("fcf_yield"),
        "revenue_growth":     fundamentals.get("revenue_growth"),
        "debt_equity":        fundamentals.get("debt_equity"),
        "pe":                 fundamentals.get("pe"),
        "rsi":                fundamentals.get("rsi"),
        "drawdown_from_high": fundamentals.get("drawdown_from_high"),
        "sector":             fundamentals.get("sector"),
    }
    return calculate_score(features, ml_prob=ml_prob)


# ---------------------------------------------------------------------------
# 11. Formata o breakdown para Telegram
# ---------------------------------------------------------------------------

def format_score_v2_breakdown(result: dict) -> str:
    """
    Gera um bloco de texto legível para o Telegram a partir do resultado
    de calculate_score() / score_from_fundamentals().
    """
    fs   = result["final_score"]
    q    = result["quality_score"]
    v    = result["value_score"]
    t    = result["timing_score"]
    conf = result["confidence"]
    vt   = result["is_value_trap"]
    skip = result["skip_recommended"]
    miss = result["missing_fields"]

    badge = "🔥" if fs >= 80 else ("⭐" if fs >= 55 else "📊")
    lines = [
        f"{badge} *Score V2: {fs:.1f}/100*",
        f"  🏗️  Quality *{q*100:.0f}%*  \u00b7  💰 Value *{v*100:.0f}%*  \u00b7  ⏱️ Timing *{t*100:.0f}%*",
        f"  📊 Confiança: *{conf*100:.0f}%*" + (" — dados insuficientes ⚠️" if skip else ""),
    ]
    if vt:
        lines.append("  🔴 *Value Trap detectada* — quality penalizada em 50%")
    if miss:
        lines.append(f"  _Em falta: {', '.join(miss)}_")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 12. is_bluechip — mantido sem alterações
# ---------------------------------------------------------------------------

def is_bluechip(fundamentals: dict) -> bool:
    """
    Determina se um stock qualifica como blue chip.
    Critérios:
      - Market cap >= $50B
      - Dividend yield >= 1.5% OU (revenue growth > 5% E gross margin > threshold sectorial)
    """
    mc           = fundamentals.get("market_cap") or 0
    div_yield    = fundamentals.get("dividend_yield") or 0
    rev_growth   = fundamentals.get("revenue_growth") or 0
    gross_margin = fundamentals.get("gross_margin") or 0
    sector       = fundamentals.get("sector", "")
    if mc < 50_000_000_000:
        return False
    threshold = _MARGIN_THRESHOLD.get(sector, 0.40)
    return (div_yield >= 0.015) or (rev_growth > 0.05 and gross_margin > threshold)


# ---------------------------------------------------------------------------
# 13. classify_dip_category — mantido sem alterações
# ---------------------------------------------------------------------------

def classify_dip_category(fundamentals: dict, dip_score: float, is_bluechip_flag: bool) -> str:
    """
    Classifica o dip em uma de 3 categorias estratégicas.
    Devolve sempre uma das constantes CATEGORY_* definidas neste módulo.

    🏗️ Hold Forever — blue chip, score ≥70, margens e balanço excelentes.
    🏠 Apartamento  — drawdown estrutural + dividendo sectorial acima do threshold.
    🔄 Rotação       — fallback táctico para o resto.
    """
    dividend_yield   = fundamentals.get("dividend_yield") or 0
    drawdown         = fundamentals.get("drawdown_from_high") or 0
    fcf_yield        = fundamentals.get("fcf_yield")
    gross_margin     = fundamentals.get("gross_margin") or 0
    debt_equity      = fundamentals.get("debt_equity")
    sector           = fundamentals.get("sector", "")
    margin_threshold = _MARGIN_THRESHOLD.get(sector, 0.40)

    # Hold Forever
    hf_fcf_ok    = (fcf_yield is None) or (fcf_yield > -0.01)
    hf_margin_ok = gross_margin >= margin_threshold
    hf_de_ok     = (debt_equity is None) or (debt_equity < 150)
    if is_bluechip_flag and dip_score >= 70 and hf_fcf_ok and hf_margin_ok and hf_de_ok:
        return CATEGORY_HOLD_FOREVER

    # Apartamento
    apt_yield_min = _APARTAMENTO_YIELD_THRESHOLD.get(sector, 0.020)
    apt_fcf_ok    = (fcf_yield is None) or (fcf_yield > -0.03)
    if (
        dividend_yield >= apt_yield_min
        and drawdown <= -20
        and apt_fcf_ok
        and dip_score >= 45
    ):
        return CATEGORY_APARTAMENTO

    return CATEGORY_ROTACAO


# ---------------------------------------------------------------------------
# 14. Shims de retro-compatibilidade
#     Permitem que main.py e bot_commands.py funcionem sem qualquer alteração.
# ---------------------------------------------------------------------------

def calculate_dip_score(
    fundamentals: dict,
    symbol: str,
    earnings_days: int | None = None,
    sector_change: float | None = None,
    stock_change_pct: float | None = None,
) -> tuple[float, str | None]:
    """
    Shim de compatibilidade. Chama o motor quantitativo internamente.
    Assinatura idêntica à função antiga — zero alterações em main.py.

    Devolve (final_score: float, rsi_str: str | None).
    """
    result  = score_from_fundamentals(fundamentals)
    rsi_val = fundamentals.get("rsi")
    rsi_str = f"{float(rsi_val):.0f}" if rsi_val is not None else None
    return result["final_score"], rsi_str


def build_score_breakdown(
    fundamentals: dict,
    symbol: str,
    earnings_days: int | None = None,
    sector_change: float | None = None,
    stock_change_pct: float | None = None,
) -> str:
    """
    Shim de compatibilidade. Devolve o bloco Telegram do motor quantitativo.
    Assinatura idêntica à função antiga — zero alterações em main.py.
    """
    result = score_from_fundamentals(fundamentals)
    return format_score_v2_breakdown(result)


# ---------------------------------------------------------------------------
# 15. Smoke test  (python score.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = {
        "roic":              0.22,
        "fcf_margin":        0.14,
        "fcf_yield":         0.06,
        "revenue_growth":    0.12,
        "debt_equity":       45.0,
        "pe":                18.0,
        "rsi":               28.0,
        "drawdown_from_high": -32.0,
        "sector":            "Technology",
        "market_cap":        200_000_000_000,
        "gross_margin":      0.65,
        "dividend_yield":    0.008,
    }

    print("=== calculate_score ===")
    res = calculate_score(sample, ml_prob=0.85)
    for k, v in res.items():
        print(f"  {k}: {v}")
    print()
    print(format_score_v2_breakdown(res))
    print()

    print("=== shim calculate_dip_score ===")
    score, rsi_str = calculate_dip_score(sample, "AAPL")
    print(f"  score={score:.1f}  rsi_str={rsi_str}")
    bc = is_bluechip(sample)
    cat = classify_dip_category(sample, score, bc)
    print(f"  is_bluechip={bc}  category={cat}")
    print()

    print("=== Value Trap ===")
    trap = dict(sample, revenue_growth=-0.05, fcf_margin=-0.04, fcf_yield=-0.02)
    res2 = calculate_score(trap, ml_prob=0.70)
    for k, v in res2.items():
        print(f"  {k}: {v}")
    print()

    print("=== Baixa confiança ===")
    sparse = {"pe": 22.0, "rsi": 45.0, "sector": "Healthcare"}
    res3 = calculate_score(sparse, ml_prob=0.60)
    for k, v in res3.items():
        print(f"  {k}: {v}")
