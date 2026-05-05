"""
score.py — Motor Quantitativo Institucional (DipRadar 2.0)

SUBSTITUI as heurísticas de limites fixos pelo motor estatístico:
  Z-Scores + Sigmóide  →  normalização contínua e resistente a outliers
  Quality / Value / Timing  →  três hemisférios ponderados
  Confidence Penalty  →  penaliza scores com dados em falta
  ML Prob Multiplier  →  integra a probabilidade WIN do classificador

Hemisfério Timing — 3 componentes:
  RSI          (peso base 50% quando volume ausente, 35% quando presente)
  Drawdown 52w (peso base 50% quando volume ausente, 35% quando presente)
  Volume Spike (peso 30% — bonus, não penaliza confidence se ausente)

Penalização near-earnings (earnings_days < 14):
  Confidence multiplicada por 0.85 — zona de incerteza pré-relatório.

Equação final:
  base_score  = 0.50 * quality + 0.30 * value + 0.20 * timing
  final_score = base_score * ml_prob * confidence * 100

API pública (compatível com toda a base de código existente):
  calculate_score(features, ml_prob)                    — motor puro
  score_from_fundamentals(fund, ml_prob, earnings_days) — adaptador market_client
  format_score_v2_breakdown(result)                     — bloco Telegram
  calculate_dip_score(fund, sym, ..., ml_prob)          — shim retro-compat
  build_score_breakdown(fund, sym, ..., ml_prob)        — shim retro-compat
  is_bluechip(fund)                                     — mantido sem alterações
  classify_dip_category(fund, score, bc_flag)           — mantido sem alterações
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

_SECTOR_ROIC_STD: dict[str, float] = {
    "Technology":             0.12,
    "Healthcare":             0.10,
    "Communication Services": 0.09,
    "Financial Services":     0.07,
    "Consumer Cyclical":      0.09,
    "Consumer Defensive":     0.08,
    "Industrials":            0.07,
    "Energy":                 0.07,
    "Utilities":              0.04,
    "Real Estate":            0.05,
    "Basic Materials":        0.07,
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
    roic_std  = _SECTOR_ROIC_STD.get(sector, _DEFAULT_STD["roic"])

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
    """
    Três componentes de timing:
      1. RSI          — sinal sobrevendido (sem Z-score, já adimensional)
      2. Drawdown 52w — queda face ao máximo (maior queda = mais oportunidade)
      3. Volume Spike — confirmação de capitulação (BONUS: não conta para confidence)

    Quando volume está disponível:
      timing = 0.70 * base(RSI + drawdown) + 0.30 * volume_score
    Quando volume está ausente:
      timing = base(RSI + drawdown)   ← sem penalização de confidence
    """
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

    base_timing = float(np.mean(scores))

    # ── Volume Spike (bonus — não afecta confidence) ───────────────────────
    # vol/avg = 1.0 → z=0 → 0.50 (neutro)
    # vol/avg = 1.5 → z=+1.0 → 0.73
    # vol/avg = 2.0 → z=+2.0 → 0.88 (capitulação/climax — sinal de reversão)
    # vol/avg = 0.5 → z=-1.0 → 0.27 (volume seco — interesse a decair)
    vol     = _safe_float(features.get("volume"))
    avg_vol = _safe_float(features.get("average_volume"))

    if math.isnan(vol) or math.isnan(avg_vol) or avg_vol <= 0:
        # Sem dados de volume: usa apenas base timing, sem penalização
        return base_timing

    vol_ratio  = vol / avg_vol
    z_vol      = (vol_ratio - 1.0) / 0.5   # std empírico 0.5x de ratio
    vol_score  = z_to_score(z_vol)

    # Blend: 70% base (RSI + drawdown) + 30% volume spike
    return base_timing * 0.70 + vol_score * 0.30


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
# 8a. Funções compostas — Quality Dislocation
# ---------------------------------------------------------------------------

def quality_dislocation(gross_margin: float, drawdown_52w: float, fcf_yield: float) -> float:
    if fcf_yield < 0:
        return 0.0
    return gross_margin * abs(drawdown_52w) / 100.0


def fcf_trend(fcf_yield_current: float | None, fcf_yield_previous: float | None) -> float:
    if fcf_yield_current is not None and fcf_yield_previous is not None:
        if fcf_yield_current > 0 and fcf_yield_previous <= 0:
            return 0.08
        return fcf_yield_current - fcf_yield_previous
    return 0.0


def peg_implicit(pe: float, revenue_growth: float) -> float:
    if pe > 0 and revenue_growth > 0:
        return pe / (revenue_growth * 100.0)
    return 3.0


def unjustified_drawdown(
    drawdown_52w: float,
    gross_margin: float,
    fcf_yield: float,
    revenue_growth: float,
) -> float:
    if abs(drawdown_52w) < 20:
        return 0.0
    health = 0
    if gross_margin > 0.45:
        health += 1
    if fcf_yield > 0.02:
        health += 1
    if revenue_growth > 0.08:
        health += 1
    return abs(drawdown_52w) * (health / 3.0)

# ---------------------------------------------------------------------------
# 8b. Red Flags + Quality Multiplier  (Fase 1 — penalty pós z-score)
# ---------------------------------------------------------------------------
#
# Subtrai pontos a um multiplicador inicial 1.0 conforme detecta sinais
# negativos vindos directamente dos fundamentals brutos. O multiplicador é
# aplicado ao final_score e à confidence — não substitui o motor z-score,
# complementa-o com penalties absolutas que o z-score sectorial dilui demais.
#
# Limites:
#   - PE: yfinance trailingPE; >200 ou <0 sugere distorção (lucro irrelevante)
#   - debt_equity: yfinance devolve em % (3.0 → 300); >300 = alavancado demais
#   - revenue_growth: <0.30 + ROE<0 = pre-profit sem crescimento explosivo
#   - fcf_yield ou fcf_margin: <0 = empresa queima caixa (pre-profit)
#
# is_preprofit dispara skip_recommended=True automaticamente.

def _detect_red_flags(features: dict) -> tuple[float, list[str], bool]:
    """
    Avalia red flags e devolve (quality_multiplier, red_flags, is_preprofit).

    quality_multiplier ∈ [0.10, 1.00] — clipado para baixo a 0.10.
    is_preprofit = True quando FCF/FCF_margin negativo (empresa queima caixa).
    """
    quality_multiplier = 1.0
    red_flags: list[str] = []
    is_preprofit = False

    pe         = features.get("pe")
    fcf_yield  = features.get("fcf_yield")
    fcf_margin = features.get("fcf_margin")
    roe        = features.get("roe")  # opcional — adapter pode não passar
    rev_growth = features.get("revenue_growth")
    debt_equity = features.get("debt_equity")

    # FCF negativo (pre-profit) — proxy: yield ou margin negativos
    fcf = None
    for cand in (fcf_yield, fcf_margin):
        if cand is not None:
            try:
                v = float(cand)
                if math.isfinite(v):
                    fcf = v
                    break
            except (TypeError, ValueError):
                continue
    if fcf is not None and fcf < 0:
        quality_multiplier -= 0.20
        red_flags.append("FCF Negativo")
        is_preprofit = True

    # PE extremo (>200 ou <0) — lucro distorcido
    if pe is not None:
        try:
            pe_v = float(pe)
            if math.isfinite(pe_v) and (pe_v > 200 or pe_v < 0):
                quality_multiplier -= 0.25
                red_flags.append(f"PE Extremo ({pe_v:.0f}x)")
        except (TypeError, ValueError):
            pass

    # ROE negativo sem crescimento forte (revenue_growth < 30%)
    if roe is not None and rev_growth is not None:
        try:
            roe_v = float(roe)
            rg_v  = float(rev_growth)
            if (math.isfinite(roe_v) and math.isfinite(rg_v)
                and roe_v < 0 and rg_v < 0.30):
                quality_multiplier -= 0.30
                red_flags.append("ROE Negativo s/ Crescimento Forte")
        except (TypeError, ValueError):
            pass

    # Dívida/Capitalização > 300% (yfinance devolve em pontos percentuais)
    if debt_equity is not None:
        try:
            de_v = float(debt_equity)
            if math.isfinite(de_v) and de_v > 300:
                quality_multiplier -= 0.20
                red_flags.append(f"D/E Elevado ({de_v:.0f}%)")
        except (TypeError, ValueError):
            pass

    # Combinação Letal — preprofit + PE distorcido alto
    if is_preprofit and pe is not None:
        try:
            pe_v = float(pe)
            if math.isfinite(pe_v) and pe_v > 100:
                quality_multiplier -= 0.15
                red_flags.append("Letal: FCF Neg + PE>100")
        except (TypeError, ValueError):
            pass

    quality_multiplier = max(0.10, quality_multiplier)
    return quality_multiplier, red_flags, is_preprofit

# ---------------------------------------------------------------------------
# 8c. Hemisfério D — Divergência (20%)
# ---------------------------------------------------------------------------

def _compute_divergence(features: dict) -> float:
    """
    Mede a dislocation entre qualidade do negócio e preço de mercado.

    Componentes:
      quality_dislocation  → 40%
      unjustified_drawdown → 35%  (normalizado ÷ 100)
      fcf_trend            → 25%  (clipped e escalado para [0, 1])

    Devolve valor em [0, 1]:
      0.00–0.15 → sem dislocation significativa
      0.15–0.30 → dislocation moderada
      0.30+     → dislocation forte (padrão PINS/NOW)
    """
    gm  = _safe_float(features.get("gross_margin"), 0.0)
    dd  = _safe_float(features.get("drawdown_from_high"), 0.0)
    fcf = _safe_float(features.get("fcf_yield"), 0.0)
    rg  = _safe_float(features.get("revenue_growth"), 0.0)

    fcf_prev = _safe_float(features.get("fcf_yield_previous"), fallback=None)
    fcf_prev = None if math.isnan(fcf_prev) else fcf_prev  # type: ignore[arg-type]

    qd  = quality_dislocation(gm, dd, fcf)
    ud  = unjustified_drawdown(dd, gm, fcf, rg) / 100.0
    ft  = fcf_trend(fcf if not math.isnan(fcf) else None, fcf_prev)

    # Escalar fcf_trend de [-0.10, +0.10] → [0, 1]
    ft_clipped = float(np.clip(ft, -0.10, 0.10))
    ft_scaled  = (ft_clipped + 0.10) / 0.20

    raw = 0.40 * qd + 0.35 * ud + 0.25 * ft_scaled
    return float(np.clip(raw, 0.0, 1.0))

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
        debt_equity, pe, rsi, drawdown_from_high, sector, gross_margin,
        volume, average_volume, earnings_days
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

    # Confidence (baseada apenas nas métricas formais — volume não conta)
    total_attempted = len(n_total)
    n_missing       = len(missing)
    n_valid         = max(0, total_attempted - n_missing)
    data_coverage   = (n_valid / total_attempted) if total_attempted > 0 else 0.0

    # Near-earnings confidence penalty — zona de incerteza pré-relatório
    # earnings_days < 14: −15% na confiança (resultado iminente = risco binário)
    earnings_days = features.get("earnings_days")
    if earnings_days is not None:
        try:
            ed = float(earnings_days)
            if math.isfinite(ed) and ed < 14:
                data_coverage *= 0.85
                logging.debug(
                    f"[score] earnings in {ed:.0f}d — coverage penalised to {data_coverage:.2f}"
                )
        except (TypeError, ValueError):
            pass

    # Red Flags + Quality Multiplier (Fase 1)
    quality_multiplier, red_flags, is_preprofit = _detect_red_flags(features)

    # Camada 2 — Quality Confidence (nova)
    fcf_yield_v  = _safe_float(features.get("fcf_yield"), 0.0)
    pe_v         = _safe_float(features.get("pe"), 0.0)
    roe_v        = _safe_float(features.get("roe"), 0.0)
    rev_growth_v = _safe_float(features.get("revenue_growth"), 0.0)

    quality_conf = 1.0
    if fcf_yield_v < 0:
        quality_conf -= 0.20
    if pe_v > 200 or pe_v <= 0:
        quality_conf -= 0.25
    if roe_v < 0 and rev_growth_v < 0.15:
        quality_conf -= 0.20
    if fcf_yield_v < 0 and pe_v > 100:
        quality_conf -= 0.15
    quality_conf = max(0.20, quality_conf)

    # Confiança final = cobertura × qualidade
    confidence = data_coverage * quality_conf
    if red_flags:
        logging.debug(
            f"[score] quality_multiplier={quality_multiplier:.2f} "
            f"quality_conf={quality_conf:.2f} "
            f"red_flags={red_flags} preprofit={is_preprofit}"
        )

    # skip_recommended actualizado
    qd_val = quality_dislocation(
        _safe_float(features.get("gross_margin"), 0.0),
        _safe_float(features.get("drawdown_from_high"), 0.0),
        fcf_yield_v,
    )
    skip = (
        confidence < 0.50
        or (fcf_yield_v < 0 and pe_v > 200 and rev_growth_v < 0.15)
        or (qd_val < 0.05 and (0.40 * _safe_float(features.get("quality_score_raw"), quality) + 0.20 * value + 0.20 * timing) * confidence * 100 < 45)
    )
    if skip:
        logging.debug(
            f"[score] skip_recommended=True — confidence={confidence:.2f} "
            f"preprofit={is_preprofit} qd={qd_val:.3f} (missing: {missing})"
        )

    # ML weight
    ml_weight = 1.0 if ml_prob is None else float(np.clip(ml_prob, 0.0, 1.0))

    try:
        divergence = _compute_divergence(features)
    except Exception as exc:
        logging.warning(f"[score] divergence hemisphere error: {exc}")
        divergence = 0.0
      
    # Score final — multiplica pelo quality_multiplier (penalty pós-zscore)
    base_score  = (0.40 * quality) + (0.20 * value) + (0.20 * timing) + (0.20 * divergence)
    raw_final   = base_score * ml_weight * confidence * 100.0
    final_score = float(np.clip(raw_final, 0.0, 100.0))

    # Score "puro fundamental" — sem o ml_weight, para o conflict resolver
    # poder cruzar fundamentais vs. ML sem dupla contagem.
    raw_fund_only = base_score * confidence * 100.0
    fund_only_score = float(np.clip(raw_fund_only, 0.0, 100.0))

    return {
        "final_score":         round(final_score, 2),
        "fund_only_score":     round(fund_only_score, 2),
        "quality_score":       round(quality,    4),
        "value_score":         round(value,      4),
        "timing_score":        round(timing,     4),
        "divergence_score":    round(divergence, 4),
        "confidence":          round(confidence, 4),
        "data_coverage":       round(data_coverage, 4),
        "quality_multiplier":  round(quality_multiplier, 4),
        "is_value_trap":       vt,
        "is_preprofit":        is_preprofit,
        "red_flags":           red_flags,
        "skip_recommended":    skip,
        "missing_fields":      missing,
    }


# ---------------------------------------------------------------------------
# 10. Bridge de compatibilidade — adaptador para o dicionário do market_client
# ---------------------------------------------------------------------------

def score_from_fundamentals(
    fundamentals: dict,
    ml_prob: float | None = None,
    earnings_days: int | None = None,
) -> dict:
    """
    Adapta o dicionário de get_fundamentals() ao motor calculate_score().

    Mapeamento:
      gross_margin    → fcf_margin proxy (quando fcf_margin não existe)
      volume          → componente volume spike no hemisfério Timing
      average_volume  → referência para normalizar o volume spike
      earnings_days   → penalização de confiança se < 14 dias
    """
    features = {
        "roic":               fundamentals.get("roic"),
        "roe":                fundamentals.get("roe"),  # Fase 1 — red flag detection
        "fcf_margin":         fundamentals.get("fcf_margin") or fundamentals.get("gross_margin"),
        "fcf_yield":          fundamentals.get("fcf_yield"),
        "revenue_growth":     fundamentals.get("revenue_growth"),
        "debt_equity":        fundamentals.get("debt_equity"),
        "pe":                 fundamentals.get("pe"),
        "rsi":                fundamentals.get("rsi"),
        "drawdown_from_high": fundamentals.get("drawdown_from_high"),
        "sector":             fundamentals.get("sector"),
        # Timing bonus — volume spike
        "volume":             fundamentals.get("volume"),
        "average_volume":     fundamentals.get("average_volume"),
        # Near-earnings uncertainty
        "earnings_days":      earnings_days,
    }
    return calculate_score(features, ml_prob=ml_prob)


# ---------------------------------------------------------------------------
# 11. Formata o breakdown para Telegram
# ---------------------------------------------------------------------------

def format_score_v2_breakdown(
    result: dict,
    conflict_state: object | None = None,
    conflict_msg: str | None = None,
) -> str:
    """
    Gera um bloco de texto legível para o Telegram a partir do resultado
    de calculate_score() / score_from_fundamentals().

    Se `conflict_state` (ConflictState) e `conflict_msg` forem passados,
    inclui um veredicto final cruzando ML × fundamentais (Fase 2).
    """
    fs   = result["final_score"]
    q    = result["quality_score"]
    v    = result["value_score"]
    t    = result["timing_score"]
    conf = result["confidence"]
    vt   = result["is_value_trap"]
    skip = result["skip_recommended"]
    miss = result["missing_fields"]
    red_flags     = result.get("red_flags") or []
    is_preprofit  = result.get("is_preprofit", False)

    badge = "🔥" if fs >= 80 else ("⭐" if fs >= 55 else "📊")
    lines = [
        f"{badge} *Score V2: {fs:.1f}/100*",
        f"  🏗️  Quality *{q*100:.0f}%*  \u00b7  💰 Value *{v*100:.0f}%*  \u00b7  ⏱️ Timing *{t*100:.0f}%*",
        f"  📊 Confiança: *{conf*100:.0f}%*" + (" — dados insuficientes ⚠️" if skip else ""),
    ]

    # Aviso quando Value é estruturalmente baixo (pre-profit + métricas distorcidas)
    if v < 0.20 and is_preprofit:
        lines.append("  ⚠️ _Valorização inaplicável (empresa pre-profit)._")

    if vt:
        lines.append("  🔴 *Value Trap detectada* — quality penalizada em 50%")

    if red_flags:
        lines.append(f"  🔴 *Red Flags:* {', '.join(red_flags)}")

    if is_preprofit:
        lines.append(
            "  ℹ️ _Empresa de crescimento pré-lucro. Score baixo estrutural "
            "(métricas de valor distorcidas)._"
        )

    # Remove volume_spike dos campos em falta (é bonus, não obrigatório)
    reportable_miss = [m for m in miss if m != "volume_spike"]
    if reportable_miss:
        lines.append(f"  _Em falta: {', '.join(reportable_miss)}_")

    # Veredicto final cruzando ML × fundamentais
    if conflict_state is not None and conflict_msg is not None:
        verdict_label = getattr(conflict_state, "value", str(conflict_state))
        lines.append("")
        lines.append(f"⚖️ *Veredicto:* {verdict_label}")
        lines.append(f"💡 _{conflict_msg}_")

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
#     ml_prob e earnings_days propagados ao motor.
# ---------------------------------------------------------------------------

def calculate_dip_score(
    fundamentals: dict,
    symbol: str,
    earnings_days: int | None = None,
    sector_change: float | None = None,
    stock_change_pct: float | None = None,
    ml_prob: float | None = None,
) -> tuple[float, str | None]:
    """
    Shim de compatibilidade. Chama o motor quantitativo internamente.

    earnings_days propagado → penalização de confiança se < 14 dias.
    ml_prob propagado       → ponderador ML no score final.

    Devolve (final_score: float, rsi_str: str | None).
    """
    result  = score_from_fundamentals(fundamentals, ml_prob=ml_prob, earnings_days=earnings_days)
    rsi_val = fundamentals.get("rsi")
    rsi_str = f"{float(rsi_val):.0f}" if rsi_val is not None else None
    return result["final_score"], rsi_str


def build_score_breakdown(
    fundamentals: dict,
    symbol: str,
    earnings_days: int | None = None,
    sector_change: float | None = None,
    stock_change_pct: float | None = None,
    ml_prob: float | None = None,
    ml_label: str | None = None,
) -> str:
    """
    Shim de compatibilidade. Devolve o bloco Telegram do motor quantitativo.

    earnings_days propagado → reflectido na confiança e na breakdown.
    ml_prob propagado       → reflectido no score final da breakdown.
    ml_label propagado      → quando presente, calcula e injecta o veredicto
                               cruzado (Fase 2) no fim do bloco.
    """
    result = score_from_fundamentals(fundamentals, ml_prob=ml_prob, earnings_days=earnings_days)
    state = None
    msg = None
    if ml_label is not None:
        try:
            from conflict_resolver import resolve_conflict
            # Usar fund_only_score para o resolver — o final_score já tem ML
            # baked-in via ml_prob multiplier, o que tornaria CONFLICT_FUND
            # inalcançável (ML bear afundaria sempre o final_score < 55).
            fund_score = result.get("fund_only_score", result["final_score"])
            state, msg = resolve_conflict(fund_score, ml_label)
        except Exception as exc:
            logging.warning(f"[score] conflict_resolver error: {exc}")
            state, msg = None, None
    return format_score_v2_breakdown(result, conflict_state=state, conflict_msg=msg)


# ---------------------------------------------------------------------------
# 15. Smoke test  (python score.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = {
        "roic":               0.22,
        "fcf_margin":         0.14,
        "fcf_yield":          0.06,
        "revenue_growth":     0.12,
        "debt_equity":        45.0,
        "pe":                 18.0,
        "rsi":                28.0,
        "drawdown_from_high": -32.0,
        "sector":             "Technology",
        "market_cap":         200_000_000_000,
        "gross_margin":       0.65,
        "dividend_yield":     0.008,
        "volume":             8_000_000,
        "average_volume":     4_000_000,  # 2× spike
    }

    print("=== calculate_score (com volume spike 2×) ===")
    res = calculate_score(sample, ml_prob=0.85)
    for k, v in res.items():
        print(f"  {k}: {v}")
    print()
    print(format_score_v2_breakdown(res))
    print()

    print("=== shim calculate_dip_score (com earnings_days=10 → penalty) ===")
    score, rsi_str = calculate_dip_score(sample, "AAPL", earnings_days=10, ml_prob=0.85)
    print(f"  score={score:.1f}  rsi_str={rsi_str}  (earnings em 10d → conf×0.85)")
    bc  = is_bluechip(sample)
    cat = classify_dip_category(sample, score, bc)
    print(f"  is_bluechip={bc}  category={cat}")
    print()

    print("=== shim calculate_dip_score (sem ml_prob — retro-compat) ===")
    score2, _ = calculate_dip_score(sample, "AAPL")
    print(f"  score={score2:.1f}  (ml_prob=None → usa 1.0, sem earnings penalty)")
    print()

    print("=== Value Trap ===")
    trap = dict(sample, revenue_growth=-0.05, fcf_margin=-0.04, fcf_yield=-0.02)
    res2 = calculate_score(trap, ml_prob=0.70)
    for k, v in res2.items():
        print(f"  {k}: {v}")
    print()

    print("=== Sem volume (volume_spike ausente — timing usa só RSI+drawdown) ===")
    no_vol = {k: v for k, v in sample.items() if k not in ("volume", "average_volume")}
    res4 = calculate_score(no_vol, ml_prob=0.85)
    print(f"  timing_score: {res4['timing_score']:.4f}")
    print(f"  confidence:   {res4['confidence']:.4f}  (volume não conta para conf)")
    print()

    print("=== Baixa confiança ===")
    sparse = {"pe": 22.0, "rsi": 45.0, "sector": "Healthcare"}
    res3 = calculate_score(sparse, ml_prob=0.60)
    for k, v in res3.items():
        print(f"  {k}: {v}")
