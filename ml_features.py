"""
ml_features.py — Unified Feature Store for DipRadar ML Pipeline.

Builds the complete feature vector for a single stock at a given moment.
Used identically during training (with labels) and inference in production (labels=None).

Architecture: 4-stage pipeline
  Stage 0 — Macro:    macro regime, VIX, SPY/sector drawdown, FRED recession prob
  Stage 1 — Quality:  gross margin, D/E, P/E vs sector fair,
                       analyst upside, ROIC, FCF yield — sourced from score.py hemispheres
  Stage 2 — Timing:   drop today, drawdown 52w, RSI, ATR ratio, volume spike, BB width
  Stage 3 — Engineered: non-linear interactions (rsi_oversold_strength, etc.)
  Stage 3b — Momentum: multi-window momentum (1m, 3m, 6m, 12m),
                        sector-relative (3m and 6m), beta_60d, vol_of_vol
  Stage 3c — Dislocation: quality_dislocation, peg_implicit, relative_drop,
                        month_of_year
  Stage 3d — Context: sector_alert_count_7d, days_since_52w_high
  Stage 3e — Short/Earnings: short_interest_ratio, earnings_surprise_avg, earnings_distance_days
  Stage 3f — Regime:  vix_percentile_1y, spy_rsi_14, yield_10y_change_5d

NaN contract:
  Every feature has an explicit fallback. No raw NaN reaches the model.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from macro_data import get_macro_context
from score import score_from_fundamentals, _safe_float

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: list[str] = [
    "vix",
    "drop_pct_today",
    "drawdown_52w",
    "rsi_14",
    "bb_width",
    "rsi_oversold_strength",
    "vix_regime",
    "drop_x_drawdown",
    "vol_x_drop",
    "return_1m",
    "return_3m_pre",
    "return_6m_pre",
    "sector_relative",
    "beta_60d",
    "vol_of_vol",
    "quality_dislocation",
    "month_of_year",
    "sector_alert_count_7d",
    "vix_percentile_1y",
    "spy_rsi_14",
    # PR #30 (Phase B): 4 raw-OHLCV features ortogonais aos existentes
    # (vol_of_vol, bb_width já cobrem volatilidade close-to-close).
    # Profiling IC sobre o parquet completo eliminou 3 candidatos iniciais
    # (realized_vol_20d ρ=0.81 com vol_of_vol → redundante; mom_5d_slope
    # ρ=0.0; gap_pct_5d ρ=0.0). Estas 4 capturam dimensões diferentes:
    "volume_zscore_20d",   # (today_vol − mean_20d) / std_20d  — flow institucional
    "close_in_range_20d",  # mean((C−L)/(H−L)) last 20 — fecho perto de high/low
    "up_days_pct_20d",     # % de dias com retorno positivo nos últimos 20
    "true_range_pct_20d",  # mean(TR_d / Close_d) últimos 20 — intraday vol
    # ── Novos sinais de qualidade e sentimento (Fase 5) ──────────────────────
    # Computados via yfinance + SEC EDGAR + Alpha Vantage/FMP (free tiers).
    # Cada um captura uma dimensão ortogonal às features técnicas existentes.
    "consecutive_red_days",  # dias consecutivos em queda antes do alerta — capitulação
    "ma_200d_ratio",         # price / 200-day MA — quão abaixo da média de longo prazo
    "insider_buy_recent",    # 1 se houve compra insider nos últimos 30d (SEC Form 4)
    "earnings_beat_rate",    # % de últimos 4 trimestres onde bateu EPS estimativas
    "analyst_rating",        # consenso analistas: 1=Strong Buy … 5=Strong Sell
    "short_interest_pct",    # short interest como % do float — squeeze potential
    # Fase 5b: sinais avançados (Form 4 montante + 8-K + short trend + earnings tone)
    "insider_buy_amount_score",  # magnitude normalizada das compras insider [0,1]
    "recent_8k_score",           # tipo de evento 8-K: -1=restatement, +1=M&A target
    "short_interest_trend",      # variação do short interest vs mês anterior
    "earnings_call_tone",        # sentimento da última earnings call [-1=pessimista, +1=confiante]
    # Sector-conditioned features (v4.2): RSI, VIX e drawdown ajustados ao tipo de sector.
    # Um VIX=35 significa coisas diferentes para growth vs commodity vs defensive.
    # Captura o desvio relativo ao regime normal do sector — dimensão ortogonal ao modelo global.
    "rsi_sector_adjusted",       # max(0, threshold_sector - rsi) — quão oversold vs norma do sector
    "vix_sector_signal",         # VIX × factor_sector / 40 — growth reage negativamente, defensive positivamente
    "drawdown_sector_normalized",# drawdown / typical_drawdown_sector — 0=normal, -3=capitulação extrema
]
# PR #28 (Phase A): drop 14 dead features after IC profiling on full parquet:
#   • 9 CONSTANTES (std=0, IC indefinida — _FALLBACK em todas as linhas):
#       gross_margin, de_ratio, pe_vs_fair, analyst_upside, quality_score,
#       atr_ratio, volume_spike, pe_attractive, peg_implicit
#     Razão: bootstrap original do parquet não tinha fontes point-in-time
#     destas features → todas as 36929 linhas carregam o mesmo _FALLBACK.
#     A inference live (build_features no position_monitor) continua a
#     computá-las correctamente — _FALLBACK abaixo é mantido para compat.
#   • 5 NOISE (|IC| < 0.01 com alpha_60d no parquet completo):
#       days_since_52w_high (+0.007), sector_drawdown_5d (-0.007),
#       macro_score (+0.005), spy_drawdown_5d (+0.003), relative_drop (+0.000)
#     Razão: estas features VARIAM por linha mas não correlacionam com
#     o target → desperdiçam capacidade do modelo.
# Net: 34 → 20 features. Smoke test mostrou IC single best 0.1744 → 0.1946 (+11%).
# Adicionalmente: month_of_year é fixed em load_base_dataset (era constante
# 5.0 no parquet por bug do bootstrap; agora é computada de alert_date.dt.month).
# Veja PR #25 para a remoção anterior (7 features ruidosas adicionadas em PR #23).

N_FEATURES = len(FEATURE_COLUMNS)

_SECTOR_FAIR_PE: dict[str, float] = {
    "Technology": 35.0,
    "Healthcare": 22.0,
    "Communication Services": 22.0,
    "Financial Services": 13.0,
    "Financials": 13.0,
    "Consumer Cyclical": 20.0,
    "Consumer Defensive": 22.0,
    "Industrials": 20.0,
    "Energy": 12.0,
    "Utilities": 18.0,
    "Real Estate": 40.0,
    "Basic Materials": 14.0,
    "Materials": 14.0,
}

_FALLBACK: dict[str, float] = {
    "macro_score": 2.0,
    "vix": 20.0,
    "spy_drawdown_5d": 0.0,
    "sector_drawdown_5d": 0.0,
    "gross_margin": 0.35,
    "de_ratio": 80.0,
    "pe_vs_fair": 1.0,
    "analyst_upside": 0.10,
    "quality_score": 0.50,
    "fcf_yield": 0.04,
    "drop_pct_today": -8.0,
    "drawdown_52w": -15.0,
    "rsi_14": 40.0,
    "atr_ratio": 0.02,
    "volume_spike": 1.0,
    "bb_width": 0.12,
    "rsi_oversold_strength": 0.0,
    "vix_regime": 1.0,
    "pe_attractive": 0.0,
    "drop_x_drawdown": 1.2,
    "vol_x_drop": 8.0,
    "return_1m": 0.0,
    "return_3m_pre": 0.0,
    "return_6m_pre": 0.0,
    "return_12m_pre": 0.0,
    "sector_relative": 0.0,
    "sector_relative_6m": 0.0,
    "beta_60d": 1.0,
    "vol_of_vol": 0.01,
    "quality_dislocation": 0.08,
    "peg_implicit": 2.0,
    "relative_drop": 0.0,
    "month_of_year": 6.0,
    "sector_alert_count_7d": 0.0,
    "days_since_52w_high": 180.0,
    "short_interest_ratio": 3.5,
    "earnings_surprise_avg": 0.0,
    "earnings_distance_days": 45.0,
    "vix_percentile_1y": 0.5,
    "spy_rsi_14": 50.0,
    "yield_10y_change_5d": 0.0,
    # PR #30 (Phase B): raw OHLCV features (sliced point-in-time <= alert_date)
    "volume_zscore_20d": 0.0,    # neutral volume vs baseline
    "close_in_range_20d": 0.5,   # 0.5 = neutral (closing mid-range)
    "up_days_pct_20d": 0.5,      # 50% — random walk baseline
    "true_range_pct_20d": 0.02,  # 2% intraday — typical equity
    # Fase 5: sinais de qualidade e sentimento
    "consecutive_red_days": 3.0,      # 3 dias consecutivos — neutro
    "ma_200d_ratio": 0.90,            # 10% abaixo da MA200 — dip típico
    "insider_buy_recent": 0.0,        # sem compra insider — caso mais comum
    "earnings_beat_rate": 0.50,       # 50% — neutro (random)
    "analyst_rating": 2.5,            # entre Buy e Hold — neutro
    "short_interest_pct": 0.05,       # 5% — short interest típico
    # Fase 5b: sinais avançados
    "insider_buy_amount_score": 0.0,  # sem compras — caso mais comum
    "recent_8k_score": 0.0,           # sem 8-K recente ou neutro
    "short_interest_trend": 0.0,      # estável
    "earnings_call_tone": 0.0,        # neutro — sem earnings call recente
    # Sector-conditioned features (v4.2)
    "rsi_sector_adjusted": 0.0,       # 0 = RSI na norma do sector (sem sinal de oversold)
    "vix_sector_signal": 0.0,         # 0 = VIX neutro ou sector tipo desconhecido
    "drawdown_sector_normalized": -1.0,  # -1 = drawdown leve (metade do típico)
}

def _tz_normalize(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t

def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return df
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df

# Mapeamento sector → tipo de dinâmica para sector-conditioned features.
# O modelo aprende que VIX=35 significa coisas diferentes consoante o sector.
_SECTOR_TYPE: dict[str, str] = {
    "Technology":             "growth",
    "Healthcare":             "growth",
    "Communication Services": "growth",
    "Financial Services":     "cyclical",
    "Financials":             "cyclical",
    "Industrials":            "cyclical",
    "Real Estate":            "cyclical",
    "Energy":                 "commodity",
    "Basic Materials":        "commodity",
    "Consumer Defensive":     "defensive",
    "Utilities":              "defensive",
    "Consumer Cyclical":      "cyclical",   # cíclico por definição — reage negativamente ao VIX
    "Unknown":                "cyclical",   # fallback neutro
}

# Limiar de "oversold" varia por sector:
# Growth: RSI<35 é forte sinal (tech recupera rápido de vendas de pânico)
# Commodity: RSI<30 é necessário (movem com ciclos mais longos)
# Defensive: RSI<40 suficiente (baixa volatilidade naturalmente)
_RSI_OVERSOLD_THRESHOLD: dict[str, float] = {
    "growth":    35.0,
    "cyclical":  32.0,
    "commodity": 28.0,
    "defensive": 40.0,
}


def add_derived_features(features: dict, alert_date: Optional[Any] = None) -> dict:
    """Computa features derivadas (interacções) + sector-conditioned features.

    Sector-conditioned features: o mesmo sinal (VIX, RSI, drawdown) tem pesos
    diferentes por tipo de sector. Em vez de modelos separados por sector
    (impossível com 46 linhas/stock em média), adicionamos versões condicionadas
    ao sector que o modelo global pode aprender a ponderar correctamente.

    Exemplo: RSI<30 em tech → sinal forte (pânico temporário).
             RSI<30 em energy → sinal fraco (pode reflectir queda do crude).
    """
    rsi = float(features.get("rsi_14", _FALLBACK["rsi_14"]))
    vix = float(features.get("vix", _FALLBACK["vix"]))
    pe_vf = float(features.get("pe_vs_fair", _FALLBACK["pe_vs_fair"]))
    drop = float(features.get("drop_pct_today", _FALLBACK["drop_pct_today"]))
    dd52 = float(features.get("drawdown_52w", _FALLBACK["drawdown_52w"]))
    volsp = float(features.get("volume_spike", _FALLBACK["volume_spike"]))

    # ── Features derivadas standard ───────────────────────────────────────────
    features["rsi_oversold_strength"] = round(max(0.0, 40.0 - rsi), 4)
    features["vix_regime"] = 0.0 if vix < 15.0 else (1.0 if vix < 25.0 else 2.0)
    features["pe_attractive"] = round(max(0.0, 1.0 - pe_vf), 4)
    features["drop_x_drawdown"] = round(drop * dd52 / 100.0, 4)
    features["vol_x_drop"] = round(volsp * abs(drop), 4)
    gm = float(features.get("gross_margin", _FALLBACK["gross_margin"]))
    fcf = float(features.get("fcf_yield", _FALLBACK["fcf_yield"]))
    revenue_growth = float(features.get("revenue_growth", 0.05))
    features["quality_dislocation"] = round(gm * abs(dd52) / 100.0, 4) if fcf >= 0 else 0.0
    if revenue_growth > 0 and pe_vf > 0:
        features["peg_implicit"] = round(min(pe_vf / (revenue_growth * 100.0), 5.0), 4)
    else:
        features["peg_implicit"] = 3.0
    sec_dd = float(features.get("sector_drawdown_5d", _FALLBACK["sector_drawdown_5d"]))
    features["relative_drop"] = round(drop - sec_dd, 4)
    try:
        features["month_of_year"] = float(pd.Timestamp(alert_date).month) if alert_date is not None else float(datetime.now().month)
    except Exception:
        features["month_of_year"] = float(datetime.now().month)

    # ── Sector-conditioned features ───────────────────────────────────────────
    # O modelo global aprende pesos diferentes para os mesmos sinais consoante
    # o tipo de sector. Não são modelos separados (impossível com ~46 linhas/stock)
    # mas o modelo aprende que VIX=35 em growth ≠ VIX=35 em commodity.
    sector   = str(features.get("sector", "Unknown") or "Unknown")
    sec_type = _SECTOR_TYPE.get(sector, "cyclical")
    rsi_thr  = _RSI_OVERSOLD_THRESHOLD.get(sec_type, 32.0)

    # RSI ajustado ao sector: quão oversold está, relativo ao limiar sectorial
    # growth: RSI<35 é "muito oversold"; commodity: RSI<28 é "muito oversold"
    features["rsi_sector_adjusted"] = round(max(0.0, rsi_thr - rsi), 4)

    # VIX × sector_type: alta volatilidade é positiva para defensivos (flight to safety)
    # mas negativa para growth (risk-off → sell tech).
    _vix_factor = {"growth": -1.0, "cyclical": -0.5, "commodity": 0.0, "defensive": +0.5}
    features["vix_sector_signal"] = round(vix * _vix_factor.get(sec_type, -0.5) / 40.0, 4)

    # Drawdown normalizado pelo sector: em energy, drawdown de -30% é comum;
    # em tech, -30% é extremo. Normaliza pelo drawdown típico do sector.
    _dd_typical = {"growth": -25.0, "cyclical": -30.0, "commodity": -40.0, "defensive": -20.0}
    dd_norm = dd52 / abs(_dd_typical.get(sec_type, -30.0))
    features["drawdown_sector_normalized"] = round(float(max(-3.0, min(0.0, dd_norm))), 4)

    return features


# ────────────────────────────────────────────────────────────────────────────────
# build_features — thin orchestrator usado pelo position_monitor para
# re-scoring diário das posições activas. Replica o contrato antigo:
#   build_features(ticker, fundamentals) → dict com todas as FEATURE_COLUMNS.
# Sem price_history/sector → as features técnicas e momentum caem para
# _FALLBACK determinístico (aceitável para vigilância diária).
# ────────────────────────────────────────────────────────────────────────────────
def build_features(
    ticker: str,
    fundamentals: dict,
    *,
    sector: str = "Unknown",
    macro_context: Optional[dict] = None,
    alert_date: Optional[Any] = None,
    price_history: Optional[pd.DataFrame] = None,
) -> dict:
    """Constrói um dict de features para inference (predict_dip / ml_score).

    Designed for **position_monitor** path — re-scoring diário das posições
    activas sem price_history nem sector ETF disponíveis. Para training
    histórico ou snapshot do universo, usar ``ml_training.data.build_dataset``
    ou ``universe_snapshot._build_snapshot_row`` (que constroem o dict
    manualmente e chamam apenas ``add_derived_features``).

    Parameters
    ----------
    ticker         : str   identificador (apenas para logs).
    fundamentals   : dict  output de ``_fetch_fundamentals_snapshot`` ou
                           equivalente. Chaves opcionais: ``pe``, ``debt_equity``,
                           ``revenue_growth``, ``gross_margin``, ``fcf_yield``,
                           ``analyst_upside``, ``price``, ``market_cap``.
    sector         : str   GICS sector (defaults "Unknown" → fair PE 22.0).
    macro_context  : dict  ``get_macro_context`` pre-calculado (evita API call
                           extra). ``None`` = chama ``get_macro_context``.
    alert_date     : Any   timestamp do alerta (usado para ``month_of_year``).
                           ``None`` = hoje.

    Returns
    -------
    dict com todas as FEATURE_COLUMNS preenchidas (valores fundamentais
    vindos do dict, técnicos/momentum/regime em fallback).
    """
    fund = fundamentals or {}

    # Stage 0 — Macro
    if macro_context is None:
        try:
            macro_context = get_macro_context(sector=sector)
        except Exception as e:
            logger.debug(f"build_features[{ticker}]: get_macro_context falhou: {e}")
            macro_context = {}
    macro_context = macro_context or {}

    features: dict = {col: _FALLBACK[col] for col in FEATURE_COLUMNS}

    # Macro overrides
    for k in ("macro_score", "vix", "spy_drawdown_5d", "sector_drawdown_5d"):
        if k in macro_context and macro_context[k] is not None:
            try:
                features[k] = float(macro_context[k])
            except (TypeError, ValueError):
                pass

    # Stage 1 — Quality / Value (do dict de fundamentais)
    def _set(col: str, val: Any) -> None:
        if val is None:
            return
        try:
            f = float(val)
        except (TypeError, ValueError):
            return
        if math.isfinite(f):
            features[col] = f

    _set("gross_margin",   fund.get("gross_margin"))
    _set("de_ratio",       fund.get("debt_equity") or fund.get("de_ratio"))
    _set("analyst_upside", fund.get("analyst_upside") or fund.get("upside"))
    _set("fcf_yield",      fund.get("fcf_yield"))

    # pe_vs_fair = pe / fair_pe_sector (ratio > 1 = overvalued)
    pe_raw = _safe_float(fund.get("pe"))
    fair_pe = _SECTOR_FAIR_PE.get(sector, 22.0)
    if not math.isnan(pe_raw) and pe_raw > 0 and fair_pe > 0:
        features["pe_vs_fair"] = round(pe_raw / fair_pe, 4)

    # Quality score via score.py (hemisphere A normalizado [0,1])
    try:
        sr = score_from_fundamentals(fund)
        if isinstance(sr, dict) and "quality_score" in sr:
            features["quality_score"] = float(sr["quality_score"])
        elif isinstance(sr, (int, float)):
            features["quality_score"] = float(sr)
    except Exception as e:
        logger.debug(f"build_features[{ticker}]: score_from_fundamentals falhou: {e}")

    # Stage 2 — Timing (do dict; sem price_history → fallback)
    _set("drop_pct_today", fund.get("drop_pct_today") or fund.get("change_pct"))
    _set("drawdown_52w",   fund.get("drawdown_52w"))
    _set("rsi_14",         fund.get("rsi_14") or fund.get("rsi"))
    _set("atr_ratio",      fund.get("atr_ratio"))
    _set("volume_spike",   fund.get("volume_spike"))

    # Stage 3 — Derived (interactions). Sobrescreve os 7 derivados a partir
    # do dict actual; o resto fica em fallback.
    add_derived_features(features, alert_date=alert_date)

    # Stage 4 — Raw OHLCV (PR #30). Opt-in: só corre se o caller passar
    # price_history; caso contrário cai nos fallbacks neutros.
    if price_history is not None:
        add_raw_ohlcv_features(features, price_history)

    # Stage 5 — Sinais de qualidade e sentimento (Fase 5).
    # NaN onde não disponível — o modelo trata NaN como "informação em falta",
    # não como valor neutro inventado.
    try:
        from fundamental_signals import compute_fundamental_signals
        signals = compute_fundamental_signals(
            ticker,
            price_history=price_history,
            alert_date=alert_date,
        )
        for key, val in signals.items():
            if key in FEATURE_COLUMNS:
                features[key] = float(val) if math.isfinite(float(val)) else float("nan")
    except Exception as e:
        logger.error(f"build_features[{ticker}]: fundamental_signals falhou: {e}")
        # Não preencher com fallback — o modelo vê NaN e sabe que faltam dados

    return features


def add_context_features(features: dict, price_history: Optional[pd.DataFrame] = None, sector_alert_count_7d: Optional[float] = None) -> dict:
    if sector_alert_count_7d is not None:
        v = float(sector_alert_count_7d)
        features["sector_alert_count_7d"] = v if math.isfinite(v) else _FALLBACK["sector_alert_count_7d"]
    else:
        features["sector_alert_count_7d"] = _FALLBACK["sector_alert_count_7d"]
    if price_history is not None and "Close" in price_history.columns:
        try:
            closes = price_history["Close"].dropna()
            lookback = closes.tail(252)
            if len(lookback) >= 5:
                idx_max = lookback.values.argmax()
                days_ago = len(lookback) - 1 - idx_max
                cal_days = round(days_ago * 1.4)
                features["days_since_52w_high"] = float(max(0, cal_days))
            else:
                features["days_since_52w_high"] = _FALLBACK["days_since_52w_high"]
        except Exception:
            features["days_since_52w_high"] = _FALLBACK["days_since_52w_high"]
    else:
        features["days_since_52w_high"] = _FALLBACK["days_since_52w_high"]
    return features

def add_short_earnings_features(features: dict, ticker_info: Optional[dict] = None, alert_date: Optional[Any] = None) -> dict:
    info = ticker_info or {}
    sr = _safe_float(info.get("shortRatio"))
    features["short_interest_ratio"] = float(min(sr, 30.0)) if math.isfinite(sr) and sr >= 0 else _FALLBACK["short_interest_ratio"]
    try:
        hist = info.get("earningsHistory", {})
        if isinstance(hist, dict):
            hist = hist.get("history", [])
        surprises = []
        next_earnings_distance = _FALLBACK["earnings_distance_days"]
        now_ts = _tz_normalize(alert_date) if alert_date is not None else None
        for entry in (hist or [])[:8]:
            if isinstance(entry, dict):
                sp = _safe_float(entry.get("surprisePercent"))
                if math.isfinite(sp):
                    surprises.append(sp * 100.0)
                if now_ts is not None:
                    dt_raw = entry.get("quarter") or entry.get("date") or entry.get("earningsDate")
                    if dt_raw is not None:
                        try:
                            dt = _tz_normalize(dt_raw)
                            diff = (dt - now_ts).days
                            if diff >= 0:
                                next_earnings_distance = min(next_earnings_distance, float(diff))
                        except Exception:
                            pass
        features["earnings_surprise_avg"] = float(np.clip(np.mean(surprises[:2]), -50.0, 50.0)) if surprises else _FALLBACK["earnings_surprise_avg"]
        features["earnings_distance_days"] = float(np.clip(next_earnings_distance, 0.0, 180.0))
    except Exception:
        features["earnings_surprise_avg"] = _FALLBACK["earnings_surprise_avg"]
        features["earnings_distance_days"] = _FALLBACK["earnings_distance_days"]
    return features

def add_raw_ohlcv_features(features: dict, price_history: Optional[pd.DataFrame]) -> dict:
    """Compute 4 raw-OHLCV features from a point-in-time price slice.

    All features assume ``price_history`` is already sliced to dates
    ``<= alert_date`` (no look-ahead) and indexed by date in ascending order.
    Falls back to neutral defaults if the slice is too short.

    Features chosen for orthogonality to the existing top features
    (vol_of_vol, bb_width, drawdown_52w, return_1m/3m/6m):

      • ``volume_zscore_20d`` — z-score of today's volume against the prior
        20 sessions. Institutional flow / capitulation. Bounded to ±10.
      • ``close_in_range_20d`` — mean of ``(Close − Low) / (High − Low)``
        across the last 20 sessions. Captures *where* in the daily range
        the stock typically closes (>0.5 → strong closes; <0.5 → weak).
        Orthogonal to close-to-close vol.
      • ``up_days_pct_20d`` — fraction of the last 20 sessions with strictly
        positive daily return. Captures trend *persistence* (regular up-days
        vs occasional big rallies). Bounded [0, 1].
      • ``true_range_pct_20d`` — mean ATR-like intraday range as a percent
        of close. Captures intraday volatility (different from close-to-close).
    """
    out_keys = ("volume_zscore_20d", "close_in_range_20d", "up_days_pct_20d", "true_range_pct_20d")
    for k in out_keys:
        features.setdefault(k, _FALLBACK[k])

    if price_history is None or price_history.empty:
        return features
    hist = _normalize_index(price_history)
    if "Close" not in hist.columns or len(hist) < 6:
        return features

    close = hist["Close"].astype(float)

    # volume_zscore_20d
    try:
        if "Volume" in hist.columns:
            vol = hist["Volume"].astype(float).tail(21)
            if len(vol) == 21:
                today_v = float(vol.iloc[-1])
                window = vol.iloc[:-1]
                mu = float(window.mean())
                sd = float(window.std())
                if sd > 0 and math.isfinite(today_v) and math.isfinite(mu):
                    z = (today_v - mu) / sd
                    if math.isfinite(z):
                        features["volume_zscore_20d"] = float(np.clip(z, -10.0, 10.0))
    except Exception:
        pass

    # close_in_range_20d: requer High e Low
    try:
        if "High" in hist.columns and "Low" in hist.columns:
            tail = hist.tail(20)
            high = tail["High"].astype(float)
            low = tail["Low"].astype(float)
            close_t = tail["Close"].astype(float)
            rng = high - low
            mask = (rng > 0) & np.isfinite(rng) & np.isfinite(close_t)
            if mask.sum() >= 5:
                cir = (close_t[mask] - low[mask]) / rng[mask]
                v = float(cir.mean())
                if math.isfinite(v):
                    features["close_in_range_20d"] = float(np.clip(v, 0.0, 1.0))
    except Exception:
        pass

    # up_days_pct_20d
    try:
        rets = close.pct_change().tail(20).dropna()
        if len(rets) >= 5:
            features["up_days_pct_20d"] = float((rets > 0).mean())
    except Exception:
        pass

    # true_range_pct_20d
    try:
        if "High" in hist.columns and "Low" in hist.columns:
            tail = hist.tail(21)  # need prev close → 21 rows for 20 TR values
            high = tail["High"].astype(float)
            low = tail["Low"].astype(float)
            close_t = tail["Close"].astype(float)
            prev_close = close_t.shift(1)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).iloc[1:]  # drop first NaN
            close_for_div = close_t.iloc[1:]
            mask = (close_for_div > 0) & np.isfinite(tr) & np.isfinite(close_for_div)
            if mask.sum() >= 5:
                tr_pct = (tr[mask] / close_for_div[mask])
                v = float(tr_pct.mean())
                if math.isfinite(v):
                    # Clip a 100% intraday range (extremo absoluto).
                    features["true_range_pct_20d"] = float(np.clip(v, 0.0, 1.0))
    except Exception:
        pass

    return features


def add_regime_features(features: dict, spy_history: Optional[pd.DataFrame], tnx_history: Optional[pd.DataFrame], alert_date: Any, vix_history: Optional[pd.DataFrame] = None) -> dict:
    alert_ts = _tz_normalize(alert_date)
    vix_hist = _normalize_index(vix_history)
    spy_hist = _normalize_index(spy_history)
    tnx_hist = _normalize_index(tnx_history)
    try:
        vix_val = float(features.get("vix", _FALLBACK["vix"]))
        pct = _FALLBACK["vix_percentile_1y"]
        if vix_hist is not None and "Close" in vix_hist.columns:
            vix_slice = vix_hist[vix_hist.index <= alert_ts]
            window = vix_slice["Close"].dropna().tail(252)
            if len(window) >= 20:
                arr = window.values
                pct = float(np.clip(np.sum(arr <= vix_val) / len(arr), 0.0, 1.0))
        elif spy_hist is not None and "Close" in spy_hist.columns:
            rets = spy_hist[spy_hist.index <= alert_ts]["Close"].pct_change().dropna().tail(252)
            if len(rets) >= 20:
                rv_window = rets.rolling(5).std().dropna()
                if len(rv_window) >= 20:
                    cur_rv = float(rv_window.iloc[-1])
                    pct = float(np.clip(np.sum(rv_window.values <= cur_rv) / len(rv_window), 0.0, 1.0))
        features["vix_percentile_1y"] = round(pct, 4)
    except Exception:
        features["vix_percentile_1y"] = _FALLBACK["vix_percentile_1y"]
    try:
        rsi_val = _FALLBACK["spy_rsi_14"]
        if spy_hist is not None and "Close" in spy_hist.columns:
            closes = spy_hist[spy_hist.index <= alert_ts]["Close"].dropna()
            if len(closes) >= 16:
                delta = closes.diff().dropna()
                gain = delta.clip(lower=0).rolling(14).mean()
                loss = (-delta.clip(upper=0)).rolling(14).mean()
                rs = gain / loss.replace(0, np.nan)
                rsi_s = (100 - 100 / (1 + rs)).iloc[-1]
                if pd.notna(rsi_s):
                    rsi_val = float(np.clip(rsi_s, 0.0, 100.0))
        features["spy_rsi_14"] = round(rsi_val, 2)
    except Exception:
        features["spy_rsi_14"] = _FALLBACK["spy_rsi_14"]
    try:
        chg = _FALLBACK["yield_10y_change_5d"]
        if tnx_hist is not None and "Close" in tnx_hist.columns:
            tnx_slice = tnx_hist[tnx_hist.index <= alert_ts]["Close"].dropna()
            if len(tnx_slice) >= 6:
                chg = float(tnx_slice.iloc[-1] - tnx_slice.iloc[-6])
                if not math.isfinite(chg) or abs(chg) > 5.0:
                    chg = _FALLBACK["yield_10y_change_5d"]
        features["yield_10y_change_5d"] = round(chg, 4)
    except Exception:
        features["yield_10y_change_5d"] = _FALLBACK["yield_10y_change_5d"]
    return features

def _pct_return(prices: np.ndarray, lookback: int) -> float:
    if len(prices) < lookback + 1:
        return 0.0
    p_end = float(prices[-1])
    p_start = float(prices[-lookback - 1])
    if p_start <= 0:
        return 0.0
    return round((p_end / p_start - 1.0) * 100.0, 4)

def _vol_of_vol(prices: np.ndarray, rv_window: int = 5, vov_window: int = 60) -> float:
    min_needed = rv_window + vov_window
    if len(prices) < min_needed:
        return _FALLBACK["vol_of_vol"]
    try:
        returns = np.diff(prices[-min_needed:]) / prices[-min_needed:-1]
        rvs = np.array([float(np.std(returns[i:i + rv_window], ddof=1)) for i in range(len(returns) - rv_window + 1)])
        if len(rvs) < 10:
            return _FALLBACK["vol_of_vol"]
        vov = float(np.std(rvs[-vov_window:], ddof=1))
        return round(float(np.clip(vov, 0.0, 0.1)), 6)
    except Exception:
        return _FALLBACK["vol_of_vol"]

def _bb_width(prices: np.ndarray, window: int = 20, n_std: float = 2.0) -> float:
    if len(prices) < window:
        return _FALLBACK["bb_width"]
    try:
        s = pd.Series(prices[-window:])
        ma = float(s.mean())
        sd = float(s.std(ddof=1))
        if ma <= 0 or not math.isfinite(ma) or not math.isfinite(sd):
            return _FALLBACK["bb_width"]
        upper = ma + n_std * sd
        lower = ma - n_std * sd
        width = (upper - lower) / ma
        return round(float(np.clip(width, 0.0, 2.0)), 6)
    except Exception:
        return _FALLBACK["bb_width"]

def add_momentum_features(features: dict, price_history: Optional[pd.DataFrame], sector_history: Optional[pd.DataFrame] = None, spy_history: Optional[pd.DataFrame] = None) -> dict:
    keys = ["return_1m", "return_3m_pre", "return_6m_pre", "return_12m_pre", "sector_relative", "sector_relative_6m", "beta_60d", "vol_of_vol", "bb_width"]
    if price_history is None or price_history.empty:
        for k in keys:
            features.setdefault(k, _FALLBACK[k])
        return features
    closes = price_history["Close"].dropna().values
    features["return_1m"] = _pct_return(closes, 21)
    features["return_3m_pre"] = _pct_return(closes, 63)
    features["return_6m_pre"] = _pct_return(closes, 126)
    features["return_12m_pre"] = _pct_return(closes, 252)
    features["bb_width"] = _bb_width(closes)
    if sector_history is not None and not sector_history.empty:
        sec_closes = sector_history["Close"].dropna().values
        sec_ret_3m = _pct_return(sec_closes, 63)
        sec_ret_6m = _pct_return(sec_closes, 126)
        features["sector_relative"] = round(features["return_3m_pre"] - sec_ret_3m, 4)
        features["sector_relative_6m"] = round(features["return_6m_pre"] - sec_ret_6m, 4)
    else:
        features["sector_relative"] = _FALLBACK["sector_relative"]
        features["sector_relative_6m"] = _FALLBACK["sector_relative_6m"]
    features["beta_60d"] = _FALLBACK["beta_60d"]
    if spy_history is not None and not spy_history.empty:
        try:
            spy_closes = spy_history["Close"].dropna().values
            n = 60
            if len(closes) >= n + 1 and len(spy_closes) >= n + 1:
                stock_rets = np.diff(closes[-n - 1:]) / closes[-n - 1:-1]
                spy_rets = np.diff(spy_closes[-n - 1:]) / spy_closes[-n - 1:-1]
                min_len = min(len(stock_rets), len(spy_rets))
                stock_rets = stock_rets[-min_len:]
                spy_rets = spy_rets[-min_len:]
                cov = np.cov(stock_rets, spy_rets)
                var_spy = float(cov[1, 1])
                if var_spy > 1e-10:
                    features["beta_60d"] = round(float(cov[0, 1]) / var_spy, 4)
        except Exception:
            pass
    features["vol_of_vol"] = _vol_of_vol(closes)
    return features
