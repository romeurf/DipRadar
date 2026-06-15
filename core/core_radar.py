"""
core_radar.py — CoreRadar: scanner de oportunidades em compounders de qualidade.

Filosofia: monitoriza um universo curado de empresas de alta qualidade e alerta
quando estão em territory de dip (queda ≥5% em 20d, tendência longa intacta).
Diferenças face ao DipRadar:
  - Universo pequeno e curado pelo utilizador (MSFT, NVO, GOOGL, etc.)
  - Dip com tendência secular intacta (MA200 ainda positiva)
  - Alocação maior (maior convicção)
  - Paper trade registado como CORE (sem auto-close)
  - Score adicional por yield: quanto mais yield atrativo no momento do dip, melhor

Universo: core_universe.json  {tickers: [{ticker, annual_yield_pct, note}]}
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger("core_radar")

# 2026-06-14: was `Path(__file__).parent` which resolved to `core/`, not
# the repo root — so _UNIVERSE_FILE looked for `core/core_universe.json`
# instead of `core_universe.json`. _load_universe() returned [] silently,
# scan_core_universe iterated zero tickers, /core_picks always said
# "Nenhum ticker", and /cash never proposed any CORE buy. The single dot
# was hiding the entire universe from production for who knows how long.
_REPO_DIR      = Path(__file__).parent.parent
_UNIVERSE_FILE = _REPO_DIR / "core_universe.json"
_DATA_DIR      = Path("/data") if Path("/data").exists() else _REPO_DIR
_MODEL_PATH    = _DATA_DIR / "core_us_model.pkl"

# Import CoreModelBundle at module load time so its _register_in_main() runs
# BEFORE any joblib.load — otherwise the unpickler raises AttributeError on
# __main__.CoreModelBundle. Mirrors the DipModels / ml_predictor pattern.
from core.core_bundle import CoreModelBundle  # noqa: F401  pickle-compat

# ── ML model (loaded lazily on first call to is_model_ready()) ────────────────
_core_model_bundle = None
_core_model_checked = False


# Primary quality gate: Spearman IC ≥ 0.04. The Core trainer now reports IC
# alongside R²; IC tolerates alpha_1y's heavy tails much better than R² does
# and is the industry-standard quant metric. We keep _MIN_CV_R2_FOR_USE as a
# legacy fallback for bundles trained before cv_ic existed (those default to
# cv_ic=0.0 via the dataclass default, which fails the IC gate just like a
# missing R² did the old gate).
_MIN_CV_IC_FOR_USE = 0.04
_MIN_CV_R2_FOR_USE = 0.05  # kept for compat with older bundles (cv_ic absent)


def _load_core_model():
    global _core_model_bundle, _core_model_checked
    if _core_model_checked:
        return _core_model_bundle
    _core_model_checked = True
    if not _MODEL_PATH.exists():
        log.warning(f"[core_radar] ML model not found at {_MODEL_PATH}. Run /admin_train_core_model.")
        return None
    try:
        import joblib
        _core_model_bundle = joblib.load(_MODEL_PATH)
        cv_ic = float(getattr(_core_model_bundle, "cv_ic", 0.0))
        cv_r2 = float(getattr(_core_model_bundle, "cv_r2", 0.0))
        champion = str(getattr(_core_model_bundle, "champion_name", "GBR"))
        log.info(
            f"[core_radar] ML model loaded: champion={champion} "
            f"n_train={_core_model_bundle.n_train} "
            f"train_date={_core_model_bundle.train_date} "
            f"CV_IC={cv_ic:+.4f} CV_RMSE={_core_model_bundle.cv_rmse:.4f} "
            f"CV_R²={cv_r2:+.3f}"
        )
        if cv_ic < _MIN_CV_IC_FOR_USE:
            log.warning(
                f"[core_radar] ML model loaded but CV_IC={cv_ic:+.4f} < "
                f"{_MIN_CV_IC_FOR_USE} floor — predictions barely better than random. "
                f"Falling back to rule-based scoring. Either retrain after fresh data "
                f"arrives or accept that Core stays rule-based for this generation."
            )
        return _core_model_bundle
    except Exception as e:
        log.error(f"[core_radar] Failed to load ML model: {e}")
        return None


def is_model_ready() -> bool:
    """Return True if the core_us ML model passes the CV IC floor.

    Older bundles without cv_ic default to 0.0, which correctly fails the IC
    gate — equivalent to the old "below R² floor" behaviour.
    """
    bundle = _load_core_model()
    if bundle is None:
        return False
    cv_ic = float(getattr(bundle, "cv_ic", 0.0))
    return cv_ic >= _MIN_CV_IC_FOR_USE


def _is_investment_phase(
    rev_growth: float | None,
    gross_margin: float | None,
    roe: float | None,
    fcf_yield: float | None,
) -> bool:
    """Classify a negative-FCF stock as 'reinvesting compounder' vs 'deterioration'.

    Returns True when the financial pattern matches a healthy business that
    happens to be in a heavy investment cycle (CapEx > OperatingCF for now):

      - fcf_yield < 0                  (i.e., this IS an FCF-negative case)
      - revenue_growth > 10%           (expanding fast — investment is funding growth)
      - gross_margin > 40%             (real pricing power — not commoditized)
      - roe > 15%                      (reinvested capital is productive)

    All 4 conditions must be true. The strictness reflects that we're
    OVERRIDING the model's penalty, so we want a high-precision detector
    rather than a high-recall one. False positives (treating deterioration
    as investment) are worse than false negatives.

    Historical examples that match: AMZN 1997-2015, NFLX during content
    buildout, hyperscalers (MSFT/GOOG/META/AMZN) building AI data centers
    in 2024-2025, NVO 2025-2026 GLP-1 manufacturing capacity expansion.
    """
    if fcf_yield is None or fcf_yield >= 0:
        return False
    if rev_growth is None or rev_growth < 0.10:
        return False
    if gross_margin is None or gross_margin < 0.40:
        return False
    if roe is None or roe < 0.15:
        return False
    return True


def _ml_score(signals: dict, annual_yield_pct: float) -> float | None:
    """Score a candidate using the ML model. Returns predicted alpha_1y mapped
    to 0-100 via tanh compression, or None if the model is not ready or has
    poor CV quality.

    Conversion: score = 50 + tanh(alpha_1y × 3) × 50  (asymptotic, never saturates)
      alpha_1y =  0.00 → 50.0
      alpha_1y = +0.10 → 64.6
      alpha_1y = +0.20 → 76.9
      alpha_1y = +0.50 → 95.2
      alpha_1y = +1.00 → 99.75 (never exactly 100)
      alpha_1y = -0.20 → 23.1

    Why tanh, not linear: the previous formula `clamp(50 + alpha * 250, 0, 100)`
    saturated at alpha ≥ 0.20, tying NVO/GIS/CMG all at 100 even when their
    underlying alpha predictions were materially different. Tanh keeps the
    distinction across the high end.

    Quality gate: if the bundle's CV IC is below _MIN_CV_IC_FOR_USE, predictions
    aren't predictive enough → return None so the caller falls back to rule-based.

    Global ML disable: set env var DIPRADAR_USE_ML_SCORE=0 (default) to skip
    ML entirely and force rule-based scoring across all radars. This is the
    2026-06-15 user decision while ML signal is weak (DIP rho_alpha=0.063,
    CORE bucket calibration has U-shape pattern not strictly monotonic).
    """
    import os as _os
    if _os.environ.get("DIPRADAR_USE_ML_SCORE", "0") != "1":
        return None
    bundle = _load_core_model()
    if bundle is None:
        return None
    cv_ic = float(getattr(bundle, "cv_ic", 0.0))
    if cv_ic < _MIN_CV_IC_FOR_USE:
        return None
    try:
        import numpy as np
        medians = bundle.impute_medians

        # Option B — Investment-phase FCF override:
        # If the stock pattern matches a reinvesting compounder, replace
        # fcf_yield_fund with the dataset median before feeding to the
        # model. The model was trained on a mix that includes both
        # deteriorating businesses (FCF negative = bad signal) and rare
        # investment-phase compounders (FCF negative = neutral/good).
        # Since it can't distinguish, we strip the signal in cases where
        # we have orthogonal evidence (rev growth + margin + ROE) that
        # the negative FCF is investment, not deterioration.
        #
        # Diagnostics: signals["fcf_override_applied"] tracks when this
        # fires so the user can audit via /analisar or log dumps.
        invest_phase = bool(signals.get("is_investment_phase", False))
        signals["fcf_override_applied"] = False

        row = []
        for col in bundle.feature_cols:
            if col == "annual_yield_pct":
                val = annual_yield_pct
            elif invest_phase and col == "fcf_yield_fund":
                val = float(medians.get("fcf_yield_fund", 0.0))
                signals["fcf_override_applied"] = True
                log.info(
                    f"[core_radar] FCF override applied "
                    f"(investment-phase compounder)"
                )
            else:
                raw = signals.get(col)
                val = float(raw) if raw is not None and not math.isnan(float(raw)) else medians.get(col, 0.0)
            row.append(val)
        X = np.array(row).reshape(1, -1)
        X_scaled = bundle.scaler.transform(X)
        alpha_pred = float(bundle.model.predict(X_scaled)[0])
        score = 50.0 + math.tanh(alpha_pred * 3.0) * 50.0
        return round(score, 1)
    except Exception as e:
        log.debug(f"[core_radar] ML score failed: {e}")
        return None

# ── Thresholds de entrada (ajustados para CORE — qualidade alta, comprar no dip) ─
_ENTRY_RETURN_20D    = -0.05   # queda ≥5% em 20d
_MAX_DD_FROM_52W     = -0.55   # descarte apenas em colapso extremo (>55%) — evita filtrar NVO
_MA200_MIN_RATIO     = 0.65    # descarte apenas se >35% abaixo da MA200d (crise severa)
_RSI_MAX             = 55.0    # preferência por zonas de oversold
_YIELD_BONUS_PER_PCT = 3.0     # pontos de score extra por 1% de yield anual atrativo

# ── Thresholds de qualidade fundamental ──────────────────────────────────────────
_FCF_YIELD_STRONG    = 0.04    # FCF yield >4% = geração de cash a preço atrativo
_FCF_YIELD_OK        = 0.02    # FCF yield >2% = positivo mas não premium
_REV_GROWTH_STRONG   = 0.10    # revenue >10% y/y = negócio em expansão clara
_REV_GROWTH_OK       = 0.05    # revenue >5% y/y = crescimento moderado
_REV_DECLINE_WARN    = -0.05   # revenue < -5% y/y = negócio em contracção
_GROSS_MARGIN_MOAT   = 0.50    # >50% gross margin = pricing power forte (MSFT, GOOGL)
_GROSS_MARGIN_OK     = 0.30    # >30% gross margin = razoável
_ROE_COMPOUNDER      = 0.20    # ROE >20% = máquina de compounding (MSFT, V, MA)
_ROE_OK              = 0.10    # ROE >10% = aceitável

# ── Alocação CORE (mais generosa que tática por ser alta convicção) ─────────────
_CORE_ALLOC_FACTOR = 1.5     # 1.5× a alocação base de um DIP normal


# ─────────────────────────────────────────────────────────────────────────────
# Universo CORE
# ─────────────────────────────────────────────────────────────────────────────

def _load_universe() -> list[dict]:
    if not _UNIVERSE_FILE.exists():
        return []
    try:
        return json.loads(_UNIVERSE_FILE.read_text(encoding="utf-8")).get("tickers", [])
    except Exception:
        return []


def _save_universe(entries: list[dict]) -> None:
    tmp = _UNIVERSE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"tickers": entries}, indent=2), encoding="utf-8")
    tmp.replace(_UNIVERSE_FILE)


def add_to_universe(ticker: str, annual_yield_pct: float = 0.0, note: str = "") -> bool:
    """Adiciona ticker ao universo CORE. Retorna False se já existia."""
    entries = _load_universe()
    ticker  = ticker.upper().strip()
    if any(e["ticker"] == ticker for e in entries):
        return False
    entries.append({"ticker": ticker, "annual_yield_pct": annual_yield_pct, "note": note})
    _save_universe(entries)
    return True


def remove_from_universe(ticker: str) -> bool:
    """Remove ticker do universo CORE. Retorna False se não existia."""
    entries = _load_universe()
    ticker  = ticker.upper().strip()
    new     = [e for e in entries if e["ticker"] != ticker]
    if len(new) == len(entries):
        return False
    _save_universe(new)
    return True


def get_universe() -> list[dict]:
    return _load_universe()


# ─────────────────────────────────────────────────────────────────────────────
# Fetch de dados de preço
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_hist(ticker: str, lookback_days: int = 1400):
    """Devolve DataFrame com histórico de preços (Close, High, Low, Volume).

    Usa 1400 dias por defeito (≈5.5 anos) para computar a MA200 semanal,
    que é o indicador de suporte de longo prazo para entradas CORE de qualidade.
    """
    try:
        from data.data_feed import get_eod_prices
        import pandas as pd
        df = get_eod_prices(ticker, lookback_days=lookback_days)
        if df is None or df.empty:
            return None
        if "date" in df.columns:
            df = df.set_index(pd.to_datetime(df["date"])).sort_index()
        df.columns = [
            c.capitalize() if c.lower() in ("open","high","low","close","volume") else c
            for c in df.columns
        ]
        return df if len(df) >= 25 else None
    except Exception as e:
        log.debug(f"[core_radar] fetch {ticker}: {e}")
        return None


def _fetch_quality_metrics(ticker: str) -> dict:
    """Busca métricas de qualidade fundamental via yfinance.

    Retorna dict com: revenue_growth, gross_margin, roe, fcf_yield,
    analyst_upside, pe_forward, dividend_yield. Valores None se não disponíveis.

    Chamado no scan para enriquecer o score com sinais fundamentais —
    distingue "compounder sólido em desconto" de "negócio a deteriorar-se".

    dividend_yield é live (não o valor estático em core_universe.json) porque
    o yield varia inversamente com o preço — uma posição que estava a 1.5%
    yield a $135 pode estar a 4.2% a $41 (mesmo valor de dividendo nominal).
    O scan reflete a realidade actual.
    """
    try:
        import yfinance as yf
        info      = yf.Ticker(ticker).info
        price     = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
        mktcap    = float(info.get("marketCap") or 0)
        fcf       = float(info.get("freeCashflow") or 0)
        target    = float(info.get("targetMeanPrice") or 0)

        fcf_yield       = (fcf / mktcap) if (mktcap > 0 and fcf != 0) else None
        analyst_upside  = ((target / price) - 1) if (price > 0 and target > 0) else None

        # Live dividend yield — compute from dividendRate / currentPrice
        # instead of using info["dividendYield"] which yfinance has returned
        # in TWO formats (decimal 0.042 = 4.2% in some versions, percentage
        # 4.2 = 4.2% in others). The heuristic to detect which format breaks
        # for very-low-yield stocks (e.g., NVDA at 0.03% could be either):
        #
        #   - 0.03 raw could mean 3% (decimal interpretation) OR 0.03%
        #     (already percentage). The heuristic can't tell.
        #
        # Computing yield ourselves from dividendRate (annual USD per share)
        # divided by currentPrice gives a single, unambiguous value. Always
        # decimal, always correct, agnostic to yfinance version drift.
        div_yield_pct = None
        try:
            div_rate = info.get("dividendRate")
            if div_rate is not None and price and price > 0:
                yield_decimal = float(div_rate) / float(price)
                # Sanity bound — 30% is loose enough for REITs/MLPs but
                # tight enough to catch yfinance data errors (e.g., rate
                # accidentally being market cap)
                if 0 < yield_decimal <= 0.30:
                    div_yield_pct = yield_decimal * 100.0
        except (TypeError, ValueError):
            div_yield_pct = None

        return {
            "revenue_growth":  info.get("revenueGrowth"),      # decimal, y/y
            "gross_margin":    info.get("grossMargins"),        # decimal
            "roe":             info.get("returnOnEquity"),      # decimal
            "fcf_yield":       fcf_yield,                       # decimal
            "analyst_upside":  analyst_upside,                  # decimal
            "pe_forward":      info.get("forwardPE"),           # ratio
            "earnings_growth": info.get("earningsGrowth"),      # decimal, y/y
            "dividend_yield":  div_yield_pct,                   # PERCENTAGE (e.g., 4.2 not 0.042)
        }
    except Exception as e:
        log.debug(f"[core_radar] quality_metrics {ticker}: {e}")
        return {}


def _compute_volume_signals(close, volume) -> dict:
    """Analisa padrões de volume para detetar acumulação/distribuição institucional.

    Quando um compounder toca a MA200 semanal, volume alto em dias verdes indica
    que os grandes players (fundos, instituições) estão a aproveitar o dip para
    comprar — exactamente o padrão de MSFT/NVO nas grandes entradas históricas.

    Métricas:
      obv_trend_20d        : OBV acumulado nos últimos 20d (normalizado por avg_vol)
                             Positivo + preço em queda = divergência bullish clássica
      up_down_vol_ratio_20d: volume em dias verdes / volume em dias vermelhos (20d)
                             > 1.2 = mais capital a entrar do que a sair
      high_vol_up_days_20d : dias com volume > 1.5× média E preço positivo (20d)
                             ≥ 3 = assinatura de acumulação institucional
      vol_spike_5d_vs_60d  : volume médio 5d / volume médio 60d
                             > 1.5 = interesse elevado recente
    """
    result: dict = {}
    try:
        import math as _math
        n = len(close)
        if n < 21 or volume is None or len(volume) < 21:
            return result

        # ── OBV (On-Balance Volume) ────────────────────────────────────────
        price_change = close.diff()
        direction    = price_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv          = (volume * direction).cumsum()

        avg_vol_60 = float(volume.rolling(min(60, n)).mean().iloc[-1])
        if avg_vol_60 > 0:
            obv_20d_change = float(obv.iloc[-1] - obv.iloc[-21])
            # Normalizar em múltiplos de volume médio diário para ser interpretável
            result["obv_trend_20d"] = round(obv_20d_change / avg_vol_60, 2)

        # ── Up/Down volume ratio (20d) ─────────────────────────────────────
        chg_20   = price_change.iloc[-20:]
        vol_20   = volume.iloc[-20:]
        up_vol   = float(vol_20[chg_20 > 0].sum())
        down_vol = float(vol_20[chg_20 < 0].sum())
        if down_vol > 0:
            result["up_down_vol_ratio_20d"] = round(up_vol / down_vol, 3)

        # ── High-volume up days (20d) ──────────────────────────────────────
        # Dias onde instituições compram em força: volume > 1.5× média E preço sobe
        hvup = int(((vol_20 > avg_vol_60 * 1.5) & (chg_20 > 0)).sum())
        result["high_vol_up_days_20d"] = hvup

        # ── Volume spike recente (5d vs 60d) ──────────────────────────────
        avg_vol_5 = float(volume.iloc[-5:].mean())
        if avg_vol_60 > 0:
            result["vol_spike_5d_vs_60d"] = round(avg_vol_5 / avg_vol_60, 3)

    except Exception as e:
        log.debug(f"[core_radar] volume_signals: {e}")

    return result


def _compute_weekly_ma200(close_daily) -> Optional[float]:
    """Calcula a MA200 semanal (200 closes semanais ≈ 4 anos de dados).

    Resamples os closes diários para closes de sexta-feira e calcula a média
    dos últimos 200 closes semanais. Este é o indicador que o utilizador usa
    para identificar suporte de longo prazo (ex: MSFT a 359$ abaixo da MA200w).
    """
    try:
        import pandas as pd
        weekly = close_daily.resample("W-FRI").last().dropna()
        if len(weekly) < 20:
            return None
        ma200w = weekly.rolling(min(200, len(weekly))).mean().iloc[-1]
        return float(ma200w) if not math.isnan(ma200w) else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Scoring de uma oportunidade CORE
# ─────────────────────────────────────────────────────────────────────────────

def _score_core_dip(
    close,                 # pd.Series de closes diários (≥1000 barras para MA200w)
    current_price: float,
    annual_yield_pct: float,
    ticker: str = "",
    volume=None,           # pd.Series de volume diário (para análise de fluxo institucional)
) -> tuple[float, dict]:
    """Calcula score de 0-100 para uma oportunidade CORE dip.

    Dois eixos de avaliação:
      1. Preço: MA200 semanal (sinal primário), MA200d trend, RSI, dip depth
      2. Qualidade fundamental: FCF yield, revenue growth, gross margin, ROE,
         analyst consensus — distingue "desconto temporário em negócio intacto"
         de "negócio em deterioração estrutural"

    Retorna (score, signals_dict).
    """
    signals: dict = {}
    n = len(close)

    # Return 20d — confirmar que estamos num dip tático
    if n < 21:
        return 0.0, {}
    ret_20d = float(close.iloc[-1] / close.iloc[-21] - 1)
    signals["return_20d"] = round(ret_20d, 4)

    if ret_20d >= _ENTRY_RETURN_20D:
        return 0.0, signals  # não é um dip

    # ── MA200 diária (tendência de médio prazo) ────────────────────────────
    ma_p   = min(200, n - 1)
    ma200d = float(close.rolling(ma_p).mean().iloc[-1]) if ma_p >= 20 else None
    signals["ma200d"] = round(ma200d, 2) if ma200d else None

    # Descarte apenas em crise severa: preço >35% abaixo da MA200d
    # (limiar relaxado de 28% → 35% para não filtrar compounders em dip profundo como NVO)
    if ma200d and current_price < ma200d * _MA200_MIN_RATIO:
        return 0.0, signals

    # ── MA200 semanal (suporte secular — principal sinal CORE) ─────────────
    ma200w = _compute_weekly_ma200(close)
    signals["ma200w"] = round(ma200w, 2) if ma200w else None
    near_ma200w  = False
    below_ma200w = False
    if ma200w and ma200w > 0:
        ratio_to_ma200w = current_price / ma200w
        signals["price_vs_ma200w"] = round(ratio_to_ma200w - 1, 4)
        near_ma200w  = ratio_to_ma200w <= 1.05   # preço ≤5% acima da MA200w
        below_ma200w = ratio_to_ma200w < 1.00    # preço abaixo da MA200w (MSFT 359$)

    # Máximo de 52 semanas
    high_52w = float(close.rolling(min(252, n)).max().iloc[-1])
    dd_52w   = (current_price / high_52w - 1) if high_52w > 0 else 0.0
    signals["dd_from_52w_high"] = round(dd_52w, 4)

    # Descarte apenas em colapso extremo (>55%) — a menos que fundamentais sejam péssimos
    # NVO a -36% não é descarte; CISCO a -80% seria
    if dd_52w < _MAX_DD_FROM_52W:
        return 0.0, signals

    # RSI 14
    try:
        delta = close.diff()
        gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs    = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] > 0 else 100.0
        rsi   = round(100 - 100 / (1 + rs), 1)
    except Exception:
        rsi = 50.0
    signals["rsi_14"] = rsi

    # MA200 diária trend (crescente = tese secular intacta)
    ma200d_trend = 0.0
    if ma200d and n >= ma_p + 60:
        ma200d_60d = float(close.rolling(ma_p).mean().iloc[-61])
        ma200d_trend = (ma200d / ma200d_60d - 1) if ma200d_60d > 0 else 0.0
    signals["ma200d_trend_60d"] = round(ma200d_trend, 4)

    # Return 5d
    ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1) if n >= 6 else 0.0
    signals["return_5d"] = round(ret_5d, 4)

    # ── Métricas de qualidade fundamental ─────────────────────────────────
    quality: dict = _fetch_quality_metrics(ticker) if ticker else {}
    rev_growth    = quality.get("revenue_growth")   # decimal (None se indisponível)
    gross_margin  = quality.get("gross_margin")
    roe           = quality.get("roe")
    fcf_yield_q   = quality.get("fcf_yield")
    analyst_up    = quality.get("analyst_upside")
    earn_growth   = quality.get("earnings_growth")

    signals["revenue_growth"]  = round(rev_growth, 4)  if rev_growth  is not None else None
    signals["gross_margin"]    = round(gross_margin, 4) if gross_margin is not None else None
    signals["roe"]             = round(roe, 4)          if roe          is not None else None
    signals["fcf_yield_fund"]  = round(fcf_yield_q, 4)  if fcf_yield_q  is not None else None
    signals["analyst_upside"]  = round(analyst_up, 4)   if analyst_up   is not None else None
    signals["earnings_growth"] = round(earn_growth, 4)  if earn_growth  is not None else None
    # Live dividend yield (yfinance) — overrides the static value in
    # core_universe.json which can drift with price moves.
    div_yield_live = quality.get("dividend_yield")
    signals["dividend_yield_live"] = round(div_yield_live, 4) if div_yield_live is not None else None

    # Investment-phase detection: distinguish capex-driven negative FCF from
    # deterioration-driven negative FCF. A reinvesting compounder (AMZN
    # 1997-2015, NVO 2025 GLP-1 capacity buildout, hyperscalers building
    # AI data centers) has negative FCF temporarily because OpEx + CapEx >
    # OperatingCF — but topline growth + margins + ROE show the business
    # is fundamentally healthy and the investment is productive.
    #
    # When TRUE, the ML model's fcf_yield_fund input is neutralized at
    # inference (Option B) so the model doesn't false-flag the stock as
    # deteriorating. This is a heuristic override; Option C (retrain with
    # is_investment_phase as a model feature) is the long-term fix.
    signals["is_investment_phase"] = _is_investment_phase(
        rev_growth, gross_margin, roe, fcf_yield_q
    )

    # ── Mandatory fundamental quality gate ──────────────────────────────────
    # CoreRadar is for quality compounders only. A business with badly declining
    # revenue or negligible margins is not a compounder — don't alert regardless
    # of price action.
    if rev_growth is not None and rev_growth < -0.10:
        return 0.0, signals   # structural revenue decline — not a compounder
    if gross_margin is not None and gross_margin < 0.25:
        return 0.0, signals   # commodity-level margins — no pricing power

    # ── Mandatory price gate: near or below MA200 weekly ─────────────────────
    # CoreRadar's entire thesis is "buy quality compounders at or below the
    # 200-week MA". If the stock is not near that level, we have no entry thesis.
    # This is the gate that was missing and caused 40 alerts per night.
    if ma200w and not near_ma200w and not below_ma200w:
        return 0.0, signals   # not near MA200w — no entry thesis

    # ── Score calculation (0-based; the stock must earn every point) ─────────
    score = 0.0

    # Axis 1: Price — MA200 weekly proximity (primary signal, max 35 pts)
    if below_ma200w:
        score += 35.0   # at/below MA200w = historical entry zone (MSFT 2022, NVO 2024)
    elif near_ma200w:
        score += 20.0   # within 5% of MA200w = zone to watch

    # Dip depth — more pain, more opportunity (max 12 pts)
    dip_depth = abs(ret_20d)
    score += min(dip_depth * 150, 12.0)

    # MA200d trend — secular thesis intact?
    if ma200d_trend > 0.01:
        score += 8.0    # MA200d rising: long-term trend intact
    elif ma200d_trend > 0:
        score += 3.0
    else:
        score -= 10.0   # MA200d falling: compounding thesis at risk

    # RSI oversold (max 8 pts)
    if rsi < 30:
        score += 8.0
    elif rsi < 40:
        score += 5.0
    elif rsi < _RSI_MAX:
        score += 2.0

    # Drawdown from 52w high (max 5 pts)
    score += min(abs(dd_52w) * 20, 5.0)

    # Momentum still negative in last 5d — small penalty
    if ret_5d < -0.03:
        score -= 3.0

    # Dividend yield — explicit user signal (max 8 pts)
    if annual_yield_pct > 0 and current_price > 0:
        score += min(annual_yield_pct * _YIELD_BONUS_PER_PCT, 8.0)

    # Axis 2: Fundamental quality (max 32 pts)
    # These gates are real — a compounder must show quality fundamentals.
    # The CoreML model will replace this with learned weights once trained.
    if rev_growth is not None:
        if rev_growth >= _REV_GROWTH_STRONG:
            score += 10.0   # accelerating compounder
        elif rev_growth >= _REV_GROWTH_OK:
            score += 6.0
        elif rev_growth >= 0:
            score += 2.0
        elif rev_growth >= _REV_DECLINE_WARN:
            score -= 5.0    # mild revenue decline — reducing conviction
        # < -10% is already gated above

    if gross_margin is not None:
        if gross_margin >= _GROSS_MARGIN_MOAT:
            score += 10.0   # strong pricing power
        elif gross_margin >= _GROSS_MARGIN_OK:
            score += 5.0
        # < 25% gated above

    if roe is not None:
        if roe >= _ROE_COMPOUNDER:
            score += 8.0    # capital allocation machine
        elif roe >= _ROE_OK:
            score += 4.0
        elif roe < 0.05:
            score -= 5.0    # low ROE = not actually compounding

    if fcf_yield_q is not None:
        if fcf_yield_q >= _FCF_YIELD_STRONG:
            score += 7.0
        elif fcf_yield_q >= _FCF_YIELD_OK:
            score += 3.0
        elif fcf_yield_q < 0:
            score -= 6.0    # burning cash in a dip = danger

    if analyst_up is not None:
        if analyst_up >= 0.20:
            score += 5.0
        elif analyst_up >= 0.10:
            score += 2.0
        elif analyst_up < 0:
            score -= 4.0    # analysts see downside — unusual for quality compounder

    # ★ Eixo 3: Fluxo Institucional (Volume) ★ ──────────────────────────────
    # Quando o preço toca a MA200w, volume alto em dias verdes = fundos a acumular.
    # É o padrão observado em MSFT (2022, 2025) e NVO (2024): queda dramática com
    # acumulação silenciosa de grandes players = turning point de alta convicção.
    vol_sigs  = _compute_volume_signals(close, volume)
    signals.update(vol_sigs)

    obv_trend  = vol_sigs.get("obv_trend_20d", 0.0)
    ud_ratio   = vol_sigs.get("up_down_vol_ratio_20d", 1.0)
    hvup_days  = vol_sigs.get("high_vol_up_days_20d", 0)
    vol_spike  = vol_sigs.get("vol_spike_5d_vs_60d", 1.0)

    # OBV divergência bullish: OBV a subir enquanto preço cai = acumulação institucional
    # Sinal mais poderoso: as instituições estão a posicionar-se ANTES da viragem
    if obv_trend > 3.0 and ret_20d < -0.03:
        score += 10.0   # divergência forte: net buying pesado num dip
    elif obv_trend > 1.0 and ret_20d < -0.03:
        score += 6.0    # divergência moderada: mais compradores que vendedores
    elif obv_trend > 0:
        score += 3.0    # OBV positivo sem divergência: neutro/levemente bullish
    elif obv_trend < -3.0:
        score -= 8.0    # distribuição pesada: instituições a sair
    elif obv_trend < -1.0:
        score -= 4.0

    # Up/Down volume ratio: compara capital a entrar vs a sair nos últimos 20 dias
    if ud_ratio > 1.8:
        score += 8.0    # muito mais capital a comprar do que a vender = acumulação clara
    elif ud_ratio > 1.3:
        score += 5.0
    elif ud_ratio > 1.1:
        score += 2.0
    elif ud_ratio < 0.6:
        score -= 7.0    # distribuição: vendedores dominam com volume
    elif ud_ratio < 0.8:
        score -= 3.0

    # High-volume up days: assinatura directa de compra institucional em força
    # ≥3 dias de "block buying" em 20d = padrão de acumulação claro
    if hvup_days >= 4:
        score += 8.0
    elif hvup_days >= 2:
        score += 5.0
    elif hvup_days >= 1:
        score += 2.0

    # Volume spike recente + ratio positivo = catalisador de compra identificado
    # (Se volume disparou mas são vendedores → não pontua)
    if vol_spike > 2.0 and ud_ratio > 1.1:
        score += 5.0    # spike de interesse com compradores a dominar
    elif vol_spike > 1.5 and ud_ratio > 1.0:
        score += 2.0

    # Try ML model first — if loaded, it replaces the heuristic score entirely
    ml_score = _ml_score(signals, annual_yield_pct)
    if ml_score is not None:
        # ML model gates: apply the same mandatory gates even with ML scoring
        # (the model may not have seen extreme cases during training)
        if rev_growth is not None and rev_growth < -0.10:
            ml_score = 0.0
        if gross_margin is not None and gross_margin < 0.25:
            ml_score = 0.0
        signals["core_score"]    = ml_score
        signals["score_source"]  = "ml"
        signals["near_ma200w"]   = near_ma200w
        signals["below_ma200w"]  = below_ma200w
        return ml_score, signals

    # Heuristic fallback (used until ML model is trained)
    score = max(0.0, min(score, 100.0))
    signals["core_score"]    = round(score, 1)
    signals["score_source"]  = "heuristic"
    signals["near_ma200w"]   = near_ma200w
    signals["below_ma200w"]  = below_ma200w
    return round(score, 1), signals


# ─────────────────────────────────────────────────────────────────────────────
# Scanner principal
# ─────────────────────────────────────────────────────────────────────────────

def scan_core_universe(min_score: float = 60.0) -> list[dict]:
    """Varre o universo CORE e devolve oportunidades de dip com score >= min_score."""
    universe = _load_universe()
    if not universe:
        log.info("[core_radar] Universo CORE vazio — usa /core_universe_add para adicionar tickers")
        return []

    results: list[dict] = []
    today = date.today()

    for entry in universe:
        ticker     = entry["ticker"]
        ann_yield_static = float(entry.get("annual_yield_pct") or 0.0)
        note       = entry.get("note", "")

        hist = _fetch_hist(ticker)
        if hist is None:
            continue

        close         = hist["Close"]
        volume        = hist["Volume"] if "Volume" in hist.columns else None
        current_price = float(close.iloc[-1])

        # Score uses the live dividend yield when yfinance returns it; falls
        # back to the static seed value in core_universe.json (which can drift
        # as prices move — e.g., NVO was 1.5% at $135, now ~4.2% at $41).
        # _fetch_quality_metrics is called inside _score_core_dip, so to
        # avoid two yfinance round-trips we pass the static value through
        # and let the scoring layer override from its own quality fetch.
        score, signals = _score_core_dip(close, current_price, ann_yield_static, ticker=ticker, volume=volume)
        live_yield = signals.get("dividend_yield_live")
        ann_yield = float(live_yield) if live_yield is not None and live_yield > 0 else ann_yield_static
        if score < min_score:
            continue

        # Alocação sugerida: 1.5× a base da allocation engine para DIP tático
        suggested_eur = _suggest_core_allocation()

        opp = {
            "ticker":           ticker,
            "score":            score,
            "current_price":    round(current_price, 4),
            "annual_yield_pct": ann_yield,
            "note":             note,
            "suggested_eur":    round(suggested_eur, 0),
            "scan_date":        today.isoformat(),
            **signals,
        }
        results.append(opp)
        _append_core_alert_db(opp)
        log.info(
            f"[core_radar] CORE DIP {ticker} score={score:.0f} "
            f"price={current_price:.2f} yield={ann_yield:.1f}% "
            f"ret20d={signals.get('return_20d',0):+.1%} eur={suggested_eur:.0f}"
        )

    results.sort(key=lambda x: -x["score"])
    return results


_CORE_ALERT_DB = _DATA_DIR / "core_alert_db.csv"

_CORE_ALERT_COLS = [
    "alert_date", "ticker", "price", "core_score",
    "price_vs_ma200w", "ma200d_trend_60d", "return_20d", "dd_from_52w_high",
    "rsi_14", "obv_trend_20d", "up_down_vol_ratio_20d", "high_vol_up_days_20d",
    "vol_spike_5d_vs_60d", "revenue_growth", "gross_margin", "roe",
    "fcf_yield_fund", "analyst_upside", "earnings_growth", "annual_yield_pct",
    "alpha_1y", "alpha_2y",  # filled later by outcome filler
]


def _append_core_alert_db(opp: dict) -> None:
    """Appends one CoreRadar alert to the persistent feature store for CoreML training."""
    row = {col: opp.get(col, "") for col in _CORE_ALERT_COLS}
    row["alert_date"] = opp.get("scan_date", date.today().isoformat())
    row["price"]      = opp.get("current_price", "")
    row["core_score"] = opp.get("score", "")
    write_header = not _CORE_ALERT_DB.exists()
    try:
        with _CORE_ALERT_DB.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CORE_ALERT_COLS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as exc:
        log.warning(f"[core_radar] core_alert_db write failed: {exc}")


def _suggest_core_allocation() -> float:
    """Alocação sugerida para uma entrada CORE — 1.5× a base mensal per-trade."""
    try:
        from portfolio.paper_trading import _get_monthly_budget, _capital_used_this_month
        budget    = _get_monthly_budget()
        used      = _capital_used_this_month()
        remaining = budget - used
        # Assume máx 3 trades CORE por mês: 1/3 do remaining × factor
        base = remaining / 3
        return max(min(base * _CORE_ALLOC_FACTOR, remaining * 0.5), 50.0)
    except Exception:
        return 500.0  # fallback razoável


# ─────────────────────────────────────────────────────────────────────────────
# Formatação do alerta Telegram
# ─────────────────────────────────────────────────────────────────────────────

def format_core_alert(opp: dict) -> str:
    """Formata alerta CORE DIP para Telegram."""
    ticker    = opp["ticker"]
    score     = opp["score"]
    price     = opp["current_price"]
    ret20d    = opp.get("return_20d", 0)
    rsi       = opp.get("rsi_14", "-")
    dd_52w    = opp.get("dd_from_52w_high", 0)
    ma_trend  = opp.get("ma200d_trend_60d", 0)
    ann_yield = opp.get("annual_yield_pct", 0)
    eur       = opp.get("suggested_eur", 0)
    note      = opp.get("note", "")

    # MA200 semanal
    ma200w        = opp.get("ma200w")
    below_ma200w  = opp.get("below_ma200w", False)
    near_ma200w   = opp.get("near_ma200w", False)
    price_vs_ma200w = opp.get("price_vs_ma200w")

    # Fundamental quality signals
    rev_growth  = opp.get("revenue_growth")
    fcf_yield_f = opp.get("fcf_yield_fund")
    gross_m     = opp.get("gross_margin")
    roe_v       = opp.get("roe")
    analyst_up  = opp.get("analyst_upside")

    # Volume / fluxo institucional
    obv_trend  = opp.get("obv_trend_20d", 0.0)
    ud_ratio   = opp.get("up_down_vol_ratio_20d")
    hvup_days  = opp.get("high_vol_up_days_20d")
    vol_spike  = opp.get("vol_spike_5d_vs_60d")

    # MA200 semanal line
    if below_ma200w and price_vs_ma200w is not None:
        ma200w_line = f"  ★ Abaixo da MA200w: *{price_vs_ma200w:+.1%}* vs ${ma200w:.2f}"
    elif near_ma200w and price_vs_ma200w is not None:
        ma200w_line = f"  ★ Perto da MA200w: *{price_vs_ma200w:+.1%}* vs ${ma200w:.2f}"
    else:
        ma200w_line = ""

    # MA200 diária trend
    ma_icon = "" if ma_trend > 0 else "⚠️"
    ma_line = f"  MA200d {ma_trend:+.1%} (60d) {ma_icon}"

    # Fundamentais block
    fund_parts = []
    if rev_growth  is not None: fund_parts.append(f"Revenue {rev_growth:+.0%}")
    if fcf_yield_f is not None: fund_parts.append(f"FCF yield {fcf_yield_f:.1%}")
    if gross_m     is not None: fund_parts.append(f"Margem {gross_m:.0%}")
    if roe_v       is not None: fund_parts.append(f"ROE {roe_v:.0%}")
    if analyst_up  is not None: fund_parts.append(f"Analysts +{analyst_up:.0%}")
    fund_line = f"  {' | '.join(fund_parts)}" if fund_parts else ""

    # Volume / fluxo institucional block
    vol_parts = []
    if ud_ratio   is not None:
        arrow = "↑" if ud_ratio > 1.1 else ("↓" if ud_ratio < 0.9 else "→")
        vol_parts.append(f"Up/Down vol {arrow} {ud_ratio:.1f}×")
    if hvup_days  is not None and hvup_days > 0:
        vol_parts.append(f"{hvup_days} block-buy days")
    if obv_trend and abs(obv_trend) > 0.5:
        obv_arrow = "↑" if obv_trend > 0 else "↓"
        vol_parts.append(f"OBV {obv_arrow} {obv_trend:+.1f}")
    if vol_spike  is not None and vol_spike > 1.3:
        vol_parts.append(f"vol spike {vol_spike:.1f}×")
    vol_line = f"  Vol: {' | '.join(vol_parts)}" if vol_parts else ""

    yield_line = f"\n  Yield: *{ann_yield:.1f}%/ano*" if ann_yield >= 1 else ""
    note_line  = f"\n  _{note}_" if note else ""

    return (
        f"💎 *CoreRadar — {ticker}*  `score {score:.0f}/100`\n\n"
        + (f"{ma200w_line}\n" if ma200w_line else "")
        + f"  Preço: ${price:.2f} | DIP: {ret20d:+.1%} (20d) | vs Pico: {dd_52w:+.1%}\n"
        f"  RSI: {rsi}\n"
        f"{ma_line}\n"
        + (f"{vol_line}\n" if vol_line else "")
        + (f"{fund_line}\n" if fund_line else "")
        + f"{yield_line}{note_line}\n\n"
        f"  Alocação sugerida: *€{eur:.0f}*\n"
        f"  _Posição CORE — hold de longo prazo, sem auto-close_"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Option C — inline CORE block to enrich DIP alerts
#
# When the DIP scan fires for a ticker that lives in the CORE universe, we want
# a single consolidated Telegram message — not a DIP alert at 16h plus a CORE
# alert at 22h for the same event. These three helpers exist for that path:
#
#   get_core_universe_set()         — O(1) "is this a CORE ticker?" lookup
#   score_dip_for_inline_alert(...) — score the CORE thesis ad-hoc for one ticker
#   build_core_dip_block(...)       — compact CORE block to append to the DIP alert
#
# The full scan path (run_core_scan @ 22h) is unchanged. This is purely additive.
# ─────────────────────────────────────────────────────────────────────────────


def get_core_universe_set() -> set[str]:
    """Return CORE universe tickers as a set for O(1) membership tests.

    Called once per DIP scan iteration. _load_universe() reads /data/
    core_universe.json which is tiny — no caching needed."""
    return {e["ticker"] for e in _load_universe()}


def score_dip_for_inline_alert(ticker: str) -> Optional[tuple[float, dict]]:
    """Score the CORE thesis for a single ticker that just fired a DIP alert.

    Cost: two yfinance calls (one history fetch, one quality metrics fetch
    inside _score_core_dip). Only invoked when the DIP-alert ticker is also
    in the CORE universe — typically 0-2 events per day.

    Returns (score, signals) or None if the score gate (45) fails. The gate
    is lower than the standalone 22h scan (60) because the DIP scanner has
    already validated the intraday move ≥ DROP_THRESHOLD — a CORE score of
    45+ on a stock that just dropped 8%+ is informative even if it wouldn't
    fire as a standalone CORE alert.
    """
    entries = {e["ticker"]: e for e in _load_universe()}
    entry   = entries.get(ticker.upper())
    if entry is None:
        return None

    hist = _fetch_hist(ticker)
    if hist is None or hist.empty:
        return None

    close  = hist["Close"]
    volume = hist["Volume"] if "Volume" in hist.columns else None
    if len(close) < 30:
        return None

    current_price    = float(close.iloc[-1])
    ann_yield_static = float(entry.get("annual_yield_pct") or 0.0)

    score, signals = _score_core_dip(
        close, current_price, ann_yield_static,
        ticker=ticker, volume=volume,
    )
    if score < 45.0:
        return None

    live_yield    = signals.get("dividend_yield_live")
    ann_yield     = float(live_yield) if live_yield is not None and live_yield > 0 else ann_yield_static
    suggested_eur = _suggest_core_allocation()

    signals["annual_yield_pct"] = ann_yield
    signals["suggested_eur"]    = round(suggested_eur, 0)
    signals["note"]             = entry.get("note", "")
    signals["current_price"]    = round(current_price, 4)
    signals["score"]            = score

    _append_core_alert_db({
        "ticker":           ticker,
        "score":            score,
        "current_price":    current_price,
        "scan_date":        date.today().isoformat(),
        "annual_yield_pct": ann_yield,
        **signals,
    })

    log.info(
        f"[core_radar] CORE inline match {ticker} score={score:.0f} "
        f"yield={ann_yield:.1f}% eur={suggested_eur:.0f} "
        f"(triggered by DIP scan)"
    )
    return score, signals


def build_core_dip_block(score: float, signals: dict) -> str:
    """Compact CORE block to append to an existing DIP alert text.

    The DIP alert already has header, price, score breakdown, narrative.
    This adds only the CORE-specific layer: MA200w status, live yield,
    suggested allocation, and the compounder thesis note.
    """
    ma200w          = signals.get("ma200w")
    below_ma200w    = signals.get("below_ma200w", False)
    near_ma200w     = signals.get("near_ma200w", False)
    price_vs_ma200w = signals.get("price_vs_ma200w")
    ann_yield       = signals.get("annual_yield_pct", 0.0)
    eur             = signals.get("suggested_eur", 500)
    note            = signals.get("note", "")
    score_src       = signals.get("score_source", "heuristic")
    ma_trend        = signals.get("ma200d_trend_60d", 0.0)

    if below_ma200w and price_vs_ma200w is not None and ma200w:
        ma200w_line = f"MA200 semanal: ${ma200w:.2f} (*{price_vs_ma200w:+.1%}*, abaixo — breakdown técnico)"
    elif near_ma200w and price_vs_ma200w is not None and ma200w:
        ma200w_line = f"MA200 semanal: ${ma200w:.2f} (*{price_vs_ma200w:+.1%}*, zona de suporte)"
    elif ma200w:
        ma200w_line = f"MA200 semanal: ${ma200w:.2f} (acima da MA200w)"
    else:
        ma200w_line = "MA200 semanal: n/d"

    yield_line = f"Yield live: *{ann_yield:.2f}%*" if ann_yield >= 0.5 else ""

    ma_icon  = "✅" if ma_trend > 0.01 else ("⚠️" if ma_trend < 0 else "→")
    ma_d_line = f"MA200d trend: {ma_trend:+.1%} (60d) {ma_icon}"

    badge = "🤖 ML" if score_src == "ml" else "📐 heuristic"

    note_line = f'_"{note}"_' if note else ""

    lines = [
        f"\n💎 *CORE radar — match* `score {score:.0f}` {badge}",
        ma200w_line,
        ma_d_line,
        yield_line,
        f"Suggested: *€{eur:.0f}* (1.5× base, CORE conviction)",
        note_line,
        "_Posição CORE — hold longo prazo, sem auto-close_",
    ]
    return "\n".join(l for l in lines if l)
