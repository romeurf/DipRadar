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

import json
import logging
import math
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger("core_radar")

_REPO_DIR     = Path(__file__).parent
_UNIVERSE_FILE = _REPO_DIR / "core_universe.json"
_DATA_DIR      = Path("/data") if Path("/data").exists() else _REPO_DIR

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
        from data_feed import get_eod_prices
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
    analyst_upside, pe_forward. Valores None se não disponíveis.

    Chamado no scan para enriquecer o score com sinais fundamentais —
    distingue "compounder sólido em desconto" de "negócio a deteriorar-se".
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

        return {
            "revenue_growth":  info.get("revenueGrowth"),      # decimal, y/y
            "gross_margin":    info.get("grossMargins"),        # decimal
            "roe":             info.get("returnOnEquity"),      # decimal
            "fcf_yield":       fcf_yield,                       # decimal
            "analyst_upside":  analyst_upside,                  # decimal
            "pe_forward":      info.get("forwardPE"),           # ratio
            "earnings_growth": info.get("earningsGrowth"),      # decimal, y/y
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

    # ── Cálculo do score ─────────────────────────────────────────────────────
    score = 50.0  # base

    # ★ Eixo 1: Preço ★ ──────────────────────────────────────────────────────

    # Sinal primário: MA200 semanal
    if below_ma200w:
        score += 25.0   # abaixo da MA200w = oportunidade histórica (MSFT 359$, NVO 38$)
    elif near_ma200w:
        score += 15.0   # perto da MA200w = zona de suporte a vigiar

    # Profundidade do dip tático
    dip_depth = abs(ret_20d)
    score += min(dip_depth * 250, 15.0)  # até +15 pts

    # MA200 diária crescente (tese secular intacta)
    if ma200d_trend > 0.01:
        score += 8.0
    elif ma200d_trend > 0:
        score += 4.0
    else:
        score -= 8.0  # MA200d em queda = cuidado

    # RSI oversold
    if rsi < 30:
        score += 8.0
    elif rsi < 40:
        score += 5.0
    elif rsi < _RSI_MAX:
        score += 2.0

    # Drawdown 52w: quanto mais profundo, mais atrativo (até certo ponto)
    score += min(abs(dd_52w) * 30, 6.0)

    # Momentum ainda negativo nos últimos 5d — penalizar levemente
    if ret_5d < -0.03:
        score -= 4.0

    # Yield de dividendo — sinal explícito definido pelo utilizador
    if annual_yield_pct > 0 and current_price > 0:
        score += min(annual_yield_pct * _YIELD_BONUS_PER_PCT, 10.0)  # até +10 pts

    # Nota: FCF yield, revenue growth, gross margin, ROE, analyst upside são
    # recolhidos como features (acima, em signals) mas NÃO aplicam scoring
    # hardcoded. O CoreML aprenderá os pesos correctos a partir de dados históricos.
    # Aplicar pesos manuais seria arbitrário — o ML fá-lo melhor.

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

    score = max(0.0, min(score, 100.0))
    signals["core_score"]   = round(score, 1)
    signals["near_ma200w"]  = near_ma200w
    signals["below_ma200w"] = below_ma200w
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
        ann_yield  = float(entry.get("annual_yield_pct") or 0.0)
        note       = entry.get("note", "")

        hist = _fetch_hist(ticker)
        if hist is None:
            continue

        close         = hist["Close"]
        volume        = hist["Volume"] if "Volume" in hist.columns else None
        current_price = float(close.iloc[-1])

        score, signals = _score_core_dip(close, current_price, ann_yield, ticker=ticker, volume=volume)
        if score < min_score:
            continue

        # Alocação sugerida: 1.5× a base da allocation engine para DIP tático
        suggested_eur = _suggest_core_allocation()

        results.append({
            "ticker":           ticker,
            "score":            score,
            "current_price":    round(current_price, 4),
            "annual_yield_pct": ann_yield,
            "note":             note,
            "suggested_eur":    round(suggested_eur, 0),
            "scan_date":        today.isoformat(),
            **signals,
        })
        log.info(
            f"[core_radar] CORE DIP {ticker} score={score:.0f} "
            f"price={current_price:.2f} yield={ann_yield:.1f}% "
            f"ret20d={signals.get('return_20d',0):+.1%} eur={suggested_eur:.0f}"
        )

    results.sort(key=lambda x: -x["score"])
    return results


def _suggest_core_allocation() -> float:
    """Alocação sugerida para uma entrada CORE — 1.5× a base mensal per-trade."""
    try:
        from paper_trading import _get_monthly_budget, _capital_used_this_month
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
