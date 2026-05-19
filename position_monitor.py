"""
position_monitor.py — The Vigilante: daily thesis surveillance for DipRadar.

Orchestration flow (called once per day by APScheduler in main.py):
  run_daily_check(bot, chat_id)
    ├─ 1. Load all ACTIVE positions from position_db
    ├─ 2. For each ticker:
    │     a. Fetch today's market + macro data
    │     b. Build feature row via ml_features.build_feature_row()
    │     c. Re-run ml_engine.predict_dip() → new_win_prob, new_sell_target
    │     d. Evaluate triggers (ordered by severity):
    │          TAKE_PROFIT   → price ≥ current_sell_target
    │          DETERIORATION → Δwin_prob ≥ +0.15 (prob fell)
    │          TIME_DECAY    → days_held ≥ current_hold_days
    │          IMPROVEMENT   → Δwin_prob ≤ -0.10 (prob rose)
    │          ROUTINE       → everything nominal
    │     e. Update record in position_db (revised targets, health, history)
    │     f. Send Telegram alert if trigger fired
    └─ 3. Log summary

Trigger thresholds (tune via env vars):
  DIPR_DETERIORATION_THRESHOLD  float  default 0.15
  DIPR_IMPROVEMENT_THRESHOLD    float  default 0.10
  DIPR_ROUTINE_SILENT_DAYS      int    default 2   (send routine update every N days)

Error isolation:
  Each ticker is wrapped in try/except. One bad API response never kills
  the loop for the remaining positions. Errors are logged and a fallback
  Telegram message is sent for the failing ticker.

FIX (contract): a feature dict actual é produzido por
  ml_features.build_features(ticker, fundamentals); o `build_feature_row`
  só converte um dict para list ordenada — não aceita (ticker, fundamentals).
  Recolhemos os fundamentais via yfinance e passamos directamente um dict
  ao predictor (que aceita dict ou list). Erros isolados por ticker.

FIX (price): current_price is now fetched via yfinance directly (no
  MarketClient dependency). MarketClient import removed entirely.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from datetime import datetime, date
from typing import Optional

import yfinance as yf

import position_db
from position_db import PositionRecord
from ml_engine import (
    load_predictor,
    predict_dip,
    extract_shap_top3,
    format_shap_drivers,
)
from ml_features import build_features

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────────────────────

DETERIORATION_THRESHOLD = float(os.getenv("DIPR_DETERIORATION_THRESHOLD", "0.15"))
IMPROVEMENT_THRESHOLD   = float(os.getenv("DIPR_IMPROVEMENT_THRESHOLD",   "0.10"))
ROUTINE_SILENT_DAYS     = int(os.getenv("DIPR_ROUTINE_SILENT_DAYS",       "2"))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_current_price(ticker: str, fallback: float) -> float:
    """
    Obtém o preço actual via yfinance (fast_info.last_price).
    Usa fallback se a chamada falhar ou retornar None.
    Não depende de MarketClient.
    """
    try:
        info = yf.Ticker(ticker).fast_info
        price = getattr(info, "last_price", None)
        if price and float(price) > 0:
            return float(price)
    except Exception as e:
        logger.debug(f"[monitor] fast_info falhou para {ticker}: {e}")
    return fallback


def _fetch_fundamentals_snapshot(ticker: str, current_price: float = 0.0) -> dict:
    """
    Devolve um dict com fundamentais básicos para alimentar build_features.
    Retorna dict vazio em caso de erro — build_features imputa fallbacks internamente.

    Notas sobre o mapeamento yfinance → internal keys:
      - `freeCashflow` (USD) / `marketCap` → fcf_yield (ratio)
      - `targetMeanPrice` (USD) vs current_price → analyst_upside (ratio)
      - `debtToEquity` mapeia para a key `debt_equity` que build_features espera
    """
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info or {}

        market_cap = info.get("marketCap")
        free_cf    = info.get("freeCashflow")
        target_pr  = info.get("targetMeanPrice")

        # fcf_yield = FCF / market_cap (ratio); None se algum em falta ou inválido
        fcf_yield: Optional[float] = None
        if market_cap and free_cf and float(market_cap) > 0:
            try:
                fcf_yield = float(free_cf) / float(market_cap)
            except (TypeError, ValueError):
                fcf_yield = None

        # analyst_upside = (target - current) / current (ratio)
        analyst_upside: Optional[float] = None
        if target_pr and current_price and float(current_price) > 0:
            try:
                analyst_upside = (float(target_pr) - float(current_price)) / float(current_price)
            except (TypeError, ValueError):
                analyst_upside = None

        # Computar quality_score via score.py para detecção de deterioração estrutural.
        # Guardado em fundamentals_snap["entry_quality_score"] quando o record é criado.
        try:
            from score import score_from_fundamentals
            _fund_for_score = {
                "pe": info.get("trailingPE"),
                "debt_equity": info.get("debtToEquity"),
                "revenue_growth": info.get("revenueGrowth"),
                "gross_margin": info.get("grossMargins"),
                "fcf_yield": fcf_yield,
            }
            _score_result = score_from_fundamentals(_fund_for_score)
            quality_score = float(_score_result.get("quality_score", 0.5)) if isinstance(_score_result, dict) else 0.5
        except Exception:
            quality_score = None

        return {
            # build_features procura `pe` directo
            "pe":               info.get("trailingPE"),
            # build_features procura `debt_equity` (não `de_ratio`)
            "debt_equity":      info.get("debtToEquity"),
            "revenue_growth":   info.get("revenueGrowth"),
            "gross_margin":     info.get("grossMargins"),
            "market_cap":       market_cap,
            "fcf_yield":        fcf_yield,
            "analyst_upside":   analyst_upside,
            "price":            current_price if current_price else None,
            "quality_score":    quality_score,
            # Sector necessário para sector-conditioned features no modelo ML
            "sector":           info.get("sector") or "Unknown",
        }
    except Exception as e:
        logger.debug(f"[monitor] Fundamentals snapshot falhou para {ticker}: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Trigger classification
# ─────────────────────────────────────────────────────────────────────────────

TRIGGER_TAKE_PROFIT        = "TAKE_PROFIT"
TRIGGER_STOP_LOSS          = "STOP_LOSS"            # preço caiu X% do entry — sair
TRIGGER_EARLY_ALPHA        = "EARLY_ALPHA_CAPTURE"  # 70%+ do alpha em <50% do tempo
TRIGGER_STRUCTURAL_DECLINE = "STRUCTURAL_DECLINE"   # dip virou queda estrutural
TRIGGER_DETERIORATION      = "DETERIORATION"
TRIGGER_TIME_DECAY         = "TIME_DECAY"
TRIGGER_IMPROVEMENT        = "IMPROVEMENT"
TRIGGER_ROUTINE            = "ROUTINE"

# Stop-loss: se o preço cair STOP_LOSS_PCT% abaixo do preço de entrada, sair.
# Configurável via env var POSITION_STOP_LOSS_PCT (default 12%).
# Protege contra quedas estruturais que o modelo não detectou a tempo.
_STOP_LOSS_PCT = float(os.getenv("POSITION_STOP_LOSS_PCT", "0.12"))


def _detect_structural_decline(
    record: PositionRecord,
    current_price: float,
    new_win_prob: float,
    feature_row_today: dict,
) -> tuple[bool, str]:
    """Detecta se o dip se transformou numa queda estrutural.

    A Cisco, a IBM dos anos 2000, o WeWork — todas pareciam dips mas eram
    colapsos estruturais. Este detector usa múltiplos critérios:

    1. Win prob colapso severo (≥ 30pp): o modelo perdeu completamente
       a confiança na tese
    2. ML score invertido: previsão actual < -3% de alpha (território de VENDER)
    3. Recovery estagnada: > 45 dias com P&L ≤ -15%  (não está a recuperar)
    4. Deterioração de qualidade fundamental: quality_score actual caiu ≥ 15pp
       em relação ao valor na entrada (guardado em fundamentals_snap)

    Trigger quando ≥ 2 critérios (1 pode ser ruído de mercado):
    """
    reasons: list[str] = []
    delta = record.alert_win_prob - new_win_prob

    # 1. Win prob colapso severo
    if delta >= 0.30:
        reasons.append(f"Win prob colapsou {delta*100:.0f}pp (entrada → {record.alert_win_prob*100:.0f}%, hoje → {new_win_prob*100:.0f}%)")

    # 2. ML score territory invertido (abaixo de -3%)
    score_today = float(feature_row_today.get("ml_score_raw", float("nan")))
    if not __import__("math").isnan(score_today) and score_today < -0.03:
        reasons.append(f"Modelo prevê alpha negativo ({score_today*100:.1f}%) — tese invertida")

    # 3. Recovery estagnada: dias ≥ 45 e P&L ≤ -15%
    pnl = (current_price / record.alert_price - 1) if record.alert_price > 0 else 0.0
    if record.days_held >= 45 and pnl <= -0.15:
        reasons.append(f"Sem recuperação: {pnl*100:.0f}% após {record.days_held}d (limite 45d)")

    # 4. Deterioração de qualidade fundamental (guardada na entrada)
    snap = record.fundamentals_snap or {}
    entry_quality = snap.get("entry_quality_score")
    if entry_quality is not None:
        current_quality = feature_row_today.get("quality_score")
        if current_quality is not None:
            quality_drop = float(entry_quality) - float(current_quality)
            if quality_drop >= 0.15:
                reasons.append(f"Qualidade fundamental caiu {quality_drop*100:.0f}pp — deterioração operacional")

    is_structural = len(reasons) >= 2
    return is_structural, " | ".join(reasons)


def _check_early_alpha(record: PositionRecord, current_price: float) -> bool:
    """Detecta quando capturaste 70%+ do alpha esperado em <50% do tempo previsto.

    Ex: ML previu +12% em 90 dias. Ao fim de 30 dias já subiu +9% (75% do target
    em 33% do tempo). O mercado pode virar — considera realizar o lucro agora.
    """
    initial_pred = getattr(record, "initial_pred_alpha", None)
    if not initial_pred or initial_pred <= 0.02:
        return False  # sem previsão ou previsão muito baixa
    if record.alert_price <= 0:
        return False

    actual_return = (current_price / record.alert_price) - 1.0
    if actual_return <= 0:
        return False

    # Fracção do alpha previsto já capturado
    alpha_fraction = actual_return / initial_pred
    # Fracção do tempo previsto já decorrido
    time_fraction  = record.days_held / max(1, getattr(record, "initial_hold_days", 90))

    # Trigger: capturámos >70% do alpha esperado em <50% do tempo
    return alpha_fraction >= 0.70 and time_fraction < 0.50


def _classify_trigger(
    record: PositionRecord,
    current_price: float,
    new_win_prob: float,
    feature_row_today: dict | None = None,
) -> str:
    """Avalia triggers por ordem de severidade. Devolve o trigger de maior prioridade.

    Δwin_prob = alert_win_prob - new_win_prob:
      positivo → modelo perdeu confiança → deterioração
      negativo → modelo ganhou confiança → melhoria
    """
    delta = record.alert_win_prob - new_win_prob

    # 1. Preço atingiu o target → realizar lucro
    if current_price >= record.current_sell_target:
        return TRIGGER_TAKE_PROFIT

    # 1b. Stop-loss: preço caiu >STOP_LOSS_PCT% abaixo do entry → sair
    # Protege contra quedas estruturais que o modelo não detectou.
    # Só activa após 5 dias para não sair em volatilidade normal de curto prazo.
    if (record.alert_price > 0
            and record.days_held >= 5
            and current_price < record.alert_price * (1 - _STOP_LOSS_PCT)):
        return TRIGGER_STOP_LOSS

    # 1c. Early Alpha Capture: 70%+ do alpha em <50% do tempo → saída parcial
    if _check_early_alpha(record, current_price):
        return TRIGGER_EARLY_ALPHA

    # 2. Deterioração estrutural (multi-critério) — mais severo que deterioração simples
    if feature_row_today is not None:
        is_structural, _ = _detect_structural_decline(
            record, current_price, new_win_prob, feature_row_today
        )
        if is_structural:
            return TRIGGER_STRUCTURAL_DECLINE

    # 3. Win prob desceu ≥ threshold → tese deteriorando
    if delta >= DETERIORATION_THRESHOLD:
        return TRIGGER_DETERIORATION

    # 4. Período de holding expirou sem atingir target
    if record.days_held >= record.current_hold_days:
        return TRIGGER_TIME_DECAY

    # 5. Win prob subiu ≥ threshold → boas notícias
    if delta <= -IMPROVEMENT_THRESHOLD:
        return TRIGGER_IMPROVEMENT

    return TRIGGER_ROUTINE


def _build_early_alpha_alert(
    record: PositionRecord,
    current_price: float,
) -> str:
    pnl_pct = (current_price / record.alert_price - 1) * 100
    initial_pred = getattr(record, "initial_pred_alpha", 0) or 0
    alpha_fraction = (current_price / record.alert_price - 1) / initial_pred if initial_pred > 0 else 0
    time_fraction  = record.days_held / max(1, getattr(record, "initial_hold_days", 90))
    return "\n".join([
        f"💰 *EARLY ALPHA CAPTURE — {record.ticker}*  [Dia {record.days_held}/{record.initial_hold_days}]",
        f"_Capturaste {alpha_fraction*100:.0f}% do alpha previsto em apenas {time_fraction*100:.0f}% do tempo._",
        "",
        f"📈 P&L actual: *{_pct(pnl_pct)}* | ML previu: {initial_pred*100:.1f}% em {record.initial_hold_days}d",
        "",
        "💡 *Ação sugerida:* Considera vender 50-60% para garantir lucro.",
        "_Deixa o restante a rolar (moonbag) caso continue a subir._",
        f"_Target original: ${record.current_sell_target:.2f} ainda activo para a moonbag._",
    ])


def _resolve_thesis_health(trigger: str, delta: float) -> str:
    mapping = {
        TRIGGER_TAKE_PROFIT:        "STRONG",
        TRIGGER_STOP_LOSS:          "STRUCTURAL_DECLINE",
        TRIGGER_EARLY_ALPHA:        "STRONG",
        TRIGGER_IMPROVEMENT:        "IMPROVING",
        TRIGGER_ROUTINE:            "STRONG" if delta < 0.05 else "WEAKENING",
        TRIGGER_DETERIORATION:      "DETERIORATING",
        TRIGGER_STRUCTURAL_DECLINE: "STRUCTURAL_DECLINE",
        TRIGGER_TIME_DECAY:         "WEAKENING",
    }
    return mapping.get(trigger, "WEAKENING")


# ─────────────────────────────────────────────────────────────────────────────
# Telegram message builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_structural_decline_alert(
    record: PositionRecord,
    current_price: float,
    new_win_prob: float,
    reasons: str,
) -> str:
    pnl_pct = (current_price / record.alert_price - 1) * 100
    return "\n".join([
        f"🚨 *DETERIORAÇÃO ESTRUTURAL — {record.ticker}*  [Dia {record.days_held}/{record.current_hold_days}]",
        "",
        f"⚠️ *O que parecia um dip pode ser uma queda estrutural.*",
        f"",
        f"🔍 *Critérios activados:*",
        *[f"  • {r}" for r in reasons.split(" | ") if r],
        "",
        f"💰 *Posição actual:* ${current_price:.2f}  ({_pct(pnl_pct)} desde entrada ${record.alert_price:.2f})",
        f"🤖 *Win prob actual:* {new_win_prob*100:.0f}%  (entrada: {record.alert_win_prob*100:.0f}%)",
        "",
        "💡 *Ação sugerida:* Revê a tese. Considera sair ou cortar a posição.",
        "_A história está cheia de casos (Cisco, Nokia, GE) onde o 'dip' era início de queda secular._",
        f"_Monitorização diária activa — próxima verificação amanhã._",
    ])


def _pct(v, decimals=1) -> str:
    if v is None:
        return "N/A"
    sign = "+" if float(v) >= 0 else ""
    return f"{sign}{float(v):.{decimals}f}%"


def _build_deterioration_alert(
    record: PositionRecord,
    current_price: float,
    new_win_prob: float,
    new_sell_target: float,
    new_hold_days: int,
    drivers: list,
) -> str:
    delta_prob  = record.alert_win_prob - new_win_prob
    delta_sell  = new_sell_target - record.current_sell_target
    delta_hold  = new_hold_days - record.current_hold_days
    pnl_pct     = (current_price / record.alert_price - 1) * 100

    sell_arrow  = "↓" if delta_sell < 0 else "→"
    hold_arrow  = "↓" if delta_hold < 0 else "→"

    return "\n".join([
        f"🔴 *DETERIORAÇÃO — {record.ticker}*  [Dia {record.days_held}/{record.current_hold_days}]",
        f"_A tese para *{record.ticker}* está a enfraquecer — o modelo perdeu confiança {delta_prob*100:.0f}pp. Revê o stop-loss._",
        "",
        f"📉 *Confiança caiu {delta_prob*100:.0f} pontos*",
        f"  Alerta: {record.alert_win_prob*100:.0f}%  →  Hoje: {new_win_prob*100:.0f}%",
        "",
        "🔍 *Top 3 culpados (SHAP Δ)*",
        format_shap_drivers(drivers),
        "",
        "📊 *Revisão de targets*",
        f"  🎯 Venda:   ${record.current_sell_target:.2f} {sell_arrow} ${new_sell_target:.2f}  ({_pct(delta_sell/record.current_sell_target*100 if record.current_sell_target else 0)})",
        f"  ⏳ Holding: {record.current_hold_days}d {hold_arrow} {new_hold_days}d  ({delta_hold:+d} dias)",
        "",
        f"💰 *Posição actual:*  ${current_price:.2f}  ({_pct(pnl_pct)} desde entrada ${record.alert_price:.2f})",
        "",
        "💡 *Ação sugerida:* Revê stop-loss. Considera fechar parcialmente.",
        f"_Tese em DETERIORATING — monitorizando diariamente_",
    ])


def _build_take_profit_alert(
    record: PositionRecord,
    current_price: float,
) -> str:
    pnl_pct = (current_price / record.alert_price - 1) * 100
    return "\n".join([
        f"✅ *TAKE PROFIT — {record.ticker}*  [Dia {record.days_held}]",
        f"_*{record.ticker}* atingiu o target com {_pct(pnl_pct)} de ganho em {record.days_held} dias. Está na hora de fechar e rodar o capital._",
        "",
        f"🎯 Preço actual ${current_price:.2f} atingiu o target ${record.current_sell_target:.2f}",
        f"📈 P&L estimado: *{_pct(pnl_pct)}* desde alerta (${record.alert_price:.2f})",
        "",
        f"💡 *Ação sugerida:* Fecha a posição. Roda o capital.",
        f"_Win prob final: {record.last_win_prob*100:.0f}% | Holding: {record.days_held} dias_",
    ])


def _build_time_decay_alert(
    record: PositionRecord,
    current_price: float,
    new_win_prob: float,
) -> str:
    pnl_pct = (current_price / record.alert_price - 1) * 100
    _acao = (
        "O modelo ainda acredita na tese — considera dar mais alguns dias."
        if new_win_prob >= 0.55 else
        "O modelo perdeu convicção — pondera fechar e realocar o capital."
    )
    return "\n".join([
        f"⏰ *TEMPO ESGOTADO — {record.ticker}*  [Dia {record.days_held}/{record.initial_hold_days}]",
        f"_{record.ticker} chegou ao fim do período de holding sem atingir o target ({_pct(pnl_pct)} desde entrada). {_acao}_",
        "",
        f"⚠️ Target original: ${record.initial_sell_target:.2f} | Preço actual: ${current_price:.2f}",
        f"🤖 *Confiança actual:* {new_win_prob*100:.0f}%  (na entrada: {record.alert_win_prob*100:.0f}%)",
    ])


def _build_improvement_alert(
    record: PositionRecord,
    current_price: float,
    new_win_prob: float,
    new_sell_target: float,
    new_hold_days: int,
) -> str:
    delta_prob = record.alert_win_prob - new_win_prob  # negative = improvement
    delta_sell = new_sell_target - record.current_sell_target
    pnl_pct    = (current_price / record.alert_price - 1) * 100
    return "\n".join([
        f"📈 *TESE A MELHORAR — {record.ticker}*  [Dia {record.days_held}/{new_hold_days}]",
        f"_A tese para *{record.ticker}* ficou mais forte desde que entraste. O modelo está mais confiante — mantém e considera reforçar._",
        "",
        f"✅ Confiança subiu {abs(delta_prob)*100:.0f}pp: {record.alert_win_prob*100:.0f}% → *{new_win_prob*100:.0f}%*",
        f"💰 Posição actual: ${current_price:.2f}  ({_pct(pnl_pct)} desde entrada a ${record.alert_price:.2f})",
        f"🎯 Novo target: ${new_sell_target:.2f}  |  Prazo estendido: {new_hold_days}d",
    ])


def _build_routine_update(
    record: PositionRecord,
    current_price: float,
    new_win_prob: float,
) -> str:
    pnl_pct  = (current_price / record.alert_price - 1) * 100
    delta_p  = new_win_prob - record.alert_win_prob
    pnl_sign = "em ganho" if pnl_pct >= 0 else "em perda"
    return "\n".join([
        f"📡 *{record.ticker}* — Dia {record.days_held}/{record.current_hold_days}",
        f"_Tese intacta. {_pct(pnl_pct)} {pnl_sign}. Nenhuma acção necessária._",
        f"  Confiança: {new_win_prob*100:.0f}% ({delta_p:+.0%} vs entrada)  |  Target: ${record.current_sell_target:.2f}  |  {record.days_remaining}d restantes",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker monitoring logic
# ─────────────────────────────────────────────────────────────────────────────

def _monitor_one(
    record: PositionRecord,
    bundle: dict,
) -> tuple[str, Optional[str]]:
    """
    Process a single active position.

    Returns:
      (trigger, telegram_message | None)
      telegram_message is None for ROUTINE updates on silent days.
    """
    today_str = date.today().isoformat()

    # ── 1. Fetch current price (yfinance directo, sem MarketClient) ────────────
    current_price = _fetch_current_price(record.ticker, fallback=record.alert_price)

    # ── 2. Fetch fundamentals + price history e construir features do dia ────────
    fundamentals = _fetch_fundamentals_snapshot(record.ticker, current_price=current_price)

    # Fetch price history para RSI, ATR, momentum, MA200 correctos.
    # Sem price_history, ~15 features técnicas ficam em fallback → modelo não detecta
    # deterioração de momentum nem oversold/overbought em tempo real.
    _price_hist = None
    try:
        import yfinance as yf
        _raw = yf.Ticker(record.ticker).history(period="1y", auto_adjust=True)
        if _raw is not None and not _raw.empty:
            import pandas as pd
            _idx = pd.DatetimeIndex(_raw.index)
            if _idx.tz is not None:
                _raw.index = _idx.tz_convert(None)
            _price_hist = _raw
    except Exception as _he:
        logger.debug(f"[monitor] price_history {record.ticker}: {_he}")

    feature_row_today = build_features(
        record.ticker,
        fundamentals,
        price_history=_price_hist,
        sector=fundamentals.get("sector") or "Unknown",
    )

    # ── 3. Re-run inference ───────────────────────────────────────────────────
    pred = predict_dip(
        feature_row=feature_row_today,
        current_price=current_price,
        ticker=record.ticker,
        bundle=bundle,
    )
    new_win_prob    = pred.win_prob
    new_sell_target = pred.sell_target
    new_hold_days   = max(record.days_held + pred.hold_days, record.current_hold_days)

    # ── 4. Classify trigger ───────────────────────────────────────────────────
    # Passa feature_row_today para a detecção de deterioração estrutural.
    trigger = _classify_trigger(record, current_price, new_win_prob, feature_row_today)
    delta   = record.alert_win_prob - new_win_prob

    # ── 5. Update record ─────────────────────────────────────────────────────
    record.last_win_prob      = new_win_prob
    record.last_checked_date  = today_str
    record.thesis_health      = _resolve_thesis_health(trigger, delta)
    record.current_sell_target = new_sell_target
    record.current_hold_days   = new_hold_days

    # Append to history trail (keep last 90 entries)
    record.history.append({
        "date":         today_str,
        "price":        round(current_price, 4),
        "win_prob":     round(new_win_prob, 4),
        "sell_target":  round(new_sell_target, 4),
        "hold_days":    new_hold_days,
        "trigger":      trigger,
        "health":       record.thesis_health,
    })
    if len(record.history) > 90:
        record.history = record.history[-90:]

    # ── 6. Handle terminal triggers ───────────────────────────────────────────
    if trigger == TRIGGER_TAKE_PROFIT:
        record.status       = "TAKE_PROFIT"
        record.close_reason = "TAKE_PROFIT"
        record.closed_at    = datetime.utcnow().isoformat()
        record.close_price  = current_price

    # ── 7. Persist ────────────────────────────────────────────────────────────
    position_db.update_record(record)

    # ── 8. Build Telegram message ─────────────────────────────────────────────
    msg: Optional[str] = None

    if trigger == TRIGGER_TAKE_PROFIT:
        msg = _build_take_profit_alert(record, current_price)

    elif trigger == TRIGGER_STOP_LOSS:
        pnl_pct = (current_price / record.alert_price - 1) * 100
        msg = "\n".join([
            f"STOP-LOSS atingido — {record.ticker}  [Dia {record.days_held}]",
            f"Preco caiu {abs(pnl_pct):.1f}% abaixo do entry (limite: {_STOP_LOSS_PCT*100:.0f}%).",
            "",
            f"Entry: ${record.alert_price:.2f}  |  Actual: ${current_price:.2f}  |  P&L: {_pct(pnl_pct)}",
            "",
            "Acao: Fechar posicao. Proteger capital para proxima oportunidade.",
            "_O stop-loss existe para que uma posicao errada nao destrua multiplas certas._",
        ])
        record.status       = "CLOSED"
        record.close_reason = "STOP_LOSS"
        record.closed_at    = datetime.utcnow().isoformat()
        record.close_price  = current_price

    elif trigger == TRIGGER_EARLY_ALPHA:
        msg = _build_early_alpha_alert(record, current_price)

    elif trigger == TRIGGER_STRUCTURAL_DECLINE:
        _, reasons_str = _detect_structural_decline(
            record, current_price, new_win_prob, feature_row_today
        )
        msg = _build_structural_decline_alert(record, current_price, new_win_prob, reasons_str)

    elif trigger == TRIGGER_DETERIORATION:
        drivers = extract_shap_top3(
            bundle=bundle,
            row_alert=record.alert_feature_row,
            row_today=feature_row_today,
        )
        msg = _build_deterioration_alert(
            record, current_price, new_win_prob,
            new_sell_target, new_hold_days, drivers,
        )

    elif trigger == TRIGGER_TIME_DECAY:
        msg = _build_time_decay_alert(record, current_price, new_win_prob)

    elif trigger == TRIGGER_IMPROVEMENT:
        msg = _build_improvement_alert(
            record, current_price, new_win_prob, new_sell_target, new_hold_days
        )

    elif trigger == TRIGGER_ROUTINE:
        # Send routine update only every ROUTINE_SILENT_DAYS days
        if record.days_held % ROUTINE_SILENT_DAYS == 0:
            msg = _build_routine_update(record, current_price, new_win_prob)

    logger.info(
        f"[monitor] {record.ticker}  trigger={trigger}  "
        f"win_prob={new_win_prob:.3f}  price={current_price:.2f}  "
        f"health={record.thesis_health}"
    )
    return trigger, msg


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_daily_check(send_fn) -> dict:
    """Vigilância diária de posições activas. Chamado pelo APScheduler em main.py.

    Parameters
    ----------
    send_fn : callable
        Função send_telegram(text) — mesma que o resto do bot usa.
        Não usa async/await para ser compatível com APScheduler síncrono.
    """
    active = position_db.get_active()
    if not active:
        logger.info("[monitor] Sem posições activas — check ignorado")
        return {"n_active": 0}

    logger.info(f"[monitor] A verificar {len(active)} posições activas")

    try:
        bundle = load_predictor()
    except FileNotFoundError:
        logger.error("[monitor] Bundle ML não encontrado — a abortar check diário")
        send_fn("Monitor diário falhou: modelo ML não encontrado. Faz /admin_retrain.")
        return {"error": "model_not_found"}

    stats = {
        "n_active":        len(active),
        "n_take_profit":   0,
        "n_deterioration": 0,
        "n_time_decay":    0,
        "n_improvement":   0,
        "n_routine":       0,
        "n_errors":        0,
    }

    for record in active:
        try:
            trigger, msg = _monitor_one(record, bundle)
            stats[f"n_{trigger.lower()}"] = stats.get(f"n_{trigger.lower()}", 0) + 1
            if msg:
                send_fn(msg)
        except Exception as exc:
            stats["n_errors"] += 1
            logger.exception(f"[monitor] Erro em {record.ticker}: {exc}")
            try:
                send_fn(
                    f"Monitor {record.ticker}: erro {type(exc).__name__}: {exc}\n"
                    "Os restantes tickers continuam a ser verificados."
                )
            except Exception:
                pass

    summary = (
        f"📡 Vigilante — resumo\n"
        f"  Activas: {stats['n_active']} | TP: {stats['n_take_profit']} "
        f"| Det: {stats['n_deterioration']} | TD: {stats['n_time_decay']} "
        f"| Imp: {stats['n_improvement']} | Err: {stats['n_errors']}"
    )
    logger.info(f"[monitor] Check concluído — {summary}")
    send_fn(summary)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI test  (python position_monitor.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    print("=== position_monitor dry-run ===")

    from ml_features import FEATURE_COLUMNS
    import numpy as np

    # Inject a fake ACTIVE position into the DB for testing
    np.random.seed(0)
    fake_features = [float(np.random.uniform(0, 1)) for _ in FEATURE_COLUMNS]
    fake_features[FEATURE_COLUMNS.index("macro_score")]    = 2.0
    fake_features[FEATURE_COLUMNS.index("rsi_14")]         = 32.0
    fake_features[FEATURE_COLUMNS.index("drop_pct_today")] = -5.0
    fake_features[FEATURE_COLUMNS.index("vix")]            = 20.0

    test_record = PositionRecord(
        ticker="TEST",
        status="ACTIVE",
        alert_date="2026-04-01",
        alert_price=100.0,
        alert_win_prob=0.72,
        alert_feature_row=fake_features,
        initial_buy_target=98.0,
        initial_sell_target=115.0,
        initial_hold_days=30,
        dip_score=71.5,
        fundamentals_snap={"pe": 22.0, "fcf_yield": 0.058},
        current_sell_target=115.0,
        current_hold_days=30,
        last_win_prob=0.72,
        thesis_health="STRONG",
    )

    position_db.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    position_db.add_position(test_record)
    print(f"Inserted test position for TEST")
    print()
    print("Active positions summary:")
    print(position_db.summary_text())
    print()
    print("Trigger classification tests:")
    rec = position_db.get_by_ticker("TEST")
    print(f"  price=116 (≥ sell_target) → {_classify_trigger(rec, 116.0, 0.72)}")
    print(f"  win_prob=0.50 (Δ=0.22)    → {_classify_trigger(rec, 105.0, 0.50)}")
    print(f"  days_held>=30, no target  → {_classify_trigger(rec, 104.0, 0.68)}")
    print(f"  win_prob=0.85 (Δ=-0.13)   → {_classify_trigger(rec, 105.0, 0.85)}")
    print(f"  nominal                   → {_classify_trigger(rec, 105.0, 0.70)}")
