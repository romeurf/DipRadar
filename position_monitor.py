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
            # current_price pode entrar no dict para build_features e cair em fallback
            "price":            current_price if current_price else None,
        }
    except Exception as e:
        logger.debug(f"[monitor] Fundamentals snapshot falhou para {ticker}: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Trigger classification
# ─────────────────────────────────────────────────────────────────────────────

TRIGGER_TAKE_PROFIT   = "TAKE_PROFIT"
TRIGGER_DETERIORATION = "DETERIORATION"
TRIGGER_TIME_DECAY    = "TIME_DECAY"
TRIGGER_IMPROVEMENT   = "IMPROVEMENT"
TRIGGER_ROUTINE       = "ROUTINE"


def _classify_trigger(
    record: PositionRecord,
    current_price: float,
    new_win_prob: float,
) -> str:
    """
    Evaluate triggers in severity order. Return the highest-priority trigger.
    Δwin_prob is defined as (alert_win_prob - new_win_prob):
      positive → model became less confident → deterioration
      negative → model became more confident → improvement
    """
    delta = record.alert_win_prob - new_win_prob  # positive = deterioration

    # 1. Price hit the sell target — take profit
    if current_price >= record.current_sell_target:
        return TRIGGER_TAKE_PROFIT

    # 2. Win prob dropped ≥ threshold — thesis deteriorating
    if delta >= DETERIORATION_THRESHOLD:
        return TRIGGER_DETERIORATION

    # 3. Hold period expired without hitting target
    if record.days_held >= record.current_hold_days:
        return TRIGGER_TIME_DECAY

    # 4. Win prob improved ≥ threshold — good news
    if delta <= -IMPROVEMENT_THRESHOLD:
        return TRIGGER_IMPROVEMENT

    return TRIGGER_ROUTINE


def _resolve_thesis_health(trigger: str, delta: float) -> str:
    mapping = {
        TRIGGER_TAKE_PROFIT:   "STRONG",
        TRIGGER_IMPROVEMENT:   "IMPROVING",
        TRIGGER_ROUTINE:       "STRONG" if delta < 0.05 else "WEAKENING",
        TRIGGER_DETERIORATION: "DETERIORATING",
        TRIGGER_TIME_DECAY:    "WEAKENING",
    }
    return mapping.get(trigger, "WEAKENING")


# ─────────────────────────────────────────────────────────────────────────────
# Telegram message builders
# ─────────────────────────────────────────────────────────────────────────────

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
        "",
        f"📉 *Win prob caiu {delta_prob*100:.0f} pontos*",
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
    return "\n".join([
        f"⏰ *TIME DECAY — {record.ticker}*  [Dia {record.days_held}/{record.initial_hold_days}]",
        "",
        f"⚠️ Período de holding expirou sem atingir o target.",
        f"  Target original: ${record.initial_sell_target:.2f}",
        f"  Preço actual:    ${current_price:.2f}  ({_pct(pnl_pct)})",
        "",
        f"🤖 *Win prob actual:* {new_win_prob*100:.0f}%  (alerta: {record.alert_win_prob*100:.0f}%)",
        "",
        "💡 *Ação sugerida:* " + (
            "Win prob ainda razoável — considera estender o holding mais alguns dias."
            if new_win_prob >= 0.55 else
            "Win prob degradada — pondera fechar e realocar capital."
        ),
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
        f"📈 *MELHORIA DE TESE — {record.ticker}*  [Dia {record.days_held}/{new_hold_days}]",
        "",
        f"✅ *Win prob subiu {abs(delta_prob)*100:.0f} pontos*",
        f"  Alerta: {record.alert_win_prob*100:.0f}%  →  Hoje: {new_win_prob*100:.0f}%",
        "",
        "📊 *Targets revistos em alta*",
        f"  🎯 Venda:   ${record.current_sell_target:.2f} → ${new_sell_target:.2f}  ({_pct(delta_sell/record.current_sell_target*100 if record.current_sell_target else 0)})",
        f"  ⏳ Holding: {record.current_hold_days}d → {new_hold_days}d",
        "",
        f"💰 *Posição actual:*  ${current_price:.2f}  ({_pct(pnl_pct)} desde entrada)",
        "",
        "💡 *Ação sugerida:* Mantém. Considera aumentar posição.",
        f"_Tese em IMPROVING — o modelo está cada vez mais confiante_",
    ])


def _build_routine_update(
    record: PositionRecord,
    current_price: float,
    new_win_prob: float,
) -> str:
    pnl_pct  = (current_price / record.alert_price - 1) * 100
    delta_p  = new_win_prob - record.alert_win_prob
    return "\n".join([
        f"📡 *Monitor diário — {record.ticker}*  [Dia {record.days_held}/{record.current_hold_days}]",
        "",
        f"🟢 Tese estável",
        f"  Win prob: {new_win_prob*100:.0f}%  ({delta_p:+.0%} vs alerta)",
        f"  Preço actual: ${current_price:.2f}  ({_pct(pnl_pct)})",
        f"  Target venda: ${record.current_sell_target:.2f}  |  {record.days_remaining}d restantes",
        f"_Nenhuma acção necessária_",
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

    # ── 2. Fetch fundamentals snapshot e construir o dict de features ao dia ────
    # build_features(ticker, fundamentals) → dict com 23 keys (FEATURE_COLUMNS).
    # Sem price_history/sector → atr/volume/momentum caem em fallback determinístico,
    # o que é aceitável para vigilância diária (não exige PIT exacto).
    fundamentals = _fetch_fundamentals_snapshot(record.ticker, current_price=current_price)
    feature_row_today = build_features(record.ticker, fundamentals)

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
    trigger = _classify_trigger(record, current_price, new_win_prob)
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

async def run_daily_check(bot, chat_id: int) -> dict:
    """
    Daily surveillance loop. Called by APScheduler in main.py.

    Parameters
    ----------
    bot     : telegram.Bot   Telegram bot instance
    chat_id : int            Chat ID to send alerts to

    Returns
    -------
    dict with summary stats:
      { n_active, n_take_profit, n_deterioration, n_time_decay,
        n_improvement, n_routine, n_errors }
    """
    active = position_db.get_active()
    if not active:
        logger.info("[monitor] No active positions — daily check skipped")
        return {"n_active": 0}

    logger.info(f"[monitor] Starting daily check — {len(active)} active positions")

    # Load the model bundle once (cached after first load)
    try:
        bundle = load_predictor()
    except FileNotFoundError:
        logger.error("[monitor] Model bundle not found — aborting daily check")
        await bot.send_message(
            chat_id=chat_id,
            text="⚠️ *Monitor diário falhou*: modelo ML não encontrado. "
                 "Executa o treino primeiro.",
            parse_mode="Markdown",
        )
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
                await bot.send_message(
                    chat_id=chat_id,
                    text=msg,
                    parse_mode="Markdown",
                )

        except Exception as exc:
            stats["n_errors"] += 1
            logger.exception(f"[monitor] Error processing {record.ticker}: {exc}")
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=(
                        f"⚠️ *Monitor — {record.ticker}*\n"
                        f"Erro ao processar posição: `{type(exc).__name__}: {exc}`\n"
                        "_A monitorização dos restantes tickers continua._"
                    ),
                    parse_mode="Markdown",
                )
            except Exception:
                pass  # Don't let Telegram errors cascade

    summary = (
        f"📡 *Vigilante — resumo diário*\n"
        f"  Activas: {stats['n_active']}  "
        f"| ✅ TP: {stats['n_take_profit']}  "
        f"| 🔴 Det: {stats['n_deterioration']}  "
        f"| ⏰ TD: {stats['n_time_decay']}  "
        f"| 📈 Imp: {stats['n_improvement']}  "
        f"| ⚠️ Err: {stats['n_errors']}"
    )
    logger.info(f"[monitor] Daily check complete — {summary}")
    await bot.send_message(chat_id=chat_id, text=summary, parse_mode="Markdown")

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
