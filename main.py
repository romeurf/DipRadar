"""
DipRadar — Stock Alert Bot v2.0
Scheduler: APScheduler (duas janelas EOD: 17h45 EU, 21h15 US)
Data feed: Tiingo EOD + yfinance fallback

Variáveis Railway obrigatórias:
  TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
  TZ=Europe/Lisbon

Variáveis opcionais:
  TIINGO_API_KEY               (Chunk 3a — data feed principal)
  DROP_THRESHOLD=8
  MIN_MARKET_CAP=2000000000
  SCAN_EVERY_MINUTES=30
  MIN_DIP_SCORE=50
  TAVILY_API_KEY
  PORTFOLIO_STRESS_PCT=5
  RECOVERY_TARGET_PCT=15
  WATCHLIST_SCAN_ENABLED=true
  DEGRADATION_DROP_THRESHOLD=15
"""

import os
import time
import logging
import requests
from pathlib import Path
from datetime import datetime, timedelta

# APScheduler — substitui `schedule` para cron jobs precisos
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron       import CronTrigger
import pytz

from market_client import (
    screen_global_dips, screen_structural_dips, screen_period_dips,
    get_fundamentals, get_news, get_historical_pe,
    get_52w_drawdown, get_earnings_date, get_earnings_days,
    get_catalyst, get_spy_change, is_market_open,
    get_usdeur, get_portfolio_snapshot,
    get_macro_context,
)
from portfolio import (
    HOLDINGS, PPR_SHARES, PPR_AVG_COST,
    DIRECT_TICKERS, FLIP_FUND_EUR, suggest_position_size,
    get_positions, update_position_data,
    mark_degradation_alerted, reset_degradation_flag,
)
from sectors import get_sector_config, score_fundamentals
from valuation import format_valuation_block
from score import (
    calculate_dip_score, build_score_breakdown, classify_dip_category,
    is_bluechip,
    score_from_fundamentals, format_score_v2_breakdown,
    CATEGORY_HOLD_FOREVER, CATEGORY_APARTAMENTO, CATEGORY_ROTACAO,
)
from state import (
    load_alerts, save_alerts, clear_alerts,
    load_weekly_log, save_weekly_log, append_weekly_log,
    load_rejected_log, append_rejected_log,
    append_backtest_entry,
    load_recovery_watch, add_recovery_position,
    mark_recovery_alerted, remove_recovery_position,
    get_stale_recovery_positions, mark_stale_alerted,
    # ── Feature 8: Dip Persistente ──────────────────────────────────
    record_dip_day, mark_persistent_alerted, expire_missing_streaks,
)
from backtest import backtest_runner, build_backtest_summary
from watchlist import run_watchlist_scan, build_watchlist_morning_summary, WATCHLIST
from alert_db import log_alert_snapshot, get_db_stats, fill_db_outcomes
import bot_commands
from position_monitor import run_daily_check

# Chunk 3a — data feed
from data_feed import get_eod_prices, get_latest_price, is_tiingo_available

# Chunk 3 — universe
from universe import get_full_universe, get_ml_universe, ETF_TICKERS, is_etf

# Chunk 6 — ML predictor
from ml_predictor import ml_score, ml_badge, is_model_ready, MLResult

# Feature 8 — Dip Persistente
from persistent_dip import check_and_alert_streak

# ── Label Resolver — Flywheel de ML ──────────────────────────────────────────
from label_resolver import run_label_resolver_job

# ── Backup diário ──────────────────────────────────────────────────────────────
from backup_data import run_backup, build_telegram_summary as backup_telegram_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

TELEGRAM_TOKEN    = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID  = os.environ.get("TELEGRAM_CHAT_ID", "")
DROP_THRESHOLD    = float(os.environ.get("DROP_THRESHOLD", "8"))
MIN_MARKET_CAP    = int(os.environ.get("MIN_MARKET_CAP", "2000000000"))
SCAN_MINUTES      = int(os.environ.get("SCAN_EVERY_MINUTES", "30"))
MIN_DIP_SCORE     = int(os.environ.get("MIN_DIP_SCORE", "50"))
STRESS_PCT        = float(os.environ.get("PORTFOLIO_STRESS_PCT", "5"))
RECOVERY_PCT      = float(os.environ.get("RECOVERY_TARGET_PCT", "15"))
WATCHLIST_ENABLED = os.environ.get("WATCHLIST_SCAN_ENABLED", "true").lower() == "true"
DEGRADATION_DROP_THRESHOLD = int(os.environ.get("DEGRADATION_DROP_THRESHOLD", "15"))

LISBON_TZ = pytz.timezone("Europe/Lisbon")

_alerted_today:  set  = load_alerts()
_scan_running:   bool = False
_stress_alerted: set  = set()
_last_tier3:     list = []


# ── Sector ETF map ────────────────────────────────────────────────────────────

_SECTOR_ETF = {
    "Technology":             "XLK",
    "Healthcare":             "XLV",
    "Communication Services": "XLC",
    "Financial Services":     "XLF",
    "Consumer Cyclical":      "XLY",
    "Consumer Defensive":     "XLP",
    "Industrials":            "XLI",
    "Energy":                 "XLE",
    "Utilities":              "XLU",
    "Real Estate":            "XLRE",
    "Basic Materials":        "XLB",
}
_sector_etf_cache: dict = {}

def get_sector_change(sector: str) -> float | None:
    etf = _SECTOR_ETF.get(sector)
    if not etf:
        return None
    if etf in _sector_etf_cache:
        return _sector_etf_cache[etf]
    try:
        import yfinance as yf
        info = yf.Ticker(etf).fast_info
        prev = getattr(info, "previous_close", None)
        last = getattr(info, "last_price", None)
        if prev and last and prev > 0:
            chg = (last - prev) / prev * 100
            _sector_etf_cache[etf] = chg
            return chg
    except Exception as e:
        logging.warning(f"Sector ETF {etf}: {e}")
    return None


# ── Helpers de badge ──────────────────────────────────────────────────────────

def score_badge(score: float) -> str:
    if score >= 80:  return "🔥"
    if score >= 55:  return "⭐"
    return "📊"


# ── Insider buying & short interest flags ────────────────────────────────────

def get_insider_buy_flag(symbol: str) -> str:
    try:
        import yfinance as yf
        transactions = yf.Ticker(symbol).insider_transactions
        if transactions is None or transactions.empty:
            return ""
        cutoff = datetime.now(LISBON_TZ) - timedelta(days=90)
        recent = transactions[
            (transactions.index >= cutoff) &
            (transactions["Shares"].fillna(0) > 0)
        ]
        if not recent.empty:
            total_shares = int(recent["Shares"].sum())
            return f" | 👤 Insider buy: +{total_shares:,} acções (90d)"
    except Exception as e:
        logging.debug(f"Insider flag {symbol}: {e}")
    return ""

def get_short_interest_flag(fundamentals: dict) -> str:
    short_pct = fundamentals.get("short_percent_of_float") or 0
    if short_pct >= 0.20:
        return f" | ⚠️ Short {short_pct*100:.0f}% float (risco/squeeze)"
    if short_pct >= 0.10:
        return f" | 🔻 Short {short_pct*100:.0f}% float"
    return ""


# ── Telegram ──────────────────────────────────────────────────────────────────

def send_telegram(message: str, retries: int = 2) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(message)
        return True
    chunks = []
    while len(message) > 4000:
        split_at = message.rfind("\n", 0, 4000)
        if split_at == -1: split_at = 4000
        chunks.append(message[:split_at])
        message = message[split_at:].lstrip("\n")
    chunks.append(message)
    url    = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    all_ok = True
    for chunk in chunks:
        sent = False
        for attempt in range(retries + 1):
            try:
                r = requests.post(url, json={
                    "chat_id": TELEGRAM_CHAT_ID, "text": chunk,
                    "parse_mode": "Markdown", "disable_web_page_preview": True,
                }, timeout=10)
                r.raise_for_status()
                sent = True
                break
            except Exception as e:
                logging.error(f"Telegram (tentativa {attempt+1}/{retries+1}): {e}")
                if attempt < retries: time.sleep(3)
        if not sent: all_ok = False
        elif len(chunks) > 1: time.sleep(1)
    return all_ok


# ── Heartbeat das 9h ──────────────────────────────────────────────────────────

def _pnl_emoji(v: float) -> str:
    return "🟢" if v > 0 else ("🔴" if v < 0 else "⚪")

def _get_snapshot() -> dict:
    usd_eur = get_usdeur()
    return get_portfolio_snapshot(
        HOLDINGS, PPR_SHARES, PPR_AVG_COST, usd_eur
    )

def send_heartbeat() -> None:
    logging.info("A gerar heartbeat de carteira...")
    now_lisbon = datetime.now(LISBON_TZ)
    try:
        snapshot = _get_snapshot()
    except Exception as e:
        logging.error(f"Heartbeat snapshot: {e}")
        send_telegram(
            f"🤖 *DipRadar* ativo — {now_lisbon.strftime('%d/%m %H:%M')}\n"
            f"_Erro ao calcular carteira: {e}_"
        )
        return

    total      = snapshot["total_eur"]
    total_cost = snapshot.get("total_cost", 0)
    if total == 0:
        send_telegram(
            f"🤖 *DipRadar — Bom dia!* {now_lisbon.strftime('%d/%m/%Y')}\n"
            f"⚠️ *Variáveis da carteira em falta no Railway.*\n"
            f"_Adiciona HOLDING_*, PPR_SHARES, PPR_AVG_COST nas env vars._"
        )
        return

    pnl_d   = snapshot["pnl_day"]
    pnl_w   = snapshot["pnl_week"]
    pnl_m   = snapshot["pnl_month"]
    pnl_tot = snapshot["pnl_total"]
    fx      = snapshot["usd_eur"]

    def _pct(pnl, base):
        if base and base != 0:
            return f" ({'+' if pnl >= 0 else ''}{pnl / base * 100:.2f}%)"
        return ""

    yesterday = (now_lisbon - timedelta(days=1)).strftime("%d/%m")
    lines = [
        f"🤖 *DipRadar — Bom dia!* {now_lisbon.strftime('%d/%m/%Y')}",
        f"_USD/EUR: {fx:.4f}_",
        "",
        f"*📦 Carteira total: €{total:,.2f}*",
        "",
        f"  {_pnl_emoji(pnl_d)}  *Ontem ({yesterday}):*  €{pnl_d:+,.2f}{_pct(pnl_d, total - pnl_d)}",
        f"  {_pnl_emoji(pnl_w)}  *Semana:*            €{pnl_w:+,.2f}{_pct(pnl_w, total - pnl_w)}",
        f"  {_pnl_emoji(pnl_m)}  *Mês:*               €{pnl_m:+,.2f}{_pct(pnl_m, total - pnl_m)}",
    ]
    # FIX Bug 2: só mostra % se a base (total_cost) for fiável (> 100 €)
    if pnl_tot is not None:
        pnl_pct_str = f" ({pnl_tot / total_cost * 100:+.1f}%)" if total_cost and total_cost > 100 else ""
        lines.append(f"  {_pnl_emoji(pnl_tot)}  *P&L acumulado:*     €{pnl_tot:+,.2f}{pnl_pct_str}")

    movers = sorted(
        [p for p in snapshot["positions"] if p["pnl_day"] is not None],
        key=lambda x: x["pnl_day"], reverse=True,
    )
    if movers:
        lines += ["", f"*📈 Top movers {yesterday}:*"]
        for p in movers[:3]:
            base = p["value_eur"] - p["pnl_day"]
            pct  = p["pnl_day"] / base * 100 if base else 0
            lines.append(f"  {_pnl_emoji(p['pnl_day'])} *{p['symbol']}*: €{p['pnl_day']:+,.2f} ({pct:+.1f}%)")
        worst = movers[-1]
        if worst["pnl_day"] < 0 and worst["symbol"] != movers[0]["symbol"]:
            base = worst["value_eur"] - worst["pnl_day"]
            pct  = worst["pnl_day"] / base * 100 if base else 0
            lines.append(f"  {_pnl_emoji(worst['pnl_day'])} *{worst['symbol']}*: €{worst['pnl_day']:+,.2f} ({pct:+.1f}%) ← pior")

    lines += [
        "",
        f"  📊 PPR: €{snapshot['ppr_value']:,.2f}",
    ]

    if FLIP_FUND_EUR:
        lines.append(f"  🎯 *Flip Fund:* €{FLIP_FUND_EUR:,.2f} disponíveis")

    if total > 0:
        concentration_warnings = []
        for pos in snapshot["positions"]:
            weight = pos["value_eur"] / total * 100
            sym = pos["symbol"]
            if sym == "NVO" and weight >= 28:
                concentration_warnings.append(
                    f"  ⚠️ *NVO*: {weight:.1f}% do portfólio — limite é 30%"
                )
            elif sym == "ADBE" and weight >= 11:
                concentration_warnings.append(
                    f"  ⚠️ *ADBE*: {weight:.1f}% — limite é 10–12%, não reforçar"
                )
            elif weight >= 20 and sym not in ("NVO",):
                concentration_warnings.append(
                    f"  ⚠️ *{sym}*: {weight:.1f}% — posição pesada, monitorizar"
                )
        eunl_val = next((p["value_eur"] for p in snapshot["positions"] if "EUNL" in p["symbol"]), 0)
        core_pct = (eunl_val + snapshot["ppr_value"]) / total * 100
        if core_pct < 35:
            concentration_warnings.append(
                f"  ⚠️ *Core (EUNL+PPR)*: {core_pct:.1f}% — abaixo do mínimo de 35%"
            )
        if concentration_warnings:
            lines += ["", "*🚨 Alertas de concentração:*"] + concentration_warnings

    earnings_alerts = []
    for sym, shares, _ in HOLDINGS:
        if not shares:
            continue
        # FIX Bug 1: ETFs não têm earnings — evita 404 no yfinance (ex: EUNL.DE)
        if is_etf(sym):
            continue
        clean_sym = sym.split(".")[0]
        try:
            days = get_earnings_days(clean_sym)
            if days is not None:
                dt_str = (now_lisbon + timedelta(days=days)).strftime("%d/%m")
                urgency = "🔴" if days <= 7 else "🟡" if days <= 21 else "📅"
                earnings_alerts.append(f"  {urgency} *{clean_sym}*: earnings em {days}d ({dt_str})")
        except Exception:
            pass

    if earnings_alerts:
        lines += ["", "*📅 Earnings próximos (carteira):*"] + sorted(earnings_alerts)

    # Semáforo macro
    try:
        mc_ctx = get_macro_context()
        if mc_ctx.get("summary"):
            lines += ["", mc_ctx["summary"]]
            if mc_ctx.get("alert"):
                lines.append("_⚠️ Condições macro adversas — redobrar cautela nas entradas_")
    except Exception as e:
        logging.warning(f"Heartbeat macro context: {e}")

    # Indicador Tiingo + ML
    tiingo_ok = is_tiingo_available()
    ml_ok     = is_model_ready()
    lines += [
        "",
        f"_{'🟢 Tiingo EOD activo' if tiingo_ok else '🟡 yfinance (sem chave Tiingo)'}_",
        f"_{'🤖 Modelo ML pronto' if ml_ok else '🤖 ML: modelo não treinado'}_",
        "_Mercado abre às 14h30 Lisboa_",
    ]

    if WATCHLIST_ENABLED and WATCHLIST:
        try:
            wl_block = build_watchlist_morning_summary(DIRECT_TICKERS)
            lines += ["", "─" * 25, "", wl_block]
        except Exception as e:
            logging.warning(f"Watchlist morning summary: {e}")

    send_telegram("\n".join(lines))
    logging.info("Heartbeat enviado.")


# ── Portfolio Stress Alert ────────────────────────────────────────────────────

def check_portfolio_stress() -> None:
    global _stress_alerted
    try:
        snapshot = _get_snapshot()
    except Exception as e:
        logging.warning(f"Portfolio stress snapshot: {e}")
        return

    positions = snapshot.get("positions", [])
    total_eur = snapshot.get("total_eur", 0)
    pnl_day   = snapshot.get("pnl_day", 0)

    for p in positions:
        sym   = p["symbol"]
        pnl_d = p.get("pnl_day")
        val   = p.get("value_eur", 0)
        if pnl_d is None or val == 0 or sym in _stress_alerted:
            continue
        base = val - pnl_d
        if base <= 0:
            continue
        pct_d = pnl_d / base * 100
        if pct_d <= -STRESS_PCT:
            in_portfolio = " 📦" if sym in DIRECT_TICKERS else ""
            send_telegram(
                f"🚨 *Alerta de stress: {sym}*{in_portfolio}\n"
                f"Queda de *{pct_d:.1f}%* no dia de hoje\n"
                f"P&L dia: €{pnl_d:+,.2f} | Valor actual: €{val:,.2f}\n"
                f"_⏰ {datetime.now(LISBON_TZ).strftime('%d/%m %H:%M')}_"
            )
            _stress_alerted.add(sym)
            logging.warning(f"Portfolio stress: {sym} {pct_d:.1f}%")

    if total_eur > 0 and pnl_day is not None:
        base_total = total_eur - pnl_day
        if base_total > 0:
            pct_total = pnl_day / base_total * 100
            if pct_total <= -3.0 and "_TOTAL_" not in _stress_alerted:
                _stress_alerted.add("_TOTAL_")
                spy_change = get_spy_change()
                spy_str    = f" | SPY: {spy_change:+.1f}%" if spy_change is not None else ""
                send_telegram(
                    f"🚨 *Carteira em stress macro*\n"
                    f"Queda de *{pct_total:.1f}%* no dia de hoje{spy_str}\n"
                    f"P&L dia: €{pnl_day:+,.2f} | Total: €{total_eur:,.2f}\n"
                    f"_⏰ {datetime.now(LISBON_TZ).strftime('%d/%m %H:%M')}_"
                )
                logging.warning(f"Portfolio stress MACRO: {pct_total:.1f}%")


# ── Thesis Degradation Check (EOD-aware) ──────────────────────────────────────

def check_thesis_degradation(region: str = "ALL") -> None:
    """
    Verifica degradação de tese usando preços EOD limpos (Tiingo/yfinance).
    Chamado em boleia pelo scan europeu (17h45) e americano (21h15).

    region: "EU" | "US" | "ALL"
      EU  — tickers com sufixo de exchange europeia (.DE, .PA, .L, etc.)
      US  — tickers sem sufixo (mercado americano)
      ALL — todos
    """
    positions = get_positions()
    if not positions:
        logging.info("[degradation] Sem posições activas.")
        return

    _EU_SUFFIXES = (".DE", ".PA", ".AS", ".MC", ".MI", ".L", ".SW",
                   ".ST", ".CO", ".OL", ".HE", ".BR", ".LS", ".VI", ".WA", ".I")

    def _is_eu(ticker: str) -> bool:
        return any(ticker.upper().endswith(s) for s in _EU_SUFFIXES)

    to_check = {
        sym: pos for sym, pos in positions.items()
        if (
            pos.get("entry_score") is not None
            and pos.get("category", "") != "ETF"
            and (region == "ALL"
                 or (region == "EU" and _is_eu(sym))
                 or (region == "US" and not _is_eu(sym)))
        )
    }

    if not to_check:
        logging.info(f"[degradation] Sem posições para região '{region}'.")
        return

    logging.info(f"[degradation] A verificar {len(to_check)} posições ({region})...")

    for sym, pos in to_check.items():
        entry_score = pos["entry_score"]
        try:
            time.sleep(2)
            price = get_latest_price(sym) or 0

            fund = get_fundamentals(sym, min_market_cap=0)
            if fund.get("skip"):
                continue
            if price:
                fund["price"] = price

            earnings_days  = get_earnings_days(sym)
            sector_chg     = get_sector_change(fund.get("sector", ""))
            _ml_feat_deg   = {
                **fund,
                "spy_change":        get_spy_change(),
                "sector_etf_change": sector_chg,
                "earnings_days":     earnings_days,
            }
            _ml_res_deg  = ml_score(_ml_feat_deg, symbol=sym)
            _ml_prob_deg = _ml_res_deg.win_prob if _ml_res_deg.model_ready else None
            score, _      = calculate_dip_score(fund, sym, earnings_days, sector_change=sector_chg, ml_prob=_ml_prob_deg)
            category      = pos.get("category", "")

            update_position_data(sym, price=price, score=score, category=category)

            drop            = entry_score - score
            already_alerted = pos.get("degradation_alerted", False)

            if already_alerted and score >= entry_score - 5:
                reset_degradation_flag(sym)
                send_telegram(
                    f"♻️ *Tese recuperada: {sym}*\n"
                    f"Score actual: *{score:.0f}/100* — voltou perto do entry score ({entry_score}/100)\n"
                    f"_Flag de degradação limpa — posição estabilizada_\n"
                    f"_⏰ {datetime.now(LISBON_TZ).strftime('%d/%m %H:%M')}_"
                )
                continue

            if drop >= DEGRADATION_DROP_THRESHOLD and not already_alerted:
                entry_date = pos.get("entry_date", "N/D")
                avg_price  = pos.get("avg_price", 0)
                pnl_pct    = (price - avg_price) / avg_price * 100 if avg_price else 0
                sector_cfg = get_sector_config(fund.get("sector", ""))
                score_breakdown = build_score_breakdown(fund, sym, earnings_days, sector_change=sector_chg)

                lines = [
                    f"🧨 *Degradação de Tese: {sym}*",
                    f"_Posição em carteira desde {entry_date}_",
                    f"_Dados EOD — mercado já fechado_",
                    "",
                    f"  📉 Score entry: *{entry_score}/100*",
                    f"  📊 Score actual: *{score:.0f}/100*",
                    f"  🔻 Drop: *-{drop:.0f} pts*",
                    "",
                    f"  💰 Preço médio: ${avg_price:.2f} | Fecho EOD: ${price:.2f}",
                    f"  {_pnl_emoji(pnl_pct)} P&L posição: {pnl_pct:+.1f}%",
                    f"  🏷️ Categoria: {category} | {sector_cfg.get('label', '')}",
                    "",
                    score_breakdown,
                    "",
                    f"*⚠️ Reavalia a tese de investimento.*",
                    f"_Preço de fecho limpo — considera a tua ordem para amanhã_",
                    f"_⏰ {datetime.now(LISBON_TZ).strftime('%d/%m %H:%M')}_",
                ]
                send_telegram("\n".join(lines))
                mark_degradation_alerted(sym)
                logging.warning(f"[degradation] {sym}: drop {drop:.0f}pts (entry={entry_score} actual={score:.0f})")

        except Exception as e:
            logging.error(f"[degradation] Erro em {sym}: {e}", exc_info=True)

    logging.info(f"[degradation] Check ({region}) concluído.")


# ── Recovery Alert ────────────────────────────────────────────────────────────

def check_recovery_alerts() -> None:
    import yfinance as yf
    positions = load_recovery_watch()
    if not positions:
        return

    try:
        stale = get_stale_recovery_positions(days=60)
        for p in stale:
            sym          = p["symbol"]
            in_portfolio = " 📦" if sym in DIRECT_TICKERS else ""
            days_held    = (datetime.now(LISBON_TZ) - datetime.fromisoformat(p["date_iso"]).replace(tzinfo=LISBON_TZ)).days
            send_telegram(
                f"⏹️ *Recovery Watch — Stop Temporal: {sym}*{in_portfolio}\n"
                f"*{days_held} dias* sem atingir o target de recuperação.\n"
                f"Entrada: {p['date']} @ *${p['price_alert']:.2f}*\n"
                f"Target: *+{p['target_pct']}%* (${p['target_price']:.2f})\n"
                f"Score original: {p.get('score', 'N/A')}/100\n"
                f"_Considera fechar a posição ou manter manualmente._\n"
                f"_⏰ {datetime.now(LISBON_TZ).strftime('%d/%m %H:%M')}_"
            )
            mark_stale_alerted(sym)
    except Exception as e:
        logging.warning(f"[recovery] Stale check: {e}")

    for pos in positions:
        if pos.get("alerted"):
            continue
        sym          = pos["symbol"]
        target_price = pos.get("target_price", 0)
        price_alert  = pos.get("price_alert", 0)
        try:
            time.sleep(2)
            current = yf.Ticker(sym).info.get("regularMarketPrice") or 0
            if current and current >= target_price:
                pct_recovery = (current - price_alert) / price_alert * 100
                in_portfolio = " 📦" if sym in DIRECT_TICKERS else ""
                send_telegram(
                    f"🔔 *Recovery Alert: {sym}*{in_portfolio}\n"
                    f"Preço actual: *${current:.2f}* | Alerta foi a *${price_alert:.2f}*\n"
                    f"Recuperação: *+{pct_recovery:.1f}%* ✅\n"
                    f"_Score original: {pos.get('score', 'N/A')}/100 | Alerta em {pos.get('date', '')}_\n"
                    f"_⏰ {datetime.now(LISBON_TZ).strftime('%d/%m %H:%M')}_"
                )
                mark_recovery_alerted(sym)
        except Exception as e:
            logging.warning(f"Recovery check {sym}: {e}")


# ── Weekly structural dip scan ────────────────────────────────────────────────

def send_weekly_dip_scan() -> None:
    if datetime.now(LISBON_TZ).weekday() != 0:
        return
    logging.info("A correr weekly structural dip scan...")
    candidates = screen_structural_dips(min_drawdown_pct=25.0, min_market_cap=MIN_MARKET_CAP)
    if not candidates:
        return
    scored = []
    for stock in candidates[:40]:
        sym = stock["symbol"]
        try:
            fund = get_fundamentals(sym, min_market_cap=MIN_MARKET_CAP)
            if fund.get("skip"): continue
            earnings_days  = get_earnings_days(sym)
            sector_chg     = get_sector_change(fund.get("sector", ""))
            _ml_feat_wk    = {
                **fund,
                "spy_change":        None,
                "sector_etf_change": sector_chg,
                "earnings_days":     earnings_days,
            }
            _ml_res_wk   = ml_score(_ml_feat_wk, symbol=sym)
            _ml_prob_wk  = _ml_res_wk.win_prob if _ml_res_wk.model_ready else None
            score, rsi_str = calculate_dip_score(fund, sym, earnings_days, sector_change=sector_chg, ml_prob=_ml_prob_wk)
            if score >= 70:
                bc_flag  = is_bluechip(fund)
                category = classify_dip_category(fund, score, bc_flag)
                scored.append({
                    "symbol":         sym,
                    "score":          score,
                    "drawdown":       stock["drawdown_52w"],
                    "price":          fund.get("price") or stock["price"],
                    "mc_b":           (fund.get("market_cap") or 0) / 1e9,
                    "sector":         get_sector_config(fund.get("sector", "")).get("label", "") or fund.get("sector", ""),
                    "rsi":            rsi_str,
                    "bluechip":       bc_flag,
                    "category":       category,
                    "earnings_days":  earnings_days,
                    "analyst_upside": fund.get("analyst_upside") or 0,
                })
            time.sleep(5)
        except Exception as e:
            logging.warning(f"Weekly scan {sym}: {e}")
    if not scored:
        return
    scored.sort(key=lambda x: x["score"], reverse=True)
    lines = [
        f"*📶 Weekly Structural Dip Scan — {datetime.now(LISBON_TZ).strftime('%d/%m/%Y')}*",
        f"_Stocks ≥25% abaixo dos máximos de 52 semanas com score ≥70/100_",
        "",
    ]
    for s in scored[:12]:
        badge      = score_badge(s["score"])
        bc_tag     = " 💎" if s["bluechip"] else ""
        rsi_tag    = f" | RSI {s['rsi']}" if s["rsi"] else ""
        earn_tag   = f" | 📅 Earnings em {s['earnings_days']}d" if s["earnings_days"] is not None else ""
        upside_tag = f" | 📡 +{s['analyst_upside']:.0f}% analistas" if s["analyst_upside"] > 20 else ""
        lines.append(
            f"{badge} *{s['symbol']}*{bc_tag} — Score {s['score']:.0f}/100 | "
            f"${s['price']:.2f} | ${s['mc_b']:.1f}B | {s['drawdown']:.0f}% do topo"
        )
        lines.append(f"   _{s['category']} | {s['sector']}{rsi_tag}{earn_tag}{upside_tag}_")
        lines.append("")
    send_telegram("\n".join(lines))


# ── Saturday Weekly Report ────────────────────────────────────────────────────

def send_saturday_report() -> None:
    if datetime.now(LISBON_TZ).weekday() != 5:
        return
    logging.info("A gerar Saturday Weekly Report...")
    entries    = load_weekly_log()
    now        = datetime.now(LISBON_TZ)
    week_start = (now - timedelta(days=now.weekday() + 1)).strftime("%d/%m")
    week_end   = (now - timedelta(days=1)).strftime("%d/%m")
    lines      = [f"*📊 Weekly Report — {week_start} a {week_end}*", ""]

    if not entries:
        lines.append("_Sem alertas diários esta semana._")
    else:
        by_date: dict[str, list] = {}
        for e in entries:
            by_date.setdefault(e["date"], []).append(e)
        total_alerts  = len(entries)
        comprar_count = sum(1 for e in entries if e["verdict"] == "COMPRAR")
        avg_score     = sum(e["score"] for e in entries) / total_alerts
        best          = max(entries, key=lambda x: x["score"])
        lines += [
            "*📋 Alertas da semana:*",
            f"  Total: *{total_alerts}* | COMPRAR: *{comprar_count}* | MONITORIZAR: *{total_alerts - comprar_count}*",
            f"  Score médio: *{avg_score:.1f}/100* | Melhor: *{best['symbol']}* (score {best['score']:.0f}, {best['date']})",
            "", "*🗓️ Por dia:*",
        ]
        for date_str in sorted(by_date.keys()):
            day_entries = by_date[date_str]
            lines.append(f"  *{date_str}* — {len(day_entries)} alerta(s)")
            for e in sorted(day_entries, key=lambda x: x["score"], reverse=True):
                badge = "🟢" if e["verdict"] == "COMPRAR" else "🟡"
                lines.append(
                    f"    {badge} *{e['symbol']}* Score {e['score']:.0f}/100 | "
                    f"{e['change']:+.1f}% | _{e['sector']}_ | {e.get('time','')}"
                )
        lines.append("")
        top3 = sorted(entries, key=lambda x: x["score"], reverse=True)[:3]
        lines.append("*🏆 Top 3 da semana:*")
        for i, e in enumerate(top3, 1):
            lines.append(f"  {i}. *{e['symbol']}* — Score {e['score']:.0f}/100 | {e['verdict']} | {e['date']}")
        lines.append("")

    try:
        bt_block = build_backtest_summary()
        lines += ["─" * 30, "", bt_block, ""]
    except Exception as e:
        logging.warning(f"Backtest block: {e}")

    rejected = load_rejected_log()
    if rejected:
        lines += [
            "*🗑️ Rejeitados hoje:*",
            "_Stocks analisados que não cumpriram os critérios_",
            "",
        ]
        for r in sorted(rejected, key=lambda x: x.get("score") or 0, reverse=True)[:10]:
            score_str   = f" | Score {r['score']:.0f}/100" if r.get("score") is not None else ""
            verdict_str = f" | {r['verdict']}" if r.get("verdict") else ""
            lines.append(
                f"  ⛔ *{r['symbol']}* {r['change']:+.1f}% | "
                f"_{r['reason']}{score_str}{verdict_str}_"
            )
        lines.append("")

    lines += ["─" * 30, "",
              "*📉 Weekly Dips — quedas ≥10% nos últimos 7 dias:*",
              "_Top 5 com score; restantes só listados_", ""]
    try:
        period_dips = screen_period_dips(
            min_market_cap=MIN_MARKET_CAP, week_threshold=10.0, month_threshold=15.0,
        )
        weekly  = period_dips.get("weekly", [])
        monthly = period_dips.get("monthly", [])

        def _score_list(stocks, label):
            for i, s in enumerate(stocks[:10]):
                sym          = s["symbol"]
                mc_b         = (s.get("market_cap") or 0) / 1e9
                in_portfolio = " 📦" if sym in DIRECT_TICKERS else ""
                if i < 5:
                    try:
                        fund = get_fundamentals(sym, min_market_cap=MIN_MARKET_CAP)
                        if not fund.get("skip"):
                            ed        = get_earnings_days(sym)
                            sec_chg   = get_sector_change(fund.get("sector", ""))
                            score, _  = calculate_dip_score(fund, sym, ed, sector_change=sec_chg)
                            badge     = score_badge(score)
                            lines.append(
                                f"  {badge} *{sym}*{in_portfolio}: *{s['change_pct']:.1f}%* ({label}) "
                                f"| Score {score:.0f}/100 | ${s['price']} | ${mc_b:.1f}B"
                            )
                        else:
                            lines.append(f"  📉 *{sym}*{in_portfolio}: *{s['change_pct']:.1f}%* ({label}) | ${s['price']} | ${mc_b:.1f}B")
                        time.sleep(4)
                    except Exception as ex:
                        logging.warning(f"Period score {sym}: {ex}")
                        lines.append(f"  📉 *{sym}*{in_portfolio}: *{s['change_pct']:.1f}%* ({label}) | ${s['price']} | ${mc_b:.1f}B")
                else:
                    lines.append(f"  • *{sym}*{in_portfolio}: {s['change_pct']:.1f}% | ${s['price']} | ${mc_b:.1f}B")

        if weekly:  _score_list(weekly, "7d")
        else:       lines.append("  _Nenhum stock com queda ≥10% na semana._")

        lines += ["", "*📉 Monthly Dips — quedas ≥15% no último mês:*",
                  "_Excluídos stocks já no weekly_", ""]
        if monthly: _score_list(monthly, "30d")
        else:       lines.append("  _Nenhum stock com queda ≥15% no mês._")

    except Exception as e:
        logging.error(f"Saturday period dips: {e}")
        lines.append(f"  _Erro: {e}_")

    save_weekly_log([])
    lines += ["", "_⏰ Próximo report: sábado às 10h_"]
    send_telegram("\n".join(lines))


# ── Target Sell / Flip ────────────────────────────────────────────────────────

_SECTOR_FLIP_CAP = {
    "Technology":             0.55,
    "Communication Services": 0.45,
    "Consumer Cyclical":      0.40,
    "Healthcare":             0.35,
    "Industrials":            0.30,
    "Basic Materials":        0.30,
    "Energy":                 0.28,
    "Real Estate":            0.25,
    "Financial Services":     0.22,
    "Consumer Defensive":     0.18,
    "Utilities":              0.15,
}

def calculate_flip_target(
    fundamentals: dict,
    dip_score: float,
    earnings_date: str | None = None,
    catalyst: dict | None = None,
    spy_change: float | None = None,
    category: str | None = None,
) -> tuple[str, str]:
    price = fundamentals.get("price") or 0
    if not price or price <= 0:
        return "N/D", "SEM DADOS"

    if category == CATEGORY_HOLD_FOREVER or (is_bluechip(fundamentals) and dip_score >= 60):
        return "HOLD ETERNO", "💎 Hold Forever — Acumular em dips, nunca vender"

    sector         = fundamentals.get("sector", "")
    sector_cfg     = get_sector_config(sector)
    pe_current     = fundamentals.get("pe") or 0
    pe_fair        = sector_cfg.get("pe_fair", 22)
    analyst_target = fundamentals.get("analyst_target")
    analyst_upside = fundamentals.get("analyst_upside") or 0
    beta           = fundamentals.get("beta") or 1.0
    anchors = []
    if pe_current and pe_current > 0 and pe_fair and pe_current < pe_fair:
        anchors.append(("PE rerating", price * (pe_fair / pe_current)))
    if analyst_target and analyst_target > price:
        anchors.append(("Analistas", float(analyst_target)))
    anchors.append(("Beta recovery", price * (1 + (beta or 1.0) * 0.12)))
    weights         = {"PE rerating": 2, "Analistas": 2, "Beta recovery": 1}
    total_w         = sum(weights[n] for n, _ in anchors)
    weighted_price  = sum(weights[n] * t for n, t in anchors) / total_w
    weighted_upside = (weighted_price / price) - 1
    sector_cap      = _SECTOR_FLIP_CAP.get(sector, 0.30)
    if dip_score >= 90:   sector_cap *= 1.20
    elif dip_score >= 75: sector_cap *= 1.10
    final_upside = min(weighted_upside, sector_cap)
    final_target = price * (1 + final_upside)
    macro_flag   = f" | 🌍 Queda macro SPY {spy_change:.1f}% — timeline incerta" if spy_change and spy_change <= -2.0 else ""
    if catalyst and catalyst.get("found"):
        snippet  = f": _{catalyst['snippet']}_" if catalyst.get("snippet") else ""
        cat_flag = f" | {catalyst['label']}{snippet}"
    elif earnings_date:
        cat_flag = f" | ✅ Earnings {earnings_date}"
    elif analyst_upside and analyst_upside > 20:
        cat_flag = " | 📡 Analistas veem upside forte"
    else:
        cat_flag = " | ⚠️ Sem catalisador identificado"

    if category == CATEGORY_APARTAMENTO:
        dividend_yield = fundamentals.get("dividend_yield") or 0
        dy_str = f" | 💰 Yield {dividend_yield*100:.1f}%/ano" if dividend_yield > 0 else ""
        strategy = f"🏠 Apartamento: ${final_target:.1f} (+{final_upside*100:.0f}%){dy_str}{cat_flag}{macro_flag}"
    else:
        strategy = f"🔄 Flip: ${final_target:.1f} (+{final_upside*100:.0f}%){cat_flag}{macro_flag}"

    return f"${final_target:.1f} (+{final_upside*100:.0f}%)", strategy


# ── Ranking Flip ──────────────────────────────────────────────────────────────

def build_flip_ranking(ranked_entries: list[dict], spy_change: float | None, exclude_syms: set | None = None) -> str:
    if not ranked_entries:
        return ""
    entries = [e for e in ranked_entries if not (exclude_syms and e["symbol"] in exclude_syms)]
    if not entries:
        return ""
    lines = ["", "*🏆 RANKING FLIP — Top compras de hoje*"]
    if spy_change is not None:
        sign = "+" if spy_change >= 0 else ""
        lines.append(f"_SPY hoje: {sign}{spy_change:.1f}%_")
    lines.append("")
    top = sorted(entries, key=lambda x: x["dip_score"], reverse=True)[:8]
    for i, entry in enumerate(top, 1):
        sym           = entry["symbol"]
        s             = entry["dip_score"]
        tier          = entry["tier"]
        f             = entry["fundamentals"]
        earnings      = entry.get("earnings_date")
        earnings_days = entry.get("earnings_days")
        catalyst      = entry.get("catalyst")
        category      = entry.get("category", CATEGORY_ROTACAO)
        price         = f.get("price", 0)
        mc_b          = (f.get("market_cap") or 0) / 1e9
        beta          = f.get("beta")
        in_portfolio  = " 📦" if sym in DIRECT_TICKERS else ""
        _, strategy   = calculate_flip_target(f, s, earnings, catalyst, spy_change, category=category)
        badge         = score_badge(s)
        tier_badge    = {1: "🔴T1", 2: "🟡T2", 3: "🔵T3"}.get(tier, "")
        _, sizing_str = suggest_position_size(
            score=s, beta=beta, earnings_days=earnings_days, spy_change=spy_change,
        )
        lines.append(f"*{i}. {sym}*{in_portfolio} {tier_badge} | Score {s:.0f}/100 {badge} | {category}")
        lines.append(f"   💰 ${price} | 🏦 ${mc_b:.1f}B")
        lines.append(f"   {strategy}")
        if sizing_str:
            lines.append(f"   💶 *Sizing:* {sizing_str}")
        lines.append("")
    return "\n".join(lines)


# ── Alerta individual ─────────────────────────────────────────────────────────

def build_alert(
    stock, fundamentals, historical_pe, news,
    verdict, emoji, reasons, dip_score, rsi_str,
    category: str = CATEGORY_ROTACAO,
    ml_result: MLResult | None = None,
) -> str:
    sector       = fundamentals.get("sector", "")
    sector_cfg   = get_sector_config(sector)
    name         = fundamentals.get("name", stock.get("name", stock.get("symbol", "N/A")))
    symbol       = stock["symbol"]
    change       = stock["change_pct"]
    price        = fundamentals.get("price") or stock.get("price", "N/D")
    mc_b         = (fundamentals.get("market_cap") or 0) / 1e9
    drawdown     = fundamentals.get("drawdown_from_high")
    drawdown_str = f" | 52w: {drawdown:.0f}%" if drawdown is not None else ""
    region_part  = f" ({stock['region']})" if stock.get("region") else ""
    in_portfolio = " 📦 *Já em carteira*" if symbol in DIRECT_TICKERS else ""

    badge_str  = score_badge(dip_score)
    score_line = f"{badge_str} Score: {dip_score:.0f}/100"

    vol          = fundamentals.get("volume") or 0
    avg_vol      = fundamentals.get("average_volume") or 0
    vol_flag     = " | 📈 Volume spike" if vol and avg_vol and vol > avg_vol * 1.5 else ""
    insider_flag = get_insider_buy_flag(symbol)
    short_flag   = get_short_interest_flag(fundamentals)

    sector_chg  = get_sector_change(sector)
    sector_warn = f" | 🟠 Sector {sector_chg:+.1f}% (rotation?)" if sector_chg is not None and sector_chg <= -2.0 else ""

    rsi_part = f" | RSI: {rsi_str}" if rsi_str else ""

    # FIX: pass sector as third argument (required by format_valuation_block)
    valuation_block = format_valuation_block(fundamentals, historical_pe, sector)

    spy_change   = get_spy_change()
    earnings_d   = get_earnings_days(symbol)
    earn_warn    = f"\n⚠️ *Earnings em {earnings_d} dias!* — Cautela" if earnings_d is not None and earnings_d < 14 else ""
    catalyst_data = None
    try:
        catalyst_data = get_catalyst(symbol, name)
    except Exception:
        pass
    _, strategy = calculate_flip_target(
        fundamentals, dip_score,
        earnings_date=get_earnings_date(symbol),
        catalyst=catalyst_data,
        spy_change=spy_change,
        category=category,
    )

    macro_ctx = ""
    try:
        mc = get_macro_context()
        if mc.get("alert"):
            macro_ctx = f"\n🌎 *Macro:* {mc['summary']}"
    except Exception:
        pass

    ml_label = ml_result.label if (ml_result is not None) else None
    score_breakdown = build_score_breakdown(
        fundamentals, symbol, earnings_d,
        sector_change=sector_chg,
        ml_label=ml_label,
    )

    _, sizing_str = suggest_position_size(
        score=dip_score,
        beta=fundamentals.get("beta"),
        earnings_days=earnings_d,
        spy_change=spy_change,
    )

    # ML badge (Chunk 6)
    ml_line = ""
    if ml_result is not None:
        ml_line = f"\n{ml_badge(ml_result)}"

    body = (
        f"{emoji} *{name} ({symbol})*{region_part}{in_portfolio}\n"
        f"Sector: {sector_cfg.get('label', sector)} | ≈{mc_b:.1f}B\n"
        f"Hoje: *{change:+.1f}%* — ${price}{drawdown_str}{vol_flag}{insider_flag}{short_flag}{rsi_part}{sector_warn}\n"
        f"{earn_warn}"
        f"{valuation_block}"
        f"\n*Categoria:* {category}\n"
        f"{score_line}"
        f"{ml_line}\n"
        f"{score_breakdown}\n"
        f"\n*Estratégia:* {strategy}"
        f"{macro_ctx}"
    )
    if sizing_str:
        body += f"\n💶 *Sizing:* {sizing_str}"
    return body


# ── On-demand ticker analysis (usado pelo /analisar command) ──────────────────

def analyze_ticker(symbol: str) -> str:
    symbol = symbol.upper().strip()
    try:
        fund = get_fundamentals(symbol, min_market_cap=0)
        if fund.get("skip"):
            return (
                f"⚠️ *{symbol}* — dados insuficientes\n"
                f"_{fund.get('skip_reason', 'Ticker sem dados disponíveis')}_"
            )

        hist_pe       = get_historical_pe(symbol)
        news          = get_news(symbol)
        earnings_days = get_earnings_days(symbol)
        sector_chg    = get_sector_change(fund.get("sector", ""))
        spy_change    = get_spy_change()
        score, rsi_str = calculate_dip_score(fund, symbol, earnings_days, sector_change=sector_chg)

        bc_flag  = is_bluechip(fund)
        category = classify_dip_category(fund, score, bc_flag)

        price      = fund.get("price") or 0
        change_pct = 0.0
        try:
            import yfinance as yf
            fi = yf.Ticker(symbol).fast_info
            prev = getattr(fi, "previous_close", None)
            last = getattr(fi, "last_price", None)
            if prev and last and prev > 0:
                change_pct = (last - prev) / prev * 100
        except Exception:
            pass

        # ML score (Chunk 6)
        ml_features = {
            **fund,
            "score":            score,
            "spy_change":       spy_change,
            "sector_etf_change": sector_chg,
            "earnings_days":    earnings_days,
            "change_day_pct":   change_pct,
        }
        ml_result = ml_score(ml_features, symbol=symbol)

        # ── Score V2 Engine (Chunk 9) ──────────────────────────────
        ml_prob_v2 = ml_result.win_prob if ml_result.model_ready else None
        score_data = score_from_fundamentals(fund, ml_prob=ml_prob_v2)
        score      = score_data["final_score"]
        v2_note    = format_score_v2_breakdown(score_data)
        logging.info(
            f"[v2/analisar] {symbol}: {score:.1f} | "
            f"skip={score_data.get('skip_recommended')} "
            f"trap={score_data.get('is_value_trap')}"
        )

        stock_dict = {"symbol": symbol, "change_pct": change_pct, "region": ""}

        if score >= 75:
            verdict, emoji_str = "COMPRAR", "🟢"
        elif score >= 60:
            verdict, emoji_str = "MONITORIZAR", "🟡"
        elif score >= MIN_DIP_SCORE:
            verdict, emoji_str = "MONITORIZAR", "🔵"
        else:
            verdict, emoji_str = "EVITAR", "🔴"

        alert_text = build_alert(
            stock_dict, fund, hist_pe, news,
            verdict, emoji_str, [], score, rsi_str,
            category=category,
            ml_result=ml_result,
        )

        header = f"🔍 *Análise on-demand — {symbol}*\n_Pedido via /analisar — {datetime.now(LISBON_TZ).strftime('%d/%m %H:%M')}_\n\n"
        return header + alert_text

    except Exception as e:
        logging.error(f"analyze_ticker({symbol}): {e}", exc_info=True)
        return f"❌ *Erro ao analisar {symbol}*\n_`{e}`_"


# ── Allocation engine (read-only, Fase 1) ─────────────────────────────────────
#
# Orquestra fundamentals + dip score + ML + macro regime + liquidez e invoca
# o motor `allocation_engine.suggest_allocation`. Devolve uma string formatada
# para Telegram (resposta do /allocate).
#
# Esta função é puramente informativa — nenhuma ordem é executada.

def allocate_ticker(symbol: str) -> str:
    from allocation_engine import (
        AllocationContext,
        suggest_allocation,
        format_allocation_telegram,
    )
    from macro_semaphore import get_macro_regime
    from portfolio import get_liquidity, get_position_pct

    symbol = symbol.upper().strip()
    try:
        # 1) Fundamentals + dip score (pipeline existente do /analisar)
        fund = get_fundamentals(symbol, min_market_cap=0)
        if fund.get("skip"):
            return (
                f"⚠️ *{symbol}* — sem dados\n"
                f"_{fund.get('skip_reason', 'Ticker indisponível')}_"
            )

        earnings_days = get_earnings_days(symbol)
        sector_chg    = get_sector_change(fund.get("sector", ""))
        spy_change    = get_spy_change()
        score, _      = calculate_dip_score(fund, symbol, earnings_days, sector_change=sector_chg)

        # Score V2 com fund_only_score + is_preprofit (Fases 1+4)
        try:
            from score import score_from_fundamentals
            v2_result      = score_from_fundamentals(fund, earnings_days=earnings_days)
            fund_only_v2   = float(v2_result.get("fund_only_score", v2_result.get("final_score", score)))
            is_preprofit_v = bool(v2_result.get("is_preprofit", False))
        except Exception as e:
            logging.warning(f"[allocate] score_from_fundamentals falhou para {symbol}: {e}")
            fund_only_v2, is_preprofit_v = float(score or 0.0), False

        bc_flag       = is_bluechip(fund)
        category_str  = classify_dip_category(fund, score, bc_flag)

        # 2) Preço actual + change_day_pct (mesma lógica do analyze_ticker)
        price      = fund.get("price") or 0.0
        change_pct = 0.0
        try:
            import yfinance as yf
            fi   = yf.Ticker(symbol).fast_info
            prev = getattr(fi, "previous_close", None)
            last = getattr(fi, "last_price", None)
            if prev and last and prev > 0:
                change_pct = (last - prev) / prev * 100
                price      = last
        except Exception:
            pass

        # 3) ML score
        ml_features = {
            **fund,
            "score":            score,
            "spy_change":       spy_change,
            "sector_etf_change": sector_chg,
            "earnings_days":    earnings_days,
            "change_day_pct":   change_pct,
        }
        ml_result = ml_score(ml_features, symbol=symbol)

        # 4) Macro regime (4 indicadores: SPY, VIX, credit, yield curve)
        try:
            regime = get_macro_regime()
            regime_color = regime.get("color", "GREEN")
            regime_mult  = float(regime.get("position_multiplier", 1.0))
        except Exception as e:
            logging.warning(f"[allocate] regime macro indisponível: {e}")
            regime_color, regime_mult = "GREEN", 1.0

        # 5) Liquidez disponível
        try:
            cash_eur = float(get_liquidity() or 0.0)
        except Exception:
            cash_eur = 0.0

        # 5b) Concentração actual deste ticker no portfolio (cost basis EUR)
        try:
            existing_pct = float(get_position_pct(symbol, usd_eur=get_usdeur()))
        except Exception as e:
            logging.warning(f"[allocate] get_position_pct falhou para {symbol}: {e}")
            existing_pct = 0.0

        # 6) Drawdown 52w (em fundamentals está em pontos percentuais — converter)
        dd_pct       = fund.get("drawdown_from_high")
        drawdown_52w = (float(dd_pct) / 100.0) if dd_pct is not None else None

        # 7) Construir contexto e correr motor (Fase 4: usa fund_only_score V2)
        ctx = AllocationContext(
            ticker                = symbol,
            fund_score            = fund_only_v2,
            is_preprofit          = is_preprofit_v,
            dip_score             = float(score or 0.0),  # backward compat
            is_etf                = is_etf(symbol),
            is_bluechip           = bool(bc_flag),
            sector                = fund.get("sector", "") or "",
            drawdown_52w          = drawdown_52w,
            dividend_yield        = fund.get("dividend_yield"),
            classify_category     = category_str,
            pred_up               = ml_result.pred_up if ml_result.model_ready else None,
            pred_down             = ml_result.pred_down if ml_result.model_ready else None,
            win_prob              = ml_result.win_prob if ml_result.model_ready else None,
            ml_label              = ml_result.label,
            model_ready           = ml_result.model_ready,
            macro_regime_color    = regime_color,
            macro_multiplier      = regime_mult,
            cash_available_eur    = cash_eur,
            monthly_budget_eur    = float(os.environ.get("MONTHLY_BUDGET_EUR", "1050")),
            existing_position_pct = existing_pct,
        )
        decision = suggest_allocation(ctx)

        # 8) Formatar resposta Telegram
        header = (
            f"💼 *Allocation read-only — {symbol}*\n"
            f"_Pedido via /allocate — {datetime.now(LISBON_TZ).strftime('%d/%m %H:%M')}_\n\n"
        )
        return header + format_allocation_telegram(decision, ctx, current_price=price)

    except Exception as e:
        logging.error(f"allocate_ticker({symbol}): {e}", exc_info=True)
        return f"❌ *Erro ao calcular alocação para {symbol}*\n_`{e}`_"


# ── Scan principal ────────────────────────────────────────────────────────────

def run_scan() -> None:
    global _scan_running, _alerted_today
    if _scan_running:
        logging.warning("Scan já a correr, a ignorar...")
        return
    _scan_running = True
    # ── Feature 8: conjunto de symbols que passaram score >= MIN_DIP_SCORE ──
    _scan_syms_scored: set[str] = set()
    try:
        if not is_market_open():
            logging.info("Mercado fechado, scan ignorado.")
            return

        logging.info("A iniciar scan de dips...")
        spy_change = get_spy_change()
        stocks     = screen_global_dips(DROP_THRESHOLD, MIN_MARKET_CAP)
        if not stocks:
            logging.info("Sem stocks na triagem.")
            return

        ranked_entries = []
        comprar_syms   = set()

        for stock in stocks:
            sym = stock["symbol"]
            if sym in _alerted_today:
                continue
            try:
                time.sleep(3)
                fund = get_fundamentals(sym, min_market_cap=MIN_MARKET_CAP)
                if fund.get("skip"):
                    append_rejected_log({
                        "symbol": sym, "change": stock["change_pct"],
                        "reason": fund.get("skip_reason", "skip"),
                        "score": None, "verdict": None,
                    })
                    continue

                hist_pe       = get_historical_pe(sym)
                news          = get_news(sym)
                earnings_days = get_earnings_days(sym)
                sector_chg    = get_sector_change(fund.get("sector", ""))
                score, rsi_str = calculate_dip_score(
                    fund, sym, earnings_days, sector_change=sector_chg
                )

                bc_flag  = is_bluechip(fund)
                category = classify_dip_category(fund, score, bc_flag)

                rsi_val = None
                try:
                    rsi_val = float(str(rsi_str).split()[0])
                except Exception:
                    pass

                if score < MIN_DIP_SCORE:
                    append_rejected_log({
                        "symbol": sym, "change": stock["change_pct"],
                        "reason": f"score baixo ({score:.0f})",
                        "score": score, "verdict": "EVITAR",
                    })
                    continue

                if score >= 75:
                    verdict, emoji_str, tier = "COMPRAR", "🟢", 1
                elif score >= 60:
                    verdict, emoji_str, tier = "MONITORIZAR", "🟡", 2
                else:
                    verdict, emoji_str, tier = "MONITORIZAR", "🔵", 3

                # Chunk 6 — ML score em tempo real
                ml_features = {
                    **fund,
                    "score":             score,
                    "spy_change":        spy_change,
                    "sector_etf_change": sector_chg,
                    "earnings_days":     earnings_days,
                    "change_day_pct":    stock["change_pct"],
                }
                ml_result = ml_score(ml_features, symbol=sym)
                if ml_result.model_ready:
                    logging.info(
                        f"[ml] {sym}: label={ml_result.label} "
                        f"prob={ml_result.win_prob:.2f} conf={ml_result.confidence}"
                    )

                # ── Score V2 Engine (Chunk 9) ──────────────────────────────
                ml_prob_v2 = ml_result.win_prob if ml_result.model_ready else None
                score_data = score_from_fundamentals(fund, ml_prob=ml_prob_v2)

                if score_data.get("skip_recommended"):
                    append_rejected_log({
                        "symbol": sym, "change": stock["change_pct"],
                        "reason": f"v2_skip: conf={score_data['confidence']:.2f}",
                        "score": score_data["final_score"], "verdict": "SKIP",
                    })
                    logging.info(f"[v2] {sym} rejeitado — confidence insuficiente")
                    continue

                score = score_data["final_score"]
                logging.info(
                    f"[v2] {sym}: final={score:.1f} | "
                    f"Q={score_data['quality_score']:.2f} "
                    f"V={score_data['value_score']:.2f} "
                    f"T={score_data['timing_score']:.2f} "
                    f"conf={score_data['confidence']:.2f} "
                    f"trap={score_data['is_value_trap']}"
                )

                # ── Feature 8: Dip Persistente ────────────────────────────
                dip_state = record_dip_day(
                    sym,
                    score,
                    fund.get("price", 0),
                    stock["change_pct"],
                    verdict,
                )
                _scan_syms_scored.add(sym)
                check_and_alert_streak(
                    sym, dip_state, score, category,
                    fund, send_telegram, DIRECT_TICKERS, LISBON_TZ,
                )
                # ──────────────────────────────────────────────────────────

                log_alert_snapshot(
                    symbol=sym, fundamentals=fund, score=score,
                    verdict=verdict, category=category,
                    change_day_pct=stock["change_pct"], rsi_val=rsi_val,
                    historical_pe=hist_pe, spy_change=spy_change,
                    sector_etf_change=sector_chg,
                )

                alert_text = build_alert(
                    stock, fund, hist_pe, news,
                    verdict, emoji_str, [], score, rsi_str,
                    category=category,
                    ml_result=ml_result,
                )

                ranked_entries.append({
                    "symbol": sym, "dip_score": score, "tier": tier,
                    "fundamentals": fund, "earnings_date": get_earnings_date(sym),
                    "earnings_days": earnings_days, "catalyst": None, "category": category,
                })

                send_telegram(alert_text)
                _alerted_today.add(sym)
                save_alerts(_alerted_today)

                if verdict == "COMPRAR":
                    comprar_syms.add(sym)
                    add_recovery_position(
                        sym,
                        price_alert=fund.get("price") or stock.get("price", 0),
                        score=score,
                        target_pct=RECOVERY_PCT,
                        verdict=verdict,
                        category=category,
                    )

                append_weekly_log({
                    "symbol": sym, "score": score, "change": stock["change_pct"],
                    "verdict": verdict, "sector": fund.get("sector", ""),
                    "date": datetime.now(LISBON_TZ).strftime("%d/%m"),
                    "time": datetime.now(LISBON_TZ).strftime("%H:%M"),
                })
                append_backtest_entry({
                    "symbol": sym, "date": datetime.now(LISBON_TZ).strftime("%Y-%m-%d"),
                    "price": fund.get("price") or stock.get("price", 0),
                    "score": score, "verdict": verdict,
                })

            except Exception as e:
                logging.error(f"Erro no scan de {sym}: {e}", exc_info=True)

        if ranked_entries:
            ranking_text = build_flip_ranking(ranked_entries, spy_change, exclude_syms=comprar_syms)
            if ranking_text:
                send_telegram(ranking_text)

    finally:
        expire_missing_streaks(_scan_syms_scored)
        _scan_running = False
        logging.info("Scan concluído.")


# ── Watchlist Scan ────────────────────────────────────────────────────────────

def run_watchlist_job() -> None:
    if not WATCHLIST_ENABLED:
        return
    logging.info("A correr watchlist scan...")
    try:
        count = run_watchlist_scan(send_telegram, DIRECT_TICKERS)
        logging.info(f"Watchlist scan: {count} alertas enviados")
    except Exception as e:
        logging.error(f"Watchlist scan: {e}")


# ── ML Outcomes Job (domingo 08:00) ───────────────────────────────────────────

def run_ml_outcomes_job() -> None:
    logging.info("[ml_outcomes] A actualizar outcomes da alert_db...")
    try:
        stats = fill_db_outcomes()
    except Exception as e:
        logging.error(f"[ml_outcomes] Erro: {e}")
        send_telegram(f"🤖 *ML Outcomes — Erro*\n`{e}`")
        return

    if stats.get("updated", 0) == 0 and stats.get("total", 0) == 0:
        return

    db_stats = get_db_stats()
    outcomes = db_stats.get("outcomes", {})
    labeled  = db_stats.get("labeled", 0)
    total    = db_stats.get("total", 0)

    outcome_lines = []
    for label, count in sorted(outcomes.items(), key=lambda x: x[1], reverse=True):
        emoji_map = {"WIN_40": "🟢", "WIN_20": "✅", "NEUTRAL": "🟡", "LOSS_15": "🔴"}
        e = emoji_map.get(label, "📊")
        pct = count / labeled * 100 if labeled > 0 else 0
        outcome_lines.append(f"  {e} {label}: {count} ({pct:.0f}%)")

    lines = [
        f"🤖 *ML Outcomes — Update semanal*",
        f"_{datetime.now(LISBON_TZ).strftime('%d/%m/%Y %H:%M')}_",
        "",
        f"*🗓️ Alertas na base de dados:* {total}",
        f"*📊 Classificados:* {labeled} ({labeled/total*100:.0f}% do total)" if total > 0 else "*📊 Classificados:* 0",
        f"*✅ Actualizados agora:* {stats.get('updated', 0)}",
        f"*⏩ Ignorados (dados insuficientes):* {stats.get('skipped', 0)}",
        "",
    ]
    if outcome_lines:
        lines.append("*🎯 Distribuição de outcomes:*")
        lines.extend(outcome_lines)
        lines.append("")
    if labeled >= 50:
        lines.append("🚦 *Podes considerar treinar o primeiro modelo ML!*")
    elif labeled > 0:
        lines.append(f"_🕐 {50 - labeled} alertas até poder treinar o modelo ML_")

    send_telegram("\n".join(lines))


# ── Monthly ML Retrain Job (dia 1 de cada mês, 06:00) ─────────────────────────

def run_monthly_retrain() -> None:
    """
    Corre no dia 1 de cada mês às 06:00 Lisboa.

    Delega para `monthly_retrain.run_monthly_retrain_v2` que faz:
      1. Build training input (bootstrap + alert_db + universe_snapshot)
      2. Treina candidate em /data/candidate/
      3. Walk-forward gating: candidate só substitui produção se
         AUC-PR ≥ prod × 0.95
      4. Atomic deploy + archive da versão antiga

    Telegram: status final (PROMOTED / KEPT / FAILED) + delta de AUC-PR.
    """
    logging.info("[monthly_retrain] A iniciar retreino mensal v2...")
    now_str = datetime.now(LISBON_TZ).strftime("%d/%m/%Y %H:%M")

    try:
        from monthly_retrain import run_monthly_retrain_v2
        result = run_monthly_retrain_v2()
    except Exception as e:
        logging.error(f"[monthly_retrain] v2 falhou: {e}", exc_info=True)
        send_telegram(
            f"❌ *Retreino Mensal — Erro*\n"
            f"`{e}`\n_⏰ {now_str}_"
        )
        return

    decision  = result.get("decision", "?")
    cand_auc  = result.get("candidate_auc_pr")
    prod_auc  = result.get("production_auc_pr")
    reason    = result.get("reason", "")
    elapsed   = result.get("elapsed_s", 0)
    s1        = result.get("candidate_stage1") or {}
    s2        = result.get("candidate_stage2") or {}
    outcomes  = result.get("outcome_stats") or {}

    # Header com decisão
    icon = {"PROMOTED": "🚀", "KEPT": "🛑", "FAILED": "❌", "DRY-RUN": "🧪"}.get(decision, "⚙️")

    def _fmt(v) -> str:
        if v is None:
            return "N/A"
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return str(v)

    lines = [
        f"{icon} *Retreino Mensal v2 — {decision}*",
        f"_{now_str} ({elapsed:.0f}s)_",
        "",
    ]
    if reason:
        lines.append(f"_{reason}_")
        lines.append("")

    if cand_auc is not None or prod_auc is not None:
        lines += [
            f"*Stage 1 AUC-PR:*",
            f"  candidate : *{_fmt(cand_auc)}*",
            f"  produção  : *{_fmt(prod_auc)}*",
        ]
        if cand_auc and prod_auc:
            delta = (cand_auc - prod_auc) / prod_auc * 100
            sign  = "+" if delta >= 0 else ""
            lines.append(f"  delta     : *{sign}{delta:.1f}%*")
        lines.append("")

    if s1:
        lines += [
            f"*Candidate Stage 1:*",
            f"  threshold: {s1.get('threshold', 'N/A')}",
            f"  precision: {s1.get('test_precision', 'N/A')}",
            f"  recall   : {s1.get('test_recall', 'N/A')}",
            f"  n_test   : {s1.get('n_test', 'N/A')}",
            "",
        ]
    if s2 and s2.get("auc_pr_test"):
        lines += [
            "*Candidate Stage 2:*",
            f"  AUC-PR: {_fmt(s2.get('auc_pr_test'))}",
            "",
        ]

    if outcomes:
        lines += [
            f"*alert_db outcomes preenchidos:* {outcomes.get('updated', 0)}",
            f"*ignorados:* {outcomes.get('skipped', 0)}",
            "",
        ]

    lines.append("_Próximo retreino: dia 1 do mês seguinte às 06h_")
    send_telegram("\n".join(lines))
    logging.info(f"[monthly_retrain] decision={decision} cand={cand_auc} prod={prod_auc}")


# ── Daily Universe Snapshot (após fecho US) ────────────────────────────────────

def run_universe_snapshot() -> None:
    """
    Snapshot diário dos ~780 tickers do ML universe.

    Corre seg-sex às 22:30 Lisboa (após fecho US 22:00 UTC inverno / 21:00 UTC verão).
    Idempotente: skip dos que já têm linha para hoje.
    """
    logging.info("[snapshot] A iniciar snapshot diário do universo...")
    now_str = datetime.now(LISBON_TZ).strftime("%d/%m/%Y %H:%M")

    try:
        from universe_snapshot import run_daily_snapshot
        stats = run_daily_snapshot()
    except Exception as e:
        logging.error(f"[snapshot] Erro: {e}", exc_info=True)
        send_telegram(f"❌ *Universe Snapshot — Erro*\n`{e}`\n_⏰ {now_str}_")
        return

    rows_added = stats.get("rows_added", 0)
    rows_skip  = stats.get("rows_skipped", 0)
    tickers_ok = stats.get("tickers_ok", 0)
    elapsed    = stats.get("elapsed_s", 0)

    send_telegram(
        f"📸 *Universe Snapshot — {now_str}*\n"
        f"  ✅ {tickers_ok} tickers | +{rows_added} linhas | skip {rows_skip} | {elapsed:.0f}s"
    )
    logging.info(f"[snapshot] {tickers_ok} tickers, +{rows_added} linhas, {elapsed:.0f}s")


# ── Daily Backup Job (23:30 Lisboa, seg-sex) ──────────────────────────────────

def run_daily_backup() -> None:
    """
    Backup diário dos ficheiros críticos: alert_db, state JSONs,
    parquet datasets, dip_models_v3.pkl e ml_report_v3.json.
    Arquiva em /data/backups/YYYY-MM-DD_HH-MM.zip e mantém os últimos 30.
    """
    logging.info("[backup] A iniciar backup diário...")
    try:
        result = run_backup()
        msg    = backup_telegram_summary(result)
        send_telegram(msg)
        logging.info(
            f"[backup] Concluído: {result.get('file_count')} ficheiros, "
            f"{result.get('size_mb')} MB"
        )
    except Exception as e:
        logging.error(f"[backup] Erro inesperado: {e}", exc_info=True)
        send_telegram(f"❌ *Backup Diário — Erro*\n`{e}`")


# ── Scheduler ─────────────────────────────────────────────────────────────────

def main() -> None:
    global _alerted_today

    bot_commands.setup(
        send_fn=send_telegram,
        analyze_fn=analyze_ticker,
        allocate_fn=allocate_ticker,
        clear_alerts_fn=lambda: (
            clear_alerts(),
            globals().update(_alerted_today=set()),
        ),
    )

    scheduler = BlockingScheduler(timezone=LISBON_TZ)

    # ── Heartbeat: 09:00 Lisboa, seg-sex ──────────────────────────────────────
    scheduler.add_job(
        send_heartbeat,
        CronTrigger(day_of_week="mon-fri", hour=9, minute=0, timezone=LISBON_TZ),
        id="heartbeat", name="Heartbeat 09h",
    )

    # ── Scan EU: 17:45 Lisboa, seg-sex ────────────────────────────────────────
    scheduler.add_job(
        lambda: (
            run_scan(),
            check_thesis_degradation("EU"),
            check_recovery_alerts(),
        ),
        CronTrigger(day_of_week="mon-fri", hour=17, minute=45, timezone=LISBON_TZ),
        id="scan_eu", name="Scan EU 17h45",
    )

    # ── Scan US: 21:15 Lisboa, seg-sex ────────────────────────────────────────
    scheduler.add_job(
        lambda: (
            run_scan(),
            check_thesis_degradation("US"),
            check_recovery_alerts(),
            run_watchlist_job(),
            send_weekly_dip_scan(),
        ),
        CronTrigger(day_of_week="mon-fri", hour=21, minute=15, timezone=LISBON_TZ),
        id="scan_us", name="Scan US 21h15",
    )

    # ── Stress check: cada 30 min, 14h30-22h30 Lisboa, seg-sex ────────────────
    scheduler.add_job(
        check_portfolio_stress,
        CronTrigger(
            day_of_week="mon-fri",
            hour="14-22", minute=f"*/{SCAN_MINUTES}",
            timezone=LISBON_TZ,
        ),
        id="stress", name="Portfolio Stress",
    )

    # ── Universe Snapshot: 22:30 Lisboa, seg-sex ──────────────────────────────
    scheduler.add_job(
        run_universe_snapshot,
        CronTrigger(day_of_week="mon-fri", hour=22, minute=30, timezone=LISBON_TZ),
        id="universe_snapshot", name="Universe Snapshot 22h30",
    )

    # ── Daily Backup: 23:30 Lisboa, seg-sex ───────────────────────────────────
    scheduler.add_job(
        run_daily_backup,
        CronTrigger(day_of_week="mon-fri", hour=23, minute=30, timezone=LISBON_TZ),
        id="daily_backup", name="Daily Backup 23h30",
    )

    # ── Label Resolver: 22:45 Lisboa, todos os dias ───────────────────────────
    scheduler.add_job(
        lambda: run_label_resolver_job(send_telegram),
        CronTrigger(hour=22, minute=45, timezone=LISBON_TZ),
        id="label_resolver", name="Label Resolver 22h45",
    )

    # ── Position monitor: 22:00 Lisboa, seg-sex ───────────────────────────────
    scheduler.add_job(
        lambda: run_daily_check(send_telegram),
        CronTrigger(day_of_week="mon-fri", hour=22, minute=0, timezone=LISBON_TZ),
        id="position_monitor", name="Position Monitor 22h00",
    )

    # ── Saturday Weekly Report: sábado 10:00 Lisboa ───────────────────────────
    scheduler.add_job(
        send_saturday_report,
        CronTrigger(day_of_week="sat", hour=10, minute=0, timezone=LISBON_TZ),
        id="saturday_report", name="Saturday Weekly Report",
    )

    # ── ML Outcomes: domingo 08:00 Lisboa ─────────────────────────────────────
    scheduler.add_job(
        run_ml_outcomes_job,
        CronTrigger(day_of_week="sun", hour=8, minute=0, timezone=LISBON_TZ),
        id="ml_outcomes", name="ML Outcomes domingo",
    )

    # ── Monthly ML Retrain: dia 1, 06:00 Lisboa ───────────────────────────────
    scheduler.add_job(
        run_monthly_retrain,
        CronTrigger(day=1, hour=6, minute=0, timezone=LISBON_TZ),
        id="monthly_retrain", name="Monthly ML Retrain",
    )

    logging.info("Scheduler iniciado. Jobs activos:")
    for job in scheduler.get_jobs():
        next_time = getattr(job, "next_run_time", "Pendente (aguarda arranque do scheduler)")
        logging.info(f"  • {job.name} — próxima execução: {next_time}")

    send_telegram(
        f"🤖 *DipRadar v2 iniciado*\n"
        f"_{datetime.now(LISBON_TZ).strftime('%d/%m/%Y %H:%M')}_\n"
        f"Scans: EU 17h45 | US 21h15\n"
        f"Backup: 23h30 → /data/backups/"
    )

    scheduler.start()


if __name__ == "__main__":
    main()
