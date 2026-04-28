"""
DipRadar — Stock Alert Bot
Trigger: Yahoo Finance day_losers
Fundamentais: yfinance
Catalisadores: Tavily Search API
Deploy: Railway.app

Variáveis Railway obrigatórias:
  TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
  TZ=Europe/Lisbon
Variáveis opcionais:
  DROP_THRESHOLD=8
  MIN_MARKET_CAP=2000000000
  SCAN_EVERY_MINUTES=30
  MIN_DIP_SCORE=50
  TAVILY_API_KEY
  PORTFOLIO_STRESS_PCT=5
  RECOVERY_TARGET_PCT=15
  WATCHLIST_SCAN_ENABLED=true
"""

import os
import time
import logging
import schedule
import requests
from datetime import datetime, timedelta
from market_client import (
    screen_global_dips, screen_structural_dips, screen_period_dips,
    get_fundamentals, get_news, get_historical_pe,
    get_52w_drawdown, get_earnings_date, get_earnings_days,
    get_catalyst, get_spy_change, is_market_open,
    get_usdeur, get_portfolio_snapshot,
)
from portfolio import HOLDINGS, CASHBACK_EUR_VALUES, PPR_SHARES, PPR_AVG_COST, DIRECT_TICKERS, FLIP_FUND_EUR
from sectors import get_sector_config, score_fundamentals
from valuation import format_valuation_block
from score import calculate_dip_score, build_score_breakdown
from state import (
    load_alerts, save_alerts, clear_alerts,
    load_weekly_log, save_weekly_log, append_weekly_log,
    load_rejected_log, append_rejected_log,
    append_backtest_entry,
    load_recovery_watch, add_recovery_position,
    mark_recovery_alerted, remove_recovery_position,
)
from backtest import backtest_runner, build_backtest_summary
from watchlist import run_watchlist_scan, build_watchlist_morning_summary, WATCHLIST
import bot_commands

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

TELEGRAM_TOKEN    = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID  = os.environ.get("TELEGRAM_CHAT_ID", "")
DROP_THRESHOLD    = float(os.environ.get("DROP_THRESHOLD", "8"))
MIN_MARKET_CAP    = int(os.environ.get("MIN_MARKET_CAP", "2000000000"))
SCAN_MINUTES      = int(os.environ.get("SCAN_EVERY_MINUTES", "30"))
MIN_DIP_SCORE     = int(os.environ.get("MIN_DIP_SCORE", "50"))  # escala 0-100
STRESS_PCT        = float(os.environ.get("PORTFOLIO_STRESS_PCT", "5"))
RECOVERY_PCT      = float(os.environ.get("RECOVERY_TARGET_PCT", "15"))
WATCHLIST_ENABLED = os.environ.get("WATCHLIST_SCAN_ENABLED", "true").lower() == "true"

_alerted_today:  set  = load_alerts()
_scan_running:   bool = False
_stress_alerted: set  = set()
_last_tier3:     list = []  # cache para comando /tier3


# ── Sector ETF map (para sector rotation signal) ───────────────────────────

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
    """Retorna a variação do ETF sectorial no dia. Cache por sessão."""
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


# ── Helpers de badge (escala 0-100) ─────────────────────────────────────

def score_badge(score: float) -> str:
    if score >= 80:  return "🔥"
    if score >= 55:  return "⭐"
    return "📊"


# ── Blue chip detection ───────────────────────────────────────────────────

_BLUECHIP_MARGIN_THRESHOLD = {
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

def is_bluechip(fundamentals: dict) -> bool:
    mc             = fundamentals.get("market_cap") or 0
    dividend_yield = fundamentals.get("dividend_yield") or 0
    rev_growth     = fundamentals.get("revenue_growth") or 0
    gross_margin   = fundamentals.get("gross_margin") or 0
    sector         = fundamentals.get("sector", "")
    if mc < 50_000_000_000:
        return False
    threshold = _BLUECHIP_MARGIN_THRESHOLD.get(sector, 0.40)
    return (dividend_yield >= 0.015) or (rev_growth > 0.05 and gross_margin > threshold)


# ── Insider buying & short interest flags ────────────────────────────────

def get_insider_buy_flag(symbol: str) -> str:
    try:
        import yfinance as yf
        transactions = yf.Ticker(symbol).insider_transactions
        if transactions is None or transactions.empty:
            return ""
        cutoff = datetime.now() - timedelta(days=90)
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


# ── Telegram ─────────────────────────────────────────────────────────────

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


# ── Heartbeat das 9h ──────────────────────────────────────────────────────

def _pnl_emoji(v: float) -> str:
    return "🟢" if v > 0 else ("🔴" if v < 0 else "⚪")

def _get_snapshot() -> dict:
    usd_eur = get_usdeur()
    return get_portfolio_snapshot(
        HOLDINGS, CASHBACK_EUR_VALUES, PPR_SHARES, PPR_AVG_COST, usd_eur
    )

def send_heartbeat() -> None:
    logging.info("A gerar heartbeat de carteira...")
    try:
        snapshot = _get_snapshot()
    except Exception as e:
        logging.error(f"Heartbeat snapshot: {e}")
        send_telegram(
            f"🤖 *DipRadar* ativo — {datetime.now().strftime('%d/%m %H:%M')}\n"
            f"_Erro ao calcular carteira: {e}_"
        )
        return

    total      = snapshot["total_eur"]
    total_cost = snapshot.get("total_cost", 0)
    if total == 0:
        send_telegram(
            f"🤖 *DipRadar — Bom dia!* {datetime.now().strftime('%d/%m/%Y')}\n"
            f"⚠️ *Variáveis da carteira em falta no Railway.*\n"
            f"_Adiciona HOLDING_*, CASHBACK_*, PPR_SHARES, PPR_AVG_COST nas env vars._"
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

    yesterday = (datetime.now() - timedelta(days=1)).strftime("%d/%m")
    lines = [
        f"🤖 *DipRadar — Bom dia!* {datetime.now().strftime('%d/%m/%Y')}",
        f"_USD/EUR: {fx:.4f}_",
        "",
        f"*📦 Carteira total: €{total:,.2f}*",
        "",
        f"  {_pnl_emoji(pnl_d)}  *Ontem ({yesterday}):*  €{pnl_d:+,.2f}{_pct(pnl_d, total - pnl_d)}",
        f"  {_pnl_emoji(pnl_w)}  *Semana:*            €{pnl_w:+,.2f}{_pct(pnl_w, total - pnl_w)}",
        f"  {_pnl_emoji(pnl_m)}  *Mês:*               €{pnl_m:+,.2f}{_pct(pnl_m, total - pnl_m)}",
    ]
    if pnl_tot is not None:
        lines.append(f"  {_pnl_emoji(pnl_tot)}  *Total investido:*   €{pnl_tot:+,.2f}{_pct(pnl_tot, total_cost)}")

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
        f"  📊 PPR (proxy ACWI): €{snapshot['ppr_value']:,.2f}",
        f"  💜 CashBack Pie: €{snapshot['cashback_eur']:,.2f}",
    ]

    # ── Flip Fund ────────────────────────────────────────────────────────
    if FLIP_FUND_EUR:
        lines.append(f"  🎯 *Flip Fund:* €{FLIP_FUND_EUR:,.2f} disponíveis")

    lines += [
        "",
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


# ── Portfolio Stress Alert ────────────────────────────────────────────────

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
                f"_⏰ {datetime.now().strftime('%d/%m %H:%M')}_"
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
                    f"_⏰ {datetime.now().strftime('%d/%m %H:%M')}_"
                )
                logging.warning(f"Portfolio stress MACRO: {pct_total:.1f}%")


# ── Recovery Alert ────────────────────────────────────────────────────────

def check_recovery_alerts() -> None:
    import yfinance as yf
    positions = load_recovery_watch()
    if not positions:
        return
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
                    f"_⏰ {datetime.now().strftime('%d/%m %H:%M')}_"
                )
                mark_recovery_alerted(sym)
                logging.info(f"Recovery alert: {sym} ${current:.2f} (+{pct_recovery:.1f}%)")
        except Exception as e:
            logging.warning(f"Recovery check {sym}: {e}")


# ── Weekly structural dip scan (segunda-feira 8h45) ───────────────────────

def send_weekly_dip_scan() -> None:
    if datetime.now().weekday() != 0:
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
            score, rsi_str = calculate_dip_score(fund, sym, earnings_days, sector_change=sector_chg)
            if score >= 70:
                scored.append({
                    "symbol":         sym,
                    "score":          score,
                    "drawdown":       stock["drawdown_52w"],
                    "price":          fund.get("price") or stock["price"],
                    "mc_b":           (fund.get("market_cap") or 0) / 1e9,
                    "sector":         get_sector_config(fund.get("sector", "")).get("label", "") or fund.get("sector", ""),
                    "rsi":            rsi_str,
                    "bluechip":       is_bluechip(fund),
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
        f"*📶 Weekly Structural Dip Scan — {datetime.now().strftime('%d/%m/%Y')}*",
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
        lines.append(f"   _{s['sector']}{rsi_tag}{earn_tag}{upside_tag}_")
        lines.append("")
    send_telegram("\n".join(lines))
    logging.info(f"Weekly scan enviado: {len(scored)} candidatos")


# ── Saturday Weekly Report ────────────────────────────────────────────────

def send_saturday_report() -> None:
    if datetime.now().weekday() != 5:
        return
    logging.info("A gerar Saturday Weekly Report...")
    entries    = load_weekly_log()
    now        = datetime.now()
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
    logging.info("Saturday report enviado.")


# ── Target Sell / Flip ────────────────────────────────────────────────────

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
) -> tuple[str, str]:
    price = fundamentals.get("price") or 0
    if not price or price <= 0:
        return "N/D", "SEM DADOS"
    if is_bluechip(fundamentals) and dip_score >= 60:
        return "HOLD ETERNO", "💎 Blue chip — Adicionar em dips, nunca vender"
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
    strategy = f"🎯 Flip: ${final_target:.1f} (+{final_upside*100:.0f}%){cat_flag}{macro_flag}"
    return f"${final_target:.1f} (+{final_upside*100:.0f}%)", strategy


# ── Ranking Flip ──────────────────────────────────────────────────────────

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
        sym      = entry["symbol"]
        s        = entry["dip_score"]
        tier     = entry["tier"]
        f        = entry["fundamentals"]
        earnings = entry.get("earnings_date")
        catalyst = entry.get("catalyst")
        price    = f.get("price", 0)
        mc_b     = (f.get("market_cap") or 0) / 1e9
        in_portfolio = " 📦" if sym in DIRECT_TICKERS else ""
        _, strategy  = calculate_flip_target(f, s, earnings, catalyst, spy_change)
        badge        = score_badge(s)
        tier_badge   = {1: "🔴T1", 2: "🟡T2", 3: "🔵T3"}.get(tier, "")
        lines.append(f"*{i}. {sym}*{in_portfolio} {tier_badge} | Score {s:.0f}/100 {badge}")
        lines.append(f"   💰 ${price} | 🏦 ${mc_b:.1f}B")
        lines.append(f"   {strategy}")
        lines.append("")
    return "\n".join(lines)


# ── Alerta individual ─────────────────────────────────────────────────────

def build_alert(
    stock, fundamentals, historical_pe, news,
    verdict, emoji, reasons, dip_score, rsi_str,
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

    lines = [
        f"📉 *{symbol} — {name}{region_part}*{in_portfolio}",
        f"Queda: *{change:.1f}%*{drawdown_str}",
        f"💰 Preço: ${price} | 🏦 Cap: ${mc_b:.1f}B",
        f"🏢 Sector: {sector_cfg.get('label', sector) or sector}",
        f"{score_line}{rsi_part}{vol_flag}{insider_flag}{short_flag}{sector_warn}",
        "",
        f"*{emoji} Veredito: {verdict}*",
    ]
    for reason in reasons:
        lines.append(f"  _{reason}_")
    lines += ["", "*📊 Fundamentos:*"]
    lines.append(format_valuation_block(fundamentals, historical_pe, sector))
    if news:
        lines += ["", "*📰 Notícias:*"]
        for item in news[:3]:
            title  = item["title"][:70]
            url    = item["url"]
            source = item.get("source", "")
            lines.append(f"  [{title}]({url}){' _' + source + '_' if source else ''}")
    lines.append(f"_⏰ {datetime.now().strftime('%d/%m %H:%M')}_")
    return "\n".join(lines)


# ── Análise on-demand (/analisar) ────────────────────────────────────────────

def handle_analyze_ticker(symbol: str) -> str:
    """
    Análise completa de qualquer ticker a pedido do utilizador.
    Devolve uma string formatada para envio via Telegram.
    Sem filtro de score mínimo — mostra sempre o resultado.
    Inclui score breakdown detalhado critério a critério.
    """
    symbol = symbol.upper().strip()
    logging.info(f"[/analisar] A analisar {symbol}...")
    try:
        fund = get_fundamentals(symbol, "", min_market_cap=0)  # sem filtro de cap
        if not fund or fund.get("skip"):
            return (
                f"⚠️ *{symbol}* não encontrado ou dados insuficientes.\n"
                f"_Verifica se o ticker está correcto (ex: AAPL, NVDA, MSFT)_"
            )

        sector         = fund.get("sector", "")
        sector_cfg     = get_sector_config(sector)
        verdict, emoji, reasons = score_fundamentals(fund, sector)
        earnings_days  = get_earnings_days(symbol)
        earnings_date  = get_earnings_date(symbol)
        sector_chg     = get_sector_change(sector)
        spy_change     = get_spy_change()
        dip_score, rsi_str = calculate_dip_score(
            fund, symbol, earnings_days, sector_change=sector_chg
        )
        historical_pe  = get_historical_pe(symbol)
        news           = get_news(symbol)
        catalyst       = get_catalyst(symbol, fund.get("name", ""))
        _, strategy    = calculate_flip_target(fund, dip_score, earnings_date, catalyst, spy_change)

        name         = fund.get("name", symbol)
        price        = fund.get("price") or 0
        mc_b         = (fund.get("market_cap") or 0) / 1e9
        drawdown     = fund.get("drawdown_from_high")
        drawdown_str = f" | 52w: {drawdown:.0f}%" if drawdown is not None else ""
        in_portfolio = " 📦 *Já em carteira*" if symbol in DIRECT_TICKERS else ""
        bc_tag       = " 💎 Blue chip" if is_bluechip(fund) else ""

        badge_str  = score_badge(dip_score)
        score_line = f"{badge_str} Score: {dip_score:.0f}/100"

        vol      = fund.get("volume") or 0
        avg_vol  = fund.get("average_volume") or 0
        vol_flag     = " | 📈 Volume spike" if vol and avg_vol and vol > avg_vol * 1.5 else ""
        insider_flag = get_insider_buy_flag(symbol)
        short_flag   = get_short_interest_flag(fund)
        sector_warn  = f" | 🟠 Sector {sector_chg:+.1f}% hoje" if sector_chg is not None and sector_chg <= -2.0 else ""
        rsi_part     = f" | RSI: {rsi_str}" if rsi_str else ""
        earn_tag     = f" | 📅 Earnings em {earnings_days}d" if earnings_days is not None else ""
        spy_str      = f" | SPY: {spy_change:+.1f}%" if spy_change is not None else ""

        lines = [
            f"🔍 *Análise: {symbol} — {name}*{in_portfolio}{bc_tag}",
            f"💰 Preço: *${price}*{drawdown_str}{spy_str}",
            f"🏦 Market cap: ${mc_b:.1f}B | 🏢 {sector_cfg.get('label', sector) or sector}",
            f"{score_line}{rsi_part}{earn_tag}{vol_flag}{insider_flag}{short_flag}{sector_warn}",
            "",
            f"*{emoji} Veredito: {verdict}*",
        ]
        for reason in reasons:
            lines.append(f"  _{reason}_")

        lines += ["", f"*🎯 Target de venda:*", f"  {strategy}", ""]

        # ── Score Breakdown ──────────────────────────────────────────────
        try:
            breakdown = build_score_breakdown(fund, symbol, earnings_days, sector_chg)
            if breakdown:
                lines += ["", "*🔬 Score Breakdown:*"]
                for bd_line in breakdown:
                    lines.append(f"  {bd_line}")
        except Exception as bd_err:
            logging.warning(f"[/analisar] score breakdown {symbol}: {bd_err}")

        lines += ["", "*📊 Fundamentos:*"]
        lines.append(format_valuation_block(fund, historical_pe, sector))

        if news:
            lines += ["", "*📰 Notícias:*"]
            for item in news[:3]:
                title  = item["title"][:70]
                url    = item["url"]
                source = item.get("source", "")
                lines.append(f"  [{title}]({url}){' _' + source + '_' if source else ''}")

        lines.append(f"_⏰ {datetime.now().strftime('%d/%m %H:%M')} — via /analisar_")
        return "\n".join(lines)

    except Exception as e:
        logging.error(f"[/analisar] {symbol}: {e}")
        return f"❌ Erro ao analisar *{symbol}*: `{e}`"


# ── Scan contínuo ─────────────────────────────────────────────────────────

def run_scan() -> None:
    global _scan_running, _sector_etf_cache
    if not is_market_open():
        logging.info("Mercado fechado — scan ignorado.")
        return
    if _scan_running:
        logging.warning("Scan anterior ainda em curso — a saltar.")
        return
    _scan_running = True
    _sector_etf_cache = {}
    today = datetime.now().date().isoformat()
    logging.info(f"A correr scan — {datetime.now().strftime('%H:%M')}")
    try:
        check_portfolio_stress()
        check_recovery_alerts()

        if WATCHLIST_ENABLED:
            try:
                wl_hits = run_watchlist_scan(send_telegram, DIRECT_TICKERS)
                if wl_hits:
                    logging.info(f"Watchlist: {wl_hits} alertas enviados")
            except Exception as e:
                logging.warning(f"Watchlist scan: {e}")

        losers = screen_global_dips(min_drop_pct=DROP_THRESHOLD, min_market_cap=MIN_MARKET_CAP)
        if not losers:
            logging.info("Sem candidatos hoje.")
            return

        n         = len(losers)
        dyn_sleep = max(2, min(8, 60 // max(n, 1)))
        logging.info(f"  {n} candidatos, sleep dinâmico: {dyn_sleep}s/stock")

        for stock in losers:
            symbol    = stock.get("symbol")
            alert_key = f"{symbol}_{today}"
            if not symbol or alert_key in _alerted_today:
                continue
            try:
                logging.info(f"A analisar {symbol} ({stock['change_pct']:.1f}%)...")
                fundamentals = get_fundamentals(symbol, stock.get("region", ""), MIN_MARKET_CAP)
                if fundamentals.get("skip"):
                    append_rejected_log(symbol, stock["change_pct"], "cap_insuficiente")
                    _alerted_today.add(alert_key); save_alerts(_alerted_today)
                    continue
                sector = fundamentals.get("sector", "")
                verdict, emoji, reasons = score_fundamentals(fundamentals, sector)
                if verdict == "EVITAR":
                    append_rejected_log(symbol, stock["change_pct"], "EVITAR", verdict=verdict, sector=sector)
                    _alerted_today.add(alert_key); save_alerts(_alerted_today)
                    continue
                earnings_days      = get_earnings_days(symbol)
                sector_chg         = get_sector_change(sector)
                dip_score, rsi_str = calculate_dip_score(
                    fundamentals, symbol, earnings_days, sector_change=sector_chg
                )
                if dip_score < MIN_DIP_SCORE:
                    append_rejected_log(symbol, stock["change_pct"], "score_baixo",
                                        score=dip_score, verdict=verdict, sector=sector)
                    _alerted_today.add(alert_key); save_alerts(_alerted_today)
                    continue
                historical_pe = get_historical_pe(symbol)
                news          = get_news(symbol)
                message       = build_alert(
                    stock, fundamentals, historical_pe,
                    news, verdict, emoji, reasons, dip_score, rsi_str,
                )
                if send_telegram(message):
                    _alerted_today.add(alert_key)
                    save_alerts(_alerted_today)
                    append_weekly_log(symbol, verdict, dip_score, stock["change_pct"], sector)
                    price_now = fundamentals.get("price") or 0
                    if price_now:
                        append_backtest_entry(symbol, verdict, dip_score, stock["change_pct"], price_now, sector)
                    _, strategy = calculate_flip_target(fundamentals, dip_score)
                    if price_now and "HOLD" not in strategy:
                        add_recovery_position(symbol, price_now, dip_score, RECOVERY_PCT, verdict)
                    logging.info(f"  ✅ Alerta: {symbol} ({verdict}, score {dip_score}/100)")
                time.sleep(dyn_sleep)
            except Exception as e:
                logging.error(f"Erro {symbol}: {e}")
    finally:
        _scan_running = False


# ── Resumo abertura (15h30) ───────────────────────────────────────────────

def send_open_summary() -> None:
    tier1 = screen_global_dips(min_drop_pct=DROP_THRESHOLD, min_market_cap=MIN_MARKET_CAP)
    if not tier1:
        return
    spy_change = get_spy_change()
    spy_str    = f" | SPY: {'+' if (spy_change or 0) >= 0 else ''}{spy_change:.1f}%" if spy_change is not None else ""
    lines = [
        f"*⚡ Abertura +1h — {datetime.now().strftime('%d/%m %H:%M')}*",
        f"_{len(tier1)} candidato(s) com queda ≥{DROP_THRESHOLD:.0f}%{spy_str}_",
        "",
    ]
    for s in tier1[:8]:
        mc_b         = (s.get("market_cap") or 0) / 1e9
        in_portfolio = " 📦" if s["symbol"] in DIRECT_TICKERS else ""
        lines.append(f"  📉 *{s['symbol']}*{in_portfolio}: {s['change_pct']:.1f}% (${mc_b:.1f}B)")
    lines += ["", "_Resumo completo às 21h15_"]
    send_telegram("\n".join(lines))


# ── Resumo fecho (21h15) ──────────────────────────────────────────────────

def send_close_summary() -> None:
    global _last_tier3
    start_time = time.time()
    WATCHDOG   = 25 * 60
    spy_change = get_spy_change()
    all_losers = screen_global_dips(min_drop_pct=3.0, min_market_cap=MIN_MARKET_CAP)
    if not all_losers:
        return

    tier1            = [s for s in all_losers if s["change_pct"] <= -DROP_THRESHOLD]
    tier2            = [s for s in all_losers if -DROP_THRESHOLD < s["change_pct"] <= -7.0]
    tier3_candidates = [s for s in all_losers if -8.0 < s["change_pct"] <= -3.0]

    fund_cache:  dict = {}
    score_cache: dict = {}

    def _timed_out():          return (time.time() - start_time) > WATCHDOG
    def _get_fund(sym, reg=""):
        if sym not in fund_cache:
            if _timed_out(): return {"skip": True}
            fund_cache[sym] = get_fundamentals(sym, reg, MIN_MARKET_CAP)
        return fund_cache[sym]
    def _get_score(sym, fund):
        if sym not in score_cache:
            ed        = get_earnings_days(sym)
            sec_chg   = get_sector_change(fund.get("sector", ""))
            score_cache[sym], _ = calculate_dip_score(fund, sym, ed, sector_change=sec_chg)
        return score_cache[sym]

    tier3 = []
    for s in tier3_candidates:
        if _timed_out(): break
        sym  = s["symbol"]
        fund = _get_fund(sym, s.get("region", ""))
        if fund.get("skip"): continue
        score = _get_score(sym, fund)
        if score >= 80:
            s["_score"] = score
            tier3.append(s)
    tier3.sort(key=lambda x: x.get("_score", 0), reverse=True)
    _last_tier3 = tier3

    spy_header = ""
    if spy_change is not None:
        sign       = "+" if spy_change >= 0 else ""
        macro_warn = " 🌍 DIA MACRO" if spy_change <= -2.0 else ""
        spy_header = f"_SPY: {sign}{spy_change:.1f}%{macro_warn}_"

    lines = [f"*📋 Resumo Fecho — {datetime.now().strftime('%d/%m/%Y')}*"]
    if spy_header: lines.append(spy_header)
    lines.append("")

    tier1_syms = {s["symbol"] for s in tier1}

    if tier1:
        lines.append(f"*🔴 TIER 1 — Análise completa (≥{DROP_THRESHOLD:.0f}%):*")
        for s in tier1[:6]:
            mc_b         = (s.get("market_cap") or 0) / 1e9
            drawdown     = get_52w_drawdown(s["symbol"])
            d_str        = f" | 52w: *{drawdown:.0f}%*" if drawdown is not None else ""
            in_portfolio = " 📦" if s["symbol"] in DIRECT_TICKERS else ""
            lines.append(f"  📉 *{s['symbol']}*{in_portfolio}: {s['change_pct']:.1f}% hoje{d_str} — ${mc_b:.1f}B")
        lines += ["", "_→ Verifica catalisador, FCF e 4 critérios Flip_", ""]
    else:
        lines += [f"_Sem quedas ≥{DROP_THRESHOLD:.0f}% hoje_", ""]

    if tier2:
        lines.append(f"*🟡 TIER 2 — Watchlist (7–{DROP_THRESHOLD:.0f}%):*")
        for s in tier2[:6]:
            mc_b         = (s.get("market_cap") or 0) / 1e9
            in_portfolio = " 📦" if s["symbol"] in DIRECT_TICKERS else ""
            lines.append(f"  👀 *{s['symbol']}*{in_portfolio}: {s['change_pct']:.1f}% (${mc_b:.1f}B)")
        lines += ["", "_→ Monitorizar apenas_", ""]

    if tier3:
        lines.append("*🔵 TIER 3 — Gems Raras (-3/-8%, score ≥80):*")
        lines.append("")
        for s in tier3[:5]:
            sym          = s["symbol"]
            score        = s.get("_score", 0)
            fund         = fund_cache.get(sym, {})
            mc_b         = (fund.get("market_cap") or 0) / 1e9
            sector_label = get_sector_config(fund.get("sector", "")).get("label", "") or fund.get("sector", "")
            price        = fund.get("price", 0)
            earnings     = get_earnings_date(sym)
            catalyst     = get_catalyst(sym, fund.get("name", ""))
            _, strategy  = calculate_flip_target(fund, score, earnings, catalyst, spy_change)
            in_portfolio = " 📦" if sym in DIRECT_TICKERS else ""
            lines.append(f"  🔥 *{sym}*{in_portfolio} — Score {score:.0f}/100 | ${price} | ${mc_b:.1f}B | {sector_label}")
            lines.append(f"     {strategy}")
            lines.append("")
        rest_high = [s for s in tier3[5:] if s.get("_score", 0) >= 80]
        if rest_high:
            lines.append(f"  _Score 80+ adicionais: {', '.join(s['symbol'] for s in rest_high)}_")
            lines.append("")

    ranking_entries = []
    for tier_num, tier_list in [(1, tier1), (2, tier2), (3, tier3)]:
        for s in tier_list:
            sym  = s["symbol"]
            fund = _get_fund(sym, s.get("region", ""))
            if fund.get("skip"): continue
            score = _get_score(sym, fund)
            if score >= 55:
                ranking_entries.append({
                    "symbol":        sym,
                    "dip_score":     score,
                    "fundamentals":  fund,
                    "tier":          tier_num,
                    "earnings_date": get_earnings_date(sym),
                    "catalyst":      get_catalyst(sym, fund.get("name", "")),
                })

    ranking_block = build_flip_ranking(ranking_entries, spy_change, exclude_syms=tier1_syms)
    if ranking_block: lines.append(ranking_block)

    elapsed = int(time.time() - start_time)
    lines.append(f"_⏱ Resumo gerado em {elapsed}s_")
    send_telegram("\n".join(lines))


# ── Comando /tier3 ───────────────────────────────────────────────────────────

def handle_tier3_command() -> str:
    if not _last_tier3:
        return "🔵 *Tier 3 — Gems Raras*\n_Nenhuma gem detectada no último resumo de fecho._"
    lines = [
        "🔵 *Tier 3 — Gems Raras* _(do último fecho)_",
        "_Apenas stocks com queda 3–8% e score ≥80/100_",
        "",
    ]
    for s in _last_tier3:
        sym   = s["symbol"]
        score = s.get("_score", 0)
        chg   = s.get("change_pct", 0)
        mc_b  = (s.get("market_cap") or 0) / 1e9
        in_portfolio = " 📦" if sym in DIRECT_TICKERS else ""
        lines.append(f"  🔥 *{sym}*{in_portfolio} — Score {score:.0f}/100 | {chg:.1f}% | ${mc_b:.1f}B")
    lines.append(f"\n_⏰ {datetime.now().strftime('%d/%m %H:%M')}_")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pathlib
    logging.info("=" * 60)
    logging.info("DipRadar iniciado")
    logging.info(f"Threshold: {DROP_THRESHOLD}% | Min cap: ${MIN_MARKET_CAP/1e9:.0f}B")
    logging.info(f"Scan a cada {SCAN_MINUTES} minutos | Min score: {MIN_DIP_SCORE}/100")
    logging.info(f"Stress: >{STRESS_PCT:.0f}% | Recovery target: +{RECOVERY_PCT:.0f}%")
    logging.info(f"Watchlist: {'ACTIVA' if WATCHLIST_ENABLED else 'INACTIVA'} ({len(WATCHLIST)} stocks)")
    logging.info("=" * 60)

    bot_commands.register_callbacks(
        send_telegram=send_telegram,
        run_scan=run_scan,
        get_snapshot=_get_snapshot,
        backtest_summary=build_backtest_summary,
        rejected_log=load_rejected_log,
        is_market_open=is_market_open,
        tier3_handler=handle_tier3_command,
        analyze_ticker=handle_analyze_ticker,
    )
    bot_commands.start_bot_listener()

    has_volume = pathlib.Path("/data").exists()
    tavily_ok  = bool(os.environ.get("TAVILY_API_KEY"))

    send_telegram(
        f"🤖 *DipRadar iniciado* ⚠️\n"
        f"_Restart detectado às {datetime.now().strftime('%d/%m %H:%M')}_\n"
        f"Tier 1: ≥{DROP_THRESHOLD}% | Tier 2: 7–{DROP_THRESHOLD:.0f}% | Tier 3: 3–8% (score≥80 🔥)\n"
        f"Score: 0–100 | Min alerta: {MIN_DIP_SCORE}/100\n"
        f"Badges: 🔥≥80 · ⭐55–79 · 📊<55\n"
        f"Portfolio stress: >{STRESS_PCT:.0f}% posição | >3% total\n"
        f"Recovery alert: +{RECOVERY_PCT:.0f}% do preço de alerta\n"
        f"Backtesting: ✅ automático às 21h30\n"
        f"Watchlist pessoal: {'✅ ' + str(len(WATCHLIST)) + ' stocks' if WATCHLIST_ENABLED else '⚠️ inactiva'}\n"
        f"Comandos: /help /status /carteira /scan /analisar /backtest /rejeitados /tier3\n"
        f"Persistência: {('✅ /data/') if has_volume else '⚠️ /tmp/ — configura Railway Volume'}\n"
        f"Tavily: {('✅') if tavily_ok else '⚠️ não configurado'}\n"
        f"_Scan a cada {SCAN_MINUTES} minutos (só horas de mercado)_"
    )

    schedule.every(SCAN_MINUTES).minutes.do(run_scan)
    schedule.every().day.at("08:45").do(send_weekly_dip_scan)
    schedule.every().day.at("09:00").do(send_heartbeat)
    schedule.every().day.at("10:00").do(send_saturday_report)
    schedule.every().day.at("15:30").do(send_open_summary)
    schedule.every().day.at("21:15").do(send_close_summary)
    schedule.every().day.at("21:30").do(backtest_runner)
    schedule.every().day.at("00:01").do(
        lambda: (
            _alerted_today.clear(), save_alerts(_alerted_today),
            _stress_alerted.clear(),
        )
    )

    run_scan()

    while True:
        schedule.run_pending()
        time.sleep(60)
