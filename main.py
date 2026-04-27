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
  MIN_DIP_SCORE=5
  TAVILY_API_KEY
"""

import os
import time
import json
import logging
import schedule
import requests
from datetime import datetime, timedelta
from pathlib import Path
from market_client import (
    screen_global_dips, screen_structural_dips, screen_period_dips,
    get_fundamentals, get_news, get_historical_pe,
    get_52w_drawdown, get_earnings_date, get_earnings_days,
    get_catalyst, get_spy_change, is_market_open,
    get_usdeur, get_portfolio_snapshot,
)
from portfolio import HOLDINGS, CASHBACK_EUR_VALUES, PPR_SHARES, PPR_AVG_COST, DIRECT_TICKERS
from sectors import get_sector_config, score_fundamentals
from valuation import format_valuation_block
from score import calculate_dip_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
DROP_THRESHOLD   = float(os.environ.get("DROP_THRESHOLD", "8"))
MIN_MARKET_CAP   = int(os.environ.get("MIN_MARKET_CAP", "2000000000"))
SCAN_MINUTES     = int(os.environ.get("SCAN_EVERY_MINUTES", "30"))
MIN_DIP_SCORE    = int(os.environ.get("MIN_DIP_SCORE", "5"))

_ALERTS_FILE  = Path("/tmp/dipadar_alerts.json")
_WEEKLY_FILE  = Path("/tmp/dipadar_weekly.json")


# ── Cache de alertas ─────────────────────────────────────────────────────────────

def _load_alerts() -> set:
    try:
        if _ALERTS_FILE.exists():
            data  = json.loads(_ALERTS_FILE.read_text())
            today = datetime.now().date().isoformat()
            return {k for k in data.get("keys", []) if k.endswith(today)}
    except Exception:
        pass
    return set()

def _save_alerts(alert_set: set) -> None:
    try:
        _ALERTS_FILE.write_text(json.dumps({"keys": list(alert_set)}))
    except Exception as e:
        logging.warning(f"Alert cache save: {e}")

_alerted_today: set = _load_alerts()
_scan_running:  bool = False


# ── Weekly log (acumula alertas da semana) ───────────────────────────────────

def _load_weekly_log() -> list:
    try:
        if _WEEKLY_FILE.exists():
            return json.loads(_WEEKLY_FILE.read_text()).get("alerts", [])
    except Exception:
        pass
    return []

def _save_weekly_log(entries: list) -> None:
    try:
        _WEEKLY_FILE.write_text(json.dumps({"alerts": entries}))
    except Exception as e:
        logging.warning(f"Weekly log save: {e}")

def _log_alert(symbol: str, verdict: str, score: float, change_pct: float, sector: str) -> None:
    entries = _load_weekly_log()
    entries.append({
        "symbol":  symbol,
        "verdict": verdict,
        "score":   score,
        "change":  change_pct,
        "sector":  sector,
        "date":    datetime.now().strftime("%d/%m"),
        "time":    datetime.now().strftime("%H:%M"),
    })
    _save_weekly_log(entries)


# ── Blue chip detection (por fundamentais, não por sector) ───────────────────

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
    margin_threshold = _BLUECHIP_MARGIN_THRESHOLD.get(sector, 0.40)
    if mc < 50_000_000_000:
        return False
    return (dividend_yield >= 0.015) or (rev_growth > 0.05 and gross_margin > margin_threshold)


# ── Telegram ─────────────────────────────────────────────────────────────────

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


# ── Heartbeat das 9h ─────────────────────────────────────────────────────────

def _pnl_emoji(v: float) -> str:
    return "🟢" if v > 0 else ("🔴" if v < 0 else "⚪")

def send_heartbeat() -> None:
    logging.info("A gerar heartbeat de carteira...")
    try:
        usd_eur  = get_usdeur()
        snapshot = get_portfolio_snapshot(
            HOLDINGS, CASHBACK_EUR_VALUES, PPR_SHARES, PPR_AVG_COST, usd_eur
        )
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
        f"*💼 Carteira total: €{total:,.2f}*",
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
        "",
        "_Mercado abre às 14h30 Lisboa_",
    ]
    send_telegram("\n".join(lines))
    logging.info("Heartbeat enviado.")


# ── Weekly structural dip scan (segunda-feira 8h45) ─────────────────────────

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
            score, rsi_str = calculate_dip_score(fund, sym, earnings_days)
            if score >= 7:
                scored.append({
                    "symbol":        sym,
                    "score":         score,
                    "drawdown":      stock["drawdown_52w"],
                    "price":         fund.get("price") or stock["price"],
                    "mc_b":          (fund.get("market_cap") or 0) / 1e9,
                    "sector":        get_sector_config(fund.get("sector", "")).get("label", "") or fund.get("sector", ""),
                    "rsi":           rsi_str,
                    "bluechip":      is_bluechip(fund),
                    "earnings_days": earnings_days,
                    "analyst_upside": fund.get("analyst_upside") or 0,
                })
            time.sleep(5)
        except Exception as e:
            logging.warning(f"Weekly scan {sym}: {e}")

    if not scored:
        logging.info("Weekly scan: nenhum candidato com score ≧7")
        return

    scored.sort(key=lambda x: x["score"], reverse=True)
    lines = [
        f"*📶 Weekly Structural Dip Scan — {datetime.now().strftime('%d/%m/%Y')}*",
        f"_Stocks ≥25% abaixo dos máximos de 52 semanas com score ≧7 (escala 0-20)_",
        "",
    ]
    for s in scored[:12]:
        badge      = "🔥" if s["score"] >= 16 else ("⭐" if s["score"] >= 11 else "📊")
        bc_tag     = " 💎" if s["bluechip"] else ""
        rsi_tag    = f" | RSI {s['rsi']}" if s["rsi"] else ""
        earn_tag   = f" | 📅 Earnings em {s['earnings_days']}d" if s["earnings_days"] is not None else ""
        upside_tag = f" | 📡 +{s['analyst_upside']:.0f}% analistas" if s["analyst_upside"] > 20 else ""
        lines.append(
            f"{badge} *{s['symbol']}*{bc_tag} — Score {s['score']:.0f}/20 | "
            f"${s['price']:.2f} | ${s['mc_b']:.1f}B | {s['drawdown']:.0f}% do topo"
        )
        lines.append(f"   _{s['sector']}{rsi_tag}{earn_tag}{upside_tag}_")
        lines.append("")
    send_telegram("\n".join(lines))
    logging.info(f"Weekly scan enviado: {len(scored)} candidatos")


# ── Saturday Weekly Report (sábado 10h) ────────────────────────────────────

def send_saturday_report() -> None:
    if datetime.now().weekday() != 5:
        return
    logging.info("A gerar Saturday Weekly Report...")

    entries = _load_weekly_log()
    now       = datetime.now()
    week_start = (now - timedelta(days=now.weekday() + 1)).strftime("%d/%m")
    week_end   = (now - timedelta(days=1)).strftime("%d/%m")

    lines = [
        f"*📊 Weekly Report — {week_start} a {week_end}*",
        "",
    ]

    # ── Bloco 1: Resumo dos alertas diários da semana ──────────────────────
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
            f"  Score médio: *{avg_score:.1f}/20* | Melhor: *{best['symbol']}* (score {best['score']:.0f}, {best['date']})",
            "",
        ]

        lines.append("*🗓️ Por dia:*")
        for date_str in sorted(by_date.keys()):
            day_entries = by_date[date_str]
            lines.append(f"  *{date_str}* — {len(day_entries)} alerta(s)")
            for e in sorted(day_entries, key=lambda x: x["score"], reverse=True):
                verdict_badge = "🟢" if e["verdict"] == "COMPRAR" else "🟡"
                lines.append(
                    f"    {verdict_badge} *{e['symbol']}* Score {e['score']:.0f}/20 | "
                    f"{e['change']:+.1f}% | _{e['sector']}_"
                )
        lines.append("")

        top3 = sorted(entries, key=lambda x: x["score"], reverse=True)[:3]
        lines.append("*🏆 Top 3 da semana:*")
        for i, e in enumerate(top3, 1):
            lines.append(f"  {i}. *{e['symbol']}* — Score {e['score']:.0f}/20 | {e['verdict']} | {e['date']}")
        lines.append("")

    # ── Bloco 2: Weekly dips (quedas ≥10% nos últimos 7 dias) ─────────────
    lines += [
        "─" * 30,
        "",
        "*📉 Weekly Dips — quedas ≥10% nos últimos 7 dias:*",
        "_Stocks dos feeds Yahoo com maior queda acumulada semanal_",
        "",
    ]
    try:
        logging.info("Saturday: a calcular period dips...")
        period_dips = screen_period_dips(
            min_market_cap=MIN_MARKET_CAP,
            week_threshold=10.0,
            month_threshold=15.0,
        )
        weekly  = period_dips.get("weekly", [])
        monthly = period_dips.get("monthly", [])

        if weekly:
            for s in weekly[:10]:
                mc_b         = (s.get("market_cap") or 0) / 1e9
                in_portfolio = " 💼" if s["symbol"] in DIRECT_TICKERS else ""
                lines.append(
                    f"  📉 *{s['symbol']}*{in_portfolio}: *{s['change_pct']:.1f}%* (7d) "
                    f"| ${s['price']} | ${mc_b:.1f}B"
                )
        else:
            lines.append("  _Nenhum stock com queda ≥10% na semana._")

        # ── Bloco 3: Monthly dips (quedas ≥15% no último mês) ─────────────
        lines += [
            "",
            "*📉 Monthly Dips — quedas ≥15% no último mês:*",
            "_Excluídos stocks já listados no weekly dip_",
            "",
        ]
        if monthly:
            for s in monthly[:10]:
                mc_b         = (s.get("market_cap") or 0) / 1e9
                in_portfolio = " 💼" if s["symbol"] in DIRECT_TICKERS else ""
                lines.append(
                    f"  📉 *{s['symbol']}*{in_portfolio}: *{s['change_pct']:.1f}%* (30d) "
                    f"| ${s['price']} | ${mc_b:.1f}B"
                )
        else:
            lines.append("  _Nenhum stock com queda ≥15% no mês._")

    except Exception as e:
        logging.error(f"Saturday period dips: {e}")
        lines.append(f"  _Erro ao calcular period dips: {e}_")

    # ── Limpa log semanal e fecha mensagem ────────────────────────────────
    _save_weekly_log([])
    lines += ["", f"_⏰ Próximo report: sábado às 10h_"]
    send_telegram("\n".join(lines))
    logging.info("Saturday report enviado e log semanal limpo.")


# ── Target Sell ─────────────────────────────────────────────────────────────

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

    if is_bluechip(fundamentals) and dip_score >= 12:
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

    sector_cap = _SECTOR_FLIP_CAP.get(sector, 0.30)
    if dip_score >= 18:   sector_cap *= 1.20
    elif dip_score >= 15: sector_cap *= 1.10

    final_upside = min(weighted_upside, sector_cap)
    final_target = price * (1 + final_upside)

    macro_flag = f" | 🌍 Queda macro SPY {spy_change:.1f}% — timeline incerta" if spy_change and spy_change <= -2.0 else ""

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


# ── Ranking Flip ──────────────────────────────────────────────────────────────

def build_flip_ranking(ranked_entries: list[dict], spy_change: float | None) -> str:
    if not ranked_entries:
        return ""
    lines = ["", "*🏆 RANKING FLIP — Top compras de hoje*"]
    if spy_change is not None:
        sign = "+" if spy_change >= 0 else ""
        lines.append(f"_SPY hoje: {sign}{spy_change:.1f}%_")
    lines.append("")
    top = sorted(ranked_entries, key=lambda x: x["dip_score"], reverse=True)[:8]
    for i, entry in enumerate(top, 1):
        sym      = entry["symbol"]
        score    = entry["dip_score"]
        tier     = entry["tier"]
        f        = entry["fundamentals"]
        earnings = entry.get("earnings_date")
        catalyst = entry.get("catalyst")
        price    = f.get("price", 0)
        mc_b     = (f.get("market_cap") or 0) / 1e9
        in_portfolio = " 💼" if sym in DIRECT_TICKERS else ""
        _, strategy  = calculate_flip_target(f, score, earnings, catalyst, spy_change)
        badge        = "🔥" if score >= 16 else ("⭐" if score >= 11 else "📊")
        tier_badge   = {1: "🔴T1", 2: "🟡T2", 3: "🔵T3"}.get(tier, "")
        lines.append(f"*{i}. {sym}*{in_portfolio} {tier_badge} | Score {score:.0f}/20 {badge}")
        lines.append(f"   💰 ${price} | 🏦 ${mc_b:.1f}B")
        lines.append(f"   {strategy}")
        lines.append("")
    return "\n".join(lines)


# ── Alerta individual ─────────────────────────────────────────────────────────

def build_alert(
    stock, fundamentals, historical_pe, news,
    verdict, emoji, reasons, dip_score, rsi_str,
) -> str:
    sector      = fundamentals.get("sector", "")
    sector_cfg  = get_sector_config(sector)
    name        = fundamentals.get("name", stock.get("name", stock.get("symbol", "N/A")))
    symbol      = stock["symbol"]
    change      = stock["change_pct"]
    price       = fundamentals.get("price") or stock.get("price", "N/D")
    mc_b        = (fundamentals.get("market_cap") or 0) / 1e9
    drawdown    = fundamentals.get("drawdown_from_high")
    drawdown_str = f" | 52w: {drawdown:.0f}%" if drawdown is not None else ""
    region_part  = f" ({stock['region']})" if stock.get("region") else ""
    in_portfolio = " 💼 *Já em carteira*" if symbol in DIRECT_TICKERS else ""
    if dip_score >= 16:   score_badge = f"🔥 Score: {dip_score:.0f}/20"
    elif dip_score >= 11: score_badge = f"⭐ Score: {dip_score:.0f}/20"
    else:                 score_badge = f"📊 Score: {dip_score:.0f}/20"
    rsi_part = f" | RSI: {rsi_str}" if rsi_str else ""
    vol     = fundamentals.get("volume") or 0
    avg_vol = fundamentals.get("average_volume") or 0
    vol_flag = " | 📈 Volume spike" if vol and avg_vol and vol > avg_vol * 1.5 else ""
    lines = [
        f"📉 *{symbol} — {name}{region_part}*{in_portfolio}",
        f"Queda: *{change:.1f}%*{drawdown_str}",
        f"💰 Preço: ${price} | 🏦 Cap: ${mc_b:.1f}B",
        f"🏢 Sector: {sector_cfg.get('label', sector) or sector}",
        f"{score_badge}{rsi_part}{vol_flag}",
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


# ── Scan contínuo ─────────────────────────────────────────────────────────────

def run_scan() -> None:
    global _scan_running
    if not is_market_open():
        logging.info("Mercado fechado — scan ignorado.")
        return
    if _scan_running:
        logging.warning("Scan anterior ainda em curso — a saltar.")
        return
    _scan_running = True
    today = datetime.now().date().isoformat()
    logging.info(f"A correr scan — {datetime.now().strftime('%H:%M')}")
    try:
        losers = screen_global_dips(min_drop_pct=DROP_THRESHOLD, min_market_cap=MIN_MARKET_CAP)
        if not losers:
            logging.info("Sem candidatos hoje.")
            return
        for stock in losers:
            symbol    = stock.get("symbol")
            alert_key = f"{symbol}_{today}"
            if not symbol or alert_key in _alerted_today:
                continue
            try:
                logging.info(f"A analisar {symbol} ({stock['change_pct']:.1f}%)...")
                fundamentals = get_fundamentals(symbol, stock.get("region", ""), MIN_MARKET_CAP)
                if fundamentals.get("skip"):
                    _alerted_today.add(alert_key); _save_alerts(_alerted_today)
                    continue
                sector = fundamentals.get("sector", "")
                verdict, emoji, reasons = score_fundamentals(fundamentals, sector)
                if verdict == "EVITAR":
                    logging.info(f"  {symbol}: EVITAR — a saltar")
                    _alerted_today.add(alert_key); _save_alerts(_alerted_today)
                    continue
                earnings_days      = get_earnings_days(symbol)
                dip_score, rsi_str = calculate_dip_score(fundamentals, symbol, earnings_days)
                if dip_score < MIN_DIP_SCORE:
                    logging.info(f"  {symbol}: score {dip_score} < {MIN_DIP_SCORE} — a saltar")
                    _alerted_today.add(alert_key); _save_alerts(_alerted_today)
                    continue
                historical_pe = get_historical_pe(symbol)
                news          = get_news(symbol)
                message       = build_alert(
                    stock, fundamentals, historical_pe,
                    news, verdict, emoji, reasons, dip_score, rsi_str,
                )
                if send_telegram(message):
                    _alerted_today.add(alert_key)
                    _save_alerts(_alerted_today)
                    _log_alert(symbol, verdict, dip_score, stock["change_pct"], sector)
                    logging.info(f"  ✅ Alerta: {symbol} ({verdict}, score {dip_score}/20)")
                time.sleep(5)
            except Exception as e:
                logging.error(f"Erro {symbol}: {e}")
    finally:
        _scan_running = False


# ── Resumo abertura (15h30) ───────────────────────────────────────────────────

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
        in_portfolio = " 💼" if s["symbol"] in DIRECT_TICKERS else ""
        lines.append(f"  📉 *{s['symbol']}*{in_portfolio}: {s['change_pct']:.1f}% (${mc_b:.1f}B)")
    lines += ["", "_Resumo completo às 21h15_"]
    send_telegram("\n".join(lines))


# ── Resumo fecho (21h15) ──────────────────────────────────────────────────────

def send_close_summary() -> None:
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
            if _timed_out(): logging.warning(f"Watchdog: a saltar {sym}"); return {"skip": True}
            fund_cache[sym] = get_fundamentals(sym, reg, MIN_MARKET_CAP)
        return fund_cache[sym]
    def _get_score(sym, fund):
        if sym not in score_cache:
            ed = get_earnings_days(sym)
            score_cache[sym], _ = calculate_dip_score(fund, sym, ed)
        return score_cache[sym]

    tier3 = []
    for s in tier3_candidates:
        if _timed_out(): logging.warning("Watchdog: Tier 3 truncado"); break
        sym  = s["symbol"]
        fund = _get_fund(sym, s.get("region", ""))
        if fund.get("skip"): continue
        score = _get_score(sym, fund)
        if score >= 11:
            s["_score"] = score
            tier3.append(s)
    tier3.sort(key=lambda x: x.get("_score", 0), reverse=True)

    spy_header = ""
    if spy_change is not None:
        sign       = "+" if spy_change >= 0 else ""
        macro_warn = " 🌍 DIA MACRO" if spy_change <= -2.0 else ""
        spy_header = f"_SPY: {sign}{spy_change:.1f}%{macro_warn}_"

    lines = [f"*📋 Resumo Fecho — {datetime.now().strftime('%d/%m/%Y')}*"]
    if spy_header: lines.append(spy_header)
    lines.append("")

    if tier1:
        lines.append(f"*🔴 TIER 1 — Análise completa (≥{DROP_THRESHOLD:.0f}%):*")
        for s in tier1[:6]:
            mc_b         = (s.get("market_cap") or 0) / 1e9
            drawdown     = get_52w_drawdown(s["symbol"])
            d_str        = f" | 52w: *{drawdown:.0f}%*" if drawdown is not None else ""
            in_portfolio = " 💼" if s["symbol"] in DIRECT_TICKERS else ""
            lines.append(f"  📉 *{s['symbol']}*{in_portfolio}: {s['change_pct']:.1f}% hoje{d_str} — ${mc_b:.1f}B")
        lines += ["", "_→ Verifica catalisador, FCF e 4 critérios Flip_", ""]
    else:
        lines += [f"_Sem quedas ≥{DROP_THRESHOLD:.0f}% hoje_", ""]

    if tier2:
        lines.append(f"*🟡 TIER 2 — Watchlist (7–{DROP_THRESHOLD:.0f}%):*")
        for s in tier2[:6]:
            mc_b         = (s.get("market_cap") or 0) / 1e9
            in_portfolio = " 💼" if s["symbol"] in DIRECT_TICKERS else ""
            lines.append(f"  👀 *{s['symbol']}*{in_portfolio}: {s['change_pct']:.1f}% (${mc_b:.1f}B)")
        lines += ["", "_→ Monitorizar apenas_", ""]

    if tier3:
        lines.append("*🔵 TIER 3 — Gems Raras (-3/-8%, score ≥11):*")
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
            badge        = "🔥" if score >= 16 else "⭐"
            in_portfolio = " 💼" if sym in DIRECT_TICKERS else ""
            lines.append(f"  {badge} *{sym}*{in_portfolio} — Score {score:.0f}/20 | ${price} | ${mc_b:.1f}B | {sector_label}")
            lines.append(f"     {strategy}")
            lines.append("")
        rest_high = [s for s in tier3[5:] if s.get("_score", 0) >= 16]
        if rest_high:
            lines.append(f"  _Score 16+ adicionais: {', '.join(s['symbol'] for s in rest_high)}_")
            lines.append("")

    ranking_entries = []
    for tier_num, tier_list in [(1, tier1), (2, tier2), (3, tier3)]:
        for s in tier_list:
            sym  = s["symbol"]
            fund = _get_fund(sym, s.get("region", ""))
            if fund.get("skip"): continue
            score = _get_score(sym, fund)
            if score >= 11:
                ranking_entries.append({
                    "symbol":        sym,
                    "dip_score":     score,
                    "fundamentals":  fund,
                    "tier":          tier_num,
                    "earnings_date": get_earnings_date(sym),
                    "catalyst":      get_catalyst(sym, fund.get("name", "")),
                })

    ranking_block = build_flip_ranking(ranking_entries, spy_change)
    if ranking_block: lines.append(ranking_block)

    elapsed = int(time.time() - start_time)
    lines.append(f"_⏱ Resumo gerado em {elapsed}s_")
    send_telegram("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("DipRadar iniciado")
    logging.info(f"Threshold: {DROP_THRESHOLD}% | Min cap: ${MIN_MARKET_CAP/1e9:.0f}B")
    logging.info(f"Scan a cada {SCAN_MINUTES} minutos | Min score: {MIN_DIP_SCORE}/20")
    logging.info("=" * 60)

    send_telegram(
        f"🤖 *DipRadar iniciado*\n"
        f"Tier 1: ≥{DROP_THRESHOLD}% | Tier 2: 7–{DROP_THRESHOLD:.0f}% | Tier 3: 3–8% (score≥11)\n"
        f"Score: 0–20 | Min alerta: {MIN_DIP_SCORE}/20\n"
        f"Weekly scan: ✅ Segunda 8h45 | Saturday report: ✅ Sábado 10h\n"
        f"Weekly dips (7d ≥10%) + Monthly dips (30d ≥15%): ✅ No Saturday report\n"
        f"Catalisadores Tavily: {'✅' if os.environ.get('TAVILY_API_KEY') else '⚠️ não configurado'}\n"
        f"_Scan a cada {SCAN_MINUTES} minutos (só horas de mercado)_"
    )

    schedule.every(SCAN_MINUTES).minutes.do(run_scan)
    schedule.every().day.at("08:45").do(send_weekly_dip_scan)
    schedule.every().day.at("09:00").do(send_heartbeat)
    schedule.every().day.at("10:00").do(send_saturday_report)
    schedule.every().day.at("15:30").do(send_open_summary)
    schedule.every().day.at("21:15").do(send_close_summary)
    schedule.every().day.at("00:01").do(
        lambda: (_alerted_today.clear(), _save_alerts(_alerted_today))
    )

    run_scan()

    while True:
        schedule.run_pending()
        time.sleep(60)
