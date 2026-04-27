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
    screen_global_dips, get_fundamentals, get_news,
    get_historical_pe, get_52w_drawdown, get_earnings_date,
    get_catalyst, get_spy_change, is_market_open,
    get_usdeur, get_portfolio_snapshot,
)
from portfolio import HOLDINGS, CASHBACK_EUR_VALUES, PPR_SHARES, PPR_AVG_COST
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

# ── Cache de alertas persistido em ficheiro ──────────────────────────────────
_ALERTS_FILE = Path("/tmp/dipadar_alerts.json")

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

# ── Lock para evitar scans sobrepostos ──────────────────────────────────────
_scan_running: bool = False


# ── Telegram ─────────────────────────────────────────────────────────────────

def send_telegram(message: str, retries: int = 2) -> bool:
    """Envia mensagem Telegram com retry automático e divisão em chunks >4000 chars."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(message)
        return True

    chunks = []
    while len(message) > 4000:
        split_at = message.rfind("\n", 0, 4000)
        if split_at == -1:
            split_at = 4000
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
                    "chat_id":                  TELEGRAM_CHAT_ID,
                    "text":                     chunk,
                    "parse_mode":               "Markdown",
                    "disable_web_page_preview": True,
                }, timeout=10)
                r.raise_for_status()
                sent = True
                break
            except Exception as e:
                logging.error(f"Telegram (tentativa {attempt+1}/{retries+1}): {e}")
                if attempt < retries:
                    time.sleep(3)
        if not sent:
            all_ok = False
        elif len(chunks) > 1:
            time.sleep(1)

    return all_ok


# ── Heartbeat das 9h — carteira pessoal ─────────────────────────────────────

def _pnl_emoji(v: float) -> str:
    if v > 0:  return "🟢"
    if v < 0:  return "🔴"
    return "⚪"


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

    total    = snapshot["total_eur"]
    # Às 9h o mercado ainda está fechado — o P&L "hoje" seria sempre 0.
    # Mostramos o P&L do dia anterior (ontem), semanal e mensal confirmados.
    pnl_d    = snapshot["pnl_day"]    # = variação do dia anterior (fecho-1 vs fecho-2)
    pnl_w    = snapshot["pnl_week"]
    pnl_m    = snapshot["pnl_month"]
    pnl_tot  = snapshot["pnl_total"]
    fx       = snapshot["usd_eur"]
    total_cost = snapshot.get("total_cost", 0)

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
        lines.append(
            f"  {_pnl_emoji(pnl_tot)}  *Total investido:*   €{pnl_tot:+,.2f}{_pct(pnl_tot, total_cost)}"
        )

    # Top 3 winners + pior do dia anterior
    movers = sorted(
        [p for p in snapshot["positions"] if p["pnl_day"] is not None],
        key=lambda x: x["pnl_day"],
        reverse=True,
    )
    if movers:
        lines.append("")
        lines.append(f"*📈 Top movers {yesterday}:*")
        for p in movers[:3]:
            base = p["value_eur"] - p["pnl_day"]
            pct  = p["pnl_day"] / base * 100 if base != 0 else 0
            lines.append(
                f"  {_pnl_emoji(p['pnl_day'])} *{p['symbol']}*: "
                f"€{p['pnl_day']:+,.2f} ({pct:+.1f}%)"
            )
        worst = movers[-1]
        if worst["pnl_day"] < 0 and worst["symbol"] != movers[0]["symbol"]:
            base = worst["value_eur"] - worst["pnl_day"]
            pct  = worst["pnl_day"] / base * 100 if base != 0 else 0
            lines.append(
                f"  {_pnl_emoji(worst['pnl_day'])} *{worst['symbol']}*: "
                f"€{worst['pnl_day']:+,.2f} ({pct:+.1f}%) ← pior"
            )

    lines += [
        "",
        f"  📊 PPR (proxy ACWI): €{snapshot['ppr_value']:,.2f}",
        f"  💜 CashBack Pie: €{snapshot['cashback_eur']:,.2f}",
        "",
        "_Mercado abre às 14h30 Lisboa_",
    ]

    send_telegram("\n".join(lines))
    logging.info("Heartbeat enviado.")


# ── Target Sell ───────────────────────────────────────────────────────────────

_BLUECHIP_SECTORS = {"Consumer Defensive", "Utilities", "Financial Services"}
_SECTOR_FLIP_CAP  = {
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

    sector         = fundamentals.get("sector", "")
    sector_cfg     = get_sector_config(sector)
    pe_current     = fundamentals.get("pe") or 0
    pe_fair        = sector_cfg.get("pe_fair", 22)
    analyst_target = fundamentals.get("analyst_target")
    analyst_upside = fundamentals.get("analyst_upside") or 0
    beta           = fundamentals.get("beta") or 1.0
    dividend_yield = fundamentals.get("dividend_yield") or 0

    is_bluechip = sector in _BLUECHIP_SECTORS or bool(dividend_yield and dividend_yield > 0.025)
    if is_bluechip and dip_score >= 8:
        return "HOLD ETERNO", "💎 Blue chip — Adicionar em dips, nunca vender"

    anchors = []
    if pe_current and pe_current > 0 and pe_fair and pe_current < pe_fair:
        anchors.append(("PE rerating", price * (pe_fair / pe_current)))
    if analyst_target and analyst_target > price:
        anchors.append(("Analistas", float(analyst_target)))
    anchors.append(("Beta recovery", price * (1 + (beta or 1.0) * 0.12)))

    weights        = {"PE rerating": 2, "Analistas": 2, "Beta recovery": 1}
    total_w        = sum(weights[n] for n, _ in anchors)
    weighted_price = sum(weights[n] * t for n, t in anchors) / total_w
    weighted_upside = (weighted_price / price) - 1

    sector_cap = _SECTOR_FLIP_CAP.get(sector, 0.30)
    if dip_score >= 9:   sector_cap *= 1.20
    elif dip_score >= 8: sector_cap *= 1.10

    final_upside = min(weighted_upside, sector_cap)
    final_target = price * (1 + final_upside)

    macro_flag = ""
    if spy_change is not None and spy_change <= -2.0:
        macro_flag = f" | 🌍 Queda macro SPY {spy_change:.1f}% — timeline incerta"

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
        _, strategy = calculate_flip_target(f, score, earnings, catalyst, spy_change)
        score_stars = "⭐" * min(int(score // 2), 5)
        tier_badge  = {1: "🔴T1", 2: "🟡T2", 3: "🔵T3"}.get(tier, "")
        lines.append(f"*{i}. {sym}* {tier_badge} | Score {score:.0f} {score_stars}")
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
    if dip_score >= 8:   score_badge = f"🔥 Score: {dip_score:.0f}/10"
    elif dip_score >= 6: score_badge = f"⭐ Score: {dip_score:.0f}/10"
    else:                score_badge = f"📊 Score: {dip_score:.0f}/10"
    rsi_part = f" | RSI: {rsi_str}" if rsi_str else ""
    lines = [
        f"📉 *{symbol} — {name}{region_part}*",
        f"Queda: *{change:.1f}%*{drawdown_str}",
        f"💰 Preço: ${price} | 🏦 Cap: ${mc_b:.1f}B",
        f"🏢 Sector: {sector_cfg.get('label', sector) or sector}",
        f"{score_badge}{rsi_part}",
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
                dip_score, rsi_str = calculate_dip_score(fundamentals, symbol)
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
                    logging.info(f"  ✅ Alerta: {symbol} ({verdict}, score {dip_score})")
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
        mc_b = (s.get("market_cap") or 0) / 1e9
        lines.append(f"  📉 *{s['symbol']}*: {s['change_pct']:.1f}% (${mc_b:.1f}B)")
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
            if _timed_out():
                logging.warning(f"Watchdog: a saltar {sym}")
                return {"skip": True}
            fund_cache[sym] = get_fundamentals(sym, reg, MIN_MARKET_CAP)
        return fund_cache[sym]
    def _get_score(sym, fund):
        if sym not in score_cache:
            score_cache[sym], _ = calculate_dip_score(fund, sym)
        return score_cache[sym]

    tier3 = []
    for s in tier3_candidates:
        if _timed_out(): logging.warning("Watchdog: Tier 3 truncado"); break
        sym  = s["symbol"]
        fund = _get_fund(sym, s.get("region", ""))
        if fund.get("skip"): continue
        score = _get_score(sym, fund)
        if score >= 8:
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
            mc_b     = (s.get("market_cap") or 0) / 1e9
            drawdown = get_52w_drawdown(s["symbol"])
            d_str    = f" | 52w: *{drawdown:.0f}%*" if drawdown is not None else ""
            lines.append(f"  📉 *{s['symbol']}*: {s['change_pct']:.1f}% hoje{d_str} — ${mc_b:.1f}B")
        lines += ["", "_→ Verifica catalisador, FCF e 4 critérios Flip_", ""]
    else:
        lines += [f"_Sem quedas ≥{DROP_THRESHOLD:.0f}% hoje_", ""]

    if tier2:
        lines.append(f"*🟡 TIER 2 — Watchlist (7–{DROP_THRESHOLD:.0f}%):*")
        for s in tier2[:6]:
            mc_b = (s.get("market_cap") or 0) / 1e9
            lines.append(f"  👀 *{s['symbol']}*: {s['change_pct']:.1f}% (${mc_b:.1f}B)")
        lines += ["", "_→ Monitorizar apenas_", ""]

    if tier3:
        lines.append("*🔵 TIER 3 — Gems Raras (-3/-8%, score ≥8):*")
        lines.append("")
        top5       = tier3[:5]
        rest_9plus = [s for s in tier3[5:] if s.get("_score", 0) >= 9]
        for s in top5:
            sym          = s["symbol"]
            score        = s.get("_score", 0)
            fund         = fund_cache.get(sym, {})
            mc_b         = (fund.get("market_cap") or 0) / 1e9
            sector_label = get_sector_config(fund.get("sector", "")).get("label", "") or fund.get("sector", "")
            price        = fund.get("price", 0)
            earnings     = get_earnings_date(sym)
            catalyst     = get_catalyst(sym, fund.get("name", ""))
            _, strategy  = calculate_flip_target(fund, score, earnings, catalyst, spy_change)
            badge        = "🔥" if score >= 9 else "⭐"
            lines.append(f"  {badge} *{sym}* — Score {score:.0f} | ${price} | ${mc_b:.1f}B | {sector_label}")
            lines.append(f"     {strategy}")
            lines.append("")
        if rest_9plus:
            lines.append(f"  _Score 9+ adicionais: {', '.join(s['symbol'] for s in rest_9plus)}_")
            lines.append("")

    ranking_entries = []
    for tier_num, tier_list in [(1, tier1), (2, tier2), (3, tier3)]:
        for s in tier_list:
            sym  = s["symbol"]
            fund = _get_fund(sym, s.get("region", ""))
            if fund.get("skip"): continue
            score = _get_score(sym, fund)
            if score >= 7:
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
    logging.info(f"Scan a cada {SCAN_MINUTES} minutos | Min score: {MIN_DIP_SCORE}")
    logging.info(f"Timezone activo: {datetime.now().strftime('%Z %z')}")
    logging.info("=" * 60)

    send_telegram(
        f"🤖 *DipRadar iniciado*\n"
        f"Tier 1: ≥{DROP_THRESHOLD}% | Tier 2: 7–{DROP_THRESHOLD:.0f}% | Tier 3: 3–8% (score≥8)\n"
        f"Cap mínimo: ${MIN_MARKET_CAP/1e9:.0f}B | Score mínimo: {MIN_DIP_SCORE}/10\n"
        f"Catalisadores Tavily: {'✅' if os.environ.get('TAVILY_API_KEY') else '⚠️ não configurado'}\n"
        f"Heartbeat carteira: ✅ 9h diário (P&L de ontem + semana + mês)\n"
        f"_Scan a cada {SCAN_MINUTES} minutos (só horas de mercado)_"
    )

    schedule.every(SCAN_MINUTES).minutes.do(run_scan)
    schedule.every().day.at("09:00").do(send_heartbeat)
    schedule.every().day.at("15:30").do(send_open_summary)
    schedule.every().day.at("21:15").do(send_close_summary)
    schedule.every().day.at("00:01").do(
        lambda: (_alerted_today.clear(), _save_alerts(_alerted_today))
    )

    run_scan()

    while True:
        schedule.run_pending()
        time.sleep(60)
