"""
DipRadar — Stock Alert Bot
Trigger: Yahoo Finance day_losers (gratuito)
Fundamentais: yfinance (gratuito)
Deploy: Railway.app

Variáveis Railway obrigatórias:
  TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
  TZ=Europe/Lisbon
Variáveis opcionais:
  DROP_THRESHOLD=8          (% queda mínima para Tier 1)
  MIN_MARKET_CAP=2000000000
  SCAN_EVERY_MINUTES=30
  MIN_DIP_SCORE=5           (score mínimo quantitativo para enviar alerta)
"""

import os
import time
import logging
import schedule
import requests
from datetime import datetime
from market_client import (
    screen_global_dips, get_fundamentals, get_news,
    get_historical_pe, get_52w_drawdown, get_earnings_date,
)
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

_alerted_today: set = set()


# ── Telegram ───────────────────────────────────────────────────────────────────

def send_telegram(message: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(message)
        return True
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message[:4096],
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        logging.error(f"Telegram: {e}")
        return False


# ── Target Sell robusto (estratégia Flip 2-3 meses) ───────────────────────────

# Sectores considerados blue chip / hold eterno (sem flip forçado)
_BLUECHIP_SECTORS = {"Consumer Defensive", "Utilities", "Financial Services"}

# Cap de upside realista em 2-3 meses por sector (baseado em beta médio)
_SECTOR_FLIP_CAP = {
    "Technology": 0.55,
    "Communication Services": 0.45,
    "Consumer Cyclical": 0.40,
    "Healthcare": 0.35,
    "Industrials": 0.30,
    "Basic Materials": 0.30,
    "Energy": 0.28,
    "Real Estate": 0.25,
    "Financial Services": 0.22,
    "Consumer Defensive": 0.18,
    "Utilities": 0.15,
}

def calculate_flip_target(
    fundamentals: dict,
    dip_score: float,
    earnings_date: str | None = None,
) -> tuple[str, str]:
    """
    Devolve (target_str, strategy_label).

    Lógica (3 âncoras):
      1. PE rerating: preço justo se o PE reverter ao pe_fair do sector
      2. Analyst target: consenso externo de 20-30 analistas
      3. Beta recovery: upside realista dado o beta e um recovery de mercado de +12%
         (estimativa conservadora de recuperação de mercado após dip)

    Target final = min(âncoras válidas) com cap por sector.
    Blue chips (Consumer Defensive, Utilities, Financials) → HOLD ETERNO.
    """
    price = fundamentals.get("price") or 0
    if not price or price <= 0:
        return "N/D", "SEM DADOS"

    sector = fundamentals.get("sector", "")
    sector_cfg = get_sector_config(sector)
    pe_current = fundamentals.get("pe") or 0
    pe_fair = sector_cfg.get("pe_fair", 22)
    analyst_target = fundamentals.get("analyst_target")
    beta = fundamentals.get("beta") or 1.0
    dividend_yield = fundamentals.get("dividend_yield") or 0

    # ── Blue chip: nunca flip forçado ────────────────────────────────────────
    is_bluechip = sector in _BLUECHIP_SECTORS or (dividend_yield and dividend_yield > 0.025)
    if is_bluechip and dip_score >= 8:
        return "HOLD ETERNO", "💎 Blue chip — Adicionar em dips, nunca vender"

    # ── Âncora 1: PE rerating ────────────────────────────────────────────────
    anchors = []
    if pe_current and pe_current > 0 and pe_current < pe_fair:
        # só calcula se estiver abaixo do fair (verdadeiro dip de valuation)
        pe_target_price = price * (pe_fair / pe_current)
        pe_upside = (pe_target_price / price) - 1
        anchors.append(("PE rerating", pe_target_price, pe_upside))

    # ── Âncora 2: Analyst target ─────────────────────────────────────────────
    if analyst_target and analyst_target > price:
        analyst_upside = (analyst_target / price) - 1
        anchors.append(("Analistas", analyst_target, analyst_upside))

    # ── Âncora 3: Beta recovery (+12% mercado em recovery, conservador) ──────
    market_recovery = 0.12
    beta_target_price = price * (1 + beta * market_recovery)
    beta_upside = (beta_target_price / price) - 1
    anchors.append(("Beta recovery", beta_target_price, beta_upside))

    if not anchors:
        return "N/D", "SEM DADOS"

    # ── Target = média ponderada das âncoras disponíveis ─────────────────────
    # PE rerating peso 2x (mais fundamental), analistas 2x, beta 1x
    weights = {"PE rerating": 2, "Analistas": 2, "Beta recovery": 1}
    total_w = sum(weights[name] for name, _, _ in anchors)
    weighted_price = sum(
        weights[name] * target for name, target, _ in anchors
    ) / total_w
    weighted_upside = (weighted_price / price) - 1

    # ── Cap por sector ───────────────────────────────────────────────────────
    sector_cap = _SECTOR_FLIP_CAP.get(sector, 0.30)
    if dip_score >= 9:
        sector_cap *= 1.20   # gem rara: cap 20% mais alto
    elif dip_score >= 8:
        sector_cap *= 1.10

    final_upside = min(weighted_upside, sector_cap)
    final_target = price * (1 + final_upside)

    # ── Catalisador de earnings ───────────────────────────────────────────────
    catalyst_str = ""
    if earnings_date:
        catalyst_str = f" | ✅ Earnings {earnings_date}"
    else:
        catalyst_str = " | ⚠️ Sem catalisador próximo"

    strategy = f"🎯 Flip target: ${final_target:.1f} (+{final_upside*100:.0f}%){catalyst_str}"
    return f"${final_target:.1f} (+{final_upside*100:.0f}%)", strategy


# ── Ranking Flip (todos os tiers com score ≥7) ─────────────────────────────────

def build_flip_ranking(ranked_entries: list[dict]) -> str:
    """
    ranked_entries: lista de dicts com keys:
      symbol, dip_score, fundamentals, tier, earnings_date
    Ordenado por dip_score DESC.
    """
    if not ranked_entries:
        return ""

    lines = ["", "*🏆 RANKING FLIP — Top compras de hoje*", ""]
    top = sorted(ranked_entries, key=lambda x: x["dip_score"], reverse=True)[:8]

    for i, entry in enumerate(top, 1):
        sym = entry["symbol"]
        score = entry["dip_score"]
        tier = entry["tier"]
        f = entry["fundamentals"]
        earnings = entry.get("earnings_date")
        price = f.get("price", 0)
        mc_b = (f.get("market_cap") or 0) / 1e9

        target_str, strategy = calculate_flip_target(f, score, earnings)

        score_stars = "⭐" * min(int(score // 2), 5)
        tier_badge = {1: "🔴T1", 2: "🟡T2", 3: "🔵T3"}.get(tier, "")

        lines.append(
            f"*{i}. {sym}* {tier_badge} | Score {score:.1f} {score_stars}"
        )
        lines.append(f"   💰 ${price} | 🏦 ${mc_b:.1f}B")
        lines.append(f"   {strategy}")
        lines.append("")

    return "\n".join(lines)


# ── Alerta individual ──────────────────────────────────────────────────────────

def build_alert(
    stock: dict,
    fundamentals: dict,
    historical_pe: dict | None,
    news: list,
    verdict: str,
    emoji: str,
    reasons: list,
    dip_score: float,
    rsi_str: str | None,
) -> str:

    sector = fundamentals.get("sector", "")
    sector_cfg = get_sector_config(sector)
    name = fundamentals.get("name", stock.get("name", stock.get("symbol", "N/A")))

    symbol = stock["symbol"]
    change = stock["change_pct"]
    price = fundamentals.get("price") or stock.get("price", "N/D")
    mc_b = (fundamentals.get("market_cap") or 0) / 1e9

    drawdown = fundamentals.get("drawdown_from_high")
    drawdown_str = f" | 52w: {drawdown:.0f}%" if drawdown is not None else ""

    region = stock.get("region")
    region_part = f" ({region})" if region else ""

    if dip_score >= 8:
        score_badge = f"🔥 Score: {dip_score}/10"
    elif dip_score >= 6:
        score_badge = f"⭐ Score: {dip_score}/10"
    else:
        score_badge = f"📊 Score: {dip_score}/10"

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
            title = item["title"][:70]
            url = item["url"]
            source = item.get("source", "")
            lines.append(f"  [{title}]({url}){' _' + source + '_ ' if source else ''}")

    lines.append(f"_⏰ {datetime.now().strftime('%d/%m %H:%M')}_")
    return "\n".join(lines)


# ── Scan contínuo ──────────────────────────────────────────────────────────────

def run_scan() -> None:
    today = datetime.now().date().isoformat()
    logging.info(f"A correr scan — {datetime.now().strftime('%H:%M')}")

    losers = screen_global_dips(
        min_drop_pct=DROP_THRESHOLD,
        min_market_cap=MIN_MARKET_CAP,
    )
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
            fundamentals = get_fundamentals(symbol, stock.get("region", ""))
            if fundamentals.get("skip"):
                _alerted_today.add(alert_key)
                continue

            sector = fundamentals.get("sector", "")
            verdict, emoji, reasons = score_fundamentals(fundamentals, sector)

            if verdict == "EVITAR":
                logging.info(f"  {symbol}: EVITAR — a saltar")
                _alerted_today.add(alert_key)
                continue

            dip_score, rsi_str = calculate_dip_score(fundamentals, symbol)
            if dip_score < MIN_DIP_SCORE:
                logging.info(f"  {symbol}: score {dip_score} < {MIN_DIP_SCORE} — a saltar")
                _alerted_today.add(alert_key)
                continue

            historical_pe = get_historical_pe(symbol)
            news          = get_news(symbol)
            message       = build_alert(
                stock, fundamentals, historical_pe,
                news, verdict, emoji, reasons,
                dip_score, rsi_str,
            )

            if send_telegram(message):
                _alerted_today.add(alert_key)
                logging.info(f"  ✅ Alerta enviado: {symbol} ({verdict}, score {dip_score})")
            time.sleep(5)

        except Exception as e:
            logging.error(f"Erro {symbol}: {e}")


# ── Resumo abertura + 1h (15h30 Lisboa) ───────────────────────────────────────

def send_open_summary() -> None:
    tier1 = screen_global_dips(
        min_drop_pct=DROP_THRESHOLD,
        min_market_cap=MIN_MARKET_CAP,
    )
    if not tier1:
        return

    lines = [
        f"*⚡ Abertura +1h — {datetime.now().strftime('%d/%m %H:%M')}*",
        f"_{len(tier1)} candidato(s) com queda ≥{DROP_THRESHOLD:.0f}% — investiga antes do fecho_",
        "",
    ]
    for s in tier1[:8]:
        mc_b = (s.get("market_cap") or 0) / 1e9
        region = s.get("region", "")
        region_str = f" ({region})" if region else ""
        lines.append(f"  📉 *{s['symbol']}*{region_str}: {s['change_pct']:.1f}% (${mc_b:.1f}B)")

    lines += [
        "",
        "_Resumo completo com drawdown 52w às 21h15_",
    ]
    send_telegram("\n".join(lines))


# ── Resumo fecho (21h15 Lisboa) ────────────────────────────────────────────────

def send_close_summary() -> None:
    # Scan amplo: apanha Tier 1 (>=DROP_THRESHOLD), Tier 2 (7–DROP), Tier 3 (3–8)
    all_losers = screen_global_dips(
        min_drop_pct=3.0,
        min_market_cap=MIN_MARKET_CAP,
    )
    if not all_losers:
        return

    tier1 = [s for s in all_losers if s["change_pct"] <= -DROP_THRESHOLD]
    tier2 = [s for s in all_losers if -DROP_THRESHOLD < s["change_pct"] <= -7.0]
    tier3_candidates = [s for s in all_losers if -8.0 < s["change_pct"] <= -3.0]

    # Cache de fundamentals partilhado entre todos os tiers
    fund_cache: dict[str, dict] = {}
    score_cache: dict[str, float] = {}

    def _get_fund(sym, region=""):
        if sym not in fund_cache:
            fund_cache[sym] = get_fundamentals(sym, region)
        return fund_cache[sym]

    def _get_score(sym, fund):
        if sym not in score_cache:
            score_cache[sym], _ = calculate_dip_score(fund, sym)
        return score_cache[sym]

    # ── Tier 3: filtrar por score ≥8 ──────────────────────────────────────────
    tier3 = []
    for s in tier3_candidates:
        sym = s["symbol"]
        fund = _get_fund(sym, s.get("region", ""))
        if fund.get("skip"):
            continue
        score = _get_score(sym, fund)
        if score >= 8:
            s["_score"] = score
            tier3.append(s)
    tier3.sort(key=lambda x: x.get("_score", 0), reverse=True)

    lines = [
        f"*📋 Resumo Fecho — {datetime.now().strftime('%d/%m/%Y')}*",
        "",
    ]

    # ── TIER 1 ────────────────────────────────────────────────────────────────
    if tier1:
        lines.append(f"*🔴 TIER 1 — Análise completa (≥{DROP_THRESHOLD:.0f}%):*")
        for s in tier1[:6]:
            mc_b = (s.get("market_cap") or 0) / 1e9
            sym  = s["symbol"]
            region = s.get("region", "")
            region_str = f" ({region})" if region else ""
            drawdown = get_52w_drawdown(sym)
            d_str = f" | 52w: *{drawdown:.0f}%*" if drawdown is not None else ""
            lines.append(
                f"  📉 *{sym}*{region_str}: {s['change_pct']:.1f}% hoje{d_str} — ${mc_b:.1f}B"
            )
        lines += [
            "",
            "_→ Para cada Tier 1: verifica catalisador, FCF e 4 critérios Flip_",
            "",
        ]
    else:
        lines += [f"_Sem quedas ≥{DROP_THRESHOLD:.0f}% hoje_", ""]

    # ── TIER 2 ────────────────────────────────────────────────────────────────
    if tier2:
        lines.append(f"*🟡 TIER 2 — Watchlist (7–{DROP_THRESHOLD:.0f}%):*")
        for s in tier2[:6]:
            mc_b = (s.get("market_cap") or 0) / 1e9
            region = s.get("region", "")
            region_str = f" ({region})" if region else ""
            lines.append(f"  👀 *{s['symbol']}*{region_str}: {s['change_pct']:.1f}% (${mc_b:.1f}B)")
        lines += [
            "",
            "_→ Tier 2: monitorizar apenas, sem acção imediata_",
            "",
        ]

    # ── TIER 3: TOP 5 detalhadas + lista score 9+ restantes ───────────────────
    if tier3:
        lines.append("*🔵 TIER 3 — Gems Raras (-3/-8%, score ≥8):*")
        lines.append("")

        top5 = tier3[:5]
        rest_9plus = [s for s in tier3[5:] if s.get("_score", 0) >= 9]

        for s in top5:
            sym   = s["symbol"]
            score = s.get("_score", 0)
            fund  = fund_cache.get(sym, {})
            mc_b  = (fund.get("market_cap") or 0) / 1e9
            sector = fund.get("sector", "")
            sector_cfg = get_sector_config(sector)
            sector_label = sector_cfg.get("label", sector) or sector
            price = fund.get("price", 0)

            earnings = get_earnings_date(sym)
            target_str, strategy = calculate_flip_target(fund, score, earnings)

            score_badge = "🔥" if score >= 9 else "⭐"
            lines.append(f"  {score_badge} *{sym}* — Score {score:.1f} | ${price} | ${mc_b:.1f}B | {sector_label}")
            lines.append(f"     {strategy}")
            lines.append("")

        if rest_9plus:
            tickers_9 = ", ".join(s["symbol"] for s in rest_9plus)
            lines.append(f"  _Score 9+ adicionais: {tickers_9}_")
            lines.append("")

    # ── RANKING FLIP (todos os tiers com score ≥7) ────────────────────────────
    ranking_entries = []

    for tier_num, tier_list in [(1, tier1), (2, tier2), (3, tier3)]:
        for s in tier_list:
            sym  = s["symbol"]
            fund = _get_fund(sym, s.get("region", ""))
            if fund.get("skip"):
                continue
            score = _get_score(sym, fund)
            if score >= 7:
                earnings = get_earnings_date(sym)
                ranking_entries.append({
                    "symbol": sym,
                    "dip_score": score,
                    "fundamentals": fund,
                    "tier": tier_num,
                    "earnings_date": earnings,
                })

    ranking_block = build_flip_ranking(ranking_entries)
    if ranking_block:
        lines.append(ranking_block)

    send_telegram("\n".join(lines))


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("DipRadar iniciado")
    logging.info(f"Threshold: {DROP_THRESHOLD}% | Min cap: ${MIN_MARKET_CAP/1e9:.0f}B")
    logging.info(f"Scan a cada {SCAN_MINUTES} minutos | Min score: {MIN_DIP_SCORE}")
    logging.info(f"Timezone activo: {datetime.now().strftime('%Z %z')}")
    logging.info("=" * 60)

    send_telegram(
        f"🤖 *DipRadar iniciado*\n"
        f"Threshold Tier 1: ≥{DROP_THRESHOLD}% | Tier 2: 7–{DROP_THRESHOLD:.0f}% | Tier 3: 3–8% (score≥8)\n"
        f"Cap mínimo: ${MIN_MARKET_CAP/1e9:.0f}B | Score mínimo: {MIN_DIP_SCORE}/10\n"
        f"Resumos: 15h30 (abertura) e 21h15 (fecho) Lisboa\n"
        f"_Scan a cada {SCAN_MINUTES} minutos_"
    )

    schedule.every(SCAN_MINUTES).minutes.do(run_scan)
    schedule.every().day.at("15:30").do(send_open_summary)
    schedule.every().day.at("21:15").do(send_close_summary)
    schedule.every().day.at("00:01").do(_alerted_today.clear)

    run_scan()

    while True:
        schedule.run_pending()
        time.sleep(60)
