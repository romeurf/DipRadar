"""
Stock Alert Bot
Trigger: Yahoo Finance day_losers (gratuito)
Fundamentais: yfinance (gratuito)
Deploy: Railway.app

Variáveis Railway obrigatórias:
  TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
  TZ=Europe/Lisbon
Variáveis opcionais:
  DROP_THRESHOLD=10         (% queda mínima para Tier 1)
  MIN_MARKET_CAP=2000000000
  SCAN_EVERY_MINUTES=30
  MIN_DIP_SCORE=4           (score mínimo quantitativo para enviar alerta)
"""

import os
import time
import logging
import schedule
import requests
from datetime import datetime
from market_client import (
    screen_global_dips, get_fundamentals, get_news,
    get_historical_pe, get_52w_drawdown,
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
DROP_THRESHOLD   = float(os.environ.get("DROP_THRESHOLD", "10"))
MIN_MARKET_CAP   = int(os.environ.get("MIN_MARKET_CAP", "2000000000"))
SCAN_MINUTES     = int(os.environ.get("SCAN_EVERY_MINUTES", "30"))
MIN_DIP_SCORE    = int(os.environ.get("MIN_DIP_SCORE", "4"))

_alerted_today: set = set()


# ── Telegram ──────────────────────────────────────────────────────────────────

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


# ── Alerta individual (scan contínuo) ─────────────────────────────────────────

def buildalertstock(stock: dict, fundamentals: dict, historical_pe: float | None,
                   news: list, verdict: str, emoji: str, reasons: list,
                   dip_score: int, rsi_str: str | None) -> str:

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

    # Score badge
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

            # Score quantitativo — filtro adicional
            dip_score, rsi_str = calculate_dip_score(fundamentals, symbol)
            if dip_score < MIN_DIP_SCORE:
                logging.info(f"  {symbol}: score {dip_score} < {MIN_DIP_SCORE} — a saltar")
                _alerted_today.add(alert_key)
                continue

            historical_pe = get_historical_pe(symbol)
            news          = get_news(symbol)
            message       = buildalertstock(stock, fundamentals, historical_pe,
                                            news, verdict, emoji, reasons,
                                            dip_score, rsi_str)

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


# ── Resumo fecho (21h15 Lisboa) ───────────────────────────────────────────────

def send_close_summary() -> None:
    all_losers = screen_global_dips(
        min_drop_pct=7.0,
        min_market_cap=MIN_MARKET_CAP,
    )
    if not all_losers:
        return

    tier1 = [s for s in all_losers if s["change_pct"] <= -DROP_THRESHOLD]
    tier2 = [s for s in all_losers if -DROP_THRESHOLD < s["change_pct"] <= -7.0]

    lines = [
        f"*📋 Resumo Fecho — {datetime.now().strftime('%d/%m/%Y')}*",
        "",
    ]

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

    if tier2:
        lines.append("*🟡 TIER 2 — Watchlist (7–10%):*")
        for s in tier2[:6]:
            mc_b = (s.get("market_cap") or 0) / 1e9
            region = s.get("region", "")
            region_str = f" ({region})" if region else ""
            lines.append(f"  👀 *{s['symbol']}*{region_str}: {s['change_pct']:.1f}% (${mc_b:.1f}B)")
        lines += [
            "",
            "_→ Tier 2: monitorizar apenas, sem acção imediata_",
        ]

    send_telegram("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("Stock Alert Bot iniciado")
    logging.info(f"Threshold: {DROP_THRESHOLD}% | Min cap: ${MIN_MARKET_CAP/1e9:.0f}B")
    logging.info(f"Scan a cada {SCAN_MINUTES} minutos | Min score: {MIN_DIP_SCORE}")
    logging.info(f"Timezone activo: {datetime.now().strftime('%Z %z')}")
    logging.info("=" * 60)

    send_telegram(
        f"🤖 *Bot iniciado*\n"
        f"Threshold Tier 1: ≥{DROP_THRESHOLD}% | Tier 2: 7–{DROP_THRESHOLD:.0f}%\n"
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
