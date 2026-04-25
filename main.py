"""
Stock Alert Bot
Trigger: Yahoo Finance day_losers (gratuito)
Fundamentais: yfinance (gratuito)
Deploy: Railway.app
"""
import os
import time
import logging
import schedule
import requests
from datetime import datetime

from market_client import screen_big_drops, get_fundamentals, get_news, get_historical_pe
from sectors import get_sector_config, score_fundamentals
from valuation import format_valuation_block

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
DROP_THRESHOLD   = float(os.environ.get("DROP_THRESHOLD", "10"))
MIN_MARKET_CAP   = int(os.environ.get("MIN_MARKET_CAP", "2000000000"))
SCAN_MINUTES     = int(os.environ.get("SCAN_EVERY_MINUTES", "30"))

_alerted_today: set = set()


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


def build_alert(stock: dict, fundamentals: dict, historical_pe: float | None,
                news: list, verdict: str, emoji: str, reasons: list) -> str:
    sector     = fundamentals.get("sector", "")
    sector_cfg = get_sector_config(sector)
    name       = fundamentals.get("name", stock["symbol"])
    change     = stock["change_pct"]
    price      = fundamentals.get("price") or stock.get("price", "N/D")
    mc_b       = (fundamentals.get("market_cap") or 0) / 1e9

    lines = [
        f"📉 *{stock['symbol']} — {name}*",
        f"Queda: *{change:.1f}%* | Preço: ${price} | Cap: ${mc_b:.1f}B",
        f"Sector: {sector_cfg.get('label', sector)}",
        "",
        f"*Veredito: {emoji} {verdict}*",
    ]
    for r in reasons:
        lines.append(f"  _{r}_")

    lines += ["", "*📊 Fundamentos:*",
              format_valuation_block(fundamentals, historical_pe, sector)]

    lines += ["", "*📰 Notícias:*"]
    for item in news[:3]:
        t = item["title"][:70]
        u = item["url"]
        s = item.get("source", "")
        lines.append(f"  • [{t}]({u})" + (f" _{s}_" if s else ""))

    lines.append(f"\n_⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}_")
    return "\n".join(lines)


def run_scan() -> None:
    today = datetime.now().date().isoformat()
    logging.info(f"A correr scan — {datetime.now().strftime('%H:%M')}")

    losers = screen_big_drops(
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
            fundamentals = get_fundamentals(symbol)

            if fundamentals.get("skip"):
                _alerted_today.add(alert_key)
                continue

            sector         = fundamentals.get("sector", "")
            verdict, emoji, reasons = score_fundamentals(fundamentals, sector)

            if verdict == "EVITAR":
                logging.info(f"  {symbol}: EVITAR — a saltar")
                _alerted_today.add(alert_key)
                continue

            historical_pe = get_historical_pe(symbol)
            news          = get_news(symbol)
            message       = build_alert(stock, fundamentals, historical_pe,
                                        news, verdict, emoji, reasons)

            if send_telegram(message):
                _alerted_today.add(alert_key)
                logging.info(f"  ✅ Alerta enviado: {symbol} ({verdict})")

            time.sleep(2)

        except Exception as e:
            logging.error(f"Erro {symbol}: {e}")


def send_daily_summary() -> None:
    losers = screen_big_drops(min_drop_pct=3.0, min_market_cap=MIN_MARKET_CAP)
    if not losers:
        return
    lines = [f"*📋 Resumo — {datetime.now().strftime('%d/%m/%Y %H:%M')}*", ""]
    for s in losers[:10]:
        mc_b = (s.get("market_cap") or 0) / 1e9
        lines.append(f"📉 *{s['symbol']}*: {s['change_pct']:.1f}% (${mc_b:.1f}B)")
    send_telegram("\n".join(lines))


if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("Stock Alert Bot iniciado")
    logging.info(f"Trigger: Yahoo Finance | Threshold: {DROP_THRESHOLD}% | Min cap: ${MIN_MARKET_CAP/1e9:.0f}B")
    logging.info(f"Scan a cada {SCAN_MINUTES} minutos")
    logging.info("=" * 60)

    send_telegram(
        f"🤖 *Bot iniciado*\n"
        f"Trigger: Yahoo Finance (gratuito)\n"
        f"Threshold: ≥{DROP_THRESHOLD}% | Cap mínimo: ${MIN_MARKET_CAP/1e9:.0f}B\n"
        f"_Scan a cada {SCAN_MINUTES} minutos_"
    )

    schedule.every(SCAN_MINUTES).minutes.do(run_scan)
    schedule.every().day.at("18:00").do(send_daily_summary)
    schedule.every().day.at("00:01").do(_alerted_today.clear)

    run_scan()

    while True:
        schedule.run_pending()
        time.sleep(60)
