"""
Stock Alert Bot — monitoriza quedas de 10%+ e analisa fundamentos por sector.
Deploy: Railway.app ou Render.com (gratuito)

Variáveis de ambiente necessárias:
  TELEGRAM_TOKEN   — token do @BotFather
  TELEGRAM_CHAT_ID — o teu chat ID
  FMP_API_KEY      — API key da Financial Modeling Prep (gratuita)
"""
import os
import time
import logging
import schedule
import requests
from datetime import datetime

from fmp_client import screen_big_drops, get_fundamentals, get_news, get_historical_pe
from sectors import get_sector_config, score_fundamentals
from valuation import format_valuation_block

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
FMP_API_KEY      = os.environ.get("FMP_API_KEY", "demo")

# Threshold de queda para alertar (%)
DROP_THRESHOLD   = float(os.environ.get("DROP_THRESHOLD", "10"))

# Market cap mínimo para filtrar micro-caps (default 1B)
MIN_MARKET_CAP   = int(os.environ.get("MIN_MARKET_CAP", "1000000000"))

# Cache de alertas do dia
_alerted_today: set = set()


# ── Telegram ─────────────────────────────────────────────────────────────────

def send_telegram(message: str, parse_mode: str = "Markdown") -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(message)
        return True
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message[:4096],  # Telegram limit
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        logging.error(f"Telegram error: {e}")
        return False


# ── Alert formatter ───────────────────────────────────────────────────────────

def build_alert(stock: dict, fundamentals: dict, historical_pe: float | None,
                news: list, verdict: str, emoji: str, reasons: list) -> str:

    sector = fundamentals.get("sector", "")
    sector_cfg = get_sector_config(sector)
    name = fundamentals.get("name", stock["symbol"])
    change = stock["change_pct"]
    price = fundamentals.get("price") or stock.get("price", "N/D")
    mc_b = (fundamentals.get("market_cap") or 0) / 1e9

    header = (
        f"📉 *{stock['symbol']} — {name}*\n"
        f"Queda: *{change:.1f}%* hoje | Preço: ${price} | Cap: ${mc_b:.1f}B\n"
        f"Sector: {sector_cfg.get('label', sector)}\n"
    )

    verdict_block = f"\n*Veredito: {emoji} {verdict}*\n"
    if reasons:
        verdict_block += "\n".join(f"  _{r}_" for r in reasons) + "\n"

    valuation_block = "\n*📊 Fundamentos:*\n"
    valuation_block += format_valuation_block(fundamentals, historical_pe, sector)

    news_block = "\n\n*📰 Notícias:*\n"
    if news:
        for item in news[:3]:
            title = item["title"][:70]
            url = item["url"]
            src = item.get("source", "")
            news_block += f"  • [{title}]({url})" + (f" _{src}_" if src else "") + "\n"
    else:
        news_block += "  • Sem notícias recentes\n"

    footer = f"\n_⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}_"

    return header + verdict_block + valuation_block + news_block + footer


# ── Main scan ────────────────────────────────────────────────────────────────

def run_scan() -> None:
    today = datetime.now().date().isoformat()
    logging.info(f"A correr scan — {datetime.now().strftime('%H:%M')}")

    losers = screen_big_drops(
        min_drop_pct=DROP_THRESHOLD,
        min_market_cap=MIN_MARKET_CAP,
    )

    if not losers:
        logging.info("Nenhuma acção com queda suficiente.")
        return

    logging.info(f"{len(losers)} acção(ões) com queda ≥{DROP_THRESHOLD}%")

    alerts_sent = 0
    for stock in losers:
        symbol = stock.get("symbol")
        if not symbol:
            continue

        # Não alertar duas vezes no mesmo dia
        alert_key = f"{symbol}_{today}"
        if alert_key in _alerted_today:
            continue

        try:
            logging.info(f"A analisar {symbol} ({stock['change_pct']:.1f}%)...")

            fundamentals = get_fundamentals(symbol)
            sector = fundamentals.get("sector", "")

            # Score fundamentals pelo sector
            verdict, emoji, reasons = score_fundamentals(fundamentals, sector)

            # Só alertar se COMPRAR ou MONITORIZAR — evitar spam de value traps
            if verdict == "EVITAR":
                logging.info(f"  {symbol}: EVITAR — a saltar")
                _alerted_today.add(alert_key)
                continue

            historical_pe = get_historical_pe(symbol)
            news = get_news(symbol)

            message = build_alert(stock, fundamentals, historical_pe, news, verdict, emoji, reasons)
            success = send_telegram(message)

            if success:
                _alerted_today.add(alert_key)
                alerts_sent += 1
                logging.info(f"  ✅ Alerta enviado: {symbol} ({verdict})")

            time.sleep(1.5)  # evitar rate limit

        except Exception as e:
            logging.error(f"Erro a processar {symbol}: {e}")
            continue

    if alerts_sent == 0 and losers:
        logging.info("Acções com queda encontradas mas fundamentos fracos — sem alertas.")


def send_daily_summary() -> None:
    """Resumo diário das maiores quedas."""
    losers = screen_big_drops(min_drop_pct=3.0, min_market_cap=MIN_MARKET_CAP)
    if not losers:
        return

    lines = [f"*📋 Resumo — {datetime.now().strftime('%d/%m/%Y %H:%M')}*\n"]
    lines.append(f"Acções com queda ≥3% hoje:\n")

    for s in losers[:10]:
        sym = s.get("symbol", "?")
        chg = s.get("change_pct", 0)
        mc_b = (s.get("market_cap") or 0) / 1e9
        lines.append(f"  📉 *{sym}*: {chg:.1f}% (cap ${mc_b:.1f}B)")

    send_telegram("\n".join(lines))


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.info("=" * 50)
    logging.info("Stock Alert Bot iniciado")
    logging.info(f"Threshold: {DROP_THRESHOLD}% | Min cap: ${MIN_MARKET_CAP/1e9:.1f}B")
    logging.info("=" * 50)

    send_telegram(
        f"🤖 *Bot iniciado*\n"
        f"Threshold: ≥{DROP_THRESHOLD}% de queda\n"
        f"Análise por sector com DCF, FCF, P/E histórico\n"
        f"_Scan a cada 30 minutos_"
    )

    # Horários
    schedule.every(30).minutes.do(run_scan)
    schedule.every().day.at("18:00").do(send_daily_summary)
    schedule.every().day.at("00:01").do(_alerted_today.clear)

    # Scan imediato ao arrancar
    run_scan()

    while True:
        schedule.run_pending()
        time.sleep(60)
