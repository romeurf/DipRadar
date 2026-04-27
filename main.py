"""
Stock Alert Bot
Trigger: Yahoo Finance day_losers (gratuito)
Fundamentais: yfinance (gratuito)
Deploy: Railway.app

Variáveis Railway obrigatórias:
  TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
  TZ=Europe/Lisbon          ← NOVO: resolve o problema do resumo chegar 1h tarde
Variáveis opcionais:
  DROP_THRESHOLD=10         (% queda mínima para Tier 1)
  MIN_MARKET_CAP=2000000000
  SCAN_EVERY_MINUTES=30
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
                   news: list, verdict: str, emoji: str, reasons: list) -> str:
    
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
    
    lines = [
        f"📉 *{symbol} — {name}{region_part}*",
        f"Queda: *{change:.1f}%*{drawdown_str}",
        f"💰 Preço: ${price} | 🏦 Cap: ${mc_b:.1f}B",
        f"🏢 Sector: {sector_cfg.get('label', sector) or sector}",
        "",
        f"*🟢 Veredito: {emoji} {verdict}*",
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
    return "\\n".join(lines)


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

            historical_pe = get_historical_pe(symbol)
            news          = get_news(symbol)
            message       = buildalertstock(stock, fundamentals, historical_pe,
                                        news, verdict, emoji, reasons)

            if send_telegram(message):
                _alerted_today.add(alert_key)
                logging.info(f"  ✅ Alerta enviado: {symbol} ({verdict})")
            time.sleep(2)

        except Exception as e:
            logging.error(f"Erro {symbol}: {e}")


# ── Resumo abertura + 1h (15h30 Lisboa) ───────────────────────────────────────
# NYSE abre 9h30 ET = 14h30 Lisboa. Às 15h30 a poeira do open já assentou.
# Mostra só Tier 1 (≥10%) — candidatos para investigares durante o dia.

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
        lines.append(f"  📉 *{s['symbol']}*: {s['change_pct']:.1f}% (${mc_b:.1f}B)")

    lines += [
        "",
        "_Resumo completo com drawdown 52w às 21h15_",
    ]
    send_telegram("\n".join(lines))


# ── Resumo fecho (21h15 Lisboa) ───────────────────────────────────────────────
# NYSE fecha 16h00 ET = 21h00 Lisboa. 15 minutos depois os números são definitivos.
# Dois tiers + drawdown desde o máximo de 52 semanas para cada Tier 1.

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

    # ── Tier 1 ──
    if tier1:
        lines.append(f"*🔴 TIER 1 — Análise completa (≥{DROP_THRESHOLD:.0f}%):*")
        for s in tier1[:6]:
            mc_b = (s.get("market_cap") or 0) / 1e9
            sym  = s["symbol"]

            # Drawdown desde máximo 52 semanas (só para Tier 1 — máx 6 stocks)
            drawdown = get_52w_drawdown(sym)
            d_str = f" | 52w: *{drawdown:.0f}%*" if drawdown is not None else ""

            lines.append(
                f"  📉 *{sym}*: {s['change_pct']:.1f}% hoje{d_str} — ${mc_b:.1f}B"
            )
        lines += [
            "",
            "_→ Para cada Tier 1: verifica catalisador, FCF e 4 critérios Flip_",
            "",
        ]
    else:
        lines += [f"_Sem quedas ≥{DROP_THRESHOLD:.0f}% hoje_", ""]

    # ── Tier 2 ──
    if tier2:
        lines.append("*🟡 TIER 2 — Watchlist (7–10%):*")
        for s in tier2[:6]:
            mc_b = (s.get("market_cap") or 0) / 1e9
            lines.append(f"  👀 *{s['symbol']}*: {s['change_pct']:.1f}% (${mc_b:.1f}B)")
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
    logging.info(f"Scan a cada {SCAN_MINUTES} minutos")
    logging.info(f"Timezone activo: {datetime.now().strftime('%Z %z')}")
    logging.info("=" * 60)

    send_telegram(
        f"🤖 *Bot iniciado*\n"
        f"Threshold Tier 1: ≥{DROP_THRESHOLD}% | Tier 2: 7–{DROP_THRESHOLD:.0f}%\n"
        f"Cap mínimo: ${MIN_MARKET_CAP/1e9:.0f}B\n"
        f"Resumos: 15h30 (abertura) e 21h15 (fecho) Lisboa\n"
        f"_Scan a cada {SCAN_MINUTES} minutos_"
    )

    # Scan contínuo
    schedule.every(SCAN_MINUTES).minutes.do(run_scan)

    # Resumo abertura +1h — 15h30 Lisboa (NYSE open + 1h)
    schedule.every().day.at("15:30").do(send_open_summary)

    # Resumo fecho definitivo — 21h15 Lisboa (NYSE close + 15min)
    schedule.every().day.at("21:15").do(send_close_summary)

    # Reset diário de alertas
    schedule.every().day.at("00:01").do(_alerted_today.clear)

    # Scan imediato ao arrancar
    run_scan()

    while True:
        schedule.run_pending()
        time.sleep(60)
