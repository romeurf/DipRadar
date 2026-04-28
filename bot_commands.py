"""
bot_commands.py — Comandos Telegram para o DipRadar.

Comandos disponíveis:
  /status     → Estado do bot (uptime, próximo scan, mercado aberto/fechado)
  /carteira   → Snapshot instantâneo da carteira
  /scan       → Força scan imediato (só horas de mercado)
  /backtest   → Resumo do backtest de alertas
  /rejeitados → Log de rejeitados de hoje
  /tier3      → Gems Raras do último resumo de fecho (score ≥16)
  /help       → Lista de comandos

Uso em main.py:
  Corre start_bot_listener() numa thread separada no arranque.
"""

import os
import time
import logging
import threading
import requests
from datetime import datetime

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

_last_update_id: int = 0
_bot_start_time: datetime = datetime.now()

# Callbacks injectados pelo main.py
_cb_send_telegram    = None  # fn(msg) -> bool
_cb_run_scan         = None  # fn() -> None
_cb_get_snapshot     = None  # fn() -> dict
_cb_backtest_summary = None  # fn() -> str
_cb_rejected_log     = None  # fn() -> list
_cb_is_market_open   = None  # fn() -> bool
_cb_tier3_handler    = None  # fn() -> str


def register_callbacks(
    send_telegram,
    run_scan,
    get_snapshot,
    backtest_summary,
    rejected_log,
    is_market_open,
    tier3_handler=None,
) -> None:
    global _cb_send_telegram, _cb_run_scan, _cb_get_snapshot
    global _cb_backtest_summary, _cb_rejected_log, _cb_is_market_open, _cb_tier3_handler
    _cb_send_telegram    = send_telegram
    _cb_run_scan         = run_scan
    _cb_get_snapshot     = get_snapshot
    _cb_backtest_summary = backtest_summary
    _cb_rejected_log     = rejected_log
    _cb_is_market_open   = is_market_open
    _cb_tier3_handler    = tier3_handler


def _reply(text: str) -> None:
    if _cb_send_telegram:
        _cb_send_telegram(text)


def _handle_command(text: str) -> None:
    cmd = text.strip().lower().split()[0] if text.strip() else ""

    if cmd in ("/help", "/start"):
        _reply(
            "*🤖 DipRadar — Comandos disponíveis:*\n\n"
            "`/status`     → Estado do bot\n"
            "`/carteira`   → Snapshot da carteira agora\n"
            "`/scan`       → Forçar scan imediato\n"
            "`/backtest`   → Resumo backtesting\n"
            "`/rejeitados` → Rejeitados de hoje\n"
            "`/tier3`      → Gems Raras do último fecho (score ≥16)\n"
            "`/help`       → Esta mensagem"
        )

    elif cmd == "/status":
        uptime = datetime.now() - _bot_start_time
        hours, rem = divmod(int(uptime.total_seconds()), 3600)
        mins = rem // 60
        market = "🟢 Aberto" if (_cb_is_market_open and _cb_is_market_open()) else "🔴 Fechado"
        _reply(
            f"*🤖 DipRadar Status*\n"
            f"Uptime: *{hours}h {mins}m*\n"
            f"Mercado: *{market}*\n"
            f"_⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}_"
        )

    elif cmd == "/carteira":
        if not _cb_get_snapshot:
            _reply("_Snapshot não disponível._")
            return
        try:
            snap  = _cb_get_snapshot()
            total = snap.get("total_eur", 0)
            pnl_d = snap.get("pnl_day", 0)
            pnl_w = snap.get("pnl_week", 0)
            pnl_m = snap.get("pnl_month", 0)
            fx    = snap.get("usd_eur", 0)
            def _e(v): return "🟢" if v > 0 else ("🔴" if v < 0 else "⚪")
            _reply(
                f"*💼 Carteira — {datetime.now().strftime('%d/%m %H:%M')}*\n"
                f"_USD/EUR: {fx:.4f}_\n\n"
                f"*Total: €{total:,.2f}*\n\n"
                f"  {_e(pnl_d)} Hoje:   €{pnl_d:+,.2f}\n"
                f"  {_e(pnl_w)} Semana: €{pnl_w:+,.2f}\n"
                f"  {_e(pnl_m)} Mês:    €{pnl_m:+,.2f}\n"
                f"  📊 PPR: €{snap.get('ppr_value',0):,.2f} | 💜 Pie: €{snap.get('cashback_eur',0):,.2f}"
            )
        except Exception as e:
            _reply(f"_Erro ao calcular carteira: {e}_")

    elif cmd == "/scan":
        if _cb_is_market_open and not _cb_is_market_open():
            _reply("⚠️ Mercado fechado — scan não disponível fora do horário.")
            return
        _reply("_🔍 A iniciar scan manual..._")
        try:
            if _cb_run_scan:
                threading.Thread(target=_cb_run_scan, daemon=True).start()
        except Exception as e:
            _reply(f"_Erro no scan: {e}_")

    elif cmd == "/backtest":
        if not _cb_backtest_summary:
            _reply("_Backtest não disponível._")
            return
        try:
            summary = _cb_backtest_summary()
            _reply(summary)
        except Exception as e:
            _reply(f"_Erro no backtest: {e}_")

    elif cmd == "/rejeitados":
        if not _cb_rejected_log:
            _reply("_Log não disponível._")
            return
        try:
            rejected = _cb_rejected_log()
            if not rejected:
                _reply("_Nenhum stock rejeitado hoje._")
                return
            lines = [f"*🗑️ Rejeitados hoje ({len(rejected)}):*", ""]
            for r in sorted(rejected, key=lambda x: x.get("score") or 0, reverse=True)[:15]:
                score_str   = f" | score {r['score']:.0f}" if r.get("score") is not None else ""
                verdict_str = f" | {r['verdict']}" if r.get("verdict") else ""
                lines.append(
                    f"  ⛔ *{r['symbol']}* {r['change']:+.1f}% | "
                    f"_{r['reason']}{score_str}{verdict_str}_ | {r.get('time','')}"
                )
            _reply("\n".join(lines))
        except Exception as e:
            _reply(f"_Erro: {e}_")

    elif cmd == "/tier3":
        try:
            if _cb_tier3_handler:
                _reply(_cb_tier3_handler())
            else:
                _reply("🔵 *Tier 3* — _Handler não registado._")
        except Exception as e:
            _reply(f"_Erro ao obter Tier 3: {e}_")

    else:
        if text.startswith("/"):
            _reply(f"_Comando desconhecido: `{cmd}` — usa /help_")


def _poll_loop() -> None:
    global _last_update_id
    if not TELEGRAM_TOKEN:
        logging.warning("[bot_commands] TELEGRAM_TOKEN não configurado — listener inactivo.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    logging.info("[bot_commands] Listener iniciado.")
    while True:
        try:
            r = requests.get(
                url,
                params={"offset": _last_update_id + 1, "timeout": 30, "allowed_updates": ["message"]},
                timeout=35,
            )
            if r.ok:
                for update in r.json().get("result", []):
                    uid  = update["update_id"]
                    _last_update_id = max(_last_update_id, uid)
                    msg  = update.get("message", {})
                    chat = str(msg.get("chat", {}).get("id", ""))
                    text = msg.get("text", "")
                    if chat == TELEGRAM_CHAT_ID and text:
                        logging.info(f"[bot_commands] Comando recebido: {text!r}")
                        _handle_command(text)
        except Exception as e:
            logging.warning(f"[bot_commands] poll error: {e}")
            time.sleep(10)


def start_bot_listener() -> threading.Thread:
    """Inicia o listener de comandos numa daemon thread."""
    t = threading.Thread(target=_poll_loop, daemon=True, name="bot-commands")
    t.start()
    return t
