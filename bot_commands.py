"""
bot_commands.py — Comandos Telegram para o DipRadar.

Comandos disponíveis:
  /status                  → Estado do bot
  /carteira                → Snapshot instantâneo da carteira
  /scan                    → Força scan imediato (só horas de mercado)
  /analisar <TICK>         → Análise completa de qualquer ticker a pedido
  /comparar <T1> <T2> ...  → Comparar scores de 2-5 tickers lado-a-lado
  /historico <TICK>        → Histórico de scores registados para um ticker
  /backtest                → Resumo do backtest de alertas
  /rejeitados              → Log de rejeitados de hoje
  /tier3                   → Gems Raras do último resumo de fecho (score ≥80)
  /watchlist               → Ver watchlist dinâmica actual
  /watchlist add TICK      → Adicionar ticker à watchlist
  /watchlist rm TICK       → Remover ticker da watchlist
  /watchlist clear         → Limpar toda a watchlist dinâmica
  /help                    → Lista de comandos

Funções chamadas pelo scheduler (main.py):
  send_earnings_alerts()   → Notificação automática de earnings a <7 dias

Uso em main.py:
  Corre start_bot_listener() numa thread separada no arranque.
"""

import os
import time
import logging
import threading
import requests
from datetime import datetime
from rate_limiter import is_allowed, rate_status
from state import (
    load_dynamic_watchlist,
    add_to_dynamic_watchlist,
    remove_from_dynamic_watchlist,
    save_dynamic_watchlist,
    get_ticker_score_history,
)

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Dias de antecedência para o alerta automático de earnings
_EARNINGS_ALERT_DAYS: int = int(os.environ.get("EARNINGS_ALERT_DAYS", 7))

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
_cb_analyze_ticker   = None  # fn(symbol) -> str
_cb_get_fundamentals = None  # fn(symbol) -> dict  (para /comparar)
_cb_earnings_days    = None  # fn(symbol) -> int|None


def register_callbacks(
    send_telegram,
    run_scan,
    get_snapshot,
    backtest_summary,
    rejected_log,
    is_market_open,
    tier3_handler=None,
    analyze_ticker=None,
    get_fundamentals=None,
    earnings_days=None,
) -> None:
    global _cb_send_telegram, _cb_run_scan, _cb_get_snapshot
    global _cb_backtest_summary, _cb_rejected_log, _cb_is_market_open
    global _cb_tier3_handler, _cb_analyze_ticker
    global _cb_get_fundamentals, _cb_earnings_days
    _cb_send_telegram    = send_telegram
    _cb_run_scan         = run_scan
    _cb_get_snapshot     = get_snapshot
    _cb_backtest_summary = backtest_summary
    _cb_rejected_log     = rejected_log
    _cb_is_market_open   = is_market_open
    _cb_tier3_handler    = tier3_handler
    _cb_analyze_ticker   = analyze_ticker
    _cb_get_fundamentals = get_fundamentals
    _cb_earnings_days    = earnings_days


def _reply(text: str) -> None:
    if _cb_send_telegram:
        _cb_send_telegram(text)


def _check_rate(cmd: str) -> bool:
    allowed, wait = is_allowed(cmd)
    if not allowed:
        _reply(
            f"⏳ *Rate limit* — `/{cmd}` muito frequente.\n"
            f"_Tenta novamente em *{wait}s*._"
        )
        logging.info(f"[bot_commands] rate limit: /{cmd} bloqueado ({wait}s)")
    return allowed


# ── /comparar ───────────────────────────────────────────────────────────────────

def _handle_comparar(symbols: list[str]) -> None:
    """
    Compara scores de 2-5 tickers lado-a-lado usando os fundamentais
    e o score_log em memória. Mostra tabela ASCII com os principais
    indicadores para facilitar a decisão entre tickers concorrentes.
    """
    if len(symbols) < 2:
        _reply(
            "⚠️ Usa: `/comparar <TICK1> <TICK2> [TICK3...]`\n"
            "_Exemplo: `/comparar AAPL MSFT GOOGL`_\n"
            "_Máximo 5 tickers de cada vez._"
        )
        return
    if len(symbols) > 5:
        _reply("⚠️ Máximo 5 tickers por comparação.")
        return

    if not _cb_get_fundamentals:
        _reply("_Comparação não disponível — callback get_fundamentals não registado._")
        return

    _reply(f"_🔄 A comparar {' vs '.join(symbols)}... (pode demorar até 30s)_")

    def _run():
        from score import calculate_dip_score
        rows = []
        for sym in symbols:
            try:
                fund  = _cb_get_fundamentals(sym)
                edays = _cb_earnings_days(sym) if _cb_earnings_days else None
                score, rsi_str = calculate_dip_score(fund, sym, edays)
                badge = "🔥" if score >= 80 else ("⭐" if score >= 55 else "📊")
                rows.append({
                    "sym":      sym,
                    "score":    score,
                    "badge":    badge,
                    "rsi":      rsi_str or "N/D",
                    "fcf":      f"{fund.get('fcf_yield',0)*100:.1f}%" if fund.get('fcf_yield') is not None else "N/D",
                    "growth":   f"{fund.get('revenue_growth',0)*100:.1f}%" if fund.get('revenue_growth') is not None else "N/D",
                    "margin":   f"{fund.get('gross_margin',0)*100:.0f}%" if fund.get('gross_margin') is not None else "N/D",
                    "upside":   f"{fund.get('analyst_upside',0):.0f}%" if fund.get('analyst_upside') is not None else "N/D",
                    "drawdown": f"{fund.get('drawdown_from_high',0):.1f}%" if fund.get('drawdown_from_high') is not None else "N/D",
                    "edays":    str(edays) if edays is not None else "N/D",
                    "sector":   (fund.get('sector') or 'N/D')[:14],
                })
            except Exception as e:
                logging.warning(f"[comparar] {sym}: {e}")
                rows.append({"sym": sym, "score": 0, "badge": "❌", "error": str(e)})

        if not rows:
            _reply("_Nenhum dado obtido._")
            return

        # Ordena por score desc
        rows.sort(key=lambda r: r.get("score", 0), reverse=True)

        lines = [f"*🔄 Comparação — {datetime.now().strftime('%d/%m %H:%M')}*", ""]
        for r in rows:
            if r.get("error"):
                lines.append(f"  ❌ *{r['sym']}* — _erro: {r['error']}_")
                continue
            lines.append(
                f"  {r['badge']} *{r['sym']}* — score *{r['score']:.0f}/100*"
            )
            lines.append(
                f"     RSI {r['rsi']} · FCF {r['fcf']} · Growth {r['growth']}"
            )
            lines.append(
                f"     Margin {r['margin']} · Upside {r['upside']} · Drawdown {r['drawdown']}"
            )
            lines.append(
                f"     Sector: _{r['sector']}_ · Earnings: {r['edays']}d"
            )
            lines.append("")

        winner = rows[0]
        if not winner.get("error"):
            lines.append(f"_🏆 Melhor score: *{winner['sym']}* ({winner['score']:.0f} pts) — usa `/analisar {winner['sym']}` para detalhe._")

        _reply("\n".join(lines))

    threading.Thread(target=_run, daemon=True).start()


# ── /historico ──────────────────────────────────────────────────────────────────

def _handle_historico(symbol: str) -> None:
    """
    Mostra as últimas N entradas do score_log para o ticker:
    data, score, verdict, change% e preço ao momento do alerta.
    Inclui trend indicator (subida/descida de score entre entradas).
    """
    history = get_ticker_score_history(symbol)
    if not history:
        _reply(
            f"_Sem histórico de scores para *{symbol}*._ \n"
            "_O ticker ainda não foi analisado ou alertado pelo DipRadar._"
        )
        return

    # Mostra as últimas 10 entradas
    entries = history[-10:]
    lines   = [f"*📈 Histórico — {symbol} ({len(history)} entradas totais):*", ""]

    prev_score = None
    for e in entries:
        score   = e.get("score", 0)
        verdict = e.get("verdict") or ""
        change  = e.get("change", 0)
        price   = e.get("price")
        date    = e.get("date", "")
        t       = e.get("time", "")

        # Trend vs entrada anterior
        if prev_score is None:
            trend = ""
        elif score > prev_score:
            trend = f" ↑{score - prev_score:.0f}"
        elif score < prev_score:
            trend = f" ↓{prev_score - score:.0f}"
        else:
            trend = " ="
        prev_score = score

        badge = "🔥" if score >= 80 else ("⭐" if score >= 55 else "📊")
        price_str = f" @ ${price:.2f}" if price else ""
        lines.append(
            f"  {badge} `{date} {t}` — *{score:.0f}*{trend} | {change:+.1f}%{price_str} | _{verdict}_"
        )

    if len(history) > 10:
        lines.append("")
        lines.append(f"_... e mais {len(history) - 10} entradas anteriores._")

    _reply("\n".join(lines))


# ── Earnings alerts (chamado pelo scheduler do main.py) ──────────────────────

def send_earnings_alerts(watchlist: list[str] | None = None) -> int:
    """
    Verifica a watchlist dinâmica (+ watchlist estática passada em watchlist=)
    e envia um alerta Telegram para cada ticker com earnings nos próximos
    _EARNINGS_ALERT_DAYS dias.

    Deve ser chamado uma vez por dia pelo scheduler do main.py.
    Evita duplicados via _alerted_today set (reset à meia-noite).

    Devolve o número de alertas enviados.
    """
    if not _cb_earnings_days or not _cb_send_telegram:
        return 0

    # Une watchlist dinâmica com estática (sem duplicados)
    dynamic  = load_dynamic_watchlist()
    static   = list(watchlist) if watchlist else []
    all_tickers = list(dict.fromkeys(dynamic + static))  # preserva ordem, remove dups

    if not all_tickers:
        return 0

    today_str = datetime.now().date().isoformat()
    sent      = 0

    for sym in all_tickers:
        try:
            edays = _cb_earnings_days(sym)
            if edays is None:
                continue
            if 0 <= edays <= _EARNINGS_ALERT_DAYS:
                # Deduplicação via score_log: verifica se já alertamos hoje
                history = get_ticker_score_history(sym)
                already = any(
                    e.get("date_iso") == today_str and e.get("verdict", "").startswith("earnings_alert")
                    for e in history
                )
                if already:
                    continue

                urgency = "🚨 *HOJE*" if edays == 0 else (f"⏰ *{edays} dia(s)*" if edays <= 2 else f"📅 {edays} dia(s)")
                last_score = history[-1].get("score") if history else None
                score_str  = f" — último score: *{last_score:.0f}/100*" if last_score is not None else ""

                _reply(
                    f"📊 *Earnings Alert* — *{sym}*\n"
                    f"Resultados em: {urgency}{score_str}\n"
                    f"_Considera `/analisar {sym}` antes dos resultados._"
                )
                sent += 1
                logging.info(f"[earnings_alert] enviado: {sym} ({edays}d)")

        except Exception as e:
            logging.warning(f"[earnings_alert] {sym}: {e}")

    return sent


# ── /watchlist handler ────────────────────────────────────────────────────────

def _handle_watchlist(parts: list[str]) -> None:
    sub = parts[1].lower() if len(parts) > 1 else "list"

    if sub in ("list", "ls", "show", "ver"):
        tickers = load_dynamic_watchlist()
        if not tickers:
            _reply("*👀 Watchlist dinâmica*\n_Está vazia. Usa `/watchlist add TICKER` para adicionar._")
            return
        lines = [f"*👀 Watchlist dinâmica ({len(tickers)} tickers):*", ""]
        for i, t in enumerate(tickers, 1):
            lines.append(f"  {i}. `{t}`")
        lines.append("")
        lines.append("_Remove com `/watchlist rm TICKER` · Limpa com `/watchlist clear`_")
        _reply("\n".join(lines))

    elif sub in ("add", "adicionar", "+"):
        if len(parts) < 3:
            _reply("⚠️ Uso: `/watchlist add <TICKER>`\n_Exemplo: `/watchlist add NVDA`_")
            return
        ticker = parts[2].upper().strip().split(".")[0]
        if len(ticker) > 10 or not ticker.isalpha():
            _reply(f"⚠️ Ticker inválido: `{ticker}` — usa letras apenas (ex: AAPL, NVDA).")
            return
        added = add_to_dynamic_watchlist(ticker)
        if added:
            total = len(load_dynamic_watchlist())
            _reply(f"✅ *`{ticker}`* adicionado à watchlist.\n_Total: {total} tickers._")
            logging.info(f"[watchlist] adicionado: {ticker}")
        else:
            _reply(f"_`{ticker}` já está na watchlist._")

    elif sub in ("rm", "remove", "remover", "del", "delete", "-"):
        if len(parts) < 3:
            _reply("⚠️ Uso: `/watchlist rm <TICKER>`\n_Exemplo: `/watchlist rm NVDA`_")
            return
        ticker = parts[2].upper().strip()
        removed = remove_from_dynamic_watchlist(ticker)
        if removed:
            total = len(load_dynamic_watchlist())
            _reply(f"🗑️ *`{ticker}`* removido da watchlist.\n_Restam {total} tickers._")
            logging.info(f"[watchlist] removido: {ticker}")
        else:
            _reply(f"⚠️ `{ticker}` não está na watchlist.")

    elif sub in ("clear", "limpar", "reset"):
        tickers = load_dynamic_watchlist()
        count   = len(tickers)
        if count == 0:
            _reply("_A watchlist já está vazia._")
            return
        save_dynamic_watchlist([])
        _reply(f"🧹 Watchlist limpa. _{count} ticker(s) removido(s)._\n_Usa `/watchlist add TICKER` para recomeçar._")
        logging.info(f"[watchlist] clear: {count} tickers removidos")

    else:
        _reply(
            f"⚠️ Sub-comando desconhecido: `{sub}`\n\n"
            "*Uso:*\n`/watchlist`          → Ver lista\n"
            "`/watchlist add TICK` → Adicionar\n"
            "`/watchlist rm TICK`  → Remover\n"
            "`/watchlist clear`    → Limpar tudo"
        )


# ── Command router ───────────────────────────────────────────────────────────

def _handle_command(text: str) -> None:
    parts   = text.strip().split()
    cmd     = parts[0].lower() if parts else ""
    cmd     = cmd.split("@")[0]  # remove bot mention
    cmd_key = cmd.lstrip("/")

    if cmd in ("/help", "/start"):
        _reply(
            "*🤖 DipRadar — Comandos disponíveis:*\n\n"
            "`/status`                  → Estado do bot\n"
            "`/carteira`                → Snapshot da carteira\n"
            "`/scan`                    → Forçar scan\n"
            "`/analisar <TICK>`         → Análise completa\n"
            "`/comparar <T1> <T2> ...`  → Comparar 2-5 tickers\n"
            "`/historico <TICK>`        → Histórico de scores\n"
            "`/backtest`                → Resumo backtesting\n"
            "`/rejeitados`              → Rejeitados de hoje\n"
            "`/tier3`                   → Gems Raras (score ≥80)\n"
            "`/watchlist`               → Ver watchlist dinâmica\n"
            "`/watchlist add TICK`      → Adicionar ticker\n"
            "`/watchlist rm TICK`       → Remover ticker\n"
            "`/watchlist clear`         → Limpar watchlist\n"
            "`/help`                    → Esta mensagem"
        )

    elif cmd == "/status":
        if not _check_rate(cmd_key): return
        uptime     = datetime.now() - _bot_start_time
        hours, rem = divmod(int(uptime.total_seconds()), 3600)
        mins       = rem // 60
        market     = "🟢 Aberto" if (_cb_is_market_open and _cb_is_market_open()) else "🔴 Fechado"
        wl         = load_dynamic_watchlist()
        _reply(
            f"*🤖 DipRadar Status*\n"
            f"Uptime: *{hours}h {mins}m*\n"
            f"Mercado: *{market}*\n"
            f"Watchlist dinâmica: *{len(wl)} tickers*\n"
            f"_⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}_\n"
            f"_{rate_status()}_"
        )

    elif cmd == "/carteira":
        if not _check_rate(cmd_key): return
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
        if not _check_rate(cmd_key): return
        if _cb_is_market_open and not _cb_is_market_open():
            _reply("⚠️ Mercado fechado — scan não disponível fora do horário.")
            return
        _reply("_🔍 A iniciar scan manual..._")
        try:
            if _cb_run_scan:
                threading.Thread(target=_cb_run_scan, daemon=True).start()
        except Exception as e:
            _reply(f"_Erro no scan: {e}_")

    elif cmd == "/analisar":
        if not _check_rate(cmd_key): return
        if len(parts) < 2:
            _reply("⚠️ Usa: `/analisar <TICKER>`\n_Exemplo: `/analisar AAPL`_")
            return
        symbol = parts[1].upper().strip()
        if not _cb_analyze_ticker:
            _reply("_Análise não disponível._")
            return
        _reply(f"_🔍 A analisar *{symbol}*... (pode demorar 10–20s)_")
        try:
            threading.Thread(
                target=lambda: _reply(_cb_analyze_ticker(symbol)),
                daemon=True,
            ).start()
        except Exception as e:
            _reply(f"_Erro ao analisar {symbol}: {e}_")

    elif cmd == "/comparar":
        if not _check_rate(cmd_key): return
        symbols = [p.upper().strip() for p in parts[1:]]
        try:
            _handle_comparar(symbols)
        except Exception as e:
            _reply(f"_Erro na comparação: {e}_")
            logging.exception("[bot_commands] /comparar error")

    elif cmd == "/historico":
        if not _check_rate(cmd_key): return
        if len(parts) < 2:
            _reply("⚠️ Usa: `/historico <TICKER>`\n_Exemplo: `/historico NVDA`_")
            return
        symbol = parts[1].upper().strip()
        try:
            _handle_historico(symbol)
        except Exception as e:
            _reply(f"_Erro no histórico: {e}_")

    elif cmd == "/backtest":
        if not _check_rate(cmd_key): return
        if not _cb_backtest_summary:
            _reply("_Backtest não disponível._")
            return
        try:
            _reply(_cb_backtest_summary())
        except Exception as e:
            _reply(f"_Erro no backtest: {e}_")

    elif cmd == "/rejeitados":
        if not _check_rate(cmd_key): return
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
        if not _check_rate(cmd_key): return
        try:
            _reply(_cb_tier3_handler() if _cb_tier3_handler else "🔵 *Tier 3* — _Handler não registado._")
        except Exception as e:
            _reply(f"_Erro ao obter Tier 3: {e}_")

    elif cmd == "/watchlist":
        if not _check_rate(cmd_key): return
        try:
            _handle_watchlist(parts)
        except Exception as e:
            _reply(f"_Erro na watchlist: {e}_")
            logging.exception("[bot_commands] /watchlist error")

    else:
        if text.startswith("/"):
            _reply(f"_Comando desconhecido: `{cmd}` — usa /help_")


# ── Poll loop ──────────────────────────────────────────────────────────────────────

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
                    uid             = update["update_id"]
                    _last_update_id = max(_last_update_id, uid)
                    msg             = update.get("message", {})
                    chat            = str(msg.get("chat", {}).get("id", ""))
                    text            = msg.get("text", "")
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
