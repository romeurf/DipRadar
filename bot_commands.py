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
  /flip                    → Ver log e P&L do Flip Fund
  /flip add TICK ENTRY SHARES [NOTA]   → Registar entrada num trade
  /flip close ID EXIT                  → Fechar trade pelo ID com preço de saída
  /flip del ID                         → Apagar trade pelo ID
  /mldata                  → Estatísticas da base de dados ML + forçar update de outcomes
  /help                    → Lista de comandos
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
    # Flip Fund
    load_flip_log,
    add_flip_trade,
    close_flip_trade,
    delete_flip_trade,
    get_flip_summary,
)

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

_EARNINGS_ALERT_DAYS: int = int(os.environ.get("EARNINGS_ALERT_DAYS", 7))

_last_update_id: int = 0
_bot_start_time: datetime = datetime.now()

# Callbacks injectados pelo main.py via register_callbacks() ou poll()
_cb_send_telegram    = None
_cb_run_scan         = None
_cb_get_snapshot     = None
_cb_backtest_summary = None
_cb_rejected_log     = None
_cb_is_market_open   = None
_cb_tier3_handler    = None
_cb_analyze_ticker   = None
_cb_get_fundamentals = None
_cb_earnings_days    = None
_cb_get_db_stats     = None
_cb_fill_db_outcomes = None


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
    get_db_stats=None,
    fill_db_outcomes=None,
) -> None:
    global _cb_send_telegram, _cb_run_scan, _cb_get_snapshot
    global _cb_backtest_summary, _cb_rejected_log, _cb_is_market_open
    global _cb_tier3_handler, _cb_analyze_ticker
    global _cb_get_fundamentals, _cb_earnings_days
    global _cb_get_db_stats, _cb_fill_db_outcomes
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
    _cb_get_db_stats     = get_db_stats
    _cb_fill_db_outcomes = fill_db_outcomes


# ── poll() bridge ────────────────────────────────────────────────────────────────────
# O main.py chama poll() periodicamente via schedule (a cada 10s).
# Este bridge lê os updates pendentes do Telegram e despacha comandos.
# A função também re-injeta os callbacks do contexto antes de processar.

def poll(
    token: str,
    chat_id: str,
    send_fn,
    context: dict,
) -> None:
    """
    Chamado pelo scheduler do main.py a cada 10 segundos.
    Lê até 20 updates pendentes do Telegram e despacha para _handle_command.
    
    context: dict com callbacks e utilitários injectados pelo main.py:
      get_snapshot, DIRECT_TICKERS, MIN_MARKET_CAP, ..., get_db_stats, fill_db_outcomes
    """
    global _last_update_id
    if not token:
        return

    # Re-injectar callbacks do contexto
    global _cb_send_telegram, _cb_get_snapshot, _cb_get_db_stats, _cb_fill_db_outcomes
    _cb_send_telegram    = send_fn
    _cb_get_snapshot     = context.get("get_snapshot")
    _cb_backtest_summary_fn = context.get("build_backtest_summary")
    _cb_get_db_stats     = context.get("get_db_stats")
    _cb_fill_db_outcomes = context.get("fill_db_outcomes")

    # Guardar contexto completo para handlers que precisam
    _poll_context.update(context)
    _poll_context["send_fn"] = send_fn

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        r = requests.get(
            url,
            params={"offset": _last_update_id + 1, "timeout": 0, "limit": 20},
            timeout=5,
        )
        if not r.ok:
            return
        for update in r.json().get("result", []):
            uid             = update["update_id"]
            _last_update_id = max(_last_update_id, uid)
            msg  = update.get("message", {})
            cid  = str(msg.get("chat", {}).get("id", ""))
            text = msg.get("text", "")
            if cid == chat_id and text:
                logging.info(f"[bot_commands] Comando: {text!r}")
                _handle_command(text)
    except Exception as e:
        logging.debug(f"[bot_commands] poll error: {e}")


# Context partilhado entre poll() e handlers
_poll_context: dict = {}


def _reply(text: str) -> None:
    fn = _poll_context.get("send_fn") or _cb_send_telegram
    if fn:
        fn(text)


def _check_rate(cmd: str) -> bool:
    allowed, wait = is_allowed(cmd)
    if not allowed:
        _reply(
            f"⏳ *Rate limit* — `/{cmd}` muito frequente.\n"
            f"_Tenta novamente em *{wait}s*._"
        )
        logging.info(f"[bot_commands] rate limit: /{cmd} bloqueado ({wait}s)")
    return allowed


# ── /comparar ───────────────────────────────────────────────────────────────────────────

def _handle_comparar(symbols: list[str]) -> None:
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

    get_fundamentals = _poll_context.get("get_fundamentals") or _cb_get_fundamentals
    get_earnings_days = _poll_context.get("get_earnings_days") or _cb_earnings_days

    if not get_fundamentals:
        _reply("_Comparação não disponível — callback get_fundamentals não registado._")
        return

    _reply(f"_🔄 A comparar {' vs '.join(symbols)}... (pode demorar até 30s)_")

    def _run():
        from score import calculate_dip_score
        rows = []
        for sym in symbols:
            try:
                fund  = get_fundamentals(sym)
                edays = get_earnings_days(sym) if get_earnings_days else None
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

        rows.sort(key=lambda r: r.get("score", 0), reverse=True)
        lines = [f"*🔄 Comparação — {datetime.now().strftime('%d/%m %H:%M')}*", ""]
        for r in rows:
            if r.get("error"):
                lines.append(f"  ❌ *{r['sym']}* — _erro: {r['error']}_")
                continue
            lines.append(f"  {r['badge']} *{r['sym']}* — score *{r['score']:.0f}/100*")
            lines.append(f"     RSI {r['rsi']} · FCF {r['fcf']} · Growth {r['growth']}")
            lines.append(f"     Margin {r['margin']} · Upside {r['upside']} · Drawdown {r['drawdown']}")
            lines.append(f"     Sector: _{r['sector']}_ · Earnings: {r['edays']}d")
            lines.append("")

        winner = rows[0]
        if not winner.get("error"):
            lines.append(f"_🏆 Melhor score: *{winner['sym']}* ({winner['score']:.0f} pts) — usa `/analisar {winner['sym']}` para detalhe._")

        _reply("\n".join(lines))

    threading.Thread(target=_run, daemon=True).start()


# ── /historico ──────────────────────────────────────────────────────────────────────────

def _handle_historico(symbol: str) -> None:
    history = get_ticker_score_history(symbol)
    if not history:
        _reply(
            f"_Sem histórico de scores para *{symbol}*._ \n"
            "_O ticker ainda não foi analisado ou alertado pelo DipRadar._"
        )
        return

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


# ── /mldata ──────────────────────────────────────────────────────────────────────────

def _handle_mldata(force_update: bool = False) -> None:
    """
    /mldata          → Mostra estatísticas da base de dados ML
    /mldata update   → Força fill_db_outcomes() e mostra resultado
    """
    get_db_stats     = _poll_context.get("get_db_stats") or _cb_get_db_stats
    fill_db_outcomes = _poll_context.get("fill_db_outcomes") or _cb_fill_db_outcomes

    if not get_db_stats:
        _reply("_Base de dados ML não disponível. Reinicia o bot._")
        return

    def _run():
        # Forçar update se pedido
        update_stats = None
        if force_update and fill_db_outcomes:
            _reply("🔄 _A actualizar outcomes da alert_db... (pode demorar 1-2 min)_")
            try:
                update_stats = fill_db_outcomes()
            except Exception as e:
                _reply(f"⚠️ Erro no update: `{e}`")
                return

        try:
            stats    = get_db_stats()
            total    = stats.get("total", 0)
            labeled  = stats.get("labeled", 0)
            outcomes = stats.get("outcomes", {})
            by_cat   = stats.get("by_category", {})
            by_vrd   = stats.get("by_verdict", {})
            db_path  = stats.get("db_path", "N/D")
        except Exception as e:
            _reply(f"_Erro ao ler stats: {e}_")
            return

        lines = [
            f"🤖 *ML Alert Database*",
            f"_{datetime.now().strftime('%d/%m/%Y %H:%M')}_",
            "",
            f"*🗓️ Total de alertas:* {total}",
            f"*📊 Classificados:* {labeled} ({labeled/total*100:.0f}% do total)" if total > 0 else "*📊 Classificados:* 0",
            f"*🗂️ Ficheiro:* `{db_path}`",
            "",
        ]

        if by_cat:
            lines.append("*🏠 Por categoria:*")
            cat_emojis = {
                "🏗️ Hold Forever": "🏗️",
                "🏠 Apartamento":   "🏠",
                "🔄 Rotação":       "🔄",
            }
            for cat, count in sorted(by_cat.items(), key=lambda x: x[1], reverse=True):
                pct = count / total * 100 if total > 0 else 0
                lines.append(f"  {cat}: *{count}* ({pct:.0f}%)")
            lines.append("")

        if by_vrd:
            lines.append("*📊 Por verdict:*")
            vrd_em = {"COMPRAR": "🟢", "MONITORIZAR": "🟡", "EVITAR": "🔴"}
            for vrd, count in sorted(by_vrd.items(), key=lambda x: x[1], reverse=True):
                em  = vrd_em.get(vrd, "📊")
                pct = count / total * 100 if total > 0 else 0
                lines.append(f"  {em} {vrd}: *{count}* ({pct:.0f}%)")
            lines.append("")

        if outcomes:
            lines.append("*🎯 Outcomes (alertas com histórico suficiente):*")
            emoji_map = {"WIN_40": "🟢", "WIN_20": "✅", "NEUTRAL": "🟡", "LOSS_15": "🔴"}
            for label, count in sorted(outcomes.items(), key=lambda x: x[1], reverse=True):
                em  = emoji_map.get(label, "📊")
                pct = count / labeled * 100 if labeled > 0 else 0
                lines.append(f"  {em} {label}: *{count}* ({pct:.0f}%)")
            lines.append("")

            # Win rate calculado
            wins   = outcomes.get("WIN_40", 0) + outcomes.get("WIN_20", 0)
            losses = outcomes.get("LOSS_15", 0)
            if labeled > 0:
                win_rate = wins / labeled * 100
                lines.append(f"  📈 *Win rate total:* {win_rate:.0f}% ({wins}/{labeled})")
                if losses > 0:
                    lines.append(f"  ⚠️ *Value Traps detectadas:* {losses} ({losses/labeled*100:.0f}%)")
            lines.append("")

        # Se houve update, mostrar stats do update
        if update_stats:
            lines.append(f"*✅ Update concluído:* {update_stats.get('updated', 0)} novas classificações")
            if update_stats.get('errors', 0) > 0:
                lines.append(f"  ⚠️ Erros: {update_stats['errors']}")
            lines.append("")

        # Progresso para treino do modelo
        if total == 0:
            lines.append("🕐 _Base de dados vazia — aguarda os primeiros alertas._")
        elif labeled < 50:
            lines.append(f"🕐 _Precisa de mais *{50 - labeled}* alertas classificados para treinar o modelo ML._")
            lines.append("_Update automático todos os domingos às 08:00._")
        elif labeled < 100:
            lines.append(f"🟡 *{labeled} alertas classificados* — podes começar a explorar o modelo!")
            lines.append("_Para resultados sólidos, aguarda 100+._")
        else:
            lines.append(f"🟢 *{labeled} alertas classificados* — *pronto para treinar o modelo!*")
            lines.append("_Usa `train_model.py` com XGBoost/sklearn._")

        if not force_update and fill_db_outcomes:
            lines.append("")
            lines.append("_Força update agora com `/mldata update`_")

        _reply("\n".join(lines))

    threading.Thread(target=_run, daemon=True).start()


# ── Earnings alerts (chamado pelo scheduler do main.py) ────────────────────────

def send_earnings_alerts(watchlist: list[str] | None = None) -> int:
    earnings_days_fn = _poll_context.get("get_earnings_days") or _cb_earnings_days
    if not earnings_days_fn or not (_poll_context.get("send_fn") or _cb_send_telegram):
        return 0

    dynamic     = load_dynamic_watchlist()
    static      = list(watchlist) if watchlist else []
    all_tickers = list(dict.fromkeys(dynamic + static))

    if not all_tickers:
        return 0

    today_str = datetime.now().date().isoformat()
    sent      = 0

    for sym in all_tickers:
        try:
            edays = earnings_days_fn(sym)
            if edays is None:
                continue
            if 0 <= edays <= _EARNINGS_ALERT_DAYS:
                history = get_ticker_score_history(sym)
                already = any(
                    e.get("date_iso") == today_str and e.get("verdict", "").startswith("earnings_alert")
                    for e in history
                )
                if already:
                    continue

                urgency    = "🚨 *HOJE*" if edays == 0 else (f"⏰ *{edays} dia(s)*" if edays <= 2 else f"📅 {edays} dia(s)")
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


# ── /watchlist handler ────────────────────────────────────────────────────────────────

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


# ── /flip handler ───────────────────────────────────────────────────────────────────

def _handle_flip(parts: list[str]) -> None:
    sub = parts[1].lower() if len(parts) > 1 else "summary"

    # ── Resumo / lista ───────────────────────────────────────────────────────────
    if sub in ("summary", "list", "ls", "ver", "show") or len(parts) == 1:
        summary = get_flip_summary()
        lines = [
            f"*🎯 Flip Fund — {datetime.now().strftime('%d/%m/%Y')}*",
            "",
        ]
        pnl      = summary["total_pnl"]
        pnl_sign = "+" if pnl >= 0 else ""
        pnl_em   = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
        lines.append(f"  {pnl_em} *P&L realizado:* ${pnl_sign}{pnl:.2f}")
        lines.append(f"  📊 Trades fechados: *{summary['n_closed']}* | Win rate: *{summary['win_rate']:.0f}%*")

        if summary["best_trade"]:
            b = summary["best_trade"]
            lines.append(f"  🏆 Melhor: *{b['symbol']}* ${b['pnl_eur']:+.2f} ({b['date_entry']} → {b['date_exit'] or '?'})")
        if summary["worst_trade"] and summary["n_closed"] > 1:
            w = summary["worst_trade"]
            lines.append(f"  ⚠️ Pior: *{w['symbol']}* ${w['pnl_eur']:+.2f} ({w['date_entry']} → {w['date_exit'] or '?'})")

        opened = summary["trades_open"]
        if opened:
            lines += ["", f"*📂 Posições abertas ({len(opened)}):*"]
            for t in opened:
                notes_str = f" — _{t['notes']}_" if t.get("notes") else ""
                lines.append(
                    f"  #{t['id']} *{t['symbol']}* x{t['shares']} @ ${t['price_entry']:.2f}"
                    f" (desde {t['date_entry']}){notes_str}"
                )
        else:
            lines += ["", "_Sem posições abertas._"]

        if sub in ("list", "ls"):
            closed = summary["trades_closed"]
            if closed:
                lines += ["", f"*✅ Trades fechados ({len(closed)}):*"]
                for t in sorted(closed, key=lambda x: x["date_exit"] or "", reverse=True)[:10]:
                    pnl_s = f"${t['pnl_eur']:+.2f}" if t["pnl_eur"] is not None else "N/D"
                    em    = "🟢" if (t["pnl_eur"] or 0) > 0 else "🔴"
                    lines.append(
                        f"  {em} #{t['id']} *{t['symbol']}* x{t['shares']} "
                        f"${t['price_entry']:.2f}→${t['price_exit']:.2f} | *{pnl_s}* "
                        f"({t['date_entry']} → {t['date_exit']})"
                    )
                if len(closed) > 10:
                    lines.append(f"  _... e mais {len(closed) - 10} trades anteriores._")

        lines += [
            "",
            "_`/flip add TICK ENTRY SHARES` → Registar entrada_",
            "_`/flip close ID EXIT` → Fechar trade_",
        ]
        _reply("\n".join(lines))
        return

    # ── /flip add ──────────────────────────────────────────────────────────────
    if sub in ("add", "entrada", "open"):
        if len(parts) < 5:
            _reply(
                "⚠️ Uso: `/flip add <TICKER> <PREÇO_ENTRADA> <SHARES> [nota]`\n"
                "_Exemplo: `/flip add NVDA 105.50 10`_"
            )
            return
        ticker = parts[2].upper().strip()
        try:
            entry  = float(parts[3])
            shares = float(parts[4])
        except ValueError:
            _reply("⚠️ Preço e shares têm de ser números.")
            return
        if entry <= 0 or shares <= 0:
            _reply("⚠️ Preço de entrada e shares têm de ser positivos.")
            return
        notes = " ".join(parts[5:]) if len(parts) > 5 else ""
        trade = add_flip_trade(ticker, shares, entry, notes=notes)
        cost  = round(entry * shares, 2)
        _reply(
            f"✅ *Flip trade registado!*\n"
            f"  #{trade['id']} *{ticker}* — {shares} shares @ ${entry:.2f}\n"
            f"  💰 Custo total: *${cost:.2f}*\n"
            f"  📅 Data: {trade['date_entry']}\n"
            f"  _Fecha com `/flip close {trade['id']} <PREÇO_SAÍDA>`_"
            + (f"\n  📝 Nota: _{notes}_" if notes else "")
        )
        return

    # ── /flip close ────────────────────────────────────────────────────────────
    if sub in ("close", "fechar", "sell"):
        if len(parts) < 4:
            _reply("⚠️ Uso: `/flip close <ID> <PREÇO_SAÍDA>`\n_Exemplo: `/flip close 3 121.80`_")
            return
        try:
            trade_id = int(parts[2])
            exit_px  = float(parts[3])
        except ValueError:
            _reply("⚠️ ID tem de ser inteiro e preço um número.")
            return
        if exit_px <= 0:
            _reply("⚠️ Preço de saída tem de ser positivo.")
            return
        trade = close_flip_trade(trade_id, exit_px)
        if trade is None:
            _reply(f"⚠️ Trade `#{trade_id}` não encontrado ou já está fechado.")
            return
        pnl    = trade["pnl_eur"]
        pct    = (exit_px - trade["price_entry"]) / trade["price_entry"] * 100
        em     = "🟢" if pnl > 0 else "🔴"
        result = "Lucro" if pnl > 0 else "Perda"
        _reply(
            f"{em} *Flip trade fechado!*\n"
            f"  #{trade_id} *{trade['symbol']}* x{trade['shares']}\n"
            f"  Entrada: ${trade['price_entry']:.2f} → Saída: ${exit_px:.2f}\n"
            f"  *{result}: ${pnl:+.2f}* ({pct:+.1f}%)\n"
            f"  _Período: {trade['date_entry']} → {trade['date_exit']}_"
        )
        return

    # ── /flip del ───────────────────────────────────────────────────────────────
    if sub in ("del", "delete", "rm", "apagar", "remover"):
        if len(parts) < 3:
            _reply("⚠️ Uso: `/flip del <ID>`\n_Exemplo: `/flip del 2`_")
            return
        try:
            trade_id = int(parts[2])
        except ValueError:
            _reply("⚠️ ID tem de ser um número inteiro.")
            return
        removed = delete_flip_trade(trade_id)
        if removed:
            _reply(f"🗑️ Trade `#{trade_id}` removido.")
        else:
            _reply(f"⚠️ Trade `#{trade_id}` não encontrado.")
        return

    _reply(
        f"⚠️ Sub-comando desconhecido: `{sub}`\n\n"
        "*Uso do /flip:*\n"
        "`/flip`                           → Resumo P&L\n"
        "`/flip list`                      → Todos os trades\n"
        "`/flip add TICK ENTRY SHARES`     → Nova entrada\n"
        "`/flip close ID EXIT`             → Fechar trade\n"
        "`/flip del ID`                    → Apagar trade"
    )


# ── Command router ─────────────────────────────────────────────────────────────────────

def _handle_command(text: str) -> None:
    parts   = text.strip().split()
    cmd     = parts[0].lower() if parts else ""
    cmd     = cmd.split("@")[0]
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
            "`/flip`                    → P&L e trades abertos\n"
            "`/flip list`               → Todos os trades\n"
            "`/flip add TICK ENTRY SHR` → Registar entrada\n"
            "`/flip close ID EXIT`      → Fechar trade\n"
            "`/flip del ID`             → Apagar trade\n"
            "`/mldata`                  → Stats da base de dados ML\n"
            "`/mldata update`           → Forçar update de outcomes\n"
            "`/help`                    → Esta mensagem"
        )

    elif cmd == "/status":
        if not _check_rate(cmd_key): return
        uptime     = datetime.now() - _bot_start_time
        hours, rem = divmod(int(uptime.total_seconds()), 3600)
        mins       = rem // 60
        is_open_fn = _poll_context.get("is_market_open")
        market     = "🟢 Aberto" if (is_open_fn and is_open_fn()) else "🔴 Fechado"
        wl         = load_dynamic_watchlist()
        summary    = get_flip_summary()
        flip_str   = f" | Flip: {summary['n_open']} abertos / P&L ${summary['total_pnl']:+.0f}"
        db_fn      = _poll_context.get("get_db_stats") or _cb_get_db_stats
        db_str     = ""
        if db_fn:
            try:
                ds     = db_fn()
                db_str = f" | ML DB: {ds.get('total', 0)} alertas"
            except Exception:
                pass
        _reply(
            f"*🤖 DipRadar Status*\n"
            f"Uptime: *{hours}h {mins}m*\n"
            f"Mercado: *{market}*\n"
            f"Watchlist dinâmica: *{len(wl)} tickers*{flip_str}{db_str}\n"
            f"_⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}_\n"
            f"_{rate_status()}_"
        )

    elif cmd == "/carteira":
        if not _check_rate(cmd_key): return
        get_snap = _poll_context.get("get_snapshot") or _cb_get_snapshot
        if not get_snap:
            _reply("_Snapshot não disponível._")
            return
        try:
            snap  = get_snap()
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
        is_open_fn = _poll_context.get("is_market_open")
        if is_open_fn and not is_open_fn():
            _reply("⚠️ Mercado fechado — scan não disponível fora do horário.")
            return
        _reply("_🔍 A iniciar scan manual..._")
        run_scan_fn = _poll_context.get("run_scan") or _cb_run_scan
        try:
            if run_scan_fn:
                threading.Thread(target=run_scan_fn, daemon=True).start()
        except Exception as e:
            _reply(f"_Erro no scan: {e}_")

    elif cmd == "/analisar":
        if not _check_rate(cmd_key): return
        if len(parts) < 2:
            _reply("⚠️ Usa: `/analisar <TICKER>`\n_Exemplo: `/analisar AAPL`_")
            return
        symbol         = parts[1].upper().strip()
        analyze_fn     = _poll_context.get("analyze_ticker") or _cb_analyze_ticker
        if not analyze_fn:
            _reply("_Análise não disponível._")
            return
        _reply(f"_🔍 A analisar *{symbol}*... (pode demorar 10–20s)_")
        try:
            threading.Thread(
                target=lambda: _reply(analyze_fn(symbol)),
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

    elif cmd == "/historico":
        if not _check_rate(cmd_key): return
        if len(parts) < 2:
            _reply("⚠️ Usa: `/historico <TICKER>`\n_Exemplo: `/historico NVDA`_")
            return
        try:
            _handle_historico(parts[1].upper().strip())
        except Exception as e:
            _reply(f"_Erro no histórico: {e}_")

    elif cmd == "/backtest":
        if not _check_rate(cmd_key): return
        bt_fn = _poll_context.get("build_backtest_summary") or _cb_backtest_summary
        if not bt_fn:
            _reply("_Backtest não disponível._")
            return
        try:
            _reply(bt_fn())
        except Exception as e:
            _reply(f"_Erro no backtest: {e}_")

    elif cmd == "/rejeitados":
        if not _check_rate(cmd_key): return
        rej_fn = _poll_context.get("load_rejected_log") or _cb_rejected_log
        if not rej_fn:
            _reply("_Log não disponível._")
            return
        try:
            rejected = rej_fn()
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
        tier3_fn = _poll_context.get("tier3_handler") or _cb_tier3_handler
        try:
            _reply(tier3_fn() if tier3_fn else "🔵 *Tier 3* — _Handler não registado._")
        except Exception as e:
            _reply(f"_Erro ao obter Tier 3: {e}_")

    elif cmd == "/watchlist":
        if not _check_rate(cmd_key): return
        try:
            _handle_watchlist(parts)
        except Exception as e:
            _reply(f"_Erro na watchlist: {e}_")

    elif cmd == "/flip":
        if not _check_rate(cmd_key): return
        try:
            _handle_flip(parts)
        except Exception as e:
            _reply(f"_Erro no flip: {e}_")

    elif cmd == "/mldata":
        if not _check_rate(cmd_key): return
        force = len(parts) > 1 and parts[1].lower() in ("update", "atualizar", "force")
        try:
            _handle_mldata(force_update=force)
        except Exception as e:
            _reply(f"_Erro no mldata: {e}_")

    else:
        if text.startswith("/"):
            _reply(f"_Comando desconhecido: `{cmd}` — usa /help_")


# ── Poll loop (legacy — usado se main.py usar start_bot_listener) ────────────────────

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
    """Inicia o listener de comandos numa daemon thread (legacy)."""
    t = threading.Thread(target=_poll_loop, daemon=True, name="bot-commands")
    t.start()
    return t
