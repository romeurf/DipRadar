"""
bot_commands.py — Todos os comandos Telegram do DipRadar.

━━━ MERCADO E ANÁLISE ━━━
  /scan                    → Forçar scan imediato (só horas de mercado)
  /analisar <TICK>         → Análise completa: score, ML, valuation, sizing
  /comparar T1 T2 ...      → Comparar scores de 2-5 tickers
  /historico <TICK>        → Histórico de scores registados para um ticker
  /performance [data][score]→ Retorno anual + risco seguindo o bot
  /backtest                → Resumo do backtest de alertas recentes
  /rejeitados              → Stocks analisados e rejeitados hoje
  /tier3                   → Gems do último resumo de fecho

━━━ TEMAS / TRENDS ━━━
  /themes                  → Ver temas em trend (fotónica, GLP-1, IA...)
  /add_theme k l T1,T2 [c] → Adicionar tema (key label TICKERS [confiança])
  /remove_theme <key>      → Remover tema

━━━ CARTEIRA E POSIÇÕES ━━━
  /carteira                → Snapshot da carteira em tempo real
  /portfolio               → Posições activas com P&L
  /sync_portfolio          → Sincronizar carteira actual via Telegram
  /buy TICK PREÇO SHARES   → Registar compra
  /sell TICK PREÇO [SHARES]→ Registar venda (parcial ou total)
  /liquidez [+|-VALOR]     → Ver / ajustar saldo disponível
  /allocate <TICKER>       → Sugestão de alocação com sizing e sector check

━━━ FLIP FUND ━━━
  /flip                    → Log e P&L do Flip Fund
  /flip add T E S [NOTA]   → Registar entrada (ticker, entry, shares)
  /flip close ID EXIT      → Fechar trade com preço de saída
  /flip del ID             → Apagar trade

━━━ WATCHLIST ━━━
  /watchlist               → Estado da watchlist pessoal
  /watchlist add TICKER    → Adicionar ticker
  /watchlist rm TICKER     → Remover ticker
  /watchlist clear         → Limpar watchlist dinâmica

━━━ ML E RETREINO ━━━
  /mldata                  → Estatísticas da base de dados ML
  /mldata update           → Forçar update de outcomes
  /ml_accuracy             → Precisão real do modelo vs outcomes reais
  /admin_retrain [dry-run] → Disparar retreino ad-hoc
  /retrigger               → Alias de /admin_retrain (sem flags)
  /admin_regen_parquet [--targets-only] → Regenerar parquet (EDGAR PIT + alpha_90d)
  /admin_set_floor <valor> → Ajustar floor de IC mínimo para promoção (ex: 0.08)
  /admin_backfill_ml       → Semear histórico de dips (5 anos) no training set
  /admin_load_models <url> → Carregar bundle ML de URL externo

━━━ SISTEMA E DIAGNÓSTICO ━━━
  /status                  → Estado do bot + mercado + modelo
  /health                  → Dashboard: RAM, CPU, APIs, drift
  /health errors           → Últimos erros críticos
  /admin_check_config      → Verificar env vars críticas em falta
  /admin_test_feed TICKER  → Testar pipeline de dados para um ticker
  /help                    → Lista completa de comandos
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
    load_flip_log,
    add_flip_trade,
    close_flip_trade,
    delete_flip_trade,
    get_flip_summary,
)
import health_monitor

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

_EARNINGS_ALERT_DAYS: int = int(os.environ.get("EARNINGS_ALERT_DAYS", 7))

_last_update_id: int = 0
_bot_start_time: datetime = datetime.now()

# Flags de segurança para operações longas
_backfill_running: bool = False
_train_running:    bool = False
_retrain_running:  bool = False

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
_cb_clear_alerts     = None
_cb_allocate_ticker  = None


# ── setup() — bootstrap leve usado pelo main.py ───────────────────────────────
#
# main.py chama `bot_commands.setup(send_fn=..., analyze_fn=..., ...)` no
# arranque. Esta função regista os callbacks essenciais e dispara o listener
# Telegram em thread separada. É um superset minimalista de register_callbacks
# (que serve outro código legado mais detalhado).
#
# Sem esta função, `python main.py` rebenta com AttributeError no boot e o
# bot fica em crash-loop no Railway.

def setup(
    send_fn,
    analyze_fn=None,
    clear_alerts_fn=None,
    allocate_fn=None,
    **extras,
) -> None:
    """Regista callbacks essenciais e arranca o listener Telegram.

    Args:
        send_fn: função send_telegram(text) — usada por todos os _reply().
        analyze_fn: usada pelo /analisar.
        clear_alerts_fn: usada por /admin (futuramente).
        allocate_fn: usada pelo /allocate.
        **extras: chaves adicionais injectadas em _poll_context
                  (forward-compat; permite passar context ad-hoc sem
                  alterar a assinatura).
    """
    global _cb_send_telegram, _cb_analyze_ticker
    global _cb_clear_alerts, _cb_allocate_ticker
    _cb_send_telegram   = send_fn
    _cb_analyze_ticker  = analyze_fn
    _cb_clear_alerts    = clear_alerts_fn
    _cb_allocate_ticker = allocate_fn

    if extras:
        _poll_context.update(extras)
    _poll_context.setdefault("send_fn", send_fn)

    # Arranca o listener em thread (idempotente — se chamado 2x, o segundo
    # start_bot_listener() faz nada porque a thread anterior continua viva).
    start_bot_listener()


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


# ── poll() bridge ────────────────────────────────────────────────────────────────

def poll(
    token: str,
    chat_id: str,
    send_fn,
    context: dict,
) -> None:
    global _last_update_id
    if not token:
        return

    global _cb_send_telegram, _cb_get_snapshot, _cb_get_db_stats, _cb_fill_db_outcomes
    _cb_send_telegram    = send_fn
    _cb_get_snapshot     = context.get("get_snapshot")
    _cb_get_db_stats     = context.get("get_db_stats")
    _cb_fill_db_outcomes = context.get("fill_db_outcomes")

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


_poll_context: dict = {}


def _reply(text: str) -> None:
    fn = _poll_context.get("send_fn") or _cb_send_telegram
    if fn:
        fn(text)


def _md_safe(s: object) -> str:
    """Sanitiza para uso dentro de code span (backtick): remove backticks internos.

    Uso correcto: f"`{_md_safe(exception)}`"
    Não usar directamente em texto bold/italic — usar _md_escape para isso.
    """
    return str(s).replace("`", "'") if s is not None else ""


def _md_escape(s: object) -> str:
    """Escapa chars especiais MarkdownV1 para uso em texto livre (fora de code spans).

    Necessário sempre que um valor dinâmico (nome de ficheiro, mensagem de erro,
    output de API) é inserido directamente em texto Markdown. Sem este escape,
    underscores em nomes como 'alpha_90d' ou 'admin_retrain' quebram a formatação.

    Uso: f"Erro em {_md_escape(filename)}: {_md_escape(e)}"
    """
    txt = str(s) if s is not None else ""
    for ch in ["_", "*", "`", "["]:
        txt = txt.replace(ch, "\\" + ch)
    return txt


def _check_rate(cmd: str) -> bool:
    allowed, wait = is_allowed(cmd)
    if not allowed:
        _reply(
            f"⏳ *Rate limit* — `/{cmd}` muito frequente.\n"
            f"_Tenta novamente em *{wait}s*._"
        )
        logging.info(f"[bot_commands] rate limit: /{cmd} bloqueado ({wait}s)")
    return allowed


# ── /comparar ──────────────────────────────────────────────────────────────────

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

    get_fundamentals  = _poll_context.get("get_fundamentals") or _cb_get_fundamentals
    get_earnings_days = _poll_context.get("get_earnings_days") or _cb_earnings_days

    if not get_fundamentals:
        _reply("_Comparação não disponível — callback get_fundamentals não registado._")
        return

    _reply(f"_🔄 A comparar {' vs '.join(symbols)}... (pode demorar até 30s)_")

    def _run():
        from score import calculate_dip_score, classify_dip_category, is_bluechip
        rows = []
        for sym in symbols:
            try:
                fund  = get_fundamentals(sym)
                edays = get_earnings_days(sym) if get_earnings_days else None
                score, rsi_str = calculate_dip_score(fund, sym, edays)
                badge    = "🔥" if score >= 80 else ("⭐" if score >= 55 else "📊")
                bc_flag  = is_bluechip(fund)
                category = classify_dip_category(fund, score, bc_flag)
                rows.append({
                    "sym": sym, "score": score, "badge": badge,
                    "category": category, "rsi": rsi_str or "N/D",
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
            lines.append(f"  {r['badge']} *{r['sym']}* — score *{r['score']:.0f}/100* | {r['category']}")
            lines.append(f"     RSI {r['rsi']} · FCF {r['fcf']} · Growth {r['growth']}")
            lines.append(f"     Margin {r['margin']} · Upside {r['upside']} · Drawdown {r['drawdown']}")
            lines.append(f"     Sector: _{r['sector']}_ · Earnings: {r['edays']}d")
            lines.append("")

        winner = rows[0]
        if not winner.get("error"):
            lines.append(f"_🏆 Melhor score: *{winner['sym']}* ({winner['score']:.0f} pts) — usa `/analisar {winner['sym']}` para detalhe._")
        _reply("\n".join(lines))

    threading.Thread(target=_run, daemon=True).start()


# ── /historico ─────────────────────────────────────────────────────────────────

def _handle_historico(symbol: str) -> None:
    history = get_ticker_score_history(symbol)
    if not history:
        _reply(
            f"_Sem histórico de scores para *{symbol}*._\n"
            "_O ticker ainda não foi analisado ou alertado pelo DipRadar._"
        )
        return

    entries    = history[-10:]
    lines      = [f"*📈 Histórico — {symbol} ({len(history)} entradas totais):*", ""]
    prev_score = None
    for e in entries:
        score  = e.get("score", 0)
        change = e.get("change", 0)
        price  = e.get("price")
        date   = e.get("date", "")
        t      = e.get("time", "")
        verdict = e.get("verdict") or ""
        if prev_score is None: trend = ""
        elif score > prev_score: trend = f" ↑{score - prev_score:.0f}"
        elif score < prev_score: trend = f" ↓{prev_score - score:.0f}"
        else: trend = " ="
        prev_score = score
        badge     = "🔥" if score >= 80 else ("⭐" if score >= 55 else "📊")
        price_str = f" @ ${price:.2f}" if price else ""
        lines.append(f"  {badge} `{date} {t}` — *{score:.0f}*{trend} | {change:+.1f}%{price_str} | _{verdict}_")

    if len(history) > 10:
        lines.append(f"\n_... e mais {len(history) - 10} entradas anteriores._")
    _reply("\n".join(lines))


# ── /mldata ────────────────────────────────────────────────────────────────────

def _handle_mldata(force_update: bool = False) -> None:
    get_db_stats     = _poll_context.get("get_db_stats") or _cb_get_db_stats
    fill_db_outcomes = _poll_context.get("fill_db_outcomes") or _cb_fill_db_outcomes

    if not get_db_stats:
        _reply("_Base de dados ML não disponível. Reinicia o bot._")
        return

    def _run():
        update_stats = None
        if force_update and fill_db_outcomes:
            _reply("🔄 _A actualizar outcomes da alert_db... (pode demorar 1-2 min)_")
            try:
                update_stats = fill_db_outcomes()
            except Exception as e:
                _reply(f"⚠️ Erro no update: `{e}`")
                return

        try:
            stats   = get_db_stats()
            total   = stats.get("total", 0)
            labeled = stats.get("labeled", 0)
            outcomes = stats.get("outcomes", {})
            by_cat  = stats.get("by_category", {})
            by_vrd  = stats.get("by_verdict", {})
            db_path = stats.get("db_path", "N/D")
        except Exception as e:
            _reply(f"_Erro ao ler stats: {e}_")
            return

        lines = [
            "🤖 *ML Alert Database*",
            f"_{datetime.now().strftime('%d/%m/%Y %H:%M')}_", "",
            f"*🗃️ Total de alertas:* {total}",
            (f"*📊 Classificados:* {labeled} ({labeled/total*100:.0f}% do total)" if total > 0 else "*📊 Classificados:* 0"),
            f"*🗂️ Ficheiro:* `{db_path}`", "",
        ]
        if by_cat:
            lines.append("*🏠 Por categoria:*")
            for cat, count in sorted(by_cat.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {cat}: *{count}* ({count/total*100:.0f}%)")
            lines.append("")
        if outcomes:
            lines.append("*🎯 Outcomes:*")
            em_map = {"WIN_STRONG": "🟢", "WIN": "✅", "NEUTRAL": "🟡", "LOSS": "🔴"}
            for lbl, cnt in sorted(outcomes.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {em_map.get(lbl,'📊')} {lbl}: *{cnt}* ({cnt/labeled*100:.0f}%)" if labeled else f"  {lbl}: {cnt}")
            wins = outcomes.get("WIN_STRONG", 0) + outcomes.get("WIN", 0)
            if labeled > 0:
                lines.append(f"\n  📈 *Win rate:* {wins/labeled*100:.0f}% ({wins}/{labeled})")
            lines.append("")
        if update_stats:
            lines.append(f"*✅ Update:* {update_stats.get('updated', 0)} novas classificações\n")

        if labeled >= 100:
            lines.append("🟢 *Pronto para treinar!* Usa `/admin_train_ml`")
        elif labeled >= 50:
            lines.append(f"🟡 *{labeled} amostras* — quase prontos (mín. recomendado: 100)")
            lines.append("_Já podes tentar `/admin_train_ml` com resultados parciais._")
        else:
            lines.append(f"🕐 *{labeled} amostras* — precisa de mais dados (mín: 30 para treinar).")

        if not force_update and fill_db_outcomes:
            lines.append("\n_Força update agora com `/mldata update`_")
        _reply("\n".join(lines))

    threading.Thread(target=_run, daemon=True).start()


# ── /admin_backfill_ml ──────────────────────────────────────────────────────────

# Timeout máximo por ticker (segundos). Se o yfinance pendurar mais do que isto,
# o ticker é marcado como erro e o backfill continua para o próximo.
_BACKFILL_TICKER_TIMEOUT = int(os.environ.get("BACKFILL_TICKER_TIMEOUT", 90))


def _handle_admin_backfill_ml() -> None:
    global _backfill_running

    if _backfill_running:
        _reply("⚠️ *Backfill já está a correr.*\n_Aguarda a conclusão antes de lançar novamente._")
        return

    dynamic = load_dynamic_watchlist()
    static  = []
    try:
        from watchlist import WATCHLIST as _STATIC_WL
        static = [e["symbol"] for e in _STATIC_WL]
    except Exception:
        pass
    tickers = list(dict.fromkeys(dynamic + static))

    if not tickers:
        _reply("⚠️ *Watchlist vazia.*\n_Adiciona tickers com `/watchlist add TICK` antes de fazer backfill._")
        return

    n = len(tickers)
    est_min = n * 4 // 60 + 1
    est_max = n * 6 // 60 + 2
    _reply(
        f"⏳ *A iniciar a Máquina do Tempo* — {n} ticker(s) | lookback 5 anos\n"
        f"_Estimativa: {est_min}–{est_max} minutos..._\n"
        f"_Vais receber update após cada ticker. Timeout por ticker: {_BACKFILL_TICKER_TIMEOUT}s._"
    )

    def _run():
        global _backfill_running
        _backfill_running = True
        start_ts = time.time()

        # Importações necessárias dentro da thread
        from backtest import (
            run_historical_backtest,
            _load_existing_keys,
            _write_hist_csv,
            _HIST_DB_PATH,
        )
        from pathlib import Path

        csv_path      = _HIST_DB_PATH
        existing_keys = _load_existing_keys(csv_path)

        total_dips     = 0
        total_written  = 0
        total_skipped  = 0
        total_censored = 0
        total_dupes    = 0
        total_errors   = 0
        all_dip_rows: list[dict] = []

        try:
            for i, sym in enumerate(tickers, 1):
                ticker_start = time.time()
                _reply(
                    f"🔄 *[{i}/{n}]* `{sym}` — a descarregar 5 anos de dados..."
                )

                # Corre run_historical_backtest para UM ticker com timeout via thread
                result_holder: list[dict] = []
                error_holder:  list[str]  = []

                def _fetch(s=sym, rh=result_holder, eh=error_holder):
                    try:
                        stats = run_historical_backtest(
                            tickers=[s],
                            min_score=0.0,
                            dry_run=False,
                        )
                        rh.append(stats)
                    except Exception as exc:
                        eh.append(str(exc))

                t = threading.Thread(target=_fetch, daemon=True)
                t.start()
                t.join(timeout=_BACKFILL_TICKER_TIMEOUT)

                elapsed_ticker = int(time.time() - ticker_start)

                if t.is_alive():
                    # Timeout — o yfinance ficou pendurado
                    total_errors += 1
                    _reply(
                        f"⏱️ *[{i}/{n}]* `{sym}` — *timeout* ({_BACKFILL_TICKER_TIMEOUT}s)\n"
                        f"_A avançar para o próximo ticker._"
                    )
                    logging.warning(f"[admin_backfill] {sym}: timeout após {_BACKFILL_TICKER_TIMEOUT}s")
                    continue

                if error_holder:
                    total_errors += 1
                    _reply(
                        f"⚠️ *[{i}/{n}]* `{sym}` — erro: `{error_holder[0][:80]}`"
                    )
                    logging.warning(f"[admin_backfill] {sym}: {error_holder[0]}")
                    continue

                if not result_holder:
                    total_errors += 1
                    _reply(f"⚠️ *[{i}/{n}]* `{sym}` — sem resultado (dados insuficientes?)")
                    continue

                stats      = result_holder[0]
                dip_rows   = stats.get("dip_rows", [])
                n_dips     = stats.get("total_dips", 0)
                n_censored = stats.get("censored", 0)
                n_skipped  = stats.get("ignored_score", 0)
                n_err      = stats.get("errors", 0)

                # Escrita incremental para este ticker
                dupes_ticker   = sum(1 for r in dip_rows if (r["symbol"], r["date_iso"]) in existing_keys)
                written_ticker = _write_hist_csv(dip_rows, csv_path, existing_keys)
                # Actualiza existing_keys para evitar duplicados nos tickers seguintes
                for r in dip_rows:
                    existing_keys.add((r["symbol"], r["date_iso"]))

                total_dips     += n_dips
                total_written  += written_ticker
                total_skipped  += n_skipped
                total_censored += n_censored
                total_dupes    += dupes_ticker
                total_errors   += n_err
                all_dip_rows   += dip_rows

                # Distribuição de outcomes para este ticker
                label_counts: dict = {}
                for row in dip_rows:
                    lbl = row.get("outcome_label") or "pending"
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1

                em_map   = {"WIN_STRONG": "🟢", "WIN": "✅", "NEUTRAL": "🟡", "LOSS": "🔴", "pending": "⏳"}
                lbl_str  = " | ".join(
                    f"{em_map.get(k,'📊')}{k}:{v}"
                    for k, v in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
                ) if label_counts else "_sem dips_"

                status_em = "✅" if n_dips > 0 else "➖"
                _reply(
                    f"{status_em} *[{i}/{n}]* `{sym}` — "
                    f"*{n_dips}* dips | +{written_ticker} linhas | {elapsed_ticker}s\n"
                    f"  {lbl_str}"
                )

        except Exception as e:
            logging.error(f"[admin_backfill] Erro fatal no loop: {e}")
            _reply(f"❌ *Backfill interrompido por erro fatal:*\n`{e}`")
        finally:
            _backfill_running = False

        # ── Relatório final ──────────────────────────────────────────────────
        elapsed  = int(time.time() - start_ts)
        mins, secs = divmod(elapsed, 60)

        label_counts_total: dict = {}
        for row in all_dip_rows:
            lbl = row.get("outcome_label") or "pending"
            label_counts_total[lbl] = label_counts_total.get(lbl, 0) + 1

        em_map = {"WIN_STRONG": "🟢", "WIN": "✅", "NEUTRAL": "🟡", "LOSS": "🔴", "pending": "⏳"}
        lines = [
            "✅ *Backfill ML concluído!*",
            f"_{datetime.now().strftime('%d/%m/%Y %H:%M')} — {mins}m{secs:02d}s_", "",
            f"*🗂️ Ficheiro:* `{str(csv_path)}`", "",
            "*📊 Pipeline stats:*",
            f"  📡 Dips detectados:  *{total_dips}*",
            f"  ✍️  Escritos no CSV: *{total_written}*",
            f"  🚫 Ignorados (score): *{total_skipped}*",
            f"  ⏳ Censurados (edge): *{total_censored}*",
            f"  🔁 Duplicados (skip): *{total_dupes}*",
        ]
        if total_errors:
            lines.append(f"  ⚠️ Erros/timeouts: *{total_errors}*")
        lines.append("")
        if label_counts_total:
            lines.append("*🎯 Distribuição de outcomes:*")
            for lbl, cnt in sorted(label_counts_total.items(), key=lambda x: x[1], reverse=True):
                pct = cnt / len(all_dip_rows) * 100 if all_dip_rows else 0
                lines.append(f"  {em_map.get(lbl,'📊')} {lbl}: *{cnt}* ({pct:.0f}%)")
            lines.append("")

        total_labeled = sum(cnt for lbl, cnt in label_counts_total.items() if lbl != "pending")
        if total_labeled >= 100:
            lines.append("🟢 *Pronto para treinar!* Usa `/admin_train_ml`")
        elif total_labeled >= 30:
            lines.append(f"🟡 *{total_labeled} amostras* — podes já tentar `/admin_train_ml`")
        else:
            lines.append(f"🕐 *{total_labeled} amostras* — adiciona mais tickers à watchlist.")

        _reply("\n".join(lines))

    threading.Thread(target=_run, daemon=True, name="admin-backfill").start()


# ── /admin_train_ml ─────────────────────────────────────────────────────────────

def _handle_admin_train_ml() -> None:
    global _train_running

    if _train_running:
        _reply("⚠️ *Treino já está a correr.*\n_Aguarda a conclusão._")
        return

    _reply(
        "🧪 *A iniciar o Laboratório de ML...*\n"
        "_A competição AutoSelect vai correr RF vs XGBoost vs LightGBM._\n"
        "_Estimativa: 2–5 minutos dependendo do volume de dados._\n"
        "_O bot continua a responder durante o treino._"
    )
    logging.info("[admin_train] Iniciando treino ML")

    def _run():
        global _train_running
        _train_running = True
        start_ts = time.time()
        try:
            from train_model import train_all
            result = train_all(live_only=False, dry_run=False, min_precision=0.70)

            elapsed    = int(time.time() - start_ts)
            mins, secs = divmod(elapsed, 60)
            s1         = result.get("stage1", {})
            s2         = result.get("stage2")

            lines = [
                "🎓 *Treino ML concluído!*",
                f"_{datetime.now().strftime('%d/%m/%Y %H:%M')} — {mins}m{secs:02d}s_",
                "",
                "*🥊 Competição AutoSelect:*",
                f"  🏆 Vencedor Andar 1: *{s1.get('algorithm', 'N/D')}*",
                f"  📊 AUC-PR:           *{s1.get('auc_pr', 0):.4f}*",
                f"  🎯 Threshold:        *{s1.get('threshold', 0):.4f}*",
                f"  📦 Amostras:         *{s1.get('n_samples', 0)}* "
                    f"(WIN={s1.get('n_win', 0)} | NOT-WIN={s1.get('n_not_win', 0)})",
                f"  🧬 SMOTE activado:   {'Sim' if s1.get('smote_used') else 'Não'}",
                "",
            ]

            if s2:
                n_win40 = s2.get('n_win40', 0)
                n_win20 = s2.get('n_win20', 0)
                lines += [
                    "*🍷 Andar 2 (Sommelier WIN40 vs WIN20):*",
                    f"  🏆 Algoritmo: *{s2.get('algorithm', 'N/D')}*",
                    f"  📊 AUC-PR:    *{s2.get('auc_pr', 0):.4f}*",
                    f"  📦 Amostras:  *{s2.get('n_samples', 0)}* "
                        f"(WIN40={n_win40} | WIN20={n_win20})",
                    "",
                ]
            else:
                lines.append("_Andar 2: dados insuficientes (precisa ≥15 amostras WIN)._\n")

            feat_imp = s1.get("feature_importance", [])
            if feat_imp:
                lines.append("*🔬 Top-5 features (Andar 1):*")
                max_imp = feat_imp[0]["importance"] if feat_imp else 1
                for fi in feat_imp[:5]:
                    ratio    = fi["importance"] / max_imp if max_imp > 0 else 0
                    bar_len  = int(ratio * 10)
                    bar      = "█" * bar_len
                    lines.append(f"  `{fi['feature']:<20}` {bar} *{fi['importance']:.4f}*")
                lines.append("")

            auc = s1.get("auc_pr", 0)
            if auc >= 0.80:
                lines.append("🟢 *Modelo excelente!* AUC-PR ≥ 0.80 — pronto para produção.")
                lines.append("_Chunk I vai integrar este cérebro no scanner ao vivo._")
            elif auc >= 0.65:
                lines.append("🟡 *Modelo bom.* AUC-PR ≥ 0.65 — funcional, vai melhorar com mais dados.")
            else:
                lines.append("🔴 *Modelo fraco.* AUC-PR < 0.65 — precisa de mais amostras.")
                lines.append("_Corre `/admin_backfill_ml` com mais tickers na watchlist._")

            _reply("\n".join(lines))
            logging.info(
                f"[admin_train] Concluído em {mins}m{secs:02d}s — "
                f"alg={s1.get('algorithm')} auc={s1.get('auc_pr')} threshold={s1.get('threshold')}"
            )

        except ValueError as e:
            _reply(
                f"⚠️ *Dados insuficientes para treinar:*\n`{e}`\n\n"
                "*Solução:*\n"
                "1. Corre `/admin_backfill_ml` para gerar mais dados históricos.\n"
                "2. Aguarda alertas live durante algumas semanas.\n"
                "3. Volta a tentar `/admin_train_ml`."
            )
        except FileNotFoundError as e:
            _reply(
                f"⚠️ *Ficheiro de dados não encontrado:*\n`{e}`\n\n"
                "_Corre `/admin_backfill_ml` primeiro para gerar o hist_backtest.csv._"
            )
        except Exception as e:
            logging.error(f"[admin_train] Erro fatal: {e}")
            _reply(
                f"❌ *Treino falhou com erro inesperado:*\n`{e}`\n"
                "_Verifica os logs no Railway para mais detalhes._"
            )
        finally:
            _train_running = False

    threading.Thread(target=_run, daemon=True, name="admin-train").start()


# ── /admin_load_models ──────────────────────────────────────────────────────────

_load_models_running: bool = False


def _handle_admin_load_models(parts: list[str]) -> None:
    """
    /admin_load_models <url>                     → tarball (.tar.gz/.tgz/.zip)
    /admin_load_models <pkl_url> <report_url>    → v3 (2 URLs)
    /admin_load_models <s1_url> <s2_url> <r_url> → legacy (3 URLs)

    Suporta dois formatos:
      • v3      (1 .pkl + 1 .json):  dip_models.pkl + ml_report.json
        (aceita também nomes legacy dip_models_v3.pkl/ml_report_v3.json)
      • legacy  (2 .pkl + 1 .json):  dip_model_stage{1,2}.pkl + ml_report.json

    O formato é detectado automaticamente pelos nomes dos ficheiros (URLs
    individuais) ou pelo conteúdo do archive (modo tarball). Faz swap atomic
    com archive automático da versão anterior em `/data/archive/<file>_<ts>`.
    O ml_predictor recarrega automaticamente quando o mtime muda — não precisa
    reiniciar o bot.

    URLs aceites: HTTPS direct download. Para Google Drive partilha pública
    usa o formato `https://drive.google.com/uc?export=download&id=<FILE_ID>`.
    """
    global _load_models_running

    if _load_models_running:
        _reply("⚠️ *Load já está a correr.* _Aguarda a conclusão._")
        return

    args = [a for a in parts[1:] if a.strip()]
    if len(args) not in (1, 2, 3):
        _reply(
            "❌ *Uso incorrecto.*\n\n"
            "*Modo tarball* (1 URL):\n"
            "`/admin_load_models <url_tar_gz>`\n"
            "_Aceita v3 (1 .pkl + 1 .json) ou legacy (2 .pkl + 1 .json)._\n\n"
            "*Modo v3* (2 URLs):\n"
            "`/admin_load_models <dip_models.pkl> <ml_report.json>`\n\n"
            "*Modo legacy* (3 URLs):\n"
            "`/admin_load_models <s1_url> <s2_url> <report_url>`\n\n"
            "_Para Google Drive: link directo `https://drive.google.com/uc?export=download&id=<ID>`._"
        )
        return

    for u in args:
        if not (u.startswith("https://") or u.startswith("http://")):
            _reply(f"❌ URL inválida (precisa de http/https): `{u}`")
            return

    _reply(
        f"📥 *A descarregar pickles...* ({len(args)} URL{'s' if len(args)>1 else ''})\n"
        "_Validação + swap atomic em ~30s._"
    )

    def _run():
        global _load_models_running
        _load_models_running = True
        from pathlib import Path
        import shutil
        import tempfile
        import json
        import joblib
        import tarfile
        import zipfile
        import urllib.request
        import urllib.parse

        data_dir   = Path("/data") if Path("/data").exists() else Path("/tmp")
        archive_dir = data_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp  = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

        # Limites de segurança
        MAX_BYTES_PER_FILE      = 80 * 1024 * 1024  # 80 MB por ficheiro
        # Formato legacy Tier A+B+C: 2 .pkl + 1 .json
        # NOTA: o ml_report.json conflitua com o canonical v3 actual; detecta
        # primeiro a presença de dip_model_stage*.pkl para distinguir.
        LEGACY_FILES            = ["dip_model_stage1.pkl", "dip_model_stage2.pkl", "ml_report.json"]
        LEGACY_REQUIRED_KEYS    = {"model", "feature_columns", "threshold"}
        # Formato v3 (canonical actual): 1 .pkl + 1 .json
        V3_FILES                = ["dip_models.pkl", "ml_report.json"]
        # Nomes legacy v3 (PR robustez 2026-05 renomeou): mantidos para aceitar
        # archives antigos. Detecção mapeia legacy_name → canonical_name.
        V3_LEGACY_ALIASES       = {
            "dip_models_v3.pkl":  "dip_models.pkl",
            "ml_report_v3.json": "ml_report.json",
        }

        tmp_dir = Path(tempfile.mkdtemp(prefix="loadmodels_"))
        try:
            def _download(url: str, dest: Path) -> None:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "DipRadar-LoadModels/1.0"},
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    total = 0
                    with open(dest, "wb") as f:
                        while True:
                            chunk = resp.read(64 * 1024)
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > MAX_BYTES_PER_FILE:
                                raise ValueError(
                                    f"ficheiro excede {MAX_BYTES_PER_FILE//1024//1024} MB"
                                )
                            f.write(chunk)
                logging.info(f"[admin_load_models] download {dest.name}: {total} bytes")

            # ── Modo tarball (1 URL): detecta formato pelo conteúdo do archive
            if len(args) == 1:
                archive_path = tmp_dir / "models.archive"
                _download(args[0], archive_path)
                extract_dir = tmp_dir / "extracted"
                extract_dir.mkdir()
                if zipfile.is_zipfile(archive_path):
                    with zipfile.ZipFile(archive_path) as zf:
                        zf.extractall(extract_dir)
                else:
                    try:
                        with tarfile.open(archive_path, "r:*") as tf:
                            tf.extractall(extract_dir)
                    except tarfile.TarError as e:
                        raise ValueError(
                            f"ficheiro não é zip nem tar reconhecido: {e}"
                        ) from e

                # Detecta formato: v3 toma prioridade (single bundle moderno).
                # Aceita nomes legacy (dip_models_v3.pkl) ou canonical (dip_models.pkl).
                v3_pkl_modern = list(extract_dir.rglob("dip_models.pkl"))
                v3_pkl_legacy = list(extract_dir.rglob("dip_models_v3.pkl"))
                if v3_pkl_modern or v3_pkl_legacy:
                    fmt = "v3"
                    expected = V3_FILES
                else:
                    fmt = "legacy"
                    expected = LEGACY_FILES

                local: dict[str, Path] = {}
                if fmt == "v3":
                    # Mapeia legacy_name → canonical_name nos casos em que só
                    # o nome antigo está presente no archive.
                    legacy_to_canonical = {v: k for k, v in V3_LEGACY_ALIASES.items()}
                    for fname in expected:
                        matches = list(extract_dir.rglob(fname))
                        if not matches:
                            legacy_name = legacy_to_canonical.get(fname)
                            if legacy_name:
                                matches = list(extract_dir.rglob(legacy_name))
                        if not matches:
                            raise ValueError(
                                f"formato v3: ficheiro {fname} (nem alias legacy) "
                                f"encontrado no archive"
                            )
                        local[fname] = matches[0]
                else:
                    for fname in expected:
                        matches = list(extract_dir.rglob(fname))
                        if not matches:
                            raise ValueError(
                                f"formato {fmt}: ficheiro {fname} não encontrado no archive"
                            )
                        local[fname] = matches[0]

            # ── Modo 2 URLs: v3 (pkl + report)
            elif len(args) == 2:
                fmt = "v3"
                expected = V3_FILES
                local = {}
                for fname, url in zip(expected, args):
                    dest = tmp_dir / fname
                    _download(url, dest)
                    local[fname] = dest

            # ── Modo 3 URLs: legacy (stage1, stage2, report)
            else:
                fmt = "legacy"
                expected = LEGACY_FILES
                local = {}
                for fname, url in zip(expected, args):
                    dest = tmp_dir / fname
                    _download(url, dest)
                    local[fname] = dest

            # ── Validação + extração de métricas específicas do formato
            meta: dict = {}
            if fmt == "legacy":
                for stage in (1, 2):
                    fname  = f"dip_model_stage{stage}.pkl"
                    bundle = joblib.load(local[fname])
                    if not isinstance(bundle, dict):
                        raise ValueError(f"{fname}: pickle não é um dict")
                    missing = LEGACY_REQUIRED_KEYS - set(bundle.keys())
                    if missing:
                        raise ValueError(f"{fname}: faltam keys {missing} no bundle")
                    fc = bundle.get("feature_columns") or []
                    if not fc:
                        raise ValueError(f"{fname}: feature_columns vazia")
                    if stage == 1:
                        meta = {
                            "n_features": len(fc),
                            "threshold":  bundle.get("threshold"),
                            "algorithm":  bundle.get("algorithm")
                                or bundle.get("algo")
                                or "unknown",
                        }
                with open(local["ml_report.json"]) as f:
                    report = json.load(f)
                if not isinstance(report, dict):
                    raise ValueError("ml_report.json: top-level não é um dict")
                s1 = report.get("stage1") or {}
                meta["auc_pr"] = s1.get("auc_pr_test") or s1.get("auc_pr")
            else:  # v3
                # Importa o helper que normaliza dataclass → dict canonical
                from ml_predictor import _safe_load, _to_dict
                raw = _safe_load(local["dip_models.pkl"])
                bundle = _to_dict(raw)
                if not bundle:
                    raise ValueError("dip_models.pkl: bundle vazio ou não reconhecido")
                if "model_up" not in bundle or "model_down" not in bundle:
                    raise ValueError(
                        "dip_models.pkl: faltam model_up/model_down (esperado v3)"
                    )
                fc = bundle.get("feature_cols") or []
                if not fc:
                    raise ValueError("dip_models.pkl: feature_cols vazia")
                with open(local["ml_report.json"]) as f:
                    report = json.load(f)
                if not isinstance(report, dict):
                    raise ValueError("ml_report.json: top-level não é um dict")
                cv = report.get("walk_forward_cv") or {}
                meta = {
                    "n_features": len(fc),
                    "champion":   bundle.get("champion")
                        or report.get("champion_model") or "unknown",
                    "rho_up":     cv.get("rho_up_mean"),
                    "rho_down":   cv.get("rho_down_mean"),
                    "rho_mean":   cv.get("rho_mean"),
                    "topk_pnl":   cv.get("topk_pnl_mean"),
                    "trained_at": report.get("trained_at"),
                }

            # ── Atomic swap: archive existing, then replace
            promoted = []
            for fname in expected:
                target = data_dir / fname
                if target.exists():
                    arch = archive_dir / f"{Path(fname).stem}_{timestamp}{Path(fname).suffix}"
                    shutil.copy2(target, arch)
                    logging.info(f"[admin_load_models] archived {target.name} → {arch.name}")
                tmp_target = target.with_suffix(target.suffix + ".tmp")
                shutil.copy2(local[fname], tmp_target)
                tmp_target.replace(target)
                promoted.append(fname)
                logging.info(f"[admin_load_models] deployed → {target}")

            # ── Reply com métricas específicas do formato
            if fmt == "legacy":
                thr    = meta.get("threshold")
                algo   = meta.get("algorithm")
                n_feat = meta.get("n_features")
                auc_pr = meta.get("auc_pr")
                lines = [
                    "✅ *Modelos carregados (legacy Tier A+B+C)*",
                    f"_{datetime.now().strftime('%d/%m/%Y %H:%M')} — archive ts {timestamp}_",
                    "",
                    "*📊 Stage 1 (gating):*",
                    f"  Algoritmo:  *{algo}*",
                    f"  AUC-PR:     *{auc_pr:.4f}*" if isinstance(auc_pr, (int, float)) else "  AUC-PR:     _N/D_",
                    f"  Threshold:  *{thr:.4f}*"    if isinstance(thr, (int, float))    else "  Threshold:  _N/D_",
                    f"  Features:   *{n_feat}*",
                ]
            else:  # v3
                lines = [
                    "✅ *Modelo v3 carregado* (regressor dual XGB-v2)",
                    f"_{datetime.now().strftime('%d/%m/%Y %H:%M')} — archive ts {timestamp}_",
                    "",
                    f"*Champion:*    `{meta.get('champion','unknown')}`",
                    f"*Features:*    *{meta.get('n_features')}*",
                ]
                if isinstance(meta.get("rho_up"), (int, float)):
                    lines.append(f"*rho_up:*      *{meta['rho_up']:.3f}*")
                if isinstance(meta.get("rho_down"), (int, float)):
                    lines.append(f"*rho_down:*    *{meta['rho_down']:.3f}*")
                if isinstance(meta.get("topk_pnl"), (int, float)):
                    lines.append(f"*Top-k PnL:*   *{meta['topk_pnl']*100:+.1f}%* /60d")
                if meta.get("trained_at"):
                    lines.append(f"*Trained:*     `{meta['trained_at']}`")
            lines += [
                "",
                f"*Files deployed*: `{', '.join(promoted)}`",
                f"*Archive*: `/data/archive/*_{timestamp}.*`",
                "",
                "_O ml_predictor recarrega automaticamente no próximo `ml_score()`._",
            ]
            _reply("\n".join(lines))

        except Exception as e:
            logging.error(f"[admin_load_models] {e}", exc_info=True)
            _reply(
                f"❌ *Load falhou:*\n`{type(e).__name__}: {e}`\n\n"
                "_Os ficheiros existentes em /data/ NÃO foram alterados._"
            )
        finally:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
            _load_models_running = False

    threading.Thread(target=_run, daemon=True, name="admin-load-models").start()


# ── /admin_test_feed ─────────────────────────────────────────────────────────────

def _handle_admin_test_feed(parts: list[str]) -> None:
    """Testa o pipeline de dados para um ticker. /admin_test_feed AAPL"""
    ticker = parts[1].upper().strip() if len(parts) > 1 else "AAPL"

    def _safe_str(v) -> str:
        """Converte valor para string sem chars especiais Markdown."""
        s = str(v)
        # Remover chars que quebram MarkdownV1
        for ch in ["*", "_", "`", "[", "]"]:
            s = s.replace(ch, "")
        return s[:80]

    def _run():
        lines = [f"Feed test: {ticker}", ""]
        try:
            # 1. data_feed
            from data_feed import get_eod_prices
            df = get_eod_prices(ticker, lookback_days=10)
            if df is not None and not df.empty:
                cols = [_safe_str(c) for c in list(df.columns)[:5]]
                has_date = "date" in df.columns
                lines.append(f"data-feed: OK {len(df)} linhas")
                lines.append(f"  cols: {cols}")
                lines.append(f"  coluna date: {'presente' if has_date else 'AUSENTE'}")
                lines.append(f"  index type: {type(df.index).__name__}")
            else:
                lines.append("data-feed: vazio")

            # 2. yfinance download
            try:
                import yfinance as yf
                import pandas as pd
                raw = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
                if raw is not None and not raw.empty:
                    is_multi = isinstance(raw.columns, pd.MultiIndex)
                    level0 = [_safe_str(c) for c in list(raw.columns.get_level_values(0) if is_multi else raw.columns)[:5]]
                    tz_str = _safe_str(getattr(raw.index, "tz", "none"))
                    lines.append(f"yf.download: OK {len(raw)} linhas")
                    lines.append(f"  MultiIndex: {is_multi}")
                    lines.append(f"  cols: {level0}")
                    lines.append(f"  index.name: {_safe_str(raw.index.name)}")
                    lines.append(f"  tz: {tz_str}")
                else:
                    lines.append("yf.download: vazio")
            except Exception as e:
                lines.append(f"yf.download erro: {_safe_str(e)}")

            # 3. yf.Ticker.history (sem raise-errors)
            try:
                hist = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
                if hist is not None and not hist.empty:
                    tz_str = _safe_str(getattr(hist.index, "tz", "none"))
                    lines.append(f"yf.history: OK {len(hist)} linhas tz={tz_str}")
                else:
                    lines.append("yf.history: vazio")
            except Exception as e:
                lines.append(f"yf.history erro: {_safe_str(e)}")

            # 4. fast-info
            try:
                fi = yf.Ticker(ticker).fast_info
                price = getattr(fi, "last_price", None)
                lines.append(f"fast-info: price={price}")
            except Exception as e:
                lines.append(f"fast-info erro: {_safe_str(e)}")

        except Exception as e:
            lines.append(f"Erro geral: {_safe_str(e)}")

        _reply("\n".join(lines))

    threading.Thread(target=_run, daemon=True, name="test-feed").start()


# ── /sync_portfolio ──────────────────────────────────────────────────────────────

def _handle_sync_portfolio(parts: list[str]) -> None:
    """/sync_portfolio — sincroniza a carteira actual via Telegram.

    Uso:
      /sync_portfolio                         → ver estado actual
      /sync_portfolio AAPL:10:150 MSFT:5:300  → substituir carteira
      /sync_portfolio clear                   → limpar override (volta a usar env vars)

    Formato: TICKER:shares:preco_medio_usd
      Exemplo: /sync_portfolio NVO:25:85.50 ADBE:8:420 CRWD:5:280

    Os dados ficam em /data/portfolio_override.json (persistido no Railway Volume).
    As env vars HOLDING_* continuam como backup se o override for limpo.
    """
    from pathlib import Path
    import json

    _OVERRIDE_PATH = Path("/data/portfolio_override.json") if Path("/data").exists() \
        else Path("/tmp/portfolio_override.json")

    # Sem argumentos → mostrar estado actual
    if len(parts) < 2:
        if _OVERRIDE_PATH.exists():
            try:
                data = json.loads(_OVERRIDE_PATH.read_text(encoding="utf-8"))
                holdings = data.get("holdings", {})
                lines = ["Carteira actual (override activo):"]
                for ticker, pos in sorted(holdings.items()):
                    lines.append(
                        f"  {ticker}: {pos['shares']} shares @ ${pos['avg_cost']:.2f}"
                    )
                lines.append("")
                lines.append("Para actualizar: /sync_portfolio TICK:shares:preco ...")
                lines.append("Para limpar: /sync_portfolio clear")
                _reply("\n".join(lines))
            except Exception as e:
                _reply(f"Erro a ler override: {e}")
        else:
            _reply(
                "Sem override activo — a usar env vars HOLDING_* do Railway.\n\n"
                "Para sincronizar a tua carteira actual:\n"
                "/sync_portfolio NVO:25:85.50 ADBE:8:420 CRWD:5:280\n"
                "(formato: TICKER:shares:preco_medio_usd)"
            )
        return

    # Limpar override
    if parts[1].lower() == "clear":
        if _OVERRIDE_PATH.exists():
            _OVERRIDE_PATH.unlink()
            _reply("Override limpo. A usar env vars HOLDING_* do Railway.")
        else:
            _reply("Sem override activo.")
        return

    # Parsear nova carteira
    holdings: dict = {}
    errors: list[str] = []
    for entry in parts[1:]:
        try:
            parts_entry = entry.split(":")
            if len(parts_entry) < 2:
                errors.append(f"Formato inválido: {entry} (usa TICKER:shares ou TICKER:shares:preco)")
                continue
            ticker     = parts_entry[0].upper().strip()
            shares     = float(parts_entry[1])
            avg_cost   = float(parts_entry[2]) if len(parts_entry) > 2 else 0.0
            holdings[ticker] = {"shares": shares, "avg_cost": avg_cost}
        except (ValueError, IndexError):
            errors.append(f"Erro a parsear: {entry}")

    if errors:
        _reply("Erros:\n" + "\n".join(errors))
        return

    if not holdings:
        _reply("Nenhuma posição válida. Formato: TICKER:shares:preco_medio_usd")
        return

    # Guardar override
    try:
        _OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "holdings":   holdings,
            "updated_at": datetime.now().isoformat(),
            "updated_by": "telegram_sync",
        }
        _OVERRIDE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

        lines = ["Carteira actualizada:"]
        for ticker, pos in sorted(holdings.items()):
            lines.append(f"  {ticker}: {pos['shares']} shares @ ${pos['avg_cost']:.2f}")
        lines.append("")
        lines.append(f"Guardado em {_OVERRIDE_PATH}")
        lines.append("Activo a partir de agora. Para limpar: /sync_portfolio clear")
        _reply("\n".join(lines))

        # Recarregar a carteira em portfolio.py (se suportado)
        try:
            import portfolio as _pf
            _pf._reload_from_override(_OVERRIDE_PATH)
        except Exception:
            pass  # reload opcional

    except Exception as e:
        _reply(f"Erro ao guardar override: {e}")


# ── /ml_accuracy ─────────────────────────────────────────────────────────────────

def _handle_ml_accuracy() -> None:
    """Mede a precisao real do modelo comparando predicoes com outcomes reais.
    Este e o backtest continuo que fecha o loop alerta -> resultado -> modelo.
    """
    def _run():
        try:
            from prediction_log import compute_ml_accuracy
            acc = compute_ml_accuracy()
            if acc.get("skipped"):
                _reply(f"ML Accuracy: {acc.get('reason', 'sem dados')}")
                return
            n    = acc["n_resolved"]
            prec = acc["precision"]
            rec  = acc["recall"]
            accu = acc["accuracy"]
            f1   = acc["f1"]
            brier = acc["brier_live"]
            win_r = acc["win_rate_actual"]
            lines = [
                f"ML Accuracy ({n} predicoes com outcome resolvido)",
                "",
                f"Precision: {prec:.0%}  (de cada COMPRAR, quantos ganharam)",
                f"Recall: {rec:.0%}  (de todos os ganhos, quantos identificamos)",
                f"Accuracy: {accu:.0%}  (total de acertos)",
                f"F1 Score: {f1:.3f}  (equilibrio precision/recall)",
                f"Brier live: {brier:.4f}  (calibracao das probabilidades)",
                f"Win rate real: {win_r:.0%}  (stocks que efectivamente subiram)",
                "",
                f"TP:{acc['tp']}  FP:{acc['fp']}  FN:{acc['fn']}  TN:{acc['tn']}",
            ]
            _reply("\n".join(lines))
        except Exception as e:
            _reply(f"Erro ao calcular accuracy: {e}")
    threading.Thread(target=_run, daemon=True).start()


# ── /admin_check_config ──────────────────────────────────────────────────────────

def _handle_admin_check_config() -> None:
    """Verifica todas as env vars criticas e mostra o estado do sistema."""
    import os
    lines = ["Config check — DipRadar", ""]

    checks = [
        # (env_var, label, obrigatorio, descricao)
        ("MONTHLY_BUDGET_EUR",      "Orcamento mensal",      True,  "Ex: 1050"),
        ("FLIP_FUND_EUR",           "Capital Flip Fund",     True,  "Ex: 500 — o Tesoureiro usa isto para sizing"),
        ("TIINGO_API_KEY",          "Tiingo EOD",            False, "Dados EOD (fallback yfinance se ausente)"),
        ("ALPHAVANTAGE_API_KEY",    "Alpha Vantage",         False, "25 req/dia gratis — revisoes analistas"),
        ("FMP_API_KEY",             "FMP",                   False, "250 req/dia gratis — upgrades/downgrades"),
        ("FRED_API_KEY",            "FRED (Fed Reserve)",    False, "Gratis — recession probability mais precisa"),
        ("SECTOR_CONCENTRATION_CAP","Sector cap",            False, "Default 35% — limite de exposicao por sector"),
        ("TELEGRAM_TOKEN",          "Telegram Bot",          True,  "Token do bot"),
        ("TELEGRAM_CHAT_ID",        "Telegram Chat",         True,  "ID do chat"),
    ]

    ok_count = 0
    warn_count = 0
    for env, label, required, desc in checks:
        val = os.environ.get(env, "")
        if val:
            ok_count += 1
            display = "****" + val[-4:] if len(val) > 6 else "***"
            lines.append(f"OK  {label}: {display}")
        elif required:
            warn_count += 1
            lines.append(f"FALTA  {label} — {desc}")
        else:
            lines.append(f"opcional  {label} — {desc}")

    lines.append("")
    lines.append(f"Resultado: {ok_count} configuradas, {warn_count} obrigatorias em falta")

    # Estado do modelo ML
    try:
        from ml_predictor import is_model_ready, get_model_info
        if is_model_ready():
            info = get_model_info()
            lines.append(f"Modelo ML: pronto (IC={info.get('rho_mean', '?')})")
        else:
            lines.append("Modelo ML: NAO treinado — faz /admin_retrain")
    except Exception:
        lines.append("Modelo ML: erro ao verificar")

    _reply("\n".join(lines))


# ── /admin_regen_parquet ────────────────────────────────────────────────────────

def _handle_admin_regen_parquet(parts: list[str]) -> None:
    """/admin_regen_parquet [--targets-only] [--fundamentals-only] [--no-fundamentals]

    Reconstrói ml_training_base.parquet com:
      - Fundamentais point-in-time (SEC EDGAR + yfinance quarterly)
      - Target alpha_90d (sem fallback para 60d)
      - Features técnicas corrigidas

    Duração: 45-90 min (1ª execução) | 5-15 min (re-runs com cache)

    Flags:
      --targets-only       → só adiciona/actualiza alpha_90d (rápido, ~10 min)
      --fundamentals-only  → só fundamentais PIT (sem re-calcular features)
      --no-fundamentals    → pula fundamentais (só features + targets)
    """
    if _retrain_running:
        _reply("⚠️ Retrain já em curso — aguarda que termine antes de regenerar.")
        return

    flags = [p for p in parts[1:] if p.startswith("--")]
    mode_str = " ".join(flags) if flags else "completo"

    _targets_only = "--targets-only" in flags
    _dur = "~10-20 min" if _targets_only else "3-5h (1a vez) | ~30 min (re-run com cache)"
    _what = "- Target alpha-90d" if _targets_only else "- Fundamentais PIT + Target alpha-90d"
    _reply(
        f"*Regeneracao de parquet - {mode_str}*\n"
        f"{_what}\n"
        f"Cache em /data/price-cache (persiste entre restarts)\n"
        f"Duracao: {_dur}\n"
        f"Podes continuar a usar o bot."
    )
    logging.info(f"[regen] Iniciando regeneração modo={mode_str}")

    def _run():
        import subprocess, sys
        from pathlib import Path
        try:
            script = Path(__file__).parent / "scripts" / "regenerate_training_base.py"
            args   = [sys.executable, str(script)] + flags

            # Path do parquet
            data_dir = Path("/data") if Path("/data").exists() else Path("/tmp")
            # OUTPUT: SEMPRE em /data/ para persistir entre deploys.
            # Se escrevermos no repo (/app/), o próximo deploy sobrescreve e
            # o alpha_90d gerado é perdido.
            out_pq = data_dir / "ml_training_base.parquet"

            # INPUT: /data/ se existir; senão lê do repo como base.
            in_pq = out_pq
            if not in_pq.exists():
                for alt in [data_dir / "ml_training_merged.parquet",
                             Path(__file__).parent / "ml_training_base.parquet"]:
                    if alt.exists():
                        in_pq = alt
                        break

            if not in_pq.exists():
                _reply("Regeneracao falhou: parquet de treino nao encontrado.")
                return

            args += ["--in", str(in_pq), "--out", str(out_pq)]
            # Cache em /data/ (persistido no Railway Volume) para não perder
            # o progresso em restarts. /tmp seria apagado e obrigaria re-download.
            cache_dir = data_dir / "price_cache"
            args += ["--cache", str(cache_dir)]

            # Timeout generoso: full regen com EDGAR + 700 tickers demora 3-5h
            # na 1ª execução; com cache (re-runs) demora ~20-30 min.
            _TIMEOUT = 6 * 3600  # 6 horas
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=_TIMEOUT
            )
            if result.returncode == 0:
                try:
                    import pandas as pd
                    df_check = pd.read_parquet(out_pq)
                    n_90d  = int(df_check["alpha_90d"].notna().sum()) if "alpha_90d" in df_check.columns else 0
                    n_fund = int((df_check.get("gross_margin", pd.Series()) != 0.35).sum()) if "gross_margin" in df_check.columns else 0
                    total  = len(df_check)
                    pct = f"{n_90d/total:.0%}" if total else "0%"
                    _reply(
                        f"*Parquet regenerado!*\n"
                        f"  Shape: {df_check.shape[0]:,} x {df_check.shape[1]} colunas\n"
                        f"  alpha-90d resolvido: {n_90d:,}/{total:,} ({pct})\n"
                        f"  Fundamentais PIT: {n_fund:,}/{total:,} linhas\n\n"
                        f"Faz /admin\\_retrain para treinar o modelo."
                    )
                except Exception as e:
                    _reply(f"Parquet regenerado. (verificacao: {e})\nFaz /admin\\_retrain")
            else:
                stderr = (result.stderr or "")[-500:]
                _reply(f"Regeneracao falhou (exit {result.returncode})\n{stderr}")
        except subprocess.TimeoutExpired:
            _reply(
                "Regeneracao excedeu 6h. O Railway pode ter reiniciado o container.\n"
                "O cache de precos esta preservado em /data/price-cache.\n"
                "Tenta novamente com /admin\\_regen\\_parquet --targets-only"
            )
        except Exception as e:
            logging.error(f"[regen] {e}", exc_info=True)
            _reply(f"❌ *Erro na regeneração:*\n`{_md_safe(e)}`")

    threading.Thread(target=_run, daemon=True, name="regen-parquet").start()


# ── /admin_retrain ──────────────────────────────────────────────────────────────

def _handle_admin_retrain(parts: list[str]) -> None:
    """
    /admin_retrain                    → dispara retreino mensal v3 ad-hoc (full)
    /admin_retrain dry-run            → só constrói training input (não treina)
    /admin_retrain dry-run no-snap    → exclui universe_snapshot.parquet
    /admin_retrain run 0.85           → retrain real com gating_ratio override

    Útil para validar `/data/` no Railway sem esperar pelo cron mensal.
    Retorna métricas (ρ_α, Brier, Top-K PnL) ou shape do training input se dry-run.
    """
    global _retrain_running

    if _retrain_running:
        _reply("⚠️ *Retreino já está a correr.*\n_Aguarda a conclusão._")
        return

    args = [p.lower() for p in parts[1:]]
    dry_run          = "dry-run" in args or "dry_run" in args or "dryrun" in args
    include_snapshot = "no-snap" not in args and "no-snapshot" not in args
    include_alert_db = "no-alert-db" not in args and "no-alerts" not in args
    gating_ratio     = None
    for a in args:
        try:
            v = float(a)
            if 0.5 <= v <= 1.5:
                gating_ratio = v
                break
        except ValueError:
            continue

    mode_str = "DRY-RUN" if dry_run else "REAL"
    extras = []
    if not include_snapshot: extras.append("no-snap")
    if not include_alert_db: extras.append("no-alert-db")
    if gating_ratio is not None: extras.append(f"ratio={gating_ratio}")
    extras_str = f" ({', '.join(extras)})" if extras else ""

    _reply(
        f"Retrain iniciado{' (modo: ' + mode_str + ')' if mode_str != 'REAL' else ''}.\n"
        f"{'A validar dados (dry-run).' if dry_run else 'A treinar o modelo — pode demorar 10-30 min. O bot continua a funcionar.'}"
    )
    logging.info(f"[admin_retrain] start mode={mode_str} extras={extras_str}")

    def _run() -> None:
        global _retrain_running
        _retrain_running = True
        start_ts = time.time()
        try:
            from monthly_retrain import run_monthly_retrain_v3
            kwargs = dict(
                dry_run=dry_run,
                include_snapshot=include_snapshot,
                include_alert_db=include_alert_db,
            )
            if gating_ratio is not None:
                kwargs["gating_ratio"] = gating_ratio
            result = run_monthly_retrain_v3(**kwargs)
        except Exception as e:
            logging.error(f"[admin_retrain] {e}", exc_info=True)
            _reply(f"❌ *Retrain falhou:*\n`{_md_safe(e)}`")
            return
        finally:
            _retrain_running = False

        elapsed = time.time() - start_ts
        decision = result.get("decision", "?")
        icon = {
            "PROMOTED":   "🚀",
            "PENDING":    "⚠️",
            "KEPT_FLOOR": "🛑",
            "FAILED":     "❌",
            "DRY-RUN":    "🧪",
        }.get(decision, "⚙️")

        # ── Resumo em linguagem simples ────────────────────────────────────
        plain = {
            "PROMOTED":   "✅ Modelo actualizado — o candidato é melhor e já está em produção.",
            "PENDING":    "⚠️ Modelo candidato ficou guardado mas não foi para produção — não melhorou o suficiente.",
            "KEPT_FLOOR": "🛑 Modelo candidato recusado — qualidade preditiva abaixo do mínimo aceitável.",
            "FAILED":     "❌ Treino falhou — verifica os logs para perceber o que correu mal.",
            "DRY-RUN":    "🧪 Simulação concluída — dados carregados, nenhum modelo foi treinado.",
        }.get(decision, "⚙️ Treino concluído.")

        lines = [
            f"{icon} *Retrain — {decision}*",
            f"_{plain}_",
            f"_{datetime.now().strftime('%d/%m/%Y %H:%M')} ({elapsed:.0f}s)_",
            "",
        ]
        reason = result.get("reason")
        if reason:
            # Code span — dynamic strings com `_` (alert_db, paths) partem
            # Markdown V1 quando wrapped em italics. Backticks tornam o
            # conteúdo literal e evitam 400 Bad Request do Telegram.
            lines += [f"`{_md_safe(reason)}`", ""]

        if decision == "DRY-RUN":
            ti_path = result.get("training_input")
            if ti_path:
                lines.append(f"*Training input:* `{_md_safe(ti_path)}`")
                try:
                    import pandas as pd
                    df = pd.read_parquet(ti_path)
                    lines.append(f"*Shape:* {df.shape[0]} linhas × {df.shape[1]} colunas")
                    if "alert_date" in df.columns:
                        d = pd.to_datetime(df["alert_date"])
                        lines.append(f"*Período:* {d.min().date()} → {d.max().date()}")
                    if "symbol" in df.columns:
                        lines.append(f"*Tickers únicos:* {df['symbol'].nunique()}")
                    if "label_win" in df.columns:
                        n_win = int(df["label_win"].fillna(0).sum())
                        lines.append(f"*Labels WIN:* {n_win}/{len(df)}")
                    if "spy_return_ref" in df.columns:
                        n_with_target = int(df["spy_return_ref"].notna().sum())
                        lines.append(f"*Com target resolvido:* {n_with_target}/{len(df)}")
                except Exception as e:
                    lines.append(f"(Falha a ler shape: `{_md_safe(e)}`)")
            outcomes = result.get("outcome_stats") or {}
            if outcomes:
                lines += [
                    "",
                    f"*alert_db outcomes preenchidos:* {outcomes.get('updated', 0)}",
                    f"*alert_db ignorados:* {outcomes.get('skipped', 0)}",
                ]
            _reply("\n".join(lines))
            return

        # Real retrain — métricas
        cand_rho   = result.get("candidate_rho_alpha")
        prod_rho   = result.get("production_rho_alpha")
        cand_brier = result.get("candidate_brier")
        prod_brier = result.get("production_brier")
        cand_pnl   = result.get("candidate_topk_pnl")
        prod_pnl   = result.get("production_topk_pnl")

        def _fmt(v):
            if v is None: return "N/A"
            try: return f"{float(v):.4f}"
            except (TypeError, ValueError): return str(v)

        if cand_rho is not None or prod_rho is not None:
            # Explicação amigável: IC mede quão bem o modelo ordena os stocks
            # por retorno futuro. Quanto maior, melhor. 0.15+ é bom sinal.
            try:
                delta_pct = (float(cand_rho) - float(prod_rho)) / float(prod_rho) * 100
                sign = "+" if delta_pct >= 0 else ""
                quality_note = (
                    "📈 Melhor que o anterior" if delta_pct > 0
                    else "📉 Ligeiramente inferior ao anterior"
                )
                lines += [
                    f"*Qualidade preditiva (IC):* {quality_note}",
                    f"  Novo modelo : *{_fmt(cand_rho)}*   Anterior: {_fmt(prod_rho)}   ({sign}{delta_pct:.1f}%)",
                    f"  _(IC mede quão bem o modelo prevê quais stocks vão superar o SPY)_",
                    "",
                ]
            except (TypeError, ValueError, ZeroDivisionError):
                lines += [
                    f"*Qualidade preditiva (IC):* candidato *{_fmt(cand_rho)}* / produção *{_fmt(prod_rho)}*",
                    "",
                ]

        if cand_pnl is not None:
            try:
                pnl_delta = (float(cand_pnl) - float(prod_pnl)) / abs(float(prod_pnl)) * 100
                sign = "+" if pnl_delta >= 0 else ""
                lines += [
                    f"*Retorno simulado top stocks:* candidato *{float(cand_pnl)*100:.1f}%* / anterior *{float(prod_pnl or 0)*100:.1f}%* ({sign}{pnl_delta:.1f}%)",
                    f"  _(média de retorno dos 12% stocks melhor pontuados pelo modelo)_",
                    "",
                ]
            except (TypeError, ValueError, ZeroDivisionError):
                lines += [f"*Top-K PnL:* candidato *{_fmt(cand_pnl)}* / produção *{_fmt(prod_pnl)}*", ""]

        if cand_brier is not None:
            lines += [
                f"*Calibração de probabilidades (Brier):* candidato *{_fmt(cand_brier)}* / anterior *{_fmt(prod_brier)}*",
                f"  _(quanto mais baixo, mais fiáveis são os % de confiança mostrados nos alertas)_",
                "",
            ]

        # Dataset health verdict
        dh = result.get("dataset_health") or {}
        if dh and not dh.get("skipped"):
            verdict_emoji = {"production_ready": "🟢", "marginal": "🟡",
                             "unstable": "🟠", "low_volume": "🔴"}.get(dh.get("verdict", ""), "⚪")
            ic_sr = dh.get("ic_sr")
            ic_mean = dh.get("ic_mean")
            pct_pos = dh.get("pct_pos")
            lines += [
                f"*Qualidade do sinal:* {verdict_emoji} {dh.get('verdict', '?')}",
                f"  IC médio: *{ic_mean:.4f}* | IC SR: *{ic_sr:.2f}* | Folds positivos: *{pct_pos:.0%}*",
                f"  _(IC SR > 0.5 e IC > 0.05 = sinal real e consistente)_",
                "",
            ]

        # Comparação detalhada candidato vs produção (para PENDING / KEPT_FLOOR)
        detail = result.get("comparison_detail") or {}
        if detail and decision in ("PENDING", "KEPT_FLOOR"):
            rows = detail.get("rows", [])
            improved = detail.get("improved", [])
            worsened = detail.get("worsened", [])
            rec      = detail.get("recommendation", "")
            if rows:
                lines += ["", "*🔍 O que mudou vs modelo actual:*"]
                for r in rows:
                    cval = f"{r['cand']}" if r['cand'] is not None else "N/A"
                    pval = f"{r['prod']}" if r['prod'] is not None else "N/A"
                    if r.get("delta_pct") is not None:
                        sign = "+" if r["delta_pct"] >= 0 else ""
                        delta_str = f"({sign}{r['delta_pct']:.1f}%)"
                        em = "✓" if r.get("improved") else "❌" if r.get("improved") is False else "—"
                    else:
                        delta_str, em = "", "—"
                    lines.append(f"  {em} *{r['label']}*: {cval} ← {pval} {delta_str}")
            if rec:
                lines += ["", f"_💡 {rec}_"]

        if decision == "PENDING":
            lines.append("_Bundle guardado como `dip_models_pending.pkl` — revisão manual._")
        elif decision == "PROMOTED":
            lines.append("_Bundle promovido em `/data/`. Reinicia o bot ou força reload via_ `/admin_load_models`.")

        _reply("\n".join(lines))
        logging.info(f"[admin_retrain] decision={decision} cand_rho={cand_rho}")

    threading.Thread(target=_run, daemon=True, name="admin-retrain").start()

# ── /retrigger ──────────────────────────────────────────────────────────────────

def _handle_retrigger() -> None:
    """
    /retrigger  → alias rápido de /admin_retrain (modo REAL, sem flags extra).

    Útil para disparar o pipeline mensal ad-hoc sem ter de escrever o comando
    completo. Delega inteiramente para _handle_admin_retrain com parts=[""].
    """
    _handle_admin_retrain(["admin_retrain"])


# ── /admin_set_floor ────────────────────────────────────────────────────────────

def _handle_admin_set_floor(parts: list[str]) -> None:
    """
    /admin_set_floor <valor>   → escreve o novo floor absoluto em FLOOR_PATH.

    Ex.: /admin_set_floor 0.08  baixa o floor para 0.08 (range válido: [0.0, 0.5]).
    O retrain mensal seguinte usa o novo valor. Mostra valor anterior + novo.
    """
    args = [a for a in parts[1:] if a.strip()]
    if len(args) != 1:
        _reply(
            "❌ *Uso incorrecto.*\n\n"
            "`/admin_set_floor <valor>`\n\n"
            "Ex.: `/admin_set_floor 0.08`\n"
            "_Range válido: 0.0 a 0.5._"
        )
        return
    try:
        new_value = float(args[0])
    except ValueError:
        _reply(f"❌ Valor não numérico: `{args[0]}`")
        return

    try:
        from monthly_retrain import set_floor_rho_alpha
        result = set_floor_rho_alpha(new_value, comment="Ajustado via /admin_set_floor.")
    except ValueError as e:
        _reply(f"❌ *Valor fora de range:*\n`{_md_safe(e)}`")
        return
    except Exception as e:
        logging.error(f"[admin_set_floor] {e}", exc_info=True)
        _reply(f"❌ *Falhou:*\n`{_md_safe(e)}`")
        return

    _reply(
        "🔧 *Floor ajustado*\n"
        f"  • anterior : `{result['old']:.4f}`\n"
        f"  • novo     : `{result['new']:.4f}`\n"
        f"  • path     : `{result['path']}`\n\n"
        "_Aplicado no próximo `/admin_retrain` ou cron mensal._"
    )
    logging.info(f"[admin_set_floor] {result['old']:.4f} → {result['new']:.4f}")

# ── /paper_portfolio ─────────────────────────────────────────────────────────────

def _handle_paper_performance(parts: list[str]) -> None:
    """/paper_portfolio [meses]

    Mostra a performance do paper trading automatico do bot.
    O bot simula as suas proprias recomendacoes COMPRAR e mede se bate o SPY.

    Uso:
      /paper_portfolio        → ultimos 3 meses
      /paper_portfolio 6      → ultimos 6 meses
      /paper_portfolio open   → posicoes abertas actuais
    """
    def _run():
        try:
            from paper_trading import (
                get_monthly_performance, format_performance_report, _load
            )

            if len(parts) > 1 and parts[1].lower() == "open":
                trades  = _load()
                open_ps = [t for t in trades if t.get("status") == "OPEN"]
                if not open_ps:
                    _reply("Sem posicoes de paper trading abertas.")
                    return
                lines = [f"Paper portfolio — {len(open_ps)} posicoes abertas:", ""]
                for t in sorted(open_ps, key=lambda x: x["open_date"], reverse=True)[:15]:
                    from datetime import date
                    days = (date.today() - date.fromisoformat(t["open_date"])).days
                    lines.append(
                        f"  {t['ticker']}: aberto {t['open_date']} "
                        f"({days}d) @ {t['open_price']:.2f} "
                        f"| target {t['sell_target']:.2f} | €{t['amount_eur']:.0f}"
                    )
                _reply("\n".join(lines))
                return

            months = 3
            if len(parts) > 1:
                try:
                    months = int(parts[1])
                except ValueError:
                    pass

            perf   = get_monthly_performance(months_back=months)
            report = format_performance_report(perf)
            _reply(f"Paper Trading — Bot vs Mercado\n\n{report}")

            # Lista detalhada de trades por mês
            from paper_trading import format_monthly_trade_list
            trade_list = format_monthly_trade_list(months_back=months)
            if trade_list and trade_list != "Sem trades fechados no periodo.":
                _reply(f"Trades por mes:\n{trade_list}")
        except Exception as e:
            logging.error(f"[paper_performance] {e}", exc_info=True)
            _reply(f"Erro ao calcular paper performance: {e}")

    threading.Thread(target=_run, daemon=True, name="paper-perf").start()


# ── /performance ─────────────────────────────────────────────────────────────────

def _handle_performance(parts: list[str]) -> None:
    """/performance [YYYY-MM-DD] [score_min]

    Simula o portfolio DipRadar e calcula retorno anual, Sharpe e max drawdown.

    Exemplos:
      /performance               → todo o histórico disponível
      /performance 2025-01-01    → desde Jan 2025
      /performance 2024-01-01 65 → desde Jan 2024, score mínimo 65
    """
    _reply(
        "⏳ *A calcular performance do portfolio...*\n"
        "_Lê o alert\\_db.csv e simula seguir todas as recomendações. Pode demorar 30s._"
    )

    def _run():
        try:
            from portfolio_simulator import run_portfolio_backtest, format_portfolio_result

            period_start = None
            score_min    = 60.0

            for p in parts[1:]:
                p = p.strip()
                try:
                    float(p)
                    score_min = float(p)
                    continue
                except ValueError:
                    pass
                if len(p) == 10 and p[4] == "-":
                    period_start = p

            result = run_portfolio_backtest(
                period_start=period_start,
                score_threshold=score_min,
            )
            _reply(format_portfolio_result(result))
        except RuntimeError as e:
            _reply(f"⚠️ *Performance:* {e}\n_Certifica-te que o alert\\_db.csv tem outcomes preenchidos. Usa /mldata update._")
        except Exception as e:
            logging.error(f"[performance] {e}", exc_info=True)
            _reply(f"❌ *Erro ao calcular performance:*\n`{_md_safe(e)}`")

    threading.Thread(target=_run, daemon=True, name="performance").start()


# ── /themes · /add_theme · /remove_theme ────────────────────────────────────────

def _handle_themes() -> None:
    """Lista todos os temas activos com rationale e bonus de sizing."""
    try:
        from themes import format_themes_list
        _reply(format_themes_list())
    except Exception as e:
        _reply(f"_Erro ao carregar temas: {e}_")


def _handle_add_theme(parts: list[str]) -> None:
    """/add_theme <key> <label (multi-word)> <TICK1,TICK2,...> [confiança 0-1]

    Exemplos:
      /add_theme robotics Robótica IRBT,ABB,FANUY 0.75
      /add_theme biotech2 Biotech Oncologia MRNA,BEAM,CRSP
    """
    if len(parts) < 4:
        _reply(
            "⚠️ Uso: `/add\\_theme <key> <label> <TICK1,TICK2,...> [confiança]`\n\n"
            "Exemplos:\n"
            "`/add_theme robotics Robótica IRBT,ABB,FANUY 0.75`\n"
            "`/add_theme biotech2 Biotech Oncologia MRNA,BEAM,CRSP`"
        )
        return

    key     = parts[1].lower().strip()
    # Último token numérico = confiança (opcional)
    confidence = 0.75
    ticker_idx = 3
    if len(parts) >= 5:
        try:
            conf_candidate = float(parts[-1])
            if 0.0 <= conf_candidate <= 1.0:
                confidence = conf_candidate
                ticker_idx = 3
                # label = todos os tokens entre key e tickers, excluindo o último (confiança)
        except ValueError:
            pass

    label   = parts[2]
    tickers = [t.upper().strip() for t in parts[ticker_idx].split(",") if t.strip()]
    if not tickers:
        _reply("_Nenhum ticker válido. Usa formato: TICK1,TICK2,TICK3_")
        return

    try:
        from themes import add_theme
        t = add_theme(key, label, tickers, confidence=confidence, added_by="user_bot")
        _reply(
            f"✅ *Tema adicionado:* `{key}`\n"
            f"  Label: {t['label']}\n"
            f"  Tickers: {', '.join(t['tickers'])}\n"
            f"  Confiança: {t['confidence']:.0%}  |  Bonus: ver /themes"
        )
    except Exception as e:
        _reply(f"_Erro ao adicionar tema: {e}_")


def _handle_remove_theme(parts: list[str]) -> None:
    """/remove_theme <key>"""
    if len(parts) < 2:
        _reply("⚠️ Uso: `/remove\\_theme <key>`  (ver keys em /themes)")
        return
    key = parts[1].lower().strip()
    try:
        from themes import remove_theme
        removed = remove_theme(key)
        if removed:
            _reply(f"✅ Tema `{key}` removido.")
        else:
            _reply(f"⚠️ Tema `{key}` não encontrado. Usa /themes para ver os keys activos.")
    except Exception as e:
        _reply(f"_Erro: {e}_")


# ── /health handler ─────────────────────────────────────────────────────────────

def _handle_health(parts: list[str]) -> None:
    """
    /health          → dashboard completo (RAM, CPU, scans, APIs, modelos, erros)
    /health errors   → log detalhado dos últimos erros críticos
    """
    sub = parts[1].lower() if len(parts) > 1 else "dashboard"

    if sub in ("errors", "erros", "log"):
        with health_monitor._lock:
            errors = list(health_monitor._error_log)
        if not errors:
            _reply("✅ *Sem erros registados* — sistema limpo.")
            return
        lines = [f"*🚨 Log de erros — últimos {len(errors)}*", ""]
        for e in reversed(errors):
            ts_str  = e["ts"].strftime("%d/%m %H:%M:%S")
            tb_prev = e["tb"][-400:] if len(e["tb"]) > 400 else e["tb"]
            lines.append(f"🔴 `{ts_str}` *[{e['context']}]*")
            lines.append(f"   _{e['error'][:120]}_")
            lines.append(f"```\n{tb_prev}\n```")
            lines.append("")
        _reply("\n".join(lines))
        return

    # Dashboard principal — corre em thread para não bloquear o poll
    _reply("_🔄 A recolher métricas... (2-5s)_")

    def _run():
        try:
            report = health_monitor.build_health_report(ping_apis=True)
            _reply(report)
        except Exception as e:
            _reply(f"❌ Erro ao construir health report: `{e}`")

    threading.Thread(target=_run, daemon=True).start()


# ── Earnings alerts ───────────────────────────────────────────────────────────────

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
    sent = 0
    for sym in all_tickers:
        try:
            edays = earnings_days_fn(sym)
            if edays is None: continue
            if 0 <= edays <= _EARNINGS_ALERT_DAYS:
                history = get_ticker_score_history(sym)
                already = any(
                    e.get("date_iso") == today_str and e.get("verdict", "").startswith("earnings_alert")
                    for e in history
                )
                if already: continue
                urgency   = "🚨 *HOJE*" if edays == 0 else (f"⏰ *{edays} dia(s)*" if edays <= 2 else f"📅 {edays} dia(s)")
                last_score = history[-1].get("score") if history else None
                score_str  = f" — último score: *{last_score:.0f}/100*" if last_score is not None else ""
                _reply(
                    f"📊 *Earnings Alert* — *{sym}*\n"
                    f"Resultados em: {urgency}{score_str}\n"
                    f"_Considera `/analisar {sym}` antes dos resultados._"
                )
                sent += 1
        except Exception as e:
            logging.warning(f"[earnings_alert] {sym}: {e}")
    return sent


# ── /watchlist handler ─────────────────────────────────────────────────────────────

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
        lines += ["", "_Remove com `/watchlist rm TICKER` · Limpa com `/watchlist clear`_"]
        _reply("\n".join(lines))
    elif sub in ("add", "adicionar", "+"):
        if len(parts) < 3:
            _reply("⚠️ Uso: `/watchlist add <TICKER>`\n_Exemplos: `AAPL`, `IEMA.L`, `IS3N.AS`, `ALV.DE`_")
            return
        # Preserva sufixo de exchange (ex: .L, .AS, .DE) — não faz split no ponto
        ticker = parts[2].upper().strip()
        # Valida: até 15 chars, só letras/números/ponto/hífen (ex: BRK-B, IS3N.AS)
        ticker_clean = ticker.replace(".", "").replace("-", "").replace("_", "")
        if len(ticker) > 15 or not ticker_clean.isalnum() or not ticker_clean:
            _reply(f"⚠️ Ticker inválido: `{ticker}`\n_Exemplos válidos: `AAPL`, `IEMA.L`, `ALV.DE`, `BRK-B`_")
            return
        added = add_to_dynamic_watchlist(ticker)
        if added:
            _reply(f"✅ *`{ticker}`* adicionado. Total: {len(load_dynamic_watchlist())} tickers.")
        else:
            _reply(f"_`{ticker}` já está na watchlist._")
    elif sub in ("rm", "remove", "remover", "del", "delete", "-"):
        if len(parts) < 3:
            _reply("⚠️ Uso: `/watchlist rm <TICKER>`")
            return
        ticker  = parts[2].upper().strip()
        removed = remove_from_dynamic_watchlist(ticker)
        if removed:
            _reply(f"🗑️ *`{ticker}`* removido. Restam {len(load_dynamic_watchlist())} tickers.")
        else:
            _reply(f"⚠️ `{ticker}` não está na watchlist.")
    elif sub in ("clear", "limpar", "reset"):
        count = len(load_dynamic_watchlist())
        if count == 0:
            _reply("_A watchlist já está vazia._")
            return
        save_dynamic_watchlist([])
        _reply(f"🧹 Watchlist limpa — {count} ticker(s) removido(s).")
    else:
        _reply(
            f"⚠️ Sub-comando desconhecido: `{sub}`\n"
            "`/watchlist` · `/watchlist add TICK` · `/watchlist rm TICK` · `/watchlist clear`"
        )


# ── /flip handler ─────────────────────────────────────────────────────────────────

# ── /allocate ────────────────────────────────────────────────────────────────

def _handle_allocate(parts: list[str]) -> None:
    """Sugere alocação read-only para um ticker (Fase 1 — sem execução).

    Usa o callback _cb_allocate_ticker registado pelo main.py via setup().
    """
    if len(parts) < 2:
        _reply(
            "⚠️ Usa: `/allocate <TICKER>`\n"
            "_Exemplo: `/allocate NVO`_\n"
            "_Sugere categoria (Core / High Conviction / Growth / Flip)\n"
            "+ sizing em €. Não executa nenhuma ordem._"
        )
        return
    symbol = parts[1].upper().strip()
    fn = _cb_allocate_ticker
    if fn is None:
        _reply(
            "_Allocation engine não disponível neste deployment._\n"
            "_(verifica que main.py passou `allocate_fn=` ao bot_commands.setup)_"
        )
        return
    _reply(f"_💼 A calcular alocação para *{symbol}*..._")
    threading.Thread(
        target=lambda: _reply(fn(symbol)),
        daemon=True,
    ).start()


def _handle_flip(parts: list[str]) -> None:
    sub = parts[1].lower() if len(parts) > 1 else "summary"

    if sub in ("summary", "list", "ls", "ver", "show") or len(parts) == 1:
        summary  = get_flip_summary()
        pnl      = summary["total_pnl"]
        pnl_em   = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
        lines = [
            f"*🎯 Flip Fund — {datetime.now().strftime('%d/%m/%Y')}*", "",
            f"  {pnl_em} *P&L realizado:* ${'+' if pnl >= 0 else ''}{pnl:.2f}",
            f"  📊 Trades fechados: *{summary['n_closed']}* | Win rate: *{summary['win_rate']:.0f}%*",
        ]
        if summary["best_trade"]:
            b = summary["best_trade"]
            lines.append(f"  🏆 Melhor: *{b['symbol']}* ${b['pnl_eur']:+.2f}")
        opened = summary["trades_open"]
        if opened:
            lines += ["", f"*📂 Posições abertas ({len(opened)}):*"]
            for t in opened:
                notes_str = f" — _{t['notes']}_" if t.get("notes") else ""
                lines.append(f"  #{t['id']} *{t['symbol']}* x{t['shares']} @ ${t['price_entry']:.2f} (desde {t['date_entry']}){notes_str}")
        else:
            lines += ["", "_Sem posições abertas._"]
        if sub in ("list", "ls"):
            closed = summary["trades_closed"]
            if closed:
                lines += ["", f"*✅ Trades fechados ({len(closed)}):*"]
                for t in sorted(closed, key=lambda x: x["date_exit"] or "", reverse=True)[:10]:
                    pnl_s = f"${t['pnl_eur']:+.2f}" if t["pnl_eur"] is not None else "N/D"
                    em    = "🟢" if (t["pnl_eur"] or 0) > 0 else "🔴"
                    lines.append(f"  {em} #{t['id']} *{t['symbol']}* ${t['price_entry']:.2f}→${t['price_exit']:.2f} | *{pnl_s}*")
        lines += ["", "_`/flip add TICK ENTRY SHR` · `/flip close ID EXIT`_"]
        _reply("\n".join(lines))
        return

    if sub in ("add", "entrada", "open"):
        if len(parts) < 5:
            _reply("⚠️ Uso: `/flip add <TICKER> <ENTRADA> <SHARES> [nota]`")
            return
        ticker = parts[2].upper().strip()
        try: entry, shares = float(parts[3]), float(parts[4])
        except ValueError:
            _reply("⚠️ Preço e shares têm de ser números.")
            return
        if entry <= 0 or shares <= 0:
            _reply("⚠️ Valores têm de ser positivos.")
            return
        notes = " ".join(parts[5:]) if len(parts) > 5 else ""
        trade = add_flip_trade(ticker, shares, entry, notes=notes)
        _reply(
            f"✅ *Trade registado!*\n"
            f"  #{trade['id']} *{ticker}* x{shares} @ ${entry:.2f}\n"
            f"  💰 Custo: *${entry*shares:.2f}*\n"
            f"  _Fecha com `/flip close {trade['id']} <PREÇO>`_"
            + (f"\n  📝 _{notes}_" if notes else "")
        )
        return

    if sub in ("close", "fechar", "sell"):
        if len(parts) < 4:
            _reply("⚠️ Uso: `/flip close <ID> <PREÇO_SAÍDA>`")
            return
        try: trade_id, exit_px = int(parts[2]), float(parts[3])
        except ValueError:
            _reply("⚠️ ID inteiro e preço numérico.")
            return
        trade = close_flip_trade(trade_id, exit_px)
        if trade is None:
            _reply(f"⚠️ Trade `#{trade_id}` não encontrado ou já fechado.")
            return
        pnl = trade["pnl_eur"]
        pct = (exit_px - trade["price_entry"]) / trade["price_entry"] * 100
        em  = "🟢" if pnl > 0 else "🔴"
        _reply(
            f"{em} *Trade fechado!*\n"
            f"  #{trade_id} *{trade['symbol']}* — ${trade['price_entry']:.2f} → ${exit_px:.2f}\n"
            f"  *{'Lucro' if pnl > 0 else 'Perda'}: ${pnl:+.2f}* ({pct:+.1f}%)"
        )
        return

    if sub in ("del", "delete", "rm", "apagar"):
        if len(parts) < 3:
            _reply("⚠️ Uso: `/flip del <ID>`")
            return
        try: trade_id = int(parts[2])
        except ValueError:
            _reply("⚠️ ID tem de ser inteiro.")
            return
        if delete_flip_trade(trade_id):
            _reply(f"🗑️ Trade `#{trade_id}` removido.")
        else:
            _reply(f"⚠️ Trade `#{trade_id}` não encontrado.")
        return

    _reply("⚠️ Usa `/flip` · `/flip list` · `/flip add` · `/flip close` · `/flip del`")


# ── /buy handler ──────────────────────────────────────────────────────────────────

def _handle_buy(parts: list[str]) -> None:
    if len(parts) < 4:
        _reply(
            "⚠️ Uso: `/buy <TICKER> <PREÇO> <SHARES> [SCORE]`\n"
            "_Exemplo: `/buy CRWD 245.50 3 82`_\n"
            "_Score é opcional (0-100). ETFs não precisam de score._"
        )
        return

    symbol = parts[1].upper().strip()
    try:
        price  = float(parts[2])
        shares = float(parts[3])
    except ValueError:
        _reply("⚠️ Preço e shares têm de ser números.")
        return

    if price <= 0 or shares <= 0:
        _reply("⚠️ Preço e shares têm de ser positivos.")
        return

    score = None
    if len(parts) >= 5:
        try:
            score = int(float(parts[4]))
            if not (0 <= score <= 100):
                _reply("⚠️ Score tem de ser entre 0 e 100.")
                return
        except ValueError:
            _reply("⚠️ Score tem de ser um número inteiro.")
            return

    try:
        from portfolio import buy as portfolio_buy, get_liquidity
        result = portfolio_buy(symbol, price, shares, entry_score=score)
    except Exception as e:
        _reply(f"❌ Erro ao registar compra: `{e}`")
        return

    action_str = "📈 *DCA efectuado*" if result["action"] == "avg_down" else "🆕 *Nova posição aberta*"
    pos        = result["position"]
    liq        = result["liquidity"]
    liq_warn   = "\n⚠️ _Liquidez ficou negativa — faz `/liquidez +VALOR` para corrigir._" if result["liq_warning"] else ""

    _reply(
        f"✅ {action_str} — *{symbol}*\n\n"
        f"  💰 Compra: *{shares}x* @ *${price:.2f}* = *${result['cost']:.2f}*\n"
        f"  📊 Preço médio: *${pos['avg_price']:.2f}* | Total: *{pos['shares']}x*\n"
        f"  🏷️ Categoria: _{pos.get('category', 'N/D')}_"
        + (f"\n  🎯 Score entrada: *{score}/100*" if score is not None else "")
        + f"\n\n  💵 Liquidez restante: *€{liq:.2f}*"
        + liq_warn
    )


# ── /sell handler ─────────────────────────────────────────────────────────────────

def _handle_sell(parts: list[str]) -> None:
    if len(parts) < 3:
        _reply(
            "⚠️ Uso: `/sell <TICKER> <PREÇO> [SHARES]`\n"
            "_Omite SHARES para fechar toda a posição._\n"
            "_Exemplo: `/sell CRWD 280.00 2`_"
        )
        return

    symbol = parts[1].upper().strip()
    try:
        price = float(parts[2])
    except ValueError:
        _reply("⚠️ Preço tem de ser um número.")
        return

    if price <= 0:
        _reply("⚠️ Preço tem de ser positivo.")
        return

    shares = None
    if len(parts) >= 4:
        try:
            shares = float(parts[3])
            if shares <= 0:
                _reply("⚠️ Shares têm de ser positivas.")
                return
        except ValueError:
            _reply("⚠️ Shares têm de ser um número.")
            return

    try:
        from portfolio import sell as portfolio_sell
        result = portfolio_sell(symbol, price, shares)
    except Exception as e:
        _reply(f"❌ Erro ao registar venda: `{e}`")
        return

    if result is None:
        _reply(f"⚠️ Posição *{symbol}* não encontrada na carteira.\n_Usa `/portfolio` para ver as posições activas._")
        return

    pnl    = result["pnl"]
    pct    = result["pnl_pct"]
    em     = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
    action = "🏁 *Posição fechada*" if result["action"] == "closed" else "✂️ *Venda parcial*"

    remaining_str = ""
    if result["action"] == "partial":
        remaining_str = f"\n  📦 Restam: *{result['remaining']}x*"

    _reply(
        f"{em} {action} — *{symbol}*\n\n"
        f"  💰 Vendido: *{result['shares_sold']}x* @ *${price:.2f}* = *${result['proceeds']:.2f}*\n"
        f"  {'📈' if pnl >= 0 else '📉'} P&L: *${pnl:+.2f}* ({pct:+.1f}%)"
        + remaining_str
        + f"\n\n  💵 Liquidez: *€{result['liquidity']:.2f}*"
    )


# ── /liquidez handler ─────────────────────────────────────────────────────────────

def _handle_liquidez(parts: list[str]) -> None:
    try:
        from portfolio import get_liquidity, add_liquidity, set_liquidity
    except Exception as e:
        _reply(f"❌ Módulo portfolio não disponível: `{e}`")
        return

    if len(parts) < 2:
        liq = get_liquidity()
        em  = "🟢" if liq >= 0 else "🔴"
        _reply(
            f"💵 *Liquidez disponível:* {em} *€{liq:.2f}*\n\n"
            "_Comandos:_\n"
            "  `/liquidez +500`  → adicionar €500\n"
            "  `/liquidez -100`  → retirar €100\n"
            "  `/liquidez =1500` → definir saldo exacto"
        )
        return

    raw = parts[1].strip()

    if raw.startswith("="):
        try:
            amount = float(raw[1:])
        except ValueError:
            _reply("⚠️ Valor inválido. Exemplo: `/liquidez =1500`")
            return
        new_liq = set_liquidity(amount)
        _reply(f"✅ *Liquidez definida para €{new_liq:.2f}*")
        return

    if raw.startswith("+") or raw.startswith("-"):
        try:
            amount = float(raw)
        except ValueError:
            _reply("⚠️ Valor inválido. Exemplo: `/liquidez +500` ou `/liquidez -100`")
            return
        new_liq = add_liquidity(amount, note="ajuste manual Telegram")
        old_liq = new_liq - amount
        direction = "adicionado" if amount >= 0 else "retirado"
        em = "🟢" if new_liq >= 0 else "🔴"
        _reply(
            f"✅ *€{abs(amount):.2f} {direction}*\n"
            f"  €{old_liq:.2f} → {em} *€{new_liq:.2f}*"
        )
        return

    try:
        amount  = float(raw)
        new_liq = set_liquidity(amount)
        _reply(f"✅ *Liquidez definida para €{new_liq:.2f}*")
    except ValueError:
        _reply(
            "⚠️ Formato inválido.\n"
            "  `/liquidez`       → ver saldo\n"
            "  `/liquidez +500`  → adicionar\n"
            "  `/liquidez =1500` → definir"
        )


# ── helpers ───────────────────────────────────────────────────────────────────────

def _fetch_eur_usd() -> float:
    """Devolve taxa EUR/USD via yfinance. Fallback para 1.0 em caso de erro."""
    try:
        import yfinance as yf
        info = yf.Ticker("EURUSD=X").fast_info
        rate = float(getattr(info, "last_price", None) or info.get("lastPrice") or 0)
        if rate > 0:
            return rate
    except Exception:
        pass
    return 1.0


# ── /portfolio handler ────────────────────────────────────────────────────────────

def _handle_portfolio(parts: list[str]) -> None:
    try:
        from portfolio import get_positions, get_liquidity, EUR_TICKERS
    except Exception as e:
        _reply(f"❌ Módulo portfolio não disponível: `{e}`")
        return

    positions = get_positions()
    liquidity = get_liquidity()

    # ── detalhe de um ticker específico ──────────────────────────────────────────
    if len(parts) >= 2:
        symbol = parts[1].upper().strip()
        pos    = positions.get(symbol)
        if not pos:
            _reply(
                f"⚠️ *{symbol}* não está na carteira activa.\n"
                "_Usa `/portfolio` para ver todas as posições._"
            )
            return
        is_eur     = symbol in EUR_TICKERS
        cur        = "€" if is_eur else "$"
        last_price = pos.get("last_price") or pos["avg_price"]
        pnl_u      = (last_price - pos["avg_price"]) * pos["shares"]
        pnl_pct    = (last_price - pos["avg_price"]) / pos["avg_price"] * 100 if pos["avg_price"] else 0
        em         = "🟢" if pnl_u >= 0 else "🔴"
        score_str  = f"*{pos['last_score']}/100*" if pos.get("last_score") is not None else "_N/D_"
        _reply(
            f"📋 *{symbol}* — {pos.get('name', symbol)}\n\n"
            f"  📦 Shares: *{pos['shares']}x*\n"
            f"  💵 Preço médio: *{cur}{pos['avg_price']:.2f}*\n"
            f"  📈 Preço actual: *{cur}{last_price:.2f}*\n"
            f"  {em} P&L não realizado: *{cur}{pnl_u:+.2f}* ({pnl_pct:+.1f}%)\n"
            f"  🏷️ Categoria: _{pos.get('category', 'N/D')}_\n"
            f"  🎯 Score actual: {score_str}\n"
            f"  📅 Entrada: {pos.get('entry_date', 'N/D')}\n"
            f"  🔄 Última actualização: {pos.get('last_update', 'N/D')}"
            + ("\n  ⚠️ _Tese degradada_" if pos.get("degradation_alerted") else "")
        )
        return

    # ── carteira vazia ────────────────────────────────────────────────────────────
    if not positions:
        liq_em = "🟢" if liquidity >= 0 else "🔴"
        _reply(
            f"📭 *Carteira vazia*\n\n"
            f"  💵 Liquidez: {liq_em} *€{liquidity:.2f}*\n\n"
            "_Usa `/buy TICK PREÇO SHARES` para registar a primeira posição._"
        )
        return

    # ── obter FX rate EUR/USD ─────────────────────────────────────────────────────
    eur_usd = _fetch_eur_usd()
    fx_note = f"_{eur_usd:.4f} EUR/USD_" if eur_usd != 1.0 else "_FX indisponível — totais aproximados_"

    # ── lista completa agrupada por categoria ─────────────────────────────────────
    total_cost_usd    = 0.0
    total_current_usd = 0.0
    lines             = [f"*📊 Carteira Activa — {datetime.now().strftime('%d/%m %H:%M')}*", f"  {fx_note}", ""]

    by_cat: dict[str, list] = {}
    for sym, pos in positions.items():
        cat = pos.get("category", "Outras")
        by_cat.setdefault(cat, []).append((sym, pos))

    for cat, items in sorted(by_cat.items()):
        lines.append(f"*{cat}*")
        for sym, pos in sorted(items, key=lambda x: x[0]):
            is_eur     = sym in EUR_TICKERS
            cur        = "€" if is_eur else "$"
            last_price = pos.get("last_price") or pos["avg_price"]
            cost       = pos["avg_price"] * pos["shares"]
            current    = last_price * pos["shares"]
            pnl_u      = current - cost
            pnl_pct    = pnl_u / cost * 100 if cost else 0
            em         = "🟢" if pnl_u >= 0 else "🔴"
            score_str  = f" | 🎯{pos['last_score']}" if pos.get("last_score") is not None else ""
            degrade    = " ⚠️" if pos.get("degradation_alerted") else ""
            lines.append(
                f"  {em} *{sym}*{degrade} — {pos['shares']}x @ {cur}{pos['avg_price']:.2f}"
                f" → {cur}{last_price:.2f} | *{pnl_pct:+.1f}%*{score_str}"
            )
            # converter tudo para USD para o total consolidado
            fx = eur_usd if is_eur else 1.0
            total_cost_usd    += cost * fx
            total_current_usd += current * fx
        lines.append("")

    total_pnl_usd  = total_current_usd - total_cost_usd
    total_pnl_pct  = total_pnl_usd / total_cost_usd * 100 if total_cost_usd else 0
    em_total       = "🟢" if total_pnl_usd >= 0 else "🔴"
    liq_em         = "🟢" if liquidity >= 0 else "🔴"

    # converter totais para EUR
    total_cost_eur    = total_cost_usd / eur_usd if eur_usd else total_cost_usd
    total_current_eur = total_current_usd / eur_usd if eur_usd else total_current_usd
    total_pnl_eur     = total_pnl_usd / eur_usd if eur_usd else total_pnl_usd
    grand_total_eur   = total_current_eur + liquidity

    lines += [
        "─────────────────",
        f"  📥 Investido:  *€{total_cost_eur:,.0f}* (≈${total_cost_usd:,.0f})",
        f"  📦 Actual:     *€{total_current_eur:,.0f}* (≈${total_current_usd:,.0f})",
        f"  {em_total} P&L:       *€{total_pnl_eur:+,.2f}* ({total_pnl_pct:+.1f}%)",
        f"  {liq_em} Liquidez:  *€{liquidity:.2f}*",
        f"  💼 Total:      *€{grand_total_eur:,.0f}*",
        "",
        "_`/portfolio TICK` para detalhe · `/buy` · `/sell`_",
    ]
    _reply("\n".join(lines))


# ── Command router ────────────────────────────────────────────────────────────────

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
            "`/watchlist add TICK`      → Adicionar ticker (ex: IEMA.L, ALV.DE)\n"
            "`/watchlist rm TICK`       → Remover ticker\n"
            "`/watchlist clear`         → Limpar watchlist\n"
            "`/flip`                    → P&L e trades abertos\n"
            "`/flip list`               → Todos os trades\n"
            "`/flip add TICK ENTRY SHR` → Registar entrada\n"
            "`/flip close ID EXIT`      → Fechar trade\n"
            "`/flip del ID`             → Apagar trade\n"
            "`/buy TICK PREÇO SHARES`   → Registar compra\n"
            "`/sell TICK PREÇO [SHR]`   → Registar venda\n"
            "`/liquidez`               → Ver/ajustar saldo\n"
            "`/portfolio`              → Posições activas\n"
            "`/allocate <TICK>`         → Sugestão de alocação read-only (categoria + sizing)\n"
            "`/mldata`                  → Stats da base de dados ML\n"
            "`/mldata update`           → Forçar update de outcomes\n"
            "`/performance [data] [score]` → Simular portfolio: retorno anual + risco\n"
            "`/themes`                  → Ver temas/trends activos (fotónica, GLP-1, IA...)\n"
            "`/add_theme <k> <l> <T>` → Adicionar tema (key label TICK1,TICK2 [conf])\n"
            "`/remove_theme <key>`      → Remover tema\n"
            "`/admin_regen_parquet [--targets-only]` → [ADMIN] Regenerar parquet com PIT fundamentais + alpha90d\n"
            "`/admin_load_models <url>` → [ADMIN] Carregar pickles novos para /data/\n"
            "`/admin_retrain [dry-run]` → [ADMIN] Disparar retrain ad-hoc\n"
            "`/retrigger`               → [ADMIN] Alias rápido de /admin_retrain (full)\\n"
            "`/health`                  → Dashboard observabilidade\n"
            "`/health errors`           → Log de erros críticos\n"
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
        from pathlib import Path
        data_dir = Path("/data") if Path("/data").exists() else Path("/tmp")
        repo_dir = Path(__file__).parent
        bundle_paths = [
            data_dir / "dip_models.pkl",
            repo_dir / "ml_training" / "dip_models.pkl",
            data_dir / "dip_models_v3.pkl",
            repo_dir / "dip_models_v3.pkl",
        ]
        ml_status = (
            "🟢 PKL pronto"
            if any(p.exists() for p in bundle_paths)
            else "🔴 Não treinado"
        )

        try:
            from portfolio import get_liquidity, get_active_symbols
            liq      = get_liquidity()
            n_pos    = len(get_active_symbols())
            port_str = f"\nCarteira: *{n_pos} posições* | Liquidez: *€{liq:.2f}*"
        except Exception:
            port_str = ""

        _reply(
            f"*🤖 DipRadar Status*\n"
            f"Uptime: *{hours}h {mins}m* | Mercado: *{market}*\n"
            f"Watchlist: *{len(wl)} tickers*{flip_str}{db_str}\n"
            f"ML Model: *{ml_status}*"
            + port_str
            + f"\n_⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}_\n"
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
                f"_USD/EUR: {fx:.4f}_\n\n*Total: €{total:,.2f}*\n\n"
                f"  {_e(pnl_d)} Hoje:   €{pnl_d:+,.2f}\n"
                f"  {_e(pnl_w)} Semana: €{pnl_w:+,.2f}\n"
                f"  {_e(pnl_m)} Mês:    €{pnl_m:+,.2f}\n"
                f"  📊 PPR: €{snap.get('ppr_value',0):,.2f}"
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
        if run_scan_fn:
            threading.Thread(target=run_scan_fn, daemon=True).start()

    elif cmd == "/relatorio":
        if not _check_rate(cmd_key): return
        months = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 3
        try:
            from paper_trading import format_unified_report
            _reply(format_unified_report(months_back=months))
        except Exception as e:
            _reply(f"_Erro ao gerar relatório: {e}_")

    elif cmd == "/momentum":
        if not _check_rate(cmd_key): return
        _reply("_🚀 A procurar momentum plays... (pode demorar 2-3 min)_")
        def _run_momentum():
            try:
                from momentum_scanner import scan_momentum_universe, format_momentum_alert
                candidates = scan_momentum_universe(min_score=60.0, max_results=5)
                if not candidates:
                    _reply("Nenhum momentum play encontrado hoje.")
                    return
                _reply(f"🚀 *{len(candidates)} momentum play{'s' if len(candidates)>1 else ''} encontrado{'s' if len(candidates)>1 else ''}:*")
                for c in candidates:
                    _reply(format_momentum_alert(c))
            except Exception as e:
                _reply(f"_Erro no momentum scan: {e}_")
        threading.Thread(target=_run_momentum, daemon=True).start()

    elif cmd == "/analisar":
        if not _check_rate(cmd_key): return
        if len(parts) < 2:
            _reply("⚠️ Usa: `/analisar <TICKER>`\n_Exemplo: `/analisar AAPL`_")
            return
        symbol     = parts[1].upper().strip()
        analyze_fn = _poll_context.get("analyze_ticker") or _cb_analyze_ticker
        if not analyze_fn:
            _reply("_Análise não disponível._")
            return
        _reply(f"_🔍 A analisar *{symbol}*..._")
        threading.Thread(target=lambda: _reply(analyze_fn(symbol)), daemon=True).start()

    elif cmd == "/comparar":
        if not _check_rate(cmd_key): return
        _handle_comparar([p.upper() for p in parts[1:]])

    elif cmd == "/historico":
        if not _check_rate(cmd_key): return
        if len(parts) < 2:
            _reply("⚠️ Usa: `/historico <TICKER>`")
            return
        _handle_historico(parts[1].upper().strip())

    elif cmd == "/backtest":
        if not _check_rate(cmd_key): return
        bt_fn = _poll_context.get("build_backtest_summary") or _cb_backtest_summary
        if bt_fn:
            try: _reply(bt_fn())
            except Exception as e: _reply(f"_Erro no backtest: {e}_")
        else:
            _reply("_Backtest não disponível._")

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
                lines.append(
                    f"  ⛔ *{r['symbol']}* {r['change']:+.1f}% | "
                    f"_{r['reason']}_ | {r.get('time','')}"
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
            _reply(f"_Erro: {e}_")

    elif cmd == "/watchlist":
        if not _check_rate(cmd_key): return
        _handle_watchlist(parts)

    elif cmd == "/flip":
        if not _check_rate(cmd_key): return
        _handle_flip(parts)

    elif cmd == "/buy":
        if not _check_rate(cmd_key): return
        _handle_buy(parts)

    elif cmd == "/sell":
        if not _check_rate(cmd_key): return
        _handle_sell(parts)

    elif cmd == "/liquidez":
        if not _check_rate(cmd_key): return
        _handle_liquidez(parts)

    elif cmd == "/portfolio":
        if not _check_rate(cmd_key): return
        _handle_portfolio(parts)

    elif cmd == "/allocate":
        if not _check_rate(cmd_key): return
        _handle_allocate(parts)

    elif cmd == "/mldata":
        if not _check_rate(cmd_key): return
        force = len(parts) > 1 and parts[1].lower() in ("update", "atualizar", "force")
        _handle_mldata(force_update=force)

    elif cmd == "/admin_backfill_ml":
        _handle_admin_backfill_ml()

    elif cmd == "/admin_train_ml":
        _handle_admin_train_ml()

    elif cmd == "/admin_load_models":
        _handle_admin_load_models(parts)

    elif cmd in ("/admin_test_feed", "/test_feed"):
        _handle_admin_test_feed(parts)

    elif cmd in ("/sync_portfolio", "/sync"):
        _handle_sync_portfolio(parts)

    elif cmd in ("/admin_check_config", "/check_config"):
        _handle_admin_check_config()

    elif cmd in ("/ml_accuracy", "/backtest_ml"):
        _handle_ml_accuracy()

    elif cmd in ("/admin_regen_parquet", "/regen_parquet"):
        _handle_admin_regen_parquet(parts)

    elif cmd == "/admin_retrain":
        _handle_admin_retrain(parts)

    elif cmd == "/admin_set_floor":
        _handle_admin_set_floor(parts)

    elif cmd == "/retrigger":
        _handle_admin_retrain(["admin_retrain"])

    elif cmd in ("/performance", "/returns", "/portfolio_performance"):
        if not _check_rate(cmd_key): return
        _handle_performance(parts)

    elif cmd in ("/paper_portfolio", "/paper", "/paper_performance"):
        if not _check_rate(cmd_key): return
        _handle_paper_performance(parts)

    elif cmd == "/themes":
        if not _check_rate(cmd_key): return
        _handle_themes()

    elif cmd in ("/add_theme", "/addtheme"):
        _handle_add_theme(parts)

    elif cmd in ("/remove_theme", "/removetheme"):
        _handle_remove_theme(parts)

    elif cmd == "/health":
        if not _check_rate(cmd_key): return
        _handle_health(parts)

    else:
        if text.startswith("/"):
            _reply(f"_Comando desconhecido: `{cmd}` — usa /help_")


# ── Poll loop (legacy) ─────────────────────────────────────────────────────────────

def _poll_loop() -> None:
    global _last_update_id
    if not TELEGRAM_TOKEN:
        logging.warning("[bot_commands] TELEGRAM_TOKEN não configurado.")
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
                        _handle_command(text)
        except Exception as e:
            logging.warning(f"[bot_commands] poll error: {e}")
            time.sleep(10)


_listener_thread: threading.Thread | None = None


def start_bot_listener() -> threading.Thread | None:
    """Arranca o listener Telegram em thread daemon (idempotente).

    Se já existe uma thread viva, devolve-a sem criar uma nova
    (evita duplicação de getUpdates calls quando setup() ou
    start_bot_listener() é chamado mais de uma vez).
    """
    global _listener_thread
    if _listener_thread is not None and _listener_thread.is_alive():
        logging.debug("[bot_commands] listener já a correr — skip.")
        return _listener_thread
    _listener_thread = threading.Thread(target=_poll_loop, daemon=True, name="bot-commands")
    _listener_thread.start()
    return _listener_thread
