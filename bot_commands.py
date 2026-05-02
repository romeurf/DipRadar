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
  /watchlist add TICK      → Adicionar ticker à watchlist (suporta IEMA.L, IS3N.AS, ALV.DE, etc.)
  /watchlist rm TICK       → Remover ticker da watchlist
  /watchlist clear         → Limpar toda a watchlist dinâmica
  /flip                    → Ver log e P&L do Flip Fund
  /flip add TICK ENTRY SHARES [NOTA]   → Registar entrada num trade
  /flip close ID EXIT                  → Fechar trade pelo ID com preço de saída
  /flip del ID                         → Apagar trade pelo ID
  /buy <TICK> <PREÇO> <SHARES> [SCORE] → Registar compra na carteira activa
  /sell <TICK> <PREÇO> [SHARES]        → Registar venda (parcial ou total)
  /liquidez [+|-]<VALOR>               → Ver / ajustar saldo disponível
  /portfolio                           → Resumo das posições activas
  /mldata                  → Estatísticas da base de dados ML + forçar update de outcomes
  /admin_backfill_ml       → [ADMIN] Semear hist_backtest.csv com 5 anos de dips históricos
  /admin_train_ml          → [ADMIN] Treinar modelo ML e gerar dip_model.pkl
  /admin_load_models <url> → [ADMIN] Descarregar pickles + ml_report do URL para /data/ (atomic)
  /health                  → Dashboard de observabilidade (RAM, CPU, latências, last scan)
  /health errors           → Log dos últimos erros críticos
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
            em_map = {"WIN_40": "🟢", "WIN_20": "✅", "NEUTRAL": "🟡", "LOSS_15": "🔴"}
            for lbl, cnt in sorted(outcomes.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {em_map.get(lbl,'📊')} {lbl}: *{cnt}* ({cnt/labeled*100:.0f}%)" if labeled else f"  {lbl}: {cnt}")
            wins = outcomes.get("WIN_40", 0) + outcomes.get("WIN_20", 0)
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

                em_map   = {"WIN_40": "🟢", "WIN_20": "✅", "NEUTRAL": "🟡", "LOSS_15": "🔴", "pending": "⏳"}
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

        em_map = {"WIN_40": "🟢", "WIN_20": "✅", "NEUTRAL": "🟡", "LOSS_15": "🔴", "pending": "⏳"}
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
    /admin_load_models <url>                     → tarball (.tar.gz/.tgz/.zip) com 3 ficheiros
    /admin_load_models <s1_url> <s2_url> <r_url> → 3 URLs individuais

    Faz download dos pickles + ml_report.json para `/data/`, valida, e faz swap
    atomic. Versão antiga vai para `/data/archive/dip_model_stageN_<ts>.pkl`. O
    ml_predictor recarrega automaticamente quando o mtime muda — não precisa
    reiniciar o bot.

    URLs aceites: HTTPS direct download. Para Google Drive partilha pública
    usa o formato `https://drive.google.com/uc?export=download&id=<FILE_ID>`.
    """
    global _load_models_running

    if _load_models_running:
        _reply("⚠️ *Load já está a correr.* _Aguarda a conclusão._")
        return

    args = [a for a in parts[1:] if a.strip()]
    if len(args) not in (1, 3):
        _reply(
            "❌ *Uso incorrecto.*\n\n"
            "*Modo tarball* (1 URL):\n"
            "`/admin_load_models <url_tar_gz>`\n"
            "_Ficheiro deve conter dip_model_stage1.pkl, dip_model_stage2.pkl, ml_report.json._\n\n"
            "*Modo individual* (3 URLs):\n"
            "`/admin_load_models <s1_url> <s2_url> <report_url>`\n\n"
            "_Sugestão: para Google Drive usa partilha pública e o link directo `https://drive.google.com/uc?export=download&id=<ID>`._"
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
        import pickle
        import tarfile
        import zipfile
        import urllib.request
        import urllib.parse

        data_dir   = Path("/data") if Path("/data").exists() else Path("/tmp")
        archive_dir = data_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp  = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

        # Limites de segurança
        MAX_BYTES_PER_FILE = 80 * 1024 * 1024  # 80 MB por ficheiro
        REQUIRED_PKL_KEYS  = {"model", "feature_columns", "threshold"}
        EXPECTED_FILES     = ["dip_model_stage1.pkl", "dip_model_stage2.pkl", "ml_report.json"]

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

            # ── Modo 1: tarball (1 URL)
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

                # Procura os 3 ficheiros (recursivo, primeira ocorrência)
                local: dict[str, Path] = {}
                for fname in EXPECTED_FILES:
                    matches = list(extract_dir.rglob(fname))
                    if not matches:
                        raise ValueError(f"ficheiro {fname} não encontrado no archive")
                    local[fname] = matches[0]
            else:
                # ── Modo 2: 3 URLs individuais (ordem: stage1, stage2, report)
                local = {}
                for fname, url in zip(EXPECTED_FILES, args):
                    dest = tmp_dir / fname
                    _download(url, dest)
                    local[fname] = dest

            # ── Validação dos pickles
            stage1_meta: dict = {}
            for stage in (1, 2):
                fname  = f"dip_model_stage{stage}.pkl"
                with open(local[fname], "rb") as f:
                    bundle = pickle.load(f)
                if not isinstance(bundle, dict):
                    raise ValueError(f"{fname}: pickle não é um dict")
                missing = REQUIRED_PKL_KEYS - set(bundle.keys())
                if missing:
                    raise ValueError(f"{fname}: faltam keys {missing} no bundle")
                # Sanity check: feature_columns deve existir e ser não-vazia
                fc = bundle.get("feature_columns") or []
                if not fc:
                    raise ValueError(f"{fname}: feature_columns vazia")
                if stage == 1:
                    stage1_meta = {
                        "n_features": len(fc),
                        "threshold":  bundle.get("threshold"),
                        "algorithm":  bundle.get("algorithm")
                            or bundle.get("algo")
                            or "unknown",
                    }

            # Validação do report
            with open(local["ml_report.json"]) as f:
                report = json.load(f)
            if not isinstance(report, dict):
                raise ValueError("ml_report.json: top-level não é um dict")
            s1 = report.get("stage1") or {}
            auc_pr = s1.get("auc_pr_test") or s1.get("auc_pr")

            # ── Atomic swap: archive existing, then replace
            promoted = []
            for fname in EXPECTED_FILES:
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

            # ── Reply com métricas
            thr   = stage1_meta.get("threshold")
            algo  = stage1_meta.get("algorithm")
            n_feat = stage1_meta.get("n_features")
            lines = [
                "✅ *Modelos carregados com sucesso!*",
                f"_{datetime.now().strftime('%d/%m/%Y %H:%M')} — archive ts {timestamp}_",
                "",
                "*📊 Stage 1 (gating):*",
                f"  Algoritmo:  *{algo}*",
                f"  AUC-PR:     *{auc_pr:.4f}*" if isinstance(auc_pr, (int, float)) else "  AUC-PR:     _N/D_",
                f"  Threshold:  *{thr:.4f}*"    if isinstance(thr, (int, float))    else "  Threshold:  _N/D_",
                f"  Features:   *{n_feat}*",
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
            "`/mldata`                  → Stats da base de dados ML\n"
            "`/mldata update`           → Forçar update de outcomes\n"
            "`/admin_load_models <url>` → [ADMIN] Carregar pickles novos para /data/\n"
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
        data_dir  = Path("/data") if Path("/data").exists() else Path("/tmp")
        ml_status = "🟢 PKL pronto" if (data_dir / "dip_model_stage1.pkl").exists() else "🔴 Não treinado"

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
        if run_scan_fn:
            threading.Thread(target=run_scan_fn, daemon=True).start()

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


def start_bot_listener() -> threading.Thread:
    t = threading.Thread(target=_poll_loop, daemon=True, name="bot-commands")
    t.start()
    return t
