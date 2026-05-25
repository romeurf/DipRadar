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

━━━ CORE (compounders de qualidade) ━━━
  /core                    → Posições CORE abertas (hold forever + P&L + yield)
  /core_add T P EUR        → Registar compra CORE (ticker, preço, montante EUR)
  /core_close T P          → Fechar posição CORE manualmente
  /core_universe           → Ver universo CoreRadar (tickers monitorizados)
  /core_universe_add T     → Adicionar ticker ao universo CORE
  /core_universe_remove T  → Remover ticker do universo CORE

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
  /admin_run_pipeline      → Pipeline unificado (dataset + retrain, todos os modelos)
  /admin_run_pipeline --dip --retrain       → retrain DIP apenas
  /admin_run_pipeline --momentum --dataset  → só dataset momentum
  /admin_run_pipeline --core                → só dataset CORE
  /admin_run_pipeline --test AAPL,MSFT      → smoke test rápido (~2 min)
  /admin_retrain [dry-run] → Retreino DIP ad-hoc (usa parquet existente)
  /admin_regen_parquet [--targets-only] → Regenerar parquet (EDGAR PIT + alpha_90d)
  /admin_set_floor <valor> → Ajustar floor de IC mínimo para promoção (ex: 0.08)
  /admin_load_models <url> → Carregar bundle ML de URL externo

━━━ PAPER TRADING ━━━
  /paper_portfolio         → Last 3 months performance vs SPY
  /paper_portfolio 6       → Last 6 months
  /paper_portfolio open    → Open positions with live weights
  /paper_portfolio drip    → CORE DRIP audit trail (every dividend event, tax disclaimer)
  /paper_portfolio missed  → MISSED_CAPITAL signals (lost alpha due to full pool)
  /admin_publish_charts    → Publish charts + DRIP ledger to GitHub (public URL)

━━━ SISTEMA E DIAGNÓSTICO ━━━
  /status                  → Estado do bot + mercado + modelos
  /admin_check_config      → Verificar config + estado dos 3 modelos
  /ps                      → Pipeline status: fase actual, log, tempo decorrido
  /admin_kill_pipeline     → Terminar processo de pipeline em curso
  /admin_clear_price_cache → Apagar cache de preços (forçar re-download completo)
  /health                  → Dashboard: RAM, CPU, APIs, drift
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
from i18n import t, set_language, get_language
from config import SUPPORTED_LANGUAGES
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
_retrain_running:  bool = False

# Pipeline live state — updated during runs, readable via /admin_pipeline_status
_pipeline_log:   list[str] = []          # rolling log of key progress lines (last 100)
_pipeline_phase: str       = ""          # current phase label e.g. "Phase 1 — DIP dataset"
_pipeline_start: float     = 0.0         # wall-clock start time (time.time())
_pipeline_proc:  object    = None        # subprocess.Popen object — killable via /admin_kill_pipeline
_PIPELINE_LOG_MAX = 100


def _plog(msg: str) -> None:
    """Record a timestamped progress line and log it."""
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    _pipeline_log.append(entry)
    if len(_pipeline_log) > _PIPELINE_LOG_MAX:
        _pipeline_log.pop(0)
    logging.info(f"[pipeline] {msg}")


def _pipeline_elapsed() -> str:
    """Human-readable elapsed time since pipeline started."""
    import time as _t
    if not _pipeline_start:
        return "0s"
    s = int(_t.time() - _pipeline_start)
    if s < 60:   return f"{s}s"
    if s < 3600: return f"{s // 60}m {s % 60}s"
    return f"{s // 3600}h {(s % 3600) // 60}m"


def _run_script_streaming(
    cmd: list,
    phase_label: str,
    timeout_s: int = 8 * 3600,
    telegram_interval_s: int = 300,
) -> tuple[int, list[str]]:
    """Run a subprocess and stream its output in real time.

    - Updates _pipeline_log line by line (filters noise, keeps INFO/WARNING/ERROR and progress lines)
    - Automatically sends a Telegram status update every telegram_interval_s seconds
    - Stores the Popen object in _pipeline_proc so it can be killed via /admin_kill_pipeline
    - Returns (returncode, list_of_captured_lines)
    """
    import subprocess as _sp, time as _t, re as _re
    global _pipeline_proc, _pipeline_phase

    _pipeline_phase = phase_label
    _plog(f"{phase_label}: starting")

    # Patterns that are useful to surface; everything else is debug noise
    _PROGRESS_RE  = _re.compile(r"\[(\d+)/(\d+)\]")          # [42/780] AAPL
    _LEVEL_RE     = _re.compile(r"\b(INFO|WARNING|ERROR|CRITICAL)\b")
    _SUMMARY_KEYS = ("dataset", "rows", "tickers", "samples", "promoted", "failed",
                     "complete", "done", "skip", "merge", "alpha", "episodes",
                     "IC=", "rho", "elapsed", "error", "exception")

    def _is_interesting(line: str) -> bool:
        ll = line.lower()
        if _PROGRESS_RE.search(line):    return True
        if _LEVEL_RE.search(line):       return True
        if any(k in ll for k in _SUMMARY_KEYS): return True
        return False

    captured: list[str] = []
    last_tg_push = _t.time()
    last_progress = ""   # last "[X/Y]" line — shown in periodic updates

    try:
        proc = _sp.Popen(
            cmd,
            stdout=_sp.PIPE, stderr=_sp.STDOUT,
            text=True, bufsize=1,
        )
        _pipeline_proc = proc

        deadline = _t.time() + timeout_s
        for raw in proc.stdout:
            if _t.time() > deadline:
                proc.terminate()
                _plog(f"{phase_label}: TIMEOUT after {timeout_s}s — process terminated")
                break

            line = raw.rstrip()
            captured.append(line)

            if _is_interesting(line):
                # Strip leading timestamp if the script already adds one
                clean = _re.sub(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,.]?\d*\s*\|\s*", "", line).strip()
                clean = _re.sub(r"^\d{2}:\d{2}:\d{2}\s+", "", clean).strip()
                if clean:
                    _plog(clean)
                    m = _PROGRESS_RE.search(line)
                    if m:
                        last_progress = clean

            # Periodic Telegram push so the user isn't blind
            now = _t.time()
            if now - last_tg_push >= telegram_interval_s:
                tail = _pipeline_log[-8:]
                elapsed = _pipeline_elapsed()
                status_lines = [
                    f"Pipeline still running — {elapsed} elapsed",
                    f"Phase: {phase_label}",
                    *(f"  {l}" for l in tail),
                    "",
                    "Use /admin_pipeline_status for full log or /admin_kill_pipeline to stop.",
                ]
                _reply("\n".join(status_lines))
                last_tg_push = now

        proc.wait()
        returncode = proc.returncode
    except Exception as exc:
        _plog(f"{phase_label}: exception — {exc}")
        logging.error(f"[pipeline] {phase_label}: {exc}", exc_info=True)
        returncode = -1
    finally:
        _pipeline_proc = None
        _pipeline_phase = ""

    status = "OK" if returncode == 0 else f"FAILED (exit {returncode})"
    _plog(f"{phase_label}: {status}")
    return returncode, captured

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
                _poll_context["_current_chat_id"] = int(cid) if cid.lstrip("-").isdigit() else 0
                _handle_command(text)
    except Exception as e:
        logging.debug(f"[bot_commands] poll error: {e}")


_poll_context: dict = {}


def _reply(text: str) -> None:
    fn = _poll_context.get("send_fn") or _cb_send_telegram
    if fn:
        fn(text)


def _send_document(
    file_bytes: bytes,
    filename: str,
    caption: str = "",
) -> bool:
    """Send a file to the Telegram chat via sendDocument.

    Falls back silently if TELEGRAM_TOKEN / TELEGRAM_CHAT_ID are not set.
    Returns True on success.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("[send_document] No TELEGRAM_TOKEN/CHAT_ID — cannot send file.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    try:
        resp = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1024]},
            files={"document": (filename, file_bytes, "text/plain")},
            timeout=30,
        )
        if not resp.ok:
            logging.warning("[send_document] Telegram error %s: %s", resp.status_code, resp.text[:200])
            return False
        return True
    except Exception as exc:
        logging.warning("[send_document] %s", exc)
        return False


def _send_photo(file_bytes: bytes, caption: str = "") -> bool:
    """Send a PNG image to the Telegram chat via sendPhoto."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
        resp = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1024]},
            files={"photo": ("chart.png", file_bytes, "image/png")},
            timeout=30,
        )
        return resp.ok
    except Exception as exc:
        logging.warning("[send_photo] %s", exc)
        return False


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
            lines.append("🟢 *Pronto para treinar!* Usa `/admin_full_pipeline --dip-only`")
        elif labeled >= 50:
            lines.append(f"🟡 *{labeled} amostras* — quase prontos (mín. recomendado: 100)")
            lines.append("_Já podes tentar `/admin_retrain` com resultados parciais._")
        else:
            lines.append(f"🕐 *{labeled} amostras* — precisa de mais dados (mín: 30 para treinar).")

        if not force_update and fill_db_outcomes:
            lines.append("\n_Força update agora com `/mldata update`_")
        _reply("\n".join(lines))

    threading.Thread(target=_run, daemon=True).start()

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

    # ── Model status — all 3 models ──────────────────────────────────────────
    lines.append("")
    lines.append("Models:")

    # DIP model (dip_models.pkl)
    try:
        from ml_predictor import is_model_ready, get_model_info
        if is_model_ready():
            info = get_model_info()
            rho  = info.get("rho_mean")
            rho_str = f"{rho:.5f}" if isinstance(rho, float) else str(rho)
            lines.append(
                f"  DIP (dip_us): ready | rho_alpha={rho_str} | "
                f"champion={info.get('champion','?')} | "
                f"features={info.get('n_features','?')}"
            )
        else:
            lines.append("  DIP (dip_us): NOT trained — run /admin_run_pipeline --dip")
    except Exception as e:
        lines.append(f"  DIP (dip_us): error checking — {e}")

    # Momentum model (momentum_model.pkl)
    try:
        from pathlib import Path as _P
        _data = _P("/data") if _P("/data").exists() else _P("/tmp")
        _mom_pkl = _data / "momentum_model.pkl"
        _mom_repo = _P(__file__).parent / "momentum_radar" / "momentum_model.pkl"
        _mom_report = _data / "momentum_report.json"
        if not _mom_report.exists():
            _mom_report = _P(__file__).parent / "momentum_radar" / "momentum_report.json"
        if (_mom_pkl.exists() or _mom_repo.exists()) and _mom_report.exists():
            import json as _json
            rep = _json.loads(_mom_report.read_text())
            ic  = rep.get("ic_ensemble") or rep.get("ic")
            ic_str = f"{ic:.5f}" if isinstance(ic, float) else str(ic)
            lines.append(
                f"  Momentum (momentum_us): ready | IC={ic_str} | "
                f"n_train={rep.get('n_train','?')} | sectors={rep.get('n_sectors','?')}"
            )
        else:
            lines.append("  Momentum (momentum_us): NOT trained — run /admin_run_pipeline --momentum")
    except Exception as e:
        lines.append(f"  Momentum (momentum_us): error checking — {e}")

    # CORE model — rule-based scoring, no ML pickle; check dataset exists
    try:
        from pathlib import Path as _Pc
        _data_c = _Pc("/data") if _Pc("/data").exists() else _Pc("/tmp")
        _core_ds = _data_c / "core_training_base.parquet"
        _core_repo = _Pc(__file__).parent / "core_training_base.parquet"
        if _core_ds.exists() or _core_repo.exists():
            _ds_path = _core_ds if _core_ds.exists() else _core_repo
            import pandas as _pd
            _nrows = len(_pd.read_parquet(_ds_path, columns=["ticker"]))
            lines.append(
                f"  CORE (core_us): rule-based scoring (no ML pkl) | "
                f"training dataset: {_nrows:,} rows"
            )
        else:
            lines.append("  CORE (core_us): rule-based scoring | dataset missing — run /admin_run_pipeline --core")
    except Exception as e:
        lines.append(f"  CORE (core_us): error checking — {e}")

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

            # Com cache EDGAR em memória (corrigido): ~45-90 min para regen completo.
            _TIMEOUT = 8 * 3600
            log_file = data_dir / "regen_progress.log"

            # Redirigir stdout+stderr para ficheiro — permite /admin_regen_status
            with open(log_file, "w", encoding="utf-8") as lf:
                result = subprocess.run(
                    args, stdout=lf, stderr=lf, timeout=_TIMEOUT
                )

            if result.returncode == 0:
                try:
                    import pandas as pd
                    df_check = pd.read_parquet(out_pq)
                    n_90d  = int(df_check["alpha_90d"].notna().sum()) if "alpha_90d" in df_check.columns else 0
                    total  = len(df_check)
                    pct = f"{n_90d/total:.0%}" if total else "0%"

                    # Copiar para /app/ como backup imediato (mesmo container).
                    # O /data/ volume é persistente — é a fonte de verdade.
                    # /app/ serve de fallback se /data/ não estiver montado ainda.
                    repo_pq = Path(__file__).parent / "ml_training_base.parquet"
                    try:
                        import shutil
                        shutil.copy2(str(out_pq), str(repo_pq))
                        logging.info(f"[regen] Backup copiado para /app/: {repo_pq} ({repo_pq.stat().st_size/1e6:.1f}MB)")
                    except Exception as cp_e:
                        logging.warning(f"[regen] Backup /app/ falhou: {cp_e}")

                    _reply(
                        f"*Parquet regenerado e guardado!*\n"
                        f"  Shape: {df_check.shape[0]:,} x {df_check.shape[1]} colunas\n"
                        f"  alpha-90d resolvido: {n_90d:,}/{total:,} ({pct})\n\n"
                        f"A iniciar retrain automaticamente..."
                    )
                    # Pipeline automático: regen → retrain sem intervenção manual
                    import threading as _t
                    _t.Thread(
                        target=lambda: _handle_admin_retrain(["admin_retrain"]),
                        daemon=True,
                        name="auto-retrain-after-regen"
                    ).start()
                except Exception as e:
                    _reply(f"Parquet regenerado. (verificacao: {e})\nFaz /admin\\_retrain")
            else:
                # Mostrar últimas linhas do log para diagnóstico
                try:
                    last_lines = log_file.read_text(encoding="utf-8", errors="replace")[-800:]
                except Exception:
                    last_lines = "(sem log)"
                _reply(f"Regeneracao falhou (exit {result.returncode})\n`{last_lines}`")
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


# ── /admin_disk ─────────────────────────────────────────────────────────────────

def _handle_admin_disk(parts: list[str]) -> None:
    """/admin_disk          → ver utilização do volume /data/
    /admin_disk clean     → apagar EDGAR cache antigo (>30d) e libertar espaço
    """
    import os, time
    from pathlib import Path

    data_dir = Path("/data") if Path("/data").exists() else Path("/tmp")
    clean = len(parts) > 1 and parts[1].lower() == "clean"

    def _human(b: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if b < 1024:
                return f"{b:.1f} {unit}"
            b /= 1024
        return f"{b:.1f} TB"

    def _dir_size(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0

    if clean:
        # Apagar EDGAR facts cache com mais de 30 dias (edgar_*.json em price_cache)
        cache_dir = data_dir / "price_cache"
        cutoff    = time.time() - 30 * 86400
        removed   = 0
        freed     = 0
        if cache_dir.exists():
            for f in cache_dir.glob("edgar_*.json"):
                if f.stat().st_mtime < cutoff:
                    freed += f.stat().st_size
                    f.unlink()
                    removed += 1
        # Apagar universe_snapshot com mais de 180 dias (linhas antigas)
        snap_path = data_dir / "universe_snapshot.parquet"
        snap_pruned = 0
        if snap_path.exists():
            try:
                import pandas as pd
                snap = pd.read_parquet(snap_path)
                if "snapshot_date" in snap.columns:
                    cutoff_date = (pd.Timestamp.now() - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
                    before = len(snap)
                    snap = snap[snap["snapshot_date"] >= cutoff_date]
                    snap_pruned = before - len(snap)
                    if snap_pruned > 0:
                        snap.to_parquet(snap_path, index=False)
            except Exception as e:
                logging.debug(f"[disk] snapshot prune: {e}")

        # Apagar parquet incompleto do /data/ — o regen usa a versão limpa do repo
        pq_removed = ""
        pq_path = data_dir / "ml_training_base.parquet"
        repo_pq = Path(__file__).parent / "ml_training_base.parquet"
        if pq_path.exists() and repo_pq.exists():
            # Só apaga se a versão do repo existe como fallback
            freed += pq_path.stat().st_size
            pq_path.unlink()
            pq_removed = "\n  Parquet incompleto (/data/): apagado — regen usa versão limpa do repo"

        # Apagar bundle ML corrompido (< 100KB = não é um bundle real)
        bundle_removed = ""
        bundle_path = data_dir / "dip_models.pkl"
        if bundle_path.exists() and bundle_path.stat().st_size < 100 * 1024:
            freed += bundle_path.stat().st_size
            bundle_path.unlink()
            bundle_removed = "\n  Bundle ML corrompido (< 100KB): apagado"

        _reply(
            f"*Limpeza concluída:*\n"
            f"  EDGAR cache apagado: {removed} ficheiros ({_human(freed)})\n"
            f"  Universe snapshot: {snap_pruned} linhas antigas removidas"
            f"{pq_removed}"
            f"{bundle_removed}\n\n"
            f"Corre `/admin\\_regen\\_parquet` para reconstruir tudo limpo."
        )
        return

    # Listagem de tamanhos
    items = [
        ("price_cache/edgar_*.json", sum(f.stat().st_size for f in (data_dir / "price_cache").glob("edgar_*.json")) if (data_dir / "price_cache").exists() else 0),
        ("price_cache/ (total)",     _dir_size(data_dir / "price_cache")),
        ("universe_snapshot.parquet",(data_dir / "universe_snapshot.parquet").stat().st_size if (data_dir / "universe_snapshot.parquet").exists() else 0),
        ("ml_training_base.parquet", (data_dir / "ml_training_base.parquet").stat().st_size if (data_dir / "ml_training_base.parquet").exists() else 0),
        ("momentum_training.parquet",(data_dir / "momentum_training.parquet").stat().st_size if (data_dir / "momentum_training.parquet").exists() else 0),
        ("dip_models.pkl",           (data_dir / "dip_models.pkl").stat().st_size if (data_dir / "dip_models.pkl").exists() else 0),
        ("alert_db.csv",             (data_dir / "alert_db.csv").stat().st_size if (data_dir / "alert_db.csv").exists() else 0),
    ]
    total = _dir_size(data_dir)

    lines = ["*Utilização do volume /data/:*", ""]
    for name, size in sorted(items, key=lambda x: x[1], reverse=True):
        if size > 0:
            lines.append(f"  `{name}`: {_human(size)}")
    lines += ["", f"*Total: {_human(total)}*", "", "_/admin\\_disk clean → apagar EDGAR cache antigo e snapshot >180d_"]
    _reply("\n".join(lines))


# ── /admin_momentum_dataset ─────────────────────────────────────────────────────

def _handle_admin_momentum_dataset(parts: list[str]) -> None:
    """/admin_momentum_dataset [--start YYYY-MM-DD] [--tickers TICK1,TICK2] [--force-rebuild]

    Constrói o dataset histórico para treinar o Breakout ML (MomentumRadar).

    Varre os ~780 tickers desde 1993, encontra dias com momentum forte
    (return_20d ≥ 10%, volume 1.2×, RSI 40-85) e calcula o retorno futuro.

    Output: /data/momentum_training.parquet

    Flags:
      --start 2021-01-01  → data de início (default: 1993-01-01)
      --tickers AAPL,MSFT → testar com subset (rápido, ~2 min)
      --force-rebuild     → ignorar parquet existente, reconstruir desde --start

    Duração: ~3-5h (780 tickers × histórico 30 anos) | ~2 min (teste com 5 tickers)
    """
    start_val     = "1993-01-01"
    ticker_val    = None
    force_rebuild = "--force-rebuild" in parts

    for i, p in enumerate(parts[1:]):
        if p == "--start" and i + 2 < len(parts):
            start_val = parts[i + 2]
        if p == "--tickers" and i + 2 < len(parts):
            ticker_val = parts[i + 2]

    is_test = ticker_val is not None
    dur_str = "~2 min (modo teste)" if is_test else "~3-5h (universo completo)"

    _reply(
        f"*MomentumRadar — Dataset Builder*\n"
        f"Desde: {start_val}"
        + (" *(force rebuild)*" if force_rebuild else "") +
        f"\n"
        + (f"Tickers: {ticker_val}\n" if is_test else "") +
        f"Output: /data/momentum\\_training.parquet\n"
        f"Duração: {dur_str}\n"
        f"Podes continuar a usar o bot."
    )

    def _run():
        import subprocess, sys, shutil
        from pathlib import Path
        try:
            script   = Path(__file__).parent / "momentum_radar" / "scripts" / "build_dataset.py"
            data_dir = Path("/data") if Path("/data").exists() else Path("/tmp")
            out_path = data_dir / "momentum_training.parquet"
            # Se /data/ não tem o parquet (deploy novo), semear a partir do repo
            if not out_path.exists() and not force_rebuild:
                repo_seed = Path(__file__).parent / "momentum_radar" / "momentum_training.parquet"
                if repo_seed.exists():
                    shutil.copy2(str(repo_seed), str(out_path))
                    logging.info(f"[momentum_dataset] Seed de {repo_seed} → {out_path}")
            cmd      = [sys.executable, str(script), "--start", start_val, "--out", str(out_path)]
            if ticker_val:
                cmd += ["--tickers", ticker_val]
            if force_rebuild:
                cmd += ["--force-rebuild"]

            _TIMEOUT = 8 * 3600  # 8h
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT)

            if result.returncode == 0:
                try:
                    import pandas as pd
                    df = pd.read_parquet(out_path)
                    # Copiar para repo para rastreio em git
                    repo_pq = Path(__file__).parent / "momentum_radar" / "momentum_training.parquet"
                    try:
                        shutil.copy2(str(out_path), str(repo_pq))
                    except Exception as _cp:
                        logging.warning(f"[momentum_dataset] cópia repo: {_cp}")
                    n_rows    = len(df)
                    n_tickers = df["ticker"].nunique() if "ticker" in df.columns else "?"
                    n_sectors = df["sector"].nunique() if "sector" in df.columns else "?"
                    avg_fwd   = df["forward_return_30d"].mean() * 100 if "forward_return_30d" in df.columns else 0
                    pct_pos   = (df["forward_return_30d"] > 0).mean() * 100 if "forward_return_30d" in df.columns else 0
                    _reply(
                        f"*MomentumRadar Dataset — Concluído!*\n"
                        f"  Amostras: {n_rows:,}\n"
                        f"  Tickers: {n_tickers} | Sectores: {n_sectors}\n"
                        f"  Retorno médio 30d: {avg_fwd:+.1f}%\n"
                        f"  % positivos: {pct_pos:.0f}%\n\n"
                        f"Próximo passo: treinar o Breakout ML quando tiveres tempo."
                    )
                except Exception as e:
                    _reply(f"Dataset gerado. (verificação: {e})")
            else:
                stderr = (result.stderr or "")[-500:]
                _reply(f"❌ Dataset falhou (exit {result.returncode})\n`{stderr}`")

        except subprocess.TimeoutExpired:
            _reply("⏰ Dataset excedeu 4h. O processo foi terminado.")
        except Exception as e:
            logging.error(f"[momentum_dataset] {e}", exc_info=True)
            _reply(f"❌ Erro: `{e}`")

    threading.Thread(target=_run, daemon=True, name="momentum-dataset").start()


# ── /admin_momentum_retrain ─────────────────────────────────────────────────────

def _handle_admin_momentum_retrain(parts: list[str]) -> None:
    """/admin_momentum_retrain

    Treina o Breakout ML com todos os dados históricos acumulados.
    Corre automaticamente no dia 1 de cada mês (após o DipRadar retrain).

    Pipeline:
      1. Actualiza o dataset com os últimos 30 dias (incremental, nunca apaga dados)
      2. Treina ScaledRidge + sector models
      3. Gating: promove só se IC ≥ IC actual × 0.90
    """
    _reply(
        "*MomentumRadar — Retrain*\n"
        "1. Actualizar dataset (incremental)\n"
        "2. Treinar Breakout ML + sector models\n"
        "3. Gating vs modelo actual\n"
        "Duração: ~20-40 min\n"
        "Podes continuar a usar o bot."
    )

    def _run():
        import subprocess, sys, shutil
        from pathlib import Path
        try:
            data_dir    = Path("/data") if Path("/data").exists() else Path("/tmp")
            ds_path     = data_dir / "momentum_training.parquet"
            bundle_path = data_dir / "momentum_model.pkl"
            report_path = data_dir / "momentum_report.json"

            # Passo 1: actualizar dataset (só últimos 35 dias — incremental)
            script = Path(__file__).parent / "momentum_radar" / "scripts" / "build_dataset.py"
            result = subprocess.run(
                [sys.executable, str(script), "--out", str(ds_path)],
                capture_output=True, text=True, timeout=7200
            )
            if result.returncode != 0:
                _reply(f"⚠️ Dataset update falhou:\n`{(result.stderr or '')[-300:]}`\n_A tentar treinar com dados existentes..._")

            # Passo 2 + 3: treinar e fazer gating
            from momentum_radar.trainer import run_momentum_training
            result_train = run_momentum_training(
                dataset_path=ds_path,
                bundle_path=bundle_path,
                report_path=report_path,
            )
            decision    = result_train.get("decision", "?")
            ic_ensemble = result_train.get("ic_ensemble")
            ic_prod     = result_train.get("ic_prod")
            n_train     = result_train.get("n_train", 0)
            n_sec       = result_train.get("n_sectors", 0)
            elapsed     = result_train.get("elapsed_s", 0)
            ic_str      = f"{ic_ensemble:.4f}" if isinstance(ic_ensemble, float) else "N/A"
            ic_prod_str = f"{ic_prod:.4f}" if isinstance(ic_prod, float) else "N/A"

            if decision == "PROMOTED":
                # Copiar artefactos para repo (fallback local + github_sync via API)
                repo_dir = Path(__file__).parent / "momentum_radar"
                for src, name in [(ds_path, "momentum_training.parquet"),
                                  (bundle_path, "momentum_model.pkl"),
                                  (report_path, "momentum_report.json")]:
                    try:
                        if src.exists():
                            shutil.copy2(str(src), str(repo_dir / name))
                    except Exception as _cp:
                        logging.warning(f"[momentum_retrain] cópia repo {name}: {_cp}")
                try:
                    from github_sync import sync_momentum_artifacts
                    sync_momentum_artifacts(data_dir)
                except Exception as _gs:
                    logging.warning(f"[momentum_retrain] github_sync: {_gs}")
                _reply(
                    f"🚀 *MomentumRadar Retrain — PROMOVIDO*\n"
                    f"IC novo: *{ic_str}*"
                    + (f" (anterior: {ic_prod_str})" if ic_prod else "") +
                    f"\n{n_train:,} amostras | {n_sec} sector models | {elapsed:.0f}s"
                )
            elif decision == "KEPT":
                _reply(
                    f"⚠️ *MomentumRadar Retrain — MANTIDO*\n"
                    f"_{result_train.get('reason', '')}_\n"
                    f"IC novo: {ic_str} | Produção: {ic_prod_str}"
                )
            else:
                _reply(f"❌ *MomentumRadar Retrain — FALHOU*\n_{result_train.get('reason', '')}_")

        except Exception as e:
            logging.error(f"[momentum_retrain] {e}", exc_info=True)
            _reply(f"❌ Erro: `{e}`")

    threading.Thread(target=_run, daemon=True, name="momentum-retrain").start()


# ── /admin_dip_dataset ──────────────────────────────────────────────────────────

def _handle_admin_dip_dataset(parts: list[str]) -> None:
    """/admin_dip_dataset [--start YYYY-MM-DD] [--tickers TICK1,TICK2] [--no-fundamentals] [--force-rebuild]

    Constrói o dataset histórico para treinar o DipRadar com episódios simulados
    desde 1993 (equivalente a /admin_momentum_dataset mas para dips).

    Varre os ~780 tickers desde 1993, detecta episódios de queda
    (return_20d < -5%, exit quando > -2%) e calcula alpha_90d vs SPY.
    Reutiliza exactamente as mesmas funções de feature do pipeline de produção.

    Output: /data/ml_training_base.parquet (merge com alertas reais existentes)

    Flags:
      --start 2010-01-01    → data de início (default: 1993-01-01)
      --tickers AAPL,MSFT   → testar com subset (rápido, ~2 min)
      --no-fundamentals     → saltar EDGAR/Tiingo (mais rápido)
      --force-rebuild       → ignorar parquet existente, reconstruir desde --start

    Duração: ~3-5h (780 tickers × histórico 30 anos) | ~2 min (teste com 5 tickers)
    """
    start_val   = "1993-01-01"
    ticker_val  = None
    no_fund     = "--no-fundamentals" in parts
    force_rebuild = "--force-rebuild" in parts

    for i, p in enumerate(parts[1:]):
        if p == "--start" and i + 2 < len(parts):
            start_val = parts[i + 2]
        if p == "--tickers" and i + 2 < len(parts):
            ticker_val = parts[i + 2]

    is_test = ticker_val is not None
    dur_str = "~2 min (modo teste)" if is_test else "~3-5h (universo completo)"

    _reply(
        f"*DipRadar — Dataset Builder (simulação histórica)*\n"
        f"Desde: {start_val}"
        + (" *(force rebuild)*" if force_rebuild else "") +
        f"\n"
        + (f"Tickers: {ticker_val}\n" if is_test else "") +
        f"Output: /data/ml\\_training\\_base.parquet\n"
        + ("Fundamentais: não\n" if no_fund else "") +
        f"Duração: {dur_str}\n"
        f"Podes continuar a usar o bot."
    )

    def _run():
        import subprocess, sys, shutil
        from pathlib import Path
        try:
            script   = Path(__file__).parent / "scripts" / "build_dip_dataset.py"
            data_dir = Path("/data") if Path("/data").exists() else Path("/tmp")
            out_path = data_dir / "ml_training_base.parquet"
            # Se /data/ não tem o parquet (deploy novo), semear a partir do repo
            if not out_path.exists() and not force_rebuild:
                repo_seed = Path(__file__).parent / "ml_training_base.parquet"
                if repo_seed.exists():
                    shutil.copy2(str(repo_seed), str(out_path))
                    logging.info(f"[dip_dataset] Seed de {repo_seed} → {out_path}")
            cmd      = [sys.executable, str(script), "--start", start_val, "--out", str(out_path)]
            if ticker_val:
                cmd += ["--tickers", ticker_val]
            if no_fund:
                cmd += ["--no-fundamentals"]
            if force_rebuild:
                cmd += ["--force-rebuild"]

            _TIMEOUT = 6 * 3600
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT)

            if result.returncode == 0:
                try:
                    import pandas as pd
                    df = pd.read_parquet(out_path)
                    # Copiar para repo para rastreio em git
                    repo_pq = Path(__file__).parent / "ml_training_base.parquet"
                    try:
                        shutil.copy2(str(out_path), str(repo_pq))
                    except Exception as _cp:
                        logging.warning(f"[dip_dataset] cópia repo: {_cp}")
                    n_rows    = len(df)
                    n_tickers = df["ticker"].nunique() if "ticker" in df.columns else "?"
                    n_sectors = df["sector"].nunique() if "sector" in df.columns else "?"
                    avg_alpha = df["alpha_90d"].mean() * 100 if "alpha_90d" in df.columns else 0
                    pct_pos   = (df["alpha_90d"] > 0).mean() * 100 if "alpha_90d" in df.columns else 0
                    _reply(
                        f"*DipRadar Dataset — Concluído!*\n"
                        f"  Amostras: {n_rows:,}\n"
                        f"  Tickers: {n_tickers} | Sectores: {n_sectors}\n"
                        f"  Alpha médio 90d: {avg_alpha:+.1f}%\n"
                        f"  % positivos: {pct_pos:.0f}%\n\n"
                        f"Próximo passo: retreinar o DipRadar com `/admin\\_full\\_pipeline --dip-only`."
                    )
                except Exception as e:
                    _reply(f"Dataset gerado. (verificação: {e})")
            else:
                stderr = (result.stderr or "")[-500:]
                _reply(f"❌ Dataset falhou (exit {result.returncode})\n`{stderr}`")

        except subprocess.TimeoutExpired:
            _reply("⏰ Dataset excedeu 6h. O processo foi terminado.")
        except Exception as e:
            logging.error(f"[dip_dataset] {e}", exc_info=True)
            _reply(f"❌ Erro: `{e}`")

    threading.Thread(target=_run, daemon=True, name="dip-dataset").start()


# ── GitHub parquet commit helper ─────────────────────────────────────────────────

def _commit_file_to_github(local_path, repo_file_path: str, message: str) -> tuple[bool, str]:
    """Commit/update a file in the GitHub repo via Contents API.

    Requires env vars GITHUB_TOKEN and GITHUB_REPO (e.g. 'romeurf/DipRadar').
    """
    import base64
    import json as _json
    import urllib.request
    import urllib.error

    token = os.environ.get("GITHUB_TOKEN", "")
    repo  = os.environ.get("GITHUB_REPO", "")
    if not token or not repo:
        return False, "GITHUB_TOKEN ou GITHUB_REPO não configurados nas env vars do Railway"

    from pathlib import Path as _P
    content_b64 = base64.b64encode(_P(local_path).read_bytes()).decode()
    api_url = f"https://api.github.com/repos/{repo}/contents/{repo_file_path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }

    # Get existing file SHA (needed to update instead of create)
    sha = None
    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            sha = _json.loads(resp.read()).get("sha")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            return False, f"GET SHA HTTP {e.code}: {e.read().decode()[:200]}"
    except Exception as e:
        return False, f"GET SHA excepção: {e}"

    payload: dict = {"message": message, "content": content_b64}
    if sha:
        payload["sha"] = sha
    req = urllib.request.Request(
        api_url,
        data=_json.dumps(payload).encode(),
        headers=headers,
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = _json.loads(resp.read())
            url = result.get("content", {}).get("html_url", "")
            return True, url
    except urllib.error.HTTPError as e:
        return False, f"PUT HTTP {e.code}: {e.read().decode()[:300]}"
    except Exception as e:
        return False, f"PUT excepção: {e}"


# ── /admin_build_core_dataset ─────────────────────────────────────────────────────

def _handle_admin_build_core_dataset(parts: list[str]) -> None:
    """/admin_build_core_dataset [--force] [--no-fundamentals]

    Constrói o dataset de treino para o CoreML:
      - 57 quality compounders × histórico desde 1993
      - Detecta episódios MA200w (preço ≤5% acima da linha)
      - Features: price/MA/RSI/volume + sector ETF context
      - Targets: alpha_1y e alpha_2y vs SPY
      - Guarda em Railway Volume (/data) + commit ao GitHub
    """
    global _retrain_running
    if _retrain_running:
        _reply("⚠️ *Pipeline já em curso.* Aguarda que termine.")
        return

    force           = "--force" in parts
    no_fundamentals = "--no-fundamentals" in parts

    _reply(
        "*CoreML Dataset Build*\n"
        "57 quality compounders × histórico desde 1993\n"
        "Episódios MA200w + features price/MA/RSI/volume/sector ETF\n"
        "Targets: alpha\\_1y e alpha\\_2y vs SPY\n"
        "Output: Railway Volume `/data` + commit ao GitHub\n"
        "_Duração estimada: 20-40 min_"
    )

    def _run():
        import subprocess, sys, re as _re
        from pathlib import Path as _P

        global _pipeline_log
        _pipeline_log.clear()
        _plog("CoreML dataset build iniciado")

        data_dir = _P("/data") if _P("/data").exists() else _P("/tmp")
        out_path = data_dir / "core_training_base.parquet"
        script   = _P(__file__).parent / "scripts" / "build_core_dataset.py"

        cmd = [sys.executable, str(script), "--out", str(out_path), "--start", "1993-01-01"]
        if force:           cmd.append("--force-rebuild")
        if no_fundamentals: cmd.append("--no-fundamentals")

        # Streaming line-by-line → visível via /admin_pipeline_status
        # stderr=STDOUT: logging do Python vai para stderr por defeito; mergir tudo
        all_lines: list[str] = []
        returncode = 1
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )

            for raw in proc.stdout:
                line = raw.rstrip()
                all_lines.append(line)
                clean = _re.sub(r"^\d{2}:\d{2}:\d{2} \| ", "", line).strip()
                if clean:
                    _plog(clean)

            proc.wait(timeout=4 * 3600)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            _reply("❌ *Build timeout* (>4h) — verifica os logs.")
            return
        except Exception as e:
            _reply(f"❌ *Build excepção:* `{e}`")
            return

        if returncode != 0:
            err = "\n".join(all_lines[-15:])
            _plog(f"Build FALHOU (exit {returncode})")
            _reply(f"❌ *Build falhou (exit {returncode})*\n`{err[-400:]}`")
            return

        # Resumo de stats
        stats_lines = []
        for line in all_lines:
            if any(k in line for k in ("rows", "tickers", "alpha_1y", "alpha_2y", "Guardado", "episódios")):
                clean = _re.sub(r"^\d{2}:\d{2}:\d{2} \| ", "", line).strip()
                if clean:
                    stats_lines.append(f"  {clean}")

        # Commit ao GitHub
        github_line = ""
        if out_path.exists():
            _plog("A committar ao GitHub...")
            ok, msg = _commit_file_to_github(
                out_path,
                "core_training_base.parquet",
                "data: update CoreML training dataset [bot]",
            )
            if ok:
                _plog("GitHub commit OK")
                github_line = "\n✅ Commitado ao GitHub"
            else:
                _plog(f"GitHub commit falhou: {msg[:80]}")
                github_line = f"\n⚠️ GitHub commit falhou: {msg}"

        _plog("Build concluído")
        stats_text = "\n".join(stats_lines) if stats_lines else "  (sem stats)"
        _reply(
            f"✅ *CoreML Dataset Concluído!*\n"
            f"{stats_text}"
            f"{github_line}\n\n"
            f"Próximo passo: `/admin_train_core_model` para treinar o modelo ML."
        )

    threading.Thread(target=_run, daemon=True, name="core-dataset-build").start()


# ── /admin_train_core_model ────────────────────────────────────────────────────

def _handle_admin_train_core_model(parts: list[str]) -> None:
    """/admin_train_core_model [--dry-run]

    Trains the core_us ML model on core_training_base.parquet.
    Saves model bundle to /data/core_us_model.pkl.
    Once trained, the nightly CoreRadar scan uses ML scoring instead of heuristics.
    """
    global _retrain_running
    if _retrain_running:
        _reply("⚠️ Pipeline already running. Wait for it to finish.")
        return

    dry_run = "--dry-run" in parts

    _reply(
        "<b>CoreML Model Training</b>\n"
        f"{'Dry-run mode — model will NOT be saved.\n' if dry_run else ''}"
        "Training GradientBoostingRegressor on core_training_base.parquet.\n"
        "Features: price/MA/RSI/volume/sector/fundamentals (PIT EDGAR XBRL)\n"
        "Target: alpha_1y (1-year excess return vs SPY)\n"
        "<i>Estimated duration: 1-3 minutes</i>"
    )

    def _run():
        import subprocess, sys, re as _re
        from pathlib import Path as _P

        data_dir   = _P("/data") if _P("/data").exists() else _P(".")
        parquet    = data_dir / "core_training_base.parquet"
        script     = _P(__file__).parent / "scripts" / "train_core_model.py"

        if not parquet.exists():
            # Fallback: try repo root
            parquet = _P(__file__).parent / "core_training_base.parquet"
        if not parquet.exists():
            _reply("❌ core_training_base.parquet not found. Run /admin_build_core_dataset first.")
            return

        cmd = [sys.executable, str(script), "--parquet", str(parquet)]
        if dry_run:
            cmd.append("--dry-run")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            _reply("❌ Training timeout (>10 min) — check logs.")
            return
        except Exception as e:
            _reply(f"❌ Training exception: <code>{e}</code>")
            return

        output = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0:
            _reply(f"❌ Training failed (exit {result.returncode})\n<code>{output[-600:]}</code>")
            return

        # Reload model in the running process so the scan picks it up immediately
        try:
            import core_radar as _cr
            _cr._core_model_checked = False
            _cr._core_model_bundle  = None
            _cr._load_core_model()
            model_ready = _cr.is_model_ready()
        except Exception:
            model_ready = False

        model_status = "✅ Model loaded in memory — CoreRadar scan is now ML-driven." if model_ready else "⚠️ Model saved but not yet loaded — will load on next scan."

        _reply(
            f"{'🔬 DRY RUN — ' if dry_run else ''}✅ <b>core_us model trained!</b>\n\n"
            f"<pre>{output[-800:]}</pre>\n\n"
            + (f"{model_status}" if not dry_run else "Model NOT saved (dry run).")
        )

    threading.Thread(target=_run, daemon=True, name="core-model-train").start()


# ── /admin_full_pipeline ────────────────────────────────────────────────────────

def _handle_admin_full_pipeline(parts: list[str]) -> None:
    """/admin_full_pipeline [--force] [--dip-only] [--momentum-only]

    Pipeline completo com cache inteligente — tudo automatizado:
      Fase 1 (paralelo): DipRadar parquet regen + Momentum dataset build
      Fase 2 (sequencial): DipRadar retrain → Momentum retrain

    Cache: salta regen se parquet tiver menos de PIPELINE_MAX_AGE_DAYS dias (default: 7).
    Retrain é sempre executado quando o parquet foi (re)gerado com sucesso.

    Flags:
      --force          → ignora cache, reconstrói tudo
      --dip-only       → só pipeline DipRadar
      --momentum-only  → só pipeline MomentumRadar
    """
    global _retrain_running
    if _retrain_running:
        _reply("⚠️ *Retrain já em curso.* Aguarda que termine antes de lançar o pipeline.")
        return

    force         = "--force" in parts
    dip_only      = "--dip-only" in parts
    momentum_only = "--momentum-only" in parts
    max_age_days  = int(os.environ.get("PIPELINE_MAX_AGE_DAYS", "7"))

    scope = "DipRadar + MomentumRadar"
    if dip_only:      scope = "DipRadar"
    elif momentum_only: scope = "MomentumRadar"

    _reply(
        f"*Pipeline Completo — {scope}*\n"
        + (f"--force: ignora cache\n" if force else f"Cache: reconstrói se > {max_age_days} dias\n") +
        f"Fase 1: parquets em paralelo → Fase 2: modelos sequencial\n"
        f"_O bot continua activo durante todo o pipeline._"
    )

    def _run():
        import subprocess, sys, time as _t
        from pathlib import Path
        global _pipeline_log
        _pipeline_log.clear()
        _plog("Pipeline iniciado")

        data_dir    = Path("/data") if Path("/data").exists() else Path("/tmp")
        dip_parquet = data_dir / "ml_training_base.parquet"
        mom_parquet = data_dir / "momentum_training.parquet"

        def _is_fresh(path: Path) -> tuple[bool, float]:
            if force or not path.exists():
                return False, 999.0
            age = (_t.time() - path.stat().st_mtime) / 86400
            return age < max_age_days, round(age, 1)

        dip_fresh, dip_age = _is_fresh(dip_parquet)
        mom_fresh, mom_age = _is_fresh(mom_parquet)

        run_dip = (not dip_fresh) and (not momentum_only)
        run_mom = (not mom_fresh) and (not dip_only)

        # Cache report
        cache_lines = ["*Cache check:*"]
        if not momentum_only:
            cache_lines.append(f"  DipRadar parquet: {'fresco ({:.0f}d) — skip'.format(dip_age) if not run_dip else 'a regenerar...'}")
        if not dip_only:
            cache_lines.append(f"  Momentum parquet: {'fresco ({:.0f}d) — skip'.format(mom_age) if not run_mom else 'a regenerar...'}")
        _reply("\n".join(cache_lines))

        # ── Fase 1: parquets em paralelo ──────────────────────────────────────
        dip_regen_ok = not run_dip
        mom_regen_ok = not run_mom
        dip_regen_err: str = ""
        mom_regen_err: str = ""

        def _regen_dip():
            nonlocal dip_regen_ok, dip_regen_err
            _plog("Fase 1 DipRadar: regen parquet iniciado")
            try:
                script = Path(__file__).parent / "scripts" / "regenerate_training_base.py"
                r = subprocess.run(
                    [sys.executable, str(script)],
                    capture_output=True, text=True, timeout=6 * 3600,
                )
                if r.returncode == 0:
                    dip_regen_ok = True
                    _plog("Fase 1 DipRadar: parquet OK")
                else:
                    dip_regen_err = (r.stderr or r.stdout or "")[-300:]
                    _plog(f"Fase 1 DipRadar: FALHOU — {dip_regen_err[:100]}")
            except Exception as e:
                dip_regen_err = str(e)
                _plog(f"Fase 1 DipRadar: excepção — {e}")

        def _regen_mom():
            nonlocal mom_regen_ok, mom_regen_err
            _plog("Fase 1 Momentum: build dataset iniciado")
            try:
                script = Path(__file__).parent / "momentum_radar" / "scripts" / "build_dataset.py"
                r = subprocess.run(
                    [sys.executable, str(script)],
                    capture_output=True, text=True, timeout=8 * 3600,
                )
                if r.returncode == 0:
                    mom_regen_ok = True
                    _plog("Fase 1 Momentum: parquet OK")
                else:
                    mom_regen_err = (r.stderr or r.stdout or "")[-300:]
                    _plog(f"Fase 1 Momentum: FALHOU — {mom_regen_err[:100]}")
            except Exception as e:
                mom_regen_err = str(e)
                _plog(f"Fase 1 Momentum: excepção — {e}")

        phase1_threads = []
        if run_dip: phase1_threads.append(threading.Thread(target=_regen_dip, daemon=True, name="pipeline-dip-regen"))
        if run_mom: phase1_threads.append(threading.Thread(target=_regen_mom, daemon=True, name="pipeline-mom-regen"))

        if phase1_threads:
            _reply(f"*Fase 1 — {len(phase1_threads)} parquet(s) a regenerar em paralelo...*")
            for t in phase1_threads:
                t.start()
            for t in phase1_threads:
                t.join()

            ph1_lines = ["*Fase 1 concluída:*"]
            if run_dip:
                ph1_lines.append(f"  {'✅' if dip_regen_ok else '❌'} DipRadar parquet" + (f": {dip_regen_err}" if dip_regen_err else ""))
            if run_mom:
                ph1_lines.append(f"  {'✅' if mom_regen_ok else '❌'} Momentum parquet" + (f": {mom_regen_err}" if mom_regen_err else ""))
            _reply("\n".join(ph1_lines))

        # ── Fase 2: treinar modelos em paralelo ───────────────────────────────
        # Só treinar se o parquet respectivo foi regenerado com sucesso (ou era fresh e estamos em force).
        # force + fresh → parquet não foi regenerado mas o utilizador quer retreinar → train anyway.
        train_dip = dip_regen_ok and not momentum_only
        train_mom = mom_regen_ok and not dip_only

        if not train_dip and not train_mom:
            _reply("*Pipeline terminado.* Nada a treinar.")
            return

        dip_train_res: dict  = {}
        mom_train_res: dict  = {}
        dip_train_err2: str  = ""
        mom_train_err2: str  = ""

        def _train_dip():
            nonlocal dip_train_res, dip_train_err2
            _plog("Fase 2 DipRadar: treino iniciado")
            try:
                from monthly_retrain import run_monthly_retrain_v3
                dip_train_res = run_monthly_retrain_v3() or {}
                dec = dip_train_res.get("decision", "?")
                _plog(f"Fase 2 DipRadar: {dec}")
            except Exception as e:
                dip_train_err2 = str(e)
                _plog(f"Fase 2 DipRadar: excepção — {e}")
                logging.error(f"[pipeline] DipRadar train: {e}", exc_info=True)

        def _train_mom():
            nonlocal mom_train_res, mom_train_err2
            _plog("Fase 2 Momentum: treino iniciado")
            try:
                from pathlib import Path as _P
                _dd = _P("/data") if _P("/data").exists() else _P("/tmp")
                from momentum_radar.trainer import run_momentum_training
                mom_train_res = run_momentum_training(
                    dataset_path=_dd / "momentum_training.parquet",
                    bundle_path =_dd / "momentum_model.pkl",
                    report_path =_dd / "momentum_report.json",
                ) or {}
                dec = mom_train_res.get("decision", "?")
                ic  = mom_train_res.get("ic_ensemble")
                _plog(f"Fase 2 Momentum: {dec} | IC={ic:.4f}" if isinstance(ic, float) else f"Fase 2 Momentum: {dec}")
            except Exception as e:
                mom_train_err2 = str(e)
                _plog(f"Fase 2 Momentum: excepção — {e}")
                logging.error(f"[pipeline] Momentum train: {e}", exc_info=True)

        n_models = (1 if train_dip else 0) + (1 if train_mom else 0)
        _reply(f"*Fase 2 — {n_models} modelo(s) a treinar sequencialmente...*\n_Usa /admin\\_pipeline\\_status para ver progresso._")
        if train_dip:
            _reply("  ⏳ DipRadar a treinar...")
            _train_dip()
        if train_mom:
            _reply("  ⏳ MomentumRadar a treinar...")
            _train_mom()

        # ── Relatório final ───────────────────────────────────────────────────
        def _f(v): return f"{float(v):.4f}" if isinstance(v, (int, float)) else "N/A"

        final_lines = ["*Pipeline concluído!*", ""]

        if train_dip:
            if dip_train_err2:
                final_lines += [f"  ❌ *DipRadar: excepção*", f"  `{dip_train_err2[:200]}`"]
            else:
                dec    = dip_train_res.get("decision", "?")
                rho    = dip_train_res.get("candidate_rho_alpha")
                p_rho  = dip_train_res.get("production_rho_alpha")
                brier  = dip_train_res.get("candidate_brier")
                reason = dip_train_res.get("reason", "")
                icon   = {"PROMOTED": "🚀", "PENDING": "⚠️", "KEPT_FLOOR": "🛑"}.get(dec, "❌")
                prod_s = f" (prod: {_f(p_rho)})" if p_rho is not None else ""
                brier_s = f" | Brier={_f(brier)}" if brier is not None else ""
                final_lines.append(f"  {icon} *DipRadar: {dec}* | ρ={_f(rho)}{prod_s}{brier_s}")
                if reason:
                    final_lines.append(f"    _{reason[:200]}_")
        else:
            final_lines.append("  ⏩ DipRadar retrain: saltado")

        if train_mom:
            if mom_train_err2:
                final_lines += [f"  ❌ *MomentumRadar: excepção*", f"  `{mom_train_err2[:200]}`"]
            else:
                dec_m     = mom_train_res.get("decision", "?")
                ic_m      = mom_train_res.get("ic_ensemble")
                ic_prod_m = mom_train_res.get("ic_prod")
                n_m       = mom_train_res.get("n_train", 0)
                horizons  = mom_train_res.get("horizons", {})
                sec_ics   = mom_train_res.get("sector_ics", {})
                reason_m  = mom_train_res.get("reason", "")
                icon_m    = {"PROMOTED": "🚀", "KEPT": "⚠️", "FAILED": "❌"}.get(dec_m, "❌")
                prod_m_s  = f" (prod: {_f(ic_prod_m)})" if ic_prod_m is not None else ""
                final_lines.append(f"  {icon_m} *MomentumRadar: {dec_m}* | IC={_f(ic_m)}{prod_m_s} | {n_m:,} amostras")
                if reason_m:
                    final_lines.append(f"    _{reason_m[:200]}_")
                if horizons:
                    h_str = "  |  ".join(f"{k}: {_f(v)}" for k, v in sorted(horizons.items()))
                    final_lines.append(f"    Horizontes: {h_str}")
                if sec_ics:
                    sec_sorted = sorted(sec_ics.items(), key=lambda x: -(x[1] or -99))
                    sec_lines = [f"    *Sectores (IC train):*"]
                    for s, ic in sec_sorted:
                        ok = "✅" if isinstance(ic, float) and ic >= 0.05 else "⚠️"
                        sec_lines.append(f"      {ok} {s}: {_f(ic)}")
                    final_lines.extend(sec_lines)
        else:
            final_lines.append("  ⏩ Momentum retrain: saltado")

        # GitHub sync para modelos promovidos
        try:
            from pathlib import Path as _Pg
            from github_sync import sync_training_artifacts, sync_momentum_artifacts
            _dd_g = _Pg("/data") if _Pg("/data").exists() else _Pg("/tmp")
            if train_dip and dip_train_res.get("decision") == "PROMOTED":
                sync_training_artifacts(_dd_g)
            if train_mom and mom_train_res.get("decision") == "PROMOTED":
                sync_momentum_artifacts(_dd_g)
        except Exception as _gs_e:
            logging.warning(f"[pipeline] github_sync: {_gs_e}")

        _plog("Pipeline concluído")
        _reply("\n".join(final_lines))

    threading.Thread(target=_run, daemon=True, name="admin-full-pipeline").start()


# ── /admin_run_pipeline ────────────────────────────────────────────────────────

def _handle_admin_run_pipeline(parts: list[str]) -> None:
    """/admin_run_pipeline [model flags] [stage flags] [--force] [--test T1,T2]

    Unified pipeline: dataset build (parallel) → retrain (sequential).

    ── Model flags (default: all three) ──
      --dip           DIP model only
      --momentum      Momentum model only
      --core          CORE model only
      (combine freely: --dip --core)

    ── Stage flags (default: both) ──
      --dataset       Build datasets only — do not retrain
      --retrain       Retrain only — skip dataset build (use existing parquets)

    ── Other ──
      --force         Force full rebuild AND clear price cache (re-downloads all tickers)
      --test T1,T2    Smoke-test: run on these tickers only (~2 min)

    Examples:
      /admin_run_pipeline                       → full pipeline, all models
      /admin_run_pipeline --dip                 → DIP dataset + retrain
      /admin_run_pipeline --momentum --dataset  → build momentum dataset only
      /admin_run_pipeline --retrain             → retrain DIP + momentum (skip dataset)
      /admin_run_pipeline --dip --retrain       → retrain DIP only
      /admin_run_pipeline --core                → build CORE dataset (no ML retrain yet)
      /admin_run_pipeline --force               → full rebuild, ignore cache
      /admin_run_pipeline --test AAPL,MSFT      → quick smoke test on 2 tickers
    """
    global _retrain_running
    if _retrain_running:
        _reply("A pipeline is already running. Wait for it to finish (/admin_pipeline_status).")
        return

    flags = [p.lower() for p in parts[1:]]

    # ── parse model selection ─────────────────────────────────────────────────
    want_dip  = "--dip"  in flags
    want_mom  = "--momentum" in flags
    want_core = "--core" in flags
    if not want_dip and not want_mom and not want_core:
        want_dip = want_mom = want_core = True  # default: all

    # ── parse stage selection ─────────────────────────────────────────────────
    want_dataset = "--dataset" in flags
    want_retrain = "--retrain" in flags
    if not want_dataset and not want_retrain:
        want_dataset = want_retrain = True       # default: both

    # ── other flags ───────────────────────────────────────────────────────────
    force = "--force" in flags
    test_tickers = None
    for i, f in enumerate(flags):
        if f == "--test" and i + 1 < len(flags):
            test_tickers = flags[i + 1]
            break

    # ── build human-readable summary ──────────────────────────────────────────
    models_str = " + ".join(
        m for m, sel in [("DIP", want_dip), ("Momentum", want_mom), ("CORE", want_core)] if sel
    )
    stages_str = " + ".join(
        s for s, sel in [("dataset build", want_dataset), ("retrain", want_retrain)] if sel
    )
    test_str = f" | TEST MODE: {test_tickers}" if test_tickers else ""
    force_str = " | FORCE (ignore cache)" if force else ""

    _reply(
        f"Pipeline starting: {models_str}\n"
        f"Stages: {stages_str}{force_str}{test_str}\n"
        f"Dataset builds run in parallel. Retrains run sequentially after.\n"
        f"Use /admin_pipeline_status to follow progress."
    )

    def _run():
        import sys, time as _t
        from pathlib import Path
        global _retrain_running, _pipeline_log, _pipeline_start
        _retrain_running = True
        _pipeline_log.clear()
        _pipeline_start = _t.time()

        data_dir = Path("/data") if Path("/data").exists() else Path("/tmp")

        # ─────────────────────────────────────────────────────────────────────
        # Phase 1 — Dataset builds (parallel threads, each streams its output)
        # ─────────────────────────────────────────────────────────────────────
        dip_ds_ok   = not want_dataset or not want_dip
        mom_ds_ok   = not want_dataset or not want_mom
        core_ds_ok  = not want_dataset or not want_core
        dip_ds_err  = mom_ds_err = core_ds_err = ""

        def _build_dip():
            nonlocal dip_ds_ok, dip_ds_err
            script = Path(__file__).parent / "scripts" / "build_dip_dataset.py"
            out    = data_dir / "ml_training_base.parquet"
            cmd    = [sys.executable, str(script), "--start", "1993-01-01", "--out", str(out)]
            if test_tickers: cmd += ["--tickers", test_tickers]
            if force:        cmd += ["--force-rebuild", "--clear-cache"]
            rc, lines = _run_script_streaming(cmd, "Phase 1 — DIP dataset")
            if rc == 0:
                dip_ds_ok = True
            else:
                dip_ds_err = next((l for l in reversed(lines) if "error" in l.lower()), lines[-1] if lines else "")[:120]

        def _build_momentum():
            nonlocal mom_ds_ok, mom_ds_err
            script = Path(__file__).parent / "momentum_radar" / "scripts" / "build_dataset.py"
            out    = data_dir / "momentum_training.parquet"
            cmd    = [sys.executable, str(script), "--start", "1993-01-01", "--out", str(out)]
            if test_tickers: cmd += ["--tickers", test_tickers]
            if force:        cmd += ["--force-rebuild", "--clear-cache"]
            rc, lines = _run_script_streaming(cmd, "Phase 1 — Momentum dataset")
            if rc == 0:
                mom_ds_ok = True
            else:
                mom_ds_err = next((l for l in reversed(lines) if "error" in l.lower()), lines[-1] if lines else "")[:120]

        def _build_core():
            nonlocal core_ds_ok, core_ds_err
            script = Path(__file__).parent / "scripts" / "build_core_dataset.py"
            out    = data_dir / "core_training_base.parquet"
            cmd    = [sys.executable, str(script), "--start", "1993-01-01", "--out", str(out)]
            if test_tickers: cmd += ["--tickers", test_tickers]
            if force:        cmd += ["--force-rebuild", "--clear-cache"]
            rc, lines = _run_script_streaming(cmd, "Phase 1 — CORE dataset")
            if rc == 0:
                core_ds_ok = True
            else:
                core_ds_err = next((l for l in reversed(lines) if "error" in l.lower()), lines[-1] if lines else "")[:120]

        # Datasets run SEQUENTIALLY — one at a time, zero risk of partial/corrupt data.
        # If one fails, the pipeline stops before wasting time on the next.
        if want_dataset:
            ds_sequence = []
            if want_dip:  ds_sequence.append(("DIP",      _build_dip))
            if want_mom:  ds_sequence.append(("Momentum", _build_momentum))
            if want_core: ds_sequence.append(("CORE",     _build_core))

            for i, (model_name, build_fn) in enumerate(ds_sequence):
                n = len(ds_sequence)
                _reply(f"Phase 1 [{i+1}/{n}] — {model_name} dataset starting… (/ps for progress)")
                build_fn()

                ok  = dip_ds_ok  if model_name == "DIP"      else (mom_ds_ok if model_name == "Momentum" else core_ds_ok)
                err = dip_ds_err if model_name == "DIP"      else (mom_ds_err if model_name == "Momentum" else core_ds_err)

                if ok:
                    _reply(f"Phase 1 [{i+1}/{n}] — {model_name} dataset: OK")
                else:
                    _reply(
                        f"Phase 1 [{i+1}/{n}] — {model_name} dataset: FAILED\n"
                        f"{err}\n"
                        f"Fix the error above before continuing. Pipeline stopped."
                    )
                    _retrain_running = False
                    return
        else:
            _plog("Phase 1 — skipped (--retrain only)")

        # ─────────────────────────────────────────────────────────────────────
        # Phase 2 — Retrains (sequential — CPU-bound, resource intensive)
        # ─────────────────────────────────────────────────────────────────────
        if not want_retrain:
            elapsed = round(_t.time() - start_wall)
            _plog("Phase 2 — skipped (--dataset only)")
            _reply(f"Pipeline complete (dataset only). Elapsed: {elapsed}s")
            _retrain_running = False
            return

        train_dip = want_dip  and want_retrain and dip_ds_ok
        train_mom = want_mom  and want_retrain and mom_ds_ok
        # CORE has no ML retrain yet — rule-based scoring in core_radar.py
        # When CORE ML training is added, wire it here.

        if not train_dip and not train_mom:
            elapsed = round(_t.time() - start_wall)
            _reply(
                "Phase 2 — nothing to retrain "
                "(datasets failed or only CORE was selected — CORE uses rule-based scoring, no ML retrain).\n"
                f"Elapsed: {elapsed}s"
            )
            _retrain_running = False
            return

        models_training = " → ".join(
            m for m, sel in [("DIP", train_dip), ("Momentum", train_mom)] if sel
        )
        _plog(f"Phase 2 — retraining sequentially: {models_training}")
        _reply(f"Phase 2 — retraining: {models_training} (use /admin_pipeline_status to follow)")

        dip_result: dict = {}
        mom_result: dict = {}

        if train_dip:
            _plog("Phase 2 — DIP retrain: starting")
            try:
                from monthly_retrain import run_monthly_retrain_v3
                dip_result = run_monthly_retrain_v3() or {}
                dec = dip_result.get("decision", "?")
                rho = dip_result.get("rho_alpha_mean")
                rho_str = f" | rho_alpha={rho:.4f}" if isinstance(rho, float) else ""
                _plog(f"Phase 2 — DIP retrain: {dec}{rho_str}")
            except Exception as e:
                dip_result = {"decision": "ERROR", "error": str(e)}
                _plog(f"Phase 2 — DIP retrain: exception — {e}")
                logging.error(f"[run_pipeline] DIP retrain: {e}", exc_info=True)

        if train_mom:
            _plog("Phase 2 — Momentum retrain: starting")
            try:
                from momentum_radar.trainer import run_momentum_training
                mom_result = run_momentum_training(
                    dataset_path=data_dir / "momentum_training.parquet",
                    bundle_path =data_dir / "momentum_model.pkl",
                    report_path =data_dir / "momentum_report.json",
                ) or {}
                dec = mom_result.get("decision", "?")
                ic  = mom_result.get("ic_ensemble")
                ic_str = f" | IC={ic:.4f}" if isinstance(ic, float) else ""
                _plog(f"Phase 2 — Momentum retrain: {dec}{ic_str}")
            except Exception as e:
                mom_result = {"decision": "ERROR", "error": str(e)}
                _plog(f"Phase 2 — Momentum retrain: exception — {e}")
                logging.error(f"[run_pipeline] Momentum retrain: {e}", exc_info=True)

        # ── GitHub sync for promoted models ───────────────────────────────────
        try:
            from github_sync import sync_training_artifacts, sync_momentum_artifacts
            if train_dip and dip_result.get("decision") == "PROMOTED":
                sync_training_artifacts(data_dir)
            if train_mom and mom_result.get("decision") == "PROMOTED":
                sync_momentum_artifacts(data_dir)
        except Exception as e:
            logging.warning(f"[run_pipeline] github_sync: {e}")

        # ── auto-publish charts after successful retrain ───────────────────────
        try:
            from chart_generator import generate_all_charts
            generate_all_charts()
            if os.environ.get("GITHUB_TOKEN") and os.environ.get("GITHUB_CHARTS_REPO"):
                from github_publisher import push_all_charts
                push_all_charts()
                _plog("Charts published to GitHub")
        except Exception as e:
            logging.warning(f"[run_pipeline] chart publish: {e}")

        # ── final summary ─────────────────────────────────────────────────────
        elapsed = round(_t.time() - start_wall)
        summary = [f"Pipeline complete in {elapsed}s", ""]

        def _dec_line(model: str, result: dict) -> str:
            dec = result.get("decision", "ERROR")
            err = result.get("error", "")
            if dec == "PROMOTED":
                rho = result.get("rho_alpha_mean") or result.get("ic_ensemble")
                metric = f"metric={rho:.4f}" if isinstance(rho, float) else ""
                return f"  {model}: PROMOTED {metric}"
            elif dec == "KEPT":
                return f"  {model}: kept (candidate did not beat production)"
            elif dec == "PENDING":
                return f"  {model}: saved as pending (manual review needed)"
            else:
                return f"  {model}: FAILED — {err[:80]}"

        if train_dip: summary.append(_dec_line("DIP",      dip_result))
        if train_mom: summary.append(_dec_line("Momentum", mom_result))
        if want_core: summary.append("  CORE: dataset built (rule-based scoring — no ML retrain)")

        _reply("\n".join(summary))
        _retrain_running = False

    threading.Thread(target=_run, daemon=True, name="admin-run-pipeline").start()


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

        # Quando promovido, copiar artefactos para repo (git-trackable)
        if decision == "PROMOTED":
            import shutil as _shutil
            from pathlib import Path as _P
            data_dir  = _P("/data") if _P("/data").exists() else _P("/tmp")
            repo_ml   = _P(__file__).parent / "ml_training"
            for src, dst in [
                (data_dir / "dip_models.pkl",  repo_ml / "dip_models.pkl"),
                (data_dir / "ml_report.json",  repo_ml / "ml_report.json"),
            ]:
                try:
                    if src.exists():
                        _shutil.copy2(str(src), str(dst))
                except Exception as _cp:
                    logging.warning(f"[admin_retrain] cópia repo {src.name}: {_cp}")

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
    """/paper_portfolio [months | open | drip | missed]

    Paper trading portfolio report. The bot simulates its own BUY signals and
    tracks whether it beats SPY over 90 days.

    Usage:
      /paper_portfolio          → last 3 months
      /paper_portfolio 6        → last 6 months
      /paper_portfolio open     → all open positions with live weights
      /paper_portfolio drip     → DRIP audit trail for CORE positions
      /paper_portfolio missed   → MISSED_CAPITAL signals (lost alpha due to full pool)
    """
    def _run():
        try:
            from paper_trading import (
                get_monthly_performance, format_performance_report, _load
            )

            sub = parts[1].lower() if len(parts) > 1 else ""

            # ── open positions ────────────────────────────────────────────────
            if sub == "open":
                from paper_trading import (
                    get_paper_portfolio_value, get_paper_allocated,
                    get_paper_available, PAPER_PORTFOLIO_EUR,
                )
                from datetime import date as _date
                trades  = _load()
                open_ps = [t for t in trades if t.get("status") == "OPEN"]
                if not open_ps:
                    _reply("No open paper trading positions.")
                    return
                port_val  = get_paper_portfolio_value(trades)
                allocated = get_paper_allocated(trades)
                available = get_paper_available(trades)
                port_ret  = (port_val - PAPER_PORTFOLIO_EUR) / max(PAPER_PORTFOLIO_EUR, 1) * 100
                lines = [
                    f"Paper portfolio — {len(open_ps)} open positions",
                    f"Value: €{port_val:.0f} ({port_ret:+.1f}%) | "
                    f"Allocated: €{allocated:.0f} | Available: €{available:.0f}",
                    "",
                ]
                for t in sorted(open_ps, key=lambda x: x["open_date"], reverse=True)[:15]:
                    days = (_date.today() - _date.fromisoformat(t["open_date"])).days
                    wt   = t.get("weight_pct", 0)
                    kind = t.get("trade_type", "DIP")
                    lines.append(
                        f"  [{kind}] {t['ticker']}: {t['open_date']} ({days}d) "
                        f"@ {t['open_price']:.2f} | {wt:.1f}% | €{t['amount_eur']:.0f}"
                    )
                _reply("\n".join(lines))
                return

            # ── DRIP audit trail ──────────────────────────────────────────────
            if sub == "drip":
                from paper_trading import get_drip_history, _TAX_DISCLAIMER
                data = get_drip_history()
                pos  = data["positions"]
                tot  = data["totals"]

                header = [
                    "CORE DRIP Ledger — Dividend Reinvestment Audit Trail",
                    "",
                    f"Positions: {tot['total_positions']} | "
                    f"Events: {tot['total_drip_events']} | "
                    f"Total reinvested: €{tot['total_reinvested_eur']:,.2f} (gross)",
                    "",
                    f"TAX: {_TAX_DISCLAIMER}",
                    "",
                ]
                _reply("\n".join(header))

                if not pos:
                    _reply(
                        "No DRIP events recorded yet.\n"
                        "DRIP tracking activates automatically on the next dividend ex-date "
                        "for each open CORE position."
                    )
                    return

                for p in pos:
                    events = p["drip_events"]
                    block = [
                        f"{p['ticker']} — opened {p['open_date']} | "
                        f"€{p['amount_eur']:,.2f} deployed",
                        f"  Shares: {p['shares_original']:.4f} → {p['shares_current']:.4f} "
                        f"(+{p['shares_added_via_drip']:.4f} via DRIP)",
                        f"  Total reinvested: €{p['total_reinvested_eur']:,.4f} "
                        f"across {len(events)} event(s)",
                    ]
                    if not events:
                        block.append("  No dividend events recorded yet.")
                    else:
                        block.append("  Events:")
                        for ev in events:
                            block.append(
                                f"    {ev['date']}: ${ev['per_share_usd']:.4f}/sh × "
                                f"{ev['shares_held']:.4f} sh = "
                                f"€{ev['gross_eur']:.4f} → "
                                f"+{ev['new_shares']:.6f} sh @ ${ev['reinvest_price_usd']:.2f} "
                                f"[{ev['tax_note']}]"
                            )
                    if p.get("note"):
                        block.append(f"  Note: {p['note']}")
                    _reply("\n".join(block))

                # Send the full ledger as a downloadable Markdown file
                from paper_trading import format_drip_report_md
                md_content = format_drip_report_md()
                sent = _send_document(
                    md_content.encode("utf-8"),
                    filename="dipr_drip_ledger.md",
                    caption=(
                        "DipRadar CORE DRIP Ledger — full audit trail.\n"
                        "Every dividend event, date, amount, shares added.\n"
                        "GROSS (pre-tax). See disclaimer inside."
                    ),
                )
                if not sent:
                    _reply("Could not send file via Telegram. Use /admin_publish_charts to push to GitHub.")
                return

            # ── missed signals ────────────────────────────────────────────────
            if sub == "missed":
                from paper_trading import get_missed_trades_stats, _load as _pt_load
                trades = _pt_load()
                missed = [t for t in trades if t.get("status") == "MISSED_CAPITAL"]
                stats  = get_missed_trades_stats(trades)

                if not missed:
                    _reply("No MISSED_CAPITAL signals recorded yet.")
                    return

                lines = [
                    f"MISSED_CAPITAL — {len(missed)} signals not opened (pool was full)",
                    f"Resolved: {stats.get('resolved', 0)} | Pending: {stats.get('pending', 0)}",
                    "",
                ]
                for m in sorted(missed, key=lambda x: x.get("signal_date", ""), reverse=True)[:20]:
                    resolved = m.get("actual_return_90d")
                    ret_str  = f"{resolved:+.1f}%" if resolved is not None else "pending"
                    alpha_str = (
                        f" vs SPY {m.get('actual_alpha_90d', 0):+.1f}%"
                        if m.get("actual_alpha_90d") is not None else ""
                    )
                    lines.append(
                        f"  [{m.get('trade_type','?')}] {m.get('ticker','?')} "
                        f"{m.get('signal_date','?')}: "
                        f"requested €{m.get('requested_eur',0):.0f}, "
                        f"available €{m.get('available_eur',0):.0f} → {ret_str}{alpha_str}"
                    )
                _reply("\n".join(lines))
                return

            # ── default: monthly summary ──────────────────────────────────────
            months = 3
            if sub:
                try:
                    months = int(sub)
                except ValueError:
                    pass

            perf   = get_monthly_performance(months_back=months)
            report = format_performance_report(perf)
            _reply(f"Paper Trading — Bot vs Market\n\n{report}")

            from paper_trading import format_monthly_trade_list
            trade_list = format_monthly_trade_list(months_back=months)
            if trade_list and "Sem trades" not in trade_list:
                _reply(f"Trades by month:\n{trade_list}")

        except Exception as e:
            logging.error(f"[paper_performance] {e}", exc_info=True)
            _reply(f"Error calculating paper performance: {e}")

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


# ── /core handlers ────────────────────────────────────────────────────────────────


def _handle_core(parts: list[str]) -> None:
    """/core — mostra todas as posições CORE com P&L e yield acumulado."""
    try:
        from paper_trading import format_core_holdings_report
        _reply(format_core_holdings_report())
    except Exception as e:
        _reply(f"❌ Erro ao ler posições CORE: `{e}`")


def _handle_core_add(parts: list[str]) -> None:
    """/core_add TICKER PRICE AMOUNT_EUR [--yield YLD] [--note "texto"]

    Exemplos:
      /core_add MSFT 359 2000
      /core_add NVO 38 1500 --yield 5.0 --note "alta yield, comprado no dip"
    """
    if len(parts) < 4:
        _reply(
            "*Uso:* `/core_add TICKER PRICE AMOUNT_EUR [--yield YLD%] [--note texto]`\n\n"
            "Exemplo: `/core_add MSFT 359 2000`\n"
            "Exemplo: `/core_add NVO 38 1500 --yield 5.0 --note \"comprado no dip\"`"
        )
        return

    try:
        ticker     = parts[1].upper().strip()
        open_price = float(parts[2])
        amount_eur = float(parts[3])
    except (ValueError, IndexError):
        _reply("❌ Preço e montante têm de ser números. Ex: `/core_add MSFT 359 2000`")
        return

    # Flags opcionais
    annual_yield = 0.0
    note = ""
    i = 4
    while i < len(parts):
        if parts[i] == "--yield" and i + 1 < len(parts):
            try:
                annual_yield = float(parts[i + 1])
                i += 2
            except ValueError:
                i += 1
        elif parts[i] == "--note" and i + 1 < len(parts):
            note = " ".join(parts[i + 1:])
            break
        else:
            i += 1

    try:
        from paper_trading import record_core_holding
        trade_id = record_core_holding(
            ticker=ticker, open_price=open_price, amount_eur=amount_eur,
            annual_yield_pct=annual_yield, note=note,
            model_id="core_us", sector="",
        )
        if trade_id:
            yield_str = f" | Yield: {annual_yield:.1f}%/ano" if annual_yield > 0 else ""
            note_str  = f"\n_{note}_" if note else ""
            _reply(
                f"✅ *CORE {ticker}* registado `[{trade_id}]`\n"
                f"  €{amount_eur:.0f} @ ${open_price:.2f}{yield_str}{note_str}\n"
                f"_Posição de longo prazo — sem gestão automática._"
            )
        else:
            _reply("❌ Não foi possível registar a posição CORE.")
    except Exception as e:
        _reply(f"❌ Erro: `{e}`")


def _handle_core_close(parts: list[str]) -> None:
    """/core_close TICKER PRICE — fecha manualmente uma posição CORE."""
    if len(parts) < 3:
        _reply("*Uso:* `/core_close TICKER PRICE`\nEx: `/core_close MSFT 420`")
        return
    try:
        ticker      = parts[1].upper().strip()
        close_price = float(parts[2])
    except (ValueError, IndexError):
        _reply("❌ Preço inválido.")
        return
    try:
        from paper_trading import close_core_holding
        closed_trade = close_core_holding(ticker, close_price)
        if closed_trade:
            ret_s   = f"{closed_trade.get('return_pct', 0):+.1f}%"
            alpha_s = f"{closed_trade.get('alpha_pct', 0):+.1f}%" if closed_trade.get("alpha_pct") is not None else "N/A"
            _reply(
                f"✅ <b>CORE {ticker}</b> closed manually @ ${close_price:.2f}\n"
                f"Return: {ret_s}  Alpha vs SPY: {alpha_s}"
            )
            # Generate and send trade card (CORE cards include yield income line)
            try:
                from chart_generator import generate_trade_card
                card = generate_trade_card(closed_trade)
                if card and card.exists():
                    _send_photo(card.read_bytes(), caption=f"CORE {ticker} — trade card")
            except Exception as _ce:
                logging.debug(f"[trade_card] core close card: {_ce}")
        else:
            _reply(f"⚠️ No open CORE position found for {ticker}.")
    except Exception as e:
        _reply(f"❌ Error: <code>{e}</code>")


# ── /core_universe handlers ───────────────────────────────────────────────────────


def _handle_core_universe(parts: list[str]) -> None:
    """/core_universe — lista o universo CoreRadar."""
    try:
        from core_radar import get_universe
        entries = get_universe()
        if not entries:
            _reply(
                "_Universo CORE vazio._\n\n"
                "Adiciona tickers com:\n"
                "`/core_universe_add TICKER [yield%] [nota]`\n\n"
                "Exemplo: `/core_universe_add MSFT 0.7 compounder secular`"
            )
            return
        lines = [f"💎 *Universo CoreRadar — {len(entries)} tickers*", ""]
        for e in entries:
            ticker   = e["ticker"]
            yld      = e.get("annual_yield_pct", 0)
            note     = e.get("note", "")
            yld_str  = f" | *{yld:.1f}%/ano*" if yld > 0 else ""
            note_str = f" — _{note}_" if note else ""
            lines.append(f"  • *{ticker}*{yld_str}{note_str}")
        lines += [
            "",
            "_/core_universe_add TICK [yield] [nota] — adicionar_",
            "_/core_universe_remove TICK — remover_",
        ]
        _reply("\n".join(lines))
    except Exception as e:
        _reply(f"❌ Erro: `{e}`")


def _handle_core_universe_add(parts: list[str]) -> None:
    """/core_universe_add TICKER [yield_pct] [nota]"""
    if len(parts) < 2:
        _reply(
            "*Uso:* `/core_universe_add TICKER [yield%] [nota]`\n\n"
            "Exemplos:\n"
            "`/core_universe_add MSFT 0.7 compounder secular cloud+AI`\n"
            "`/core_universe_add ABBV 3.8 pharma high-yield Humira+Skyrizi`"
        )
        return
    ticker       = parts[1].upper().strip()
    annual_yield = 0.0
    note         = ""
    if len(parts) >= 3:
        try:
            annual_yield = float(parts[2])
            note = " ".join(parts[3:]) if len(parts) > 3 else ""
        except ValueError:
            note = " ".join(parts[2:])
    try:
        from core_radar import add_to_universe
        added = add_to_universe(ticker, annual_yield_pct=annual_yield, note=note)
        if added:
            yld_str  = f" | Yield: {annual_yield:.1f}%/ano" if annual_yield > 0 else ""
            note_str = f"\n_{note}_" if note else ""
            _reply(
                f"✅ *{ticker}* adicionado ao universo CoreRadar{yld_str}{note_str}\n"
                f"_Será escaneado no próximo CoreRadar scan (22h Lisboa)._"
            )
        else:
            _reply(f"⚠️ *{ticker}* já está no universo CoreRadar.")
    except Exception as e:
        _reply(f"❌ Erro: `{e}`")


def _handle_core_universe_remove(parts: list[str]) -> None:
    """/core_universe_remove TICKER"""
    if len(parts) < 2:
        _reply("*Uso:* `/core_universe_remove TICKER`\nEx: `/core_universe_remove NVO`")
        return
    ticker = parts[1].upper().strip()
    try:
        from core_radar import remove_from_universe
        removed = remove_from_universe(ticker)
        if removed:
            _reply(f"✅ *{ticker}* removido do universo CoreRadar.")
        else:
            _reply(f"⚠️ *{ticker}* não estava no universo CoreRadar.")
    except Exception as e:
        _reply(f"❌ Erro: `{e}`")


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
            "`/core`                    → Posições CORE abertas (hold forever)\n"
            "`/core_add TICK P EUR`     → Registar compra CORE manual\n"
            "`/core_close TICK P`       → Fechar posição CORE manualmente\n"
            "`/core_universe`           → Ver universo CoreRadar\n"
            "`/core_universe_add TICK`  → Adicionar ticker ao universo CORE\n"
            "`/core_universe_remove T`  → Remover ticker do universo CORE\n"
            "`/admin_regen_parquet [--targets-only]` → [ADMIN] Regenerar parquet com PIT fundamentais + alpha90d\n"
            "`/admin_load_models <url>` → [ADMIN] Carregar pickles novos para /data/\n"
            "`/admin_retrain [dry-run]` → [ADMIN] Disparar retrain ad-hoc\n"
            "`/admin_dip_dataset`       → [ADMIN] Construir dataset histórico DipRadar (desde 1993)\n"
            "`/admin_build_core_dataset` → [ADMIN] Constrói dataset CoreML (57 compounders, desde 1993) + commit GitHub\n"
            "`/admin_full_pipeline`     → [ADMIN] Pipeline completo (parquets + retrain)\n"
            "`/health`                  → Dashboard observabilidade\n"
            "`/health errors`           → Log de erros críticos\n"
            "`/language <en|pt>`        → Definir idioma (default: pt)\n"
            "`/help`                    → Esta mensagem"
        )

    elif cmd == "/language":
        if not _check_rate(cmd_key): return
        # Derive chat_id from the current update context (best-effort; fall back to 0)
        _chat_id: int = int(_poll_context.get("_current_chat_id", 0))
        _lang_now = get_language(_chat_id)
        if len(parts) < 2:
            avail = ", ".join(f"`{l}`" for l in SUPPORTED_LANGUAGES)
            _reply(
                f"*Idioma actual:* `{_lang_now}`\n"
                f"Uso: `/language <{' | '.join(SUPPORTED_LANGUAGES)}>`\n"
                f"Idiomas disponíveis: {avail}"
            )
        else:
            _requested = parts[1].lower().strip()
            if set_language(_chat_id, _requested):
                _reply(t("language_set", lang=_requested))
            else:
                avail = ", ".join(SUPPORTED_LANGUAGES)
                _reply(t("language_invalid", lang=_requested, available=avail))  # TODO(i18n): lang param needed here; use default

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
            snap      = get_snap()
            total     = snap.get("total_eur", 0)
            pnl_d     = snap.get("pnl_day", 0)
            pnl_w     = snap.get("pnl_week", 0)
            pnl_m     = snap.get("pnl_month", 0)
            fx        = snap.get("usd_eur", 0)
            positions = snap.get("positions", [])
            def _e(v): return "🟢" if v > 0 else ("🔴" if v < 0 else "⚪")
            lines = [
                f"*💼 Carteira — {datetime.now().strftime('%d/%m %H:%M')}*",
                f"_USD/EUR: {fx:.4f}_",
                "",
                f"*Total: €{total:,.2f}*",
                f"  {_e(pnl_d)} Hoje:   €{pnl_d:+,.2f}",
                f"  {_e(pnl_w)} Semana: €{pnl_w:+,.2f}",
                f"  {_e(pnl_m)} Mês:    €{pnl_m:+,.2f}",
                f"  📊 PPR: €{snap.get('ppr_value',0):,.2f}",
            ]
            if positions:
                lines.append("")
                lines.append("*Posições:*")
                for p in sorted(positions, key=lambda x: x.get("value_eur", 0), reverse=True):
                    sym   = p.get("symbol","")
                    val   = p.get("value_eur", 0)
                    pd_   = p.get("pnl_day")
                    pt_   = p.get("pnl_total")
                    pd_s  = f" {_e(pd_)} €{pd_:+.0f}" if pd_ is not None else ""
                    pt_s  = f" | total €{pt_:+.0f}" if pt_ is not None else ""
                    lines.append(f"  *{sym}* €{val:,.0f}{pd_s}{pt_s}")
            _reply("\n".join(lines))
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
                from momentum_radar.scanner import scan_momentum_universe, format_momentum_alert
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

    elif cmd == "/admin_bootstrap":
        # /admin_bootstrap [ano]  — gera dataset histórico desde o ano dado
        # Default: 1993 (máximo cobertura yfinance para S&P 500)
        _year  = parts[1] if len(parts) > 1 and parts[1].isdigit() else "1993"
        _start = f"{_year}-01-01"
        _reply(
            f"*Bootstrap histórico — desde {_year}*\n"
            f"Simula o DipRadar desde {_year}: dot-com, 2008, COVID, 2022...\n"
            f"Notificações a 25%, 50%, 75% e conclusão de cada etapa.\n\n"
            f"Pipeline completo:\n"
            f"  1. Bootstrap (dips históricos) ~30 min\n"
            f"  2. Regen (features completas) ~50 min\n"
            f"  3. Retrain automático ~90 min\n"
            f"  4. Push para GitHub\n\n"
            f"_Total estimado: 3-4 horas. Recebes notificação em cada etapa._"
        )
        def _run_bootstrap():
            import subprocess, sys
            from pathlib import Path as _P
            _log_file = (_P("/data") if _P("/data").exists() else _P("/tmp")) / "bootstrap_progress.log"
            try:
                script = _P(__file__).parent / "scripts" / "build_bootstrap_dataset.py"
                with open(_log_file, "w", encoding="utf-8") as _lf:
                    result = subprocess.run(
                        [sys.executable, str(script), "--start", _start],
                        stdout=_lf, stderr=_lf, timeout=6*3600
                    )
                if result.returncode == 0:
                    # Etapa 2: regen automático (enriquecer com todas as features)
                    _handle_admin_regen_parquet(["admin_regen_parquet"])
                else:
                    try:
                        last = _log_file.read_text(encoding="utf-8", errors="replace")[-500:]
                    except Exception:
                        last = "(sem log)"
                    _reply(f"❌ Bootstrap falhou:\n`{last}`")
            except subprocess.TimeoutExpired:
                _reply("⏰ Bootstrap excedeu 6h.")
            except Exception as e:
                _reply(f"❌ Erro: `{e}`")
        import threading
        threading.Thread(target=_run_bootstrap, daemon=True, name="bootstrap").start()

    elif cmd == "/admin_bootstrap_status":
        from pathlib import Path as _P
        _dd = _P("/data") if _P("/data").exists() else _P("/tmp")
        for _log_name in ["bootstrap_progress.log", "regen_progress.log", "retrain_progress.log"]:
            _log = _dd / _log_name
            if _log.exists():
                try:
                    _txt  = _log.read_text(encoding="utf-8", errors="replace")
                    _last = _txt[-800:].strip()
                    _reply(f"*{_log_name}:*\n```\n{_last}\n```")
                except Exception as e:
                    _reply(f"Erro: {e}")
                return
        _reply("Nenhum processo em curso.")

    elif cmd == "/admin_momentum_bootstrap":
        _year  = parts[1] if len(parts) > 1 and parts[1].isdigit() else "1993"
        _start = f"{_year}-01-01"
        _reply(
            f"*Bootstrap Momentum — desde {_year}*\n"
            f"Detecta dias de breakout em cada ticker desde {_year}.\n"
            f"Target: forward\\_return\\_30d (retorno 30 dias após o breakout)\n\n"
            f"Pipeline automático:\n"
            f"  1. Bootstrap momentum ~30 min\n"
            f"  2. Treino modelo + sector models ~10 min\n\n"
            f"_Notificações automáticas em cada etapa._"
        )
        def _run_mom_bootstrap():
            import subprocess, sys
            from pathlib import Path as _P
            _log = (_P("/data") if _P("/data").exists() else _P("/tmp")) / "bootstrap_progress.log"
            try:
                script = _P(__file__).parent / "momentum_radar" / "scripts" / "build_dataset.py"
                with open(_log, "w", encoding="utf-8") as _lf:
                    result = subprocess.run(
                        [sys.executable, str(script), "--start", _start],
                        stdout=_lf, stderr=_lf, timeout=4*3600
                    )
                if result.returncode == 0:
                    # Pipeline automático: dataset → retrain
                    _dd2 = _P("/data") if _P("/data").exists() else _P("/tmp")
                    from momentum_radar.trainer import run_momentum_training
                    res = run_momentum_training(
                        dataset_path=_dd2 / "momentum_training.parquet",
                        bundle_path =_dd2 / "momentum_model.pkl",
                        report_path =_dd2 / "momentum_report.json",
                    )
                    dec = res.get("decision", "?")
                    ic  = res.get("ic_ensemble", 0)
                    n   = res.get("n_train", 0)
                    ic_str = f"{ic:.4f}" if isinstance(ic, float) else "N/A"
                    _reply(
                        f"*MomentumRadar — {dec}*\n"
                        f"IC: {ic_str} | {n:,} amostras\n"
                        + ("_Usa `/admin\\_promote` para activar._" if dec == "PENDING" else "")
                    )
                else:
                    try: last = _log.read_text(errors="replace")[-400:]
                    except Exception: last = ""
                    _reply(f"❌ Bootstrap momentum falhou:\n`{last}`")
            except subprocess.TimeoutExpired:
                _reply("⏰ Bootstrap momentum excedeu 4h.")
            except Exception as e:
                _reply(f"❌ Erro: `{e}`")
        import threading
        threading.Thread(target=_run_mom_bootstrap, daemon=True, name="momentum-bootstrap").start()

    elif cmd == "/admin_momentum_dataset":
        _handle_admin_momentum_dataset(parts)

    elif cmd == "/admin_dip_dataset":
        _handle_admin_dip_dataset(parts)

    elif cmd in ("/admin_pipeline_status", "/pipeline_status", "/ps"):
        import time as _t
        if _retrain_running or _pipeline_log:
            elapsed = _pipeline_elapsed() if _pipeline_start else "?"
            phase   = _pipeline_phase or "idle"
            running = _retrain_running
            tail    = _pipeline_log[-20:]  # last 20 lines
            lines   = [
                f"Pipeline {'RUNNING' if running else 'last run'} — {elapsed} elapsed",
                f"Phase: {phase}",
                "",
            ]
            lines += [f"  {e}" for e in tail]
            if running:
                lines += [
                    "",
                    "Commands:",
                    "  /admin_pipeline_status  — refresh this log",
                    "  /admin_kill_pipeline    — terminate the current process",
                ]
            _reply("\n".join(lines))
        else:
            _reply("No pipeline running. Use /admin_run_pipeline to start one.")

    elif cmd in ("/admin_kill_pipeline", "/kill_pipeline"):
        import bot_commands as _self
        _proc = _pipeline_proc
        if _proc is None and not _retrain_running:
            _reply("No pipeline process running.")
        else:
            try:
                _phase = _pipeline_phase or "?"
                if _proc is not None:
                    _proc.terminate()
                    _reply(
                        f"Sent SIGTERM to pipeline process (phase: {_phase}).\n"
                        f"The process will stop within seconds.\n"
                        f"Partial parquet data may be incomplete — use --force-rebuild next time."
                    )
                else:
                    _reply("Pipeline thread running but no subprocess found. Forcing flag reset.")
                _self._retrain_running = False
                _self._pipeline_phase  = ""
                _self._pipeline_proc   = None
            except Exception as e:
                _reply(f"Kill failed: {e}")

    elif cmd in ("/admin_clear_price_cache", "/clear_price_cache"):
        from pathlib import Path as _P
        _dd = _P("/data") if _P("/data").exists() else _P("/tmp")
        _cache = _dd / "price_cache"
        if not _cache.exists():
            _reply("Price cache directory does not exist — nothing to clear.")
        else:
            _deleted = 0
            _skipped = 0
            for _f in _cache.glob("*.parquet"):
                if _f.name.startswith("edgar_"):
                    _skipped += 1
                    continue
                try:
                    _f.unlink()
                    _deleted += 1
                except Exception:
                    pass
            _reply(
                f"Price cache cleared: {_deleted} files deleted, {_skipped} EDGAR files preserved.\n"
                f"Next dataset build will re-download all price histories from yfinance.\n"
                f"Run /admin_run_pipeline --dip --dataset to rebuild."
            )

    elif cmd == "/admin_disk":
        _handle_admin_disk(parts)

    elif cmd == "/admin_regen_status":
        from pathlib import Path as _P
        _dd = _P("/data") if _P("/data").exists() else _P("/tmp")
        _log = _dd / "regen_progress.log"
        if not _log.exists():
            _reply("Nenhum regen em curso ou log não encontrado.")
        else:
            try:
                _txt = _log.read_text(encoding="utf-8", errors="replace")
                _last = _txt[-1200:].strip()
                _reply(f"*Regen log (últimas linhas):*\n```\n{_last}\n```")
            except Exception as e:
                _reply(f"Erro ao ler log: {e}")

    elif cmd == "/admin_pending_info":
        from pathlib import Path as _P
        import json as _json
        _dd = _P("/data") if _P("/data").exists() else _P("/tmp")
        _pending = _dd / "dip_models_pending.pkl"
        _report  = _dd / "ml_report_pending.json"
        if not _pending.exists():
            _reply("Nenhum modelo pending. Corre `/admin\\_retrain` primeiro.")
        else:
            try:
                lines = ["*Modelo Pending — Relatório Completo*", ""]
                if _report.exists():
                    r = _json.loads(_report.read_text())
                    m = r.get("metrics", {})
                    def _fmt(v, spec): return format(v, spec) if isinstance(v, (int, float)) else "N/A"
                    lines += [
                        f"  IC (ρ_α):   *{_fmt(m.get('rho_alpha_mean'), '.4f')}*",
                        f"  Brier OOF:  *{_fmt(m.get('brier_oof'), '.4f')}*",
                        f"  Top-K PnL:  *{_fmt(m.get('topk_pnl_mean'), '.4f')}*",
                        f"  Win rate:   *{_fmt(m.get('win_rate_alpha'), '.1%')}*",
                        f"  Champion:   {r.get('champion', '?')}",
                        f"  Features:   {r.get('n_features', '?')}",
                        f"  Treino:     {r.get('n_train', '?'):,} amostras",
                        "",
                    ]
                    # Sector models
                    sector_m = r.get("sector_models", {})
                    if sector_m:
                        lines.append("*Modelos sectoriais:*")
                        for sname, sinfo in sorted(sector_m.items()):
                            ic   = sinfo.get("ic_holdout")
                            n    = sinfo.get("n_train", 0)
                            ic_s = f"{ic:.4f}" if isinstance(ic, (int, float)) else "N/A"
                            ok   = "✅" if isinstance(ic, (int, float)) and ic >= 0.02 else "⚠️"
                            lines.append(f"  {ok} {sname}: IC={ic_s} (n={n:,})")
                        lines.append("")
                    else:
                        lines.append("_Nenhum sector model aceite no gating (IC < -0.02)._")
                        lines.append("")
                    lines.append("Usa `/admin\\_promote` para colocar em produção.")
                else:
                    lines.append(f"Modelo pending existe ({_pending.stat().st_size/1e6:.1f}MB) mas sem report.")
                    lines.append("Usa `/admin\\_promote` para promover.")
                _reply("\n".join(lines))
            except Exception as e:
                _reply(f"Erro: `{e}`")

    elif cmd == "/admin_promote":
        # Promove o modelo pending para produção
        from pathlib import Path as _P
        _dd = _P("/data") if _P("/data").exists() else _P("/tmp")
        _pending = _dd / "dip_models_pending.pkl"
        _prod    = _dd / "dip_models.pkl"
        _report_pending = _dd / "ml_report_pending.json"
        _report_prod    = _dd / "ml_report.json"
        if not _pending.exists():
            _reply("Nenhum modelo pending encontrado. Corre `/admin\\_retrain` primeiro.")
        else:
            try:
                import shutil, json
                # Arquivar modelo actual se existir
                if _prod.exists():
                    shutil.copy2(str(_prod), str(_dd / "dip_models_archived.pkl"))
                # Promover pending
                shutil.copy2(str(_pending), str(_prod))
                _pending.unlink()
                if _report_pending.exists():
                    shutil.copy2(str(_report_pending), str(_report_prod))
                    _report_pending.unlink()
                # Forçar reload do bundle em memória
                try:
                    import ml_predictor as _mp
                    _mp._bundle = None
                    _mp._load_bundle(force=True)
                except Exception:
                    pass
                # Ler IC do report
                _ic_str = ""
                try:
                    _rdata = json.loads(_report_prod.read_text()) if _report_prod.exists() else {}
                    _ic = _rdata.get("metrics", {}).get("rho_alpha_mean")
                    if _ic: _ic_str = f"\n  IC: {_ic:.4f}"
                except Exception:
                    pass
                _reply(
                    f"*Modelo promovido para producao!*{_ic_str}\n"
                    f"Modelo anterior arquivado em `dip\\_models\\_archived.pkl`.\n"
                    f"O proximo `/analisar` ja usa o novo modelo."
                )
            except Exception as e:
                _reply(f"Erro ao promover: `{e}`")

    elif cmd == "/admin_retrain_status":
        from pathlib import Path as _P
        _dd = _P("/data") if _P("/data").exists() else _P("/tmp")
        _log = _dd / "retrain_progress.log"
        if not _log.exists():
            _reply("Nenhum retrain em curso ou log não encontrado. Usa /admin\\_retrain para iniciar.")
        else:
            try:
                _txt = _log.read_text(encoding="utf-8", errors="replace")
                _last = _txt[-1200:].strip()
                _reply(f"*Retrain log (últimas linhas):*\n```\n{_last}\n```")
            except Exception as e:
                _reply(f"Erro ao ler log: {e}")

    elif cmd == "/admin_momentum_retrain":
        _handle_admin_momentum_retrain(parts)

    elif cmd in ("/admin_build_core_dataset", "/admin_core_dataset"):
        _handle_admin_build_core_dataset(parts)

    elif cmd in ("/admin_train_core_model", "/admin_core_train"):
        _handle_admin_train_core_model(parts)

    elif cmd == "/admin_full_pipeline":
        _handle_admin_full_pipeline(parts)

    elif cmd == "/admin_run_pipeline":
        _handle_admin_run_pipeline(parts)

    elif cmd in ("/admin_refresh_core_universe", "/core_refresh_universe"):
        dry = "--dry-run" in parts
        _reply(
            f"CORE universe refresh starting{'(dry-run)' if dry else ''}.\n"
            f"Screening full US universe (~3,500 tickers) for quality compounders.\n"
            f"Criteria: gross_margin>=35%, 5y_rev_growth>=5%, market_cap>=$5B "
            f"OR Dividend Aristocrat.\n"
            f"Duration: ~30-60 min. Use /ps to follow progress."
        )
        def _run_refresh():
            try:
                import subprocess, sys
                from pathlib import Path as _P
                script = _P(__file__).parent / "scripts" / "refresh_core_universe.py"
                cmd_args = [sys.executable, str(script)]
                if dry: cmd_args.append("--dry-run")
                for p in parts[1:]:
                    if p.startswith("--min"): cmd_args.append(p)
                rc, lines = _run_script_streaming(cmd_args, "CORE universe refresh", timeout_s=7200)
                # Parse result summary from last lines
                summary_lines = [l for l in lines[-30:] if any(k in l for k in ("Added:", "Removed:", "Kept:", "Skipped:", "Errors:", "DRY RUN"))]
                _reply(
                    f"CORE universe refresh {'dry-run ' if dry else ''}complete (exit {rc}).\n"
                    + "\n".join(summary_lines) +
                    f"\nUse /core_universe to see the updated list."
                )
            except Exception as e:
                _reply(f"CORE refresh error: {e}")
        threading.Thread(target=_run_refresh, daemon=True, name="core-refresh").start()

    elif cmd == "/admin_publish_charts":
        has_github = bool(os.environ.get("GITHUB_TOKEN") and os.environ.get("GITHUB_CHARTS_REPO"))
        if has_github:
            _reply("Generating performance charts + DRIP ledger and publishing to GitHub…")
        else:
            _reply(
                "GITHUB_TOKEN or GITHUB_CHARTS_REPO not set — skipping GitHub upload.\n"
                "Generating charts + DRIP ledger and sending here via Telegram…"
            )
        def _run_publish():
            try:
                from chart_generator import generate_all_charts
                chart_paths = generate_all_charts()
                if not chart_paths:
                    _reply("No files generated — no closed trades yet.")
                    return

                # Step 1: always send charts as Telegram photos + DRIP as document
                png_paths = [p for p in chart_paths if p.suffix == ".png"]
                md_paths  = [p for p in chart_paths if p.suffix == ".md"]

                for png in png_paths:
                    try:
                        _send_photo(png.read_bytes(), caption=png.stem.replace("_", " ").title())
                    except Exception as e:
                        logging.warning("[publish_charts] photo send %s: %s", png.name, e)

                for md in md_paths:
                    try:
                        _send_document(
                            md.read_bytes(),
                            filename=md.name,
                            caption="DipRadar DRIP Ledger — full dividend audit trail (gross, pre-tax)",
                        )
                    except Exception as e:
                        logging.warning("[publish_charts] doc send %s: %s", md.name, e)

                # Step 2: push to GitHub if configured
                if has_github:
                    from github_publisher import push_all_charts
                    results = push_all_charts()
                    charts_ok  = [(n, u) for n, u in results.items() if u and not u.startswith("ERROR") and not n.endswith(".md")]
                    reports_ok = [(n, u) for n, u in results.items() if u and not u.startswith("ERROR") and n.endswith(".md")]
                    errors     = [(n, u) for n, u in results.items() if not u or u.startswith("ERROR")]

                    lines = ["", "GitHub URLs:"]
                    for name, url in charts_ok:
                        lines.append(f"  {name}: {url}")
                    if reports_ok:
                        lines.append("")
                        lines.append("DRIP ledger (share with clients):")
                        for name, url in reports_ok:
                            lines.append(f"  {name}: {url}")
                    if errors:
                        lines.append("")
                        lines.append("Errors:")
                        for name, url in errors:
                            lines.append(f"  {name}: {url or 'skipped'}")
                    _reply("\n".join(lines))
                else:
                    _reply(
                        "Charts and DRIP ledger sent above.\n"
                        "To publish permanently to GitHub, set GITHUB_TOKEN and GITHUB_CHARTS_REPO in Railway env vars."
                    )
            except Exception as e:
                _reply(f"Error publishing: {e}")
        import threading
        threading.Thread(target=_run_publish, daemon=True, name="publish-charts").start()

    elif cmd == "/admin_set_floor":
        _handle_admin_set_floor(parts)

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

    elif cmd in ("/core", "/core_holdings"):
        if not _check_rate(cmd_key): return
        _handle_core(parts)

    elif cmd in ("/core_add", "/add_core"):
        _handle_core_add(parts)

    elif cmd in ("/core_close", "/close_core"):
        _handle_core_close(parts)

    elif cmd in ("/core_universe",):
        if not _check_rate(cmd_key): return
        _handle_core_universe(parts)

    elif cmd in ("/core_universe_add", "/add_core_universe"):
        _handle_core_universe_add(parts)

    elif cmd in ("/core_universe_remove", "/remove_core_universe"):
        _handle_core_universe_remove(parts)

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
                        _poll_context["_current_chat_id"] = int(chat) if chat.lstrip("-").isdigit() else 0
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
