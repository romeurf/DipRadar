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
    """/sync_portfolio — bulk-import existing holdings into the Telegram ledger.

    Use case: you already own positions (from before the bot, or from a broker
    export). /buy registers AND debits liquidity (purchase semantics). This
    command registers WITHOUT touching liquidity — for migrating existing
    holdings.

    Uso:
      /sync_portfolio                                    → ver estado actual
      /sync_portfolio NVO:25:85.50 MSFT:12:410          → importar várias
      /sync_portfolio NVO:25:85.50 --no-conviction      → não criar convicção

    Formato: TICKER:shares:preço_médio_usd

    Cada ticker importado:
      1. Vai para _cache["positions"] via portfolio.register_existing_position
         (sem afectar liquidez — assume-se que já é tua há tempo)
      2. Auto-cria convicção via conviction_radar (excepto ETFs/PPR/foreign
         e --no-conviction). Tipo auto-classificado por ML.

    Idempotente: tickers já no ledger são saltados (não sobrescrevem).
    Para alterar preço médio: /sell + /buy, ou /conviction transition.

    As env vars HOLDING_* foram DESCONTINUADAS — usa este comando ou /buy.
    """
    from portfolio import get_positions, register_existing_position

    no_conviction = "--no-conviction" in parts
    entries = [p for p in parts[1:] if p != "--no-conviction"]

    # Sem argumentos → mostrar estado actual
    if not entries:
        positions = get_positions()
        if not positions:
            _reply(
                "💼 *Carteira vazia.*\n\n"
                "Para bulk-import (sem afectar liquidez):\n"
                "`/sync_portfolio NVO:25:85.50 MSFT:12:410 UBER:30:65`\n\n"
                "Para comprar (debita liquidez):\n"
                "`/buy TICKER PREÇO SHARES`\n\n"
                "_Formato bulk: TICKER:shares:preço_médio_usd_"
            )
            return
        lines = [f"💼 *Carteira actual* ({len(positions)} posições)"]
        for sym in sorted(positions.keys()):
            pos = positions[sym]
            lines.append(
                f"  • *{sym}*: {pos.get('shares', 0):g} @ ${pos.get('avg_price', 0):.2f}"
                f" _({pos.get('category', '?')})_"
            )
        lines += [
            "",
            "_Para adicionar mais: `/sync_portfolio TICK:shares:preço ...`_",
            "_Para remover: `/sell TICKER PREÇO`_",
        ]
        _reply("\n".join(lines))
        return

    # Parse entries
    parsed: list[tuple[str, float, float]] = []
    errors: list[str] = []
    for entry in entries:
        try:
            bits = entry.split(":")
            if len(bits) < 3:
                errors.append(f"`{entry}` — falta preço (TICKER:shares:preço)")
                continue
            ticker = bits[0].upper().strip()
            shares = float(bits[1])
            price  = float(bits[2])
            if shares <= 0 or price <= 0:
                errors.append(f"`{entry}` — shares e preço têm de ser > 0")
                continue
            parsed.append((ticker, shares, price))
        except (ValueError, IndexError):
            errors.append(f"`{entry}` — erro a parsear")

    if errors and not parsed:
        _reply("⚠️ *Sem entradas válidas:*\n" + "\n".join(f"  • {e}" for e in errors))
        return

    imported: list[tuple[str, float, float]] = []
    already:  list[str] = []
    convictions_created: list[tuple[str, str]] = []
    conviction_errors:   list[tuple[str, str]] = []

    for ticker, shares, price in parsed:
        try:
            result = register_existing_position(ticker, price, shares)
            if result["action"] == "imported":
                imported.append((ticker, shares, price))
            else:
                already.append(ticker)
        except Exception as e:
            errors.append(f"`{ticker}` — registo falhou: {e}")
            continue

        # Auto-create conviction (unless --no-conviction or deny-list)
        if no_conviction:
            continue
        try:
            import conviction_radar as cr
            deny = cr._get_deny_list()
            if ticker in deny or "." in ticker:
                continue
            if cr.get_by_ticker(ticker):
                continue  # already a conviction
            current = cr._fetch_price(ticker, fallback=price)
            hist    = cr._fetch_price_history(ticker, period="6mo")
            auto_type = cr._classify_type_from_signals(ticker, current, hist)
            cr.add_conviction(
                ticker, price, shares, auto_type,
                trailing_high = current if auto_type == "MOMENTUM" else None,
                note          = "imported via /sync_portfolio",
            )
            convictions_created.append((ticker, auto_type))
        except Exception as e:
            conviction_errors.append((ticker, str(e)))

    # Build reply
    lines = ["💼 *Bulk-import resultado*"]
    if imported:
        lines.append(f"\n✅ *{len(imported)} posições importadas* (sem alteração de liquidez):")
        for t, s, p in imported:
            lines.append(f"  • *{t}*: {s:g} @ ${p:.2f}")
    if already:
        lines.append(f"\n♻️ *{len(already)} já no ledger* (saltadas):")
        lines.append("  " + ", ".join(already))
    if convictions_created:
        lines.append(f"\n🧠 *{len(convictions_created)} convicções auto-criadas:*")
        for t, tp in convictions_created:
            lines.append(f"  • {t} → {tp}")
    if conviction_errors:
        lines.append(f"\n⚠️ *Convicção falhou para:*")
        for t, e in conviction_errors:
            lines.append(f"  • {t}: _{e}_")
    if errors:
        lines.append(f"\n❌ *Erros:*")
        for e in errors:
            lines.append(f"  • {e}")

    lines += [
        "",
        "_Vê com `/portfolio` ou `/conviction list`._",
        "_Faz `/conviction transition TICKER CORE` para os holds de longo prazo._",
    ]
    _reply("\n".join(lines))


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


# ── /admin_features ──────────────────────────────────────────────────────────

def _handle_admin_features(parts: list[str]) -> None:
    """Print per-feature stats for a model's training parquet.

    Usage:
      /admin_features dip       — DIP parquet feature stats
      /admin_features core      — Core parquet feature stats
      /admin_features momentum  — Momentum parquet feature stats

    Output (one line per feature):
      col_name  n=<rows>  mean=<float>  std=<float>  nan=<count>
    """
    from pathlib import Path as _P
    import pandas as _pd

    model = (parts[1].lower() if len(parts) > 1 else "").strip()
    if model not in ("dip", "core", "momentum"):
        _reply(
            "<b>/admin_features [dip|core|momentum]</b>\n\n"
            "Pick one model to inspect its training parquet feature stats:\n"
            "  <code>/admin_features dip</code>\n"
            "  <code>/admin_features core</code>\n"
            "  <code>/admin_features momentum</code>"
        )
        return

    _dd = _P("/data") if _P("/data").exists() else _P("/tmp")
    _repo = _P(__file__).parent

    if model == "dip":
        path = _dd / "ml_training_base.parquet"
        if not path.exists():
            path = _repo / "ml_training_base.parquet"
        try:
            from ml_features import get_feature_columns
            feat_cols = get_feature_columns("dip_us")
        except Exception:
            feat_cols = None
        title = "DIP (dip_us)"
    elif model == "core":
        path = _dd / "core_training_base.parquet"
        if not path.exists():
            path = _repo / "core_training_base.parquet"
        # The Core trainer uses scripts/train_core_model.py:CORE_FEATURE_COLS,
        # NOT ml_features.FEATURE_SETS["core_us"] (which is the aspirational
        # registry shape, not what the trainer actually consumes). Match the
        # trainer's truth.
        try:
            import sys
            _scripts_path = _repo / "scripts"
            if str(_scripts_path) not in sys.path:
                sys.path.insert(0, str(_scripts_path))
            from train_core_model import CORE_FEATURE_COLS as feat_cols
        except Exception:
            feat_cols = None
        title = "CORE (core_us)"
    else:  # momentum
        path = _dd / "momentum_training.parquet"
        if not path.exists():
            path = _repo / "momentum_radar" / "momentum_training.parquet"
        try:
            from momentum_radar.trainer import FEATURE_COLS as feat_cols
        except Exception:
            feat_cols = None
        title = "Momentum (momentum_us)"

    if not path.exists():
        _reply(f"<b>{title}</b>\n\nParquet not found.\nExpected: <code>{path}</code>")
        return

    try:
        df = _pd.read_parquet(path)
    except Exception as exc:
        _reply(f"<b>{title}</b>\n\nFailed to read parquet:\n<code>{exc}</code>")
        return

    n_total = len(df)
    lines = [
        f"<b>{title}</b>",
        f"<code>{path}</code>",
        f"{n_total:,} rows total",
        "",
    ]

    # Targets (always show first if present)
    target_cols = ["alpha_30d", "alpha_90d", "alpha_180d",
                   "alpha_1y", "alpha_2y",
                   "forward_return_30d", "forward_return_90d", "forward_return_180d",
                   "max_drawdown_60d", "max_drawdown_90d"]
    targets_present = [c for c in target_cols if c in df.columns]

    # Features
    if feat_cols is None:
        # No registry — show everything except meta columns
        meta = {"ticker", "alert_date", "date", "snapshot_date", "sector", "price",
                "_source", "source", "label_win"}
        feats_present = [c for c in df.columns if c not in meta and c not in target_cols]
    else:
        feats_present = list(feat_cols)

    def _stat_line(col: str) -> str:
        if col not in df.columns:
            return f"  {col:30s} — MISSING"
        s = _pd.to_numeric(df[col], errors="coerce")
        n = int(s.notna().sum())
        nan = int(s.isna().sum())
        if n == 0:
            return f"  {col:30s} — NO DATA (nan={nan})"
        return (
            f"  {col:30s} n={n:>6} mean={s.mean():>9.4f} "
            f"std={s.std():>8.4f} nan={nan}"
        )

    if targets_present:
        lines.append("<b>Targets:</b>")
        for c in targets_present:
            lines.append(_stat_line(c))
        lines.append("")

    lines.append(f"<b>Features ({len(feats_present)}):</b>")
    missing = []
    for c in feats_present:
        if feat_cols is not None and c not in df.columns:
            missing.append(c)
        lines.append(_stat_line(c))

    if missing:
        lines.append("")
        lines.append(f"<b>⚠️ MISSING features ({len(missing)}):</b> {', '.join(missing)}")

    # Telegram has a 4096 char limit per message — chunk if needed
    body = "\n".join(lines)
    if len(body) <= 3900:
        _reply(f"<pre>{body}</pre>")
        return
    # Chunk by lines, keep each msg under 3900 chars
    chunk: list[str] = []
    size = 0
    for ln in lines:
        if size + len(ln) + 1 > 3900:
            _reply("<pre>" + "\n".join(chunk) + "</pre>")
            chunk = []
            size = 0
        chunk.append(ln)
        size += len(ln) + 1
    if chunk:
        _reply("<pre>" + "\n".join(chunk) + "</pre>")


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
            lines.append("  DIP (dip_us): NOT trained — run /admin_retrain")
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
            lines.append("  Momentum (momentum_us): NOT trained — run /admin_momentum_retrain")
    except Exception as e:
        lines.append(f"  Momentum (momentum_us): error checking — {e}")

    # CORE model — has ML scoring (champion-of GBR vs Ridge on alpha_1y) gated
    # by CV IC ≥ core_radar._MIN_CV_IC_FOR_USE. Below that, falls back to
    # rule-based scoring even if the pkl is on disk.
    try:
        import core_radar as _cr
        if _cr.is_model_ready():
            b = _cr._core_model_bundle
            cv_ic = float(getattr(b, "cv_ic", 0.0))
            champion = str(getattr(b, "champion_name", "GBR"))
            lines.append(
                f"  CORE (core_us): ML active | champion={champion} | "
                f"n_train={b.n_train} | CV_IC={cv_ic:+.4f} | "
                f"CV_R²={b.cv_r2:+.3f} | alpha_p50={b.alpha_p50:.3f} | "
                f"train_date={b.train_date}"
            )
        elif _cr._core_model_bundle is not None:
            # Loaded but failed quality gate.
            b = _cr._core_model_bundle
            cv_ic = float(getattr(b, "cv_ic", 0.0))
            champion = str(getattr(b, "champion_name", "GBR"))
            lines.append(
                f"  CORE (core_us): rule-based fallback "
                f"(ML pkl loaded — champion={champion}, CV_IC={cv_ic:+.4f} < "
                f"{_cr._MIN_CV_IC_FOR_USE} floor)"
            )
        else:
            lines.append(
                "  CORE (core_us): rule-based fallback "
                "(ML pkl missing — run /admin_train_core_model)"
            )
    except Exception as e:
        lines.append(f"  CORE (core_us): error checking — {e}")

    # ── Training datasets — verify all 3 parquets are complete ───────────────
    lines.append("")
    lines.append("Training datasets:")

    from pathlib import Path as _Pp
    import pandas as _pdp
    _data_p = _Pp("/data") if _Pp("/data").exists() else _Pp("/tmp")

    def _parquet_stats(label: str, primary: _Pp, repo_fallback: _Pp | None = None) -> str:
        """Return a one-line summary of a parquet's size, rows, tickers, date range."""
        path = primary if primary.exists() else (repo_fallback if repo_fallback and repo_fallback.exists() else None)
        if path is None:
            return f"  {label}: MISSING"
        try:
            size_mb = path.stat().st_size / 1e6
            df = _pdp.read_parquet(path)
            n_rows  = len(df)
            n_tick  = df["ticker"].nunique() if "ticker" in df.columns else "?"
            # Date column may be "alert_date" (DIP), "date" (Momentum), or "snapshot_date" (Core)
            date_col = next((c for c in ("alert_date", "date", "snapshot_date") if c in df.columns), None)
            if date_col is not None:
                _dates = _pdp.to_datetime(df[date_col], errors="coerce").dropna()
                if not _dates.empty:
                    d_first = _dates.min().strftime("%Y-%m-%d")
                    d_last  = _dates.max().strftime("%Y-%m-%d")
                    date_str = f" | {d_first} → {d_last}"
                else:
                    date_str = ""
            else:
                date_str = ""
            return (
                f"  {label}: {n_rows:,} rows | {n_tick} tickers | "
                f"{size_mb:.1f} MB{date_str} | {path.name}"
            )
        except Exception as exc:
            return f"  {label}: error reading {path.name} — {exc}"

    lines.append(_parquet_stats(
        "DIP      (ml_training_base.parquet)",
        _data_p / "ml_training_base.parquet",
        _Pp(__file__).parent / "ml_training_base.parquet",
    ))
    lines.append(_parquet_stats(
        "CORE     (core_training_base.parquet)",
        _data_p / "core_training_base.parquet",
        _Pp(__file__).parent / "core_training_base.parquet",
    ))
    lines.append(_parquet_stats(
        "Momentum (momentum_training.parquet)",
        _data_p / "momentum_training.parquet",
        _Pp(__file__).parent / "momentum_radar" / "momentum_training.parquet",
    ))

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
    """/admin_disk          → disk usage breakdown for /data/
    /admin_disk clean     → delete old EDGAR cache (>30d) and prune snapshot
    /admin_disk purge     → delete ALL cache and training parquets — keep only
                            alert_db.csv and model pickles (use before overnight rebuild)
    """
    import os, time, shutil
    from pathlib import Path

    data_dir = Path("/data") if Path("/data").exists() else Path("/tmp")
    subcmd   = parts[1].lower() if len(parts) > 1 else ""
    clean    = subcmd == "clean"
    purge    = subcmd == "purge"

    def _human(b: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if b < 1024:
                return f"{b:.1f} {unit}"
            b /= 1024
        return f"{b:.1f} TB"

    def _dir_size(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0

    if purge:
        # Delete all non-essential data to free disk for the overnight rebuild.
        # NEVER deleted: position/state files, alert databases, backups/ dir.
        # Model pickles ARE deleted — the overnight rebuild retrains all 3.
        KEEP_FILES = {
            # Positions / trades (irreplaceable without manual entry)
            "alert_db.csv",
            "core_alert_db.csv",
            "paper_trades.json",
            "positions.json",
            "persistent_dips.json",
            "portfolio_override.json",
            # Bot state (session continuity)
            "_dipr_alerts.json",
            "_dipr_recovery.json",
            "_dipr_watchlist.json",
            "_dipr_dip_streaks.json",
            "_dipr_weekly.json",
            "_dipr_backtest.json",
            "_dipr_rejected.json",
            "alerted_today.json",
            # Logs (historical record)
            "backtest_log.json",
            "hist_backtest.csv",
            "rejected_log.json",
            "weekly_log.json",
            "ml_predictions.csv",
        }
        KEEP_DIRS  = {"backups"}   # /data/backups/*.zip — daily backups, always preserved
        freed = 0
        deleted_dirs: list[str] = []
        deleted_files: list[str] = []

        # 1. Nuke price_cache/ (OHLCV parquets + EDGAR json + fundamental_cache subdir)
        price_cache_dir = data_dir / "price_cache"
        if price_cache_dir.exists():
            sz = sum(f.stat().st_size for f in price_cache_dir.rglob("*") if f.is_file())
            freed += sz
            shutil.rmtree(str(price_cache_dir), ignore_errors=True)
            deleted_dirs.append(f"price_cache/ ({_human(sz)})")

        # 2. Delete training parquets, model pickles, reports — will be rebuilt
        for name in (
            "ml_training_base.parquet",
            "core_training_base.parquet",
            "momentum_training.parquet",
            "ml_momentum_training.parquet",
            "ml_training_input.parquet",
            "dip_models.pkl",
            "core_us_model.pkl",
            "momentum_model.pkl",
            "momentum_report.json",
            "ml_report.json",
        ):
            p = data_dir / name
            if p.exists():
                sz = p.stat().st_size
                freed += sz
                p.unlink()
                deleted_files.append(f"{name} ({_human(sz)})")

        # 3. Any other top-level file not in KEEP_FILES (directories are skipped)
        for p in data_dir.iterdir():
            if p.is_dir():
                continue  # never touch subdirectories (preserves backups/)
            if p.name in KEEP_FILES:
                continue
            if p.suffix in (".parquet", ".json", ".csv", ".tmp", ".pkl"):
                sz = p.stat().st_size
                freed += sz
                try:
                    p.unlink()
                    deleted_files.append(f"{p.name} ({_human(sz)})")
                except Exception:
                    pass

        backup_count = len(list((data_dir / "backups").glob("*.zip"))) if (data_dir / "backups").exists() else 0
        lines = [
            "<b>Disk purge complete</b>",
            f"Freed: <b>{_human(freed)}</b>",
            "",
        ]
        if deleted_dirs:
            lines += ["Directories deleted:"] + [f"  • {d}" for d in deleted_dirs]
        if deleted_files:
            lines += ["Files deleted:"] + [f"  • {f}" for f in deleted_files]
        lines += [
            "",
            f"✅ Kept: <code>alert_db.csv</code> (open positions)",
            f"✅ Kept: <code>backups/</code> — {backup_count} daily ZIP(s) untouched",
            "Model pickles deleted — overnight will retrain all 3.",
            "",
            "Ready for <code>/admin_overnight</code>",
        ]
        _reply("\n".join(lines))
        return

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

    # Disk usage breakdown
    def _sz(name: str) -> int:
        p = data_dir / name
        return p.stat().st_size if p.exists() else 0

    backup_dir   = data_dir / "backups"
    backup_count = len(list(backup_dir.glob("*.zip"))) if backup_dir.exists() else 0
    items = [
        ("price_cache/ (total)",         _dir_size(data_dir / "price_cache")),
        ("price_cache/fundamental_cache", _dir_size(data_dir / "price_cache" / "fundamental_cache")),
        ("price_cache/edgar_*.json",     sum(f.stat().st_size for f in (data_dir / "price_cache").glob("edgar_*.json")) if (data_dir / "price_cache").exists() else 0),
        (f"backups/ ({backup_count} ZIPs)", _dir_size(backup_dir)),
        ("ml_training_base.parquet",     _sz("ml_training_base.parquet")),
        ("core_training_base.parquet",   _sz("core_training_base.parquet")),
        ("momentum_training.parquet",    _sz("momentum_training.parquet")),
        ("dip_models.pkl",               _sz("dip_models.pkl")),
        ("core_us_model.pkl",            _sz("core_us_model.pkl")),
        ("momentum_model.pkl",           _sz("momentum_model.pkl")),
        ("alert_db.csv",                 _sz("alert_db.csv")),
    ]
    total = _dir_size(data_dir)

    lines = ["<b>Disk usage — /data/</b>", ""]
    for name, size in sorted(items, key=lambda x: x[1], reverse=True):
        if size > 0:
            lines.append(f"  <code>{name}</code>: {_human(size)}")
    lines += [
        "",
        f"<b>Total: {_human(total)}</b>",
        "",
        "<code>/admin_disk clean</code> — delete old EDGAR cache (>30d) + prune snapshot",
        "<code>/admin_disk purge</code> — nuke ALL cache/parquets, keep alert_db + pickles",
    ]
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

            # Passo 1: actualizar dataset (só últimos 35 dias — incremental).
            # 12h timeout — momentum dataset now queries EDGAR per episode for
            # the 4 insider features (commit 0c4703c), which dramatically
            # slowed the bulk build. Incremental updates are still fast.
            script = Path(__file__).parent / "momentum_radar" / "scripts" / "build_dataset.py"
            result = subprocess.run(
                [sys.executable, str(script), "--out", str(ds_path)],
                capture_output=True, text=True, timeout=12 * 3600
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

            # 12h timeout: with the sec_guard fix, EDGAR is now called for every
            # US ticker (1500+). Even with the per-ticker submissions cache,
            # each filing requires XML parsing — the bulk historical run is
            # significantly slower than the old "skip-EDGAR" path. 12h leaves
            # margin over the observed 6h-baseline + EDGAR overhead.
            _TIMEOUT = 12 * 3600
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
      - core_universe.json × histórico desde 1993
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
        "core_universe.json × histórico desde 1993\n"
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
                from monthly_retrain import run_monthly_retrain
                dip_train_res = run_monthly_retrain() or {}
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


# ── /admin_overnight ──────────────────────────────────────────────────────────

def _handle_admin_overnight(parts: list[str]) -> None:
    """/admin_overnight [flags]

    Full sequential rebuild of all 3 models — designed to run overnight.

    Steps:
      0. Purge — CONSERVATIVE by default: only deletes parquets (which
         will be rebuilt anyway). Preserves price_cache, EDGAR cache,
         and pickle files so:
         - yfinance doesn't re-download 30+ years × 1500 tickers (saves 4-6h)
         - EDGAR cache stays warm (saves significant per-episode time)
         - Existing pickles stay loaded if retrain fails (no model loss)
      1. DIP dataset build  (scripts/build_dip_dataset.py --force-rebuild)
      2. DIP model retrain  (monthly_retrain.run_monthly_retrain)
      3. Core dataset build (scripts/build_core_dataset.py --force-rebuild)
      4. Core model train   (scripts/train_core_model.py)
      5. Momentum dataset   (momentum_radar/scripts/build_dataset.py)
      6. Momentum retrain   (momentum_radar.trainer.run_momentum_training)

    Flags:
      --no-purge       Skip the disk purge step entirely
      --purge-cache    ALSO wipe price_cache (forces full yfinance re-download)
                       Use when you suspect cache corruption or want hard reset.
      --purge-models   ALSO wipe .pkl files (forces fallback if retrain fails)
                       DANGEROUS — bot loses ML models if pipeline doesn't
                       complete. Only use if you have a backup or are willing
                       to drop back to rule-based.
      --purge-all      Equivalent to --purge-cache --purge-models (legacy
                       behaviour of older /admin_overnight runs).
      --dip-only       Only DIP build + retrain
      --core-only      Only Core build + retrain
      --momentum-only  Only Momentum build + retrain

    The bot stays fully responsive throughout. Use /admin_pipeline_status to monitor.
    Each step sends a Telegram update on start and completion.
    """
    global _retrain_running
    if _retrain_running:
        _reply("⚠️ Pipeline already running. Wait for it to finish.")
        return

    no_purge       = "--no-purge"       in parts
    purge_cache    = "--purge-cache"    in parts or "--purge-all" in parts
    purge_models   = "--purge-models"   in parts or "--purge-all" in parts
    dip_only       = "--dip-only"       in parts
    core_only      = "--core-only"      in parts
    momentum_only  = "--momentum-only"  in parts

    run_dip      = not core_only and not momentum_only
    run_core     = not dip_only  and not momentum_only
    run_momentum = not dip_only  and not core_only

    scope_parts = []
    if run_dip:      scope_parts.append("DIP")
    if run_core:     scope_parts.append("Core")
    if run_momentum: scope_parts.append("Momentum")
    scope = " + ".join(scope_parts)

    _purge_summary = (
        "Skipping purge (--no-purge)"
        if no_purge
        else (
            f"Purge: parquets always; "
            f"cache={'WIPE' if purge_cache else 'KEEP'}; "
            f"models={'WIPE' if purge_models else 'KEEP'}"
        )
    )
    _reply(
        f"<b>Overnight Pipeline — {scope}</b>\n"
        f"{_purge_summary}\n"
        f"Steps: dataset build → retrain for each model, sequentially.\n"
        f"Bot stays active. Use <code>/admin_pipeline_status</code> to monitor.\n"
        f"<i>Estimated total time: 4–8 hours with warm caches, 6–12 hours cold</i>"
    )

    def _run():
        global _retrain_running, _pipeline_log, _pipeline_phase, _pipeline_start
        import time as _t
        _retrain_running = True
        _pipeline_log.clear()
        _pipeline_start = _t.time()
        import subprocess, sys, shutil
        from pathlib import Path

        data_dir = Path("/data") if Path("/data").exists() else Path("/tmp")
        scripts  = Path(__file__).parent / "scripts"

        # Track each step's outcome for the final summary.
        # Format: {step_name: ("OK"|"FAIL"|"SKIP"|"PARTIAL", detail_str)}
        step_results: dict[str, tuple[str, str]] = {}

        def _try_promote_checkpoint(parquet_path: Path, label: str) -> bool:
            """If a *.checkpoint.parquet exists for this dataset, promote it to
            the full parquet path so the retrain can proceed with partial data.
            Returns True if promoted, False otherwise.

            Rationale: 678/1526 tickers' worth of episodes is still much better
            than zero data. The retrain can run on partial — it just won't
            generalize as broadly. Operator can re-run later for full rebuild.
            """
            ckpt = parquet_path.with_name(parquet_path.stem + ".checkpoint.parquet")
            if not ckpt.exists():
                return False
            try:
                import pandas as pd
                _df = pd.read_parquet(ckpt)
                _rows = len(_df)
                shutil.copy2(str(ckpt), str(parquet_path))
                _plog(
                    f"  [{label}] checkpoint recovered: {_rows:,} rows from "
                    f"{ckpt.name} → {parquet_path.name}"
                )
                return True
            except Exception as e:
                _plog(f"  [{label}] checkpoint recovery failed: {e}")
                return False

        def _sub(cmd: list[str], phase: str, timeout_h: float) -> tuple[bool, str]:
            """Run subprocess, set pipeline phase, return (ok, tail_output)."""
            global _pipeline_phase
            _pipeline_phase = phase
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=int(timeout_h * 3600))
                tail = ((r.stdout or "") + (r.stderr or ""))[-600:]
                return r.returncode == 0, tail
            except subprocess.TimeoutExpired:
                return False, f"Timeout after {timeout_h}h"
            except Exception as e:
                return False, str(e)

        try:
            # ── Step 0: Purge disk (CONSERVATIVE by default) ──────────────────
            # Default: only delete parquets (they get rebuilt anyway).
            # --purge-cache: also wipe price_cache (forces yfinance re-download).
            # --purge-models: also wipe .pkl files (DANGEROUS).
            if not no_purge:
                _pipeline_phase = "Step 0 — purging disk"
                _plog(
                    f"Step 0: purging disk... "
                    f"(cache: {'YES' if purge_cache else 'preserved'}, "
                    f"models: {'YES' if purge_models else 'preserved'})"
                )
                freed = 0
                items_removed: list[str] = []

                # Always: training parquets + intermediate JSON reports
                always_purge = (
                    "ml_training_base.parquet", "core_training_base.parquet",
                    "momentum_training.parquet", "ml_momentum_training.parquet",
                    "ml_training_input.parquet",
                    "momentum_report.json", "ml_report.json",
                    # Old checkpoint files from interrupted runs
                    "ml_training_base.checkpoint.parquet",
                    "core_training_base.checkpoint.parquet",
                    "momentum_training.checkpoint.parquet",
                )
                for name in always_purge:
                    p = data_dir / name
                    if p.exists():
                        freed += p.stat().st_size
                        p.unlink()
                        items_removed.append(name)

                # Opt-in: price cache (yfinance OHLCV cache, expensive to rebuild)
                if purge_cache:
                    price_cache_dir = data_dir / "price_cache"
                    if price_cache_dir.exists():
                        freed += sum(
                            f.stat().st_size for f in price_cache_dir.rglob("*") if f.is_file()
                        )
                        shutil.rmtree(str(price_cache_dir), ignore_errors=True)
                        items_removed.append("price_cache/")

                # Opt-in: model pickles (loses ML on retrain failure!)
                if purge_models:
                    for name in ("dip_models.pkl", "core_us_model.pkl", "momentum_model.pkl"):
                        p = data_dir / name
                        if p.exists():
                            freed += p.stat().st_size
                            p.unlink()
                            items_removed.append(name)

                mb = freed / 1024 / 1024
                _plog(f"Step 0: purge done — freed {mb:.0f} MB ({len(items_removed)} items)")
                _reply(
                    f"✅ <b>Step 0 — Disk purged</b>: freed <b>{mb:.0f} MB</b>\n"
                    f"<i>Removed: {len(items_removed)} items. "
                    f"{'price_cache preserved' if not purge_cache else 'price_cache WIPED'}, "
                    f"{'models preserved' if not purge_models else 'models WIPED'}.</i>\n"
                    f"Starting dataset builds..."
                )

            # ── Step 1+2: DIP ─────────────────────────────────────────────────
            if run_dip:
                _plog("Step 1: DIP dataset build started (~4-6h)...")
                _reply("⏳ <b>Step 1/6 — DIP dataset</b> building (4-6h)...")
                t0 = _t.time()
                out_dip = data_dir / "ml_training_base.parquet"
                ok, tail = _sub(
                    [sys.executable, str(scripts / "build_dip_dataset.py"),
                     "--start", "1993-01-01", "--out", str(out_dip), "--force-rebuild"],
                    phase="Step 1/6 — DIP dataset build",
                    timeout_h=12,
                )
                elapsed = (_t.time() - t0) / 60
                if ok:
                    try:
                        import pandas as pd
                        df = pd.read_parquet(out_dip)
                        rows = len(df)
                        tickers = df["ticker"].nunique() if "ticker" in df.columns else "?"
                        _plog(f"Step 1: DIP dataset OK — {rows:,} rows, {tickers} tickers, {elapsed:.0f}m")
                        _reply(f"✅ <b>Step 1/6 — DIP dataset done</b> ({elapsed:.0f} min)\n{rows:,} rows | {tickers} tickers")
                        step_results["DIP dataset"] = ("OK", f"{rows:,} rows, {tickers} tickers, {elapsed:.0f}m")
                    except Exception:
                        _plog(f"Step 1: DIP dataset OK ({elapsed:.0f}m) — stats unavailable")
                        _reply(f"✅ <b>Step 1/6 — DIP dataset done</b> ({elapsed:.0f} min)")
                        step_results["DIP dataset"] = ("OK", f"{elapsed:.0f}m (stats unavailable)")
                else:
                    _plog(f"Step 1: DIP dataset FAILED — {tail[-200:]}")
                    # Try to recover checkpoint as partial dataset
                    _ckpt_ok = _try_promote_checkpoint(out_dip, "Step 1/DIP")
                    if _ckpt_ok:
                        ok = True  # let retrain proceed with partial data
                        try:
                            import pandas as pd
                            df = pd.read_parquet(out_dip)
                            rows = len(df)
                            tickers = df["ticker"].nunique() if "ticker" in df.columns else "?"
                            _reply(
                                f"⚠️ <b>Step 1/6 — DIP dataset PARTIAL</b>\n"
                                f"Build failed but checkpoint recovered: {rows:,} rows | {tickers} tickers\n"
                                f"<i>Retrain will run on partial dataset. Re-run later for full rebuild.</i>"
                            )
                            step_results["DIP dataset"] = ("PARTIAL", f"{rows:,} rows (from checkpoint)")
                        except Exception:
                            step_results["DIP dataset"] = ("PARTIAL", "checkpoint recovered (stats unavailable)")
                    else:
                        _reply(
                            f"❌ <b>Step 1/6 — DIP dataset FAILED</b>\n<code>{tail[-400:]}</code>\n\n"
                            f"<i>No checkpoint to recover. Skipping DIP retrain, continuing with Core...</i>"
                        )
                        step_results["DIP dataset"] = ("FAIL", tail[-200:])

                if ok:
                    _pipeline_phase = "Step 2/6 — DIP retrain"
                    _plog("Step 2: DIP retrain started...")
                    _reply("⏳ <b>Step 2/6 — DIP retrain</b>...")
                    t0 = _t.time()
                    try:
                        from monthly_retrain import run_monthly_retrain
                        result = run_monthly_retrain() or {}
                        dec    = result.get("decision", "?")
                        rho    = result.get("candidate_rho_alpha")
                        elapsed = (_t.time() - t0) / 60
                        icon   = {"PROMOTED": "🚀", "KEPT": "⚠️"}.get(dec, "❌")
                        rho_s  = f" | ρ={rho:.4f}" if isinstance(rho, float) else ""
                        _plog(f"Step 2: DIP retrain {dec} ({elapsed:.0f}m){rho_s}")
                        _reply(f"{icon} <b>Step 2/6 — DIP retrain: {dec}</b> ({elapsed:.0f} min){rho_s}")
                        step_results["DIP retrain"] = (
                            "OK" if dec == "PROMOTED" else ("PARTIAL" if dec == "KEPT" else "FAIL"),
                            f"{dec}{rho_s} ({elapsed:.0f}m)"
                        )
                    except Exception as e:
                        _plog(f"Step 2: DIP retrain exception — {e}")
                        _reply(f"❌ <b>Step 2/6 — DIP retrain FAILED</b>\n<code>{e}</code>")
                        step_results["DIP retrain"] = ("FAIL", str(e))
                else:
                    step_results["DIP retrain"] = ("SKIP", "dataset failed")

            # ── Step 3+4: Core ────────────────────────────────────────────────
            if run_core:
                _plog("Step 3: Core dataset build started (~1-2h)...")
                _reply("⏳ <b>Step 3/6 — Core dataset</b> building (1-2h)...")
                t0 = _t.time()
                out_core = data_dir / "core_training_base.parquet"
                ok, tail = _sub(
                    [sys.executable, str(scripts / "build_core_dataset.py"),
                     "--out", str(out_core), "--start", "1993-01-01", "--force-rebuild"],
                    phase="Step 3/6 — Core dataset build",
                    timeout_h=10,
                )
                elapsed = (_t.time() - t0) / 60
                if ok:
                    try:
                        import pandas as pd
                        df = pd.read_parquet(out_core)
                        rows = len(df)
                        tickers = df["ticker"].nunique() if "ticker" in df.columns else "?"
                        _plog(f"Step 3: Core dataset OK — {rows:,} rows, {tickers} tickers, {elapsed:.0f}m")
                        _reply(f"✅ <b>Step 3/6 — Core dataset done</b> ({elapsed:.0f} min)\n{rows:,} rows | {tickers} tickers")
                        step_results["Core dataset"] = ("OK", f"{rows:,} rows, {tickers} tickers, {elapsed:.0f}m")
                    except Exception:
                        _plog(f"Step 3: Core dataset OK ({elapsed:.0f}m) — stats unavailable")
                        _reply(f"✅ <b>Step 3/6 — Core dataset done</b> ({elapsed:.0f} min)")
                        step_results["Core dataset"] = ("OK", f"{elapsed:.0f}m")
                else:
                    _plog(f"Step 3: Core dataset FAILED — {tail[-200:]}")
                    _reply(f"❌ <b>Step 3/6 — Core dataset FAILED</b>\n<code>{tail[-400:]}</code>\n\n<i>Continuing with Momentum...</i>")
                    step_results["Core dataset"] = ("FAIL", tail[-200:])

                if ok:
                    _plog("Step 4: Core model train started...")
                    _reply("⏳ <b>Step 4/6 — Core model training</b>...")
                    t0 = _t.time()
                    ok4, tail4 = _sub(
                        [sys.executable, str(scripts / "train_core_model.py"),
                         "--parquet", str(out_core)],
                        phase="Step 4/6 — Core model train",
                        timeout_h=1,
                    )
                    elapsed = (_t.time() - t0) / 60
                    if ok4:
                        _plog(f"Step 4: Core model trained ({elapsed:.0f}m)")
                        # Reload model in-process
                        try:
                            import core_radar as _cr
                            _cr._core_model_checked = False
                            _cr._core_model_bundle  = None
                            _cr._load_core_model()
                        except Exception:
                            pass
                        _reply(f"✅ <b>Step 4/6 — Core model trained</b> ({elapsed:.0f} min)\nML scoring active.")
                        step_results["Core retrain"] = ("OK", f"{elapsed:.0f}m")
                    else:
                        _plog(f"Step 4: Core model FAILED — {tail4[-200:]}")
                        _reply(f"❌ <b>Step 4/6 — Core model FAILED</b>\n<code>{tail4[-400:]}</code>")
                        step_results["Core retrain"] = ("FAIL", tail4[-200:])
                else:
                    step_results["Core retrain"] = ("SKIP", "dataset failed")

            # ── Step 5+6: Momentum ────────────────────────────────────────────
            if run_momentum:
                _plog("Step 5: Momentum dataset build started (~2-3h)...")
                _reply("⏳ <b>Step 5/6 — Momentum dataset</b> building (2-3h)...")
                t0 = _t.time()
                out_mom = data_dir / "momentum_training.parquet"
                mom_script = Path(__file__).parent / "momentum_radar" / "scripts" / "build_dataset.py"
                ok, tail = _sub(
                    [sys.executable, str(mom_script), "--out", str(out_mom)],
                    phase="Step 5/6 — Momentum dataset build",
                    timeout_h=8,
                )
                elapsed = (_t.time() - t0) / 60
                if ok:
                    try:
                        import pandas as pd
                        df = pd.read_parquet(out_mom)
                        rows = len(df)
                        _plog(f"Step 5: Momentum dataset OK — {rows:,} rows, {elapsed:.0f}m")
                        _reply(f"✅ <b>Step 5/6 — Momentum dataset done</b> ({elapsed:.0f} min)\n{rows:,} rows")
                        step_results["Momentum dataset"] = ("OK", f"{rows:,} rows, {elapsed:.0f}m")
                    except Exception:
                        _plog(f"Step 5: Momentum dataset OK ({elapsed:.0f}m)")
                        _reply(f"✅ <b>Step 5/6 — Momentum dataset done</b> ({elapsed:.0f} min)")
                        step_results["Momentum dataset"] = ("OK", f"{elapsed:.0f}m")
                else:
                    _plog(f"Step 5: Momentum dataset FAILED — {tail[-200:]}")
                    _reply(f"❌ <b>Step 5/6 — Momentum dataset FAILED</b>\n<code>{tail[-400:]}</code>")
                    step_results["Momentum dataset"] = ("FAIL", tail[-200:])

                if ok:
                    _pipeline_phase = "Step 6/6 — Momentum retrain"
                    _plog("Step 6: Momentum retrain started...")
                    _reply("⏳ <b>Step 6/6 — Momentum retrain</b>...")
                    t0 = _t.time()
                    try:
                        from momentum_radar.trainer import run_momentum_training
                        result = run_momentum_training(
                            dataset_path=out_mom,
                            bundle_path =data_dir / "momentum_model.pkl",
                            report_path =data_dir / "momentum_report.json",
                        ) or {}
                        dec  = result.get("decision", "?")
                        ic   = result.get("ic_ensemble")
                        elapsed = (_t.time() - t0) / 60
                        icon = {"PROMOTED": "🚀", "KEPT": "⚠️"}.get(dec, "❌")
                        ic_s = f" | IC={ic:.4f}" if isinstance(ic, float) else ""
                        _plog(f"Step 6: Momentum retrain {dec} ({elapsed:.0f}m){ic_s}")
                        _reply(f"{icon} <b>Step 6/6 — Momentum retrain: {dec}</b> ({elapsed:.0f} min){ic_s}")
                        step_results["Momentum retrain"] = (
                            "OK" if dec == "PROMOTED" else ("PARTIAL" if dec == "KEPT" else "FAIL"),
                            f"{dec}{ic_s} ({elapsed:.0f}m)"
                        )
                    except Exception as e:
                        _plog(f"Step 6: Momentum retrain exception — {e}")
                        _reply(f"❌ <b>Step 6/6 — Momentum retrain FAILED</b>\n<code>{e}</code>")
                        step_results["Momentum retrain"] = ("FAIL", str(e))
                else:
                    step_results["Momentum retrain"] = ("SKIP", "dataset failed")

            # ── Final summary: honest about what worked vs failed ────────────
            _plog("Overnight pipeline complete")
            _summary_lines = ["<b>Overnight pipeline complete</b>", ""]
            _ok_count   = sum(1 for s, _ in step_results.values() if s == "OK")
            _part_count = sum(1 for s, _ in step_results.values() if s == "PARTIAL")
            _fail_count = sum(1 for s, _ in step_results.values() if s == "FAIL")
            _skip_count = sum(1 for s, _ in step_results.values() if s == "SKIP")
            _total      = len(step_results)
            _summary_lines.append(
                f"<b>{_ok_count}/{_total} OK</b> | {_part_count} partial | "
                f"{_fail_count} failed | {_skip_count} skipped"
            )
            _summary_lines.append("")
            _emoji = {"OK": "✅", "PARTIAL": "⚠️", "FAIL": "❌", "SKIP": "⏭️"}
            for _step_name, (_status, _detail) in step_results.items():
                _summary_lines.append(f"{_emoji.get(_status, '?')} <b>{_step_name}</b>: {_detail}")
            _summary_lines += [
                "",
                "<i>Verifica com:</i>",
                "<code>/admin_check_config</code> — estado dos 3 modelos",
                "<code>/admin_features dip|core|momentum</code> — qualidade dos features",
            ]
            if _fail_count > 0 or _part_count > 0:
                _summary_lines += [
                    "",
                    "<i>Para retentar só os que falharam (mantém caches/modelos):</i>",
                    "<code>/admin_overnight --no-purge --dip-only</code> (ou --core-only / --momentum-only)",
                ]
            _reply("\n".join(_summary_lines))

        except Exception as e:
            logging.error(f"[overnight] unexpected: {e}", exc_info=True)
            _reply(f"❌ <b>Overnight pipeline crashed:</b>\n<code>{e}</code>")
        finally:
            _retrain_running = False
            _pipeline_phase  = ""
            _pipeline_start  = 0.0

    threading.Thread(target=_run, daemon=True, name="admin-overnight").start()


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
                from monthly_retrain import run_monthly_retrain
                dip_result = run_monthly_retrain() or {}
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
    /admin_retrain                    → triggers monthly retrain v3 ad-hoc (full)
    /admin_retrain dry-run            → build training input only (no train)
    /admin_retrain run 0.85           → real retrain with gating_ratio override

    Training data: bootstrap parquet + alert_db (universe_snapshot removed).
    Returns metrics (ρ_α, Brier, Top-K PnL) or training input shape if dry-run.
    """
    global _retrain_running

    if _retrain_running:
        _reply("⚠️ *Retreino já está a correr.*\n_Aguarda a conclusão._")
        return

    args = [p.lower() for p in parts[1:]]
    dry_run          = "dry-run" in args or "dry_run" in args or "dryrun" in args
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
            from monthly_retrain import run_monthly_retrain
            kwargs = dict(
                dry_run=dry_run,
                include_alert_db=include_alert_db,
            )
            if gating_ratio is not None:
                kwargs["gating_ratio"] = gating_ratio
            result = run_monthly_retrain(**kwargs)
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

    # Auto-create conviction if not already tracked AND not in deny-list.
    # Rationale: "if you /buy, you have a thesis — let the radar watch it".
    # Skips ETFs/foreign-listed/PPR via _AUTO_TRACK_DENY.
    # If conviction already exists, surface the DCA hint instead of duplicating.
    conviction_tail = ""
    try:
        import conviction_radar as cr
        existing_conv = cr.get_by_ticker(symbol)
        if existing_conv:
            conviction_tail = (
                f"\n\n🧠 _Convicção {symbol} já activa "
                f"({existing_conv.position_type} @ ${existing_conv.entry_price:.2f}). "
                f"Preço médio mudou para ${pos['avg_price']:.2f} — se quiseres "
                f"re-ancorar a tese ao novo preço médio:_\n"
                f"  `/conviction close {symbol} {price:.2f} dca`\n"
                f"  `/conviction add {symbol} {pos['avg_price']:.2f} {pos['shares']} {existing_conv.position_type}`"
            )
        else:
            deny = cr._get_deny_list()
            if symbol in deny or "." in symbol:
                conviction_tail = (
                    f"\n\n🧠 _Carteira actualizada. {symbol} é ETF/foreign-listed — "
                    f"não auto-criada como convicção (deny-list)._"
                )
            else:
                # Auto-classify and create
                try:
                    current = cr._fetch_price(symbol, fallback=price)
                    hist    = cr._fetch_price_history(symbol, period="6mo")
                    auto_type = cr._classify_type_from_signals(symbol, current, hist)
                    cr.add_conviction(
                        symbol, pos["avg_price"], pos["shares"], auto_type,
                        trailing_high=current if auto_type == "MOMENTUM" else None,
                        note="auto-created from /buy",
                    )
                    conviction_tail = (
                        f"\n\n🧠 _Convicção {symbol} criada auto como **{auto_type}** "
                        f"(classificação por ML + technicals). "
                        f"`/conviction transition {symbol} CORE` se queres long-term hold._"
                    )
                except Exception as e:
                    conviction_tail = (
                        f"\n\n🧠 _Convicção {symbol} não criada (auto-classifier falhou: {e}). "
                        f"Faz `/conviction add {symbol} {pos['avg_price']:.2f} {pos['shares']} TIPO` manual._"
                    )
    except Exception as e:
        logging.debug(f"[buy→conviction auto-create] {symbol}: {e}")

    _reply(
        f"✅ {action_str} — *{symbol}*\n\n"
        f"  💰 Compra: *{shares}x* @ *${price:.2f}* = *${result['cost']:.2f}*\n"
        f"  📊 Preço médio: *${pos['avg_price']:.2f}* | Total: *{pos['shares']}x*\n"
        f"  🏷️ Categoria: _{pos.get('category', 'N/D')}_"
        + (f"\n  🎯 Score entrada: *{score}/100*" if score is not None else "")
        + f"\n\n  💵 Liquidez restante: *€{liq:.2f}*"
        + liq_warn
        + conviction_tail
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

    # Auto-sync conviction tracker: full close → auto-close conviction;
    # partial → leave a hint for the user.
    conviction_tail = ""
    try:
        import conviction_radar as cr
        existing_conv = cr.get_by_ticker(symbol)
        if existing_conv:
            if result["action"] == "closed":
                outcome = cr.close_conviction(symbol, price, reason="sold-via-/sell")
                if outcome:
                    conviction_tail = (
                        f"\n\n🧠 _Convicção {symbol} fechada automaticamente "
                        f"({outcome['pnl_pct']:+.1f}% / {outcome['days_held']}d). "
                        f"Gravado em conviction_outcomes.json._"
                    )
            else:
                conviction_tail = (
                    f"\n\n🧠 _Convicção {symbol} continua activa (venda parcial). "
                    f"Usa `/conviction close {symbol} {price:.2f}` se também quiseres "
                    f"encerrar a tese._"
                )
    except Exception as e:
        logging.debug(f"[sell→conviction sync] {symbol}: {e}")

    _reply(
        f"{em} {action} — *{symbol}*\n\n"
        f"  💰 Vendido: *{result['shares_sold']}x* @ *${price:.2f}* = *${result['proceeds']:.2f}*\n"
        f"  {'📈' if pnl >= 0 else '📉'} P&L: *${pnl:+.2f}* ({pct:+.1f}%)"
        + remaining_str
        + f"\n\n  💵 Liquidez: *€{result['liquidity']:.2f}*"
        + conviction_tail
    )


# ── /conviction handler ──────────────────────────────────────────────────────────
#
# Parallel tracker for positions opened by personal conviction (outside the bot's
# alert flow). Uses the same DIP/Momentum/Core analysis engine but writes to
# personal_positions.json + conviction_outcomes.json — files NEVER touched by
# any ML training script. See conviction_radar.py for the isolation contract.

def _handle_conviction(parts: list[str]) -> None:
    sub = parts[1].lower() if len(parts) > 1 else "help"

    try:
        import conviction_radar as cr
    except Exception as e:
        _reply(f"❌ Módulo conviction não disponível: `{e}`")
        return

    if sub == "help" or sub in ("?", "-h", "--help"):
        _reply(
            "🧠 *Convicção — radar paralelo*\n"
            "_Trackeia posições com o motor do DipRadar mas sem contaminar o ML._\n\n"
            "*Comandos:*\n"
            "  `/conviction add TICKER PREÇO SHARES TIPO [PICO]`\n"
            "    Tipos: `DIP`, `MOMENTUM`, `CORE`. CORE → never_sell automático.\n"
            "    Auto-faz `/buy` se o ticker não está na carteira.\n"
            "  `/conviction list` — todas as activas + P&L (🔒 = never_sell)\n"
            "  `/conviction TICKER` — deep-dive condensado (sinais hold/close)\n"
            "  `/conviction close TICKER PREÇO [motivo]` — fechar + gravar outcome\n"
            "  `/conviction delete TICKER` — remover sem outcome (corrigir enganos)\n"
            "  `/conviction transition TICKER TIPO` — mudar tipo (DIP→MOMENTUM etc.)\n"
            "  `/conviction lock TICKER` — never_sell=True (triggers de fase só alertam)\n"
            "  `/conviction unlock TICKER` — never_sell=False (stop ATR pode fechar)\n"
            "  `/conviction sync` — varrer carteira e criar convicções em falta\n"
            "  `/conviction outcomes` — histórico de saídas\n\n"
            "_Auto-sync diário às 06:00 Lisboa. ETFs/PPR/foreign-listed saltados._"
        )
        return

    if sub == "add":
        # /conviction add TICKER PRICE SHARES TYPE [TRAILING_HIGH]
        if len(parts) < 6:
            _reply("⚠️ Uso: `/conviction add TICKER PREÇO SHARES TIPO [PICO]`\n"
                   "_Tipos: DIP, MOMENTUM, CORE_")
            return
        ticker = parts[2].upper().strip()
        try:
            price  = float(parts[3])
            shares = float(parts[4])
        except ValueError:
            _reply("⚠️ Preço e shares têm de ser números.")
            return
        ptype = parts[5].upper().strip()
        if ptype not in cr.VALID_TYPES:
            _reply(f"⚠️ Tipo inválido `{ptype}`. Usa: {', '.join(cr.VALID_TYPES)}")
            return
        trailing_high = None
        if len(parts) >= 7:
            try:
                trailing_high = float(parts[6])
            except ValueError:
                _reply("⚠️ PICO tem de ser um número.")
                return
        try:
            cr.add_conviction(ticker, price, shares, ptype, trailing_high=trailing_high)
        except ValueError as e:
            _reply(f"⚠️ {e}")
            return
        except Exception as e:
            _reply(f"❌ Erro ao adicionar: `{e}`")
            return

        # Auto-sync to portfolio: "if you added with price+shares, you bought it".
        # Only auto-buy if ticker not already in portfolio (avoids accidental DCA).
        buy_tail = ""
        try:
            from portfolio import get_position, buy as portfolio_buy
            if get_position(ticker) is None:
                _br = portfolio_buy(ticker, price, shares)
                buy_tail = (
                    f"\n\n💼 _Também adicionado à carteira: "
                    f"{shares:g}x @ ${price:.2f} (liquidez: €{_br.get('liquidity', 0):.2f})_"
                )
            else:
                buy_tail = (
                    f"\n\n💼 _{ticker} já na carteira — não duplicado. "
                    f"Se queres fazer DCA usa `/buy {ticker} PREÇO SHARES`._"
                )
        except Exception as e:
            buy_tail = f"\n\n⚠️ _Auto-/buy falhou: {e}. Faz `/buy {ticker} {price} {shares}` manual se quiseres._"

        # Show the full deep-dive immediately — much more informative than
        # a one-line score snapshot, and tailored to the position type.
        try:
            dive = cr.render_deep_dive(ticker)
            _reply(f"✅ *Convicção aberta — {ticker}*{buy_tail}\n\n{dive}")
        except Exception as e:
            _reply(
                f"✅ *Convicção aberta — {ticker}* ({ptype}){buy_tail}\n"
                f"_(deep-dive falhou: {e}. Tenta `/conviction {ticker}`.)_"
            )
        return

    if sub == "list":
        _reply(cr.render_list())
        return

    if sub == "outcomes":
        _reply(cr.render_outcomes())
        return

    if sub == "delete":
        # /conviction delete TICKER — silent removal (for fixing mistakes).
        # Does NOT write to conviction_outcomes.json. Use `close` for real exits.
        if len(parts) < 3:
            _reply("⚠️ Uso: `/conviction delete TICKER`\n"
                   "_Remove sem gravar outcome. Para fechar uma trade real usa `close`._")
            return
        ticker = parts[2].upper().strip()
        try:
            removed = cr.delete_conviction(ticker)
        except Exception as e:
            _reply(f"❌ Erro: `{e}`")
            return
        if not removed:
            _reply(f"⚠️ Convicção *{ticker}* não encontrada.")
            return
        _reply(
            f"🗑️ *{ticker} removida* (sem outcome gravado).\n"
            f"_Podes voltar a `/conviction add {ticker} …` com os valores correctos._"
        )
        return

    if sub in ("lock", "unlock"):
        # /conviction lock TICKER       → never_sell = True  (phase triggers suprimidos)
        # /conviction unlock TICKER     → never_sell = False (phase triggers podem fechar)
        if len(parts) < 3:
            _reply(f"⚠️ Uso: `/conviction {sub} TICKER`")
            return
        ticker = parts[2].upper().strip()
        new_val = (sub == "lock")
        try:
            updated = cr.set_never_sell(ticker, new_val)
        except Exception as e:
            _reply(f"❌ Erro: `{e}`")
            return
        if not updated:
            _reply(f"⚠️ Convicção *{ticker}* não encontrada.")
            return
        if new_val:
            _reply(
                f"🔒 *{ticker} bloqueada — never_sell = TRUE*\n"
                f"_Triggers de fase (trailing stop, momentum cooling) viram alertas 🟡 — "
                f"não recomendam fechar. Triggers estruturais (deterioração de tese, 8-K "
                f"adverso, queda de qualidade) **continuam** a recomendar fechar._"
            )
        else:
            _reply(
                f"🔓 *{ticker} desbloqueada — never_sell = FALSE*\n"
                f"_Triggers de fase podem recomendar fechar. Stop ATR activo._"
            )
        return

    if sub == "sync":
        # /conviction sync — manually trigger auto-sync from portfolio (debug aid)
        try:
            result = cr.auto_sync_from_portfolio(send_telegram=None)
        except Exception as e:
            _reply(f"❌ Erro no auto-sync: `{e}`")
            return

        port_count = result.get("portfolio_count", 0)
        if port_count == 0:
            _reply(
                "🔄 *Auto-sync — carteira vazia*\n\n"
                "_O `portfolio.get_positions()` devolveu 0 entradas — não há nada para sincronizar._\n\n"
                "*Como popular a carteira:*\n"
                "  1️⃣ `/conviction add TICKER PREÇO SHARES TIPO` — auto-faz `/buy` também\n"
                "  2️⃣ `/buy TICKER PREÇO SHARES` — auto-cria convicção também\n"
                "  3️⃣ Definir env vars `HOLDING_X=TICKER:SHARES:PREÇO` no Railway e reiniciar\n\n"
                "_Vê o que tens com `/portfolio`._"
            )
            return

        lines = [
            f"🔄 *Auto-sync resultado* (carteira: {port_count} posições)",
            f"  ✅ Adicionadas: {result['added']}",
            f"  ⏭️ Saltadas: {result['skipped']} (ETFs/foreign/erro)",
            f"  ♻️ Já trackeadas: {result['already_tracked']}",
            "",
        ]
        for item in result.get("items", []):
            t, p, r = item
            if r == "added":
                lines.append(f"  ✅ {t} → {p}")
            elif r == "já em convicção":
                lines.append(f"  ♻️ {t} _(já em convicção)_")
            else:
                lines.append(f"  ⏭️ {t} _({r})_")
        _reply("\n".join(lines))
        return

    if sub == "close":
        # /conviction close TICKER PRICE [reason]
        if len(parts) < 4:
            _reply("⚠️ Uso: `/conviction close TICKER PREÇO [motivo]`")
            return
        ticker = parts[2].upper().strip()
        try:
            price = float(parts[3])
        except ValueError:
            _reply("⚠️ Preço tem de ser um número.")
            return
        reason = " ".join(parts[4:]).strip() if len(parts) >= 5 else "manual"
        try:
            outcome = cr.close_conviction(ticker, price, reason=reason)
        except Exception as e:
            _reply(f"❌ Erro ao fechar: `{e}`")
            return
        if not outcome:
            _reply(f"⚠️ Convicção *{ticker}* não encontrada.")
            return
        pnl_emo = "🟢" if outcome["pnl_pct"] >= 0 else "🔴"
        _reply(
            f"{pnl_emo} *Convicção fechada — {ticker}*\n\n"
            f"  💰 {outcome['shares']:g}x @ ${outcome['exit_price']:.2f}\n"
            f"  📊 P&L: *${outcome['pnl_abs']:+.2f}* ({outcome['pnl_pct']:+.1f}%) em {outcome['days_held']}d\n"
            f"  🏷️ Motivo: _{outcome['reason']}_\n\n"
            f"_Gravado em `conviction_outcomes.json` (não entra em ML)._"
        )
        return

    if sub == "transition":
        # /conviction transition TICKER TYPE
        if len(parts) < 4:
            _reply("⚠️ Uso: `/conviction transition TICKER TIPO`")
            return
        ticker = parts[2].upper().strip()
        new_type = parts[3].upper().strip()
        if new_type not in cr.VALID_TYPES:
            _reply(f"⚠️ Tipo inválido `{new_type}`. Usa: {', '.join(cr.VALID_TYPES)}")
            return
        try:
            pos = cr.transition_type(ticker, new_type)
        except Exception as e:
            _reply(f"❌ Erro: `{e}`")
            return
        if not pos:
            _reply(f"⚠️ Convicção *{ticker}* não encontrada.")
            return
        if new_type == "MOMENTUM":
            _reply(f"🔄 *{ticker}* → {new_type}\n_Trailing high ancorado a ${pos.trailing_high:.2f}._")
        else:
            _reply(f"🔄 *{ticker}* → {new_type}")
        return

    # Default: treat first arg as TICKER → deep-dive
    ticker = parts[1].upper().strip()
    try:
        _reply(cr.render_deep_dive(ticker))
    except Exception as e:
        _reply(f"❌ Erro no deep-dive de {ticker}: `{e}`")


# ── /ppr handler ─────────────────────────────────────────────────────────────────
#
# PPR (Plano Poupança Reforma) — Telegram-managed, replaces the deprecated
# PPR_SHARES / PPR_AVG_COST env vars which caused a €19B snapshot bug when
# set wrong. State lives in /data/ppr.json (state.get_ppr / set_ppr).

def _handle_ppr(parts: list[str]) -> None:
    from state import get_ppr, set_ppr, set_ppr_nav, clear_ppr

    sub = parts[1].lower() if len(parts) > 1 else "show"

    if sub in ("show", ""):
        s = get_ppr()
        shares = s.get("shares", 0)
        avg    = s.get("avg_cost", 0)
        nav    = s.get("current_nav")
        nav_dt = s.get("nav_updated")
        if shares <= 0 or avg <= 0:
            _reply(
                "💰 *PPR* — sem valores configurados.\n\n"
                "*Setup inicial (custo base):*\n"
                "  `/ppr set SHARES PREÇO_MÉDIO`\n"
                "  _Ex: `/ppr set 100 50.25` → 100 unidades a €50.25_\n\n"
                "*Update cotação (NAV actual do banco):*\n"
                "  `/ppr nav NOVO_NAV`\n"
                "  _Ex: `/ppr nav 52.80` → cotação actual €52.80/unidade_\n\n"
                "_Sem NAV: valor = custo (não há live tracking automático para PPRs)._\n"
                "_Para limpar: `/ppr clear`_"
            )
            return
        cost = shares * avg
        if nav and nav > 0:
            value = shares * nav
            pnl   = value - cost
            pnl_pct = (nav / avg - 1) * 100 if avg > 0 else 0
            arrow = "📈" if pnl >= 0 else "📉"
            stale_note = ""
            if nav_dt:
                from datetime import date as _date
                try:
                    age = (_date.today() - _date.fromisoformat(nav_dt)).days
                    if age > 30:
                        stale_note = f" ⚠️ NAV com {age} dias (desactualizada — corre `/ppr nav` se possível)"
                    else:
                        stale_note = f" _(NAV de há {age} dias)_"
                except Exception:
                    pass
            _reply(
                f"💰 *PPR* — configurada com NAV live\n\n"
                f"  Unidades: *{shares:g}*\n"
                f"  Custo médio: *€{avg:.4f}*\n"
                f"  NAV actual: *€{nav:.4f}*{stale_note}\n"
                f"  Custo total: *€{cost:,.2f}*\n"
                f"  Valor actual: *€{value:,.2f}*\n"
                f"  {arrow} P&L: *€{pnl:+,.2f}* ({pnl_pct:+.2f}%)\n\n"
                f"_Update NAV: `/ppr nav NOVO_VALOR`_"
            )
        else:
            _reply(
                f"💰 *PPR* — sem NAV (valor frozen no custo)\n\n"
                f"  Unidades: *{shares:g}*\n"
                f"  Preço médio: *€{avg:.4f}*\n"
                f"  Valor (= custo): *€{cost:,.2f}*\n\n"
                f"⚠️ _Sem NAV definida, o snapshot trata a PPR como sem variação._\n"
                f"_Para tracking de P&L: lê a cotação actual no teu banco e corre:_\n"
                f"  `/ppr nav NOVO_NAV`"
            )
        return

    if sub == "set":
        if len(parts) < 4:
            _reply("⚠️ Uso: `/ppr set SHARES PREÇO_MÉDIO`\n_Exemplo: `/ppr set 100 50.25`_")
            return
        try:
            shares = float(parts[2])
            avg    = float(parts[3])
        except ValueError:
            _reply("⚠️ Shares e preço têm de ser números.")
            return
        if shares < 0 or avg < 0:
            _reply("⚠️ Shares e preço não podem ser negativos.")
            return
        value = shares * avg
        if value > 1_000_000:
            _reply(
                f"⚠️ *Valor suspeito:* {shares} × €{avg:.4f} = *€{value:,.2f}*\n\n"
                f"_Mais de €1M — provável typo. Confirma com `--force` se for mesmo:_\n"
                f"  `/ppr set {parts[2]} {parts[3]} --force`"
            )
            if "--force" not in parts:
                return
        set_ppr(shares, avg)
        existing_nav = get_ppr().get("current_nav")
        nav_hint = ""
        if not existing_nav:
            nav_hint = (
                f"\n\n_Para ver P&L real, faz `/ppr nav <cotação_actual>` "
                f"quando souberes a NAV do dia (lê no teu banco)._"
            )
        _reply(
            f"✅ *PPR actualizada* (custo base)\n\n"
            f"  Unidades: *{shares:g}*\n"
            f"  Preço médio: *€{avg:.4f}*\n"
            f"  Custo total: *€{value:,.2f}*"
            f"{nav_hint}"
        )
        return

    if sub == "nav":
        if len(parts) < 3:
            _reply(
                "⚠️ Uso: `/ppr nav NOVA_COTAÇÃO`\n\n"
                "_Lê a cotação actual da PPR no teu banco (extracto/app) e "
                "introduz aqui. Snapshot passa a usar este valor para calcular "
                "P&L real até actualizares de novo._\n\n"
                "_Exemplo: `/ppr nav 52.80`_"
            )
            return
        try:
            nav = float(parts[2])
        except ValueError:
            _reply("⚠️ NAV tem de ser um número.")
            return
        if nav <= 0:
            _reply("⚠️ NAV tem de ser positiva.")
            return
        existing = get_ppr()
        shares = existing.get("shares", 0)
        avg    = existing.get("avg_cost", 0)
        if shares <= 0 or avg <= 0:
            _reply("⚠️ Define primeiro o custo base com `/ppr set SHARES PREÇO_MÉDIO`.")
            return
        # Sanity: refuse NAVs absurdly far from avg_cost (likely typo)
        if avg > 0 and (nav > avg * 10 or nav < avg / 10):
            _reply(
                f"⚠️ *NAV suspeita:* €{nav:.4f} vs preço médio €{avg:.4f} "
                f"(diferença >10×). Provável typo. Confirma o valor antes de inserir."
            )
            return
        set_ppr_nav(nav)
        value = shares * nav
        cost  = shares * avg
        pnl   = value - cost
        pnl_pct = (nav / avg - 1) * 100
        arrow = "📈" if pnl >= 0 else "📉"
        _reply(
            f"✅ *NAV da PPR actualizada*\n\n"
            f"  NAV: *€{nav:.4f}* (era €{avg:.4f} na entrada)\n"
            f"  Valor actual: *€{value:,.2f}*\n"
            f"  {arrow} P&L: *€{pnl:+,.2f}* ({pnl_pct:+.2f}%)\n\n"
            f"_Snapshot passa a usar este NAV até nova actualização._\n"
            f"_Recomendado: actualizar pelo menos 1× por mês._"
        )
        return

    if sub == "scrape":
        # /ppr scrape — auto-fetch NAV from investing.com using stored ISIN
        from state import set_ppr_nav
        existing = get_ppr()
        if not existing or existing.get("shares", 0) <= 0:
            _reply("⚠️ Define primeiro o custo base com `/ppr set SHARES PREÇO_MÉDIO`.")
            return
        isin = existing.get("isin", "PTARMJHM0003")
        _reply(f"_🔍 A consultar cotação de `{isin}` no investing.com..._")
        try:
            import ppr_scraper
            result = ppr_scraper.fetch_ppr_nav(isin)
        except Exception as e:
            _reply(f"❌ Erro no scraper: `{e}`")
            return
        if not result or not result.get("nav"):
            _reply(
                f"⚠️ Não foi possível obter NAV de `{isin}` via investing.com.\n\n"
                f"_Causa provável: alteração na estrutura da página ou rate limit._\n"
                f"_Faz `/ppr nav <valor>` manual com a cotação do banco._"
            )
            return
        nav = result["nav"]
        avg = existing.get("avg_cost", 0)
        if avg > 0 and (nav > avg * 10 or nav < avg / 10):
            _reply(
                f"⚠️ Cotação obtida (€{nav:.4f}) suspeita vs preço médio "
                f"(€{avg:.4f}). Não aplicada. Verifica manualmente."
            )
            return
        set_ppr_nav(nav, source=result.get("source", "investing.com"))

        shares = existing.get("shares", 0)
        cost   = shares * avg
        value  = shares * nav
        pnl    = value - cost
        pnl_pct = (nav / avg - 1) * 100 if avg > 0 else 0
        arrow = "📈" if pnl >= 0 else "📉"
        extras = []
        if result.get("change_pct") is not None:
            extras.append(f"variação dia: {result['change_pct']:+.2f}%")
        if result.get("status"):
            extras.append(result["status"])
        extras_str = " | ".join(extras) if extras else ""
        _reply(
            f"✅ *NAV actualizada via scraper*\n\n"
            f"  ISIN: `{isin}`\n"
            f"  NAV: *€{nav:.4f}*\n"
            f"  Fonte: _{result.get('source', 'investing.com')}_"
            + (f"\n  Estado: _{extras_str}_" if extras_str else "")
            + f"\n\n"
            f"  Valor actual: *€{value:,.2f}*\n"
            f"  {arrow} P&L: *€{pnl:+,.2f}* ({pnl_pct:+.2f}%)\n\n"
            f"_Auto-scrape diário 22h30 Lisboa._"
        )
        return

    if sub == "isin":
        # /ppr isin <NEW_ISIN> — change which fund the scraper targets
        from state import set_ppr_isin
        if len(parts) < 3:
            existing = get_ppr()
            cur_isin = existing.get("isin", "PTARMJHM0003")
            _reply(
                f"💰 *PPR ISIN actual:* `{cur_isin}`\n\n"
                f"_Para alterar (caso tenhas outro PPR):_\n"
                f"  `/ppr isin NOVO_ISIN`\n\n"
                f"_Exemplo: `/ppr isin PTBPPHM0008`_\n"
                f"_Confirma no investing.com que o fundo aparece em "
                f"pt.investing.com/funds/<ISIN_minusculo>_"
            )
            return
        new_isin = parts[2].upper().strip()
        if len(new_isin) != 12:
            _reply(f"⚠️ ISIN inválido (deve ter 12 caracteres): `{new_isin}`")
            return
        set_ppr_isin(new_isin)
        _reply(
            f"✅ ISIN actualizado: `{new_isin}`\n"
            f"_Próximo `/ppr scrape` vai usar este ISIN._"
        )
        return

    if sub == "clear":
        clear_ppr()
        _reply("✅ *PPR limpa.* Snapshot da carteira já não a inclui.")
        return

    _reply(f"⚠️ Sub-comando `{sub}` desconhecido. Usa: `show`, `set`, `nav`, `scrape`, `isin`, `clear`.")


# ── /etf handler ─────────────────────────────────────────────────────────────────
#
# Passive ETF tracker. Same pattern as /ppr — Telegram-managed state, lives
# outside the active scan / conviction / ML pipeline. Live prices fetched
# via yfinance when ticker resolves (use exchange suffix: SXR8.DE, IS3N.DE,
# CSPX.L, etc.), falls back to avg_cost if the ticker doesn't resolve.

def _handle_etf(parts: list[str]) -> None:
    from state import get_etfs, set_etf, remove_etf, clear_etfs
    import yfinance as _yf

    sub = parts[1].lower() if len(parts) > 1 else "show"

    if sub in ("show", "list", ""):
        etfs = get_etfs()
        if not etfs:
            _reply(
                "📊 *ETFs* — nenhum configurado.\n\n"
                "Para adicionar (com sufixo do mercado: `.DE` Xetra, `.L` Londres):\n"
                "  `/etf set SXR8.DE 5 450.50`\n"
                "  `/etf set IS3N.DE 10 32.40`\n\n"
                "_Tickers sem ponto assumem USD (NYSE/Nasdaq)._"
            )
            return
        lines = [f"📊 *ETFs* ({len(etfs)} posições)"]
        total_value = 0.0
        total_cost  = 0.0
        for tkr in sorted(etfs.keys()):
            pos = etfs[tkr]
            shares = pos["shares"]
            avg    = pos["avg_cost"]
            # Live price attempt
            live = None
            try:
                _info = _yf.Ticker(tkr).fast_info
                _p = getattr(_info, "last_price", None)
                if _p and float(_p) > 0:
                    live = float(_p)
            except Exception:
                pass
            cost   = shares * avg
            if live:
                value = shares * live
                pnl   = value - cost
                pnl_pct = (live / avg - 1) * 100 if avg > 0 else 0
                arrow = "📈" if pnl >= 0 else "📉"
                lines.append(
                    f"  {arrow} *{tkr}* | {shares:g} @ €{avg:.2f} → €{live:.2f}"
                    f"  ({pnl:+,.2f} / {pnl_pct:+.1f}%)"
                )
                total_value += value
            else:
                value = cost  # fallback
                lines.append(
                    f"  ⚪ *{tkr}* | {shares:g} @ €{avg:.2f} _(sem live price)_"
                    f"  €{value:,.2f}"
                )
                total_value += value
            total_cost += cost
        total_pnl = total_value - total_cost
        pnl_pct_total = (total_value / total_cost - 1) * 100 if total_cost > 0 else 0
        lines += [
            "",
            f"💼 *Total ETFs:* €{total_value:,.2f}",
            f"   Custo: €{total_cost:,.2f} | P&L: €{total_pnl:+,.2f} ({pnl_pct_total:+.1f}%)",
            "",
            "_Adicionar: `/etf set TICKER SHARES PREÇO`_",
            "_Remover:   `/etf remove TICKER`_",
        ]
        _reply("\n".join(lines))
        return

    if sub == "set":
        if len(parts) < 5:
            _reply(
                "⚠️ Uso: `/etf set TICKER SHARES PREÇO_MÉDIO`\n"
                "_Exemplo: `/etf set SXR8.DE 5 450.50`_\n"
                "_Usa o sufixo do mercado (.DE Xetra, .L Londres) para live prices._"
            )
            return
        ticker = parts[2].upper().strip()
        try:
            shares = float(parts[3])
            avg    = float(parts[4])
        except ValueError:
            _reply("⚠️ Shares e preço têm de ser números.")
            return
        if shares < 0 or avg < 0:
            _reply("⚠️ Shares e preço não podem ser negativos.")
            return
        # Sanity check
        cost = shares * avg
        if cost > 500_000:
            _reply(
                f"⚠️ *Valor suspeito:* {shares} × €{avg:.2f} = *€{cost:,.2f}*\n"
                f"_Mais de €500k num único ETF — confirma os valores._"
            )
            return
        # Try live price for confirmation
        live_str = ""
        try:
            _info = _yf.Ticker(ticker).fast_info
            _p = getattr(_info, "last_price", None)
            if _p and float(_p) > 0:
                live_val = shares * float(_p)
                live_str = f"\n  📡 Preço actual: €{float(_p):.2f} → valor live: *€{live_val:,.2f}*"
            else:
                live_str = f"\n  ⚠️ _yfinance sem dados para `{ticker}` — vai usar custo médio no snapshot._"
        except Exception:
            live_str = f"\n  ⚠️ _yfinance falhou para `{ticker}` — usa fallback custo médio._"

        set_etf(ticker, shares, avg)
        _reply(
            f"✅ *ETF actualizado — {ticker}*\n\n"
            f"  Unidades: *{shares:g}*\n"
            f"  Custo médio: *€{avg:.4f}*\n"
            f"  Custo total: *€{cost:,.2f}*"
            f"{live_str}\n\n"
            f"_Vê todos: `/etf`_"
        )
        return

    if sub == "remove":
        if len(parts) < 3:
            _reply("⚠️ Uso: `/etf remove TICKER`")
            return
        ticker = parts[2].upper().strip()
        if remove_etf(ticker):
            _reply(f"🗑️ *{ticker}* removida do tracker de ETFs.")
        else:
            _reply(f"⚠️ *{ticker}* não está no tracker.")
        return

    if sub == "clear":
        clear_etfs()
        _reply("✅ *Tracker de ETFs limpo.* Snapshot já não os inclui.")
        return

    _reply(f"⚠️ Sub-comando `{sub}` desconhecido. Usa: `show`, `set`, `remove`, `clear`.")


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
        parts_after_cmd = (parts[1:] if len(parts) > 1 else [])
        topic = parts_after_cmd[0].lower() if parts_after_cmd else ""

        if topic in ("market", "scan", "analise", "análise"):
            _reply(
                "*📊 MARKET ANALYSIS*\n\n"
                "`/scan`                       → Force a dip scan now\n"
                "`/analisar <TICK>`            → Full analysis for one ticker\n"
                "`/comparar <T1> <T2> ...`     → Compare 2-5 tickers side by side\n"
                "`/historico <TICK>`           → Historical scores for a ticker\n"
                "`/tier3`                      → Rare gems (score ≥ 80)\n"
                "`/rejeitados`                 → Tickers rejected today + reason\n"
                "`/backtest`                   → Backtest summary\n"
                "`/themes`                     → Active themes / trends\n"
                "`/add_theme <k> <l> <T>`      → Add theme (key label TICK1,TICK2)\n"
                "`/remove_theme <key>`         → Remove theme\n\n"
                "_← Back: `/help`_"
            )
        elif topic in ("portfolio", "carteira", "pf"):
            _reply(
                "*💼 PORTFOLIO*\n\n"
                "*Holdings & balance*\n"
                "`/carteira`                   → Portfolio snapshot\n"
                "`/portfolio`                  → Active positions\n"
                "`/liquidez`                   → View / adjust cash balance\n"
                "`/buy TICK PRICE SHARES`      → Register a buy\n"
                "`/sell TICK PRICE [SHR]`      → Register a sell\n"
                "`/allocate <TICK>`            → Allocation suggestion (sizing)\n"
                "`/performance [date] [score]` → Simulate annual return + risk\n\n"
                "*Flip trades (bot-tracked)*\n"
                "`/flip`                       → Open trades + P&L\n"
                "`/flip list`                  → All trades\n"
                "`/flip add TICK ENTRY SHR`    → Register entry\n"
                "`/flip close ID EXIT`         → Close trade\n"
                "`/flip del ID`                → Delete trade\n\n"
                "*Core positions (hold-forever)*\n"
                "`/core`                       → Open CORE positions\n"
                "`/core_add TICK P EUR`        → Register manual CORE buy\n"
                "`/core_close TICK P`          → Close CORE position\n"
                "`/core_universe`              → View CoreRadar universe\n"
                "`/core_universe_add TICK`     → Add ticker to CORE universe\n"
                "`/core_universe_remove TICK`  → Remove from CORE universe\n\n"
                "*Watchlist*\n"
                "`/watchlist`                  → View dynamic watchlist\n"
                "`/watchlist add TICK`         → Add ticker\n"
                "`/watchlist rm TICK`          → Remove ticker\n"
                "`/watchlist clear`            → Clear watchlist\n\n"
                "_← Back: `/help`_"
            )
        elif topic in ("build", "ml", "train", "pipeline"):
            _reply(
                "*🔧 ADMIN — ML PIPELINE*\n"
                "_Dataset building, training and gating._\n\n"
                "*Full pipeline*\n"
                "`/admin_overnight`            → Sequential rebuild (DIP + Core + Momentum)\n"
                "`/admin_full_pipeline`        → Parquets + retrain (parallel)\n"
                "`/admin_run_pipeline`         → Flexible pipeline with flags\n\n"
                "*DIP*\n"
                "`/admin_dip_dataset`          → Build DIP dataset (since 1993)\n"
                "`/admin_retrain [dry-run]`    → Retrain DIP model (uses existing parquet)\n"
                "`/admin_regen_parquet`        → Regen parquet (PIT fundamentals + alpha_90d)\n\n"
                "*Momentum*\n"
                "`/admin_momentum_dataset`     → Build Momentum dataset\n"
                "`/admin_momentum_retrain`     → Retrain Momentum model\n\n"
                "*Core*\n"
                "`/admin_build_core_dataset`   → Build Core dataset + commit to GitHub\n"
                "`/admin_train_core_model`     → Train Core ML model\n"
                "`/admin_refresh_core_universe` → Refresh CORE universe from sources\n\n"
                "*Gating / promotion*\n"
                "`/admin_set_floor <value>`    → Set ρ_α absolute floor\n"
                "`/admin_promote`              → Promote pending bundle to production\n"
                "`/admin_pending_info`         → Inspect pending candidate bundle\n\n"
                "*Data*\n"
                "`/mldata`                     → ML training database stats\n"
                "`/mldata update`              → Force outcome resolution update\n"
                "`/admin_features <model>`     → Per-feature stats (n / mean / std / nan)\n\n"
                "_← Back: `/help`_"
            )
        elif topic in ("admin", "sys", "system"):
            _reply(
                "*⚙️ ADMIN — SYSTEM & MONITORING*\n\n"
                "*Status*\n"
                "`/status`                     → Bot status\n"
                "`/health`                     → Observability dashboard\n"
                "`/health errors`              → Critical error log\n\n"
                "*Pipeline control*\n"
                "`/admin_pipeline_status` (`/ps`) → Current pipeline phase + log\n"
                "`/admin_kill_pipeline`        → Terminate running pipeline\n"
                "`/admin_retrain_status`       → Last retrain result\n"
                "`/admin_regen_status`         → Last parquet regen result\n\n"
                "*Disk & cache*\n"
                "`/admin_disk`                 → Disk usage breakdown (/data/)\n"
                "`/admin_clear_price_cache`    → Clear price cache\n"
                "`/admin_check_config`         → Verify model bundles + config\n\n"
                "*Bundles & charts*\n"
                "`/admin_load_models <url>`    → Load pickles into /data/\n"
                "`/admin_publish_charts`       → Publish charts to GitHub\n"
                "`/admin_test_feed`            → Diagnose market data feed\n\n"
                "*Settings*\n"
                "`/language <en|pt>`           → Set bot language\n\n"
                "_← Back: `/help`_"
            )
        else:
            _reply(
                "*🤖 DipRadar — Help*\n\n"
                "Pick a topic:\n"
                "`/help market`         → Scans, analysis, themes\n"
                "`/help portfolio`      → Holdings, flip, core, watchlist\n"
                "`/help build`          → [ADMIN] ML datasets + training\n"
                "`/help admin`          → [ADMIN] System + monitoring\n\n"
                "*Quick commands*\n"
                "`/status`              → Bot status\n"
                "`/scan`                → Force a dip scan\n"
                "`/carteira`            → Portfolio snapshot\n"
                "`/analisar <TICK>`     → Full ticker analysis\n"
                "`/language <en|pt>`    → Set language (default: pt)"
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

    elif cmd == "/conviction":
        if not _check_rate(cmd_key): return
        _handle_conviction(parts)

    elif cmd == "/ppr":
        if not _check_rate(cmd_key): return
        _handle_ppr(parts)

    elif cmd == "/etf":
        if not _check_rate(cmd_key): return
        _handle_etf(parts)

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

    elif cmd in ("/admin_portfolio_clear", "/admin_carteira_clear"):
        # Wipe corrupted portfolio cache + persisted JSON. For recovering from
        # the Google Sheets locale bug (values inflated 10^N×). After running,
        # use /sync_portfolio to re-import positions with correct values.
        # Requires explicit --yes to prevent accidental wipe.
        if "--yes" not in parts:
            _reply(
                "⚠️ *Confirmação obrigatória*\n\n"
                "Este comando vai APAGAR todas as posições da carteira "
                "(cache em memória + ficheiro `_dipr_portfolio.json`).\n\n"
                "_PPR, ETFs e convicções não são tocados._\n\n"
                "Para confirmar:\n"
                "  `/admin_portfolio_clear --yes`\n\n"
                "_Depois usa `/sync_portfolio` para re-importar._"
            )
        else:
            try:
                from portfolio import _cache, _PORTFOLIO_FILE
                _cache["positions"] = {}
                # _cache["liquidity"] preserved — that's a separate concern
                if _PORTFOLIO_FILE.exists():
                    _PORTFOLIO_FILE.unlink()
                _reply(
                    "✅ *Carteira limpa*\n\n"
                    "  Cache: zerada (positions={})\n"
                    "  Ficheiro: `_dipr_portfolio.json` removido\n\n"
                    "Para re-importar com valores corrigidos:\n"
                    "`/sync_portfolio NVO:159.59197811:50.56 ADBE:16.27745882:268.6 …`"
                )
            except Exception as e:
                _reply(f"❌ Erro a limpar: `{e}`")

    elif cmd in ("/admin_portfolio_diag", "/admin_carteira_diag"):
        # Diagnostic: dump exact contents of _cache["positions"] + computed values
        try:
            from portfolio import get_positions, get_holdings_tuples, _cache
            from market_client import get_usdeur, get_portfolio_snapshot
            from state import get_etfs, get_ppr
            import json, yfinance as _yf

            raw_positions = get_positions()
            holdings_tuples = get_holdings_tuples()
            usd_eur = get_usdeur()
            etfs = get_etfs()
            ppr  = get_ppr()

            lines = ["🔍 *Portfolio Diagnostic*\n"]
            lines.append(f"USD/EUR: `{usd_eur:.6f}`\n")
            lines.append(f"_cache[positions] has *{len(raw_positions)}* entries:\n")
            for sym, pos in raw_positions.items():
                _shares = pos.get("shares", "MISSING")
                _avg    = pos.get("avg_price", "MISSING")
                _cost   = pos.get("total_cost", "MISSING")
                lines.append(
                    f"  *{sym}*: shares=`{_shares}` avg_price=`{_avg}` "
                    f"total_cost=`{_cost}`"
                )
            lines.append("")
            lines.append(f"get_holdings_tuples() returns *{len(holdings_tuples)}* tuples:")
            for t in holdings_tuples:
                lines.append(f"  `{t}`")
            lines.append("")
            lines.append(f"Live prices + value_eur (what snapshot computes):")
            for sym, shares, avg in holdings_tuples:
                try:
                    _info = _yf.Ticker(sym).fast_info
                    _now = float(getattr(_info, "last_price", 0) or 0)
                    _fx = usd_eur  # all stocks USD
                    _value = shares * _now * _fx
                    lines.append(
                        f"  *{sym}*: {shares:g} × ${_now:.2f} × {_fx:.4f} = "
                        f"*€{_value:,.2f}*"
                    )
                except Exception as _e:
                    lines.append(f"  *{sym}*: live price failed ({_e})")
            lines.append("")
            lines.append(f"PPR state: shares=`{ppr.get('shares')}` avg=`{ppr.get('avg_cost')}` nav=`{ppr.get('current_nav')}`")
            lines.append(f"ETFs ({len(etfs)}): `{list(etfs.keys())}`")
            for etf, p in etfs.items():
                lines.append(f"  {etf}: shares=`{p.get('shares')}` avg=`{p.get('avg_cost')}`")

            # Run actual snapshot to compare
            lines.append("")
            lines.append(f"get_portfolio_snapshot() returns:")
            _snap = get_portfolio_snapshot(
                holdings_tuples,
                float(ppr.get("shares") or 0),
                float(ppr.get("avg_cost") or 0),
                usd_eur,
                etfs=etfs,
                ppr_current_nav=ppr.get("current_nav"),
            )
            lines.append(f"  total_eur: *€{_snap.get('total_eur'):,.2f}*")
            lines.append(f"  total_cost: *€{_snap.get('total_cost'):,.2f}*")
            lines.append(f"  ppr_value: *€{_snap.get('ppr_value'):,.2f}*")
            lines.append(f"  etf_total: *€{_snap.get('etf_total', 0):,.2f}*")
            lines.append(f"  positions ({len(_snap.get('positions', []))}):")
            for pos in _snap.get("positions", []):
                lines.append(
                    f"    {pos.get('symbol')}: "
                    f"shares={pos.get('shares')} "
                    f"value=€{pos.get('value_eur'):,.2f} "
                    f"pnl_d={pos.get('pnl_day')}"
                )
            _reply("\n".join(lines))
        except Exception as e:
            _reply(f"❌ Diag falhou: `{e}`")

    elif cmd == "/admin_features":
        _handle_admin_features(parts)

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
        # `threading` is already imported at module top; importing it inside
        # _handle_command would shadow it as a local variable and break the
        # other elif branches that use threading.Thread() above this line.
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

    elif cmd in ("/admin_overnight", "/overnight"):
        _handle_admin_overnight(parts)

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
