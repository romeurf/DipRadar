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
from utils.rate_limiter import is_allowed, rate_status
from utils.i18n import t, set_language, get_language
from config import SUPPORTED_LANGUAGES
from state import (
    get_ticker_score_history,
)
# Flip log imports removed Phase 6 — Flip Fund deprecated
from utils import health_monitor
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
# _cb_allocate_ticker removed Phase 6 — /allocate replaced by /cash


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
    **extras,
) -> None:
    """Regista callbacks essenciais e arranca o listener Telegram.

    Args:
        send_fn: função send_telegram(text) — usada por todos os _reply().
        analyze_fn: usada pelo /analisar.
        clear_alerts_fn: usada por /admin (futuramente).
        **extras: chaves adicionais injectadas em _poll_context
                  (forward-compat; permite passar context ad-hoc sem
                  alterar a assinatura).
    """
    global _cb_send_telegram, _cb_analyze_ticker, _cb_clear_alerts
    _cb_send_telegram  = send_fn
    _cb_analyze_ticker = analyze_fn
    _cb_clear_alerts   = clear_alerts_fn

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

# /comparar and /historico handlers moved to commands/analysis.py (Phase 8.2)
from commands.analysis import (
    handle_comparar as _handle_comparar_impl,
    handle_historico as _handle_historico_impl,
)


def _handle_comparar(symbols: list[str]) -> None:
    _handle_comparar_impl(symbols)


def _handle_historico(symbol: str) -> None:
    _handle_historico_impl(symbol)
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
                from ml.ml_predictor import _safe_load, _to_dict
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
            from data.data_feed import get_eod_prices
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

_VALID_BUCKETS_SYNC = ("CORE", "ROTATION", "MOMENTUM", "DIP")


def _handle_sync_portfolio(parts: list[str]) -> None:
    """/sync_portfolio — bulk-import existing holdings + optional bucket tag.

    Use case: you already own positions (from before the bot, or from a broker
    export). /buy registers AND debits liquidity (purchase semantics). This
    command registers WITHOUT touching liquidity — for migrating existing
    holdings — and assigns a strategy bucket so /cash, rotation engine, etc.
    know how each position should be managed.

    Uso:
      /sync_portfolio                                              → ver estado actual
      /sync_portfolio NVO:159.59:48.0:CORE                         → 1 posição com bucket
      /sync_portfolio LIF:10.16:32.6:MOMENTUM:2026-02-15           → MOMENTUM com entry_date
      /sync_portfolio NVO:159.59:48.0:CORE MSFT:5.82:336:CORE      → várias

    Formato:  TICKER:shares:preço_médio[:BUCKET[:ENTRY_DATE]]

    ENTRY_DATE (YYYY-MM-DD) é OPCIONAL mas RECOMENDADO para MOMENTUM/DIP —
    sem ela, o trailing_high anchora no current price (perde info do peak
    real que viste durante a tua posse). Com ela, fetcha max(price) desde
    entry_date e ancora corretamente. Para CORE não importa (sem auto-exit).

    Buckets:  CORE | ROTATION | MOMENTUM | DIP
      CORE     — hold-for-life quality compounders (NVO, MSFT, VICI, ...)
      ROTATION — geridos pelo sector rotation engine (SEC0, SXR8, IS3N, ...)
      MOMENTUM — DIP positions que recuperaram, aguardam exit signal
      DIP      — DIP positions ativas, aguardam recovery target

    Idempotente: re-correr para o mesmo ticker UPDATE o bucket sem duplicar.

    As env vars HOLDING_* foram DESCONTINUADAS — usa este comando ou /buy.
    """
    from portfolio.portfolio import get_positions, register_existing_position, _cache, _ensure_cache, _flush

    no_conviction = "--no-conviction" in parts
    update_avg    = "--update-avg" in parts  # allow overwrite of avg_price + shares
    entries = [p for p in parts[1:] if not p.startswith("--")]

    # Sem argumentos → mostrar estado actual agrupado por bucket
    if not entries:
        positions = get_positions()
        if not positions:
            _reply(
                "💼 *Carteira vazia.*\n\n"
                "Para bulk-import com buckets (sem afectar liquidez):\n"
                "`/sync_portfolio NVO:159.59:48.0:CORE SEC0:94:18:ROTATION`\n\n"
                "Formato: `TICKER:shares:preço_médio_usd:BUCKET`\n"
                "Buckets: `CORE | ROTATION | MOMENTUM | DIP`\n\n"
                "Para comprar (debita liquidez):\n"
                "`/buy TICKER PREÇO SHARES`"
            )
            return

        # Group by bucket (category field). Map legacy categories to buckets.
        def _normalise_bucket(cat: str) -> str:
            c = (cat or "").upper().strip()
            if c in _VALID_BUCKETS_SYNC:
                return c
            # Legacy mappings: ETF, Manual import → require user to update
            return "UNCATEGORIZED"

        by_bucket: dict[str, list] = {}
        total_cost = 0.0
        for sym in sorted(positions.keys()):
            pos = positions[sym]
            bucket = _normalise_bucket(pos.get("category", ""))
            by_bucket.setdefault(bucket, []).append((sym, pos))
            total_cost += pos.get("total_cost", 0)

        lines = [
            f"💼 *Carteira* — {len(positions)} posições, cost basis €{total_cost:.2f}",
            "",
        ]
        for bucket in ("CORE", "ROTATION", "MOMENTUM", "DIP", "UNCATEGORIZED"):
            if bucket not in by_bucket:
                continue
            entries_b = by_bucket[bucket]
            bucket_cost = sum(p.get("total_cost", 0) for _, p in entries_b)
            bucket_pct = (bucket_cost / total_cost * 100) if total_cost > 0 else 0.0
            lines.append(f"*{bucket}* — €{bucket_cost:.2f} ({bucket_pct:.1f}%)")
            for sym, pos in entries_b:
                lines.append(
                    f"  • `{sym}`: {pos.get('shares', 0):g} @ ${pos.get('avg_price', 0):.2f}"
                    f" (cost €{pos.get('total_cost', 0):.2f})"
                )
            lines.append("")

        if "UNCATEGORIZED" in by_bucket:
            lines.append(
                "_⚠️ Posições UNCATEGORIZED: re-corre `/sync_portfolio TICKER:shares:price:BUCKET` "
                "para classificar._"
            )
        _reply("\n".join(lines))
        return

    # Parse entries
    # Tuple: (ticker, shares, price, bucket, entry_date_iso_or_empty)
    parsed: list[tuple[str, float, float, str, str]] = []
    errors: list[str] = []
    for entry in entries:
        try:
            bits = entry.split(":")
            if len(bits) < 3:
                errors.append(f"`{entry}` — falta preço (TICKER:shares:preço[:BUCKET[:ENTRY_DATE]])")
                continue
            ticker = bits[0].upper().strip()
            shares = float(bits[1])
            price  = float(bits[2])
            if shares <= 0 or price <= 0:
                errors.append(f"`{entry}` — shares e preço têm de ser > 0")
                continue
            # Optional 4th token: bucket
            bucket = ""
            if len(bits) >= 4:
                bucket_candidate = bits[3].upper().strip()
                if bucket_candidate not in _VALID_BUCKETS_SYNC:
                    errors.append(
                        f"`{entry}` — bucket inválido `{bucket_candidate}`. "
                        f"Usa: {', '.join(_VALID_BUCKETS_SYNC)}"
                    )
                    continue
                bucket = bucket_candidate
            # Optional 5th token: entry_date (YYYY-MM-DD)
            entry_date = ""
            if len(bits) >= 5:
                from datetime import date as _date
                try:
                    entry_date = bits[4].strip()
                    # Validate format
                    _date.fromisoformat(entry_date)
                except ValueError:
                    errors.append(
                        f"`{entry}` — entry_date `{bits[4]}` inválido. "
                        f"Usa YYYY-MM-DD."
                    )
                    continue
            parsed.append((ticker, shares, price, bucket, entry_date))
        except (ValueError, IndexError):
            errors.append(f"`{entry}` — erro a parsear")

    if errors and not parsed:
        _reply("⚠️ *Sem entradas válidas:*\n" + "\n".join(f"  • {e}" for e in errors))
        return

    imported: list[tuple[str, float, float, str]] = []
    updated:  list[tuple[str, str]] = []
    already:  list[str] = []
    convictions_created: list[tuple[str, str]] = []
    convictions_updated: list[tuple[str, str, float]] = []  # (ticker, entry_date, trailing_high)
    conviction_errors:   list[tuple[str, str]] = []

    for ticker, shares, price, bucket, entry_date in parsed:
        try:
            result = register_existing_position(ticker, price, shares, category=bucket)
            if result["action"] == "imported":
                imported.append((ticker, shares, price, bucket or "(no bucket)"))
            else:
                # Position already in ledger. Two paths:
                # --update-avg: overwrite avg_price / shares (use when fixing
                #               currency mismatch or correcting a typo)
                # default:      keep existing avg_price (idempotent); only
                #               adjust bucket if changed
                _ensure_cache()
                existing = _cache["positions"].get(ticker)
                if existing is None:
                    already.append(ticker)
                    continue
                if update_avg:
                    old_avg = existing.get("avg_price", 0)
                    old_shares = existing.get("shares", 0)
                    existing["avg_price"]  = round(price, 4)
                    existing["shares"]     = round(shares, 6)
                    existing["total_cost"] = round(price * shares, 2)
                    if bucket and existing.get("category") != bucket:
                        existing["category"] = bucket
                    _flush()
                    updated.append((
                        ticker,
                        f"avg ${old_avg}→${price} | shares {old_shares}→{shares}"
                    ))
                elif bucket and existing.get("category") != bucket:
                    existing["category"] = bucket
                    _flush()
                    updated.append((ticker, bucket))
                else:
                    already.append(ticker)
        except Exception as e:
            errors.append(f"`{ticker}` — registo falhou: {e}")
            continue

        if no_conviction:
            continue
        if bucket == "ROTATION":
            continue
        try:
            from core import conviction_radar as cr
            deny = cr._get_deny_list()
            if ticker in deny or "." in ticker:
                continue

            current = cr._fetch_price(ticker, fallback=price)
            hist    = cr._fetch_price_history(ticker, period="6mo")
            if bucket and bucket in ("DIP", "MOMENTUM", "CORE"):
                auto_type = bucket
            else:
                auto_type = cr._classify_type_from_signals(ticker, current, hist)

            # Compute trailing_high anchored on entry_date when provided.
            # Without entry_date we use current price (safe default — the
            # daily monitor will update trailing_high as price climbs).
            trailing_high_val = current  # safe default
            if auto_type == "MOMENTUM" and entry_date and hist is not None \
                    and not hist.empty and "Close" in hist.columns:
                try:
                    import pandas as _pd
                    entry_ts = _pd.Timestamp(entry_date)
                    # tz-strip both sides so the comparison doesn't raise
                    if hist.index.tz is not None:
                        hist_naive = hist.copy()
                        hist_naive.index = hist_naive.index.tz_localize(None)
                    else:
                        hist_naive = hist
                    hist_since = hist_naive[hist_naive.index >= entry_ts]
                    if not hist_since.empty:
                        since_max = float(hist_since["Close"].max())
                        if since_max > 0:
                            trailing_high_val = max(since_max, current)
                except Exception:
                    pass

            existing_conv = cr.get_by_ticker(ticker)
            if existing_conv is not None:
                # Existing conviction: UPDATE in place.
                # • position_type if bucket override changed it
                # • entry_date if user provided (anchors trailing stop)
                # • trailing_high recomputed from new entry_date
                # • entry_price ONLY when --update-avg flag set (used when
                #   fixing currency mismatch or correcting a typo)
                changed = False
                if existing_conv.position_type != auto_type:
                    existing_conv.position_type = auto_type
                    changed = True
                if entry_date:
                    iso_date = entry_date
                    if getattr(existing_conv, "entry_date", "") != iso_date:
                        existing_conv.entry_date = iso_date
                        changed = True
                if update_avg and abs(existing_conv.entry_price - price) > 0.001:
                    existing_conv.entry_price = float(price)
                    existing_conv.shares = float(shares)
                    changed = True
                if auto_type == "MOMENTUM" and trailing_high_val:
                    if abs(float(existing_conv.trailing_high or 0) - trailing_high_val) > 0.01:
                        existing_conv.trailing_high = float(trailing_high_val)
                        changed = True
                if changed:
                    cr.update_conviction(existing_conv)
                    convictions_updated.append(
                        (ticker, entry_date or "(unchanged)",
                         float(existing_conv.trailing_high or 0))
                    )
                continue  # done with this ticker, skip add_conviction below

            cr.add_conviction(
                ticker, price, shares, auto_type,
                trailing_high = trailing_high_val,
                note          = "imported via /sync_portfolio",
            )
            convictions_created.append((ticker, auto_type))
        except Exception as e:
            conviction_errors.append((ticker, str(e)))

    # Build reply
    lines = ["💼 *Bulk-import resultado*"]
    if imported:
        lines.append(f"\n✅ *{len(imported)} posições importadas* (sem alteração de liquidez):")
        for t, s, p, b in imported:
            bucket_tag = f" → `{b}`" if b and b != "(no bucket)" else ""
            lines.append(f"  • *{t}*: {s:g} @ ${p:.2f}{bucket_tag}")
    if updated:
        lines.append(f"\n🔄 *{len(updated)} buckets actualizados*:")
        for t, b in updated:
            lines.append(f"  • *{t}* → `{b}`")
    if already:
        lines.append(f"\n♻️ *{len(already)} já no ledger* (sem alteração):")
        lines.append("  " + ", ".join(already))
    if convictions_created:
        lines.append(f"\n🧠 *{len(convictions_created)} convicções auto-criadas:*")
        for t, tp in convictions_created:
            lines.append(f"  • {t} → {tp}")
    if convictions_updated:
        lines.append(f"\n🔄 *{len(convictions_updated)} convicções actualizadas:*")
        # Display trailing_high in native currency. yfinance returns USD
        # for US-listed tickers, so the stored trailing_high is USD —
        # display with $. (Currency awareness is per-ticker; UCITS would
        # be EUR, but those are ROTATION bucket and skipped here.)
        for t, ed, th in convictions_updated:
            th_str = f"${th:.2f}" if th > 0 else "n/a"
            lines.append(f"  • `{t}` entry_date `{ed}` | trailing_high {th_str}")
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
            from ml.ml_features import get_feature_columns
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

    # Targets (always show first if present). 60d removed 2026-06-12 (uniformized 90d).
    target_cols = ["alpha_30d", "alpha_90d", "alpha_180d",
                   "alpha_1y", "alpha_2y", "alpha_3y",
                   "forward_return_30d", "forward_return_90d", "forward_return_180d",
                   "max_drawdown_90d"]
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
        ("FLIP_FUND_EUR",           "Orçamento mensal",      True,  "Ex: 500 — usado para sizing por trade (env var keeps legacy name)"),
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
        from ml.ml_predictor import is_model_ready, get_model_info
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
        from core import core_radar as _cr
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
        f"Output: {'/data/test/' if is_test else '/data/'}momentum\\_training.parquet\n"
        f"Duração: {dur_str}\n"
        f"Podes continuar a usar o bot."
    )

    def _run():
        import subprocess, sys, shutil
        from pathlib import Path
        from utils.dataset_paths import resolve_dataset_path, is_test_mode
        try:
            script   = Path(__file__).parent / "momentum_radar" / "scripts" / "build_dataset.py"
            is_test  = is_test_mode(ticker_val)
            out_path = resolve_dataset_path("momentum_training.parquet", is_test=is_test)
            # Seed a partir do repo só em produção. Em test mode, parquet vazio
            # é o estado correcto — não queremos arrastar contexto de produção.
            if not is_test and not out_path.exists() and not force_rebuild:
                repo_seed = Path(__file__).parent / "momentum_radar" / "momentum_training.parquet"
                if repo_seed.exists():
                    shutil.copy2(str(repo_seed), str(out_path))
                    logging.info(f"[momentum_dataset] Seed de {repo_seed} → {out_path}")
            cmd      = [sys.executable, str(script), "--start", start_val, "--out", str(out_path)]
            if is_test:
                cmd += ["--tickers", ticker_val]
            if force_rebuild:
                cmd += ["--force-rebuild"]

            _TIMEOUT = 8 * 3600  # 8h
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT)

            if result.returncode == 0:
                try:
                    import pandas as pd
                    df = pd.read_parquet(out_path)
                    # Copiar para repo SÓ em produção. Em test mode, copiar o
                    # smoke-test parquet para o repo envenenaria o seed de
                    # deploys futuros — exactamente o que aconteceu antes deste fix.
                    if not is_test:
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
    """/admin_momentum_retrain [force]

    Treina o Breakout ML com todos os dados históricos acumulados.
    Corre automaticamente no dia 1 de cada mês (após o DipRadar retrain).

    Modes:
      /admin_momentum_retrain        → incremental (~20-40 min)
        - lê parquet existente, computa só últimos ~220 dias
        - preserva todo o histórico anterior intacto

      /admin_momentum_retrain force  → FORCE-REBUILD (~1-3 horas)
        - reconstrói parquet do zero
        - aplica fixes recentes (ex: alpha horizon trading-day fix em 7a6f000)
          a TODOS os históricos
        - usar quando há mudança no schema ou bug fix no target

    Pipeline (ambos os modos):
      1. Build dataset (incremental ou force)
      2. Treina LightGBM + sector models
      3. Gating: promove só se IC ≥ IC actual × 0.90
    """
    force_rebuild = len(parts) >= 2 and parts[1].lower() in ("force", "rebuild", "--force-rebuild")

    if force_rebuild:
        _reply(
            "*MomentumRadar — FORCE REBUILD*\n"
            "1. ⚠️ Reconstruir parquet do zero\n"
            "2. Treinar LightGBM + sector models\n"
            "3. Gating vs modelo actual\n"
            "Duração: ~1-3 horas\n"
            "Podes continuar a usar o bot."
        )
    else:
        _reply(
            "*MomentumRadar — Retrain*\n"
            "1. Actualizar dataset (incremental)\n"
            "2. Treinar LightGBM + sector models\n"
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

            # Passo 1: actualizar dataset.
            # 12h timeout — momentum dataset now queries EDGAR per episode for
            # the 4 insider features (commit 0c4703c), which dramatically
            # slowed the bulk build. Incremental updates are still fast.
            script = Path(__file__).parent / "momentum_radar" / "scripts" / "build_dataset.py"
            cmd = [sys.executable, str(script), "--out", str(ds_path)]
            if force_rebuild:
                cmd.append("--force-rebuild")
            result = subprocess.run(
                cmd,
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
                    from utils.github_sync import sync_momentum_artifacts
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
        f"Output: {'/data/test/' if is_test else '/data/'}ml\\_training\\_base.parquet\n"
        + ("Fundamentais: não\n" if no_fund else "") +
        f"Duração: {dur_str}\n"
        f"Podes continuar a usar o bot."
    )

    def _run():
        import subprocess, sys, shutil
        from pathlib import Path
        from utils.dataset_paths import resolve_dataset_path, is_test_mode
        try:
            script   = Path(__file__).parent / "scripts" / "build_dip_dataset.py"
            is_test  = is_test_mode(ticker_val)
            out_path = resolve_dataset_path("ml_training_base.parquet", is_test=is_test)
            # Seed a partir do repo só em produção. Em test mode, parquet vazio
            # é o estado correcto — não queremos arrastar contexto de produção.
            if not is_test and not out_path.exists() and not force_rebuild:
                repo_seed = Path(__file__).parent / "ml_training_base.parquet"
                if repo_seed.exists():
                    shutil.copy2(str(repo_seed), str(out_path))
                    logging.info(f"[dip_dataset] Seed de {repo_seed} → {out_path}")
            cmd      = [sys.executable, str(script), "--start", start_val, "--out", str(out_path)]
            if is_test:
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
                    # Copiar para repo SÓ em produção. Em test mode, copiar o
                    # smoke-test parquet para o repo envenenaria o seed de
                    # deploys futuros — exactamente o que aconteceu antes deste fix.
                    if not is_test:
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
        from utils.dataset_paths import production_dataset_path

        global _pipeline_log
        _pipeline_log.clear()
        _plog("CoreML dataset build iniciado")

        # Standalone /admin_build_core_dataset is always a full production
        # build — there is no --test surface here. Test mode goes through
        # /admin_run_pipeline --test which uses resolve_dataset_path(...).
        out_path = production_dataset_path("core_training_base.parquet")
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
            from core import core_radar as _cr
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
                from ml.monthly_retrain import run_monthly_retrain
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
            from utils.github_sync import sync_training_artifacts, sync_momentum_artifacts
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
      --incremental    Use INCREMENTAL builds — don't pass --force-rebuild to
                       dataset builders, so they only process new dates since
                       the last record in the parquet. 10× faster than full
                       rebuild. Implies --no-purge automatically. RECOMMENDED
                       FOR MONTHLY RETRAIN. Use when you have a healthy
                       parquet and just want to add new month's data.
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
    # --incremental: pass NO --force-rebuild to dataset builders, so they
    # detect existing parquet and only process new dates since last record.
    # 10× faster than full rebuild. Perfect for monthly retrain cadence.
    # Implies --no-purge automatically (can't keep parquets if purge wipes them).
    incremental    = "--incremental"    in parts
    if incremental:
        no_purge = True  # forced

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
                _dip_cmd = [sys.executable, str(scripts / "build_dip_dataset.py"),
                            "--start", "1993-01-01", "--out", str(out_dip)]
                if not incremental:
                    _dip_cmd.append("--force-rebuild")
                ok, tail = _sub(
                    _dip_cmd,
                    phase="Step 1/6 — DIP dataset build" + (" (incremental)" if incremental else ""),
                    # Incremental builds finish in 10-30 min; full rebuild in 4-18h.
                    timeout_h=2 if incremental else 18,
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
                        from ml.monthly_retrain import run_monthly_retrain
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
                _core_cmd = [sys.executable, str(scripts / "build_core_dataset.py"),
                             "--out", str(out_core), "--start", "1993-01-01"]
                if not incremental:
                    _core_cmd.append("--force-rebuild")
                ok, tail = _sub(
                    _core_cmd,
                    phase="Step 3/6 — Core dataset build" + (" (incremental)" if incremental else ""),
                    timeout_h=1 if incremental else 10,
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
                            from core import core_radar as _cr
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
                _mom_cmd = [sys.executable, str(mom_script), "--out", str(out_mom)]
                if not incremental:
                    _mom_cmd.append("--force-rebuild")
                ok, tail = _sub(
                    _mom_cmd,
                    phase="Step 5/6 — Momentum dataset build" + (" (incremental)" if incremental else ""),
                    # Incremental: ~30-90 min. Full rebuild: 6-12h.
                    timeout_h=2 if incremental else 12,
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

        from utils.dataset_paths import resolve_dataset_path, is_test_mode
        # `test_tickers` carries the `--test T1,T2` value through the whole
        # pipeline. When set, every parquet write goes to /data/test/ instead
        # of /data/, and the retrain/github-sync paths below skip persistence
        # so a smoke test cannot promote a 2-ticker model or push test data
        # to GitHub. Production rebuilds (test_tickers=None) are unaffected.
        is_test = is_test_mode(test_tickers)

        def _build_dip():
            nonlocal dip_ds_ok, dip_ds_err
            script = Path(__file__).parent / "scripts" / "build_dip_dataset.py"
            out    = resolve_dataset_path("ml_training_base.parquet", is_test=is_test)
            cmd    = [sys.executable, str(script), "--start", "1993-01-01", "--out", str(out)]
            if is_test:      cmd += ["--tickers", test_tickers]
            if force:        cmd += ["--force-rebuild", "--clear-cache"]
            rc, lines = _run_script_streaming(cmd, "Phase 1 — DIP dataset")
            if rc == 0:
                dip_ds_ok = True
            else:
                dip_ds_err = next((l for l in reversed(lines) if "error" in l.lower()), lines[-1] if lines else "")[:120]

        def _build_momentum():
            nonlocal mom_ds_ok, mom_ds_err
            script = Path(__file__).parent / "momentum_radar" / "scripts" / "build_dataset.py"
            out    = resolve_dataset_path("momentum_training.parquet", is_test=is_test)
            cmd    = [sys.executable, str(script), "--start", "1993-01-01", "--out", str(out)]
            if is_test:      cmd += ["--tickers", test_tickers]
            if force:        cmd += ["--force-rebuild", "--clear-cache"]
            rc, lines = _run_script_streaming(cmd, "Phase 1 — Momentum dataset")
            if rc == 0:
                mom_ds_ok = True
            else:
                mom_ds_err = next((l for l in reversed(lines) if "error" in l.lower()), lines[-1] if lines else "")[:120]

        def _build_core():
            nonlocal core_ds_ok, core_ds_err
            script = Path(__file__).parent / "scripts" / "build_core_dataset.py"
            out    = resolve_dataset_path("core_training_base.parquet", is_test=is_test)
            cmd    = [sys.executable, str(script), "--start", "1993-01-01", "--out", str(out)]
            if is_test:      cmd += ["--tickers", test_tickers]
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

        # Test mode: smoke test verified dataset builds finish. Don't retrain
        # — a 2-ticker model is useless and the retrain path reads from the
        # production parquet (run_monthly_retrain has no test-path knob), so
        # letting it run would either error out or train on stale production
        # data and (worse) promote it. Stop here cleanly.
        if is_test:
            elapsed = round(_t.time() - start_wall)
            _plog("Phase 2 — skipped (test mode: datasets built in /data/test/, retrain not run)")
            _reply(
                f"Smoke test OK in {elapsed}s.\n"
                f"Parquets written to /data/test/ (production untouched).\n"
                f"Phase 2 retrain skipped — a 2-ticker model is useless. "
                f"Re-run without `--test` for a real retrain."
            )
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
                from ml.monthly_retrain import run_monthly_retrain
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
            from utils.github_sync import sync_training_artifacts, sync_momentum_artifacts
            if train_dip and dip_result.get("decision") == "PROMOTED":
                sync_training_artifacts(data_dir)
            if train_mom and mom_result.get("decision") == "PROMOTED":
                sync_momentum_artifacts(data_dir)
        except Exception as e:
            logging.warning(f"[run_pipeline] github_sync: {e}")

        # ── auto-publish charts after successful retrain ───────────────────────
        try:
            from utils.chart_generator import generate_all_charts
            generate_all_charts()
            if os.environ.get("GITHUB_TOKEN") and os.environ.get("GITHUB_CHARTS_REPO"):
                from utils.github_publisher import push_all_charts
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
            from ml.monthly_retrain import run_monthly_retrain
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
        from ml.monthly_retrain import set_floor_rho_alpha
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
# /themes, /add_theme, /remove_theme handlers moved to commands/themes.py
# (Phase 8 split of bot_commands.py monolith)
from commands.themes import (
    handle_themes as _handle_themes_impl,
    handle_add_theme as _handle_add_theme_impl,
    handle_remove_theme as _handle_remove_theme_impl,
)


# Misc handlers (mldata, ml_accuracy, performance, paper_performance,
# health) moved to commands/misc.py (Phase 8.5)
from commands.misc import (
    handle_mldata as _handle_mldata_impl,
    handle_ml_accuracy as _handle_ml_accuracy_impl,
    handle_paper_performance as _handle_paper_performance_impl,
    handle_performance as _handle_performance_impl,
    handle_health as _handle_health_impl,
)


# Core handlers (core, core_add, core_close, core_universe, core_universe_add,
# core_universe_remove) moved to commands/core_cmds.py (Phase 8.6)
from commands.core_cmds import (
    handle_core as _handle_core_impl,
    handle_core_add as _handle_core_add_impl,
    handle_core_close as _handle_core_close_impl,
    handle_core_universe as _handle_core_universe_impl,
    handle_core_universe_add as _handle_core_universe_add_impl,
    handle_core_universe_remove as _handle_core_universe_remove_impl,
)


# Portfolio handlers (buy, sell, ppr, etf, liquidez, portfolio) — Phase 8.4
# extracted these to commands/portfolio_extras.py but the import block was
# never added to bot_commands.py, so every /buy, /sell, /ppr, /etf, /liquidez
# and /portfolio call NameErrored in production. handle_portfolio itself was
# orphaned mid-extraction (body stranded inside _handle_liquidez); the proper
# extraction lives in portfolio_extras now.
from commands.portfolio_extras import (
    handle_buy       as _handle_buy_impl,
    handle_sell      as _handle_sell_impl,
    handle_ppr       as _handle_ppr_impl,
    handle_etf       as _handle_etf_impl,
    handle_liquidez  as _handle_liquidez_impl,
    handle_portfolio as _handle_portfolio_impl,
)


# Conviction handler — Phase 8.3 extracted but import was also missed.
from commands.conviction import (
    handle_conviction as _handle_conviction_impl,
)


def _handle_core(parts: list[str]) -> None:
    _handle_core_impl(parts)


def _handle_core_add(parts: list[str]) -> None:
    _handle_core_add_impl(parts)


def _handle_core_close(parts: list[str]) -> None:
    _handle_core_close_impl(parts)


def _handle_core_universe(parts: list[str]) -> None:
    _handle_core_universe_impl(parts)


def _handle_core_universe_add(parts: list[str]) -> None:
    _handle_core_universe_add_impl(parts)


def _handle_core_universe_remove(parts: list[str]) -> None:
    _handle_core_universe_remove_impl(parts)



def _handle_mldata(force_update: bool = False) -> None:
    _handle_mldata_impl(force_update)


def _handle_ml_accuracy() -> None:
    _handle_ml_accuracy_impl()


def _handle_paper_performance(parts: list[str]) -> None:
    _handle_paper_performance_impl(parts)


def _handle_performance(parts: list[str]) -> None:
    _handle_performance_impl(parts)


def _handle_health(parts: list[str]) -> None:
    _handle_health_impl(parts)


def _handle_themes() -> None:
    _handle_themes_impl()


def _handle_add_theme(parts: list[str]) -> None:
    _handle_add_theme_impl(parts)


def _handle_remove_theme(parts: list[str]) -> None:
    _handle_remove_theme_impl(parts)
# ── Earnings alerts ───────────────────────────────────────────────────────────────

def send_earnings_alerts(watchlist: list[str] | None = None) -> int:
    earnings_days_fn = _poll_context.get("get_earnings_days") or _cb_earnings_days
    if not earnings_days_fn or not (_poll_context.get("send_fn") or _cb_send_telegram):
        return 0
    all_tickers = list(dict.fromkeys(watchlist)) if watchlist else []
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
def _handle_buy(parts: list[str]) -> None:
    _handle_buy_impl(parts)


def _handle_sell(parts: list[str]) -> None:
    _handle_sell_impl(parts)


def _handle_ppr(parts: list[str]) -> None:
    _handle_ppr_impl(parts)


def _handle_etf(parts: list[str]) -> None:
    _handle_etf_impl(parts)


def _handle_liquidez(parts: list[str]) -> None:
    _handle_liquidez_impl(parts)


def _handle_portfolio(parts: list[str]) -> None:
    _handle_portfolio_impl(parts)


def _handle_conviction(parts: list[str]) -> None:
    _handle_conviction_impl(parts)


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
        # Flip log removed Phase 6 — Flip Fund deprecated
        flip_str   = ""
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
            from portfolio.portfolio import get_liquidity, get_active_symbols
            liq      = get_liquidity()
            n_pos    = len(get_active_symbols())
            port_str = f"\nCarteira: *{n_pos} posições* | Liquidez: *€{liq:.2f}*"
        except Exception:
            port_str = ""

        _reply(
            f"*🤖 DipRadar Status*\n"
            f"Uptime: *{hours}h {mins}m* | Mercado: *{market}*\n"
            f"{flip_str}{db_str}\n"
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
            from portfolio.paper_trading import format_unified_report
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

    # /allocate removed in Phase 6 — use /cash for unified allocation

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

    elif cmd in ("/admin_dip_fill_missing", "/admin_fill_missing"):
        # Generic fill-missing: applies to DIP, Core, or Momentum parquets.
        # Identifies tickers in the universe but NOT yet in the parquet, then
        # processes only those with full history. Final merge is concat +
        # dedup, preferring source="real" over "simulated" on conflicts.
        # Usage:
        #   /admin_fill_missing dip       (or /admin_dip_fill_missing — back-compat)
        #   /admin_fill_missing core
        #   /admin_fill_missing momentum

        # Determine which model
        if cmd == "/admin_dip_fill_missing":
            model = "dip"
        elif len(parts) > 1:
            model = parts[1].lower().strip()
        else:
            _reply(
                "🔍 *Fill missing tickers*\n\n"
                "Uso: `/admin_fill_missing <model>`\n"
                "  • `dip`      — completa `ml_training_base.parquet`\n"
                "  • `core`     — completa `core_training_base.parquet`\n"
                "  • `momentum` — completa `momentum_training.parquet`\n\n"
                "_Útil quando um `/admin_overnight` parou a meio e quer-se acabar "
                "só o que falta, sem refazer tudo._"
            )
            return

        if model not in ("dip", "core", "momentum"):
            _reply(f"⚠️ Modelo inválido `{model}`. Usa: `dip`, `core`, `momentum`.")
            return

        # Per-model configuration
        def _load_universe_for(m: str) -> set[str]:
            if m in ("dip", "momentum"):
                from core.universe import get_ml_universe
                return set(get_ml_universe())
            elif m == "core":
                # Read core_universe.json directly (avoids subprocess hop)
                import json as _json
                from pathlib import Path as _P
                _uni_file = _P(__file__).parent / "core_universe.json"
                if not _uni_file.exists():
                    return set()
                try:
                    data = _json.loads(_uni_file.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        # may be {"tickers": [...]} or {ticker: {...}}
                        if "tickers" in data and isinstance(data["tickers"], list):
                            return {str(t).upper() for t in data["tickers"]}
                        return {str(t).upper() for t in data.keys()}
                    if isinstance(data, list):
                        return {str(t).upper() for t in data}
                except Exception:
                    return set()
            return set()

        from pathlib import Path as _P
        configs = {
            "dip": {
                "parquet_name": "ml_training_base.parquet",
                "script":       _P(__file__).parent / "scripts" / "build_dip_dataset.py",
                "extra_args":   ["--start", "1993-01-01"],
                "dedup_keys":   ["ticker", "alert_date"],
                "date_col":     "alert_date",
                "retrain_hint": "/admin_retrain",
            },
            "core": {
                "parquet_name": "core_training_base.parquet",
                "script":       _P(__file__).parent / "scripts" / "build_core_dataset.py",
                "extra_args":   ["--start", "1993-01-01"],
                "dedup_keys":   ["ticker", "date"],
                "date_col":     "date",
                "retrain_hint": "/admin_overnight --no-purge --core-only",
            },
            "momentum": {
                "parquet_name": "momentum_training.parquet",
                "script":       _P(__file__).parent / "momentum_radar" / "scripts" / "build_dataset.py",
                "extra_args":   [],
                "dedup_keys":   ["ticker", "date"],
                "date_col":     "date",
                "retrain_hint": "/admin_overnight --no-purge --momentum-only",
            },
        }
        cfg = configs[model]

        try:
            import pandas as _pd
            _dd = _P("/data") if _P("/data").exists() else _P("/tmp")
            parquet = _dd / cfg["parquet_name"]
            if not parquet.exists():
                _reply(
                    f"⚠️ `{parquet.name}` não existe. Corre `/admin_overnight --{model}-only` "
                    f"primeiro para criar a base."
                )
                return

            _df = _pd.read_parquet(parquet)
            existing_tickers = set(_df["ticker"].unique()) if "ticker" in _df.columns else set()
            universe = _load_universe_for(model)
            if not universe:
                _reply(f"⚠️ Universo {model} vazio — impossível determinar missing.")
                return
            missing = sorted(universe - existing_tickers)

            if not missing:
                _reply(
                    f"✅ Parquet *{model.upper()}* já está completo.\n"
                    f"  Tickers no parquet: *{len(existing_tickers)}*\n"
                    f"  Universo: *{len(universe)}*"
                )
                return

            _reply(
                f"🔍 *Fill missing — {model.upper()}*\n\n"
                f"  No parquet: *{len(existing_tickers)}* tickers ({len(_df):,} rows)\n"
                f"  Universo: *{len(universe)}* tickers\n"
                f"  *{len(missing)} em falta* — vou processar agora.\n\n"
                f"_Tickers: {', '.join(missing[:20])}{'...' if len(missing) > 20 else ''}_\n\n"
                f"⏳ Estimativa: ~{max(5, len(missing)*2)} min. Notificação quando terminar."
            )

            def _run_fill():
                import subprocess as _sp, sys as _sys
                import time as _t
                t0 = _t.time()
                tmp_out = _dd / f"{cfg['parquet_name'].rsplit('.', 1)[0]}.fill.parquet"
                cmd = [
                    _sys.executable, str(cfg["script"]),
                    "--tickers", ",".join(missing),
                    "--out", str(tmp_out),
                    *cfg["extra_args"],
                ]
                try:
                    proc = _sp.run(cmd, capture_output=True, text=True, timeout=4 * 3600)
                except _sp.TimeoutExpired:
                    _reply(f"❌ Fill {model} timed out após 4h. Logs no Railway.")
                    return
                except Exception as e:
                    _reply(f"❌ Fill {model} falhou: `{e}`")
                    return

                if proc.returncode != 0:
                    tail = ((proc.stdout or "") + (proc.stderr or ""))[-400:]
                    _reply(f"❌ Build {model} dos missing falhou (exit {proc.returncode}):\n<code>{tail}</code>")
                    return

                if not tmp_out.exists():
                    _reply(f"⚠️ Build {model} OK mas `{tmp_out.name}` não foi escrito. Sem dados novos para merge.")
                    return

                try:
                    new_df = _pd.read_parquet(tmp_out)
                    existing_df = _pd.read_parquet(parquet)

                    # source column: ensure it exists, default existing=real, new=simulated
                    if "source" not in existing_df.columns:
                        existing_df["source"] = "real"
                    if "source" not in new_df.columns:
                        new_df["source"] = "simulated"

                    combined = _pd.concat([existing_df, new_df], ignore_index=True)
                    combined["_sort"] = (combined["source"] == "real").astype(int)
                    combined = combined.sort_values("_sort", ascending=False).drop(columns=["_sort"])

                    # Dedup keys differ per model (DIP=alert_date, others=date)
                    dedup_keys = [k for k in cfg["dedup_keys"] if k in combined.columns]
                    if not dedup_keys:
                        dedup_keys = ["ticker"]
                    combined = combined.drop_duplicates(subset=dedup_keys, keep="first")

                    sort_col = cfg["date_col"] if cfg["date_col"] in combined.columns else dedup_keys[0]
                    combined = combined.sort_values(sort_col).reset_index(drop=True)

                    n_before = len(existing_df)
                    n_after  = len(combined)
                    n_new    = n_after - n_before
                    new_tickers_actual = set(new_df["ticker"].unique()) - existing_tickers

                    combined.to_parquet(parquet, index=False)
                    tmp_out.unlink()

                    elapsed = (_t.time() - t0) / 60
                    _reply(
                        f"✅ *Fill missing — {model.upper()} concluído* ({elapsed:.0f} min)\n\n"
                        f"  Rows antes: *{n_before:,}*\n"
                        f"  Rows depois: *{n_after:,}*\n"
                        f"  Novos: *+{n_new:,}*\n"
                        f"  Tickers novos: *{len(new_tickers_actual)}/{len(missing)}*\n"
                        f"  ({len(missing) - len(new_tickers_actual)} sem dados — provavelmente delisted)\n\n"
                        f"_Próximo passo: `{cfg['retrain_hint']}` para treinar o modelo no parquet completo._"
                    )
                except Exception as e:
                    _reply(f"❌ Merge {model} falhou: `{e}`")

            threading.Thread(target=_run_fill, daemon=True, name=f"{model}-fill-missing").start()
        except Exception as e:
            _reply(f"❌ Erro: `{e}`")

    elif cmd in ("/admin_test_retrain", "/admin_dry_retrain", "/admin_micro_test"):
        # Quick health check before committing to 8-12h of full retrain.
        # Runs in 3-8 min:
        #   1. Schema check — all FEATURE_COLUMNS present
        #   2. NaN coverage report (per-column real vs NaN)
        #   3. Distribution sanity (no Inf, std > 0, mean within sane range)
        #   4. Trainer dry run on 5% sample with 2 folds — verifies pipeline
        #      end-to-end without producing a real model
        #   5. Time estimate for full retrain based on sample timing
        try:
            from pathlib import Path as _P
            import pandas as _pd
            import math as _math
            import time as _time

            _dd = _P("/data") if _P("/data").exists() else _P("/tmp")
            parquet = _dd / "ml_training_base.parquet"
            if not parquet.exists():
                _reply(f"⚠️ `{parquet.name}` não existe. Corre `/admin_overnight` primeiro.")
                return

            _reply(
                f"🧪 *Micro-test* do retrain a iniciar\n\n"
                f"  Schema check + NaN report + 5% sample training + ETA full\n"
                f"  Estimativa: 3-8 min\n\n"
                f"_Sem pickle gerado, sem alteração no parquet._"
            )

            def _run_test():
                t0 = _time.time()
                lines = ["🧪 *Micro-test retrain — resultado*\n"]
                try:
                    df = _pd.read_parquet(parquet)
                except Exception as e:
                    _reply(f"❌ Parquet load falhou: `{e}`")
                    return

                lines.append(f"📊 *Parquet:* {len(df):,} rows × {len(df.columns)} cols")
                lines.append(f"   Range: {df['alert_date'].min()} → {df['alert_date'].max()}")
                lines.append("")

                # 1. Schema check
                try:
                    from ml.ml_features import FEATURE_COLUMNS
                    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
                    if missing:
                        lines.append(f"❌ *Schema check FAIL:* {len(missing)} cols em falta:")
                        for c in missing[:10]:
                            lines.append(f"   • {c}")
                        _reply("\n".join(lines))
                        return
                    else:
                        lines.append(f"✅ *Schema check:* {len(FEATURE_COLUMNS)} features presentes")
                except Exception as e:
                    lines.append(f"⚠️ Schema check error: `{e}`")

                # 2. NaN coverage + distribution sanity
                lines.append("")
                lines.append("📉 *NaN coverage + sanidade:*")
                problems_severe = []
                problems_warn = []
                from ml.ml_features import FEATURE_COLUMNS as _FC
                for col in _FC:
                    if col not in df.columns:
                        continue
                    series = df[col]
                    n_real = int(series.notna().sum())
                    pct_real = n_real / len(df) * 100
                    if n_real == 0:
                        problems_severe.append(f"{col}: 0% real (dead column)")
                        continue
                    valid = series.dropna()
                    # Check for Inf
                    if _math.isinf(float(valid.min() if len(valid) else 0)) or _math.isinf(float(valid.max() if len(valid) else 0)):
                        problems_severe.append(f"{col}: contains Inf")
                    # Check for zero variance
                    elif len(valid) >= 2 and valid.std() == 0:
                        problems_severe.append(f"{col}: std=0 ({valid.iloc[0]} constante) — likely broken")
                    elif pct_real < 5:
                        problems_warn.append(f"{col}: {pct_real:.1f}% real (very sparse)")

                if problems_severe:
                    lines.append(f"❌ *{len(problems_severe)} features problemáticas:*")
                    for p in problems_severe[:10]:
                        lines.append(f"   • {p}")
                else:
                    lines.append("✅ Sem features com 0% real ou std=0")
                if problems_warn:
                    lines.append(f"⚠️ *{len(problems_warn)} sparse features (<5% real):*")
                    for p in problems_warn[:5]:
                        lines.append(f"   • {p}")

                # 3. Sample trainer dry run (5% rows, 2 folds, fast)
                lines.append("")
                lines.append("🔬 *Trainer dry run* (5% sample, 2 folds, no pickle)...")
                sample_n = max(1000, int(len(df) * 0.05))
                sample_n = min(sample_n, 10000)  # cap
                t_train_start = _time.time()
                try:
                    from ml_training.train import run_training
                    result = run_training(
                        input_parquet=parquet,
                        output_bundle=None,   # NO write
                        output_report=None,
                        n_folds=2,            # vs 10
                        use_ensemble_champion=False,
                        use_sector_neutral=False,
                        max_rows=sample_n,
                        log_summary=False,
                    ) or {}
                    train_elapsed = _time.time() - t_train_start
                    decision = result.get("decision", "?")
                    rho = result.get("candidate_rho_alpha")
                    icon = "✅" if decision in ("PROMOTED", "KEPT") else "⚠️"
                    rho_str = f" | ρ_alpha={rho:.4f}" if isinstance(rho, float) else ""
                    lines.append(
                        f"   {icon} Trainer corre limpo em {train_elapsed:.0f}s "
                        f"({sample_n:,} rows × 2 folds){rho_str}"
                    )
                    # Estimate full run time
                    full_rows = len(df)
                    # Heuristic: training scales ~O(N) per fold; 10 folds vs 2 → 5x
                    # ensemble champion adds ~2x; full universe so multiply
                    est_full_minutes = (train_elapsed / 60) * (full_rows / sample_n) * (10/2) * 2.0
                    lines.append(f"   📐 ETA full retrain: ~{est_full_minutes:.0f} min ({est_full_minutes/60:.1f}h)")
                except Exception as e:
                    train_elapsed = _time.time() - t_train_start
                    lines.append(f"   ❌ Trainer FALHOU após {train_elapsed:.0f}s:")
                    lines.append(f"      <code>{str(e)[:200]}</code>")
                    lines.append("   _NÃO disparar /admin_retrain antes de investigar o erro acima._")

                # 4. Verdict
                lines.append("")
                if problems_severe or "FALHOU" in "\n".join(lines):
                    lines.append("🔴 *Veredicto: NÃO RECOMENDA retrain completo*")
                    lines.append("   Corrige os problemas acima antes de comprometer com 12h.")
                elif problems_warn:
                    lines.append("🟡 *Veredicto: OK COM RESSALVAS*")
                    lines.append("   Sparse features vão funcionar mas podem reduzir IC.")
                else:
                    lines.append("🟢 *Veredicto: SEGURO disparar full retrain*")
                    lines.append("   Parquet saudável, pipeline limpo, sem flags vermelhos.")

                total_elapsed = (_time.time() - t0) / 60
                lines.append(f"\n_Test demorou {total_elapsed:.1f} min._")
                _reply("\n".join(lines))

            threading.Thread(target=_run_test, daemon=True, name="micro-test-retrain").start()
        except Exception as e:
            _reply(f"❌ Micro-test falhou: `{e}`")

    elif cmd in ("/admin_backfill_fundamentals", "/admin_backfill"):
        # Recompute ONLY the fundamental columns in the DIP parquet, in place,
        # without re-fetching yfinance OHLCV or re-detecting episodes. Updates:
        # gross_margin, de_ratio, roe, revenue_growth, earnings_growth,
        # _eps_ttm, trailing_pe, peg_ratio, earnings_beat_rate, analyst_rating,
        # fcf_yield, quality_score. Everything else (RSI, drawdown, vix,
        # alpha_90d targets, etc.) is preserved untouched.
        try:
            from pathlib import Path as _P
            _dd = _P("/data") if _P("/data").exists() else _P("/tmp")
            parquet = _dd / "ml_training_base.parquet"
            if not parquet.exists():
                _reply(f"⚠️ `{parquet.name}` não existe — corre `/admin_overnight` primeiro.")
                return

            # Optional filters
            since_arg = None
            tickers_arg = None
            max_rows_arg = None
            for token in parts[1:]:
                if token.startswith("--since="):
                    since_arg = token.split("=", 1)[1]
                elif token.startswith("--tickers="):
                    tickers_arg = token.split("=", 1)[1]
                elif token.startswith("--max-rows="):
                    max_rows_arg = token.split("=", 1)[1]

            _reply(
                f"🔄 *Backfill fundamentals* a iniciar\n\n"
                f"  Parquet: `{parquet.name}` ({parquet.stat().st_size//1024//1024} MB)\n"
                f"  EDGAR PIT TTM + YoY para `revenue_growth`, `earnings_growth`, `_eps_ttm`\n"
                f"  yfinance para `earnings_beat_rate`, `analyst_rating`\n"
                f"  Derivado: `peg_ratio`, `trailing_pe`\n\n"
                f"⏳ Estimativa: 45-90 min para 211k rows\n"
                f"_Preserva intacto: RSI, drawdown_52w, alpha_90d, targets, e tudo o resto._\n"
                f"_Atomic write: parquet original safe até ao fim._"
            )

            def _run_backfill():
                import subprocess as _sp, sys as _sys
                import time as _t
                t0 = _t.time()
                script = _P(__file__).parent / "scripts" / "backfill_fundamentals.py"
                cmd = [_sys.executable, str(script), "--parquet", str(parquet)]
                if since_arg:
                    cmd += ["--since", since_arg]
                if tickers_arg:
                    cmd += ["--tickers", tickers_arg]
                if max_rows_arg:
                    cmd += ["--max-rows", max_rows_arg]
                # Timeout: first full run fetches Form 4 XMLs for the entire
                # parquet (potentially 17-30h). Set to 48h so the user can
                # genuinely "let it finish". The early-skip in _process_row
                # eliminates ~20k pre-2003 rows that have no fetchable data;
                # the remaining work is bounded by EDGAR rate limit and the
                # permanent Form 4 cache (second run is sub-minute).
                _BACKFILL_TIMEOUT = 48 * 3600
                try:
                    proc = _sp.run(cmd, capture_output=True, text=True,
                                   timeout=_BACKFILL_TIMEOUT)
                except _sp.TimeoutExpired as _te:
                    # Capture whatever the subprocess managed to write before
                    # being killed. Without this, the user has no idea how
                    # far the run got. Railway logs don't show subprocess
                    # output (it's captured silently), so this is the only
                    # window into what happened.
                    _so = (getattr(_te, "stdout", "") or "")
                    _se = (getattr(_te, "stderr", "") or "")
                    _combined = (_so + "\n" + _se).strip()
                    _tail = _combined[-2500:] if _combined else "(sem output capturado)"
                    _reply(
                        f"❌ *Backfill timed out após {_BACKFILL_TIMEOUT/3600:.0f}h*\n\n"
                        f"<code>{_tail}</code>\n\n"
                        f"_Sugestão: re-corre com `--max-rows=50000` para sessão "
                        f"incremental (cache Form 4 é permanente, próxima sessão "
                        f"é mais rápida)._"
                    )
                    return
                except Exception as e:
                    _reply(f"❌ Backfill falhou: `{e}`")
                    return
                elapsed = (_t.time() - t0) / 60
                if proc.returncode == 0:
                    # Extract the FULL backfill log — covers everything from
                    # "Loading parquet" through "Total rows" / "Ticker filter"
                    # / "Processing: N rows" / per-1000-row progress / final
                    # NaN coverage report. Python logging writes to stderr by
                    # default so the bulk is in proc.stderr. Searching both is
                    # robust to any log redirect.
                    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
                    if "Loading parquet" in combined:
                        idx = combined.find("Loading parquet")
                        tail = combined[idx:]
                    elif "NaN coverage report" in combined:
                        idx = combined.find("NaN coverage report")
                        tail = combined[idx:]
                    else:
                        tail = combined[-3000:]
                    # Telegram code blocks ≤ ~4000 chars. If too long, keep
                    # the head (filter / processing summary) AND tail (NaN
                    # report) — the per-1000-row progress lines in the
                    # middle are skippable.
                    if len(tail) > 3500:
                        head = tail[:1500]
                        rep_idx = tail.find("NaN coverage report")
                        if rep_idx >= 0:
                            tail = head + "\n... [progress lines truncated] ...\n" + tail[rep_idx:rep_idx + 2000]
                        else:
                            tail = head + "\n... [truncated] ...\n" + tail[-1500:]
                    _reply(
                        f"✅ *Backfill fundamentals — concluído* ({elapsed:.0f} min)\n\n"
                        f"<code>{tail}</code>\n\n"
                        f"_Próximo passo: `/admin_test_retrain` para validar o pipeline "
                        f"antes de comprometer com retrain completo._"
                    )
                else:
                    tail = ((proc.stdout or "") + (proc.stderr or ""))[-600:]
                    _reply(f"❌ Backfill exit {proc.returncode}:\n<code>{tail}</code>")

            threading.Thread(target=_run_backfill, daemon=True, name="dip-backfill-fundamentals").start()
        except Exception as e:
            _reply(f"❌ Erro: `{e}`")

    elif cmd in ("/admin_rotation_backtest", "/admin_rotation"):
        # Top-N rotation backtest — capital flows monthly to whichever
        # sectors have the highest cross-sectional rotation_score.
        # Compares vs SPY buy-and-hold AND equal-weight basket buy-and-hold.
        #
        # The honest test: does rotation generate alpha at retail scale,
        # or is buy-and-hold still better?
        #
        # Usage:
        #   /admin_rotation_backtest                              # 3y, top-3, new_only
        #   /admin_rotation_backtest --years=5                    # 5y
        #   /admin_rotation_backtest --top=5                      # top-5
        #   /admin_rotation_backtest --floor=30                   # require score ≥30
        #   /admin_rotation_backtest --alloc=new_only             # cash to NEW entries only
        #   /admin_rotation_backtest --alloc=equal_top_n          # cash equal to all top-N
        #   /admin_rotation_backtest --alloc=score_weighted       # cash weighted by score
        try:
            n_years = 3
            top_n = 3
            score_floor = 0.0
            allocation_mode = "new_only"
            for token in parts[1:]:
                if token.startswith("--years="):
                    try:
                        n_years = int(token.split("=", 1)[1])
                    except Exception:
                        pass
                elif token.startswith("--top="):
                    try:
                        top_n = int(token.split("=", 1)[1])
                    except Exception:
                        pass
                elif token.startswith("--floor="):
                    try:
                        score_floor = float(token.split("=", 1)[1])
                    except Exception:
                        pass
                elif token.startswith("--alloc="):
                    val = token.split("=", 1)[1].strip().lower()
                    if val in ("new_only", "equal_top_n", "score_weighted"):
                        allocation_mode = val

            _reply(
                f"🧪 *Rotation backtest a iniciar*\n"
                f"  Period: `{n_years}y`\n"
                f"  Top-N: `{top_n}`\n"
                f"  Score floor: `{score_floor}`\n"
                f"  Allocation: `{allocation_mode}`\n\n"
                f"_~5-10 min para fetch + simulação. Vou postar resumo no fim._"
            )

            def _run():
                try:
                    from core.etf_universe import all_monitored_tickers
                    from core.sector_rotation_backtest import backtest_rotation

                    tickers = all_monitored_tickers()

                    def _progress(label, i, n):
                        if i > 0 and i % 8 == 0:
                            _reply(f"📊 Progress: `{i}/{n}` — {label}")

                    res = backtest_rotation(
                        tickers,
                        n_years=n_years,
                        top_n=top_n,
                        score_floor=score_floor,
                        allocation_mode=allocation_mode,
                        progress_callback=_progress,
                    )
                    if res is None:
                        _reply("⚠️ Backtest sem resultados — verifica conectividade yfinance.")
                        return

                    # Verdict
                    verdict_spy = "✅ VENCE SPY" if res.alpha_vs_spy > 0 else "❌ Perde para SPY"
                    verdict_basket = "✅ VENCE basket" if res.alpha_vs_basket > 0 else "❌ Perde para basket"

                    lines = [
                        f"🧪 *Rotation backtest — {res.n_years}y, top-{res.top_n}*",
                        f"",
                        f"```",
                        f"                Strategy    SPY         Basket",
                        f"Total return    {res.strategy_total_return*100:>+7.1f}%   "
                        f"{res.spy_total_return*100:>+7.1f}%   "
                        f"{res.basket_total_return*100:>+7.1f}%",
                        f"CAGR            {res.strategy_cagr*100:>+7.1f}%   "
                        f"{res.spy_cagr*100:>+7.1f}%   "
                        f"{res.basket_cagr*100:>+7.1f}%",
                        f"Max drawdown    {res.strategy_max_drawdown*100:>+7.1f}%   "
                        f"{res.spy_max_drawdown*100:>+7.1f}%   "
                        f"{res.basket_max_drawdown*100:>+7.1f}%",
                        f"Sharpe          {res.strategy_sharpe:>7.2f}    "
                        f"{res.spy_sharpe:>7.2f}    "
                        f"{res.basket_sharpe:>7.2f}",
                        f"```",
                        f"",
                        f"*Alphas* (strategy − benchmark):",
                        f"• vs SPY:    `{res.alpha_vs_spy*100:+.1f}pp` — {verdict_spy}",
                        f"• vs Basket: `{res.alpha_vs_basket*100:+.1f}pp` — {verdict_basket}",
                        f"",
                        f"*Activity*:",
                        f"• Rebalances: `{res.n_rebalances}`",
                        f"• Trades: `{res.n_trades}`",
                        f"• Total cost: `{res.total_cost_pct*100:.2f}%`",
                        f"• Holding period médio: `{res.avg_holding_period_days:.0f} dias`",
                        f"",
                        f"*Top 5 ETFs mais ocupados*:",
                    ]
                    for t in res.top_tickers:
                        days = res.ticker_appearances.get(t, 0)
                        pct = (days / res.n_days) * 100 if res.n_days else 0
                        lines.append(f"• `{t}`: {pct:.0f}% do tempo")

                    # Final verdict
                    lines.append("")
                    if res.alpha_vs_spy > 0.05 and res.alpha_vs_basket > 0.05:
                        lines.append(
                            "🎯 *VEREDICTO: ROTATION GERA ALPHA*. "
                            "Vence ambos benchmarks por margem real (>5pp). "
                            "Worth pursuing para live trading."
                        )
                    elif res.alpha_vs_spy > 0 and res.alpha_vs_basket > 0:
                        lines.append(
                            "🟡 *VEREDICTO: ALPHA MARGINAL*. "
                            "Vence mas margem pequena (<5pp). "
                            "Depende se a complexidade vale a pena."
                        )
                    elif res.alpha_vs_basket > 0:
                        lines.append(
                            "🟡 *VEREDICTO: BATE BASKET, NÃO SPY*. "
                            "Diversificação rotacional vence concentração no SPY, "
                            "mas SPY puro continua melhor. Filosoficamente "
                            "indica que SPY é o trade simples superior."
                        )
                    else:
                        lines.append(
                            "❌ *VEREDICTO: ROTATION NÃO FUNCIONA*. "
                            "Perde para benchmarks. Buy-and-hold continua "
                            "campeão a retail scale. Aceitar e simplificar."
                        )

                    _reply("\n".join(lines))
                except Exception as _e:
                    _reply(f"❌ Rotation backtest falhou: `{_e}`")

            threading.Thread(
                target=_run, daemon=True, name="rotation-backtest"
            ).start()
        except Exception as e:
            _reply(f"❌ Erro: `{e}`")

    elif cmd in ("/admin_sector_backtest", "/admin_backtest_sectors"):
        # Backtest the scaling-exit policy over historical data.
        # Compares strategy total return + max drawdown vs buy-and-hold.
        # The honest test of whether trimming early on caution signals
        # nets you ahead, or whether you're cutting too soon.
        #
        # Usage:
        #   /admin_sector_backtest          → full universe (~15-20 min bg)
        #   /admin_sector_backtest XLK      → single ticker (~1 min sync)
        #   /admin_sector_backtest --years=5 → custom history length
        try:
            n_years = 3
            single_ticker = None
            for token in parts[1:]:
                if token.startswith("--years="):
                    try:
                        n_years = int(token.split("=", 1)[1])
                    except (ValueError, IndexError):
                        pass
                elif not token.startswith("--"):
                    single_ticker = token.upper().strip()

            if single_ticker:
                # Foreground — fast enough for 1 ticker (~30-60s)
                _reply(
                    f"🧪 *Backtest a iniciar* — {single_ticker} ({n_years}y)\n"
                    f"_Pode demorar 30-60s..._"
                )
                from core.sector_backtest import backtest_sector

                def _run_single():
                    try:
                        res = backtest_sector(single_ticker, n_years=n_years)
                        if res is None:
                            _reply(f"⚠️ Sem dados suficientes para `{single_ticker}`.")
                            return
                        _strategy_dd_pct = res.strategy_max_drawdown * 100
                        _bh_dd_pct = res.buyhold_max_drawdown * 100
                        _alpha_pct = res.alpha * 100
                        _str_ret = res.strategy_total_return * 100
                        _bh_ret = res.buyhold_total_return * 100
                        _verdict = "✅ STRATEGY VENCE" if res.alpha > 0 else "❌ Buy-hold vence"
                        _reply(
                            f"🧪 *Backtest — {single_ticker} ({n_years}y)*\n\n"
                            f"```\n"
                            f"                Strategy    Buy-Hold     Diff\n"
                            f"Total return    {_str_ret:>7.1f}%    "
                            f"{_bh_ret:>7.1f}%    {_alpha_pct:>+5.1f}pp\n"
                            f"CAGR            {res.strategy_cagr*100:>7.1f}%    "
                            f"{res.buyhold_cagr*100:>7.1f}%\n"
                            f"Max drawdown    {_strategy_dd_pct:>7.1f}%    "
                            f"{_bh_dd_pct:>7.1f}%    "
                            f"{res.drawdown_reduction*100:>+5.1f}pp\n"
                            f"Sharpe          {res.strategy_sharpe:>7.2f}     "
                            f"{res.buyhold_sharpe:>7.2f}\n"
                            f"```\n"
                            f"• Trades: `{res.n_trades}`\n"
                            f"• Custo total: `{res.total_cost_pct*100:.2f}%`\n"
                            f"• % do tempo totalmente investido: `{res.pct_time_invested*100:.0f}%`\n"
                            f"• Posição final: `{res.final_position*100:.0f}%`\n\n"
                            f"*{_verdict}*"
                        )
                    except Exception as _e:
                        _reply(f"❌ Backtest falhou: `{_e}`")

                threading.Thread(
                    target=_run_single, daemon=True,
                    name=f"backtest-{single_ticker}"
                ).start()
                return

            # Background — full universe
            _reply(
                f"🧪 *Backtest universe a iniciar* — {n_years}y\n"
                f"_~15-20 min para 17 ETFs. Vou postar resumo no fim._"
            )

            def _run_universe():
                try:
                    from core.etf_universe import all_monitored_tickers
                    from core.sector_backtest import backtest_universe
                    tickers = all_monitored_tickers()

                    def _progress(ticker, i, n):
                        if i % 5 == 0 and i > 0:
                            _reply(
                                f"📊 Backtest progresso: `{i}/{n}` "
                                f"(actual: {ticker})"
                            )

                    results = backtest_universe(
                        tickers, n_years=n_years,
                        progress_callback=_progress,
                    )
                    if not results:
                        _reply("⚠️ Backtest sem resultados — verifica conectividade yfinance.")
                        return

                    # Aggregate by alpha
                    results.sort(key=lambda r: r.alpha, reverse=True)

                    n_wins = sum(1 for r in results if r.alpha > 0)
                    n_losses = sum(1 for r in results if r.alpha <= 0)
                    avg_alpha = sum(r.alpha for r in results) / len(results)
                    avg_dd_reduction = (
                        sum(r.drawdown_reduction for r in results) / len(results)
                    )

                    lines = [
                        f"🧪 *Backtest universe — {n_years}y*",
                        f"",
                        f"Strategy vence em `{n_wins}/{len(results)}` ETFs "
                        f"(perde em `{n_losses}`)",
                        f"Alpha médio: `{avg_alpha*100:+.1f}pp`",
                        f"Redução média de DD: `{avg_dd_reduction*100:+.1f}pp`",
                        f"",
                        f"```",
                        f"{'Ticker':>7} {'Alpha':>7} {'StrRet':>7} {'BHRet':>7} "
                        f"{'StrDD':>7} {'BHDD':>7} {'Trades':>6}",
                        f"─" * 60,
                    ]
                    for r in results[:25]:
                        lines.append(
                            f"{r.ticker:>7} "
                            f"{r.alpha*100:>+6.1f}% "
                            f"{r.strategy_total_return*100:>+6.1f}% "
                            f"{r.buyhold_total_return*100:>+6.1f}% "
                            f"{r.strategy_max_drawdown*100:>+6.1f}% "
                            f"{r.buyhold_max_drawdown*100:>+6.1f}% "
                            f"{r.n_trades:>6}"
                        )
                    lines.append("```")
                    lines.append("")
                    if avg_alpha > 0:
                        lines.append(
                            "✅ *Veredicto*: scaling-exit policy é net POSITIVE "
                            "vs buy-and-hold no período testado."
                        )
                    else:
                        lines.append(
                            "❌ *Veredicto*: scaling-exit policy é net NEGATIVE — "
                            "estás a cortar cedo demais. Considera relaxar thresholds."
                        )
                    _reply("\n".join(lines))
                except Exception as _e:
                    _reply(f"❌ Backtest universe falhou: `{_e}`")

            threading.Thread(
                target=_run_universe, daemon=True,
                name="backtest-universe"
            ).start()
        except Exception as e:
            _reply(f"❌ Erro: `{e}`")

    elif cmd in ("/admin_reset_sector_states", "/admin_reset_sectors"):
        # Wipe the sector_radar FSM state file. Useful after fixing a
        # scoring bug — the persisted state may have wrong transitions
        # baked in, and you want a fresh snapshot to re-classify.
        try:
            from pathlib import Path as _P
            _dd = _P("/data") if _P("/data").exists() else _P("/tmp")
            f = _dd / "sector_radar_state.json"
            if f.exists():
                f.unlink()
                _reply(f"🧹 `{f.name}` apagado. Próximo `/sectors` arranca limpo.")
            else:
                _reply(f"ℹ️ `{f.name}` já não existia.")
        except Exception as e:
            _reply(f"❌ Erro: `{e}`")

    elif cmd in ("/sectors", "/sector_radar"):
        # Daily sector radar snapshot — ranks all sector ETFs + broad
        # market + thematic by combined buy score. Highlights BUY_ZONE
        # and SELL_ZONE candidates (FSM state from state.py).
        #
        # Usage:
        #   /sectors           → full snapshot (all monitored ETFs)
        #   /sectors XLK       → deep dive on one ticker
        #   /sectors holdings  → only the user's current ETF positions
        try:
            from core.sector_radar import rank_universe, fetch_and_compute
            from core.etf_universe import (
                all_monitored_tickers,
                sector_etf_tickers,
                broad_etf_tickers,
                thematic_etf_tickers,
                user_holding_proxies,
                proxy_to_ucits,
                get_etf,
                display_ticker_pair,
            )
            import state as _st

            # Deep-dive on one ticker
            if len(parts) >= 2 and parts[1].upper() not in ("HOLDINGS",):
                t = parts[1].upper().strip()
                sig = fetch_and_compute(t)
                if sig is None:
                    _reply(f"⚠️ Sem dados para `{t}` (não está no universe ou yfinance falhou).")
                    return
                etf_meta = get_etf(t)
                meta_line = ""
                if etf_meta:
                    if etf_meta.ucits_ticker:
                        meta_line = (
                            f"\n_{etf_meta.name}_\n"
                            f"_UCITS equivalente: `{etf_meta.ucits_ticker}` ({etf_meta.ucits_name})_"
                        )
                    else:
                        meta_line = f"\n_{etf_meta.name}_"

                state_record = _st.get_sector_state(t)
                fsm_state = state_record.get("state", "NEUTRAL")

                # Apply the FSM (this updates persistent state too)
                new_state, should_alert = _st.update_sector_state(
                    t,
                    combined_buy=sig.combined_buy,
                    exit_strength=sig.exit_strength,
                )

                _reply(
                    f"📊 *Sector Radar — {t}*{meta_line}\n"
                    f"Preço: `${sig.price:.2f}` | RSI: `{sig.rsi_14:.0f}` | "
                    f"DD 52w: `{sig.drawdown_52w*100:+.1f}%`\n"
                    f"Pullback do peak 30d: `{sig.pullback_from_recent_peak*100:+.1f}%`\n"
                    f"MA200 ratio: `{sig.ma200_ratio:.3f}` "
                    f"({sig.days_below_ma200}d abaixo, {sig.dd_duration_months:.0f}m em DD)\n"
                    f"Retorno 7d: `{sig.return_7d*100:+.1f}%` "
                    f"(vs SPY: `{sig.return_7d_vs_spy*100:+.1f}pp`)\n"
                    f"Retorno 30d: `{sig.return_30d*100:+.1f}%` "
                    f"(vs SPY: `{sig.return_30d_vs_spy*100:+.1f}pp`)\n"
                    f"Retorno YTD: `{sig.return_ytd*100:+.1f}%` "
                    f"(vs SPY: `{sig.return_ytd_vs_spy*100:+.1f}pp`)\n\n"
                    f"*Scores* (0-100):\n"
                    f"• Dip:           `{sig.dip_score:>5.1f}`\n"
                    f"• Momentum:      `{sig.momentum_score:>5.1f}`\n"
                    f"• Secular Risk:  `{sig.secular_risk:>5.1f}` "
                    f"{'⚠️ value trap?' if sig.secular_risk > 60 else ''}\n"
                    f"• Caution:       `{sig.caution_score:>5.1f}` "
                    f"{'⚠️ deterioração ativa' if sig.caution_score > 50 else ''}\n"
                    f"• Combined BUY:  `{sig.combined_buy:>5.1f}` "
                    f"{'🔥' if sig.combined_buy > 70 else ''}\n"
                    f"• Exit strength: `{sig.exit_strength:>5.1f}` "
                    f"{'🛑' if sig.exit_strength > 70 else ''}\n"
                    f"• Entry strength:`{sig.entry_strength:>5.1f}`\n\n"
                    f"FSM state: `{fsm_state}` → `{new_state}` "
                    f"{'(🚨 alert!)' if should_alert else ''}"
                )
                return

            # Holdings-only mode
            holdings_mode = len(parts) >= 2 and parts[1].upper() == "HOLDINGS"
            if holdings_mode:
                proxies = user_holding_proxies()
                tickers = list(proxies.values())
            else:
                tickers = all_monitored_tickers()

            results = rank_universe(tickers)
            if not results:
                _reply("⚠️ Sem resultados — verifica conectividade yfinance.")
                return

            # Apply FSM to each + collect transitions
            transitions: list[str] = []
            for sig in results:
                new_state, should_alert = _st.update_sector_state(
                    sig.ticker,
                    combined_buy=sig.combined_buy,
                    exit_strength=sig.exit_strength,
                )
                sig.state = new_state
                if should_alert:
                    transitions.append(
                        f"{sig.ticker} → {new_state} "
                        f"(buy={sig.combined_buy:.0f}, exit={sig.exit_strength:.0f})"
                    )

            # Build snapshot table
            header = "📊 *Sector Radar — snapshot*" + (
                " (holdings only)" if holdings_mode else ""
            )
            lines = [header, ""]
            lines.append("```")
            lines.append(
                f"{'Ticker':>7} {'BUY':>4} {'Dip':>4} {'Mom':>4} "
                f"{'Sec':>4} {'Caut':>4} {'Exit':>4} {'State':>9}"
            )
            lines.append("─" * 53)
            for sig in results[:20]:  # cap at 20 to avoid spam
                state_marker = ""
                if sig.state == "BUY_ZONE":
                    state_marker = "🔥"
                elif sig.state == "SELL_ZONE":
                    state_marker = "🛑"
                # Caution overlay: even if state=BUY_ZONE, caution>50 means
                # "be careful, deterioration in progress" — show ⚠️
                caut_marker = "⚠️" if sig.caution_score >= 50 else ""
                ucits = proxy_to_ucits(sig.ticker)
                ucits_tag = f" ({ucits})" if ucits else ""
                ticker_disp = sig.ticker + ucits_tag if holdings_mode else sig.ticker
                lines.append(
                    f"{ticker_disp:>7} "
                    f"{sig.combined_buy:>4.0f} "
                    f"{sig.dip_score:>4.0f} "
                    f"{sig.momentum_score:>4.0f} "
                    f"{sig.secular_risk:>4.0f} "
                    f"{sig.caution_score:>4.0f} "
                    f"{sig.exit_strength:>4.0f} "
                    f"{caut_marker + state_marker + sig.state:>9}"
                )
            lines.append("```")

            # Highlight transitions
            if transitions:
                lines.append("")
                lines.append("*Alertas de transição:*")
                for tr in transitions:
                    lines.append(f"• {tr}")

            # Summary footer
            summary = _st.sector_states_summary()
            lines.append("")
            lines.append(
                f"_Estados activos: BUY={summary['counts']['BUY_ZONE']}, "
                f"SELL={summary['counts']['SELL_ZONE']}, "
                f"NEUTRAL={summary['counts']['NEUTRAL']}._"
            )
            lines.append("_Detalhe de um ETF: `/sectors XLK`. Holdings: `/sectors holdings`._")

            _reply("\n".join(lines))
        except Exception as e:
            _reply(f"❌ Sector radar falhou: `{e}`")

    elif cmd in ("/holdings_etf_status", "/holdings_etf"):
        # Focused view: just the user's UCITS holdings (SEC0, SXR8, IS3N)
        # via their US proxies. Shows exit_strength and entry_strength so
        # you can decide whether to trim / hold / re-enter.
        try:
            from core.sector_radar import fetch_and_compute, fetch_spy_history
            from core.etf_universe import user_holding_proxies, proxy_to_ucits, get_etf
            import state as _st

            proxies = user_holding_proxies()  # {UCITS: US_proxy}
            if not proxies:
                _reply("ℹ️ Nenhuma holding UCITS registada em `etf_universe.py`.")
                return

            # Fetch SPY once for the loop — otherwise momentum gate fails
            # for each proxy and the FSM writes corrupt state.
            spy_hist = fetch_spy_history()

            lines = ["💼 *Holdings ETF status*", ""]
            for ucits, us_proxy in proxies.items():
                sig = fetch_and_compute(us_proxy, spy_hist=spy_hist)
                if sig is None:
                    lines.append(f"• `{ucits}` ({us_proxy}): ⚠️ sem dados")
                    continue
                state_record = _st.get_sector_state(us_proxy)
                fsm_state = state_record.get("state", "NEUTRAL")

                # Apply FSM (also persists)
                new_state, should_alert = _st.update_sector_state(
                    us_proxy,
                    combined_buy=sig.combined_buy,
                    exit_strength=sig.exit_strength,
                )

                state_emoji = {
                    "NEUTRAL":   "⚪",
                    "BUY_ZONE":  "🔥",
                    "SELL_ZONE": "🛑",
                }.get(new_state, "⚪")

                # Policy: only exit_strength (structural breakdown) triggers
                # a SELL recommendation. caution_score is informational —
                # backtest of caution-based TRIM showed 0/20 wins vs
                # buy-and-hold over 3y. Leading signals catch dozens of
                # false positives that cost more than the drawdowns saved.
                #
                # User retains discretion: if caution ≥ 70 and you JUDGE
                # the deterioration is real (earnings miss + macro + chart
                # all bad), you can trim manually — the radar gives you
                # visibility, you make the call.
                action = "MANTER"
                action_detail = ""
                if new_state == "SELL_ZONE":
                    action = "VENDER — sinal estrutural confirmado"
                    action_detail = (
                        f" (exit_strength {sig.exit_strength:.0f}: "
                        f"MA200 / death cross / RSI rollover)"
                    )
                elif sig.exit_strength > 50:
                    action = "VIGIAR — exit a aproximar-se"
                    action_detail = f" (exit_strength {sig.exit_strength:.0f})"
                elif sig.caution_score >= 70:
                    action = "MANTER — caution alto, vigilância manual"
                    action_detail = (
                        f" (caution {sig.caution_score:.0f}, "
                        f"pullback {sig.pullback_from_recent_peak*100:+.1f}%, "
                        f"7d {sig.return_7d*100:+.1f}%) — *avalia tu se trims*"
                    )
                elif sig.caution_score >= 50:
                    action = "MANTER — sinais leading ativos"
                    action_detail = (
                        f" (caution {sig.caution_score:.0f}, "
                        f"vigia evolução)"
                    )
                elif new_state == "BUY_ZONE":
                    action = "REFORÇAR"

                lines.append(
                    f"*{ucits}* (via `{us_proxy}`) {state_emoji} `{new_state}`"
                )
                lines.append(
                    f"  Preço: `${sig.price:.2f}` | RSI: `{sig.rsi_14:.0f}` | "
                    f"DD 52w: `{sig.drawdown_52w*100:+.1f}%`"
                )
                lines.append(
                    f"  7d: `{sig.return_7d*100:+.1f}%` "
                    f"(vs SPY `{sig.return_7d_vs_spy*100:+.1f}pp`) | "
                    f"Pullback peak 30d: `{sig.pullback_from_recent_peak*100:+.1f}%`"
                )
                lines.append(
                    f"  Caution: `{sig.caution_score:.0f}` | "
                    f"Exit: `{sig.exit_strength:.0f}` | "
                    f"Entry: `{sig.entry_strength:.0f}` | "
                    f"BUY: `{sig.combined_buy:.0f}`"
                )
                lines.append(f"  → *{action}*{action_detail}")
                if should_alert:
                    lines.append(f"  🚨 *Transição: {fsm_state} → {new_state}*")
                lines.append("")

            _reply("\n".join(lines).rstrip())
        except Exception as e:
            _reply(f"❌ Holdings status falhou: `{e}`")

    elif cmd in ("/admin_calibrate_rotation", "/calibrate_rotation"):
        # Run scripts/calibrate_rotation_alpha.py in the background.
        # Reads N years of rotation backtest history, fits alpha-vs-score
        # regression, writes data/rotation_alpha_calibration.json. /cash
        # picks up the result on next invocation.
        #
        # Usage:
        #   /admin_calibrate_rotation        → 4-year window (default)
        #   /admin_calibrate_rotation 5      → 5-year window
        years = 4
        if len(parts) >= 2:
            try:
                years = max(2, min(10, int(parts[1])))
            except ValueError:
                pass

        _reply(
            f"🔧 *Rotation alpha calibration* — iniciado\n"
            f"_Janela: {years}y | ~10-15min | escreve "
            f"`data/rotation_alpha_calibration.json`_"
        )

        def _run_calibration():
            try:
                import subprocess
                import sys as _sys
                result = subprocess.run(
                    [_sys.executable, "scripts/calibrate_rotation_alpha.py", "--years", str(years)],
                    capture_output=True, text=True, timeout=1800,
                )
                out_tail = (result.stdout or "")[-1500:]
                err_tail = (result.stderr or "")[-800:]

                # Read the produced JSON to surface key numbers.
                # Supports both legacy (single-method linear) and new
                # multi-method format with `method` + `params` + `all_methods`.
                summary = ""
                try:
                    import json as _json
                    from pathlib import Path as _Path
                    cal_path = _Path("data/rotation_alpha_calibration.json")
                    if cal_path.exists():
                        cal = _json.loads(cal_path.read_text())
                        chosen_method = cal.get("method", "linear")
                        cv_r2 = cal.get("cv_r_squared", cal.get("r_squared"))
                        lines_sum = [
                            f"\n📊 *Resultado*:",
                            f"  Método escolhido: `{chosen_method}`",
                            f"  CV r² (out-of-fold) = `{cv_r2}`",
                            f"  In-sample r² = `{cal.get('r_squared')}`",
                            f"  n_picks = `{cal.get('n_picks')}`",
                            f"  accepted = `{cal.get('accepted')}` "
                            f"(threshold {cal.get('accept_threshold')})",
                        ]
                        # Show comparison if all_methods present
                        all_m = cal.get("all_methods", [])
                        if all_m:
                            lines_sum.append("\n  Comparação dos 4 métodos (CV r²):")
                            for m in all_m:
                                err = (
                                    f" ({m.get('error')})"
                                    if m.get("error") else ""
                                )
                                marker = " ←" if m.get("method") == chosen_method else ""
                                lines_sum.append(
                                    f"    {m.get('method'):11} "
                                    f"in={m.get('r_squared'):.3f} "
                                    f"cv={m.get('cv_r_squared'):.3f}{err}{marker}"
                                )
                        # Method-specific params
                        params = cal.get("params", {})
                        if chosen_method == "linear" and params:
                            lines_sum.append(
                                f"  slope = `{params.get('slope_alpha_per_point_monthly')}` | "
                                f"intercept = `{params.get('intercept')}`"
                            )
                        elif chosen_method == "polynomial" and params:
                            lines_sum.append(
                                f"  degree = `{params.get('degree')}` | "
                                f"coeffs = `{params.get('coeffs')}`"
                            )
                        elif chosen_method == "buckets" and params:
                            lines_sum.append(
                                f"  boundaries = `{params.get('boundaries')}`"
                            )
                            lines_sum.append(
                                f"  alpha/bucket = `{params.get('alpha_per_bucket')}`"
                            )
                        summary = "\n".join(lines_sum)
                except Exception as _e:
                    summary = f"\n_(erro a ler JSON: {_e})_"

                if result.returncode == 0:
                    _reply(
                        f"✅ Calibration done.{summary}\n\n"
                        f"_Log tail:_\n```\n{out_tail}\n```"
                    )
                elif result.returncode == 2:
                    _reply(
                        f"⚠️ Calibration completed but r² below threshold.{summary}\n"
                        f"_Rotation alpha defaultará a 0 no `/cash` até próxima calibration._\n\n"
                        f"_Log tail:_\n```\n{out_tail}\n```"
                    )
                else:
                    _reply(
                        f"❌ Calibration failed (exit {result.returncode}).\n"
                        f"_stderr:_\n```\n{err_tail}\n```\n"
                        f"_stdout tail:_\n```\n{out_tail}\n```"
                    )
            except subprocess.TimeoutExpired:
                _reply("❌ Calibration timeout (30min). Reduce --years ou investiga.")
            except Exception as _e:
                _reply(f"❌ Erro: `{_e}`")

        threading.Thread(
            target=_run_calibration, daemon=True, name="rotation-calibrate"
        ).start()

    elif cmd in ("/monthly_check", "/monthly", "/portfolio_check"):
        # Comprehensive monthly portfolio review. Iterates EVERY registered
        # position and reports its current state with HOLD/SELL/WATCH
        # decision + reason. Designed to be the "I want to know everything
        # the system thinks about my portfolio right now" command.
        #
        # Output structure (per bucket):
        #   • Current price, PnL, since-entry days
        #   • State (HOLD/WATCH/SELL) with concrete reason
        #   • Triggers fired this month (if any)
        #
        # Plus system-wide:
        #   • Rotation top-3 change vs current ROTATION holdings
        #   • New Core picks worth attention
        #   • DIP scanner alerts this week
        #
        # Read-only — does NOT update any state.
        _reply(
            "📊 *Monthly Portfolio Check* — scan completo iniciado\n"
            "_~60-90s: ~13 posições × evaluation + rotation + Core picks..._"
        )

        def _run_monthly():
            try:
                from portfolio.portfolio import get_positions
                from datetime import datetime as _dt

                positions = get_positions()
                if not positions:
                    _reply("ℹ️ Carteira vazia — corre `/sync_portfolio` primeiro.")
                    return

                portfolio_by_bucket: dict = {
                    "CORE": [], "ROTATION": [], "MOMENTUM": [], "DIP": [],
                    "UNCATEGORIZED": [],
                }
                total_cost = 0.0
                for sym, pos in positions.items():
                    bucket = (pos.get("category") or "").upper()
                    if bucket not in portfolio_by_bucket:
                        bucket = "UNCATEGORIZED"
                    portfolio_by_bucket[bucket].append((sym, pos))
                    total_cost += pos.get("total_cost", 0.0)

                # Compute current values to derive bucket weights
                from core import conviction_radar as cr
                bucket_value: dict[str, float] = {b: 0.0 for b in portfolio_by_bucket}
                position_states: list[dict] = []

                for bucket_name, plist in portfolio_by_bucket.items():
                    for sym, pos in plist:
                        avg_price = float(pos.get("avg_price", 0) or 0)
                        shares    = float(pos.get("shares", 0) or 0)
                        cost      = float(pos.get("total_cost", 0) or 0)

                        # Live price:
                        # - US tickers: yfinance via conviction_radar fetcher
                        # - UCITS (ROTATION): yfinance has no data, so estimate
                        #   from the US proxy's % change since entry_date
                        #   (USER_HOLDINGS_UCITS mapping). PnL is approximate.
                        current = avg_price
                        ucits_estimated = False
                        if bucket_name == "ROTATION":
                            try:
                                from core.ucits_proxy import estimate_ucits_current_price
                                est = estimate_ucits_current_price(
                                    sym, avg_price,
                                    entry_date_iso=pos.get("entry_date_iso"),
                                )
                                if est is not None and est > 0:
                                    current = est
                                    ucits_estimated = True
                            except Exception:
                                pass
                        else:
                            try:
                                current = float(cr._fetch_price(sym, fallback=avg_price))
                            except Exception:
                                current = avg_price

                        value_now = current * shares
                        pnl_eur   = value_now - cost
                        pnl_pct   = (pnl_eur / cost * 100) if cost > 0 else 0.0
                        bucket_value[bucket_name] += value_now

                        # Build state
                        state = "HOLD"
                        reason = ""
                        days_held = 0

                        # Get conviction record for advanced analysis
                        conv_state: str | None = None
                        if bucket_name in ("CORE", "MOMENTUM", "DIP"):
                            try:
                                conv = cr.get_by_ticker(sym)
                                if conv is not None:
                                    ev = cr.evaluate_conviction(conv)
                                    triggers = ev.get("triggers", [])
                                    high_sev = [t for t in triggers if t.get("severity") == "high"]
                                    med_sev  = [t for t in triggers if t.get("severity") == "med"]
                                    if high_sev:
                                        state = "SELL"
                                        reason = high_sev[0].get("message", "trigger HIGH")
                                    elif med_sev:
                                        state = "WATCH"
                                        reason = med_sev[0].get("message", "trigger MED")
                                    # Days held from entry_date
                                    try:
                                        ed = _dt.fromisoformat(conv.entry_date)
                                        days_held = (_dt.utcnow() - ed).days
                                    except Exception:
                                        days_held = 0
                                    # Trailing stop info for MOMENTUM
                                    if bucket_name == "MOMENTUM" and conv.trailing_high > 0:
                                        gap = (current - conv.trailing_high) / conv.trailing_high
                                        conv_state = (
                                            f"trail ${conv.trailing_high:.2f}, "
                                            f"gap {gap*100:+.1f}%"
                                        )
                            except Exception:
                                pass

                        position_states.append({
                            "ticker":           sym,
                            "bucket":           bucket_name,
                            "shares":           shares,
                            "avg_price":        avg_price,
                            "current":          current,
                            "value_now":        value_now,
                            "cost":             cost,
                            "pnl_eur":          pnl_eur,
                            "pnl_pct":          pnl_pct,
                            "state":            state,
                            "reason":           reason,
                            "days_held":        days_held,
                            "conv_state":       conv_state,
                            "ucits_estimated":  ucits_estimated,
                        })

                total_value = sum(bucket_value.values())
                total_pnl   = total_value - total_cost
                total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0

                # ── Build output ──────────────────────────────────────
                lines: list[str] = []
                lines.append(f"📊 *Monthly Portfolio Check* — {_dt.utcnow().strftime('%Y-%m-%d')}")
                lines.append("")
                lines.append(f"💼 *Total*: €{total_value:.2f} (cost €{total_cost:.2f})")
                lines.append(
                    f"   PnL: €{total_pnl:+.2f} ({total_pnl_pct:+.2f}%)"
                )
                lines.append("")

                # Bucket allocation breakdown
                lines.append("📐 *Allocation por bucket*:")
                for b in ("CORE", "ROTATION", "MOMENTUM", "DIP", "UNCATEGORIZED"):
                    v = bucket_value[b]
                    pct = (v / total_value * 100) if total_value > 0 else 0.0
                    n = len(portfolio_by_bucket[b])
                    if n > 0:
                        lines.append(f"   {b}: €{v:.2f} ({pct:.1f}%) — {n} posições")
                lines.append("")

                # Per-bucket position details with state
                for b in ("CORE", "ROTATION", "MOMENTUM", "DIP", "UNCATEGORIZED"):
                    plist = [s for s in position_states if s["bucket"] == b]
                    if not plist:
                        continue
                    # Sort by absolute value desc
                    plist.sort(key=lambda s: -s["value_now"])

                    lines.append(f"*{b}*")
                    for s in plist:
                        state_emoji = {
                            "HOLD": "🟢", "WATCH": "🟡", "SELL": "🛑",
                        }.get(s["state"], "⚪")

                        # Compact line with concentration %
                        days_str = f"d{s['days_held']}" if s["days_held"] > 0 else "—"
                        # UCITS positions use proxy-derived PnL when available
                        # (marked with ~ to flag the approximation). Otherwise
                        # show "no live" rather than misleading 0%.
                        if b == "ROTATION":
                            if s.get("ucits_estimated"):
                                pnl_str = f"~{s['pnl_pct']:+.1f}%"
                            else:
                                pnl_str = "no live"
                        else:
                            pnl_str = f"{s['pnl_pct']:+.1f}%"
                        conc_pct = (
                            (s['value_now'] / total_value * 100)
                            if total_value > 0 else 0.0
                        )
                        # Mark positions near or over the 25%/30% caps
                        conc_marker = ""
                        if conc_pct >= 30.0:
                            conc_marker = " 🛑"
                        elif conc_pct >= 25.0:
                            conc_marker = " ⚠️"
                        lines.append(
                            f"  {state_emoji} `{s['ticker']:<5}` "
                            f"€{s['value_now']:>7.0f} ({pnl_str}) "
                            f"{conc_pct:>4.1f}%{conc_marker} {days_str}"
                        )
                        # Reason on a separate line if non-HOLD
                        if s["state"] != "HOLD" and s["reason"]:
                            lines.append(f"      _{s['reason']}_")
                        if s["conv_state"]:
                            lines.append(f"      _{s['conv_state']}_")
                    lines.append("")

                # ── System signals: rotation + Core + DIP queue ──────
                lines.append("📡 *Signals do sistema*:")
                lines.append("")

                # Rotation top-3
                try:
                    from core.sector_radar import rank_universe
                    from core.etf_universe import all_monitored_tickers, get_etf
                    all_t = all_monitored_tickers()
                    ranked = rank_universe(all_t)
                    if ranked:
                        held_rotation = {sym for sym, _ in portfolio_by_bucket["ROTATION"]}
                        top_3 = ranked[:3]
                        lines.append("🔄 ROTATION top-3:")
                        for i, sig in enumerate(top_3, 1):
                            etf = get_etf(sig.ticker)
                            ucits = etf.ucits_ticker if etf and etf.ucits_ticker else sig.ticker
                            marker = "✓ tens" if ucits in held_rotation else "NOVO"
                            lines.append(
                                f"  {i}. `{sig.ticker}` → `{ucits}` "
                                f"score {sig.combined_buy:.0f} ({marker})"
                            )
                        # Rotation holdings out of top-3 → suggest SELL
                        top3_ucits = {
                            (get_etf(s.ticker).ucits_ticker if get_etf(s.ticker) and get_etf(s.ticker).ucits_ticker else s.ticker)
                            for s in top_3
                        }
                        out_of_top = [sym for sym in held_rotation if sym not in top3_ucits]
                        if out_of_top:
                            lines.append(f"   ⚠️ Holdings fora do top-3: {', '.join(out_of_top)}")
                        lines.append("")
                except Exception as _e:
                    lines.append(f"  ⚠️ Rotation falhou: `{_e}`")
                    lines.append("")

                # Core picks
                try:
                    from core.core_radar import scan_core_universe
                    core_opps = scan_core_universe(min_score=70.0)
                    if core_opps:
                        core_held = {sym.upper() for sym, _ in portfolio_by_bucket["CORE"]}
                        fresh = [o for o in core_opps if o["ticker"].upper() not in core_held]
                        top5 = fresh[:5]
                        lines.append(f"🏛 CORE picks (score ≥70, top 5 não-detidos):")
                        if not top5:
                            lines.append("  _Nenhum pick novo, todos os top scorers já são tuas posições._")
                        for opp in top5:
                            lines.append(
                                f"  • `{opp['ticker']}` score {opp['score']:.0f} "
                                f"(${opp.get('current_price', 0):.2f})"
                            )
                        lines.append("")
                except Exception as _e:
                    lines.append(f"  ⚠️ Core picks falhou: `{_e}`")
                    lines.append("")

                # DIP alerts (real tickers only — WL_ watchlist pseudo-tickers filtered)
                try:
                    import state as _st
                    raw_alerts = list(_st.load_alerts())
                    today_alerts = [
                        t for t in raw_alerts
                        if not str(t).upper().startswith("WL_")
                        and "_" not in str(t)
                    ]
                    wl_count = len(raw_alerts) - len(today_alerts)
                    if today_alerts:
                        suffix = f" (+{wl_count} watchlist)" if wl_count else ""
                        lines.append(f"🎯 DIP alerts hoje ({len(today_alerts)}{suffix}):")
                        for t in today_alerts[:5]:
                            lines.append(f"  • `{t}`")
                        lines.append("")
                    else:
                        lines.append(
                            f"🎯 DIP alerts hoje: 0"
                            + (f" ({wl_count} watchlist apenas)" if wl_count else "")
                        )
                        lines.append("")
                except Exception as _e:
                    lines.append(f"  ⚠️ DIP queue falhou: `{_e}`")

                # ── Summary actions ─────────────────────────────────
                sell_states = [s for s in position_states if s["state"] == "SELL"]
                watch_states = [s for s in position_states if s["state"] == "WATCH"]

                if sell_states or watch_states:
                    lines.append("")
                    lines.append("📋 *Decisões pendentes*:")
                    for s in sell_states:
                        lines.append(
                            f"  🛑 SELL `{s['ticker']}` "
                            f"({s['shares']:g} sh, €{s['value_now']:.0f})"
                        )
                    for s in watch_states:
                        lines.append(
                            f"  🟡 WATCH `{s['ticker']}` "
                            f"({s['shares']:g} sh, €{s['value_now']:.0f})"
                        )
                else:
                    lines.append("")
                    lines.append("✅ *Nenhuma decisão pendente* — tudo HOLD")

                _reply("\n".join(lines))
            except Exception as _e:
                _reply(f"❌ Monthly check falhou: `{_e}`")

        threading.Thread(
            target=_run_monthly, daemon=True, name="monthly-check"
        ).start()

    elif cmd in ("/cash_reserve", "/reserve"):
        # View or clear the cash reserve that /cash recommended to hold.
        # Reserve is the residual after /cash deploys signals; it's
        # available for opportunistic deploy when high-conviction
        # alerts fire during the month.
        #
        # Usage:
        #   /cash_reserve         → view current reserve
        #   /cash_reserve clear   → manually clear (after physical T212 deploy)
        try:
            import state as _st_r
            data = _st_r.load_cash_reserve()
            current = float(data.get("amount_eur", 0))

            if len(parts) >= 2 and parts[1].lower() in ("clear", "reset", "zero"):
                _st_r.consume_cash_reserve(current, reason="manual clear")
                _reply(
                    f"🧹 Cash reserve limpa.\n"
                    f"Valor anterior: €{current:.2f}\n"
                    f"_Use quando já deployaste manualmente em T212 e queres "
                    f"que os alerts deixem de mencionar a reserva._"
                )
                return

            if current <= 0:
                _reply(
                    "📥 *Cash reserve*: €0.00\n\n"
                    "_Sem reserva pendente. O próximo /cash pode definir "
                    "uma se houver budget sem signals para deploy._"
                )
                return

            set_at = data.get("set_at", "?")
            source = data.get("source", "?")
            history = data.get("history", [])
            lines = [
                f"📥 *Cash reserve*: *€{current:,.2f}*",
                f"",
                f"_Definida em_ `{set_at[:16]}` _via_ `{source}`",
                f"_Yield em T212 @ 2.2%/ano = €{current * 0.022:.2f}/ano_",
                f"",
                f"Será mostrada em DIP/Core alerts (score ≥ 70) "
                f"para te lembrar de deploy parcial sem esperar inflow novo.",
            ]
            if history:
                lines.append("")
                lines.append(f"_Histórico ({len(history)} entries anteriores):_")
                for h in history[-3:]:
                    lines.append(
                        f"  • €{h.get('amount_eur', 0):.0f} "
                        f"({h.get('reason', '?')[:30]}) "
                        f"até {h.get('ended_at', '?')[:10]}"
                    )
            lines.append("")
            lines.append(
                "_Para limpar manualmente_ (e.g., após deploy físico em T212): "
                "`/cash_reserve clear`"
            )
            _reply("\n".join(lines))
        except Exception as e:
            _reply(f"❌ Cash reserve falhou: `{e}`")

    elif cmd in ("/cash", "/allocation", "/alloc"):
        # Allocation flow with freed-cash incorporation.
        #
        # Two-pass design:
        #   Pass 1 — Compute all proposed SELLs (rotation positions out of
        #            top-3, MOMENTUM/DIP positions below ATR trailing stop).
        #            Sum their values into liberated_cash.
        #   Pass 2 — Total available = inflow + liberated_cash.
        #            Distribute total across BUYs (rotation top-3,
        #            Core picks, DIP entries) with explicit budgets per
        #            radar. Whatever's left becomes cash bucket.
        #
        # The output presents the proposal as a contract: "execute these
        # sells AND these buys for the proposed result". User can selectively
        # execute. /cash 0 means "no inflow, just show what to rebalance".
        #
        # Usage:
        #   /cash 1050              → full proposal with €1050 inflow
        #   /cash 0                 → health check, no inflow
        #   /cash 1050 --no-sells   → ignore SELL recommendations (treat
        #                             freed cash as 0, only deploy inflow)
        #   /cash 1050 --no-rot     → skip rotation evaluation
        if len(parts) < 2:
            _reply(
                "Uso: `/cash <amount_eur>`\n"
                "Ex: `/cash 1050` — proposta com €1050 inflow + freed cash.\n"
                "Ex: `/cash 0` — só health check / rebalance suggestions.\n\n"
                "_Antes de correr, garante que `/sync_portfolio` está atualizado._"
            )
            return
        try:
            cash_amount = float(parts[1])
            if cash_amount < 0:
                _reply("⚠️ Amount tem que ser ≥ 0.")
                return
        except ValueError:
            _reply(f"❌ `{parts[1]}` não é um número válido.")
            return

        skip_rotation = "--no-rot" in parts or "--skip-rotation" in parts
        ignore_sells  = "--no-sells" in parts

        _reply(
            f"💰 *Allocation €{cash_amount:.2f}* — scan iniciado\n"
            f"_~30-60s: sells primeiro (freed cash), depois buys..._"
        )

        def _run_allocation():
            try:
                from portfolio.portfolio import get_positions
                from core import conviction_radar as cr
                from core.tax_engine import should_rotate as tax_should_rotate

                # Concentration caps (per-ticker share of total portfolio)
                CONCENTRATION_SOFT = 0.25
                CONCENTRATION_HARD = 0.30

                # Load rotation alpha calibration. Supports multiple model
                # methods (linear/polynomial/buckets/lightgbm). If the file
                # is missing or the chosen method has CV r² below threshold,
                # falls back to a FLAT estimate (~12%/yr, from backtest
                # empirical: +15pp/yr - costs ≈ 12%).
                import json as _json
                from pathlib import Path as _Path
                import numpy as _np

                _CAL: dict = {}
                _CAL_ACCEPTED = False
                # Backtest-empirical fallback: 4-year backtest showed +15pp
                # annual alpha vs SPY with monthly rotation. Subtract cost
                # drag and forward uncertainty → conservative 12% estimate.
                _ROTATION_FLAT_FALLBACK = 0.12

                try:
                    _cal_path = _Path("data/rotation_alpha_calibration.json")
                    if _cal_path.exists():
                        _CAL = _json.loads(_cal_path.read_text())
                        _CAL_ACCEPTED = bool(_CAL.get("accepted", False))
                except Exception:
                    pass

                def _rotation_alpha_from_calibration(score: float) -> float | None:
                    """Predict monthly alpha from score using the calibrated method.

                    Returns None if calibration unavailable or method unknown.
                    """
                    if not _CAL or not _CAL_ACCEPTED:
                        return None
                    method = _CAL.get("method")
                    params = _CAL.get("params", {})
                    try:
                        if method == "linear":
                            return (
                                float(params.get("intercept", 0))
                                + float(params.get("slope_alpha_per_point_monthly", 0)) * score
                            )
                        if method == "polynomial":
                            coeffs = params.get("coeffs", [])
                            return float(_np.polyval(coeffs, score))
                        if method == "buckets":
                            boundaries = params.get("boundaries", [])
                            means = params.get("alpha_per_bucket", {})
                            idx = int(_np.digitize([score], boundaries)[0])
                            return float(means.get(str(idx), 0.0))
                        if method == "lightgbm":
                            grid_x = params.get("grid_x", [])
                            grid_y = params.get("grid_y", [])
                            if grid_x and grid_y:
                                return float(_np.interp(score, grid_x, grid_y))
                    except Exception:
                        return None
                    return None

                def _annualized_alpha(signal_type: str, **kw) -> float:
                    """Map each signal to annualized expected alpha."""
                    if signal_type == "DIP":
                        return float(kw.get("pred_alpha_90d", 0)) * (365.0 / 90.0)
                    if signal_type == "MOMENTUM":
                        return float(kw.get("pred_alpha_30d", 0)) * (365.0 / 30.0)
                    if signal_type == "CORE":
                        return float(kw.get("pred_alpha_1y", 0))
                    if signal_type == "ROTATION":
                        score = float(kw.get("score", 0))
                        monthly = _rotation_alpha_from_calibration(score)
                        if monthly is not None:
                            return monthly * 12.0
                        # Calibration not accepted → flat fallback (backtest-empirical)
                        return _ROTATION_FLAT_FALLBACK
                    return 0.0

                # ── State current portfolio (cost basis + live value) ──────
                positions = get_positions()
                portfolio_by_bucket: dict = {
                    "CORE": [], "ROTATION": [], "MOMENTUM": [], "DIP": [],
                    "UNCATEGORIZED": [],
                }
                portfolio_cost = 0.0
                portfolio_value = 0.0
                ticker_current_value: dict[str, float] = {}  # for concentration math
                for sym, pos in positions.items():
                    bucket = (pos.get("category") or "").upper()
                    if bucket not in portfolio_by_bucket:
                        bucket = "UNCATEGORIZED"
                    portfolio_by_bucket[bucket].append((sym, pos))
                    portfolio_cost += pos.get("total_cost", 0.0)
                    # Live value: yfinance current × shares. ROTATION UCITS
                    # fall back to avg_price (yfinance has no UCITS data).
                    avg = float(pos.get("avg_price", 0) or 0)
                    shares = float(pos.get("shares", 0) or 0)
                    current_px = avg
                    if bucket != "ROTATION":
                        try:
                            current_px = float(cr._fetch_price(sym, fallback=avg))
                        except Exception:
                            current_px = avg
                    val = current_px * shares
                    ticker_current_value[sym.upper()] = val
                    portfolio_value += val

                sell_actions: list[dict] = []
                buy_actions:  list[dict] = []
                info_lines:   list[str] = []

                def _can_top_up(ticker: str, requested_eur: float) -> tuple[bool, float, str]:
                    """Concentration cap gate.

                    Returns (allowed, max_alloc_eur, reason). Allows up to
                    CONCENTRATION_HARD (30%) — typical buys are sized to land
                    within CONCENTRATION_SOFT (25%) but the hard cap is the
                    binding constraint for the alloc.
                    """
                    if portfolio_value <= 0:
                        return True, requested_eur, ""
                    held = ticker_current_value.get(ticker.upper(), 0.0)
                    held_pct = held / portfolio_value
                    if held_pct >= CONCENTRATION_HARD:
                        return False, 0.0, (
                            f"concentration {held_pct*100:.1f}% ≥ hard cap "
                            f"{CONCENTRATION_HARD*100:.0f}%"
                        )
                    # Cap so post-buy stays under hard cap on the new total
                    new_total = portfolio_value + requested_eur
                    max_held_eur = new_total * CONCENTRATION_HARD
                    headroom_eur = max(0.0, max_held_eur - held)
                    allowed = min(requested_eur, headroom_eur)
                    if allowed < requested_eur - 0.5:  # capped
                        return True, allowed, (
                            f"capped at €{allowed:.0f} (post-buy would be "
                            f"{(held + allowed) / new_total * 100:.1f}%, "
                            f"hard cap {CONCENTRATION_HARD*100:.0f}%)"
                        )
                    return True, allowed, ""

                # ╔════════════════════════════════════════════════════════╗
                # ║ PASS 1 — Compute all proposed SELLs (freed_cash)       ║
                # ╚════════════════════════════════════════════════════════╝

                # ── ROTATION sells: tax-friction-aware rotation gate ─────
                # A ROTATION holding sells ONLY when (target - current) score
                # gap exceeds the tax friction implied by realising the
                # unrealized gain (see tax_engine.should_rotate). Override
                # cases: structural exit, severe deterioration, sector decay.
                rotation_top3 = []
                rotation_holdings = {sym: pos for sym, pos in portfolio_by_bucket["ROTATION"]}
                rotation_holds_kept: list[dict] = []  # HOLD lines for display
                rotation_ranked = []
                if not skip_rotation:
                    try:
                        from core.sector_radar import rank_universe
                        from core.etf_universe import all_monitored_tickers, get_etf
                        all_t = all_monitored_tickers()
                        ranked = rank_universe(all_t)
                        rotation_ranked = ranked
                        top_signals = ranked[:3] if ranked else []
                        for sig in top_signals:
                            etf = get_etf(sig.ticker)
                            ucits = etf.ucits_ticker if etf and etf.ucits_ticker else sig.ticker
                            rotation_top3.append({
                                "us_ticker":     sig.ticker,
                                "ucits":         ucits,
                                "score":         sig.combined_buy,
                                "exit_strength": getattr(sig, "exit_strength", 0.0),
                                "secular_risk":  getattr(sig, "secular_risk", 0.0),
                                "already_held":  ucits in rotation_holdings,
                            })

                        top3_ucits_set = {r["ucits"] for r in rotation_top3}
                        best_target_score = rotation_top3[0]["score"] if rotation_top3 else 0.0

                        # Build US-proxy → current score map for held UCITS
                        # (so we can compute the "current score" of each
                        # holding by looking up its US proxy in ranked).
                        from core.etf_universe import USER_HOLDINGS_UCITS
                        held_score_map: dict[str, float] = {}
                        for sig in ranked:
                            for ucits, us_proxy in USER_HOLDINGS_UCITS.items():
                                if sig.ticker == us_proxy and ucits in rotation_holdings:
                                    held_score_map[ucits] = sig.combined_buy

                        # Use US proxy for UCITS PnL estimation (yfinance
                        # has no UCITS data). proxy % change since entry →
                        # apply to cost basis → estimate current value.
                        from core.ucits_proxy import estimate_ucits_current_price

                        for sym, pos in rotation_holdings.items():
                            if sym in top3_ucits_set:
                                continue  # in top-3, keep
                            avg_price = float(pos.get("avg_price", 0.0))
                            shares    = float(pos.get("shares", 0.0))
                            cost      = avg_price * shares

                            # Estimate UCITS current price via US proxy
                            current_est = estimate_ucits_current_price(
                                sym, avg_price,
                                entry_date_iso=pos.get("entry_date_iso"),
                            )
                            if current_est is not None and current_est > 0:
                                value = current_est * shares
                                unrealized_gain_pct: float | None = (current_est / avg_price) - 1.0
                            else:
                                # Proxy fetch failed → unknown PnL (None signals
                                # tax_engine to use conservative default)
                                value = cost
                                unrealized_gain_pct = None

                            # current_score lookup: held_score_map returns the US
                            # proxy's combined_buy. Missing key → None (data-starved).
                            score_for_sym = held_score_map.get(sym)

                            do_sell, reason = tax_should_rotate(
                                unrealized_gain_pct=unrealized_gain_pct,
                                current_score=score_for_sym,
                                target_score=best_target_score,
                            )
                            pnl_pct_display = (
                                unrealized_gain_pct * 100
                                if unrealized_gain_pct is not None else 0.0
                            )
                            if do_sell:
                                sell_actions.append({
                                    "ticker":  sym,
                                    "bucket":  "ROTATION",
                                    "reason":  reason,
                                    "shares":  shares,
                                    "value":   value,
                                    "current_pnl_pct": pnl_pct_display,
                                })
                            else:
                                rotation_holds_kept.append({
                                    "ticker":  sym,
                                    "shares":  shares,
                                    "value":   value,
                                    "reason":  reason,
                                })
                    except Exception as _e:
                        info_lines.append(f"⚠️ Rotation scoring falhou: `{_e}`")
                        rotation_top3 = []

                # ── MOMENTUM/DIP sells: positions below ATR trailing stop ─
                exit_warnings_list = []
                try:
                    from core import conviction_radar as cr
                    momentum_positions = portfolio_by_bucket.get("MOMENTUM", []) + \
                                         portfolio_by_bucket.get("DIP", [])
                    for sym, pos in momentum_positions:
                        try:
                            conv = cr.get_by_ticker(sym)
                            if conv is None:
                                continue
                            ev = cr.evaluate_conviction(conv)
                            triggers = ev.get("triggers", [])
                            high_sev = any(t.get("severity") == "high" for t in triggers)
                            med_sev = any(t.get("severity") == "med" for t in triggers)
                            current_price = float(ev.get("current_price", 0) or pos.get("avg_price", 0))
                            avg_price = pos.get("avg_price", 0.0)
                            shares = pos.get("shares", 0.0)
                            pnl_pct = (current_price / avg_price - 1.0) * 100 if avg_price > 0 else 0.0
                            value_at_current = current_price * shares

                            if high_sev:
                                sell_actions.append({
                                    "ticker":  sym,
                                    "bucket":  pos.get("category", "MOMENTUM"),
                                    "reason":  next(
                                        (t.get("message", "trigger HIGH") for t in triggers if t.get("severity") == "high"),
                                        "trigger HIGH"
                                    ),
                                    "shares":  shares,
                                    "value":   value_at_current,
                                    "current_pnl_pct": pnl_pct,
                                })
                            elif med_sev:
                                exit_warnings_list.append({
                                    "ticker":  sym,
                                    "bucket":  pos.get("category", "MOMENTUM"),
                                    "shares":  shares,
                                    "value":   value_at_current,
                                    "current_pnl_pct": pnl_pct,
                                    "reason":  next(
                                        (t.get("message", "trigger MED") for t in triggers if t.get("severity") == "med"),
                                        "trigger MED"
                                    ),
                                })
                        except Exception:
                            continue
                except Exception as _e:
                    info_lines.append(f"⚠️ Conviction monitor check falhou: `{_e}`")

                # ── Total freed cash from SELLs ────────────────────────
                if ignore_sells:
                    freed_cash = 0.0
                    sell_actions_to_show = sell_actions  # display but exclude from budget
                else:
                    freed_cash = sum(s["value"] for s in sell_actions)
                    sell_actions_to_show = sell_actions

                total_budget = cash_amount + freed_cash

                # ╔════════════════════════════════════════════════════════╗
                # ║ PASS 2 — Build candidate BUY queue, rank by ROI        ║
                # ╚════════════════════════════════════════════════════════╝
                # No per-radar budget reserves. Every candidate buy across
                # Rotation, Core, and DIP is annualized to a comparable
                # expected alpha, ranked DESC, and deployed top-down until
                # budget exhausted (or concentration cap kicks in).

                candidates: list[dict] = []
                # Default suggested allocation per signal — radar-specific
                # but adjustable. We size big for high-conviction (Core 100)
                # and modest for short-horizon DIP (more positions, smaller
                # each, since the model's pred_alpha already prices the bet).
                # Allocation: Full Kelly Criterion driven by ML predictions
                # + calibration-derived downside. NO magic numbers — the
                # math comes from "maximize log-wealth growth" theory.
                #
                # Kelly fraction: f* = (p × b − (1 − p)) / b
                #   where p = win_prob (from ML model)
                #         b = alpha_90d / |downside_q05|  (both data-driven)
                #
                # Position size: f* × budget × KELLY_FRACTION
                #
                # KELLY_FRACTION:
                #   1.0 → Full Kelly: maximum theoretical growth, brutal
                #         drawdowns (-50% common before recovery)
                #   0.5 → Half Kelly: 75% of growth, ~half the drawdown
                #         (academic consensus)
                #   0.25 → Quarter Kelly: very conservative
                #
                # User chose Full Kelly (1.0) on 2026-06-12 ('máximo ROI').
                # Override via env var DIPRADAR_KELLY_FRACTION.
                import os as _os_kelly
                try:
                    KELLY_FRACTION = float(_os_kelly.environ.get("DIPRADAR_KELLY_FRACTION", "1.0"))
                except (TypeError, ValueError):
                    KELLY_FRACTION = 1.0
                KELLY_FRACTION = max(0.05, min(1.0, KELLY_FRACTION))

                # Fallback downside when no calibration bucket has data
                # (e.g. ROTATION + CORE if their own calibrations aren't
                # built yet). Not a magic number — conservative historical
                # estimate from DIP backtest tail behaviour.
                _DEFAULT_DOWNSIDE_FALLBACK = 0.15

                def _kelly_full(win_prob, alpha, downside, budget, fraction=KELLY_FRACTION):
                    """Full Kelly position size given p, α, downside, budget.

                    Returns 0 when any input is missing/non-positive. This
                    correctly handles ADBE-style cases where the ML model
                    sees no edge → no deploy.
                    """
                    if budget <= 0:
                        return 0.0
                    if win_prob is None or alpha is None or downside is None:
                        return 0.0
                    try:
                        p   = float(win_prob)
                        a   = float(alpha)
                        d   = float(downside)
                    except (TypeError, ValueError):
                        return 0.0
                    if p <= 0 or a <= 0 or d <= 0:
                        # Model says no edge → don't deploy. This is the right
                        # behaviour: better to hold cash than bet against the
                        # model's own probability assessment.
                        return 0.0
                    b = a / d
                    if b <= 0:
                        return 0.0
                    f_kelly = (p * b - (1.0 - p)) / b
                    if f_kelly <= 0:
                        return 0.0
                    return float(budget) * f_kelly * fraction

                # ── Rotation candidates: top-3 (NEW or top-up) ──────────
                # Equal allocation within top-3: the calibration test
                # showed CV r² ≤ 0.007 across all methods (linear, poly,
                # buckets, lightgbm). rotation_score identifies WHO is
                # in top-3 but doesn't predict per-pick alpha within it.
                # Score-weighted sizing would be adding noise — equal
                # is the honest choice given the data.
                if not skip_rotation and rotation_top3:
                    n_picks = len(rotation_top3)
                    # ROTATION uses Kelly per pick (not equal_size anymore).
                    # win_prob estimated from rotation_score (60 ≈ 55%,
                    # 80 ≈ 75%) since rotation has no per-pick ML.
                    for r in rotation_top3:
                        ann_alpha = _annualized_alpha(
                            "ROTATION", score=r["score"]
                        )
                        _rot_score = float(r["score"])
                        _rot_winp  = max(0.5, min(0.85, 0.5 + (_rot_score - 50.0) / 100.0))
                        _rot_kelly = _kelly_full(
                            win_prob=_rot_winp,
                            alpha=ann_alpha / (365.0 / 90.0),  # back to 90d alpha for Kelly
                            downside=_DEFAULT_DOWNSIDE_FALLBACK,
                            budget=total_budget,
                        )
                        candidates.append({
                            "ticker":     r["ucits"],
                            "bucket":     "ROTATION",
                            "reason":     f"rotation top-{rotation_top3.index(r)+1} score {r['score']:.0f}",
                            "suggested":  _rot_kelly,
                            "ann_alpha":  ann_alpha,
                            "confidence": 1.0,
                            "extra":      {
                                "us_proxy":     r["us_ticker"],
                                "score":        r["score"],
                                "win_prob_est": _rot_winp,
                                "kelly_alloc":  _rot_kelly,
                            },
                        })

                # ── Core picks: all (including held — concentration cap gates) ─
                try:
                    from core.core_radar import scan_core_universe
                    core_opps = scan_core_universe(min_score=70.0)
                    for p in core_opps:
                        # Convert score back to alpha estimate for ranking.
                        # Score uses tanh(alpha × 3) × 50 + 50, so inverse:
                        # alpha = atanh((score - 50) / 50) / 3
                        import math as _m
                        score_norm = (float(p["score"]) - 50.0) / 50.0
                        score_norm = max(-0.999, min(0.999, score_norm))
                        alpha_1y = _m.atanh(score_norm) / 3.0
                        # Kelly for CORE: estimate win_prob from score
                        # (no ML for CORE per-ticker yet). score 50 ≈ random
                        # (50%), score 100 ≈ near-certainty (90%). Linear map.
                        _core_score = float(p["score"])
                        _core_winp  = max(0.5, min(0.9, 0.5 + (_core_score - 50.0) / 100.0))
                        _core_kelly = _kelly_full(
                            win_prob=_core_winp,
                            alpha=alpha_1y,
                            downside=_DEFAULT_DOWNSIDE_FALLBACK,  # no CORE calibration yet
                            budget=total_budget,
                        )
                        candidates.append({
                            "ticker":     p["ticker"],
                            "bucket":     "CORE",
                            "reason":     f"Core score {p['score']:.0f} (${p.get('current_price', 0):.2f})",
                            "suggested":  _core_kelly,
                            "ann_alpha":  alpha_1y,
                            "confidence": 1.0,
                            "extra":      {
                                "score":          p["score"],
                                "price":          p.get("current_price"),
                                "win_prob_est":   _core_winp,
                                "kelly_alloc":    _core_kelly,
                            },
                        })
                except Exception as _e:
                    info_lines.append(f"⚠️ Core picks falhou: `{_e}`")

                # ── DIP entries (real tickers only — WL_ filtered) ──────
                try:
                    import state as _st
                    from ml.prediction_log import get_recent_predictions_by_ticker
                    from scanners.alert_db import get_recent_alerts_pred_alpha

                    raw_alerts = list(_st.load_alerts())
                    today_alerts = [
                        t for t in raw_alerts
                        if not str(t).upper().startswith("WL_")
                        and "_" not in str(t)
                    ]
                    # Per-alert α resolution: prefer alert_db.csv (durable,
                    # migrate_schema archives instead of deleting) and fall
                    # back to ml_predictions.csv only if missing. Avoids the
                    # hardcoded-fallback case where every DIP buy gets the
                    # same α=20.3%/y just because prediction_log was reset
                    # by a recent FEATURE_COLUMNS schema drift.
                    recent_alerts_pred = get_recent_alerts_pred_alpha(today_alerts, lookback_days=7)
                    recent_preds = get_recent_predictions_by_ticker(today_alerts, lookback_days=7)

                    for ticker in today_alerts:
                        # Source priority:
                        #   1. alert_db (pred_alpha_90d + ml_win_prob captured
                        #      at signal time, durable across schema resets)
                        #   2. prediction_log (score_raw or win_prob from
                        #      ml_predictor's per-call log)
                        #   3. hardcoded fallback 0.05 (truly nothing known)
                        adb = recent_alerts_pred.get(ticker.upper(), {})
                        pred = recent_preds.get(ticker.upper(), {})

                        adb_alpha = adb.get("pred_alpha_90d")
                        adb_wp    = adb.get("ml_win_prob")
                        score_raw = pred.get("score_raw")
                        win_prob = pred.get("win_prob")

                        if adb_alpha is not None and adb_alpha > 0:
                            pred_alpha_90d = float(adb_alpha)
                            source = "alert_db.pred_alpha"
                            confidence = min(1.0, max(0.3, float(adb_wp or 0.6)))
                        elif score_raw is not None and score_raw > 0:
                            pred_alpha_90d = float(score_raw)
                            source = "score_raw"
                            confidence = min(1.0, max(0.3, (win_prob or 0.6)))
                        elif adb_wp is not None and adb_wp > 0:
                            # alert_db has win_prob but no alpha (e.g. low-conf
                            # alert pre-fix). Estimate alpha linearly.
                            pred_alpha_90d = max(0.0, (float(adb_wp) - 0.50) * 0.5)
                            source = f"alert_db.win_prob {adb_wp:.2f}"
                            confidence = min(1.0, max(0.3, float(adb_wp)))
                        elif win_prob is not None:
                            pred_alpha_90d = max(0.0, (float(win_prob) - 0.50) * 0.5)
                            source = f"win_prob {win_prob:.2f}"
                            confidence = min(1.0, max(0.3, float(win_prob)))
                        else:
                            # No data — alert pre-dates this fix.
                            pred_alpha_90d = 0.05
                            source = "no log entry — fallback"
                            confidence = 0.4

                        # Calibration: if data/dip_alpha_calibration.json
                        # exists AND is accepted (r²≥0.10, monotonic), look
                        # up the bucket whose pred range contains our
                        # pred_alpha_90d and replace with the bucket's
                        # mean_actual_alpha_90d. This is the "historical
                        # observation" override the user asked for — α
                        # reflects what tickers with this model output
                        # ACTUALLY delivered, not what the model said they
                        # would. V1: parquet in-sample (synthetic). V2:
                        # weekly refresh blends real mature alerts.
                        _calibrated_alpha = pred_alpha_90d
                        _bucket_downside  = _DEFAULT_DOWNSIDE_FALLBACK
                        _calib_note       = source
                        _horizon_days     = 90
                        try:
                            for _try in (
                                _Path("/data/dip_alpha_calibration.json"),
                                _Path("data/dip_alpha_calibration.json"),
                            ):
                                if _try.exists():
                                    _cd = _json.loads(_try.read_text())
                                    _horizon_days = int(_cd.get("horizon_days") or 90)
                                    if _cd.get("accepted"):
                                        for _b in _cd.get("buckets", []):
                                            if (pred_alpha_90d >= _b["pred_min"]
                                                    and pred_alpha_90d <= _b["pred_max"]):
                                                _calibrated_alpha = float(_b["mean_actual_alpha_90d"])
                                                _bucket_downside  = float(
                                                    _b.get("downside_q05_90d")
                                                    or _DEFAULT_DOWNSIDE_FALLBACK
                                                )
                                                _calib_note = (
                                                    f"{source} → calibrated bucket "
                                                    f"[{_b['pred_min']:+.3f},{_b['pred_max']:+.3f}]"
                                                )
                                                break
                                    break
                        except Exception:
                            pass

                        ann_alpha = _calibrated_alpha * (365.0 / float(_horizon_days))
                        # Full Kelly: pure math from ML predictions +
                        # calibration downside. No magic curve.
                        _dip_score = adb.get("score")
                        _kelly_alloc_eur = _kelly_full(
                            win_prob=win_prob,
                            alpha=_calibrated_alpha,
                            downside=_bucket_downside,
                            budget=total_budget,
                        )
                        candidates.append({
                            "ticker":     ticker,
                            "bucket":     "DIP",
                            "reason":     f"DIP alert ({_calib_note})",
                            "suggested":  _kelly_alloc_eur,
                            "ann_alpha":  ann_alpha,
                            "confidence": confidence,
                            "extra":      {
                                "pred_alpha_90d":    pred_alpha_90d,
                                "calibrated_alpha":  _calibrated_alpha,
                                "bucket_downside":   _bucket_downside,
                                "win_prob":          win_prob,
                                "score":             _dip_score,
                                "kelly_alloc":       _kelly_alloc_eur,
                            },
                        })
                except Exception as _e:
                    info_lines.append(f"⚠️ DIP queue falhou: `{_e}`")

                # ── HOLD: SELL+BUY same-ticker = contradictory → don't churn ─
                # If a ticker is queued to SELL (e.g. HUBS MOMENTUM stop break)
                # AND would also be bought from another radar (e.g. DIP alert),
                # the two signals contradict each other on the same name. Tax
                # friction + bid/ask + slippage + the conviction signal of
                # "model still likes this at lower price" all favor HOLD over
                # churn. Pull both rows out, surface the contradiction, and
                # recompute the budget against the smaller sell pool.
                _sell_set = {s["ticker"].upper() for s in sell_actions}
                _cand_set = {c["ticker"].upper() for c in candidates}
                _hold_tickers = _sell_set & _cand_set
                hold_actions: list[dict] = []
                if _hold_tickers:
                    for _ht in _hold_tickers:
                        _sell_row = next(s for s in sell_actions if s["ticker"].upper() == _ht)
                        _buy_row  = next(c for c in candidates    if c["ticker"].upper() == _ht)
                        hold_actions.append({
                            "ticker":       _sell_row["ticker"],
                            "bucket":       _sell_row.get("bucket", "?"),
                            "shares":       _sell_row.get("shares", 0),
                            "value":        _sell_row.get("value", 0.0),
                            "sell_reason":  _sell_row.get("reason", ""),
                            "buy_reason":   _buy_row.get("reason", ""),
                            "would_alloc":  _buy_row.get("suggested", 0.0),
                        })
                    sell_actions = [s for s in sell_actions if s["ticker"].upper() not in _hold_tickers]
                    candidates   = [c for c in candidates   if c["ticker"].upper() not in _hold_tickers]
                    # Recompute freed_cash + total_budget against filtered sells
                    if ignore_sells:
                        freed_cash = 0.0
                    else:
                        freed_cash = sum(s["value"] for s in sell_actions)
                    total_budget = cash_amount + freed_cash
                    sell_actions_to_show = sell_actions

                # ── Rank by ROI score = ann_alpha × max(0.5, confidence) ───
                for c in candidates:
                    c["roi_score"] = c["ann_alpha"] * max(0.5, c["confidence"])

                # Drop signals with confidence < 0.3 (per plan)
                candidates = [c for c in candidates if c["confidence"] >= 0.3]

                # Cap top 3 per bucket BEFORE the unified ranking — user
                # feedback (2026-06-12): showing 8 DIP candidates is noise,
                # they want max top-3 per radar to keep the proposal
                # actionable. ROTATION already returns 3 by design; this
                # caps DIP and CORE (which can each produce 5+ candidates
                # on volatile days).
                _PER_BUCKET_CAP = 3
                _by_bucket: dict[str, list[dict]] = {}
                for c in candidates:
                    _by_bucket.setdefault(c["bucket"], []).append(c)
                _capped: list[dict] = []
                for bucket, group in _by_bucket.items():
                    group.sort(key=lambda x: -x["roi_score"])
                    _capped.extend(group[:_PER_BUCKET_CAP])
                candidates = _capped

                candidates.sort(key=lambda c: -c["roi_score"])

                # ── Deploy top-down respecting concentration cap ───────
                remaining = total_budget
                ranked_deploys: list[dict] = []  # for display

                for c in candidates:
                    if remaining <= 5:
                        break  # negligible
                    requested = min(c["suggested"], remaining)
                    allowed, capped_alloc, cap_reason = _can_top_up(c["ticker"], requested)
                    if not allowed or capped_alloc < 25:
                        if cap_reason:
                            info_lines.append(
                                f"⏭️ `{c['ticker']}` [{c['bucket']}] skipped — {cap_reason}"
                            )
                        continue
                    final_alloc = round(capped_alloc, 0)
                    if final_alloc < 25:
                        continue
                    reason = c["reason"]
                    if cap_reason:
                        reason = f"{reason} | {cap_reason}"
                    # Update virtual portfolio for next iteration's cap math
                    ticker_current_value[c["ticker"].upper()] = (
                        ticker_current_value.get(c["ticker"].upper(), 0.0) + final_alloc
                    )
                    buy_actions.append({
                        "ticker": c["ticker"],
                        "bucket": c["bucket"],
                        "reason": reason,
                        "value":  final_alloc,
                        "ann_alpha_pct": c["ann_alpha"] * 100,
                        "extra":  c.get("extra") or {},
                    })
                    ranked_deploys.append({
                        "ticker":   c["ticker"],
                        "bucket":   c["bucket"],
                        "roi":      c["roi_score"] * 100,
                        "alpha":    c["ann_alpha"] * 100,
                    })
                    remaining -= final_alloc

                # ── Cash bucket: whatever's left ───────────────────────
                cash_bucket = max(0.0, remaining)

                # Persist reserve so future scanner alerts can surface it.
                # The user explicitly opted for "deploy when opportunity
                # arises, never forced" — we never auto-buy, only display
                # availability alongside high-conviction signals.
                if cash_bucket > 0:
                    try:
                        import state as _st_cr
                        _st_cr.set_cash_reserve(cash_bucket, source="/cash")
                    except Exception as _e:
                        info_lines.append(f"⚠️ Cash reserve persist failed: `{_e}`")

                # ╔════════════════════════════════════════════════════════╗
                # ║ Build output                                           ║
                # ╚════════════════════════════════════════════════════════╝

                lines: list[str] = []
                lines.append(f"💰 *Allocation Proposal*")
                lines.append("")
                lines.append(
                    f"📊 *Portfolio*: cost €{portfolio_cost:.2f} | "
                    f"live €{portfolio_value:.2f} | "
                    f"{len(positions)} posições"
                )
                lines.append(
                    f"   CORE: {len(portfolio_by_bucket['CORE'])} | "
                    f"ROTATION: {len(portfolio_by_bucket['ROTATION'])} | "
                    f"MOMENTUM: {len(portfolio_by_bucket['MOMENTUM'])} | "
                    f"DIP: {len(portfolio_by_bucket['DIP'])}"
                )
                lines.append("")
                lines.append(f"💵 *Budget*:")
                lines.append(f"   Cash inflow:      €{cash_amount:,.2f}")
                lines.append(
                    f"   Freed from sells: €{freed_cash:,.2f}"
                    f"{' (ignorado)' if ignore_sells else ''}"
                )
                lines.append(f"   *TOTAL budget*:   €{total_budget:,.2f}")
                lines.append("")

                # SELLs section
                if sell_actions_to_show:
                    lines.append(f"🔻 *SELLs recomendados* ({len(sell_actions_to_show)}):")
                    for s in sell_actions_to_show:
                        pnl_str = f"({s['current_pnl_pct']:+.1f}%)" if s['current_pnl_pct'] != 0 else ""
                        lines.append(
                            f"  • `{s['ticker']}` [{s['bucket']}]: "
                            f"{s['shares']:g} → €{s['value']:.0f} {pnl_str}"
                        )
                        lines.append(f"      _{s['reason']}_")
                    lines.append("")

                # ROTATION HOLDs (kept by tax-aware gate — informational)
                if rotation_holds_kept:
                    lines.append(
                        f"🟢 *ROTATION HOLDs* ({len(rotation_holds_kept)} — "
                        f"tax-friction gate kept):"
                    )
                    for h in rotation_holds_kept:
                        lines.append(
                            f"  • `{h['ticker']}`: {h['shares']:g} sh "
                            f"(€{h['value']:.0f}) — _{h['reason']}_"
                        )
                    lines.append("")

                # HOLDs from SELL+BUY contradiction (same-ticker, two radars)
                if hold_actions:
                    lines.append(
                        f"⚖️ *HOLDs* ({len(hold_actions)} — sinais contraditórios, "
                        f"sem churn):"
                    )
                    for h in hold_actions:
                        lines.append(
                            f"  • `{h['ticker']}` [{h['bucket']}]: "
                            f"{h['shares']:g} sh (€{h['value']:.0f})"
                        )
                        lines.append(f"      SELL: _{h['sell_reason']}_")
                        lines.append(
                            f"      BUY would re-enter @ €{h['would_alloc']:.0f}: "
                            f"_{h['buy_reason']}_"
                        )
                    lines.append("")

                # EXIT WARNINGS (medium severity, not yet sell)
                if exit_warnings_list:
                    lines.append(f"⚠️ *EXIT WARNINGS* ({len(exit_warnings_list)} — sinais MED, monitorizar):")
                    for w in exit_warnings_list:
                        lines.append(
                            f"  • `{w['ticker']}` [{w['bucket']}]: "
                            f"{w['shares']:g} ({w['current_pnl_pct']:+.1f}%) — _{w['reason']}_"
                        )
                    lines.append("")

                # ROTATION top-3 status (informational)
                if not skip_rotation and rotation_top3:
                    lines.append(f"🔄 *ROTATION* top-3 hoje:")
                    for i, r in enumerate(rotation_top3, 1):
                        held_tag = " ✓ tens" if r["already_held"] else " (NOVO)"
                        lines.append(
                            f"  {i}. `{r['us_ticker']}` → `{r['ucits']}` "
                            f"score {r['score']:.0f}{held_tag}"
                        )
                    lines.append("")

                # BUYs section — ranked by ROI
                if buy_actions:
                    lines.append(f"✅ *BUYs propostos* ({len(buy_actions)}, ROI-ranked):")
                    for b in buy_actions:
                        ann_str = (
                            f" • α={b.get('ann_alpha_pct', 0):.1f}%/y"
                            if b.get("ann_alpha_pct") else ""
                        )
                        # Surface the score that drove the candidate into
                        # the queue — was missing per user feedback. Pulled
                        # from extra (rotation/core write it directly; DIP
                        # now also writes via alert_db lookup).
                        _bscore = (b.get("extra") or {}).get("score")
                        score_tag = f" score *{float(_bscore):.0f}*" if _bscore is not None else ""
                        lines.append(
                            f"  • `{b['ticker']}` [{b['bucket']}]{score_tag}: "
                            f"€{b['value']:.0f}{ann_str}"
                        )
                        lines.append(f"      _{b['reason']}_")
                    lines.append("")

                # Cash bucket — explicit before/after summary so the user
                # sees exactly what stays liquid post-execution (was
                # ambiguous in the previous format).
                _deployed_total = sum(b["value"] for b in buy_actions)
                lines.append("📥 *Pós-execução*:")
                lines.append(
                    f"   Deployed em {len(buy_actions)} compras: *€{_deployed_total:,.0f}*"
                )
                lines.append(
                    f"   Cash que fica em T212: *€{cash_bucket:,.2f}*"
                )
                if cash_bucket > 0:
                    lines.append(
                        f"   _T212 @ 2.2%/ano = €{cash_bucket * 0.022:.2f}/ano. "
                        f"Dry powder para próximos Core/DIP signals._"
                    )
                lines.append("")

                # Info messages (warnings about failed checks)
                if info_lines:
                    lines.append("ℹ️ *Notas técnicas*:")
                    lines.extend(info_lines)
                    lines.append("")

                # Action checklist for T212
                if sell_actions_to_show or buy_actions:
                    lines.append("📋 *CHECKLIST para T212*:")
                    step = 1
                    for s in sell_actions_to_show:
                        lines.append(
                            f"  {step}. SELL `{s['ticker']}` ({s['shares']:g} shares)"
                        )
                        step += 1
                    for b in buy_actions:
                        lines.append(
                            f"  {step}. BUY `{b['ticker']}` €{b['value']:.0f}"
                        )
                        step += 1

                # Persist the proposal so /deployed can look up the
                # recommendation per ticker afterward. Stored to /data/
                # (Railway volume) to survive deploys.
                try:
                    from datetime import datetime as _dt
                    _proposal_path = (
                        _Path("/data") if _Path("/data").exists() else _Path(__file__).parent / "data"
                    ) / "last_cash_proposal.json"
                    _proposal_path.parent.mkdir(parents=True, exist_ok=True)
                    _proposal = {
                        "timestamp":    _dt.now().isoformat(timespec="seconds"),
                        "budget":       float(total_budget),
                        "cash_inflow":  float(cash_amount),
                        "freed_cash":   float(freed_cash),
                        "actions":      [
                            {
                                "ticker":            b["ticker"],
                                "bucket":            b["bucket"],
                                "recommended_eur":   float(b["value"]),
                                "ann_alpha_pct":     b.get("ann_alpha_pct"),
                                "score":             (b.get("extra") or {}).get("score"),
                                "win_prob":          (b.get("extra") or {}).get("win_prob"),
                                "kelly_alloc":       (b.get("extra") or {}).get("kelly_alloc"),
                            }
                            for b in buy_actions
                        ],
                    }
                    _proposal_path.write_text(_json.dumps(_proposal, indent=2))
                except Exception as _pe:
                    logging.debug(f"[cash] persist proposal failed: {_pe}")

                _reply("\n".join(lines))
            except Exception as _e:
                _reply(f"❌ Allocation falhou: `{_e}`")

        threading.Thread(
            target=_run_allocation, daemon=True, name="cash-allocation"
        ).start()

    elif cmd in ("/deployed", "/deploy"):
        # Record an actual deploy + optional reason for deviation from the
        # last /cash recommendation. Usage:
        #   /deployed TICKER AMOUNT_EUR [REASON...]
        # Examples:
        #   /deployed PTC 100
        #   /deployed ADBE 50 achei agressivo
        #   /deployed NVO 200 livre vontade — gosto mais
        if len(parts) < 3:
            _reply(
                "*Uso:* `/deployed TICKER AMOUNT [reason]`\n\n"
                "Examples:\n"
                "  `/deployed PTC 100`\n"
                "  `/deployed ADBE 50 achei agressivo demais`\n"
                "  `/deployed NVO 200 livre vontade`"
            )
        else:
            from pathlib import Path as _P
            import json as _J
            from datetime import datetime as _dt
            _ticker  = parts[1].upper().strip()
            try:
                _amount = float(parts[2])
            except ValueError:
                _reply(f"❌ Amount inválido: `{parts[2]}`")
                return
            _reason  = " ".join(parts[3:]).strip() if len(parts) > 3 else ""

            # Load last proposal for comparison
            _dd = _P("/data") if _P("/data").exists() else _P(__file__).parent / "data"
            _proposal_path = _dd / "last_cash_proposal.json"
            _recommended = None
            _proposal_ts = None
            if _proposal_path.exists():
                try:
                    _prop = _J.loads(_proposal_path.read_text())
                    _proposal_ts = _prop.get("timestamp")
                    for _a in _prop.get("actions", []):
                        if _a.get("ticker", "").upper() == _ticker:
                            _recommended = float(_a.get("recommended_eur") or 0)
                            break
                except Exception:
                    pass

            _delta_eur = (_amount - _recommended) if _recommended is not None else None
            _delta_pct = (
                (_delta_eur / _recommended * 100.0)
                if (_delta_eur is not None and _recommended and _recommended > 0)
                else None
            )

            # Classify reason if not provided + delta is significant
            _category = None
            if not _reason and _delta_pct is not None and abs(_delta_pct) >= 10:
                # Significant delta with no reason — ask user to amend
                _reply(
                    f"⚠️ *{_ticker}* deploy €{_amount:.0f} vs recomendado €{_recommended:.0f} "
                    f"(*{_delta_pct:+.0f}%*).\n\n"
                    f"_Adiciona razão para o bot aprender:_\n"
                    f"  `/deployed {_ticker} {_amount:g} achei agressivo` (deploy menor)\n"
                    f"  `/deployed {_ticker} {_amount:g} forte convicção` (deploy maior)\n"
                    f"  `/deployed {_ticker} {_amount:g} livre vontade` (sem razão específica)"
                )
                return

            # Auto-classify from reason keywords
            _r_lower = _reason.lower()
            if any(w in _r_lower for w in ("agressivo", "muito", "demasiado", "menos", "cauteloso")):
                _category = "more_conservative"
            elif any(w in _r_lower for w in ("convicção", "convicao", "forte", "mais", "deploy maior")):
                _category = "more_aggressive"
            elif any(w in _r_lower for w in ("livre", "vontade", "intuição", "intuicao", "feeling")):
                _category = "free_choice"
            elif _reason:
                _category = "other"
            else:
                _category = "no_deviation" if _delta_pct is None or abs(_delta_pct) < 10 else "unspecified"

            # Append to history
            _hist_path = _dd / "deploy_history.json"
            _hist_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                _hist = _J.loads(_hist_path.read_text()) if _hist_path.exists() else []
            except Exception:
                _hist = []
            _entry = {
                "timestamp":           _dt.now().isoformat(timespec="seconds"),
                "ticker":              _ticker,
                "deployed_eur":        _amount,
                "recommended_eur":     _recommended,
                "delta_eur":           _delta_eur,
                "delta_pct":           _delta_pct,
                "reason":              _reason or None,
                "category":            _category,
                "proposal_timestamp":  _proposal_ts,
            }
            _hist.append(_entry)
            # Keep last 200 entries
            if len(_hist) > 200:
                _hist = _hist[-200:]
            _hist_path.write_text(_J.dumps(_hist, indent=2))

            # Reply
            if _recommended is None:
                _reply(
                    f"✅ *{_ticker}*: €{_amount:.0f} registado.\n"
                    f"_(Sem recomendação prévia no /cash para comparação.)_\n\n"
                    f"Reason: _{_reason or '—'}_"
                )
            elif _delta_pct is None or abs(_delta_pct) < 1:
                _reply(
                    f"✅ *{_ticker}*: €{_amount:.0f} = recomendação. *Match perfeito.*"
                )
            else:
                _emoji = "📈" if _delta_pct > 0 else "📉"
                _reply(
                    f"✅ *{_ticker}*: €{_amount:.0f} registado.\n"
                    f"{_emoji} Recomendado: €{_recommended:.0f} | Delta: *{_delta_pct:+.0f}%* ({_delta_eur:+.0f}€)\n"
                    f"Reason: _{_reason or '—'}_  ·  Category: `{_category}`"
                )

    elif cmd in ("/deploy_history", "/deploys"):
        # Show last N deploys with deltas + reasons. Default N=20.
        from pathlib import Path as _P
        import json as _J
        _dd = _P("/data") if _P("/data").exists() else _P(__file__).parent / "data"
        _hist_path = _dd / "deploy_history.json"
        if not _hist_path.exists():
            _reply("📭 _Sem deploys registados ainda. Use `/deployed TICKER AMOUNT` após executar no T212._")
        else:
            try:
                _hist = _J.loads(_hist_path.read_text())
            except Exception:
                _hist = []
            if not _hist:
                _reply("📭 _Histórico vazio._")
            else:
                _n = 20
                if len(parts) > 1:
                    try:
                        _n = max(1, min(100, int(parts[1])))
                    except ValueError:
                        pass
                _recent = _hist[-_n:]
                lines = [f"*📜 Deploy history (últimos {len(_recent)})*", ""]
                # Stats
                _matches      = sum(1 for e in _recent if e.get("delta_pct") is not None and abs(e["delta_pct"]) < 1)
                _conservative = sum(1 for e in _recent if e.get("category") == "more_conservative")
                _aggressive   = sum(1 for e in _recent if e.get("category") == "more_aggressive")
                _free         = sum(1 for e in _recent if e.get("category") == "free_choice")
                lines.append(
                    f"_Match: {_matches} · Conservador: {_conservative} · "
                    f"Agressivo: {_aggressive} · Livre: {_free}_"
                )
                lines.append("")
                for e in reversed(_recent):
                    _t  = e.get("ticker", "?")
                    _a  = e.get("deployed_eur", 0)
                    _r  = e.get("recommended_eur")
                    _dp = e.get("delta_pct")
                    _ts = (e.get("timestamp") or "")[:16].replace("T", " ")
                    if _r is None or _dp is None:
                        lines.append(f"`{_ts}` *{_t}* €{_a:.0f} _(no rec)_")
                    elif abs(_dp) < 1:
                        lines.append(f"`{_ts}` *{_t}* €{_a:.0f} ✓ match")
                    else:
                        _arrow = "📈" if _dp > 0 else "📉"
                        lines.append(
                            f"`{_ts}` *{_t}* €{_a:.0f} (rec €{_r:.0f}, {_dp:+.0f}%) {_arrow}"
                        )
                    if e.get("reason"):
                        lines.append(f"   _{e['reason'][:80]}_")
                _reply("\n".join(lines))

    elif cmd in ("/admin_core_debug", "/core_debug"):
        # Debug a single ticker through the Core radar scoring pipeline.
        # Useful when a stock unexpectedly drops in or out of /core_picks
        # (e.g., NVO at score 80 one scan, missing the next). Surfaces:
        #   - all yfinance fundamentals fetched
        #   - is_investment_phase result + reason
        #   - gate triggers (rev_growth, gross_margin, MA200w)
        #   - final score + source (ML vs heuristic)
        if len(parts) < 2:
            _reply("Uso: `/admin_core_debug TICKER`\nEx: `/admin_core_debug NVO`")
            return
        ticker = parts[1].upper().strip()
        _reply(f"🔬 *Core debug* — {ticker} — scan iniciado...")

        def _run_debug():
            try:
                from core.core_radar import (
                    _fetch_hist, _fetch_quality_metrics, _score_core_dip,
                    _is_investment_phase, _load_universe,
                )
                # Static yield from universe
                universe = _load_universe() or []
                ann_yield_static = 0.0
                for e in universe:
                    if e.get("ticker") == ticker:
                        ann_yield_static = float(e.get("annual_yield_pct") or 0.0)
                        break

                # Fetch history
                hist = _fetch_hist(ticker)
                if hist is None or hist.empty:
                    _reply(f"❌ Sem histórico para {ticker} (yfinance falhou)")
                    return

                close = hist["Close"]
                volume = hist["Volume"] if "Volume" in hist.columns else None
                current_price = float(close.iloc[-1])

                # Fetch quality metrics
                quality = _fetch_quality_metrics(ticker)

                # Investment phase detector
                rev = quality.get("revenue_growth")
                gm = quality.get("gross_margin")
                roe = quality.get("roe")
                fcf = quality.get("fcf_yield")
                ip = _is_investment_phase(rev, gm, roe, fcf)

                # Run full scoring
                score, signals = _score_core_dip(
                    close, current_price, ann_yield_static,
                    ticker=ticker, volume=volume,
                )

                # Build report
                lines = [
                    f"🔬 *Core debug — {ticker}*",
                    f"",
                    f"💰 *Price*: ${current_price:.2f}",
                    f"📥 *Static yield (universe)*: {ann_yield_static:.2f}%",
                    f"📤 *Live yield (yfinance)*: "
                    f"{signals.get('dividend_yield_live'):.2f}%"
                    if signals.get('dividend_yield_live') is not None
                    else "📤 *Live yield (yfinance)*: N/A",
                    f"",
                    f"🏗 *Fundamentals (yfinance)*:",
                    f"  revenue_growth: `{rev}`",
                    f"  gross_margin:   `{gm}`",
                    f"  roe:            `{roe}`",
                    f"  fcf_yield:      `{fcf}`",
                    f"  analyst_upside: `{quality.get('analyst_upside')}`",
                    f"",
                    f"🌱 *Investment-phase detector*: `{ip}`",
                    f"  • fcf < 0: `{fcf is not None and fcf < 0}`",
                    f"  • rev > 10%: `{rev is not None and rev > 0.10}`",
                    f"  • gm > 40%: `{gm is not None and gm > 0.40}`",
                    f"  • roe > 15%: `{roe is not None and roe > 0.15}`",
                    f"",
                    f"⛔ *Hard gates*:",
                    f"  rev_growth < -10%: `{rev is not None and rev < -0.10}` (kill if True)",
                    f"  gross_margin < 25%: `{gm is not None and gm < 0.25}` (kill if True)",
                    f"",
                    f"📊 *Score*: `{score}`",
                    f"  source: `{signals.get('score_source', '?')}`",
                    f"  fcf_override_applied: `{signals.get('fcf_override_applied', False)}`",
                    f"",
                    f"📉 *Price signals*:",
                    f"  near_ma200w: `{signals.get('near_ma200w')}`",
                    f"  below_ma200w: `{signals.get('below_ma200w')}`",
                    f"  return_20d: `{signals.get('return_20d')}`",
                    f"  rsi_14: `{signals.get('rsi_14')}`",
                    f"  dd_from_52w_high: `{signals.get('dd_from_52w_high')}`",
                ]
                _reply("\n".join(lines))
            except Exception as _e:
                _reply(f"❌ Debug falhou: `{_e}`")
                import traceback
                _reply(f"```\n{traceback.format_exc()[-1500:]}\n```")

        threading.Thread(
            target=_run_debug, daemon=True, name=f"core-debug-{ticker}"
        ).start()

    elif cmd in ("/core_picks", "/admin_core_picks", "/core_top"):
        # Scan the Core universe (~300 quality compounders) and return the
        # current top-scored opportunities. Uses core_radar.scan_core_universe
        # which combines the ML model (champion=GBR, CV_IC=+0.118) with
        # rule-based gates (rev_growth, gross_margin) and dip detection.
        #
        # Without this, the Core radar runs blind — alerts fire automatically
        # but there's no way to query "who's the best Core buy today?".
        # The /cash command depends on this to suggest Core deployments.
        #
        # Usage:
        #   /core_picks            → top 10 with default min_score=60
        #   /core_picks 15         → top 15
        #   /core_picks 10 50      → top 10 with min_score=50 (more lenient)
        try:
            from core.core_radar import scan_core_universe

            top_n = 10
            min_score = 60.0
            if len(parts) >= 2:
                try:
                    top_n = max(1, min(50, int(parts[1])))
                except (ValueError, IndexError):
                    pass
            if len(parts) >= 3:
                try:
                    min_score = float(parts[2])
                except (ValueError, IndexError):
                    pass

            _reply(
                f"🏛 *Core picks a calcular* — top {top_n}, min score {min_score:.0f}\n"
                f"_~30-60s para varrer 300 tickers + scoring ML..._"
            )

            def _run_core_scan():
                try:
                    opportunities = scan_core_universe(min_score=min_score)
                    if not opportunities:
                        _reply(
                            f"ℹ️ Nenhum ticker do universo Core "
                            f"com score ≥ {min_score:.0f}. "
                            f"Mercado pode estar overheated — espera ou baixa floor."
                        )
                        return

                    # Already sorted by -score in scan_core_universe
                    top = opportunities[:top_n]

                    lines = [
                        f"🏛 *Core Picks — top {len(top)} de {len(opportunities)} opportunities*",
                        f"_Score ≥ {min_score:.0f}, source: {'ML' if any(o.get('score_source') == 'ml' for o in top) else 'heurístico'}_",
                        f"",
                        f"```",
                        f"{'Ticker':>7} {'Score':>6} {'Preço':>8} {'DD52w':>7} {'Yld%':>5} {'Sugerido':>9}",
                        f"─" * 55,
                    ]
                    for opp in top:
                        ticker = str(opp.get("ticker", "?"))[:6]
                        score = float(opp.get("score", 0))
                        price = float(opp.get("current_price", 0))
                        dd52w = float(opp.get("dd_from_52w_high", 0)) * 100
                        yld = float(opp.get("annual_yield_pct", 0))
                        sug_eur = float(opp.get("suggested_eur", 0))
                        # Mark stocks where FCF override fired (investment-phase compounder)
                        marker = " 🌱" if opp.get("fcf_override_applied") else ""
                        lines.append(
                            f"{ticker:>7} "
                            f"{score:>6.0f} "
                            f"${price:>7.2f} "
                            f"{dd52w:>+6.1f}% "
                            f"{yld:>4.1f}% "
                            f"€{sug_eur:>7.0f}{marker}"
                        )
                    lines.append("```")

                    # Notes on the top picks (first 3)
                    lines.append("")
                    lines.append("*Notas* (top 3):")
                    for opp in top[:3]:
                        ticker = opp.get("ticker", "?")
                        note = opp.get("note", "")
                        if note:
                            lines.append(f"• `{ticker}`: {note}")

                    # Investment-phase footer if any pick triggered override
                    invest_phase_picks = [
                        opp for opp in top
                        if opp.get("fcf_override_applied")
                    ]
                    if invest_phase_picks:
                        lines.append("")
                        lines.append(
                            "🌱 _= FCF negativo neutralizado (investment-phase "
                            "compounder: rev growth >10%, GM >40%, ROE >15%). "
                            "Capex spike, não deterioração._"
                        )

                    lines.append("")
                    lines.append(
                        f"_Sugerido EUR é a alocação base. Em /cash o sistema "
                        f"vai ajustar conforme conviction + capital disponível._"
                    )

                    # Surface cash reserve if available — opportunity-driven
                    # deploy reminder per user's stated policy
                    try:
                        from state import get_cash_reserve_eur
                        reserve = get_cash_reserve_eur()
                        if reserve >= 50:
                            lines.append("")
                            lines.append(
                                f"📥 _Reserva disponível: *€{reserve:.0f}* "
                                f"(definida pelo último /cash). "
                                f"Se algum pick acima tem conviction alta, "
                                f"podes deploy parcial sem esperar inflow novo._"
                            )
                    except Exception:
                        pass

                    _reply("\n".join(lines))
                except Exception as _e:
                    _reply(f"❌ Core scan falhou: `{_e}`")

            threading.Thread(
                target=_run_core_scan, daemon=True, name="core-picks"
            ).start()
        except Exception as e:
            _reply(f"❌ Erro: `{e}`")

    elif cmd in ("/admin_disk_usage", "/admin_disk"):
        # Quick breakdown of what's eating the 5GB Railway volume.
        # Useful before launching a multi-hour backfill that will grow
        # form4_cache / edgar_submissions_chunks significantly.
        try:
            from pathlib import Path as _P
            import shutil as _sh

            _dd = _P("/data") if _P("/data").exists() else _P("/tmp")
            total, used, free = _sh.disk_usage(str(_dd))

            def _dir_size(path: _P) -> tuple[int, int]:
                """Return (total_bytes, file_count) for a directory."""
                if not path.exists():
                    return 0, 0
                total_b = 0
                count = 0
                try:
                    for fp in path.rglob("*"):
                        if fp.is_file():
                            try:
                                total_b += fp.stat().st_size
                                count += 1
                            except OSError:
                                pass
                except OSError:
                    pass
                return total_b, count

            def _fmt(b: int) -> str:
                if b >= 1024**3:
                    return f"{b/1024**3:.2f} GB"
                if b >= 1024**2:
                    return f"{b/1024**2:.1f} MB"
                if b >= 1024:
                    return f"{b/1024:.0f} KB"
                return f"{b} B"

            # Inspect the top consumers — cache dirs + the big parquets
            candidates = [
                ("form4_cache",              _dd / "form4_cache"),
                ("edgar_submissions",        _dd / "edgar_submissions"),
                ("edgar_submissions_chunks", _dd / "edgar_submissions_chunks"),
                ("fundamentals_cache",       _dd / "fundamentals_cache"),
                ("fundamental_cache",        _dd / "fundamental_cache"),
                ("price_cache",              _dd / "price_cache"),
                ("yf_cache",                 _dd / "yf_cache"),
                ("ml_predictions",           _dd / "ml_predictions"),
                ("backups",                  _dd / "backups"),
                ("dip_model_stage",          _dd / "dip_model_stage"),
                ("alert_db",                 _dd / "alert_db"),
            ]
            entries = []
            for name, path in candidates:
                sz, ct = _dir_size(path)
                if sz > 0:
                    entries.append((name, sz, ct))

            # Sort biggest first
            entries.sort(key=lambda x: -x[1])
            lines = [f"💾 *Disk usage — `{_dd}`*"]
            lines.append(
                f"Volume: `{_fmt(used)}` used / `{_fmt(total)}` total "
                f"(`{_fmt(free)}` free, {used/total*100:.0f}%)"
            )
            lines.append("")
            lines.append("*Top consumers:*")
            for name, sz, ct in entries[:12]:
                lines.append(f"• `{name}`: {_fmt(sz)} ({ct:,} files)")

            # Big parquets — nomes têm de bater certo com os do build pipeline.
            # Antes de 2026-06-12 estavam aqui "core_dataset.parquet" e
            # "momentum_dataset.parquet" que NÃO existem; só o DIP aparecia.
            lines.append("")
            lines.append("*Key parquets:*")
            for name in ("ml_training_base.parquet",
                         "core_training_base.parquet",
                         "momentum_training.parquet"):
                fp = _dd / name
                if fp.exists():
                    lines.append(f"• `{name}`: {_fmt(fp.stat().st_size)}")
                else:
                    lines.append(f"• `{name}`: _ausente_")
            # Test-mode artefacts em /data/test/ — útil para confirmar isolamento.
            test_dir = _dd / "test"
            if test_dir.exists():
                lines.append("")
                lines.append("*Test parquets (sandbox):*")
                for name in ("ml_training_base.parquet",
                             "core_training_base.parquet",
                             "momentum_training.parquet"):
                    fp = test_dir / name
                    if fp.exists():
                        lines.append(f"• `test/{name}`: {_fmt(fp.stat().st_size)}")

            # Warning if free space getting tight
            if free < 500 * 1024**2:  # <500 MB
                lines.append("")
                lines.append(
                    "⚠️ *Menos de 500 MB livres* — backfill pode falhar a "
                    "escrever cache ou .tmp parquet. Considera limpar "
                    "`fundamental_cache` (yfinance, rebuilds-on-demand)."
                )

            _reply("\n".join(lines))
        except Exception as e:
            _reply(f"❌ Erro: `{e}`")

    elif cmd in ("/admin_threshold_analysis", "/admin_threshold", "/admin_thresh"):
        # Find the threshold that maximises Sharpe-of-selected-trades.
        # Preferentially uses honest out-of-fold predictions persisted
        # during /admin_retrain. Falls back to in-sample predictions
        # (with a loud warning) when OOF file doesn't exist yet.
        #
        # OOF semantics: for each row, the prediction comes from a fold
        # where THAT row was in the test set — the model never saw it.
        # This eliminates the in-sample bias that would otherwise inflate
        # Sharpe estimates by ~30-50%.
        #
        # Read-only — does NOT change any production constants. Output is
        # a recommendation; set _SCORE_HIGH / _SCORE_MED / _SCORE_FLOOR
        # in ml_predictor.py based on what the table shows.
        try:
            from pathlib import Path as _P
            import pandas as _pd
            import numpy as _np
            import joblib as _jl

            _dd = _P("/data") if _P("/data").exists() else _P("/tmp")
            parquet_p = _dd / "ml_training_base.parquet"
            oof_p     = _dd / "dip_oof_predictions.parquet"
            bundle_candidates = [
                _dd / "dip_models.pkl",
                _P(__file__).parent / "ml_training" / "dip_models.pkl",
            ]
            bundle_p = next((p for p in bundle_candidates if p.exists()), None)

            # Path A: prefer OOF predictions file (rigorous, no bias)
            df = None
            source_tag = ""
            bias_warning = ""
            if oof_p.exists():
                oof_df = _pd.read_parquet(oof_p)
                if "oof_pred" in oof_df.columns and "actual" in oof_df.columns:
                    df = oof_df.rename(columns={"oof_pred": "_pred", "actual": "_actual"})
                    n_total = len(df)
                    source_tag = "OOF (walk-forward, sem bias)"
                    _PT = "oof"  # display only
                else:
                    _reply("⚠️ `dip_oof_predictions.parquet` corrupto. Re-corre `/admin_retrain`.")
                    return
            else:
                # Path B: fallback to in-sample. Loud warning.
                if not parquet_p.exists():
                    _reply(f"⚠️ Nem OOF nem parquet existem. Corre `/admin_overnight` + `/admin_retrain`.")
                    return
                if bundle_p is None:
                    _reply(
                        "⚠️ OOF file ausente E bundle ausente. Corre `/admin_retrain` — "
                        "vai gerar ambos (modelo + dip_oof_predictions.parquet)."
                    )
                    return

                try:
                    from ml_training.config import PRIMARY_TARGET as _PT
                except Exception:
                    _PT = "alpha_90d"
                df_raw = _pd.read_parquet(parquet_p)
                if _PT not in df_raw.columns:
                    _reply(
                        f"⚠️ Coluna `{_PT}` ausente no parquet. "
                        f"Se mudaste DIP_PRIMARY_TARGET, corre `/admin_overnight`."
                    )
                    return
                df_raw = df_raw[df_raw[_PT].notna()].reset_index(drop=True)
                if len(df_raw) < 200:
                    _reply(f"⚠️ Apenas `{len(df_raw)}` rows com `{_PT}` resolvido — análise pouco fiável.")
                    return

                bundle = _jl.load(bundle_p)
                feature_cols = list(getattr(bundle, "feature_cols", []))
                if not feature_cols:
                    _reply("⚠️ Bundle sem `feature_cols`. Retrain necessário.")
                    return
                missing_in_parquet = [c for c in feature_cols if c not in df_raw.columns]
                if missing_in_parquet:
                    _reply(
                        f"⚠️ Bundle pede `{len(missing_in_parquet)}` features ausentes: "
                        f"`{missing_in_parquet[:5]}`. Regenera parquet."
                    )
                    return

                X = df_raw[feature_cols].values.astype("float32")
                try:
                    preds = bundle.model_up.predict(X)
                except Exception as e:
                    _reply(f"❌ Falha a predizer: `{e}`")
                    return
                df = _pd.DataFrame({
                    "_pred":   preds,
                    "_actual": df_raw[_PT].values.astype("float64"),
                })
                n_total = len(df)
                source_tag = "IN-SAMPLE (biased ↑ ~30-50%)"
                bias_warning = (
                    "\n\n⚠️ *AVISO*: Predictions in-sample (modelo viu estes dados no treino). "
                    "Sharpe inflacionado. Corre `/admin_retrain` para gerar OOF correto, "
                    "depois re-corre este comando."
                )

            # Common analysis (whatever path produced df)
            actual = df["_actual"].values
            preds  = df["_pred"].values

            # Pick threshold ladder spanning typical alpha range
            thresholds = [0.005, 0.010, 0.015, 0.020, 0.030, 0.040,
                          0.050, 0.060, 0.080, 0.100, 0.150]

            rows_table = []
            for thr in thresholds:
                sel = df[df["_pred"] >= thr]
                n_sel = len(sel)
                if n_sel == 0:
                    rows_table.append({"thr": thr, "n": 0})
                    continue
                a   = sel["_actual"].values
                mu  = float(_np.mean(a))
                sd  = float(_np.std(a))
                hit = float((a > 0).sum()) / n_sel
                sharpe = mu / sd if sd > 1e-6 else 0.0
                rows_table.append({
                    "thr":    thr,
                    "n":      n_sel,
                    "mean":   mu,
                    "hit":    hit,
                    "sharpe": sharpe,
                })

            # Baseline: ALL rows (no threshold filter)
            base_mean = float(actual.mean())
            base_hit  = float((actual > 0).sum()) / n_total
            base_sd   = float(actual.std())
            base_sharpe = base_mean / base_sd if base_sd > 1e-6 else 0.0

            # Recommend by Sharpe among thresholds with ≥30 selections
            viable = [r for r in rows_table if r.get("n", 0) >= 30 and "sharpe" in r]
            best = max(viable, key=lambda r: r["sharpe"]) if viable else None

            lines = [
                f"📊 *Threshold Analysis — DIP*",
                f"Source: `{source_tag}`",
                f"Target: `{_PT}`",
                f"Base: `{n_total:,}` rows, μ={base_mean*100:+.2f}%, "
                f"hit={base_hit*100:.0f}%, Sharpe={base_sharpe:+.2f}",
                "",
                "```",
                f"{'Thr':>6} {'N':>6} {'μ-α':>7} {'Hit%':>5} {'Sharpe':>7}",
                "─" * 38,
            ]
            for r in rows_table:
                if r.get("n", 0) == 0:
                    lines.append(f"{r['thr']:>6.3f} {0:>6}     --     --      --")
                    continue
                marker = " ←" if best and abs(r["thr"] - best["thr"]) < 1e-9 else ""
                lines.append(
                    f"{r['thr']:>6.3f} {r['n']:>6,} "
                    f"{r['mean']*100:>+6.2f}% {r['hit']*100:>4.0f}% "
                    f"{r['sharpe']:>+7.2f}{marker}"
                )
            lines.append("```")
            lines.append("")
            if best:
                lines.append(
                    f"*Recomendação*: threshold ≈ `{best['thr']:.3f}` "
                    f"(Sharpe {best['sharpe']:+.2f}, "
                    f"{best['n']:,} alertas, μ-α {best['mean']*100:+.2f}%)"
                )
                # Suggest mapping to existing constants
                if best["thr"] <= 0.02:
                    lines.append(f"_Sugestão: `_SCORE_FLOOR = {best['thr']:.3f}` em `ml_predictor.py`._")
                elif best["thr"] <= 0.04:
                    lines.append(f"_Sugestão: `_SCORE_MED = {best['thr']:.3f}` em `ml_predictor.py`._")
                else:
                    lines.append(f"_Sugestão: `_SCORE_HIGH = {best['thr']:.3f}` em `ml_predictor.py`._")
            else:
                lines.append("⚠️ Nenhum threshold com ≥30 alertas selecionados — modelo pouco discriminante.")
            if bias_warning:
                lines.append(bias_warning)
            _reply("\n".join(lines))
        except Exception as e:
            _reply(f"❌ Threshold analysis falhou: `{e}`")

    elif cmd in ("/admin_purge_sec_caches", "/admin_purge_sec"):
        # Reclaim disk by removing the three SEC Filings cache directories
        # that supported the now-removed Form 4 / 8-K / earnings call
        # pipeline. Safe to run anytime — XBRL fundamentals (gross_margin,
        # roe, etc.) use a different path (fundamentals_cache) which is
        # NOT touched here.
        try:
            from pathlib import Path as _P
            import shutil as _sh

            _dd     = _P("/data") if _P("/data").exists() else _P("/tmp")
            targets = [
                _dd / "form4_cache",
                _dd / "edgar_submissions",
                _dd / "edgar_submissions_chunks",
            ]
            lines = ["🧹 *SEC caches purge*"]
            total_freed = 0
            for d in targets:
                if not d.exists():
                    lines.append(f"• `{d.name}`: (não existe)")
                    continue
                # Sum size before delete for the report
                size = 0
                n    = 0
                try:
                    for fp in d.rglob("*"):
                        if fp.is_file():
                            try:
                                size += fp.stat().st_size
                                n    += 1
                            except OSError:
                                pass
                except OSError:
                    pass
                try:
                    _sh.rmtree(d)
                    total_freed += size
                    if size >= 1024**2:
                        lines.append(f"• `{d.name}`: {size/1024**2:.1f} MB, {n:,} files → ✅ deleted")
                    else:
                        lines.append(f"• `{d.name}`: {size/1024:.0f} KB, {n:,} files → ✅ deleted")
                except OSError as e:
                    lines.append(f"• `{d.name}`: ⚠️ falha ({e})")

            if total_freed >= 1024**2:
                lines.append(f"\n*Total libertado: {total_freed/1024**2:.1f} MB*")
            else:
                lines.append(f"\n*Total libertado: {total_freed/1024:.0f} KB*")
            lines.append("_XBRL fundamentals (gross_margin, roe, fcf) usam path separado, intacto._")
            _reply("\n".join(lines))
        except Exception as e:
            _reply(f"❌ Erro: `{e}`")

    elif cmd in ("/admin_diag_yf", "/admin_yf_test"):
        # Diagnose what yfinance actually returns for a ticker, so we can see
        # if peg_ratio/earnings_beat_rate/analyst_rating are 100% NaN due to:
        #   (a) yfinance returning empty (data limit, real)
        #   (b) yfinance returning data with renamed columns (parsing bug)
        if len(parts) < 2:
            _reply("Uso: `/admin_diag_yf TICKER` (ex: `/admin_diag_yf AAPL`)")
            return
        ticker = parts[2].upper().strip() if len(parts) > 2 else parts[1].upper().strip()
        try:
            import yfinance as _yf
            import pandas as _pd
            tk = _yf.Ticker(ticker)

            lines = [f"🔬 *yfinance diagnostic — {ticker}*\n"]

            # 1. earnings_dates
            lines.append("📅 *earnings_dates:*")
            try:
                edf = tk.earnings_dates
                if edf is None or edf.empty:
                    lines.append("  ❌ Empty/None")
                else:
                    lines.append(f"  ✅ {len(edf)} rows | cols: `{list(edf.columns)}`")
                    # Show first row to see actual data
                    first = edf.iloc[0]
                    lines.append(f"  First row: {dict(first.head(5).items())}")
            except Exception as e:
                lines.append(f"  ❌ Exception: `{e}`")

            # 2. recommendations
            lines.append("\n👨‍💼 *recommendations:*")
            try:
                rdf = tk.recommendations
                if rdf is None or rdf.empty:
                    lines.append("  ❌ Empty/None")
                else:
                    lines.append(f"  ✅ {len(rdf)} rows | cols: `{list(rdf.columns)}`")
                    first = rdf.iloc[0]
                    lines.append(f"  First row: {dict(first.head(6).items())}")
            except Exception as e:
                lines.append(f"  ❌ Exception: `{e}`")

            # 3. quarterly_financials
            lines.append("\n💰 *quarterly_financials:*")
            try:
                qf = tk.quarterly_financials
                if qf is None or qf.empty:
                    lines.append("  ❌ Empty/None")
                else:
                    lines.append(f"  ✅ {qf.shape[0]} rows × {qf.shape[1]} cols")
                    lines.append(f"  Index sample: {list(qf.index[:5])}")
                    lines.append(f"  Cols (dates): {[str(c)[:10] for c in qf.columns[:4]]}")
            except Exception as e:
                lines.append(f"  ❌ Exception: `{e}`")

            # 4. Run actual pipeline functions
            lines.append("\n🔧 *Pipeline functions:*")
            try:
                from pathlib import Path as _P
                from datetime import date as _date
                from data.fundamental_history import _yf_pit_earnings_signals, get_pit_fundamentals
                _dd = _P("/data") if _P("/data").exists() else _P("/tmp")
                cache = _dd / "fundamentals_cache"
                cache.mkdir(parents=True, exist_ok=True)

                # Force fresh (delete cache file if exists)
                cache_path = cache / f"yf_pit_earnings_{ticker.replace('.', '_')}.json"
                if cache_path.exists():
                    cache_path.unlink()

                earn = _yf_pit_earnings_signals(ticker, _date.today(), cache)
                lines.append(f"  `_yf_pit_earnings_signals`: `{earn}`")

                fund = get_pit_fundamentals(ticker, _date.today(), cache)
                # Show only key fields
                key_fund = {k: fund.get(k) for k in ("revenue_growth", "gross_margin", "earnings_growth", "_eps_ttm", "fcf_yield") if k in fund}
                lines.append(f"  `get_pit_fundamentals` (key fields): `{key_fund}`")
            except Exception as e:
                lines.append(f"  ❌ Exception: `{e}`")

            lines.append("\n_Esta info diz se yfinance funciona vs se o nosso parser está partido._")
            _reply("\n".join(lines))
        except Exception as e:
            _reply(f"❌ Diag falhou: `{e}`")

    elif cmd in ("/admin_missed_alpha", "/admin_missed_dips"):
        # Surface dip episodes that the bot DID NOT alert on but recovered
        # strongly (alpha_90d ≥ threshold). These are "money left on table"
        # — patterns the current alert filter is over-restrictive on.
        #
        # Usage:
        #   /admin_missed_alpha                  → last 180 days, alpha ≥ 0.15
        #   /admin_missed_alpha 365              → last year, default 15%
        #   /admin_missed_alpha 180 25           → 25%+ alpha threshold
        try:
            lookback_days = int(parts[1]) if len(parts) > 1 else 180
        except ValueError:
            lookback_days = 180
        try:
            min_alpha_pct = float(parts[2]) if len(parts) > 2 else 15.0
        except ValueError:
            min_alpha_pct = 15.0
        min_alpha = min_alpha_pct / 100.0

        try:
            from pathlib import Path as _P
            import pandas as _pd
            from datetime import date as _date, timedelta as _td

            _dd = _P("/data") if _P("/data").exists() else _P("/tmp")
            parquet = _dd / "ml_training_base.parquet"
            if not parquet.exists():
                _reply(f"⚠️ Parquet `{parquet.name}` não existe — corre `/admin_overnight` primeiro.")
                return

            df = _pd.read_parquet(parquet)
            n_total = len(df)
            if "alpha_90d" not in df.columns:
                _reply("⚠️ Parquet sem coluna `alpha_90d` — formato antigo. Re-builds via `/admin_overnight`.")
                return

            cutoff = _date.today() - _td(days=lookback_days)
            df["_alert_date"] = _pd.to_datetime(df["alert_date"]).dt.date
            recent = df[df["_alert_date"] >= cutoff].copy()

            # Episodes the bot DIDN'T alert on (source=simulated) AND had high alpha
            if "source" in recent.columns:
                missed = recent[(recent["source"] == "simulated") & (recent["alpha_90d"] >= min_alpha)]
                # "Caught" = real alerts (bot fired on it)
                caught = recent[(recent["source"] == "real")]
            else:
                # Older parquet without source column — assume all are simulated
                missed = recent[recent["alpha_90d"] >= min_alpha]
                caught = _pd.DataFrame()

            if missed.empty:
                _reply(
                    f"✅ *Missed Alpha Report* — {lookback_days}d\n\n"
                    f"Nenhum dip não-alertado com alpha ≥ {min_alpha_pct:.0f}% nos últimos {lookback_days} dias.\n"
                    f"_Parquet: {n_total:,} rows_"
                )
                return

            # Aggregate
            missed = missed.sort_values("alpha_90d", ascending=False)
            n_missed = len(missed)
            total_alpha = missed["alpha_90d"].sum() * 100
            avg_alpha = missed["alpha_90d"].mean() * 100
            n_caught = len(caught)
            catch_rate = n_caught / (n_caught + n_missed) if (n_caught + n_missed) > 0 else 0

            lines = [
                f"🔍 *Missed Alpha Report* ({lookback_days}d, alpha ≥ {min_alpha_pct:.0f}%)\n",
                f"📊 *Bot performance:*",
                f"  ✅ Caught (alertou): *{n_caught}* episódios",
                f"  ❌ Missed (não alertou): *{n_missed}* episódios",
                f"  🎯 Catch rate: *{catch_rate*100:.1f}%*\n",
                f"📈 *Money left on table:*",
                f"  Total alpha somado: *{total_alpha:+.0f}%*",
                f"  Avg alpha por miss: *{avg_alpha:+.1f}%*\n",
                f"🥇 *Top 15 misses (alpha desc):*",
            ]
            for _, row in missed.head(15).iterrows():
                tkr = row.get("ticker", "?")
                dt  = row["_alert_date"]
                alpha = row.get("alpha_90d", 0) * 100
                drop = (row.get("drop_pct_today") or 0)
                rsi  = (row.get("rsi_14") or 0)
                sec  = (row.get("sector") or "?")[:12]
                lines.append(
                    f"  📈 *{tkr}* {dt} | drop {drop:+.1f}% | alpha *{alpha:+.0f}%* "
                    f"| RSI {rsi:.0f} | _{sec}_"
                )

            # Sector breakdown
            if "sector" in missed.columns:
                sec_counts = missed["sector"].value_counts().head(5)
                lines += [
                    "",
                    "🏭 *Por sector (top 5):*",
                ]
                for s, c in sec_counts.items():
                    pct = c / n_missed * 100
                    lines.append(f"  • {s}: *{c}* ({pct:.0f}%)")

            # Month breakdown
            try:
                missed["_month"] = _pd.to_datetime(missed["_alert_date"]).dt.to_period("M").astype(str)
                month_counts = missed["_month"].value_counts().sort_index(ascending=False).head(6)
                lines += [
                    "",
                    "📅 *Por mês (últimos 6):*",
                ]
                for m, c in month_counts.items():
                    lines.append(f"  • {m}: *{c}* misses")
            except Exception:
                pass

            lines += [
                "",
                "_💡 Como reagir:_",
                "  • Sectores recorrentes → revê filtros sectoriais",
                "  • RSI baixo recorrente → considera baixar threshold de RSI",
                "  • Drops moderados recorrentes → relaxa o threshold de drop_pct_today",
                "  • Para forçar o modelo a aprender estes padrões, ver _Phase B_ na roadmap",
            ]
            _reply("\n".join(lines))
        except Exception as e:
            _reply(f"❌ Missed alpha report falhou: `{e}`")

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
                from portfolio.portfolio import _cache, _PORTFOLIO_FILE
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
            from portfolio.portfolio import get_positions, get_holdings_tuples, _cache
            from data.market_client import get_usdeur, get_portfolio_snapshot
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

    elif cmd in ("/digest", "/digest_now"):
        # On-demand digest peek — emits the current buffer without clearing,
        # so the user can check 'what's queued right now' anytime during the
        # session without affecting the scheduled 17:45 midday or 21:30
        # evening digests.
        _run_digest = _poll_context.get("run_digest_fn")
        if _run_digest is None:
            _reply("⚠️ run_digest_fn não configurada — verifica setup().")
        else:
            try:
                n = _run_digest()
                if n == 0:
                    _reply(
                        "📭 *Buffer vazio* — sem oportunidades acumuladas hoje.\n"
                        "_O próximo scan adiciona alertas. Digest auto às 17:45 e 21:30._"
                    )
                # If n > 0, run_digest_fn already sent the message
            except Exception as e:
                _reply(f"❌ Erro a emitir digest: `{e}`")

    elif cmd in ("/admin_calibrate_dip", "/calibrate_dip"):
        # Run scripts/calibrate_dip_alpha.py via subprocess (the user can't
        # SSH into Railway, so this is the only entry point). Streams the
        # output line-by-line into a single message so the user sees the
        # buckets table and the final r²/monotonicity/accepted flags.
        import subprocess as _subp
        import threading as _thr
        from pathlib import Path as _P

        _script = _P(__file__).parent / "scripts" / "calibrate_dip_alpha.py"
        if not _script.exists():
            _reply(f"⚠️ Script não encontrado: `{_script}`")
        else:
            _reply(
                "🧪 *DIP α calibration — V1 (parquet, in-sample)*\n"
                "_A correr `scripts/calibrate_dip_alpha.py`... pode demorar 1-3 min._"
            )

            def _run_calib():
                try:
                    proc = _subp.Popen(
                        ["python", str(_script)],
                        stdout=_subp.PIPE,
                        stderr=_subp.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    captured: list[str] = []
                    if proc.stdout is not None:
                        for line in proc.stdout:
                            captured.append(line.rstrip())
                            if len(captured) > 200:
                                captured = captured[-200:]
                    rc = proc.wait()
                    tail = "\n".join(captured[-40:])
                    status = "✅" if rc == 0 else "❌"
                    # Read the resulting JSON for a clean summary
                    summary = ""
                    try:
                        import json as _json
                        _cd = _P("/data") if _P("/data").exists() else _P("/tmp")
                        _alt = _P(__file__).parent / "data"
                        for _candidate in (_cd / "dip_alpha_calibration.json",
                                           _alt / "dip_alpha_calibration.json"):
                            if _candidate.exists():
                                _cf = _json.loads(_candidate.read_text())
                                summary = (
                                    f"\n\n📊 *Resultado*: r²=*{_cf.get('r_squared', 0):.3f}* "
                                    f"| monotonic=*{_cf.get('is_monotonic', False)}* "
                                    f"| accepted=*{_cf.get('accepted', False)}*\n"
                                    f"  n_total={_cf.get('n_samples_total', 0)} "
                                    f"(real={_cf.get('n_real', 0)}, "
                                    f"synthetic={_cf.get('n_synthetic', 0)})\n"
                                    f"  version=`{_cf.get('version', '?')}`\n"
                                    f"  Buckets: {len(_cf.get('buckets', []))}"
                                )
                                break
                    except Exception:
                        pass
                    _reply(
                        f"{status} *Calibração terminou (exit={rc})*\n\n"
                        f"```\n{tail[-3000:]}\n```{summary}"
                    )
                except Exception as e:
                    _reply(f"❌ Erro a correr calibração: `{e}`")

            _thr.Thread(target=_run_calib, daemon=True, name="dip-calib").start()

    elif cmd in ("/admin_clear_alerts", "/clear_alerts_now"):
        # Manual trigger for the daily dedup reset. Useful when stale
        # tickers from previous sessions are clogging the /cash DIP queue
        # (e.g. entries that pre-date the alert_db pred_alpha_90d schema
        # fix and now appear with 'no log entry — fallback' α=20.3%/y).
        if _cb_clear_alerts is None:
            _reply("⚠️ clear_alerts callback not wired — restart bot.")
        else:
            try:
                _cb_clear_alerts()
                _reply(
                    "🧹 *Alerted-today set limpo manualmente.*\n"
                    "_Próximos scans podem re-alertar tickers que tinham sido suprimidos hoje. "
                    "O auto-reset diário corre às 06:00 Lisboa._"
                )
            except Exception as e:
                _reply(f"❌ Erro a limpar: `{e}`")

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
                    from ml import ml_predictor as _mp
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
                from utils.chart_generator import generate_all_charts
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
                    from utils.github_publisher import push_all_charts
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
