"""
health_monitor.py — Chunk 8 · Observabilidade e Health Checks

Responsabilidades:
  1. Registar o timestamp de cada scan bem-sucedido  (mark_scan_ok)
  2. Registar erros críticos e enviá-los via Telegram (record_error)
  3. Expor métricas de sistema: RAM, CPU, uptime, latência Tiingo/yfinance
  4. Construir o bloco /health para o bot_commands.py

Integração (main.py):
  • Chamar health_monitor.mark_scan_ok("EU") / mark_scan_ok("US") no fim de
    cada eod_scan_* com sucesso.
  • Envolver run_scan / eod_scan_* com health_monitor.guarded() ou colocar um
    try/except global que chame health_monitor.record_error(context, exc).

Não tem dependências externas além de psutil (já no requirements.txt).
"""

from __future__ import annotations

import logging
import os
import time
import threading
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

# psutil é opcional — degrada graciosamente se não estiver instalado
try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

# ── Estado interno (thread-safe) ──────────────────────────────────────────────

_lock = threading.Lock()

# { "EU": datetime | None, "US": datetime | None, "WATCHLIST": datetime | None, ... }
_last_scan_ok: dict[str, datetime | None] = {
    "EU":        None,
    "US":        None,
    "WATCHLIST": None,
    "HEARTBEAT": None,
    "ML_OUTCOMES": None,
}

# Fila circular de erros recentes (max 20)
_MAX_ERRORS = 20
_error_log: list[dict] = []

# Tempo de arranque do processo
_start_time: datetime = datetime.now()

# Callback de envio Telegram (injectado por main.py)
_send_fn: Callable[[str], None] | None = None

# Limiar de silêncio: se um scan demorar mais do que este valor sem reportar
# sucesso, o /health mostra um aviso.
SCAN_STALE_HOURS: dict[str, float] = {
    "EU":          28.0,   # escandaloso se falhar mais de 1 dia de mercado
    "US":          28.0,
    "WATCHLIST":   28.0,
    "HEARTBEAT":   26.0,   # heartbeat diário das 9h
    "ML_OUTCOMES": 170.0,  # semanal ao domingo — 7 dias + margem
}

# ── Registo de callback ────────────────────────────────────────────────────────

def register_send_fn(fn: Callable[[str], None]) -> None:
    """Injecta o send_telegram do main.py para que o health_monitor possa
    enviar alertas autónomos sem criar dependência circular."""
    global _send_fn
    _send_fn = fn


# ── API pública ────────────────────────────────────────────────────────────────

def mark_scan_ok(scan_name: str) -> None:
    """Chama no fim de cada job agendado que terminou sem excepção."""
    with _lock:
        _last_scan_ok[scan_name] = datetime.now()
    logging.debug(f"[health] mark_scan_ok: {scan_name}")


def record_error(context: str, exc: Exception, *, send_alert: bool = True) -> None:
    """
    Regista um erro crítico no log interno e envia alerta Telegram se
    send_alert=True e _send_fn estiver configurado.

    Parâmetros:
        context   — nome do job / função onde ocorreu o erro
        exc       — excepção capturada
        send_alert — se False, só regista; não envia mensagem
    """
    tb_str = traceback.format_exc()
    entry = {
        "ts":      datetime.now(),
        "context": context,
        "error":   str(exc),
        "tb":      tb_str,
    }
    with _lock:
        _error_log.append(entry)
        if len(_error_log) > _MAX_ERRORS:
            _error_log.pop(0)

    logging.error(f"[health] ERROR em '{context}': {exc}\n{tb_str}")

    if send_alert and _send_fn:
        # Trunca o traceback para não rebentar o limite do Telegram
        tb_preview = tb_str[-600:] if len(tb_str) > 600 else tb_str
        try:
            _send_fn(
                f"⚠️ *SYSTEM ERROR — DipRadar*\n"
                f"_Job:_ `{context}`\n"
                f"_Hora:_ {entry['ts'].strftime('%d/%m %H:%M:%S')}\n\n"
                f"*Erro:* `{str(exc)[:200]}`\n\n"
                f"```\n{tb_preview}\n```"
            )
        except Exception as send_exc:
            logging.error(f"[health] Falha ao enviar alerta de erro: {send_exc}")


def guarded(job_name: str) -> Callable:
    """
    Decorador / wrapper que envolve uma função com captura de erros e
    marcação automática de sucesso.

    Uso em main.py:
        @health_monitor.guarded("EU")
        def eod_scan_europe(): ...

    Ou inline:
        health_monitor.guarded("US")(eod_scan_us)()
    """
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                result = fn(*args, **kwargs)
                mark_scan_ok(job_name)
                return result
            except Exception as exc:
                record_error(job_name, exc)
                raise
        wrapper.__name__ = fn.__name__
        wrapper.__doc__  = fn.__doc__
        return wrapper
    return decorator


# ── Métricas de sistema ────────────────────────────────────────────────────────

def _ram_usage() -> tuple[float, float]:
    """Devolve (rss_mb, percent) do processo actual. (-1, -1) se psutil indisponível."""
    if not _PSUTIL:
        return -1.0, -1.0
    proc = psutil.Process(os.getpid())
    mem  = proc.memory_info()
    rss  = mem.rss / 1024 / 1024
    pct  = proc.memory_percent()
    return round(rss, 1), round(pct, 1)


def _cpu_percent() -> float:
    """CPU do processo (intervalo 0.5s). -1 se psutil indisponível."""
    if not _PSUTIL:
        return -1.0
    return psutil.Process(os.getpid()).cpu_percent(interval=0.5)


def _disk_data_dir() -> tuple[float, float]:
    """
    Espaço usado / disponível (GB) no volume /data (Railway).
    Cai para /tmp se /data não existir.
    """
    if not _PSUTIL:
        return -1.0, -1.0
    data_dir = Path("/data") if Path("/data").exists() else Path("/tmp")
    try:
        usage = psutil.disk_usage(str(data_dir))
        return round(usage.used / 1e9, 2), round(usage.free / 1e9, 2)
    except Exception:
        return -1.0, -1.0


def _ping_tiingo() -> float | None:
    """Latência HTTP ao endpoint Tiingo (ms). None se falhar."""
    import requests
    token = os.environ.get("TIINGO_API_KEY", "")
    if not token:
        return None
    try:
        t0 = time.monotonic()
        r  = requests.get(
            "https://api.tiingo.com/api/test",
            headers={"Authorization": f"Token {token}"},
            timeout=6,
        )
        if r.ok:
            return round((time.monotonic() - t0) * 1000, 1)
    except Exception:
        pass
    return None


def _ping_yfinance() -> float | None:
    """Latência de uma chamada rápida ao yfinance (ms). None se falhar."""
    try:
        import yfinance as yf
        t0   = time.monotonic()
        info = yf.Ticker("SPY").fast_info
        _    = getattr(info, "last_price", None)
        return round((time.monotonic() - t0) * 1000, 1)
    except Exception:
        return None


# ── Construtor do bloco /health ────────────────────────────────────────────────

def build_health_report(*, ping_apis: bool = True) -> str:
    """
    Constrói a mensagem completa do comando /health.

    ping_apis=False salta os pings de latência (útil em testes unitários).
    """
    now    = datetime.now()
    uptime = now - _start_time
    h, rem = divmod(int(uptime.total_seconds()), 3600)
    m      = rem // 60

    lines: list[str] = [
        f"🩺 *DipRadar — Health Check*",
        f"_{now.strftime('%d/%m/%Y %H:%M:%S')}_",
        "",
    ]

    # ── Uptime & recursos ────────────────────────────────────────────────────
    rss, pct = _ram_usage()
    cpu      = _cpu_percent()
    d_used, d_free = _disk_data_dir()

    lines.append("*🖥️ Sistema:*")
    lines.append(f"  ⏱️ Uptime: *{h}h {m:02d}m*")

    if rss >= 0:
        ram_emoji = "🟢" if rss < 300 else ("🟡" if rss < 500 else "🔴")
        lines.append(f"  {ram_emoji} RAM: *{rss} MB* ({pct}%)")
    else:
        lines.append("  ⚪ RAM: _psutil indisponível_")

    if cpu >= 0:
        cpu_emoji = "🟢" if cpu < 40 else ("🟡" if cpu < 75 else "🔴")
        lines.append(f"  {cpu_emoji} CPU: *{cpu}%*")

    if d_used >= 0:
        disk_emoji = "🟢" if d_free > 0.5 else "🔴"
        lines.append(f"  {disk_emoji} Disco /data: *{d_used} GB usados* | *{d_free} GB livres*")

    lines.append("")

    # ── Último scan bem-sucedido ──────────────────────────────────────────────
    lines.append("*📡 Último scan OK:*")
    with _lock:
        snap = dict(_last_scan_ok)

    scan_labels = {
        "EU":          "EOD Europa  (17h45)",
        "US":          "EOD EUA     (21h15)",
        "WATCHLIST":   "Watchlist",
        "HEARTBEAT":   "Heartbeat  (09h00)",
        "ML_OUTCOMES": "ML Outcomes (dom)",
    }
    any_stale = False
    for key, label in scan_labels.items():
        ts      = snap.get(key)
        stale_h = SCAN_STALE_HOURS.get(key, 26.0)
        if ts is None:
            age_str = "_nunca registado_"
            emoji   = "⚪"
        else:
            age     = now - ts
            h_age   = age.total_seconds() / 3600
            age_str = ts.strftime("%d/%m %H:%M")
            if h_age < stale_h:
                emoji = "🟢"
            else:
                emoji = "🔴"
                any_stale = True
        lines.append(f"  {emoji} *{label}*: {age_str}")

    lines.append("")

    # ── APIs externas ─────────────────────────────────────────────────────────
    if ping_apis:
        lines.append("*🌐 Latência APIs:*")

        tiingo_ms = _ping_tiingo()
        if tiingo_ms is None:
            tiingo_str   = "_sem chave / timeout_"
            tiingo_emoji = "⚪"
        elif tiingo_ms < 400:
            tiingo_str   = f"*{tiingo_ms} ms*"
            tiingo_emoji = "🟢"
        elif tiingo_ms < 1200:
            tiingo_str   = f"*{tiingo_ms} ms*"
            tiingo_emoji = "🟡"
        else:
            tiingo_str   = f"*{tiingo_ms} ms* ⚠️"
            tiingo_emoji = "🔴"
        lines.append(f"  {tiingo_emoji} Tiingo: {tiingo_str}")

        yf_ms = _ping_yfinance()
        if yf_ms is None:
            yf_str   = "_timeout_"
            yf_emoji = "🔴"
        elif yf_ms < 800:
            yf_str   = f"*{yf_ms} ms*"
            yf_emoji = "🟢"
        elif yf_ms < 2000:
            yf_str   = f"*{yf_ms} ms*"
            yf_emoji = "🟡"
        else:
            yf_str   = f"*{yf_ms} ms* ⚠️"
            yf_emoji = "🔴"
        lines.append(f"  {yf_emoji} yfinance (SPY): {yf_str}")

        lines.append("")

    # ── ML model ─────────────────────────────────────────────────────────────
    data_dir   = Path("/data") if Path("/data").exists() else Path("/tmp")
    pkl_s1     = data_dir / "dip_model_stage1.pkl"
    pkl_s2     = data_dir / "dip_model_stage2.pkl"
    ml_s1_str  = f"🟢 *pronto* (modificado {datetime.fromtimestamp(pkl_s1.stat().st_mtime).strftime('%d/%m %H:%M')})" \
                 if pkl_s1.exists() else "🔴 _não treinado_"
    ml_s2_str  = f"🟢 *pronto*" if pkl_s2.exists() else "⚪ _não treinado_"

    lines.append("*🤖 Modelos ML:*")
    lines.append(f"  Andar 1: {ml_s1_str}")
    lines.append(f"  Andar 2: {ml_s2_str}")
    lines.append("")

    # ── Erros recentes ────────────────────────────────────────────────────────
    with _lock:
        recent_errors = list(_error_log[-5:])

    if recent_errors:
        lines.append(f"*🚨 Últimos {len(recent_errors)} erro(s):*")
        for e in reversed(recent_errors):
            ts_str = e["ts"].strftime("%d/%m %H:%M")
            lines.append(f"  🔴 `{ts_str}` [{e['context']}] _{e['error'][:80]}_")
        lines.append("")
    else:
        lines.append("*✅ Sem erros registados*")
        lines.append("")

    # ── Resumo final ─────────────────────────────────────────────────────────
    if any_stale:
        lines.append("⚠️ _Um ou mais jobs estão em silêncio há demasiado tempo._")
        lines.append("_Verifica os logs no Railway: `railway logs --tail 200`_")
    else:
        lines.append("_Todos os sistemas operacionais. 🟢_")

    return "\n".join(lines)
