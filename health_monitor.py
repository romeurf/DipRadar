"""
health_monitor.py — Chunk 8 · Observabilidade e Health Checks

Responsabilidades:
  1. Registar o timestamp de cada scan bem-sucedido  (mark_scan_ok)
  2. Registar erros críticos e enviá-los via Telegram (record_error)
  3. Expor métricas de sistema: RAM, CPU, uptime, latência Tiingo/yfinance
  4. Construir o bloco /health para o bot_commands.py
  5. Detectar feature drift comparando distribuições live vs treino (PSI)

Integração (main.py):
  • Chamar health_monitor.mark_scan_ok("EU") / mark_scan_ok("US") no fim de
    cada eod_scan_* com sucesso.
  • Envolver run_scan / eod_scan_* com health_monitor.guarded() ou colocar um
    try/except global que chame health_monitor.record_error(context, exc).
  • Chamar health_monitor.check_feature_drift() após cada scan para detectar
    derivações de distribuição nas features do modelo.

Não tem dependências externas além de psutil (já no requirements.txt).
"""

from __future__ import annotations

import json
import logging
import os
import time
import threading
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np

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

# Buffer circular de feature vectors recentes (max 500 observações)
# Cada entrada: dict {feature_name: float_value}
_MAX_LIVE_ROWS = 500
_live_feature_buffer: list[dict] = []

# Cache do último resultado de drift (para /health não recomputar)
_last_drift_result: dict | None = None


# ── Registo de callback ──────────────────────────────────────────────────

def register_send_fn(fn: Callable[[str], None]) -> None:
    """Injecta o send_telegram do main.py para que o health_monitor possa
    enviar alertas autónomos sem criar dependência circular."""
    global _send_fn
    _send_fn = fn


# ── API pública ──────────────────────────────────────────────────────────────

def mark_scan_ok(scan_name: str) -> None:
    """Chama no fim de cada job agendado que terminou sem exceção."""
    with _lock:
        _last_scan_ok[scan_name] = datetime.now()
    logging.debug(f"[health] mark_scan_ok: {scan_name}")


def record_error(context: str, exc: Exception, *, send_alert: bool = True) -> None:
    """
    Regista um erro crítico no log interno e envia alerta Telegram se
    send_alert=True e _send_fn estiver configurado.

    Parâmetros:
        context   — nome do job / função onde ocorreu o erro
        exc       — exceção capturada
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


def record_live_features(feature_vector: dict) -> None:
    """
    Regista um vector de features live no buffer circular.
    Chamar após cada alerta processado em ml_predictor.py:

        import health_monitor
        health_monitor.record_live_features(feature_vector)

    Apenas as colunas numéricas (float/int) são guardadas.
    Labels e valores None são ignorados.
    """
    row = {
        k: float(v)
        for k, v in feature_vector.items()
        if isinstance(v, (int, float)) and v is not None
    }
    if not row:
        return
    with _lock:
        _live_feature_buffer.append(row)
        if len(_live_feature_buffer) > _MAX_LIVE_ROWS:
            _live_feature_buffer.pop(0)


# ── Feature Drift (PSI) ────────────────────────────────────────────────────────

_PSI_WARN  = 0.10   # amarelo — drift moderado
_PSI_ALERT = 0.25   # vermelho — drift severo, envia Telegram
_PSI_BINS  = 10     # número de bins para o histograma PSI


def _psi(ref_vals: np.ndarray, live_vals: np.ndarray, bins: int = _PSI_BINS) -> float:
    """
    Population Stability Index entre distribuição de referência (treino)
    e distribuição live.

    PSI = sum( (live_pct - ref_pct) * ln(live_pct / ref_pct) )

    Limites clássicos:
      < 0.10  -> estável
      0.10–0.25 -> drift moderado (monitorizar)
      > 0.25  -> drift severo (retraining necessário)
    """
    if len(ref_vals) < 10 or len(live_vals) < 5:
        return 0.0  # amostra insuficiente — não reportar

    # Usar os percentis da referência como bordas dos bins
    breakpoints = np.nanpercentile(ref_vals, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # remove duplicados em dist. constantes
    if len(breakpoints) < 3:
        return 0.0  # distribuição constante — sem drift mensurável

    ref_counts, _ = np.histogram(ref_vals, bins=breakpoints)
    live_counts, _ = np.histogram(live_vals, bins=breakpoints)

    # Converter para proporções (evita divisão por zero com epsilon)
    eps = 1e-6
    ref_pct  = (ref_counts  + eps) / (ref_counts.sum()  + eps * len(ref_counts))
    live_pct = (live_counts + eps) / (live_counts.sum() + eps * len(live_counts))

    psi_val = float(np.sum((live_pct - ref_pct) * np.log(live_pct / ref_pct)))
    return round(max(0.0, psi_val), 4)


def check_feature_drift(*, send_alert: bool = True) -> dict:
    """
    Compara a distribuição das features live (buffer) com a baseline de
    treino guardada em ml_report_v3.json (com fallback para ml_report.json
    legacy).

    Devolve um dict com PSI por feature e listas de features em cada zona.
    Envia alerta Telegram se alguma feature tiver PSI > _PSI_ALERT.

    A baseline é lida de report['feature_stats']. O notebook v3.1 actual
    não grava feature_stats — nesse caso a função devolve
    {"skipped": True} graciosamente.
    """
    global _last_drift_result

    data_dir = Path("/data") if Path("/data").exists() else Path("/tmp")
    repo_dir = Path(__file__).parent

    # Procura em /data, depois no repo, v3 antes de legacy
    candidates = [
        data_dir / "ml_report_v3.json",
        repo_dir / "ml_report_v3.json",
        data_dir / "ml_report.json",
        repo_dir / "ml_report.json",
    ]
    report_path = next((p for p in candidates if p.exists()), None)
    if report_path is None:
        logging.debug("[drift] ml_report_v3.json não encontrado — skip")
        return {"skipped": True, "reason": "ml_report ausente"}

    try:
        with open(report_path) as f:
            report = json.load(f)
    except Exception as e:
        logging.warning(f"[drift] Falha ao ler {report_path.name}: {e}")
        return {"skipped": True, "reason": str(e)}

    feature_stats: dict | None = report.get("feature_stats")
    if not feature_stats:
        logging.debug(f"[drift] {report_path.name} sem 'feature_stats' — skip")
        return {"skipped": True, "reason": "feature_stats ausente no relatório"}

    # ─ Snapshot do buffer live ──────────────────────────────────────────
    with _lock:
        buffer_snap = list(_live_feature_buffer)

    if len(buffer_snap) < 5:
        return {"skipped": True, "reason": f"buffer insuficiente ({len(buffer_snap)} obs.)"}

    # ─ Computar PSI por feature ───────────────────────────────────────
    psi_per_feature: dict[str, float] = {}
    stable, moderate, severe = [], [], []

    for feat, stats in feature_stats.items():
        # A baseline é representada pela média e std do treino.
        # Reconstruímos uma distribuição Gaussiana sintética como referência.
        mean = stats.get("mean")
        std  = stats.get("std")
        if mean is None or std is None or std < 1e-9:
            continue

        # Distribuição de referência: 1000 pontos Gaussianos (sintéticos)
        rng = np.random.default_rng(seed=42)
        ref_vals = rng.normal(loc=mean, scale=std, size=1000)

        # Distribuição live: extrair coluna do buffer
        live_vals = np.array([
            row[feat] for row in buffer_snap if feat in row
        ], dtype=float)

        if len(live_vals) < 5:
            continue

        psi_val = _psi(ref_vals, live_vals)
        psi_per_feature[feat] = psi_val

        if psi_val < _PSI_WARN:
            stable.append(feat)
        elif psi_val < _PSI_ALERT:
            moderate.append((feat, psi_val))
        else:
            severe.append((feat, psi_val))

    result = {
        "skipped":    False,
        "n_live":     len(buffer_snap),
        "psi":        psi_per_feature,
        "stable":     stable,
        "moderate":   moderate,
        "severe":     severe,
        "checked_at": datetime.now().isoformat(),
    }

    with _lock:
        _last_drift_result = result

    # ─ Alerta Telegram se houver drift severo ─────────────────────────
    if severe and send_alert and _send_fn:
        severe_lines = "\n".join(
            f"  🔴 `{f}`: PSI={v:.3f}" for f, v in severe
        )
        moderate_lines = ("\n".join(
            f"  🟡 `{f}`: PSI={v:.3f}" for f, v in moderate
        ) if moderate else "  _nenhuma_")
        try:
            _send_fn(
                f"🚨 *Feature Drift Detectado — DipRadar*\n"
                f"_{datetime.now().strftime('%d/%m/%Y %H:%M')}_ | {len(buffer_snap)} obs. live\n\n"
                f"*Drift Severo (PSI > {_PSI_ALERT}):*\n{severe_lines}\n\n"
                f"*Drift Moderado (PSI {_PSI_WARN}–{_PSI_ALERT}):*\n{moderate_lines}\n\n"
                f"⚠️ O mercado mudou de regime. Considera retraining antes do próximo ciclo mensal."
            )
        except Exception as e:
            logging.error(f"[drift] Falha ao enviar alerta de drift: {e}")

    logging.info(
        f"[drift] PSI check: {len(stable)} estáveis, "
        f"{len(moderate)} moderadas, {len(severe)} severas — "
        f"({len(buffer_snap)} obs. live)"
    )
    return result


# ── Métricas de sistema ─────────────────────────────────────────────────────────

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


# ── Construtor do bloco /health ──────────────────────────────────────────────

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

    # ── Uptime & recursos ───────────────────────────────────────────────
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

    # ── Último scan bem-sucedido ────────────────────────────────────────────
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

    # ── APIs externas ───────────────────────────────────────────────────
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

    # ── ML model (bundle v3 single-file) ────────────────────────────────────
    data_dir = Path("/data") if Path("/data").exists() else Path("/tmp")
    repo_dir = Path(__file__).parent
    pkl_v3 = next(
        (p for p in (data_dir / "dip_models_v3.pkl", repo_dir / "dip_models_v3.pkl")
         if p.exists()),
        None,
    )
    if pkl_v3 is not None:
        mtime = datetime.fromtimestamp(pkl_v3.stat().st_mtime).strftime("%d/%m %H:%M")
        ml_str = f"🟢 *pronto* (modificado {mtime})"
    else:
        ml_str = "🔴 _não treinado_"

    lines.append("*🤖 Modelo ML (v3):*")
    lines.append(f"  Bundle: {ml_str}")
    lines.append("")

    # ── Feature Drift ──────────────────────────────────────────────────────
    lines.append("*📊 Feature Drift (PSI):*")
    with _lock:
        drift = _last_drift_result

    if drift is None:
        lines.append("  ⚪ _Ainda não computado (aguarda primeiro scan)_")
    elif drift.get("skipped"):
        reason = drift.get("reason", "desconhecido")
        lines.append(f"  ⚪ _Skipped: {reason}_")
    else:
        n_live    = drift.get("n_live", 0)
        stable    = drift.get("stable", [])
        moderate  = drift.get("moderate", [])
        severe    = drift.get("severe", [])
        checked   = drift.get("checked_at", "")
        checked_str = checked[:16].replace("T", " ") if checked else ""

        lines.append(f"  _Última verificação: {checked_str} | {n_live} obs. live_")

        if severe:
            for feat, val in severe:
                lines.append(f"  🔴 `{feat}`: PSI={val:.3f} (severo)")
        if moderate:
            for feat, val in moderate:
                lines.append(f"  🟡 `{feat}`: PSI={val:.3f} (moderado)")
        if not severe and not moderate:
            lines.append(f"  🟢 Todas as {len(stable)} features estáveis")

    lines.append("")

    # ── Erros recentes ──────────────────────────────────────────────────────────
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

    # ── Resumo final ─────────────────────────────────────────────────────────────
    drift_severe = drift and not drift.get("skipped") and drift.get("severe")
    if any_stale or drift_severe:
        if any_stale:
            lines.append("⚠️ _Um ou mais jobs estão em silêncio há demasiado tempo._")
            lines.append("_Verifica os logs no Railway: `railway logs --tail 200`_")
        if drift_severe:
            lines.append("⚠️ _Drift severo detectado — considera retraining antecipado._")
    else:
        lines.append("_Todos os sistemas operacionais. 🟢_")

    return "\n".join(lines)
