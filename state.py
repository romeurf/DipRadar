"""
state.py — Persistência de estado entre restarts do Railway.

Estratégia:
  - Tenta Railway Volume (/data/) se existir.
  - Fallback para /tmp/ (perde-se em restart, mas é melhor do que nada).
  - Todos os ficheiros de estado passam por cá — nunca por Path("/tmp/") directo.

Ficheiros geridos:
  alerts   → _dipr_alerts.json   (cache de alertas do dia)
  weekly   → _dipr_weekly.json   (log semanal de alertas)
  rejected → _dipr_rejected.json (log diário de rejeitados)
"""

import json
import logging
from datetime import datetime
from pathlib import Path

# Preferir volume persistente do Railway; fallback /tmp
_DATA_DIR = Path("/data") if Path("/data").exists() else Path("/tmp")
logging.info(f"[state] Directoria de estado: {_DATA_DIR}")

_ALERTS_FILE   = _DATA_DIR / "_dipr_alerts.json"
_WEEKLY_FILE   = _DATA_DIR / "_dipr_weekly.json"
_REJECTED_FILE = _DATA_DIR / "_dipr_rejected.json"


# ── helpers genéricos ─────────────────────────────────────────────────────

def _read(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as e:
        logging.warning(f"[state] read {path.name}: {e}")
    return {}

def _write(path: Path, data: dict) -> None:
    try:
        path.write_text(json.dumps(data))
    except Exception as e:
        logging.warning(f"[state] write {path.name}: {e}")


# ── Alerts cache (diário) ────────────────────────────────────────────────

def load_alerts() -> set:
    data  = _read(_ALERTS_FILE)
    today = datetime.now().date().isoformat()
    return {k for k in data.get("keys", []) if k.endswith(today)}

def save_alerts(alert_set: set) -> None:
    _write(_ALERTS_FILE, {"keys": list(alert_set)})

def clear_alerts() -> None:
    _write(_ALERTS_FILE, {"keys": []})


# ── Weekly log ────────────────────────────────────────────────────────────

def load_weekly_log() -> list:
    return _read(_WEEKLY_FILE).get("alerts", [])

def save_weekly_log(entries: list) -> None:
    _write(_WEEKLY_FILE, {"alerts": entries})

def append_weekly_log(symbol: str, verdict: str, score: float, change_pct: float, sector: str) -> None:
    entries = load_weekly_log()
    entries.append({
        "symbol":  symbol,
        "verdict": verdict,
        "score":   score,
        "change":  change_pct,
        "sector":  sector,
        "date":    datetime.now().strftime("%d/%m"),
        "time":    datetime.now().strftime("%H:%M"),
    })
    save_weekly_log(entries)


# ── Rejected log (diário) ─────────────────────────────────────────────────

def load_rejected_log() -> list:
    data  = _read(_REJECTED_FILE)
    today = datetime.now().date().isoformat()
    return [r for r in data.get("entries", []) if r.get("date_iso") == today]

def append_rejected_log(
    symbol: str,
    change_pct: float,
    reason: str,
    score: float | None = None,
    verdict: str | None = None,
    sector: str = "",
) -> None:
    """
    Regista um stock que foi analisado mas não gerou alerta, com o motivo.
    Motivos possíveis: 'EVITAR', 'score_baixo', 'cap_insuficiente', 'sem_dados'
    """
    data    = _read(_REJECTED_FILE)
    entries = data.get("entries", [])
    today   = datetime.now().date().isoformat()
    # Limpa entradas de dias anteriores
    entries = [e for e in entries if e.get("date_iso") == today]
    entries.append({
        "symbol":   symbol,
        "change":   change_pct,
        "reason":   reason,
        "score":    score,
        "verdict":  verdict,
        "sector":   sector,
        "time":     datetime.now().strftime("%H:%M"),
        "date_iso": today,
    })
    _write(_REJECTED_FILE, {"entries": entries})
