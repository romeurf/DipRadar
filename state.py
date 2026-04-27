"""
state.py — Persistência de estado entre restarts do Railway.

Estrutura de ficheiros:
  _dipr_alerts.json    → cache de alertas do dia
  _dipr_weekly.json    → log semanal de alertas
  _dipr_rejected.json  → log diário de rejeitados
  _dipr_backtest.json  → histórico de alertas para backtesting
  _dipr_recovery.json  → posições em aberto aguardando recovery alert
"""

import json
import logging
from datetime import datetime
from pathlib import Path

_DATA_DIR = Path("/data") if Path("/data").exists() else Path("/tmp")
logging.info(f"[state] Directoria de estado: {_DATA_DIR}")

_ALERTS_FILE    = _DATA_DIR / "_dipr_alerts.json"
_WEEKLY_FILE    = _DATA_DIR / "_dipr_weekly.json"
_REJECTED_FILE  = _DATA_DIR / "_dipr_rejected.json"
_BACKTEST_FILE  = _DATA_DIR / "_dipr_backtest.json"
_RECOVERY_FILE  = _DATA_DIR / "_dipr_recovery.json"


# ── helpers genéricos ────────────────────────────────────────────────────────

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


# ── Alerts cache (diário) ────────────────────────────────────────────────────

def load_alerts() -> set:
    data  = _read(_ALERTS_FILE)
    today = datetime.now().date().isoformat()
    return {k for k in data.get("keys", []) if k.endswith(today)}

def save_alerts(alert_set: set) -> None:
    _write(_ALERTS_FILE, {"keys": list(alert_set)})

def clear_alerts() -> None:
    _write(_ALERTS_FILE, {"keys": []})


# ── Weekly log ───────────────────────────────────────────────────────────────

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


# ── Rejected log (diário) ────────────────────────────────────────────────────

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
    data    = _read(_REJECTED_FILE)
    entries = data.get("entries", [])
    today   = datetime.now().date().isoformat()
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


# ── Backtest log (persistente, acumula todos os alertas) ─────────────────────

def load_backtest_log() -> list:
    """Devolve lista de todos os alertas históricos com campos de resultado."""
    return _read(_BACKTEST_FILE).get("entries", [])

def save_backtest_log(entries: list) -> None:
    _write(_BACKTEST_FILE, {"entries": entries})

def append_backtest_entry(
    symbol: str,
    verdict: str,
    score: float,
    change_pct: float,
    price_alert: float,
    sector: str = "",
) -> None:
    """
    Regista um alerta para futura avaliação de resultado.
    Os campos price_5d, price_10d, price_20d e pnl_* são preenchidos
    pelo backtest_runner quando os dados estiverem disponíveis.
    """
    entries = load_backtest_log()
    entries.append({
        "symbol":      symbol,
        "verdict":     verdict,
        "score":       score,
        "change":      change_pct,
        "price_alert": price_alert,
        "sector":      sector,
        "date":        datetime.now().strftime("%d/%m/%Y"),
        "date_iso":    datetime.now().date().isoformat(),
        "price_5d":    None,
        "price_10d":   None,
        "price_20d":   None,
        "pnl_5d":      None,
        "pnl_10d":     None,
        "pnl_20d":     None,
        "resolved":    False,
    })
    save_backtest_log(entries)


# ── Recovery watch (posições em aberto) ──────────────────────────────────────

def load_recovery_watch() -> list:
    return _read(_RECOVERY_FILE).get("positions", [])

def save_recovery_watch(positions: list) -> None:
    _write(_RECOVERY_FILE, {"positions": positions})

def add_recovery_position(
    symbol: str,
    price_alert: float,
    score: float,
    target_pct: float,
    verdict: str,
) -> None:
    """
    Adiciona posição ao watch de recovery.
    target_pct: % de recuperação a partir do preço de alerta (ex: 15.0)
    Dispara alerta quando preço >= price_alert * (1 + target_pct/100).
    """
    positions = load_recovery_watch()
    # Evita duplicados
    if any(p["symbol"] == symbol for p in positions):
        return
    positions.append({
        "symbol":      symbol,
        "price_alert": price_alert,
        "score":       score,
        "target_pct":  target_pct,
        "target_price": round(price_alert * (1 + target_pct / 100), 2),
        "verdict":     verdict,
        "date":        datetime.now().strftime("%d/%m/%Y"),
        "alerted":     False,
    })
    save_recovery_watch(positions)

def mark_recovery_alerted(symbol: str) -> None:
    positions = load_recovery_watch()
    for p in positions:
        if p["symbol"] == symbol:
            p["alerted"] = True
    save_recovery_watch(positions)

def remove_recovery_position(symbol: str) -> None:
    positions = [p for p in load_recovery_watch() if p["symbol"] != symbol]
    save_recovery_watch(positions)
