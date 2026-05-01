"""
state.py — Persistência de estado do DipRadar.

Todos os ficheiros JSON são gravados em /data/ (Railway volume)
com fallback para /tmp/ em ambiente local.

Funções exportadas:
  load_alerts / save_alerts / clear_alerts
  load_weekly_log / save_weekly_log / append_weekly_log
  load_rejected_log / append_rejected_log
  load_backtest_log / save_backtest_log / append_backtest_entry
  load_recovery_watch / save_recovery_watch
  add_recovery_position / mark_recovery_alerted / remove_recovery_position
  get_stale_recovery_positions / mark_stale_alerted
  record_dip_day / mark_persistent_alerted / expire_missing_streaks
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

# ── Directório de persistência ─────────────────────────────────────────────

_DATA_DIR = Path("/data") if Path("/data").exists() else Path("/tmp")

_REJECTED_FILE = "rejected_log.json"


def _path(filename: str) -> Path:
    return _DATA_DIR / filename


def _load_json(filename: str, default):
    p = _path(filename)
    if not p.exists():
        return default
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"[state] Erro ao ler {filename}: {e}")
        return default


def _save_json(filename: str, data) -> None:
    p = _path(filename)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"[state] Erro ao gravar {filename}: {e}")


def _read(filename: str) -> dict:
    p = _path(filename)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"[state] Erro ao ler {filename}: {e}")
        return {}


def _write(filename: str, data: dict) -> None:
    _save_json(filename, data)


# ── Alertas diários ────────────────────────────────────────────────────────

def load_alerts() -> set:
    return set(_load_json("alerted_today.json", []))


def save_alerts(alerted: set) -> None:
    _save_json("alerted_today.json", list(alerted))


def clear_alerts() -> None:
    _save_json("alerted_today.json", [])


# ── Weekly log ────────────────────────────────────────────────────────────

def load_weekly_log() -> list:
    return _load_json("weekly_log.json", [])


def save_weekly_log(entries: list) -> None:
    _save_json("weekly_log.json", entries)


def append_weekly_log(entry: dict) -> None:
    entries = load_weekly_log()
    entries.append(entry)
    save_weekly_log(entries)


# ── Rejected log ──────────────────────────────────────────────────────────

def load_rejected_log() -> list:
    return _load_json("rejected_log.json", [])


def append_rejected_log(entry: dict) -> None:
    data    = _read(_REJECTED_FILE)
    entries = data.get("entries", [])
    today   = datetime.now().date().isoformat()
    entries = [e for e in entries if e.get("date_iso") == today]
    entries.append({
        **entry,
        "time":     datetime.now().strftime("%H:%M"),
        "date_iso": today,
    })
    _write(_REJECTED_FILE, {"entries": entries})


# ── Backtest log ──────────────────────────────────────────────────────────

def load_backtest_log() -> list:
    return _load_json("backtest_log.json", [])


def save_backtest_log(entries: list) -> None:
    _save_json("backtest_log.json", entries)


def append_backtest_entry(entry: dict) -> None:
    entries = load_backtest_log()
    entries.append({
        **entry,
        "price_5d":  None, "price_10d": None, "price_20d": None,
        "pnl_5d":    None, "pnl_10d":   None, "pnl_20d":   None,
        "resolved":  False,
    })
    save_backtest_log(entries)


# ── Recovery watch ────────────────────────────────────────────────────────

def load_recovery_watch() -> list:
    return _load_json("recovery_watch.json", [])


def save_recovery_watch(positions: list) -> None:
    _save_json("recovery_watch.json", positions)


def add_recovery_position(
    symbol: str,
    score: float,
    price_alert: float,
    target_pct: float = 15.0,
    verdict: str = "",
    category: str = "",
) -> None:
    positions = load_recovery_watch()
    # Não duplica — se já existe, actualiza
    positions = [p for p in positions if p["symbol"] != symbol]
    now = datetime.now()
    target_price = price_alert * (1 + target_pct / 100) if price_alert > 0 else 0
    positions.append({
        "symbol":       symbol,
        "score":        score,
        "price_alert":  round(price_alert, 4),
        "target_price": round(target_price, 4),
        "target_pct":   target_pct,
        "verdict":      verdict,
        "category":     category,
        "date":         now.strftime("%d/%m/%Y"),
        "date_iso":     now.strftime("%Y-%m-%d"),
        "alerted":      False,
        "stale_alerted": False,
    })
    save_recovery_watch(positions)


def mark_recovery_alerted(symbol: str) -> None:
    positions = load_recovery_watch()
    for p in positions:
        if p["symbol"] == symbol:
            p["alerted"] = True
    save_recovery_watch(positions)


def remove_recovery_position(symbol: str) -> None:
    positions = load_recovery_watch()
    positions = [p for p in positions if p["symbol"] != symbol]
    save_recovery_watch(positions)


def get_stale_recovery_positions(days: int = 60) -> list:
    positions = load_recovery_watch()
    cutoff    = datetime.now() - timedelta(days=days)
    stale     = []
    for p in positions:
        if p.get("alerted") or p.get("stale_alerted"):
            continue
        try:
            dt = datetime.fromisoformat(p["date_iso"])
            if dt < cutoff:
                stale.append(p)
        except Exception:
            pass
    return stale


def mark_stale_alerted(symbol: str) -> None:
    positions = load_recovery_watch()
    for p in positions:
        if p["symbol"] == symbol:
            p["stale_alerted"] = True
    save_recovery_watch(positions)


# ── Persistent Dip (Feature 8) ─────────────────────────────────────────────

_DIP_STREAKS_FILE = "dip_streaks.json"


def _load_streaks() -> dict:
    return _load_json(_DIP_STREAKS_FILE, {})


def _save_streaks(data: dict) -> None:
    _save_json(_DIP_STREAKS_FILE, data)


def record_dip_day(
    symbol: str,
    score: float,
    price: float,
    change_pct: float,
    verdict: str,
) -> dict:
    """
    Regista um dia de dip para `symbol` e devolve o estado actual da streak.
    Devolve dict com: symbol, days, scores, prices, first_date, last_date,
    alerted_persistent.
    """
    streaks = _load_streaks()
    today   = datetime.now().strftime("%Y-%m-%d")
    entry   = streaks.get(symbol, {
        "symbol":             symbol,
        "days":               0,
        "scores":             [],
        "prices":             [],
        "first_date":         today,
        "last_date":          today,
        "alerted_persistent": False,
    })

    # Só conta um dia por ticker por sessão
    if entry.get("last_date") != today:
        entry["days"]      = entry.get("days", 0) + 1
        entry["last_date"] = today
        entry["scores"].append(round(score, 1))
        entry["prices"].append(round(price, 4))

    streaks[symbol] = entry
    _save_streaks(streaks)
    return entry


def mark_persistent_alerted(symbol: str) -> None:
    streaks = _load_streaks()
    if symbol in streaks:
        streaks[symbol]["alerted_persistent"] = True
        _save_streaks(streaks)


def expire_missing_streaks(active_symbols: set) -> None:
    """
    Remove streaks de symbols que não apareceram no scan de hoje.
    Chamado no finally do run_scan() com o conjunto de symbols pontuados.
    """
    streaks = _load_streaks()
    to_remove = [
        sym for sym in list(streaks.keys())
        if sym not in active_symbols
    ]
    for sym in to_remove:
        del streaks[sym]
    if to_remove:
        _save_streaks(streaks)
        logging.info(f"[state] Streaks expiradas: {to_remove}")
