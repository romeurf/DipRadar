"""
state.py — Persistência de estado entre restarts do Railway.

Estrutura de ficheiros:
  _dipr_alerts.json      → cache de alertas do dia
  _dipr_weekly.json      → log semanal de alertas
  _dipr_rejected.json    → log diário de rejeitados
  _dipr_backtest.json    → histórico de alertas para backtesting
  _dipr_recovery.json    → posições em aberto aguardando recovery alert
  _dipr_watchlist.json   → watchlist dinâmica (add/remove via Telegram)
  _dipr_score_log.json   → histórico de scores por ticker (para upgrades + /historico)
  _dipr_flip_log.json    → log de trades do Flip Fund (/flip)
  _dipr_dip_streaks.json → Feature 8: rasteia dips consecutivos dia-a-dia
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

_DATA_DIR = Path("/data") if Path("/data").exists() else Path("/tmp")
logging.info(f"[state] Directoria de estado: {_DATA_DIR}")

_ALERTS_FILE    = _DATA_DIR / "_dipr_alerts.json"
_WEEKLY_FILE    = _DATA_DIR / "_dipr_weekly.json"
_REJECTED_FILE  = _DATA_DIR / "_dipr_rejected.json"
_BACKTEST_FILE  = _DATA_DIR / "_dipr_backtest.json"
_RECOVERY_FILE  = _DATA_DIR / "_dipr_recovery.json"
_WATCHLIST_FILE = _DATA_DIR / "_dipr_watchlist.json"
_SCORE_LOG_FILE = _DATA_DIR / "_dipr_score_log.json"
_FLIP_LOG_FILE  = _DATA_DIR / "_dipr_flip_log.json"
_DIP_STREAKS_FILE = _DATA_DIR / "_dipr_dip_streaks.json"


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
    category: str = "",
) -> None:
    positions = load_recovery_watch()
    if any(p["symbol"] == symbol for p in positions):
        return
    positions.append({
        "symbol":        symbol,
        "price_alert":   price_alert,
        "score":         score,
        "target_pct":    target_pct,
        "target_price":  round(price_alert * (1 + target_pct / 100), 2),
        "verdict":       verdict,
        "category":      category,
        "date":          datetime.now().strftime("%d/%m/%Y"),
        "date_iso":      datetime.now().date().isoformat(),
        "alerted":       False,
        "stale_alerted": False,
    })
    save_recovery_watch(positions)

def mark_recovery_alerted(symbol: str) -> None:
    positions = load_recovery_watch()
    for p in positions:
        if p["symbol"] == symbol:
            p["alerted"] = True
    save_recovery_watch(positions)

def mark_stale_alerted(symbol: str) -> None:
    """Marca a posição como já tendo recebido o aviso de stop temporal."""
    positions = load_recovery_watch()
    for p in positions:
        if p["symbol"] == symbol:
            p["stale_alerted"] = True
    save_recovery_watch(positions)

def remove_recovery_position(symbol: str) -> None:
    positions = [p for p in load_recovery_watch() if p["symbol"] != symbol]
    save_recovery_watch(positions)

def get_stale_recovery_positions(days: int = 60) -> list[dict]:
    """
    Devolve posições que estão no recovery_watch há mais de `days` dias
    sem terem atingido o target, e cujo aviso de stale ainda não foi enviado.
    """
    positions = load_recovery_watch()
    stale     = []
    cutoff    = datetime.now() - timedelta(days=days)
    for p in positions:
        if p.get("alerted"):
            continue
        if p.get("stale_alerted"):
            continue
        date_iso = p.get("date_iso")
        if not date_iso:
            continue
        try:
            entry_date = datetime.fromisoformat(date_iso)
        except ValueError:
            continue
        if entry_date <= cutoff:
            stale.append(p)
    return stale


# ── Watchlist dinâmica (gerida via /watchlist add|remove) ────────────────────

def load_dynamic_watchlist() -> list[str]:
    return _read(_WATCHLIST_FILE).get("tickers", [])

def save_dynamic_watchlist(tickers: list[str]) -> None:
    _write(_WATCHLIST_FILE, {"tickers": list(dict.fromkeys(t.upper() for t in tickers))})

def add_to_dynamic_watchlist(ticker: str) -> bool:
    """Adiciona ticker. Devolve True se foi adicionado, False se já existia."""
    tickers = load_dynamic_watchlist()
    t = ticker.upper().strip()
    if t in tickers:
        return False
    tickers.append(t)
    save_dynamic_watchlist(tickers)
    return True

def remove_from_dynamic_watchlist(ticker: str) -> bool:
    """Remove ticker. Devolve True se existia e foi removido, False caso contrário."""
    tickers = load_dynamic_watchlist()
    t = ticker.upper().strip()
    if t not in tickers:
        return False
    tickers = [x for x in tickers if x != t]
    save_dynamic_watchlist(tickers)
    return True


# ── Score log (histórico de scores para /historico e upgrades) ───────────────

def load_score_log() -> dict:
    return _read(_SCORE_LOG_FILE).get("scores", {})

def save_score_log(data: dict) -> None:
    _write(_SCORE_LOG_FILE, {"scores": data})

def append_score_log(
    symbol: str,
    score: float,
    verdict: str,
    change_pct: float = 0.0,
    price: float = 0.0,
) -> None:
    data = load_score_log()
    entries = data.get(symbol, [])
    entries.append({
        "score":    round(score, 1),
        "verdict":  verdict,
        "change":   round(change_pct, 2),
        "price":    round(price, 4) if price else None,
        "date":     datetime.now().strftime("%d/%m/%Y"),
        "time":     datetime.now().strftime("%H:%M"),
        "date_iso": datetime.now().date().isoformat(),
    })
    data[symbol] = entries[-30:]
    save_score_log(data)

def get_ticker_score_history(symbol: str) -> list:
    return load_score_log().get(symbol.upper(), [])

def get_last_score(symbol: str) -> float | None:
    history = get_ticker_score_history(symbol)
    if history:
        return history[-1]["score"]
    return None


# ── Flip Fund trade log ───────────────────────────────────────────────────────

def load_flip_log() -> list:
    """
    Devolve lista de todos os trades do Flip Fund.
    Cada entrada tem:
      id, symbol, shares, price_entry, price_exit (None se aberto),
      date_entry, date_exit (None se aberto), pnl_eur, status ('open'|'closed'),
      notes
    """
    return _read(_FLIP_LOG_FILE).get("trades", [])

def save_flip_log(trades: list) -> None:
    _write(_FLIP_LOG_FILE, {"trades": trades})

def add_flip_trade(
    symbol: str,
    shares: float,
    price_entry: float,
    date_entry: str | None = None,
    notes: str = "",
) -> dict:
    """
    Regista entrada num trade Flip Fund (posição aberta).
    Devolve o dict do trade criado.
    """
    trades = load_flip_log()
    trade_id = (max((t["id"] for t in trades), default=0) + 1)
    trade = {
        "id":          trade_id,
        "symbol":      symbol.upper().strip(),
        "shares":      round(shares, 6),
        "price_entry": round(price_entry, 4),
        "price_exit":  None,
        "date_entry":  date_entry or datetime.now().strftime("%d/%m/%Y"),
        "date_exit":   None,
        "pnl_eur":     None,
        "status":      "open",
        "notes":       notes,
    }
    trades.append(trade)
    save_flip_log(trades)
    logging.info(f"[flip_log] Trade aberto: #{trade_id} {symbol} x{shares} @ ${price_entry}")
    return trade

def close_flip_trade(
    trade_id: int,
    price_exit: float,
    date_exit: str | None = None,
) -> dict | None:
    """
    Fecha um trade Flip Fund e calcula o P&L em EUR (usando preços USD; sem conversão automática).
    Devolve o trade actualizado, ou None se o ID não for encontrado.
    """
    trades = load_flip_log()
    for t in trades:
        if t["id"] == trade_id and t["status"] == "open":
            t["price_exit"] = round(price_exit, 4)
            t["date_exit"]  = date_exit or datetime.now().strftime("%d/%m/%Y")
            t["pnl_eur"]    = round((price_exit - t["price_entry"]) * t["shares"], 2)
            t["status"]     = "closed"
            save_flip_log(trades)
            logging.info(
                f"[flip_log] Trade fechado: #{trade_id} {t['symbol']} "
                f"P&L ${t['pnl_eur']:+.2f}"
            )
            return t
    return None

def delete_flip_trade(trade_id: int) -> bool:
    """Remove um trade pelo ID (para correcções). Devolve True se removido."""
    trades = load_flip_log()
    new    = [t for t in trades if t["id"] != trade_id]
    if len(new) == len(trades):
        return False
    save_flip_log(new)
    logging.info(f"[flip_log] Trade removido: #{trade_id}")
    return True

def get_flip_summary() -> dict:
    """
    Devolve resumo do Flip Fund:
      - trades_open: list de trades abertos
      - trades_closed: list de trades fechados
      - total_pnl: P&L total realizado (só fechados)
      - best_trade / worst_trade: dict dos melhores/piores trades fechados
      - win_rate: % trades fechados com lucro
    """
    trades  = load_flip_log()
    closed  = [t for t in trades if t["status"] == "closed"]
    opened  = [t for t in trades if t["status"] == "open"]
    pnl_sum = sum(t["pnl_eur"] or 0 for t in closed)
    winners = [t for t in closed if (t["pnl_eur"] or 0) > 0]
    win_rate = len(winners) / len(closed) * 100 if closed else 0
    best  = max(closed, key=lambda x: x["pnl_eur"] or 0, default=None)
    worst = min(closed, key=lambda x: x["pnl_eur"] or 0, default=None)
    return {
        "trades_open":   opened,
        "trades_closed": closed,
        "total_pnl":     round(pnl_sum, 2),
        "best_trade":    best,
        "worst_trade":   worst,
        "win_rate":      round(win_rate, 1),
        "n_open":        len(opened),
        "n_closed":      len(closed),
    }


# ── Feature 8 — Dip Persistente (streak dia-a-dia) ───────────────────────────
#
# Estrutura do JSON (_dipr_dip_streaks.json):
# {
#   "streaks": {
#     "AAPL": {
#       "first_seen":       "2026-04-28",
#       "last_seen":        "2026-04-30",
#       "streak_days":      3,
#       "alerted_at_streak": 2,      ← último streak_day em que enviámos alerta
#       "entries": [
#         {"date": "2026-04-28", "score": 65.2, "price": 180.5,
#          "change": -5.2, "verdict": "MONITORIZAR"},
#         ...
#       ]
#     }
#   }
# }

def load_dip_streaks() -> dict:
    """Devolve o dicionário completo de streaks activos."""
    return _read(_DIP_STREAKS_FILE).get("streaks", {})

def _save_dip_streaks(streaks: dict) -> None:
    _write(_DIP_STREAKS_FILE, {"streaks": streaks})


def record_dip_day(
    symbol: str,
    score: float,
    price: float,
    change_pct: float,
    verdict: str,
) -> dict:
    """
    Regista que `symbol` apareceu hoje no scan com score >= MIN_DIP_SCORE.

    Regras:
      - Se já foi registado hoje (scan correu duas vezes), devolve o estado
        actual sem duplicar a entrada.
      - Se `last_seen` é de um dia anterior, incrementa streak_days.
      - Se symbol é novo, cria entrada com streak_days=1.

    Devolve o dict do streak actualizado (útil para decidir o alerta).
    """
    streaks = load_dip_streaks()
    today   = datetime.now().date().isoformat()
    sym     = symbol.upper()

    if sym in streaks:
        s = streaks[sym]
        if s.get("last_seen") == today:
            # Já registado hoje — devolve sem alterar
            return s
        # Dia novo — incrementa streak
        s["streak_days"] += 1
        s["last_seen"]    = today
    else:
        # Primeira vez que este ticker aparece no scan
        streaks[sym] = {
            "first_seen":        today,
            "last_seen":         today,
            "streak_days":       1,
            "alerted_at_streak": 0,
            "entries":           [],
        }

    streaks[sym]["entries"].append({
        "date":    today,
        "score":   round(score, 1),
        "price":   round(price, 4) if price else None,
        "change":  round(change_pct, 2),
        "verdict": verdict,
    })
    # Mantém apenas os últimos 30 dias de histórico por ticker
    streaks[sym]["entries"] = streaks[sym]["entries"][-30:]

    _save_dip_streaks(streaks)
    logging.debug(
        f"[streak] {sym}: streak_days={streaks[sym]['streak_days']} "
        f"score={score:.1f} last_seen={today}"
    )
    return streaks[sym]


def get_dip_streak(symbol: str) -> dict | None:
    """Devolve o estado do streak de `symbol`, ou None se não existe."""
    return load_dip_streaks().get(symbol.upper())


def reset_dip_streak(symbol: str) -> None:
    """Remove o streak de `symbol` (stock recuperou ou saiu do scan)."""
    streaks = load_dip_streaks()
    sym     = symbol.upper()
    if sym in streaks:
        del streaks[sym]
        _save_dip_streaks(streaks)
        logging.debug(f"[streak] {sym}: streak resetado")


def mark_persistent_alerted(symbol: str, streak_day: int) -> None:
    """
    Marca o streak de `symbol` como tendo sido alertado no dia `streak_day`.
    Evita enviar o mesmo alerta de persistência múltiplas vezes no mesmo nível.
    """
    streaks = load_dip_streaks()
    sym     = symbol.upper()
    if sym in streaks:
        streaks[sym]["alerted_at_streak"] = streak_day
        _save_dip_streaks(streaks)


def expire_missing_streaks(seen_symbols: set) -> list[str]:
    """
    Remove da store os streaks de symbols que NÃO apareceram no scan de hoje
    (i.e., o stock saiu do radar — recuperou ou desceu abaixo do threshold).

    Chamado no finally de run_scan() com o conjunto de symbols que passaram
    score >= MIN_DIP_SCORE neste ciclo de scan.

    Devolve lista de symbols expirados (para logging).
    """
    streaks = load_dip_streaks()
    today   = datetime.now().date().isoformat()
    seen    = {s.upper() for s in seen_symbols}
    expired = []

    to_delete = [
        sym for sym, s in streaks.items()
        if sym not in seen and s.get("last_seen") != today
    ]
    for sym in to_delete:
        del streaks[sym]
        expired.append(sym)

    if expired:
        _save_dip_streaks(streaks)
        logging.info(f"[streak] Expirados (saíram do scan): {expired}")

    return expired


def get_all_active_streaks(min_days: int = 2) -> list[dict]:
    """
    Devolve lista de streaks activos com streak_days >= min_days.
    Útil para o heartbeat ou um comando /streaks.
    Cada item: {symbol, streak_days, first_seen, last_seen, last_score, last_price}
    """
    streaks = load_dip_streaks()
    result  = []
    for sym, s in streaks.items():
        if s.get("streak_days", 0) >= min_days:
            entries    = s.get("entries", [])
            last_entry = entries[-1] if entries else {}
            result.append({
                "symbol":      sym,
                "streak_days": s["streak_days"],
                "first_seen":  s["first_seen"],
                "last_seen":   s["last_seen"],
                "last_score":  last_entry.get("score"),
                "last_price":  last_entry.get("price"),
                "last_change": last_entry.get("change"),
            })
    return sorted(result, key=lambda x: x["streak_days"], reverse=True)
