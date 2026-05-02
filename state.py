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
  load_dynamic_watchlist / save_dynamic_watchlist
  add_to_dynamic_watchlist / remove_from_dynamic_watchlist
  load_score_log / save_score_log / append_score_log
  get_ticker_score_history / get_last_score
  load_flip_log / save_flip_log
  add_flip_trade / close_flip_trade / delete_flip_trade
  get_flip_summary

NOTA DE SEGURANÇA (race condition):
  _save_json usa escrita atómica via ficheiro temporário + os.replace().
  Garante que nenhum leitor vê um JSON parcialmente escrito, mesmo que
  o APScheduler dispare macro_data.py e o Vigilante em simultâneo.
"""

import json
import logging
import os
import tempfile
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
    """
    Escrita atómica: serializa para um ficheiro temporário no mesmo
    directório e depois faz os.replace() (operação atómica no SO).
    Elimina o risco de corrupção por race condition ou falha a meio.
    """
    p = _path(filename)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        # Ficheiro temporário no mesmo directório → mesmo filesystem → replace atómico
        fd, tmp_path = tempfile.mkstemp(
            dir=str(p.parent),
            prefix=f".{p.name}.tmp.",
            suffix=".json",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, str(p))  # atómico em Linux/macOS/Windows
        except Exception:
            # Limpa o temporário se algo correu mal antes do replace
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
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


# ── Watchlist dinâmica (gerida via /watchlist add|remove) ────────────────────
#
# Mantém-se o nome de ficheiro `_dipr_watchlist.json` para preservar o estado
# já gravado no volume Railway de versões anteriores do bot.

_WATCHLIST_FILE = "_dipr_watchlist.json"


def load_dynamic_watchlist() -> list:
    return _read(_WATCHLIST_FILE).get("tickers", [])


def save_dynamic_watchlist(tickers: list) -> None:
    _write(_WATCHLIST_FILE, {
        "tickers": list(dict.fromkeys(t.upper() for t in tickers)),
    })


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

_SCORE_LOG_FILE = "_dipr_score_log.json"


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
    data    = load_score_log()
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


def get_last_score(symbol: str):
    history = get_ticker_score_history(symbol)
    if history:
        return history[-1]["score"]
    return None


# ── Flip Fund trade log ───────────────────────────────────────────────────────

_FLIP_LOG_FILE = "_dipr_flip_log.json"


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
    date_entry: str = "",
    notes: str = "",
) -> dict:
    """Regista entrada num trade Flip Fund (posição aberta). Devolve o dict criado."""
    trades   = load_flip_log()
    trade_id = max((t["id"] for t in trades), default=0) + 1
    trade    = {
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
    date_exit: str = "",
):
    """
    Fecha um trade Flip Fund e calcula o P&L em EUR (sem conversão automática
    de USD→EUR). Devolve o trade actualizado, ou None se o ID não existir.
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
    """Remove um trade pelo ID (correcções). Devolve True se removido."""
    trades = load_flip_log()
    new    = [t for t in trades if t["id"] != trade_id]
    if len(new) == len(trades):
        return False
    save_flip_log(new)
    logging.info(f"[flip_log] Trade removido: #{trade_id}")
    return True


def get_flip_summary() -> dict:
    """
    Devolve resumo do Flip Fund (trades abertos/fechados, P&L total, win rate).
    """
    trades   = load_flip_log()
    closed   = [t for t in trades if t["status"] == "closed"]
    opened   = [t for t in trades if t["status"] == "open"]
    pnl_sum  = sum((t["pnl_eur"] or 0) for t in closed)
    winners  = [t for t in closed if (t["pnl_eur"] or 0) > 0]
    win_rate = (len(winners) / len(closed) * 100) if closed else 0
    best     = max(closed, key=lambda x: x["pnl_eur"] or 0, default=None)
    worst    = min(closed, key=lambda x: x["pnl_eur"] or 0, default=None)
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
