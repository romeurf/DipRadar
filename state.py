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
from datetime import date, datetime, timedelta
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


# ── Daily flip spend tracker ──────────────────────────────────────────────
# Tracks how much EUR the bot's allocation engine has already committed today,
# across multi-alert scans. Persisted to /data/daily_flip_spent.json so it
# survives Railway restarts (in-process dict would silently reset on deploy).
#
# Used by allocation_engine to shrink "available cash" passed into sizing as
# alerts stack within the same day. Auto-prunes to last 7 days.

def _today_iso() -> str:
    return date.today().isoformat()


# ── PPR (Plano Poupança Reforma) — Telegram-managed ────────────────────────
# Migrated from PPR_SHARES / PPR_AVG_COST env vars (which caused a €19B
# bug when set with wrong values and were never wired into /sync_portfolio).
# Single source of truth: /data/ppr.json, mutated via /ppr set|clear.

_PPR_FILE = "ppr.json"


_DEFAULT_PPR_ISIN = "PTARMJHM0003"  # Invest Tendências Globais PPR (Banco Invest)


def get_ppr() -> dict:
    """Return {shares, avg_cost, current_nav, nav_updated, isin, nav_source}.

    current_nav: optional manually-updated NAV per unit (€). When set,
    the snapshot uses it to compute live value + P&L. When None, value
    falls back to shares × avg_cost (frozen at cost basis, no P&L).

    nav_updated: ISO date of last NAV update (helps user see staleness).
    isin: which PPR fund (default PTARMJHM0003); used by the scraper.
    nav_source: "manual" or "investing.com" or fallback identifier.
    """
    data = _load_json(_PPR_FILE, default={}) or {}
    if not isinstance(data, dict):
        return {}
    return {
        "shares":      float(data.get("shares", 0) or 0),
        "avg_cost":    float(data.get("avg_cost", 0) or 0),
        "current_nav": float(data.get("current_nav", 0) or 0) or None,
        "nav_updated": data.get("nav_updated") or None,
        "updated":     data.get("updated") or None,
        "isin":        (data.get("isin") or _DEFAULT_PPR_ISIN).upper().strip(),
        "nav_source":  data.get("nav_source") or None,
    }


def set_ppr(shares: float, avg_cost: float) -> dict:
    """Overwrite PPR core holdings (shares + avg_cost). Keeps any current_nav
    that's already set, so /ppr set after /ppr nav doesn't lose the NAV."""
    existing = get_ppr()
    payload = {
        "shares":      float(shares),
        "avg_cost":    float(avg_cost),
        "current_nav": existing.get("current_nav"),
        "nav_updated": existing.get("nav_updated"),
        "updated":     _today_iso(),
    }
    _save_json(_PPR_FILE, payload)
    return payload


def set_ppr_nav(current_nav: float, source: str = "manual") -> dict:
    """Update just the NAV (price per unit).

    source: "manual" (default, /ppr nav), "investing.com" (scraper auto-update),
            or any other identifier the caller wants to record. Used by the
            display to show provenance ("Cotação obtida de investing.com").
    """
    existing = get_ppr()
    payload = {
        "shares":      existing.get("shares", 0.0),
        "avg_cost":    existing.get("avg_cost", 0.0),
        "current_nav": float(current_nav),
        "nav_updated": _today_iso(),
        "updated":     existing.get("updated") or _today_iso(),
        "isin":        existing.get("isin") or _DEFAULT_PPR_ISIN,
        "nav_source":  source,
    }
    _save_json(_PPR_FILE, payload)
    return payload


def set_ppr_isin(isin: str) -> dict:
    """Change the PPR ISIN (which fund the scraper queries). Preserves all
    other PPR state."""
    existing = get_ppr()
    payload = dict(existing)
    payload["isin"]    = isin.upper().strip()
    payload["updated"] = _today_iso()
    _save_json(_PPR_FILE, payload)
    return payload


def clear_ppr() -> None:
    _save_json(_PPR_FILE, {
        "shares":      0.0,
        "avg_cost":    0.0,
        "current_nav": None,
        "nav_updated": None,
        "updated":     _today_iso(),
    })


# ── ETFs (passive holdings, snapshot-only) ─────────────────────────────────
# Same pattern as PPR: Telegram-managed, no env vars, no contamination of
# the active scan / conviction / ML pipeline. For European-listed ETFs
# (Xetra, LSE) that the user holds but doesn't actively trade — they should
# show up in the aggregate snapshot value but never trigger alerts.

_ETFS_FILE = "etfs.json"


def get_etfs() -> dict:
    """Return {ticker: {shares, avg_cost}} dict. Empty if none configured."""
    data = _load_json(_ETFS_FILE, default={}) or {}
    if not isinstance(data, dict):
        return {}
    out: dict = {}
    for tkr, pos in data.items():
        if not isinstance(pos, dict):
            continue
        try:
            shares   = float(pos.get("shares", 0) or 0)
            avg_cost = float(pos.get("avg_cost", 0) or 0)
        except (TypeError, ValueError):
            continue
        if shares <= 0:
            continue
        out[str(tkr).upper().strip()] = {
            "shares":   shares,
            "avg_cost": avg_cost,
            "updated":  pos.get("updated", ""),
        }
    return out


def set_etf(ticker: str, shares: float, avg_cost: float) -> dict:
    """Upsert one ETF. Use shares=0 to remove via the same code path."""
    data = _load_json(_ETFS_FILE, default={}) or {}
    if not isinstance(data, dict):
        data = {}
    key = ticker.upper().strip()
    if shares <= 0:
        data.pop(key, None)
    else:
        data[key] = {
            "shares":   float(shares),
            "avg_cost": float(avg_cost),
            "updated":  _today_iso(),
        }
    _save_json(_ETFS_FILE, data)
    return get_etfs()


def remove_etf(ticker: str) -> bool:
    """Remove one ETF. Returns True if removed."""
    data = _load_json(_ETFS_FILE, default={}) or {}
    if not isinstance(data, dict):
        return False
    key = ticker.upper().strip()
    if key not in data:
        return False
    data.pop(key)
    _save_json(_ETFS_FILE, data)
    return True


def clear_etfs() -> None:
    _save_json(_ETFS_FILE, {})


def load_daily_flip_spent() -> dict:
    """Return {YYYY-MM-DD: amount_eur} for today + recent prior days."""
    return _load_json("daily_flip_spent.json", default={}) or {}


def get_today_flip_spent() -> float:
    data = load_daily_flip_spent()
    return float(data.get(_today_iso(), 0.0))


def add_today_flip_spent(amount_eur: float) -> float:
    """Increment today's committed amount; return new running total."""
    if amount_eur <= 0:
        return get_today_flip_spent()
    data = load_daily_flip_spent()
    today = _today_iso()
    data[today] = float(data.get(today, 0.0)) + float(amount_eur)
    cutoff = (date.today() - timedelta(days=7)).isoformat()
    data = {k: v for k, v in data.items() if k >= cutoff}
    _save_json("daily_flip_spent.json", data)
    return data[today]


def reset_today_flip_spent() -> None:
    """Manual reset (e.g. after a sell or pipeline correction)."""
    data = load_daily_flip_spent()
    data[_today_iso()] = 0.0
    _save_json("daily_flip_spent.json", data)


# ── Scan heartbeat ────────────────────────────────────────────────────────
# Persistent telemetry of every DIP scan run. The point is to make silence
# legible: when a week passes with 0 alerts, the heartbeats let us tell
# "the scanner ran 5 times, screened 80 tickers each, top score was 48
# (threshold 50)" from "the scanner is broken — 0 tickers screened, all
# features NaN". One is healthy patience; the other is a real problem.
#
# Stored as a rolling list. Pruned to last 60 days on each write so the
# JSON stays small.

def append_scan_heartbeat(entry: dict) -> None:
    """Append one heartbeat record.

    Caller provides at minimum:
      - timestamp (ISO datetime string)
      - n_screened (raw universe size that entered the loop)
      - n_scored (how many got a valid score)
      - n_alerted (how many passed the score threshold)
      - top_score (max score in the scan, or None if none scored)
      - threshold (effective MIN_DIP_SCORE used)
    Extra keys are stored verbatim.
    """
    if not isinstance(entry, dict):
        return
    data = _load_json("scan_heartbeat.json", default=[]) or []
    if not isinstance(data, list):
        data = []
    data.append(entry)
    # Prune to last 60 days
    cutoff_iso = (date.today() - timedelta(days=60)).isoformat()
    pruned = []
    for h in data:
        ts = str(h.get("timestamp") or "")
        if ts[:10] >= cutoff_iso:
            pruned.append(h)
    _save_json("scan_heartbeat.json", pruned)


def load_scan_heartbeats(days: int = 7) -> list:
    """Return heartbeats from the last `days` days (most recent first)."""
    data = _load_json("scan_heartbeat.json", default=[]) or []
    if not isinstance(data, list):
        return []
    cutoff_iso = (date.today() - timedelta(days=days)).isoformat()
    out = [h for h in data if str(h.get("timestamp") or "")[:10] >= cutoff_iso]
    return sorted(out, key=lambda h: h.get("timestamp", ""), reverse=True)


# ── Weekly log ────────────────────────────────────────────────────────────

def load_weekly_log() -> list:
    return _load_json("weekly_log.json", [])


def save_weekly_log(entries: list) -> None:
    _save_json("weekly_log.json", entries)


def append_weekly_log(entry: dict) -> None:
    entries = load_weekly_log()
    # Adiciona data ISO para filtrar por semana (o campo "date" só tem dd/mm)
    if "date_iso" not in entry:
        entry = {**entry, "date_iso": datetime.now().date().isoformat()}
    # Remove entradas com mais de 8 dias para não acumular semanas antigas
    cutoff = (datetime.now().date() - timedelta(days=8)).isoformat()
    entries = [e for e in entries if e.get("date_iso", "9999-99-99") >= cutoff]
    entries.append(entry)
    save_weekly_log(entries)


# ── Rejected log ──────────────────────────────────────────────────────────

def load_rejected_log() -> list:
    data = _load_json("rejected_log.json", [])
    # append_rejected_log() escreve {"entries": [...]}.
    # load_rejected_log() devolvia o dict inteiro → sorted() iterava sobre
    # as chaves ("entries") → AttributeError 'str' has no attribute 'get'.
    if isinstance(data, dict):
        return data.get("entries", [])
    return data if isinstance(data, list) else []


def append_rejected_log(entry: dict) -> None:
    data    = _read(_REJECTED_FILE)
    entries = data.get("entries", [])
    today   = datetime.now().date().isoformat()
    # Guardar historicamente (máx 180 dias) — não apagar entradas anteriores
    cutoff  = (datetime.now() - timedelta(days=180)).date().isoformat()
    entries = [e for e in entries if e.get("date_iso", "") >= cutoff]
    entries.append({
        **entry,
        "time":        datetime.now().strftime("%H:%M"),
        "date_iso":    today,
        "outcome_pct": None,   # preenchido pelo shadow tracker semanal
        "outcome_spy": None,
        "outcome_date": None,
    })
    _write(_REJECTED_FILE, {"entries": entries})


_REJECTED_HORIZONS = {
    "DIP":      [30, 60, 90],
    "MOMENTUM": [30, 60, 90],
    "DEFAULT":  [30, 60, 90],
}


def _spy_return_between(alert_date, target_date, spy_hist) -> float | None:
    """Retorno SPY entre alert_date e target_date usando hist já descarregado."""
    try:
        start_row = spy_hist[spy_hist.index.date >= alert_date]  # type: ignore[attr-defined]
        end_row   = spy_hist[spy_hist.index.date <= target_date]  # type: ignore[attr-defined]
        if start_row.empty or end_row.empty:
            return None
        p0 = float(start_row["Close"].iloc[0])
        p1 = float(end_row["Close"].iloc[-1])
        return round((p1 / p0 - 1) * 100, 2) if p0 > 0 else None
    except Exception:
        return None


def fill_rejected_outcomes() -> dict:
    """Preenche outcomes a 30d/60d/90d para rejeições com histórico suficiente.

    Horizontes por radar:
      DIP:      30d, 60d, 90d  (hold estimado do DipRadar)
      MOMENTUM: 30d, 60d       (hold estimado do MomentumRadar)

    Chamado semanalmente (domingo, antes do ML Outcomes).
    """
    data    = _read(_REJECTED_FILE)
    entries = data.get("entries", [])
    today   = datetime.now().date()
    filled  = 0

    try:
        import yfinance as yf
        spy_hist = yf.Ticker("SPY").history(period="200d", auto_adjust=True)
        if not spy_hist.empty:
            spy_hist.index = spy_hist.index.tz_localize(None) if spy_hist.index.tz else spy_hist.index
        else:
            spy_hist = None
    except Exception:
        spy_hist = None

    ticker_hist_cache: dict = {}

    for e in entries:
        date_iso = e.get("date_iso", "")
        symbol   = e.get("symbol", "")
        price_in = e.get("price") or e.get("close")
        if not date_iso or not symbol or not price_in:
            continue
        try:
            alert_date = datetime.strptime(date_iso, "%Y-%m-%d").date()
        except ValueError:
            continue

        radar    = e.get("radar", "DEFAULT")
        horizons = _REJECTED_HORIZONS.get(radar, _REJECTED_HORIZONS["DEFAULT"])
        days_old = (today - alert_date).days

        # Só processar se pelo menos o horizonte mínimo passou
        if days_old < min(horizons):
            continue

        # Descarregar histórico do ticker uma vez
        if symbol not in ticker_hist_cache:
            try:
                h = yf.Ticker(symbol).history(
                    start=str(alert_date), end=str(today), auto_adjust=True
                )
                h.index = h.index.tz_localize(None) if h.index.tz else h.index
                ticker_hist_cache[symbol] = h if not h.empty else None
            except Exception:
                ticker_hist_cache[symbol] = None
        hist = ticker_hist_cache.get(symbol)
        if hist is None:
            continue

        entry_changed = False
        for h_days in horizons:
            key_ret = f"outcome_{h_days}d"
            key_spy = f"outcome_spy_{h_days}d"
            if e.get(key_ret) is not None:
                continue  # já preenchido
            if days_old < h_days:
                continue  # ainda não chegou o tempo

            target_date = alert_date + timedelta(days=h_days)
            valid = hist[hist.index.date <= target_date]  # type: ignore[attr-defined]
            if valid.empty:
                continue
            price_out = float(valid["Close"].iloc[-1])
            ret_pct   = round((price_out / float(price_in) - 1) * 100, 2)
            e[key_ret] = ret_pct
            if spy_hist is not None:
                e[key_spy] = _spy_return_between(alert_date, target_date, spy_hist)
            entry_changed = True

        if entry_changed:
            e["outcome_updated"] = str(today)
            filled += 1

    if filled > 0:
        _write(_REJECTED_FILE, {"entries": entries})

    pending = sum(
        1 for e in entries
        if e.get("date_iso")
        and e.get("outcome_30d") is None
        and (datetime.now().date() - datetime.strptime(e["date_iso"], "%Y-%m-%d").date()).days >= 30
    )
    return {"filled": filled, "pending": pending, "total": len(entries)}


def get_rejected_stats(lookback_days: int = 90) -> dict:
    """Estatísticas do shadow portfolio de rejeições por horizonte e razão."""
    entries = load_rejected_log()
    cutoff  = (datetime.now() - timedelta(days=lookback_days)).date().isoformat()
    entries = [e for e in entries if e.get("date_iso", "") >= cutoff]

    result: dict = {"lookback_days": lookback_days, "total_rejected": len(entries)}

    for h_days in [30, 60, 90]:
        key = f"outcome_{h_days}d"
        key_spy = f"outcome_spy_{h_days}d"
        scored = [e for e in entries if e.get(key) is not None]
        if not scored:
            continue
        returns = [e[key] for e in scored]
        alphas  = [e[key] - e[key_spy] for e in scored if e.get(key_spy) is not None]
        by_reason: dict = {}
        by_reason_alpha: dict = {}
        for e in scored:
            r = e.get("reason", "unknown")
            by_reason.setdefault(r, []).append(e[key])
            if e.get(key_spy) is not None:
                by_reason_alpha.setdefault(r, []).append(e[key] - e[key_spy])

        by_reason_detail = {
            k: {
                "n":          len(v),
                "avg_return": round(sum(v) / len(v), 2),
                "avg_alpha":  round(sum(by_reason_alpha[k]) / len(by_reason_alpha[k]), 2)
                              if k in by_reason_alpha else None,
            }
            for k, v in by_reason.items()
        }

        result[f"h{h_days}d"] = {
            "n":              len(scored),
            "win_rate":       round(sum(1 for r in returns if r > 0) / len(returns), 3),
            "avg_return":     round(sum(returns) / len(returns), 2),
            "avg_alpha":      round(sum(alphas) / len(alphas), 2) if alphas else None,
            "by_reason":      {k: round(sum(v) / len(v), 2) for k, v in sorted(by_reason.items(), key=lambda x: -abs(sum(x[1])/len(x[1])))},
            "by_reason_detail": by_reason_detail,
        }

    return result


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


# ─── Sector Radar state machine ──────────────────────────────────────────────
#
# Each ticker tracks one state. Transitions fire alerts; staying in a state
# is silent (hysteresis: trigger >70, reset <50). The Telegram alert layer
# calls sector_state_should_alert() to decide whether to send.
#
# Schema per ticker:
#   {
#     "state":             "NEUTRAL" | "SELL_ZONE" | "BUY_ZONE",
#     "last_alert_at":     ISO-8601 timestamp,
#     "last_alert_state":  the state that triggered the last alert,
#     "last_combined_buy": float — for diagnostic
#     "last_exit_strength": float
#     "updated_at":        ISO-8601 timestamp
#   }
#
# State transitions (with hysteresis, threshold 70 / reset 50):
#   NEUTRAL → SELL_ZONE  when exit_strength crosses ≥70   — ALERT
#   NEUTRAL → BUY_ZONE   when combined_buy   crosses ≥70   — ALERT
#   SELL_ZONE → NEUTRAL  when exit_strength drops <50      — SILENT (reset)
#   BUY_ZONE  → NEUTRAL  when combined_buy   drops <50     — SILENT (reset)
#
# A ticker in NEUTRAL is mentioned in the daily morning report only if a
# signal recently fired (handled at the report layer, not here).

_SECTOR_STATE_FILE = "sector_radar_state.json"

_BUY_TRIGGER_THRESHOLD  = 70.0
_BUY_RESET_THRESHOLD    = 50.0
_SELL_TRIGGER_THRESHOLD = 70.0
_SELL_RESET_THRESHOLD   = 50.0


def load_sector_states() -> dict:
    """Return the full sector-state dict {ticker: state_record}."""
    return _read(_SECTOR_STATE_FILE)


def save_sector_states(states: dict) -> None:
    _write(_SECTOR_STATE_FILE, states)


def get_sector_state(ticker: str) -> dict:
    """Return state record for a ticker, or a default NEUTRAL record."""
    states = load_sector_states()
    return states.get(ticker.upper(), {
        "state":              "NEUTRAL",
        "last_alert_at":      None,
        "last_alert_state":   None,
        "last_combined_buy":  0.0,
        "last_exit_strength": 0.0,
        "updated_at":         None,
    })


def update_sector_state(
    ticker: str,
    *,
    combined_buy: float,
    exit_strength: float,
) -> tuple[str, bool]:
    """Apply the FSM to the new scores and persist.

    Returns ``(new_state, should_alert)``. The caller is responsible for
    actually sending the alert when ``should_alert=True``.
    """
    states = load_sector_states()
    t = ticker.upper()
    record = states.get(t, {
        "state":              "NEUTRAL",
        "last_alert_at":      None,
        "last_alert_state":   None,
        "last_combined_buy":  0.0,
        "last_exit_strength": 0.0,
        "updated_at":         None,
    })

    prev_state = record.get("state", "NEUTRAL")
    new_state = prev_state
    should_alert = False

    # Transition logic — hysteresis built in
    if prev_state == "NEUTRAL":
        # SELL takes priority: a holding flagged for exit is more urgent
        # than another buy opportunity (defensive bias).
        if exit_strength >= _SELL_TRIGGER_THRESHOLD:
            new_state = "SELL_ZONE"
            should_alert = True
        elif combined_buy >= _BUY_TRIGGER_THRESHOLD:
            new_state = "BUY_ZONE"
            should_alert = True

    elif prev_state == "SELL_ZONE":
        if exit_strength < _SELL_RESET_THRESHOLD:
            new_state = "NEUTRAL"
            # Silent — signal faded
        # else: stay in SELL_ZONE, no new alert

    elif prev_state == "BUY_ZONE":
        if combined_buy < _BUY_RESET_THRESHOLD:
            new_state = "NEUTRAL"
        # else: stay in BUY_ZONE, no new alert

    # Persist
    now_iso = datetime.utcnow().isoformat()
    record["state"] = new_state
    record["last_combined_buy"]  = float(combined_buy)
    record["last_exit_strength"] = float(exit_strength)
    record["updated_at"] = now_iso
    if should_alert:
        record["last_alert_at"]    = now_iso
        record["last_alert_state"] = new_state

    states[t] = record
    save_sector_states(states)
    return new_state, should_alert


def reset_sector_state(ticker: str) -> bool:
    """Force a ticker back to NEUTRAL (admin use; manually clear stuck state)."""
    states = load_sector_states()
    t = ticker.upper()
    if t not in states:
        return False
    states[t]["state"] = "NEUTRAL"
    states[t]["updated_at"] = datetime.utcnow().isoformat()
    save_sector_states(states)
    return True


def sector_states_summary() -> dict:
    """Counts per state across all tracked tickers — for diagnostics."""
    states = load_sector_states()
    counts = {"NEUTRAL": 0, "SELL_ZONE": 0, "BUY_ZONE": 0}
    for record in states.values():
        s = record.get("state", "NEUTRAL")
        if s in counts:
            counts[s] += 1
    return {
        "counts":   counts,
        "n_total":  len(states),
        "tickers_in_buy_zone":  [t for t, r in states.items() if r.get("state") == "BUY_ZONE"],
        "tickers_in_sell_zone": [t for t, r in states.items() if r.get("state") == "SELL_ZONE"],
    }
