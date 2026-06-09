"""
cache.py — Cache TTL em memória para chamadas yfinance/Tavily custosas.

Evita pedidos duplicados quando /analisar e o scan automático apanham
o mesmo ticker em menos de TTL_MINUTES minutos.

Uso:
    from utils.cache import get_cached, set_cached, invalidate, cache_stats

    data = get_cached("fundamentals", "AAPL")
    if data is None:
        data = get_fundamentals("AAPL")
        set_cached("fundamentals", "AAPL", data)

Buckets disponíveis (TTL independente, sobreposto por env vars):
    fundamentals   — CACHE_TTL_FUNDAMENTALS  (default 15 min)
    news           — CACHE_TTL_NEWS          (default 30 min)
    catalyst       — CACHE_TTL_CATALYST      (default 60 min)
    rsi            — CACHE_TTL_RSI           (default 10 min)
    earnings_days  — CACHE_TTL_EARNINGS      (default 60 min)

Env vars (segundos):
    CACHE_TTL_FUNDAMENTALS=900
    CACHE_TTL_NEWS=1800
    CACHE_TTL_CATALYST=3600
    CACHE_TTL_RSI=600
    CACHE_TTL_EARNINGS=3600
    CACHE_PURGE_INTERVAL=300   # segundos entre purges automáticos (0=desligado)
"""

import os
import time
import threading
import logging
from typing import Any


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (ValueError, TypeError):
        return default


_TTL: dict[str, int] = {
    "fundamentals":  _env_int("CACHE_TTL_FUNDAMENTALS", 15 * 60),
    "news":          _env_int("CACHE_TTL_NEWS",          30 * 60),
    "catalyst":      _env_int("CACHE_TTL_CATALYST",      60 * 60),
    "rsi":           _env_int("CACHE_TTL_RSI",           10 * 60),
    "earnings_days": _env_int("CACHE_TTL_EARNINGS",      60 * 60),
}

_PURGE_INTERVAL: int = _env_int("CACHE_PURGE_INTERVAL", 5 * 60)  # 0 = desligado

# _store[bucket][key] = (value, timestamp)
_store: dict[str, dict[str, tuple[Any, float]]] = {b: {} for b in _TTL}
_lock  = threading.Lock()

# Hit/miss counters por bucket
_hits:   dict[str, int] = {b: 0 for b in _TTL}
_misses: dict[str, int] = {b: 0 for b in _TTL}


# ── API pública ────────────────────────────────────────────────────────────

def get_cached(bucket: str, key: str) -> Any | None:
    """
    Devolve o valor em cache se existir e não tiver expirado.
    Devolve None caso contrário. Regista hit/miss.
    """
    ttl = _TTL.get(bucket)
    if ttl is None:
        return None
    with _lock:
        entry = _store[bucket].get(key)
    if entry is None:
        with _lock:
            _misses[bucket] = _misses.get(bucket, 0) + 1
        return None
    value, ts = entry
    if time.monotonic() - ts > ttl:
        with _lock:
            _store[bucket].pop(key, None)
            _misses[bucket] = _misses.get(bucket, 0) + 1
        return None
    with _lock:
        _hits[bucket] = _hits.get(bucket, 0) + 1
    return value


def set_cached(bucket: str, key: str, value: Any) -> None:
    """Guarda valor em cache com timestamp actual."""
    if bucket not in _store:
        logging.warning(f"[cache] bucket desconhecido: {bucket!r}")
        return
    with _lock:
        _store[bucket][key] = (value, time.monotonic())


def invalidate(bucket: str, key: str | None = None) -> None:
    """
    Invalida uma chave específica ou todo o bucket se key=None.
    """
    with _lock:
        if key is None:
            _store[bucket].clear()
        else:
            _store[bucket].pop(key, None)


def purge_expired() -> int:
    """
    Remove todas as entradas expiradas de todos os buckets.
    Devolve o número de entradas removidas.
    """
    now     = time.monotonic()
    removed = 0
    with _lock:
        for bucket, entries in _store.items():
            ttl  = _TTL[bucket]
            dead = [k for k, (_, ts) in entries.items() if now - ts > ttl]
            for k in dead:
                del entries[k]
            removed += len(dead)
    if removed:
        logging.debug(f"[cache] purge: {removed} entradas expiradas removidas")
    return removed


def cache_stats() -> str:
    """Linha de debug com contagem de entradas vivas e hit-rate por bucket."""
    now   = time.monotonic()
    parts = []
    with _lock:
        for bucket, entries in _store.items():
            ttl    = _TTL[bucket]
            live   = sum(1 for _, (_, ts) in entries.items() if now - ts <= ttl)
            hits   = _hits.get(bucket, 0)
            misses = _misses.get(bucket, 0)
            total  = hits + misses
            rate   = f"{hits/total*100:.0f}%" if total else "n/a"
            parts.append(f"{bucket}:{live}e h{rate}")
    return "Cache — " + " | ".join(parts)


def reset_stats() -> None:
    """Reinicia contadores de hit/miss (ex: ao iniciar novo scan)."""
    with _lock:
        for b in _hits:
            _hits[b]   = 0
            _misses[b] = 0


# ── Auto-purge background thread ─────────────────────────────────────────

def _purge_loop() -> None:
    while True:
        time.sleep(_PURGE_INTERVAL)
        try:
            purge_expired()
        except Exception as e:
            logging.warning(f"[cache] purge_loop error: {e}")


if _PURGE_INTERVAL > 0:
    _purge_thread = threading.Thread(
        target=_purge_loop, daemon=True, name="cache-purge"
    )
    _purge_thread.start()
    logging.debug(f"[cache] auto-purge activo a cada {_PURGE_INTERVAL}s")


# Log dos TTLs activos no arranque
logging.debug(
    "[cache] TTLs: " +
    ", ".join(f"{b}={v//60}m" for b, v in _TTL.items())
)
