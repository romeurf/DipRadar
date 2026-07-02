"""utils/sector_cache.py — ticker → sector lookup with persistent cache.

Sectors virtually never change, so a 90-day TTL disk cache turns N
yfinance .info calls per scan into dict lookups. Used by the watchlist
triage layer, which needs sectors for BOTH holdings and candidates —
resolving both through the same cache guarantees the cluster math
compares like-for-like strings.

Cache file: /data/sector_cache.json (Railway volume) or local data/.
Schema: {"TICKER": {"sector": "Technology", "fetched_at": iso}}
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger("sector_cache")

_DATA_DIR = (
    Path("/data") if Path("/data").exists()
    else Path(__file__).resolve().parent.parent / "data"
)
_CACHE_PATH = _DATA_DIR / "sector_cache.json"
_TTL_DAYS = 90

_cache: dict = {}
_loaded = False


def _load() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    if _CACHE_PATH.exists():
        try:
            raw = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                _cache.update(raw)
        except Exception as e:
            log.warning(f"[sector_cache] read failed: {e}")


def _persist() -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(_cache, indent=2), encoding="utf-8")
    except Exception as e:
        log.debug(f"[sector_cache] write failed: {e}")


def _fresh(entry: dict) -> bool:
    try:
        age_days = (
            datetime.now() - datetime.fromisoformat(entry.get("fetched_at", ""))
        ).total_seconds() / 86400
        return age_days < _TTL_DAYS
    except Exception:
        return False


def get_sector(ticker: str) -> str:
    """Return the yfinance sector string for a ticker; "" when unresolvable.

    Cache-first (90d TTL); one lazy yfinance .info call on miss. A
    successful empty answer ("" — e.g. ETFs have no sector) is cached
    too, so we don't re-hit yfinance for known sector-less symbols.
    """
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return ""

    _load()
    entry = _cache.get(ticker)
    if entry is not None and _fresh(entry):
        return entry.get("sector", "") or ""

    sector = ""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        sector = info.get("sector", "") or ""
    except Exception as e:
        log.debug(f"[sector_cache] fetch {ticker}: {e}")
        # Fetch failure: don't cache — retry next call. But if we have a
        # stale entry, serving it beats returning nothing.
        if entry is not None:
            return entry.get("sector", "") or ""
        return ""

    _cache[ticker] = {
        "sector": sector,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }
    _persist()
    return sector
