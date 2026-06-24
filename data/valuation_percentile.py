"""data/valuation_percentile.py — Current P/E vs own 3y historical range.

Used by the dislocation scorer (sub-score #5, "absolute valuation guard")
to enforce real cheapness in absolute fundamentals — not just "the stock
dropped a lot from the high" (which is what the OLD biggest-drops screen
did and is precisely the value-trap pattern).

The signal: `valuation_percentile(ticker)` returns the current forward P/E
expressed as a percentile within the ticker's OWN 3-year history. A name
trading at its own P25 (cheaper than 75% of its own history) is "real
cheap"; one at P85+ is expensive regardless of how far it has dropped.

Cached to /data/pe_history_cache.json with a 7-day TTL per ticker. A 3y
monthly P/E series doesn't move materially in a week, and we want to
avoid re-fetching the same 3-year monthly history on every scan (caller-
side rate limits would otherwise dominate).

Bypasses `market_client.get_historical_pe`'s `time.sleep(5)` because that
guard was sized for batch backfills, not live scans. On-demand single
fetches are short enough that explicit throttling isn't required —
yfinance handles rate-limits via 429 retries.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("valuation_percentile")

_DATA_DIR = (
    Path("/data") if Path("/data").exists()
    else Path(__file__).resolve().parent.parent / "data"
)
_CACHE_PATH = _DATA_DIR / "pe_history_cache.json"
_CACHE_TTL_DAYS = 7

# In-process cache hydrated from disk on first call. Map ticker → cache entry.
_cache: dict = {}
_cache_loaded = False


@dataclass
class ValuationBreakdown:
    ticker:                          str
    current_fwd_pe:                  Optional[float]
    pe_hist_p10:                     Optional[float]
    pe_hist_p25:                     Optional[float]
    pe_hist_p50:                     Optional[float]
    pe_hist_p75:                     Optional[float]
    pe_hist_p90:                     Optional[float]
    pe_hist_n_samples:               int
    current_pe_percentile_in_own_3y: Optional[float]  # 0.0 = cheaper than all hist; 1.0 = more expensive
    source:                          str   # "cached" | "live" | "missing"
    fetched_at:                      str   # ISO

    def to_dict(self) -> dict:
        return asdict(self)


def _load_cache() -> None:
    global _cache_loaded
    if _cache_loaded:
        return
    _cache_loaded = True
    if not _CACHE_PATH.exists():
        return
    try:
        raw = json.loads(_CACHE_PATH.read_text())
        if isinstance(raw, dict):
            _cache.update(raw)
            log.info(f"[valuation_pct] loaded {len(_cache)} entries from cache")
    except Exception as e:
        log.warning(f"[valuation_pct] cache read failed: {e}")


def _persist_cache() -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(_cache, indent=2))
    except Exception as e:
        log.warning(f"[valuation_pct] cache write failed: {e}")


def _cache_is_fresh(entry: dict) -> bool:
    """Cache hit when entry is < _CACHE_TTL_DAYS old."""
    fetched_at = entry.get("fetched_at")
    if not fetched_at:
        return False
    try:
        from datetime import datetime
        age_days = (datetime.now() - datetime.fromisoformat(fetched_at)).total_seconds() / 86400
        return age_days < _CACHE_TTL_DAYS
    except Exception:
        return False


def _fetch_pe_history(ticker: str, years: int = 3) -> Optional[list[float]]:
    """Pull 3y monthly close / trailingEps. Returns list of PE samples or None.

    Bypasses market_client.get_historical_pe's sleep(5) — that sleep was
    sized for batch backfills, not live single-name calls.
    """
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        eps = info.get("trailingEps")
        if not eps or eps <= 0:
            return None
        hist = tk.history(period=f"{years}y", interval="1mo", auto_adjust=True)
        if hist is None or hist.empty:
            return None
        closes = hist["Close"].dropna()
        if len(closes) < 6:
            return None
        pe_series = (closes / float(eps)).tolist()
        # Drop non-finite + outlier values (PE > 500 or < 0 = data error)
        pe_clean = [
            float(x) for x in pe_series
            if x is not None and x == x and 0 < x < 500
        ]
        if len(pe_clean) < 4:
            return None
        return pe_clean
    except Exception as e:
        log.debug(f"[valuation_pct] _fetch_pe_history({ticker}): {e}")
        return None


def _percentile_of(value: float, samples: list[float]) -> float:
    """Return what percentile `value` sits in within `samples` (0.0-1.0)."""
    if not samples:
        return 0.5
    arr = np.asarray(samples, dtype=float)
    return float((arr <= value).sum()) / float(len(arr))


def valuation_percentile(
    ticker: str,
    *,
    force_refresh: bool = False,
) -> ValuationBreakdown:
    """Return current forward P/E + own 3y percentile + breakdown.

    Cache hit when on-disk entry is < 7 days old AND `force_refresh=False`.
    Otherwise fetches fresh from yfinance and updates cache.
    Always returns a `ValuationBreakdown` — fields are None when data
    isn't available so the caller can degrade gracefully.
    """
    ticker = (ticker or "").upper().strip()
    from datetime import datetime
    now_iso = datetime.now().isoformat(timespec="seconds")

    _load_cache()

    cached = _cache.get(ticker)
    if cached and not force_refresh and _cache_is_fresh(cached):
        # Refresh only `current_fwd_pe` from live, reuse cached percentiles
        try:
            import yfinance as yf
            live_info = yf.Ticker(ticker).info or {}
            live_fwd_pe = live_info.get("forwardPE")
            if live_fwd_pe is not None and live_fwd_pe > 0:
                cached_samples = cached.get("pe_samples") or []
                if cached_samples:
                    pct = _percentile_of(float(live_fwd_pe), cached_samples)
                    cached["current_fwd_pe"] = float(live_fwd_pe)
                    cached["current_pe_percentile_in_own_3y"] = pct
                    cached["source"] = "cached"
                    return _entry_to_breakdown(ticker, cached)
        except Exception as e:
            log.debug(f"[valuation_pct] live fwd_pe refresh({ticker}): {e}")
        # Fallback: serve cached as-is
        cached["source"] = "cached"
        return _entry_to_breakdown(ticker, cached)

    # Cache miss or expired → fetch fresh history
    pe_samples = _fetch_pe_history(ticker)
    if pe_samples is None:
        return ValuationBreakdown(
            ticker=ticker,
            current_fwd_pe=None,
            pe_hist_p10=None, pe_hist_p25=None, pe_hist_p50=None,
            pe_hist_p75=None, pe_hist_p90=None,
            pe_hist_n_samples=0,
            current_pe_percentile_in_own_3y=None,
            source="missing",
            fetched_at=now_iso,
        )

    arr = np.asarray(pe_samples, dtype=float)
    p10, p25, p50, p75, p90 = (
        float(np.quantile(arr, q)) for q in (0.10, 0.25, 0.50, 0.75, 0.90)
    )

    # Pull current forward PE from live
    current_fwd_pe = None
    try:
        import yfinance as yf
        live_info = yf.Ticker(ticker).info or {}
        live_fwd_pe = live_info.get("forwardPE")
        if live_fwd_pe is not None and live_fwd_pe > 0:
            current_fwd_pe = float(live_fwd_pe)
    except Exception as e:
        log.debug(f"[valuation_pct] current_fwd_pe({ticker}): {e}")

    pct = (
        _percentile_of(current_fwd_pe, pe_samples)
        if current_fwd_pe is not None else None
    )

    entry = {
        "ticker":                          ticker,
        "current_fwd_pe":                  current_fwd_pe,
        "pe_hist_p10":                     p10,
        "pe_hist_p25":                     p25,
        "pe_hist_p50":                     p50,
        "pe_hist_p75":                     p75,
        "pe_hist_p90":                     p90,
        "pe_hist_n_samples":               len(pe_samples),
        "current_pe_percentile_in_own_3y": pct,
        "source":                          "live",
        "fetched_at":                      now_iso,
        "pe_samples":                      pe_samples,  # kept in cache for live re-percentile
    }
    _cache[ticker] = entry
    _persist_cache()
    return _entry_to_breakdown(ticker, entry)


def _entry_to_breakdown(ticker: str, entry: dict) -> ValuationBreakdown:
    return ValuationBreakdown(
        ticker=ticker,
        current_fwd_pe=entry.get("current_fwd_pe"),
        pe_hist_p10=entry.get("pe_hist_p10"),
        pe_hist_p25=entry.get("pe_hist_p25"),
        pe_hist_p50=entry.get("pe_hist_p50"),
        pe_hist_p75=entry.get("pe_hist_p75"),
        pe_hist_p90=entry.get("pe_hist_p90"),
        pe_hist_n_samples=entry.get("pe_hist_n_samples", 0),
        current_pe_percentile_in_own_3y=entry.get("current_pe_percentile_in_own_3y"),
        source=entry.get("source", "cached"),
        fetched_at=entry.get("fetched_at", ""),
    )


def valuation_subscore(
    breakdown: ValuationBreakdown,
) -> Optional[float]:
    """Convert a ValuationBreakdown into sub-score #5's contribution.

    Plan B formula:
        score = 1 - clip(current_fwd_pe / pe_hist_p25, 0.3, 2.0)
        hard_floor -1 if current_fwd_pe > pe_hist_p75 * 1.2

    Range: [-1, +1]. Positive = cheap relative to own history. Negative =
    expensive (the "still expensive after drop" hard penalty).
    Returns None when data is missing — caller treats as neutral 0 + flag.
    """
    if breakdown.current_fwd_pe is None or breakdown.pe_hist_p25 is None:
        return None
    if breakdown.pe_hist_p25 <= 0:
        return None
    ratio = breakdown.current_fwd_pe / breakdown.pe_hist_p25
    score = 1.0 - float(np.clip(ratio, 0.3, 2.0))

    # Hard floor: still expensive vs own P75
    if (breakdown.pe_hist_p75 is not None
            and breakdown.current_fwd_pe > breakdown.pe_hist_p75 * 1.2):
        return -1.0

    return float(np.clip(score, -1.0, 1.0))


def clear_cache() -> None:
    """Wipe in-process cache. Disk cache is separate; use force_refresh=True
    on per-call basis when you need fresh data."""
    _cache.clear()
    global _cache_loaded
    _cache_loaded = False
