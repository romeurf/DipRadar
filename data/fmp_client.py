"""data/fmp_client.py — Financial Modeling Prep free tier client.

Used by the dislocation scorer for sub-score #1 (estimate revisions
direction). FMP free tier exposes consensus EPS/revenue estimate
history via /v3/analyst-estimates/{ticker}, which lets us compute the
direction of consensus over the last ~30 days — the user's PRIMARY
signal for "fundamentals holding vs deteriorating".

API key reads from `FMP_API_KEY` env var (Railway). When unset, every
call returns None → caller degrades to proxy mode (see
`data/estimate_revisions.py`).

Daily quota: free tier is 250 calls/day shared bucket. A CORE universe
sweep of ~230 tickers exhausts ~92% in one pass. Quota state tracked
in /data/fmp_quota.json with daily reset at midnight UTC (FMP's reset
boundary). When >= 240/250, subsequent calls short-circuit to None.

The endpoint can change without notice (FMP free tier has shifted
twice in 2024-2026). The client is defensive: any non-200, exception,
or empty response → returns None. Caller treats None as "data not
available, fall back".
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("fmp")

_BASE_URL          = "https://financialmodelingprep.com/api/v3"
_DAILY_QUOTA       = 250
_QUOTA_SOFT_LIMIT  = 240  # short-circuit after this to leave buffer for /analisar etc.
_DEFAULT_TIMEOUT_S = 15.0

_DATA_DIR   = (
    Path("/data") if Path("/data").exists()
    else Path(__file__).resolve().parent.parent / "data"
)
_QUOTA_PATH = _DATA_DIR / "fmp_quota.json"

# In-process quota state; hydrated from disk on first call.
_quota_state: dict = {}


# ── API key ────────────────────────────────────────────────────────────

def get_api_key() -> Optional[str]:
    """Return FMP_API_KEY from env, or None when not set.

    Used to gate calls — when None, no HTTP is issued and the caller
    degrades to proxy mode.
    """
    key = os.environ.get("FMP_API_KEY")
    if key and key.strip():
        return key.strip()
    return None


# ── Quota tracking ─────────────────────────────────────────────────────

def _load_quota_state() -> None:
    """Hydrate the in-process quota state from /data/fmp_quota.json.

    Resets the counter when the persisted date != today UTC (FMP's
    reset is UTC midnight regardless of local timezone).
    """
    global _quota_state
    today = datetime.now(timezone.utc).date().isoformat()
    if _quota_state.get("date") == today and "calls" in _quota_state:
        return
    if _QUOTA_PATH.exists():
        try:
            on_disk = json.loads(_QUOTA_PATH.read_text())
            if on_disk.get("date") == today:
                _quota_state = on_disk
                return
        except Exception as e:
            log.debug(f"[fmp] quota read failed: {e}")
    _quota_state = {"date": today, "calls": 0}
    _persist_quota_state()


def _persist_quota_state() -> None:
    try:
        _QUOTA_PATH.parent.mkdir(parents=True, exist_ok=True)
        _QUOTA_PATH.write_text(json.dumps(_quota_state, indent=2))
    except Exception as e:
        log.debug(f"[fmp] quota write failed: {e}")


def get_quota_status() -> dict:
    """Return current quota state. Useful for /admin telemetry."""
    _load_quota_state()
    used = int(_quota_state.get("calls", 0))
    return {
        "date":            _quota_state.get("date"),
        "calls_used":      used,
        "calls_remaining": max(0, _DAILY_QUOTA - used),
        "soft_limit":      _QUOTA_SOFT_LIMIT,
        "can_call":        used < _QUOTA_SOFT_LIMIT,
    }


def _can_call() -> bool:
    _load_quota_state()
    return int(_quota_state.get("calls", 0)) < _QUOTA_SOFT_LIMIT


def _increment_quota() -> None:
    _load_quota_state()
    _quota_state["calls"] = int(_quota_state.get("calls", 0)) + 1
    _persist_quota_state()


# ── HTTP layer ─────────────────────────────────────────────────────────

def _http_get(endpoint: str, params: Optional[dict] = None) -> Optional[object]:
    """GET an FMP endpoint with quota + auth handled.

    Returns parsed JSON (dict or list) on success, None on:
      - no API key
      - quota exhausted
      - HTTP error (4xx/5xx)
      - network/parsing exception
    Always safe to call; caller treats None as "no data, fall back".
    """
    api_key = get_api_key()
    if not api_key:
        log.debug(f"[fmp] FMP_API_KEY not set — skipping {endpoint}")
        return None

    if not _can_call():
        log.info(
            f"[fmp] quota soft-limit reached "
            f"({_quota_state.get('calls')}/{_DAILY_QUOTA}) — skipping {endpoint}"
        )
        return None

    import requests
    url = f"{_BASE_URL}{endpoint}"
    p = dict(params or {})
    p["apikey"] = api_key
    try:
        resp = requests.get(url, params=p, timeout=_DEFAULT_TIMEOUT_S)
    except Exception as e:
        log.debug(f"[fmp] {endpoint} network: {e}")
        return None

    # Count the call regardless of outcome — FMP counts attempts, not successes.
    _increment_quota()

    if resp.status_code == 429:
        log.warning(f"[fmp] 429 rate-limit on {endpoint}")
        return None
    if resp.status_code == 401 or resp.status_code == 403:
        log.warning(
            f"[fmp] {resp.status_code} auth/access denied on {endpoint} — "
            f"check FMP_API_KEY validity / tier"
        )
        return None
    if resp.status_code != 200:
        log.debug(f"[fmp] {endpoint} HTTP {resp.status_code}")
        return None

    try:
        return resp.json()
    except Exception as e:
        log.debug(f"[fmp] {endpoint} JSON parse: {e}")
        return None


# ── Endpoint: analyst estimates ────────────────────────────────────────

def get_analyst_estimates(ticker: str) -> Optional[list]:
    """Return annual consensus EPS/revenue estimates history.

    Free tier endpoint: /v3/analyst-estimates/{ticker}
    Returns a list of annual snapshots ordered newest-first. Each entry:
      { "date": "YYYY-MM-DD", "symbol": "...",
        "estimatedRevenueAvg": ..., "estimatedEpsAvg": ...,
        "numberAnalystEstimatedEps": ..., ... }

    None when the ticker is unsupported, key invalid, quota out, or
    response is empty. Caller (estimate_revisions.consensus_delta_30d)
    handles the missing-data degradation.
    """
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return None
    data = _http_get(f"/analyst-estimates/{ticker}")
    if not isinstance(data, list) or len(data) == 0:
        return None
    return data


def consensus_eps_delta_pct(ticker: str, lookback_days: int = 35) -> Optional[float]:
    """Return % change in next-FY consensus EPS over the last ~lookback_days.

    Implementation: the FMP /analyst-estimates endpoint returns annual
    snapshots, not daily revisions. We approximate "delta over 30d" by
    comparing the MOST RECENT snapshot for the next fiscal year with
    the SAME fiscal-year estimate from the snapshot taken closest to
    `lookback_days` ago. When there isn't a snapshot that old, returns
    None (data too thin to compute delta).

    Sign: positive = consensus EPS rising (good).
          negative = consensus EPS falling (bad, value-trap signal).

    Returns None on any data issue. Caller treats None as proxy-mode.
    """
    estimates = get_analyst_estimates(ticker)
    if not estimates or len(estimates) < 2:
        return None

    from datetime import datetime as _dt
    # Parse + sort newest-first
    parsed = []
    for e in estimates:
        try:
            d = _dt.strptime(e.get("date", ""), "%Y-%m-%d").date()
            eps = e.get("estimatedEpsAvg")
            if eps is None or float(eps) <= 0:
                continue
            parsed.append((d, float(eps)))
        except Exception:
            continue

    if len(parsed) < 2:
        return None

    parsed.sort(key=lambda x: x[0], reverse=True)
    latest_date, latest_eps = parsed[0]

    # Find the closest older snapshot to (latest - lookback_days)
    from datetime import timedelta
    target_date = latest_date - timedelta(days=lookback_days)
    best = None
    best_gap = None
    for d, eps in parsed[1:]:
        gap = abs((d - target_date).days)
        if best is None or gap < best_gap:
            best = (d, eps)
            best_gap = gap

    if best is None or best[1] <= 0:
        return None

    delta_pct = (latest_eps / best[1]) - 1.0
    return float(delta_pct)
