"""
http_timeout.py — Force a default timeout on every requests.Session call.

Why this exists
===============
yfinance, Tiingo, and the EDGAR helpers all use `requests` under the hood.
EDGAR helpers (data/fundamental_history.py) pass `timeout=` explicitly,
so they're safe. yfinance does NOT — its internal `requests.get` calls
have no timeout, so a stalled socket (DNS hang, slow TLS handshake,
yfinance API holding a connection open) blocks the calling thread
indefinitely.

On 2026-06-12 the CORE build hung 17 minutes after "Preços carregados",
with zero progress on a 237-ticker run. The orchestrator's
ThreadPoolExecutor.as_completed waited forever because the underlying
yfinance call inside _process_one_ticker never returned. /admin_kill_pipeline
was the only escape.

This module monkey-patches `requests.Session.request` to inject a default
`timeout` if the caller did not supply one. Callers passing `timeout=N`
explicitly still win. Install once on import — every module that imports
this gets the protection process-wide.

Tuning
======
Env var `DIPRADAR_HTTP_TIMEOUT` overrides the default (15 seconds). Set
higher for slow networks; lower if you want failures to surface faster.
"""
from __future__ import annotations

import logging
import os

_INSTALLED = False
log = logging.getLogger(__name__)


def install(timeout_seconds: float = 15.0) -> None:
    """Patch requests.Session.request once. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        import requests
    except ImportError:
        # requests not installed — nothing to protect, nothing to patch.
        return

    _orig = requests.Session.request

    def _patched(self, method, url, *args, **kwargs):
        kwargs.setdefault("timeout", timeout_seconds)
        return _orig(self, method, url, *args, **kwargs)

    requests.Session.request = _patched
    _INSTALLED = True
    log.info(
        f"[http_timeout] default {timeout_seconds}s timeout installed "
        f"on requests.Session.request"
    )


# Install on import — every build script that pulls this in gets coverage
# without having to remember to call install() in main().
install(float(os.environ.get("DIPRADAR_HTTP_TIMEOUT", "15")))
