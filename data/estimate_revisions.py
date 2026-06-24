"""data/estimate_revisions.py — Sub-score #1 source: revisions direction.

Wraps `fmp_client.consensus_eps_delta_pct` (primary) with a proxy
fallback (`earnings_beat_rate` from yfinance via `fundamental_signals`)
when FMP is unavailable: missing API key, quota exhausted, network
error, ticker not covered.

Returns a structured result so /analisar can show the data source
transparently:

    {
      "score": -1.0..+1.0,
      "source": "fmp_primary" | "proxy_degraded" | "missing",
      "raw_eps_delta_pct": float | None,
      "raw_beat_rate":     float | None,
      "explanation":       str,
    }

Per the validation plan (G.1): when source = "proxy_degraded", the
dislocation scorer scales this signal's contribution by 0.5 in the
composite. The honest proxy signal magnitude is also capped at ±0.5
(half what FMP can produce) to reflect the weaker confidence.

Why the proxy is degraded: `earnings_beat_rate` measures management
credibility (history of meeting/beating consensus), NOT the DIRECTION
of recent consensus revisions. They correlate weakly. A name with
strong beat history but currently being cut by analysts will look fine
in the proxy and bad in the FMP primary. The plan acknowledges this
in G.1.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Optional

log = logging.getLogger("estimate_revisions")


@dataclass
class EstimateRevisions:
    ticker:             str
    score:              float  # in [-1, +1]
    source:             str    # "fmp_primary" | "proxy_degraded" | "missing"
    raw_eps_delta_pct:  Optional[float]
    raw_beat_rate:      Optional[float]
    explanation:        str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def degraded(self) -> bool:
        return self.source != "fmp_primary"


def _normalize_fmp_delta(eps_delta_pct: float) -> float:
    """Convert FMP eps delta % to a [-1, +1] score.

    Uses tanh with k=20 so:
       +5% EPS delta → +0.76 (clear positive)
      +10%          → +0.96 (saturating strong)
        0%          →  0.00
       -5%          → -0.76
      -10%          → -0.96
    Caps naturally; no clipping required.
    """
    import math
    return math.tanh(float(eps_delta_pct) * 20.0)


def _proxy_from_beat_rate(beat_rate: float) -> float:
    """Convert earnings_beat_rate (0..1) into degraded proxy score.

    beat_rate 0.50 = neutral (50% beat rate is random-baseline guess).
    beat_rate 1.00 = always beats → strongest positive proxy
    beat_rate 0.00 = always misses → strongest negative proxy

    Returns capped to [-0.5, +0.5] — half the magnitude of an FMP-primary
    signal — to reflect that beat history is a noisy proxy for revision
    direction. The dislocation scorer further scales contribution by 0.5
    when source = proxy_degraded (per plan G.1), so a beat_rate of 1.00
    contributes at most ±0.25 of weighted composite.
    """
    centered = float(beat_rate) - 0.50  # [-0.5, +0.5]
    return max(-0.5, min(0.5, centered))


def estimate_revisions_for_ticker(ticker: str) -> EstimateRevisions:
    """Resolve sub-score #1 for a ticker. Always returns a result.

    Order of attempts:
      1. FMP primary via consensus_eps_delta_pct(ticker, 35d) → "fmp_primary"
      2. Proxy via earnings_beat_rate from yfinance → "proxy_degraded"
      3. Both missing → score 0.0, "missing"
    """
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return EstimateRevisions(
            ticker="", score=0.0, source="missing",
            raw_eps_delta_pct=None, raw_beat_rate=None,
            explanation="empty ticker",
        )

    # --- Primary: FMP consensus EPS delta -----------------------------
    try:
        from data.fmp_client import consensus_eps_delta_pct
        eps_delta = consensus_eps_delta_pct(ticker, lookback_days=35)
    except Exception as e:
        log.debug(f"[est_rev] FMP call({ticker}): {e}")
        eps_delta = None

    if eps_delta is not None:
        score = _normalize_fmp_delta(eps_delta)
        direction = (
            "rising" if eps_delta > 0.01 else
            "falling" if eps_delta < -0.01 else
            "flat"
        )
        return EstimateRevisions(
            ticker=ticker,
            score=score,
            source="fmp_primary",
            raw_eps_delta_pct=eps_delta,
            raw_beat_rate=None,
            explanation=f"FMP consensus EPS {direction} {eps_delta*100:+.1f}% over 35d",
        )

    # --- Fallback: yfinance earnings_beat_rate proxy ------------------
    beat_rate = None
    try:
        from data.fundamental_signals import yf_fundamental_signals
        sigs = yf_fundamental_signals(ticker) or {}
        beat_rate = sigs.get("earnings_beat_rate")
        if beat_rate is None or (isinstance(beat_rate, float) and beat_rate != beat_rate):
            beat_rate = None
    except Exception as e:
        log.debug(f"[est_rev] proxy beat_rate({ticker}): {e}")
        beat_rate = None

    if beat_rate is not None:
        proxy_score = _proxy_from_beat_rate(float(beat_rate))
        return EstimateRevisions(
            ticker=ticker,
            score=proxy_score,
            source="proxy_degraded",
            raw_eps_delta_pct=None,
            raw_beat_rate=float(beat_rate),
            explanation=(
                f"FMP unavailable — proxy from beat_rate {beat_rate*100:.0f}% "
                f"(weak signal of revision direction)"
            ),
        )

    # --- Both missing → neutral with honest flag ----------------------
    return EstimateRevisions(
        ticker=ticker,
        score=0.0,
        source="missing",
        raw_eps_delta_pct=None,
        raw_beat_rate=None,
        explanation="no FMP coverage and no beat_rate proxy available",
    )
