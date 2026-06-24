"""core/scoring_dispatch.py — Dispatcher between dislocation_z and composite_z.

Single point of dispatch for every scoring caller during the F.6
side-by-side transition window. Reads `DIPRADAR_USE_DISLOCATION` env
(default "1" = dislocation) ONCE per scoring session and routes all
per-ticker calls to the same scorer for that session.

Why a ScoringSession class instead of a free function:
  - The flag MUST be observed at one point per scan, not per ticker.
    If the flag flipped mid-scan (env reload, race, manual edit), a
    free function reading os.environ each call would produce a mixed
    output that's harder to audit.
  - The plan F.6 protocol relies on knowing exactly which scorer
    produced each /cash, /scan_now, /analisar, watchdog notification.
    Every ScoreResult is tagged with .scorer ("dislocation"|"composite")
    so the 2-week log review (per plan F.6) is unambiguous.
  - Universe stats are cached on the session, so a CORE-universe sweep
    of 230 tickers hits disk once, not 230 times.

Flag semantics:
  DIPRADAR_USE_DISLOCATION=1 (default)  → dislocation_z (new rule-based)
  DIPRADAR_USE_DISLOCATION=0            → composite_z   (legacy heuristic)

When the F.6 side-by-side passes, F.7 deletes the composite path and
this dispatcher reduces to a thin shim. F.8 deletes the dispatcher too.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Union

log = logging.getLogger("scoring_dispatch")

ScorerName = str  # "dislocation" | "composite" — Literal omitted for py<3.10 compat


@dataclass
class ScoreResult:
    """Unified scoring result across both scorers.

    Callers should rely on .scorer + .z + .hard_excluded +
    .explanation_lines() rather than reaching into .breakdown directly,
    so the same display code works for both engines during F.6 and the
    composite path can be cleanly removed in F.7.
    """
    ticker:         str
    scorer:         ScorerName
    z:              Optional[float]
    hard_excluded:  bool
    exclude_reason: Optional[str]
    summary_line:   str
    breakdown:      object  # CompositeBreakdown | DislocationBreakdown

    def explanation_lines(self, threshold_abs: float = 0.3) -> list[str]:
        """Multi-line audit dump for /analisar appendix and /scan_now."""
        if self.scorer == "dislocation":
            lines = [self.summary_line]
            for s in self.breakdown.sub_scores.values():
                if abs(s.value) >= threshold_abs:
                    weighted = s.value * s.weight
                    deg = " ⚠️degraded" if s.degraded else ""
                    lines.append(
                        f"  • {s.name}: raw {s.value:+.2f} × w {s.weight:.1f} "
                        f"= {weighted:+.2f}  ({s.explanation}){deg}"
                    )
            return lines
        # composite path
        z_str = f"{self.z:+.2f}z" if self.z is not None else "?z"
        lines = [f"{self.ticker}  {z_str}"]
        b = self.breakdown
        if hasattr(b, "quality_z"):
            lines.append(f"  • quality_z:   {b.quality_z:+.2f}")
        if hasattr(b, "dd_z"):
            lines.append(f"  • dd_z:        {b.dd_z:+.2f}")
        if hasattr(b, "valuation_z"):
            lines.append(f"  • valuation_z: {b.valuation_z:+.2f}")
        return lines


def get_active_scorer() -> ScorerName:
    """Return "dislocation" or "composite" based on env. Default dislocation."""
    val = os.environ.get("DIPRADAR_USE_DISLOCATION", "1").strip()
    if val in ("0", "false", "False", "FALSE"):
        return "composite"
    return "dislocation"


class ScoringSession:
    """One-flag-read-per-scan scorer with shared universe stats cache.

    Usage:
        session = ScoringSession()              # logs scorer name once
        for ticker, signals in candidates:
            r = session.score(ticker, signals)  # uses session's frozen scorer
            if not r.hard_excluded:
                rank_by_z(r.z)

    Override the auto-detected scorer at construction (testing /
    validation harness):
        session = ScoringSession(scorer="dislocation")
        session = ScoringSession(scorer="composite")

    `variant` controls the dislocation scorer's gate variant (a = full,
    b = signal #1 zeroed for PIT-clean BINDING gate). Only used when
    self.scorer == "dislocation".

    `strict_pit` ONLY applies to the dislocation path. The backtest
    harness sets it True so that a forgotten as_of_date raises instead
    of silently using all-data ("now").
    """

    def __init__(
        self,
        *,
        scorer: Optional[ScorerName] = None,
        variant: str = "a",
        strict_pit: bool = False,
    ):
        self.scorer: ScorerName = scorer or get_active_scorer()
        self.variant = variant
        self.strict_pit = strict_pit
        self._stats: Optional[dict] = None
        log.info(
            f"[score] session start  scorer={self.scorer}  "
            f"variant={self.variant}  strict_pit={self.strict_pit}"
        )

    def _load_stats(self) -> dict:
        if self._stats is not None:
            return self._stats
        if self.scorer == "dislocation":
            from core.dislocation_score import build_dislocation_stats
            self._stats = build_dislocation_stats() or {}
        else:
            from core.composite_score import build_universe_stats
            self._stats = build_universe_stats() or {}
        return self._stats

    def score(
        self,
        ticker: str,
        signals: dict,
        *,
        as_of_date: Optional[date] = None,
        horizon: str = "CORE",
    ) -> ScoreResult:
        """Score a single ticker via the session's frozen scorer.

        `as_of_date`: PIT date (validation harness). Production callers
        pass None.
        `horizon`: composite-only ("CORE" or "DIP"). Ignored by dislocation.
        """
        stats = self._load_stats()

        if self.scorer == "dislocation":
            from core.dislocation_score import dislocation_for_ticker
            b = dislocation_for_ticker(
                ticker, signals,
                as_of_date=as_of_date,
                variant=self.variant,
                strict_pit=self.strict_pit,
                stats=stats,
            )
            return ScoreResult(
                ticker=ticker,
                scorer="dislocation",
                z=b.composite_z,
                hard_excluded=b.hard_excluded,
                exclude_reason=b.exclude_reason,
                summary_line=b.summary_line,
                breakdown=b,
            )

        # composite legacy path
        from core.composite_score import composite_from_signals
        b = composite_from_signals(ticker, signals, horizon=horizon, stats=stats)
        return ScoreResult(
            ticker=ticker,
            scorer="composite",
            z=b.composite_z,
            hard_excluded=False,
            exclude_reason=None,
            summary_line=f"{ticker}  {b.composite_z:+.2f}z (composite legacy)",
            breakdown=b,
        )
