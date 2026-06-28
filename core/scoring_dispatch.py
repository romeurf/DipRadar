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
    """Unified scoring result across all three scorers.

    Callers should rely on .scorer + .z + .qualifies + .hard_excluded +
    .explanation_lines() rather than reaching into .breakdown directly.

    Semantics by scorer:
      watchlist:   z=None (no ranking by edge); qualifies tells you if
                   it's in the list; sort callers by entry.dd_from_52w_high
      composite:   z=composite_z (legacy heuristic); qualifies=z>=threshold
      dislocation: z=composite_z (experimental); qualifies=z>=threshold
    """
    ticker:         str
    scorer:         ScorerName
    z:              Optional[float]
    qualifies:      bool             # True if name passes the scorer's bar
    hard_excluded:  bool
    exclude_reason: Optional[str]
    summary_line:   str
    breakdown:      object           # CompositeBreakdown | DislocationBreakdown | WatchlistEntry
    # Optional secondary sort key for watchlist mode (dd_from_52w_high, neg = deeper)
    sort_key:       Optional[float] = None

    def explanation_lines(self, threshold_abs: float = 0.3) -> list[str]:
        """Multi-line audit dump for /analisar appendix and /scan_now."""
        if self.scorer == "watchlist":
            # WatchlistEntry has pass/fail per criterion + display fields
            b = self.breakdown
            lines = [self.summary_line]
            if not b.quality_passes:
                lines.append(f"  ❌ quality gate: {b.quality_reason}")
            if b.dd_from_52w_high is None or b.dd_from_52w_high > -0.25:
                dd_str = f"{b.dd_from_52w_high*100:+.0f}%" if b.dd_from_52w_high is not None else "n/a"
                lines.append(f"  ❌ DD: {dd_str} (need <= -25%)")
            if b.days_to_catalyst is None or not (-10 <= b.days_to_catalyst <= 30):
                cat = b.days_to_catalyst
                cat_str = f"{cat}d" if cat is not None else "n/a"
                lines.append(f"  ❌ catalyst: {cat_str} (need within -10d to +30d)")
            if b.sector_relative_60d is not None:
                lines.append(
                    f"  ℹ️ vs sector 60d: {b.sector_relative_60d*100:+.0f}pp "
                    f"({'fell with market' if abs(b.sector_relative_60d) < 0.05 else 'specific move'})"
                )
            return lines

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
    """Return active scorer based on env. Default = watchlist.

    Reads `DIPRADAR_SCORING_MODE` (preferred). Accepts:
        "watchlist"   — honest filter (default ship after F.5 REVERT)
        "composite"   — legacy heuristic z-score
        "dislocation" — experimental rule-based (failed F.5 binding gate)

    Backward-compat: reads legacy `DIPRADAR_USE_DISLOCATION` when
    `DIPRADAR_SCORING_MODE` is unset:
        DIPRADAR_USE_DISLOCATION=0 -> composite
        DIPRADAR_USE_DISLOCATION=1 -> dislocation
    The legacy flag will be deprecated in a future release; once both
    are unset, default is "watchlist".
    """
    explicit = os.environ.get("DIPRADAR_SCORING_MODE", "").strip().lower()
    if explicit in ("watchlist", "composite", "dislocation"):
        return explicit

    # Backward-compat: legacy flag if no explicit mode set
    legacy = os.environ.get("DIPRADAR_USE_DISLOCATION", "").strip()
    if legacy == "1":
        return "dislocation"
    if legacy in ("0", "false", "False", "FALSE"):
        return "composite"

    # Default after F.5 REVERT: watchlist is the honest ship
    return "watchlist"


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
        if self.scorer == "watchlist":
            # Watchlist doesn't compute against a stats distribution;
            # criteria are absolute thresholds. Empty dict keeps the
            # interface uniform without an unnecessary universe walk.
            self._stats = {}
        elif self.scorer == "dislocation":
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

        `as_of_date`: PIT date (validation harness; dislocation only).
        `horizon`: composite-only ("CORE" or "DIP"). Ignored elsewhere.
        """
        stats = self._load_stats()

        if self.scorer == "watchlist":
            from core.honest_watchlist import evaluate_ticker
            entry = evaluate_ticker(ticker, signals)
            return ScoreResult(
                ticker=ticker,
                scorer="watchlist",
                z=None,  # watchlist explicitly does NOT rank by z
                qualifies=entry.qualifies,
                hard_excluded=not entry.quality_passes,
                exclude_reason=entry.quality_reason if not entry.quality_passes else None,
                summary_line=entry.summary_line,
                breakdown=entry,
                # Sort key for ordering: deepest DD first (most negative)
                sort_key=entry.dd_from_52w_high,
            )

        if self.scorer == "dislocation":
            from core.dislocation_score import dislocation_for_ticker
            b = dislocation_for_ticker(
                ticker, signals,
                as_of_date=as_of_date,
                variant=self.variant,
                strict_pit=self.strict_pit,
                stats=stats,
            )
            # Qualifies threshold for dislocation: z >= 0.3 (same as
            # backtest top-decile-ish cutoff)
            qualifies = (b.composite_z is not None
                         and b.composite_z >= 0.3
                         and not b.hard_excluded)
            return ScoreResult(
                ticker=ticker,
                scorer="dislocation",
                z=b.composite_z,
                qualifies=qualifies,
                hard_excluded=b.hard_excluded,
                exclude_reason=b.exclude_reason,
                summary_line=b.summary_line,
                breakdown=b,
                sort_key=None,
            )

        # composite legacy path
        from core.composite_score import composite_from_signals
        b = composite_from_signals(ticker, signals, horizon=horizon, stats=stats)
        qualifies = b.composite_z is not None and b.composite_z >= 0.3
        return ScoreResult(
            ticker=ticker,
            scorer="composite",
            z=b.composite_z,
            qualifies=qualifies,
            hard_excluded=False,
            exclude_reason=None,
            summary_line=f"{ticker}  {b.composite_z:+.2f}z (composite legacy)",
            breakdown=b,
            sort_key=None,
        )
