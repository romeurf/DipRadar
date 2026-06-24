"""core/dislocation_score.py — Master rule-based dislocation scorer.

Replaces the ML alpha-rank pipeline with a transparent weighted sum
of 7 named sub-scores. Each candidate name produces a
DislocationBreakdown the user can read and audit ("cheap P/E p15 +
estimates holding + sector_drop -28% + catalyst in 24d").

Sub-scores (see plan §B):
  #1 Estimate revisions      (PRIMARY thesis-wise; demoted weight 1.0)
  #2 Guidance change         (8-K Item 2.02 + news keywords)
  #3 Drop attribution        (sector-relative + bad-news subtraction,
                              with relief-keyword context guard)
  #4 Catalyst proximity      (triangular around 21-42d earnings window)
  #5 Absolute valuation      (P/E vs own 3y percentile + hard floor)
  #6 Quality                 (hard gate + graded composite)
  #7 Governance              (insider sales, 8-K restatements, mgmt depart)

Composition (plan §C):
  raw = 1.0·#1 + 1.0·#2 + 1.3·#3 + 0.6·#4 + 1.0·#5 + 0.8·#6 + 1.0·#7
  z   = (raw - μ_universe) / σ_universe   ← against dislocation_stats.json

Hard exclusion gates fire BEFORE scoring:
  - quality gate fail (rev/margin/FCF — same gates as core_radar.py)
  - governance == -2 (capped; means multiple severe red flags)

PIT (point-in-time) discipline is BLOCKING:

  All sub-scorers that touch news headlines, SEC filings, or insider
  transactions accept an `as_of_date` parameter. When None, current
  behavior (production = "now"). When set, ONLY data with publish/
  filing_date <= as_of_date is considered. This converts the plan's
  ±15d windows into "15d backwards only" automatically.

  When `strict_pit=True` (backtest mode), `as_of_date=None` raises
  ValueError — the failure mode of silently running with no PIT
  filter (forgotten parameter) is exactly how the original ML
  pipeline let look-ahead leak in. The strict_pit gate ensures
  validation can NEVER run a fold without an explicit as_of_date.

Variant (b) for the binding GO gate (plan §E):
  When `variant="b"`, sub-score #1's weight is forced to 0. This is
  the PIT-clean fallback — the gate must pass on variant (b) for the
  scorer to ship, defending against the ML-style failure where a
  signal with non-PIT data flatters the backtest. See plan §E
  "Reading the results" for interpretation.

Universe stats:
  Z-scoring uses /data/dislocation_stats.json built by walking the
  CORE universe once per day (analogous to composite_score.build_universe_stats).
  Built lazily on first call if missing; daily TTL.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("dislocation")

_DATA_DIR = (
    Path("/data") if Path("/data").exists()
    else Path(__file__).resolve().parent.parent / "data"
)
_STATS_PATH = _DATA_DIR / "dislocation_stats.json"
_STATS_TTL_HOURS = 24

# Weights from plan §C. Overridable via env var DIPRADAR_DISLOCATION_WEIGHTS_JSON.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "estimate_revisions":  1.0,
    "guidance":            1.0,
    "drop_attribution":    1.3,
    "catalyst":            0.6,
    "valuation":           1.0,
    "quality":             0.8,
    "governance":          1.0,
}


def _load_weights() -> dict[str, float]:
    """Load weights from DIPRADAR_DISLOCATION_WEIGHTS_JSON env or default."""
    raw = os.environ.get("DIPRADAR_DISLOCATION_WEIGHTS_JSON")
    if not raw:
        return dict(_DEFAULT_WEIGHTS)
    try:
        overrides = json.loads(raw)
        merged = dict(_DEFAULT_WEIGHTS)
        merged.update({k: float(v) for k, v in overrides.items() if k in _DEFAULT_WEIGHTS})
        return merged
    except Exception as e:
        log.warning(f"[dislocation] invalid DIPRADAR_DISLOCATION_WEIGHTS_JSON: {e}")
        return dict(_DEFAULT_WEIGHTS)


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class SubScore:
    name:        str            # "estimate_revisions", etc.
    value:       float          # in component-specific range (see plan §B)
    weight:      float          # weight used in this composition (variant b zeros #1)
    raw_inputs:  dict           # human-inspectable inputs
    source:      str            # "fmp_primary" / "live" / "8k_filing" / "missing" / ...
    explanation: str
    degraded:    bool = False   # True when proxy/missing — caller may scale weight


@dataclass
class DislocationBreakdown:
    ticker:          str
    as_of_date:      Optional[str]   # ISO; None = production "now"
    variant:         str             # "a" (full) | "b" (signal #1 zeroed for PIT gate)
    hard_excluded:   bool
    exclude_reason:  Optional[str]
    sub_scores:      dict[str, SubScore]
    composite_raw:   float
    composite_z:     Optional[float]  # None when universe stats unavailable
    summary_line:    str             # 1-line explanation built from sub-scores >= |0.5|

    def to_dict(self) -> dict:
        d = asdict(self)
        # asdict serializes nested dataclasses → SubScore dicts already serialize
        return d


# ── Hard gates ────────────────────────────────────────────────────────

def _passes_quality_gate(signals: dict) -> tuple[bool, str]:
    """Quality hard gate (mirrors core_radar.py:594-597).

    Excludes:
      - revenue_growth < -10% y/y
      - gross_margin   < 25%
      - FCF yield < 0 AND not investment-phase

    Missing values default to PASS (we can't exclude on missing data).
    """
    rg = signals.get("revenue_growth")
    if rg is not None and rg < -0.10:
        return False, f"revenue_growth {rg*100:+.0f}% < -10%"

    gm = signals.get("gross_margin")
    if gm is not None and gm < 0.25:
        return False, f"gross_margin {gm*100:.0f}% < 25%"

    fcf = signals.get("fcf_yield_fund") or signals.get("fcf_yield")
    if fcf is not None and fcf < 0:
        if not signals.get("is_investment_phase"):
            return False, f"fcf_yield {fcf*100:+.1f}% < 0 (not investment phase)"

    return True, ""


# ── Sub-score #1 — estimate revisions ─────────────────────────────────

def _sub_estimate_revisions(ticker: str) -> SubScore:
    """Wraps data.estimate_revisions.estimate_revisions_for_ticker.

    Note: no as_of_date support — the FMP endpoint returns historical
    snapshots; we don't backfill point-in-time consensus for the
    backtest. This is precisely why the validation runs variant (b)
    with this sub-score's weight forced to 0 (plan §E).
    """
    try:
        from data.estimate_revisions import estimate_revisions_for_ticker
        er = estimate_revisions_for_ticker(ticker)
        return SubScore(
            name="estimate_revisions",
            value=float(er.score),
            weight=0.0,  # set by caller from weights dict
            raw_inputs={
                "eps_delta_pct": er.raw_eps_delta_pct,
                "beat_rate":     er.raw_beat_rate,
            },
            source=er.source,
            explanation=er.explanation,
            degraded=er.degraded,
        )
    except Exception as e:
        log.debug(f"[dislocation] _sub_estimate_revisions({ticker}): {e}")
        return SubScore(
            name="estimate_revisions",
            value=0.0, weight=0.0, raw_inputs={}, source="error",
            explanation=f"estimate_revisions error: {e}",
            degraded=True,
        )


# ── Sub-score #2 — guidance change ────────────────────────────────────

# Keyword lists for headline classification. Conservative bias: a cut
# fires more easily than a raise (cuts are typically clearer in titles).
_GUIDANCE_RAISE_KW = {
    "raises guidance", "raised guidance", "raises outlook", "raises forecast",
    "upgrades guidance", "boosts forecast", "lifts forecast", "raises full-year",
    "above prior", "raises fy", "raises 20",  # raises 2025/2026
}
_GUIDANCE_REAFFIRM_KW = {
    "reaffirms guidance", "reiterates guidance", "maintains guidance",
    "reaffirmed outlook", "reiterated outlook",
}
_GUIDANCE_CUT_KW = {
    "cuts guidance", "lowers guidance", "slashes guidance", "reduces guidance",
    "cuts outlook", "lowers outlook", "cuts forecast", "lowers forecast",
    "warns on", "guidance cut", "outlook cut", "guides below", "guides lower",
    "withdraws guidance", "pulls guidance", "trims forecast",
}


def _classify_guidance_in_title(title: str) -> Optional[float]:
    """Return +1.0 / +0.3 / -1.0 / None based on guidance keywords in headline title.

    None means no guidance signal detected (don't penalize ambiguity).
    """
    t = (title or "").lower()
    if any(kw in t for kw in _GUIDANCE_CUT_KW):
        return -1.0
    if any(kw in t for kw in _GUIDANCE_RAISE_KW):
        return 1.0
    if any(kw in t for kw in _GUIDANCE_REAFFIRM_KW):
        return 0.3
    return None


def _sub_guidance(
    ticker: str,
    *,
    as_of_date: Optional[date] = None,
    lookback_days: int = 15,
) -> SubScore:
    """Sub-score #2: guidance direction from headlines (±lookback_days) + 8-K Item 2.02.

    PIT-aware: when `as_of_date` is set, only headlines and 8-K filings
    with date <= as_of_date are considered. The plan's "±15d" window
    becomes "15d backwards only" because future headlines are filtered.
    Production (as_of=None) sees all available headlines naturally.

    Returns the MOST EXTREME signal found (strongest cut or raise).
    """
    raw_inputs = {"headline_hits": [], "8k_score": None}
    signal = None
    explanation_parts = []

    # Headlines path
    try:
        from utils.news_headlines import fetch_headlines
        headlines = fetch_headlines(
            ticker, max_results=20, as_of_date=as_of_date,
        )
        # Filter by lookback window (additionally restrict to as_of - lookback_days
        # when as_of_date is set, to honor the "±15d" plan window in the past-only sense)
        cutoff = None
        if as_of_date is not None:
            cutoff = as_of_date - timedelta(days=lookback_days)
        else:
            cutoff = date.today() - timedelta(days=lookback_days)

        for h in headlines:
            try:
                pub = date.fromisoformat((h.published_at or "")[:10])
                if pub < cutoff:
                    continue
            except Exception:
                continue
            cls = _classify_guidance_in_title(h.title)
            if cls is not None:
                raw_inputs["headline_hits"].append({
                    "date": pub.isoformat(),
                    "title": h.title[:100],
                    "signal": cls,
                })
                if signal is None or abs(cls) > abs(signal):
                    signal = cls
    except Exception as e:
        log.debug(f"[dislocation] _sub_guidance headlines({ticker}): {e}")

    # 8-K Item 2.02 path (results/guidance) — already PIT via as_of_date
    try:
        from data.fundamental_signals import classify_recent_8k
        k_score = classify_recent_8k(
            ticker, lookback_days=lookback_days, as_of_date=as_of_date,
        )
        # k_score returns NaN when no recent 8-K
        if k_score is not None and k_score == k_score:  # not NaN
            raw_inputs["8k_score"] = float(k_score)
            # 8-K Item 2.02 is Earnings/Guidance, scored 0.0 by default in the
            # _8K_ITEM_SCORES table. Item 4.02 restatement (-0.9) is governance
            # not guidance, so we map the 8-K score conservatively into the
            # guidance signal only when |score| >= 0.4 (i.e. material 8-K event).
            if abs(k_score) >= 0.4:
                # Same blending rule: most-extreme wins
                if signal is None or abs(k_score) > abs(signal):
                    signal = float(k_score)
    except Exception as e:
        log.debug(f"[dislocation] _sub_guidance 8k({ticker}): {e}")

    if signal is None:
        return SubScore(
            name="guidance",
            value=0.0, weight=0.0, raw_inputs=raw_inputs,
            source="no_signal",
            explanation="no guidance keyword/8-K hits in window",
            degraded=False,
        )

    direction = "raise" if signal > 0.5 else ("reaffirm" if signal > 0 else "cut")
    return SubScore(
        name="guidance",
        value=float(np.clip(signal, -1.0, 1.0)),
        weight=0.0,
        raw_inputs=raw_inputs,
        source="news+8k" if raw_inputs["headline_hits"] and raw_inputs["8k_score"] else (
            "8k" if raw_inputs["8k_score"] else "news"
        ),
        explanation=f"guidance {direction} detected ({len(raw_inputs['headline_hits'])} headlines, "
                    f"8k_score={raw_inputs['8k_score']})",
        degraded=False,
    )


# ── Sub-score #3 — drop attribution (HIGHEST-WEIGHTED sub-score) ──────

# Bad-news keywords that subtract -0.5 from the sector-relative signal.
_BAD_NEWS_KW = {
    "miss", "missed", "misses",
    "cut", "cuts", "cutting", "lowered", "slashes",
    "trial fail", "trial failed", "failed trial",
    "fraud", "fraudulent",
    "accounting", "restatement", "restated",
    "sec investigation", "sec probe", "subpoena",
    "fda rejection", "fda reject", "fda complete response letter",
    "lawsuit", "class action",
}

# Relief keywords that NEGATE bad-news detection. Context guard per user
# feedback: a headline like "miss fears overblown" or "cleared in SEC
# probe" has bad keywords but is REASSURING — must not penalize.
_RELIEF_KW = {
    "overblown", "cleared", "exonerated", "vindicated", "dismissed",
    "rebuts", "rebutted", "denies", "denied", "addresses concerns",
    "no wrongdoing", "no merit", "rejected", "thrown out", "withdraws complaint",
    "reassures", "reaffirms", "settles for nominal", "minor",
}


def _has_bad_news(title_lower: str) -> bool:
    """True iff bad-news keyword present AND no relief keyword present."""
    has_bad = any(kw in title_lower for kw in _BAD_NEWS_KW)
    if not has_bad:
        return False
    has_relief = any(kw in title_lower for kw in _RELIEF_KW)
    return not has_relief


def _sub_drop_attribution(
    ticker: str,
    signals: dict,
    *,
    as_of_date: Optional[date] = None,
    lookback_days: int = 15,
) -> SubScore:
    """Sub-score #3: classifies the drop as sector-wide (reward) vs stock-specific (penalize).

    Formula (plan §B):
      base = 0.7 * tanh((sector_rel_60d - own_60d_implicit) / 0.10)
           + 0.3 * tanh((sector_rel_20d - own_20d_implicit) / 0.05)
      penalty = -0.5 if bad-news keyword OR 8-K Item 4.02 in window, ELSE 0

    Note: sector_relative_return already gives us (ret_ticker - ret_sector_etf)
    directly, which IS the "stock minus sector" signal. The plan's
    "(sector_rel_60d - own_60d)" double-subtraction is redundant —
    sector_relative_return already encodes that. We use it directly.

    A negative sector-relative (ticker underperformed sector) = positive
    score (forced selling / sentiment dislocation candidate, the thing we
    want to reward). A positive sector-relative = ticker is outperforming
    sector = negative score (no dislocation).

    PIT-aware:
      - Headlines fetched with as_of_date filter (no future news)
      - sector_relative_return uses price history that naturally cuts at
        the as_of_date if hist is fetched from yfinance ending on that
        date (caller responsibility in backtest spine builder)
      - 8-K Item 4.02 filings via classify_recent_8k (already PIT)
    """
    # _score_core_dip doesn't populate signals["sector"] today.
    # Lazy-fetch from yfinance when missing so the highest-weighted
    # sub-score has the data it needs to actually run. Cached by yfinance
    # internally so a CORE sweep won't re-hit the network per ticker.
    sector = signals.get("sector") or ""
    if not sector:
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info or {}
            sector = info.get("sector") or ""
        except Exception:
            sector = ""

    try:
        from data.sector_relative import sector_relative_pair
        long_rel, short_rel = sector_relative_pair(
            ticker, sector,
            long_window=60, short_window=20,
        )
    except Exception as e:
        log.debug(f"[dislocation] _sub_drop_attribution sector_rel({ticker}): {e}")
        long_rel, short_rel = None, None

    if long_rel is None and short_rel is None:
        return SubScore(
            name="drop_attribution",
            value=0.0, weight=0.0,
            raw_inputs={"sector": sector, "long_rel": None, "short_rel": None},
            source="missing",
            explanation=f"no sector-relative data (sector={sector!r})",
            degraded=True,
        )

    # Negate the rel to make "underperformed sector" positive (=
    # candidate dislocation). Plan §B intent.
    # When both windows available, blend 70/30 (long primary, short tiebreaker).
    # When only one available, use it at full magnitude.
    if long_rel is not None and short_rel is not None:
        base = 0.7 * math.tanh((-long_rel) / 0.10) + 0.3 * math.tanh((-short_rel) / 0.05)
    elif long_rel is not None:
        base = math.tanh((-long_rel) / 0.10)
    else:  # short_rel is not None (else we returned early above)
        base = math.tanh((-short_rel) / 0.05)

    # Bad-news penalty (PIT-filtered headlines)
    bad_news_hits = []
    relief_hits = []
    try:
        from utils.news_headlines import fetch_headlines
        headlines = fetch_headlines(
            ticker, max_results=20, as_of_date=as_of_date,
        )
        cutoff = (as_of_date or date.today()) - timedelta(days=lookback_days)
        for h in headlines:
            try:
                pub = date.fromisoformat((h.published_at or "")[:10])
                if pub < cutoff:
                    continue
            except Exception:
                continue
            title_lower = (h.title or "").lower()
            has_bad = any(kw in title_lower for kw in _BAD_NEWS_KW)
            if not has_bad:
                continue
            has_relief = any(kw in title_lower for kw in _RELIEF_KW)
            if has_relief:
                relief_hits.append({
                    "date": pub.isoformat(),
                    "title": h.title[:100],
                })
                continue
            bad_news_hits.append({
                "date": pub.isoformat(),
                "title": h.title[:100],
            })
    except Exception as e:
        log.debug(f"[dislocation] _sub_drop_attribution headlines({ticker}): {e}")

    # Restatement 8-K (Item 4.02) in window — same penalty
    has_restatement = False
    try:
        from data.fundamental_signals import classify_recent_8k
        k_score = classify_recent_8k(
            ticker, lookback_days=lookback_days, as_of_date=as_of_date,
        )
        if k_score is not None and k_score <= -0.8:
            has_restatement = True
    except Exception:
        pass

    penalty = 0.0
    penalty_reason_parts = []
    if bad_news_hits:
        penalty = -0.5
        penalty_reason_parts.append(f"{len(bad_news_hits)} bad-news headlines")
    if has_restatement:
        penalty = min(penalty, -0.5)
        penalty_reason_parts.append("restatement 8-K (Item 4.02)")

    final = float(np.clip(base + penalty, -1.5, 1.0))

    expl = []
    if long_rel is not None:
        expl.append(f"sector_rel_60d {long_rel*100:+.0f}pp")
    if short_rel is not None:
        expl.append(f"sector_rel_20d {short_rel*100:+.0f}pp")
    if penalty_reason_parts:
        expl.append(f"penalty (-0.5): {'; '.join(penalty_reason_parts)}")
    if relief_hits:
        expl.append(f"({len(relief_hits)} relief hits negated penalty)")

    return SubScore(
        name="drop_attribution",
        value=final,
        weight=0.0,
        raw_inputs={
            "sector":         sector,
            "long_rel_60d":   long_rel,
            "short_rel_20d":  short_rel,
            "bad_news_hits":  bad_news_hits,
            "relief_hits":    relief_hits,
            "has_restatement": has_restatement,
        },
        source="sector_rel+news+8k",
        explanation="  ·  ".join(expl) if expl else "no signal",
        degraded=False,
    )


# ── Sub-score #4 — catalyst proximity ─────────────────────────────────

def _sub_catalyst(ticker: str) -> SubScore:
    """Triangular reward around 21-42d earnings window. No PIT issue:
    earnings calendar is forward-looking by definition.

    Formula (plan §B):
      d = 0..20    → linear ramp 0 → +1
      d = 21..42   → +1 (sweet spot, "not stuck")
      d = 42..90   → linear decay +1 → 0
      d > 90 or no earnings → -0.2 ("stuck" penalty)
    """
    try:
        from data.market_client import get_earnings_days
        d = get_earnings_days(ticker)
    except Exception as e:
        log.debug(f"[dislocation] _sub_catalyst({ticker}): {e}")
        d = None

    if d is None:
        return SubScore(
            name="catalyst", value=-0.2, weight=0.0,
            raw_inputs={"days_to_earnings": None},
            source="missing",
            explanation="no earnings date within 45d (stuck penalty)",
            degraded=False,
        )

    if d < 0:
        # Reported recently; treat as no near catalyst
        val = -0.2
        expl = f"earnings reported {-d}d ago (no near catalyst)"
    elif d <= 20:
        val = d / 20.0  # 0..+1
        expl = f"earnings in {d}d (ramping up)"
    elif d <= 42:
        val = 1.0
        expl = f"earnings in {d}d (sweet spot 21-42d)"
    elif d <= 90:
        val = 1.0 - (d - 42) / 48.0   # +1 → 0
        expl = f"earnings in {d}d (decaying toward window)"
    else:
        val = -0.2
        expl = f"earnings in {d}d (too far)"

    return SubScore(
        name="catalyst", value=float(val), weight=0.0,
        raw_inputs={"days_to_earnings": d},
        source="yfinance_calendar",
        explanation=expl,
        degraded=False,
    )


# ── Sub-score #5 — absolute valuation guard ───────────────────────────

def _sub_valuation(ticker: str) -> SubScore:
    """Wraps data.valuation_percentile. PIT-clean (uses live + cached
    samples; cache TTL 7d but historical PE series at as_of_date isn't
    something we have — accept the small drift in backtest)."""
    try:
        from data.valuation_percentile import valuation_percentile, valuation_subscore
        b = valuation_percentile(ticker)
        sub = valuation_subscore(b)
        if sub is None:
            return SubScore(
                name="valuation", value=0.0, weight=0.0,
                raw_inputs={
                    "current_fwd_pe": b.current_fwd_pe,
                    "pe_hist_p25":    b.pe_hist_p25,
                },
                source=b.source,
                explanation="valuation data missing — neutral",
                degraded=True,
            )
        expl_parts = []
        if b.current_fwd_pe and b.pe_hist_p25:
            expl_parts.append(f"P/E {b.current_fwd_pe:.1f} vs p25 {b.pe_hist_p25:.1f}")
        if b.current_pe_percentile_in_own_3y is not None:
            expl_parts.append(f"own-pct {b.current_pe_percentile_in_own_3y*100:.0f}%")
        return SubScore(
            name="valuation",
            value=float(sub), weight=0.0,
            raw_inputs={
                "current_fwd_pe":   b.current_fwd_pe,
                "pe_hist_p25":      b.pe_hist_p25,
                "pe_hist_p75":      b.pe_hist_p75,
                "own_pct":          b.current_pe_percentile_in_own_3y,
            },
            source=b.source,
            explanation="  ·  ".join(expl_parts),
            degraded=False,
        )
    except Exception as e:
        log.debug(f"[dislocation] _sub_valuation({ticker}): {e}")
        return SubScore(
            name="valuation", value=0.0, weight=0.0, raw_inputs={},
            source="error", explanation=f"valuation error: {e}",
            degraded=True,
        )


# ── Sub-score #6 — quality (hard gate + graded) ───────────────────────

def _sub_quality(signals: dict) -> SubScore:
    """Reuses composite_score._quality_composite for the graded portion.

    The hard gate is checked separately in `passes_gates()` — if quality
    fails the gate, the whole name is excluded before scoring. This
    sub-score only fires when the gate has already passed.

    Range [0, 1]; value of 0.5 = neutral (no quality data).
    """
    try:
        from core.composite_score import _quality_composite
        q = _quality_composite(signals)
        # Convert [0,1] → [0, +1] (already in range)
        return SubScore(
            name="quality", value=float(q), weight=0.0,
            raw_inputs={
                "roe":            signals.get("roe"),
                "gross_margin":   signals.get("gross_margin"),
                "revenue_growth": signals.get("revenue_growth"),
                "fcf_yield":      signals.get("fcf_yield_fund") or signals.get("fcf_yield"),
            },
            source="yfinance_fundamentals",
            explanation=f"quality composite {q:.2f} (post-gate)",
            degraded=q == 0.5,
        )
    except Exception as e:
        log.debug(f"[dislocation] _sub_quality: {e}")
        return SubScore(
            name="quality", value=0.5, weight=0.0,
            raw_inputs={}, source="error",
            explanation=f"quality error: {e}", degraded=True,
        )


# ── Sub-score #7 — governance ─────────────────────────────────────────

# Anomalous insider sale threshold per user plan (>$10M by CEO/CFO in 60d).
_INSIDER_SELL_ANOMALY_USD = 10_000_000

# Keywords in news headlines indicating mgmt departures (CEO/CFO out).
_MGMT_DEPARTURE_KW = {
    "ceo resigns", "ceo steps down", "ceo leaves",
    "cfo resigns", "cfo steps down", "cfo leaves",
    "chairman resigns", "chairman steps down",
    "fires ceo", "fires cfo", "ousts ceo", "ousts cfo",
}

# Material weakness keywords (governance red flag)
_MATERIAL_WEAKNESS_KW = {
    "material weakness", "internal controls weakness", "deficiency in controls",
}


def _sub_governance(
    ticker: str,
    *,
    as_of_date: Optional[date] = None,
    lookback_days: int = 60,
) -> SubScore:
    """Sum (capped at -2): restatement, material weakness, big insider sale, mgmt depart.

    PIT-aware via classify_recent_8k + insider_sell_amount_score (both
    accept as_of_date) + news headlines (filtered).
    """
    score = 0.0
    raw = {}
    expl_parts = []

    # 8-K Item 4.02 restatement
    try:
        from data.fundamental_signals import classify_recent_8k
        k = classify_recent_8k(ticker, lookback_days=lookback_days, as_of_date=as_of_date)
        if k is not None and k == k and k <= -0.8:  # -0.9 for restatement
            score -= 1.0
            raw["restatement_8k_score"] = float(k)
            expl_parts.append("restatement 8-K (-1.0)")
    except Exception:
        pass

    # Insider sell amount anomaly
    try:
        from data.fundamental_signals import insider_sell_amount_score
        # Function returns (score_norm, amount_usd) tuple per signature
        result = insider_sell_amount_score(
            ticker, lookback_days=lookback_days, as_of_date=as_of_date,
        )
        amount_usd = None
        if isinstance(result, tuple) and len(result) >= 2:
            amount_usd = float(result[1] or 0.0)
        elif isinstance(result, (int, float)):
            amount_usd = float(result)
        if amount_usd and amount_usd >= _INSIDER_SELL_ANOMALY_USD:
            score -= 0.5
            raw["insider_sell_usd"] = amount_usd
            expl_parts.append(f"insider sell ${amount_usd/1e6:.0f}M (-0.5)")
    except Exception:
        pass

    # Headlines: mgmt departure + material weakness
    try:
        from utils.news_headlines import fetch_headlines
        headlines = fetch_headlines(
            ticker, max_results=20, as_of_date=as_of_date,
        )
        cutoff = (as_of_date or date.today()) - timedelta(days=lookback_days)
        depart_hits = []
        weakness_hits = []
        for h in headlines:
            try:
                pub = date.fromisoformat((h.published_at or "")[:10])
                if pub < cutoff:
                    continue
            except Exception:
                continue
            title_lower = (h.title or "").lower()
            if any(kw in title_lower for kw in _MGMT_DEPARTURE_KW):
                depart_hits.append(h.title[:100])
            if any(kw in title_lower for kw in _MATERIAL_WEAKNESS_KW):
                weakness_hits.append(h.title[:100])
        if depart_hits:
            score -= 0.3
            raw["mgmt_departure_hits"] = depart_hits
            expl_parts.append(f"mgmt departure (-0.3)")
        if weakness_hits:
            score -= 0.7
            raw["material_weakness_hits"] = weakness_hits
            expl_parts.append(f"material weakness (-0.7)")
    except Exception:
        pass

    score = float(np.clip(score, -2.0, 0.0))
    return SubScore(
        name="governance", value=score, weight=0.0,
        raw_inputs=raw,
        source="8k+insider+news",
        explanation=("; ".join(expl_parts) if expl_parts else "no red flags"),
        degraded=False,
    )


# ── Master scorer ─────────────────────────────────────────────────────

def passes_gates(
    signals: dict,
    *,
    governance_subscore: Optional[SubScore] = None,
) -> tuple[bool, str]:
    """Hard exclusion: quality gate fail OR governance == -2.

    Returns (passes, reason).
    """
    quality_ok, quality_reason = _passes_quality_gate(signals)
    if not quality_ok:
        return False, f"quality_gate_fail: {quality_reason}"
    if governance_subscore is not None and governance_subscore.value <= -1.99:
        return False, f"governance_capped: {governance_subscore.explanation}"
    return True, ""


def _build_summary_line(
    ticker: str,
    composite_z: Optional[float],
    sub_scores: dict[str, SubScore],
    threshold_abs: float = 0.5,
) -> str:
    """Auto-build a one-line explanation from sub-scores ≥ |threshold_abs|.

    Format: "TICKER  +Xz  | reason1 + reason2 + reason3"
    """
    z_part = f"{composite_z:+.2f}z" if composite_z is not None else "?z"
    notes = []
    for s in sub_scores.values():
        if abs(s.value) >= threshold_abs:
            # Short tag per sub-score
            if s.name == "estimate_revisions" and s.value > 0:
                notes.append("estimates_holding")
            elif s.name == "estimate_revisions" and s.value < 0:
                notes.append("estimates_falling")
            elif s.name == "guidance" and s.value > 0:
                notes.append("guidance_raise")
            elif s.name == "guidance" and s.value < 0:
                notes.append("guidance_cut")
            elif s.name == "drop_attribution" and s.value > 0:
                rel = s.raw_inputs.get("long_rel_60d")
                if rel is not None:
                    notes.append(f"sector_drop {rel*100:+.0f}%")
                else:
                    notes.append("sector_drop")
            elif s.name == "drop_attribution" and s.value < 0:
                notes.append("stock_specific_bad_news")
            elif s.name == "catalyst" and s.value > 0:
                d = s.raw_inputs.get("days_to_earnings")
                if d is not None:
                    notes.append(f"catalyst in {d}d")
            elif s.name == "valuation" and s.value > 0:
                pct = s.raw_inputs.get("own_pct")
                if pct is not None:
                    notes.append(f"cheap (P/E p{pct*100:.0f})")
                else:
                    notes.append("cheap")
            elif s.name == "valuation" and s.value < 0:
                notes.append("expensive_vs_own_history")
            elif s.name == "quality" and s.value >= 0.7:
                notes.append("strong_quality")
            elif s.name == "governance" and s.value <= -0.5:
                notes.append("governance_red_flag")
    body = " + ".join(notes) if notes else "no strong components"
    return f"{ticker}  {z_part}  | {body}"


def dislocation_for_ticker(
    ticker: str,
    signals: dict,
    *,
    as_of_date: Optional[date] = None,
    variant: str = "a",
    strict_pit: bool = False,
    stats: Optional[dict] = None,
) -> DislocationBreakdown:
    """Compute the full dislocation breakdown for a single ticker.

    Args:
        ticker: ticker symbol (uppercase recommended)
        signals: pre-fetched dict from core_radar._score_core_dip (has
            sector, roe, gross_margin, revenue_growth, fcf_yield_fund,
            return_20d, dd_from_52w_high, etc.). Required for hard gates
            and sub-scores #5/#6.
        as_of_date: PIT cut-off date. None = production "now" (no PIT
            filter). Required when strict_pit=True.
        variant: "a" = full 7-signal scorer. "b" = signal #1 weight
            forced to 0 (PIT-clean BINDING gate for validation, plan §E).
        strict_pit: when True, requires as_of_date to be set. Used by
            the backtest harness — guards against silently running a
            fold with no PIT filter (look-ahead leakage). Defaults False
            for production callers.
        stats: optional pre-loaded dislocation_stats dict to avoid disk
            I/O when scoring many names. Built on demand if None.
    """
    if strict_pit and as_of_date is None:
        raise ValueError(
            "strict_pit=True requires explicit as_of_date — refusing to run "
            "without PIT filter. This guards against silent look-ahead leakage "
            "in the validation harness."
        )

    ticker = (ticker or "").upper().strip()
    weights = _load_weights()

    # Variant (b): force signal #1 weight to 0 — PIT-clean BINDING gate.
    if variant == "b":
        weights = dict(weights)
        weights["estimate_revisions"] = 0.0

    sub_scores: dict[str, SubScore] = {}

    sub_scores["estimate_revisions"] = _sub_estimate_revisions(ticker)
    sub_scores["guidance"]           = _sub_guidance(ticker, as_of_date=as_of_date)
    sub_scores["drop_attribution"]   = _sub_drop_attribution(ticker, signals, as_of_date=as_of_date)
    sub_scores["catalyst"]           = _sub_catalyst(ticker)
    sub_scores["valuation"]          = _sub_valuation(ticker)
    sub_scores["quality"]            = _sub_quality(signals)
    sub_scores["governance"]         = _sub_governance(ticker, as_of_date=as_of_date)

    # Assign weights now that we have the sub-scores
    for name, s in sub_scores.items():
        s.weight = float(weights.get(name, 0.0))

    # Hard gates
    gate_pass, gate_reason = passes_gates(signals, governance_subscore=sub_scores["governance"])
    if not gate_pass:
        return DislocationBreakdown(
            ticker=ticker,
            as_of_date=(as_of_date.isoformat() if as_of_date else None),
            variant=variant,
            hard_excluded=True, exclude_reason=gate_reason,
            sub_scores=sub_scores,
            composite_raw=0.0, composite_z=None,
            summary_line=f"{ticker} EXCLUDED: {gate_reason}",
        )

    # Per-name degraded-proxy guard on signal #1 (plan §C)
    er = sub_scores["estimate_revisions"]
    if er.degraded:
        er.weight = er.weight * 0.5

    composite_raw = sum(s.value * s.weight for s in sub_scores.values())

    # z-score against universe distribution
    if stats is None:
        stats = build_dislocation_stats()
    composite_z: Optional[float] = None
    if stats:
        mu = stats.get("mean")
        sigma = stats.get("std")
        if sigma is not None and sigma > 0:
            composite_z = (composite_raw - mu) / sigma

    summary = _build_summary_line(ticker, composite_z, sub_scores)

    return DislocationBreakdown(
        ticker=ticker,
        as_of_date=(as_of_date.isoformat() if as_of_date else None),
        variant=variant,
        hard_excluded=False, exclude_reason=None,
        sub_scores=sub_scores,
        composite_raw=composite_raw,
        composite_z=composite_z,
        summary_line=summary,
    )


# ── Universe stats (μ, σ for z-scoring) ───────────────────────────────

def build_dislocation_stats(force_refresh: bool = False) -> dict:
    """Walk the CORE universe to compute μ and σ of composite_raw.

    Cached daily at /data/dislocation_stats.json. Lazy build on first
    call. Walks ~230 tickers — heavy (~3-8 min). Use the env override
    DIPRADAR_DISLOCATION_STATS_REFRESH=1 to force a refresh.
    """
    if not force_refresh and _STATS_PATH.exists():
        try:
            cached = json.loads(_STATS_PATH.read_text())
            computed_at = cached.get("computed_at")
            if computed_at:
                age_h = (datetime.now() - datetime.fromisoformat(computed_at)).total_seconds() / 3600
                if age_h < _STATS_TTL_HOURS:
                    log.info(
                        f"[dislocation] stats cache hit "
                        f"({age_h:.1f}h old, n={cached.get('n')})"
                    )
                    return cached
        except Exception as e:
            log.warning(f"[dislocation] stats cache read failed: {e}")

    log.info("[dislocation] rebuilding dislocation_stats (3-8 min)")
    try:
        from core.core_radar import scan_core_universe
        opps = scan_core_universe(min_score=0.0)
    except Exception as e:
        log.warning(f"[dislocation] scan_core_universe failed: {e}")
        return {}

    if not opps:
        return {}

    raws: list[float] = []
    for o in opps:
        ticker = o.get("ticker", "")
        if not ticker:
            continue
        try:
            b = dislocation_for_ticker(ticker, o, as_of_date=None, variant="a", stats={})
            if not b.hard_excluded:
                raws.append(b.composite_raw)
        except Exception as e:
            log.debug(f"[dislocation] stats raw({ticker}): {e}")
            continue

    if not raws:
        return {}

    arr = np.asarray(raws, dtype=float)
    stats = {
        "computed_at": datetime.now().isoformat(timespec="seconds"),
        "n":           int(len(arr)),
        "mean":        float(arr.mean()),
        "std":         float(arr.std()) or 0.01,
        "p10":         float(np.quantile(arr, 0.10)),
        "p50":         float(np.quantile(arr, 0.50)),
        "p90":         float(np.quantile(arr, 0.90)),
    }
    try:
        _STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATS_PATH.write_text(json.dumps(stats, indent=2))
        log.info(f"[dislocation] persisted stats to {_STATS_PATH}")
    except Exception as e:
        log.warning(f"[dislocation] stats persist failed: {e}")
    return stats
