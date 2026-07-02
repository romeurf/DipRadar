"""tests/test_honest_watchlist.py — watchlist criteria + sector-rel column.

Per F.5 user review post-REVERT:
  - watchlist FILTERS by quality + DD>=25% + catalyst<=30d
  - sector_relative is COMPUTED and DISPLAYED but NOT used to filter
    (filtering by it reopens value-trap door for absolute drops)
  - dispatcher 3rd mode + DIPRADAR_SCORING_MODE flag default=watchlist
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from core import honest_watchlist as hw
from core.scoring_dispatch import (
    get_active_scorer, ScoringSession, ScoreResult,
)


# ── Quality gate ─────────────────────────────────────────────────────

class TestQualityGate:
    """Per F.5 design: 2-of-3 signal sufficiency + data-driven thresholds."""

    # 2-of-3 SUFFICIENCY (counts of {ROE, GM, revenue_growth} present)

    def test_sufficiency_pass_with_all_three(self):
        sigs = {"revenue_growth": 0.10, "gross_margin": 0.50, "roe": 0.30}
        ok, _ = hw._quality_passes(sigs)
        assert ok

    def test_sufficiency_pass_with_two_of_three(self):
        """Per user: partial data OK if at least 2 present."""
        ok, _ = hw._quality_passes({"roe": 0.30, "gross_margin": 0.50})
        assert ok

    def test_sufficiency_fail_with_only_one(self):
        ok, reason = hw._quality_passes({"roe": 0.30})
        assert not ok
        assert "1/3" in reason

    def test_sufficiency_fail_with_zero(self):
        ok, reason = hw._quality_passes({})
        assert not ok
        assert "0/3" in reason

    # FCF graded check (allows investment-phase compounders)

    def test_fcf_negative_excludes_without_investment_phase(self):
        sigs = {"revenue_growth": 0.10, "gross_margin": 0.50, "roe": 0.30,
                "fcf_yield_fund": -0.05}
        ok, reason = hw._quality_passes(sigs)
        assert not ok and "fcf" in reason.lower()

    def test_fcf_negative_ok_when_investment_phase(self):
        sigs = {"revenue_growth": 0.10, "gross_margin": 0.50, "roe": 0.30,
                "fcf_yield_fund": -0.05, "is_investment_phase": True}
        ok, _ = hw._quality_passes(sigs)
        assert ok

    # DATA-DRIVEN THRESHOLDS (not "acho que")

    def test_threshold_fail_below_universe_median_roe(self):
        sigs = {"roe": 0.05, "gross_margin": 0.50, "revenue_growth": 0.10}
        # Universe median ROE = 0.20 → ticker ROE 0.05 fails
        thresholds = {"roe": 0.20, "gross_margin": 0.30, "revenue_growth": 0.0}
        ok, reason = hw._quality_passes(sigs, thresholds=thresholds)
        assert not ok and "roe" in reason.lower() and "median" in reason.lower()

    def test_threshold_fail_below_universe_median_gm(self):
        sigs = {"roe": 0.30, "gross_margin": 0.20, "revenue_growth": 0.10}
        thresholds = {"roe": 0.20, "gross_margin": 0.30, "revenue_growth": 0.0}
        ok, reason = hw._quality_passes(sigs, thresholds=thresholds)
        assert not ok and "gross_margin" in reason.lower()

    def test_threshold_pass_above_all_universe_medians(self):
        sigs = {"roe": 0.30, "gross_margin": 0.50, "revenue_growth": 0.10}
        thresholds = {"roe": 0.20, "gross_margin": 0.30, "revenue_growth": 0.0}
        ok, _ = hw._quality_passes(sigs, thresholds=thresholds)
        assert ok

    def test_threshold_missing_signal_neutral(self):
        """Missing signal does NOT vote — neither fails nor passes its
        threshold. As long as 2 of 3 are present and pass theirs, the
        name qualifies."""
        sigs = {"roe": 0.30, "gross_margin": 0.50}  # no revenue_growth
        thresholds = {"roe": 0.20, "gross_margin": 0.30, "revenue_growth": 0.0}
        ok, _ = hw._quality_passes(sigs, thresholds=thresholds)
        assert ok

    # compute_quality_thresholds helper

    def test_compute_thresholds_from_universe(self):
        opps = [
            {"roe": 0.10, "gross_margin": 0.30, "revenue_growth": 0.05},
            {"roe": 0.20, "gross_margin": 0.40, "revenue_growth": 0.10},
            {"roe": 0.30, "gross_margin": 0.50, "revenue_growth": 0.15},
        ]
        t = hw.compute_quality_thresholds(opps)
        # Medians of [0.10, 0.20, 0.30] = 0.20 etc
        assert t["roe"] == 0.20
        assert t["gross_margin"] == 0.40
        assert t["revenue_growth"] == 0.10

    def test_compute_thresholds_handles_missing(self):
        """Tickers missing a field shouldn't break median computation;
        only PRESENT values count toward median."""
        opps = [
            {"roe": 0.10, "gross_margin": 0.30},  # no rev_growth
            {"roe": 0.20, "revenue_growth": 0.05},  # no gm
            {"gross_margin": 0.50, "revenue_growth": 0.15},  # no roe
        ]
        t = hw.compute_quality_thresholds(opps)
        # ROE median of [0.10, 0.20] = 0.15
        assert abs(t["roe"] - 0.15) < 1e-9
        # GM median of [0.30, 0.50] = 0.40
        assert abs(t["gross_margin"] - 0.40) < 1e-9


# ── Watchlist criteria assembly ──────────────────────────────────────

class TestEvaluateTicker:
    def _stub_externals(self, monkeypatch, sector="Healthcare",
                        sector_rel_60d=-0.30, catalyst_days=24):
        monkeypatch.setattr(hw, "_fetch_sector_for", lambda t: sector)
        monkeypatch.setattr(hw, "_fetch_sector_relative_60d",
                            lambda t, s: sector_rel_60d)
        monkeypatch.setattr(hw, "_fetch_catalyst_days",
                            lambda t: catalyst_days)

    def test_full_qualifier(self, monkeypatch):
        self._stub_externals(monkeypatch)
        sigs = {
            "revenue_growth": 0.08, "gross_margin": 0.85, "fcf_yield_fund": 0.04,
            "roe": 0.65, "dd_from_52w_high": -0.44,
        }
        e = hw.evaluate_ticker("NVO", sigs)
        assert e.qualifies
        assert e.quality_passes
        assert e.dd_from_52w_high == -0.44
        assert e.days_to_catalyst == 24
        assert e.sector_relative_60d == -0.30
        assert "−44%" in e.summary_line or "-44%" in e.summary_line
        assert "24d" in e.summary_line

    def test_dd_too_shallow_disqualifies(self, monkeypatch):
        """DD = -20% < -25% threshold — does NOT qualify."""
        self._stub_externals(monkeypatch)
        sigs = {
            "revenue_growth": 0.08, "gross_margin": 0.85, "fcf_yield_fund": 0.04,
            "dd_from_52w_high": -0.20,
        }
        e = hw.evaluate_ticker("XYZ", sigs)
        assert not e.qualifies
        assert e.quality_passes  # gate still passed
        # the DD is the failing criterion

    def test_catalyst_far_still_qualifies_when_quality_and_dd_pass(self, monkeypatch):
        """Per F.5 followup: catalyst is a SIGNAL not a FILTER.
        A name with deep DD + quality passing must qualify even when
        catalyst is 60d away — we don't want to miss good dislocations
        just because earnings are distant."""
        self._stub_externals(monkeypatch, catalyst_days=60)
        sigs = {
            "revenue_growth": 0.08, "gross_margin": 0.85, "fcf_yield_fund": 0.04,
            "dd_from_52w_high": -0.44,
        }
        e = hw.evaluate_ticker("ZZZ", sigs)
        assert e.qualifies, "Far catalyst must NOT exclude (now signal, not filter)"
        # Catalyst info still shown in summary for prioritization
        assert "60d" in e.summary_line
        # But no 🎯 badge (not near)
        assert "🎯" not in e.summary_line

    def test_just_reported_recent_qualifies(self, monkeypatch):
        """Catalyst -5d (just reported) is WITHIN the post-earnings
        revaluation window (catalyst_min_days=-10) — qualifies."""
        self._stub_externals(monkeypatch, catalyst_days=-5)
        sigs = {
            "revenue_growth": 0.08, "gross_margin": 0.85, "fcf_yield_fund": 0.04,
            "dd_from_52w_high": -0.30,
        }
        e = hw.evaluate_ticker("YYY", sigs)
        assert e.qualifies

    def test_quality_fail_via_insufficient_data(self, monkeypatch):
        """Without threshold dict (single-ticker /analisar path), quality
        passes if 2 of 3 sufficiency holds. Failure here is only on
        insufficient data (≤1 of 3 fields). Threshold-based exclusion
        is tested separately in TestQualityGate."""
        self._stub_externals(monkeypatch)
        sigs = {
            "gross_margin": 0.85,  # only 1 of 3 signals — fails sufficiency
            "fcf_yield_fund": 0.04,
            "dd_from_52w_high": -0.44,
        }
        e = hw.evaluate_ticker("BAD", sigs)
        assert not e.qualifies
        assert not e.quality_passes
        assert "1/3" in e.quality_reason


# ── CRITICAL: sector-relative is DISPLAY, not FILTER ─────────────────

class TestSectorRelativeAsColumn:
    """The user-mandated nuance: showing vs sector but NOT filtering by
    it. An absolute drop of 26% with vs sector = -2% means 'fell with
    market' — the user reads the column and visually discards. But the
    watchlist does NOT auto-exclude it."""

    def _stub_basic(self, monkeypatch):
        monkeypatch.setattr(hw, "_fetch_sector_for", lambda t: "Tech")
        monkeypatch.setattr(hw, "_fetch_catalyst_days", lambda t: 20)

    def test_qualifier_with_weak_sector_rel_still_qualifies(self, monkeypatch):
        """Name dropped 26% (DD passes) with vs sector = -2% (fell with
        market) — must STILL be in watchlist. User reads the column and
        decides if it's a real dislocation or noise."""
        self._stub_basic(monkeypatch)
        monkeypatch.setattr(hw, "_fetch_sector_relative_60d",
                            lambda t, s: -0.02)  # near-zero sector-rel
        sigs = {
            "revenue_growth": 0.05, "gross_margin": 0.60, "fcf_yield_fund": 0.02,
            "dd_from_52w_high": -0.26,
        }
        e = hw.evaluate_ticker("MKT", sigs)
        assert e.qualifies, "Weak sector-rel must NOT exclude — it's display only"
        assert e.sector_relative_60d == -0.02

    def test_qualifier_with_strong_sector_rel_qualifies(self, monkeypatch):
        """Name fell -38% vs sector (real underperformance) — same path."""
        self._stub_basic(monkeypatch)
        monkeypatch.setattr(hw, "_fetch_sector_relative_60d",
                            lambda t, s: -0.38)
        sigs = {
            "revenue_growth": 0.05, "gross_margin": 0.85, "fcf_yield_fund": 0.04,
            "dd_from_52w_high": -0.47,
        }
        e = hw.evaluate_ticker("ADBE", sigs)
        assert e.qualifies
        assert e.sector_relative_60d == -0.38

    def test_summary_line_shows_sector_rel_always(self, monkeypatch):
        """Even when sector-rel is weak/positive, must show in display."""
        self._stub_basic(monkeypatch)
        for sr in [-0.40, -0.02, +0.05, None]:
            monkeypatch.setattr(hw, "_fetch_sector_relative_60d", lambda t, s, _sr=sr: _sr)
            sigs = {
                "revenue_growth": 0.05, "gross_margin": 0.60, "fcf_yield_fund": 0.02,
                "dd_from_52w_high": -0.30,
            }
            e = hw.evaluate_ticker("XYZ", sigs)
            assert "vs sector" in e.summary_line


# ── Dispatcher 3rd mode + flag default ───────────────────────────────

class TestDispatcherWatchlistMode:
    def test_default_no_env_is_watchlist(self, monkeypatch):
        monkeypatch.delenv("DIPRADAR_SCORING_MODE", raising=False)
        monkeypatch.delenv("DIPRADAR_USE_DISLOCATION", raising=False)
        assert get_active_scorer() == "watchlist"

    def test_explicit_watchlist(self, monkeypatch):
        monkeypatch.setenv("DIPRADAR_SCORING_MODE", "watchlist")
        assert get_active_scorer() == "watchlist"

    def test_explicit_composite(self, monkeypatch):
        monkeypatch.setenv("DIPRADAR_SCORING_MODE", "composite")
        assert get_active_scorer() == "composite"

    def test_explicit_dislocation(self, monkeypatch):
        monkeypatch.setenv("DIPRADAR_SCORING_MODE", "dislocation")
        assert get_active_scorer() == "dislocation"

    def test_legacy_flag_backward_compat_composite(self, monkeypatch):
        monkeypatch.delenv("DIPRADAR_SCORING_MODE", raising=False)
        monkeypatch.setenv("DIPRADAR_USE_DISLOCATION", "0")
        assert get_active_scorer() == "composite"

    def test_legacy_flag_backward_compat_dislocation(self, monkeypatch):
        monkeypatch.delenv("DIPRADAR_SCORING_MODE", raising=False)
        monkeypatch.setenv("DIPRADAR_USE_DISLOCATION", "1")
        assert get_active_scorer() == "dislocation"

    def test_invalid_mode_falls_back_to_watchlist(self, monkeypatch):
        monkeypatch.setenv("DIPRADAR_SCORING_MODE", "invalid_garbage")
        monkeypatch.delenv("DIPRADAR_USE_DISLOCATION", raising=False)
        assert get_active_scorer() == "watchlist"

    def test_session_freezes_mode_at_construction(self, monkeypatch):
        monkeypatch.setenv("DIPRADAR_SCORING_MODE", "watchlist")
        session = ScoringSession()
        assert session.scorer == "watchlist"
        # Mid-scan flag flip ignored
        monkeypatch.setenv("DIPRADAR_SCORING_MODE", "composite")
        assert session.scorer == "watchlist"

    def test_watchlist_session_score_returns_qualifies_flag(self, monkeypatch):
        """ScoringSession.score in watchlist mode returns ScoreResult
        with z=None and qualifies bool, NOT z-based ranking."""
        # Stub all watchlist externals
        monkeypatch.setattr(hw, "_fetch_sector_for", lambda t: "Tech")
        monkeypatch.setattr(hw, "_fetch_sector_relative_60d", lambda t, s: -0.30)
        monkeypatch.setattr(hw, "_fetch_catalyst_days", lambda t: 24)

        session = ScoringSession(scorer="watchlist")
        sigs = {
            "revenue_growth": 0.05, "gross_margin": 0.60, "fcf_yield_fund": 0.02,
            "dd_from_52w_high": -0.30,
        }
        r = session.score("XYZ", sigs)
        assert r.scorer == "watchlist"
        assert r.z is None, "watchlist must NOT rank by z"
        assert r.qualifies is True
        assert r.sort_key == -0.30, "sort_key = DD for watchlist ordering"


# ── Summary line format ──────────────────────────────────────────────

class TestSummaryLineFormat:
    def test_full_line(self, monkeypatch):
        monkeypatch.setattr(hw, "_fetch_sector_for", lambda t: "Healthcare")
        monkeypatch.setattr(hw, "_fetch_sector_relative_60d", lambda t, s: -0.31)
        monkeypatch.setattr(hw, "_fetch_catalyst_days", lambda t: 24)
        sigs = {
            "revenue_growth": 0.08, "gross_margin": 0.84, "fcf_yield_fund": 0.04,
            "roe": 0.71, "dd_from_52w_high": -0.44,
        }
        e = hw.evaluate_ticker("NVO", sigs)
        # Expect format like: NVO  -44% from 52w | vs sector -31% | catalyst 24d | ROE 71%, GM 84%
        s = e.summary_line
        assert "NVO" in s
        assert "44%" in s and "from 52w" in s
        assert "vs sector" in s and "31%" in s
        assert "catalyst 24d" in s
        assert "ROE 71%" in s
        assert "GM 84%" in s


# ── Triage layer (CLEAN / BORDERLINE / EXCLUDED) ─────────────────────

class _FakeHeadline:
    def __init__(self, title):
        self.title = title
        self.publisher = "Test"
        self.published_at = "2026-07-01T10:00:00"
        self.url = ""


def _mk_entry(ticker="XYZ", sector="Technology", sector_rel=-0.12,
              dd=-0.40, catalyst=20):
    """Minimal qualifying WatchlistEntry for triage tests."""
    return hw.WatchlistEntry(
        ticker=ticker, qualifies=True,
        dd_from_52w_high=dd,
        sector=sector, sector_relative_60d=sector_rel,
        days_to_catalyst=catalyst, catalyst_source="earnings_calendar",
        quality_passes=True, quality_reason="",
        roe=0.30, gross_margin=0.60, revenue_growth=0.10, fcf_yield=0.03,
        current_price=100.0, core_score=80.0,
        summary_line=f"{ticker} test line",
    )


def _no_headlines(ticker, max_results=8):
    return []


def _no_llm(*args, **kwargs):
    """classify_fn stub: LLM unavailable → triage must use keyword fallback.
    Also shields tests from a real ANTHROPIC_API_KEY in the dev env."""
    return None


class TestTriage:
    """Automates the manual review of the 2026-06-29 /cash proposal:
    NVO at 30.3% (cap), HOOD +34pp (outperformer), MKC merger (event),
    software-AI cluster. Machine excludes clear-cut only; borderlines
    are generous and human-reviewed."""

    def test_excluded_at_cap(self):
        """NVO case: position >= 30% -> EXCLUDED, headroom zero."""
        e = _mk_entry("NVO")
        ctx = {"position_pct": {"NVO": 0.31}, "sector_cost_pct": {}}
        hw.triage_watchlist([e], ctx, fetch_headlines_fn=_no_headlines)
        assert e.triage_tier == "EXCLUDED"
        assert any("hard cap" in r for r in e.triage_reasons)

    def test_excluded_sector_outperform(self):
        """HOOD case: vs sector +34pp -> not a dislocation -> EXCLUDED."""
        e = _mk_entry("HOOD", sector_rel=+0.34)
        ctx = {"position_pct": {}, "sector_cost_pct": {}}
        hw.triage_watchlist([e], ctx, fetch_headlines_fn=_no_headlines)
        assert e.triage_tier == "EXCLUDED"
        assert any("OUTPERFORMOU" in r for r in e.triage_reasons)

    def test_borderline_near_cap(self):
        """27% position: between soft (25%) and hard (30%) -> BORDERLINE."""
        e = _mk_entry("ADBE")
        ctx = {"position_pct": {"ADBE": 0.27}, "sector_cost_pct": {}}
        hw.triage_watchlist([e], ctx, fetch_headlines_fn=_no_headlines)
        assert e.triage_tier == "BORDERLINE"
        assert any("perto do cap" in r for r in e.triage_reasons)

    def test_borderline_fell_with_market(self):
        """vs sector -2pp: fell WITH market, not specific -> BORDERLINE."""
        e = _mk_entry("META", sector_rel=-0.02)
        ctx = {"position_pct": {}, "sector_cost_pct": {}}
        hw.triage_watchlist([e], ctx, fetch_headlines_fn=_no_headlines)
        assert e.triage_tier == "BORDERLINE"
        assert any("COM o mercado" in r for r in e.triage_reasons)

    def test_borderline_event_risk_carries_snippet(self):
        """MKC case: merger headline -> BORDERLINE, and the reason MUST
        carry the headline snippet, never just the keyword."""
        e = _mk_entry("MKC", sector="Consumer Defensive")
        headline = "McCormick announces $44.8B merger with Unilever food business"
        def fetch(t, max_results=8):
            return [_FakeHeadline(headline)]
        ctx = {"position_pct": {}, "sector_cost_pct": {}}
        hw.triage_watchlist([e], ctx, fetch_headlines_fn=fetch,
                            classify_fn=_no_llm)
        assert e.triage_tier == "BORDERLINE"
        event_reasons = [r for r in e.triage_reasons if "event risk" in r]
        assert event_reasons, f"no event-risk reason: {e.triage_reasons}"
        assert "$44.8B merger" in event_reasons[0], \
            "reason must carry the headline SNIPPET, not just the keyword"
        assert "matched: merger" in event_reasons[0]

    def test_borderline_cluster(self):
        """Candidate sector already 40% of portfolio -> adensa flag."""
        e = _mk_entry("PAYC", sector="Technology")
        ctx = {"position_pct": {},
               "sector_cost_pct": {"Technology": 0.40}}
        hw.triage_watchlist([e], ctx, fetch_headlines_fn=_no_headlines)
        assert e.triage_tier == "BORDERLINE"
        assert any("adensa cluster" in r for r in e.triage_reasons)

    def test_borderline_event_and_cluster_both_reasons(self):
        """Event-risk AND cluster simultaneously -> BOTH reasons appear,
        headline snippet still present alongside the cluster reason."""
        e = _mk_entry("CRM", sector="Technology")
        def fetch(t, max_results=8):
            return [_FakeHeadline("Salesforce in talks for $20B acquisition of rival")]
        ctx = {"position_pct": {},
               "sector_cost_pct": {"Technology": 0.45}}
        hw.triage_watchlist([e], ctx, fetch_headlines_fn=fetch,
                            classify_fn=_no_llm)
        assert e.triage_tier == "BORDERLINE"
        assert any("$20B acquisition" in r for r in e.triage_reasons), \
            "snippet must survive when multiple reasons accumulate"
        assert any("adensa cluster" in r for r in e.triage_reasons)
        assert len(e.triage_reasons) >= 2

    def test_clean_deep_dislocation(self):
        """Deep DD, fell alone (-12pp), no flags, diversifies -> CLEAN."""
        e = _mk_entry("SPGI", sector="Financial Services", sector_rel=-0.12)
        ctx = {"position_pct": {},
               "sector_cost_pct": {"Technology": 0.50}}  # different sector
        hw.triage_watchlist([e], ctx, fetch_headlines_fn=_no_headlines)
        assert e.triage_tier == "CLEAN"
        assert e.triage_reasons == []

    def test_multiple_reasons_accumulate(self):
        """Near-cap + fell-with-market together -> both reasons listed."""
        e = _mk_entry("MSFT", sector_rel=-0.03)
        ctx = {"position_pct": {"MSFT": 0.26}, "sector_cost_pct": {}}
        hw.triage_watchlist([e], ctx, fetch_headlines_fn=_no_headlines)
        assert e.triage_tier == "BORDERLINE"
        assert len(e.triage_reasons) == 2

    def test_no_context_all_clean(self):
        """portfolio_context=None -> all CLEAN with explicit skip reason
        (legacy callers see no behavior change)."""
        e = _mk_entry("XYZ", sector_rel=+0.50)  # would be EXCLUDED with ctx
        hw.triage_watchlist([e], None, fetch_headlines_fn=_no_headlines)
        assert e.triage_tier == "CLEAN"
        assert e.triage_reasons == ["triage skipped (sem contexto de portfólio)"]

    def test_sector_relative_none_does_not_vote(self):
        """sector_rel=None abstains: neither excludes (outperform rule)
        nor flags (fell-with-market rule)."""
        e = _mk_entry("UNK", sector_rel=None)
        ctx = {"position_pct": {}, "sector_cost_pct": {}}
        hw.triage_watchlist([e], ctx, fetch_headlines_fn=_no_headlines)
        assert e.triage_tier == "CLEAN"

    def test_excluded_skips_headline_fetch(self):
        """EXCLUDED names must not incur the headline fetch (1-2s each)."""
        calls = []
        def fetch(t, max_results=8):
            calls.append(t)
            return []
        e = _mk_entry("NVO")
        ctx = {"position_pct": {"NVO": 0.35}, "sector_cost_pct": {}}
        hw.triage_watchlist([e], ctx, fetch_headlines_fn=fetch)
        assert e.triage_tier == "EXCLUDED"
        assert calls == [], "headline fetch must be skipped for EXCLUDED"

    def test_get_watchlist_applies_triage(self, monkeypatch):
        """Integration: get_watchlist(portfolio_context=...) triages and
        records tier counts in _last_run_stats."""
        opps = [{
            "ticker": "NVO", "score": 80.0, "current_price": 48.0,
            "roe": 0.71, "gross_margin": 0.83, "revenue_growth": 0.12,
            "fcf_yield_fund": 0.05, "dd_from_52w_high": -0.30,
        }]
        monkeypatch.setattr("core.core_radar.scan_core_universe",
                            lambda min_score=0.0: opps)
        monkeypatch.setattr(hw, "_fetch_sector_for", lambda t: "Healthcare")
        monkeypatch.setattr(hw, "_fetch_sector_relative_60d", lambda t, s: -0.12)
        monkeypatch.setattr(hw, "_fetch_catalyst_days", lambda t: 24)
        # Route triage through the real function but with no-network headlines
        _real_triage = hw.triage_watchlist
        def _triage_no_net(entries, ctx, **kw):
            return _real_triage(entries, ctx, fetch_headlines_fn=_no_headlines)
        monkeypatch.setattr(hw, "triage_watchlist", _triage_no_net)

        ctx = {"position_pct": {"NVO": 0.31}, "sector_cost_pct": {}}
        entries = hw.get_watchlist(source="core", portfolio_context=ctx)
        assert len(entries) == 1
        assert entries[0].triage_tier == "EXCLUDED"
        stats = hw.get_last_run_stats()
        assert stats.get("n_excluded") == 1
        assert stats.get("n_clean") == 0


# ── LLM drop-attribution classifier ──────────────────────────────────

from core.llm_attribution import DropAttribution  # noqa: E402 — pure dataclass, no SDK import
import core.llm_attribution as la  # noqa: E402


def _llm(category, explanation="", key_headline="", confidence="high"):
    """Build a classify_fn stub returning a fixed attribution."""
    def fn(ticker, headlines, **kwargs):
        return DropAttribution(category=category, explanation=explanation,
                               key_headline=key_headline, confidence=confidence)
    return fn


def _llm_must_not_be_called(ticker, headlines, **kwargs):
    raise AssertionError("classifier must not be called")


class TestLLMTriage:
    """LLM drop-attribution inside triage_watchlist: corporate_event and
    deterioration flag BORDERLINE with explanation + evidence; the other
    categories affirmatively clear the name (no keyword noise); any LLM
    failure falls back to the keyword scan unchanged."""

    _CTX = {"position_pct": {}, "sector_cost_pct": {}}

    def test_corporate_event_borderline_with_explanation_and_snippet(self):
        """MKC case via LLM: BORDERLINE, reason carries the PT explanation
        AND the decisive headline as evidence."""
        e = _mk_entry("MKC", sector="Consumer Defensive")
        headline = "McCormick announces $44.8B merger with Unilever food business"
        def fetch(t, max_results=8):
            return [_FakeHeadline(headline)]
        hw.triage_watchlist(
            [e], dict(self._CTX), fetch_headlines_fn=fetch,
            classify_fn=_llm("corporate_event",
                             explanation="MKC anunciou fusão de $44.8B com a Unilever",
                             key_headline=headline),
        )
        assert e.triage_tier == "BORDERLINE"
        ev = [r for r in e.triage_reasons if "evento corporativo" in r]
        assert ev, f"no corporate-event reason: {e.triage_reasons}"
        assert "fusão de $44.8B" in ev[0], "reason must carry the explanation"
        assert "$44.8B merger" in ev[0], "reason must carry the headline snippet"
        assert e.drop_attribution["category"] == "corporate_event"

    def test_deterioration_borderline(self):
        e = _mk_entry("XYZ")
        headline = "XYZ slashes full-year guidance on customer churn"
        def fetch(t, max_results=8):
            return [_FakeHeadline(headline)]
        hw.triage_watchlist(
            [e], dict(self._CTX), fetch_headlines_fn=fetch,
            classify_fn=_llm("deterioration",
                             explanation="Corte de guidance anual por churn de clientes",
                             key_headline=headline),
        )
        assert e.triage_tier == "BORDERLINE"
        assert any("deterioração" in r for r in e.triage_reasons)

    def test_specific_fear_stays_clean_and_kills_keyword_noise(self):
        """ESTC case: headline contains 'cut' (keyword would flag) but the
        LLM judged specific_fear -> CLEAN, no event reason, attribution
        stored for display."""
        e = _mk_entry("ESTC")
        def fetch(t, max_results=8):
            # 'cut' would keyword-match; it's about ANOTHER company
            return [_FakeHeadline("Datadog price target cut by Morgan Stanley")]
        hw.triage_watchlist(
            [e], dict(self._CTX), fetch_headlines_fn=fetch,
            classify_fn=_llm("specific_fear",
                             explanation="Queda parece medo específico; headline é sobre a Datadog"),
        )
        assert e.triage_tier == "CLEAN"
        assert not any("event risk" in r for r in e.triage_reasons), \
            "keyword scan must NOT run when the LLM affirmatively classified"
        assert e.drop_attribution["category"] == "specific_fear"

    def test_llm_failure_falls_back_to_keywords(self):
        """classify_fn returns None (API down / disabled) -> keyword scan
        fires exactly as pre-LLM."""
        e = _mk_entry("MKC")
        def fetch(t, max_results=8):
            return [_FakeHeadline("McCormick announces $44.8B merger with Unilever")]
        hw.triage_watchlist([e], dict(self._CTX), fetch_headlines_fn=fetch,
                            classify_fn=_no_llm)
        assert e.triage_tier == "BORDERLINE"
        assert any("matched: merger" in r for r in e.triage_reasons)

    def test_llm_exception_falls_back_to_keywords(self):
        """classify_fn raising must not break triage — keyword fallback."""
        e = _mk_entry("MKC")
        def fetch(t, max_results=8):
            return [_FakeHeadline("McCormick announces $44.8B merger with Unilever")]
        def boom(*a, **k):
            raise RuntimeError("API down")
        hw.triage_watchlist([e], dict(self._CTX), fetch_headlines_fn=fetch,
                            classify_fn=boom)
        assert e.triage_tier == "BORDERLINE"
        assert any("matched: merger" in r for r in e.triage_reasons)

    def test_llm_never_touches_excluded(self):
        """EXCLUDED names skip headline fetch AND the classifier — the LLM
        can never override the quantitative exclusion rules."""
        e = _mk_entry("NVO")
        ctx = {"position_pct": {"NVO": 0.35}, "sector_cost_pct": {}}
        hw.triage_watchlist([e], ctx, fetch_headlines_fn=_no_headlines,
                            classify_fn=_llm_must_not_be_called)
        assert e.triage_tier == "EXCLUDED"

    def test_llm_not_called_without_headlines(self):
        e = _mk_entry("SPGI", sector_rel=-0.12)
        hw.triage_watchlist([e], dict(self._CTX),
                            fetch_headlines_fn=_no_headlines,
                            classify_fn=_llm_must_not_be_called)
        assert e.triage_tier == "CLEAN"

    def test_flag_without_key_headline_still_reasons(self):
        """Attribution without key_headline: reason is label + explanation,
        no dangling quotes."""
        e = _mk_entry("XYZ")
        def fetch(t, max_results=8):
            return [_FakeHeadline("Some headline")]
        hw.triage_watchlist(
            [e], dict(self._CTX), fetch_headlines_fn=fetch,
            classify_fn=_llm("deterioration", explanation="Guidance cortado"),
        )
        assert e.triage_tier == "BORDERLINE"
        r = [x for x in e.triage_reasons if "deterioração" in x][0]
        assert "Guidance cortado" in r
        assert '"' not in r  # no empty-snippet quotes


class TestLLMAttributionParsing:
    """_parse_response is the defensive boundary between the API and
    triage — anything unparseable must become None (keyword fallback)."""

    def test_valid_json(self):
        raw = ('{"category": "corporate_event", "explanation": "Fusão anunciada",'
               ' "key_headline": "X to merge with Y", "confidence": "high"}')
        r = la._parse_response(raw)
        assert r is not None
        assert r.category == "corporate_event"
        assert r.explanation == "Fusão anunciada"
        assert r.key_headline == "X to merge with Y"
        assert r.confidence == "high"

    def test_invalid_category_returns_none(self):
        raw = '{"category": "banana", "explanation": "", "key_headline": "", "confidence": "low"}'
        assert la._parse_response(raw) is None

    def test_malformed_json_returns_none(self):
        assert la._parse_response("not json at all {") is None

    def test_empty_returns_none(self):
        assert la._parse_response("") is None
        assert la._parse_response("   ") is None

    def test_non_dict_json_returns_none(self):
        assert la._parse_response('["corporate_event"]') is None

    def test_missing_fields_tolerated(self):
        """Schema guarantees the fields, but the parser must survive a
        degraded response (e.g. older SDK without output_config)."""
        r = la._parse_response('{"category": "unclear"}')
        assert r is not None
        assert r.category == "unclear"
        assert r.explanation == ""
        assert r.confidence == "low"
