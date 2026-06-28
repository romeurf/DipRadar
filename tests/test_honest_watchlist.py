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
    def test_pass(self):
        sigs = {"revenue_growth": 0.10, "gross_margin": 0.50, "fcf_yield_fund": 0.05}
        ok, reason = hw._quality_passes(sigs)
        assert ok and reason == ""

    def test_fail_revenue_collapse(self):
        sigs = {"revenue_growth": -0.20, "gross_margin": 0.50, "fcf_yield_fund": 0.05}
        ok, reason = hw._quality_passes(sigs)
        assert not ok and "revenue_growth" in reason

    def test_fail_low_gm(self):
        sigs = {"revenue_growth": 0.10, "gross_margin": 0.15, "fcf_yield_fund": 0.05}
        ok, reason = hw._quality_passes(sigs)
        assert not ok and "gross_margin" in reason

    def test_fail_fcf_unless_investment_phase(self):
        sigs = {"revenue_growth": 0.10, "gross_margin": 0.50, "fcf_yield_fund": -0.05}
        ok, reason = hw._quality_passes(sigs)
        assert not ok and "fcf" in reason.lower()

    def test_investment_phase_exempts_fcf(self):
        sigs = {"revenue_growth": 0.10, "gross_margin": 0.50,
                "fcf_yield_fund": -0.05, "is_investment_phase": True}
        ok, _ = hw._quality_passes(sigs)
        assert ok

    def test_all_signals_missing_excludes(self):
        """Data sufficiency requirement: at least one of {roe, gross_margin,
        revenue_growth} must be present. Without ANY signal we can't assert
        quality — exclude with a clear reason. This prevents non-CORE
        tickers from DIP alert union (which often lack fundamentals data)
        from sliding through the gate silently."""
        ok, reason = hw._quality_passes({})
        assert not ok
        assert "no quality signals" in reason.lower()

    def test_one_signal_present_passes(self):
        """At least one signal is enough to assert quality (when that
        signal passes its threshold). E.g. just ROE with no GM/rev_growth."""
        ok, _ = hw._quality_passes({"roe": 0.35})
        assert ok


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

    def test_quality_fail_excludes_via_hard_excluded(self, monkeypatch):
        """Hard quality fail means not in watchlist regardless of DD/catalyst."""
        self._stub_externals(monkeypatch)
        sigs = {
            "revenue_growth": -0.30,  # collapse
            "gross_margin": 0.85, "fcf_yield_fund": 0.04,
            "dd_from_52w_high": -0.44,
        }
        e = hw.evaluate_ticker("BAD", sigs)
        assert not e.qualifies
        assert not e.quality_passes
        assert "revenue_growth" in e.quality_reason


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
