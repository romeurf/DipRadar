"""tests/test_scoring_dispatch.py — ScoringSession + dispatch contract.

Per plan F.4 user review:
  1. Read DIPRADAR_USE_DISLOCATION flag ONCE per session (not per ticker).
     A mid-scan flag flip must not produce a mixed output.
  2. Tag every result with .scorer so 2-week side-by-side logs are
     unambiguous about which engine produced what.
  3. The same session backs /scan_now, /analisar, and the watchdog
     cron — one flag, all paths.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from core.scoring_dispatch import (
    get_active_scorer, ScoringSession, ScoreResult,
)


# ── Default + env semantics ──────────────────────────────────────────

class TestFlagSemantics:
    def test_default_is_dislocation(self, monkeypatch):
        monkeypatch.delenv("DIPRADAR_USE_DISLOCATION", raising=False)
        assert get_active_scorer() == "dislocation"

    def test_flag_1_is_dislocation(self, monkeypatch):
        monkeypatch.setenv("DIPRADAR_USE_DISLOCATION", "1")
        assert get_active_scorer() == "dislocation"

    def test_flag_0_is_composite(self, monkeypatch):
        monkeypatch.setenv("DIPRADAR_USE_DISLOCATION", "0")
        assert get_active_scorer() == "composite"

    def test_flag_false_string_is_composite(self, monkeypatch):
        # Accept common falsey string variants
        for v in ("false", "False", "FALSE"):
            monkeypatch.setenv("DIPRADAR_USE_DISLOCATION", v)
            assert get_active_scorer() == "composite", f"Failed for v={v!r}"


# ── Session freezes flag at construction (CORE F.4 requirement) ──────

class TestSessionFlagFrozen:
    def test_session_ignores_mid_scan_flag_flip(self, monkeypatch):
        """The user-mandated contract: flag is read ONCE at session
        construction. A subsequent env change must NOT switch the engine
        for already-running scans."""
        monkeypatch.setenv("DIPRADAR_USE_DISLOCATION", "1")
        session = ScoringSession()
        assert session.scorer == "dislocation"

        # Flag flips mid-scan (race, manual edit, env reload)
        monkeypatch.setenv("DIPRADAR_USE_DISLOCATION", "0")
        assert session.scorer == "dislocation", \
            "Session must freeze scorer at construction. Mid-scan flag flip leaked."

    def test_explicit_override_beats_env(self, monkeypatch):
        monkeypatch.setenv("DIPRADAR_USE_DISLOCATION", "0")
        session = ScoringSession(scorer="dislocation")
        assert session.scorer == "dislocation"

        monkeypatch.setenv("DIPRADAR_USE_DISLOCATION", "1")
        session = ScoringSession(scorer="composite")
        assert session.scorer == "composite"


# ── ScoreResult.scorer tag — log-auditability contract ───────────────

class TestScorerTag:
    def test_dislocation_session_tags_dislocation(self, monkeypatch):
        _stub_for_dislocation(monkeypatch)
        session = ScoringSession(scorer="dislocation")
        signals = _quality_pass_signals()
        r = session.score("XYZ", signals)
        assert r.scorer == "dislocation"

    def test_composite_session_tags_composite(self, monkeypatch):
        _stub_for_composite(monkeypatch)
        session = ScoringSession(scorer="composite")
        signals = _quality_pass_signals()
        r = session.score("XYZ", signals)
        assert r.scorer == "composite"


# ── Variant b propagation (BINDING gate enabler) ─────────────────────

class TestVariantPropagation:
    def test_variant_b_propagates_to_dislocation_call(self, monkeypatch):
        """Validation script constructs session with variant='b'; the
        underlying dislocation_for_ticker call MUST receive it."""
        _stub_for_dislocation(monkeypatch)
        from core.dislocation_score import DislocationBreakdown

        captured = {}

        def fake_dft(ticker, signals, **kw):
            captured["variant"] = kw.get("variant")
            captured["strict_pit"] = kw.get("strict_pit")
            captured["as_of_date"] = kw.get("as_of_date")
            return DislocationBreakdown(
                ticker=ticker, as_of_date=None, variant=kw.get("variant", "a"),
                hard_excluded=False, exclude_reason=None,
                sub_scores={}, composite_raw=0.0, composite_z=0.0,
                summary_line="stub",
            )

        monkeypatch.setattr("core.dislocation_score.dislocation_for_ticker", fake_dft)

        session = ScoringSession(scorer="dislocation", variant="b", strict_pit=True)
        from datetime import date
        session.score("XYZ", _quality_pass_signals(), as_of_date=date(2024, 3, 15))

        assert captured["variant"] == "b"
        assert captured["strict_pit"] is True
        assert captured["as_of_date"].isoformat() == "2024-03-15"


# ── explanation_lines unified across engines (display layer) ─────────

class TestExplanationLinesUnified:
    def test_dislocation_explanation_uses_sub_scores(self, monkeypatch):
        from core.dislocation_score import DislocationBreakdown, SubScore
        result = ScoreResult(
            ticker="NVO", scorer="dislocation", z=2.3,
            hard_excluded=False, exclude_reason=None,
            summary_line="NVO  +2.30z  | cheap + sector_drop -28%",
            breakdown=DislocationBreakdown(
                ticker="NVO", as_of_date=None, variant="a",
                hard_excluded=False, exclude_reason=None,
                sub_scores={
                    "valuation": SubScore(
                        name="valuation", value=0.8, weight=1.0,
                        raw_inputs={}, source="live",
                        explanation="P/E 15 vs p25 20",
                    ),
                    "drop_attribution": SubScore(
                        name="drop_attribution", value=0.9, weight=1.3,
                        raw_inputs={}, source="sector_rel",
                        explanation="ticker -28% vs sector +0%",
                    ),
                },
                composite_raw=0.0, composite_z=2.3, summary_line="stub",
            ),
        )
        lines = result.explanation_lines(threshold_abs=0.3)
        assert any("valuation" in l for l in lines)
        assert any("drop_attribution" in l for l in lines)

    def test_composite_explanation_uses_q_dd_val(self):
        from core.composite_score import CompositeBreakdown
        result = ScoreResult(
            ticker="NVO", scorer="composite", z=1.5,
            hard_excluded=False, exclude_reason=None,
            summary_line="NVO  +1.50z (composite legacy)",
            breakdown=CompositeBreakdown(
                ticker="NVO", horizon="CORE",
                composite_z=1.5, quality_z=0.5, dd_z=0.7, valuation_z=0.3,
                quality_raw=0.7, dd_raw=0.4, valuation_raw=0.1,
            ),
        )
        lines = result.explanation_lines()
        assert any("quality_z" in l for l in lines)
        assert any("dd_z" in l for l in lines)
        assert any("valuation_z" in l for l in lines)


# ── Helpers ──────────────────────────────────────────────────────────

def _quality_pass_signals():
    return {
        "sector":            "Healthcare",
        "roe":               0.40,
        "gross_margin":      0.80,
        "revenue_growth":    0.10,
        "fcf_yield_fund":    0.05,
        "is_investment_phase": False,
        "dd_from_52w_high":  -0.30,
        "return_20d":        -0.10,
        "price_vs_ma200w":   -0.15,
    }


def _stub_for_dislocation(monkeypatch):
    """Stub external calls so dislocation_for_ticker returns offline."""
    from utils.news_headlines import Headline
    from data.valuation_percentile import ValuationBreakdown
    from data.estimate_revisions import EstimateRevisions
    monkeypatch.setattr("utils.news_headlines.fetch_headlines",
                        lambda t, max_results=3, *, as_of_date=None: [])
    monkeypatch.setattr("data.sector_relative.sector_relative_pair",
                        lambda *a, **k: (-0.05, -0.02))
    monkeypatch.setattr("data.fundamental_signals.classify_recent_8k",
                        lambda *a, **k: float("nan"))
    monkeypatch.setattr("data.fundamental_signals.insider_sell_amount_score",
                        lambda *a, **k: (0.0, 0.0))
    monkeypatch.setattr("data.market_client.get_earnings_days",
                        lambda t: 30)
    monkeypatch.setattr(
        "data.estimate_revisions.estimate_revisions_for_ticker",
        lambda t: EstimateRevisions(
            ticker=t, score=0.0, source="missing",
            raw_eps_delta_pct=None, raw_beat_rate=None,
            explanation="stub",
        ),
    )
    monkeypatch.setattr(
        "data.valuation_percentile.valuation_percentile",
        lambda t, **k: ValuationBreakdown(
            ticker=t, current_fwd_pe=20.0,
            pe_hist_p10=18.0, pe_hist_p25=20.0, pe_hist_p50=22.0,
            pe_hist_p75=26.0, pe_hist_p90=30.0,
            pe_hist_n_samples=36,
            current_pe_percentile_in_own_3y=0.25,
            source="live",
            fetched_at="2024-03-15T12:00:00",
        ),
    )
    # Stub build_dislocation_stats to skip the universe walk
    monkeypatch.setattr(
        "core.dislocation_score.build_dislocation_stats",
        lambda force_refresh=False: {"mean": 0.0, "std": 1.0, "n": 1},
    )


def _stub_for_composite(monkeypatch):
    """Stub build_universe_stats for composite path."""
    monkeypatch.setattr(
        "core.composite_score.build_universe_stats",
        lambda force_refresh=False: {
            "quality":   {"mean": 0.5, "std": 0.15, "n": 100},
            "dd_core":   {"mean": 0.15, "std": 0.10, "n": 100},
            "dd_dip":    {"mean": 0.04, "std": 0.04, "n": 100},
            "valuation": {"mean": 0.0, "std": 0.10, "n": 100},
        },
    )
