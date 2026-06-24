"""tests/test_dislocation_score.py — unit tests for the rule-based scorer.

Coverage priorities (per plan §F.3 and user review):
  1. PIT correctness for sub-score #3 (HIGHEST priority — it's the
     heaviest-weighted signal in variant b and consumes headlines)
  2. PIT correctness for sub-scores #2 and #7
  3. Relief-keyword context guard in #3 (false positives matter most
     here since #3 carries the gate)
  4. strict_pit mode raises when as_of_date missing
  5. Variant (b) forces sub-score #1 weight to 0
  6. Quality hard gate triggers exclusion
  7. Per-sub-scorer formula behavior with synthetic signals

Tests avoid network: monkeypatch fetch_headlines, classify_recent_8k,
estimate_revisions_for_ticker, sector_relative_pair, valuation_percentile,
get_earnings_days, insider_sell_amount_score, scan_core_universe.

Synthetic Headline objects feed the news path.
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from core import dislocation_score as ds


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_signals_quality_pass():
    """A signals dict that passes the quality gate (real compounder-like)."""
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


@pytest.fixture
def synthetic_signals_quality_fail():
    """A signals dict that FAILS the quality gate (revenue collapsing)."""
    return {
        "sector":            "Healthcare",
        "roe":               -0.20,
        "gross_margin":      0.40,
        "revenue_growth":    -0.25,  # < -10% → gate fail
        "fcf_yield_fund":    -0.05,
        "is_investment_phase": False,
        "dd_from_52w_high":  -0.50,
    }


def _make_headline(title: str, days_ago_from: date):
    """Helper to make a Headline-like object the scorer can consume."""
    from utils.news_headlines import Headline
    iso = days_ago_from.isoformat() + "T12:00:00"
    return Headline(title=title, publisher="Test", published_at=iso, url="")


# ── PIT correctness tests (BLOCKING per user review) ─────────────────

class TestPITCorrectness:
    """Verifies that as_of_date filtering works in every sub-scorer that
    touches headlines or filings. Without this, variant (b) is not
    PIT-clean and the binding gate is contaminated by look-ahead."""

    def test_sub3_excludes_headlines_after_as_of_date(self, monkeypatch, synthetic_signals_quality_pass):
        """#3 (drop attribution) bad-news subtraction must NOT see headlines
        published after as_of_date. This is the highest-stakes PIT check —
        #3 carries the most weight in variant (b) and uses headlines."""
        as_of = date(2024, 3, 15)
        from datetime import timedelta
        # Synthetic: ONE bad-news headline 3 days AFTER as_of_date.
        # Without PIT filter, the -0.5 penalty would fire. With PIT, must not.
        future_bad_headline = _make_headline("Company misses earnings badly",
                                              as_of + timedelta(days=3))
        past_neutral_headline = _make_headline("Analyst raises price target",
                                                as_of - timedelta(days=5))

        captured_kwargs = {}

        def fake_fetch_headlines(ticker, max_results=3, *, as_of_date=None):
            captured_kwargs["as_of_date"] = as_of_date
            # Simulate proper PIT filter behavior (what fetch_headlines really does)
            all_headlines = [future_bad_headline, past_neutral_headline]
            if as_of_date is None:
                return all_headlines[:max_results]
            return [h for h in all_headlines
                    if date.fromisoformat(h.published_at[:10]) <= as_of_date][:max_results]

        monkeypatch.setattr("utils.news_headlines.fetch_headlines",
                            fake_fetch_headlines)
        monkeypatch.setattr("data.sector_relative.sector_relative_pair",
                            lambda *a, **k: (-0.10, -0.05))  # ticker underperformed
        monkeypatch.setattr("data.fundamental_signals.classify_recent_8k",
                            lambda *a, **k: float("nan"))

        sub = ds._sub_drop_attribution(
            "XYZ", synthetic_signals_quality_pass, as_of_date=as_of,
        )

        assert captured_kwargs["as_of_date"] == as_of, \
            "fetch_headlines must be called with as_of_date for PIT"
        bad_news_hits = sub.raw_inputs["bad_news_hits"]
        bad_titles = [h["title"] for h in bad_news_hits]
        assert "Company misses earnings badly" not in str(bad_titles), \
            f"PIT VIOLATION: future bad-news headline leaked into score. " \
            f"bad_news_hits={bad_news_hits}"

    def test_sub3_includes_headlines_before_as_of_date(self, monkeypatch, synthetic_signals_quality_pass):
        """Mirror: a bad-news headline BEFORE as_of_date must fire the penalty."""
        as_of = date(2024, 3, 15)
        from datetime import timedelta
        # Headline 5 days before as_of, within 15d window
        past_bad_headline = _make_headline("Company misses earnings",
                                            as_of - timedelta(days=5))

        def fake_fetch_headlines(ticker, max_results=3, *, as_of_date=None):
            if as_of_date is None:
                return [past_bad_headline][:max_results]
            return [h for h in [past_bad_headline]
                    if date.fromisoformat(h.published_at[:10]) <= as_of_date][:max_results]

        monkeypatch.setattr("utils.news_headlines.fetch_headlines",
                            fake_fetch_headlines)
        monkeypatch.setattr("data.sector_relative.sector_relative_pair",
                            lambda *a, **k: (-0.10, -0.05))
        monkeypatch.setattr("data.fundamental_signals.classify_recent_8k",
                            lambda *a, **k: float("nan"))

        sub = ds._sub_drop_attribution(
            "XYZ", synthetic_signals_quality_pass, as_of_date=as_of,
        )
        bad_news_hits = sub.raw_inputs["bad_news_hits"]
        assert len(bad_news_hits) == 1, \
            f"Past bad-news headline should fire penalty. Got: {bad_news_hits}"

    def test_sub2_excludes_headlines_after_as_of_date(self, monkeypatch):
        """Same PIT discipline for guidance sub-score."""
        as_of = date(2024, 3, 15)
        from datetime import timedelta
        future_cut = _make_headline("Company cuts guidance for FY26",
                                     as_of + timedelta(days=3))
        past_neutral = _make_headline("New product launched",
                                       as_of - timedelta(days=5))

        def fake_fetch_headlines(ticker, max_results=3, *, as_of_date=None):
            all_h = [future_cut, past_neutral]
            if as_of_date is None:
                return all_h[:max_results]
            return [h for h in all_h
                    if date.fromisoformat(h.published_at[:10]) <= as_of_date][:max_results]

        monkeypatch.setattr("utils.news_headlines.fetch_headlines",
                            fake_fetch_headlines)
        monkeypatch.setattr("data.fundamental_signals.classify_recent_8k",
                            lambda *a, **k: float("nan"))

        sub = ds._sub_guidance("XYZ", as_of_date=as_of)
        hit_titles = [h["title"] for h in sub.raw_inputs["headline_hits"]]
        assert "Company cuts guidance for FY26" not in str(hit_titles), \
            "PIT VIOLATION: future guidance-cut headline leaked into sub #2"

    def test_sub7_excludes_headlines_after_as_of_date(self, monkeypatch):
        """Governance sub-score must respect PIT for mgmt-departure headlines."""
        as_of = date(2024, 3, 15)
        from datetime import timedelta
        future_depart = _make_headline("CEO resigns effective immediately",
                                        as_of + timedelta(days=3))

        def fake_fetch_headlines(ticker, max_results=3, *, as_of_date=None):
            if as_of_date is None:
                return [future_depart][:max_results]
            return [h for h in [future_depart]
                    if date.fromisoformat(h.published_at[:10]) <= as_of_date][:max_results]

        monkeypatch.setattr("utils.news_headlines.fetch_headlines",
                            fake_fetch_headlines)
        monkeypatch.setattr("data.fundamental_signals.classify_recent_8k",
                            lambda *a, **k: float("nan"))
        monkeypatch.setattr("data.fundamental_signals.insider_sell_amount_score",
                            lambda *a, **k: (0.0, 0.0))

        sub = ds._sub_governance("XYZ", as_of_date=as_of)
        assert "mgmt_departure_hits" not in sub.raw_inputs, \
            f"PIT VIOLATION: future mgmt-departure headline leaked into #7. " \
            f"raw_inputs={sub.raw_inputs}"


# ── strict_pit guard (footgun closer) ────────────────────────────────

class TestStrictPITGuard:
    """The strict_pit=True mode is the validation script's safety net.
    Without it, forgetting to pass as_of_date in a backtest would silently
    revert to production mode (all data visible) = look-ahead."""

    def test_strict_pit_raises_when_as_of_missing(self, synthetic_signals_quality_pass):
        with pytest.raises(ValueError, match="strict_pit"):
            ds.dislocation_for_ticker(
                "XYZ", synthetic_signals_quality_pass,
                strict_pit=True,
                as_of_date=None,  # forgotten by mistake
            )

    def test_strict_pit_ok_when_as_of_set(self, synthetic_signals_quality_pass, monkeypatch):
        # Stub all external calls so the test runs offline.
        _stub_all_external(monkeypatch)
        b = ds.dislocation_for_ticker(
            "XYZ", synthetic_signals_quality_pass,
            strict_pit=True,
            as_of_date=date(2024, 3, 15),
        )
        assert b.ticker == "XYZ"


# ── Relief-keyword context guard for #3 (user-mandated) ──────────────

class TestReliefKeywordGuard:
    """User feedback: 'fears of a miss are overblown' has the keyword 'miss'
    but is REASSURING. The relief-keyword guard must negate the penalty."""

    @pytest.mark.parametrize("title,should_penalize", [
        ("Company misses earnings",                           True),
        ("Company misses earnings — analysts say overblown",  False),  # relief
        ("CEO addresses miss concerns, no wrongdoing found",  False),  # relief
        ("Cleared in SEC probe over alleged fraud",           False),  # relief
        ("SEC fraud investigation deepens",                   True),
        ("Class-action lawsuit dismissed",                    False),  # relief
        ("Trial failed — drug ineffective",                   True),
    ])
    def test_relief_kw_negates_bad_news(self, title, should_penalize):
        """Direct test of _has_bad_news for the context guard logic."""
        result = ds._has_bad_news(title.lower())
        assert result == should_penalize, \
            f"For title={title!r}: expected has_bad_news={should_penalize}, got {result}"

    def test_relief_kw_in_full_subscore(self, monkeypatch, synthetic_signals_quality_pass):
        """End-to-end: relief headline must NOT add the -0.5 penalty in #3."""
        as_of = date(2024, 3, 15)
        from datetime import timedelta
        # Both have 'miss' keyword. Relief headline should be filtered out.
        bad = _make_headline("Company misses Q1 earnings",
                              as_of - timedelta(days=2))
        relief = _make_headline("Miss concerns overblown say analysts",
                                 as_of - timedelta(days=4))

        def fake_fetch(ticker, max_results=3, *, as_of_date=None):
            return [bad, relief][:max_results]

        monkeypatch.setattr("utils.news_headlines.fetch_headlines", fake_fetch)
        monkeypatch.setattr("data.sector_relative.sector_relative_pair",
                            lambda *a, **k: (-0.10, -0.05))
        monkeypatch.setattr("data.fundamental_signals.classify_recent_8k",
                            lambda *a, **k: float("nan"))

        sub = ds._sub_drop_attribution(
            "XYZ", synthetic_signals_quality_pass, as_of_date=as_of,
        )
        # 'bad' fires penalty; 'relief' goes to relief_hits and does NOT
        # subtract additionally. The score should be (base - 0.5).
        bad_count = len(sub.raw_inputs["bad_news_hits"])
        relief_count = len(sub.raw_inputs["relief_hits"])
        assert bad_count == 1, f"Expected 1 bad-news hit; got {bad_count}"
        assert relief_count == 1, f"Expected 1 relief hit; got {relief_count}"


# ── Variant (b) — signal #1 zeroed ───────────────────────────────────

class TestVariantB:
    """Variant b is the BINDING gate. Signal #1 weight must be 0 so that
    look-ahead in the estimate-revisions data cannot inflate the score."""

    def test_variant_b_zeroes_signal_1_weight(self, monkeypatch, synthetic_signals_quality_pass):
        _stub_all_external(monkeypatch)

        b_var_a = ds.dislocation_for_ticker(
            "XYZ", synthetic_signals_quality_pass, variant="a",
            stats={"mean": 0.0, "std": 1.0},
        )
        b_var_b = ds.dislocation_for_ticker(
            "XYZ", synthetic_signals_quality_pass, variant="b",
            stats={"mean": 0.0, "std": 1.0},
        )

        assert b_var_a.sub_scores["estimate_revisions"].weight > 0
        assert b_var_b.sub_scores["estimate_revisions"].weight == 0.0
        # Other weights identical
        for k in ["guidance", "drop_attribution", "catalyst", "valuation", "quality", "governance"]:
            assert b_var_a.sub_scores[k].weight == b_var_b.sub_scores[k].weight, \
                f"Weight for {k} should be identical in both variants"


# ── Hard quality gate ────────────────────────────────────────────────

class TestQualityGate:
    def test_revenue_collapse_excludes(self, synthetic_signals_quality_fail, monkeypatch):
        _stub_all_external(monkeypatch)
        b = ds.dislocation_for_ticker(
            "XYZ", synthetic_signals_quality_fail,
            stats={"mean": 0.0, "std": 1.0},
        )
        assert b.hard_excluded
        assert "revenue_growth" in b.exclude_reason

    def test_quality_pass_does_not_exclude(self, synthetic_signals_quality_pass, monkeypatch):
        _stub_all_external(monkeypatch)
        b = ds.dislocation_for_ticker(
            "XYZ", synthetic_signals_quality_pass,
            stats={"mean": 0.0, "std": 1.0},
        )
        assert not b.hard_excluded


# ── Sub-scorer formulas (sanity, no network) ─────────────────────────

class TestSubScorerFormulas:
    def test_catalyst_sweet_spot(self, monkeypatch):
        # 30d to earnings → in sweet spot 21-42d → score +1.0
        monkeypatch.setattr("data.market_client.get_earnings_days",
                            lambda t: 30)
        s = ds._sub_catalyst("XYZ")
        assert s.value == 1.0

    def test_catalyst_stuck(self, monkeypatch):
        # No earnings → -0.2 penalty
        monkeypatch.setattr("data.market_client.get_earnings_days",
                            lambda t: None)
        s = ds._sub_catalyst("XYZ")
        assert s.value == -0.2

    def test_catalyst_ramping_up(self, monkeypatch):
        # 10d → 0.5 (halfway through ramp)
        monkeypatch.setattr("data.market_client.get_earnings_days",
                            lambda t: 10)
        s = ds._sub_catalyst("XYZ")
        assert s.value == 0.5


# ── Test helpers ─────────────────────────────────────────────────────

def _stub_all_external(monkeypatch):
    """Stub every external call so dislocation_for_ticker runs offline."""
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
