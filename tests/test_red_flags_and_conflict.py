"""
Tests para Fase 1 (red flags + quality_multiplier) e Fase 2 (conflict_resolver).

Corre com: python3 tests/test_red_flags_and_conflict.py
(sem dependência de pytest)
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

# Garante import a partir do raiz do repo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from score import _detect_red_flags, calculate_score, format_score_v2_breakdown
from conflict_resolver import ConflictState, resolve_conflict


# ---------------------------------------------------------------------------
# Fase 1 — _detect_red_flags
# ---------------------------------------------------------------------------

class TestRedFlags(unittest.TestCase):

    def test_clean_company_no_flags(self):
        f = {
            "pe": 18.0,
            "fcf_yield": 0.05,
            "fcf_margin": 0.12,
            "roe": 0.18,
            "revenue_growth": 0.08,
            "debt_equity": 80.0,
        }
        mult, flags, preprofit = _detect_red_flags(f)
        self.assertEqual(mult, 1.0)
        self.assertEqual(flags, [])
        self.assertFalse(preprofit)

    def test_fcf_negative_triggers_preprofit(self):
        f = {"fcf_yield": -0.04, "pe": 50.0}
        mult, flags, preprofit = _detect_red_flags(f)
        self.assertTrue(preprofit)
        self.assertIn("FCF Negativo", flags)
        self.assertAlmostEqual(mult, 0.80, places=2)

    def test_fcf_margin_fallback(self):
        f = {"fcf_margin": -0.10, "pe": 30.0}
        mult, flags, preprofit = _detect_red_flags(f)
        self.assertTrue(preprofit)
        self.assertIn("FCF Negativo", flags)

    def test_pe_extreme_high(self):
        f = {"pe": 250.0, "fcf_yield": 0.03}
        mult, flags, preprofit = _detect_red_flags(f)
        self.assertFalse(preprofit)
        self.assertTrue(any("PE Extremo" in f for f in flags))
        self.assertAlmostEqual(mult, 0.75, places=2)

    def test_pe_extreme_negative(self):
        f = {"pe": -15.0, "fcf_yield": 0.03}
        mult, flags, preprofit = _detect_red_flags(f)
        self.assertTrue(any("PE Extremo" in f for f in flags))
        self.assertAlmostEqual(mult, 0.75, places=2)

    def test_pe_normal_no_flag(self):
        f = {"pe": 25.0, "fcf_yield": 0.03}
        mult, flags, preprofit = _detect_red_flags(f)
        self.assertEqual(mult, 1.0)
        self.assertEqual(flags, [])

    def test_roe_neg_with_low_growth(self):
        f = {"roe": -0.10, "revenue_growth": 0.10, "fcf_yield": 0.05}
        mult, flags, preprofit = _detect_red_flags(f)
        self.assertIn("ROE Negativo s/ Crescimento Forte", flags)
        self.assertAlmostEqual(mult, 0.70, places=2)

    def test_roe_neg_with_high_growth_no_flag(self):
        # Crescimento >= 30% absorve o ROE negativo (high-growth tech típico)
        f = {"roe": -0.10, "revenue_growth": 0.45, "fcf_yield": 0.05}
        mult, flags, preprofit = _detect_red_flags(f)
        self.assertNotIn("ROE Negativo s/ Crescimento Forte", flags)

    def test_debt_equity_extreme(self):
        f = {"debt_equity": 450.0, "fcf_yield": 0.03}
        mult, flags, preprofit = _detect_red_flags(f)
        self.assertTrue(any("D/E Elevado" in f for f in flags))
        self.assertAlmostEqual(mult, 0.80, places=2)

    def test_letal_combination(self):
        # FCF negativo + PE extremo → FCF (-0.20) + PE (-0.25) + Letal (-0.15) = 0.40
        f = {"pe": 250.0, "fcf_yield": -0.04}
        mult, flags, preprofit = _detect_red_flags(f)
        self.assertTrue(preprofit)
        self.assertIn("FCF Negativo", flags)
        self.assertTrue(any("PE Extremo" in f for f in flags))
        self.assertTrue(any("Letal" in f for f in flags))
        self.assertAlmostEqual(mult, 0.40, places=2)

    def test_min_clip_at_010(self):
        # Todos os flags juntos: -0.20 -0.25 -0.30 -0.20 -0.15 = -1.10 → clip 0.10
        f = {
            "pe": 250.0,
            "fcf_yield": -0.05,
            "roe": -0.20,
            "revenue_growth": 0.05,
            "debt_equity": 500.0,
        }
        mult, flags, preprofit = _detect_red_flags(f)
        self.assertGreaterEqual(mult, 0.10)
        self.assertLessEqual(mult, 0.15)

    def test_nan_values_ignored(self):
        f = {"pe": float("nan"), "fcf_yield": float("nan"), "roe": float("nan")}
        mult, flags, preprofit = _detect_red_flags(f)
        self.assertEqual(mult, 1.0)
        self.assertEqual(flags, [])
        self.assertFalse(preprofit)

    def test_none_values_ignored(self):
        f = {"pe": None, "fcf_yield": None, "roe": None, "debt_equity": None}
        mult, flags, preprofit = _detect_red_flags(f)
        self.assertEqual(mult, 1.0)
        self.assertEqual(flags, [])


# ---------------------------------------------------------------------------
# Fase 1 — integration with calculate_score
# ---------------------------------------------------------------------------

class TestCalculateScoreWithFlags(unittest.TestCase):

    def test_clean_score_unaffected(self):
        f = {
            "roic": 0.22, "fcf_margin": 0.14, "fcf_yield": 0.06,
            "revenue_growth": 0.12, "debt_equity": 45.0, "pe": 18.0,
            "rsi": 28.0, "drawdown_from_high": -32.0, "sector": "Technology",
            "roe": 0.18,
        }
        res = calculate_score(f, ml_prob=0.85)
        self.assertEqual(res["quality_multiplier"], 1.0)
        self.assertEqual(res["red_flags"], [])
        self.assertFalse(res["is_preprofit"])
        self.assertGreaterEqual(res["final_score"], 45.0)

    def test_rklb_like_profile_score_collapses(self):
        # Pre-profit + PE distorcido + alto crescimento (típico aerospace/biotech early)
        f = {
            "roic": -0.05, "fcf_margin": -0.18, "fcf_yield": -0.04,
            "revenue_growth": 0.78, "debt_equity": 180.0, "pe": 250.0,
            "rsi": 32.0, "drawdown_from_high": -45.0, "sector": "Industrials",
            "roe": -0.30,
        }
        res = calculate_score(f, ml_prob=0.65)
        self.assertTrue(res["is_preprofit"])
        self.assertTrue(res["skip_recommended"])
        self.assertIn("FCF Negativo", res["red_flags"])
        # Score deve afundar para zona "EVITAR" (<55)
        self.assertLess(res["final_score"], 30.0)

    def test_preprofit_forces_skip(self):
        # FCF negativo sozinho (sem outros red flags) → ainda força skip
        f = {
            "roic": 0.10, "fcf_margin": -0.05, "fcf_yield": -0.02,
            "revenue_growth": 0.40, "debt_equity": 60.0, "pe": 80.0,
            "rsi": 40.0, "drawdown_from_high": -20.0, "sector": "Technology",
        }
        res = calculate_score(f, ml_prob=0.70)
        self.assertTrue(res["is_preprofit"])
        self.assertTrue(res["skip_recommended"])

    def test_format_breakdown_shows_red_flags(self):
        f = {"pe": 250.0, "fcf_yield": -0.04, "roe": -0.20, "revenue_growth": 0.05}
        res = calculate_score(f)
        msg = format_score_v2_breakdown(res)
        self.assertIn("Red Flags", msg)
        self.assertIn("FCF Negativo", msg)
        self.assertIn("pré-lucro", msg)


# ---------------------------------------------------------------------------
# Fase 2 — resolve_conflict
# ---------------------------------------------------------------------------

class TestConflictResolver(unittest.TestCase):

    def _rc(self, *args, **kwargs):
        """Helper: desempacota os 4 valores de resolve_conflict → (state, msg)."""
        state, msg, _emoji, _sizing = resolve_conflict(*args, **kwargs)
        return state, msg

    def test_consensus_bull_high_score_winning(self):
        state, msg = self._rc(80, "WIN_STRONG")
        self.assertEqual(state, ConflictState.CONSENSUS_BULL)
        self.assertIn("forte", msg.lower())

    def test_consensus_bull_threshold_65(self):
        state, _ = self._rc(65, "WIN")
        self.assertEqual(state, ConflictState.CONSENSUS_BULL)

    def test_consensus_bear_low_score_no_win(self):
        state, msg = self._rc(40, "NO_WIN")
        self.assertEqual(state, ConflictState.CONSENSUS_BEAR)

    def test_consensus_bear_threshold_below_50(self):
        # threshold CONSENSUS_BEAR é fund_score < 50 (docstring)
        state, _ = self._rc(49, "WEAK")
        self.assertEqual(state, ConflictState.CONSENSUS_BEAR)

    def test_technical_only_low_fund_high_ml(self):
        # fund < 55, ml bull, sem dislocation → TECHNICAL_ONLY
        state, msg = self._rc(45, "WIN")
        self.assertEqual(state, ConflictState.TECHNICAL_ONLY)
        self.assertIn("especula", msg.lower())

    def test_technical_with_dislocation(self):
        # fund < 55, ml bull, com dislocation → TECHNICAL_WITH_DISLOCATION
        state, msg = self._rc(45, "WIN", has_dislocation=True)
        self.assertEqual(state, ConflictState.TECHNICAL_WITH_DISLOCATION)
        self.assertIn("dislocation", msg.lower())

    def test_fundamental_only_high_fund_low_ml(self):
        # fund >= 65, ml não-bull → FUNDAMENTAL_ONLY
        state, msg = self._rc(75, "NO_WIN")
        self.assertEqual(state, ConflictState.FUNDAMENTAL_ONLY)
        self.assertIn("dca", msg.lower())

    def test_neutral_grey_zone_60(self):
        # fund=60: nem >= 65 nem < 55 → NEUTRAL
        state, _ = self._rc(60, "WIN")
        self.assertEqual(state, ConflictState.NEUTRAL)

    def test_neutral_grey_zone_55_to_65(self):
        # fund=58: zona cinzenta
        state, _ = self._rc(58, "WEAK")
        self.assertEqual(state, ConflictState.NEUTRAL)

    def test_no_ml_label_treated_as_bear(self):
        # ml_label=None com score baixo → CONSENSUS_BEAR
        state, _ = self._rc(40, None)
        self.assertEqual(state, ConflictState.CONSENSUS_BEAR)

    def test_no_model_treated_as_bear(self):
        state, _ = self._rc(40, "NO_MODEL")
        self.assertEqual(state, ConflictState.CONSENSUS_BEAR)

    def test_win_40_counts_as_bull(self):
        state, _ = self._rc(70, "WIN_40")
        self.assertEqual(state, ConflictState.CONSENSUS_BULL)


# ---------------------------------------------------------------------------
# Fase 3 — format_score_v2_breakdown verdict integration
# ---------------------------------------------------------------------------

class TestFormatBreakdownWithVerdict(unittest.TestCase):

    def test_verdict_appears_in_output(self):
        f = {
            "roic": 0.22, "fcf_margin": 0.14, "fcf_yield": 0.06,
            "revenue_growth": 0.12, "debt_equity": 45.0, "pe": 18.0,
            "rsi": 28.0, "drawdown_from_high": -32.0, "sector": "Technology",
        }
        res = calculate_score(f, ml_prob=0.85)
        state, msg, _emoji, _sizing = resolve_conflict(res["final_score"], "WIN")
        out = format_score_v2_breakdown(res, conflict_state=state, conflict_msg=msg)
        self.assertIn("Veredicto", out)
        self.assertIn(state.value, out)

    def test_verdict_omitted_without_conflict_args(self):
        f = {"roic": 0.10, "rsi": 50.0}
        res = calculate_score(f)
        out = format_score_v2_breakdown(res)
        self.assertNotIn("Veredicto", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
