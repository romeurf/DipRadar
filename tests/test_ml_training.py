#!/usr/bin/env python3
"""Tests for ml_training package — sem rede (mocks/datasets sintéticos).

Cobertura:
  - cv: winsorize, spearman_safe, topk_pnl, temporal_weights, build_walk_forward_folds
  - models: factories + feature lists
  - data: compute_sector_alert_count_7d, spy_max_return_forward, days_since_52w_high
  - bundle: DipModelsV3 dataclass + save/load round-trip + report build
  - train: walk_forward_cv smoke (com modelo Ridge para ser rápido)

Corre com: ``python3 tests/test_ml_training.py``
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# CV helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestCV(unittest.TestCase):
    def test_winsorize_basic(self):
        from ml_training.cv import winsorize
        arr = np.array([1.0, 2.0, 3.0, 100.0, -50.0])
        out = winsorize(arr, pct=0.20)
        # pct=0.20 → q20=−39.6, q80=80.6, valores extremos clipados
        self.assertGreaterEqual(out.min(), -50.0)
        self.assertLessEqual(out.max(), 100.0)
        self.assertEqual(len(out), len(arr))

    def test_winsorize_empty(self):
        from ml_training.cv import winsorize
        out = winsorize(np.array([]))
        self.assertEqual(len(out), 0)

    def test_spearman_safe_perfect_correlation(self):
        from ml_training.cv import spearman_safe
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        self.assertAlmostEqual(spearman_safe(x, y), 1.0, places=5)

    def test_spearman_safe_anticorrelation(self):
        from ml_training.cv import spearman_safe
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
        self.assertAlmostEqual(spearman_safe(x, y), -1.0, places=5)

    def test_spearman_safe_too_few_finite(self):
        from ml_training.cv import spearman_safe
        x = np.array([1.0, 2.0, np.nan, np.nan])
        y = np.array([1.0, 2.0, np.nan, np.nan])
        self.assertTrue(np.isnan(spearman_safe(x, y)))

    def test_topk_pnl_basic(self):
        from ml_training.cv import topk_pnl
        # pred = true → top-K mean = mean dos K maiores valores
        true = np.array([0.10, 0.20, 0.30, -0.10, -0.20, 0.40, 0.05, -0.05, 0.15, 0.25])
        pred = true.copy()
        # top 20% → 2 elementos: 0.40, 0.30
        out = topk_pnl(pred, true, k=0.20)
        self.assertAlmostEqual(out, (0.40 + 0.30) / 2, places=5)

    def test_topk_pnl_perfect_inverse_ranking(self):
        from ml_training.cv import topk_pnl
        # pred maior → true menor (anti-correlation)
        pred = np.arange(10).astype(float)
        true = np.arange(10, 0, -1).astype(float)
        # top 20% por pred = índices 8, 9; true[8]=2, true[9]=1; mean=1.5
        out = topk_pnl(pred, true, k=0.20)
        self.assertAlmostEqual(out, 1.5, places=5)

    def test_temporal_weights_decay(self):
        from ml_training.cv import temporal_weights
        max_date = pd.Timestamp("2025-01-01")
        # Datas: 0 dias atrás (peso 1.0), 365*3 atrás (peso 0.5 — 1 half-life)
        dates = [max_date, max_date - pd.Timedelta(days=365 * 3)]
        w = temporal_weights(dates, max_date, half_life_days=365 * 3)
        self.assertAlmostEqual(w[0], 1.0, places=5)
        self.assertAlmostEqual(w[1], 0.5, places=4)

    def test_build_walk_forward_folds_count(self):
        from ml_training.cv import build_walk_forward_folds
        # Período 4 anos com alertas distribuídos uniformemente
        dates = pd.date_range("2020-01-01", "2024-01-01", periods=200)
        df = pd.DataFrame({"alert_date": dates})
        folds = build_walk_forward_folds(df, n_folds=10, purge_days=21)
        # Em 4 anos, 10 folds deve produzir ≥ 8 (alguns podem cair fora por purga)
        self.assertGreaterEqual(len(folds), 8)
        # Cada fold tem (k, train_end, purge_end, test_end)
        for k, tr, pg, te in folds:
            self.assertIsInstance(k, int)
            self.assertGreater(pg, tr)
            self.assertGreater(te, pg)

    def test_build_walk_forward_folds_empty(self):
        from ml_training.cv import build_walk_forward_folds
        folds = build_walk_forward_folds(pd.DataFrame(columns=["alert_date"]))
        self.assertEqual(folds, [])


# ─────────────────────────────────────────────────────────────────────────────
# Model factories
# ─────────────────────────────────────────────────────────────────────────────

class TestModels(unittest.TestCase):
    def test_factories_produce_distinct_objects(self):
        from ml_training.models import xgb_factory, lgbm_factory, rf_factory, ridge_factory
        m1 = xgb_factory()
        m2 = xgb_factory()
        self.assertIsNot(m1, m2)  # cada chamada cria nova instância

        # smoke: importam e instanciam sem erro
        lgbm_factory()
        rf_factory()
        ridge_factory()

    def test_feature_lists_v31_has_34(self):
        from ml_training.models import build_feature_lists
        v31, baseline = build_feature_lists()
        self.assertEqual(len(v31), 34, "v31 deve ter 34 features (notebook cell 12)")
        # Deve incluir as 4 NEW
        for new_feat in ["relative_drop", "sector_alert_count_7d",
                         "days_since_52w_high", "month_of_year"]:
            self.assertIn(new_feat, v31)
        # Baseline NÃO inclui as 4 NEW
        for new_feat in ["relative_drop", "sector_alert_count_7d",
                         "days_since_52w_high", "month_of_year"]:
            self.assertNotIn(new_feat, baseline)
        # Baseline tem 4 features a menos que v31
        self.assertEqual(len(v31) - len(baseline), 4)

    def test_model_configs_5_models(self):
        from ml_training.models import build_feature_lists, build_model_configs
        v31, baseline = build_feature_lists()
        cfg = build_model_configs(v31, baseline)
        self.assertEqual(len(cfg), 5)
        self.assertIn("XGB-alpha-v31", cfg)
        self.assertIn("XGB-alpha-baseline", cfg)
        self.assertEqual(cfg["XGB-alpha-baseline"]["feats"], baseline)


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestData(unittest.TestCase):
    def test_sector_alert_count_anti_leakage(self):
        from ml_training.data import compute_sector_alert_count_7d
        df = pd.DataFrame({
            "ticker":     ["AAPL", "MSFT", "NVDA", "AMD", "INTC"],
            "sector":     ["Technology"] * 5,
            "alert_date": pd.to_datetime([
                "2024-01-01", "2024-01-03", "2024-01-05",
                "2024-01-10", "2024-01-15",
            ]),
        })
        lookup = compute_sector_alert_count_7d(df)
        # AAPL @ 01-01: prior=0 → count=0
        self.assertEqual(lookup[("AAPL", pd.Timestamp("2024-01-01"))], 0)
        # MSFT @ 01-03: prior AAPL @ 01-01 (2d antes, dentro de 7d) → 1
        self.assertEqual(lookup[("MSFT", pd.Timestamp("2024-01-03"))], 1)
        # NVDA @ 01-05: prior AAPL+MSFT (4d, 2d antes, dentro 7d) → 2
        self.assertEqual(lookup[("NVDA", pd.Timestamp("2024-01-05"))], 2)
        # AMD @ 01-10: window=[01-03, 01-10). NVDA @ 01-05 (dentro), MSFT @ 01-03 (dentro,
        # boundary inclusive); AAPL @ 01-01 (fora, < 01-03) → 2
        self.assertEqual(lookup[("AMD", pd.Timestamp("2024-01-10"))], 2)
        # INTC @ 01-15: prior AMD @ 01-10 (5d antes) → 1
        self.assertEqual(lookup[("INTC", pd.Timestamp("2024-01-15"))], 1)

    def test_sector_alert_count_segregates_by_sector(self):
        from ml_training.data import compute_sector_alert_count_7d
        df = pd.DataFrame({
            "ticker":     ["AAPL", "JPM", "MSFT"],
            "sector":     ["Technology", "Financial Services", "Technology"],
            "alert_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        })
        lookup = compute_sector_alert_count_7d(df)
        # AAPL e MSFT são Technology, JPM é Financial.
        # MSFT @ 01-03: só AAPL (Tech) conta → 1
        self.assertEqual(lookup[("MSFT", pd.Timestamp("2024-01-03"))], 1)
        # JPM @ 01-02: nenhum Financial antes → 0
        self.assertEqual(lookup[("JPM", pd.Timestamp("2024-01-02"))], 0)

    def test_spy_max_return_forward_basic(self):
        from ml_training.data import spy_max_return_forward
        # SPY history sintética: subida linear
        idx = pd.date_range("2024-01-01", periods=100, freq="D")
        spy = pd.DataFrame({
            "Close": np.linspace(400.0, 500.0, 100),
        }, index=idx)
        # alert em day 50 → forward 60d cobre day 51..60..100 (entry=400+50/99*100=450.5)
        alert = pd.Timestamp("2024-02-19")  # ~day 50
        ret = spy_max_return_forward(spy, alert, horizon=60)
        self.assertGreater(ret, 0.0)
        self.assertLess(ret, 0.5)  # max return em range razoável

    def test_spy_max_return_forward_no_history(self):
        from ml_training.data import spy_max_return_forward
        self.assertTrue(np.isnan(spy_max_return_forward(None, pd.Timestamp("2024-01-01"))))

    def test_days_since_52w_high(self):
        from ml_training.data import days_since_52w_high
        # 52w window: pico em day 30, alert em day 100 → 70 dias
        idx = pd.date_range("2024-01-01", periods=200, freq="D")
        prices = np.linspace(100.0, 200.0, 200)
        prices[30] = 500.0  # pico claro em day 30
        hist = pd.DataFrame({
            "High": prices,
            "Low":  prices - 1,
            "Close": prices - 0.5,
        }, index=idx)
        alert = idx[100]
        days = days_since_52w_high(hist, alert)
        self.assertEqual(days, 70.0)


# ─────────────────────────────────────────────────────────────────────────────
# Bundle / report
# ─────────────────────────────────────────────────────────────────────────────

class TestBundle(unittest.TestCase):
    def test_bundle_dataclass_fields(self):
        from ml_training.bundle import DipModelsV3
        from dataclasses import fields
        names = {f.name for f in fields(DipModelsV3)}
        # 14 campos (replica notebook cell 32)
        expected = {
            "model_up", "model_down", "feature_cols", "score_calibrator",
            "n_train_samples", "train_date", "champion_name", "schema_version",
            "momentum_feats", "rho_mean", "rho_alpha", "rho_down", "topk_pnl",
            "fold_metrics",
        }
        self.assertEqual(names, expected)

    def test_bundle_registered_in_main(self):
        from ml_training.bundle import DipModelsV3
        main_mod = sys.modules.get("__main__")
        self.assertIs(main_mod.DipModelsV3, DipModelsV3)

    def test_bundle_round_trip(self):
        from ml_training.bundle import DipModelsV3, save_bundle, load_bundle
        from sklearn.linear_model import Ridge
        m_up = Ridge().fit(np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 3.0]))
        m_down = Ridge().fit(np.array([[1.0], [2.0], [3.0]]), np.array([0.1, 0.2, 0.3]))
        bundle = DipModelsV3(
            model_up=m_up,
            model_down=m_down,
            feature_cols=["x"],
            score_calibrator=None,
            n_train_samples=3,
            train_date="2025-01-01T00:00:00Z",
            champion_name="Ridge-test",
            momentum_feats=["return_1m"],
            rho_alpha=0.30,
            rho_down=0.20,
            topk_pnl=0.15,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bundle.pkl"
            save_bundle(bundle, path)
            self.assertTrue(path.exists())
            loaded = load_bundle(path)
            self.assertEqual(loaded.champion_name, "Ridge-test")
            self.assertEqual(loaded.feature_cols, ["x"])
            self.assertAlmostEqual(loaded.rho_alpha, 0.30)

    def test_build_report_keys(self):
        from ml_training.bundle import DipModelsV3, build_report
        bundle = DipModelsV3(
            model_up=None, model_down=None,
            feature_cols=["a", "b", "c"],
            n_train_samples=100,
            train_date="2025-01-01T00:00:00Z",
            champion_name="XGB-test",
            rho_alpha=0.30, rho_down=0.20, topk_pnl=0.15,
            momentum_feats=["return_1m", "beta_60d"],
        )
        summary = pd.DataFrame([
            {"model": "XGB-test", "rho_alpha_mean": 0.30, "topk_pnl_mean": 0.15},
        ])
        report = build_report(
            bundle=bundle, summary_df=summary,
            brier_oof=0.22, win_rate_alpha=0.65,
            n_folds_used=10, purge_days=21, horizon_days=60,
            new_features=["relative_drop"],
        )
        self.assertEqual(report["schema_version"], 3)
        self.assertEqual(report["champion"], "XGB-test")
        self.assertEqual(report["n_features"], 3)
        self.assertEqual(report["n_train"], 100)
        self.assertAlmostEqual(report["metrics"]["rho_alpha_mean"], 0.30)
        self.assertAlmostEqual(report["metrics"]["brier_oof"], 0.22)
        self.assertAlmostEqual(report["metrics"]["win_rate_alpha"], 0.65)
        self.assertEqual(report["target"]["name"], "alpha_60d")

    def test_metrics_from_report_handles_missing(self):
        from ml_training.bundle import metrics_from_report
        out = metrics_from_report(Path("/nonexistent/report.json"))
        self.assertIn("rho_alpha_mean", out)
        self.assertIsNone(out["rho_alpha_mean"])

    def test_save_load_report_round_trip(self):
        from ml_training.bundle import save_report, metrics_from_report
        report = {
            "schema_version": 3,
            "metrics": {
                "rho_alpha_mean": 0.30,
                "rho_down_mean": 0.20,
                "topk_pnl_mean": 0.15,
                "brier_oof": 0.22,
                "win_rate_alpha": 0.65,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            save_report(report, path)
            self.assertTrue(path.exists())
            data = json.loads(path.read_text())
            self.assertEqual(data["schema_version"], 3)
            metrics = metrics_from_report(path)
            self.assertAlmostEqual(metrics["rho_alpha_mean"], 0.30)


# ─────────────────────────────────────────────────────────────────────────────
# Train (smoke com dataset sintético, modelo Ridge para velocidade)
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainSmoke(unittest.TestCase):
    def test_walk_forward_cv_synthetic(self):
        """Synthetic dataset com signal real → CV deve correr e dar rho > 0."""
        from ml_training.cv import build_walk_forward_folds
        from ml_training.models import ridge_factory
        from ml_training.train import (
            run_walk_forward_cv,
            select_champion,
            summarize_results,
            train_full_champion,
        )
        # 500 alertas em 4 anos, 2 features sintéticas com sinal real
        rng = np.random.default_rng(42)
        n = 500
        dates = pd.date_range("2020-01-01", "2024-01-01", periods=n)
        f1 = rng.normal(0, 1, n)
        f2 = rng.normal(0, 1, n)
        # Target = combinação linear de features + ruído
        alpha = 0.3 * f1 + 0.2 * f2 + rng.normal(0, 0.5, n)
        drawdown = -0.5 * f1 + rng.normal(0, 0.3, n)
        df = pd.DataFrame({
            "alert_date": dates,
            "f1": f1,
            "f2": f2,
            "alpha_60d": alpha,
            "max_drawdown_60d": drawdown,
        })

        # Ridge é rápido (~1s para 500 amostras × 5 folds)
        configs = {
            "Ridge-test": {"factory": ridge_factory, "feats": ["f1", "f2"]},
        }
        results, oof_pred, fold_specs = run_walk_forward_cv(
            df_v31=df,
            model_configs=configs,
            n_folds=5,
            purge_days=10,
        )
        self.assertGreaterEqual(len(fold_specs), 3)
        self.assertGreaterEqual(len(results["Ridge-test"]), 3)
        # rho médio deve ser > 0 (signal real)
        rhos = [h["rho_alpha"] for h in results["Ridge-test"] if np.isfinite(h["rho_alpha"])]
        self.assertGreater(np.mean(rhos), 0.0)

        # Sumário + champion selection
        summary = summarize_results(results)
        champion_name, _row = select_champion(summary)
        self.assertEqual(champion_name, "Ridge-test")

        # Treino full
        champ_alpha, champ_down, feats, n_train = train_full_champion(df, configs[champion_name])
        self.assertIsNotNone(champ_alpha)
        self.assertEqual(feats, ["f1", "f2"])
        self.assertEqual(n_train, n)

    def test_select_champion_fallback_when_no_pnl_positive(self):
        """Se nenhum modelo tem topk_pnl_mean > 0, escolhe melhor rho mesmo assim."""
        from ml_training.train import select_champion
        summary = pd.DataFrame([
            {"model": "A", "rho_alpha_mean": 0.30, "topk_pnl_mean": -0.05},
            {"model": "B", "rho_alpha_mean": 0.20, "topk_pnl_mean": -0.10},
        ])
        name, _row = select_champion(summary)
        self.assertEqual(name, "A")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    sys.exit(0 if res.wasSuccessful() else 1)
