"""
test_production_quality.py — Testes de qualidade de produção para componentes críticos.

Cobertura:
  - ScaledRidge: fit/predict/scaling correctness
  - themes.py: get_stock_themes, get_theme_bonus, add/remove theme
  - monthly_retrain._load_alert_db_as_training: CSV parsing e rename map
  - run_training: target selection (alpha_90d fallback para alpha_60d)
  - prediction_log: compute_win_prob_drift

Corre com: python -m unittest tests.test_production_quality
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("MONTHLY_BUDGET_EUR", "1050")


# ─────────────────────────────────────────────────────────────────────────────
# ScaledRidge
# ─────────────────────────────────────────────────────────────────────────────

class TestScaledRidge(unittest.TestCase):

    def _make(self, alpha=10.0):
        from ml_training.models import ScaledRidge
        return ScaledRidge(alpha=alpha)

    def test_fit_predict_smoke(self):
        """ScaledRidge treina e prevê sem erros."""
        m = self._make()
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randn(50)
        m.fit(X, y)
        preds = m.predict(X)
        self.assertEqual(len(preds), 50)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_scaling_applied(self):
        """ScaledRidge deve produzir predições diferentes de Ridge sem scaler
        quando features têm escalas muito diferentes."""
        from sklearn.linear_model import Ridge
        # Feature 0: 0-1,  Feature 1: 0-1000 (escala muito diferente)
        rng = np.random.default_rng(42)
        X = np.column_stack([
            rng.uniform(0, 1, 100),
            rng.uniform(0, 1000, 100),
        ]).astype(np.float32)
        y = X[:, 0] * 5 + rng.normal(0, 0.1, 100)   # sinal só na feature pequena

        scaled = self._make(alpha=1.0)
        scaled.fit(X, y)
        pred_scaled = scaled.predict(X)

        raw = Ridge(alpha=1.0)
        raw.fit(X, y)
        pred_raw = raw.predict(X)

        # ScaledRidge deve ter correlação mais alta com y (sinal na feature pequena)
        from scipy.stats import spearmanr
        ic_scaled, _ = spearmanr(pred_scaled, y)
        ic_raw, _    = spearmanr(pred_raw, y)
        self.assertGreater(ic_scaled, ic_raw,
            f"ScaledRidge (IC={ic_scaled:.3f}) deveria superar Ridge sem scaler (IC={ic_raw:.3f})")

    def test_sample_weight_accepted(self):
        """fit() aceita sample_weight sem erros."""
        m = self._make()
        X = np.random.randn(30, 4).astype(np.float32)
        y = np.random.randn(30)
        w = np.ones(30)
        w[:10] = 2.0   # pesos diferentes nos primeiros 10
        m.fit(X, y, sample_weight=w)   # não deve lançar
        self.assertTrue(m._fitted)

    def test_predict_before_fit_raises(self):
        """predict() antes de fit() levanta RuntimeError."""
        m = self._make()
        with self.assertRaises(RuntimeError):
            m.predict(np.ones((1, 4), dtype=np.float32))

    def test_coef_accessible(self):
        """coef_ e intercept_ acessíveis após fit."""
        m = self._make()
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.random.randn(20)
        m.fit(X, y)
        self.assertEqual(len(m.coef_), 3)
        self.assertIsInstance(m.intercept_, float)

    def test_feature_importance_dict(self):
        """feature_importance_dict devolve dict com chaves e valores normalizados."""
        m = self._make()
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.random.randn(20)
        m.fit(X, y)
        imp = m.feature_importance_dict(["a", "b", "c"])
        self.assertEqual(set(imp.keys()), {"a", "b", "c"})
        total = sum(imp.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_picklable(self):
        """ScaledRidge deve ser picklável para joblib."""
        import pickle
        m = self._make()
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.random.randn(20)
        m.fit(X, y)
        data = pickle.dumps(m)
        m2 = pickle.loads(data)
        np.testing.assert_array_almost_equal(m.predict(X), m2.predict(X))


# ─────────────────────────────────────────────────────────────────────────────
# themes.py
# ─────────────────────────────────────────────────────────────────────────────

class TestThemes(unittest.TestCase):

    def setUp(self):
        """Usa directório temporário para não poluir /data/themes.json."""
        self._tmp = tempfile.TemporaryDirectory()
        import themes as tm
        self._orig_path = tm._THEMES_PATH
        tm._THEMES_PATH = Path(self._tmp.name) / "themes_test.json"

    def tearDown(self):
        import themes as tm
        tm._THEMES_PATH = self._orig_path
        self._tmp.cleanup()

    def test_builtin_themes_loaded(self):
        """Sem ficheiro, deve carregar os temas built-in."""
        from themes import list_themes
        ts = list_themes()
        self.assertGreater(len(ts), 0)
        keys = [t["key"] for t in ts]
        self.assertIn("photonics", keys)
        self.assertIn("glp1", keys)

    def test_match_by_ticker(self):
        """CIEN deve matchar fotónica."""
        from themes import get_stock_themes
        matches = get_stock_themes("CIEN", "Technology", "Ciena Corp")
        self.assertTrue(any(m["key"] == "photonics" for m in matches),
                        f"CIEN não matchou photonics: {matches}")

    def test_match_by_keyword(self):
        """Empresa com 'optical' no nome deve matchar fotónica."""
        from themes import get_stock_themes
        matches = get_stock_themes("UNKWN", "", "Optical Systems Inc")
        self.assertTrue(any(m["key"] == "photonics" for m in matches))

    def test_match_by_sector(self):
        """Ticker desconhecido mas sector Healthcare deve matchar glp1."""
        from themes import get_stock_themes
        matches = get_stock_themes("XYZPH", "Healthcare", "Pharma Co")
        self.assertTrue(any(m["key"] == "glp1" for m in matches))

    def test_no_match_irrelevant(self):
        """Ticker e sector sem tema não deve dar match."""
        from themes import get_stock_themes
        matches = get_stock_themes("UTIL123", "Utilities", "Water Utility Corp")
        # Utilities está em energy_transition, mas 'water' não está nos keywords
        # e UTIL123 não está em nenhuma lista de tickers
        irrelevant_keys = [m["key"] for m in matches
                           if m["key"] not in ("energy_transition",)]
        self.assertEqual(len(irrelevant_keys), 0)

    def test_get_theme_bonus_returns_one_for_no_match(self):
        """Stock sem tema deve dar bonus 1.0 (sem alteração de sizing)."""
        from themes import get_theme_bonus
        bonus = get_theme_bonus("ZZZNONE", "Unknown", "Random Corp")
        self.assertAlmostEqual(bonus, 1.0, places=3)

    def test_get_theme_bonus_above_one_for_match(self):
        """Stock em tema deve dar bonus > 1.0."""
        from themes import get_theme_bonus
        bonus = get_theme_bonus("NVO", "Healthcare", "Novo Nordisk")
        self.assertGreater(bonus, 1.0)
        self.assertLessEqual(bonus, 1.21)  # max THEME_BONUS_MAX

    def test_add_remove_theme(self):
        """add_theme e remove_theme devem persistir e remover correctamente."""
        from themes import add_theme, remove_theme, list_themes
        add_theme(
            key="test_robotics",
            label="Robótica Test",
            tickers=["IRBT", "ABB"],
            rationale="Test theme",
            confidence=0.70,
        )
        ts = list_themes()
        keys = [t["key"] for t in ts]
        self.assertIn("test_robotics", keys)

        removed = remove_theme("test_robotics")
        self.assertTrue(removed)
        ts2 = list_themes()
        keys2 = [t["key"] for t in ts2]
        self.assertNotIn("test_robotics", keys2)

    def test_remove_nonexistent_returns_false(self):
        from themes import remove_theme
        self.assertFalse(remove_theme("nao_existe_mesmo"))


# ─────────────────────────────────────────────────────────────────────────────
# monthly_retrain._load_alert_db_as_training (CSV parsing)
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadAlertDb(unittest.TestCase):

    def _write_csv(self, rows: list[dict], path: Path) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                                    quoting=csv.QUOTE_NONNUMERIC)
            writer.writeheader()
            writer.writerows(rows)

    def test_basic_parsing(self):
        """CSV válido com outcome_label preenchido deve ser carregado."""
        import monthly_retrain as mr
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "alert_db.csv"
            rows = [
                {
                    "date_iso": "2025-01-15",
                    "symbol": "AAPL",
                    "sector": "Technology",
                    "change_day_pct": "-8.2",
                    "drawdown_from_high": "-25.3",
                    "rsi": "32.0",
                    "pe": "22.0",
                    "pe_fair": "35.0",
                    "fcf_yield": "0.05",
                    "revenue_growth": "0.10",
                    "gross_margin": "0.40",
                    "debt_equity": "50.0",
                    "analyst_upside": "0.15",
                    "spy_change": "-1.2",
                    "sector_etf_change": "-0.8",
                    "score": "72.0",
                    "verdict": "COMPRAR",
                    "category": "Rotacao",
                    "outcome_label": "WIN_20",
                    "return_60d": "18.5",
                    "return_3m": "22.1",
                    "return_6m": "30.0",
                    "price": "165.0",
                    "market_cap_b": "2600.0",
                    "beta": "1.1",
                    "dividend_yield": "0.6",
                },
            ]
            self._write_csv(rows, p)

            orig = mr.ALERT_DB_PATH
            mr.ALERT_DB_PATH = p
            try:
                df = mr._load_alert_db_as_training()
            finally:
                mr.ALERT_DB_PATH = orig

        self.assertFalse(df.empty, "DataFrame não deve estar vazio")
        self.assertIn("ticker", df.columns, "symbol deve ser renomeado para ticker")
        self.assertIn("drawdown_52w", df.columns, "drawdown_from_high → drawdown_52w")
        self.assertIn("drop_pct_today", df.columns, "change_day_pct → drop_pct_today")
        self.assertEqual(df["ticker"].iloc[0], "AAPL")

    def test_empty_outcome_label_excluded(self):
        """Linhas sem outcome_label devem ser excluídas."""
        import monthly_retrain as mr
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "alert_db.csv"
            rows = [
                {"date_iso": "2025-01-15", "symbol": "AAPL", "outcome_label": "WIN_20",
                 "score": "70"},
                {"date_iso": "2025-02-10", "symbol": "MSFT", "outcome_label": "",
                 "score": "68"},
            ]
            self._write_csv(rows, p)
            orig = mr.ALERT_DB_PATH
            mr.ALERT_DB_PATH = p
            try:
                df = mr._load_alert_db_as_training()
            finally:
                mr.ALERT_DB_PATH = orig

        # Só a linha com outcome_label deve aparecer
        self.assertEqual(len(df), 1)
        self.assertEqual(df["ticker"].iloc[0], "AAPL")

    def test_nonexistent_file_returns_empty(self):
        """Ficheiro inexistente deve retornar DataFrame vazio."""
        import monthly_retrain as mr
        orig = mr.ALERT_DB_PATH
        mr.ALERT_DB_PATH = Path("/tmp/nonexistent_alert_db_xyz.csv")
        try:
            df = mr._load_alert_db_as_training()
        finally:
            mr.ALERT_DB_PATH = orig
        self.assertTrue(df.empty)


# ─────────────────────────────────────────────────────────────────────────────
# run_training: target selection (alpha_90d vs alpha_60d fallback)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import pyarrow  # noqa: F401
    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False


@unittest.skipUnless(_HAS_PYARROW, "pyarrow não instalado — skip testes de parquet")
class TestTargetSelection(unittest.TestCase):

    def _make_df(self, n: int = 200, has_90d: bool = True) -> pd.DataFrame:
        """Cria um DataFrame sintético de treino com ou sem alpha_90d."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        df = pd.DataFrame({
            "ticker":     [f"T{i%50:03d}" for i in range(n)],
            "alert_date": dates,
            "sector":     ["Technology"] * n,
            "alpha_60d":  rng.normal(0.0, 0.1, n),
            "max_drawdown_60d": -rng.uniform(0.05, 0.25, n),
            "month_of_year":    np.tile(np.arange(1, 13), n // 12 + 1)[:n].astype(float),
        })
        # Adicionar todas as FEATURE_COLS com valores neutros
        try:
            from ml_training.config import FEATURE_COLS
            for col in FEATURE_COLS:
                if col not in df.columns:
                    df[col] = 0.0
        except Exception:
            pass
        if has_90d:
            df["alpha_90d"] = rng.normal(0.0, 0.12, n)
        return df

    def test_uses_alpha_90d_when_available(self):
        """run_training usa alpha_90d como target quando disponível."""
        from ml_training.train import run_training
        df = self._make_df(n=300, has_90d=True)
        with tempfile.TemporaryDirectory() as tmp:
            pq = Path(tmp) / "train.parquet"
            df.to_parquet(pq, index=False)
            result = run_training(
                input_parquet=pq,
                n_folds=2, purge_days=10, min_train=50, min_test=10,
                log_summary=False,
            )
        # Quando alpha_90d está disponível, o target activo deve ser alpha_90d
        report = result.get("report") or {}
        self.assertIn("alpha_90d", str(report.get("target", {}).get("name", "")))

    def test_fails_without_alpha_90d(self):
        """run_training falha com KeyError quando alpha_90d está ausente — sem fallback silencioso."""
        from ml_training.train import run_training
        df = self._make_df(n=300, has_90d=False)
        with tempfile.TemporaryDirectory() as tmp:
            pq = Path(tmp) / "train.parquet"
            df.to_parquet(pq, index=False)
            with self.assertRaises(KeyError):
                run_training(
                    input_parquet=pq,
                    n_folds=2, purge_days=10, min_train=50, min_test=10,
                    log_summary=False,
                )


# ─────────────────────────────────────────────────────────────────────────────
# prediction_log.compute_win_prob_drift
# ─────────────────────────────────────────────────────────────────────────────

class TestWinProbDrift(unittest.TestCase):

    def _write_predictions(self, path: Path, win_probs: list[float]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["ts", "symbol", "win_prob", "outcome_label"])
            writer.writeheader()
            from datetime import datetime, timedelta
            base = datetime(2026, 5, 1, 12, 0, 0)
            for i, p in enumerate(win_probs):
                writer.writerow({
                    "ts":            (base + timedelta(days=i)).isoformat(),
                    "symbol":        f"T{i:03d}",
                    "win_prob":      p,
                    "outcome_label": "",
                })

    def test_no_drift_stable(self):
        """Previsões estáveis vs baseline não devem levantar drift_flag."""
        import prediction_log as pl
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ml_predictions.csv"
            self._write_predictions(p, [0.55] * 30)
            orig = pl.PREDICTIONS_PATH
            pl.PREDICTIONS_PATH = p
            try:
                result = pl.compute_win_prob_drift(window_days=60, baseline_win_prob=0.55)
            finally:
                pl.PREDICTIONS_PATH = orig
        self.assertFalse(result.get("skipped"))
        self.assertFalse(result.get("drift_flag"))

    def test_large_drift_raises_flag(self):
        """Shift de >10pp deve activar drift_flag."""
        import prediction_log as pl
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ml_predictions.csv"
            self._write_predictions(p, [0.35] * 30)   # 35% vs baseline 50%
            orig = pl.PREDICTIONS_PATH
            pl.PREDICTIONS_PATH = p
            try:
                result = pl.compute_win_prob_drift(window_days=60, baseline_win_prob=0.55)
            finally:
                pl.PREDICTIONS_PATH = orig
        self.assertFalse(result.get("skipped"))
        self.assertTrue(result.get("drift_flag"),
                        f"drift_flag devia ser True: delta={result.get('delta')}")

    def test_no_file_returns_skipped(self):
        import prediction_log as pl
        orig = pl.PREDICTIONS_PATH
        pl.PREDICTIONS_PATH = Path("/tmp/no_predictions_xyz.csv")
        try:
            result = pl.compute_win_prob_drift()
        finally:
            pl.PREDICTIONS_PATH = orig
        self.assertTrue(result.get("skipped"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
