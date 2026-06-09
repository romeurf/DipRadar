"""Unit tests for the retrain pipeline (gating + atomic deploy).

Foco em **lógica pura** (não exige network / yfinance). Testa:
- gate_and_promote_v3: 5 ramos (FAILED, KEPT_FLOOR, cold start, PROMOTED, PENDING)
- _read_floor_rho_alpha: criação automática + ler de ficheiro existente
- _do_promote / _save_pending: copy/archive correctos

NOTE: Não testa `run_monthly_retrain` end-to-end (precisaria de yfinance).
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestGateAndPromote(unittest.TestCase):

    def setUp(self):
        # Each test uses an isolated tmpdir; we patch _data_dir so that
        # monthly_retrain._paths() resolves all paths under it.
        from tempfile import mkdtemp
        self.tmp = Path(mkdtemp(prefix="dipradar_retrain_"))
        self._patcher = patch("monthly_retrain._data_dir", return_value=self.tmp)
        self._patcher.start()
        (self.tmp / "candidate").mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self._patcher.stop()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_candidate(self, rho: float = 0.40):
        from ml import monthly_retrain as m
        m.CANDIDATE_BUNDLE.write_text("fake_pkl_bytes")
        m.CANDIDATE_REPORT.write_text(json.dumps({"metrics": {"rho_alpha_mean": rho}}))

    def _make_production(self, rho: float = 0.35):
        from ml import monthly_retrain as m
        m.PRODUCTION_BUNDLE.write_text("old_prod_pkl_bytes")
        m.PRODUCTION_REPORT.write_text(json.dumps({"metrics": {"rho_alpha_mean": rho}}))

    def test_failed_when_candidate_rho_missing(self):
        from ml.monthly_retrain import gate_and_promote_v3
        result = gate_and_promote_v3(
            cand_metrics={"rho_alpha_mean": None},
            prod_metrics={"rho_alpha_mean": 0.35},
        )
        self.assertEqual(result["decision"], "FAILED")

    def test_kept_floor_when_below_absolute_floor(self):
        from ml.monthly_retrain import gate_and_promote_v3, FLOOR_RHO_ALPHA_DEFAULT
        # Pick a candidate ρ_α clearly below the default floor (currently 0.08).
        rho_below = FLOOR_RHO_ALPHA_DEFAULT / 2
        result = gate_and_promote_v3(
            cand_metrics={"rho_alpha_mean": rho_below},
            prod_metrics={"rho_alpha_mean": 0.30},
        )
        self.assertEqual(result["decision"], "KEPT_FLOOR")
        self.assertIn("floor", result["reason"].lower())

    def test_promoted_cold_start(self):
        """Sem produção, candidate só precisa passar o floor."""
        self._make_candidate(rho=0.40)
        from ml.monthly_retrain import gate_and_promote_v3
        result = gate_and_promote_v3(
            cand_metrics={"rho_alpha_mean": 0.40},
            prod_metrics={"rho_alpha_mean": None},
        )
        self.assertEqual(result["decision"], "PROMOTED")
        # Bundle deve ter sido copiado para production
        from ml import monthly_retrain as m
        self.assertTrue(m.PRODUCTION_BUNDLE.exists())

    def test_promoted_with_better_candidate(self):
        self._make_candidate(rho=0.40)
        self._make_production(rho=0.35)
        from ml.monthly_retrain import gate_and_promote_v3
        result = gate_and_promote_v3(
            cand_metrics={"rho_alpha_mean": 0.40},
            prod_metrics={"rho_alpha_mean": 0.35},
            gating_ratio=0.90,
        )
        self.assertEqual(result["decision"], "PROMOTED")
        self.assertGreater(result["delta_pct"], 0)

    def test_promoted_with_slight_drop_within_threshold(self):
        """ρ_α drop de 5% (cand 0.333, prod 0.350) — deve promover (90% threshold)."""
        self._make_candidate(rho=0.333)
        self._make_production(rho=0.350)
        from ml.monthly_retrain import gate_and_promote_v3
        result = gate_and_promote_v3(
            cand_metrics={"rho_alpha_mean": 0.333},
            prod_metrics={"rho_alpha_mean": 0.350},
            gating_ratio=0.90,
        )
        self.assertEqual(result["decision"], "PROMOTED")

    def test_pending_when_drop_exceeds_gating(self):
        """ρ_α drop de 20% — recusa promoção, guarda pending."""
        self._make_candidate(rho=0.28)
        self._make_production(rho=0.35)
        from ml.monthly_retrain import gate_and_promote_v3
        from ml import monthly_retrain as m
        prod_content_before = m.PRODUCTION_BUNDLE.read_text()
        result = gate_and_promote_v3(
            cand_metrics={"rho_alpha_mean": 0.28},
            prod_metrics={"rho_alpha_mean": 0.35},
            gating_ratio=0.90,
        )
        self.assertEqual(result["decision"], "PENDING")
        # Pending bundle deve existir
        self.assertTrue(m.PENDING_BUNDLE.exists())
        # Production NÃO foi sobreescrita
        self.assertEqual(m.PRODUCTION_BUNDLE.read_text(), prod_content_before)
        # E não foi criado archive (porque não promoveu)
        self.assertFalse(any(m.ARCHIVE_DIR.glob("*.pkl")))

    def test_archive_created_on_promote(self):
        self._make_candidate(rho=0.40)
        self._make_production(rho=0.35)
        from ml.monthly_retrain import gate_and_promote_v3
        gate_and_promote_v3(
            cand_metrics={"rho_alpha_mean": 0.40},
            prod_metrics={"rho_alpha_mean": 0.35},
        )
        from ml import monthly_retrain as m
        archives = list(m.ARCHIVE_DIR.glob("dip_models_*.pkl"))
        self.assertEqual(len(archives), 1, "Deve criar exactamente 1 archive da prod anterior")

    def test_floor_file_created_with_default(self):
        from ml.monthly_retrain import _read_floor_rho_alpha
        from ml import monthly_retrain as m
        floor = _read_floor_rho_alpha()
        self.assertEqual(floor, m.FLOOR_RHO_ALPHA_DEFAULT)
        self.assertTrue(m.FLOOR_PATH.exists())
        data = json.loads(m.FLOOR_PATH.read_text())
        self.assertEqual(data["floor_rho_alpha"], m.FLOOR_RHO_ALPHA_DEFAULT)

    def test_floor_file_respects_user_override(self):
        from ml.monthly_retrain import _read_floor_rho_alpha
        from ml import monthly_retrain as m
        m.FLOOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        m.FLOOR_PATH.write_text(json.dumps({"floor_rho_alpha": 0.30}))
        self.assertEqual(_read_floor_rho_alpha(), 0.30)


class TestBuildTrainingInput(unittest.TestCase):
    """Smoke do builder: bootstrap-only, sem alert_db nem snapshot."""

    def setUp(self):
        from tempfile import mkdtemp
        import pandas as pd
        self.tmp = Path(mkdtemp(prefix="dipradar_input_"))
        self._patcher = patch("monthly_retrain._data_dir", return_value=self.tmp)
        self._patcher.start()
        bootstrap = self.tmp / "ml_training_base.parquet"
        df = pd.DataFrame({
            "symbol":         ["AAPL", "MSFT", "GOOG"],
            "alert_date":     pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "drop_pct_today": [-3.5, -4.0, -3.8],
            "label_win":      [1, 0, 1],
            "spy_return_ref": [-1.0, -0.5, -2.0],
        })
        df.to_parquet(bootstrap, index=False)

    def tearDown(self):
        self._patcher.stop()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_bootstrap_only_input(self):
        from ml.monthly_retrain import build_training_input
        path = build_training_input(include_snapshot=False, include_alert_db=False)
        self.assertTrue(path.exists())
        import pandas as pd
        df = pd.read_parquet(path)
        self.assertEqual(len(df), 3)
        self.assertIn("symbol", df.columns)
        self.assertIn("alert_date", df.columns)


class TestBootstrapFallback(unittest.TestCase):
    """Regression: Railway cold-start — /data/ empty, fallback to repo-root parquet."""

    def setUp(self):
        from tempfile import mkdtemp
        import pandas as pd
        self.tmp_data = Path(mkdtemp(prefix="dipradar_data_"))
        self.tmp_repo = Path(mkdtemp(prefix="dipradar_repo_"))
        # _data_dir() returns empty tmp_data; _REPO_ROOT points to tmp_repo
        self._patcher_dd   = patch("monthly_retrain._data_dir", return_value=self.tmp_data)
        self._patcher_repo = patch("monthly_retrain._REPO_ROOT", self.tmp_repo)
        self._patcher_dd.start()
        self._patcher_repo.start()
        df = pd.DataFrame({
            "symbol":         ["AAPL", "MSFT"],
            "alert_date":     pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "drop_pct_today": [-3.5, -4.0],
            "label_win":      [1, 0],
            "spy_return_ref": [-1.0, -0.5],
        })
        df.to_parquet(self.tmp_repo / "ml_training_base.parquet", index=False)

    def tearDown(self):
        self._patcher_dd.stop()
        self._patcher_repo.stop()
        import shutil
        shutil.rmtree(self.tmp_data, ignore_errors=True)
        shutil.rmtree(self.tmp_repo, ignore_errors=True)

    def test_falls_back_to_repo_parquet_when_data_dir_empty(self):
        from ml.monthly_retrain import build_training_input
        path = build_training_input(include_snapshot=False, include_alert_db=False)
        self.assertTrue(path.exists())
        import pandas as pd
        df = pd.read_parquet(path)
        self.assertEqual(len(df), 2)
        self.assertIn("AAPL", df["symbol"].tolist())

    def test_prefers_data_dir_when_both_exist(self):
        """Se `/data/ml_training_base.parquet` existe, é usado e ignora o
        fallback (volume é a fonte da verdade depois do primeiro retrain)."""
        from ml import monthly_retrain as m
        import pandas as pd

        df_data = pd.DataFrame({
            "symbol":         ["NVDA"],
            "alert_date":     pd.to_datetime(["2024-06-15"]),
            "drop_pct_today": [-5.0],
            "label_win":      [1],
            "spy_return_ref": [-2.0],
        })
        df_data.to_parquet(m.BOOTSTRAP_PATH, index=False)

        from ml.monthly_retrain import build_training_input
        path = build_training_input(include_snapshot=False, include_alert_db=False)
        df = pd.read_parquet(path)
        # Tem de ter NVDA (do volume), não AAPL/MSFT (do fallback)
        self.assertIn("NVDA", df["symbol"].tolist())
        self.assertNotIn("AAPL", df["symbol"].tolist())

    def test_raises_when_neither_exists(self):
        """Sem volume e sem repo → erro claro."""
        from ml import monthly_retrain as m
        m.BOOTSTRAP_FALLBACK = self.tmp_repo / "does_not_exist.parquet"

        from ml.monthly_retrain import build_training_input
        with self.assertRaises(RuntimeError) as ctx:
            build_training_input(include_snapshot=False, include_alert_db=False)
        self.assertIn("Sem dados de treino", str(ctx.exception))


class TestMetricsRead(unittest.TestCase):

    def test_read_v3_metrics_canonical(self):
        from tempfile import NamedTemporaryFile
        from ml.monthly_retrain import _read_v3_metrics
        with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"metrics": {
                "rho_alpha_mean": 0.42,
                "brier_oof":      0.18,
                "topk_pnl_mean":  0.12,
            }}, f)
            f.flush()
            metrics = _read_v3_metrics(Path(f.name))

        self.assertAlmostEqual(metrics["rho_alpha_mean"], 0.42)
        self.assertAlmostEqual(metrics["brier_oof"], 0.18)
        self.assertAlmostEqual(metrics["topk_pnl_mean"], 0.12)

    def test_read_v3_metrics_missing_file(self):
        from ml.monthly_retrain import _read_v3_metrics
        metrics = _read_v3_metrics(Path("/tmp/nonexistent_report_v3.json"))
        # Função tolera ficheiro ausente — devolve dict com Nones
        self.assertIsNone(metrics.get("rho_alpha_mean"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
