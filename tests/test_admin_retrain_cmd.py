"""Tests for the /admin_retrain command parser + dispatch logic.

Avoids importing the full bot_commands module side-effects by mocking
the threading.Thread + monthly_retrain calls. Verifies:

- Argument parsing (dry-run, no-snap, gating ratio)
- Concurrency lock (`_retrain_running` flag)
- Dispatch into `run_monthly_retrain_v3` with correct kwargs
- Result formatting branches (DRY-RUN, PROMOTED, PENDING, FAILED)
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestAdminRetrainParser(unittest.TestCase):
    """Verifica que o handler chama `run_monthly_retrain_v3` com os
    kwargs correctos consoante os args do utilizador."""

    def setUp(self):
        # Mock _reply para capturar mensagens
        import bot_commands
        self.replies: list[str] = []
        bot_commands._reply = lambda txt: self.replies.append(txt)
        # Reset flag
        bot_commands._retrain_running = False

    def _invoke(self, parts: list[str], result: dict) -> dict:
        """Invoca o handler de forma síncrona (sem thread real)."""
        import bot_commands
        captured: dict = {}

        def fake_thread(target, daemon=None, name=None):
            class _T:
                def start(self_t):
                    target()
            return _T()

        with patch("bot_commands.threading.Thread", fake_thread), \
             patch("monthly_retrain.run_monthly_retrain_v3") as mock_v3:
            mock_v3.return_value = result
            bot_commands._handle_admin_retrain(parts)
            captured["mock_v3"] = mock_v3
        return captured

    def test_default_full_retrain(self):
        captured = self._invoke(
            parts=["/admin_retrain"],
            result={
                "decision": "PROMOTED",
                "candidate_rho_alpha": 0.40,
                "production_rho_alpha": 0.35,
                "candidate_brier": 0.18,
                "candidate_topk_pnl": 0.12,
            },
        )
        captured["mock_v3"].assert_called_once_with(
            dry_run=False,
            include_snapshot=True,
            include_alert_db=True,
        )
        self.assertTrue(any("Retrain v3" in r for r in self.replies))
        self.assertTrue(any("PROMOTED" in r for r in self.replies))

    def test_dry_run_flag(self):
        captured = self._invoke(
            parts=["/admin_retrain", "dry-run"],
            result={
                "decision":       "DRY-RUN",
                "training_input": "/data/ml_training_input.parquet",
                "outcome_stats":  {"updated": 5, "skipped": 0},
            },
        )
        kwargs = captured["mock_v3"].call_args.kwargs
        self.assertTrue(kwargs["dry_run"])
        self.assertTrue(any("DRY-RUN" in r for r in self.replies))

    def test_no_snapshot_flag(self):
        captured = self._invoke(
            parts=["/admin_retrain", "dry-run", "no-snap"],
            result={"decision": "DRY-RUN", "training_input": ""},
        )
        kwargs = captured["mock_v3"].call_args.kwargs
        self.assertTrue(kwargs["dry_run"])
        self.assertFalse(kwargs["include_snapshot"])
        self.assertTrue(kwargs["include_alert_db"])

    def test_no_alert_db_flag(self):
        captured = self._invoke(
            parts=["/admin_retrain", "dry-run", "no-alert-db"],
            result={"decision": "DRY-RUN", "training_input": ""},
        )
        kwargs = captured["mock_v3"].call_args.kwargs
        self.assertFalse(kwargs["include_alert_db"])
        self.assertTrue(kwargs["include_snapshot"])

    def test_gating_ratio_override(self):
        captured = self._invoke(
            parts=["/admin_retrain", "0.85"],
            result={"decision": "KEPT_FLOOR", "reason": "below floor"},
        )
        kwargs = captured["mock_v3"].call_args.kwargs
        self.assertEqual(kwargs["gating_ratio"], 0.85)
        self.assertFalse(kwargs["dry_run"])

    def test_invalid_ratio_ignored(self):
        """Args fora de [0.5, 1.5] não devem ser tratados como ratio."""
        captured = self._invoke(
            parts=["/admin_retrain", "100"],
            result={"decision": "PROMOTED"},
        )
        kwargs = captured["mock_v3"].call_args.kwargs
        self.assertNotIn("gating_ratio", kwargs)

    def test_concurrent_lock(self):
        """Segundo invocation enquanto _retrain_running=True deve abortar."""
        import bot_commands
        bot_commands._retrain_running = True
        bot_commands._handle_admin_retrain(["/admin_retrain"])
        self.assertTrue(any("já está a correr" in r for r in self.replies))
        bot_commands._retrain_running = False

    def test_failed_decision_formats_error(self):
        captured = self._invoke(
            parts=["/admin_retrain"],
            result={
                "decision": "FAILED",
                "reason":   "candidate report missing",
            },
        )
        self.assertTrue(any("FAILED" in r for r in self.replies))
        self.assertTrue(any("candidate report missing" in r for r in self.replies))

    def test_pending_decision_mentions_pending_bundle(self):
        self._invoke(
            parts=["/admin_retrain"],
            result={
                "decision":            "PENDING",
                "candidate_rho_alpha": 0.28,
                "production_rho_alpha": 0.35,
                "reason":              "below gating threshold",
            },
        )
        self.assertTrue(any("pending" in r.lower() for r in self.replies))

    def test_exception_in_run_caught(self):
        """Erro dentro do thread → reply de erro, flag liberta."""
        import bot_commands

        def fake_thread(target, daemon=None, name=None):
            class _T:
                def start(self_t):
                    target()  # roda síncrono
            return _T()

        with patch("bot_commands.threading.Thread", fake_thread), \
             patch("monthly_retrain.run_monthly_retrain_v3", side_effect=RuntimeError("boom")):
            bot_commands._handle_admin_retrain(["/admin_retrain"])

        self.assertTrue(any("Retrain falhou" in r for r in self.replies))
        self.assertTrue(any("boom" in r for r in self.replies))
        self.assertFalse(bot_commands._retrain_running, "flag deve ser liberta após erro")


class TestDispatcherRouting(unittest.TestCase):
    """Verifica que o /admin_retrain está registado no _handle_command."""

    def test_admin_retrain_in_dispatcher(self):
        import bot_commands
        with open(bot_commands.__file__) as f:
            source = f.read()
        # Procura a string exacta que registaria o comando
        self.assertIn('cmd == "/admin_retrain":', source)
        self.assertIn("_handle_admin_retrain(parts)", source)

    def test_help_mentions_admin_retrain(self):
        import bot_commands
        with open(bot_commands.__file__) as f:
            source = f.read()
        self.assertIn("/admin_retrain", source)


class TestMarkdownSafety(unittest.TestCase):
    """Regressão para o bug 400 Bad Request quando a `reason` continha
    underscores (e.g. `alert_db`) — italics partia o parser do Telegram.

    Fix: reason vai dentro de backticks (code span), nunca italics.
    """

    def setUp(self):
        import bot_commands
        self.replies: list[str] = []
        bot_commands._reply = lambda txt: self.replies.append(txt)
        bot_commands._retrain_running = False

    def _invoke_failed_with_reason(self, reason: str) -> None:
        import bot_commands

        def fake_thread(target, daemon=None, name=None):
            class _T:
                def start(self_t):
                    target()
            return _T()

        with patch("bot_commands.threading.Thread", fake_thread), \
             patch("monthly_retrain.run_monthly_retrain_v3") as mock_v3:
            mock_v3.return_value = {"decision": "FAILED", "reason": reason}
            bot_commands._handle_admin_retrain(["/admin_retrain"])

    def test_md_safe_strips_backticks(self):
        from bot_commands import _md_safe
        # Backticks na string interna partem o code span. Têm de ser
        # substituídos para preservar a sintaxe Markdown.
        self.assertNotIn("`", _md_safe("path with `backtick` inside"))
        self.assertEqual(_md_safe(None), "")
        self.assertEqual(_md_safe(123), "123")

    def test_reason_with_underscores_uses_backticks_not_italics(self):
        """O bug original: `_input build failed: ... alert_db ..._` parte o
        parser. Garantimos que a `reason` vai sempre dentro de backticks.
        """
        original_bug_reason = (
            "input build failed: Sem dados de treino — bootstrap, "
            "alert_db e snapshot todos vazios."
        )
        self._invoke_failed_with_reason(original_bug_reason)

        # A reason tem de aparecer dentro de backticks (code span), nunca
        # dentro de underscores (italics) — caso contrário 400 Bad Request.
        backtick_wrapped = f"`{original_bug_reason}`"
        italic_wrapped = f"_{original_bug_reason}_"

        joined = "\n".join(self.replies)
        self.assertIn(backtick_wrapped, joined,
                      "reason deve estar dentro de `...` (code span)")
        self.assertNotIn(italic_wrapped, joined,
                         "reason NÃO deve estar dentro de _..._ (italics)")

    def test_reason_with_path_underscores_safe(self):
        """Path com `/data/ml_training_merged.parquet` — underscores múltiplos."""
        reason = "Cannot read /data/ml_training_merged.parquet — file missing"
        self._invoke_failed_with_reason(reason)
        self.assertIn(f"`{reason}`", "\n".join(self.replies))


if __name__ == "__main__":
    unittest.main(verbosity=2)
