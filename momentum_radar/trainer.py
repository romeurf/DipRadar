"""momentum_radar/trainer.py — RETIRED.

The MomentumRadar ML training pipeline was retired after walk-forward
validation showed it was anti-predictive (CV IC = -0.0042 — worse than
random). Keeping the trainer in production would have shipped a model
that confidently ranked names in the WRONG direction.

The rule-based scanner in `momentum_radar/scanner.py` remains the only
shipped MomentumRadar component. It uses transparent heuristics
(return_20d >= 12%, volume_ratio >= 1.4, RSI 52-78, beats sector,
quality >= 0.55) and does NOT depend on this trainer.

This file is kept as an import-compatibility stub so existing admin
commands and crons that wired in the trainer (~30 call sites in
bot_commands.py + main.py) get a CLEAR error rather than ImportError
or, worse, silently quiet behaviour. Stubs are honest about retirement.

If you need the original code: see git history before commit that
introduced this stub.
"""
from __future__ import annotations

# Empty so iteration is safe; old code that does
# `for col in FEATURE_COLS` gets zero iterations instead of NameError.
FEATURE_COLS: list[str] = []


def run_momentum_training(*args, **kwargs):
    """Retired entrypoint. Raises with a clear message.

    Original behaviour: trained multi-horizon momentum model, persisted
    a bundle, returned metrics dict. Anti-predictive (CV IC -0.0042),
    retired to avoid ML masquerading as edge.
    """
    raise RuntimeError(
        "MomentumRadar ML training is RETIRED. The model was anti-"
        "predictive (CV IC = -0.0042). Use the rule-based scanner at "
        "momentum_radar/scanner.py - it does not depend on training "
        "and is already the only momentum component in production."
    )
