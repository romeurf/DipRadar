"""tax_engine.py — Tax-friction-aware rotation decisions.

Used by /cash to decide whether selling a profitable position to rotate into
a higher-scoring alternative is +EV after paying CGT on the realised gain.

User context: 25yo on IRS Jovem 75% exemption (≤ €20k income, first 5 years
of employment). Effective CGT ~7% (28% statutory × 25% of taxable gain after
exemption). Override via EFFECTIVE_CGT_PCT env var if status changes.
"""

from __future__ import annotations

import os


# How many rotation score points equal 1% of forward-return advantage.
# Used as the conversion factor: tax_cost_pct × MULTIPLIER = required score gap.
#
# Value of 500 means: 1% tax friction = 5 score points required.
#
# PROVENANCE: chosen intuitively, NOT calibrated from data. Sanity bound:
# the sector rotation backtest gave ~+15pp annual alpha at mean picked
# score ~60 → ~0.25% alpha per score point per year → ~0.02%/month.
# Inverse: 1% of friction ≈ 50 score points/month, ~4 per annual horizon.
# Mid-anchor "5 points per 1%" sits between the per-month and per-year
# extremes and biases toward holding (less rotation, fewer tax events).
#
# REVISIT after 6 months of production:
#   - Log every (rotate, hold) decision and forward 1m/3m/6m outcome
#     (see rotation_decisions.parquet, written by /cash when this fires).
#   - Compute the multiplier that would have maximized after-tax alpha.
#   - If observed-optimal differs from 500 by > 2× either direction,
#     update this constant and the calibration date below.
#
# Last calibration: 2026-06-08 (initial, intuitive — no production data yet)
TAX_FRICTION_MULTIPLIER = 500.0


def _default_effective_tax() -> float:
    """Effective CGT rate. Override via EFFECTIVE_CGT_PCT env var."""
    raw = os.environ.get("EFFECTIVE_CGT_PCT", "0.07")
    try:
        v = float(raw)
        if 0.0 <= v <= 0.50:
            return v
    except ValueError:
        pass
    return 0.07  # IRS Jovem 75% exemption default


# Used when the caller can't determine the real unrealized gain (e.g., UCITS
# where yfinance has no price feed). Conservative middle estimate — keeps
# tax friction non-zero so we don't rotate "for free" on positions where the
# real PnL might be substantially positive.
_UNKNOWN_GAIN_DEFAULT = 0.10


def should_rotate(
    unrealized_gain_pct: float | None,
    current_score: float | None,
    target_score: float,
    effective_tax_pct: float | None = None,
    *,
    exit_strength: float = 0.0,
    secular_risk: float = 0.0,
) -> tuple[bool, str]:
    """Decide if rotating out of a position is +EV after tax friction.

    Args:
      unrealized_gain_pct: position PnL as decimal (e.g., 0.18 for +18%).
        Negative = loss position (no tax friction).
        None or exact 0 = unknown (e.g., UCITS without price feed) →
        substituted with _UNKNOWN_GAIN_DEFAULT to keep tax friction
        non-trivial. Don't trust a "free rotation" inference from missing
        data.
      current_score: rotation score of the position currently held.
        None = unknown → HOLD (don't fire SELL on missing signal).
      target_score: rotation score of the candidate to rotate INTO.
      effective_tax_pct: defaults to env var or IRS Jovem 7%.
      exit_strength: structural exit signal (0-100). >70 forces rotate.
        This is the ONLY structural override — a low rotation score alone
        is "no signal", not "active deterioration".
      secular_risk: sector decay signal (0-100). >60 forces rotate.

    Returns:
      (should_rotate: bool, reason: str)
    """
    if effective_tax_pct is None:
        effective_tax_pct = _default_effective_tax()

    # If we have no score for the current position, the rotation decision
    # is data-starved — default to HOLD rather than triggering a sell on
    # a missing-data fallback (the old behaviour was to fire "severe
    # deterioration" when held_score_map returned 0, which conflated
    # "no signal" with "active breakdown").
    if current_score is None:
        return False, "current_score unavailable — HOLD on missing signal"

    # Overrides: structural reasons that override tax math.
    # NOTE: "current_score < 30" used to be an override here but is removed
    # because a low rotation score is just "not a top pick today" — not
    # active deterioration. Use exit_strength / secular_risk for that.
    if exit_strength > 70:
        return True, f"exit_strength {exit_strength:.0f} > 70 (structural breakdown)"
    if secular_risk > 60:
        return True, f"secular_risk {secular_risk:.0f} > 60 (sector dying)"

    score_gap = target_score - current_score
    if score_gap <= 0:
        return False, f"target score {target_score:.0f} ≤ current {current_score:.0f}"

    # Loss override: ONLY for explicit losses. unrealized=0 is "unknown"
    # or "exactly break-even" — neither warrants the friction-free path.
    if unrealized_gain_pct is not None and unrealized_gain_pct < 0:
        return True, f"unrealized loss {unrealized_gain_pct*100:.1f}% — no tax cost"

    # Unknown gain → use conservative default so we don't rotate "for free"
    if unrealized_gain_pct is None or unrealized_gain_pct == 0:
        effective_gain = _UNKNOWN_GAIN_DEFAULT
        gain_label = f"gain unknown — using {_UNKNOWN_GAIN_DEFAULT*100:.0f}% default"
    else:
        effective_gain = unrealized_gain_pct
        gain_label = f"gain {unrealized_gain_pct*100:.1f}%"

    tax_cost_pct = effective_gain * effective_tax_pct
    required_gap = tax_cost_pct * TAX_FRICTION_MULTIPLIER

    if score_gap >= required_gap:
        return True, (
            f"score gap {score_gap:.0f} ≥ required {required_gap:.1f} "
            f"({gain_label}, tax {tax_cost_pct*100:.2f}%)"
        )
    return False, (
        f"score gap {score_gap:.0f} < required {required_gap:.1f} "
        f"(tax friction at {effective_tax_pct*100:.0f}% on {gain_label})"
    )
