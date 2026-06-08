"""calibrate_rotation_alpha.py — Fit slope/intercept of rotation alpha vs score.

Walks the backtest history of monthly rotation checkpoints. For each pick at
each checkpoint, records (score_at_pick, forward_1m_alpha_vs_spy). Fits a
linear regression and writes the result to data/rotation_alpha_calibration.json
so /cash can use empirically-grounded numbers instead of guessed multipliers.

Usage:
    python scripts/calibrate_rotation_alpha.py
    python scripts/calibrate_rotation_alpha.py --years 5

Run from project root. Re-run quarterly (the relationship can drift as the
sector universe or market regime changes).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure project root on path so we can import sibling modules when invoked
# as `python scripts/calibrate_rotation_alpha.py`.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("calibrate")


_OUTPUT_PATH = _ROOT / "data" / "rotation_alpha_calibration.json"
_R_SQUARED_ACCEPT = 0.15


def _ensure_data_dir() -> None:
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def _gather_picks(n_years: int) -> tuple[list[float], list[float], dict]:
    """Walk the rotation backtest, collecting (score, fwd_1m_alpha) per pick.

    Returns (scores, alphas, meta_dict).
    """
    from etf_universe import all_monitored_tickers
    from sector_radar import compute_signals, compute_rotation_scores
    from sector_rotation_backtest import _INITIAL_LOOKBACK_DAYS

    import yfinance as yf
    import pandas as pd

    tickers = all_monitored_tickers()
    log.info(f"Fetching {n_years}y history for {len(tickers)} ETFs + SPY")

    histories: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            h = yf.Ticker(t).history(period=f"{n_years}y", auto_adjust=True)
            if h is not None and len(h) >= _INITIAL_LOOKBACK_DAYS:
                histories[t] = h
        except Exception as e:
            log.debug(f"{t} fetch failed: {e}")
    spy_hist = yf.Ticker("SPY").history(period=f"{n_years}y", auto_adjust=True)
    if spy_hist is None or len(spy_hist) < _INITIAL_LOOKBACK_DAYS:
        raise RuntimeError("SPY history insufficient")

    # Use the union of all dates as the daily index; align all on this.
    all_dates = sorted({d for h in histories.values() for d in h.index})

    # Walk monthly: pick first trading day of each month after lookback.
    # For each, compute scores, take top-3 by combined_buy, record their
    # forward 1-month return vs SPY.
    scores_out: list[float] = []
    alphas_out: list[float] = []

    # Get monthly checkpoints (first trading day of each month)
    checkpoint_dates = []
    last_month = None
    for d in all_dates[_INITIAL_LOOKBACK_DAYS:]:
        if last_month is None or d.month != last_month or d.year != all_dates[all_dates.index(d) - 1].year:
            checkpoint_dates.append(d)
            last_month = d.month

    log.info(f"Found {len(checkpoint_dates)} monthly checkpoints")

    for cp in checkpoint_dates:
        # Compute scores at this checkpoint
        score_list = []
        for t, h in histories.items():
            window = h[h.index <= cp]
            if len(window) < _INITIAL_LOOKBACK_DAYS:
                continue
            spy_window = spy_hist[spy_hist.index <= cp]
            try:
                sig = compute_signals(t, window, spy_hist=spy_window)
                if sig is not None:
                    score_list.append((t, sig.combined_buy))
            except Exception:
                continue
        if not score_list:
            continue

        # Apply cross-sectional rotation_score (same logic as backtest)
        try:
            rotation_ranked = compute_rotation_scores(score_list)
            sorted_tickers = [t for t, _ in rotation_ranked[:3]]
        except Exception:
            score_list.sort(key=lambda x: -x[1])
            sorted_tickers = [t for t, _ in score_list[:3]]

        score_map = {t: s for t, s in score_list}

        # Look up next checkpoint (or end) for forward 1m
        idx = checkpoint_dates.index(cp)
        if idx + 1 >= len(checkpoint_dates):
            continue
        cp_next = checkpoint_dates[idx + 1]

        # SPY forward return for this window
        try:
            spy_now = float(spy_hist[spy_hist.index <= cp]["Close"].iloc[-1])
            spy_then = float(spy_hist[spy_hist.index <= cp_next]["Close"].iloc[-1])
            spy_ret = spy_then / spy_now - 1.0
        except Exception:
            continue

        for t in sorted_tickers:
            h = histories.get(t)
            if h is None:
                continue
            try:
                px_now = float(h[h.index <= cp]["Close"].iloc[-1])
                px_then = float(h[h.index <= cp_next]["Close"].iloc[-1])
                t_ret = px_then / px_now - 1.0
                alpha_1m = t_ret - spy_ret
                scores_out.append(score_map[t])
                alphas_out.append(alpha_1m)
            except Exception:
                continue

    meta = {
        "n_checkpoints": len(checkpoint_dates),
        "n_picks":       len(scores_out),
        "n_tickers":     len(histories),
    }
    return scores_out, alphas_out, meta


def _fit_linear(x: list[float], y: list[float]) -> dict:
    """OLS fit y = intercept + slope × x."""
    if len(x) < 30:
        return {
            "slope": 0.0, "intercept": 0.0, "r_squared": 0.0,
            "n": len(x), "error": "insufficient samples (need ≥30)",
        }
    xa = np.array(x, dtype=float)
    ya = np.array(y, dtype=float)
    # Filter NaNs
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    if len(xa) < 30:
        return {
            "slope": 0.0, "intercept": 0.0, "r_squared": 0.0,
            "n": len(xa), "error": "too few finite samples",
        }
    x_mean = xa.mean()
    y_mean = ya.mean()
    cov = float(((xa - x_mean) * (ya - y_mean)).sum())
    var_x = float(((xa - x_mean) ** 2).sum())
    slope = cov / var_x if var_x > 1e-9 else 0.0
    intercept = y_mean - slope * x_mean
    y_pred = intercept + slope * xa
    ss_res = float(((ya - y_pred) ** 2).sum())
    ss_tot = float(((ya - y_mean) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "n": len(xa),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=4, help="History window")
    args = parser.parse_args()

    _ensure_data_dir()

    log.info(f"Calibrating rotation alpha — window={args.years}y")
    scores, alphas, meta = _gather_picks(args.years)
    log.info(
        f"Collected {len(scores)} picks across {meta['n_checkpoints']} monthly checkpoints"
    )

    fit = _fit_linear(scores, alphas)
    log.info(
        f"Fit: slope={fit['slope']:.6f}, intercept={fit['intercept']:.6f}, "
        f"r²={fit['r_squared']:.3f}, n={fit['n']}"
    )

    output = {
        "slope_alpha_per_point_monthly": round(fit["slope"], 6),
        "intercept":                      round(fit["intercept"], 6),
        "r_squared":                      round(fit["r_squared"], 3),
        "n_picks":                        fit["n"],
        "n_checkpoints":                  meta["n_checkpoints"],
        "n_tickers":                      meta["n_tickers"],
        "backtest_window":                f"{args.years}y",
        "calibrated_on":                  datetime.utcnow().strftime("%Y-%m-%d"),
        "accept_threshold":               _R_SQUARED_ACCEPT,
        "accepted":                       fit["r_squared"] >= _R_SQUARED_ACCEPT,
    }
    if "error" in fit:
        output["error"] = fit["error"]

    _OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    log.info(f"Wrote {_OUTPUT_PATH}")

    if not output["accepted"]:
        log.warning(
            f"r² {fit['r_squared']:.3f} < {_R_SQUARED_ACCEPT}: rotation alpha "
            f"will default to 0 in /cash (last in ROI queue) until "
            f"recalibration improves the fit."
        )
        return 2  # non-zero but distinct from hard error

    log.info(f"Calibration ACCEPTED (r² ≥ {_R_SQUARED_ACCEPT})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
