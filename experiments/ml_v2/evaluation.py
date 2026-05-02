"""
experiments/ml_v2/evaluation.py

Evaluation suite for DipRadar v2 regression models.

Metrics used (and why):
  - Spearman correlation     : ranking quality — do we rank good alerts above bad ones?
  - Top-K precision          : of the top-10% scored alerts, how many are actually good trades?
  - Directional accuracy     : does pred_up > 0 match real_up > 0? (model gets the "side" right)
  - Profit simulation        : simple backtest — if we only take alerts above score_threshold,
                               what is the average and cumulative real return?

NOT used (and why):
  - MSE / RMSE : penalises magnitude errors, irrelevant in trading (ranking is what matters)
  - Accuracy   : was built for classification, meaningless for regression targets

Usage:
    from experiments.ml_v2.evaluation import evaluate_v2
    report = evaluate_v2(pred_df, y_up_real, y_down_real)
    print(report)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ─────────────────────────────────────────────────────────────────────────────
# Individual metrics
# ─────────────────────────────────────────────────────────────────────────────

def spearman_score(y_pred: np.ndarray, y_real: np.ndarray) -> float:
    """
    Spearman rank correlation between predicted and real values.
    Range [-1, 1]. Target: > 0.25 is useful, > 0.40 is strong.
    """
    corr, _ = spearmanr(y_pred, y_real)
    return float(corr)


def top_k_precision(
    scores: np.ndarray,
    y_real_up: np.ndarray,
    top_fraction: float = 0.10,
    win_threshold: float = 0.15,
) -> dict:
    """
    Of the top-(top_fraction)% alerts by score, what fraction achieved
    real upside >= win_threshold?

    Parameters
    ----------
    scores        : model's risk/reward score per alert
    y_real_up     : real max_return_60d per alert
    top_fraction  : e.g. 0.10 = top 10%
    win_threshold : e.g. 0.15 = 15% upside counts as a win

    Returns
    -------
    dict with keys: n_top, n_wins, precision, avg_real_return
    """
    n_top = max(1, int(len(scores) * top_fraction))
    top_idx = np.argsort(scores)[::-1][:n_top]
    top_real = y_real_up[top_idx]

    n_wins = int(np.sum(top_real >= win_threshold))
    precision = n_wins / n_top
    avg_real_return = float(np.mean(top_real))

    return {
        "n_top":           n_top,
        "n_wins":          n_wins,
        "precision":       round(precision, 4),
        "avg_real_return": round(avg_real_return, 4),
    }


def directional_accuracy(
    pred_up: np.ndarray,
    y_real_up: np.ndarray,
    threshold: float = 0.0,
) -> float:
    """
    Fraction of alerts where sign(pred_up) == sign(real_up).
    i.e. model predicts positive return AND real return is positive (or both negative).

    threshold: minimum real upside to count as "truly positive" (default 0 = any positive)
    """
    pred_pos = pred_up > threshold
    real_pos = y_real_up > threshold
    return float(np.mean(pred_pos == real_pos))


def profit_simulation(
    scores: np.ndarray,
    y_real_up: np.ndarray,
    score_threshold: float = 1.5,
    equal_weight: bool = True,
) -> dict:
    """
    Simple paper-trading simulation:
      - Take every alert where score > score_threshold
      - Measure the real max_return_60d for those trades
      - Assumes equal position sizing (equal_weight=True)

    Parameters
    ----------
    scores          : predicted risk/reward scores
    y_real_up       : real max_return_60d
    score_threshold : minimum score to "take" a trade
    equal_weight    : if True, each trade has the same weight

    Returns
    -------
    dict: n_trades, avg_return, median_return, win_rate (>0%), cumulative_return
    """
    mask = scores > score_threshold
    n_trades = int(np.sum(mask))

    if n_trades == 0:
        return {
            "n_trades":         0,
            "avg_return":       0.0,
            "median_return":    0.0,
            "win_rate":         0.0,
            "cumulative_return": 0.0,
            "score_threshold":  score_threshold,
        }

    selected = y_real_up[mask]
    avg_ret    = float(np.mean(selected))
    med_ret    = float(np.median(selected))
    win_rate   = float(np.mean(selected > 0.0))

    # Simple additive (not compounded) cumulative return
    cum_ret = float(np.sum(selected)) if not equal_weight else float(np.mean(selected) * n_trades)

    return {
        "n_trades":          n_trades,
        "avg_return":        round(avg_ret, 4),
        "median_return":     round(med_ret, 4),
        "win_rate":          round(win_rate, 4),
        "cumulative_return": round(cum_ret, 4),
        "score_threshold":   score_threshold,
    }


def calibrate_tp_factor(
    pred_up: np.ndarray,
    y_real_up: np.ndarray,
    quantile: float = 0.70,
) -> float:
    """
    Calibrate the take-profit factor empirically instead of using a fixed 0.7.

    Logic: on validation data, the 70th percentile of (real_up / pred_up)
    gives a conservative factor that TP is hit ~70% of the time.

    Returns the recommended tp_factor (replaces the hardcoded 0.7 in pipeline.py).
    """
    valid = (pred_up > 0.01) & (y_real_up > 0)
    if valid.sum() < 5:
        return 0.7  # fallback
    ratio = y_real_up[valid] / pred_up[valid]
    return float(np.percentile(ratio, quantile * 100))


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation report
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_v2(
    pred_df: pd.DataFrame,
    y_up_real: np.ndarray,
    y_down_real: np.ndarray,
    score_thresholds: Optional[list[float]] = None,
    top_fractions: Optional[list[float]] = None,
) -> dict:
    """
    Full evaluation report for v2 predictions.

    Parameters
    ----------
    pred_df       : output of predict_v2() — must have 'pred_up', 'pred_down', 'score'
    y_up_real     : real max_return_60d
    y_down_real   : real max_drawdown_60d
    score_thresholds : list of thresholds for profit simulation (default [1.0, 1.5, 2.0])
    top_fractions    : list of fractions for Top-K precision (default [0.1, 0.2])

    Returns
    -------
    dict with full evaluation breakdown
    """
    if score_thresholds is None:
        score_thresholds = [1.0, 1.5, 2.0]
    if top_fractions is None:
        top_fractions = [0.10, 0.20]

    pred_up   = pred_df["pred_up"].values
    pred_down = pred_df["pred_down"].values
    scores    = pred_df["score"].values

    y_up_real   = np.array(y_up_real)
    y_down_real = np.array(y_down_real)

    # 1. Spearman correlations
    spearman_up   = spearman_score(pred_up,   y_up_real)
    spearman_down = spearman_score(pred_down, y_down_real)
    spearman_score_vs_real = spearman_score(scores, y_up_real)

    # 2. Top-K precision
    topk = {}
    for frac in top_fractions:
        key = f"top_{int(frac * 100)}pct"
        topk[key] = top_k_precision(scores, y_up_real, top_fraction=frac)

    # 3. Directional accuracy (upside only)
    dir_acc = directional_accuracy(pred_up, y_up_real)

    # 4. Profit simulation at multiple thresholds
    sims = {}
    for thr in score_thresholds:
        key = f"score_gt_{str(thr).replace('.', '_')}"
        sims[key] = profit_simulation(scores, y_up_real, score_threshold=thr)

    # 5. Calibrated TP factor
    tp_factor = calibrate_tp_factor(pred_up, y_up_real)

    # 6. Baseline comparison: random ranking (shuffle scores)
    rng = np.random.default_rng(42)
    random_scores = rng.permutation(scores)
    baseline_topk = top_k_precision(random_scores, y_up_real, top_fraction=0.10)

    report = {
        "n_samples":                   len(pred_up),
        "n_rejected":                  int(pred_df["rejected"].sum()),
        "spearman_up":                  round(spearman_up, 4),
        "spearman_down":                round(spearman_down, 4),
        "spearman_score_vs_real_up":    round(spearman_score_vs_real, 4),
        "directional_accuracy_up":      round(dir_acc, 4),
        "top_k_precision":              topk,
        "top_10pct_baseline_random":    baseline_topk,
        "profit_simulation":            sims,
        "calibrated_tp_factor":         round(tp_factor, 4),
    }

    return report


def print_report(report: dict) -> None:
    """Pretty-print the evaluation report to stdout."""
    print("\n" + "=" * 55)
    print("  DipRadar v2 — Evaluation Report")
    print("=" * 55)
    print(f"  Samples           : {report['n_samples']}")
    print(f"  Rejected (filters): {report['n_rejected']}")
    print()
    print("  Spearman Correlations")
    print(f"    upside   : {report['spearman_up']:+.4f}   (target > 0.25)")
    print(f"    downside : {report['spearman_down']:+.4f}   (target > 0.25)")
    print(f"    score↔up : {report['spearman_score_vs_real_up']:+.4f}")
    print()
    print(f"  Directional Accuracy (upside): {report['directional_accuracy_up']:.2%}")
    print()
    print("  Top-K Precision")
    for k, v in report["top_k_precision"].items():
        print(
            f"    {k:10s}: precision={v['precision']:.2%}  "
            f"avg_return={v['avg_real_return']:.2%}  "
            f"({v['n_wins']}/{v['n_top']} wins)"
        )
    baseline = report["top_10pct_baseline_random"]
    print(
        f"    random_10%: precision={baseline['precision']:.2%}  "
        f"avg_return={baseline['avg_real_return']:.2%}  (baseline)"
    )
    print()
    print("  Profit Simulation")
    for k, v in report["profit_simulation"].items():
        print(
            f"    score > {v['score_threshold']:.1f}: "
            f"{v['n_trades']} trades  "
            f"win_rate={v['win_rate']:.2%}  "
            f"avg_ret={v['avg_return']:.2%}  "
            f"cum_ret={v['cumulative_return']:.2%}"
        )
    print()
    print(f"  Calibrated TP factor: {report['calibrated_tp_factor']:.3f}")
    print("  (replace 0.7 in pipeline.py → predict_v2 if val set is large enough)")
    print("=" * 55 + "\n")
