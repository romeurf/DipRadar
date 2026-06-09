"""calibrate_rotation_alpha.py — Multi-method calibration of rotation score → forward alpha.

Walks the rotation backtest history of monthly checkpoints. For each pick at
each checkpoint, records (combined_buy_score, forward_1m_alpha_vs_spy).
Fits four candidate models and picks the one with best CV r²:

  - linear         OLS slope + intercept
  - polynomial     degree-2 polynomial in score
  - buckets        5 bins of score, mean alpha per bin (non-parametric)
  - lightgbm       gradient-boosted regression (if lightgbm installed)

Persists the winner (params + CV r² + provenance) to
data/rotation_alpha_calibration.json. /cash loads the file and uses the
selected method's predictor at runtime.

Usage:
    python scripts/calibrate_rotation_alpha.py
    python scripts/calibrate_rotation_alpha.py --years 5
    python scripts/calibrate_rotation_alpha.py --method buckets  # force one
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("calibrate")


_OUTPUT_PATH = _ROOT / "data" / "rotation_alpha_calibration.json"
_R_SQUARED_ACCEPT = 0.15
_N_BUCKETS = 5


def _ensure_data_dir() -> None:
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def _gather_picks(n_years: int) -> tuple[list[float], list[float], dict]:
    """Walk the rotation backtest, collecting (score, fwd_1m_alpha) per pick."""
    from core.etf_universe import all_monitored_tickers
    from core.sector_radar import compute_signals, compute_rotation_scores
    from core.sector_rotation_backtest import _INITIAL_LOOKBACK_DAYS

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

    all_dates = sorted({d for h in histories.values() for d in h.index})

    checkpoint_dates = []
    last_month = None
    for d in all_dates[_INITIAL_LOOKBACK_DAYS:]:
        if last_month is None or d.month != last_month or d.year != all_dates[all_dates.index(d) - 1].year:
            checkpoint_dates.append(d)
            last_month = d.month

    log.info(f"Found {len(checkpoint_dates)} monthly checkpoints")

    scores_out: list[float] = []
    alphas_out: list[float] = []

    for cp in checkpoint_dates:
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

        try:
            rotation_ranked = compute_rotation_scores(score_list)
            sorted_tickers = [t for t, _ in rotation_ranked[:3]]
        except Exception:
            score_list.sort(key=lambda x: -x[1])
            sorted_tickers = [t for t, _ in score_list[:3]]

        score_map = {t: s for t, s in score_list}

        idx = checkpoint_dates.index(cp)
        if idx + 1 >= len(checkpoint_dates):
            continue
        cp_next = checkpoint_dates[idx + 1]

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


# ── Model fits ──────────────────────────────────────────────────────────────

def _kfold_cv(x: np.ndarray, y: np.ndarray, fit_predict_fn, k: int = 5) -> float:
    """K-fold CV → return out-of-fold r²."""
    if len(x) < k * 5:
        return 0.0
    idx = np.arange(len(x))
    np.random.seed(42)
    np.random.shuffle(idx)
    folds = np.array_split(idx, k)
    y_true_all = []
    y_pred_all = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        try:
            pred = fit_predict_fn(x[train_idx], y[train_idx], x[test_idx])
            y_true_all.extend(y[test_idx].tolist())
            y_pred_all.extend(pred.tolist() if hasattr(pred, "tolist") else list(pred))
        except Exception:
            continue
    if not y_true_all:
        return 0.0
    yt = np.array(y_true_all)
    yp = np.array(y_pred_all)
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0


def _fit_linear(x: list[float], y: list[float]) -> dict:
    xa = np.array(x, dtype=float)
    ya = np.array(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    if len(xa) < 30:
        return {"method": "linear", "r_squared": 0.0, "cv_r_squared": 0.0,
                "n": len(xa), "error": "too few samples"}

    def fp(x_tr, y_tr, x_te):
        slope = float(np.cov(x_tr, y_tr, ddof=0)[0, 1] / max(np.var(x_tr), 1e-9))
        intercept = float(y_tr.mean() - slope * x_tr.mean())
        return intercept + slope * x_te

    cv = _kfold_cv(xa, ya, fp)

    # Final fit on all data
    slope = float(np.cov(xa, ya, ddof=0)[0, 1] / max(np.var(xa), 1e-9))
    intercept = float(ya.mean() - slope * xa.mean())
    y_pred = intercept + slope * xa
    r2 = 1.0 - float(((ya - y_pred) ** 2).sum()) / max(float(((ya - ya.mean()) ** 2).sum()), 1e-9)

    return {
        "method": "linear",
        "params": {"slope_alpha_per_point_monthly": round(slope, 6),
                   "intercept": round(intercept, 6)},
        "r_squared": round(r2, 3),
        "cv_r_squared": round(cv, 3),
        "n": len(xa),
    }


def _fit_polynomial(x: list[float], y: list[float], degree: int = 2) -> dict:
    xa = np.array(x, dtype=float)
    ya = np.array(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    if len(xa) < 30:
        return {"method": "polynomial", "r_squared": 0.0, "cv_r_squared": 0.0,
                "n": len(xa), "error": "too few samples"}

    def fp(x_tr, y_tr, x_te):
        coeffs = np.polyfit(x_tr, y_tr, deg=degree)
        return np.polyval(coeffs, x_te)

    cv = _kfold_cv(xa, ya, fp)

    coeffs = np.polyfit(xa, ya, deg=degree)
    y_pred = np.polyval(coeffs, xa)
    r2 = 1.0 - float(((ya - y_pred) ** 2).sum()) / max(float(((ya - ya.mean()) ** 2).sum()), 1e-9)

    return {
        "method": "polynomial",
        "params": {"degree": degree, "coeffs": [float(c) for c in coeffs]},
        "r_squared": round(r2, 3),
        "cv_r_squared": round(cv, 3),
        "n": len(xa),
    }


def _fit_buckets(x: list[float], y: list[float], n_buckets: int = _N_BUCKETS) -> dict:
    xa = np.array(x, dtype=float)
    ya = np.array(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    if len(xa) < n_buckets * 5:
        return {"method": "buckets", "r_squared": 0.0, "cv_r_squared": 0.0,
                "n": len(xa), "error": "too few samples"}

    boundaries = list(np.quantile(xa, np.linspace(0, 1, n_buckets + 1)[1:-1]))

    def predict_bucket(boundaries_sorted, bucket_means, x_arr):
        idx = np.digitize(x_arr, boundaries_sorted)
        # Some buckets might be empty in train → use global mean as fallback
        return np.array([bucket_means.get(int(i), float(ya.mean())) for i in idx])

    def fp(x_tr, y_tr, x_te):
        # Compute boundaries on TRAIN only
        b_tr = list(np.quantile(x_tr, np.linspace(0, 1, n_buckets + 1)[1:-1]))
        ids_tr = np.digitize(x_tr, b_tr)
        means = {}
        for i in range(n_buckets):
            mask_i = ids_tr == i
            if mask_i.sum() > 0:
                means[i] = float(y_tr[mask_i].mean())
            else:
                means[i] = float(y_tr.mean())
        return predict_bucket(b_tr, means, x_te)

    cv = _kfold_cv(xa, ya, fp)

    # Final fit on all data
    ids = np.digitize(xa, boundaries)
    bucket_means = {}
    for i in range(n_buckets):
        mask_i = ids == i
        if mask_i.sum() > 0:
            bucket_means[i] = float(ya[mask_i].mean())
        else:
            bucket_means[i] = float(ya.mean())
    y_pred = predict_bucket(boundaries, bucket_means, xa)
    r2 = 1.0 - float(((ya - y_pred) ** 2).sum()) / max(float(((ya - ya.mean()) ** 2).sum()), 1e-9)

    return {
        "method": "buckets",
        "params": {
            "boundaries": [round(b, 2) for b in boundaries],
            "alpha_per_bucket": {str(k): round(v, 6) for k, v in bucket_means.items()},
            "n_buckets": n_buckets,
        },
        "r_squared": round(r2, 3),
        "cv_r_squared": round(cv, 3),
        "n": len(xa),
    }


def _fit_lightgbm(x: list[float], y: list[float]) -> dict:
    try:
        import lightgbm as lgb
    except ImportError:
        return {"method": "lightgbm", "r_squared": 0.0, "cv_r_squared": 0.0,
                "n": len(x), "error": "lightgbm not installed"}
    xa = np.array(x, dtype=float).reshape(-1, 1)
    ya = np.array(y, dtype=float)
    mask = np.isfinite(xa.flatten()) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    if len(xa) < 50:
        return {"method": "lightgbm", "r_squared": 0.0, "cv_r_squared": 0.0,
                "n": len(xa), "error": "too few samples for lightgbm"}

    def fp(x_tr, y_tr, x_te):
        model = lgb.LGBMRegressor(
            num_leaves=8, min_child_samples=10, n_estimators=50,
            learning_rate=0.05, verbosity=-1,
        )
        model.fit(x_tr.reshape(-1, 1), y_tr)
        return model.predict(x_te.reshape(-1, 1))

    cv = _kfold_cv(xa.flatten(), ya, fp)

    # Final fit
    model = lgb.LGBMRegressor(
        num_leaves=8, min_child_samples=10, n_estimators=50,
        learning_rate=0.05, verbosity=-1,
    )
    model.fit(xa, ya)
    y_pred = model.predict(xa)
    r2 = 1.0 - float(((ya - y_pred) ** 2).sum()) / max(float(((ya - ya.mean()) ** 2).sum()), 1e-9)

    # Save the model's predictions at a grid for /cash to reconstruct without lightgbm
    grid_x = np.linspace(xa.min(), xa.max(), 50)
    grid_y = model.predict(grid_x.reshape(-1, 1))

    return {
        "method": "lightgbm",
        "params": {
            "grid_x": [round(float(v), 3) for v in grid_x],
            "grid_y": [round(float(v), 6) for v in grid_y],
        },
        "r_squared": round(r2, 3),
        "cv_r_squared": round(cv, 3),
        "n": len(xa),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=4, help="History window")
    parser.add_argument(
        "--method", type=str, default="auto",
        choices=["auto", "linear", "polynomial", "buckets", "lightgbm"],
        help="Force a specific method; auto picks best by CV r²",
    )
    args = parser.parse_args()

    _ensure_data_dir()

    log.info(f"Calibrating rotation alpha — window={args.years}y, method={args.method}")
    scores, alphas, meta = _gather_picks(args.years)
    log.info(f"Collected {len(scores)} picks across {meta['n_checkpoints']} monthly checkpoints")

    # Fit all methods
    fits: list[dict] = []
    fits.append(_fit_linear(scores, alphas))
    fits.append(_fit_polynomial(scores, alphas, degree=2))
    fits.append(_fit_buckets(scores, alphas, n_buckets=_N_BUCKETS))
    fits.append(_fit_lightgbm(scores, alphas))

    log.info("Method comparison (in-sample r² | CV r²):")
    for f in fits:
        err = f" [{f['error']}]" if "error" in f else ""
        log.info(f"  {f['method']:11} {f['r_squared']:>6.3f} | {f['cv_r_squared']:>6.3f}{err}")

    # Pick best by CV r²
    if args.method == "auto":
        best = max(fits, key=lambda f: f.get("cv_r_squared", 0.0))
    else:
        match = [f for f in fits if f["method"] == args.method]
        best = match[0] if match else max(fits, key=lambda f: f.get("cv_r_squared", 0.0))

    log.info(f"Selected: {best['method']} (CV r²={best['cv_r_squared']})")

    output = {
        "method":           best["method"],
        "params":           best.get("params", {}),
        "r_squared":        best["r_squared"],
        "cv_r_squared":     best["cv_r_squared"],
        "n_picks":          best["n"],
        "n_checkpoints":    meta["n_checkpoints"],
        "n_tickers":        meta["n_tickers"],
        "backtest_window":  f"{args.years}y",
        "calibrated_on":    datetime.utcnow().strftime("%Y-%m-%d"),
        "accept_threshold": _R_SQUARED_ACCEPT,
        "accepted":         best["cv_r_squared"] >= _R_SQUARED_ACCEPT,
        "all_methods":      [
            {"method": f["method"], "r_squared": f["r_squared"],
             "cv_r_squared": f["cv_r_squared"], "error": f.get("error")}
            for f in fits
        ],
    }
    if "error" in best:
        output["error"] = best["error"]

    _OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    log.info(f"Wrote {_OUTPUT_PATH}")

    if not output["accepted"]:
        log.warning(
            f"CV r² {best['cv_r_squared']:.3f} < {_R_SQUARED_ACCEPT}: best method "
            f"still below acceptance threshold. /cash falls back to flat rotation alpha."
        )
        return 2

    log.info(f"Calibration ACCEPTED (CV r² ≥ {_R_SQUARED_ACCEPT})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
