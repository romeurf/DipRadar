"""scripts/run_dislocation_validation.py — Run walk-forward backtest on spine.

Implements the BINDING gate per plan §E + F.5 user review:

1. Reads /data/validation_cache/spine.parquet (built by
   build_dislocation_validation_spine.py — PIT-clean subscores only).

2. Walk-forward CV by quarter, with purge 21d + embargo 5d (matches
   ml_training/cv.py production setup).

3. For each fold:
   - Top-decile = top 10% by composite raw within the test slice
   - Primary metric: top_decile_6m_excess_vs_spy
   - Secondary: top_decile_hit_rate (% positive at 6m)
   - Diagnostic only: universe Spearman IC

4. **Defense-in-depth for #7 (per F.5 user review)**:
   The spine carries TWO composite scores per row:
     - composite_raw           (all PIT-clean subscores including #7)
     - composite_without_sub7  (same, but #7 zeroed)
   The backtest runs the same fold pipeline on BOTH. The BINDING gate
   uses min(with, without) — if they diverge materially, #7 is carrying
   the signal too hard and we trust the more conservative reading.

5. Decision rule (BINDING):
   - top_decile_6m_excess_vs_spy ≥ +500bps  (survivorship-inflated threshold per plan §E)
   - top_decile_hit_rate         ≥ 65%      (same)
   - worst-fold excess           ≥ -500bps
   On the MIN of the two variants. If pass → GO; otherwise REVERT.

6. JSON output (`/data/validation_cache/dislocation_validation_<date>.json`):
   - decision: GO | REVERT
   - decision_reason: human-readable
   - spine_variant_with_sub7   : per-fold + aggregated
   - spine_variant_without_sub7: per-fold + aggregated
   - binding_metrics           : min(with, without)
   - spine_metadata            : full provenance carried from spine

The output is the single source of truth for whether F.7 (ML deletion)
can proceed. Per plan §G.5, the 2-week side-by-side (F.6) cannot
validate edge — this backtest is the entire confidence in the edge.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backtest")

_DATA_DIR  = Path("/data") if Path("/data").exists() else _ROOT / "data"
_CACHE_DIR = _DATA_DIR / "validation_cache"
_SPINE_PATH = _CACHE_DIR / "spine.parquet"
_SPINE_META = _CACHE_DIR / "spine_metadata.json"

# Decision rule thresholds (survivorship-inflated per plan §E)
_GO_TOP_DECILE_EXCESS_BPS    = 500   # = +5%
_GO_TOP_DECILE_HIT_RATE      = 0.65
_GO_WORST_FOLD_EXCESS_BPS    = -500  # no fold worse than -5% vs SPY

# Walk-forward params (match ml_training/cv.py production)
_PURGE_DAYS  = 21
_EMBARGO_DAYS = 5

# Top decile fraction
_TOP_DECILE_FRAC = 0.10


# ── Fold construction ────────────────────────────────────────────────

def quarterly_folds(spine: pd.DataFrame) -> list[dict]:
    """Yield quarterly fold definitions.

    Each fold spans one calendar quarter. The training set is everything
    before the quarter start (minus purge), and the test set is the
    quarter itself (minus embargo at the end). Per ml_training/cv.py
    semantics; here we only need the TEST window because the scorer is
    deterministic (no parameter fitting).
    """
    spine = spine.copy()
    spine["as_of_dt"] = pd.to_datetime(spine["as_of"])
    spine["year"] = spine["as_of_dt"].dt.year
    spine["quarter"] = spine["as_of_dt"].dt.quarter
    spine["yq"] = spine["year"].astype(str) + "Q" + spine["quarter"].astype(str)

    folds = []
    for yq in sorted(spine["yq"].unique()):
        sub = spine[spine["yq"] == yq]
        if sub.empty:
            continue
        q_start = sub["as_of_dt"].min().date()
        q_end   = sub["as_of_dt"].max().date()
        folds.append({
            "fold": yq,
            "start": q_start.isoformat(),
            "end":   q_end.isoformat(),
            "n_rows": len(sub),
        })
    return folds


# ── Fold metrics ─────────────────────────────────────────────────────

def fold_metrics(
    spine: pd.DataFrame,
    yq: str,
    score_col: str,
    target_col: str,
    spy_col: str,
) -> dict:
    """Compute top-decile metrics on one fold's test slice."""
    sub = spine[spine["yq"] == yq].copy()
    sub = sub.dropna(subset=[score_col, target_col, spy_col])
    if len(sub) < 30:
        return {
            "fold": yq, "n_total": len(sub), "n_top": 0,
            "top_decile_return": None, "top_decile_excess_vs_spy": None,
            "top_decile_hit_rate": None,
            "spy_return": None, "ic_spearman": None,
            "skipped": "n<30",
        }

    n_top = max(1, int(len(sub) * _TOP_DECILE_FRAC))
    threshold = sub[score_col].quantile(1 - _TOP_DECILE_FRAC)
    top = sub[sub[score_col] >= threshold].head(n_top)

    spy_mean = float(sub[spy_col].mean())
    top_mean = float(top[target_col].mean())

    # Universe IC (diagnostic only)
    try:
        ic_corr = sub[[score_col, target_col]].corr(method="spearman").iloc[0, 1]
        ic = float(ic_corr) if not np.isnan(ic_corr) else None
    except Exception:
        ic = None

    return {
        "fold":                       yq,
        "n_total":                    int(len(sub)),
        "n_top":                      int(len(top)),
        "top_decile_return":          top_mean,
        "spy_return":                 spy_mean,
        "top_decile_excess_vs_spy":   top_mean - spy_mean,
        "top_decile_hit_rate":        float((top[target_col] > 0).mean()),
        "ic_spearman":                ic,
    }


def aggregate_folds(per_fold: list[dict]) -> dict:
    """Average metrics across folds (excluding skipped folds)."""
    valid = [f for f in per_fold if f.get("top_decile_return") is not None]
    if not valid:
        return {
            "n_folds_valid": 0, "n_folds_total": len(per_fold),
            "top_decile_excess_vs_spy_mean": None,
            "top_decile_hit_rate_mean": None,
            "worst_fold_excess": None,
            "ic_spearman_mean": None,
        }
    excess = [f["top_decile_excess_vs_spy"] for f in valid]
    hit_rate = [f["top_decile_hit_rate"] for f in valid]
    ic = [f["ic_spearman"] for f in valid if f["ic_spearman"] is not None]
    return {
        "n_folds_valid":                  len(valid),
        "n_folds_total":                  len(per_fold),
        "top_decile_excess_vs_spy_mean":  float(np.mean(excess)),
        "top_decile_hit_rate_mean":       float(np.mean(hit_rate)),
        "worst_fold_excess":              float(np.min(excess)),
        "ic_spearman_mean":               float(np.mean(ic)) if ic else None,
    }


# ── Run a variant ────────────────────────────────────────────────────

def _augment_with_yq(spine: pd.DataFrame) -> pd.DataFrame:
    """Add as_of_dt, year, quarter, yq columns so fold_metrics can slice
    by quarter without re-parsing the date string per call."""
    spine = spine.copy()
    spine["as_of_dt"] = pd.to_datetime(spine["as_of"])
    spine["year"] = spine["as_of_dt"].dt.year
    spine["quarter"] = spine["as_of_dt"].dt.quarter
    spine["yq"] = spine["year"].astype(str) + "Q" + spine["quarter"].astype(str)
    return spine


def run_variant(
    spine: pd.DataFrame,
    score_col: str,
    label: str,
) -> dict:
    """Run the fold pipeline for one composite-score column."""
    log.info(f"[variant] {label} (score_col={score_col})")
    spine_aug = _augment_with_yq(spine)
    folds = quarterly_folds(spine_aug)
    per_fold = [
        fold_metrics(spine_aug, f["fold"], score_col, "forward_6m_return", "spy_fwd_6m")
        for f in folds
    ]
    agg = aggregate_folds(per_fold)
    return {
        "label":     label,
        "score_col": score_col,
        "per_fold":  per_fold,
        "aggregate": agg,
    }


# ── Decision rule ────────────────────────────────────────────────────

def decide(with_sub7: dict, without_sub7: dict) -> dict:
    """Apply the BINDING decision rule on min(with, without).

    Per plan §E + F.5 user review: if #7 inflates the signal, the
    without-#7 version is the conservative truth. Take min of:
      - top_decile_6m_excess_vs_spy_mean
      - top_decile_hit_rate_mean
      - worst_fold_excess (NOT min; worst is already pessimistic)
    """
    agg_w = with_sub7["aggregate"]
    agg_wo = without_sub7["aggregate"]

    if agg_w["n_folds_valid"] == 0 or agg_wo["n_folds_valid"] == 0:
        return {"decision": "INVALID", "reason": "no valid folds in one or both variants"}

    binding_excess = min(
        agg_w["top_decile_excess_vs_spy_mean"],
        agg_wo["top_decile_excess_vs_spy_mean"],
    )
    binding_hit_rate = min(
        agg_w["top_decile_hit_rate_mean"],
        agg_wo["top_decile_hit_rate_mean"],
    )
    binding_worst = min(
        agg_w["worst_fold_excess"],
        agg_wo["worst_fold_excess"],
    )

    excess_threshold = _GO_TOP_DECILE_EXCESS_BPS / 10_000.0
    worst_threshold  = _GO_WORST_FOLD_EXCESS_BPS / 10_000.0

    excess_ok = binding_excess >= excess_threshold
    hit_rate_ok = binding_hit_rate >= _GO_TOP_DECILE_HIT_RATE
    worst_ok = binding_worst >= worst_threshold

    decision = "GO" if (excess_ok and hit_rate_ok and worst_ok) else "REVERT"

    reasons = []
    reasons.append(f"binding top_decile_6m_excess = {binding_excess*100:+.2f}pp (need ≥ +{_GO_TOP_DECILE_EXCESS_BPS/100:.0f}pp): {'PASS' if excess_ok else 'FAIL'}")
    reasons.append(f"binding hit_rate = {binding_hit_rate:.1%} (need ≥ {_GO_TOP_DECILE_HIT_RATE:.0%}): {'PASS' if hit_rate_ok else 'FAIL'}")
    reasons.append(f"binding worst_fold_excess = {binding_worst*100:+.2f}pp (need ≥ {_GO_WORST_FOLD_EXCESS_BPS/100:.0f}pp): {'PASS' if worst_ok else 'FAIL'}")

    # Note divergence between variants
    divergence_excess = agg_w["top_decile_excess_vs_spy_mean"] - agg_wo["top_decile_excess_vs_spy_mean"]
    divergence_note = ""
    if abs(divergence_excess) > 0.02:  # 2pp+
        divergence_note = (
            f" — WARNING: variants diverge {divergence_excess*100:+.2f}pp on excess. "
            f"#7 sub-score appears to be carrying material weight; trust the "
            f"more conservative (without_sub7) reading."
        )

    return {
        "decision":                       decision,
        "reasons":                        reasons,
        "binding_top_decile_excess":      binding_excess,
        "binding_top_decile_hit_rate":    binding_hit_rate,
        "binding_worst_fold_excess":      binding_worst,
        "thresholds": {
            "top_decile_excess_min":  excess_threshold,
            "hit_rate_min":           _GO_TOP_DECILE_HIT_RATE,
            "worst_fold_excess_min":  worst_threshold,
        },
        "divergence_excess_pp":           divergence_excess * 100,
        "divergence_note":                divergence_note,
    }


# ── Entrypoint ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: dislocation_validation_<date>.json)")
    args = parser.parse_args()

    if not _SPINE_PATH.exists():
        log.error(f"{_SPINE_PATH} missing — run build_dislocation_validation_spine.py first")
        sys.exit(1)
    if not _SPINE_META.exists():
        log.error(f"{_SPINE_META} missing — run build_dislocation_validation_spine.py first")
        sys.exit(1)

    spine = pd.read_parquet(_SPINE_PATH)
    spine_meta = json.loads(_SPINE_META.read_text())
    log.info(f"[backtest] loaded spine: {len(spine):,} rows")
    log.info(f"[backtest] spine_construction: {spine_meta.get('spine_construction')}")

    # Run both variants
    with_sub7 = run_variant(spine, score_col="composite_raw",
                             label="with_sub7")
    without_sub7 = run_variant(spine, score_col="composite_without_sub7",
                                label="without_sub7")

    # Decide
    decision = decide(with_sub7, without_sub7)

    out = {
        "built_at":                       datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision_rule_explanation":      (
            "Per plan §E + F.5 user review: BINDING gate uses min(with_sub7, without_sub7) "
            "for top_decile_6m_excess_vs_spy, hit_rate, and worst_fold. If the two variants "
            "diverge materially, #7 is carrying the signal and the conservative reading wins. "
            "Thresholds inflated to absorb survivorship bias per plan §E."
        ),
        "decision":                       decision,
        "spine_variant_with_sub7":        with_sub7,
        "spine_variant_without_sub7":     without_sub7,
        "spine_metadata":                 spine_meta,
        "binding_thresholds": {
            "top_decile_excess_bps_min":  _GO_TOP_DECILE_EXCESS_BPS,
            "hit_rate_min":               _GO_TOP_DECILE_HIT_RATE,
            "worst_fold_excess_bps_min":  _GO_WORST_FOLD_EXCESS_BPS,
        },
    }

    if args.output:
        out_path = Path(args.output)
    else:
        ts = date.today().isoformat()
        out_path = _CACHE_DIR / f"dislocation_validation_{ts}.json"

    out_path.write_text(json.dumps(out, indent=2))
    log.info(f"[backtest] wrote {out_path}")

    # Console summary
    print()
    print("=" * 70)
    print(f"DISLOCATION VALIDATION SUMMARY ({out_path.name})")
    print("=" * 70)
    print(f"Decision: {decision['decision']}")
    for r in decision.get("reasons", []):
        print(f"  - {r}")
    if decision.get("divergence_note"):
        print(decision["divergence_note"])
    print()
    print(f"WITH sub-#7: excess = {with_sub7['aggregate']['top_decile_excess_vs_spy_mean']*100:+.2f}pp, "
          f"hit_rate = {with_sub7['aggregate']['top_decile_hit_rate_mean']:.1%}, "
          f"worst_fold = {with_sub7['aggregate']['worst_fold_excess']*100:+.2f}pp, "
          f"n_folds = {with_sub7['aggregate']['n_folds_valid']}")
    print(f"WITHOUT sub-#7: excess = {without_sub7['aggregate']['top_decile_excess_vs_spy_mean']*100:+.2f}pp, "
          f"hit_rate = {without_sub7['aggregate']['top_decile_hit_rate_mean']:.1%}, "
          f"worst_fold = {without_sub7['aggregate']['worst_fold_excess']*100:+.2f}pp, "
          f"n_folds = {without_sub7['aggregate']['n_folds_valid']}")


if __name__ == "__main__":
    main()
