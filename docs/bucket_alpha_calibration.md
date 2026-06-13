# 7-Bucket Alpha Calibration — Implementation Guide

A pattern that converts a model's raw prediction into an empirically-grounded expected return and percentile-based downside, so the prediction can be used directly by Kelly Criterion sizing.

Currently lives in DIP (`scripts/calibrate_dip_alpha.py`). The same pattern should be replicated to CORE, ROTATION and Momentum. This document is the reference for that work.

---

## Why

ML regressors are typically over-confident in the centre of their prediction distribution and under-confident in the tails. Using `pred_alpha` directly:

- Inflates allocations on mediocre predictions ("the model says 8% so I'll deploy 8% conviction").
- Provides no per-bucket downside, so Kelly's `b = alpha / downside` has to fall back to a generic constant.
- Loses the information that "predictions around X% historically delivered Y%, with a 5%-tail of Z%".

Bucketing the OOF predictions by quantile and computing `mean(actual_alpha)` plus percentiles per bucket fixes all three.

---

## Pattern overview

1. Generate **out-of-fold** predictions (NOT in-sample) — see [OOF section](#oof-out-of-fold-predictions).
2. Sort samples by `pred`.
3. Divide into 7 equal-quantile buckets.
4. Per bucket, persist `mean_actual_alpha` and percentile envelope (`q05/q25/q50/q75/q95`).
5. Apply an acceptance gate (`r² >= 0.10` OR monotonic + spread ≥ 4%). If failed, do not persist — fall back to raw alpha at runtime.
6. At runtime, look up which bucket the raw pred falls into; return `calibrated_alpha = bucket.mean_actual` and `downside = bucket.downside_q05` for Kelly.

---

## Implementation checklist

Use this ordered list when adding calibration to a new model.

### 1. Choose the target horizon

| Model | Alpha target | Risk target |
|---|---|---|
| DIP | `alpha_90d` | `max_drawdown_90d` |
| CORE | `alpha_1y` (or `alpha_3y`) | matching horizon |
| ROTATION | monthly excess return | (use realised drawdown 1m) |
| Momentum | `alpha_30d` or `alpha_90d` | `max_drawdown_30d` or `_90d` |

The risk target must match the alpha target's horizon. Mixing horizons breaks the percentile interpretation.

### 2. Identify the prediction column

- DIP: trained regressor's `pred_alpha`
- Others: their model output or combined score column

### 3. Create the calibration script

Path: `scripts/calibrate_<model>_alpha.py`. Copy structure from `scripts/calibrate_dip_alpha.py`. Skeleton:

```python
def calibrate(
    parquet_path: Path,
    bundle_path: Path,
    output_json: Path,
    n_buckets: int = 7,
    min_samples_per_bucket: int = 30,
) -> dict:
    df = pd.read_parquet(parquet_path)
    bundle = load_bundle(bundle_path)

    # 4. Generate OOF predictions — see OOF section
    oof_preds, oof_actuals = generate_oof_predictions(df, bundle)

    # 5. Quantile bins
    quantiles = np.linspace(0.0, 1.0, n_buckets + 1)
    edges = np.quantile(oof_preds, quantiles)

    # 6. Per-bucket stats
    buckets = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bucket = (oof_preds >= lo) & (oof_preds < hi)
        n = int(in_bucket.sum())
        if n < min_samples_per_bucket:
            continue
        samples = oof_actuals[in_bucket]
        buckets.append({
            "lo": float(lo),
            "hi": float(hi),
            "mean_actual": float(samples.mean()),
            "q05": float(np.quantile(samples, 0.05)),
            "q25": float(np.quantile(samples, 0.25)),
            "q50": float(np.quantile(samples, 0.50)),
            "q75": float(np.quantile(samples, 0.75)),
            "q95": float(np.quantile(samples, 0.95)),
            "downside_q05": abs(float(np.quantile(samples, 0.05))),
            "n_samples": n,
        })

    # 7. Acceptance gate
    means = [b["mean_actual"] for b in buckets]
    monotonic = all(means[i] <= means[i + 1] for i in range(len(means) - 1))
    spread = means[-1] - means[0] if means else 0.0
    r_squared = compute_bucket_r2(buckets)  # bucket index vs mean_actual
    accepted = r_squared >= 0.10 or (monotonic and spread >= 0.04)
    if not accepted:
        log.warning(
            f"calibration gate failed: r²={r_squared:.3f} "
            f"monotonic={monotonic} spread={spread:.3f} — not persisting"
        )
        return {"accepted": False, "r_squared": r_squared, "monotonic": monotonic}

    # 8. Persist JSON
    result = {
        "buckets": buckets,
        "r_squared": r_squared,
        "monotonic": monotonic,
        "spread_top_bottom": spread,
        "n_samples_total": int(sum(b["n_samples"] for b in buckets)),
        "training_horizon_days": horizon_days,
        "calibrated_on": datetime.now().isoformat(),
        "oof_method": "purged_kfold_N=10_purge=90_embargo=20",
    }
    output_json.write_text(json.dumps(result, indent=2))
    return result
```

### 4–8. (covered by the skeleton above)

### 9. Wire into the predictor / radar

In the predictor (e.g. `ml/ml_predictor.py` for DIP, `core_radar.py` for CORE):

```python
_CALIB: dict | None = None

def _load_calibration():
    global _CALIB
    if _CALIB is not None:
        return _CALIB
    p = Path("/data") / f"{MODEL_NAME}_alpha_calibration.json"
    if not p.exists():
        log.warning(f"[calib] no calibration file — using raw alpha")
        _CALIB = {}
        return _CALIB
    try:
        _CALIB = json.loads(p.read_text())
    except Exception as e:
        log.warning(f"[calib] load failed: {e} — using raw alpha")
        _CALIB = {}
    return _CALIB

def calibrated_alpha(raw_pred: float) -> tuple[float, float | None]:
    """Return (calibrated_alpha, downside_q05). downside_q05 is None when no calibration."""
    cal = _load_calibration()
    buckets = cal.get("buckets", [])
    if not buckets:
        return raw_pred, None  # fallback
    for b in buckets:
        if b["lo"] <= raw_pred < b["hi"]:
            return b["mean_actual"], b["downside_q05"]
    # Extrapolation: clamp to nearest bucket
    if raw_pred < buckets[0]["lo"]:
        return buckets[0]["mean_actual"], buckets[0]["downside_q05"]
    return buckets[-1]["mean_actual"], buckets[-1]["downside_q05"]
```

### 10. Wire into `/cash` Kelly Full

In `bot_commands.py` around the Kelly sizing block:

```python
alpha_cal, downside = calibrated_alpha(raw_pred_alpha)
if downside is None or downside <= 0:
    alloc_eur = 0.0  # safe degraded mode
else:
    b = alpha_cal / downside
    f_kelly = (win_prob * b - (1 - win_prob)) / b
    alloc_eur = budget * max(0.0, f_kelly) * KELLY_FRACTION
```

### 11. Admin command

Add `/admin_calibrate_<model>` for manual reruns. Wire automatic post-`PROMOTED` hook in the retrain decision path:

```python
if decision == "PROMOTED":
    try:
        from scripts.calibrate_X_alpha import calibrate
        calibrate(...)
    except Exception as e:
        log.warning(f"post-retrain calibration failed: {e}")
```

Skip when decision == "KEPT" — calibration of the old model is still valid.

### 12. Backup

Add `/data/<model>_alpha_calibration.json` to `utils/backup_data.py` `_BACKUP_PATTERNS`. Calibration files are small (KB-scale) but irreplaceable without rerunning OOF.

### 13. Surface in `/admin_check_config`

Add a line per model:

```
<MODEL> calibration: loaded=Y r²=0.XX buckets=N monotonic=Y last=YYYY-MM-DD oof=purged_kfold_N=10
```

So a quick check confirms each model's calibration is healthy without diving into JSON files.

---

## OOF (out-of-fold) predictions

**Most important section of this guide.**

If you fit a model on the full parquet and then call `model.predict()` on the same parquet to get `(pred, actual)` pairs:

- The model has seen every sample during training.
- The predictions are in-sample → optimistically biased.
- Bucket `mean_actual` becomes ≈ `pred`, r² inflates artificially.
- The deployed calibration looks great in the validation report but produces over-confident allocations in production.

**Correct approach**: out-of-fold predictions from purged K-fold cross-validation. Each sample gets its `pred` from a fold where that sample was in the **test set**, never the train set.

### Reuse the trainer's CV split

The DIP trainer (`ml_training/train.py`) already runs purged K-fold CV with `N_FOLDS=10`, `PURGE_DAYS=90`, `EMBARGO_DAYS=20`. The cleanest pattern is to:

1. Inside `run_training`, collect `oof_preds[i] = model.predict(X_test_fold[i])` after each fold is fit.
2. Persist `oof_preds` alongside the bundle (e.g. `bundle["oof_preds"]` or a sibling parquet).
3. The calibration script reads `bundle["oof_preds"]` directly — no re-fit needed, no CV duplication.

### Or re-run OOF inside the calibrator

If reusing the trainer's split is too invasive, replicate the CV inside `calibrate_<model>_alpha.py`:

```python
from ml_training.cv import purged_kfold_indices

oof_preds = np.full(len(df), np.nan)
for fold_idx, (train_idx, test_idx) in enumerate(
    purged_kfold_indices(df["alert_date"], N_FOLDS, PURGE_DAYS, EMBARGO_DAYS)
):
    model_fold = clone_bundle_estimator(bundle)
    model_fold.fit(X[train_idx], y[train_idx], sample_weight=w[train_idx])
    oof_preds[test_idx] = model_fold.predict(X[test_idx])

mask = ~np.isnan(oof_preds)
oof_preds = oof_preds[mask]
oof_actuals = y[mask]
```

The downside is that the calibration takes nearly as long as a fresh retrain. Worth it for correctness.

### Verifying OOF was actually used

In the persisted JSON, the `oof_method` field documents the CV configuration. In `/admin_check_config`, print it. If absent or `"in_sample"`, the calibration is unsafe.

---

## Acceptance gates

A calibration that fails its gate must NOT be persisted. Fallback to raw `pred` is safer than a bad calibration.

### Primary gate: r² >= 0.10

Linear regression of bucket index (1..7) against bucket `mean_actual`. R² measures how well the bucket ordering explains the actual mean return ordering.

| r² value | Interpretation |
|---|---|
| ≥ 0.50 | strong calibration signal — buckets meaningfully differentiate outcomes |
| 0.10–0.50 | acceptable — model has predictive power, calibration is useful |
| < 0.10 | weak/no signal — model's predictions are mostly noise; calibration would mislead |

### Fallback gate: monotonic + spread ≥ 0.04

Sometimes r² is low because of one bad bucket but the overall trend is correct. Accept if:

- Bucket means are weakly monotonic (`means[i] <= means[i+1]` for all i)
- AND top-bucket mean − bottom-bucket mean ≥ 4%

This catches "the calibration ranks correctly even if it's not linear".

### When both fail

Log a warning, don't persist, and the predictor falls back to raw alpha (with `downside=None` so Kelly degrades to "skip sizing"). Do NOT silently use a bad calibration.

---

## Where each piece goes

| Artefact | Location |
|---|---|
| Calibration script | `scripts/calibrate_<model>_alpha.py` |
| Persisted JSON | `/data/<model>_alpha_calibration.json` |
| Runtime loader | predictor module (e.g. `ml/ml_predictor.py`) |
| Kelly integration | `bot_commands.py` (Kelly Full block in `/cash`) |
| Admin trigger | `/admin_calibrate_<model>` in `bot_commands.py` |
| Auto-trigger | retrain decision path (after `PROMOTED`) |
| Backup pattern | `utils/backup_data.py` `_BACKUP_PATTERNS` |
| Health surface | `/admin_check_config` |

---

## Testing the calibration

Before deploying a new calibration:

1. **Bucket monotonicity plot**: bucket index (x) vs `mean_actual` (y). Should look monotonic.
2. **Spread check**: top vs bottom bucket. Confirms differentiation.
3. **OOF coverage**: verify `oof_preds` covers all (or near-all) samples — no fold should be silently dropped.
4. **Production parity**: pick 10 historical predictions, compute `calibrated_alpha` via the live loader, confirm against the calibration script's output for the same samples.
5. **Kelly sanity**: pick a high-conviction case (top bucket, downside ~5%), compute Kelly allocation. Should match intuition. A negative or zero allocation when alpha > 0 means the bucket's downside is too aggressive.

---

## Anti-patterns to avoid

- **In-sample calibration**: see OOF section.
- **Single-fold OOF**: if K-fold isn't feasible, at least split train/validation chronologically (no shuffle) and calibrate on validation. Not as robust but better than in-sample.
- **Ignoring the acceptance gate**: persisting a bad calibration is worse than no calibration.
- **Forgetting backup**: a poisoned `/data` wipes calibration; without backup it takes hours to regenerate via the calibrator.
- **Stale calibration after retrain**: if you skip the auto-trigger after `PROMOTED`, the predictor uses old buckets with new model outputs — silent drift.

---

## Related work

- `ml_training/cv.py` — purged K-fold implementation used by trainer (reuse for calibrator).
- `scripts/calibrate_dip_alpha.py` — current production reference for DIP.
- `MEMORY.md` → `bucket-calibration-pattern` — quick reference index.
