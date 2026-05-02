# DipRadar ML v2 — XGBoost Regression Pipeline

> **Status:** experiment — run locally, do not deploy to Railway until validated.

## What Changed vs. Production (v1)

| | v1 (production) | v2 (this experiment) |
|---|---|---|
| Task | Binary classification (WIN/LOSS) | Regression (upside + downside separately) |
| Model | RandomForest (Stage 1 + 2) | 2× XGBoost (model_up + model_down) |
| Targets | `label_win` (0/1) | `max_return_60d`, `max_drawdown_60d` |
| Target transform | None | `log1p` (stabilises heavy tails) |
| Target logic | Peak in window | Sequence-aware: drawdown THEN recovery |
| Primary metric | Accuracy | Spearman + Top-K Precision + Profit Sim |
| Output | Probability of win | Score = upside / \|downside\|, TP, SL |
| New features | — | `return_5d/10d/20d`, `zscore_20d`, `distance_from_ma50`, `volatility_20d`, `sp500_trend` |

## File Structure

```
experiments/ml_v2/
  pipeline.py     — build_targets(), train_v2(), predict_v2()
  evaluation.py   — Spearman, Top-K, directional accuracy, profit sim
  README.md       — this file
```

## Quick Start

```python
import pandas as pd
import numpy as np
from experiments.ml_v2.pipeline import build_targets, train_v2, predict_v2, save_models_v2
from experiments.ml_v2.evaluation import evaluate_v2, print_report

# 1. Build feature matrix X and targets from your alert_db + price history
#    (see build_targets() docstring for anti-leakage requirements)

# 2. Train
models = train_v2(X_train, y_up_train, y_down_train)

# 3. Predict on held-out validation set
pred_df = predict_v2(X_val, models, entry_prices=entry_prices_val)

# 4. Evaluate
report = evaluate_v2(pred_df, y_up_val, y_down_val)
print_report(report)

# 5. Save if results are good
save_models_v2(models)
```

## Key Design Decisions

### Anti-Leakage
`build_targets()` receives only `future_prices` (data **after** `alert_date`).  
`build_v2_features()` receives only `price_history` up to and including `alert_date`.  
The caller is responsible for slicing correctly.

### Sequence-Aware Targets
Instead of measuring upside from the alert price:
1. Find deepest price in the 60d window → `max_drawdown_60d`
2. Measure recovery from that point → `max_return_60d`

This models the real buy-the-dip scenario: stock drops further, then recovers.

### Log Transform
```python
y_up_t   = log1p(max_return_60d)       # upside ≥ 0
y_down_t = -log1p(abs(max_drawdown_60d))  # downside ≤ 0
```
Reduces impact of outliers, stabilises XGBoost training.

### Evaluation Priority
1. **Spearman > 0.25** — model ranks alerts correctly
2. **Top-10% precision > baseline** — top picks are actually good
3. **Profit sim win rate > 55%** — usable in practice

Do NOT optimise for MSE or RMSE.

## Promoting to Production

Only replace `dip_model_stage1.pkl` / `dip_model_stage2.pkl` after:
- [ ] Walk-forward validation passes (no single time window dominates)
- [ ] Spearman ≥ 0.25 on out-of-sample data
- [ ] Top-10% precision > random baseline by ≥ 5pp
- [ ] Profit sim avg return > 0% at score threshold ≥ 1.5
- [ ] `ml_predictor.py` and `monthly_retrain.py` updated to use v2 API
