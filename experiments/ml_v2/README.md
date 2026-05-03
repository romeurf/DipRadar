# DipRadar ML v3 — Dual-regressor (alpha) Pipeline

> **Status:** v3.1 (notebook em `DipRadar_v3_Training.ipynb`) é o único pipeline de treino activo.

## O que é v3

| | v3 (actual) |
|---|---|
| Tarefa | Regressão dupla — `model_up` prevê `alpha_60d`, `model_down` prevê drawdown |
| Modelo | XGBRegressor (champion seleccionado por walk-forward CV) |
| Target principal | `alpha_60d = max_return_60d − spy_max_return_60d` (60d forward window) |
| Métrica primária | Spearman ρ_alpha (out-of-fold) |
| Métricas secundárias | Top-20% PnL alpha, Brier score (calibrator), per-fold ρ stability |
| Output | Score = `pred_up` (alpha estimado), prob calibrada via IsotonicRegression |
| Bundle | `dip_models_v3.pkl` (single file, joblib) |

## Estrutura

```
experiments/ml_v2/
├── README.md                       — este ficheiro
├── DipRadar_v3_Training.ipynb      — notebook completo (Colab Run-all)
├── pipeline.py                     — FEATURE_COLUMNS_V2 (30) + build_v2_features + build_targets
├── build_dataset.py                — helpers para juntar parquet + price_history
└── evaluation.py                   — Spearman, top-K, profit sim, walk-forward
```

## Quick Start (notebook)

1. Abre `DipRadar_v3_Training.ipynb` no Colab
2. Runtime → Run all (~10-20 min)
3. No fim, escolhe o deploy:
   - **A**: `git push` directo do `dip_models_v3.pkl` (Railway redeploy automático)
   - **B**: Upload via GitHub API → `/admin_load_models <url>` no Telegram

## Anti-leakage

- `build_v2_features()` só recebe `price_history` até ao `alert_date` (inclusive).
- `build_targets()` só recebe `future_prices` (após `alert_date`).
- `alpha_60d` é calculado entry-vs-max em `[alert_date+1, alert_date+60d]`.
- O caller (notebook) é responsável por slice correcto.

## Critérios de promoção (v3.1)

Antes de promover um candidate v3.x para `dip_models_v3.pkl` em produção:

- [ ] `rho_alpha_mean > 0.30` (out-of-fold)
- [ ] `topk_pnl_alpha > 0` (top-20%, alpha real, sem beta de mercado)
- [ ] `Brier score OOF < 0.20` (calibrator)
- [ ] Walk-forward estável: nenhum fold individual com ρ < −0.05
- [ ] `score_calibrator` presente no bundle e callable
- [ ] `ml_predictor._to_dict(bundle)` normaliza todos os campos sem erro

Ver `docs/retrain_plan_v3_1.md` § Métricas para detalhes.

## Histórico

- **v1**: classificação binária (RandomForest, `dip_model_stage{1,2}.pkl`) — removido em PR #14.
- **v2**: regressão dupla XGBoost com `max_return_60d` / `max_drawdown_60d` — substituído por v3 (target alpha-vs-SPY).
- **v3**: target = `max_return_60d`, 5 folds CV, sigmoide hardcoded para win_prob.
- **v3.1**: target = `alpha_60d`, 10 folds + purga 21d, +4 features, IsotonicRegression calibrator, sample weights half-life 3y.
