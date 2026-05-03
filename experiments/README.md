# experiments/

Código ML exploratório e helpers do pipeline de treino. **Não é importado em produção** directamente pelo `main.py`, mas o notebook v3.1 importa de `experiments/ml_v2/` para construir o dataset de treino.

## Conteúdo actual

```
experiments/
├── README.md                    ← este ficheiro
└── ml_v2/
    ├── README.md                ← documentação da pipeline v2/v3
    ├── DipRadar_v3_Training.ipynb  ← **notebook activo** (Colab — gera dip_models_v3.pkl)
    ├── pipeline.py              ← FEATURE_COLUMNS_V2 + build_v2_features + build_targets
    ├── build_dataset.py         ← helpers para construir dataset_v3* a partir de price_history
    └── evaluation.py            ← Spearman, top-K, walk-forward metrics
```

## Stack de produção activa (raiz)

```
ml_predictor.py     ← predição ao vivo (importado por main.py)
ml_features.py      ← feature engineering (FEATURE_COLUMNS, _FALLBACK, add_*)
ml_engine.py        ← shim de compat (delega para ml_predictor.ml_score)
monthly_retrain.py  ← retreino mensal — actualmente DESACTUALIZADO p/ v3 (ver docs/auto_retrain_plan.md)
label_resolver.py   ← back-fill diário de labels (alert_db outcomes)
```

## Notebooks anteriores (removidos)

`colab_bootstrap.ipynb`, `DipRadar_v1_Backtest.ipynb`, `DipRadar_v1_Production.ipynb` e `DipRadar_v2_Training.ipynb` foram removidos no PR #14 — eram da era v1/v2 (modelo binário com `dip_model_stage{1,2}.pkl`). O fluxo actual é único: `DipRadar_v3_Training.ipynb`.
