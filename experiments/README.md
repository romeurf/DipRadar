# experiments/

Código ML exploratório e scripts de uso local que **não são usados em produção** (não importados pelo `main.py`).

## Ficheiros aqui presentes

| Ficheiro | Origem | Motivo |
|---|---|---|
| `ml_engine.py` | raiz | Ensemble XGBoost+LightGBM alternativo — substituído por `ml_predictor.py` |
| `ml_pipeline.py` | raiz | Pipeline de treino alternativo — substituído por `monthly_retrain.py` |
| `ml_scan.py` | raiz | Scan ML alternativo — não chamado em lado nenhum |
| `ml_ensemble.py` | raiz | Exploração de ensemble — código morto |
| `ml_walk_forward.py` | raiz | Walk-forward validation — útil localmente, não em prod |
| `bootstrap_ml.py` | raiz | Bootstrap histórico do dataset — uso manual/local |
| `train_model.py` | raiz | Script de treino manual — só para uso local |

## Stack de produção activa

```
ml_predictor.py     ← predição ao vivo (importado por main.py)
ml_features.py      ← feature engineering (importado por ml_predictor.py)
monthly_retrain.py  ← retreino mensal agendado (importado por main.py)
label_resolver.py   ← flywheel de dados diário (importado por main.py)
```
