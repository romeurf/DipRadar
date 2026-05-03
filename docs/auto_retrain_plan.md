# Auto-Retrain Plan (v3.x → mensal sem intervenção)

> Estado actual: `monthly_retrain.py` está agendado no cron mas **partido** desde o refactor v3 (importa `train_model.train_all` que foi removido; produz `dip_model_stage{1,2}.pkl` que não existe na arquitectura v3 single-bundle).

## O que queremos

1. No dia 1 de cada mês, o bot retreina o modelo automaticamente, **sem qualquer intervenção manual** (sem Colab, sem upload de pickles).
2. O dataset cresce mensalmente com:
   - **Alertas reais** (`alert_db.csv`) cujo outcome 60d já maturou → label resolvido por `label_resolver.py`.
   - **Snapshots diários** (`universe_snapshot.parquet`) onde a janela 60d já passou → outcome resolvido via lookup intra-tabela.
3. Walk-forward gating: candidate só substitui produção se passar critérios objectivos (ρ_alpha, top-k PnL, Brier).
4. Atomic deploy: rollback automático se o gating falhar.
5. Telegram notification: decisão (PROMOTED / KEPT / FAILED) + delta de métricas.

## Dependências (já existem)

- `alert_db.py` + `label_resolver.py` — back-fill de outcomes 60d (chamado do cron diário)
- `universe_snapshot.py` — append diário ao parquet
- `data_feed.py` — Tiingo / yfinance / Stooq fallback
- `experiments/ml_v2/pipeline.py` — `FEATURE_COLUMNS_V2`, `build_v2_features`, `build_targets`
- `ml_features.py` — `FEATURE_COLUMNS`, `add_derived_features`, `add_momentum_features`
- `ml_predictor.py` — `_to_dict`, `joblib.load`, hot-reload por `mtime`

## Dependências (a criar / corrigir)

### Bloco A — extrair `train_v31()` para módulo importável

O notebook v3.1 tem toda a lógica de treino, mas é Jupyter — não chamável a partir de `monthly_retrain.py`. Solução: extrair para `ml_training/train_v31.py`:

```
ml_training/
├── __init__.py
├── train_v31.py        ← train_v31(parquet_path, output_dir) → DipModelsV3
├── walk_forward.py     ← purged_walk_forward_cv(...) — usado pelo train + gating
├── calibrator.py       ← fit_isotonic_oof(...) — score_calibrator
└── feature_pipeline.py ← build_dataset_v31(price_history, alerts) — anti-leakage
```

Cada função deve ser testável isoladamente. O notebook passa a ser uma fina camada que importa estes módulos + faz Colab-specific stuff (download, pip installs, plots).

### Bloco B — Reescrever `monthly_retrain.py` para v3

Substituir os imports e caminhos:

| Antes (v1/v2) | Depois (v3) |
|---|---|
| `from train_model import train_all` | `from ml_training.train_v31 import train_v31` |
| `dip_model_stage{1,2}.pkl` | `dip_models_v3.pkl` |
| `ml_report.json` | `ml_report_v3.json` |
| Gating via `auc_pr_test` | Gating via `rho_alpha_mean` |
| `FLOOR_AUC_PR_DEFAULT = 0.18` | `FLOOR_RHO_ALPHA_DEFAULT = 0.20` |
| Floor file `ml_floor_auc_pr.json` | `ml_floor_rho_alpha.json` |

Manter:
- Estrutura `build_training_input()` com 3 fontes (bootstrap + alert_db + snapshot)
- Atomic copy + replace (tmpfile → rename)
- Archive em `/data/archive/dip_models_v3_<timestamp>.pkl`
- Telegram notification

### Bloco C — Resolver origem do `ml_training_merged.parquet`

**Pergunta do utilizador**: "porque usamos o `ml_training_merged.parquet`? É antigo, não será melhor retificar?"

Resposta: o parquet é o output histórico do `bootstrap_ml.py` (Tier A+B+C, 2014-2025). Foi gerado **uma vez** durante a inicialização e nunca mais foi actualizado. **Limitações actuais**:
- Estrutura de features incompleta (não inclui `relative_drop`, `sector_alert_count_7d`, etc.)
- Cobertura termina em ~2025 (o sample do utilizador atinge May 2026 só porque o dataset foi preparado num momento ambíguo)
- 17.368 rows é uma boa base mas **não cresce** — todo o crescimento vem do `alert_db` (~10/dia) e `universe_snapshot` (~800/dia)

**Plano de regeneração**:

Opção 1 (simples) — manter o parquet como **âncora histórica imutável**:
- Não regenerar; tratar como o "Big Bang" do dataset.
- O crescimento vem dos 2 outros caminhos.
- Risco: o parquet velho pode ter features ligeiramente diferentes do que produzimos hoje (skew). Mitigação: manter o `build_training_input()` a fazer schema unification + fillna.

Opção 2 (limpa) — regenerar o parquet com o pipeline v3.1 actual:
- Correr `experiments/ml_v2/DipRadar_v3_Training.ipynb` cell de "build dataset" sobre os mesmos tickers + datas históricas
- Output: `ml_training_merged_v31.parquet` com schema 100% compatível com v3.1
- Vantagem: zero skew com features actuais. Desvantagem: ~30 min de fetch yfinance, e qualquer divergência futura repete o problema.

**Recomendação**: Opção 1 inicialmente — adicionar schema unification em `build_training_input()` (preencher features novas com `_FALLBACK[col]` se ausente). Migrar para Opção 2 só quando houver dispersão notável entre o parquet histórico e os dados frescos.

### Bloco D — Watchdog de saúde do retreino

`/admin_health` deve mostrar:
- Última data de retreino bem-sucedido
- N de retreinos consecutivos KEPT (sem promoção) — alerta se >3 (degradação contínua)
- Último N candidate ρ_alpha vs production ρ_alpha
- Cobertura: % de outcomes resolvidos no `alert_db` no último mês
- Idade do `dip_models_v3.pkl` em produção

## Ordem de implementação sugerida

| Sprint | Entrega | Esforço | Bloqueio |
|---|---|---|---|
| 1 | Bloco A — extrair lógica do notebook para `ml_training/` | ~3h | nenhum |
| 2 | Bloco B — `monthly_retrain.py` chama o novo pipeline | ~2h | depende do Sprint 1 |
| 3 | Bloco C — schema unification em `build_training_input()` | ~1h | depende do Sprint 1 |
| 4 | Bloco D — `/admin_health` e Telegram notifications detalhadas | ~1h | depende dos Sprints 1-3 |
| 5 | Validação end-to-end: dry-run em produção 2-3 semanas antes de Activar cron | — | — |

**Total**: ~7h de trabalho dev + 2-3 semanas de observação.

## Anti-leakage no auto-retrain

Garantias que o pipeline tem de manter quando crescer mensalmente:

1. **Outcome resolution só usa dados maduros**: `alert_date + 60d ≤ today`.
2. **Features só são computadas a partir de dados ≤ alert_date**: `build_v2_features` recebe `price_history[:alert_date]`.
3. **Walk-forward CV mantém purga de 21d**: cada fold tem `train_end + 21d ≤ test_start`.
4. **Sample weights mantêm half-life 3y**: regimes antigos não são zerados.

## Próximos passos imediatos (após este PR)

1. **PR seguinte**: extrair `ml_training/train_v31.py` (Bloco A). Smoke test local com o `ml_training_merged.parquet` actual.
2. **PR seguinte+1**: reescrever `monthly_retrain.py` (Bloco B). Dry-run no Railway primeiro (`python monthly_retrain.py --dry-run`).
3. **Activar cron**: depois de 1-2 retreinos manuais bem-sucedidos via Colab v3.1.

## Riscos

- **Retreino mensal pode degradar**: se o último mês teve regime atípico (ex: crash 2020), o ρ_alpha cai e o gating mantém o modelo antigo — **comportamento desejado**.
- **alert_db poluído**: se houver alertas com `outcome_label` errado (back-fill bug), contaminam o training. Mitigação: `_validate_outcomes()` em `build_training_input` que descarta linhas com `return_3m` improvável (ex: > 200% sem catalisador conhecido).
- **Volume `/data/` cheio**: archive de pickles pode crescer. Cleanup automático: manter os últimos 6 archives, descartar os mais antigos.

## Critérios de sucesso

- ≥3 retreinos automáticos consecutivos sem intervenção humana
- ρ_alpha em produção mantém-se ≥ 0.25 nos últimos 90 dias de observação
- Drift detection (via `prediction_log.py`) não acende alarme nos 90 dias
- `/admin_health` reporta tudo verde
