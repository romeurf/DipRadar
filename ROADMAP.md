# DipRadar — Roadmap

Última actualização: 2026-06-11.

Para o plano detalhado da próxima fase de trabalho ver
[`docs/plans/cash_tax_aware.md`](docs/plans/cash_tax_aware.md) (quando criado)
ou o plan-file activo em `.claude/plans/`.

---

## Estado actual

O sistema está estável em produção com 5 radares activos:

| Radar | Estado | Notas |
|---|---|---|
| DIP scanner | ✅ | ML model treinado, scoring v2, alertas intraday |
| CORE radar | ✅ | ML model treinado (CV IC ≥ 0.04), scan 22h Lisboa + bloco inline em alertas DIP (Option C) |
| Sector rotation | ✅ | FSM com hysteresis (BUY_ZONE/SELL_ZONE/NEUTRAL), backtest histórico activo |
| Momentum radar | ⚠️ | IC ≈ −0.004 — modelo anti-predictivo. Investigação pendente em Phase 3 |
| Conviction radar | ✅ | Paralelo ao DipRadar mas isolado do ML (não contamina o treino) |
| Position monitor | ✅ | Vigilante diário com trailing stop ATR + stop-loss + deterioração estrutural |

**Arquitectura:**
- Estrutura modular: `core/`, `data/`, `ml/`, `portfolio/`, `scanners/`, `utils/`, `commands/`, `momentum_radar/`, `ml_training/`.
- `bot_commands.py` em split incremental (Phase 8); ~~~7900 linhas, alvo final < 50KB com lógica em `commands/*.py`.
- Persistência em `/data` (Railway volume) com escrita atómica via `_save_json`.
- Live price via yfinance `fast_info.last_price` no momento de gerar alertas; RSI re-injecta o tick intraday.

---

## Em curso

### Phase 2 — Tax-aware `/cash` + DCA (CRÍTICO antes de 2026-06-24)

Plano completo em `.claude/plans/snappy-tumbling-chipmunk.md`. Próximo inflow €1050 — sem este fix o `/cash` toma decisões mal calibradas:

- **2.1 Rotation gate tax-aware**: novo `tax_engine.should_rotate()` com IRS Jovem 7% effective rate. Hoje usa baseline implícito de 28% → recomenda HOLD demasiado conservador.
- **2.2 DCA em CORE existentes**: remover filtro `core_held` no `/cash`. Adicionar concentration cap 25% soft / 30% hard. NVO no dip a -15% deveria ser top-up, não exclusão.
- **2.3 100% deploy + ROI-ranked queue**: substituir os 33%/30%/20% fixos por ranking unificado de annualized expected alpha. Cash bucket ≈ €0 quando há sinais; cresce só genuinamente.
- **2.4 portfolio_value wiring**: passar valor da carteira a `/cash` e `/monthly_check` para a math de concentração.

Pré-requisito: `scripts/calibrate_rotation_alpha.py` (existe, foi calibrada na sessão `c2e127f`).

### Phase 8 — Split `bot_commands.py` incremental

- 8.1 themes (-70 linhas) ✅
- 8.2 analysis (-150) ✅
- 8.3 conviction (-255) ✅
- 8.4 portfolio_extras (-680) ✅
- 8.5 misc (-450) ✅
- 8.6 core_cmds (-203) ✅
- **8.7 sync_portfolio** (~318) — pendente
- **8.8 admin handlers** (~3000) — sessão dedicada

Restam ~3500 linhas em bot_commands.py; alvo final < 1500 (só dispatcher + wiring).

---

## Próximas iterações

### Phase 3 — Momentum radar fix

Modelo actual com IC ≈ −0.004 (anti-predictivo). Suspeitos:
- `_has_infra` fallback no `momentum_radar/trainer.py` constrói numpy ndarray sem feature names → ordem inconsistente entre treino e inferência.
- `dropna` cascade pode estar a decimar o dataset de 85k linhas.
- Alpha-target alignment em `build_dataset.py` — possível erro de sinal ou off-by-one temporal.

Plano em `snappy-tumbling-chipmunk.md` Phase 3.

### Phase 1 — Quick wins (não-bloqueantes)

- **1.1 Core score saturation**: `core_radar.py:133` usa clamp linear que satura em `alpha_pred ≥ 0.20`. NVO/GIS/CMG todos em 100 esconde ranking. Trocar por `tanh` compression (~5 linhas).
- **1.2 DIP `rho_alpha` persistence**: bundle do KEPT_FLOOR retrain salvou `rho_mean=None`. Investigar `_to_dict` no trainer ou alias-on-read.

### Phase 4 (já completado parcialmente) — Estrutura de pastas

Phase 7 já migrou para `core/`, `data/`, `ml/`, `portfolio/`, `scanners/`, `utils/`, `commands/`, `momentum_radar/`. Falta:

- Limpeza dos imports ainda absolutos para o root antigo.
- Mover `scripts/` para a raiz (já lá está).
- Consolidação de duplicação (`get_eod_prices` em `data_feed.py` vs `tiingo_client.py` etc.).

---

## Bugs reais (Perplexity audit 2026-06)

| Bug | Estado |
|---|---|
| BUG-26 trailing_high persist gap | ✅ Fixed `f1aa66c` |
| BUG-27 yfinance calls duplicadas em momentum | ✅ Fixed `f1aa66c` |
| BUG-28 utcnow vs Lisbon tz inconsistency | ✅ Fixed `f1aa66c` |
| BUG-32 `_detect_structural_decline` 2× por tick | ✅ Fixed `f1aa66c` |
| BUG-29 potencial circular `core_cmds → bot_commands` | ⏭️ Não confirmado (lazy-proxy resolve) |
| BUG-30 `_send_photo` undefined | ✅ Fixed `be29701` |
| BUG-31 `initial_pred_alpha` morto em posições legacy | ❓ Discussão pendente (sintoma desejado?) |
| BUG-33-35 minor | ⏭️ Não prioritário |
| BUG-01 sector bonus viola sector cap | ❌ Pendente (allocation_engine ainda existe? verificar) |
| BUG-02 `target_price` sempre None | ❌ Pendente |
| BUG-16 parquets de treino no git | ❌ Pendente (`.gitignore` update) |
| BUG-18 posições legacy invisíveis no monitor | ❌ Pendente (migração via `position_db.add_position`) |

---

## Aspirações de longo prazo

- **Markdown V2 no Telegram**: elimina os "400 Bad Request" actuais.
- **Inline keyboards** para alertas COMPRAR: botões "Ver análise" / "Alocar capital" / "Ignorar".
- **Dashboard web** (FastAPI + HTMX): portfolio em tempo real, histórico de alertas, curva de equity vs SPY, feature importance.
- **Multi-utilizador**: cada utilizador com `TELEGRAM_CHAT_ID`, `MONTHLY_BUDGET_EUR`, portfolio e paper trades separados; modelo ML partilhado.
- **Postgres**: migrar `alert_db.csv`, `paper_trades.json`, `positions.json` para Railway Postgres. Multi-tenant + queries complexas.
- **Email summaries** (mensal) via SendGrid free tier.

---

## Princípios de evolução

1. **Stability first**: fix de bugs em produção tem prioridade sobre features novas. O staleness fix (`838c6b7`, `36784e3`) e os 4 bugs do monitor (`f1aa66c`) são exemplos típicos.
2. **Refactor incremental**: split do bot_commands.py em batches pequenos (Phase 8.1 → 8.8) com smoke-tests entre cada commit.
3. **Aditivo, não destrutivo**: Option C inline CORE block adicionou helpers em vez de tocar nos scan paths existentes.
4. **Tax-aware allocation**: tudo o que decide allocation tem de considerar IRS Jovem 7% effective rate enquanto o utilizador qualificar (< €20k income).
5. **No premature abstraction**: 3 linhas duplicadas é preferível a uma abstracção mal pensada.
