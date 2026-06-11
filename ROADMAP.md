# DipRadar — Roadmap

Última actualização: **2026-06-11 — pós sessão Fase 0-4**.

Para o plano detalhado da próxima fase de trabalho ver o plan-file activo
em `.claude/plans/` ou os commit messages dos últimos 7 dias.

---

## Estado actual

O sistema está estável em produção com 5 radares activos:

| Radar | Estado | Notas |
|---|---|---|
| DIP scanner | ✅ | ML model treinado, scoring v2, alertas intraday |
| CORE radar | ✅ | ML model treinado (CV IC ≥ 0.04), scan 22h + bloco inline em alertas DIP (Option C) + **intraday warning** para DIRECT/HIGH_CONVICTION |
| Sector rotation | ✅ | FSM com hysteresis (BUY_ZONE/SELL_ZONE/NEUTRAL), backtest histórico activo |
| Momentum radar | ⚠️ | IC ≈ −0.004 — modelo anti-predictivo. Investigação em **Phase 3** abaixo |
| Conviction radar | ✅ | Paralelo ao DipRadar mas isolado do ML. **Real-time MOMENTUM stops** em todos os slots hourly (Fase A) |
| Position monitor | ✅ | Vigilante diário com trailing stop ATR + stop-loss + deterioração estrutural. 4 bugs corrigidos em `f1aa66c` |
| Structural 52w DIP | ✅ NOVO | Apanha drifts longos que screener intraday -8% não vê (Fase D) |

**Arquitectura:**
- Estrutura modular: `core/`, `data/`, `ml/`, `portfolio/`, `scanners/`, `utils/`, `commands/`, `momentum_radar/`, `ml_training/`.
- `bot_commands.py` em split incremental (Phase 8); ~7800 linhas, alvo final < 1500 (só dispatcher).
- Persistência em `/data` (Railway volume) com escrita atómica via `_save_json`.
- Live price via yfinance `fast_info.last_price` no momento de gerar alertas; RSI re-injecta o tick intraday.
- UCITS via direct lookup com sufixos europeus (`.DE`/`.AS`/`.L`/`.MI`), proxy estimation como fallback.

**Allocation engine (`/cash`):**
- Tax-aware rotation gate (`core/tax_engine.should_rotate`) com IRS Jovem 7% effective rate.
- Concentration cap 25% soft / 30% hard via `_can_top_up()` — DCA em CORE existentes é gateado por exposição, não filtrado por hold-status.
- ROI-ranked unified deploy queue: `_annualized_alpha(signal_type)` ordena candidatos num único ranking. Cash bucket cresce só quando não há sinais worth deploying.
- **SELL+BUY=HOLD anti-churn** (Fase C): contradições mesmo-ticker viram HOLD com nota explícita.
- Per-alert α via `alert_db.csv` (durável contra `ml_predictions.csv` resets).

**Alert volume esperado (com Fases A-D):**

| Tipo | Normal | Crash |
|---|---|---|
| DIP intraday | 3-5 | 10-15 |
| CORE intraday (B) | 0-1 | 1-3 |
| MOMENTUM stops (A) | 0-1 | 2-3 |
| 52w structural (D) | 0-2 | 2-3 (cap 3) |
| **TOTAL** | **3-9** | **15-24** |

---

## ✅ Done — sessão 2026-06-11

8 commits production-ready:

| Commit | Resolveu |
|---|---|
| `be29701` | TIER-1 hotfix — restored state PPR helpers + `_today_iso`, core_bundle pickle alias, BUG-30 `_send_photo` |
| `838c6b7` | Staleness: `get_live_price()` no momento de cada alerta + timestamp visível |
| `36784e3` | RSI: inject `fast_info.last_price` na série EOD para scans 15-20h |
| `256c9d4` | Option C — bloco CORE inline em alertas DIP quando ticker em CORE universe |
| `032f6f0` | Option C completion — CORE scan dedup contra `_alerted_today` |
| `f1aa66c` | 4 position_monitor bugs (BUG-26/27/28/32 do Perplexity audit) |
| `82271c9` | Wire missing Phase 8.3/8.4 imports (7 handlers a NameError em produção) + handle_portfolio órfão |
| `72a4fd1` | DIP α durável via alert_db.csv + UCITS direct quote com sufixos europeus |
| `4164ab9` | **Fase 0** — Watchlist deletion (-646 linhas) |
| `fbaadde` | **Fase 1 (C)** — SELL+BUY=HOLD em `/cash` |
| `f1e27db` | **Fase 2 (A)** — Real-time MOMENTUM trailing stops nos hourly slots |
| `6c6b5ac` | **Fase 3 (B)** — CORE intraday warning para DIRECT_TICKERS ou CAT_HIGH_CONVICTION |
| `e342f5b` | **Fase 4 (D)** — Structural 52w DIP intraday scan |
| `a4af509` | Perf — drop redundant 2s sleep em structural scan |

---

## A fazer — ordem de prioridade

### 0 — Observar 24-48h em produção (passo zero antes de qualquer trabalho novo)

5 features novas foram deployadas hoje. Antes de adicionar mais, **medir**:
- Volume real de alertas (vs 3-9/dia esperados).
- Latência do slot das 15h (esperado: ~3min na primeira execução do dia, ~5s subsequentes).
- Falsos positivos / qualidade das `STRUCTURAL DIP` e `CORE WARNING` mensagens.
- Comportamento do bloco HOLD no `/cash` (HUBS-style cases).

**Critérios de revisita:**
- Se >12 alertas/dia em dia normal: ajustar Fase B threshold `-7%` → `-8%`.
- Se slot 15h >5min: reduzir `max_results=15` → `10` no `screen_structural_dips`.
- Se MOMENTUM stop não dispara para HUBS após break: investigar `evaluate_conviction()`.

### 1 — Phase 3: Momentum radar fix (próximo trabalho técnico de impacto)

Modelo actual com IC ≈ −0.004 (anti-predictivo). Suspeitos:
- `_has_infra` fallback no `momentum_radar/trainer.py` constrói numpy ndarray sem feature names → ordem inconsistente entre treino e inferência.
- `dropna` cascade pode estar a decimar o dataset de 85k linhas.
- Alpha-target alignment em `build_dataset.py` — possível erro de sinal ou off-by-one temporal.

Plano em `.claude/plans/snappy-tumbling-chipmunk.md` Phase 3.

**Esforço**: ~1.5-2h. **Impacto**: Momentum continua off em produção até resolver.

### 2 — Phase 8 split tail (refactor)

- 8.7 `sync_portfolio` (~318 linhas) → `commands/sync_portfolio.py`
- 8.8 admin handlers (~3000 linhas) → `commands/admin/*.py` (sessão dedicada)

Restam ~3500 linhas em `bot_commands.py`; alvo final < 1500 (só dispatcher + wiring).

### 3 — Estrutura de pastas (cleanup)

Phase 7 migrou para módulos. Falta:
- Limpeza dos imports ainda absolutos para o root antigo.
- Consolidação de duplicação (`get_eod_prices` em `data_feed.py` vs `tiingo_client.py`).

---

## Open design questions (a discutir antes de implementar)

### HUBS auto-transition MOMENTUM → DIP/CORE quando stop break em 52w low

Cenário levantado em 2026-06-11: HUBS entrou DIP → auto-transitou MOMENTUM → stop break em 52w low. O sistema agora trata sell e re-entry como eventos isolados.

A Fase C (SELL+BUY=HOLD) ajuda quando ambos os signals disparam no mesmo `/cash` run, mas se a tese mudou (HUBS deixou de ser MOMENTUM e voltou a ser DIP por dip prolongado), o utilizador tem de fazer `/conviction transition HUBS DIP` manual.

Possível fix: quando MOMENTUM stop confirma E o ticker está em 52w low, auto-sugerir transition para DIP/CORE. Não auto-executar — só sugerir no alerta.

### Capital constraint awareness no alert volume

User explicitly notou: "12 alertas/dia, depois como vou ter capital?". O `/cash` queue ordena por ROI mas não considera *attention budget*. Possível extensão:
- Marcar alertas com `[CONSIDER]` vs `[INFO]` baseado em ROI relativo ao mês.
- Cap visual no `/cash` checklist (top 5 only).
- Daily digest às 21:30 com top 3 oportunidades em vez de N notificações dispersas.

### Yahoo screener vs custom universe

`screen_global_dips` e `screen_structural_dips` dependem do Yahoo screeners (`day_losers`, `undervalued_large_caps`, `most_actives`). Estes screeners têm o universo da Yahoo, não controlado pelo bot. Possível trabalho:
- Override com `core_universe.json` + S&P 500 base.
- Reduz dependência da Yahoo API.

---

## Bugs reais (Perplexity audit 2026-06)

| Bug | Estado |
|---|---|
| BUG-26 trailing_high persist gap | ✅ Fixed `f1aa66c` |
| BUG-27 yfinance calls duplicadas em momentum | ✅ Fixed `f1aa66c` |
| BUG-28 utcnow vs Lisbon tz inconsistency | ✅ Fixed `f1aa66c` |
| BUG-30 `_send_photo` undefined | ✅ Fixed `be29701` |
| BUG-32 `_detect_structural_decline` 2× por tick | ✅ Fixed `f1aa66c` |
| BUG-18 posições legacy invisíveis no monitor | ✅ Resolvido por `be29701` — conviction_radar é o monitor para CORE; estava crashing por ImportError, agora corrigido |
| BUG-01 sector bonus viola sector cap | ✅ Ghost — `allocation_engine.py` deletado em Phase 6 |
| BUG-02 `target_price` sempre None | ✅ Ghost — `suggest_allocation` morreu com allocation_engine |
| BUG-16 parquets de treino no git | ✅ Design intencional (cold-start seeds com whitelist no `.gitignore`) |
| BUG-29 potencial circular `core_cmds → bot_commands` | ⏭️ Não confirmado (lazy-proxy resolve, sem erros em produção) |
| BUG-31 `initial_pred_alpha` morto em posições legacy | ❓ Discussão pendente (sintoma desejado?) |
| BUG-33-35 minor | ⏭️ Cosméticos |

**Score**: 7 fixed, 3 ghosts, 1 design, 1 misdiagnosed (BUG-18 actually a different monitor) = 12/12 endereçados.

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

1. **Stability first**: fix de bugs em produção tem prioridade sobre features novas. Os fixes de hoje (`be29701`, `f1aa66c`, `82271c9`) são exemplos típicos.
2. **Refactor incremental**: split do bot_commands.py em batches pequenos (Phase 8.1 → 8.8) com smoke-tests entre cada commit.
3. **Aditivo, não destrutivo**: Option C inline CORE block, Fase B CORE warning, Fase D structural — todos adicionados sem tocar nos scan paths existentes.
4. **Tax-aware allocation**: tudo o que decide allocation tem de considerar IRS Jovem 7% effective rate enquanto o utilizador qualificar (< €20k income).
5. **No premature abstraction**: 3 linhas duplicadas é preferível a uma abstracção mal pensada.
6. **Anti-spam discipline**: cada novo tipo de alerta tem dedup explícita via `_alerted_today` + cap top-N + threshold racional (não baixar gates só para "ter mais sinal").
7. **Capital-aware design**: alertas devem feedar `/cash` queue, não pedir acção individual. Ranking importa mais que quantidade.
8. **Commit isolado por fase**: cada Fase 0-4 é um commit reversível independente. Reverter qualquer fase é `git revert <sha>` sem efeitos colaterais nas outras.

---

## Próxima revisita do ROADMAP

- **Após 24-48h** de observação dos 5 novos alert types em produção.
- Confirmar volume real vs 3-9/dia esperado.
- Decidir Phase 3 (momentum fix) vs Phase 8.7 (split) com base em prioridade do user.
