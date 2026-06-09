# 📡 DipRadar — Multi-Strategy Quant Portfolio System

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Railway](https://img.shields.io/badge/Deploy-Railway-000000?style=for-the-badge&logo=railway&logoColor=white)](https://railway.app/)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)](https://core.telegram.org/bots)

> **Sistema de trading quantitativo multi-estratégia.** Cinco radares paralelos identificam oportunidades, um motor unificado (`/cash`) distribui capital com awareness fiscal e gestão de concentração, e a execução é sempre manual pelo utilizador no T212.

---

## 🏛️ Os 5 Radares

| Radar | Objetivo | Métrica | Status |
|---|---|---|---|
| **DIP Scanner** | Quedas abruptas em empresas de qualidade | IC=+0.118 | ✅ Funcional |
| **Core Radar** | Compounders em desconto vs MA200 semanal | IC=+0.119 | ✅ Funcional |
| **Sector Rotation** | Top-3 sectores mensais | +15pp alpha vs SPY (backtest) | ✅ Funcional |
| **Momentum Scanner** | Breakouts com momentum durável | IC=0.0045 | ⚠️ Em retreinamento |
| **Conviction Radar** | Exit signals para posições MOMENTUM/DIP | Trailing stop ATR | ✅ Funcional |

---

## 💰 O Motor de Alocação: `/cash`

Comando único que substitui o legacy `/allocate` e `/flip`. Quando recebes inflow novo (ou só queres verificar estado):

```
/cash 1050        → propõe allocation completa
/cash 0           → health check sem deploy
/cash 1050 --no-rot      → skip rotation evaluation
/cash 1050 --no-sells    → ignora SELLs sugeridos
```

**Two-pass design**:

1. **Pass 1 — SELLs**: identifica posições para sair
   - ROTATION holdings fora do top-3 → tax-aware via `tax_engine.should_rotate()`
   - MOMENTUM/DIP com triggers HIGH severity → SELL imediato
   - MOMENTUM/DIP com triggers MED → EXIT WARNING (não vende)

2. **Pass 2 — BUYs**: ROI-ranked unified queue
   - Cada candidato (Rotation/Core/DIP) tem `annualized_expected_alpha × confidence`
   - Sort DESC → deploy top-down até budget esgotar
   - Concentration cap 25% soft / 30% hard por ticker
   - Resto → cash reserve (persistido para alerts futuros)

**Tax-aware rotation** (IRS Jovem 7% effective):
- Vende só se `score gap ≥ tax_friction × 500` OR override (exit_strength > 70 / secular_risk > 60 / current_score < 30)
- Para gain de 25% com 7% tax → required gap = 8.75 pontos

---

## 🤖 Comandos Principais

### Análise e alocação
| Comando | Descrição |
|---------|-----------|
| `/analisar <TICKER>` | Análise full: score, ML, valuation, sizing |
| `/cash <eur>` | Allocation proposal multi-estratégia |
| `/cash_reserve` | Vê reserva de cash + histórico |
| `/cash_reserve clear` | Limpa reserva (após deploy manual) |
| `/monthly_check` | Relatório mensal do portfolio com concentration % |
| `/core_picks [N] [min_score]` | Top picks do Core radar (default top 10, score≥60) |
| `/comparar T1 T2 ...` | Comparar scores de múltiplos tickers |
| `/historico <TICKER>` | Histórico de scores |

### Portfolio
| Comando | Descrição |
|---------|-----------|
| `/sync_portfolio` | Vê portfolio actual agrupado por bucket |
| `/sync_portfolio TICK:shares:price:BUCKET[:DATE]` | Bulk import positions |
| `/sync_portfolio --update-avg TICK:...` | Overwrite avg_price + shares |
| `/buy TICK PREÇO SHARES` | Comprar (debita liquidez) |
| `/sell TICK PREÇO [SHARES]` | Vender |
| `/carteira` | Snapshot real-time |

**Buckets**: `CORE` (hold-for-life) | `ROTATION` (sector ETFs) | `MOMENTUM` (DIP que recuperou) | `DIP` (active recovery)

### Sectors e Rotation
| Comando | Descrição |
|---------|-----------|
| `/sectors` | Rotation engine ranking real-time |
| `/sectors TICKER` | Detail per ETF |
| `/sectors holdings` | Status dos teus ETFs |
| `/admin_calibrate_rotation [years]` | Calibrar rotation alpha (linear/poly/buckets/lightgbm) |

### Conviction (exit signals para CORE/MOMENTUM/DIP holdings)
| Comando | Descrição |
|---------|-----------|
| `/conviction list` | Posições activas com trailing stop |
| `/conviction TICKER` | Deep dive de uma posição |
| `/conviction close TICKER PRECO [motivo]` | Fechar posição |
| `/conviction transition TICKER TIPO` | Mudar tipo (DIP→MOMENTUM→CORE) |

### Watchlist
| Comando | Descrição |
|---------|-----------|
| `/watchlist` | Estado da watchlist |
| `/watchlist add/rm TICKER` | Add/remove |

### ML e admin
| Comando | Descrição |
|---------|-----------|
| `/ml_accuracy` | Precision/recall/F1 vs outcomes reais |
| `/admin_check_config` | Estado dos modelos + IC |
| `/admin_retrain` | Retreinar DIP model |
| `/admin_momentum_retrain [force]` | Retreinar Momentum (force = full rebuild) |
| `/admin_core_debug TICKER` | Debug Core score per ticker |
| `/admin_calibrate_rotation` | Calibrar rotation alpha |
| `/scan` | Forçar scan imediato |
| `/health` | Dashboard sistema |
| `/backtest` | Resumo backtest |

---

## 🧮 Architecture overview

### Modelos ML

**DIP scanner** (`ml_predictor.py`):
- Target: `alpha_90d = log(stock_close_90d) - log(spy_close_90d)`
- IC=+0.118 validado (Spearman, walk-forward purged 5-fold)
- Champion: XGB-v2 / Ridge ensemble
- Sector models per GICS sector
- Score persiste `score_raw` + `win_prob` no prediction_log

**Core radar** (`core_radar.py`):
- Target: `alpha_1y` (1-year excess vs SPY)
- IC=+0.119, champion=GBR
- 20 features: price/MA, volume, sector context, fundamentals (rev_growth, gross_margin, ROE, fcf_yield_fund)
- **Option B FCF override**: stocks com investment-phase signature (rev growth >10% + GM >40% + ROE >15% + FCF <0) têm o feature FCF substituído pela mediana de treino antes da inferência. Permite que reinvesting compounders (AMZN-style, NVO em GLP-1 buildout) não sejam falsamente classificados como deterioração

**Momentum** (`momentum_radar/`):
- Multi-horizonte: alpha_30d, alpha_90d, alpha_180d
- IC actualmente baixo (em retreinamento — ver Phase 3)

**Sector rotation** (`sector_radar.py`):
- 24 ETFs (sector SPDR + thematic + alternatives)
- US tickers para data + UCITS mapping para deploy (SXR8/SEC0/IS3N/IUIT/etc.)
- Backtest validado: +15pp alpha vs SPY em 4 anos com monthly rebalance
- Calibration multi-method: linear → polynomial → buckets → lightgbm (best CV r²)

### Tax engine (`tax_engine.py`)
- IRS Jovem 75% exemption default (effective CGT 7%)
- Override via `EFFECTIVE_CGT_PCT` env var
- `should_rotate(unrealized_gain, current_score, target_score)` decide quando vender:
  - Override structural: `exit_strength > 70`, `secular_risk > 60`, `current_score < 30`
  - Loss positions → sempre rotate (loss = tax deduction)
  - Profit → `score gap ≥ tax_cost_pct × TAX_FRICTION_MULTIPLIER (500)`

### UCITS proxy (`ucits_proxy.py`)
- yfinance não tem UCITS (SEC0, SXR8, IS3N) — só US listed
- Computa PnL via US proxy (SOXX/SPY/EEM) % change desde entry_date
- Approximation: não captura FX drift EUR/USD

### Conviction radar (`conviction_radar.py`)
- Daily check 22h05 (Lisbon) seg-sex
- Monitoriza CORE/MOMENTUM/DIP positions sincronizadas
- Triggers: trailing stop ATR (12-21% dependendo de vol), structural decline, 8-K events
- Auto-sync `/sync_portfolio` → conviction records (ROTATION skipped)

### Cash reserve
- `/cash` persiste residual em `state.cash_reserve.json`
- DIP/Core alerts (verdict=COMPRAR, score≥70) mostram "reserva disponível €X"
- Sistema **nunca** auto-compra; apenas surface availability para deploy opportunistic
- `/cash_reserve clear` para limpar após deploy físico

---

## ⚙️ Env Vars

### Obrigatórias
| Variável | Descrição |
|----------|-----------|
| `TELEGRAM_TOKEN` | Token do bot |
| `TELEGRAM_CHAT_ID` | Chat ID |
| `MONTHLY_BUDGET_EUR` | Orçamento mensal (ex: `1050`) |
| `TZ` | `Europe/Lisbon` |

### Tax + allocation
| Variável | Default | Descrição |
|----------|---------|-----------|
| `EFFECTIVE_CGT_PCT` | `0.07` | Effective CGT (IRS Jovem 75% = 7%) |
| `SECTOR_CONCENTRATION_CAP` | `0.35` | Sector cap (legacy, /cash usa per-ticker) |

### Gestão de risco
| Variável | Default | Descrição |
|----------|---------|-----------|
| `POSITION_STOP_LOSS_PCT` | `0.12` | Stop-loss DIP positions |
| `MIN_DIP_SCORE` | `50` | DIP score mínimo para alert |
| `ML_SCORE_FLOOR` | `0.01` | Floor de alpha previsto |

### Retrain
| Variável | Default | Descrição |
|----------|---------|-----------|
| `RETRAIN_MAX_HOURS` | `3` | Watchdog timeout retrain |
| `INLINE_TICKER_TIMEOUT` | `25` | Download timeout por ticker |

### APIs
| Variável | Fonte | Descrição |
|----------|-------|-----------|
| `TIINGO_API_KEY` | api.tiingo.com | EOD prices de qualidade |
| `FRED_API_KEY` | fred.stlouisfed.org | Macro: recession prob |

---

## 🚀 Arranque

### Cold start
1. Configure Railway com `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, `MONTHLY_BUDGET_EUR`, `TZ=Europe/Lisbon`
2. Deploy → bot inicia automaticamente
3. Telegram: `/admin_check_config` para validar estado dos modelos
4. Se primeiro arranque: `/sync_portfolio TICK:shares:price:BUCKET` para registar posições existentes

### Fluxo mensal típico
1. Receber inflow (~€1050)
2. `/cash 1050` para proposta de allocation
3. Executar SELLs + BUYs manualmente em T212
4. Resto (se houver) fica como cash_reserve automaticamente
5. Durante o mês: DIP/Core alerts mostram reserve quando high-conviction signal → decide deploy opportunistic

### Conviction monitoring (automático)
- 22h05 Lisbon seg-sex: conviction daily check
- Trailing stop ATR + structural decline → alert push se trigger fire
- Tu decides em T212 se vendes (sistema nunca auto-executa)

---

## 🗑️ Deprecated (Phase 6 cleanup)

Removidos em favor do `/cash` unificado:
- ❌ `/allocate <TICKER>` (allocation_engine.py)
- ❌ `/flip` (Flip Fund concept)
- ❌ `HOLDING_*` env vars (use `/sync_portfolio`)
- ❌ `daily_flip_spent.json` tracker
- ❌ `_dipr_flip_log.json`

Categorias legacy (`CAT_HIGH_CONVICTION`, `CAT_GROWTH`) ainda em uso interno por score.py, watchlist.py, alert_db.py — vão ser removidas numa próxima sessão de refactor.

---

## 📊 Status atual (Junho 2026)

✅ **Funcionais**: DIP, Core, Sector Rotation, Conviction, Cash engine, Tax engine, UCITS proxy
⚠️ **Em retreino**: Momentum (IC=0.0045, debugging em curso)
🟡 **Tech debt**: `bot_commands.py` 424KB (Phase 7 split pendente), folder restructure pendente

---

## 📁 Estrutura

```
DipRadar/
├── main.py                          # Entry point + APScheduler
├── bot_commands.py                  # Telegram bot dispatcher (monolithic; split TBD)
├── state.py                         # JSON persistence layer
├── config.py                        # Central config
├── categories.py                    # Position category constants (legacy)
│
├── core_radar.py                    # Core radar (compounders)
├── core_bundle.py                   # Core ML bundle dataclass
├── sector_radar.py                  # Sector rotation engine
├── sector_backtest.py / sector_rotation_backtest.py
├── conviction_radar.py              # CORE/MOMENTUM/DIP exit monitor
│
├── ml_predictor.py / ml_engine.py / ml_features.py
├── monthly_retrain.py / model_registry.py
├── ml_training/                     # Trainer infra
│
├── momentum_radar/                  # Momentum scanner
├── scripts/                         # Build/calibrate/retrain scripts
│
├── tax_engine.py                    # IRS Jovem-aware rotation logic
├── ucits_proxy.py                   # UCITS PnL via US proxy
├── etf_universe.py                  # ETF ↔ UCITS mapping
│
├── portfolio.py                     # Cost basis ledger
├── position_db.py / position_monitor.py
├── paper_trading.py                 # Paper trading parallel ledger
│
├── fundamental_data.py / fundamental_history.py / fundamental_signals.py
├── tiingo_client.py / tiingo_fundamentals_client.py
├── macro_data.py / macro_semaphore.py
│
├── alert_db.py / score.py / signal_narrative.py
├── watchlist.py / persistent_dip.py
├── insider_radar.py / sectors.py / themes.py / universe.py
│
└── README.md (este ficheiro)
```

Para detalhes técnicos por módulo, ver docstrings inline.
