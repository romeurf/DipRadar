# 📡 DipRadar — Caçador Quantitativo de Assimetrias

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Railway](https://img.shields.io/badge/Deploy-Railway-000000?style=for-the-badge&logo=railway&logoColor=white)](https://railway.app/)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)](https://core.telegram.org/bots)

> **Bot de trading quantitativo que caça dips em empresas de qualidade, dimensiona o capital de forma inteligente e evolui continuamente com dados reais.**

---

## 💡 A Filosofia

O DipRadar executa uma única filosofia com precisão institucional: **Dip Hunting**. Varre o mercado à procura de quedas abruptas, separa empresas de excelência do lixo especulativo, e diz-te exactamente quanto capital arriscar na recuperação — sem intervenção humana.

---

## 🏛️ Os 4 Pilares

### 1. O Radar (Machine Learning)
Modelo preditivo treinado com dados históricos validados com regras de walk-forward CV (padrão institucional):
- **Target**: `alpha_90d = log1p(stock_return_90d) − log1p(spy_return_90d)` — excesso sobre o S&P 500 em 90 dias
- **IC actual**: 0.124 (IC SR: 4.49) | 100% folds positivos
- **Features (33)**: técnicas + fundamentais PIT + sentimento SEC + short interest + earnings call tone
- **Modelos**: ScaledRidge (champion), XGBoost DART, LightGBM GOSS, RF com walk-forward CV purged
- **Sector models**: modelos especializados por grupo sectorial (Tech/HC, Fin/Industrial, Commodity, Defensivos)
- **Retreino**: automático Dia 1 do mês, watchdog 3h, notificação de início + conclusão

### 2. O Escudo (Análise Fundamental)
Sistema de filtragem em 3 camadas:
- **Score V2 (0–100)**: Quality (40%) + Value (20%) + Timing (20%) + Divergência (20%)
- **Red Flags**: FCF negativo → pre-profit cap; P/E > 200 → rejeição; D/E extremo → penalidade
- **Dividend Safety**: FCF/dividendos < 0.8 → penalidade 15% (excl. REITs/Utilities que usam FFO)
- **8-K Veto**: restatement ou default → muda COMPRAR para MONITORIZAR automaticamente
- **RSI Crossover**: só entra em COMPRAR se RSI < 42 (oversold confirmado)
- **Volume Capitulação**: preferencialmente dips com volume > média (pânico esgotado)

### 3. O Tesoureiro (Allocation Engine)
Não diz apenas "Compra ServiceNow". Calcula exactamente quanto:
- **Sizing dinâmico**: edge × R:R × max_position (Kelly-inspired)
- **Correlação de posições**: penaliza entries correlacionadas (>0.65) com portfolio actual
- **Sector concentration cap**: 35% máximo por sector (configurável)
- **Portfolio VaR**: se VaR > 8%, reduz sizing automaticamente
- **Pyramiding**: entrada faseada 65%/35% para dips severos
- **Scale Out / Moonbag**: realiza 50% no target, deixa 50% a rolar
- **Early Alpha Capture**: sai quando captura 70%+ do alpha em <50% do tempo
- **Stop-loss**: sai automaticamente se preço cair >12% do entry (configurável)

### 4. Evolução Contínua (MLOps)
Sistema vivo que aprende com os seus próprios erros:
- **Paper trading autónomo**: cada COMPRAR cria posição simulada com o budget real (€850/mês)
- **Self-evaluation**: mede se bate o SPY sem intervenção humana
- **Saída antecipada inteligente**: fecha paper trades quando target atingido cedo
- **ML Accuracy**: `/ml_accuracy` mede precision, recall, F1, Brier score vs outcomes reais
- **Retreino mensal**: watchdog 3h, notificação de início e conclusão
- **Drift detection**: win_prob médio monitorizado vs baseline de treino

---

## 🤖 Comandos Telegram

### Mercado e análise
| Comando | Descrição |
|---------|-----------|
| `/scan` | Força scan imediato |
| `/analisar <TICKER>` | Análise completa com narrativa natural dos sinais |
| `/comparar T1 T2 ...` | Comparar scores de 2-5 tickers |
| `/historico <TICKER>` | Histórico de scores |
| `/rejeitados` | Stocks rejeitados hoje (com razão) |
| `/tier3` | Gems do último resumo de fecho |

### Performance e simulação
| Comando | Descrição |
|---------|-----------|
| `/performance [data] [score]` | Retorno histórico seguindo o bot |
| `/paper_portfolio` | Paper trading: retorno do bot vs SPY |
| `/paper_portfolio open` | Posições simuladas abertas |
| `/paper_portfolio 6` | Últimos 6 meses de simulação |
| `/ml_accuracy` | Precisão real do modelo vs outcomes reais |

### Temas / Trends
| Comando | Descrição |
|---------|-----------|
| `/themes` | Ver temas em trend (fotónica, GLP-1, IA...) |
| `/add_theme key label TICK1,TICK2` | Adicionar tema |
| `/remove_theme key` | Remover tema |

### Carteira e posições
| Comando | Descrição |
|---------|-----------|
| `/carteira` | Snapshot em tempo real |
| `/portfolio` | Posições activas com P&L |
| `/sync_portfolio TICK:shares:preco ...` | Sincronizar carteira via Telegram |
| `/buy TICK PREÇO SHARES` | Registar compra |
| `/sell TICK PREÇO [SHARES]` | Registar venda |
| `/liquidez [+\|-VALOR]` | Ajustar saldo |
| `/allocate <TICKER>` | Sugestão de alocação com VaR e sector check |
| `/flip` | Flip Fund P&L |

### Watchlist
| Comando | Descrição |
|---------|-----------|
| `/watchlist` | Estado da watchlist |
| `/watchlist add TICKER` | Adicionar |
| `/watchlist rm TICKER` | Remover |

### ML e retreino
| Comando | Descrição |
|---------|-----------|
| `/mldata` | Stats da base de dados ML |
| `/mldata update` | Forçar update de outcomes |
| `/admin_retrain [dry-run]` | Disparar retreino (notificação de início imediata) |
| `/retrigger` | Alias de `/admin_retrain` |
| `/admin_regen_parquet [--targets-only]` | Regenerar parquet (EDGAR PIT + alpha_90d) |
| `/admin_set_floor <valor>` | Ajustar floor de IC |

### Sistema e diagnóstico
| Comando | Descrição |
|---------|-----------|
| `/status` | Estado do bot |
| `/health` | Dashboard: RAM, CPU, drift, APIs |
| `/admin_check_config` | Verificar env vars |
| `/admin_test_feed TICKER` | Testar pipeline de dados |
| `/backtest` | Resumo do backtest |

---

## ⚙️ Env Vars

### Obrigatórias
| Variável | Descrição |
|----------|-----------|
| `TELEGRAM_TOKEN` | Token do bot |
| `TELEGRAM_CHAT_ID` | Chat ID |
| `MONTHLY_BUDGET_EUR` | Orçamento mensal total (ex: `1050`) |
| `TZ` | `Europe/Lisbon` |

### Orçamento e alocação
| Variável | Default | Descrição |
|----------|---------|-----------|
| `PAPER_BUDGET_EUR` | — | Capital para dip hunting (ex: `850` se investes €200 em ETFs) |
| `ETF_DCA_EUR` | `0` | ETFs mensais a excluir do paper trading (ex: `200`) |
| `FLIP_FUND_EUR` | 10% do budget | Capital do Flip Fund (auto-deriva se não definido) |
| `SECTOR_CONCENTRATION_CAP` | `0.35` | Limite de exposição por sector (35%) |

### Gestão de risco
| Variável | Default | Descrição |
|----------|---------|-----------|
| `POSITION_STOP_LOSS_PCT` | `0.12` | Stop-loss automático (12% abaixo do entry) |
| `MIN_DIP_SCORE` | `50` | Score V2 mínimo (sobe automaticamente em stress sectorial) |

### Retrain
| Variável | Default | Descrição |
|----------|---------|-----------|
| `INLINE_TICKER_TIMEOUT` | `25` | Segundos por ticker no download de preços |
| `INLINE_BUDGET_MINUTES` | `60` | Minutos máximo para computar alpha_90d |
| `RETRAIN_MAX_HOURS` | `3` | Watchdog: mata o retrain se demorar mais |

### APIs gratuitas
| Variável | Fonte | Descrição |
|----------|-------|-----------|
| `TIINGO_API_KEY` | api.tiingo.com | EOD de qualidade |
| `ALPHAVANTAGE_API_KEY` | alphavantage.co (25 req/dia) | Revisões de analistas |
| `FMP_API_KEY` | financialmodelingprep.com (250 req/dia) | Upgrades/downgrades |
| `FRED_API_KEY` | fred.stlouisfed.org | Recession probability |
| `TAVILY_API_KEY` | tavily.com | Catalisadores e notícias |

### Carteira (privados — nunca no repo)
```
HOLDING_NVO=<shares>,<avg_cost>
HOLDING_ADBE=<shares>,<avg_cost>
HOLDING_MSFT=<shares>,<avg_cost>
...
PPR_SHARES=<shares>
PPR_AVG_COST=<preco_medio>
```
> **Alternativa**: `/sync_portfolio NVO:25:85.50 ADBE:8:420` no Telegram

---

## 🧠 Features do Modelo ML (33 features)

### Técnicas (base)
RSI, drawdown 52w, momentum 1m/3m/6m, beta, vol_of_vol, bb_width, VIX regime, VIX percentil 1y, SPY RSI, sector relative, volume zscore, up days pct, true range, drop × drawdown

### Macro
VIX, sector alert count 7d, sector drawdown 5d

### Sentimento e qualidade (Fase 5 — SEC EDGAR + yfinance)
| Feature | Fonte | O que mede |
|---------|-------|-----------|
| `insider_buy_recent` | SEC Form 4 | Houve compra de insider nos últimos 30 dias? |
| `insider_buy_amount_score` | SEC Form 4 XML | Magnitude normalizada da compra (0-1) |
| `recent_8k_score` | SEC 8-K item codes | Tipo de evento: -1=restatement, +1=M&A target |
| `earnings_call_tone` | SEC 8-K texto | Sentimento da earnings call [-1=pessimista, +1=confiante] |
| `short_interest_pct` | yfinance | Short interest % do float |
| `short_interest_trend` | yfinance | Variação do short interest vs mês anterior |
| `consecutive_red_days` | Price history | Dias consecutivos em queda (capitulação) |
| `ma_200d_ratio` | Price history | Price / MA200 (quão abaixo da média de longo prazo) |
| `earnings_beat_rate` | yfinance | % de últimos 4 trimestres com EPS beat |
| `analyst_rating` | yfinance | Consenso analistas: 1=Strong Buy … 5=Strong Sell |

---

## 📊 Arquitectura de Risco

```
Novo alerta COMPRAR →
  ├── RSI < 42? (oversold confirmado)     ← se não: MONITORIZAR
  ├── 8-K restatement recente?            ← se sim: veto → MONITORIZAR
  ├── Volume spike no dip?                ← se não: menos confiança
  ├── Score V2 > threshold macro-ajustado?← threshold sobe em stress sectorial
  │
  ├── Allocation Engine:
  │   ├── Sector concentration < 35%?
  │   ├── Correlação < 0.65 com portfolio?
  │   ├── Portfolio VaR OK?
  │   └── Sizing = edge × R:R × budget
  │
  └── Paper trade criado automaticamente

Position Monitor (diário 22h):
  ├── Target atingido → TAKE_PROFIT
  ├── 70%+ alpha em <50% tempo → EARLY_ALPHA_CAPTURE
  ├── Preço < entry × 0.88 → STOP_LOSS
  ├── 2+ critérios de deterioração → STRUCTURAL_DECLINE
  └── Rotina → update silencioso
```

---

## 🎯 Paper Trading Autónomo

O bot simula as suas próprias recomendações para provar se bate o mercado:

```
Dia 1 do mês → budget renovado (PAPER_BUDGET_EUR)
Cada COMPRAR → posição criada automaticamente
22h50 diariamente → fecha posições que:
  - Atingiram o target de preço
  - 70%+ do alpha em <50% do tempo (early exit)
  - Esgotaram o período de holding (90 dias)
Dia 1 → relatório: "Se seguisses o bot, ganharias X% vs SPY Y%"
```

**Comandos**:
- `/paper_portfolio` — performance dos últimos 3 meses
- `/paper_portfolio 6` — últimos 6 meses
- `/paper_portfolio open` — posições abertas

---

## ⚠️ Disclaimer
*DipRadar é uma ferramenta de research e screening. Não constitui aconselhamento financeiro. Faz sempre a tua própria análise antes de investir.*
