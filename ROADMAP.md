# DipRadar — Roadmap de Melhorias

## Backend & ML

### Dados e Pipeline de Treino
- **Tiingo Fundamentals (Starter plan)**: O plano gratuito só dá EOD prices. Com o plano Starter (~$10/mês) o regen usa Tiingo para fundamentais PIT em vez de EDGAR XBRL, muito mais rápido e completo. Fase 5 (Form 4, 8-K) passaria de 45 min para ~5 min.
- **Fase 5 EDGAR no regen**: Actualmente o EDGAR cache é em memória por sessão. Com disco persistente por ticker (ex: `edgar_submissions_{ticker}.json` com TTL 7 dias), o regen re-run seria instantâneo para a parte EDGAR.
- **Universe snapshot maturidade**: O dataset de treino cresce com alertas de produção. Com 6+ meses de alertas maturados (90d), a qualidade do modelo vai melhorar significativamente sem qualquer intervenção.
- **MomentumRadar ML**: Actualmente rule-based. Quando houver dados suficientes (3-6 meses de snapshots), treinar modelo ScaledRidge com `forward_return_30d` como target.

### Modelo ML
- **Sector models com walk-forward CV completo**: Actualmente os 11 sector models usam split 80/20 simples. Com dados suficientes, implementar 5-fold walk-forward purged por sector.
- **`max_drawdown_90d` como risk target**: Actualmente `model_down` treina em `max_drawdown_60d`. Deveria ser 90d para alinhar com o horizonte do modelo. Requer parquet regenerado com a coluna correcta.
- **Ensemble stacking**: Os resultados do walk-forward CV mostram que o ensemble IC^2-weighted já supera o single best. Explorar meta-learner com features de contexto (VIX regime, sector stress).
- **yfinance → Tiingo para preços PIT**: yfinance tem instabilidade na API. Com Tiingo Starter, os preços históricos são mais fiáveis e mais rápidos de descarregar.

### Infraestrutura
- **Railway Volume**: Aumentar para 5GB quando o dataset de treino crescer (~500MB/ano).
- **Separar processo de regen do bot**: O regen como subprocess do bot é frágil. Com Railway Jobs (ou Cron separado), o regen correria independentemente e o bot nunca ficaria bloqueado.
- **Parquet incremental no regen**: Em vez de re-processar todas as 36,929 linhas, só processar linhas novas (alert_dates após o último regen). Reduziria o regen mensal para ~5 min.
- **Health checks automáticos**: Adicionar `/health` check periódico (ex: ping às 09h) que detecta se o bot ficou preso e envia alerta.

---

## Frontend & Comunicação com Utilizador

### Telegram
- **Markdown V2**: Migrar de Markdown V1 para MarkdownV2 do Telegram. Elimina todos os "400 Bad Request" actuais. Requer escape de caracteres especiais (`_`, `*`, `[`, `)`, `.`, `!`).
- **Inline keyboards**: Para alertas COMPRAR, adicionar botões inline "Ver análise" / "Alocar capital" / "Ignorar" em vez de comandos manuais.
- **Alertas silenciosos**: Modo silencioso para alertas MONITORIZAR (sem notificação sonora) e modo urgente para COMPRAR com stock em queda acelerada.
- **Relatório semanal automático**: Às segundas-feiras, resumo da semana: P&L do portfolio, alertas gerados, posições abertas, comparação vs SPY.

### Análise e Reporting
- **`/analisar` com gráfico**: Adicionar chart ASCII ou imagem (via matplotlib) com preço + RSI + volume nos últimos 90 dias. Telegram suporta imagens.
- **`/relatorio` com comparação histórica**: Mostrar os últimos 6 meses de performance DipRadar vs MomentumRadar vs SPY lado a lado.
- **Alertas de sector**: Quando ≥3 stocks do mesmo sector disparam em simultâneo, enviar alerta de "rotação sectorial" com contexto macro.
- **Watchlist proactiva**: Em vez de só alertar quando o preço atinge o critério, adicionar análise semanal da watchlist com score e distância ao trigger.

### Interface Web (futuro)
- **Dashboard simples**: FastAPI + HTMX com:
  - Portfolio overview em tempo real
  - Histórico de alertas com P&L realizado
  - Curva de equity vs SPY
  - Feature importance do modelo
- **Mobile-friendly**: O Telegram já é mobile. O dashboard seria para análise mais profunda no PC.

---

## Escalabilidade (multi-utilizador)

### Quando houver mais utilizadores
- **Multi-tenant**: Cada utilizador tem o seu `TELEGRAM_CHAT_ID`, `MONTHLY_BUDGET_EUR`, portfolio e paper trades separados. O modelo ML é partilhado.
- **Base de dados**: Migrar `alert_db.csv`, `paper_trades.json`, `positions.json` para PostgreSQL (Railway inclui). Permite queries complexas, histórico ilimitado, e multi-utilizador.
- **API REST**: Expor endpoints para que um frontend web possa consumir os dados do bot (portfolio, alertas, performance).
- **Notificações por email**: Além do Telegram, enviar resumo mensal por email (via SendGrid free tier).

---

## Quick Wins (implementar na próxima iteração)

1. Migrar Telegram Markdown V1 → V2 (elimina todos os 400 errors)
2. Tiingo Starter (~$10/mês) → regen de 45 min para 5 min
3. Parquet incremental no regen → regen mensal em 5 min
4. Railway Jobs para regen separado do bot
5. Relatório semanal automático (segundas, 09h)
6. Gráfico de preço no `/analisar`
