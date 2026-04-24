# 📈 Stock Alert Bot

Monitoriza **todas as acções** para quedas ≥10% e filtra automaticamente
por fundamentos intactos, com análise específica por sector.

## Como funciona

```
1. A cada 30min → FMP API: lista de maiores quedas do dia
2. Filtro: queda ≥10% + market cap ≥$1B (sem penny stocks)
3. Para cada acção que passa: análise fundamental por sector
4. Score: COMPRAR / MONITORIZAR / EVITAR
5. Se COMPRAR ou MONITORIZAR → alerta Telegram com:
   - P/E actual vs histórico 5 anos vs sector
   - FCF yield
   - DCF simplificado com WACC por sector
   - Revenue growth
   - EV/EBITDA
   - Upside dos analistas
   - 3 notícias recentes
6. Às 18h: resumo diário das maiores quedas
```

## Setup

### 1. FMP API (Financial Modeling Prep)
- Regista em https://site.financialmodelingprep.com/register
- Free tier: 250 calls/dia — chega para uso normal
- Copia a tua API key

### 2. Telegram Bot
- Fala com @BotFather → `/newbot` → copia o **token**
- Fala com o teu bot para activar
- Vai a `https://api.telegram.org/bot<TOKEN>/getUpdates` → copia o `chat.id`

### 3. Deploy Railway (gratuito)

```bash
# 1. Cria conta em railway.app com GitHub
# 2. New Project → Deploy from GitHub
# 3. Faz push deste código para um repo teu
# 4. Em Variables adiciona:
TELEGRAM_TOKEN=xxxxx
TELEGRAM_CHAT_ID=xxxxx
FMP_API_KEY=xxxxx

# Opcional:
DROP_THRESHOLD=10      # % de queda mínima (default 10)
MIN_MARKET_CAP=1000000000  # cap mínima em $ (default 1B)
```

### 4. (Alternativa) Render.com
- New Web Service → GitHub → Start Command: `python main.py`
- Adiciona as mesmas env vars

---

## Ficheiros

```
stock_alert_bot/
├── main.py         # Loop principal + Telegram
├── fmp_client.py   # API client FMP (screening + fundamentals + news)
├── sectors.py      # Thresholds por sector + scoring
├── valuation.py    # DCF, WACC, margem de segurança
├── requirements.txt
└── railway.toml
```

## Personalizar

**Threshold de queda:**
```python
# Em main.py ou via env var DROP_THRESHOLD
DROP_THRESHOLD = 10  # alerta com queda ≥10%
```

**Sectores e thresholds** (em `sectors.py`):
Cada sector tem: pe_max, pe_fair, fcf_yield_min, ev_ebitda_max, etc.
Podes ajustar os valores para o teu critério de "fundamentos intactos".

**Formato da mensagem Telegram (exemplo):**
```
📉 NVO — Novo Nordisk A/S
Queda: -12.3% hoje | Preço: $36.5 | Cap: $160B
Sector: 🏥 Saúde

Veredito: 🟢 COMPRAR
  P/E 18x — 18% abaixo do justo (22x) para o sector
  FCF yield 5.2% — muito atrativo

📊 Fundamentos:
  • P/E: 18.2x vs hist. 35.0x (-48%)
  • FCF Yield: 5.2%
  • DCF intrínseco: $58.2 (margem: +59%)
  • EV/EBITDA: 12.1x
  • Revenue growth: 22.1%
  • Gross margin: 84.0%
  • Dividendo: 7.10% (payout 52%)
  • Target analistas: $47.0 (+29%)

📰 Notícias:
  • Novo Nordisk beats Q1 earnings... (reuters.com)
  • FDA approves new GLP-1 indication... (bloomberg.com)
```
