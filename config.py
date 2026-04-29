"""
config.py — Watchlist estática do DipRadar.

Estes tickers servem de base para o backfill ML e os earnings alerts.
A watchlist dinâmica (adicionada via /watchlist add) é unida a esta lista
em runtime — duplicados são eliminados automaticamente.

Para corrigir um ticker inválido no Yahoo Finance, substitui aqui.
Exemplo: IS3N.L foi substituído por IEMA.L (iShares MSCI EM ESG Leaders, LSE).
"""

WATCHLIST: list[str] = [
    # ── US Large Cap / Blue Chip ───────────────────────────────────────────
    "AAPL",    # Apple
    "MSFT",    # Microsoft
    "GOOGL",   # Alphabet
    "AMZN",    # Amazon
    "NVDA",    # NVIDIA
    "META",    # Meta Platforms
    "BRK-B",   # Berkshire Hathaway B
    "JPM",     # JPMorgan Chase
    "V",       # Visa
    "MA",      # Mastercard
    "UNH",     # UnitedHealth
    "JNJ",     # Johnson & Johnson
    "PG",      # Procter & Gamble
    "HD",      # Home Depot
    "COST",    # Costco

    # ── US Growth / Tech ──────────────────────────────────────────────────
    "CRWD",    # CrowdStrike
    "DDOG",    # Datadog
    "SNOW",    # Snowflake
    "NET",     # Cloudflare
    "SHOP",    # Shopify
    "ENPH",    # Enphase Energy
    "ADBE",    # Adobe
    "NOW",     # ServiceNow
    "PANW",    # Palo Alto Networks

    # ── ETFs (LSE / Euronext) ─────────────────────────────────────────────
    "VWRL.L",  # Vanguard FTSE All-World (LSE)
    "CSPX.L",  # iShares Core S&P 500 (LSE)
    "EQQQ.L",  # Invesco NASDAQ-100 (LSE)
    "IEMA.L",  # iShares MSCI EM ESG Leaders (LSE) — substituiu IS3N.L
    "VUSA.L",  # Vanguard S&P 500 (LSE)
    "IWDA.L",  # iShares Core MSCI World (LSE)

    # ── Semiconductors ────────────────────────────────────────────────────
    "AMD",     # Advanced Micro Devices
    "AVGO",    # Broadcom
    "AMAT",    # Applied Materials
    "ASML",    # ASML Holding (Nasdaq)
    "TSM",     # Taiwan Semiconductor (ADR)
    "QCOM",    # Qualcomm
    "MU",      # Micron Technology
    "ARM",     # Arm Holdings

    # ── Healthcare / Biotech ──────────────────────────────────────────────
    "LLY",     # Eli Lilly
    "NVO",     # Novo Nordisk (ADR)
    "ABBV",    # AbbVie
    "BMY",     # Bristol-Myers Squibb
    "MRNA",    # Moderna

    # ── Financials ────────────────────────────────────────────────────────
    "GS",      # Goldman Sachs
    "MS",      # Morgan Stanley
    "BAC",     # Bank of America
    "SCHW",    # Charles Schwab

    # ── Energy / Commodities ──────────────────────────────────────────────
    "XOM",     # ExxonMobil
    "CVX",     # Chevron
    "NEE",     # NextEra Energy (renováveis)

    # ── Consumer / Retail ─────────────────────────────────────────────────
    "TSLA",    # Tesla
    "NKE",     # Nike
    "SBUX",    # Starbucks
    "MCD",     # McDonald's
]
