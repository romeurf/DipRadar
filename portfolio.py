"""
Carteira pessoal — tickers, shares e preço médio de compra.
O heartbeat das 9h usa este módulo para calcular valor e P&L.

CashBack Pie (CRWD, PLTR, NOW, DUOL): valores em EUR directamente
da Trading212 — actualiza aqui quando fizeres depósitos.

PPR Invest Tendências Globais (Banco Invest): sem ticker público.
Usamos ACWI como proxy directional do NAV (correlação alta com MSCI World).
"""

# (symbol, shares, avg_cost_eur)
# avg_cost_eur = preço médio de compra em EUR (None = não disponível)
HOLDINGS = [
    # ── Posições directas ──────────────────────────────────────────────────────
    ("NVO",  142.33678955, None),
    ("ADBE",  16.27745882, None),
    ("UBER",  42.73462592, None),
    ("EUNL",  19.88552887, None),   # cotado em EUR (LSE)
    ("MSFT",   5.81970441, None),
    ("PINS",  95.00488077, None),
    ("ADP",    6.85764136, None),
    ("CRM",    6.17179094, None),
    ("VICI",  20.36983514, None),
]

# CashBack Pie — valor total em EUR por ticker (sem shares exactas).
# Actualiza após cada depósito mensal.
CASHBACK_EUR_VALUES = {
    "CRWD": 2.52,
    "PLTR": 2.20,
    "NOW":  6.45,
    "DUOL": 2.51,
}

# PPR Invest Tendências Globais
PPR_SHARES    = 917.2796
PPR_AVG_COST  = 7.2432          # EUR por unidade NAV
PPR_COST_TOTAL = PPR_SHARES * PPR_AVG_COST  # ~6 643 EUR

# Tickers cotados em USD (necessitam conversão USD→EUR)
USD_TICKERS = {
    "NVO", "ADBE", "UBER", "MSFT", "PINS", "ADP", "CRM", "VICI",
    "CRWD", "PLTR", "NOW", "DUOL", "ACWI",
}
EUR_TICKERS = {"EUNL"}
