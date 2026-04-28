"""
Carteira pessoal — apenas tickers públicos.

Os valores privados (shares, custos, valores CashBack, PPR)
estão em variáveis de ambiente no Railway.
O repositório público não contém nenhum dado financeiro pessoal.

Ver README.md → "Portfolio Env Vars" para a lista completa.

Env vars necessárias por posição directa:
  HOLDING_{SYM}          — número de acções (ex: HOLDING_ADBE=5.2)
  HOLDING_{SYM}_AVG      — custo médio em EUR (ex: HOLDING_ADBE_AVG=320.50)

Para o PPR:
  PPR_SHARES             — número de unidades de participação
  PPR_AVG_COST           — custo médio por unidade em EUR

CashBack Pie (valor EUR actual por ticker):
  CASHBACK_{SYM}         — ex: CASHBACK_CRWD=13.68

Flip Fund (capital separado para flips táticos):
  FLIP_FUND_EUR          — actualiza no Railway após cada depósito ou flip
"""

import os

# ── Tickers ────────────────────────────────────────────────────────────────────
# EUNL.DE  = iShares Core MSCI World (Xetra)
# IS3N.AS  = iShares Core MSCI EM IMI (Euronext Amsterdam)
DIRECT_TICKERS   = ["NVO", "ADBE", "UBER", "EUNL.DE", "MSFT", "PINS", "ADP", "CRM", "VICI"]
CASHBACK_TICKERS = ["CRWD", "PLTR", "NOW", "DUOL"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def _float_env(key: str, default: float = 0.0) -> float:
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default

# ── Shares + custo médio das posições directas (via env vars) ────────────────
# HOLDING_{SYM}      = número de acções
# HOLDING_{SYM}_AVG  = custo médio em EUR por acção
HOLDINGS = [
    (
        sym,
        _float_env(f"HOLDING_{sym}") or None,
        _float_env(f"HOLDING_{sym}_AVG") or None,
    )
    for sym in DIRECT_TICKERS
]

# ── CashBack Pie (valores EUR por ticker, via env vars) ─────────────────────
CASHBACK_EUR_VALUES = {
    sym: _float_env(f"CASHBACK_{sym}")
    for sym in CASHBACK_TICKERS
}

# ── PPR Invest Tendências Globais (proxy ACWI) ──────────────────────────
PPR_SHARES     = _float_env("PPR_SHARES")
PPR_AVG_COST   = _float_env("PPR_AVG_COST")
PPR_COST_TOTAL = PPR_SHARES * PPR_AVG_COST

# ── Helpers de FX ───────────────────────────────────────────────────────
USD_TICKERS = {
    "NVO", "ADBE", "UBER", "MSFT", "PINS", "ADP", "CRM", "VICI",
    "CRWD", "PLTR", "NOW", "DUOL", "ACWI",
}
EUR_TICKERS = {"EUNL.DE", "IS3N.AS", "ALV.DE"}

# ── Flip Fund (capital separado para flips táticos) ──────────────────────────
# Actualiza FLIP_FUND_EUR no Railway após cada depósito ou execução de flip.
# Não entra no total da carteira — é capital separado para operações rápidas.
FLIP_FUND_EUR = _float_env("FLIP_FUND_EUR")
