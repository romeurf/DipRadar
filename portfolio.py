"""
Carteira pessoal — apenas tickers públicos.

Os valores privados (shares, custos, valores CashBack, PPR)
estão em variáveis de ambiente no Railway.
O repositório público não contém nenhum dado financeiro pessoal.

Ver README.md → "Portfolio Env Vars" para a lista completa.
"""

import os

# ── Tickers ────────────────────────────────────────────────────────────
# EUNL.DE  = iShares Core MSCI World (Xetra)
# IS3N.AS  = iShares Core MSCI EM IMI (Euronext Amsterdam)
DIRECT_TICKERS   = ["NVO", "ADBE", "UBER", "EUNL.DE", "MSFT", "PINS", "ADP", "CRM", "VICI"]
CASHBACK_TICKERS = ["CRWD", "PLTR", "NOW", "DUOL"]

# ── Shares das posições directas (via env vars) ──────────────────────────
def _float_env(key: str, default: float = 0.0) -> float:
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default

HOLDINGS = [
    (sym, _float_env(f"HOLDING_{sym}") or None, None)
    for sym in DIRECT_TICKERS
]

# ── CashBack Pie (valores EUR por ticker, via env vars) ──────────────────
CASHBACK_EUR_VALUES = {
    sym: _float_env(f"CASHBACK_{sym}")
    for sym in CASHBACK_TICKERS
}

# ── PPR Invest Tendências Globais (proxy ACWI) ──────────────────────────
PPR_SHARES     = _float_env("PPR_SHARES")
PPR_AVG_COST   = _float_env("PPR_AVG_COST")
PPR_COST_TOTAL = PPR_SHARES * PPR_AVG_COST

# ── Helpers de FX ──────────────────────────────────────────────────────
USD_TICKERS = {
    "NVO", "ADBE", "UBER", "MSFT", "PINS", "ADP", "CRM", "VICI",
    "CRWD", "PLTR", "NOW", "DUOL", "ACWI",
}
EUR_TICKERS = {"EUNL.DE", "IS3N.AS", "ALV.DE"}
