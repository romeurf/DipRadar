"""
portfolio.py — Gestão de carteira activa e liquidez.

Estrutura do ficheiro _dipr_portfolio.json:
{
  "liquidity": 0.0,          # saldo disponivel em EUR
  "positions": {
    "CRWD": {
      "symbol":       "CRWD",
      "name":         "CrowdStrike",
      "shares":       3.0,
      "avg_price":    245.50,    # USD
      "total_cost":   736.50,    # USD (shares * avg_price)
      "category":     "Rotação",
      "entry_score":  82,
      "entry_date":   "29/04/2026",
      "entry_date_iso": "2026-04-29",
      "last_score":   82,        # actualizado no scan diário
      "last_price":   245.50,    # actualizado no scan diário
      "last_update":  "29/04 12:00",
      "degradation_alerted": false
    }
  }
}

ETFs (EUNL, IEMA, etc.) são registados com entry_score=None e category="ETF".
Não participam no motor de degradação de tese.

Compatibilidade com código legado:
  DIRECT_TICKERS / CASHBACK_TICKERS — mantidos para não quebrar imports existentes.
  suggest_position_size() — mantida para o Flip Fund.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from universe import is_etf


# ─────────────────────────────────────────────────────────────────────────
# Compatibilidade legado — não remover (usados em watchlist.py / bot_commands.py)
# ─────────────────────────────────────────────────────────────────────────
DIRECT_TICKERS   = ["NVO", "ADBE", "UBER", "EUNL", "MSFT", "PINS", "ADP", "CRM", "VICI"]
CASHBACK_TICKERS = ["CRWD", "PLTR", "NOW", "DUOL"]

USD_TICKERS = {
    "NVO", "ADBE", "UBER", "MSFT", "PINS", "ADP", "CRM", "VICI",
    "CRWD", "PLTR", "NOW", "DUOL", "ACWI",
}
EUR_TICKERS = {"EUNL", "EUNL.DE", "IS3N.DE", "ALV.DE", "IEMA"}


def _float_env(key: str, default: float = 0.0) -> float:
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


FLIP_FUND_EUR = _float_env("FLIP_FUND_EUR")


# ─────────────────────────────────────────────────────────────────────────
# Persistência
# ─────────────────────────────────────────────────────────────────────────
_DATA_DIR       = Path("/data") if Path("/data").exists() else Path("/tmp")
_PORTFOLIO_FILE = _DATA_DIR / "_dipr_portfolio.json"


def _read_raw() -> dict:
    try:
        if _PORTFOLIO_FILE.exists():
            return json.loads(_PORTFOLIO_FILE.read_text())
    except Exception as e:
        logging.warning(f"[portfolio] read error: {e}")
    return {"liquidity": 0.0, "positions": {}}


def _write_raw(data: dict) -> None:
    try:
        _PORTFOLIO_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logging.warning(f"[portfolio] write error: {e}")


# ─────────────────────────────────────────────────────────────────────────
# API pública — leitura
# ─────────────────────────────────────────────────────────────────────────

def get_liquidity() -> float:
    return _read_raw().get("liquidity", 0.0)


def get_positions() -> dict:
    return _read_raw().get("positions", {})


def get_position(symbol: str) -> dict | None:
    return get_positions().get(symbol.upper())


def get_active_symbols() -> list[str]:
    return list(get_positions().keys())


# ─────────────────────────────────────────────────────────────────────────
# /buy — registar compra
# ─────────────────────────────────────────────────────────────────────────

def buy(
    symbol:      str,
    price:       float,
    shares:      float,
    category:    str       = "",
    entry_score: int | None = None,
    name:        str       = "",
) -> dict:
    """
    Regista uma compra.
    - Se posição já existir: DCA (preço médio ponderado).
    - Desconta custo da liquidez (pode ficar negativa — aviso no output).
    - ETFs: category="ETF", entry_score=None automático.
    Devolve dict com resultado.
    """
    symbol = symbol.upper().strip()
    data   = _read_raw()
    cost   = round(price * shares, 2)

    if is_etf(symbol):
        category    = "ETF"
        entry_score = None

    positions = data.get("positions", {})
    now_str   = datetime.now().strftime("%d/%m/%Y")
    now_iso   = datetime.now().date().isoformat()
    now_hm    = datetime.now().strftime("%d/%m %H:%M")

    if symbol in positions:
        # DCA
        pos        = positions[symbol]
        new_shares = round(pos["shares"] + shares, 6)
        new_cost   = round(pos["total_cost"] + cost, 2)
        new_avg    = round(new_cost / new_shares, 4)
        pos["shares"]      = new_shares
        pos["total_cost"]  = new_cost
        pos["avg_price"]   = new_avg
        pos["last_update"] = now_hm
        if name:
            pos["name"] = name
        action = "avg_down"
    else:
        positions[symbol] = {
            "symbol":              symbol,
            "name":                name or symbol,
            "shares":              round(shares, 6),
            "avg_price":           round(price, 4),
            "total_cost":          cost,
            "category":            category or "Desconhecida",
            "entry_score":         entry_score,
            "entry_date":          now_str,
            "entry_date_iso":      now_iso,
            "last_score":          entry_score,
            "last_price":          price,
            "last_update":         now_hm,
            "degradation_alerted": False,
        }
        action = "new"

    old_liq           = data.get("liquidity", 0.0)
    new_liq           = round(old_liq - cost, 2)
    data["liquidity"] = new_liq
    data["positions"] = positions
    _write_raw(data)

    logging.info(
        f"[portfolio] BUY {symbol} x{shares} @ ${price} "
        f"| custo ${cost} | liq {old_liq:.2f}€ → {new_liq:.2f}€"
    )
    return {
        "symbol":      symbol,
        "shares":      shares,
        "price":       price,
        "cost":        cost,
        "action":      action,
        "liquidity":   new_liq,
        "liq_warning": new_liq < 0,
        "position":    positions[symbol],
    }


# ─────────────────────────────────────────────────────────────────────────
# /sell — registar venda
# ─────────────────────────────────────────────────────────────────────────

def sell(
    symbol: str,
    price:  float,
    shares: float | None = None,
) -> dict | None:
    """
    Regista uma venda. shares=None = venda total.
    Adiciona proceeds à liquidez.
    Devolve None se posição não existir.
    """
    symbol = symbol.upper().strip()
    data   = _read_raw()
    positions = data.get("positions", {})

    if symbol not in positions:
        logging.warning(f"[portfolio] SELL {symbol}: posição não encontrada")
        return None

    pos         = positions[symbol]
    sell_shares = min(shares if shares is not None else pos["shares"], pos["shares"])
    proceeds    = round(price * sell_shares, 2)
    avg         = pos["avg_price"]
    pnl         = round((price - avg) * sell_shares, 2)
    pnl_pct     = round((price - avg) / avg * 100, 2) if avg else 0
    remaining   = round(pos["shares"] - sell_shares, 6)

    if remaining <= 0.0001:
        del positions[symbol]
        action = "closed"
    else:
        pos["shares"]      = remaining
        pos["total_cost"]  = round(avg * remaining, 2)
        pos["last_update"] = datetime.now().strftime("%d/%m %H:%M")
        action = "partial"

    old_liq           = data.get("liquidity", 0.0)
    new_liq           = round(old_liq + proceeds, 2)
    data["liquidity"] = new_liq
    data["positions"] = positions
    _write_raw(data)

    logging.info(
        f"[portfolio] SELL {symbol} x{sell_shares} @ ${price} "
        f"| P&L ${pnl:+.2f} ({pnl_pct:+.1f}%) | liq {old_liq:.2f} → {new_liq:.2f}€"
    )
    return {
        "symbol":      symbol,
        "shares_sold": sell_shares,
        "price":       price,
        "proceeds":    proceeds,
        "pnl":         pnl,
        "pnl_pct":     pnl_pct,
        "action":      action,
        "remaining":   remaining,
        "liquidity":   new_liq,
    }


# ─────────────────────────────────────────────────────────────────────────
# Liquidez — ajuste manual
# ─────────────────────────────────────────────────────────────────────────

def add_liquidity(amount: float, note: str = "") -> float:
    """Adiciona `amount` à liquidez. Devolve novo saldo."""
    data = _read_raw()
    old  = data.get("liquidity", 0.0)
    new  = round(old + amount, 2)
    data["liquidity"] = new
    _write_raw(data)
    logging.info(f"[portfolio] Liquidez +{amount}€ | {old:.2f} → {new:.2f} | {note}")
    return new


def set_liquidity(amount: float) -> float:
    """Define liquidez directamente (correcção manual)."""
    data = _read_raw()
    data["liquidity"] = round(amount, 2)
    _write_raw(data)
    logging.info(f"[portfolio] Liquidez definida para {amount:.2f}€")
    return round(amount, 2)


# ─────────────────────────────────────────────────────────────────────────
# Actualização diária (chamada pelo scan em main.py)
# ─────────────────────────────────────────────────────────────────────────

def update_position_data(
    symbol:   str,
    price:    float,
    score:    int | None = None,
    category: str | None = None,
) -> None:
    """Actualiza preço e score de uma posição. Chamado pelo scan diário."""
    symbol = symbol.upper().strip()
    data   = _read_raw()
    pos    = data.get("positions", {}).get(symbol)
    if not pos:
        return
    pos["last_price"]  = price
    pos["last_update"] = datetime.now().strftime("%d/%m %H:%M")
    if score is not None:
        pos["last_score"] = score
    if category is not None:
        pos["category"] = category
    _write_raw(data)


def mark_degradation_alerted(symbol: str) -> None:
    """Marca que já foi enviado alerta de degradação para este ticker."""
    symbol = symbol.upper().strip()
    data   = _read_raw()
    pos    = data.get("positions", {}).get(symbol)
    if pos:
        pos["degradation_alerted"] = True
        _write_raw(data)


def reset_degradation_flag(symbol: str) -> None:
    """Limpa a flag de degradação (ex: após melhoria de score)."""
    symbol = symbol.upper().strip()
    data   = _read_raw()
    pos    = data.get("positions", {}).get(symbol)
    if pos:
        pos["degradation_alerted"] = False
        _write_raw(data)


# ─────────────────────────────────────────────────────────────────────────
# Flip Fund — mantido do código legado
# ─────────────────────────────────────────────────────────────────────────

def suggest_position_size(
    score:         float,
    beta:          float | None = None,
    earnings_days: int   | None = None,
    spy_change:    float | None = None,
) -> tuple[float, str]:
    """
    Sugere montante em EUR a investir do Flip Fund.
    Fazía parte do código legado — mantida intacta.
    """
    if not FLIP_FUND_EUR or FLIP_FUND_EUR <= 0:
        return 0.0, "⚠️ FLIP_FUND_EUR não configurado"

    raw       = FLIP_FUND_EUR * (score / 100.0)
    beta_val  = max(0.0, min(float(beta or 1.0), 3.0))
    beta_mult = max(0.40, 1.0 - beta_val * 0.15)

    earn_mult = 1.0
    earn_note = ""
    if earnings_days is not None and 0 <= earnings_days <= 7:
        earn_mult = 0.50
        earn_note = f" ✂️×0.5 (earnings em {earnings_days}d)"
    elif earnings_days is not None and earnings_days <= 14:
        earn_mult = 0.75
        earn_note = f" ✂️×0.75 (earnings em {earnings_days}d)"

    macro_mult = 1.0
    macro_note = ""
    if spy_change is not None and spy_change <= -2.0:
        macro_mult = 0.75
        macro_note = " 🌍×0.75 (SPY stress)"

    amount  = raw * beta_mult * earn_mult * macro_mult
    amount  = max(20.0, min(amount, FLIP_FUND_EUR * 0.40))
    amount  = round(amount, 0)
    pct     = amount / FLIP_FUND_EUR * 100
    explanation = (
        f"€{amount:.0f} ({pct:.0f}% do Flip Fund)"
        f" | β={beta_val:.1f}{earn_note}{macro_note}"
    )
    return amount, explanation
