"""
portfolio.py — Gestão de carteira activa e liquidez.

Backend dual:
  1. Google Sheets (primário) — se GOOGLE_SHEETS_CREDENTIALS e GOOGLE_SHEET_ID
     estiverem definidos como variáveis de ambiente no Railway.
  2. Ficheiro local JSON (fallback) — comportamento anterior, garante que o bot
     arranca mesmo sem Google Sheets configurado.

Estrutura da folha Google Sheets:
  Aba "Liquidez"  → linha 2, coluna A: valor float
  Aba "Posicoes"  → cabeçalho na linha 1:
    Ticker | Entry_Date | Entry_Price | Quantity | Entry_Category |
    Entry_Score | Last_Price | Last_Score | Last_Update | Degradation_Alerted

Cache write-through:
  Dados são carregados para memória na primeira leitura.
  Escritas vão SEMPRE para Google Sheets primeiro, depois actualizam a cache.
  Leituras usam sempre a cache (zero chamadas extra à API).

Variáveis de ambiente esperadas no Railway:
  HOLDING_<TICKER>=shares,avg_cost  — posições holding (ex: HOLDING_CRWD=10,180.50)
  HOLDING_EUNL=shares,avg_cost      — será mapeado automaticamente para EUNL.DE
  PPR_SHARES=<float>                — actualizado manualmente no Railway
  PPR_AVG_COST=<float>              — actualizado manualmente no Railway
  FLIP_FUND_EUR=<float>             — capital do Flip Fund
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from universe import is_etf

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────
DIRECT_TICKERS = ["NVO", "ADBE", "UBER", "EUNL.DE", "MSFT", "PINS", "ADP", "CRM", "VICI",
                   "CRWD", "PLTR", "NOW", "DUOL"]

USD_TICKERS = {
    "NVO", "ADBE", "UBER", "MSFT", "PINS", "ADP", "CRM", "VICI",
    "CRWD", "PLTR", "NOW", "DUOL",
}
EUR_TICKERS = {"EUNL.DE", "IS3N.DE", "ALV.DE", "IEMA"}

# Mapa de aliases: env vars sem sufixo → ticker correcto para yfinance
# Permite usar HOLDING_EUNL=... no Railway sem ter de configurar HOLDING_EUNL.DE
_TICKER_ALIASES: dict[str, str] = {
    "EUNL":  "EUNL.DE",
    "IS3N":  "IS3N.DE",
    "ALV":   "ALV.DE",
    "IEMA":  "IEMA.L",
}


def _float_env(key: str, default: float = 0.0) -> float:
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


FLIP_FUND_EUR = _float_env("FLIP_FUND_EUR")

# ─────────────────────────────────────────────────────────────────────────
# HOLDINGS — lê env vars com o padrão HOLDING_<TICKER>=shares,avg_cost
# ─────────────────────────────────────────────────────────────────────────

def _parse_holdings_env() -> list[tuple[str, float, float]]:
    """Constrói HOLDINGS a partir das env vars HOLDING_<TICKER>=shares[,avg_cost].
    Aplica _TICKER_ALIASES para normalizar tickers sem sufixo de bolsa
    (ex: HOLDING_EUNL → EUNL.DE).
    """
    result: list[tuple[str, float, float]] = []
    for key, val in os.environ.items():
        if not key.startswith("HOLDING_"):
            continue
        raw_ticker = key[len("HOLDING_"):].strip().upper()
        if not raw_ticker:
            continue
        # Normaliza alias (ex: EUNL → EUNL.DE)
        ticker = _TICKER_ALIASES.get(raw_ticker, raw_ticker)
        parts = [p.strip() for p in val.split(",")]
        try:
            shares = float(parts[0]) if parts else 0.0
            avg    = float(parts[1]) if len(parts) > 1 else 0.0
            result.append((ticker, shares, avg))
        except (ValueError, IndexError):
            log.warning(f"[portfolio] env var malformada: {key}={val}")
    return result


# Variáveis públicas exigidas pelo main.py
HOLDINGS: list[tuple[str, float, float]] = _parse_holdings_env()
PPR_SHARES   = _float_env("PPR_SHARES")
PPR_AVG_COST = _float_env("PPR_AVG_COST")


# ─────────────────────────────────────────────────────────────────────────
# Google Sheets backend
# ─────────────────────────────────────────────────────────────────────────

_GS_CREDENTIALS = os.getenv("GOOGLE_SHEETS_CREDENTIALS", "")
_GS_SHEET_ID    = os.getenv("GOOGLE_SHEET_ID", "")
_USE_GS         = bool(_GS_CREDENTIALS and _GS_SHEET_ID)

# Cache em memória
_cache: dict = {"liquidity": None, "positions": None}  # None = não carregado ainda
_gs_client  = None
_gs_sheet   = None

GS_COLS = [
    "Ticker", "Entry_Date", "Entry_Price", "Quantity",
    "Entry_Category", "Entry_Score", "Last_Price",
    "Last_Score", "Last_Update", "Degradation_Alerted",
]


def _get_gs_sheet():
    """Devolve o objecto Spreadsheet, inicializando a ligação se necessário."""
    global _gs_client, _gs_sheet
    if _gs_sheet is not None:
        return _gs_sheet
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        scopes = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds_dict = json.loads(_GS_CREDENTIALS)
        creds      = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        _gs_client = gspread.authorize(creds)
        _gs_sheet  = _gs_client.open_by_key(_GS_SHEET_ID)
        log.info("[portfolio] Google Sheets ligado com sucesso")
        return _gs_sheet
    except Exception as e:
        log.error(f"[portfolio] Falha ao ligar ao Google Sheets: {e}")
        return None


def _gs_load_liquidity() -> float:
    sheet = _get_gs_sheet()
    if sheet is None:
        return 0.0
    try:
        try:
            ws = sheet.worksheet("Liquidez")
        except Exception:
            # Aba não existe — cria automaticamente no primeiro arranque
            log.info("[portfolio] GS: criando aba 'Liquidez'")
            ws = sheet.add_worksheet(title="Liquidez", rows=10, cols=2)
            ws.update([["Liquidez"], [0.0]])
        val = ws.acell("A2").value
        return float(val) if val else 0.0
    except Exception as e:
        log.warning(f"[portfolio] GS read liquidez erro: {e}")
        return 0.0


def _gs_write_liquidity(amount: float) -> None:
    sheet = _get_gs_sheet()
    if sheet is None:
        return
    try:
        try:
            ws = sheet.worksheet("Liquidez")
        except Exception:
            ws = sheet.add_worksheet(title="Liquidez", rows=10, cols=2)
            ws.update([["Liquidez"], [0.0]])
        ws.update("A2", [[round(amount, 2)]])
    except Exception as e:
        log.warning(f"[portfolio] GS write liquidez erro: {e}")


def _gs_load_positions() -> dict:
    sheet = _get_gs_sheet()
    if sheet is None:
        return {}
    try:
        try:
            ws = sheet.worksheet("Posicoes")
        except Exception:
            # Aba não existe — cria automaticamente com cabeçalho
            log.info("[portfolio] GS: criando aba 'Posicoes'")
            ws = sheet.add_worksheet(title="Posicoes", rows=100, cols=len(GS_COLS))
            ws.update([GS_COLS])
            return {}
        records = ws.get_all_records()
        positions = {}
        for row in records:
            ticker = str(row.get("Ticker", "")).strip().upper()
            if not ticker:
                continue
            positions[ticker] = {
                "symbol":              ticker,
                "name":                ticker,
                "shares":              float(row.get("Quantity", 0) or 0),
                "avg_price":           float(row.get("Entry_Price", 0) or 0),
                "total_cost":          round(
                    float(row.get("Entry_Price", 0) or 0) *
                    float(row.get("Quantity", 0) or 0), 2
                ),
                "category":            str(row.get("Entry_Category", "") or ""),
                "entry_score":         int(row["Entry_Score"]) if row.get("Entry_Score") not in ("", None) else None,
                "entry_date":          str(row.get("Entry_Date", "") or ""),
                "entry_date_iso":      str(row.get("Entry_Date", "") or ""),
                "last_score":          int(row["Last_Score"]) if row.get("Last_Score") not in ("", None) else None,
                "last_price":          float(row.get("Last_Price", 0) or 0),
                "last_update":         str(row.get("Last_Update", "") or ""),
                "degradation_alerted": str(row.get("Degradation_Alerted", "False")).lower() == "true",
            }
        return positions
    except Exception as e:
        log.warning(f"[portfolio] GS read positions erro: {e}")
        return {}


def _gs_write_positions(positions: dict) -> None:
    sheet = _get_gs_sheet()
    if sheet is None:
        return
    try:
        try:
            ws = sheet.worksheet("Posicoes")
        except Exception:
            ws = sheet.add_worksheet(title="Posicoes", rows=100, cols=len(GS_COLS))
        rows = [GS_COLS]
        now_hm = datetime.now().strftime("%d/%m %H:%M")
        for ticker, pos in positions.items():
            rows.append([
                ticker,
                pos.get("entry_date", ""),
                pos.get("avg_price", 0),
                pos.get("shares", 0),
                pos.get("category", ""),
                pos.get("entry_score", "") if pos.get("entry_score") is not None else "",
                pos.get("last_price", 0),
                pos.get("last_score", "") if pos.get("last_score") is not None else "",
                pos.get("last_update", now_hm),
                str(pos.get("degradation_alerted", False)),
            ])
        ws.clear()
        ws.update(rows)
    except Exception as e:
        log.warning(f"[portfolio] GS write positions erro: {e}")


# ─────────────────────────────────────────────────────────────────────────
# Fallback local JSON
# ─────────────────────────────────────────────────────────────────────────
_DATA_DIR       = Path("/data") if Path("/data").exists() else Path("/tmp")
_PORTFOLIO_FILE = _DATA_DIR / "_dipr_portfolio.json"


def _read_raw() -> dict:
    try:
        if _PORTFOLIO_FILE.exists():
            return json.loads(_PORTFOLIO_FILE.read_text())
    except Exception as e:
        log.warning(f"[portfolio] read local erro: {e}")
    return {"liquidity": 0.0, "positions": {}}


def _write_raw(data: dict) -> None:
    try:
        _PORTFOLIO_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        log.warning(f"[portfolio] write local erro: {e}")


# ─────────────────────────────────────────────────────────────────────────
# Cache — load lazy
# ─────────────────────────────────────────────────────────────────────────

def _ensure_cache() -> None:
    """Carrega dados para a cache em memória (apenas uma vez por sessão)."""
    if _cache["liquidity"] is None:
        if _USE_GS:
            _cache["liquidity"] = _gs_load_liquidity()
            _cache["positions"] = _gs_load_positions()
            log.info("[portfolio] Cache carregada do Google Sheets")
        else:
            raw = _read_raw()
            _cache["liquidity"] = raw.get("liquidity", 0.0)
            _cache["positions"] = raw.get("positions", {})
            log.info("[portfolio] Cache carregada do ficheiro local")


def _flush() -> None:
    """Persiste cache actual para o backend activo."""
    if _USE_GS:
        _gs_write_liquidity(_cache["liquidity"])
        _gs_write_positions(_cache["positions"])
    else:
        _write_raw({"liquidity": _cache["liquidity"], "positions": _cache["positions"]})


# ─────────────────────────────────────────────────────────────────────────
# API pública — leitura
# ─────────────────────────────────────────────────────────────────────────

def get_liquidity() -> float:
    _ensure_cache()
    return _cache["liquidity"]


def get_positions() -> dict:
    _ensure_cache()
    return dict(_cache["positions"])  # cópia defensiva


def get_position(symbol: str) -> dict | None:
    return get_positions().get(symbol.upper())


def get_active_symbols() -> list[str]:
    return list(get_positions().keys())


# ─────────────────────────────────────────────────────────────────────────
# Concentração — usado pelo allocation_engine para os caps 12%/30%
# ─────────────────────────────────────────────────────────────────────────

def _is_usd_ticker(symbol: str) -> bool:
    """Heurística simples: tickers com sufixo (.DE/.L/.PA/.AS/...) → não USD.
    Restantes (NYSE/NASDAQ/AMEX padrão) → USD.

    Mais permissiva que o set USD_TICKERS hardcoded — funciona para qualquer
    ticker novo (ex: RKLB, XPEV) sem ter de o adicionar a lista.
    """
    if symbol in EUR_TICKERS:
        return False
    if "." in symbol:
        return False
    return True


def get_position_pct(symbol: str, usd_eur: float = 0.92) -> float:
    """Devolve a % do portfolio total actualmente alocada a `symbol`.

    Cálculo (cost basis):
      • Para cada posição, soma `total_cost` convertido para EUR (USD × usd_eur).
      • Adiciona liquidez (já em EUR).
      • Devolve cost da posição (EUR) / total wealth (EUR).

    Caveat: usa cost basis (initial investment), não mark-to-market. Estável,
    sem necessidade de live prices. Para o efeito de concentration cap (12%/30%)
    é suficiente — o cap é sobre quanto foi colocado, não sobre P&L.

    Args:
        symbol: ticker normalizado (será uppercased).
        usd_eur: taxa USD→EUR (default 0.92). Passar `get_usdeur()` para precisão.

    Returns:
        Float em [0.0, 1.0]. 0.0 se o ticker não estiver no portfolio ou total=0.
    """
    symbol = _TICKER_ALIASES.get(symbol.upper().strip(), symbol.upper().strip())
    pos = get_position(symbol)
    if not pos:
        return 0.0

    fx = max(0.5, min(float(usd_eur or 0.92), 2.0))  # clamp defensivo

    total_eur = float(get_liquidity() or 0.0)
    for sym, p in get_positions().items():
        cost = float(p.get("total_cost", 0) or 0)
        if _is_usd_ticker(sym):
            cost *= fx
        total_eur += cost

    if total_eur <= 0:
        return 0.0

    pos_cost = float(pos.get("total_cost", 0) or 0)
    if _is_usd_ticker(symbol):
        pos_cost *= fx

    return max(0.0, min(pos_cost / total_eur, 1.0))


# ─────────────────────────────────────────────────────────────────────────
# /buy
# ─────────────────────────────────────────────────────────────────────────

def buy(
    symbol:      str,
    price:       float,
    shares:      float,
    category:    str        = "",
    entry_score: int | None = None,
    name:        str        = "",
) -> dict:
    symbol = _TICKER_ALIASES.get(symbol.upper().strip(), symbol.upper().strip())
    _ensure_cache()
    cost = round(price * shares, 2)

    if is_etf(symbol):
        category    = "ETF"
        entry_score = None

    positions = _cache["positions"]
    now_str   = datetime.now().strftime("%d/%m/%Y")
    now_iso   = datetime.now().date().isoformat()
    now_hm    = datetime.now().strftime("%d/%m %H:%M")

    if symbol in positions:
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

    old_liq           = _cache["liquidity"]
    new_liq           = round(old_liq - cost, 2)
    _cache["liquidity"] = new_liq
    _flush()

    log.info(
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
# seed_position — popula o GSheets a partir de env vars sem tocar na liquidez
# ─────────────────────────────────────────────────────────────────────────

def seed_position(
    symbol:      str,
    price:       float,
    shares:      float,
    category:    str        = "",
    entry_score: int | None = None,
    name:        str        = "",
    entry_date:  str        = "",
) -> dict:
    """Regista uma posição existente no backend (GSheets/JSON) SEM afectar a liquidez.
    Idempotente: se o ticker já existir, devolve action='exists' e não sobrescreve."""
    symbol = _TICKER_ALIASES.get(symbol.upper().strip(), symbol.upper().strip())
    _ensure_cache()

    if is_etf(symbol):
        category    = "ETF"
        entry_score = None

    positions = _cache["positions"]

    if symbol in positions:
        log.info(f"[portfolio] SEED {symbol} — já existe, ignorado")
        return {"symbol": symbol, "shares": shares, "price": price,
                "cost": round(price * shares, 2), "action": "exists",
                "position": positions[symbol]}

    now_str = entry_date or datetime.now().strftime("%d/%m/%Y")
    now_iso = datetime.now().date().isoformat()
    now_hm  = datetime.now().strftime("%d/%m %H:%M")
    cost    = round(price * shares, 2)

    positions[symbol] = {
        "symbol":              symbol,
        "name":                name or symbol,
        "shares":              round(shares, 6),
        "avg_price":           round(price, 4),
        "total_cost":          cost,
        "category":            category or "Holding",
        "entry_score":         entry_score,
        "entry_date":          now_str,
        "entry_date_iso":      now_iso,
        "last_score":          entry_score,
        "last_price":          price,
        "last_update":         now_hm,
        "degradation_alerted": False,
    }
    _flush()
    log.info(f"[portfolio] SEED {symbol} x{shares} @ ${price} | custo ${cost} | nova posição")
    return {
        "symbol":   symbol,
        "shares":   shares,
        "price":    price,
        "cost":     cost,
        "action":   "new",
        "position": positions[symbol],
    }


# ─────────────────────────────────────────────────────────────────────────
# /sell
# ─────────────────────────────────────────────────────────────────────────

def sell(
    symbol: str,
    price:  float,
    shares: float | None = None,
) -> dict | None:
    symbol = _TICKER_ALIASES.get(symbol.upper().strip(), symbol.upper().strip())
    _ensure_cache()
    positions = _cache["positions"]

    if symbol not in positions:
        log.warning(f"[portfolio] SELL {symbol}: posição não encontrada")
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

    old_liq             = _cache["liquidity"]
    new_liq             = round(old_liq + proceeds, 2)
    _cache["liquidity"] = new_liq
    _flush()

    log.info(
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
# Liquidez
# ─────────────────────────────────────────────────────────────────────────

def add_liquidity(amount: float, note: str = "") -> float:
    _ensure_cache()
    old                 = _cache["liquidity"]
    new                 = round(old + amount, 2)
    _cache["liquidity"] = new
    _flush()
    log.info(f"[portfolio] Liquidez +{amount}€ | {old:.2f} → {new:.2f} | {note}")
    return new


def set_liquidity(amount: float) -> float:
    _ensure_cache()
    _cache["liquidity"] = round(amount, 2)
    _flush()
    log.info(f"[portfolio] Liquidez definida para {amount:.2f}€")
    return round(amount, 2)


# ─────────────────────────────────────────────────────────────────────────
# Actualização diária
# ─────────────────────────────────────────────────────────────────────────

def update_position_data(
    symbol:   str,
    price:    float,
    score:    int | None = None,
    category: str | None = None,
) -> None:
    symbol = _TICKER_ALIASES.get(symbol.upper().strip(), symbol.upper().strip())
    _ensure_cache()
    pos = _cache["positions"].get(symbol)
    if not pos:
        return
    pos["last_price"]  = price
    pos["last_update"] = datetime.now().strftime("%d/%m %H:%M")
    if score is not None:
        pos["last_score"] = score
    if category is not None:
        pos["category"] = category
    _flush()


def mark_degradation_alerted(symbol: str) -> None:
    symbol = _TICKER_ALIASES.get(symbol.upper().strip(), symbol.upper().strip())
    _ensure_cache()
    pos = _cache["positions"].get(symbol)
    if pos:
        pos["degradation_alerted"] = True
        _flush()


def reset_degradation_flag(symbol: str) -> None:
    symbol = _TICKER_ALIASES.get(symbol.upper().strip(), symbol.upper().strip())
    _ensure_cache()
    pos = _cache["positions"].get(symbol)
    if pos:
        pos["degradation_alerted"] = False
        _flush()


# ─────────────────────────────────────────────────────────────────────────
# Flip Fund
# ─────────────────────────────────────────────────────────────────────────

def suggest_position_size(
    score:         float,
    beta:          float | None = None,
    earnings_days: int   | None = None,
    spy_change:    float | None = None,
) -> tuple[float, str]:
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
