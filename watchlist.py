"""
watchlist.py — Monitorização personalizada de stocks com critérios de entrada próprios.

Filosofia: acumular qualidade em dip profundo, longo prazo.
Não é dip→flip. É dip→hold indefinidamente.

Cada entrada define:
  - symbol        : ticker Yahoo Finance correcto (com sufixo se necessário)
  - name          : nome legível
  - slot          : P1/P2/P3 — prioridade de alocação
  - category      : intenção estratégica (CATEGORY_* de score.py) — opcional
                    Se definida, é comparada com a categoria dinâmica do score.py.
                    Divergência → aviso no alerta ("Intenção vs Realidade").
  - criteria      : lista de condições que devem ser verificadas (qualquer uma basta)
  - notes         : contexto / tese de investimento
  - alert_once    : se True, só alerta uma vez por dia por condição satisfeita

Critérios suportados:
  drawdown_52w_pct  : queda desde máximo de 52 semanas >= X %
  price_below       : preço actual <= X
  dividend_yield    : yield actual >= X %  (ex: 5.5 → 5.5%)
  price_above       : preço actual >= X  (para targets de saída)
  change_day_pct    : queda no dia >= X %
"""

from __future__ import annotations
import time
import logging
from datetime import datetime
from typing import Any
import yfinance as yf
from state import load_alerts, save_alerts
from score import CATEGORY_HOLD_FOREVER, CATEGORY_APARTAMENTO, CATEGORY_ROTACAO


# ════════════════════════════════════════════════════════════════════════════
# WATCHLIST — edita aqui para adicionar/remover stocks
# ════════════════════════════════════════════════════════════════════════════

WATCHLIST: list[dict[str, Any]] = [

    # ── CORE EUROPEU ───────────────────────────────────────────────────────────
    {
        "symbol":   "IS3N.L",
        "name":     "iShares Core MSCI EM IMI",
        "slot":     "P1",
        "category": CATEGORY_ROTACAO,
        "criteria": [
            {"type": "drawdown_52w_pct", "value": 12.0},
        ],
        "notes": "Entrada imediata com excesso do fundo de emergência. Diversificação Emerging Markets.",
    },

    # ── DIVIDENDO / REIT ──────────────────────────────────────────────────────
    {
        "symbol":   "O",
        "name":     "Realty Income",
        "slot":     "P2",
        "category": CATEGORY_APARTAMENTO,
        "criteria": [
            {"type": "dividend_yield", "value": 5.5},
            {"type": "price_below",    "value": 50.0},
        ],
        "notes": "Dividendo mensal. Entrar quando yield >5.5% ou preço <50. Esperar Fed a cortar.",
    },
    {
        "symbol":   "MDT",
        "name":     "Medtronic",
        "slot":     "P2",
        "category": CATEGORY_APARTAMENTO,
        "criteria": [
            {"type": "dividend_yield", "value": 4.0},
            {"type": "price_below",    "value": 80.0},
        ],
        "notes": "Healthtech de dividendo. Entrar com yield >4% ou preço <80. Aguarda espaço libertado por NVO.",
    },
    {
        "symbol":   "ABBV",
        "name":     "AbbVie",
        "slot":     "P2",
        "category": CATEGORY_APARTAMENTO,
        "criteria": [
            {"type": "dividend_yield",  "value": 4.0},
            {"type": "change_day_pct",  "value": 15.0},
        ],
        "notes": "Pharma dividendo. Entrar com yield >4% ou dip >15% com pipeline intacto (Skyrizi/Rinvoq).",
    },
    {
        "symbol":   "LMT",
        "name":     "Lockheed Martin",
        "slot":     "P3",
        "category": CATEGORY_ROTACAO,
        "criteria": [
            {"type": "change_day_pct",   "value": 10.0},
            {"type": "drawdown_52w_pct", "value": 20.0},
        ],
        "notes": "Defesa. Dip adicional >10% no dia ou >20% do topo.",
    },
    {
        "symbol":   "RTX",
        "name":     "RTX (Raytheon)",
        "slot":     "P3",
        "category": CATEGORY_ROTACAO,
        "criteria": [
            {"type": "change_day_pct",   "value": 10.0},
            {"type": "drawdown_52w_pct", "value": 20.0},
        ],
        "notes": "Defesa. Dip adicional >10% no dia ou >20% do topo.",
    },

    # ── TECH GROWTH ───────────────────────────────────────────────────────────
    {
        "symbol":   "CRWD",
        "name":     "CrowdStrike",
        "slot":     "P3",
        "category": CATEGORY_ROTACAO,
        "criteria": [
            {"type": "drawdown_52w_pct", "value": 20.0},
        ],
        "notes": "Cybersecurity líder. Entrar só com correção >20% com tese intacta.",
    },
    {
        "symbol":   "PANW",
        "name":     "Palo Alto Networks",
        "slot":     "P3",
        "category": CATEGORY_ROTACAO,
        "criteria": [
            {"type": "drawdown_52w_pct", "value": 20.0},
        ],
        "notes": "Cybersecurity. Correção >20% com tese intacta.",
    },
    {
        "symbol":   "TSM",
        "name":     "Taiwan Semiconductor",
        "slot":     "P2",
        "category": CATEGORY_ROTACAO,
        "criteria": [
            {"type": "drawdown_52w_pct", "value": 15.0},
            {"type": "change_day_pct",   "value": 12.0},
        ],
        "notes": "Semicondutores. Correção 15-20% com tese intacta (risco geopolítico controlado).",
    },
    {
        "symbol":   "AVGO",
        "name":     "Broadcom",
        "slot":     "P3",
        "category": CATEGORY_ROTACAO,
        "criteria": [
            {"type": "drawdown_52w_pct", "value": 30.0},
        ],
        "notes": "Semicondutores/AI infra. Slot P3. Entrar apenas com queda >30% dos máximos.",
    },

    # ── EUROPA ────────────────────────────────────────────────────────────────────
    {
        "symbol":   "ALV.DE",
        "name":     "Allianz",
        "slot":     "P3",
        "category": CATEGORY_APARTAMENTO,
        "criteria": [
            {"type": "drawdown_52w_pct", "value": 15.0},
        ],
        "notes": "Seguradora europeia de qualidade. Dip >15% dos máximos.",
    },

]


# ════════════════════════════════════════════════════════════════════════════
# ENGINE DE VERIFICAÇÃO
# ════════════════════════════════════════════════════════════════════════════

_SLOT_EMOJI = {"P1": "🔴", "P2": "🟡", "P3": "🔵"}


def _get_ticker_data(symbol: str) -> dict | None:
    """Devolve dados relevantes para avaliar critérios. None se falhar."""
    try:
        time.sleep(2)
        t    = yf.Ticker(symbol)
        info = t.info
        if not info or not info.get("regularMarketPrice"):
            return None

        price      = info.get("regularMarketPrice") or info.get("currentPrice") or 0
        high_52w   = info.get("fiftyTwoWeekHigh") or 0
        div_yield  = (info.get("dividendYield") or 0) * 100
        prev_close = info.get("regularMarketPreviousClose") or price
        change_day = abs((price - prev_close) / prev_close * 100) if prev_close else 0
        drawdown   = (high_52w - price) / high_52w * 100 if high_52w else 0
        name       = info.get("longName") or info.get("shortName") or symbol
        mc         = info.get("marketCap") or 0
        sector     = info.get("sector") or ""

        return {
            "price":       price,
            "high_52w":    high_52w,
            "drawdown":    drawdown,
            "div_yield":   div_yield,
            "change_day":  change_day,
            "name":        name,
            "mc_b":        mc / 1e9,
            "sector":      sector,
            "market_cap":      mc,
            "free_cashflow":   info.get("freeCashflow"),
            "gross_margins":   info.get("grossMargins") or 0,
            "debt_to_equity":  info.get("debtToEquity"),
            "revenue_growth":  info.get("revenueGrowth") or 0,
            "dividend_yield_raw": info.get("dividendYield") or 0,
        }
    except Exception as e:
        logging.warning(f"[watchlist] {symbol}: {e}")
        return None


def _check_criteria(data: dict, criteria: list[dict]) -> list[str]:
    triggered = []
    for c in criteria:
        ctype = c["type"]
        val   = c["value"]
        if ctype == "drawdown_52w_pct" and data["drawdown"] >= val:
            triggered.append(f"📉 Drawdown 52w: *{data['drawdown']:.1f}%* (critério ≥{val:.0f}%)")
        elif ctype == "price_below" and data["price"] <= val:
            triggered.append(f"💲 Preço: *${data['price']:.2f}* (critério ≤${val:.0f})")
        elif ctype == "dividend_yield" and data["div_yield"] >= val:
            triggered.append(f"💰 Yield: *{data['div_yield']:.2f}%* (critério ≥{val:.1f}%)")
        elif ctype == "change_day_pct" and data["change_day"] >= val:
            triggered.append(f"🔻 Queda hoje: *{data['change_day']:.1f}%* (critério ≥{val:.0f}%)")
        elif ctype == "price_above" and data["price"] >= val:
            triggered.append(f"📈 Preço: *${data['price']:.2f}* (critério ≥${val:.0f})")
    return triggered


def _check_category_divergence(
    intention: str | None,
    symbol: str,
    data: dict,
) -> str | None:
    if not intention:
        return None
    try:
        from score import classify_dip_category, is_bluechip

        mc  = data.get("market_cap", 0) or 0
        fcf = data.get("free_cashflow")
        fundamentals = {
            "dividend_yield":    data.get("dividend_yield_raw", 0),
            "drawdown_from_high": -data["drawdown"],
            "fcf_yield":         (fcf / mc) if (fcf and mc > 0) else None,
            "gross_margin":      data.get("gross_margins", 0),
            "debt_equity":       data.get("debt_to_equity"),
            "market_cap":        mc,
            "revenue_growth":    data.get("revenue_growth", 0),
            "sector":            data.get("sector", ""),
        }
        bc_flag = is_bluechip(fundamentals)
        reality = classify_dip_category(fundamentals, dip_score=50, is_bluechip_flag=bc_flag)
        if intention != reality:
            return f"⚠️ *ALERTA DE TESE:* Intenção ({intention}) → Modelo ({reality})"
    except Exception as e:
        logging.debug(f"[watchlist] divergence check {symbol}: {e}")
    return None


def _build_watchlist_alert(
    entry: dict,
    data: dict,
    triggered: list[str],
    in_portfolio: bool,
    divergence: str | None = None,
) -> str:
    symbol    = entry["symbol"]
    slot      = entry["slot"]
    notes     = entry["notes"]
    intention = entry.get("category")
    slot_e    = _SLOT_EMOJI.get(slot, "⚪")
    port_tag  = " 📦 *Já em carteira*" if in_portfolio else ""
    mc_str    = f"${data['mc_b']:.1f}B" if data["mc_b"] else "N/D"
    lines = [
        f"🎯 *Watchlist Hit: {symbol} — {data['name']}*{port_tag}",
        f"{slot_e} Slot *{slot}* | 💰 ${data['price']:.2f} | 🏦 {mc_str}",
        f"📉 52w drawdown: *{data['drawdown']:.1f}%* | Yield: *{data['div_yield']:.2f}%*",
    ]
    if intention:
        lines.append(f"🗂️ Intenção: *{intention}*")
    if divergence:
        lines.append(divergence)
    lines += [
        "",
        "*✅ Critérios satisfeitos:*",
    ]
    for t in triggered:
        lines.append(f"  {t}")
    if data.get("sector"):
        lines.append(f"\n  _Sector: {data['sector']}_")
    lines += [
        "",
        f"*📝 Tese:* _{notes}_",
        f"_⏰ {datetime.now().strftime('%d/%m %H:%M')}_",
    ]
    return "\n".join(lines)


def run_watchlist_scan(
    send_telegram,
    direct_tickers: set | list,
) -> int:
    alerted  = load_alerts()
    today    = datetime.now().date().isoformat()
    sent     = 0
    in_port  = set(direct_tickers)

    for entry in WATCHLIST:
        symbol    = entry["symbol"]
        alert_key = f"WL_{symbol}_{today}"
        if alert_key in alerted:
            continue

        logging.info(f"[watchlist] A verificar {symbol}...")
        data = _get_ticker_data(symbol)
        if not data:
            continue

        triggered = _check_criteria(data, entry["criteria"])
        if not triggered:
            continue

        divergence = _check_category_divergence(entry.get("category"), symbol, data)

        msg = _build_watchlist_alert(
            entry,
            data,
            triggered,
            in_portfolio=(symbol in in_port),
            divergence=divergence,
        )
        if send_telegram(msg):
            alerted.add(alert_key)
            save_alerts(alerted)
            sent += 1
            logging.info(f"[watchlist] ✅ Alerta: {symbol} ({entry['slot']})")

    return sent


def build_watchlist_morning_summary(direct_tickers: set | list) -> str:
    in_port = set(direct_tickers)
    lines   = ["*👀 Watchlist — Estado actual:*", ""]
    for entry in WATCHLIST:
        symbol    = entry["symbol"]
        slot      = entry["slot"]
        intention = entry.get("category", "")
        slot_e    = _SLOT_EMOJI.get(slot, "⚪")
        data      = _get_ticker_data(symbol)
        if not data:
            lines.append(f"  {slot_e} *{symbol}* — _erro ao obter dados_")
            continue
        triggered  = _check_criteria(data, entry["criteria"])
        port_tag   = " 📦" if symbol in in_port else ""
        hit_tag    = " 🎯 *CRITÉRIO ATINGIDO*" if triggered else ""
        cat_tag    = f" | {intention}" if intention else ""
        lines.append(
            f"  {slot_e} *{symbol}*{port_tag}{hit_tag}{cat_tag} — "
            f"${data['price']:.2f} | 52w \u2193{data['drawdown']:.0f}% | "
            f"Yield {data['div_yield']:.1f}%"
        )
    lines.append(f"\n_⏰ {datetime.now().strftime('%d/%m %H:%M')}_")
    return "\n".join(lines)
