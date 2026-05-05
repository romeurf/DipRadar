"""
watchlist.py — Monitorização personalizada de stocks com critérios de entrada próprios.

Filosofia: acumular qualidade em dip profundo, longo prazo.
Não é dip→flip. É dip→hold indefinidamente.

Fonte primária : yfinance (price, drawdown, dividend_yield, change_day)
Double-check   : Tiingo confirma preço e drawdown nos hits (fail-open).
                  Se Tiingo falhar/throttled/ticker não coberto, o alerta
                  é enviado na mesma com dados Yahoo e nota de estado Tiingo.

Cada entrada define:
  - symbol        : ticker Yahoo Finance correcto (com sufixo se necessário)
  - name          : nome legível
  - slot          : P1/P2/P3 — prioridade de alocação
  - category      : intenção estratégica (CATEGORY_* de score.py) — opcional
  - criteria      : lista de condições que devem ser verificadas (qualquer uma basta)
  - notes         : contexto / tese de investimento
  - alert_once    : se True, só alerta uma vez por dia por condição satisfeita

Critérios suportados:
  drawdown_52w_pct  : queda desde máximo de 52 semanas >= X %
  price_below       : preço actual <= X
  dividend_yield    : yield actual >= X %  (ex: 5.5 → 5.5%)
  price_above       : preço actual >= X  (para targets de saída)
  change_day_pct    : queda no dia >= X %
  quality_dislocation : gross_margin × |drawdown_52w| / 100 >= X (FCF negativo penaliza para 0)

Tiingo confirma apenas critérios baseados em EOD histórico:
  drawdown_52w_pct, price_below, price_above
Critérios dividend_yield, change_day_pct e quality_dislocation ficam exclusivamente em Yahoo.
"""

from __future__ import annotations
import time
import logging
from datetime import datetime
from typing import Any
import pytz
import yfinance as yf
from state import load_alerts, save_alerts
from score import CATEGORY_HOLD_FOREVER, CATEGORY_APARTAMENTO, CATEGORY_ROTACAO
from data_feed import get_tiingo_confirmation

LISBON_TZ = pytz.timezone("Europe/Lisbon")

# Critérios para os quais faz sentido pedir confirmação Tiingo
_TIINGO_CONFIRMABLE_CRITERIA: set[str] = {"drawdown_52w_pct", "price_below", "price_above"}


# ════════════════════════════════════════════════════════════════════════════
# WATCHLIST — edita aqui para adicionar/remover stocks
# ════════════════════════════════════════════════════════════════════════════

WATCHLIST: list[dict[str, Any]] = [

    # ── CORE EUROPEU ───────────────────────────────────────────────────────────
    {
        "symbol":   "IEMA.L",
        "name":     "iShares MSCI EM IMI UCITS ETF",
        "slot":     "P1",
        "category": CATEGORY_ROTACAO,
        "criteria": [
            {"type": "drawdown_52w_pct", "value": 12.0},
        ],
        "notes": "Entrada imediata com excesso do fundo de emergência. Diversificação Emerging Markets. Ticker LSE.",
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
        prev_close = info.get("regularMarketPreviousClose") or price
        change_day = abs((price - prev_close) / prev_close * 100) if prev_close else 0
        drawdown   = (high_52w - price) / high_52w * 100 if high_52w else 0
        name       = info.get("longName") or info.get("shortName") or symbol
        mc         = info.get("marketCap") or 0
        sector     = info.get("sector") or ""

        div_yield_raw = info.get("dividendYield") or 0
        div_yield     = div_yield_raw

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
            "dividend_yield_raw": div_yield_raw / 100.0,
        }
    except Exception as e:
        logging.warning(f"[watchlist] {symbol}: {e}")
        return None


def _should_confirm_with_tiingo(criteria: list[dict]) -> bool:
    """Verdadeiro se algum critério triggado é confirmável via Tiingo EOD."""
    return any(c["type"] in _TIINGO_CONFIRMABLE_CRITERIA for c in criteria)


def _build_tiingo_confirmation_line(
    symbol: str,
    yf_data: dict,
    tiingo: dict | None,
    criteria_triggered: list[str],
) -> str:
    """
    Gera linha de confirmação Tiingo para o alerta.

    Limiares de divergência:
      - Preço    : >2% entre Yahoo e Tiingo
      - Drawdown : >2pp entre Yahoo e Tiingo
    """
    # Só mostra se o critério triggado é do tipo EOD (confirmável por Tiingo)
    has_eod_criterion = any(
        ct in c for c in criteria_triggered
        for ct in ("Drawdown", "💲 Preço")
    )
    if not has_eod_criterion:
        return ""

    if tiingo is None:
        return "_ℹ️ Tiingo indisponível — dados Yahoo apenas_"

    yf_price = yf_data["price"]
    yf_dd    = yf_data["drawdown"]
    t_price  = tiingo["price"]
    t_dd     = tiingo["drawdown"]

    price_diff_pct = abs(yf_price - t_price) / yf_price * 100 if yf_price else 0
    dd_diff        = abs(yf_dd - t_dd)

    if price_diff_pct <= 2.0 and dd_diff <= 2.0:
        return f"✅ *Tiingo confirma:* ${t_price:.2f} | 52w ↓{t_dd:.1f}%"

    parts: list[str] = []
    if price_diff_pct > 2.0:
        parts.append(f"preço Yahoo ${yf_price:.2f} vs Tiingo ${t_price:.2f} ({price_diff_pct:.1f}%Δ)")
    if dd_diff > 2.0:
        parts.append(f"drawdown Yahoo {yf_dd:.1f}% vs Tiingo {t_dd:.1f}% ({dd_diff:.1f}ppΔ)")
    return f"⚠️ *Divergência Yahoo/Tiingo:* {'; '.join(parts)}"


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
        elif ctype == "quality_dislocation":
            gm      = data.get("gross_margins", 0) or 0
            dd      = data["drawdown"]
            fcf_raw = data.get("free_cashflow") or 0
            mc      = data.get("market_cap") or 1
            fcf_yield = fcf_raw / mc if mc > 0 else 0
            qd = (gm * dd / 100) if fcf_yield >= 0 else 0.0
            if qd >= val:
                triggered.append(
                    f"🎯 Quality Dislocation: *{qd:.2f}* (critério ≥{val})"
                )
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
    tiingo_conf: dict | None = None,
    tiingo_skipped: bool = False,
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

    # ── Bloco de confirmação Tiingo ──────────────────────────────────────
    if not tiingo_skipped:
        confirm_line = _build_tiingo_confirmation_line(
            symbol, data, tiingo_conf, triggered
        )
        if confirm_line:
            lines.append(f"\n{confirm_line}")
    # ─────────────────────────────────────────────────────────────────────

    if data.get("sector"):
        lines.append(f"\n  _Sector: {data['sector']}_")
    lines += [
        "",
        f"*📝 Tese:* _{notes}_",
        f"_⏰ {datetime.now(LISBON_TZ).strftime('%d/%m %H:%M')}_",
    ]
    return "\n".join(lines)


def run_watchlist_scan(
    send_telegram,
    direct_tickers: set | list,
) -> int:
    alerted  = load_alerts()
    today    = datetime.now(LISBON_TZ).date().isoformat()
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

        # ── Tiingo double-confirmer (fail-open) ──────────────────────────
        tiingo_conf    = None
        tiingo_skipped = True
        if _should_confirm_with_tiingo(entry["criteria"]):
            tiingo_skipped = False
            tiingo_conf    = get_tiingo_confirmation(symbol)
            if tiingo_conf is None:
                logging.info(
                    f"[watchlist] Tiingo confirm indisponível para {symbol} "
                    "— alerta enviado com dados Yahoo"
                )
        # ─────────────────────────────────────────────────────────────────

        msg = _build_watchlist_alert(
            entry,
            data,
            triggered,
            in_portfolio=(symbol in in_port),
            divergence=divergence,
            tiingo_conf=tiingo_conf,
            tiingo_skipped=tiingo_skipped,
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
    lines.append(f"\n_⏰ {datetime.now(LISBON_TZ).strftime('%d/%m %H:%M')}_")
    return "\n".join(lines)
