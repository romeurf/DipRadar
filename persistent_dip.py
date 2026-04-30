"""
persistent_dip.py — Feature 8: Dip Persistente

Detecta e alerta quando um stock aparece no scan por N dias consecutivos
sem recuperar — sinal de deterioração contínua ou de capitulação profunda.

Integração em run_scan() (main.py) — adicionar APÓS score_data ser calculado
e ANTES do append_weekly_log:

  # ── Feature 8 ──────────────────────────────────────────────────────
  from persistent_dip import check_and_alert_streak
  from state import record_dip_day, mark_persistent_alerted, expire_missing_streaks

  # 1. Dentro do loop de stocks, após score_data:
  dip_state = record_dip_day(sym, score, fund.get("price", 0),
                              stock["change_pct"], verdict)
  check_and_alert_streak(sym, dip_state, score, category,
                         fund, send_telegram, DIRECT_TICKERS, LISBON_TZ)

  # 2. Inicializar antes do loop:
  _scan_syms_scored: set[str] = set()
  # Dentro do loop, após record_dip_day:
  _scan_syms_scored.add(sym)

  # 3. No bloco finally de run_scan():
  expire_missing_streaks(_scan_syms_scored)

Thresholds:
  streak_days == 2        → 🔁 alerta amarelo (stock persistente)
  streak_days == 3        → 🚨 alerta vermelho (sinal sério)
  streak_days > 3,        → 🚨 alerta a cada 3 dias adicionais (dias 6, 9, ...)
  alerted_at_streak usado para deduplicar (nunca re-alerta o mesmo nível).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable

from state import mark_persistent_alerted


# ── Thresholds ────────────────────────────────────────────────────────────────

_THRESHOLD_YELLOW = 2   # 🔁 a partir do 2º dia
_THRESHOLD_RED    = 3   # 🚨 a partir do 3º dia
_RED_REPEAT_EVERY = 3   # 🚨 repete a cada 3 dias após o 3º (dias 6, 9, ...)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _should_alert(streak_days: int, alerted_at: int) -> bool:
    """
    Decide se deve enviar alerta de persistência dado o streak actual
    e o último dia em que foi alertado.

    Regras:
      - Dia 2: alerta se ainda não alertado no nível 2+
      - Dia 3: alerta se ainda não alertado no nível 3+
      - Dia 6, 9, ...: alerta se já passaram >= 3 dias desde último alerta
    """
    if streak_days < _THRESHOLD_YELLOW:
        return False
    if alerted_at == 0:
        return True
    if streak_days >= _THRESHOLD_RED:
        # Repete a cada _RED_REPEAT_EVERY dias
        return (streak_days - alerted_at) >= _RED_REPEAT_EVERY
    # Entre dia 2 e 3: alerta apenas uma vez
    return streak_days > alerted_at


def _format_history(entries: list[dict], n: int = 5) -> list[str]:
    """Devolve as últimas `n` linhas do histórico formatadas para Telegram."""
    lines = []
    for e in entries[-n:]:
        price_str = f"${e['price']:.2f}" if e.get("price") else "N/D"
        lines.append(
            f"  • {e['date']}: Score *{e['score']:.0f}*/100 "
            f"| {e['change']:+.1f}% | {price_str} | _{e['verdict']}_"
        )
    return lines


def _cumulative_drop(entries: list[dict]) -> str:
    """
    Calcula a queda acumulada desde a primeira entrada do streak.
    Devolve string formatada, ex: "-12.4%", ou "" se sem dados.
    """
    if len(entries) < 2:
        return ""
    p0 = entries[0].get("price")
    p1 = entries[-1].get("price")
    if not p0 or not p1 or p0 <= 0:
        return ""
    pct = (p1 - p0) / p0 * 100
    return f"{pct:+.1f}%"


# ── Alerta principal ──────────────────────────────────────────────────────────

def build_persistent_dip_message(
    symbol: str,
    streak_days: int,
    dip_state: dict,
    score: float,
    category: str,
    in_portfolio: bool,
    lisbon_tz,
) -> str:
    """
    Constrói a mensagem Telegram para o alerta de dip persistente.
    Chamada por check_and_alert_streak().
    """
    entries    = dip_state.get("entries", [])
    first_seen = dip_state.get("first_seen", "N/D")

    badge     = "🚨" if streak_days >= _THRESHOLD_RED else "🔁"
    portfolio = " 📦 *Carteira*" if in_portfolio else ""
    cum_drop  = _cumulative_drop(entries)
    cum_str   = f" | Queda acumulada: *{cum_drop}*" if cum_drop else ""

    ordinal = {
        2: "2º", 3: "3º", 4: "4º", 5: "5º",
        6: "6º", 7: "7º", 8: "8º", 9: "9º", 10: "10º",
    }.get(streak_days, f"{streak_days}º")

    lines = [
        f"{badge} *Dip Persistente: {symbol}*{portfolio}",
        f"_{ordinal} dia consecutivo acima do threshold de score_",
        "",
        f"  📅 Desde: {first_seen}{cum_str}",
        f"  📊 Score actual: *{score:.0f}/100* | {category}",
        "",
        "*📋 Histórico do streak:*",
    ]
    lines += _format_history(entries, n=5)
    lines += [
        "",
    ]

    if streak_days >= _THRESHOLD_RED:
        lines.append(
            "_⚠️ Sinal sério — capitalização prolongada ou deterioração de tese._"
        )
        lines.append("_Reavalia: é capitulação com oportunidade ou queda estrutural?_")
    else:
        lines.append(
            "_📌 Stock persistente no radar — confirmar se tese mantém._"
        )

    now_str = datetime.now(lisbon_tz).strftime("%d/%m %H:%M")
    lines.append(f"_⏰ {now_str}_")

    return "\n".join(lines)


# ── Entry point para run_scan() ───────────────────────────────────────────────

def check_and_alert_streak(
    symbol: str,
    dip_state: dict,
    score: float,
    category: str,
    fundamentals: dict,
    send_telegram_fn: Callable[[str], bool],
    direct_tickers: set | list,
    lisbon_tz,
) -> bool:
    """
    Verifica se o streak de `symbol` merece um alerta de persistência
    e envia via `send_telegram_fn` se sim.

    Parâmetros
    ----------
    symbol          : ticker do stock
    dip_state       : dict devolvido por record_dip_day()
    score           : score final do motor quantitativo
    category        : CATEGORY_* string
    fundamentals    : dict de get_fundamentals() (não usado no corpo, reservado)
    send_telegram_fn: função send_telegram do main.py
    direct_tickers  : set/list de tickers em carteira (DIRECT_TICKERS)
    lisbon_tz       : pytz timezone para o timestamp

    Retorna True se um alerta foi enviado, False caso contrário.
    """
    streak_days = dip_state.get("streak_days", 1)
    alerted_at  = dip_state.get("alerted_at_streak", 0)

    if not _should_alert(streak_days, alerted_at):
        return False

    in_portfolio = symbol.upper() in {t.upper() for t in direct_tickers}

    try:
        msg = build_persistent_dip_message(
            symbol=symbol,
            streak_days=streak_days,
            dip_state=dip_state,
            score=score,
            category=category,
            in_portfolio=in_portfolio,
            lisbon_tz=lisbon_tz,
        )
        sent = send_telegram_fn(msg)
        if sent:
            mark_persistent_alerted(symbol, streak_days)
            logging.info(
                f"[persistent_dip] {symbol}: alerta enviado "
                f"(streak={streak_days}, prev_alerted={alerted_at})"
            )
        return sent
    except Exception as e:
        logging.error(f"[persistent_dip] Erro ao alertar {symbol}: {e}", exc_info=True)
        return False
