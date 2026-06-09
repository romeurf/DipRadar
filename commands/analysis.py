"""commands/analysis.py — /comparar, /historico handlers.

Shared state (`_poll_context`, callbacks, `_reply`) is accessed via the
bot_commands module imported as `_bc`. The import happens at module load,
but attribute access (`_bc._reply`) happens at function call time when
bot_commands is fully loaded — Python's lazy attribute resolution handles
the circular dependency gracefully.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime

import bot_commands as _bc
from state import get_ticker_score_history


def handle_comparar(symbols: list[str]) -> None:
    if len(symbols) < 2:
        _bc._reply(
            "⚠️ Usa: `/comparar <TICK1> <TICK2> [TICK3...]`\n"
            "_Exemplo: `/comparar AAPL MSFT GOOGL`_\n"
            "_Máximo 5 tickers de cada vez._"
        )
        return
    if len(symbols) > 5:
        _bc._reply("⚠️ Máximo 5 tickers por comparação.")
        return

    get_fundamentals  = _bc._poll_context.get("get_fundamentals") or _bc._cb_get_fundamentals
    get_earnings_days = _bc._poll_context.get("get_earnings_days") or _bc._cb_earnings_days

    if not get_fundamentals:
        _bc._reply("_Comparação não disponível — callback get_fundamentals não registado._")
        return

    _bc._reply(f"_🔄 A comparar {' vs '.join(symbols)}... (pode demorar até 30s)_")

    def _run():
        from scanners.score import calculate_dip_score, classify_dip_category, is_bluechip
        rows = []
        for sym in symbols:
            try:
                fund  = get_fundamentals(sym)
                edays = get_earnings_days(sym) if get_earnings_days else None
                score, rsi_str = calculate_dip_score(fund, sym, edays)
                badge    = "🔥" if score >= 80 else ("⭐" if score >= 55 else "📊")
                bc_flag  = is_bluechip(fund)
                category = classify_dip_category(fund, score, bc_flag)
                rows.append({
                    "sym": sym, "score": score, "badge": badge,
                    "category": category, "rsi": rsi_str or "N/D",
                    "fcf":      f"{fund.get('fcf_yield',0)*100:.1f}%" if fund.get('fcf_yield') is not None else "N/D",
                    "growth":   f"{fund.get('revenue_growth',0)*100:.1f}%" if fund.get('revenue_growth') is not None else "N/D",
                    "margin":   f"{fund.get('gross_margin',0)*100:.0f}%" if fund.get('gross_margin') is not None else "N/D",
                    "upside":   f"{fund.get('analyst_upside',0):.0f}%" if fund.get('analyst_upside') is not None else "N/D",
                    "drawdown": f"{fund.get('drawdown_from_high',0):.1f}%" if fund.get('drawdown_from_high') is not None else "N/D",
                    "edays":    str(edays) if edays is not None else "N/D",
                    "sector":   (fund.get('sector') or 'N/D')[:14],
                })
            except Exception as e:
                logging.warning(f"[comparar] {sym}: {e}")
                rows.append({"sym": sym, "score": 0, "badge": "❌", "error": str(e)})

        if not rows:
            _bc._reply("_Nenhum dado obtido._")
            return

        rows.sort(key=lambda r: r.get("score", 0), reverse=True)
        lines = [f"*🔄 Comparação — {datetime.now().strftime('%d/%m %H:%M')}*", ""]
        for r in rows:
            if r.get("error"):
                lines.append(f"  ❌ *{r['sym']}* — _erro: {r['error']}_")
                continue
            lines.append(f"  {r['badge']} *{r['sym']}* — score *{r['score']:.0f}/100* | {r['category']}")
            lines.append(f"     RSI {r['rsi']} · FCF {r['fcf']} · Growth {r['growth']}")
            lines.append(f"     Margin {r['margin']} · Upside {r['upside']} · Drawdown {r['drawdown']}")
            lines.append(f"     Sector: _{r['sector']}_ · Earnings: {r['edays']}d")
            lines.append("")

        winner = rows[0]
        if not winner.get("error"):
            lines.append(
                f"_🏆 Melhor score: *{winner['sym']}* ({winner['score']:.0f} pts) — "
                f"usa `/analisar {winner['sym']}` para detalhe._"
            )
        _bc._reply("\n".join(lines))

    threading.Thread(target=_run, daemon=True).start()


def handle_historico(symbol: str) -> None:
    history = get_ticker_score_history(symbol)
    if not history:
        _bc._reply(
            f"_Sem histórico de scores para *{symbol}*._\n"
            "_O ticker ainda não foi analisado ou alertado pelo DipRadar._"
        )
        return

    entries    = history[-10:]
    lines      = [f"*📈 Histórico — {symbol} ({len(history)} entradas totais):*", ""]
    prev_score = None
    for e in entries:
        score  = e.get("score", 0)
        change = e.get("change", 0)
        price  = e.get("price")
        date   = e.get("date", "")
        t      = e.get("time", "")
        verdict = e.get("verdict") or ""
        if prev_score is None: trend = ""
        elif score > prev_score: trend = f" ↑{score - prev_score:.0f}"
        elif score < prev_score: trend = f" ↓{prev_score - score:.0f}"
        else: trend = " ="
        prev_score = score
        badge     = "🔥" if score >= 80 else ("⭐" if score >= 55 else "📊")
        price_str = f" @ ${price:.2f}" if price else ""
        lines.append(
            f"  {badge} `{date} {t}` — *{score:.0f}*{trend} | "
            f"{change:+.1f}%{price_str} | _{verdict}_"
        )

    if len(history) > 10:
        lines.append(f"\n_... e mais {len(history) - 10} entradas anteriores._")
    _bc._reply("\n".join(lines))
