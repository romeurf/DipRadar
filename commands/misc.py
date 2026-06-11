"""commands/misc.py — Smaller info commands: mldata, ml_accuracy, performance,
paper_performance, health.

Phase 8.5 of the bot_commands.py split. Uses `import bot_commands as _bc`
for shared state.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path

import bot_commands as _bc



# ── /mldata ────────────────────────────────────────────────────────────────────

def handle_mldata(force_update: bool = False) -> None:
    get_db_stats     = _bc._poll_context.get("get_db_stats") or _bc._cb_get_db_stats
    fill_db_outcomes = _bc._poll_context.get("fill_db_outcomes") or _bc._cb_fill_db_outcomes

    if not get_db_stats:
        _bc._reply("_Base de dados ML não disponível. Reinicia o bot._")
        return

    def _run():
        update_stats = None
        if force_update and fill_db_outcomes:
            _bc._reply("🔄 _A actualizar outcomes da alert_db... (pode demorar 1-2 min)_")
            try:
                update_stats = fill_db_outcomes()
            except Exception as e:
                _bc._reply(f"⚠️ Erro no update: `{e}`")
                return

        try:
            stats   = get_db_stats()
            total   = stats.get("total", 0)
            labeled = stats.get("labeled", 0)
            outcomes = stats.get("outcomes", {})
            by_cat  = stats.get("by_category", {})
            by_vrd  = stats.get("by_verdict", {})
            db_path = stats.get("db_path", "N/D")
        except Exception as e:
            _bc._reply(f"_Erro ao ler stats: {e}_")
            return

        lines = [
            "🤖 *ML Alert Database*",
            f"_{datetime.now().strftime('%d/%m/%Y %H:%M')}_", "",
            f"*🗃️ Total de alertas:* {total}",
            (f"*📊 Classificados:* {labeled} ({labeled/total*100:.0f}% do total)" if total > 0 else "*📊 Classificados:* 0"),
            f"*🗂️ Ficheiro:* `{db_path}`", "",
        ]
        if by_cat:
            lines.append("*🏠 Por categoria:*")
            for cat, count in sorted(by_cat.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {cat}: *{count}* ({count/total*100:.0f}%)")
            lines.append("")
        if outcomes:
            lines.append("*🎯 Outcomes:*")
            em_map = {"WIN_STRONG": "🟢", "WIN": "✅", "NEUTRAL": "🟡", "LOSS": "🔴"}
            for lbl, cnt in sorted(outcomes.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {em_map.get(lbl,'📊')} {lbl}: *{cnt}* ({cnt/labeled*100:.0f}%)" if labeled else f"  {lbl}: {cnt}")
            wins = outcomes.get("WIN_STRONG", 0) + outcomes.get("WIN", 0)
            if labeled > 0:
                lines.append(f"\n  📈 *Win rate:* {wins/labeled*100:.0f}% ({wins}/{labeled})")
            lines.append("")
        if update_stats:
            lines.append(f"*✅ Update:* {update_stats.get('updated', 0)} novas classificações\n")

        if labeled >= 100:
            lines.append("🟢 *Pronto para treinar!* Usa `/admin_full_pipeline --dip-only`")
        elif labeled >= 50:
            lines.append(f"🟡 *{labeled} amostras* — quase prontos (mín. recomendado: 100)")
            lines.append("_Já podes tentar `/admin_retrain` com resultados parciais._")
        else:
            lines.append(f"🕐 *{labeled} amostras* — precisa de mais dados (mín: 30 para treinar).")

        if not force_update and fill_db_outcomes:
            lines.append("\n_Força update agora com `/mldata update`_")
        _bc._reply("\n".join(lines))

    threading.Thread(target=_run, daemon=True).start()



# ── /ml_accuracy ─────────────────────────────────────────────────────────────────

def handle_ml_accuracy() -> None:
    """Mede a precisao real do modelo comparando predicoes com outcomes reais.
    Este e o backtest continuo que fecha o loop alerta -> resultado -> modelo.
    """
    def _run():
        try:
            from ml.prediction_log import compute_ml_accuracy
            acc = compute_ml_accuracy()
            if acc.get("skipped"):
                _bc._reply(f"ML Accuracy: {acc.get('reason', 'sem dados')}")
                return
            n    = acc["n_resolved"]
            prec = acc["precision"]
            rec  = acc["recall"]
            accu = acc["accuracy"]
            f1   = acc["f1"]
            brier = acc["brier_live"]
            win_r = acc["win_rate_actual"]
            lines = [
                f"ML Accuracy ({n} predicoes com outcome resolvido)",
                "",
                f"Precision: {prec:.0%}  (de cada COMPRAR, quantos ganharam)",
                f"Recall: {rec:.0%}  (de todos os ganhos, quantos identificamos)",
                f"Accuracy: {accu:.0%}  (total de acertos)",
                f"F1 Score: {f1:.3f}  (equilibrio precision/recall)",
                f"Brier live: {brier:.4f}  (calibracao das probabilidades)",
                f"Win rate real: {win_r:.0%}  (stocks que efectivamente subiram)",
                "",
                f"TP:{acc['tp']}  FP:{acc['fp']}  FN:{acc['fn']}  TN:{acc['tn']}",
            ]
            _bc._reply("\n".join(lines))
        except Exception as e:
            _bc._reply(f"Erro ao calcular accuracy: {e}")
    threading.Thread(target=_run, daemon=True).start()



# ── /paper_portfolio ─────────────────────────────────────────────────────────────

def handle_paper_performance(parts: list[str]) -> None:
    """/paper_portfolio [months | open | drip | missed]

    Paper trading portfolio report. The bot simulates its own BUY signals and
    tracks whether it beats SPY over 90 days.

    Usage:
      /paper_portfolio          → last 3 months
      /paper_portfolio 6        → last 6 months
      /paper_portfolio open     → all open positions with live weights
      /paper_portfolio drip     → DRIP audit trail for CORE positions
      /paper_portfolio missed   → MISSED_CAPITAL signals (lost alpha due to full pool)
    """
    def _run():
        try:
            from portfolio.paper_trading import (
                get_monthly_performance, format_performance_report, _load
            )

            sub = parts[1].lower() if len(parts) > 1 else ""

            # ── open positions ────────────────────────────────────────────────
            if sub == "open":
                from portfolio.paper_trading import (
                    get_paper_portfolio_value, get_paper_allocated,
                    get_paper_available, PAPER_PORTFOLIO_EUR,
                )
                from datetime import date as _date
                trades  = _load()
                open_ps = [t for t in trades if t.get("status") == "OPEN"]
                if not open_ps:
                    _bc._reply("No open paper trading positions.")
                    return
                port_val  = get_paper_portfolio_value(trades)
                allocated = get_paper_allocated(trades)
                available = get_paper_available(trades)
                port_ret  = (port_val - PAPER_PORTFOLIO_EUR) / max(PAPER_PORTFOLIO_EUR, 1) * 100
                lines = [
                    f"Paper portfolio — {len(open_ps)} open positions",
                    f"Value: €{port_val:.0f} ({port_ret:+.1f}%) | "
                    f"Allocated: €{allocated:.0f} | Available: €{available:.0f}",
                    "",
                ]
                for t in sorted(open_ps, key=lambda x: x["open_date"], reverse=True)[:15]:
                    days = (_date.today() - _date.fromisoformat(t["open_date"])).days
                    wt   = t.get("weight_pct", 0)
                    kind = t.get("trade_type", "DIP")
                    lines.append(
                        f"  [{kind}] {t['ticker']}: {t['open_date']} ({days}d) "
                        f"@ {t['open_price']:.2f} | {wt:.1f}% | €{t['amount_eur']:.0f}"
                    )
                _bc._reply("\n".join(lines))
                return

            # ── DRIP audit trail ──────────────────────────────────────────────
            if sub == "drip":
                from portfolio.paper_trading import get_drip_history, _TAX_DISCLAIMER
                data = get_drip_history()
                pos  = data["positions"]
                tot  = data["totals"]

                header = [
                    "CORE DRIP Ledger — Dividend Reinvestment Audit Trail",
                    "",
                    f"Positions: {tot['total_positions']} | "
                    f"Events: {tot['total_drip_events']} | "
                    f"Total reinvested: €{tot['total_reinvested_eur']:,.2f} (gross)",
                    "",
                    f"TAX: {_TAX_DISCLAIMER}",
                    "",
                ]
                _bc._reply("\n".join(header))

                if not pos:
                    _bc._reply(
                        "No DRIP events recorded yet.\n"
                        "DRIP tracking activates automatically on the next dividend ex-date "
                        "for each open CORE position."
                    )
                    return

                for p in pos:
                    events = p["drip_events"]
                    block = [
                        f"{p['ticker']} — opened {p['open_date']} | "
                        f"€{p['amount_eur']:,.2f} deployed",
                        f"  Shares: {p['shares_original']:.4f} → {p['shares_current']:.4f} "
                        f"(+{p['shares_added_via_drip']:.4f} via DRIP)",
                        f"  Total reinvested: €{p['total_reinvested_eur']:,.4f} "
                        f"across {len(events)} event(s)",
                    ]
                    if not events:
                        block.append("  No dividend events recorded yet.")
                    else:
                        block.append("  Events:")
                        for ev in events:
                            block.append(
                                f"    {ev['date']}: ${ev['per_share_usd']:.4f}/sh × "
                                f"{ev['shares_held']:.4f} sh = "
                                f"€{ev['gross_eur']:.4f} → "
                                f"+{ev['new_shares']:.6f} sh @ ${ev['reinvest_price_usd']:.2f} "
                                f"[{ev['tax_note']}]"
                            )
                    if p.get("note"):
                        block.append(f"  Note: {p['note']}")
                    _bc._reply("\n".join(block))

                # Send the full ledger as a downloadable Markdown file
                from portfolio.paper_trading import format_drip_report_md
                md_content = format_drip_report_md()
                sent = _send_document(
                    md_content.encode("utf-8"),
                    filename="dipr_drip_ledger.md",
                    caption=(
                        "DipRadar CORE DRIP Ledger — full audit trail.\n"
                        "Every dividend event, date, amount, shares added.\n"
                        "GROSS (pre-tax). See disclaimer inside."
                    ),
                )
                if not sent:
                    _bc._reply("Could not send file via Telegram. Use /admin_publish_charts to push to GitHub.")
                return

            # ── missed signals ────────────────────────────────────────────────
            if sub == "missed":
                from portfolio.paper_trading import get_missed_trades_stats, _load as _pt_load
                trades = _pt_load()
                missed = [t for t in trades if t.get("status") == "MISSED_CAPITAL"]
                stats  = get_missed_trades_stats(trades)

                if not missed:
                    _bc._reply("No MISSED_CAPITAL signals recorded yet.")
                    return

                lines = [
                    f"MISSED_CAPITAL — {len(missed)} signals not opened (pool was full)",
                    f"Resolved: {stats.get('resolved', 0)} | Pending: {stats.get('pending', 0)}",
                    "",
                ]
                for m in sorted(missed, key=lambda x: x.get("signal_date", ""), reverse=True)[:20]:
                    resolved = m.get("actual_return_90d")
                    ret_str  = f"{resolved:+.1f}%" if resolved is not None else "pending"
                    alpha_str = (
                        f" vs SPY {m.get('actual_alpha_90d', 0):+.1f}%"
                        if m.get("actual_alpha_90d") is not None else ""
                    )
                    lines.append(
                        f"  [{m.get('trade_type','?')}] {m.get('ticker','?')} "
                        f"{m.get('signal_date','?')}: "
                        f"requested €{m.get('requested_eur',0):.0f}, "
                        f"available €{m.get('available_eur',0):.0f} → {ret_str}{alpha_str}"
                    )
                _bc._reply("\n".join(lines))
                return

            # ── default: monthly summary ──────────────────────────────────────
            months = 3
            if sub:
                try:
                    months = int(sub)
                except ValueError:
                    pass

            perf   = get_monthly_performance(months_back=months)
            report = format_performance_report(perf)
            _bc._reply(f"Paper Trading — Bot vs Market\n\n{report}")

            from portfolio.paper_trading import format_monthly_trade_list
            trade_list = format_monthly_trade_list(months_back=months)
            if trade_list and "Sem trades" not in trade_list:
                _bc._reply(f"Trades by month:\n{trade_list}")

        except Exception as e:
            logging.error(f"[paper_performance] {e}", exc_info=True)
            _bc._reply(f"Error calculating paper performance: {e}")

    threading.Thread(target=_run, daemon=True, name="paper-perf").start()




# ── /performance ─────────────────────────────────────────────────────────────────

def handle_performance(parts: list[str]) -> None:
    """/performance [YYYY-MM-DD] [score_min]

    Simula o portfolio DipRadar e calcula retorno anual, Sharpe e max drawdown.

    Exemplos:
      /performance               → todo o histórico disponível
      /performance 2025-01-01    → desde Jan 2025
      /performance 2024-01-01 65 → desde Jan 2024, score mínimo 65
    """
    _bc._reply(
        "⏳ *A calcular performance do portfolio...*\n"
        "_Lê o alert\\_db.csv e simula seguir todas as recomendações. Pode demorar 30s._"
    )

    def _run():
        try:
            from portfolio.portfolio_simulator import run_portfolio_backtest, format_portfolio_result

            period_start = None
            score_min    = 60.0

            for p in parts[1:]:
                p = p.strip()
                try:
                    float(p)
                    score_min = float(p)
                    continue
                except ValueError:
                    pass
                if len(p) == 10 and p[4] == "-":
                    period_start = p

            result = run_portfolio_backtest(
                period_start=period_start,
                score_threshold=score_min,
            )
            _bc._reply(format_portfolio_result(result))
        except RuntimeError as e:
            _bc._reply(f"⚠️ *Performance:* {e}\n_Certifica-te que o alert\\_db.csv tem outcomes preenchidos. Usa /mldata update._")
        except Exception as e:
            logging.error(f"[performance] {e}", exc_info=True)
            _bc._reply(f"❌ *Erro ao calcular performance:*\n`{_md_safe(e)}`")

    threading.Thread(target=_run, daemon=True, name="performance").start()




# ── /health handler ─────────────────────────────────────────────────────────────

def handle_health(parts: list[str]) -> None:
    """
    /health          → dashboard completo (RAM, CPU, scans, APIs, modelos, erros)
    /health errors   → log detalhado dos últimos erros críticos
    """
    sub = parts[1].lower() if len(parts) > 1 else "dashboard"

    if sub in ("errors", "erros", "log"):
        with health_monitor._lock:
            errors = list(health_monitor._error_log)
        if not errors:
            _bc._reply("✅ *Sem erros registados* — sistema limpo.")
            return
        lines = [f"*🚨 Log de erros — últimos {len(errors)}*", ""]
        for e in reversed(errors):
            ts_str  = e["ts"].strftime("%d/%m %H:%M:%S")
            tb_prev = e["tb"][-400:] if len(e["tb"]) > 400 else e["tb"]
            lines.append(f"🔴 `{ts_str}` *[{e['context']}]*")
            lines.append(f"   _{e['error'][:120]}_")
            lines.append(f"```\n{tb_prev}\n```")
            lines.append("")
        _bc._reply("\n".join(lines))
        return

    # Dashboard principal — corre em thread para não bloquear o poll
    _bc._reply("_🔄 A recolher métricas... (2-5s)_")

    def _run():
        try:
            report = health_monitor.build_health_report(ping_apis=True)
            _bc._reply(report)
        except Exception as e:
            _bc._reply(f"❌ Erro ao construir health report: `{e}`")

    threading.Thread(target=_run, daemon=True).start()




# /buy, /sell, /ppr, /etf, /liquidez handlers moved to commands/portfolio_extras.py (Phase 8.4)
from commands.portfolio_extras import (
    handle_buy as _handle_buy_impl,
    handle_sell as _handle_sell_impl,
    handle_ppr as _handle_ppr_impl,
    handle_etf as _handle_etf_impl,
    handle_liquidez as _handle_liquidez_impl,
)


