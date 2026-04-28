"""
backtest.py — Avaliação automática dos alertas históricos do DipRadar.

Lógica:
  - Após cada alerta (COMPRAR / MONITORIZAR), regista o preço de entrada
    via append_backtest_entry() em state.py.
  - O backtest_runner() corre todos os dias às 21h30 e preenche os campos
    price_5d / price_10d / price_20d para entradas com 5/10/20 dias úteis
    já decorridos, calculando o P&L %.
  - O Saturday report chama build_backtest_summary() para mostrar o resumo.
  - Auto-calibração: suggest_min_score() calcula o score mínimo óptimo
    com base nos winners históricos e sugere ajuste ao MIN_DIP_SCORE.

Nenhuma API key necessária — usa yfinance.history().
"""

import time
import logging
from datetime import datetime, timedelta
import yfinance as yf
from state import load_backtest_log, save_backtest_log


def _business_days_since(date_iso: str) -> int:
    """Conta dias úteis (seg-sex) entre date_iso e hoje."""
    try:
        start = datetime.fromisoformat(date_iso).date()
        end   = datetime.now().date()
        count = 0
        cur   = start
        while cur < end:
            cur += timedelta(days=1)
            if cur.weekday() < 5:
                count += 1
        return count
    except Exception:
        return 0


def _get_price_n_days_ago(symbol: str, n_business_days: int) -> float | None:
    """
    Devolve o preço de fecho aproximadamente n dias úteis atrás.
    Usa history(period='60d') e indexa pelo offset.
    """
    try:
        time.sleep(3)
        hist = yf.Ticker(symbol).history(period="60d", interval="1d")["Close"].dropna()
        if len(hist) < n_business_days + 1:
            return None
        return float(hist.iloc[-(n_business_days + 1)])
    except Exception as e:
        logging.warning(f"Backtest price {symbol} -{n_business_days}d: {e}")
        return None


def backtest_runner() -> int:
    """
    Preenche os campos price_5d/10d/20d para entradas pendentes.
    Devolve o número de entradas actualizadas.
    Corre todos os dias às 21h30 (agendado em main.py).
    """
    entries = load_backtest_log()
    updated = 0

    for entry in entries:
        if entry.get("resolved"):
            continue
        symbol      = entry["symbol"]
        price_alert = entry.get("price_alert") or 0
        if not price_alert:
            continue

        bd = _business_days_since(entry["date_iso"])

        changed = False
        for days, key_p, key_pnl in [
            (5,  "price_5d",  "pnl_5d"),
            (10, "price_10d", "pnl_10d"),
            (20, "price_20d", "pnl_20d"),
        ]:
            if bd >= days and entry.get(key_p) is None:
                p = _get_price_n_days_ago(symbol, bd - days)
                if p is not None:
                    entry[key_p]   = round(p, 4)
                    entry[key_pnl] = round((p - price_alert) / price_alert * 100, 2)
                    changed = True

        if all(entry.get(k) is not None for k in ("price_5d", "price_10d", "price_20d")):
            entry["resolved"] = True

        if changed:
            updated += 1

    if updated:
        save_backtest_log(entries)
        logging.info(f"Backtest: {updated} entradas actualizadas")

    return updated


def suggest_min_score(entries: list[dict] | None = None, horizon: str = "pnl_5d") -> dict:
    """
    Analisa os resultados históricos e sugere o score mínimo óptimo.

    Algoritmo:
      - Para cada threshold candidato (45, 50, 55, 60, 65, 70, 75, 80),
        calcula a win rate e o P&L médio dos alertas com score >= threshold.
      - O threshold óptimo é o mais baixo que mantém win rate >= 55%.
      - Se nenhum atingir 55%, usa o de maior win rate.

    Devolve dict com:
      suggested_min : int  — threshold sugerido
      current_best  : dict — stats do threshold sugerido
      all_thresholds: list — stats de todos os candidatos
      reason        : str  — explicação humana
    """
    if entries is None:
        entries = load_backtest_log()

    resolved = [e for e in entries if e.get(horizon) is not None]
    if len(resolved) < 5:
        return {
            "suggested_min": None,
            "reason": f"Insuficiente (só {len(resolved)} entradas resolvidas em {horizon})",
            "all_thresholds": [],
        }

    candidates = [45, 50, 55, 60, 65, 70, 75, 80]
    results = []
    for thresh in candidates:
        subset = [e for e in resolved if (e.get("score") or 0) >= thresh]
        if len(subset) < 3:
            results.append({"threshold": thresh, "n": len(subset), "win_rate": None, "avg_pnl": None})
            continue
        pnls    = [e[horizon] for e in subset]
        win_n   = sum(1 for x in pnls if x > 0)
        win_r   = win_n / len(pnls)
        avg_pnl = sum(pnls) / len(pnls)
        results.append({
            "threshold": thresh,
            "n":         len(subset),
            "win_rate":  round(win_r, 3),
            "avg_pnl":   round(avg_pnl, 2),
        })

    # Melhor threshold: o mais baixo com win rate >= 55%
    viable = [r for r in results if r["win_rate"] is not None and r["win_rate"] >= 0.55]
    if viable:
        best = min(viable, key=lambda x: x["threshold"])
        reason = (
            f"Score ≥{best['threshold']} tem win rate {best['win_rate']*100:.0f}% "
            f"({best['n']} alertas, avg P&L {best['avg_pnl']:+.1f}%)"
        )
    else:
        # Nenhum atinge 55% — usa o de maior win rate
        valid = [r for r in results if r["win_rate"] is not None]
        if not valid:
            return {"suggested_min": None, "reason": "Sem dados suficientes", "all_thresholds": results}
        best   = max(valid, key=lambda x: x["win_rate"])
        reason = (
            f"Nenhum threshold atinge 55% win rate. "
            f"Melhor disponível: ≥{best['threshold']} com {best['win_rate']*100:.0f}% "
            f"({best['n']} alertas)"
        )

    return {
        "suggested_min":  best["threshold"],
        "current_best":   best,
        "all_thresholds": results,
        "reason":         reason,
    }


def build_backtest_summary(min_entries: int = 3) -> str:
    """
    Gera bloco Markdown para o Saturday report.
    Só mostra resultados para entradas com pelo menos price_5d preenchido.
    Inclui sugestão de auto-calibração do MIN_DIP_SCORE.
    """
    entries  = load_backtest_log()
    resolved = [e for e in entries if e.get("pnl_5d") is not None]

    if len(resolved) < min_entries:
        total   = len(entries)
        pending = total - len(resolved)
        if total == 0:
            return "_Backtest: sem alertas registados ainda._"
        return f"_Backtest: {total} alertas registados, {pending} ainda sem dados suficientes (aguarda 5 dias úteis)._"

    comprar = [e for e in resolved if e["verdict"] == "COMPRAR"]
    monitor = [e for e in resolved if e["verdict"] != "COMPRAR"]

    def _stats(lst: list, label: str) -> list[str]:
        if not lst:
            return []
        pnl5  = [e["pnl_5d"]  for e in lst if e.get("pnl_5d")  is not None]
        pnl10 = [e["pnl_10d"] for e in lst if e.get("pnl_10d") is not None]
        pnl20 = [e["pnl_20d"] for e in lst if e.get("pnl_20d") is not None]
        win5  = sum(1 for x in pnl5 if x > 0)
        lines = [f"  *{label}* ({len(lst)} alertas):"]
        if pnl5:  lines.append(f"    5d:  avg {sum(pnl5)/len(pnl5):+.1f}% | win {win5}/{len(pnl5)} ({win5/len(pnl5)*100:.0f}%)")
        if pnl10:
            win10 = sum(1 for x in pnl10 if x > 0)
            lines.append(f"    10d: avg {sum(pnl10)/len(pnl10):+.1f}% | win {win10}/{len(pnl10)} ({win10/len(pnl10)*100:.0f}%)")
        if pnl20:
            win20 = sum(1 for x in pnl20 if x > 0)
            lines.append(f"    20d: avg {sum(pnl20)/len(pnl20):+.1f}% | win {win20}/{len(pnl20)} ({win20/len(pnl20)*100:.0f}%)")
        return lines

    lines = [
        "*🔬 Backtest de Alertas:*",
        f"_Total avaliados: {len(resolved)} | COMPRAR: {len(comprar)} | MONITORIZAR: {len(monitor)}_",
        "",
    ]
    lines += _stats(comprar, "COMPRAR")
    if comprar and monitor:
        lines.append("")
    lines += _stats(monitor, "MONITORIZAR")

    all_with_5d = sorted(resolved, key=lambda x: x["pnl_5d"], reverse=True)
    if all_with_5d:
        lines += ["", "  *🏆 Melhores (5d):*"]
        for e in all_with_5d[:3]:
            lines.append(f"    ✅ *{e['symbol']}* {e['pnl_5d']:+.1f}% | score {e['score']:.0f} | {e['date']}")
        lines += ["", "  *💀 Piores (5d):*"]
        for e in all_with_5d[-3:][::-1]:
            lines.append(f"    ❌ *{e['symbol']}* {e['pnl_5d']:+.1f}% | score {e['score']:.0f} | {e['date']}")

    winners = [e for e in resolved if e.get("pnl_5d", 0) > 0]
    losers  = [e for e in resolved if e.get("pnl_5d", 0) <= 0]
    if winners and losers:
        avg_w = sum(e["score"] for e in winners) / len(winners)
        avg_l = sum(e["score"] for e in losers)  / len(losers)
        lines += [
            "",
            "  *📐 Calibração do score:*",
            f"    Score médio winners: *{avg_w:.1f}* | losers: *{avg_l:.1f}*",
        ]
        if avg_w > avg_l + 1:
            lines.append("    _Score discrimina bem winners/losers ✅_")
        else:
            lines.append("    _Score tem pouco poder discriminativo — considera ajustar thresholds ⚠️_")

    # ── Auto-calibração: sugestão de MIN_DIP_SCORE ────────────────────
    try:
        cal = suggest_min_score(entries)
        if cal.get("suggested_min") is not None:
            lines += [
                "",
                "  *🤖 Auto-calibração MIN_DIP_SCORE:*",
                f"    Sugestão baseada em histórico: *score ≥{cal['suggested_min']}*",
                f"    _{cal['reason']}_",
            ]
            # Tabela compacta de todos os thresholds
            valid_rows = [r for r in cal["all_thresholds"] if r["win_rate"] is not None]
            if valid_rows:
                lines.append("    _Thresholds:_")
                for r in valid_rows:
                    marker = " ←" if r["threshold"] == cal["suggested_min"] else ""
                    lines.append(
                        f"      ≥{r['threshold']}: win {r['win_rate']*100:.0f}% | "
                        f"avg {r['avg_pnl']:+.1f}% | n={r['n']}{marker}"
                    )
            lines.append(
                f"    _→ Actualiza MIN\_DIP\_SCORE no Railway para {cal['suggested_min']} se concordas_"
            )
        else:
            lines += ["", f"  _Auto-calibração: {cal.get('reason', 'sem dados')}_"]
    except Exception as e:
        logging.warning(f"Auto-calibração: {e}")

    return "\n".join(lines)
