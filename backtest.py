"""
backtest.py — Avaliação automática dos alertas históricos do DipRadar.

Lógica:
  - Após cada alerta (COMPRAR / MONITORIZAR), regista o preço de entrada
    via append_backtest_entry() em state.py.
  - O backtest_runner() corre todos os dias às 21h30 e preenche os campos
    price_5d / price_10d / price_20d para entradas com 5/10/20 dias úteis
    já decorridos, calculando o P&L %.
  - O Saturday report chama build_backtest_summary() para mostrar o resumo.

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
        # hist.iloc[-1] = hoje; hist.iloc[-(n+1)] = n dias atrás
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

        # Marca como resolved quando os 3 campos estiverem preenchidos
        if all(entry.get(k) is not None for k in ("price_5d", "price_10d", "price_20d")):
            entry["resolved"] = True

        if changed:
            updated += 1

    if updated:
        save_backtest_log(entries)
        logging.info(f"Backtest: {updated} entradas actualizadas")

    return updated


def build_backtest_summary(min_entries: int = 3) -> str:
    """
    Gera bloco Markdown para o Saturday report.
    Só mostra resultados para entradas com pelo menos price_5d preenchido.
    """
    entries = load_backtest_log()
    resolved = [e for e in entries if e.get("pnl_5d") is not None]

    if len(resolved) < min_entries:
        total = len(entries)
        pending = total - len(resolved)
        if total == 0:
            return "_Backtest: sem alertas registados ainda._"
        return f"_Backtest: {total} alertas registados, {pending} ainda sem dados suficientes (aguarda 5 dias úteis)._"

    # Métricas globais
    comprar   = [e for e in resolved if e["verdict"] == "COMPRAR"]
    monitor   = [e for e in resolved if e["verdict"] != "COMPRAR"]

    def _stats(lst: list, label: str) -> list[str]:
        if not lst:
            return []
        pnl5  = [e["pnl_5d"]  for e in lst if e.get("pnl_5d")  is not None]
        pnl10 = [e["pnl_10d"] for e in lst if e.get("pnl_10d") is not None]
        pnl20 = [e["pnl_20d"] for e in lst if e.get("pnl_20d") is not None]
        win5  = sum(1 for x in pnl5  if x > 0)
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

    # Top 3 melhores e piores a 5d
    all_with_5d = sorted(resolved, key=lambda x: x["pnl_5d"], reverse=True)
    if all_with_5d:
        lines += ["", "  *🏆 Melhores (5d):*"]
        for e in all_with_5d[:3]:
            lines.append(f"    ✅ *{e['symbol']}* {e['pnl_5d']:+.1f}% | score {e['score']:.0f} | {e['date']}")
        lines += ["", "  *💀 Piores (5d):*"]
        for e in all_with_5d[-3:][::-1]:
            lines.append(f"    ❌ *{e['symbol']}* {e['pnl_5d']:+.1f}% | score {e['score']:.0f} | {e['date']}")

    # Calibração: score médio dos winners vs losers
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

    return "\n".join(lines)
