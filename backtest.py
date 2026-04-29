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
  - fill_db_outcomes() vive em alert_db.py (fonte da verdade única).
    Preenche MFE/MAE/return a 1m/3m/6m na alert_db.csv ao sábado.
  - run_historical_backtest() (F2) simula dips históricos retroactivos
    para gerar dados de treino ML sem esperar 6 meses de dados live.
    F2-A: detecção de dips + scoring (anti-clustering 20d).
    F2-B: MFE/MAE/returns forward-looking + outcome_label canónico.
    F2-C: escrita do CSV idempotente + build_historical_training_set().
    Lookback fixo: 5 anos (sweet spot: apanha 2022 bear + 2023 recovery
    sem contaminar o modelo com fundamentais de empresas estruturalmente
    diferentes há mais de 5 anos).

Nenhuma API key necessária — usa yfinance.history().
"""

import csv
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
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


# ── Score calibration ─────────────────────────────────────────────────────

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

    viable = [r for r in results if r["win_rate"] is not None and r["win_rate"] >= 0.55]
    if viable:
        best = min(viable, key=lambda x: x["threshold"])
        reason = (
            f"Score \u2265{best['threshold']} tem win rate {best['win_rate']*100:.0f}% "
            f"({best['n']} alertas, avg P&L {best['avg_pnl']:+.1f}%)"
        )
    else:
        valid = [r for r in results if r["win_rate"] is not None]
        if not valid:
            return {"suggested_min": None, "reason": "Sem dados suficientes", "all_thresholds": results}
        best   = max(valid, key=lambda x: x["win_rate"])
        reason = (
            f"Nenhum threshold atinge 55% win rate. "
            f"Melhor dispon\u00edvel: \u2265{best['threshold']} com {best['win_rate']*100:.0f}% "
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
        "*\U0001f52c Backtest de Alertas:*",
        f"_Total avaliados: {len(resolved)} | COMPRAR: {len(comprar)} | MONITORIZAR: {len(monitor)}_",
        "",
    ]
    lines += _stats(comprar, "COMPRAR")
    if comprar and monitor:
        lines.append("")
    lines += _stats(monitor, "MONITORIZAR")

    all_with_5d = sorted(resolved, key=lambda x: x["pnl_5d"], reverse=True)
    if all_with_5d:
        lines += ["", "  *\U0001f3c6 Melhores (5d):*"]
        for e in all_with_5d[:3]:
            lines.append(f"    \u2705 *{e['symbol']}* {e['pnl_5d']:+.1f}% | score {e['score']:.0f} | {e['date']}")
        lines += ["", "  *\U0001f480 Piores (5d):*"]
        for e in all_with_5d[-3:][::-1]:
            lines.append(f"    \u274c *{e['symbol']}* {e['pnl_5d']:+.1f}% | score {e['score']:.0f} | {e['date']}")

    winners = [e for e in resolved if e.get("pnl_5d", 0) > 0]
    losers  = [e for e in resolved if e.get("pnl_5d", 0) <= 0]
    if winners and losers:
        avg_w = sum(e["score"] for e in winners) / len(winners)
        avg_l = sum(e["score"] for e in losers)  / len(losers)
        lines += [
            "",
            "  *\U0001f4d0 Calibração do score:*",
            f"    Score médio winners: *{avg_w:.1f}* | losers: *{avg_l:.1f}*",
        ]
        if avg_w > avg_l + 1:
            lines.append("    _Score discrimina bem winners/losers \u2705_")
        else:
            lines.append("    _Score tem pouco poder discriminativo \u2014 considera ajustar thresholds \u26a0\ufe0f_")

    try:
        cal = suggest_min_score(entries)
        if cal.get("suggested_min") is not None:
            lines += [
                "",
                "  *\U0001f916 Auto-calibração MIN\_DIP\_SCORE:*",
                f"    Sugestão baseada em histórico: *score \u2265{cal['suggested_min']}*",
                f"    _{cal['reason']}_",
            ]
            valid_rows = [r for r in cal["all_thresholds"] if r["win_rate"] is not None]
            if valid_rows:
                lines.append("    _Thresholds:_")
                for r in valid_rows:
                    marker = " \u2190" if r["threshold"] == cal["suggested_min"] else ""
                    lines.append(
                        f"      \u2265{r['threshold']}: win {r['win_rate']*100:.0f}% | "
                        f"avg {r['avg_pnl']:+.1f}% | n={r['n']}{marker}"
                    )
            lines.append(
                f"    _\u2192 Actualiza MIN_DIP_SCORE no Railway para {cal['suggested_min']} se concordas_"
            )
        else:
            lines += ["", f"  _Auto-calibração: {cal.get('reason', 'sem dados')}_"]
    except Exception as e:
        logging.warning(f"Auto-calibração: {e}")

    return "\n".join(lines)


# ── Historical backtest (F2) ──────────────────────────────────────────────

# Lookback fixo: 5 anos — sweet spot entre dados de crise suficientes
# e estabilidade fundamental aceitável (2021-2026 apanha bear 2022 + recovery 2023).
_LOOKBACK_YEARS = 5

# Janelas em dias de mercado (trading days):
#   1 mês  ≈ 21 td  | 3 meses ≈ 63 td  | 6 meses ≈ 126 td
_TD_1M  = 21
_TD_3M  = 63
_TD_6M  = 126

# Colunas do CSV de saída: super-conjunto de alert_db._FIELDS + campos extra
# de backtest. O modelo ML apenas usa as colunas que existem em ambas as
# fontes — o campo 'source' permite filtrar na fase de treino.
_HIST_FIELDS = [
    # Identificação
    "date_iso", "symbol", "name", "sector",
    # Classificação
    "category", "score",
    # Preço e mercado
    "price", "market_cap_b", "drawdown_52w", "change_day_pct",
    # Técnicos
    "rsi", "volume_ratio",
    # Fundamentais (snapshot de hoje)
    "pe", "fcf_yield", "revenue_growth", "gross_margin",
    "debt_equity", "dividend_yield", "analyst_upside",
    # Outcomes forward-looking (F2-B)
    "price_1m", "price_3m", "price_6m",
    "return_1m", "return_3m", "return_6m",
    "mfe_3m", "mae_3m",
    "outcome_label",  # WIN_40 | WIN_20 | NEUTRAL | LOSS_15 | "" (pending)
    # Metadado de rastreabilidade
    "source",         # sempre "historical_backtest"
]

# Caminho padrão do CSV histórico (separado do alert_db.csv live)
_HIST_DB_PATH = (
    Path("/data/hist_backtest.csv")
    if Path("/data").exists()
    else Path("/tmp/hist_backtest.csv")
)


def _rsi_series(close_series, period: int = 14):
    """
    RSI rolling vectorizado (Wilder EWM) — idêntico ao TradingView/yfinance.
    """
    delta    = close_series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _drawdown_from_rolling_high(close_series, window: int = 252):
    """
    Drawdown % vs máximo rolling dos últimos `window` trading days.
    Valores são sempre <= 0.
    """
    rolling_max = close_series.rolling(window=window, min_periods=1).max()
    return (close_series - rolling_max) / rolling_max * 100


def _build_hybrid_fund(info: dict, row) -> dict:
    """
    Dicionário `fund` híbrido:
      Técnicos = Dia 0 real | Fundamentais = snapshot de hoje.
    """
    price    = float(row["Close"])
    mc_today = info.get("marketCap") or 0
    fcf_raw  = info.get("freeCashflow")
    fcf_yield = (fcf_raw / mc_today) if (fcf_raw and mc_today > 0) else None

    return {
        "price":              price,
        "rsi":                float(row["rsi_14"]) if row["rsi_14"] == row["rsi_14"] else None,
        "drawdown_from_high": float(row["drawdown_52w"]),
        "volume":             float(row["Volume"]),
        "average_volume":     float(row["avg_vol_20d"]) if row["avg_vol_20d"] == row["avg_vol_20d"] else 0,
        "market_cap":         mc_today,
        "fcf_yield":          fcf_yield,
        "gross_margin":       info.get("grossMargins") or 0,
        "revenue_growth":     info.get("revenueGrowth") or 0,
        "debt_equity":        info.get("debtToEquity"),
        "dividend_yield":     info.get("dividendYield") or 0,
        "beta":               info.get("beta"),
        "analyst_upside":     _analyst_upside(info),
        "pe":                 info.get("trailingPE") or 0,
        "sector":             info.get("sector") or "",
        "name":               (info.get("longName") or info.get("shortName") or "")[:40],
    }


def _analyst_upside(info: dict) -> float:
    """% upside até ao target médio dos analistas."""
    target = info.get("targetMeanPrice")
    price  = info.get("currentPrice") or info.get("regularMarketPrice")
    if target and price and price > 0:
        return round((target - price) / price * 100, 1)
    return 0.0


def _detect_dips(df) -> object:
    """
    Indicadores vectorizados + anti-clustering 20d.
    Condições: RSI < 35 AND Drawdown 52w < -15%.
    """
    import pandas as pd

    df = df.copy()
    df["rsi_14"]       = _rsi_series(df["Close"])
    df["drawdown_52w"] = _drawdown_from_rolling_high(df["Close"], window=252)
    df["avg_vol_20d"]  = df["Volume"].rolling(window=20, min_periods=5).mean()
    df["change_pct"]   = df["Close"].pct_change() * 100
    df["is_dip"]       = (df["rsi_14"] < 35) & (df["drawdown_52w"] < -15.0)

    dip_dates = df.index[df["is_dip"]]
    if len(dip_dates) == 0:
        return df[df["is_dip"]]

    kept     = [dip_dates[0]]
    cooldown = pd.Timedelta(days=20)
    for dt in dip_dates[1:]:
        if dt - kept[-1] >= cooldown:
            kept.append(dt)

    return df.loc[kept]


# ── F2-B: forward-looking outcomes ─────────────────────────────────────

def _forward_outcomes(
    df,
    dip_iloc: int,
    price_entry: float,
    category: str,
) -> dict:
    """
    F2-B — Métricas forward-looking para um dip, usando OHLCV em memória.

    Janelas (trading days):
      T+1m = iloc+21 | T+3m = iloc+63 | T+6m = iloc+126

    Right-edge censor: se offset > len(df), devolve None ("pending").

    MFE = Max(High[iloc+1..iloc+63] - price_entry) / price_entry
    MAE = Min(Low[iloc+1..iloc+63]  - price_entry) / price_entry

    outcome_label: canónico via alert_db._resolve_outcome_label.
      Apartamento  → 6m > 3m > 1m
      Rotação      → 3m > 6m > 1m
      Hold Forever → sem label
    """
    from alert_db import _resolve_outcome_label
    from score import CATEGORY_APARTAMENTO, CATEGORY_HOLD_FOREVER

    total_rows  = len(df)
    max_forward = total_rows - dip_iloc - 1

    result = {
        "price_1m":  None, "price_3m":  None, "price_6m":  None,
        "return_1m": None, "return_3m": None, "return_6m": None,
        "mfe_3m": None, "mae_3m": None, "outcome_label": "",
    }

    if max_forward <= 0 or price_entry <= 0:
        return result

    def _close_at(offset_td):
        t = dip_iloc + offset_td
        return float(df["Close"].iloc[t]) if t < total_rows else None

    def _slice_forward(end_td):
        s, e = dip_iloc + 1, min(dip_iloc + end_td + 1, total_rows)
        return df.iloc[s:e] if s < total_rows else None

    p1m = _close_at(_TD_1M)
    if p1m:
        result["price_1m"]  = round(p1m, 4)
        result["return_1m"] = round((p1m - price_entry) / price_entry * 100, 2)

    p3m = _close_at(_TD_3M)
    if p3m:
        result["price_3m"]  = round(p3m, 4)
        result["return_3m"] = round((p3m - price_entry) / price_entry * 100, 2)

    sl = _slice_forward(_TD_3M)
    if sl is not None and not sl.empty:
        result["mfe_3m"] = round((sl["High"].max() - price_entry) / price_entry * 100, 2)
        result["mae_3m"] = round((sl["Low"].min()  - price_entry) / price_entry * 100, 2)

    p6m = _close_at(_TD_6M)
    if p6m:
        result["price_6m"]  = round(p6m, 4)
        result["return_6m"] = round((p6m - price_entry) / price_entry * 100, 2)

    if CATEGORY_HOLD_FOREVER not in category:
        prio = (
            ("return_6m", "return_3m", "return_1m")
            if CATEGORY_APARTAMENTO in category
            else ("return_3m", "return_6m", "return_1m")
        )
        for field in prio:
            val = result.get(field)
            if val is not None:
                result["outcome_label"] = _resolve_outcome_label(val)
                break

    return result


# ── F2-C: CSV writer + idempotency + public entry-point ───────────────

def _load_existing_keys(csv_path: Path) -> set:
    """
    Lê o CSV existente e devolve um set de chaves compostas
    (symbol, date_iso) para evitar duplicados.

    Idempotência: re-correr o build_historical_training_set() não
    duplica linhas — apenas acrescenta dips novos ou substitui
    dips com label pendente que entretanto ficaram resolvidos.
    """
    keys: set = set()
    if not csv_path.exists():
        return keys
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sym  = row.get("symbol", "")
                date = row.get("date_iso", "")
                if sym and date:
                    keys.add((sym, date))
    except Exception as e:
        logging.warning(f"[hist_csv] Erro a ler chaves existentes: {e}")
    return keys


def _write_hist_csv(
    dip_rows: list[dict],
    csv_path: Path,
    existing_keys: set,
) -> int:
    """
    Acrescenta ao CSV apenas as linhas que ainda não existem
    (guard by existing_keys).

    Cria o ficheiro com cabeçalho se não existir.
    Devolve o número de linhas efectivamente escritas.
    """
    new_rows = [
        r for r in dip_rows
        if (r["symbol"], r["date_iso"]) not in existing_keys
    ]
    if not new_rows:
        logging.info("[hist_csv] Nenhuma linha nova para escrever.")
        return 0

    write_header = not csv_path.exists()
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=_HIST_FIELDS,
                extrasaction="ignore",  # ignora campos extra do dip_record
            )
            if write_header:
                writer.writeheader()
            writer.writerows(new_rows)
        logging.info(f"[hist_csv] {len(new_rows)} linha(s) escritas em {csv_path}")
        return len(new_rows)
    except Exception as e:
        logging.error(f"[hist_csv] Erro ao escrever CSV: {e}")
        return 0


def run_historical_backtest(
    tickers: list[str],
    output_path: Path | None = None,
    lookback_years: int = _LOOKBACK_YEARS,
    min_score: float = 45.0,
    rsi_threshold: float = 35.0,
    drawdown_threshold: float = -15.0,
    dry_run: bool = False,
) -> dict:
    """
    F2 (A+B) — Motor interno: detecção de dips + outcomes forward-looking.

    Preferência: chama build_historical_training_set() para a orquestração
    completa com escrita CSV (F2-C). Esta função é mantida pública para
    testes unitários e dry_run.

    Retorna dict:
      total_dips, written, skipped, censored, errors, dip_rows
    """
    from score import calculate_dip_score, classify_dip_category, is_bluechip

    stats: dict = {
        "total_dips": 0, "written": 0, "skipped": 0,
        "censored":   0, "errors":  0, "dip_rows": [],
    }

    for symbol in tickers:
        logging.info(f"[hist_backtest] A processar {symbol}...")
        try:
            time.sleep(2)
            ticker_obj = yf.Ticker(symbol)
            info       = ticker_obj.info or {}

            df_full = ticker_obj.history(period=f"{lookback_years}y", interval="1d")
            if df_full is None or df_full.empty or len(df_full) < 30:
                logging.warning(f"[hist_backtest] {symbol}: histórico insuficiente")
                stats["errors"] += 1
                continue

            dip_df = _detect_dips(df_full)
            if dip_df.empty:
                logging.info(f"[hist_backtest] {symbol}: nenhum dip detectado")
                continue

            logging.info(f"[hist_backtest] {symbol}: {len(dip_df)} dip(s) candidatos")
            stats["total_dips"] += len(dip_df)

            # Recalcular indicadores no df_full completo para o slicing forward
            df_full = df_full.copy()
            df_full["rsi_14"]       = _rsi_series(df_full["Close"])
            df_full["drawdown_52w"] = _drawdown_from_rolling_high(df_full["Close"], window=252)
            df_full["avg_vol_20d"]  = df_full["Volume"].rolling(window=20, min_periods=5).mean()
            df_full["change_pct"]   = df_full["Close"].pct_change() * 100

            for dip_date, row in dip_df.iterrows():
                try:
                    fund  = _build_hybrid_fund(info, row)
                    score, _ = calculate_dip_score(
                        fundamentals=fund,
                        symbol=symbol,
                        sector_change=None,
                        stock_change_pct=float(row["change_pct"]) if row["change_pct"] == row["change_pct"] else None,
                    )

                    if score < min_score:
                        stats["skipped"] += 1
                        continue

                    bc_flag  = is_bluechip(fund)
                    category = classify_dip_category(fund, score, bc_flag)
                    dip_iloc = df_full.index.get_loc(dip_date)

                    outcomes = _forward_outcomes(
                        df=df_full, dip_iloc=dip_iloc,
                        price_entry=float(row["Close"]), category=category,
                    )

                    if outcomes["outcome_label"] == "":
                        stats["censored"] += 1

                    vol_ratio = ""
                    if row["avg_vol_20d"] and row["avg_vol_20d"] > 0:
                        vol_ratio = round(float(row["Volume"]) / float(row["avg_vol_20d"]), 2)

                    dip_record = {
                        "symbol":         symbol,
                        "date_iso":       dip_date.date().isoformat(),
                        "price":          round(float(row["Close"]), 4),
                        "score":          round(score, 1),
                        "rsi":            round(float(row["rsi_14"]), 1) if row["rsi_14"] == row["rsi_14"] else "",
                        "drawdown_52w":   round(float(row["drawdown_52w"]), 2),
                        "volume_ratio":   vol_ratio,
                        "change_day_pct": round(float(row["change_pct"]), 2) if row["change_pct"] == row["change_pct"] else "",
                        "category":       category,
                        "sector":         fund.get("sector", ""),
                        "name":           fund.get("name", ""),
                        "market_cap_b":   round((fund.get("market_cap") or 0) / 1e9, 2),
                        "fcf_yield":      round(fund["fcf_yield"], 4) if fund.get("fcf_yield") is not None else "",
                        "gross_margin":   round(fund.get("gross_margin") or 0, 4),
                        "revenue_growth": round(fund.get("revenue_growth") or 0, 4),
                        "debt_equity":    fund.get("debt_equity") or "",
                        "dividend_yield": round(fund.get("dividend_yield") or 0, 4),
                        "pe":             fund.get("pe") or "",
                        "analyst_upside": fund.get("analyst_upside") or 0,
                        "price_1m":  outcomes["price_1m"]  if outcomes["price_1m"]  is not None else "",
                        "price_3m":  outcomes["price_3m"]  if outcomes["price_3m"]  is not None else "",
                        "price_6m":  outcomes["price_6m"]  if outcomes["price_6m"]  is not None else "",
                        "return_1m": outcomes["return_1m"] if outcomes["return_1m"] is not None else "",
                        "return_3m": outcomes["return_3m"] if outcomes["return_3m"] is not None else "",
                        "return_6m": outcomes["return_6m"] if outcomes["return_6m"] is not None else "",
                        "mfe_3m":    outcomes["mfe_3m"]    if outcomes["mfe_3m"]    is not None else "",
                        "mae_3m":    outcomes["mae_3m"]    if outcomes["mae_3m"]    is not None else "",
                        "outcome_label": outcomes["outcome_label"],
                        "source":        "historical_backtest",
                    }

                    stats["dip_rows"].append(dip_record)
                    stats["written"] += 1

                    logging.info(
                        f"[hist_backtest] \u2714 {symbol} {dip_date.date()} | "
                        f"score={score:.0f} | rsi={dip_record['rsi']} | "
                        f"dd={dip_record['drawdown_52w']:.1f}% | cat={category} | "
                        f"r3m={outcomes['return_3m']} | mfe={outcomes['mfe_3m']} | "
                        f"mae={outcomes['mae_3m']} | label={outcomes['outcome_label'] or 'pending'}"
                    )

                except Exception as e:
                    logging.warning(f"[hist_backtest] {symbol} {dip_date}: {e}")
                    stats["errors"] += 1

        except Exception as e:
            logging.warning(f"[hist_backtest] Falha no ticker {symbol}: {e}")
            stats["errors"] += 1

    logging.info(
        f"[hist_backtest] Concluído — total={stats['total_dips']} | "
        f"escritos={stats['written']} | ignorados={stats['skipped']} | "
        f"censurados={stats['censored']} | erros={stats['errors']}"
    )
    return stats


def build_historical_training_set(
    tickers: list[str],
    output_path: Path | None = None,
    min_score: float = 45.0,
    dry_run: bool = False,
) -> dict:
    """
    F2-C — Ponto de entrada público do pipeline histórico completo.

    Orquestra F2-A + F2-B + escrita CSV idempotente:
      1. Chama run_historical_backtest() para detectar dips e calcular
         todos os outcomes forward-looking (A+B).
      2. Carrega as chaves (symbol, date_iso) já presentes no CSV para
         evitar duplicados (idempotência total).
      3. Escreve apenas as linhas novas no CSV histórico.
      4. Reporta stats completas ao chamador.

    O CSV de saída (hist_backtest.csv) tem a mesma estrutura de colunas
    que alert_db.csv para todas as features de ML partilhadas. O campo
    'source="historical_backtest"' distingue as duas origens.

    Uso típico (Railway, via main.py ou one-shot):
      from config import WATCHLIST
      from backtest import build_historical_training_set
      stats = build_historical_training_set(tickers=WATCHLIST)
      print(stats)

    Argumentos:
      tickers     : lista de símbolos
      output_path : caminho do CSV (None = _HIST_DB_PATH auto)
      min_score   : score mínimo (default: 45)
      dry_run     : se True, não grava nada — devolve stats e dip_rows

    Retorna dict:
      total_dips : dips detectados antes do filtro score
      written    : dips que passaram score + foram adicionados ao CSV
      skipped    : dips abaixo de min_score
      censored   : dips recentes sem outcome_label (right-edge)
      duplicates : dips já existentes no CSV (ignorados)
      errors     : falhas de ticker/dip
      csv_path   : str — caminho do CSV escrito (ou None em dry_run)
      dip_rows   : list — todos os dip_records (incluindo duplicados)
    """
    csv_path = output_path or _HIST_DB_PATH

    # F2-A + F2-B
    stats = run_historical_backtest(
        tickers=tickers,
        output_path=csv_path,
        min_score=min_score,
        dry_run=dry_run,
    )

    dip_rows = stats.get("dip_rows", [])

    if dry_run:
        logging.info(f"[hist_csv] dry_run=True — {len(dip_rows)} linhas geradas, sem escrita.")
        stats["duplicates"] = 0
        stats["csv_path"]   = None
        return stats

    # F2-C: idempotência + escrita
    existing_keys = _load_existing_keys(csv_path)
    duplicates    = sum(
        1 for r in dip_rows
        if (r["symbol"], r["date_iso"]) in existing_keys
    )

    written_now = _write_hist_csv(dip_rows, csv_path, existing_keys)

    stats["written"]    = written_now  # substitui o contador interno
    stats["duplicates"] = duplicates
    stats["csv_path"]   = str(csv_path)

    logging.info(
        f"[hist_csv] Pipeline completo — "
        f"novas={written_now} | duplicados={duplicates} | "
        f"censurados={stats['censored']} | CSV={csv_path}"
    )
    return stats
