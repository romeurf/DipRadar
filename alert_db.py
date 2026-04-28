"""
alert_db.py — Base de dados de alertas para treino futuro de ML.

Guarda uma "fotografia" de cada alerta gerado com todas as métricas
financeiras relevantes no momento do alerta.

Fase 1 (actual): só registar (fotografar).
Fase 2 (actual): fill_db_outcomes() preenche MFE/MAE a 1, 3, 6 meses.
Fase 3 (futuro): treinar modelo sklearn/xgboost com os dados acumulados.

Formato: CSV em /data/alert_db.csv (Railway Volume) ou /tmp/alert_db.csv.
"""

import csv
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

# Persistência em /data/ se Railway Volume disponível, senão /tmp/
_DB_PATH = Path("/data/alert_db.csv") if Path("/data").exists() else Path("/tmp/alert_db.csv")

_FIELDS = [
    # Identificação
    "date_iso",       # 2025-04-28
    "time_iso",       # 14:32
    "symbol",
    "name",
    "sector",
    # Classificação
    "category",       # Hold Forever | Apartamento | Rotação
    "verdict",        # COMPRAR | MONITORIZAR | EVITAR
    "score",          # 0-100
    # Preço e mercado
    "price",
    "market_cap_b",   # em billions USD
    "drawdown_52w",   # % abaixo do máximo de 52 semanas (negativo)
    "change_day_pct", # queda do dia que gerou o alerta
    # Técnicos
    "rsi",
    "volume_ratio",   # volume / average_volume (vazio se dados insuficientes)
    # Fundamentais
    "pe",
    "pe_historical",  # P/E histórico de 5 anos
    "pe_fair",        # P/E justo do sector
    "fcf_yield",
    "revenue_growth",
    "gross_margin",
    "debt_equity",
    "dividend_yield",
    "beta",
    "analyst_upside",
    # Contexto macro
    "spy_change",
    "sector_etf_change",
    # Resultados futuros (preenchidos pelo fill_db_outcomes, inicialmente vazios)
    "price_1m",       # preço 1 mês depois
    "price_3m",
    "price_6m",
    "return_1m",      # % retorno vs preço de alerta
    "return_3m",
    "return_6m",
    "mfe_3m",         # Maximum Favorable Excursion em 3 meses
    "mae_3m",         # Maximum Adverse Excursion em 3 meses
    "outcome_label",  # WIN_40 | WIN_20 | NEUTRAL | LOSS_15
]


def _ensure_header() -> None:
    """Cria o ficheiro CSV com cabeçalho se não existir."""
    if not _DB_PATH.exists():
        try:
            _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _DB_PATH.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_FIELDS)
                writer.writeheader()
            logging.info(f"[alert_db] Criado: {_DB_PATH}")
        except Exception as e:
            logging.warning(f"[alert_db] Erro ao criar header: {e}")


def _safe_volume_ratio(vol: float | None, avg_vol: float | None) -> float | str:
    """
    Calcula volume / average_volume de forma segura para ML.
    Retorna string vazia se qualquer valor for None, zero ou negativo
    — um ratio inventado (ex: vol/1) seria ruído para o modelo.
    Threshold mínimo de avg_vol: 1000 acções (exclui ETFs ilíquidos com
    average_volume = 0 reportado pelo Yahoo Finance).
    """
    if not vol or not avg_vol or avg_vol < 1000:
        return ""
    return round(vol / avg_vol, 2)


def log_alert_snapshot(
    symbol: str,
    fundamentals: dict,
    score: float,
    verdict: str,
    category: str,
    change_day_pct: float = 0.0,
    rsi_val: float | None = None,
    historical_pe: float | None = None,
    spy_change: float | None = None,
    sector_etf_change: float | None = None,
) -> None:
    """
    Regista um alerta na base de dados ML.
    Campos de resultado (return_1m, MFE, etc.) ficam vazios para serem
    preenchidos futuramente pelo fill_db_outcomes().
    """
    _ensure_header()
    try:
        from sectors import get_sector_config
        sector    = fundamentals.get("sector", "")
        pe_fair   = get_sector_config(sector).get("pe_fair", 22)
        vol_ratio = _safe_volume_ratio(
            fundamentals.get("volume"),
            fundamentals.get("average_volume"),
        )

        row = {
            "date_iso":         datetime.now().date().isoformat(),
            "time_iso":         datetime.now().strftime("%H:%M"),
            "symbol":           symbol,
            "name":             (fundamentals.get("name") or "")[:40],
            "sector":           sector,
            "category":         category,
            "verdict":          verdict,
            "score":            round(score, 1),
            "price":            fundamentals.get("price") or "",
            "market_cap_b":     round((fundamentals.get("market_cap") or 0) / 1e9, 2),
            "drawdown_52w":     fundamentals.get("drawdown_from_high") or "",
            "change_day_pct":   round(change_day_pct, 2),
            "rsi":              round(rsi_val, 1) if rsi_val is not None else "",
            "volume_ratio":     vol_ratio,
            "pe":               fundamentals.get("pe") or "",
            "pe_historical":    round(historical_pe, 1) if historical_pe is not None else "",
            "pe_fair":          pe_fair,
            "fcf_yield":        round(fundamentals.get("fcf_yield") or 0, 4) if fundamentals.get("fcf_yield") is not None else "",
            "revenue_growth":   round(fundamentals.get("revenue_growth") or 0, 4),
            "gross_margin":     round(fundamentals.get("gross_margin") or 0, 4),
            "debt_equity":      fundamentals.get("debt_equity") or "",
            "dividend_yield":   round(fundamentals.get("dividend_yield") or 0, 4),
            "beta":             fundamentals.get("beta") or "",
            "analyst_upside":   round(fundamentals.get("analyst_upside") or 0, 1),
            "spy_change":       round(spy_change, 2) if spy_change is not None else "",
            "sector_etf_change": round(sector_etf_change, 2) if sector_etf_change is not None else "",
            # Resultado — a preencher pelo fill_db_outcomes
            "price_1m": "", "price_3m": "", "price_6m": "",
            "return_1m": "", "return_3m": "", "return_6m": "",
            "mfe_3m": "", "mae_3m": "", "outcome_label": "",
        }
        with _DB_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            writer.writerow(row)
        logging.info(f"[alert_db] Snapshot gravado: {symbol} | cat={category} | score={score:.0f}")
    except Exception as e:
        logging.warning(f"[alert_db] Erro ao gravar {symbol}: {e}")


def fill_db_outcomes() -> dict:
    """
    Fase 2 — preenche os campos de resultado (MFE/MAE/returns) para alertas
    com pelo menos 30 dias de histórico disponível.

    Lógica por janela:
      - return_1m  / price_1m  : preenchido se alerta tem >= 30 dias
      - return_3m  / price_3m  : preenchido se alerta tem >= 90 dias
      - return_6m  / price_6m  : preenchido se alerta tem >= 180 dias
      - mfe_3m / mae_3m        : preenchido se alerta tem >= 90 dias
        MFE = retorno máximo intraday em qualquer dia dos 90 dias
        MAE = drawdown máximo intraday em qualquer dia dos 90 dias
      - outcome_label: WIN_40 | WIN_20 | NEUTRAL | LOSS_15
        (baseado em return_3m, ou return_6m se disponível)

    Só actualiza linhas onde os campos ainda estão vazios.
    Retorna dict com stats do run para logging/Telegram.
    """
    import yfinance as yf

    if not _DB_PATH.exists():
        logging.info("[fill_db] Ficheiro não existe ainda.")
        return {"total": 0, "updated": 0, "skipped": 0, "errors": 0}

    try:
        rows = []
        with _DB_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        logging.error(f"[fill_db] Erro a ler CSV: {e}")
        return {"total": 0, "updated": 0, "skipped": 0, "errors": 1}

    today      = datetime.now().date()
    updated    = 0
    skipped    = 0
    errors     = 0
    sym_cache: dict = {}  # cache de histórico por símbolo — evita chamadas duplicadas

    for i, row in enumerate(rows):
        # Só processar linhas que ainda têm campos em falta
        needs_1m = row.get("return_1m") == ""
        needs_3m = row.get("return_3m") == ""
        needs_6m = row.get("return_6m") == ""
        if not (needs_1m or needs_3m or needs_6m):
            skipped += 1
            continue

        date_iso = row.get("date_iso", "")
        symbol   = row.get("symbol", "")
        price_at_alert = row.get("price", "")

        if not date_iso or not symbol or not price_at_alert:
            skipped += 1
            continue

        try:
            alert_date  = datetime.fromisoformat(date_iso).date()
            price_entry = float(price_at_alert)
        except (ValueError, TypeError):
            skipped += 1
            continue

        days_elapsed = (today - alert_date).days

        # Nada a fazer se ainda não passaram 30 dias
        if days_elapsed < 30:
            skipped += 1
            continue

        # Buscar histórico (com cache por símbolo)
        if symbol not in sym_cache:
            try:
                # Pede 7 meses para cobrir T+6m + margem
                start = alert_date - timedelta(days=1)
                end   = min(today, alert_date + timedelta(days=210))
                hist  = yf.Ticker(symbol).history(start=start, end=end, interval="1d")
                sym_cache[symbol] = hist
                time.sleep(1)  # respeitar rate limit do Yahoo Finance
            except Exception as e:
                logging.warning(f"[fill_db] yfinance {symbol}: {e}")
                sym_cache[symbol] = None
                errors += 1
                continue

        hist = sym_cache[symbol]
        if hist is None or hist.empty:
            skipped += 1
            continue

        # Filtrar apenas candles APÓS a data do alerta
        try:
            hist_after = hist[hist.index.date > alert_date]
        except Exception:
            skipped += 1
            continue

        if hist_after.empty:
            skipped += 1
            continue

        changed = False

        def _get_price_at(target_date):
            """
            Preço de fecho mais próximo de target_date.
            Tenta de -3 a +5 dias úteis para cobrir feriados e fins-de-semana.
            """
            for delta in range(-3, 6):
                check = target_date + timedelta(days=delta)
                matches = hist_after[hist_after.index.date == check]
                if not matches.empty:
                    return float(matches["Close"].iloc[0])
            return None

        # ── T+1m ──────────────────────────────────────────────────────────
        if needs_1m and days_elapsed >= 30:
            p1m = _get_price_at(alert_date + timedelta(days=30))
            if p1m is not None and price_entry > 0:
                r1m = (p1m - price_entry) / price_entry * 100
                row["price_1m"]  = round(p1m, 4)
                row["return_1m"] = round(r1m, 2)
                changed = True

        # ── T+3m ──────────────────────────────────────────────────────────
        if needs_3m and days_elapsed >= 90:
            p3m = _get_price_at(alert_date + timedelta(days=91))
            if p3m is not None and price_entry > 0:
                r3m = (p3m - price_entry) / price_entry * 100
                row["price_3m"]  = round(p3m, 4)
                row["return_3m"] = round(r3m, 2)
                changed = True

                # ── MFE / MAE nos primeiros 90 dias ───────────────────────
                window_90 = hist_after[hist_after.index.date <= alert_date + timedelta(days=91)]
                if not window_90.empty:
                    highs  = window_90["High"]
                    lows   = window_90["Low"]
                    mfe    = (highs.max() - price_entry) / price_entry * 100
                    mae    = (lows.min() - price_entry) / price_entry * 100
                    row["mfe_3m"] = round(mfe, 2)
                    row["mae_3m"] = round(mae, 2)

        # ── T+6m ──────────────────────────────────────────────────────────
        if needs_6m and days_elapsed >= 180:
            p6m = _get_price_at(alert_date + timedelta(days=182))
            if p6m is not None and price_entry > 0:
                r6m = (p6m - price_entry) / price_entry * 100
                row["price_6m"]  = round(p6m, 4)
                row["return_6m"] = round(r6m, 2)
                changed = True

        # ── outcome_label ─────────────────────────────────────────────────
        if changed and row.get("outcome_label") == "":
            # Usar o retorno mais longo disponível
            ref_return = None
            for field in ("return_6m", "return_3m", "return_1m"):
                val = row.get(field)
                if val not in ("", None):
                    try:
                        ref_return = float(val)
                        break
                    except ValueError:
                        pass

            if ref_return is not None:
                if ref_return >= 40:
                    row["outcome_label"] = "WIN_40"
                elif ref_return >= 20:
                    row["outcome_label"] = "WIN_20"
                elif ref_return >= -15:
                    row["outcome_label"] = "NEUTRAL"
                else:
                    row["outcome_label"] = "LOSS_15"

        if changed:
            rows[i] = row
            updated += 1

    # Reescrever o CSV completo com as actualizações
    if updated > 0:
        try:
            with _DB_PATH.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            logging.info(f"[fill_db] CSV actualizado: {updated} linhas novas | {skipped} ignoradas | {errors} erros")
        except Exception as e:
            logging.error(f"[fill_db] Erro a escrever CSV: {e}")
            errors += 1
    else:
        logging.info(f"[fill_db] Nada a actualizar ({skipped} linhas ignoradas)")

    return {
        "total":   len(rows),
        "updated": updated,
        "skipped": skipped,
        "errors":  errors,
    }


def get_db_stats() -> dict:
    """
    Devolve estatísticas da base de dados:
    total de registos, por categoria, por verdict, últimos 5 alertas.
    """
    if not _DB_PATH.exists():
        return {"total": 0, "by_category": {}, "by_verdict": {}, "last_5": []}
    try:
        rows = []
        with _DB_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows   = list(reader)
        total       = len(rows)
        by_category = {}
        by_verdict  = {}
        outcomes    = {}
        labeled     = 0
        for r in rows:
            cat = r.get("category", "?") or "?"
            vrd = r.get("verdict", "?") or "?"
            lbl = r.get("outcome_label", "") or ""
            by_category[cat] = by_category.get(cat, 0) + 1
            by_verdict[vrd]  = by_verdict.get(vrd, 0) + 1
            if lbl:
                outcomes[lbl] = outcomes.get(lbl, 0) + 1
                labeled += 1
        last_5 = rows[-5:][::-1]  # os 5 mais recentes
        return {
            "total":       total,
            "by_category": by_category,
            "by_verdict":  by_verdict,
            "outcomes":    outcomes,
            "labeled":     labeled,
            "last_5":      last_5,
            "db_path":     str(_DB_PATH),
        }
    except Exception as e:
        logging.warning(f"[alert_db] get_db_stats: {e}")
        return {"total": 0, "error": str(e)}
