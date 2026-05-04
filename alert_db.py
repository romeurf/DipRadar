"""
alert_db.py — Base de dados de alertas para treino futuro de ML.

Guarda uma "fotografia" de cada alerta gerado com todas as métricas
financeiras relevantes no momento do alerta.

Fase 1 (actual): só registar (fotografar).
Fase 2 (actual): fill_db_outcomes() preenche MFE/MAE a 1, 3, 6 meses + return_60d.
Fase 3 (futuro): treinar modelo sklearn/xgboost com os dados acumulados.

Formato: CSV em /data/alert_db.csv (Railway Volume) ou /tmp/alert_db.csv.

Fonte de preços: Tiingo API (via tiingo_client.py).
  Mais fiável que yfinance — dados ajustados, sem rate limits agressivos.
  Requer TIINGO_API_KEY no ambiente.
"""

import csv
import logging
from datetime import datetime, timedelta, date
from pathlib import Path

# Persistência em /data/ se Railway Volume disponível, senão /tmp/
_DB_PATH = Path("/data/alert_db.csv") if Path("/data").exists() else Path("/tmp/alert_db.csv")

_FIELDS = [
    # Identificação
    "date_iso",            # 2025-04-28
    "time_iso",            # 14:32
    "symbol",
    "name",
    "sector",
    # Classificação
    "category",            # Hold Forever | Apartamento | Rotação
    "verdict",             # COMPRAR | MONITORIZAR | EVITAR
    "score",               # 0-100
    # Preço e mercado
    "price",
    "market_cap_b",        # em billions USD
    "drawdown_from_high",  # % abaixo do máximo de 52 semanas (negativo)
                           # NOTA: era drawdown_52w — renomeado para coincidir
                           # com o nome canónico do ml_predictor._FEATURE_COLS
    "change_day_pct",      # queda do dia que gerou o alerta
    # Técnicos
    "rsi",
    "volume_ratio",        # volume / average_volume
    # Fundamentais
    "pe",
    "pe_historical",
    "pe_fair",
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
    # Resultados futuros — preenchidos por fill_db_outcomes()
    "price_1m",
    "price_3m",
    "price_6m",
    "return_1m",
    "return_3m",
    "return_6m",
    # Target alinhado com o treino do XGB-v2 (max_return_60d)
    "return_60d",          # retorno no dia ~60 a contar do alerta
    "spy_return_60d",      # retorno SPY na mesma janela (para alpha_60d futuro)
    "mfe_3m",
    "mae_3m",
    "outcome_label",       # WIN_40 | WIN_20 | NEUTRAL | LOSS_15
]

# Maturidade mínima (dias) para preencher return_60d e rotular
_MIN_DAYS_60D = 62  # 60 dias de calendário + 2 dias de buffer para fecho de mercado


def _ensure_header() -> None:
    """Cria o ficheiro CSV com cabeçalho se não existir."""
    if not _DB_PATH.exists():
        try:
            _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _DB_PATH.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=_FIELDS,
                    quoting=csv.QUOTE_NONNUMERIC,
                )
                writer.writeheader()
            logging.info(f"[alert_db] Criado: {_DB_PATH}")
        except Exception as e:
            logging.warning(f"[alert_db] Erro ao criar header: {e}")


def _safe_volume_ratio(vol: float | None, avg_vol: float | None) -> float | str:
    """
    Calcula volume / average_volume de forma segura para ML.
    Retorna string vazia se qualquer valor for None, zero ou negativo.
    Threshold mínimo de avg_vol: 1000 acções.
    """
    if not vol or not avg_vol or avg_vol < 1000:
        return ""
    return round(vol / avg_vol, 2)


def _resolve_outcome_label(ref_return: float) -> str:
    """
    Converte um retorno percentual num label de outcome para treino ML.
    Thresholds:
      WIN_40  : >= +40%
      WIN_20  : >= +20%
      NEUTRAL : >= -15%
      LOSS_15 : < -15%
    """
    if ref_return >= 40:
        return "WIN_40"
    elif ref_return >= 20:
        return "WIN_20"
    elif ref_return >= -15:
        return "NEUTRAL"
    else:
        return "LOSS_15"


def log_alert_snapshot(
    symbol: str,
    fundamentals: dict,
    score: float,
    verdict: str,
    category: str,
    change_day_pct: float = 0.0,
    rsi_val: float | None = None,
    historical_pe: float | dict | None = None,
    spy_change: float | None = None,
    sector_etf_change: float | None = None,
) -> None:
    """
    Regista um alerta na base de dados ML.
    Campos de resultado (return_1m, MFE, return_60d, etc.) ficam vazios para
    serem preenchidos futuramente pelo fill_db_outcomes().

    historical_pe pode ser um dict (de get_historical_pe) ou um float/None.

    Nota: usa csv.QUOTE_NONNUMERIC para garantir que campos de texto com
    vírgulas (ex: nomes de empresas) ficam entre aspas e não corrompem o CSV.
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

        if isinstance(historical_pe, dict):
            _pe_hist_val = historical_pe.get("pe_hist_median")
        else:
            _pe_hist_val = historical_pe

        row = {
            "date_iso":           datetime.now().date().isoformat(),
            "time_iso":           datetime.now().strftime("%H:%M"),
            "symbol":             symbol,
            "name":               (fundamentals.get("name") or "")[:40],
            "sector":             sector,
            "category":           category,
            "verdict":            verdict,
            "score":              round(score, 1),
            "price":              fundamentals.get("price") or "",
            "market_cap_b":       round((fundamentals.get("market_cap") or 0) / 1e9, 2),
            "drawdown_from_high": fundamentals.get("drawdown_from_high") or "",
            "change_day_pct":     round(change_day_pct, 2),
            "rsi":                round(rsi_val, 1) if rsi_val is not None else "",
            "volume_ratio":       vol_ratio,
            "pe":                 fundamentals.get("pe") or "",
            "pe_historical":      round(_pe_hist_val, 1) if _pe_hist_val is not None else "",
            "pe_fair":            pe_fair,
            "fcf_yield":          round(fundamentals.get("fcf_yield") or 0, 4) if fundamentals.get("fcf_yield") is not None else "",
            "revenue_growth":     round(fundamentals.get("revenue_growth") or 0, 4),
            "gross_margin":       round(fundamentals.get("gross_margin") or 0, 4),
            "debt_equity":        fundamentals.get("debt_equity") or "",
            "dividend_yield":     round(fundamentals.get("dividend_yield") or 0, 4),
            "beta":               fundamentals.get("beta") or "",
            "analyst_upside":     round(fundamentals.get("analyst_upside") or 0, 1),
            "spy_change":         round(spy_change, 2) if spy_change is not None else "",
            "sector_etf_change":  round(sector_etf_change, 2) if sector_etf_change is not None else "",
            # Resultado — a preencher pelo fill_db_outcomes
            "price_1m": "", "price_3m": "", "price_6m": "",
            "return_1m": "", "return_3m": "", "return_6m": "",
            "return_60d": "",
            "spy_return_60d": "",
            "mfe_3m": "", "mae_3m": "", "outcome_label": "",
        }
        with _DB_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=_FIELDS,
                quoting=csv.QUOTE_NONNUMERIC,
            )
            writer.writerow(row)
        logging.info(f"[alert_db] Snapshot gravado: {symbol} | cat={category} | score={score:.0f}")
    except Exception as e:
        logging.warning(f"[alert_db] Erro ao gravar {symbol}: {e}")


def fill_db_outcomes() -> dict:
    """
    Fase 2 — preenche os campos de resultado para alertas maduros.

    Janelas:
      return_1m  / price_1m  : >= 30 dias
      return_3m  / price_3m  : >= 90 dias (+ MFE/MAE)
      return_6m  / price_6m  : >= 180 dias
      return_60d             : >= 62 dias (alinhado com max_return_60d do treino)
      spy_return_60d         : >= 62 dias (referência SPY para alpha futuro)

    TRAVA DE MATURIDADE: return_60d só é preenchido se
      data_actual >= alert_date + _MIN_DAYS_60D (62 dias).

    outcome_label usa return_60d como referência primária.
    Fallback: return_3m -> return_6m -> return_1m.

    Hold Forever nunca recebe label.
    Só actualiza linhas onde os campos ainda estão vazios.
    """
    from tiingo_client import get_ohlcv, get_price_at, get_mfe_mae
    from score import CATEGORY_APARTAMENTO, CATEGORY_HOLD_FOREVER

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
    sym_cache: dict = {}

    for i, row in enumerate(rows):
        needs_1m  = row.get("return_1m")   == ""
        needs_3m  = row.get("return_3m")   == ""
        needs_6m  = row.get("return_6m")   == ""
        needs_60d = row.get("return_60d")  == ""
        if not (needs_1m or needs_3m or needs_6m or needs_60d):
            skipped += 1
            continue

        date_iso       = row.get("date_iso", "")
        symbol         = row.get("symbol", "")
        price_at_alert = row.get("price", "")
        category       = row.get("category", "")

        if not date_iso or not symbol or not price_at_alert:
            skipped += 1
            continue

        try:
            alert_date  = date.fromisoformat(date_iso)
            price_entry = float(price_at_alert)
        except (ValueError, TypeError):
            skipped += 1
            continue

        days_elapsed = (today - alert_date).days

        if days_elapsed < 30:
            skipped += 1
            continue

        if symbol not in sym_cache:
            try:
                start   = alert_date - timedelta(days=1)
                end     = min(today, alert_date + timedelta(days=210))
                candles = get_ohlcv(symbol, start, end)
                sym_cache[symbol] = candles
            except EnvironmentError as e:
                logging.error(f"[fill_db] {e}")
                return {
                    "total": len(rows),
                    "updated": updated,
                    "skipped": skipped,
                    "errors": errors + 1,
                    "abort_reason": str(e),
                }
            except Exception as e:
                logging.warning(f"[fill_db] Tiingo {symbol}: {e}")
                sym_cache[symbol] = []
                errors += 1
                continue

        candles = sym_cache[symbol]
        if not candles:
            skipped += 1
            continue

        changed = False

        # ── T+1m ─────────────────────────────────────────────────────────────
        if needs_1m and days_elapsed >= 30:
            p1m = get_price_at(candles, alert_date + timedelta(days=30))
            if p1m is not None and price_entry > 0:
                row["price_1m"]  = round(p1m, 4)
                row["return_1m"] = round((p1m - price_entry) / price_entry * 100, 2)
                changed = True

        # ── T+60d ─────────────────────────────────────────────────────────────
        if needs_60d and days_elapsed >= _MIN_DAYS_60D:
            p60d = get_price_at(candles, alert_date + timedelta(days=60))
            if p60d is not None and price_entry > 0:
                row["return_60d"] = round((p60d - price_entry) / price_entry * 100, 2)
                changed = True
            if row.get("spy_return_60d") == "":
                try:
                    from tiingo_client import get_ohlcv as _get_ohlcv, get_price_at as _get_price_at
                    spy_candles = sym_cache.get("SPY")
                    if spy_candles is None:
                        spy_candles = _get_ohlcv(
                            "SPY",
                            alert_date - timedelta(days=1),
                            min(today, alert_date + timedelta(days=70)),
                        )
                        sym_cache["SPY"] = spy_candles
                    spy_entry = _get_price_at(spy_candles, alert_date)
                    spy_exit  = _get_price_at(spy_candles, alert_date + timedelta(days=60))
                    if spy_entry and spy_exit and spy_entry > 0:
                        row["spy_return_60d"] = round(
                            (spy_exit - spy_entry) / spy_entry * 100, 2
                        )
                except Exception as e:
                    logging.debug(f"[fill_db] SPY 60d para {symbol}: {e}")

        # ── T+3m ─────────────────────────────────────────────────────────────
        if needs_3m and days_elapsed >= 90:
            p3m = get_price_at(candles, alert_date + timedelta(days=91))
            if p3m is not None and price_entry > 0:
                row["price_3m"]  = round(p3m, 4)
                row["return_3m"] = round((p3m - price_entry) / price_entry * 100, 2)
                changed = True
                if row.get("mfe_3m") == "":
                    mfe, mae = get_mfe_mae(
                        candles,
                        after_date=alert_date,
                        price_entry=price_entry,
                        window_days=91,
                    )
                    if mfe is not None:
                        row["mfe_3m"] = mfe
                    if mae is not None:
                        row["mae_3m"] = mae

        # ── T+6m ─────────────────────────────────────────────────────────────
        if needs_6m and days_elapsed >= 180:
            p6m = get_price_at(candles, alert_date + timedelta(days=182))
            if p6m is not None and price_entry > 0:
                row["price_6m"]  = round(p6m, 4)
                row["return_6m"] = round((p6m - price_entry) / price_entry * 100, 2)
                changed = True

        # ── outcome_label ─────────────────────────────────────────────────────
        if changed and row.get("outcome_label") == "":
            if CATEGORY_HOLD_FOREVER in category:
                pass
            else:
                ref_return = None
                for field in ("return_60d", "return_3m", "return_6m", "return_1m"):
                    val = row.get(field)
                    if val not in ("", None):
                        try:
                            ref_return = float(val)
                            break
                        except ValueError:
                            pass
                if ref_return is not None and (
                    row.get("return_60d") not in ("", None)
                    or days_elapsed >= _MIN_DAYS_60D
                ):
                    row["outcome_label"] = _resolve_outcome_label(ref_return)

        if changed:
            rows[i] = row
            updated += 1

    if updated > 0:
        try:
            with _DB_PATH.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=_FIELDS,
                    quoting=csv.QUOTE_NONNUMERIC,
                )
                writer.writeheader()
                writer.writerows(rows)
            logging.info(
                f"[fill_db] CSV actualizado: {updated} linhas | "
                f"{skipped} ignoradas | {errors} erros"
            )
        except Exception as e:
            logging.error(f"[fill_db] Erro ao reescrever CSV: {e}")
            errors += 1

    return {
        "total":   len(rows),
        "updated": updated,
        "skipped": skipped,
        "errors":  errors,
    }


def get_db_stats() -> dict:
    """
    Estatísticas rápidas para Telegram /admin e setup_schedule jobs.
    """
    if not _DB_PATH.exists():
        return {"total": 0, "labeled": 0, "outcomes": {}, "first_date": None, "last_date": None, "db_path": str(_DB_PATH)}

    try:
        with _DB_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        logging.error(f"[get_db_stats] erro a ler CSV: {e}")
        return {"total": 0, "labeled": 0, "outcomes": {}, "error": str(e), "db_path": str(_DB_PATH)}

    total = len(rows)
    if total == 0:
        return {"total": 0, "labeled": 0, "outcomes": {}, "first_date": None, "last_date": None, "db_path": str(_DB_PATH)}

    outcomes: dict[str, int] = {}
    labeled = 0
    for row in rows:
        label = (row.get("outcome_label") or "").strip()
        if label:
            labeled += 1
            outcomes[label] = outcomes.get(label, 0) + 1

    dates = sorted([r.get("date_iso", "") for r in rows if r.get("date_iso")])
    by_category: dict[str, int] = {}
    for row in rows:
        cat = (row.get("category") or "Unknown").strip()
        by_category[cat] = by_category.get(cat, 0) + 1

    return {
        "total":       total,
        "labeled":     labeled,
        "outcomes":    outcomes,
        "first_date":  dates[0] if dates else None,
        "last_date":   dates[-1] if dates else None,
        "by_category": by_category,
        "db_path":     str(_DB_PATH),
    }
