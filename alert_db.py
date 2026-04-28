"""
alert_db.py — Base de dados de alertas para treino futuro de ML.

Guarda uma "fotografia" de cada alerta gerado com todas as métricas
financeiras relevantes no momento do alerta.

Fase 1 (actual): só registar (fotografar).
Fase 2 (futuro): backtest.py actualiza MFE/MAE a 1, 3, 6 meses.
Fase 3 (futuro): treinar modelo sklearn/xgboost com os dados acumulados.

Formato: CSV em /data/alert_db.csv (Railway Volume) ou /tmp/alert_db.csv.
"""

import csv
import logging
from datetime import datetime
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
    "volume_ratio",   # volume / average_volume
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
    # Resultados futuros (preenchidos pelo backtest, inicialmente vazios)
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
    preenchidos futuramente pelo backtest.py.
    """
    _ensure_header()
    try:
        from sectors import get_sector_config
        sector     = fundamentals.get("sector", "")
        pe_fair    = get_sector_config(sector).get("pe_fair", 22)
        vol        = fundamentals.get("volume") or 0
        avg_vol    = fundamentals.get("average_volume") or 1
        vol_ratio  = round(vol / avg_vol, 2) if avg_vol else None

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
            "volume_ratio":     vol_ratio or "",
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
            # Resultado — a preencher futuramente
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
        for r in rows:
            cat = r.get("category", "?") or "?"
            vrd = r.get("verdict", "?") or "?"
            by_category[cat] = by_category.get(cat, 0) + 1
            by_verdict[vrd]  = by_verdict.get(vrd, 0) + 1
        last_5 = rows[-5:][::-1]  # os 5 mais recentes
        return {
            "total":       total,
            "by_category": by_category,
            "by_verdict":  by_verdict,
            "last_5":      last_5,
            "db_path":     str(_DB_PATH),
        }
    except Exception as e:
        logging.warning(f"[alert_db] get_db_stats: {e}")
        return {"total": 0, "error": str(e)}
