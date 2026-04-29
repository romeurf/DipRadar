"""
alert_clustering.py — Anti-clustering de alertas (janela de 20 dias).

Garante que o bot não envia dois alertas para o mesmo ticker
na mesma janela de 20 dias de calendário.

Persistência: /data/alert_clustering.csv (Railway Volume) ou /tmp/
"""

import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path

_CLUSTER_PATH = (
    Path("/data/alert_clustering.csv")
    if Path("/data").exists()
    else Path("/tmp/alert_clustering.csv")
)

_WINDOW_DAYS = 20
_FIELDS      = ["symbol", "date_iso", "ml_class", "score"]


def _ensure_header() -> None:
    if not _CLUSTER_PATH.exists():
        try:
            _CLUSTER_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _CLUSTER_PATH.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=_FIELDS).writeheader()
        except Exception as e:
            logging.warning(f"[clustering] Erro ao criar header: {e}")


def was_alerted_recently(symbol: str, window_days: int = _WINDOW_DAYS) -> bool:
    """
    Devolve True se este ticker já gerou alerta nos últimos `window_days` dias.
    """
    _ensure_header()
    cutoff = datetime.now().date() - timedelta(days=window_days)
    try:
        with _CLUSTER_PATH.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("symbol") == symbol:
                    try:
                        row_date = datetime.fromisoformat(row["date_iso"]).date()
                        if row_date >= cutoff:
                            return True
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    except Exception as e:
        logging.warning(f"[clustering] Erro ao ler {symbol}: {e}")
    return False


def register_alert(symbol: str, ml_class: str = "", score: float = 0.0) -> None:
    """
    Regista um alerta enviado para o ticker hoje.
    """
    _ensure_header()
    try:
        with _CLUSTER_PATH.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_FIELDS).writerow({
                "symbol":   symbol,
                "date_iso": datetime.now().date().isoformat(),
                "ml_class": ml_class,
                "score":    round(score, 1),
            })
        logging.info(f"[clustering] Registado: {symbol} | {ml_class} | score={score:.0f}")
    except Exception as e:
        logging.warning(f"[clustering] Erro ao registar {symbol}: {e}")


def purge_old_entries(window_days: int = _WINDOW_DAYS) -> int:
    """
    Remove entradas mais antigas que `window_days` dias.
    Chamado no reset diário (00:01). Devolve nº de linhas removidas.
    """
    if not _CLUSTER_PATH.exists():
        return 0
    cutoff = datetime.now().date() - timedelta(days=window_days)
    try:
        with _CLUSTER_PATH.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        fresh = []
        stale = 0
        for row in rows:
            try:
                if datetime.fromisoformat(row["date_iso"]).date() >= cutoff:
                    fresh.append(row)
                else:
                    stale += 1
            except ValueError:
                fresh.append(row)  # mantém linhas com data inválida por segurança
        if stale > 0:
            with _CLUSTER_PATH.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=_FIELDS)
                w.writeheader()
                w.writerows(fresh)
            logging.info(f"[clustering] Purge: {stale} entradas removidas")
        return stale
    except Exception as e:
        logging.warning(f"[clustering] Erro no purge: {e}")
        return 0
