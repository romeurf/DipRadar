"""
prediction_log.py — Append-only log de previsões do `ml_predictor.ml_score()`.

Cada chamada em produção é registada para podermos:
  - reconstruir o desempenho real do modelo em produção (precision/recall live)
  - detectar drift (probabilidades médias mensais vs baseline do treino)
  - debug de alertas falsos / falsos negativos

Storage:
  /data/ml_predictions.csv   (Railway Volume) — append-only, leitura barata
  /tmp/ml_predictions.csv    (fallback dev)

Schema (compatível com pandas → parquet):
  ts, symbol, snapshot_date, win_prob, prob_price, prob_fund, win40_prob,
  threshold, label, vix_regime, low_coverage, coverage,
  + 21 features do FEATURE_COLUMNS
  + outcome_label (preenchido depois pelo job de outcomes — começa vazio)
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DATA_DIR = Path("/data") if Path("/data").exists() else Path("/tmp")
PREDICTIONS_PATH = _DATA_DIR / "ml_predictions.csv"

# Lazy import dentro das funções para evitar circular dependencies
_HEADER_FIELDS_CACHE: list[str] | None = None


def _header_fields() -> list[str]:
    """FEATURE_COLUMNS + metadata. Cached para evitar import repetido."""
    global _HEADER_FIELDS_CACHE
    if _HEADER_FIELDS_CACHE is not None:
        return _HEADER_FIELDS_CACHE

    try:
        from ml_features import FEATURE_COLUMNS
        feats = list(FEATURE_COLUMNS)
    except Exception:
        feats = []

    base = [
        "ts",
        "symbol",
        "snapshot_date",
        "win_prob",
        "prob_price",
        "prob_fund",
        "win40_prob",
        "threshold",
        "label",
        "vix_regime",
        "low_coverage",
        "coverage",
    ]
    outcome = [
        "outcome_label",   # preenchido depois (WIN_40 / WIN_20 / NEUTRAL / LOSS_15)
        "return_3m",
        "return_6m",
        "outcome_resolved_at",
    ]
    _HEADER_FIELDS_CACHE = base + feats + outcome
    return _HEADER_FIELDS_CACHE


def _ensure_header() -> None:
    """Cria o ficheiro com cabeçalho se não existir."""
    if PREDICTIONS_PATH.exists():
        return
    try:
        PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PREDICTIONS_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_header_fields())
            writer.writeheader()
    except Exception as e:
        log.warning(f"[prediction_log] init falhou: {e}")


def log_prediction(
    symbol: str,
    features: dict,
    result: Any,
    snapshot_date: str | None = None,
) -> None:
    """
    Append uma linha ao log de previsões. Não levanta excepções —
    failure loga mas nunca afecta o caller.

    Args:
        symbol:        ticker (ex.: "AAPL")
        features:      dict de features pré-derived (mesmo que passas a ml_score)
        result:        MLResult devolvido por ml_score()
        snapshot_date: ISO YYYY-MM-DD (default: hoje em UTC)
    """
    try:
        _ensure_header()

        # Compute derived features so logged row reflects exactly what the model saw
        try:
            from ml_features import add_derived_features
            enriched = dict(features) if features else {}
            add_derived_features(enriched)
        except Exception:
            enriched = dict(features) if features else {}

        row: dict = {f: "" for f in _header_fields()}
        row["ts"]            = datetime.utcnow().isoformat(timespec="seconds")
        row["symbol"]        = symbol
        row["snapshot_date"] = snapshot_date or datetime.utcnow().date().isoformat()
        row["win_prob"]      = getattr(result, "win_prob", None)
        row["prob_price"]    = getattr(result, "prob_price", None)
        row["prob_fund"]     = getattr(result, "prob_fund", None)
        row["win40_prob"]    = getattr(result, "win40_prob", None)
        row["threshold"]     = getattr(result, "threshold", None)
        row["label"]         = getattr(result, "label", None)
        row["vix_regime"]    = getattr(result, "vix_regime", None)
        row["low_coverage"]  = getattr(result, "low_coverage", None)
        row["coverage"]      = getattr(result, "coverage", None)

        for k, v in enriched.items():
            if k in row:
                row[k] = v

        with PREDICTIONS_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_header_fields())
            writer.writerow(row)

    except Exception as e:
        log.debug(f"[prediction_log] {symbol}: log falhou ({e}) — ignorado.")


def get_log_stats() -> dict:
    """Estatísticas rápidas para /admin (Telegram)."""
    if not PREDICTIONS_PATH.exists():
        return {"total": 0, "labeled": 0}
    try:
        import pandas as pd
        df = pd.read_csv(PREDICTIONS_PATH)
        total = len(df)
        if "outcome_label" in df.columns:
            mask = df["outcome_label"].notna() & (df["outcome_label"].astype(str) != "")
            labeled = int(mask.sum())
        else:
            labeled = 0
        last_ts = str(df["ts"].max()) if "ts" in df.columns and total else None
        return {"total": total, "labeled": labeled, "last_ts": last_ts}
    except Exception as e:
        log.warning(f"[prediction_log] stats falhou: {e}")
        return {"total": -1, "labeled": -1, "error": str(e)}
