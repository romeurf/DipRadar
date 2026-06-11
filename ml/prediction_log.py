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
        from ml.ml_features import FEATURE_COLUMNS
        feats = list(FEATURE_COLUMNS)
    except Exception:
        feats = []

    base = [
        "ts",
        "symbol",
        "snapshot_date",
        "win_prob",
        "score_raw",       # alpha_90d predicted (log-return excess vs SPY)
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
        "outcome_label",   # preenchido depois (WIN_STRONG / WIN / NEUTRAL / LOSS)
        "return_3m",
        "return_6m",
        "outcome_resolved_at",
    ]
    _HEADER_FIELDS_CACHE = base + feats + outcome
    return _HEADER_FIELDS_CACHE


def _validate_or_reset() -> None:
    """Verifica se o ficheiro tem o schema correcto. Se não, apaga e recria limpo.

    Ficheiros com schema antigo ou corrompidos são removidos.
    Diagnósticos de produção recomeçam do zero — dados inválidos não têm valor.
    """
    if not PREDICTIONS_PATH.exists():
        return
    try:
        with PREDICTIONS_PATH.open("r", encoding="utf-8", newline="") as f:
            header = next(csv.reader(f), [])
        expected = _header_fields()
        if header == expected:
            return  # schema correcto — nada a fazer
        log.warning(
            f"[prediction_log] Schema desactualizado "
            f"({len(header)} campos vs {len(expected)} esperados) — a apagar e recriar."
        )
        PREDICTIONS_PATH.unlink()
    except Exception as e:
        log.warning(f"[prediction_log] Falha ao validar schema: {e} — a apagar.")
        try:
            PREDICTIONS_PATH.unlink(missing_ok=True)
        except Exception:
            pass


def _ensure_header() -> None:
    """Valida schema e cria o ficheiro com cabeçalho se não existir."""
    _validate_or_reset()
    if PREDICTIONS_PATH.exists():
        return
    try:
        PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PREDICTIONS_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_header_fields())
            writer.writeheader()
        log.info(f"[prediction_log] Criado: {PREDICTIONS_PATH}")
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
            from ml.ml_features import add_derived_features
            enriched = dict(features) if features else {}
            add_derived_features(enriched)
        except Exception:
            enriched = dict(features) if features else {}

        row: dict = {f: "" for f in _header_fields()}
        row["ts"]            = datetime.utcnow().isoformat(timespec="seconds")
        row["symbol"]        = symbol
        row["snapshot_date"] = snapshot_date or datetime.utcnow().date().isoformat()
        row["win_prob"]      = getattr(result, "win_prob", None)
        row["score_raw"]     = getattr(result, "score_raw", None)
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


def get_recent_predictions_by_ticker(
    tickers: list[str],
    lookback_days: int = 7,
) -> dict[str, dict]:
    """Return the most recent prediction per ticker from the last N days.

    Output: {ticker: {win_prob, score_raw, ts}} for tickers found.
    Missing tickers are simply not in the result dict.

    Used by /cash to fetch per-DIP-alert alpha estimate instead of
    hardcoding a default. score_raw is the model's raw alpha_90d
    prediction; win_prob is the calibrated probability.
    """
    if not PREDICTIONS_PATH.exists() or not tickers:
        return {}
    try:
        from datetime import date, timedelta
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
        wanted = {t.upper() for t in tickers}
        latest: dict[str, dict] = {}
        with PREDICTIONS_PATH.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sym = (row.get("symbol") or "").upper()
                if sym not in wanted:
                    continue
                ts = row.get("ts") or ""
                snap = row.get("snapshot_date") or ""
                if snap and snap < cutoff:
                    continue
                # Track latest by ts per symbol
                if sym not in latest or ts > latest[sym].get("ts", ""):
                    try:
                        wp = float(row.get("win_prob") or "")
                    except (TypeError, ValueError):
                        wp = None
                    try:
                        sr = float(row.get("score_raw") or "")
                    except (TypeError, ValueError):
                        sr = None
                    latest[sym] = {
                        "win_prob":  wp,
                        "score_raw": sr,
                        "ts":        ts,
                        "snapshot_date": snap,
                    }
        return latest
    except Exception as e:
        log.debug(f"[prediction_log] get_recent_predictions failed: {e}")
        return {}


def compute_ml_accuracy() -> dict:
    """Mede a precisão real do ML comparando win_prob com outcomes.

    Para cada predição com outcome resolvido:
    - win_prob > 0.55 E retorno > 0  → TP (True Positive)
    - win_prob > 0.55 E retorno <= 0 → FP (False Positive)
    - win_prob <= 0.55 E retorno > 0 → FN (False Negative)
    - win_prob <= 0.55 E retorno <= 0 → TN (True Negative)

    Retorna precision, recall, accuracy e Brier score live.
    Este loop de feedback fecha o ciclo alerta→resultado→modelo.
    """
    if not PREDICTIONS_PATH.exists():
        return {"skipped": True, "reason": "predictions log nao encontrado"}
    try:
        import pandas as pd
        df = pd.read_csv(PREDICTIONS_PATH)
        required = ["win_prob", "outcome_label"]
        if not all(c in df.columns for c in required):
            return {"skipped": True, "reason": "colunas em falta"}

        # Só linhas com outcome resolvido
        resolved = df[
            df["outcome_label"].notna() &
            (df["outcome_label"].astype(str).str.strip() != "")
        ].copy()
        if len(resolved) < 10:
            return {"skipped": True, "reason": f"apenas {len(resolved)} outcomes resolvidos (min 10)"}

        _WIN_LABELS = {"WIN_STRONG", "WIN"}
        resolved["actual_win"] = resolved["outcome_label"].isin(_WIN_LABELS).astype(int)
        resolved["predicted_win"] = (resolved["win_prob"].astype(float) > 0.55).astype(int)

        tp = int(((resolved["predicted_win"] == 1) & (resolved["actual_win"] == 1)).sum())
        fp = int(((resolved["predicted_win"] == 1) & (resolved["actual_win"] == 0)).sum())
        fn = int(((resolved["predicted_win"] == 0) & (resolved["actual_win"] == 1)).sum())
        tn = int(((resolved["predicted_win"] == 0) & (resolved["actual_win"] == 0)).sum())

        precision  = tp / max(1, tp + fp)
        recall     = tp / max(1, tp + fn)
        accuracy   = (tp + tn) / max(1, len(resolved))
        f1         = 2 * precision * recall / max(1e-6, precision + recall)

        # Brier score live (MSE entre win_prob e actual_win)
        probs   = resolved["win_prob"].astype(float).clip(0, 1)
        actuals = resolved["actual_win"].astype(float)
        brier   = float(((probs - actuals) ** 2).mean())

        win_rate = float(resolved["actual_win"].mean())

        return {
            "n_resolved":   len(resolved),
            "precision":    round(precision, 3),
            "recall":       round(recall, 3),
            "accuracy":     round(accuracy, 3),
            "f1":           round(f1, 3),
            "brier_live":   round(brier, 4),
            "win_rate_actual": round(win_rate, 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
    except Exception as e:
        return {"skipped": True, "reason": str(e)}


def compute_win_prob_drift(
    window_days: int = 30,
    baseline_win_prob: float | None = None,
) -> dict:
    """Calcula drift do win_prob nas últimas `window_days` vs baseline de treino.

    Retorna dict com:
      - recent_mean: média das últimas window_days previsões
      - baseline_mean: baseline do treino (do ml_report.json ou parâmetro)
      - delta: diferença (recent - baseline)
      - n_recent: número de previsões no período
      - drift_flag: True se |delta| > 0.10 (10pp de shift → modelo pode ter degradado)
    """
    if not PREDICTIONS_PATH.exists():
        return {"skipped": True, "reason": "predictions log não encontrado"}
    try:
        import pandas as pd
        from datetime import timedelta

        df = pd.read_csv(PREDICTIONS_PATH, on_bad_lines="skip")
        if "win_prob" not in df.columns or "ts" not in df.columns or df.empty:
            return {"skipped": True, "reason": "colunas win_prob/ts ausentes"}

        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        cutoff = pd.Timestamp.utcnow().replace(tzinfo=None) - pd.Timedelta(days=window_days)
        recent = df[df["ts"] >= cutoff]["win_prob"].dropna().astype(float)

        if len(recent) < 5:
            return {"skipped": True, "reason": f"apenas {len(recent)} previsões no período"}

        recent_mean = float(recent.mean())

        # Baseline do ml_report.json se não fornecido
        if baseline_win_prob is None:
            try:
                import json
                from pathlib import Path
                _data_dir = Path("/data") if Path("/data").exists() else Path("/tmp")
                # Phase 7 moved this module to /app/ml/, so Path(__file__).parent
                # is now /app/ml — walk up one to /app for the repo path.
                _repo_root = Path(__file__).parent.parent
                for rp in [_data_dir / "ml_report.json",
                           _repo_root / "ml_training" / "ml_report.json",
                           Path(__file__).parent / "ml_training" / "ml_report.json"]:
                    if rp.exists():
                        rdata = json.loads(rp.read_text())
                        baseline_win_prob = float(rdata.get("metrics", {}).get("win_rate_alpha", 0.50))
                        break
            except Exception:
                pass
            if baseline_win_prob is None:
                baseline_win_prob = 0.50  # fallback neutro

        delta = recent_mean - baseline_win_prob
        drift_flag = abs(delta) > 0.10

        return {
            "recent_mean":     round(recent_mean, 4),
            "baseline_mean":   round(baseline_win_prob, 4),
            "delta":           round(delta, 4),
            "n_recent":        int(len(recent)),
            "window_days":     window_days,
            "drift_flag":      drift_flag,
        }
    except Exception as e:
        log.warning(f"[prediction_log] compute_win_prob_drift falhou: {e}")
        return {"skipped": True, "reason": str(e)}


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
