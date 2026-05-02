"""
ml_walk_forward.py — Walk-forward backtest mensal/semestral 2022-2025.

Quê faz:
  Avalia o modelo Stage 1 (camada B) num regime de "produção": divide o
  dataset em janelas mensais e, para cada janela, ou:
    a) Usa o modelo TREINADO uma só vez sobre dados anteriores à janela
       (modo "fixed") — barato, valida estabilidade do threshold;
    b) RE-TREINA o modelo em janela deslizante (modo "rolling") — caro,
       valida que o pipeline aprende correctamente em cada época.

Output:
  /tmp/diprader_v2/walk_forward.json com métricas por janela:
  win_rate, precision@thr, recall@thr, n_samples, n_positives, n_alerts.

Uso:
  python ml_walk_forward.py --parquet ml_training_merged.parquet \\
      --model /tmp/diprader_v2/dip_model_stage1.pkl \\
      --output /tmp/diprader_v2/walk_forward.json \\
      [--mode fixed|rolling] [--start 2022-01] [--end 2026-01]
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from ml_features import add_derived_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_WIN_LABELS = {"WIN_40", "WIN_20"}


def _load_clean(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["alert_date"] = pd.to_datetime(df["alert_date"])
    # Keep only the cleaning that train_model also applies
    df = df.drop_duplicates(subset=["symbol", "alert_date"]).copy()
    df = df[df["alert_date"].dt.year >= 2014].copy()
    df = df[df["return_3m"].abs() <= 200].copy()
    df = df.sort_values("alert_date").reset_index(drop=True)
    df["_y"] = df["outcome_label"].isin(_WIN_LABELS).astype(int)
    return df


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply add_derived_features per row to materialise engineered features."""
    rows: list[dict] = df.to_dict("records")
    for r in rows:
        add_derived_features(r)  # type: ignore[arg-type]
    return pd.DataFrame(rows)


def _build_X(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    X = pd.DataFrame({c: pd.to_numeric(df.get(c, np.nan), errors="coerce")
                       for c in columns})
    return X.to_numpy(dtype=np.float32)


def _safe_load(p: Path) -> dict:
    return joblib.load(p)


def _month_iter(start: pd.Timestamp, end: pd.Timestamp):
    cur = pd.Timestamp(start.year, start.month, 1)
    last = pd.Timestamp(end.year, end.month, 1)
    while cur < last:
        nxt = (cur + pd.DateOffset(months=1)).to_period("M").to_timestamp()
        yield cur, nxt
        cur = nxt


def walk_forward_fixed(
    parquet_path: Path,
    model_path: Path,
    output_path: Path,
    start: str,
    end: str,
) -> dict:
    """
    Single training (model already trained) → roll predictions month by month.
    """
    bundle = _safe_load(model_path)
    model = bundle["model"]
    columns: list[str] = bundle["feature_columns"]
    threshold = float(bundle.get("threshold", 0.5))
    regime_thresholds = bundle.get("vix_regime_thresholds") or {}

    df = _engineer(_load_clean(parquet_path))
    sdate = pd.Timestamp(start)
    edate = pd.Timestamp(end)

    monthly: list[dict[str, Any]] = []
    full_y: list[int] = []
    full_p: list[float] = []
    full_alert: list[int] = []
    full_dates: list[pd.Timestamp] = []

    for m_start, m_end in _month_iter(sdate, edate):
        chunk = df[(df["alert_date"] >= m_start) & (df["alert_date"] < m_end)]
        if len(chunk) == 0:
            continue

        X = _build_X(chunk, columns)
        proba = np.asarray(model.predict_proba(X))[:, 1]
        y = chunk["_y"].to_numpy(dtype=int)

        # Per-row threshold based on each row's vix_regime
        vix_reg = chunk.get("vix_regime", pd.Series(1.0, index=chunk.index)).to_numpy()
        per_thr = np.zeros_like(proba, dtype=float)
        for i, regime_id in enumerate(vix_reg):
            regime = "low" if regime_id == 0.0 else "high" if regime_id == 2.0 else "medium"
            block = regime_thresholds.get(regime, {}) if isinstance(regime_thresholds, dict) else {}
            if isinstance(block, dict) and not block.get("fallback") and block.get("threshold") is not None:
                per_thr[i] = float(block["threshold"])
            else:
                per_thr[i] = threshold

        alerts = (proba >= per_thr).astype(int)
        n_alerts = int(alerts.sum())
        n_pos = int(y.sum())
        n = len(chunk)
        wins_caught = int(((alerts == 1) & (y == 1)).sum())
        precision = float(wins_caught / n_alerts) if n_alerts else float("nan")
        recall = float(wins_caught / n_pos) if n_pos else float("nan")
        try:
            auc_pr = float(average_precision_score(y, proba)) if n_pos > 0 and n_pos < n else float("nan")
        except ValueError:
            auc_pr = float("nan")

        monthly.append({
            "month": m_start.strftime("%Y-%m"),
            "n_samples": n,
            "n_positives": n_pos,
            "n_alerts": n_alerts,
            "wins_caught": wins_caught,
            "win_rate_actual": round(n_pos / n, 4) if n else 0.0,
            "precision_at_thr": round(precision, 4) if not np.isnan(precision) else None,
            "recall_at_thr": round(recall, 4) if not np.isnan(recall) else None,
            "auc_pr": round(auc_pr, 4) if not np.isnan(auc_pr) else None,
            "mean_threshold": round(float(per_thr.mean()), 4),
        })

        full_y.extend(y.tolist())
        full_p.extend(proba.tolist())
        full_alert.extend(alerts.tolist())
        full_dates.extend([m_start] * len(chunk))

    # Aggregate over the full period
    fy = np.asarray(full_y)
    fp = np.asarray(full_p)
    fa = np.asarray(full_alert)
    if len(fy):
        agg_pos = int(fy.sum())
        agg_alerts = int(fa.sum())
        agg_caught = int(((fa == 1) & (fy == 1)).sum())
        agg_precision = float(agg_caught / agg_alerts) if agg_alerts else float("nan")
        agg_recall = float(agg_caught / agg_pos) if agg_pos else float("nan")
        try:
            agg_auc = float(average_precision_score(fy, fp))
        except ValueError:
            agg_auc = float("nan")
    else:
        agg_pos = agg_alerts = agg_caught = 0
        agg_precision = agg_recall = agg_auc = float("nan")

    out = {
        "mode": "fixed",
        "model": str(model_path),
        "algorithm": bundle.get("algorithm", "?"),
        "global_threshold": threshold,
        "vix_regime_thresholds": regime_thresholds,
        "period": {"start": str(sdate.date()), "end": str(edate.date())},
        "n_samples": int(len(fy)),
        "n_positives": agg_pos,
        "n_alerts": agg_alerts,
        "wins_caught": agg_caught,
        "win_rate_actual": round(agg_pos / max(len(fy), 1), 4),
        "precision_at_thr": round(agg_precision, 4) if not np.isnan(agg_precision) else None,
        "recall_at_thr": round(agg_recall, 4) if not np.isnan(agg_recall) else None,
        "auc_pr": round(agg_auc, 4) if not np.isnan(agg_auc) else None,
        "monthly": monthly,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    logging.info(
        f"[walk-forward fixed] {len(monthly)} months "
        f"P={agg_precision:.3f} R={agg_recall:.3f} AUC-PR={agg_auc:.3f}"
    )
    logging.info(f"[walk-forward] Saved → {output_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest do modelo Stage 1")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--model",   type=Path, required=True)
    parser.add_argument("--output",  type=Path, required=True)
    parser.add_argument("--mode",    type=str, default="fixed", choices=["fixed"])
    parser.add_argument("--start",   type=str, default="2022-01-01")
    parser.add_argument("--end",     type=str, default="2026-01-01")

    args = parser.parse_args()

    if args.mode == "fixed":
        walk_forward_fixed(args.parquet, args.model, args.output, args.start, args.end)


if __name__ == "__main__":
    main()
