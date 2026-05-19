"""
momentum_radar/trainer.py — Treino do Breakout ML (MomentumRadar).

Usa o mesmo padrão do DipRadar:
  - Walk-forward CV purged (5 folds, purge 30d — horizonte mais curto que dip)
  - ScaledRidge como champion (sample-efficient, interpretável)
  - Gating: novo modelo só promovido se IC ≥ IC actual × 0.90
  - Bundle guardado em /data/momentum_model.pkl
  - Histórico de dados nunca descartado — dataset acumula mensalmente

Target: forward_return_30d (retorno absoluto a 30 dias)
  Diferente do DipRadar (alpha_90d vs SPY): momentum é absoluto porque
  o objectivo é capturar movimentos fortes, não apenas bater o SPY.

Features:
  return_20d, return_5d, return_60d_pre, volume_ratio_20d,
  rsi_14, pct_from_52w_high, atr_pct, close_in_range_20d
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_DATA_DIR    = Path("/data") if Path("/data").exists() else Path("/tmp")
_DATASET     = _DATA_DIR / "momentum_training.parquet"
_BUNDLE_PATH = _DATA_DIR / "momentum_model.pkl"
_REPORT_PATH = _DATA_DIR / "momentum_report.json"

FEATURE_COLS = [
    "return_20d",
    "return_5d",
    "return_60d_pre",
    "volume_ratio_20d",
    "rsi_14",
    "pct_from_52w_high",
    "atr_pct",
    "close_in_range_20d",
]
TARGET_COL = "forward_return_30d"

PURGE_DAYS = 30
N_FOLDS    = 5


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if len(x) < 10:
        return float("nan")
    try:
        rho, _ = spearmanr(x, y)
        return float(rho) if math.isfinite(float(rho)) else 0.0
    except Exception:
        return 0.0


def _walk_forward_ic(df: pd.DataFrame, feats: list[str], target: str) -> tuple[float, float]:
    """Walk-forward CV purged — devolve (mean_IC, IC_SR)."""
    from ml_training.models import ScaledRidge
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    n = len(df)
    fold_size = n // (N_FOLDS + 1)
    ics: list[float] = []

    for k in range(N_FOLDS):
        train_end  = df["date"].iloc[fold_size * (k + 1)]
        purge_end  = train_end + pd.Timedelta(days=PURGE_DAYS)
        test_start = purge_end
        test_end   = df["date"].iloc[min(fold_size * (k + 2), n - 1)]

        tr = df[df["date"] <= train_end]
        te = df[(df["date"] > test_start) & (df["date"] <= test_end)]

        if len(tr) < 50 or len(te) < 20:
            continue

        X_tr = tr[feats].fillna(0).values.astype(np.float32)
        y_tr = tr[target].values.astype(float)
        X_te = te[feats].fillna(0).values.astype(np.float32)
        y_te = te[target].values

        m = ScaledRidge(alpha=10.0)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        ic = _spearman(preds, y_te)
        if math.isfinite(ic):
            ics.append(ic)

    if not ics:
        return 0.0, 0.0
    mean_ic = float(np.mean(ics))
    ic_sr   = mean_ic / float(np.std(ics)) if np.std(ics) > 0 else 0.0
    return round(mean_ic, 4), round(ic_sr, 2)


def run_momentum_training(dataset_path: Path = _DATASET) -> dict:
    """Treina o Breakout ML e promove se melhor que o modelo actual.

    Retorna dict com: decision, ic_new, ic_prod, n_train, elapsed_s
    """
    import time
    from ml_training.models import ScaledRidge
    import joblib

    t0 = time.time()

    if not dataset_path.exists():
        return {"decision": "FAILED", "reason": "Dataset não encontrado. Corre /admin_momentum_dataset primeiro."}

    df = pd.read_parquet(dataset_path)
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        return {"decision": "FAILED", "reason": f"Colunas em falta: {missing}"}

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    if len(df) < 200:
        return {"decision": "FAILED", "reason": f"Dados insuficientes: {len(df)} amostras (mínimo 200)"}

    log.info(f"[momentum_train] Dataset: {len(df)} amostras | {df['ticker'].nunique()} tickers")
    log.info(f"[momentum_train] A calcular IC walk-forward ({N_FOLDS} folds, purge {PURGE_DAYS}d)...")

    ic_new, ic_sr = _walk_forward_ic(df, FEATURE_COLS, TARGET_COL)
    log.info(f"[momentum_train] IC novo: {ic_new:.4f} | IC SR: {ic_sr:.2f}")

    # Ler IC do modelo em produção (se existir)
    ic_prod = None
    try:
        if _REPORT_PATH.exists():
            report = json.loads(_REPORT_PATH.read_text())
            ic_prod = report.get("ic_mean")
    except Exception:
        pass

    # Gating: promover só se IC ≥ IC actual × 0.90
    if ic_prod is not None and ic_new < ic_prod * 0.90:
        log.warning(f"[momentum_train] IC novo ({ic_new:.4f}) < IC produção ({ic_prod:.4f}) × 0.90 — a manter modelo actual")
        return {
            "decision": "KEPT",
            "reason":   f"IC novo ({ic_new:.4f}) não melhora suficientemente sobre produção ({ic_prod:.4f})",
            "ic_new":   ic_new,
            "ic_prod":  ic_prod,
            "elapsed_s": round(time.time() - t0, 1),
        }

    # Treinar modelo final no dataset completo
    log.info("[momentum_train] A treinar modelo final no dataset completo...")
    X_full = df[FEATURE_COLS].fillna(0).values.astype(np.float32)
    y_full = df[TARGET_COL].values.astype(float)
    model  = ScaledRidge(alpha=10.0)
    model.fit(X_full, y_full)

    # Sector models (um ScaledRidge por sector)
    sector_models: dict = {}
    if "sector" in df.columns:
        for sector, grp in df.groupby("sector"):
            if len(grp) < 100:
                continue
            Xs = grp[FEATURE_COLS].fillna(0).values.astype(np.float32)
            ys = grp[TARGET_COL].values.astype(float)
            sm = ScaledRidge(alpha=10.0)
            sm.fit(Xs, ys)
            sector_models[sector] = {"model": sm, "n_train": len(grp)}
            log.info(f"[momentum_train] Sector model {sector}: {len(grp)} amostras")

    bundle = {
        "model":          model,
        "sector_models":  sector_models,
        "feature_cols":   FEATURE_COLS,
        "target":         TARGET_COL,
        "n_train":        len(df),
        "train_date":     datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ic_mean":        ic_new,
        "ic_sr":          ic_sr,
    }

    # Guardar bundle e report
    _BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, _BUNDLE_PATH)

    report = {
        "trained_at": bundle["train_date"],
        "n_train":    len(df),
        "n_tickers":  int(df["ticker"].nunique()),
        "ic_mean":    ic_new,
        "ic_sr":      ic_sr,
        "ic_prod":    ic_prod,
        "feature_cols": FEATURE_COLS,
        "target":     TARGET_COL,
        "sector_models": {s: v["n_train"] for s, v in sector_models.items()},
    }
    _REPORT_PATH.write_text(json.dumps(report, indent=2))

    elapsed = round(time.time() - t0, 1)
    log.info(f"[momentum_train] PROMOVIDO em {elapsed}s | IC={ic_new:.4f} | {len(sector_models)} sector models")

    return {
        "decision":  "PROMOTED",
        "ic_new":    ic_new,
        "ic_sr":     ic_sr,
        "ic_prod":   ic_prod,
        "n_train":   len(df),
        "n_sectors": len(sector_models),
        "elapsed_s": elapsed,
    }
