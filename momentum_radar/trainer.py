"""
momentum_radar/trainer.py — Treino multi-horizonte do Breakout ML (MomentumRadar).

Três modelos por horizonte:
  L1 (30d)  — trade táctico (peso 0.25 no ensemble)
  L2 (90d)  — janela intermédia (peso 0.40)
  L3 (180d) — momentum duradouro (peso 0.35)

Ensemble: momentum_score = L1×0.25 + L2×0.40 + L3×0.35
  Scores positivos = momentum com durabilidade esperada.
  L3 >> L1 → tag "DURÁVEL" (ex: RKLB em 2024)
  L1 >> L3 → tag "TÁCTICO" (pump de curto prazo)

Padrão partilhado com DipRadar:
  - Walk-forward CV purged (5 folds, purge 30d)
  - ScaledRidge como champion
  - Gating: novo modelo só promovido se IC ≥ IC actual × 0.90
  - Bundle guardado em /data/momentum_model.pkl
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
    # Market Structure (v4.3)
    "vwap_from_low_ratio",   # positivo = preço acima do custo médio desde fundo
    "volume_poc_ratio",      # positivo = preço acima do nível de maior volume (60d)
]

# Alvos multi-horizonte e pesos no ensemble
TARGET_HORIZONS = [
    ("forward_return_30d",  "L1", 0.25),
    ("forward_return_90d",  "L2", 0.40),
    ("forward_return_180d", "L3", 0.35),
]

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
    df = df.dropna(subset=[target]).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    n = len(df)
    if n < 100:
        return 0.0, 0.0
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

        _col_means = {
            col: float(tr[col].dropna().mean()) if col in tr.columns and tr[col].notna().any() else 0.0
            for col in feats
        }
        def _prep(df_):
            r = df_[feats].copy()
            for c in feats:
                if c in r.columns and r[c].isna().any():
                    r[c] = r[c].fillna(_col_means.get(c, 0.0))
            return r.values.astype(np.float32)

        X_tr = _prep(tr)
        y_tr = tr[target].values.astype(float)
        X_te = _prep(te)
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


def _train_final_model(df: pd.DataFrame, feats: list[str], target: str):
    """Treina modelo final no dataset completo, com mean imputation."""
    from ml_training.models import ScaledRidge
    df = df.dropna(subset=[target]).reset_index(drop=True)
    _means = {
        col: float(df[col].dropna().mean()) if col in df.columns and df[col].notna().any() else 0.0
        for col in feats
    }
    r = df[feats].copy()
    for c in feats:
        if c in r.columns and r[c].isna().any():
            r[c] = r[c].fillna(_means.get(c, 0.0))
    X = r.values.astype(np.float32)
    y = df[target].values.astype(float)
    m = ScaledRidge(alpha=10.0)
    m.fit(X, y)
    return m, _means, len(df)


def run_momentum_training(dataset_path: Path = _DATASET) -> dict:
    """Treina o Breakout ML multi-horizonte e promove se melhor.

    Retorna dict com: decision, ic_ensemble, ic_prod, n_train, elapsed_s
    """
    import time
    import joblib

    t0 = time.time()

    if not dataset_path.exists():
        return {"decision": "FAILED", "reason": "Dataset não encontrado. Corre /admin_momentum_dataset primeiro."}

    df = pd.read_parquet(dataset_path)
    missing_feats = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_feats:
        return {"decision": "FAILED", "reason": f"Features em falta: {missing_feats}"}

    # Apenas exige que L1 (30d) esteja presente; L2/L3 são opcionais mas enriquecem
    if "forward_return_30d" not in df.columns:
        return {"decision": "FAILED", "reason": "Coluna forward_return_30d em falta"}

    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    if len(df) < 200:
        return {"decision": "FAILED", "reason": f"Dados insuficientes: {len(df)} amostras (mínimo 200)"}

    log.info(f"[momentum_train] Dataset: {len(df)} amostras | {df['ticker'].nunique()} tickers")

    # Walk-forward IC por horizonte
    horizon_results: dict[str, dict] = {}
    valid_horizons = []
    for target_col, label, weight in TARGET_HORIZONS:
        if target_col not in df.columns or df[target_col].dropna().__len__() < 100:
            log.warning(f"[momentum_train] {label} ({target_col}): dados insuficientes — ignorado")
            continue
        log.info(f"[momentum_train] Walk-forward IC {label} ({target_col})...")
        ic, ic_sr = _walk_forward_ic(df, FEATURE_COLS, target_col)
        log.info(f"[momentum_train] {label}: IC={ic:.4f} IC_SR={ic_sr:.2f}")
        horizon_results[label] = {"target": target_col, "weight": weight, "ic": ic, "ic_sr": ic_sr}
        valid_horizons.append((target_col, label, weight, ic))

    if not valid_horizons:
        return {"decision": "FAILED", "reason": "Nenhum horizonte com dados suficientes"}

    # IC ensemble ponderado (normalizar pesos pelos horizontes disponíveis)
    total_weight = sum(w for _, _, w, _ in valid_horizons)
    ic_ensemble = sum((w / total_weight) * ic for _, _, w, ic in valid_horizons)
    ic_ensemble = round(ic_ensemble, 4)
    log.info(f"[momentum_train] IC ensemble: {ic_ensemble:.4f}")

    # Gating: promover só se IC ≥ IC actual × 0.90
    ic_prod = None
    try:
        if _REPORT_PATH.exists():
            report = json.loads(_REPORT_PATH.read_text())
            ic_prod = report.get("ic_ensemble") or report.get("ic_mean")
    except Exception:
        pass

    if ic_prod is not None and ic_ensemble < ic_prod * 0.90:
        log.warning(f"[momentum_train] IC ({ic_ensemble:.4f}) < prod ({ic_prod:.4f}) × 0.90 — a manter modelo actual")
        return {
            "decision": "KEPT",
            "reason":   f"IC novo ({ic_ensemble:.4f}) não melhora sobre produção ({ic_prod:.4f})",
            "ic_ensemble": ic_ensemble,
            "ic_prod":  ic_prod,
            "elapsed_s": round(time.time() - t0, 1),
        }

    # Treinar modelo final por horizonte
    log.info("[momentum_train] A treinar modelos finais...")
    models: dict[str, dict] = {}
    for target_col, label, weight, ic in valid_horizons:
        m, means, n = _train_final_model(df, FEATURE_COLS, target_col)
        models[label] = {
            "model":    m,
            "means":    means,
            "target":   target_col,
            "weight":   weight / total_weight,  # normalizado
            "ic":       ic,
            "n_train":  n,
        }
        log.info(f"[momentum_train] {label}: {n} amostras")

    # Sector models por L2 (horizonte principal)
    sector_models: dict = {}
    if "sector" in df.columns and "forward_return_90d" in df.columns:
        df_l2 = df.dropna(subset=["forward_return_90d"])
        for sector, grp in df_l2.groupby("sector"):
            if len(grp) < 100:
                continue
            Xs = grp[FEATURE_COLS].fillna(0).values.astype(np.float32)
            ys = grp["forward_return_90d"].values.astype(float)
            from ml_training.models import ScaledRidge
            sm = ScaledRidge(alpha=10.0)
            sm.fit(Xs, ys)
            sector_models[sector] = {"model": sm, "n_train": len(grp)}
            log.info(f"[momentum_train] Sector L2 {sector}: {len(grp)} amostras")

    bundle = {
        "models":        models,         # {"L1": {...}, "L2": {...}, "L3": {...}}
        "sector_models": sector_models,  # sector → {model, n_train}
        "feature_cols":  FEATURE_COLS,
        "horizon_results": horizon_results,
        "n_train":       len(df),
        "train_date":    datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ic_ensemble":   ic_ensemble,
        # Backward compat com scanner que lê "model" e "ic_mean"
        "model":         models.get("L2", models.get("L1", {})).get("model"),
        "ic_mean":       ic_ensemble,
        "target":        "ensemble",
    }

    _BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    import joblib as _jl
    _jl.dump(bundle, _BUNDLE_PATH)

    report = {
        "trained_at":    bundle["train_date"],
        "n_train":       len(df),
        "n_tickers":     int(df["ticker"].nunique()),
        "ic_ensemble":   ic_ensemble,
        "ic_mean":       ic_ensemble,
        "ic_prod":       ic_prod,
        "horizons":      {lbl: {"ic": r["ic"], "ic_sr": r["ic_sr"], "weight": r["weight"]} for lbl, r in horizon_results.items()},
        "feature_cols":  FEATURE_COLS,
        "sector_models": {s: v["n_train"] for s, v in sector_models.items()},
    }
    _REPORT_PATH.write_text(json.dumps(report, indent=2))

    elapsed = round(time.time() - t0, 1)
    log.info(f"[momentum_train] PROMOVIDO em {elapsed}s | IC_ensemble={ic_ensemble:.4f} | {len(models)} horizontes | {len(sector_models)} sector models")

    return {
        "decision":    "PROMOTED",
        "ic_ensemble": ic_ensemble,
        "ic_prod":     ic_prod,
        "horizons":    {lbl: r["ic"] for lbl, r in horizon_results.items()},
        "n_train":     len(df),
        "n_sectors":   len(sector_models),
        "elapsed_s":   elapsed,
    }
