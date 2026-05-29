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
_REPO_DIR    = Path(__file__).parent
_DATASET     = _REPO_DIR / "momentum_training.parquet"
_BUNDLE_PATH = _REPO_DIR / "momentum_model.pkl"
_REPORT_PATH = _REPO_DIR / "momentum_report.json"

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
    # Volume Flow
    "mfi_14",                # Money Flow Index 14d — acumulação institucional
    "volume_up_pct_20d",     # % volume em dias de subida — pressão compradora
    # Insider activity (Form 4, PIT). Buys after a breakout = smart money
    # confirming; sells = distribution. NaN propagates for pre-2003 episodes
    # and non-US tickers (LightGBM handles natively).
    "insider_buy_recent",
    "insider_buy_amount_score",
    "insider_sell_recent",
    "insider_sell_amount_score",
]

# Multi-horizon targets — alpha vs SPY removes market regime bias.
# Trains on "will this stock outperform SPY" rather than "will market go up."
# alpha_Xd columns are built by build_dataset.py alongside forward_return_Xd.
TARGET_HORIZONS = [
    ("alpha_30d",  "L1", 0.25),
    ("alpha_90d",  "L2", 0.40),
    ("alpha_180d", "L3", 0.35),
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


def _make_momentum_model():
    """LightGBM-ES (mesmo config que DipRadar) com ScaledRidge como fallback.

    Usa a factory partilhada de ml_training/models.py para que ambos os
    modelos evoluam juntos quando os hiperparâmetros forem ajustados.
    """
    try:
        from ml_training.models import lgbm_es_factory
        return lgbm_es_factory()
    except Exception:
        from ml_training.models import ScaledRidge
        return ScaledRidge(alpha=10.0)


def _is_tree(m) -> bool:
    name = type(m).__name__.lower()
    return any(x in name for x in ("lgbm", "lightgbm", "xgb", "forest", "tree"))


def _walk_forward_ic(df: pd.DataFrame, feats: list[str], target: str) -> tuple[float, float]:
    """Walk-forward purged CV — returns (mean_IC, IC_SR).

    Always uses the ml_training infra. The previous numpy fallback dropped
    LightGBM's feature names and produced spurious predictions (LGBM warns
    "X does not have valid feature names, but LGBMRegressor was fitted with
    feature names"). If ml_training isn't importable, we crash explicitly
    so the bug surfaces instead of silently producing inverted IC.
    """
    from ml_training.train import _fit_model, _split_val, _prepare_X, _compute_col_means

    n_before = len(df)
    df = df.dropna(subset=[target]).sort_values("date").reset_index(drop=True)
    log.info(f"[momentum_train/{target}] dropna({target}): {n_before} → {len(df)} rows")
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

        m = _make_momentum_model()

        col_means = _compute_col_means(tr, feats)
        X_tr = _prepare_X(tr, feats, col_means, m)
        X_te = _prepare_X(te, feats, col_means, m)
        y_tr = tr[target].values.astype(float)
        y_te = te[target].values
        sw_tr = np.ones(len(X_tr), dtype=np.float32)
        X_t, y_t, sw_t, X_v, y_v = _split_val(X_tr, y_tr, sw_tr)
        m = _fit_model(m, X_t, y_t, sw_t, X_v, y_v)

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
    """Train final model on the full dataset.

    Same infra requirement as _walk_forward_ic. LightGBM handles NaN natively,
    so means is returned as {} for tree models (scanner attempts imputation
    but LGBM ignores it).
    """
    from ml_training.train import _fit_model, _split_val, _prepare_X, _compute_col_means

    df = df.dropna(subset=[target]).reset_index(drop=True)
    m = _make_momentum_model()

    col_means = _compute_col_means(df, feats)
    X = _prepare_X(df, feats, col_means, m)
    y = df[target].values.astype(float)
    sw = np.ones(len(X), dtype=np.float32)
    X_t, y_t, sw_t, X_v, y_v = _split_val(X, y, sw)
    m = _fit_model(m, X_t, y_t, sw_t, X_v, y_v)
    means = {} if _is_tree(m) else col_means

    return m, means, len(df)


def run_momentum_training(
    dataset_path: Path = _DATASET,
    bundle_path:  Path = _BUNDLE_PATH,
    report_path:  Path = _REPORT_PATH,
) -> dict:
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

    # Winsorize targets (p1–p99) para reduzir influência de outliers extremos
    for _tcol in ["forward_return_30d", "forward_return_90d", "forward_return_180d"]:
        if _tcol in df.columns and df[_tcol].notna().sum() > 10:
            _p1, _p99 = df[_tcol].quantile([0.01, 0.99])
            df[_tcol] = df[_tcol].clip(_p1, _p99)

    log.info(f"[momentum_train] Dataset: {len(df)} amostras | {df['ticker'].nunique()} tickers")

    # ── Alpha vs forward-return sanity check ──
    # alpha_Xd is log(stock) - log(spy); it must correlate strongly with the
    # raw forward stock return for the dataset to be usable. A correlation
    # <0.8 means SPY history is misaligned or alpha has a sign/horizon bug.
    # We log the result so it shows up even when the builder hasn't been
    # re-run since the latest fix.
    log.info("[momentum_train] Alpha sanity (alpha_Xd vs forward_return_Xd):")
    for _target_col, _label, _w in TARGET_HORIZONS:
        _fwd_col = _target_col.replace("alpha_", "forward_return_")
        if _target_col in df.columns and _fwd_col in df.columns:
            _pair = df[[_target_col, _fwd_col]].dropna()
            if len(_pair) >= 30:
                _corr = _pair[_target_col].corr(_pair[_fwd_col])
                _flag = "OK" if (_corr is not None and _corr >= 0.80) else "*** SUSPECT ***"
                log.info(
                    f"  {_label} {_target_col}: n={len(_pair)} corr={_corr:+.4f} [{_flag}]"
                )
            else:
                log.warning(f"  {_label} {_target_col}: only {len(_pair)} non-null pairs")
        else:
            log.warning(f"  {_label} {_target_col}: missing column {_target_col} or {_fwd_col}")

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
        if report_path.exists():
            report = json.loads(report_path.read_text())
            ic_prod = report.get("ic_ensemble") or report.get("ic_mean")
    except Exception:
        pass

    _IC_FLOOR = 0.06  # IC < 0.06 is not worth promoting — barely predictive
    if ic_ensemble < _IC_FLOOR:
        log.warning(f"[momentum_train] IC ({ic_ensemble:.4f}) below floor ({_IC_FLOOR}) — keeping current model")
        return {
            "decision":    "KEPT",
            "reason":      f"New IC ({ic_ensemble:.4f}) below minimum floor ({_IC_FLOOR})",
            "ic_ensemble": ic_ensemble,
            "ic_prod":     ic_prod,
            "n_train":     len(df),
            "elapsed_s":   round(time.time() - t0, 1),
        }
    if ic_prod is not None and ic_ensemble < ic_prod * 0.90:
        log.warning(f"[momentum_train] IC ({ic_ensemble:.4f}) < prod ({ic_prod:.4f}) × 0.90 — keeping current model")
        return {
            "decision":    "KEPT",
            "reason":      f"New IC ({ic_ensemble:.4f}) does not improve over production ({ic_prod:.4f})",
            "ic_ensemble": ic_ensemble,
            "ic_prod":     ic_prod,
            "n_train":     len(df),
            "elapsed_s":   round(time.time() - t0, 1),
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

    # Sector models on L2 (main horizon). Same alpha target as main models —
    # raw returns conflate market regime with stock-specific signal.
    # Mean-imputation per sector (instead of fillna(0)) so missing features
    # don't pull ScaledRidge predictions toward the wrong baseline.
    sector_models: dict = {}
    if "sector" in df.columns and "alpha_90d" in df.columns:
        from scipy.stats import spearmanr as _spr
        try:
            from ml_training.train import _compute_col_means, _prepare_X
            from ml_training.models import ScaledRidge
            _sector_infra = True
        except ImportError:
            _sector_infra = False
            log.warning("[momentum_train] ml_training infra unavailable — sector models skipped")

        if _sector_infra:
            df_l2 = df.dropna(subset=["alpha_90d"])
            for sector, grp in df_l2.groupby("sector"):
                if len(grp) < 100:
                    continue
                sm = ScaledRidge(alpha=10.0)
                col_means = _compute_col_means(grp, FEATURE_COLS)
                Xs = _prepare_X(grp, FEATURE_COLS, col_means, sm)
                ys = grp["alpha_90d"].values.astype(float)
                sm.fit(Xs, ys)
                try:
                    preds = sm.predict(Xs)
                    ic_train = round(float(_spr(preds, ys).statistic), 4)
                except Exception:
                    ic_train = None
                sector_models[sector] = {
                    "model":    sm,
                    "means":    col_means,
                    "n_train":  len(grp),
                    "ic_train": ic_train,
                }
                log.info(f"[momentum_train] Sector L2 {sector}: {len(grp)} samples | IC_train={ic_train}")

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

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib as _jl
    _jl.dump(bundle, bundle_path)

    report = {
        "trained_at":    bundle["train_date"],
        "n_train":       len(df),
        "n_tickers":     int(df["ticker"].nunique()),
        "ic_ensemble":   ic_ensemble,
        "ic_mean":       ic_ensemble,
        "ic_prod":       ic_prod,
        "horizons":      {lbl: {"ic": r["ic"], "ic_sr": r["ic_sr"], "weight": r["weight"]} for lbl, r in horizon_results.items()},
        "feature_cols":  FEATURE_COLS,
        "sector_models": {s: {"n_train": v["n_train"], "ic_train": v.get("ic_train")} for s, v in sector_models.items()},
    }
    report_path.write_text(json.dumps(report, indent=2))

    elapsed = round(time.time() - t0, 1)
    log.info(f"[momentum_train] PROMOVIDO em {elapsed}s | IC_ensemble={ic_ensemble:.4f} | {len(models)} horizontes | {len(sector_models)} sector models")

    return {
        "decision":    "PROMOTED",
        "ic_ensemble": ic_ensemble,
        "ic_prod":     ic_prod,
        "horizons":    {lbl: r["ic"] for lbl, r in horizon_results.items()},
        "sector_ics":  {s: v.get("ic_train") for s, v in sector_models.items()},
        "n_train":     len(df),
        "n_sectors":   len(sector_models),
        "elapsed_s":   elapsed,
    }
