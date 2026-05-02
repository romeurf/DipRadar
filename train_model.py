"""
train_model.py — DipRadar ML Lab v2 (robusto e walk-forward-safe).

ARQUITECTURA: Classificador em Cascata de 2 Andares com calibração temporal.

  Andar 1 — Filtro binário WIN vs NOT-WIN
    - Soft-voting ensemble (RF + XGB + LightGBM)
    - Treinado em janela cronológica antiga (T1)
    - Probabilidades calibradas isotonicamente em janela de meio (T2)
    - Threshold seleccionado em T2 maximizando F-beta com Precision mínima
    - Métricas reportadas em janela final out-of-sample (T3)

  Andar 2 — Granulosidade WIN_40 vs WIN_20
    - Mesmo pipeline (ensemble + calibração)
    - Subset cronológico de wins, mesmo split T1/T2/T3

DATA HYGIENE (aplicado por defeito, --no-clean para desligar):
  * Drop pré-2014: fundamentais são fallback constante → pollui o treino
  * Dedup (symbol, alert_date)
  * Drop outliers |return_3m| > 200%
  * Engineer 5 features derivadas (rsi_oversold_strength, vix_regime,
    pe_attractive, drop_x_drawdown, vol_x_drop) consistentes com inferência

SAMPLE WEIGHTING:
  * Decay exponencial pela idade do alerta (--half-life-years, default 3)
  * Wins ligeiramente sobreponderados (1.5x) — tunable via --pos-weight

ARTEFACTOS COMPATÍVEIS:
  * `dip_model_stage1.pkl` / `dip_model_stage2.pkl` em formato
    (chave `"model"`, `"feature_columns"`, `"threshold"`) compatível com
    `ml_predictor.py`. Inclui também `"vix_regime_thresholds"` para
    threshold dinâmico em produção.
  * `ml_report.json` enriquecido: AUC-PR train/cal/test, classification
    report OOS, P/R por threshold, feature importance, breakdown por
    sector e por regime VIX.

CLI:
  python train_model.py --parquet ml_training_merged.parquet --output-dir /tmp/m
  python train_model.py --report                              # imprime relatório
  python train_model.py --dry-run                             # valida sem treinar
  python train_model.py --parquet ... --no-clean              # mantém dados sintéticos
  python train_model.py --parquet ... --algos rf,xgb          # só 2 dos 3 base
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_ensemble import IsotonicCalibratedVote, PrefittedSoftVote
from ml_features import FEATURE_COLUMNS, add_derived_features

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Paths (overridable via CLI) ───────────────────────────────────────────────
_DATA_DIR = Path("/data") if Path("/data").exists() else Path("/tmp")
_LIVE_DB  = _DATA_DIR / "alert_db.csv"
_HIST_DB  = _DATA_DIR / "hist_backtest.csv"
_MODEL_S1 = _DATA_DIR / "dip_model_stage1.pkl"
_MODEL_S2 = _DATA_DIR / "dip_model_stage2.pkl"
_REPORT   = _DATA_DIR / "ml_report.json"

_WIN_LABELS  = {"WIN_40", "WIN_20"}
_LOSE_LABELS = {"NEUTRAL", "LOSS_15"}

# ── Defaults ──────────────────────────────────────────────────────────────────
# Date-based splits are preferred over row-based because the dataset is
# heavily skewed toward 2020 (COVID, 48% of all rows). Row-based splits would
# place T2/T3 in the COVID rally — useless for modern-regime evaluation.
#
#   Train (T1) : 2014-01 → 2021-01 (≈5564 rows, includes COVID rally for pattern learning)
#   Cal   (T2) : 2021-01 → 2023-01 (≈1472 rows, post-rally + 2022 bear regime)
#   Test  (T3) : 2023-01 → present (≈288 rows, fully out-of-sample modern regime)
_DEFAULT_TRAIN_END = "2021-01-01"
_DEFAULT_CAL_END   = "2023-01-01"
# Row-based fallbacks (used only with --row-split flag)
_DEFAULT_TRAIN_FRAC = 0.70
_DEFAULT_CAL_FRAC   = 0.15

_DEFAULT_HALF_LIFE_YEARS = 1.5    # heavier recency tilt: 2020 weight ≈ 0.06× of 2024
_DEFAULT_POS_WEIGHT = 1.5
# Precision-first defaults: user wants max wins / min losses → favour Precision
# over Recall. F-0.5 weights Precision 4x more than Recall in the F-beta score.
_DEFAULT_MIN_PRECISION = 0.40     # ~2.5x baseline win rate; achievable on calibration
_DEFAULT_BETA = 0.5               # F-0.5 → Precision 4x weight vs Recall
_DEFAULT_MIN_YEAR = 2014          # drop pre-2014 synthetic data
_RETURN_OUTLIER = 200.0           # |return_3m| > 200% → outlier
_TS_CV_SPLITS = 5

# ── VIX regime thresholds (used at inference for dynamic threshold) ───────────
_VIX_REGIMES = {
    "low":    (-float("inf"), 15.0),
    "medium": (15.0,          25.0),
    "high":   (25.0,          float("inf")),
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & cleaning
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataSplit:
    """Three chronological slices of the dataset."""
    X_train: np.ndarray
    y_train: np.ndarray
    w_train: np.ndarray
    X_cal:   np.ndarray
    y_cal:   np.ndarray
    X_test:  np.ndarray
    y_test:  np.ndarray
    meta_train: pd.DataFrame
    meta_cal:   pd.DataFrame
    meta_test:  pd.DataFrame
    feature_names: list[str]


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 5 derived features in a vectorised way (mirrors add_derived_features)."""
    rsi   = df["rsi_14"].astype(float)
    vix   = df["vix"].astype(float)
    pe_vf = df["pe_vs_fair"].astype(float)
    drop  = df["drop_pct_today"].astype(float)
    dd52  = df["drawdown_52w"].astype(float)
    volsp = df["volume_spike"].astype(float)

    df = df.copy()
    df["rsi_oversold_strength"] = (40.0 - rsi).clip(lower=0.0).round(4)
    df["vix_regime"] = pd.cut(
        vix, bins=[-np.inf, 15.0, 25.0, np.inf], labels=[0.0, 1.0, 2.0],
    ).astype(float)
    df["pe_attractive"]   = (1.0 - pe_vf).clip(lower=0.0).round(4)
    df["drop_x_drawdown"] = (drop * dd52 / 100.0).round(4)
    df["vol_x_drop"]      = (volsp * drop.abs()).round(4)
    return df


def _clean_dataset(
    df: pd.DataFrame,
    min_year: Optional[int],
    drop_outliers: bool,
    dedup: bool,
) -> pd.DataFrame:
    """Apply data hygiene: dedup, drop synthetic rows, drop outliers."""
    df = df.copy()
    n0 = len(df)

    # Parse dates once
    df["alert_date"] = pd.to_datetime(df["alert_date"], errors="coerce")
    df = df[df["alert_date"].notna()].copy()
    if len(df) < n0:
        logging.info(f"[clean] Dropped {n0 - len(df)} rows with invalid alert_date")

    if dedup:
        before = len(df)
        df = df.drop_duplicates(subset=["symbol", "alert_date"], keep="first")
        removed = before - len(df)
        if removed > 0:
            logging.info(f"[clean] Removed {removed} duplicate (symbol, alert_date) rows")

    if min_year is not None:
        before = len(df)
        df = df[df["alert_date"].dt.year >= min_year]
        removed = before - len(df)
        if removed > 0:
            logging.info(
                f"[clean] Dropped {removed} pre-{min_year} rows "
                f"(synthetic fundamentals = constant fallback)"
            )

    if drop_outliers and "return_3m" in df.columns:
        before = len(df)
        df = df[df["return_3m"].abs() <= _RETURN_OUTLIER]
        removed = before - len(df)
        if removed > 0:
            logging.info(
                f"[clean] Dropped {removed} rows with |return_3m| > {_RETURN_OUTLIER}% "
                "(corporate actions / data corruption)"
            )

    df = df.sort_values("alert_date").reset_index(drop=True)
    logging.info(f"[clean] Kept {len(df)} / {n0} rows after cleaning ({len(df)/n0*100:.1f}%)")
    return df


def _impute_in_split(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """
    Impute missing feature values with the median of the EARLIEST split only,
    to prevent test-time leakage. Caller passes the full DataFrame in chronological
    order; this function uses the first 70% as the imputation reference.
    """
    n = len(df)
    cutoff = int(n * _DEFAULT_TRAIN_FRAC)
    train_slice = df.iloc[:cutoff]
    medians = train_slice[feature_names].median(numeric_only=True)
    df = df.copy()
    df[feature_names] = df[feature_names].fillna(medians).fillna(0.0)
    return df


def _compute_sample_weights(
    dates: pd.Series,
    y: np.ndarray,
    half_life_years: float,
    pos_weight: float,
) -> np.ndarray:
    """
    Exponential-decay weighting by alert age + small upweight for positive class.

    weight = pos_factor * 0.5^((today - date) / half_life_years)

    Recent alerts (closer to "now") get weight ≈ 1; alerts from 6 years ago with
    half_life=3 years get weight ≈ 0.25.
    """
    ref_date = pd.to_datetime(dates).max()
    age_years = (ref_date - pd.to_datetime(dates)).dt.total_seconds() / (365.25 * 24 * 3600)
    decay = np.power(0.5, age_years.values / max(half_life_years, 0.1))
    pos_factor = np.where(y == 1, pos_weight, 1.0)
    w = decay * pos_factor
    # Normalise so mean weight = 1 (numerical stability for ensembles)
    w = w / max(w.mean(), 1e-9)
    return w.astype(np.float32)


def prepare_data(
    parquet_path: Path,
    train_end: Optional[str] = _DEFAULT_TRAIN_END,
    cal_end:   Optional[str] = _DEFAULT_CAL_END,
    train_frac: float = _DEFAULT_TRAIN_FRAC,
    cal_frac: float   = _DEFAULT_CAL_FRAC,
    half_life_years: float = _DEFAULT_HALF_LIFE_YEARS,
    pos_weight: float = _DEFAULT_POS_WEIGHT,
    min_year: Optional[int] = _DEFAULT_MIN_YEAR,
    drop_outliers: bool = True,
    dedup: bool = True,
) -> tuple[DataSplit, pd.DataFrame]:
    """
    Load parquet, clean, engineer features, compute weights, split chronologically.

    Two split modes (date-based is preferred):
      * If `train_end` and `cal_end` are provided, splits by date.
      * Else, splits by row fraction (`train_frac` / `cal_frac`).

    Returns (DataSplit, cleaned_dataframe).
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    logging.info(f"[data] Loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logging.info(f"[data] Raw rows: {len(df)} | cols: {list(df.columns)}")

    # Cleaning
    df = _clean_dataset(df, min_year=min_year, drop_outliers=drop_outliers, dedup=dedup)

    # Filter to valid outcome labels
    valid_labels = list(_WIN_LABELS | _LOSE_LABELS)
    df = df[df["outcome_label"].isin(valid_labels)].copy()

    # Engineer derived features
    df = _engineer(df)

    feature_names = list(FEATURE_COLUMNS)
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features in parquet (and not engineerable): {missing}")

    # Convert numerics
    for c in feature_names:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Impute (using train portion median to avoid leakage)
    df = _impute_in_split(df, feature_names)

    # Build labels
    df["_y"] = df["outcome_label"].isin(_WIN_LABELS).astype(int)

    # Chronological split (date-based preferred, falls back to row-based)
    if train_end and cal_end:
        train_end_ts = pd.Timestamp(train_end)
        cal_end_ts   = pd.Timestamp(cal_end)
        train_df = df[df["alert_date"] < train_end_ts].copy()
        cal_df   = df[(df["alert_date"] >= train_end_ts) & (df["alert_date"] < cal_end_ts)].copy()
        test_df  = df[df["alert_date"] >= cal_end_ts].copy()
        logging.info(f"[data] Date-based split: train < {train_end}, cal < {cal_end}, test ≥ {cal_end}")
    else:
        n = len(df)
        n_train = int(n * train_frac)
        n_cal   = int(n * cal_frac)
        train_df = df.iloc[:n_train].copy()
        cal_df   = df.iloc[n_train:n_train + n_cal].copy()
        test_df  = df.iloc[n_train + n_cal:].copy()
        logging.info(
            f"[data] Row-based split: train={train_frac:.0%}, cal={cal_frac:.0%}, "
            f"test={1 - train_frac - cal_frac:.0%}"
        )

    logging.info(
        f"[data] Splits — train={len(train_df)} ({train_df['alert_date'].min().date()}→{train_df['alert_date'].max().date()}) "
        f"| cal={len(cal_df)} ({cal_df['alert_date'].min().date() if len(cal_df) else 'n/a'}→"
        f"{cal_df['alert_date'].max().date() if len(cal_df) else 'n/a'}) "
        f"| test={len(test_df)} ({test_df['alert_date'].min().date() if len(test_df) else 'n/a'}→"
        f"{test_df['alert_date'].max().date() if len(test_df) else 'n/a'})"
    )

    if len(train_df) < 200 or len(cal_df) < 50 or len(test_df) < 50:
        raise ValueError(
            f"Insufficient data after splits: train={len(train_df)}, "
            f"cal={len(cal_df)}, test={len(test_df)}. Need ≥200/50/50."
        )

    X_train = train_df[feature_names].values.astype(np.float32)
    X_cal   = cal_df[feature_names].values.astype(np.float32)
    X_test  = test_df[feature_names].values.astype(np.float32)
    y_train = train_df["_y"].values.astype(int)
    y_cal   = cal_df["_y"].values.astype(int)
    y_test  = test_df["_y"].values.astype(int)

    # Compute sample weights for training set only
    w_train = _compute_sample_weights(
        train_df["alert_date"], y_train, half_life_years, pos_weight,
    )
    logging.info(
        f"[data] Sample weights — mean={w_train.mean():.3f} "
        f"min={w_train.min():.3f} max={w_train.max():.3f} | "
        f"pos_weight={pos_weight} half_life={half_life_years}y"
    )

    # Class distribution sanity
    logging.info(
        f"[data] Win rate — train={y_train.mean():.3f} "
        f"cal={y_cal.mean():.3f} test={y_test.mean():.3f}"
    )

    split = DataSplit(
        X_train=X_train, y_train=y_train, w_train=w_train,
        X_cal=X_cal, y_cal=y_cal,
        X_test=X_test, y_test=y_test,
        meta_train=train_df, meta_cal=cal_df, meta_test=test_df,
        feature_names=feature_names,
    )
    return split, df


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_base_model(name: str, stage: int = 1) -> Any:
    """
    Build a single sklearn-compatible classifier wrapped in a pipeline
    (imputer → scaler → clf). Imputer is defensive: features should already
    be filled, but this handles inference-time drift.
    """
    name = name.lower()
    if name == "rf":
        if stage == 2:
            clf = RandomForestClassifier(
                n_estimators=400, max_depth=8, min_samples_leaf=8,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
        else:
            clf = RandomForestClassifier(
                n_estimators=500, max_depth=12, min_samples_leaf=4,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
    elif name == "xgb":
        from xgboost import XGBClassifier
        if stage == 2:
            clf = XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.04,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                eval_metric="logloss", verbosity=0, n_jobs=-1, random_state=42,
            )
        else:
            clf = XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                eval_metric="logloss", verbosity=0, n_jobs=-1, random_state=42,
            )
    elif name == "lgbm":
        from lightgbm import LGBMClassifier
        if stage == 2:
            clf = LGBMClassifier(
                n_estimators=400, max_depth=-1, num_leaves=31, learning_rate=0.04,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=15,
                class_weight="balanced", verbosity=-1, n_jobs=-1, random_state=42,
            )
        else:
            clf = LGBMClassifier(
                n_estimators=500, max_depth=-1, num_leaves=63, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                class_weight="balanced", verbosity=-1, n_jobs=-1, random_state=42,
            )
    else:
        raise ValueError(f"Unknown algorithm: {name}")

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     clf),
    ])


def _supports_sample_weight(name: str) -> bool:
    """All 3 of our base models support sample_weight via fit kwargs."""
    return name.lower() in ("rf", "xgb", "lgbm")


def _fit_with_weights(pipe: Pipeline, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> Pipeline:
    """Fit a pipeline passing sample_weight to the final classifier."""
    pipe.fit(X, y, **{"clf__sample_weight": w})
    return pipe


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validated AUC-PR (TimeSeriesSplit) — for ensemble weight selection
# ─────────────────────────────────────────────────────────────────────────────

def _cv_auc_pr(name: str, X: np.ndarray, y: np.ndarray, w: np.ndarray, stage: int) -> float:
    """Mean AUC-PR over TimeSeriesSplit(5)."""
    cv = TimeSeriesSplit(n_splits=_TS_CV_SPLITS)
    scores: list[float] = []
    for tr_idx, va_idx in cv.split(X):
        if y[tr_idx].sum() < 5 or y[va_idx].sum() < 1:
            continue
        try:
            pipe = _build_base_model(name, stage=stage)
            _fit_with_weights(pipe, X[tr_idx], y[tr_idx], w[tr_idx])
            proba = pipe.predict_proba(X[va_idx])[:, 1]
            scores.append(float(average_precision_score(y[va_idx], proba)))
        except Exception as e:
            logging.warning(f"[cv:{name}] fold failed: {e}")
    if not scores:
        return 0.0
    mean_score = float(np.mean(scores))
    logging.info(f"[cv:{name}] mean AUC-PR = {mean_score:.4f} ({len(scores)} folds)")
    return mean_score


# ─────────────────────────────────────────────────────────────────────────────
# Threshold selection (calibration-set max F-beta with Precision floor)
# ─────────────────────────────────────────────────────────────────────────────

def _select_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    min_precision: float,
    beta: float,
) -> tuple[float, dict]:
    """
    Returns (threshold, metrics_dict) with optimal F-beta under Precision constraint.
    """
    prec, rec, thr = precision_recall_curve(y_true, proba)
    # precision_recall_curve returns one extra precision/recall vs thresholds
    prec, rec = prec[:-1], rec[:-1]

    # Compute F-beta for every threshold
    beta2 = beta ** 2
    f_beta = (1 + beta2) * (prec * rec) / np.maximum(beta2 * prec + rec, 1e-9)

    # Apply precision floor; if no candidate qualifies, gracefully degrade by
    # picking the threshold that maximises F-beta among the top-precision tail.
    mask = prec >= min_precision
    if mask.any():
        idxs = np.where(mask)[0]
        best_local = idxs[np.argmax(f_beta[idxs])]
        chosen_threshold = float(thr[best_local])
        chosen_label = f"F-beta-max under Precision≥{min_precision:.2f}"
    else:
        # Fallback: relax the floor to the top achievable precision and use
        # F-beta-max within that top quartile. Avoids "1 sample with R=0.01" picks.
        max_p = float(prec.max())
        relaxed_floor = max(0.5 * max_p, min_precision * 0.5)
        rmask = (prec >= relaxed_floor) & (rec >= 0.05)
        if rmask.any():
            ridxs = np.where(rmask)[0]
            best_local = ridxs[np.argmax(f_beta[ridxs])]
            chosen_label = (
                f"F-beta-max under relaxed floor {relaxed_floor:.2f} "
                f"(target {min_precision:.2f} unreachable; max achievable {max_p:.2f})"
            )
        else:
            best_local = int(np.argmax(f_beta))
            chosen_label = "F-beta-max (full fallback)"
        chosen_threshold = float(thr[best_local])
        logging.warning(
            f"[threshold] Precision floor {min_precision:.2f} unreachable. "
            f"Using {chosen_label}: P={prec[best_local]:.3f} R={rec[best_local]:.3f} "
            f"@thr={chosen_threshold:.4f}"
        )

    return chosen_threshold, {
        "selection":     chosen_label,
        "threshold":     round(chosen_threshold, 4),
        "precision":     round(float(prec[best_local]), 4),
        "recall":        round(float(rec[best_local]), 4),
        "f_beta":        round(float(f_beta[best_local]), 4),
        "min_precision": min_precision,
        "beta":          beta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 trainer (binary WIN vs NOT-WIN)
# ─────────────────────────────────────────────────────────────────────────────

def train_stage1(
    split: DataSplit,
    output_dir: Path,
    algos: list[str],
    min_precision: float,
    beta: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "dip_model_stage1.pkl"

    X_tr, y_tr, w_tr = split.X_train, split.y_train, split.w_train
    X_cal, y_cal     = split.X_cal,   split.y_cal
    X_te,  y_te      = split.X_test,  split.y_test

    logging.info(
        f"[stage1] WIN: train={int(y_tr.sum())} cal={int(y_cal.sum())} test={int(y_te.sum())}"
    )

    # ── 1) CV-derive ensemble weights ───────────────────────────────────────
    auc_per_algo: dict[str, float] = {}
    for name in algos:
        try:
            auc_per_algo[name] = _cv_auc_pr(name, X_tr, y_tr, w_tr, stage=1)
        except ImportError as e:
            logging.warning(f"[stage1] {name} unavailable: {e}")
        except Exception as e:
            logging.warning(f"[stage1] {name} CV failed: {e}")

    if not auc_per_algo:
        raise RuntimeError("No base models trained successfully.")

    total = sum(auc_per_algo.values())
    weights = {k: (v / total if total > 0 else 1.0 / len(auc_per_algo))
               for k, v in auc_per_algo.items()}
    logging.info(
        "[stage1] Ensemble weights: " +
        ", ".join(f"{k}={w:.3f}" for k, w in weights.items())
    )

    # ── 2) Fit each base model on full training set (with weights) ───────────
    fitted: list[tuple[str, Pipeline]] = []
    for name in auc_per_algo:
        pipe = _build_base_model(name, stage=1)
        _fit_with_weights(pipe, X_tr, y_tr, w_tr)
        fitted.append((name, pipe))

    # ── 3) Soft VotingClassifier (refit avoided via voting='soft' + cv='prefit'-style) ──
    # VotingClassifier requires re-fitting; we wrap manually instead so we keep
    # already-fitted base models AND can pass them straight to CalibratedClassifierCV.
    voting = PrefittedSoftVote(fitted, [weights[n] for n, _ in fitted])

    # ── 4) Isotonic calibration on T2 (cal split) ───────────────────────────
    # Manual: predict raw voting probas → fit isotonic regressor → wrap together.
    raw_cal = voting.predict_proba(X_cal)[:, 1]
    calibrated = IsotonicCalibratedVote(voting, raw_cal, y_cal)

    # ── 5) Threshold selection on T2 ────────────────────────────────────────
    proba_cal = calibrated.predict_proba(X_cal)[:, 1]
    threshold, thr_metrics = _select_threshold(y_cal, proba_cal, min_precision, beta)
    logging.info(
        f"[stage1] Threshold (cal): {threshold:.4f} | "
        f"P@T={thr_metrics['precision']:.3f} R@T={thr_metrics['recall']:.3f}"
    )

    # ── 6) VIX-regime-specific thresholds (refine on cal) ───────────────────
    vix_idx = split.feature_names.index("vix")
    vix_thresholds: dict[str, dict] = {}
    for regime, (lo, hi) in _VIX_REGIMES.items():
        mask = (split.meta_cal["vix"].values >= lo) & (split.meta_cal["vix"].values < hi)
        if mask.sum() < 30 or y_cal[mask].sum() < 3:
            vix_thresholds[regime] = {"threshold": threshold, "n": int(mask.sum()), "fallback": True}
            continue
        try:
            t_r, m_r = _select_threshold(y_cal[mask], proba_cal[mask], min_precision, beta)
            # Safeguard: if regime threshold gives < 5% recall, the model is
            # firing on noise (e.g. 1-sample-perfect-precision). Fall back to
            # the global threshold which is broader and more honest.
            if m_r.get("recall", 0.0) < 0.05:
                logging.warning(
                    f"[stage1] VIX regime '{regime}' threshold={t_r:.4f} gives "
                    f"degenerate recall={m_r.get('recall', 0):.3f}; using global threshold."
                )
                vix_thresholds[regime] = {
                    "threshold": round(threshold, 4),
                    "n":         int(mask.sum()),
                    "fallback":  True,
                    "reason":    "regime threshold had <5% recall on cal",
                }
            else:
                vix_thresholds[regime] = {
                    "threshold": round(t_r, 4),
                    "n":         int(mask.sum()),
                    "precision": m_r["precision"],
                    "recall":    m_r["recall"],
                    "fallback":  False,
                }
        except Exception as e:
            logging.debug(f"[stage1] VIX regime {regime} threshold failed: {e}")
            vix_thresholds[regime] = {"threshold": threshold, "n": int(mask.sum()), "fallback": True}

    logging.info(f"[stage1] VIX-regime thresholds: {vix_thresholds}")

    # ── 7) Out-of-sample evaluation on T3 ───────────────────────────────────
    proba_te = calibrated.predict_proba(X_te)[:, 1]
    auc_pr_te = float(average_precision_score(y_te, proba_te)) if y_te.sum() else 0.0
    auc_roc_te = float(roc_auc_score(y_te, proba_te)) if y_te.sum() else 0.0
    pred_te = (proba_te >= threshold).astype(int)
    cm_te = confusion_matrix(y_te, pred_te, labels=[0, 1])
    report_te = classification_report(
        y_te, pred_te,
        target_names=["NOT-WIN", "WIN"],
        zero_division=0, digits=3,
    )
    logging.info(f"[stage1] HOLDOUT (T3) AUC-PR={auc_pr_te:.4f} AUC-ROC={auc_roc_te:.4f}")
    logging.info(f"[stage1] HOLDOUT classification report:\n{report_te}")

    # Test-set precision/recall AT chosen threshold
    tp = int(((y_te == 1) & (pred_te == 1)).sum())
    fp = int(((y_te == 0) & (pred_te == 1)).sum())
    fn = int(((y_te == 1) & (pred_te == 0)).sum())
    p_te = tp / max(tp + fp, 1)
    r_te = tp / max(tp + fn, 1)
    logging.info(f"[stage1] HOLDOUT @threshold={threshold:.3f}: P={p_te:.3f} R={r_te:.3f}")

    # Per-sector hit-rate
    sector_metrics = _per_sector_metrics(split.meta_test, y_te, pred_te)

    # Per-VIX-regime hit-rate
    vix_metrics = _per_vix_regime_metrics(split.meta_test, y_te, pred_te)

    # PR curve points (sampled at 11 thresholds for compact JSON)
    prec, rec, thr = precision_recall_curve(y_te, proba_te) if y_te.sum() else (np.array([1.0]), np.array([0.0]), np.array([0.5]))
    pr_curve = []
    for q in np.linspace(0.05, 0.95, 19):
        idx = int(np.clip(q * (len(thr) - 1), 0, len(thr) - 1))
        pr_curve.append({
            "threshold": round(float(thr[idx]), 4),
            "precision": round(float(prec[idx]), 4),
            "recall":    round(float(rec[idx]), 4),
        })

    # ── 8) Feature importance (averaged across base models, weighted) ───────
    feat_imp = _ensemble_feature_importance(fitted, weights, split.feature_names)

    # ── 9) Persist artefact compatible with ml_predictor.py ─────────────────
    bundle = {
        "model":             calibrated,
        "feature_columns":   split.feature_names,
        "n_features":        len(split.feature_names),
        "threshold":         round(threshold, 4),
        "vix_regime_thresholds": vix_thresholds,
        "algorithm":         f"ensemble_iso[{','.join(n for n, _ in fitted)}]",
        "weights":           weights,
        "trained_at":        datetime.now().isoformat(),
        "n_train":           int(len(y_tr)),
        "n_cal":             int(len(y_cal)),
        "n_test":            int(len(y_te)),
        "auc_pr_test":       round(auc_pr_te, 4),
        "auc_roc_test":      round(auc_roc_te, 4),
        "win_rate_train":    round(float(y_tr.mean()), 4),
        "win_rate_test":     round(float(y_te.mean()), 4),
        # Backward-compat keys used by older code paths
        "classifier":        calibrated,
        "feature_names":     split.feature_names,
        "label_map":         {0: "NOT-WIN", 1: "WIN"},
    }
    joblib.dump(bundle, model_path)
    logging.info(f"[stage1] Saved → {model_path}")

    return {
        "algorithm":         bundle["algorithm"],
        "weights":           weights,
        "threshold":         bundle["threshold"],
        "vix_regime_thresholds": vix_thresholds,
        "auc_pr_cv":         {k: round(v, 4) for k, v in auc_per_algo.items()},
        "auc_pr_test":       round(auc_pr_te, 4),
        "auc_roc_test":      round(auc_roc_te, 4),
        "n_train":           int(len(y_tr)),
        "n_cal":             int(len(y_cal)),
        "n_test":            int(len(y_te)),
        "win_rate_train":    round(float(y_tr.mean()), 4),
        "win_rate_cal":      round(float(y_cal.mean()), 4),
        "win_rate_test":     round(float(y_te.mean()), 4),
        "test_precision":    round(p_te, 4),
        "test_recall":       round(r_te, 4),
        "test_confusion":    cm_te.tolist(),
        "threshold_metrics_cal": thr_metrics,
        "feature_importance":    feat_imp,
        "per_sector_test":       sector_metrics,
        "per_vix_regime_test":   vix_metrics,
        "pr_curve_test":         pr_curve,
        "classification_report": report_te,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 trainer (WIN_40 vs WIN_20, only on positives)
# ─────────────────────────────────────────────────────────────────────────────

def train_stage2(
    split: DataSplit,
    full_df: pd.DataFrame,
    output_dir: Path,
    algos: list[str],
    min_precision: float,
    beta: float,
) -> dict | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "dip_model_stage2.pkl"

    # Subset to wins, preserving chronological split
    train_wins = split.meta_train[split.meta_train["outcome_label"].isin(_WIN_LABELS)].copy()
    cal_wins   = split.meta_cal  [split.meta_cal  ["outcome_label"].isin(_WIN_LABELS)].copy()
    test_wins  = split.meta_test [split.meta_test ["outcome_label"].isin(_WIN_LABELS)].copy()

    n_tr_40 = (train_wins["outcome_label"] == "WIN_40").sum()
    n_tr_20 = (train_wins["outcome_label"] == "WIN_20").sum()
    if len(train_wins) < 100 or n_tr_40 < 20 or n_tr_20 < 20:
        logging.warning(
            f"[stage2] Insufficient: train={len(train_wins)} "
            f"WIN_40={n_tr_40} WIN_20={n_tr_20}. Skipping."
        )
        return None

    feat = split.feature_names
    X_tr  = train_wins[feat].values.astype(np.float32)
    X_cal = cal_wins  [feat].values.astype(np.float32)
    X_te  = test_wins [feat].values.astype(np.float32)
    y_tr  = (train_wins["outcome_label"] == "WIN_40").astype(int).values
    y_cal = (cal_wins  ["outcome_label"] == "WIN_40").astype(int).values
    y_te  = (test_wins ["outcome_label"] == "WIN_40").astype(int).values

    # Recompute weights for stage-2 (recency only; balance taken care of by class_weight)
    w_tr = _compute_sample_weights(
        train_wins["alert_date"], y_tr,
        half_life_years=_DEFAULT_HALF_LIFE_YEARS, pos_weight=1.0,
    )

    logging.info(
        f"[stage2] train={len(y_tr)} (WIN_40={int(y_tr.sum())}) "
        f"cal={len(y_cal)} (WIN_40={int(y_cal.sum())}) "
        f"test={len(y_te)} (WIN_40={int(y_te.sum())})"
    )

    # CV ensemble weights
    auc_per_algo: dict[str, float] = {}
    for name in algos:
        try:
            auc_per_algo[name] = _cv_auc_pr(name, X_tr, y_tr, w_tr, stage=2)
        except ImportError:
            continue
        except Exception as e:
            logging.warning(f"[stage2] {name} CV failed: {e}")

    if not auc_per_algo:
        logging.warning("[stage2] No base models trained. Skipping.")
        return None

    total = sum(auc_per_algo.values())
    weights_dict = {k: (v / total if total > 0 else 1.0 / len(auc_per_algo))
                    for k, v in auc_per_algo.items()}

    fitted: list[tuple[str, Pipeline]] = []
    for name in auc_per_algo:
        pipe = _build_base_model(name, stage=2)
        _fit_with_weights(pipe, X_tr, y_tr, w_tr)
        fitted.append((name, pipe))

    voting = PrefittedSoftVote(fitted, [weights_dict[n] for n, _ in fitted])

    # Calibration (only if cal set has both classes)
    if len(y_cal) >= 30 and y_cal.sum() >= 5 and (y_cal == 0).sum() >= 5:
        raw_cal = voting.predict_proba(X_cal)[:, 1]
        calibrated = IsotonicCalibratedVote(voting, raw_cal, y_cal)
        proba_cal = calibrated.predict_proba(X_cal)[:, 1]
        # Stage 2 threshold prefers F1 (no precision floor; we just want best WIN_40 separator)
        threshold, thr_metrics = _select_threshold(y_cal, proba_cal, min_precision=0.5, beta=beta)
    else:
        logging.warning("[stage2] Calibration set too small — using uncalibrated voting at 0.5")
        calibrated = voting
        threshold = 0.5
        thr_metrics = {"selection": "default 0.5", "threshold": 0.5}

    # Evaluate on test
    if len(y_te) >= 20 and y_te.sum() >= 3 and (y_te == 0).sum() >= 3:
        proba_te = calibrated.predict_proba(X_te)[:, 1]
        auc_pr_te  = float(average_precision_score(y_te, proba_te))
        auc_roc_te = float(roc_auc_score(y_te, proba_te))
        pred_te = (proba_te >= threshold).astype(int)
        report_te = classification_report(
            y_te, pred_te, target_names=["WIN_20", "WIN_40"], zero_division=0, digits=3,
        )
        cm_te = confusion_matrix(y_te, pred_te, labels=[0, 1]).tolist()
    else:
        logging.warning("[stage2] Test set too small for evaluation")
        auc_pr_te = auc_roc_te = 0.0
        report_te = "(test set too small)"
        cm_te = [[0, 0], [0, 0]]

    feat_imp = _ensemble_feature_importance(fitted, weights_dict, feat)

    bundle = {
        "model":             calibrated,
        "feature_columns":   feat,
        "n_features":        len(feat),
        "threshold":         round(float(threshold), 4),
        "algorithm":         f"ensemble_iso[{','.join(n for n, _ in fitted)}]",
        "weights":           weights_dict,
        "trained_at":        datetime.now().isoformat(),
        "n_train":           int(len(y_tr)),
        "n_cal":             int(len(y_cal)),
        "n_test":            int(len(y_te)),
        "auc_pr_test":       round(auc_pr_te, 4),
        "auc_roc_test":      round(auc_roc_te, 4),
        # Back-compat
        "classifier":        calibrated,
        "feature_names":     feat,
        "label_map":         {0: "WIN_20", 1: "WIN_40"},
    }
    joblib.dump(bundle, model_path)
    logging.info(f"[stage2] Saved → {model_path}")

    return {
        "algorithm":         bundle["algorithm"],
        "weights":           weights_dict,
        "threshold":         bundle["threshold"],
        "auc_pr_cv":         {k: round(v, 4) for k, v in auc_per_algo.items()},
        "auc_pr_test":       round(auc_pr_te, 4),
        "auc_roc_test":      round(auc_roc_te, 4),
        "n_train":           int(len(y_tr)),
        "n_cal":             int(len(y_cal)),
        "n_test":            int(len(y_te)),
        "test_confusion":    cm_te,
        "threshold_metrics_cal": thr_metrics,
        "feature_importance":    feat_imp,
        "classification_report": report_te,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: feature importance, sector/vix regime metrics
# ─────────────────────────────────────────────────────────────────────────────

def _ensemble_feature_importance(
    fitted: list[tuple[str, Pipeline]],
    weights: dict[str, float],
    feature_names: list[str],
) -> list[dict]:
    """Weighted average of per-model feature importance, top 15."""
    agg = np.zeros(len(feature_names), dtype=np.float64)
    for name, pipe in fitted:
        try:
            clf = pipe.named_steps["clf"]
            if hasattr(clf, "feature_importances_"):
                imp = np.asarray(clf.feature_importances_, dtype=np.float64)
            elif hasattr(clf, "coef_"):
                imp = np.abs(clf.coef_[0])
            else:
                continue
            if imp.sum() > 0:
                imp = imp / imp.sum()
            agg += weights.get(name, 0.0) * imp
        except Exception as e:
            logging.debug(f"[feat-imp] {name}: {e}")

    if agg.sum() > 0:
        agg = agg / agg.sum()

    pairs = sorted(zip(feature_names, agg.tolist()), key=lambda x: x[1], reverse=True)
    return [{"feature": f, "importance": round(float(v), 6)} for f, v in pairs[:15]]


def _per_sector_metrics(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    """Per-sector precision/recall/n on the test set."""
    if "sector" not in meta.columns:
        return []
    out: list[dict] = []
    for sector, grp in meta.assign(_y=y_true, _p=y_pred).groupby("sector", dropna=False):
        n = len(grp)
        if n < 20:
            continue
        tp = int(((grp["_y"] == 1) & (grp["_p"] == 1)).sum())
        fp = int(((grp["_y"] == 0) & (grp["_p"] == 1)).sum())
        fn = int(((grp["_y"] == 1) & (grp["_p"] == 0)).sum())
        out.append({
            "sector":    str(sector) if pd.notna(sector) else "Unknown",
            "n":         n,
            "wins":      int((grp["_y"] == 1).sum()),
            "precision": round(tp / max(tp + fp, 1), 4),
            "recall":    round(tp / max(tp + fn, 1), 4),
        })
    return sorted(out, key=lambda d: d["n"], reverse=True)


def _per_vix_regime_metrics(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    """Per-VIX-regime precision/recall/n."""
    if "vix" not in meta.columns:
        return []
    out: list[dict] = []
    for regime, (lo, hi) in _VIX_REGIMES.items():
        mask = (meta["vix"].values >= lo) & (meta["vix"].values < hi)
        n = int(mask.sum())
        if n < 20:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        out.append({
            "regime":    regime,
            "n":         n,
            "wins":      int((yt == 1).sum()),
            "precision": round(tp / max(tp + fp, 1), 4),
            "recall":    round(tp / max(tp + fn, 1), 4),
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _save_report(stage1: dict, stage2: dict | None, output_dir: Path) -> None:
    report = {
        "trained_at":  datetime.now().isoformat(),
        "schema":      "v2_ensemble_iso_chrono_split",
        "stage1":      {k: v for k, v in stage1.items() if k != "classification_report"},
        "stage1_classification_report": stage1.get("classification_report", ""),
        "stage2":      ({k: v for k, v in stage2.items() if k != "classification_report"} if stage2 else None),
        "stage2_classification_report": stage2.get("classification_report", "") if stage2 else "",
    }
    report_path = output_dir / "ml_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default))
    logging.info(f"[report] Saved → {report_path}")


def _json_default(o: Any) -> Any:
    """JSON encoder fallback for numpy / pandas types."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    raise TypeError(f"Cannot serialise {type(o)}")


def print_report(output_dir: Path) -> None:
    report_path = output_dir / "ml_report.json"
    if not report_path.exists():
        print(f"No report at {report_path}. Run training first.")
        return
    print(json.dumps(json.loads(report_path.read_text()), indent=2, ensure_ascii=False))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_all(
    parquet_path: Path,
    output_dir: Path,
    algos: list[str],
    min_precision: float = _DEFAULT_MIN_PRECISION,
    beta: float          = _DEFAULT_BETA,
    half_life_years: float = _DEFAULT_HALF_LIFE_YEARS,
    pos_weight: float    = _DEFAULT_POS_WEIGHT,
    train_end: Optional[str] = _DEFAULT_TRAIN_END,
    cal_end:   Optional[str] = _DEFAULT_CAL_END,
    train_frac: float    = _DEFAULT_TRAIN_FRAC,
    cal_frac: float      = _DEFAULT_CAL_FRAC,
    min_year: Optional[int] = _DEFAULT_MIN_YEAR,
    drop_outliers: bool  = True,
    dry_run: bool        = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 60)
    logging.info("DipRadar ML Trainer v2 — chrono split + ensemble + isotonic")
    logging.info("=" * 60)
    logging.info(f"  parquet:        {parquet_path}")
    logging.info(f"  output:         {output_dir}")
    logging.info(f"  algos:          {algos}")
    logging.info(f"  min_precision:  {min_precision}")
    logging.info(f"  beta:           {beta}")
    logging.info(f"  half_life:      {half_life_years}y")
    logging.info(f"  pos_weight:     {pos_weight}")
    if train_end and cal_end:
        logging.info(f"  split (date):   train<{train_end}, cal<{cal_end}, test≥{cal_end}")
    else:
        logging.info(f"  split (rows):   {train_frac:.0%}/{cal_frac:.0%}/{1-train_frac-cal_frac:.0%}")
    logging.info(f"  min_year:       {min_year}")
    logging.info(f"  drop_outliers:  {drop_outliers}")

    split, full_df = prepare_data(
        parquet_path=parquet_path,
        train_end=train_end,
        cal_end=cal_end,
        train_frac=train_frac,
        cal_frac=cal_frac,
        half_life_years=half_life_years,
        pos_weight=pos_weight,
        min_year=min_year,
        drop_outliers=drop_outliers,
    )

    if dry_run:
        return {
            "dry_run": True,
            "n_rows":  int(len(full_df)),
            "splits":  {
                "train": int(len(split.y_train)),
                "cal":   int(len(split.y_cal)),
                "test":  int(len(split.y_test)),
            },
            "win_rates": {
                "train": round(float(split.y_train.mean()), 4),
                "cal":   round(float(split.y_cal.mean()), 4),
                "test":  round(float(split.y_test.mean()), 4),
            },
            "feature_names": split.feature_names,
        }

    s1 = train_stage1(split, output_dir, algos, min_precision, beta)
    s2 = train_stage2(split, full_df, output_dir, algos, min_precision, beta)
    _save_report(s1, s2, output_dir)

    print()
    print("=" * 70)
    print("DipRadar ML — Training complete")
    print("=" * 70)
    print(f"  Stage 1 [{s1['algorithm']}]: AUC-PR test={s1['auc_pr_test']} | "
          f"threshold={s1['threshold']} | "
          f"P_test={s1['test_precision']} R_test={s1['test_recall']}")
    if s2:
        print(f"  Stage 2 [{s2['algorithm']}]: AUC-PR test={s2['auc_pr_test']} | "
              f"threshold={s2['threshold']} | n_test={s2['n_test']}")
    else:
        print("  Stage 2: skipped (insufficient win samples)")
    print(f"  Models:  {output_dir}/dip_model_stage{{1,2}}.pkl")
    print(f"  Report:  {output_dir}/ml_report.json")
    print("=" * 70)
    print("\nTop-10 features (Stage 1, weighted ensemble importance):")
    for fi in s1.get("feature_importance", [])[:10]:
        bar = "█" * max(1, int(fi["importance"] * 80))
        print(f"  {fi['feature']:<25} {bar} ({fi['importance']:.4f})")
    print()

    return {"status": "ok", "stage1": s1, "stage2": s2}


# ─────────────────────────────────────────────────────────────────────────────
# Production prediction (used by ml_predictor as a fallback if needed)
# ─────────────────────────────────────────────────────────────────────────────

def predict_dip(
    features: dict,
    model_s1_path: Optional[Path] = None,
    model_s2_path: Optional[Path] = None,
) -> dict:
    """
    Cascade inference. Reads bundle, supports both v1 and v2 artefact formats.

    Applies VIX-aware threshold if present in the bundle.
    """
    s1_path = model_s1_path or _MODEL_S1
    s2_path = model_s2_path or _MODEL_S2

    result = {
        "stage1_label":     "NOT-WIN",
        "stage1_proba":     0.0,
        "stage1_confident": False,
        "stage2_label":     None,
        "stage2_proba":     None,
        "ml_verdict":       "🤖 ML: sem sinal",
        "models_loaded":    False,
    }
    if not s1_path.exists():
        result["ml_verdict"] = "🤖 ML: modelo não treinado ainda"
        return result

    try:
        art1 = joblib.load(s1_path)
    except Exception as e:
        result["ml_verdict"] = f"🤖 ML: erro ao carregar modelo ({e})"
        return result

    result["models_loaded"] = True
    feat_names = art1.get("feature_columns") or art1.get("feature_names")
    if not feat_names:
        result["ml_verdict"] = "🤖 ML: bundle sem feature names"
        return result

    # Ensure derived features are computed
    feats = dict(features)
    add_derived_features(feats)
    row = np.array([float(feats.get(f, 0.0)) for f in feat_names], dtype=float).reshape(1, -1)

    clf1 = art1.get("model") or art1.get("classifier")
    proba1 = float(clf1.predict_proba(row)[0, 1])
    threshold = float(art1.get("threshold", 0.5))

    # VIX-aware dynamic threshold
    vix_thresholds = art1.get("vix_regime_thresholds") or {}
    if vix_thresholds:
        vix = float(feats.get("vix", 20.0))
        for regime, (lo, hi) in _VIX_REGIMES.items():
            if lo <= vix < hi and regime in vix_thresholds:
                t = vix_thresholds[regime].get("threshold")
                if t is not None and not vix_thresholds[regime].get("fallback"):
                    threshold = float(t)
                break

    result["stage1_proba"]     = round(proba1, 4)
    result["stage1_confident"] = proba1 >= threshold

    if proba1 < threshold:
        result["stage1_label"] = "NOT-WIN"
        result["ml_verdict"] = (
            f"🤖 ML: 🔴 NOT-WIN — confiança {proba1*100:.0f}% "
            f"(threshold {threshold*100:.0f}%)"
        )
        return result

    result["stage1_label"] = "WIN"
    grade_label, grade_proba = "WIN", None
    if s2_path.exists():
        try:
            art2 = joblib.load(s2_path)
            cols2 = art2.get("feature_columns") or art2.get("feature_names")
            row2 = np.array([float(feats.get(f, 0.0)) for f in cols2], dtype=float).reshape(1, -1)
            clf2 = art2.get("model") or art2.get("classifier")
            proba2 = float(clf2.predict_proba(row2)[0, 1])
            t2 = float(art2.get("threshold", 0.5))
            grade_label = "WIN_40" if proba2 >= t2 else "WIN_20"
            grade_proba = round(proba2, 4)
        except Exception as e:
            logging.warning(f"[predict] Stage 2 failed: {e}")

    result["stage2_label"] = grade_label if grade_label in ("WIN_40", "WIN_20") else None
    result["stage2_proba"] = grade_proba

    conf_pct = f"{proba1*100:.0f}%"
    if grade_label == "WIN_40":
        result["ml_verdict"] = f"🤖 ML: 🟢 WIN_40 — confiança {conf_pct} (home-run potencial)"
    elif grade_label == "WIN_20":
        result["ml_verdict"] = f"🤖 ML: ✅ WIN_20 — confiança {conf_pct} (retorno sólido esperado)"
    else:
        result["ml_verdict"] = f"🤖 ML: ✅ WIN — confiança {conf_pct}"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DipRadar ML Trainer v2")
    parser.add_argument("--parquet",      type=Path, default=None,
                        help="Caminho para o Parquet de treino (obrigatório se não --report)")
    parser.add_argument("--output-dir",   type=Path, default=None,
                        help="Directório onde guardar .pkl + ml_report.json (default: /data ou /tmp)")
    parser.add_argument("--report",       action="store_true",
                        help="Imprime ml_report.json existente")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Valida o pipeline de dados sem treinar")
    parser.add_argument("--algos",        type=str, default="rf,xgb,lgbm",
                        help="Lista de algoritmos do ensemble (rf,xgb,lgbm)")
    parser.add_argument("--min-precision", type=float, default=_DEFAULT_MIN_PRECISION,
                        help=f"Precisão mínima para selecção de threshold (default {_DEFAULT_MIN_PRECISION})")
    parser.add_argument("--beta",         type=float, default=_DEFAULT_BETA,
                        help="F-beta para selecção de threshold (default 1.0)")
    parser.add_argument("--half-life",    type=float, default=_DEFAULT_HALF_LIFE_YEARS,
                        help="Half-life em anos para sample weights (default 3)")
    parser.add_argument("--pos-weight",   type=float, default=_DEFAULT_POS_WEIGHT,
                        help="Multiplicador de peso para classe positiva (default 1.5)")
    parser.add_argument("--train-end",    type=str, default=_DEFAULT_TRAIN_END,
                        help=f"Fim cronológico do treino (default {_DEFAULT_TRAIN_END})")
    parser.add_argument("--cal-end",      type=str, default=_DEFAULT_CAL_END,
                        help=f"Fim cronológico da calibração (default {_DEFAULT_CAL_END})")
    parser.add_argument("--row-split",    action="store_true",
                        help="Usa --train-frac/--cal-frac em vez de datas")
    parser.add_argument("--train-frac",   type=float, default=_DEFAULT_TRAIN_FRAC)
    parser.add_argument("--cal-frac",     type=float, default=_DEFAULT_CAL_FRAC)
    parser.add_argument("--min-year",     type=int, default=_DEFAULT_MIN_YEAR,
                        help=f"Ano mínimo (default {_DEFAULT_MIN_YEAR}; usa --no-min-year para desligar)")
    parser.add_argument("--no-min-year",  action="store_true",
                        help="Não filtra por min_year (mantém pré-2014 sintético)")
    parser.add_argument("--no-clean",     action="store_true",
                        help="Desliga limpeza (mantém duplicados, outliers e pré-2014)")
    args = parser.parse_args()

    out_dir = args.output_dir or _DATA_DIR

    if args.report:
        print_report(out_dir)
        raise SystemExit(0)

    if not args.parquet:
        parser.error("--parquet is required (unless --report)")

    algos = [a.strip().lower() for a in args.algos.split(",") if a.strip()]

    train_all(
        parquet_path=args.parquet,
        output_dir=out_dir,
        algos=algos,
        min_precision=args.min_precision,
        beta=args.beta,
        half_life_years=args.half_life,
        pos_weight=args.pos_weight,
        train_end=None if args.row_split else args.train_end,
        cal_end=None if args.row_split else args.cal_end,
        train_frac=args.train_frac,
        cal_frac=args.cal_frac,
        min_year=None if args.no_min_year or args.no_clean else args.min_year,
        drop_outliers=not args.no_clean,
        dry_run=args.dry_run,
    )
