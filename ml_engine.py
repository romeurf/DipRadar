"""
ml_engine.py — DipPredictor Ensemble (DipRadar ML Core)

Architecture (3-component ensemble):

  Model A — The Strategist  (XGBoost + Platt Scaling, prefit=True)
    Input : 16 features from ml_features.FEATURE_COLUMNS
    Target: label_win (int 1/0) — recovery ≥15% in 60 calendar days
    Output: win_prob (float [0,1]) — calibrated confidence for Telegram

  Model B — The Tactician  (LightGBM Regressor)
    Input : same 16 features
    Target: label_further_drop (float, clipped to [-30, 0]) — max additional
            % drop before recovery begins. Used to compute buy_target.
    Output: further_drop_pct (float ≤ 0)

  Historical Oracle  (KNN, k=20, no ML)
    Input : (win_prob, macro_score_normalised) of the current prediction
    Lookup: 20 nearest historical dips in the same 2D space
    Output: sell_target_pct (p75 of returns in neighbourhood)
            hold_days (median of hold days in neighbourhood)

Temporal split discipline (zero leakage guarantee):
  70% oldest rows → XGBoost train
  30% newest rows → Platt calibration set (prefit=True)
  Both Model B and Oracle use the same 100% for training.

SHAP extraction discipline:
  CalibratedClassifierCV wraps the XGBoost estimator — shap.TreeExplainer
  cannot receive the wrapper directly. extract_xgb_base() navigates the
  internal structure safely and returns the raw XGBClassifier.
  extract_shap_top3() uses that raw model and computes SHAP×Δfeature to
  identify the Top 3 drivers of probability change between alert day
  and monitoring day. Used by position_monitor.py for thesis deterioration
  explanations in Telegram alerts.

Serialisation:
  Single .pkl via joblib:
    { 'model_a': CalibratedClassifierCV, 'model_b': LGBMRegressor,
      'oracle_table': DataFrame, 'trained_at': ISO timestamp,
      'n_train': int, 'feature_columns': list }

Public API:
  train_ensemble(df_train)            → metrics dict, saves bundle
  predict_dip(feature_row, price, ticker) → DipPrediction dataclass
  load_predictor()                    → cached bundle dict
  extract_xgb_base(model_a)           → raw XGBClassifier (SHAP-compatible)
  extract_shap_top3(bundle,           → list[tuple[feature_name, delta_shap]]
                    row_alert,            Top 3 drivers of win_prob change
                    row_today)
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("LIGHTGBM_VERBOSITY", "-1")

try:
    import xgboost as xgb
    from lightgbm import LGBMRegressor
except ImportError as e:
    raise ImportError(
        "ml_engine requires xgboost and lightgbm. "
        "Run: pip install xgboost lightgbm"
    ) from e

from ml_features import FEATURE_COLUMNS, LABEL_COLUMNS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = Path(os.getenv("DIPR_MODEL_PATH", "models/dip_predictor.pkl"))
MIN_TRAIN_SAMPLES    = 40
FURTHER_DROP_CLIP_MIN = -30.0
FURTHER_DROP_CLIP_MAX =   0.0
ORACLE_K             = 20
ORACLE_MACRO_WEIGHT  = 1.0
BUY_TARGET_FACTOR    = 0.5
BUY_TARGET_CAP       = -0.05
SELL_TARGET_PERCENTILE = 75


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DipPrediction:
    """
    Full output of predict_dip().
    All monetary values in same currency as current_price.
    Percentages are raw floats (e.g. -4.2 means -4.2%).
    """
    ticker:               str
    current_price:        float
    win_prob:             float   # calibrated [0,1]
    further_drop_pct:     float   # ≤0
    buy_target:           float
    sell_target:          float
    hold_days:            int
    expected_return_pct:  float
    oracle_k_used:        int
    prediction_ts:        str
    model_trained_at:     str


# ─────────────────────────────────────────────────────────────────────────────
# SHAP — safe extraction through CalibratedClassifierCV
# ─────────────────────────────────────────────────────────────────────────────

def extract_xgb_base(model_a: CalibratedClassifierCV) -> "xgb.XGBClassifier":
    """
    Extract the raw XGBClassifier from inside a CalibratedClassifierCV.

    The SHAP library cannot receive a CalibratedClassifierCV directly —
    shap.TreeExplainer requires the underlying tree model.

    Internal structure of CalibratedClassifierCV (sklearn ≥1.2, cv='prefit'):
      model_a
        .calibrated_classifiers_        # list[_CalibratedClassifier] (len=1 for prefit)
          [0]
            .estimator                  # the raw XGBClassifier ← this is what we want

    Fallback chain (handles sklearn version differences):
      1. .calibrated_classifiers_[0].estimator      (sklearn ≥1.2)
      2. .calibrated_classifiers_[0].base_estimator (sklearn <1.2 compat)
      3. .estimator                                  (direct estimator attr)

    Raises ValueError if none of the paths resolve to an XGBClassifier.
    """
    # Path 1: sklearn ≥1.2 — standard production path
    try:
        base = model_a.calibrated_classifiers_[0].estimator
        if isinstance(base, xgb.XGBClassifier):
            return base
    except (AttributeError, IndexError):
        pass

    # Path 2: sklearn <1.2 compatibility
    try:
        base = model_a.calibrated_classifiers_[0].base_estimator
        if isinstance(base, xgb.XGBClassifier):
            return base
    except (AttributeError, IndexError):
        pass

    # Path 3: direct .estimator on the wrapper (edge case)
    try:
        base = model_a.estimator
        if isinstance(base, xgb.XGBClassifier):
            return base
    except AttributeError:
        pass

    raise ValueError(
        "extract_xgb_base: could not find raw XGBClassifier inside "
        "CalibratedClassifierCV. Inspect model_a.calibrated_classifiers_ "
        "manually and update the fallback chain."
    )


def extract_shap_top3(
    bundle: dict,
    row_alert: list[float],
    row_today: list[float],
    n_top: int = 3,
) -> list[tuple[str, float]]:
    """
    Compute the Top N features driving the change in win_prob between
    the alert day and today. Used by position_monitor.py to explain
    thesis deterioration or improvement in Telegram messages.

    Algorithm:
      1. Extract raw XGBClassifier from CalibratedClassifierCV
      2. Compute SHAP values for BOTH rows (alert and today)
      3. Δshap_i = shap_today_i - shap_alert_i  (signed change per feature)
      4. Sort by |delta_shap| descending, return top N

    The sign of delta_shap tells the story:
      Negative delta: feature pulled win_prob DOWN (thesis deteriorating)
      Positive delta: feature pulled win_prob UP   (thesis improving)

    Parameters
    ----------
    bundle    : dict          Loaded model bundle (from load_predictor())
    row_alert : list[float]   16-feature vector from the original alert day
    row_today : list[float]   16-feature vector from today's monitoring run
    n_top     : int           Number of top drivers to return (default 3)

    Returns
    -------
    list of (feature_name, delta_shap) tuples, sorted by |delta_shap| desc.
    Returns empty list if shap is not installed or extraction fails.

    Example output:
      [("fcf_yield",    -0.082),   # FCF yield fell → dragged win_prob down
       ("rsi_14",       -0.041),   # RSI rose (less oversold) → negative
       ("macro_score",  +0.031)]   # macro improved slightly → positive
    """
    try:
        import shap  # optional dependency — graceful fallback if not installed
    except ImportError:
        logger.warning("extract_shap_top3: 'shap' not installed. Run: pip install shap")
        return []

    try:
        xgb_base = extract_xgb_base(bundle["model_a"])
        feature_cols = bundle["feature_columns"]

        # Build (2, N_FEATURES) matrix: row 0 = alert, row 1 = today
        X = np.array([row_alert, row_today], dtype=float)

        explainer   = shap.TreeExplainer(xgb_base)
        shap_values = explainer.shap_values(X)  # shape: (2, N_FEATURES)

        # Δshap = today - alert (signed change in SHAP contribution per feature)
        delta_shap = shap_values[1] - shap_values[0]

        # Sort by absolute delta descending
        sorted_indices = np.argsort(np.abs(delta_shap))[::-1]
        top_indices    = sorted_indices[:n_top]

        result = [
            (feature_cols[i], float(delta_shap[i]))
            for i in top_indices
        ]

        logger.debug(
            f"extract_shap_top3: top {n_top} drivers: "
            + ", ".join(f"{name}={val:+.4f}" for name, val in result)
        )
        return result

    except Exception as e:
        logger.warning(f"extract_shap_top3: failed with error: {e}")
        return []


def format_shap_drivers(drivers: list[tuple[str, float]]) -> str:
    """
    Format the SHAP top-3 drivers into a readable Telegram block.
    Called by position_monitor.py when building the daily monitoring alert.

    Example output:
      ↳ fcf_yield ↓ -0.082  (maior impacto negativo)
      ↳ rsi_14 ↓ -0.041
      ↳ macro_score ↑ +0.031
    """
    if not drivers:
        return "  _Análise SHAP indisponível_"

    # Human-readable names
    _LABELS = {
        "macro_score":          "Regime macro",
        "vix":                  "VIX",
        "spy_drawdown_5d":      "SPY (5d)",
        "sector_drawdown_5d":   "Sector ETF (5d)",
        "fcf_yield":            "FCF Yield",
        "revenue_growth":       "Revenue growth",
        "gross_margin":         "Gross margin",
        "de_ratio":             "D/E ratio",
        "pe_vs_fair":           "P/E vs fair",
        "analyst_upside":       "Upside analístas",
        "quality_score":        "Quality score",
        "drop_pct_today":       "Queda do dia",
        "drawdown_52w":         "Drawdown 52w",
        "rsi_14":               "RSI 14",
        "atr_ratio":            "Volatilidade (ATR)",
        "volume_spike":         "Volume spike",
    }

    lines = []
    for i, (feat, delta) in enumerate(drivers):
        label    = _LABELS.get(feat, feat)
        arrow    = "↑" if delta > 0 else "↓"
        suffix   = "  (maior impacto)" if i == 0 else ""
        lines.append(f"  ↳ {label} {arrow} {delta:+.3f}{suffix}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_oracle_space(df: pd.DataFrame) -> np.ndarray:
    win_prob = df["win_prob_predicted"].values.astype(float)
    macro    = (df["macro_score"].values.astype(float) / 4.0) * ORACLE_MACRO_WEIGHT
    return np.column_stack([win_prob, macro])


def _safe_oracle_result(
    oracle_table: pd.DataFrame,
    win_prob: float,
    macro_score: float,
) -> tuple[float, int, float, int]:
    n = len(oracle_table)
    k = min(ORACLE_K, n)

    if k < 3:
        logger.warning(f"Oracle: only {n} samples — using global medians")
        p75 = float(oracle_table["label_win_return_pct"].quantile(0.75)) if n > 0 else 0.15
        med = int(oracle_table["label_hold_days"].median()) if n > 0 else 30
        return p75, med, p75, k

    X_oracle = _build_oracle_space(oracle_table)
    query    = np.array([[win_prob, (macro_score / 4.0) * ORACLE_MACRO_WEIGHT]])

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_oracle)
    _, indices = nn.kneighbors(query)
    neighbours = oracle_table.iloc[indices[0]]

    returns   = neighbours["label_win_return_pct"].values.astype(float)
    hold_days = neighbours["label_hold_days"].values.astype(float)

    p75 = float(np.percentile(returns, SELL_TARGET_PERCENTILE))
    med = int(np.median(hold_days))
    return p75, med, p75, k


def _build_oracle_table(df: pd.DataFrame, model_a: CalibratedClassifierCV) -> pd.DataFrame:
    X         = df[FEATURE_COLUMNS].values
    win_probs = model_a.predict_proba(X)[:, 1]
    oracle    = df[["macro_score", "label_win_return_pct", "label_hold_days"]].copy()
    oracle["win_prob_predicted"] = win_probs
    return oracle.reset_index(drop=True)


def _compute_buy_target(current_price: float, further_drop_pct: float) -> float:
    raw_offset  = further_drop_pct * BUY_TARGET_FACTOR / 100.0
    safe_offset = max(raw_offset, BUY_TARGET_CAP)
    return round(current_price * (1.0 + safe_offset), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_ensemble(df_train: pd.DataFrame) -> dict:
    """
    Train Model A + Model B, build Oracle table, persist bundle.

    df_train must be in chronological order (oldest first) and contain:
      FEATURE_COLUMNS + LABEL_COLUMNS + ['label_win_return_pct', 'label_hold_days']
    """
    required = set(FEATURE_COLUMNS + LABEL_COLUMNS + ["label_win_return_pct", "label_hold_days"])
    missing  = required - set(df_train.columns)
    if missing:
        raise ValueError(f"train_ensemble: missing columns: {missing}")

    n_total = len(df_train)
    if n_total < MIN_TRAIN_SAMPLES:
        raise ValueError(f"train_ensemble: need ≥{MIN_TRAIN_SAMPLES} samples, got {n_total}")

    logger.info(f"[train] Starting — {n_total} samples")

    # Temporal split (Decision 1)
    split_idx = int(n_total * 0.70)
    df_xgb    = df_train.iloc[:split_idx]
    df_calib  = df_train.iloc[split_idx:]

    X_xgb, y_xgb     = df_xgb[FEATURE_COLUMNS].values,   df_xgb["label_win"].values.astype(int)
    X_calib, y_calib = df_calib[FEATURE_COLUMNS].values,  df_calib["label_win"].values.astype(int)
    logger.info(f"[train] Split → XGB: {len(df_xgb)}, Calib: {len(df_calib)}")

    # Model A — XGBoost raw
    scale_pos_weight = float((y_xgb == 0).sum()) / max(float((y_xgb == 1).sum()), 1)
    xgb_raw = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", use_label_encoder=False,
        verbosity=0, random_state=42,
    )
    xgb_raw.fit(X_xgb, y_xgb)

    # Model A — Platt calibration (prefit=True, Decision 1)
    model_a = CalibratedClassifierCV(estimator=xgb_raw, method="sigmoid", cv="prefit")
    model_a.fit(X_calib, y_calib)

    win_probs_calib = model_a.predict_proba(X_calib)[:, 1]
    auc_calib    = float(roc_auc_score(y_calib, win_probs_calib))
    brier_calib  = float(brier_score_loss(y_calib, win_probs_calib))
    logger.info(f"[train] Model A → AUC={auc_calib:.4f}  Brier={brier_calib:.4f}")

    # Verify SHAP extraction works before persisting
    try:
        extract_xgb_base(model_a)
        logger.info("[train] SHAP extraction path — verified ✓")
    except ValueError as e:
        logger.error(f"[train] SHAP extraction path broken: {e}")

    # Model B — LightGBM Regressor
    X_full = df_train[FEATURE_COLUMNS].values
    y_b    = np.clip(df_train["label_further_drop"].values.astype(float),
                     FURTHER_DROP_CLIP_MIN, FURTHER_DROP_CLIP_MAX)

    model_b = LGBMRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=10, verbose=-1, random_state=42,
    )
    model_b.fit(X_full, y_b)

    preds_b   = model_b.predict(X_full)
    lgbm_rmse = float(np.sqrt(np.mean((preds_b - y_b) ** 2)))
    logger.info(f"[train] Model B → RMSE={lgbm_rmse:.4f}%")

    # Oracle table
    oracle_table = _build_oracle_table(df_train, model_a)
    logger.info(f"[train] Oracle table built — {len(oracle_table)} rows")

    # Persist
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    trained_at = datetime.utcnow().isoformat()
    bundle = {
        "model_a":         model_a,
        "model_b":         model_b,
        "oracle_table":    oracle_table,
        "trained_at":      trained_at,
        "n_train":         n_total,
        "feature_columns": FEATURE_COLUMNS,
    }
    joblib.dump(bundle, MODEL_PATH)
    logger.info(f"[train] Bundle saved → {MODEL_PATH}")

    return {
        "n_train":     len(df_xgb),
        "n_calib":     len(df_calib),
        "auc_calib":   round(auc_calib, 4),
        "brier_calib": round(brier_calib, 4),
        "lgbm_rmse":   round(lgbm_rmse, 4),
        "trained_at":  trained_at,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

_CACHED_BUNDLE: dict | None = None


def load_predictor(force_reload: bool = False) -> dict:
    global _CACHED_BUNDLE
    if _CACHED_BUNDLE is None or force_reload:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train_ensemble() first."
            )
        _CACHED_BUNDLE = joblib.load(MODEL_PATH)
        logger.info(
            f"[load] Bundle loaded — trained_at={_CACHED_BUNDLE['trained_at']} "
            f"n_train={_CACHED_BUNDLE['n_train']}"
        )
    return _CACHED_BUNDLE


def predict_dip(
    feature_row: list[float],
    current_price: float,
    ticker: str = "???",
    bundle: Optional[dict] = None,
) -> DipPrediction:
    """
    Run the full 3-component ensemble on a single feature vector.
    feature_row: list[float] in FEATURE_COLUMNS order (from build_feature_row()).
    """
    if bundle is None:
        bundle = load_predictor()

    expected_cols = bundle["feature_columns"]
    if len(feature_row) != len(expected_cols):
        raise ValueError(
            f"predict_dip [{ticker}]: expected {len(expected_cols)} features, "
            f"got {len(feature_row)}"
        )

    X = np.array(feature_row, dtype=float).reshape(1, -1)

    # Model A
    win_prob = float(np.clip(bundle["model_a"].predict_proba(X)[0, 1], 0.0, 1.0))

    # Model B
    further_drop_pct = float(np.clip(
        bundle["model_b"].predict(X)[0],
        FURTHER_DROP_CLIP_MIN, FURTHER_DROP_CLIP_MAX
    ))

    # Buy target
    buy_target = _compute_buy_target(current_price, further_drop_pct)

    # Oracle
    macro_score = float(feature_row[expected_cols.index("macro_score")])
    sell_target_pct, hold_days, expected_return_pct, k_used = _safe_oracle_result(
        bundle["oracle_table"], win_prob, macro_score
    )
    sell_target = round(current_price * (1.0 + sell_target_pct), 4)

    if sell_target <= buy_target:
        logger.warning(
            f"[predict] [{ticker}] sell_target ≤ buy_target — bumping to buy+5%"
        )
        sell_target = round(buy_target * 1.05, 4)

    logger.info(
        f"[predict] [{ticker}] win_prob={win_prob:.3f} "
        f"further_drop={further_drop_pct:.1f}% "
        f"buy={buy_target:.4f} sell={sell_target:.4f} "
        f"hold={hold_days}d oracle_k={k_used}"
    )

    return DipPrediction(
        ticker=ticker, current_price=current_price,
        win_prob=win_prob, further_drop_pct=further_drop_pct,
        buy_target=buy_target, sell_target=sell_target,
        hold_days=hold_days, expected_return_pct=expected_return_pct,
        oracle_k_used=k_used,
        prediction_ts=datetime.utcnow().isoformat(),
        model_trained_at=bundle["trained_at"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Telegram formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_prediction_telegram(
    pred: DipPrediction,
    fundamentals: dict,
    dip_score: float,
) -> str:
    if pred.win_prob >= 0.80:
        badge = "🔥"
    elif pred.win_prob >= 0.60:
        badge = "⭐"
    else:
        badge = "📊"

    macro_labels = {0: "🔴 BEAR", 1: "🟠 WEAK", 2: "🟡 NEUTRAL", 3: "🟢 BULL", 4: "🚀 STRONG BULL"}
    macro_label  = macro_labels.get(int(fundamentals.get("macro_score", 2)), "🟡 NEUTRAL")

    pe           = fundamentals.get("pe")
    fcf_yield    = fundamentals.get("fcf_yield")
    rev_growth   = fundamentals.get("revenue_growth")
    gross_margin = fundamentals.get("gross_margin")
    de_ratio     = fundamentals.get("debt_equity")
    drawdown     = fundamentals.get("drawdown_from_high") or fundamentals.get("drawdown_52w")

    def _pct(v, d=1):
        if v is None or (isinstance(v, float) and math.isnan(v)): return "N/A"
        return f"{float(v)*100:.{d}f}%"

    def _val(v, fmt=".1f"):
        if v is None or (isinstance(v, float) and math.isnan(v)): return "N/A"
        return f"{float(v):{fmt}}"

    return "\n".join([
        f"{badge} *DIP ALERT — {pred.ticker}*  [Score: {dip_score:.0f}/100]",
        "",
        "📊 *Fundamentals*",
        f"  P/E: {_val(pe, '.1f')}x  |  FCF Yield: {_pct(fcf_yield)}",
        f"  Revenue Growth: {_pct(rev_growth)}  |  Gross Margin: {_pct(gross_margin)}",
        f"  D/E: {_val(de_ratio, '.0f')}  |  Drawdown 52w: {_val(drawdown, '.1f')}%",
        "",
        "📉 *Drawdown*",
        f"  Queda actual: {pred.further_drop_pct:.1f}%",
        f"  Modelo prevê queda adicional máx: {pred.further_drop_pct:.1f}%",
        "",
        f"🎯 *Recomendação ML*  (confiança: {pred.win_prob*100:.0f}%)",
        f"  💰 Comprar em: ${pred.buy_target:.2f}  (actual: ${pred.current_price:.2f})",
        f"  🎯 Vender em: ${pred.sell_target:.2f}",
        f"  ⏳ Holding estimado: {pred.hold_days} dias",
        f"  📈 Upside esperado: {pred.expected_return_pct*100:.1f}%",
        "",
        f"⚙️ *Macro:* {macro_label}  |  Oracle k={pred.oracle_k_used}",
        f"_Modelo treinado em: {pred.model_trained_at[:10]}_",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test  (python ml_engine.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    print("=== DipPredictor smoke test ===")
    np.random.seed(42)

    n = 120
    rows = []
    for i in range(n):
        row = {col: np.random.uniform(0.0, 1.0) for col in FEATURE_COLUMNS}
        row["macro_score"] = float(np.random.randint(0, 5))
        row["rsi_14"]      = float(np.random.uniform(10, 70))
        row["vix"]         = float(np.random.uniform(12, 40))
        row["de_ratio"]    = float(np.random.uniform(10, 150))
        row["pe_vs_fair"]  = float(np.random.uniform(0.3, 2.0))
        row["label_win"]            = int(np.random.binomial(1, 0.55))
        row["label_further_drop"]   = float(np.random.uniform(-20, 0))
        row["label_win_return_pct"] = float(np.random.uniform(0.05, 0.35))
        row["label_hold_days"]      = float(np.random.randint(10, 60))
        rows.append(row)

    df = pd.DataFrame(rows)

    print("--- Training ---")
    metrics = train_ensemble(df)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print()
    print("--- SHAP extraction test ---")
    bundle = load_predictor(force_reload=True)
    try:
        xgb_base = extract_xgb_base(bundle["model_a"])
        print(f"  XGB base extracted: {type(xgb_base).__name__} ✓")
    except ValueError as e:
        print(f"  SHAP extraction FAILED: {e}")

    # Simulate two feature rows (alert day vs today)
    row_alert = [float(np.random.uniform(0, 1)) for _ in FEATURE_COLUMNS]
    row_today  = [v * np.random.uniform(0.85, 1.15) for v in row_alert]
    row_alert[FEATURE_COLUMNS.index("macro_score")] = 2.0
    row_today[FEATURE_COLUMNS.index("macro_score")]  = 1.0  # macro deteriorated

    drivers = extract_shap_top3(bundle, row_alert, row_today)
    print(f"  Top 3 drivers: {drivers}")
    print()
    print("  Formatted:")
    print(format_shap_drivers(drivers))

    print()
    print("--- Inference ---")
    fake_row = list(row_alert)
    pred = predict_dip(fake_row, current_price=387.20, ticker="MSFT", bundle=bundle)
    print(f"  win_prob={pred.win_prob:.3f}  buy={pred.buy_target:.2f}  sell={pred.sell_target:.2f}  hold={pred.hold_days}d")

    print()
    mock_fund = {"pe": 24.1, "fcf_yield": 0.062, "revenue_growth": 0.123,
                 "gross_margin": 0.684, "debt_equity": 41.0,
                 "drawdown_from_high": -23.1, "macro_score": 2}
    print(format_prediction_telegram(pred, mock_fund, dip_score=76.3))
