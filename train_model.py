"""
train_model.py — Chunk H: Laboratório de Machine Learning do DipRadar.

ARQUITECTURA: Classificador em Cascata de 2 Andares

  Andar 1 — Filtro binário (WIN vs. NOT-WIN)
    Maximiza Precision na classe positiva.
    Threshold calibrado por Precision-Recall curve.

  Andar 2 — Classificador de granulosidade (WIN_40 vs. WIN_20)
    Só é chamado quando o Andar 1 diz WIN.

CLI:
  python train_model.py                          # usa /data como data_dir
  python train_model.py --dry-run
  python train_model.py --report
  python train_model.py --live-only
  python train_model.py --precision 0.65

  # Colab: especificar parquet e output-dir
  python train_model.py \\
      --parquet /content/drive/MyDrive/DipRadar/ml_training_price.parquet \\
      --output-dir /content/drive/MyDrive/DipRadar

Outputs:
  <output-dir>/dip_model_stage1.pkl
  <output-dir>/dip_model_stage2.pkl
  <output-dir>/ml_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Caminhos (overridable via CLI) ────────────────────────────────────────────
_DATA_DIR  = Path("/data") if Path("/data").exists() else Path("/tmp")
_LIVE_DB   = _DATA_DIR / "alert_db.csv"
_HIST_DB   = _DATA_DIR / "hist_backtest.csv"
_MODEL_S1  = _DATA_DIR / "dip_model_stage1.pkl"
_MODEL_S2  = _DATA_DIR / "dip_model_stage2.pkl"
_REPORT    = _DATA_DIR / "ml_report.json"
_PARQUET   = None  # definido via --parquet

# ── Features ──────────────────────────────────────────────────────────────────
# Contrato alinhado com ml_features.FEATURE_COLUMNS (bootstrap_ml.py source of truth).
# train_model.py aceita AMBOS os contratos:
#   A) Parquet do bootstrap (16 features ml_features)
#   B) CSVs live/hist com features legacy
# A detecção é automática baseada nas colunas disponíveis.

_FEATURES_BOOTSTRAP = [
    "macro_score", "vix", "spy_drawdown_5d", "sector_drawdown_5d",
    "fcf_yield", "revenue_growth", "gross_margin", "de_ratio",
    "pe_vs_fair", "analyst_upside", "quality_score",
    "drop_pct_today", "drawdown_52w", "rsi_14", "atr_ratio", "volume_spike",
]

_FEATURES_LEGACY = [
    "score", "rsi", "drawdown_52w", "volume_ratio", "change_day_pct",
    "market_cap_b", "pe", "fcf_yield", "revenue_growth", "gross_margin",
    "debt_equity", "dividend_yield", "analyst_upside", "mfe_3m", "mae_3m",
]

_CATEGORICAL_FEATURES = ["category", "sector"]

_WIN_LABELS  = {"WIN_40", "WIN_20"}
_LOSE_LABELS = {"NEUTRAL", "LOSS_15"}
_MIN_SAMPLES_S1  = 30
_MIN_SAMPLES_S2  = 15
_SMOTE_THRESHOLD = 200


# ── 1. Preparação de dados ─────────────────────────────────────────────────────

def prepare_ml_data(
    live_only: bool = False,
    parquet_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Carrega dados de treino a partir de:
      1. Parquet do bootstrap_ml (preferido quando --parquet é passado)
      2. CSVs legacy (alert_db.csv + hist_backtest.csv)

    Retorna (X, y_raw, feature_names).
    """
    frames = []
    feature_names: list[str] = []

    # ── Modo Parquet (bootstrap_ml output) ───────────────────────────────────
    if parquet_path and Path(parquet_path).exists():
        logging.info(f"[ml_data] A carregar Parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        logging.info(f"[ml_data] Parquet: {len(df)} registos | colunas: {list(df.columns)}")

        # Detectar contrato de features disponíveis
        available = [f for f in _FEATURES_BOOTSTRAP if f in df.columns]
        if not available:
            raise ValueError(
                f"Parquet não tem features reconhecidas. "
                f"Colunas encontradas: {list(df.columns)}"
            )
        feature_names = available
        logging.info(f"[ml_data] Usando {len(feature_names)} features do bootstrap: {feature_names}")

        # Filtrar outcomes válidos
        valid_labels = list(_WIN_LABELS | _LOSE_LABELS)
        df = df[df["outcome_label"].isin(valid_labels)].copy()
        logging.info(f"[ml_data] Com outcome válido: {len(df)} registos")

        if df.empty:
            raise ValueError("Nenhuma linha com outcome_label válido no Parquet.")

        # Converter para float e imputar
        for col in feature_names:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        medians = df[feature_names].median()
        df[feature_names] = df[feature_names].fillna(medians)

        X     = df[feature_names].copy()
        y_raw = df["outcome_label"].copy()

        logging.info(
            f"[ml_data] Dataset final: {len(X)} amostras | "
            f"Dist: {y_raw.value_counts().to_dict()}"
        )
        return X, y_raw, feature_names

    # ── Modo CSV legacy ────────────────────────────────────────────────────────
    if _LIVE_DB.exists():
        try:
            df_live = pd.read_csv(_LIVE_DB, dtype=str)
            frames.append(df_live)
            logging.info(f"[ml_data] live: {len(df_live)} linhas")
        except Exception as e:
            logging.warning(f"[ml_data] Erro ao ler live DB: {e}")

    if not live_only and _HIST_DB.exists():
        try:
            df_hist = pd.read_csv(_HIST_DB, dtype=str)
            frames.append(df_hist)
            logging.info(f"[ml_data] hist: {len(df_hist)} linhas")
        except Exception as e:
            logging.warning(f"[ml_data] Erro ao ler hist DB: {e}")

    if not frames:
        raise FileNotFoundError(
            f"Nenhum dado encontrado. Passa --parquet ou garante CSVs em {_DATA_DIR}."
        )

    df = pd.concat(frames, ignore_index=True)
    logging.info(f"[ml_data] Total bruto: {len(df)} linhas")

    valid_labels = list(_WIN_LABELS | _LOSE_LABELS)
    df = df[df["outcome_label"].isin(valid_labels)].copy()

    if df.empty:
        raise ValueError("Nenhuma linha com outcome_label válido.")

    for col in _FEATURES_LEGACY:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    le_map: dict = {}
    for col in _CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna("Unknown").astype(str))
            le_map[col] = le
        else:
            df[col] = 0

    all_feat = _FEATURES_LEGACY + _CATEGORICAL_FEATURES
    numeric_present = [c for c in _FEATURES_LEGACY if c in df.columns]
    df = df.dropna(subset=numeric_present, how="all")
    medians = df[numeric_present].median()
    df[numeric_present] = df[numeric_present].fillna(medians)

    feature_names = all_feat
    X     = df[feature_names].copy()
    y_raw = df["outcome_label"].copy()

    logging.info(
        f"[ml_data] Dataset final (legacy): {len(X)} amostras | "
        f"Dist: {y_raw.value_counts().to_dict()}"
    )
    return X, y_raw, feature_names


# ── 2. AutoSelect ──────────────────────────────────────────────────────────────

def _get_candidates(n_samples: int) -> list[tuple[str, Any]]:
    candidates = [
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_leaf=2,
                class_weight="balanced", random_state=42, n_jobs=-1,
            ),
        )
    ]
    try:
        from xgboost import XGBClassifier
        candidates.append((
            "XGBoost",
            XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                random_state=42, verbosity=0, n_jobs=-1,
            ),
        ))
    except ImportError:
        logging.warning("[autoselect] XGBoost nao disponivel.")

    try:
        from lightgbm import LGBMClassifier
        candidates.append((
            "LightGBM",
            LGBMClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
                random_state=42, verbosity=-1, n_jobs=-1,
            ),
        ))
    except ImportError:
        logging.warning("[autoselect] LightGBM nao disponivel.")

    return candidates


def _autoselect(
    X: np.ndarray,
    y_bin: np.ndarray,
    cv: StratifiedKFold,
    use_smote: bool,
) -> tuple[str, Any, float]:
    candidates = _get_candidates(len(X))
    best_name, best_model, best_score = None, None, -1.0

    for name, clf in candidates:
        aps_scores = []
        for train_idx, val_idx in cv.split(X, y_bin):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y_bin[train_idx], y_bin[val_idx]

            if use_smote and sum(y_tr) >= 2 and sum(y_tr == 0) >= 2:
                try:
                    from imblearn.over_sampling import SMOTE
                    X_tr, y_tr = SMOTE(
                        random_state=42,
                        k_neighbors=min(3, sum(y_tr) - 1)
                    ).fit_resample(X_tr, y_tr)
                except Exception as e:
                    logging.debug(f"[smote] {e}")

            try:
                clf.fit(X_tr, y_tr)
                proba = (
                    clf.predict_proba(X_val)[:, 1]
                    if hasattr(clf, "predict_proba")
                    else clf.decision_function(X_val)
                )
                aps_scores.append(average_precision_score(y_val, proba))
            except Exception as e:
                logging.warning(f"[autoselect] {name} fold falhou: {e}")

        if aps_scores:
            mean_aps = float(np.mean(aps_scores))
            logging.info(f"[autoselect] {name}: AUC-PR = {mean_aps:.4f}")
            if mean_aps > best_score:
                best_score = mean_aps
                best_name  = name
                best_model = clf

    if best_model is None:
        raise RuntimeError("AutoSelect falhou.")

    logging.info(f"[autoselect] Vencedor: {best_name} (AUC-PR={best_score:.4f})")
    return best_name, best_model, best_score


# ── 3. Calibração do Threshold ─────────────────────────────────────────────────

def _calibrate_threshold(
    clf,
    X: np.ndarray,
    y_bin: np.ndarray,
    min_precision: float = 0.70,
) -> float:
    if not hasattr(clf, "predict_proba"):
        return 0.5

    proba = clf.predict_proba(X)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_bin, proba)

    candidates = [
        (p, r, t)
        for p, r, t in zip(prec[:-1], rec[:-1], thresholds)
        if p >= min_precision
    ]

    if candidates:
        best = max(candidates, key=lambda x: x[1])
        threshold = float(best[2])
        logging.info(
            f"[threshold] Precision={best[0]:.2f} | Recall={best[1]:.2f} | "
            f"Threshold={threshold:.4f}"
        )
    else:
        f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-9)
        idx = int(np.argmax(f1))
        threshold = float(thresholds[idx])
        logging.warning(
            f"[threshold] Sem threshold com precision>={min_precision:.0%}. "
            f"Usando F1-max={threshold:.4f}"
        )

    return threshold


# ── 4. Treino Andar 1 ──────────────────────────────────────────────────────────

def train_stage1(
    X: pd.DataFrame,
    y_raw: pd.Series,
    feature_names: list[str],
    min_precision: float = 0.70,
    output_dir: Path | None = None,
) -> dict:
    model_path = (output_dir or _DATA_DIR) / "dip_model_stage1.pkl"

    X_arr = X.values.astype(float)
    y_bin = (y_raw.isin(_WIN_LABELS)).astype(int).values

    n_pos = y_bin.sum()
    n_neg = len(y_bin) - n_pos
    logging.info(f"[stage1] WIN={n_pos} | NOT-WIN={n_neg} | Total={len(y_bin)}")

    if len(y_bin) < _MIN_SAMPLES_S1:
        raise ValueError(f"Insuficiente para Andar 1: {len(y_bin)} amostras.")

    use_smote = len(y_bin) < _SMOTE_THRESHOLD
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_arr)

    best_name, best_clf, best_aps = _autoselect(X_sc, y_bin, cv, use_smote)

    X_final, y_final = X_sc.copy(), y_bin.copy()
    if use_smote and n_pos >= 2 and n_neg >= 2:
        try:
            from imblearn.over_sampling import SMOTE
            X_final, y_final = SMOTE(
                random_state=42, k_neighbors=min(3, n_pos - 1)
            ).fit_resample(X_final, y_final)
            logging.info(f"[stage1] SMOTE: {len(y_final)} amostras")
        except Exception as e:
            logging.warning(f"[stage1] SMOTE falhou: {e}")

    best_clf.fit(X_final, y_final)
    threshold = _calibrate_threshold(best_clf, X_sc, y_bin, min_precision)

    y_pred_cal = (
        (best_clf.predict_proba(X_sc)[:, 1] >= threshold).astype(int)
        if hasattr(best_clf, "predict_proba")
        else best_clf.predict(X_sc)
    )
    report_str = classification_report(
        y_bin, y_pred_cal, target_names=["NOT-WIN", "WIN"], zero_division=0,
    )
    logging.info(f"[stage1] Classification Report (threshold={threshold:.4f}):\n{report_str}")

    feat_imp = _get_feature_importance(best_clf, feature_names)

    artefact = {
        "scaler":        scaler,
        "classifier":    best_clf,
        "threshold":     threshold,
        "feature_names": feature_names,
        "label_map":     {0: "NOT-WIN", 1: "WIN"},
        "algorithm":     best_name,
        "trained_at":    datetime.now().isoformat(),
        "n_samples":     int(len(y_bin)),
        "n_pos":         int(n_pos),
        "n_neg":         int(n_neg),
        "smote_used":    use_smote,
    }
    joblib.dump(artefact, model_path)
    logging.info(f"[stage1] Modelo guardado em {model_path}")

    return {
        "algorithm":      best_name,
        "auc_pr":         round(best_aps, 4),
        "threshold":      round(threshold, 4),
        "n_samples":      int(len(y_bin)),
        "n_win":          int(n_pos),
        "n_not_win":      int(n_neg),
        "smote_used":     use_smote,
        "feature_importance": feat_imp,
        "classification_report": report_str,
    }


# ── 5. Treino Andar 2 ──────────────────────────────────────────────────────────

def train_stage2(
    X: pd.DataFrame,
    y_raw: pd.Series,
    feature_names: list[str],
    output_dir: Path | None = None,
) -> dict | None:
    model_path = (output_dir or _DATA_DIR) / "dip_model_stage2.pkl"

    mask  = y_raw.isin(_WIN_LABELS)
    X_win = X[mask]
    y_win = y_raw[mask]

    n_40 = (y_win == "WIN_40").sum()
    n_20 = (y_win == "WIN_20").sum()
    logging.info(f"[stage2] WIN_40={n_40} | WIN_20={n_20}")

    if len(y_win) < _MIN_SAMPLES_S2 or n_40 < 3 or n_20 < 3:
        logging.warning(
            f"[stage2] Insuficiente ({len(y_win)} amostras, "
            f"WIN_40={n_40}, WIN_20={n_20}). A ignorar."
        )
        return None

    y_bin2 = (y_win == "WIN_40").astype(int).values
    X_arr  = X_win.values.astype(float)
    use_smote = len(y_bin2) < _SMOTE_THRESHOLD
    cv = StratifiedKFold(n_splits=min(5, n_40, n_20), shuffle=True, random_state=42)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_arr)

    best_name, best_clf, best_aps = _autoselect(X_sc, y_bin2, cv, use_smote)

    X_final, y_final = X_sc.copy(), y_bin2.copy()
    if use_smote and n_40 >= 2 and n_20 >= 2:
        try:
            from imblearn.over_sampling import SMOTE
            X_final, y_final = SMOTE(
                random_state=42, k_neighbors=min(3, min(n_40, n_20) - 1)
            ).fit_resample(X_final, y_final)
        except Exception as e:
            logging.warning(f"[stage2] SMOTE falhou: {e}")

    best_clf.fit(X_final, y_final)

    report_str = classification_report(
        y_bin2, best_clf.predict(X_sc),
        target_names=["WIN_20", "WIN_40"], zero_division=0,
    )
    logging.info(f"[stage2] Classification Report:\n{report_str}")
    feat_imp = _get_feature_importance(best_clf, feature_names)

    artefact = {
        "scaler":        scaler,
        "classifier":    best_clf,
        "threshold":     0.5,
        "feature_names": feature_names,
        "label_map":     {0: "WIN_20", 1: "WIN_40"},
        "algorithm":     best_name,
        "trained_at":    datetime.now().isoformat(),
        "n_samples":     int(len(y_bin2)),
        "n_win40":       int(n_40),
        "n_win20":       int(n_20),
    }
    joblib.dump(artefact, model_path)
    logging.info(f"[stage2] Modelo guardado em {model_path}")

    return {
        "algorithm":     best_name,
        "auc_pr":        round(best_aps, 4),
        "n_samples":     int(len(y_bin2)),
        "n_win40":       int(n_40),
        "n_win20":       int(n_20),
        "classification_report": report_str,
        "feature_importance":    feat_imp,
    }


# ── 6. Feature Importance ──────────────────────────────────────────────────────

def _get_feature_importance(clf, feature_names: list[str]) -> list[dict]:
    try:
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            imp = np.abs(clf.coef_[0])
        else:
            return []
        pairs = sorted(zip(feature_names, imp.tolist()), key=lambda x: x[1], reverse=True)
        return [{"feature": f, "importance": round(float(v), 6)} for f, v in pairs[:10]]
    except Exception:
        return []


# ── 7. Predição em Produção ────────────────────────────────────────────────────

def predict_dip(
    features: dict,
    model_s1_path: Path | None = None,
    model_s2_path: Path | None = None,
) -> dict:
    s1_path = model_s1_path or _MODEL_S1
    s2_path = model_s2_path or _MODEL_S2

    result = {
        "stage1_label":      "NOT-WIN",
        "stage1_proba":      0.0,
        "stage1_confident":  False,
        "stage2_label":      None,
        "stage2_proba":      None,
        "ml_verdict":        "\U0001f916 ML: sem sinal",
        "models_loaded":     False,
    }

    if not s1_path.exists():
        result["ml_verdict"] = "\U0001f916 ML: modelo nao treinado ainda"
        return result

    try:
        art1 = joblib.load(s1_path)
    except Exception as e:
        logging.warning(f"[predict] Erro ao carregar Andar 1: {e}")
        result["ml_verdict"] = f"\U0001f916 ML: erro ao carregar modelo ({e})"
        return result

    result["models_loaded"] = True
    feat_names = art1["feature_names"]

    row = []
    for f in feat_names:
        val = features.get(f)
        row.append(float(val) if val is not None and val == val else np.nan)

    X_raw = np.nan_to_num(np.array(row, dtype=float).reshape(1, -1), nan=0.0)
    X_sc  = art1["scaler"].transform(X_raw)

    clf1      = art1["classifier"]
    threshold = art1["threshold"]

    proba1 = (
        float(clf1.predict_proba(X_sc)[0, 1])
        if hasattr(clf1, "predict_proba")
        else float(clf1.decision_function(X_sc)[0])
    )

    result["stage1_proba"]     = round(proba1, 4)
    result["stage1_confident"] = proba1 >= threshold

    if proba1 < threshold:
        result["stage1_label"] = "NOT-WIN"
        result["ml_verdict"] = (
            f"\U0001f916 ML: \U0001f534 NOT-WIN \u2014 confianca {proba1*100:.0f}% "
            f"(threshold {threshold*100:.0f}%)"
        )
        return result

    result["stage1_label"] = "WIN"

    grade_label, grade_proba = "WIN", None
    if s2_path.exists():
        try:
            art2   = joblib.load(s2_path)
            X_sc2  = art2["scaler"].transform(X_raw)
            clf2   = art2["classifier"]
            proba2 = (
                float(clf2.predict_proba(X_sc2)[0, 1])
                if hasattr(clf2, "predict_proba")
                else 0.5
            )
            grade_label = "WIN_40" if proba2 >= art2["threshold"] else "WIN_20"
            grade_proba = round(proba2, 4)
        except Exception as e:
            logging.warning(f"[predict] Andar 2 falhou: {e}")

    result["stage2_label"] = grade_label if grade_label in ("WIN_40", "WIN_20") else None
    result["stage2_proba"] = grade_proba

    conf_pct = f"{proba1*100:.0f}%"
    # Nota: sem backslash em f-strings — usar Unicode directo
    if grade_label == "WIN_40":
        result["ml_verdict"] = (
            f"\U0001f916 ML: \U0001f7e2 WIN_40 \u2014 confianca {conf_pct} "
            "(home-run potencial)"
        )
    elif grade_label == "WIN_20":
        result["ml_verdict"] = (
            f"\U0001f916 ML: \u2705 WIN_20 \u2014 confianca {conf_pct} "
            "(retorno solido esperado)"
        )
    else:
        result["ml_verdict"] = f"\U0001f916 ML: \u2705 WIN \u2014 confianca {conf_pct}"

    return result


# ── 8. Relatório ───────────────────────────────────────────────────────────────

def _save_report(stage1: dict, stage2: dict | None, output_dir: Path | None = None) -> None:
    report_path = (output_dir or _DATA_DIR) / "ml_report.json"
    report = {
        "trained_at": datetime.now().isoformat(),
        "stage1":     {k: v for k, v in stage1.items() if k != "classification_report"},
        "stage2":     {k: v for k, v in stage2.items() if k != "classification_report"} if stage2 else None,
    }
    try:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        logging.info(f"[report] Guardado em {report_path}")
    except Exception as e:
        logging.warning(f"[report] Erro: {e}")


def print_report(output_dir: Path | None = None) -> None:
    report_path = (output_dir or _DATA_DIR) / "ml_report.json"
    if not report_path.exists():
        print("Nenhum relatorio encontrado. Corre train_model.py primeiro.")
        return
    print(json.dumps(json.loads(report_path.read_text()), indent=2, ensure_ascii=False))


# ── 9. Entry-point ─────────────────────────────────────────────────────────────

def train_all(
    live_only: bool = False,
    dry_run: bool = False,
    min_precision: float = 0.70,
    parquet_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    logging.info("-" * 60)
    logging.info("DipRadar ML - Laboratorio de Treino")
    logging.info("-" * 60)

    X, y_raw, feature_names = prepare_ml_data(
        live_only=live_only,
        parquet_path=parquet_path,
    )

    if dry_run:
        dist = y_raw.value_counts().to_dict()
        logging.info(f"[dry_run] Dataset OK - {len(X)} amostras | dist={dist}")
        return {"dry_run": True, "n_samples": len(X), "distribution": dist}

    result_s1 = train_stage1(X, y_raw, feature_names, min_precision, output_dir)
    result_s2 = train_stage2(X, y_raw, feature_names, output_dir)

    _save_report(result_s1, result_s2, output_dir)

    print("\n" + "=" * 60)
    print("DipRadar ML - Treino Concluido")
    print("=" * 60)
    print(
        f"  Andar 1 [{result_s1['algorithm']}]: "
        f"AUC-PR={result_s1['auc_pr']} | "
        f"threshold={result_s1['threshold']} | "
        f"n={result_s1['n_samples']}"
    )
    if result_s2:
        print(
            f"  Andar 2 [{result_s2['algorithm']}]: "
            f"AUC-PR={result_s2['auc_pr']} | "
            f"n={result_s2['n_samples']}"
        )
    else:
        print("  Andar 2: ignorado (dados insuficientes)")
    print(f"  Modelos em: {output_dir or _DATA_DIR}")
    print("=" * 60)
    print("\nTop-5 features (Andar 1):")
    for fi in result_s1.get("feature_importance", [])[:5]:
        bar = "\u2588" * int(fi["importance"] * 40)
        print(f"  {fi['feature']:<22} {bar} ({fi['importance']:.4f})")
    print()

    return {"status": "ok", "stage1": result_s1, "stage2": result_s2}


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DipRadar ML Trainer")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Valida dados sem treinar")
    parser.add_argument("--report",     action="store_true",
                        help="Imprime relatorio do ultimo treino")
    parser.add_argument("--live-only",  action="store_true",
                        help="Usa so alert_db.csv (ignora hist)")
    parser.add_argument("--precision",  type=float, default=0.70,
                        help="Precision minima para threshold (default: 0.70)")
    parser.add_argument("--parquet",    type=Path, default=None, metavar="PATH",
                        help="Caminho para o Parquet do bootstrap_ml (preferido sobre CSVs)")
    parser.add_argument("--output-dir", type=Path, default=None, metavar="DIR",
                        help="Directorio de output para .pkl e ml_report.json")
    args = parser.parse_args()

    # Resolver output_dir
    out_dir = args.output_dir
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.report:
        print_report(out_dir)
    else:
        train_all(
            live_only=args.live_only,
            dry_run=args.dry_run,
            min_precision=args.precision,
            parquet_path=args.parquet,
            output_dir=out_dir,
        )
