"""
ml_trainer.py — Treino automático do modelo ML com janela rolling de 5 anos.

Lê directamente o alert_db.csv (sem Parquet, sem CLI, sem argumentos).
Remove automaticamente registos com mais de 5 anos de idade.
Treina Stage 1 (Porteiro) + Stage 2 (Sommelier) e guarda os .pkl em /data/.

API pública:
    auto_train_if_ready(send_fn=None) -> dict
        Chama-se de qualquer job do scheduler (ex: domingo 08:30).
        Só treina se houver dados suficientes (>= MIN_SAMPLES).
        Devolve dict com stats; envia resumo via send_fn (Telegram) se fornecida.

    force_train(algo="rf", send_fn=None) -> dict
        Força treino imediato independentemente do MIN_SAMPLES.
        Usado pelo comando /admin_train_ml do Telegram.
"""

from __future__ import annotations

import csv
import logging
import pickle
import warnings
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Callable

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Caminhos ──────────────────────────────────────────────────────────────────

_DATA_DIR = Path("/data") if Path("/data").exists() else Path("/tmp")
_DB_PATH  = _DATA_DIR / "alert_db.csv"
_PKL_S1   = _DATA_DIR / "dip_model_stage1.pkl"
_PKL_S2   = _DATA_DIR / "dip_model_stage2.pkl"

# ── Configuração ──────────────────────────────────────────────────────────────

ROLLING_YEARS   = 5         # janela de dados a manter (anos)
MIN_SAMPLES     = 60        # mínimo de linhas com outcome_label para treinar
MIN_WINS        = 15        # mínimo de wins (label WIN_40 ou WIN_20) para Stage 1
MIN_S2_SAMPLES  = 30        # mínimo de wins para Stage 2 ser útil
DEFAULT_ALGO    = "rf"      # rf | xgb | lgbm

# Colunas esperadas do alert_db.csv → nome canónico para treino
_COL_MAP = {
    # alert_db col  →  feature col esperada pelo ml_predictor
    "rsi":            "rsi",
    "drawdown_52w":   "drawdown_pct",
    "change_day_pct": "change_day_pct",
    "pe":             "pe_ratio",
    "fcf_yield":      "fcf_yield",
    "analyst_upside": "analyst_upside",
    "revenue_growth": "revenue_growth",
    "gross_margin":   "gross_margin",
    "debt_equity":    "debt_to_equity",
    "beta":           "beta",
    "spy_change":     "spy_change",
    "sector_etf_change": "sector_etf_change",
    "score":          "dip_score",
    "market_cap_b":   "market_cap_b",
}

FEATURE_COLS = list(dict.fromkeys(_COL_MAP.values()))  # ordem preservada, sem duplicados

TARGET_COL = "outcome_label"
TARGET_S1  = "target_s1"
TARGET_S2  = "target_s2"


# ── 1. Carregar e limpar o CSV ────────────────────────────────────────────────

def _load_csv() -> list[dict]:
    """Lê o alert_db.csv e devolve lista de dicts (todas as linhas)."""
    if not _DB_PATH.exists():
        logging.warning("[ml_trainer] alert_db.csv não encontrado em %s", _DB_PATH)
        return []
    try:
        with _DB_PATH.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        logging.error("[ml_trainer] Erro a ler CSV: %s", e)
        return []


def _save_csv(rows: list[dict]) -> None:
    """Reescreve o CSV com as linhas fornecidas."""
    if not rows:
        return
    try:
        fieldnames = list(rows[0].keys())
        with _DB_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        logging.error("[ml_trainer] Erro a reescrever CSV: %s", e)


def purge_old_rows() -> int:
    """
    Remove do CSV todos os registos com date_iso mais antigo do que
    (hoje - ROLLING_YEARS anos - 1 dia).
    Devolve o número de linhas eliminadas.
    """
    rows = _load_csv()
    if not rows:
        return 0

    cutoff: date = date.today() - timedelta(days=ROLLING_YEARS * 365 + 1)
    kept, dropped = [], 0

    for row in rows:
        date_str = row.get("date_iso", "")
        try:
            row_date = datetime.fromisoformat(date_str).date()
        except (ValueError, TypeError):
            kept.append(row)
            continue

        if row_date <= cutoff:
            dropped += 1
        else:
            kept.append(row)

    if dropped:
        _save_csv(kept)
        logging.info(
            "[ml_trainer] Purge: %d linhas eliminadas (anteriores a %s)", dropped, cutoff
        )
    return dropped


# ── 2. Converter CSV → arrays numpy ──────────────────────────────────────────

def _to_float(val) -> float:
    """Converte valor CSV (possivelmente str vazia) para float; NaN se falhar."""
    if val is None or val == "":
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def _build_dataset(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constrói X (features), y_s1 (WIN/NO_WIN) e y_s2 (WIN_40/WIN_20 | NaN).
    Só usa linhas com outcome_label válido.
    Devolve arrays em ordem cronológica.
    """
    valid = [
        r for r in rows
        if r.get(TARGET_COL) in ("WIN_40", "WIN_20", "NEUTRAL", "LOSS_15")
    ]

    # ordena por data para split temporal sem look-ahead
    valid.sort(key=lambda r: r.get("date_iso", ""))

    X, y_s1, y_s2 = [], [], []

    for r in valid:
        row_features = []
        for src_col, feat_col in _COL_MAP.items():
            row_features.append(_to_float(r.get(src_col)))

        label = r[TARGET_COL]
        s1    = 1 if label in ("WIN_40", "WIN_20") else 0
        s2    = (1 if label == "WIN_40" else 0) if s1 == 1 else np.nan

        X.append(row_features)
        y_s1.append(s1)
        y_s2.append(s2)

    X    = np.array(X, dtype=np.float32)
    y_s1 = np.array(y_s1, dtype=np.float32)
    y_s2 = np.array(y_s2, dtype=object)  # float | NaN

    return X, y_s1, y_s2


# ── 3. Construção do Pipeline sklearn ────────────────────────────────────────

def _build_pipeline(algo: str = "rf"):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps = [("imputer", SimpleImputer(strategy="median"))]
    if algo in ("rf", "lgbm"):
        steps.append(("scaler", StandardScaler()))

    if algo == "rf":
        clf = RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
    elif algo == "xgb":
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                eval_metric="logloss", verbosity=0,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            clf = GradientBoostingClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
    elif algo == "lgbm":
        try:
            from lightgbm import LGBMClassifier
            clf = LGBMClassifier(
                n_estimators=400, max_depth=8, learning_rate=0.05,
                class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
            )
        except ImportError:
            clf = RandomForestClassifier(
                n_estimators=400, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )
    else:
        raise ValueError(f"Algoritmo desconhecido: {algo!r} — usa rf | xgb | lgbm")

    steps.append(("clf", clf))
    return Pipeline(steps)


def _optimal_threshold(pipeline, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """F1-optimal threshold com Precision >= 0.40."""
    from sklearn.metrics import precision_recall_curve
    probs          = pipeline.predict_proba(X_test)[:, 1]
    prec, rec, thr = precision_recall_curve(y_test, probs)
    f1s            = 2 * prec * rec / (prec + rec + 1e-9)
    valid          = prec[:-1] >= 0.40
    best_idx       = np.where(valid, f1s[:-1], 0.0).argmax() if valid.any() else f1s[:-1].argmax()
    thr_val        = float(thr[best_idx])
    logging.info("[ml_trainer] Threshold ótimo: %.4f (F1=%.3f)", thr_val, f1s[best_idx])
    return thr_val


# ── 4. Treino Stage 1 ─────────────────────────────────────────────────────────

def _train_s1(X: np.ndarray, y: np.ndarray, algo: str) -> dict:
    from sklearn.metrics import average_precision_score

    split  = int(len(X) * 0.80)
    X_tr, y_tr = X[:split], y[:split].astype(int)
    X_te, y_te = X[split:], y[split:].astype(int)

    if algo == "xgb":
        ratio = max((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 1.0)
        pipeline = _build_pipeline(algo)
        pipeline.named_steps["clf"].set_params(scale_pos_weight=ratio)
    else:
        pipeline = _build_pipeline(algo)

    pipeline.fit(X_tr, y_tr)
    threshold = _optimal_threshold(pipeline, X_te, y_te) if len(X_te) >= 10 else 0.50

    probs  = pipeline.predict_proba(X_te)[:, 1]
    auc_pr = average_precision_score(y_te, probs) if len(set(y_te)) >= 2 else 0.0

    logging.info(
        "[ml_trainer] Stage 1 concluído — algo=%s auc_pr=%.4f thr=%.4f n=%d wins=%d",
        algo, auc_pr, threshold, len(X_tr), int(y_tr.sum()),
    )
    return {
        "model":           pipeline,
        "feature_columns": FEATURE_COLS,
        "threshold":       threshold,
        "algorithm":       algo,
        "auc_pr":          round(auc_pr, 4),
        "n_samples":       len(X_tr),
        "train_date":      datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ── 5. Treino Stage 2 ─────────────────────────────────────────────────────────

def _train_s2(X: np.ndarray, y_s1: np.ndarray, y_s2: np.ndarray, algo: str) -> dict | None:
    from sklearn.metrics import average_precision_score

    # Filtra só linhas WIN (s2 não-NaN)
    mask = ~np.isnan(y_s2.astype(float))
    if mask.sum() < MIN_S2_SAMPLES:
        logging.info("[ml_trainer] Stage 2 saltado — apenas %d wins", mask.sum())
        return None

    Xw  = X[mask]
    yw  = y_s2[mask].astype(float).astype(int)

    split   = int(len(Xw) * 0.80)
    X_tr    = Xw[:split];     y_tr = yw[:split]
    X_te    = Xw[split:] if split < len(Xw) else Xw[:5]
    y_te    = yw[split:] if split < len(yw) else yw[:5]

    pipeline = _build_pipeline(algo)
    pipeline.fit(X_tr, y_tr)

    auc_pr = 0.0
    if len(X_te) >= 5 and len(set(y_te)) >= 2:
        probs  = pipeline.predict_proba(X_te)[:, 1]
        auc_pr = average_precision_score(y_te, probs)

    logging.info(
        "[ml_trainer] Stage 2 concluído — algo=%s auc_pr=%.4f wins=%d",
        algo, auc_pr, len(Xw),
    )
    return {
        "model":           pipeline,
        "feature_columns": FEATURE_COLS,
        "threshold":       0.55,
        "algorithm":       algo,
        "auc_pr":          round(auc_pr, 4),
        "n_samples":       len(Xw),
        "train_date":      datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ── 6. Guardar bundles ────────────────────────────────────────────────────────

def _save(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info("[ml_trainer] Guardado: %s (%.1f KB)", path, path.stat().st_size / 1024)


# ── 7. Entry-points públicos ──────────────────────────────────────────────────

def _run_train(algo: str, send_fn: Callable | None) -> dict:
    """Lógica comum de treino (purge → load → train → save → notify)."""
    # 1. Purge dados antigos
    dropped = purge_old_rows()

    # 2. Carregar dados
    rows = _load_csv()
    if not rows:
        msg = "⚠️ alert_db.csv vazio ou inexistente — treino não iniciado."
        logging.warning("[ml_trainer] %s", msg)
        if send_fn:
            send_fn(f"🤖 *ML Auto-Trainer*\n{msg}")
        return {"status": "no_data", "dropped": dropped}

    # 3. Filtrar só linhas com outcome
    labeled = [r for r in rows if r.get(TARGET_COL) in ("WIN_40", "WIN_20", "NEUTRAL", "LOSS_15")]
    total   = len(rows)
    n_lab   = len(labeled)
    n_wins  = sum(1 for r in labeled if r.get(TARGET_COL) in ("WIN_40", "WIN_20"))

    if n_lab < MIN_SAMPLES or n_wins < MIN_WINS:
        msg = (
            f"⏳ Dados insuficientes para treinar:\n"
            f"  Classificados: *{n_lab}* (mín {MIN_SAMPLES})\n"
            f"  Wins: *{n_wins}* (mín {MIN_WINS})\n"
            f"  Total registos: {total}"
        )
        logging.info("[ml_trainer] %s", msg.replace("*", ""))
        if send_fn:
            send_fn(f"🤖 *ML Auto-Trainer*\n{msg}")
        return {"status": "insufficient_data", "n_labeled": n_lab, "n_wins": n_wins, "dropped": dropped}

    # 4. Construir dataset
    X, y_s1, y_s2 = _build_dataset(labeled)

    # 5. Treinar Stage 1
    bundle_s1 = _train_s1(X, y_s1, algo)
    _save(bundle_s1, _PKL_S1)

    # 6. Treinar Stage 2 (se dados suficientes)
    bundle_s2 = _train_s2(X, y_s1, y_s2, algo)
    s2_status = "não treinado"
    if bundle_s2:
        _save(bundle_s2, _PKL_S2)
        s2_status = f"AUC-PR {bundle_s2['auc_pr']:.4f}"

    # 7. Recarrega o predictor em memória
    try:
        from ml_predictor import _load_models
        _load_models(force=True)
        logging.info("[ml_trainer] ml_predictor recarregado após treino.")
    except Exception as e:
        logging.warning("[ml_trainer] Falha ao recarregar ml_predictor: %s", e)

    # 8. Notifica via Telegram
    if send_fn:
        lines = [
            f"🤖 *ML Auto-Trainer — Treino concluído*",
            f"_{datetime.now().strftime('%d/%m/%Y %H:%M')}_",
            "",
            f"*📊 Dados usados:* {n_lab} alertas classificados",
            f"*🗓️ Janela:* últimos {ROLLING_YEARS} anos",
            f"*🗑️ Registos eliminados (>5 anos):* {dropped}",
            "",
            f"*Stage 1 (Porteiro)*",
            f"  Algoritmo: `{algo.upper()}`",
            f"  AUC-PR: *{bundle_s1['auc_pr']:.4f}*",
            f"  Threshold: *{bundle_s1['threshold']:.4f}*",
            f"  Amostras treino: {bundle_s1['n_samples']}",
            f"  Wins: {n_wins} ({n_wins/n_lab*100:.0f}%)",
            "",
            f"*Stage 2 (Sommelier WIN40 vs WIN20)*",
            f"  {s2_status}",
            "",
            f"_Modelo activo imediatamente — próximos alertas já usam o novo modelo_",
        ]
        send_fn("\n".join(lines))

    return {
        "status":    "trained",
        "algo":      algo,
        "n_labeled": n_lab,
        "n_wins":    n_wins,
        "dropped":   dropped,
        "auc_pr_s1": bundle_s1["auc_pr"],
        "threshold": bundle_s1["threshold"],
        "s2":        s2_status,
    }


def auto_train_if_ready(send_fn: Callable | None = None, algo: str = DEFAULT_ALGO) -> dict:
    """
    Ponto de entrada chamado pelo scheduler semanal.
    Só treina se MIN_SAMPLES estiver satisfeito.
    Purge automático dos dados com mais de ROLLING_YEARS anos.
    """
    logging.info("[ml_trainer] auto_train_if_ready() iniciado (algo=%s)", algo)
    try:
        return _run_train(algo=algo, send_fn=send_fn)
    except Exception as e:
        logging.error("[ml_trainer] Erro inesperado: %s", e, exc_info=True)
        if send_fn:
            send_fn(f"🤖 *ML Auto-Trainer — Erro*\n`{e}`")
        return {"status": "error", "error": str(e)}


def force_train(algo: str = DEFAULT_ALGO, send_fn: Callable | None = None) -> dict:
    """
    Força treino imediato via comando Telegram /admin_train_ml.
    Não verifica MIN_SAMPLES — usa todos os dados disponíveis.
    """
    logging.info("[ml_trainer] force_train() iniciado (algo=%s)", algo)
    try:
        return _run_train(algo=algo, send_fn=send_fn)
    except Exception as e:
        logging.error("[ml_trainer] Erro em force_train: %s", e, exc_info=True)
        if send_fn:
            send_fn(f"🤖 *ML Force Train — Erro*\n`{e}`")
        return {"status": "error", "error": str(e)}
