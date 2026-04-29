"""
ml_predictor.py — Integração do modelo ML no scanner ao vivo.

Carrega dip_model_stage1.pkl (e opcionalmente stage2) do volume /data
e puntua cada dip detectado em tempo real.

API pública:
  ml_score(features: dict) -> MLResult
  is_model_ready() -> bool
  get_model_info() -> dict
  ml_badge(result: MLResult) -> str    # emoji + label para Telegram
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ── Caminhos ───────────────────────────────────────────────────────────────────

_DATA_DIR = Path("/data") if Path("/data").exists() else Path("/tmp")
_PKL_S1   = _DATA_DIR / "dip_model_stage1.pkl"
_PKL_S2   = _DATA_DIR / "dip_model_stage2.pkl"

# ── Cache em memória ───────────────────────────────────────────────────────────

_model_s1:    Any | None = None
_model_s2:    Any | None = None
_feat_cols:   list[str]  = []
_threshold:   float      = 0.50
_loaded_mtime: float     = 0.0


@dataclass
class MLResult:
    win_prob:    float       = 0.0    # probabilidade de WIN (stage 1)
    win40_prob:  float | None = None  # prob WIN_40 vs WIN_20 (stage 2, se disponível)
    label:       str         = "NO_MODEL"
    confidence:  str         = "–"
    model_ready: bool        = False
    threshold:   float       = 0.50
    features_used: list[str] = field(default_factory=list)


# ── Feature engineering ────────────────────────────────────────────────────────

_FEATURE_MAP: dict[str, str] = {
    # fundamentals
    "rsi":                   "rsi",
    "pe":                    "pe_ratio",
    "pe_ratio":              "pe_ratio",
    "pb":                    "pb_ratio",
    "pb_ratio":              "pb_ratio",
    "fcf_yield":             "fcf_yield",
    "revenue_growth":        "revenue_growth",
    "gross_margin":          "gross_margin",
    "debt_to_equity":        "debt_to_equity",
    "analyst_upside":        "analyst_upside",
    "drawdown_from_high":    "drawdown_pct",
    "drawdown_pct":          "drawdown_pct",
    "beta":                  "beta",
    "short_percent_of_float":"short_pct",
    "short_pct":             "short_pct",
    # score do motor de regras
    "score":                 "dip_score",
    "dip_score":             "dip_score",
    # contexto
    "spy_change":            "spy_change",
    "sector_etf_change":     "sector_etf_change",
    "earnings_days":         "earnings_days",
    "change_day_pct":        "change_day_pct",
    "market_cap":            "market_cap_b",
}

_SCALE_FUNCS: dict[str, Any] = {
    "market_cap_b": lambda v: v / 1e9 if v is not None else None,
}


def _build_feature_vector(raw: dict, columns: list[str]) -> np.ndarray:
    """
    Constrói um vector numpy com os valores na ordem esperada pelo modelo.
    Valores em falta → 0.0 (o modelo foi treinado com a mesma estratégia).
    """
    vec: list[float] = []
    for col in columns:
        # tenta o nome canónico e os aliases inversos
        val = raw.get(col)
        if val is None:
            for src, dst in _FEATURE_MAP.items():
                if dst == col and src in raw:
                    val = raw[src]
                    break
        if val is None:
            vec.append(0.0)
            continue
        # aplica transformação se existir
        fn = _SCALE_FUNCS.get(col)
        if fn:
            try:
                val = fn(val)
            except Exception:
                val = 0.0
        try:
            vec.append(float(val) if val is not None else 0.0)
        except (TypeError, ValueError):
            vec.append(0.0)
    return np.array(vec, dtype=np.float32).reshape(1, -1)


# ── Carregamento lazy do modelo ────────────────────────────────────────────────

def _load_models(force: bool = False) -> bool:
    """Carrega (ou recarrega) os modelos do disco. Devolve True se stage1 OK."""
    global _model_s1, _model_s2, _feat_cols, _threshold, _loaded_mtime

    if not _PKL_S1.exists():
        return False

    mtime = _PKL_S1.stat().st_mtime
    if not force and _model_s1 is not None and mtime == _loaded_mtime:
        return True  # já em cache e não mudou

    try:
        with open(_PKL_S1, "rb") as f:
            bundle = pickle.load(f)

        _model_s1  = bundle["model"]
        _feat_cols = bundle.get("feature_columns", [])
        _threshold = bundle.get("threshold", 0.50)
        _loaded_mtime = mtime
        logging.info(
            f"[ml_predictor] Stage 1 carregado — "
            f"alg={bundle.get('algorithm','?')} "
            f"threshold={_threshold:.3f} "
            f"features={len(_feat_cols)}"
        )
    except Exception as e:
        logging.error(f"[ml_predictor] Erro ao carregar stage1: {e}")
        _model_s1 = None
        return False

    # Stage 2 (opcional — Sommelier WIN40 vs WIN20)
    _model_s2 = None
    if _PKL_S2.exists():
        try:
            with open(_PKL_S2, "rb") as f:
                bundle2 = pickle.load(f)
            _model_s2 = bundle2
            logging.info(
                f"[ml_predictor] Stage 2 carregado — "
                f"alg={bundle2.get('algorithm','?')}"
            )
        except Exception as e:
            logging.warning(f"[ml_predictor] Stage 2 não carregado: {e}")

    return True


# ── API pública ────────────────────────────────────────────────────────────────

def is_model_ready() -> bool:
    """Devolve True se o modelo stage 1 estiver disponível."""
    return _PKL_S1.exists()


def get_model_info() -> dict:
    """Metadados do modelo carregado (para /status e /mldata)."""
    if not _load_models():
        return {"ready": False, "path": str(_PKL_S1)}
    try:
        with open(_PKL_S1, "rb") as f:
            bundle = pickle.load(f)
        return {
            "ready":     True,
            "algorithm": bundle.get("algorithm", "?"),
            "auc_pr":    bundle.get("auc_pr", 0),
            "threshold": bundle.get("threshold", 0.5),
            "n_samples": bundle.get("n_samples", 0),
            "features":  len(bundle.get("feature_columns", [])),
            "stage2":    _PKL_S2.exists(),
            "path":      str(_PKL_S1),
        }
    except Exception as e:
        return {"ready": False, "error": str(e)}


def ml_score(
    features: dict,
    reload_if_stale: bool = True,
) -> MLResult:
    """
    Pontua um dip com o modelo ML.

    `features` deve conter os campos do fundamentals dict + contexto:
      score, rsi, pe_ratio, fcf_yield, revenue_growth, gross_margin,
      debt_to_equity, analyst_upside, drawdown_pct, beta,
      spy_change, sector_etf_change, earnings_days,
      change_day_pct, market_cap

    Devolve MLResult com win_prob, label e confidence.
    """
    if reload_if_stale:
        if not _load_models():
            return MLResult(model_ready=False, label="NO_MODEL")

    if _model_s1 is None or not _feat_cols:
        return MLResult(model_ready=False, label="NO_MODEL")

    try:
        X = _build_feature_vector(features, _feat_cols)
        proba = _model_s1.predict_proba(X)[0]
        # classe 1 = WIN (index 1 se existir, senão último)
        win_prob: float = float(proba[1]) if len(proba) >= 2 else float(proba[-1])

        # Stage 2 — Sommelier WIN40 vs WIN20
        win40_prob: float | None = None
        if _model_s2 and win_prob >= _threshold:
            try:
                cols2 = _model_s2.get("feature_columns", _feat_cols)
                X2    = _build_feature_vector(features, cols2)
                p2    = _model_s2["model"].predict_proba(X2)[0]
                win40_prob = float(p2[1]) if len(p2) >= 2 else float(p2[-1])
            except Exception as e:
                logging.debug(f"[ml_predictor] Stage 2 score error: {e}")

        # Label
        if win_prob >= _threshold:
            if win40_prob is not None and win40_prob >= 0.55:
                label = "WIN_40"
            else:
                label = "WIN"
        else:
            label = "NO_WIN"

        # Confidence bucket
        if win_prob >= 0.80:
            confidence = "Alta"
        elif win_prob >= 0.65:
            confidence = "Média"
        elif win_prob >= _threshold:
            confidence = "Baixa"
        else:
            confidence = "–"

        return MLResult(
            win_prob=win_prob,
            win40_prob=win40_prob,
            label=label,
            confidence=confidence,
            model_ready=True,
            threshold=_threshold,
            features_used=_feat_cols,
        )

    except Exception as e:
        logging.warning(f"[ml_predictor] Erro ao pontuar: {e}")
        return MLResult(model_ready=False, label="ERROR")


def ml_badge(result: MLResult) -> str:
    """
    Gera uma linha de texto formatada para incluir no alerta Telegram.

    Exemplos:
      🤖 ML: WIN_40 | prob 0.87 | confiança Alta
      🤖 ML: WIN    | prob 0.63 | confiança Baixa
      🤖 ML: NO_WIN | prob 0.31
      🤖 ML: modelo não treinado
    """
    if not result.model_ready:
        return "🤖 ML: _modelo não treinado_"

    emoji_map = {
        "WIN_40": "🟢",
        "WIN":    "✅",
        "NO_WIN": "🔴",
        "ERROR":  "⚠️",
    }
    em = emoji_map.get(result.label, "📊")

    win40_str = ""
    if result.win40_prob is not None:
        win40_str = f" | WIN40 prob {result.win40_prob:.2f}"

    confidence_str = f" | confiança *{result.confidence}*" if result.confidence != "–" else ""

    return (
        f"🤖 *ML:* {em} `{result.label}` | prob *{result.win_prob:.2f}*"
        f"{win40_str}{confidence_str}"
    )
