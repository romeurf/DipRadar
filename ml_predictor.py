"""
ml_predictor.py — Score dual-layer: Camada A (preço) + Camada B (fundamentais).

Score final: prob_final = 0.40 × prob_preço + 0.60 × prob_fundamentais

API pública:
  ml_score(features: dict) -> MLResult
  is_model_ready() -> bool
  get_model_info() -> dict
  ml_badge(result: MLResult) -> str    # linha formatada para Telegram
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ── Caminhos ───────────────────────────────────────────────────────────────────

_DATA_DIR  = Path("/data") if Path("/data").exists() else Path("/tmp")
_PKL_PRICE = _DATA_DIR / "dip_model_price.pkl"   # Camada A — preço
_PKL_S1    = _DATA_DIR / "dip_model_stage1.pkl"  # Camada B — win vs no-win
_PKL_S2    = _DATA_DIR / "dip_model_stage2.pkl"  # Camada B — win40 vs win20

# Peso de cada camada no score final
_W_PRICE = 0.40
_W_FUND  = 0.60

# ── Cache em memória ───────────────────────────────────────────────────────────

_model_price:    Any | None = None
_model_s1:       Any | None = None
_model_s2:       Any | None = None
_mtime_price:    float      = 0.0
_mtime_s1:       float      = 0.0


@dataclass
class MLResult:
    win_prob:      float       = 0.0    # score final combinado
    prob_price:    float | None = None  # Camada A
    prob_fund:     float | None = None  # Camada B
    win40_prob:    float | None = None  # Sommelier WIN40 vs WIN20
    label:         str         = "NO_MODEL"
    confidence:    str         = "–"
    model_ready:   bool        = False
    threshold:     float       = 0.50
    features_used: list[str]   = field(default_factory=list)


# ── Feature map / aliases ─────────────────────────────────────────────────────

_FEATURE_MAP: dict[str, str] = {
    "rsi":                    "rsi",
    "pe":                     "pe_ratio",
    "pe_ratio":               "pe_ratio",
    "pb":                     "pb_ratio",
    "pb_ratio":               "pb_ratio",
    "fcf_yield":              "fcf_yield",
    "revenue_growth":         "revenue_growth",
    "gross_margin":           "gross_margin",
    "debt_to_equity":         "debt_to_equity",
    "analyst_upside":         "analyst_upside",
    "drawdown_from_high":     "drawdown_pct",
    "drawdown_pct":           "drawdown_pct",
    "beta":                   "beta",
    "short_percent_of_float": "short_pct",
    "short_pct":              "short_pct",
    "score":                  "dip_score",
    "dip_score":              "dip_score",
    "spy_change":             "spy_change",
    "sector_etf_change":      "sector_etf_change",
    "earnings_days":          "earnings_days",
    "change_day_pct":         "change_day_pct",
    "market_cap":             "market_cap_b",
    "volume_ratio":           "volume_ratio",
    "atr_pct":                "atr_pct",
}

_SCALE_FUNCS: dict[str, Any] = {
    "market_cap_b": lambda v: v / 1e9 if v is not None else None,
}


def _build_feature_vector(raw: dict, columns: list[str]) -> np.ndarray:
    vec: list[float] = []
    for col in columns:
        val = raw.get(col)
        if val is None:
            for src, dst in _FEATURE_MAP.items():
                if dst == col and src in raw:
                    val = raw[src]
                    break
        if val is None:
            vec.append(0.0)
            continue
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


# ── Carregamento lazy ─────────────────────────────────────────────────────────

def _load_models(force: bool = False) -> dict[str, bool]:
    """
    Carrega/actualiza os modelos do disco.
    Devolve {"price": bool, "fund": bool}.
    """
    global _model_price, _model_s1, _model_s2, _mtime_price, _mtime_s1
    loaded = {"price": False, "fund": False}

    # Camada A
    if _PKL_PRICE.exists():
        mtime = _PKL_PRICE.stat().st_mtime
        if force or _model_price is None or mtime != _mtime_price:
            try:
                with open(_PKL_PRICE, "rb") as f:
                    bundle = pickle.load(f)
                _model_price  = bundle
                _mtime_price  = mtime
                logging.info(
                    f"[ml_predictor] CamadaA carregada — "
                    f"alg={bundle.get('algorithm','?')} "
                    f"features={len(bundle.get('feature_columns', []))}"
                )
            except Exception as e:
                logging.error(f"[ml_predictor] Erro CamadaA: {e}")
        if _model_price is not None:
            loaded["price"] = True

    # Camada B stage 1
    if _PKL_S1.exists():
        mtime = _PKL_S1.stat().st_mtime
        if force or _model_s1 is None or mtime != _mtime_s1:
            try:
                with open(_PKL_S1, "rb") as f:
                    bundle = pickle.load(f)
                _model_s1 = bundle
                _mtime_s1 = mtime
                logging.info(
                    f"[ml_predictor] CamadaB stage1 carregada — "
                    f"alg={bundle.get('algorithm','?')} "
                    f"threshold={bundle.get('threshold',0.5):.3f}"
                )
            except Exception as e:
                logging.error(f"[ml_predictor] Erro CamadaB stage1: {e}")
        if _model_s1 is not None:
            loaded["fund"] = True

    # Camada B stage 2 (opcional)
    _model_s2 = None
    if _PKL_S2.exists():
        try:
            with open(_PKL_S2, "rb") as f:
                _model_s2 = pickle.load(f)
        except Exception:
            pass

    return loaded


# ── API pública ────────────────────────────────────────────────────────────────

def is_model_ready() -> bool:
    return _PKL_PRICE.exists() or _PKL_S1.exists()


def get_model_info() -> dict:
    loaded = _load_models()
    info: dict = {
        "ready":       loaded["price"] or loaded["fund"],
        "camada_a":    loaded["price"],
        "camada_b":    loaded["fund"],
        "stage2":      _PKL_S2.exists(),
        "weight_price": _W_PRICE,
        "weight_fund":  _W_FUND,
    }
    if loaded["price"] and _model_price:
        info["alg_price"]   = _model_price.get("algorithm", "?")
        info["features_a"]  = len(_model_price.get("feature_columns", []))
    if loaded["fund"] and _model_s1:
        info["alg_fund"]    = _model_s1.get("algorithm", "?")
        info["auc_pr"]      = _model_s1.get("auc_pr", 0)
        info["threshold"]   = _model_s1.get("threshold", 0.5)
        info["n_samples"]   = _model_s1.get("n_samples", 0)
        info["features_b"]  = len(_model_s1.get("feature_columns", []))
    return info


def ml_score(
    features: dict,
    reload_if_stale: bool = True,
) -> MLResult:
    """
    Pontua um dip com o score dual-layer.

    Score final = W_PRICE × prob_preço + W_FUND × prob_fundamentais
    Se só uma camada estiver disponível, usa apenas essa.
    """
    if reload_if_stale:
        loaded = _load_models()
        if not loaded["price"] and not loaded["fund"]:
            return MLResult(model_ready=False, label="NO_MODEL")

    prob_price: float | None = None
    prob_fund:  float | None = None
    win40_prob: float | None = None
    feat_used:  list[str]    = []
    threshold   = 0.50

    # ── Camada A ──────────────────────────────────────────────────────────────
    if _model_price is not None:
        try:
            cols_a = _model_price["feature_columns"]
            X_a    = _build_feature_vector(features, cols_a)
            p_a    = _model_price["model"].predict_proba(X_a)[0]
            prob_price = float(p_a[1]) if len(p_a) >= 2 else float(p_a[-1])
            feat_used  = cols_a
        except Exception as e:
            logging.debug(f"[ml_predictor] CamadaA score error: {e}")

    # ── Camada B stage 1 ──────────────────────────────────────────────────────
    if _model_s1 is not None:
        try:
            cols_b    = _model_s1["feature_columns"]
            threshold = _model_s1.get("threshold", 0.50)
            X_b       = _build_feature_vector(features, cols_b)
            p_b       = _model_s1["model"].predict_proba(X_b)[0]
            prob_fund = float(p_b[1]) if len(p_b) >= 2 else float(p_b[-1])
            feat_used = list(set(feat_used) | set(cols_b))
        except Exception as e:
            logging.debug(f"[ml_predictor] CamadaB score error: {e}")

    # ── Score final combinado ─────────────────────────────────────────────────
    if prob_price is not None and prob_fund is not None:
        win_prob = _W_PRICE * prob_price + _W_FUND * prob_fund
    elif prob_price is not None:
        win_prob = prob_price
        threshold = 0.50
    elif prob_fund is not None:
        win_prob = prob_fund
    else:
        return MLResult(model_ready=False, label="ERROR")

    # ── Stage 2 — Sommelier ───────────────────────────────────────────────────
    if _model_s2 and win_prob >= threshold:
        try:
            cols2 = _model_s2.get("feature_columns", [])
            X2    = _build_feature_vector(features, cols2)
            p2    = _model_s2["model"].predict_proba(X2)[0]
            win40_prob = float(p2[1]) if len(p2) >= 2 else float(p2[-1])
        except Exception as e:
            logging.debug(f"[ml_predictor] Stage 2 error: {e}")

    # ── Label ─────────────────────────────────────────────────────────────────
    if win_prob >= threshold:
        if win40_prob is not None and win40_prob >= 0.55:
            label = "WIN_40"
        else:
            label = "WIN"
    else:
        label = "NO_WIN"

    # ── Confidence ────────────────────────────────────────────────────────────
    if win_prob >= 0.80:
        confidence = "Alta"
    elif win_prob >= 0.65:
        confidence = "Média"
    elif win_prob >= threshold:
        confidence = "Baixa"
    else:
        confidence = "–"

    return MLResult(
        win_prob=round(win_prob, 3),
        prob_price=round(prob_price, 3) if prob_price is not None else None,
        prob_fund=round(prob_fund, 3) if prob_fund is not None else None,
        win40_prob=round(win40_prob, 3) if win40_prob is not None else None,
        label=label,
        confidence=confidence,
        model_ready=True,
        threshold=threshold,
        features_used=feat_used,
    )


def ml_badge(result: MLResult) -> str:
    """
    Linha formatada para o alerta Telegram.

    Exemplos:
      🤖 ML: 🟢 WIN_40 | prob 0.87 | P:0.82 F:0.90 | confiança Alta
      🤖 ML: ✅ WIN    | prob 0.63 | P:0.55 F:0.68 | confiança Baixa
      🤖 ML: 🔴 NO_WIN | prob 0.31
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

    # Detalhe das duas camadas
    layers_str = ""
    if result.prob_price is not None and result.prob_fund is not None:
        layers_str = f" | P:{result.prob_price:.2f} F:{result.prob_fund:.2f}"
    elif result.prob_price is not None:
        layers_str = f" | P:{result.prob_price:.2f}"
    elif result.prob_fund is not None:
        layers_str = f" | F:{result.prob_fund:.2f}"

    win40_str = ""
    if result.win40_prob is not None:
        win40_str = f" | WIN40:{result.win40_prob:.2f}"

    conf_str = f" | *{result.confidence}*" if result.confidence != "–" else ""

    return (
        f"🤖 *ML:* {em} `{result.label}` | prob *{result.win_prob:.2f}*"
        f"{layers_str}{win40_str}{conf_str}"
    )
