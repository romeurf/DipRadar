"""
ml_predictor.py — Score v3: regressor dual (model_up + model_down).

Champion: XGB-v2 (17 features base, rho_up=0.36, rho_down≈-0.001, topk_pnl=17.9%/60d)

Score final = pred_up (predicted max_return_60d, já transformado).
  → representa retorno máximo esperado nos 60 dias seguintes ao alerta.

Nota: versões anteriores usavam `pred_up / |pred_down|`, mas como o
model_down tem rho ≈ 0 (não prevê nada), dividir por |pred_down| era
essencialmente dividir por ruído. Mantemos `pred_down` no MLResult para
diagnóstico, mas não entra no score até termos rho_down < -0.10.

API pública (inalterada):
  ml_score(features: dict) -> MLResult
  is_model_ready() -> bool
  get_model_info() -> dict
  ml_badge(result: MLResult) -> str    # linha formatada para Telegram

Features esperadas (17 base do merged):
  macro_score, vix, spy_drawdown_5d, sector_drawdown_5d,
  fcf_yield, revenue_growth, gross_margin, de_ratio,
  pe_vs_fair, analyst_upside, quality_score,
  drop_pct_today, drawdown_52w, rsi_14, atr_ratio,
  volume_spike, market_cap_b
"""

from __future__ import annotations

import logging
import pickle
import sys
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


# ── Compatibilidade com bundles v3 picklados em Colab ─────────────────────────────────────
# O notebook colab_bootstrap.ipynb define a classe `DipModelsV3` em `__main__`
# antes de fazer pickle.dump. Quando o bundle é carregado em Railway (onde
# `__main__` é main.py, que não tem a classe), o unpickler falha com:
#   AttributeError: Can't get attribute 'DipModelsV3' on <module '__main__'>
# Definimos aqui e registamos em __main__ para que o unpickler a encontre.

@dataclass
class DipModelsV3:
    model_up:         Any
    model_down:       Any
    feature_cols:     list
    momentum_feats:   list
    n_train_samples:  int
    train_date:       str
    champion_name:    str
    schema_version:   int = 3


# Registar em __main__ para o unpickler resolver `__main__.DipModelsV3`.
# Idempotente: se já existir (e.g. re-import), não causa problemas.
try:
    _main_mod = sys.modules.get("__main__")
    if _main_mod is not None and not hasattr(_main_mod, "DipModelsV3"):
        _main_mod.DipModelsV3 = DipModelsV3
except Exception:  # pragma: no cover
    pass


def _safe_load(path: Path) -> Any:
    """Load a pickle/joblib bundle, trying joblib first."""
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def _to_dict(obj: Any) -> dict:
    """Normalize a loaded bundle to a dict with canonical keys.

    Aceita:
      - dict (passa-through)
      - dataclass instance (e.g. DipModelsV3)
      - objecto com __dict__

    Mapeia field names em alias canonónicos para que o resto do código
    possa usar `bundle.get("champion")`, `bundle.get("n_samples")`, etc.
    """
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj):
        d = {f.name: getattr(obj, f.name) for f in fields(obj)}
    elif hasattr(obj, "__dict__"):
        d = {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    else:
        return {}
    # Aliases para keys canónicas usadas pelo resto do módulo
    if "champion_name" in d:
        d.setdefault("champion", d["champion_name"])
    if "n_train_samples" in d:
        d.setdefault("n_samples", d["n_train_samples"])
    return d


# ── Caminhos ───────────────────────────────────────────────────────────────────

_REPO_DIR = Path(__file__).parent
_DATA_DIR = Path("/data") if Path("/data").exists() else Path("/tmp")

# Bundle v3 — procura primeiro em /data (Railway volume), depois no repo
_PKL_V3 = next(
    (p for p in [_DATA_DIR / "dip_models_v3.pkl", _REPO_DIR / "dip_models_v3.pkl"]
     if p.exists()),
    _REPO_DIR / "dip_models_v3.pkl",
)

# Features esperadas pelo champion XGB-v2 (17 base)
_FEATURE_COLS: list[str] = [
    "macro_score", "vix", "spy_drawdown_5d", "sector_drawdown_5d",
    "fcf_yield", "revenue_growth", "gross_margin", "de_ratio",
    "pe_vs_fair", "analyst_upside", "quality_score",
    "drop_pct_today", "drawdown_52w", "rsi_14", "atr_ratio",
    "volume_spike", "market_cap_b",
]

# Aliases de features — campo externo → nome interno
_FEATURE_MAP: dict[str, str] = {
    "rsi":                    "rsi_14",
    "rsi_14":                 "rsi_14",
    "drop_pct":               "drop_pct_today",
    "change_day_pct":         "drop_pct_today",
    "drawdown_from_high":     "drawdown_52w",
    "drawdown_pct":           "drawdown_52w",
    "spy_change":             "spy_drawdown_5d",
    "sector_etf_change":      "sector_drawdown_5d",
    "atr_pct":                "atr_ratio",
    "volume_ratio":           "volume_spike",
    "market_cap":             "market_cap_b",
    "fcf_yield":              "fcf_yield",
    "revenue_growth":         "revenue_growth",
    "gross_margin":           "gross_margin",
    "de_ratio":               "de_ratio",
    "debt_to_equity":         "de_ratio",
    "pe_vs_fair":             "pe_vs_fair",
    "analyst_upside":         "analyst_upside",
    "quality_score":          "quality_score",
    "macro_score":            "macro_score",
    "vix":                    "vix",
    "market_cap_b":           "market_cap_b",
}

_SCALE_FUNCS: dict[str, Any] = {
    "market_cap_b": lambda v: v / 1e9 if v is not None and float(v) > 1e6 else v,
}

# Thresholds de score para labels (escala: pred_up = retorno previsto a 60d)
# Ancorados na win threshold do treino (5% em 60d).
_SCORE_HIGH   = 0.10   # pred_up > 10% → WIN_STRONG
_SCORE_MED    = 0.05   # pred_up > 5%  → WIN  (== win threshold do treino)
_SCORE_FLOOR  = 0.02   # pred_up > 2%  → WEAK; abaixo → NO_WIN

# Cache em memória
_bundle:    Any | None = None
_mtime_v3: float       = 0.0


@dataclass
class MLResult:
    win_prob:      float        = 0.0     # score normalizado [0, 1] para compatibilidade
    score_raw:     float        = 0.0     # rácio upside/downside (não normalizado)
    pred_up:       float | None = None    # previsão max_return_60d
    pred_down:     float | None = None    # previsão max_drawdown_60d (negativo)
    prob_price:    float | None = None    # alias compatibilidade (= win_prob)
    prob_fund:     float | None = None    # alias compatibilidade (= win_prob)
    win40_prob:    float | None = None    # n/a em v3 — mantido para compatibilidade
    label:         str          = "NO_MODEL"
    confidence:    str          = "–"
    model_ready:   bool         = False
    threshold:     float        = _SCORE_FLOOR
    features_used: list[str]    = field(default_factory=list)
    vix_regime:    str | None   = None
    coverage:      float        = 1.0
    low_coverage:  bool         = False
    model_version: str          = "v3"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_feature(raw: dict, col: str) -> float:
    """Resolve uma feature com fallback via aliases e escala se necessário."""
    val = raw.get(col)
    if val is None:
        for src, dst in _FEATURE_MAP.items():
            if dst == col and src in raw:
                val = raw[src]
                break
    if val is None:
        return 0.0
    fn = _SCALE_FUNCS.get(col)
    if fn:
        try:
            val = fn(val)
        except Exception:
            val = 0.0
    try:
        return float(val) if val is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _build_feature_vector(raw: dict, columns: list[str]) -> np.ndarray:
    vec = [_resolve_feature(raw, col) for col in columns]
    return np.array(vec, dtype=np.float32).reshape(1, -1)


def _classify_vix(vix_value: float | None) -> str:
    if vix_value is None:
        return "medium"
    v = float(vix_value)
    if v < 15:
        return "low"
    if v < 25:
        return "medium"
    return "high"


def _score_to_prob(score: float, calibrator: Any | None = None) -> float:
    """Mapeia o score (= pred_up) para uma probabilidade calibrada [0, 1].

    Se o bundle trouxer um `score_calibrator` (sklearn IsotonicRegression
    ou objecto com .predict()), usa-o. Caso contrário cai para uma
    sigmoide ancorada na win threshold de 5% (centro), que dá:
      score=0.00 → 0.32   score=0.05 → 0.50
      score=0.10 → 0.68   score=0.15 → 0.82
    Steepness 15 => transição suave em torno de ±5% retorno previsto.
    """
    if calibrator is not None:
        try:
            arr = np.asarray([score], dtype=np.float64)
            pred = calibrator.predict(arr)
            return float(np.clip(pred[0], 0.0, 1.0))
        except Exception as e:  # pragma: no cover — fallback robusto
            logging.debug(f"[ml_predictor] calibrator falhou ({e}); fallback sigmoide")
    return float(1.0 / (1.0 + np.exp(-15.0 * (score - 0.05))))


def _inverse_transform_up(yp: float) -> float:
    return float(np.expm1(np.clip(yp, -3, 3)))


def _inverse_transform_down(yp: float) -> float:
    return float(-np.expm1(np.clip(-yp, 0, 3)))


# ── Carregamento lazy ─────────────────────────────────────────────────────────

def _load_bundle(force: bool = False) -> bool:
    global _bundle, _mtime_v3
    if not _PKL_V3.exists():
        return False
    mtime = _PKL_V3.stat().st_mtime
    if not force and _bundle is not None and mtime == _mtime_v3:
        return True
    try:
        raw_bundle = _safe_load(_PKL_V3)
        _bundle    = _to_dict(raw_bundle)
        _mtime_v3  = mtime
        cols       = _bundle.get("feature_cols", _FEATURE_COLS)
        champion   = _bundle.get("champion", "XGB-v2")
        logging.info(
            f"[ml_predictor] Bundle v3 carregado — champion={champion} "
            f"features={len(cols)} rho={_bundle.get('rho_mean', '?')}"
        )
        return True
    except Exception as e:
        logging.error(f"[ml_predictor] Erro ao carregar bundle v3: {e}")
        return False


# ── API pública ────────────────────────────────────────────────────────────────

def is_model_ready() -> bool:
    return _PKL_V3.exists()


def get_model_info() -> dict:
    ready = _load_bundle()
    if not ready or _bundle is None:
        return {"ready": False, "model_version": "v3"}
    cols = _bundle.get("feature_cols", _FEATURE_COLS)
    return {
        "ready":         True,
        "model_version": "v3",
        "champion":      _bundle.get("champion", "XGB-v2"),
        "n_features":    len(cols),
        "feature_cols":  cols,
        "rho_mean":      _bundle.get("rho_mean"),
        "topk_pnl":      _bundle.get("topk_pnl"),
        "n_samples":     _bundle.get("n_samples"),
        # Compatibilidade com código que lê camada_a/camada_b
        "camada_a":      True,
        "camada_b":      False,
        "weight_price":  1.0,
        "weight_fund":   0.0,
    }


def ml_score(
    features: dict,
    reload_if_stale: bool = True,
    symbol: str | None = None,
    log_to_file: bool = True,
) -> MLResult:
    """
    Pontua um dip com o modelo v3 (regressor dual XGB-v2).

    Score = pred_up (retorno previsto a 60 dias, após expm1).
      pred_down é mantido em MLResult para diagnóstico mas não entra
      no score (rho_down ≈ 0 → não tem sinal útil).

    Labels (ancorados na win threshold de 5% do treino):
      WIN_STRONG  — score > 0.10  (retorno previsto > 10%)
      WIN         — score > 0.05  (retorno previsto > 5%)
      WEAK        — score > 0.02
      NO_WIN      — score <= 0.02
    """
    if reload_if_stale:
        if not _load_bundle():
            return MLResult(model_ready=False, label="NO_MODEL")

    if _bundle is None:
        return MLResult(model_ready=False, label="NO_MODEL")

    enriched = dict(features) if features else {}

    # Features a usar — bundle pode sobrepor a lista default
    cols = _bundle.get("feature_cols", _FEATURE_COLS)
    X    = _build_feature_vector(enriched, cols)

    try:
        model_up   = _bundle["model_up"]
        model_down = _bundle["model_down"]
        pred_up_raw   = float(model_up.predict(X)[0])
        pred_down_raw = float(model_down.predict(X)[0])
        pred_up   = _inverse_transform_up(pred_up_raw)
        pred_down = _inverse_transform_down(pred_down_raw)
    except Exception as e:
        logging.error(f"[ml_predictor] Erro na inferência v3: {e}")
        return MLResult(model_ready=False, label="ERROR")

    # Score = pred_up directo. pred_down mantido apenas para diagnóstico.
    # Justificação: rho_down_mean ≈ -0.001 no champion v3 (XGB-v2) — dividir
    # por ruído distorce o score. Reverter quando rho_down < -0.10.
    score = float(pred_up)

    # Normalizar para [0,1] usando o calibrator do bundle (se existir)
    score_calibrator = _bundle.get("score_calibrator") if _bundle else None
    win_prob = _score_to_prob(score, calibrator=score_calibrator)

    # VIX regime (informativo)
    vix_value  = enriched.get("vix") or enriched.get("vix_value")
    vix_regime = _classify_vix(vix_value)

    # Label baseado no score raw
    if score > _SCORE_HIGH:
        label      = "WIN_STRONG"
        confidence = "Alta"
    elif score > _SCORE_MED:
        label      = "WIN"
        confidence = "Média" if score < 1.25 else "Alta"
    elif score > _SCORE_FLOOR:
        label      = "WEAK"
        confidence = "Baixa"
    else:
        label      = "NO_WIN"
        confidence = "–"

    result = MLResult(
        win_prob      = round(win_prob, 3),
        score_raw     = round(score, 3),
        pred_up       = round(pred_up, 4),
        pred_down     = round(pred_down, 4),
        prob_price    = round(win_prob, 3),   # alias compatibilidade
        prob_fund     = None,
        win40_prob    = None,
        label         = label,
        confidence    = confidence,
        model_ready   = True,
        threshold     = _SCORE_FLOOR,
        features_used = cols,
        vix_regime    = vix_regime,
        coverage      = 1.0,
        low_coverage  = False,
        model_version = "v3",
    )

    if log_to_file and symbol:
        try:
            from prediction_log import log_prediction
            log_prediction(symbol=symbol, features=features or {}, result=result)
        except Exception as e:
            logging.debug(f"[ml_predictor] log_prediction skipped: {e}")

    return result


def ml_badge(result: MLResult) -> str:
    """
    Linha formatada para o alerta Telegram.

    Exemplos:
      🤖 ML v3: 🟢 WIN_STRONG | up +18.5% | dn -8.2% | score 2.25 | VIX:med
      🤖 ML v3: ✅ WIN        | up +12.1% | dn -9.4% | score 1.29 | Alta
      🤖 ML v3: 🟡 WEAK       | up +6.3%  | dn -10.1%| score 0.62
      🤖 ML v3: 🔴 NO_WIN     | up +2.1%  | dn -12.3%| score 0.17
      🤖 ML v3: modelo não treinado
    """
    if not result.model_ready:
        return "🤖 ML v3: _modelo não treinado_"

    emoji_map = {
        "WIN_STRONG": "🟢",
        "WIN":        "✅",
        "WEAK":       "🟡",
        "NO_WIN":     "🔴",
        "ERROR":      "⚠️",
        "NO_MODEL":   "⚫",
    }
    em = emoji_map.get(result.label, "📊")

    up_str   = f"+{result.pred_up*100:.1f}%" if result.pred_up is not None else "?"
    down_str = f"{result.pred_down*100:.1f}%" if result.pred_down is not None else "?"
    score_str = f"{result.score_raw:.2f}"

    vix_str  = f" | VIX:{result.vix_regime[:3]}" if result.vix_regime else ""
    conf_str = f" | *{result.confidence}*" if result.confidence != "–" else ""

    return (
        f"🤖 *ML v3:* {em} `{result.label}` "
        f"| up *{up_str}* | dn {down_str} | score *{score_str}*"
        f"{vix_str}{conf_str}"
    )
