"""
ml_predictor.py — DIP model inference. Predicts alpha_90d (log-return excess over SPY).

The bundle stores its own feature_cols list — at inference we read it from there,
not from FEATURE_COLUMNS. This module just feeds the live features dict through it.

Score thresholds (alpha_90d scale):
  _SCORE_HIGH  = 0.06   alpha > +6pp over SPY → WIN_STRONG
  _SCORE_MED   = 0.03   alpha > +3pp over SPY → WIN
  _SCORE_FLOOR = 0.01   alpha > +1pp over SPY → WEAK

model_down (max_drawdown_60d) is kept in MLResult for diagnostics only; its IC ≈ 0.

Public API: ml_score(features) -> MLResult, is_model_ready(), get_model_info(),
            ml_badge(result).
"""

from __future__ import annotations

import logging
import math
import pickle
import sys
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


# ── Pickle compatibility: ensure DipModels resolves under __main__ ──
# Notebook/training code pickles DipModels with __main__ as the owning module.
# When loaded in the bot process (where __main__ is main.py), the unpickler
# raises AttributeError unless we register the class on __main__ ourselves.
#
# Local stub: kept as fallback for minimal installs where ml_training isn't
# importable. MUST stay field-for-field aligned with ml_training/bundle.py.
# When ml_training is available, we rebind to the canonical class so any new
# field added there is picked up without editing this stub.

@dataclass
class DipModels:
    """Fallback stub for pickled bundles. Canonical class is ml_training.bundle.DipModels."""
    model_up:         Any
    model_down:       Any
    feature_cols:     list
    score_calibrator: Any = None
    n_train_samples:  int = 0
    train_date:       str = ""
    champion_name:    str = ""
    schema_version:   int = 3
    momentum_feats:   list = field(default_factory=list)
    rho_mean:         Any  = None
    rho_alpha:        Any  = None
    rho_down:         Any  = None
    topk_pnl:         Any  = None
    fold_metrics:     list = field(default_factory=list)
    sector_models:    dict = field(default_factory=dict)


try:
    from ml_training.bundle import DipModels as _CanonicalDipModels
    DipModels = _CanonicalDipModels  # type: ignore[misc,assignment]
except Exception:  # pragma: no cover — keep local stub
    pass

# Legacy alias for unpickling old bundles that reference __main__.DipModelsV3.
DipModelsV3 = DipModels

# Register on __main__ so the unpickler resolves both names.
try:
    _main_mod = sys.modules.get("__main__")
    if _main_mod is not None:
        for _name in ("DipModels", "DipModelsV3"):
            if not hasattr(_main_mod, _name):
                setattr(_main_mod, _name, DipModels)
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
      - dataclass instance (e.g. DipModels)
      - objecto com __dict__

    Mapeia field names em alias canonónicos para que o resto do código
    possa usar `bundle.get("champion")`, `bundle.get("n_samples")`, etc.
    """
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj) or hasattr(obj, "__dict__"):
        d = {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    else:
        return {}
    # Aliases para keys canónicas usadas pelo resto do módulo
    if "champion_name" in d:
        d.setdefault("champion", d["champion_name"])
    if "n_train_samples" in d:
        d.setdefault("n_samples", d["n_train_samples"])
    # Older bundles only wrote `rho_alpha`; the canonical display key is
    # `rho_mean` (post-PR #23 the trainer writes BOTH, but in-flight bundles
    # may have only one). Alias rho_alpha → rho_mean so /admin_check_config
    # doesn't show "rho_alpha=None" just because the field name evolved.
    if d.get("rho_mean") in (None, "") and d.get("rho_alpha") not in (None, ""):
        d["rho_mean"] = d["rho_alpha"]
    return d


# ── Caminhos ───────────────────────────────────────────────────────────────────

_REPO_DIR = Path(__file__).parent
_DATA_DIR = Path("/data") if Path("/data").exists() else Path("/tmp")

# Bundle ML v3 — procura em ordem (Railway volume / repo / legacy v3 names).
# Após PR de robustez 2026-05, o nome canónico é `dip_models.pkl` em /data ou
# no sub-package `ml_training/`. Os nomes legacy `dip_models_v3.pkl` são lidos
# como fallback durante a migração para não quebrar volumes antigos.
_BUNDLE_CANDIDATES = [
    _DATA_DIR / "dip_models.pkl",
    _REPO_DIR / "ml_training" / "dip_models.pkl",
    _DATA_DIR / "dip_models_v3.pkl",
    _REPO_DIR / "dip_models_v3.pkl",
]
_PKL_V3 = next((p for p in _BUNDLE_CANDIDATES if p.exists()), _BUNDLE_CANDIDATES[0])

# Diagnostic log at module import time so production traces show which
# bundle path was actually selected. Helps debug NO_MODEL cases where
# /admin_check_config disagrees with /analisar. Logged at WARNING level
# because Railway often filters INFO logs in production.
try:
    _resolved = "FOUND" if _PKL_V3.exists() else "MISSING"
    _checked_paths = " | ".join(
        f"{p}={'✓' if p.exists() else '✗'}" for p in _BUNDLE_CANDIDATES
    )
    logging.warning(
        f"[ml_predictor] Bundle path resolved: {_PKL_V3} [{_resolved}] | "
        f"all candidates: {_checked_paths}"
    )
except Exception:  # never block import on logging
    pass

# Features esperadas pelo bundle — importadas de ml_features para evitar duplicação.
# Se o import falhar (ex: em testes isolados), usa lista vazia e o bundle
# devolve erro explícito (bundle['feature_cols'] é obrigatório após retrain).
try:
    from ml.ml_features import FEATURE_COLUMNS as _FEATURE_COLS
except Exception as _e:
    logging.warning(f"[ml_predictor] Falha ao importar FEATURE_COLUMNS: {_e} — fallback vazio")
    _FEATURE_COLS = []

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

# ── Thresholds de score (escala: alpha_90d = log-return excess sobre SPY) ────
#
#   alpha_90d = log1p(stock_close_90d) - log1p(spy_close_90d)
#
#   Distribuição típica de alpha_90d no dataset de treino (2015-2024):
#     p25 ≈ -0.10   mediana ≈ +0.00   p75 ≈ +0.10   p90 ≈ +0.18
#
#   Thresholds (para alpha_90d, excesso sobre SPY):
#     HIGH  = 0.06  → alpha > +6pp sobre SPY → WIN_STRONG
#     MED   = 0.03  → alpha > +3pp sobre SPY → WIN
#     FLOOR = 0.01  → alpha > +1pp sobre SPY → WEAK
#
#   A sigmoide em _score_to_prob é ancorada em 0.03 (MED) para que
#   win_prob=0.50 corresponda ao break-even alpha vs benchmark.
_SCORE_HIGH   = 0.06   # pred_up > +6% alpha → WIN_STRONG
_SCORE_MED    = 0.03   # pred_up > +3% alpha → WIN
_SCORE_FLOOR  = 0.01   # pred_up > +1% alpha → WEAK; abaixo → NO_WIN

# Cache em memória
_bundle:    Any | None = None
_mtime_v3: float       = 0.0


@dataclass
class MLResult:
    win_prob:           float        = 0.0     # P(alpha_90d > threshold), calibrado [0,1]
    score_raw:          float        = 0.0     # alpha_90d previsto (log-return excess SPY)
    pred_up:            float | None = None    # previsão alpha_90d (= score_raw)
    pred_down:          float | None = None    # previsão max_drawdown_60d (risco)
    prob_price:         float | None = None    # alias compatibilidade (= win_prob)
    prob_fund:          float | None = None    # alias compatibilidade (= win_prob)
    win40_prob:         float | None = None    # n/a em v3 — mantido para compatibilidade
    label:              str          = "NO_MODEL"
    confidence_label:   str          = "–"
    model_ready:        bool         = False
    threshold:          float        = _SCORE_FLOOR
    features_used:      list[str]    = field(default_factory=list)
    vix_regime:         str | None   = None
    coverage:           float        = 1.0
    low_coverage:       bool         = False
    model_version:      str          = "v3"
    # ── Novos campos v4.1 ────────────────────────────────────────────────────
    stock_type:         str          = "SPECULATIVE"  # BLUE_CHIP | QUALITY | SPECULATIVE
    recommended_hold:   str          = "90D"          # 90D | 6M | LONG_TERM
    position_size_pct:  float        = 0.0            # % do portfolio sugerida pelo modelo
    risk_reward_ratio:  float        = 0.0            # |alpha_90d| / |max_drawdown_60d|


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_feature(raw: dict, col: str) -> float:
    """Resolve a feature with alias fallback and scaling.

    NaN policy: the build_features pipeline now leaves EDGAR-dependent features
    as NaN when source data is missing (pre-2003 episodes, sparse coverage).
    At training time the trainer handles NaN per model type (native for LGBM,
    col-mean imputation for Ridge). At inference time we don't have access to
    the training col_means, so we impute to ml_features._FALLBACK — these are
    set to plausibly-neutral values close to typical col_means and were the
    semantics used by Ridge models pre-NaN-flow. LGBM models tolerate the
    constant gracefully; the alternative (passing NaN to a Ridge .predict)
    would emit NaN through the calibrator and produce no score at all.
    """
    val = raw.get(col)
    if val is None:
        for src, dst in _FEATURE_MAP.items():
            if dst == col and src in raw:
                val = raw[src]
                break
    if val is None:
        # truly absent → fall back to the trained-time neutral value
        from ml.ml_features import _FALLBACK as _FB
        return float(_FB.get(col, 0.0))

    fn = _SCALE_FUNCS.get(col)
    if fn:
        try:
            val = fn(val)
        except Exception:
            val = None

    try:
        f = float(val) if val is not None else float("nan")
    except (TypeError, ValueError):
        f = float("nan")

    if math.isnan(f):
        # Source had the column but value was NaN (e.g. pre-EDGAR episode at
        # training time leaked through, or live yfinance fetch failed) →
        # same _FALLBACK substitution as the absent case.
        from ml.ml_features import _FALLBACK as _FB
        return float(_FB.get(col, 0.0))

    return f


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
    """Mapeia o score (= alpha_90d previsto) para uma probabilidade calibrada [0, 1].

    Se o bundle trouxer um `score_calibrator` (sklearn IsotonicRegression
    ou Platt/LogisticRegression com .predict_proba()), usa-o.
    Caso contrário cai para uma sigmoide ancorada na win threshold de 3%
    (_SCORE_MED), que dá:
      score=-0.03 → 0.18   score=0.00 → 0.32
      score=+0.03 → 0.50   score=+0.06 → 0.68   score=+0.09 → 0.82
    Steepness 15 => transição suave em torno de ±3pp de alpha.
    """
    if calibrator is not None:
        try:
            arr = np.asarray([score], dtype=np.float64)
            # Suporta sklearn calibrators com .predict() ou .predict_proba()
            if hasattr(calibrator, "predict_proba"):
                pred = calibrator.predict_proba(arr.reshape(-1, 1))[:, 1]
            else:
                pred = calibrator.predict(arr)
            return float(np.clip(pred[0], 0.0, 1.0))
        except Exception as e:
            logging.debug(f"[ml_predictor] calibrator falhou ({e}); fallback sigmoide")
    # Sigmoide ancorada em _SCORE_MED (0.03) — win_prob=0.50 no break-even alpha
    return float(1.0 / (1.0 + np.exp(-15.0 * (score - _SCORE_MED))))


def _inverse_transform_up(yp: float) -> float:
    """Devolve a previsão de alpha_90d tal como o modelo a produz.

    BUG FIX v4.1: a versão anterior aplicava expm1() erradamente — o modelo
    prediz alpha_90d directamente (sem log-transform adicional ao target).
    expm1(0.06) ≈ 0.0618, o que inflacionava ligeiramente os scores exibidos
    e desalinhava os thresholds de label.
    """
    return float(np.clip(yp, -3.0, 3.0))


def _inverse_transform_down(yp: float) -> float:
    """Devolve a previsão de max_drawdown_60d tal como o modelo a produz.

    BUG FIX v4.1: idem — sem expm1(). Valores negativos = drawdown esperado.
    """
    return float(np.clip(yp, -3.0, 3.0))


# ── Classificação de stock e sizing ───────────────────────────────────────────

def classify_stock_type(market_cap_b: float, quality_score: float) -> str:
    """Classifica o stock como BLUE_CHIP, QUALITY ou SPECULATIVE.

    Critérios (configuráveis em ml_training/config.py):
      BLUE_CHIP   : cap > 100B + quality ≥ 0.65 → acumular, hold longo prazo
      QUALITY     : cap > 25B  + quality ≥ 0.55 → hold 6M se alpha forte
      SPECULATIVE : resto       → dip-and-flip, 90d
    """
    try:
        from ml_training.config import (
            BLUE_CHIP_MARKET_CAP_B, BLUE_CHIP_QUALITY_SCORE,
            QUALITY_MARKET_CAP_B, QUALITY_SCORE_MIN,
        )
    except ImportError:
        BLUE_CHIP_MARKET_CAP_B, BLUE_CHIP_QUALITY_SCORE = 100.0, 0.65
        QUALITY_MARKET_CAP_B, QUALITY_SCORE_MIN = 25.0, 0.55

    cap = float(market_cap_b) if market_cap_b else 0.0
    qs  = float(quality_score) if quality_score else 0.0
    if cap >= BLUE_CHIP_MARKET_CAP_B and qs >= BLUE_CHIP_QUALITY_SCORE:
        return "BLUE_CHIP"
    if cap >= QUALITY_MARKET_CAP_B and qs >= QUALITY_SCORE_MIN:
        return "QUALITY"
    return "SPECULATIVE"


def recommend_hold_period(stock_type: str, pred_alpha_90d: float) -> str:
    """Recomenda período de holding com base no tipo de stock e sinal do modelo.

    BLUE_CHIP   → LONG_TERM (acumular; só sair por deterioração estrutural)
    QUALITY + alpha forte → 6M (segurar; pode bater muito mais que 90d)
    Resto       → 90D (rotação de capital)
    """
    try:
        from ml_training.config import EXTEND_HOLD_ALPHA_THRESHOLD
    except ImportError:
        EXTEND_HOLD_ALPHA_THRESHOLD = 0.08

    if stock_type == "BLUE_CHIP":
        return "LONG_TERM"
    if stock_type == "QUALITY" and pred_alpha_90d >= EXTEND_HOLD_ALPHA_THRESHOLD:
        return "6M"
    return "90D"


def compute_position_size(
    pred_alpha_90d: float,
    pred_max_drawdown: float,
    win_prob: float,
    max_position: float = 0.25,
) -> float:
    """Dimensiona a posição com base nas previsões do modelo (risk-adjusted).

    Formula: edge × rr_score × max_position
      edge     = 2 × win_prob − 1   (0 sem vantagem, 1 com certeza absoluta)
      rr_score = alpha / (alpha + |drawdown|)   (0..1, penaliza alto risco)

    Interpretação:
      win_prob 50% → sem vantagem → 0% (nunca aloca sem edge)
      win_prob 70%, alpha 8%, drawdown 10% → rr=0.44 → pos=0.40×0.44×25%=4.4%
      win_prob 80%, alpha 12%, drawdown 8% → rr=0.60 → pos=0.60×0.60×25%=9.0%

    max_position é um cap de segurança (default 25%) não um valor hardcoded de
    alocação — o tamanho real depende do alpha e drawdown previstos.
    """
    if win_prob <= 0.50 or pred_alpha_90d <= 0.0:
        return 0.0
    edge   = 2.0 * win_prob - 1.0
    dd     = abs(pred_max_drawdown) if pred_max_drawdown < 0 else 0.10
    alpha  = max(1e-4, pred_alpha_90d)
    rr_score = alpha / (alpha + dd)
    return float(min(max_position, edge * rr_score * max_position))


def compute_risk_reward(pred_alpha_90d: float, pred_max_drawdown: float) -> float:
    """Rácio simples |alpha_90d| / |max_drawdown_60d|. 0 se inválido."""
    if pred_max_drawdown >= 0 or pred_alpha_90d <= 0:
        return 0.0
    return round(float(pred_alpha_90d / abs(pred_max_drawdown)), 2)


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
        # Thresholds activos
        "score_high":    _SCORE_HIGH,
        "score_med":     _SCORE_MED,
        "score_floor":   _SCORE_FLOOR,
        "target":        "alpha_90d",
    }


def ml_score(
    features: dict,
    reload_if_stale: bool = True,
    symbol: str | None = None,
    log_to_file: bool = True,
) -> MLResult:
    """
    Pontua um dip com o modelo v3 (regressor dual XGB-v2).

    Score = pred_up = alpha_90d previsto = log-return excess sobre SPY em 90d.
      pred_down é mantido em MLResult para diagnóstico mas não entra
      no score (rho_down ≈ 0 → não tem sinal útil).

    Labels (ancorados em alpha_90d — excesso sobre SPY):
      WIN_STRONG  — alpha previsto > +6pp  (top ~15% histórico)
      WIN         — alpha previsto > +3pp  (top ~35% histórico)
      WEAK        — alpha previsto > +1pp
      NO_WIN      — alpha previsto <= +1pp
    """
    if reload_if_stale:
        if not _load_bundle():
            return MLResult(model_ready=False, label="NO_MODEL")

    if _bundle is None:
        return MLResult(model_ready=False, label="NO_MODEL")

    enriched = dict(features) if features else {}

    # Option B — Investment-phase FCF override (mirrors core_radar fix).
    # If the stock pattern matches a reinvesting compounder (NVO-style:
    # rev growth > 10%, GM > 40%, ROE > 15%, FCF < 0), neutralize fcf_yield
    # at inference so the model doesn't false-flag a capex-driven negative
    # FCF as deterioration.
    #
    # The neutral-value source preference (best → fallback):
    #   1. bundle["impute_medians"]["fcf_yield"]   — exact training median
    #      (preferred: matches the distribution the model trained on)
    #   2. ml_features._FALLBACK["fcf_yield"]      — manually-maintained
    #      "plausibly-neutral" constant (~0.04)
    #
    # TECH DEBT: the DIP bundle currently doesn't store impute_medians (see
    # comment in _resolve_feature). When the DIP model is retrained, the
    # trainer should persist column medians into the bundle so this fallback
    # path becomes principled. Until then we use _FALLBACK which is "close
    # to typical col_means" by manual curation.
    try:
        from scanners.score import _is_investment_phase_compounder
        if _is_investment_phase_compounder(enriched):
            # Try bundle median first; fall back to static if unavailable
            bundle_medians = _bundle.get("impute_medians") if _bundle else None
            if bundle_medians and "fcf_yield" in bundle_medians:
                neutral = float(bundle_medians["fcf_yield"])
                source = "bundle.impute_medians"
            else:
                from ml.ml_features import _FALLBACK as _FB
                neutral = float(_FB.get("fcf_yield", 0.04))
                source = "ml_features._FALLBACK"
            enriched["fcf_yield"] = neutral
            enriched["_fcf_override_applied"] = True
            logging.info(
                f"[ml_predictor] FCF override applied for {symbol or 'unknown'} "
                f"(investment-phase compounder, neutral={neutral:.4f} from {source})"
            )
    except Exception as _e:
        logging.debug(f"[ml_predictor] investment-phase check failed: {_e}")

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
        # Detailed log so /analisar callers can see what's blocking inference.
        # Common causes: feature_cols schema drift between training and the
        # current ml_features, NaN in a feature the model can't impute,
        # KeyError on "model_up"/"model_down" if bundle is partial.
        logging.error(
            f"[ml_predictor] Inference failed for {symbol or '?'}: "
            f"{type(e).__name__}: {e}",
            exc_info=True,
        )
        return MLResult(model_ready=False, label="ERROR")

    # Sector model: se existir modelo treinado para este sector, usa-o
    # exclusivamente. O modelo global serve de fallback para sectores sem
    # dados suficientes (< 150 rows históricos) ou sector "Unknown".
    _sector_models = (_bundle.get("sector_models") or {}) if _bundle else {}
    _SECTOR_ALIASES = {"Financials": "Financial Services", "Materials": "Basic Materials"}
    _stock_sector  = str(features.get("sector") or "Unknown")
    _stock_sector  = _SECTOR_ALIASES.get(_stock_sector, _stock_sector)
    _sector_entry  = _sector_models.get(_stock_sector)
    if _sector_entry is not None:
        try:
            _sector_raw  = float(_sector_entry["model"].predict(X)[0])
            pred_up      = _inverse_transform_up(_sector_raw)
        except Exception as _se:
            logging.debug(f"[ml_predictor] sector model '{_stock_sector}' falhou, a usar global: {_se}")

    # Score = alpha_90d previsto (log-return excess sobre SPY).
    score = float(pred_up)

    # Normalize to [0,1] using the bundle's calibrator (if present).
    score_calibrator = _bundle.get("score_calibrator") if _bundle else None
    win_prob = _score_to_prob(score, calibrator=score_calibrator)

    # VIX regime (informativo)
    vix_value  = enriched.get("vix") or enriched.get("vix_value")
    vix_regime = _classify_vix(vix_value)

    # Labels baseados em win_prob (mais estáveis e independentes do threshold bruto)
    if win_prob > 0.70:
        label           = "WIN_STRONG"
        confidence_label = "Alta"
    elif win_prob > 0.55:
        label           = "WIN"
        confidence_label = "Alta" if win_prob > 0.65 else "Média"
    elif win_prob > 0.40:
        label           = "WEAK"
        confidence_label = "Baixa"
    else:
        label           = "NO_WIN"
        confidence_label = "–"

    # Classificação de stock e sizing — requerem market_cap e quality_score
    # das features (se disponíveis; fallback neutro se não estiverem).
    mkt_cap_b    = float(min(enriched.get("market_cap", 0.0) or 0.0, 1e13)) / 1e9  # cap 10T para dados corrompidos
    quality_sc   = float(enriched.get("quality_score", 0.0) or 0.0)
    stock_type   = classify_stock_type(mkt_cap_b, quality_sc)
    hold_period  = recommend_hold_period(stock_type, score)
    pos_size     = compute_position_size(score, pred_down, win_prob)
    rr_ratio     = compute_risk_reward(score, pred_down)

    result = MLResult(
        win_prob          = round(win_prob, 3),
        score_raw         = round(score, 3),
        pred_up           = round(pred_up, 4),
        pred_down         = round(pred_down, 4),
        prob_price        = round(win_prob, 3),
        prob_fund         = None,
        win40_prob        = None,
        label             = label,
        confidence_label  = confidence_label,
        model_ready       = True,
        threshold         = _SCORE_FLOOR,
        features_used     = cols,
        vix_regime        = vix_regime,
        coverage          = 1.0,
        low_coverage      = False,
        model_version     = "v3",
        stock_type        = stock_type,
        recommended_hold  = hold_period,
        position_size_pct = round(pos_size, 3),
        risk_reward_ratio = rr_ratio,
    )

    if log_to_file and symbol:
        try:
            from ml.prediction_log import log_prediction
            log_prediction(symbol=symbol, features=features or {}, result=result)
        except Exception as e:
            logging.debug(f"[ml_predictor] log_prediction skipped: {e}")

    return result


def ml_badge(result: MLResult) -> str:
    """Linha ML em linguagem natural — legível para qualquer investidor."""
    if not result.model_ready:
        # Differentiate the two not-ready states so the user knows whether
        # to retrain (NO_MODEL → file missing or load failed) vs investigate
        # (ERROR → bundle loaded but inference raised — usually feature
        # schema drift between training and runtime).
        if getattr(result, "label", None) == "ERROR":
            return "🤖 _ML carregado mas inferência falhou — ver logs (`/admin_check_config`)_"
        return "🤖 _Modelo não treinado ainda_"

    wp   = result.win_prob or 0.0
    up   = (result.pred_up or 0.0) * 100
    down = abs((result.pred_down or 0.0) * 100)

    # Descrição do sinal em linguagem simples
    if result.label == "WIN_STRONG":
        sinal = "MUITO OPTIMISTA"
    elif result.label == "WIN":
        sinal = "OPTIMISTA"
    elif result.label == "WEAK":
        sinal = "NEUTRO"
    else:
        sinal = "NÃO RECOMENDA"

    # Linha principal
    up_str = f"+{up:.1f}%" if up >= 0 else f"{up:.1f}%"
    linha = f"🤖 *Modelo:* {sinal} — {wp:.0%} de probabilidade de bater o mercado"

    # Contexto de upside/downside em linguagem simples
    contexto = ""
    if up > 3 and wp >= 0.5:
        contexto = f"\n  Prevê *{up_str}* acima do SPY nos próximos 90 dias"
    elif up > 0:
        contexto = f"\n  Upside previsto: *{up_str}* vs SPY — confiança baixa"
    else:
        contexto = f"\n  Modelo não vê vantagem clara face ao SPY agora"

    if down > 8:
        contexto += f" | risco de queda adicional até *{down:.0f}%* antes de recuperar"

    return linha + contexto
