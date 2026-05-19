"""
ml_predictor.py — Score v3: regressor dual (model_up + model_down).

Score final = pred_up (predicted alpha_90d = log-return excess over SPY em 90 dias).
  → representa o excesso de retorno sobre o SPY esperado nos 90 dias
    seguintes ao alerta (escala log-return).

Thresholds (escala alpha_90d):
  _SCORE_HIGH  = 0.06   alpha > +6pp sobre SPY → WIN_STRONG
  _SCORE_MED   = 0.03   alpha > +3pp sobre SPY → WIN
  _SCORE_FLOOR = 0.01   alpha > +1pp sobre SPY → WEAK

Nota: model_down tem rho ≈ 0 (não prevê nada), não entra no score.
Mantemos pred_down em MLResult apenas para diagnóstico.

API pública (inalterada):
  ml_score(features: dict) -> MLResult
  is_model_ready() -> bool
  get_model_info() -> dict
  ml_badge(result: MLResult) -> str    # linha formatada para Telegram

Features esperadas (37 — FEATURE_COLUMNS de ml_features.py):
  Stage 0(4): macro_score, vix, spy_drawdown_5d, sector_drawdown_5d
  Stage 1(5): gross_margin, de_ratio, pe_vs_fair, analyst_upside, quality_score
  Stage 2(5): drop_pct_today, drawdown_52w, rsi_14, atr_ratio, volume_spike
  Stage 3(5): rsi_oversold_strength, vix_regime, pe_attractive,
              drop_x_drawdown, vol_x_drop
  Stage 3b(4): return_1m, return_3m_pre, sector_relative, beta_60d
  Stage 3c(4): quality_dislocation, peg_implicit, relative_drop, month_of_year
  Stage 3d(2): sector_alert_count_7d, days_since_52w_high
  Stage 3e(2): short_interest_ratio, earnings_surprise_avg
  Stage 3f(3): vix_percentile_1y, spy_rsi_14, yield_10y_change_5d
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
    """Stub de compatibilidade para bundles picklados com __main__.DipModelsV3.

    BUG FIX v4.1: alinhado com ml_training/bundle.py (14 campos) para que
    score_calibrator, rho_alpha, topk_pnl não sejam perdidos ao carregar bundles
    gerados pelo training pipeline. Se __main__.DipModelsV3 tiver menos campos
    que o bundle, joblib.load falha silenciosamente e o calibrador não é usado.
    """
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

# Features esperadas pelo bundle — importadas de ml_features para evitar duplicação.
# Se o import falhar (ex: em testes isolados), usa lista vazia e o bundle
# devolve erro explícito (bundle['feature_cols'] é obrigatório após retrain).
try:
    from ml_features import FEATURE_COLUMNS as _FEATURE_COLS
except Exception:
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
    confidence:         str          = "–"
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

    # Sector model blend: se o bundle tiver modelos sectoriais, blendamos
    # 70% global + 30% sector. O sector model especializa-se na dinâmica
    # do seu grupo (ex: RSI + FCF dominam em Tech, macro domina em Financials).
    _sector_models = (_bundle.get("sector_models") or {}) if _bundle else {}
    _stock_sector  = str(features.get("sector") or "Unknown")
    _sector_entry  = _sector_models.get(_stock_sector)
    if _sector_entry is not None:
        try:
            _sector_pred = float(_sector_entry["model"].predict(X)[0])
            _sector_pred = _inverse_transform_up(_sector_pred)
            pred_up = 0.70 * pred_up + 0.30 * _sector_pred
        except Exception as _se:
            logging.debug(f"[ml_predictor] sector model '{_stock_sector}' falhou: {_se}")

    # Score = alpha_90d previsto (log-return excess sobre SPY).
    score = float(pred_up)

    # Normalizar para [0,1] usando o calibrator do bundle (se existir).
    # BUG FIX v4.1: DipModelsV3 stub agora tem score_calibrator → calibrador
    # é efectivamente usado (antes estava a ser descartado por campo em falta).
    score_calibrator = _bundle.get("score_calibrator") if _bundle else None
    win_prob = _score_to_prob(score, calibrator=score_calibrator)

    # VIX regime (informativo)
    vix_value  = enriched.get("vix") or enriched.get("vix_value")
    vix_regime = _classify_vix(vix_value)

    # Labels baseados em win_prob (mais estáveis e independentes do threshold bruto)
    if win_prob > 0.70:
        label      = "WIN_STRONG"
        confidence = "Alta"
    elif win_prob > 0.55:
        label      = "WIN"
        confidence = "Alta" if win_prob > 0.65 else "Média"
    elif win_prob > 0.40:
        label      = "WEAK"
        confidence = "Baixa"
    else:
        label      = "NO_WIN"
        confidence = "–"

    # Classificação de stock e sizing — requerem market_cap e quality_score
    # das features (se disponíveis; fallback neutro se não estiverem).
    mkt_cap_b    = float(enriched.get("market_cap", 0.0) or 0.0) / 1e9
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
        confidence        = confidence,
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
            from prediction_log import log_prediction
            log_prediction(symbol=symbol, features=features or {}, result=result)
        except Exception as e:
            logging.debug(f"[ml_predictor] log_prediction skipped: {e}")

    return result


def ml_badge(result: MLResult) -> str:
    """Linha formatada para o alerta Telegram (v4.1).

    Exemplos:
      🤖 ML:🟢 WIN_STRONG 💎 | α₉₀ +8.2% | dn -6.1% | P(win) 76% | R:R 1.37 | Hold: Longo prazo
      🤖 ML:✅ WIN ⭐ | α₉₀ +5.1% | dn -9.2% | P(win) 62% | Size: 4% | Hold: ~6 meses
      🤖 ML:🔄 WEAK | α₉₀ +2.1% | dn -14% | P(win) 45% | Hold: ~90 dias
      🤖 ML:modelo não treinado
    """
    if not result.model_ready:
        return "🤖 ML:_modelo não treinado_"

    emoji_map = {
        "WIN_STRONG": "🟢",
        "WIN":        "✅",
        "WEAK":       "🟡",
        "NO_WIN":     "🔴",
        "ERROR":      "⚠️",
        "NO_MODEL":   "⚫",
    }
    type_emoji = {"BLUE_CHIP": "💎", "QUALITY": "⭐", "SPECULATIVE": "🔄"}.get(result.stock_type, "")
    hold_label = {"LONG_TERM": "Longo prazo", "6M": "~6 meses", "90D": "~90 dias"}.get(
        result.recommended_hold, result.recommended_hold
    )

    em   = emoji_map.get(result.label, "📊")
    sign = "+" if (result.pred_up or 0.0) >= 0 else ""
    alpha_str = f"{sign}{(result.pred_up or 0.0)*100:.1f}%" if result.pred_up is not None else "?"
    down_str  = f"{(result.pred_down or 0.0)*100:.1f}%" if result.pred_down is not None else "?"
    prob_str  = f"{result.win_prob:.0%}"

    parts = [
        f"🤖 *ML:* {em} `{result.label}` {type_emoji}",
        f"| α₉₀ *{alpha_str}* | dn {down_str} | P(win) *{prob_str}*",
    ]

    if result.position_size_pct > 0:
        parts.append(f"| Size: *{result.position_size_pct*100:.0f}%*")
    if result.risk_reward_ratio > 0:
        parts.append(f"| R:R *{result.risk_reward_ratio:.2f}*")

    parts.append(f"| Hold: {hold_label}")

    if result.vix_regime:
        parts.append(f"| VIX:{result.vix_regime[:3]}")

    return " ".join(parts)
