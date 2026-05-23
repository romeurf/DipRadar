"""
momentum_radar/scanner.py — Detector de breakouts e momentum plays.

Complemento do DipRadar: enquanto o dip scanner procura quedas para recuperação,
o momentum scanner procura stocks em aceleração de upside (Micron, NOW, SanDisk...).

Critérios (rule-based, ML como próxima fase):
  1. Força recente:  return_20d >= 12%  (breakout confirmado)
  2. Volume:         volume_ratio_20d >= 1.4  (compradores institucionais)
  3. RSI:            52 <= rsi_14 <= 78  (momentum forte, não sobrecomprado)
  4. Força relativa: return_20d > sector_etf_return_20d  (melhor que o sector)
  5. Qualidade:      quality_score >= 0.55  (filtra pumps especulativos)
  6. Não em dip:     return_20d > -5%  (não está a cair, está a subir)

Exit: trailing stop -12% do máximo (não target fixo — momentum pode correr muito).

Integração:
  - run_momentum_scan() chamado do main.py (schedule: dia útil 17h45 + 21h15)
  - Posições criadas com position_type="MOMENTUM" no PositionRecord
  - position_monitor.py usa trailing stop para estas posições
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Critérios de entrada
# ─────────────────────────────────────────────────────────────────────────────

_LOOKBACK_DAYS    = 280    # histórico para indicadores
_MIN_RETURN_20D   = 0.12   # +12% em 20 dias — breakout confirmado
_MIN_VOLUME_RATIO = 1.40   # 40% acima da média — confirmação institucional
_RSI_MIN          = 52.0   # acima de 50 = momentum positivo
_RSI_MAX          = 78.0   # abaixo de 80 = não sobrecomprado
_MIN_QUALITY      = 0.55   # filtra pumps especulativos
# NOTA: sem restrição de drawdown — o scanner cobre TODOS os stocks com momentum,
# incluindo aqueles que nunca estiveram em dip (Sandisk, Micron na subida inicial).
# Stocks em dip E em momentum (recuperação rápida) são capturados para transição DIP→MOMENTUM.
_MIN_MARKET_CAP_B = 2.0    # em billions USD — liquidez mínima


# ─────────────────────────────────────────────────────────────────────────────
# Cálculos de indicadores
# ─────────────────────────────────────────────────────────────────────────────

def _calc_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff().dropna()
    if len(delta) < period:
        return 50.0
    gain = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs   = gain / loss.replace(0, np.nan)
    rsi  = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


def _calc_momentum_signals(hist: pd.DataFrame) -> dict:
    """Computa sinais de momentum a partir do histórico de preços."""
    if hist is None or len(hist) < 25:
        return {}
    close  = hist["Close"]
    volume = hist.get("Volume", pd.Series(dtype=float))

    # Return 20d
    if len(close) >= 21:
        ret_20d = float(close.iloc[-1] / close.iloc[-21] - 1)
    else:
        return {}

    # Volume ratio
    if len(volume) >= 20 and volume.sum() > 0:
        vol_ratio = float(volume.iloc[-1] / volume.iloc[-20:].mean()) if volume.iloc[-20:].mean() > 0 else 1.0
    else:
        vol_ratio = 1.0

    # RSI
    rsi = _calc_rsi(close)

    # Drawdown recente (20d) — para confirmar que está a subir, não a cair
    rolling_max_20d = close.iloc[-20:].max() if len(close) >= 20 else close.max()
    dd_20d = float(close.iloc[-1] / rolling_max_20d - 1)

    # Return 5d (momentum imediato — confirma tendência não reverteu)
    ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0.0

    # Distância ao máximo de 52 semanas
    high_52w = close.rolling(252, min_periods=20).max().iloc[-1] if len(close) >= 20 else close.max()
    pct_from_52w_high = float(close.iloc[-1] / high_52w - 1)

    # Market Structure + Volume Flow
    try:
        from ml_features import _anchored_vwap_ratio, _volume_poc_ratio, _compute_mfi, _compute_volume_up_pct
        vwap_low_ratio = _anchored_vwap_ratio(hist, use_high=False)
        poc_ratio      = _volume_poc_ratio(hist, lookback=60)
        mfi_val        = _compute_mfi(hist, period=14)
        vol_up_pct     = _compute_volume_up_pct(hist, lookback=20)
    except Exception:
        vwap_low_ratio = 0.0
        poc_ratio      = 0.0
        mfi_val        = 50.0
        vol_up_pct     = 0.50

    # return_60d_pre (retorno dos 60d antes dos últimos 20d — impulso prévio)
    ret_60d_pre = float(close.iloc[-21] / close.iloc[-81] - 1) if len(close) >= 81 else 0.0

    # ATR % (volatilidade relativa ao preço)
    if len(close) >= 15 and "High" in hist.columns and "Low" in hist.columns:
        h14 = hist["High"].iloc[-14:].values
        l14 = hist["Low"].iloc[-14:].values
        atr_pct = float(np.mean(h14 - l14)) / float(close.iloc[-1]) if close.iloc[-1] > 0 else 0.02
    else:
        atr_pct = 0.02

    # close_in_range_20d (força: fechamentos perto dos máximos de 20d)
    if len(close) >= 20 and "High" in hist.columns and "Low" in hist.columns:
        h20 = hist["High"].iloc[-20:].values
        l20 = hist["Low"].iloc[-20:].values
        c20 = close.iloc[-20:].values
        ranges = h20 - l20
        close_in_range_20d = float(np.mean((c20 - l20) / np.where(ranges > 0, ranges, 1)))
    else:
        close_in_range_20d = 0.5

    return {
        "return_20d":          round(ret_20d, 4),
        "return_5d":           round(ret_5d, 4),
        "return_60d_pre":      round(ret_60d_pre, 4),
        "volume_ratio_20d":    round(vol_ratio, 3),
        "rsi_14":              round(rsi, 1),
        "drawdown_20d":        round(dd_20d, 4),
        "pct_from_52w_high":   round(pct_from_52w_high, 4),
        "atr_pct":             round(atr_pct, 4),
        "close_in_range_20d":  round(close_in_range_20d, 4),
        "vwap_from_low_ratio": round(vwap_low_ratio, 4),
        "volume_poc_ratio":    round(poc_ratio, 4),
        "mfi_14":              round(mfi_val, 2),
        "volume_up_pct_20d":   round(vol_up_pct, 4),
        "price":               round(float(close.iloc[-1]), 4),
    }


# Cache por sector para o scan completo: sem cache, cada candidato que passa os filtros
# faz uma chamada API ao ETF do seu sector — 50 candidatos em 5 sectores = 50 chamadas
# em vez de 5. O cache é por instância de scan (dict mutável como default arg é intencional).
_SECTOR_ETF_CACHE: dict[str, float] = {}


def _sector_return_20d(sector: str, _cache: dict = _SECTOR_ETF_CACHE) -> float:
    """Retorno do ETF sectorial nos últimos 20 dias.

    Cache de sessão: cada ETF é descarregado apenas uma vez por scan completo.
    11 sectores = máximo 11 chamadas, independentemente do número de candidatos.
    """
    if sector in _cache:
        return _cache[sector]

    from ml_training.config import SECTOR_ETF
    from data_feed import get_eod_prices
    etf = SECTOR_ETF.get(sector, "SPY")
    result = 0.0
    try:
        df = get_eod_prices(etf, lookback_days=30)
        if df is not None and not df.empty and "Close" in df.columns:
            close = df.set_index(pd.to_datetime(df["date"]))["Close"].sort_index() if "date" in df.columns else df["Close"]
            if len(close) >= 21:
                result = float(close.iloc[-1] / close.iloc[-21] - 1)
    except Exception as e:
        log.debug(f"[momentum] sector ETF {etf} falhou: {e}")

    _cache[sector] = result
    return result


def clear_sector_cache() -> None:
    """Limpa o cache de ETFs sectoriais. Chamar entre scans se necessário."""
    _SECTOR_ETF_CACHE.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Score de momentum (0-100)
# ─────────────────────────────────────────────────────────────────────────────

def score_momentum(signals: dict, fund: dict, sector_ret_20d: float = 0.0) -> tuple[float, list[str]]:
    """Calcula score de momentum (0-100) e lista de razões.

    Diferente do score de dip (que premia quedas grandes), o score de momentum
    premia stocks com upside contínuo, volume e força relativa.
    """
    reasons: list[str] = []
    score = 0.0

    ret_20d     = signals.get("return_20d", 0.0)
    vol_ratio   = signals.get("volume_ratio_20d", 1.0)
    rsi         = signals.get("rsi_14", 50.0)
    dd_20d      = signals.get("drawdown_20d", 0.0)
    ret_5d      = signals.get("return_5d", 0.0)
    high_pct    = signals.get("pct_from_52w_high", -1.0)
    quality     = float(fund.get("quality_score") or 0.0)

    # Força recente (40 pontos)
    if ret_20d >= 0.25:
        score += 40; reasons.append(f"+{ret_20d*100:.0f}% em 20d (breakout forte)")
    elif ret_20d >= 0.18:
        score += 32; reasons.append(f"+{ret_20d*100:.0f}% em 20d (breakout)")
    elif ret_20d >= 0.12:
        score += 22; reasons.append(f"+{ret_20d*100:.0f}% em 20d (momentum)")
    else:
        return 0.0, []  # abaixo do mínimo — rejeitar

    # Volume (20 pontos)
    if vol_ratio >= 2.0:
        score += 20; reasons.append(f"Volume {vol_ratio:.1f}× médio (pressão institucional forte)")
    elif vol_ratio >= 1.6:
        score += 15; reasons.append(f"Volume {vol_ratio:.1f}× médio")
    elif vol_ratio >= 1.4:
        score += 10; reasons.append(f"Volume {vol_ratio:.1f}× médio")

    # RSI favorável (15 pontos)
    if 58 <= rsi <= 72:
        score += 15; reasons.append(f"RSI {rsi:.0f} — zona de momentum óptima")
    elif 52 <= rsi < 58 or 72 < rsi <= 78:
        score += 8; reasons.append(f"RSI {rsi:.0f}")
    elif rsi > 78:
        score -= 10; reasons.append(f"RSI {rsi:.0f} — sobrecomprado, cuidado")

    # Força relativa vs sector (10 pontos)
    rel_strength = ret_20d - sector_ret_20d
    if rel_strength >= 0.08:
        score += 10; reasons.append(f"Outperform sector +{rel_strength*100:.0f}pp")
    elif rel_strength >= 0.04:
        score += 5; reasons.append(f"Outperform sector +{rel_strength*100:.0f}pp")

    # Qualidade do negócio (10 pontos)
    if quality >= 0.70:
        score += 10; reasons.append("Qualidade fundamental alta")
    elif quality >= 0.55:
        score += 5; reasons.append("Qualidade fundamental moderada")

    # Momentum imediato confirmado (5 pontos)
    if ret_5d >= 0.03:
        score += 5; reasons.append(f"+{ret_5d*100:.0f}% nos últimos 5d")

    # Perto do máximo de 52w (bonus: breakout real)
    if high_pct >= -0.03:
        score += 5; reasons.append("A testar máximos de 52 semanas")
    elif high_pct >= -0.08:
        score += 2

    # Penalidade: RSI sobrecomprado (> 80)
    if rsi > 80:
        score -= 15

    return round(max(0.0, min(100.0, score)), 1), reasons


# ─────────────────────────────────────────────────────────────────────────────
# ML multi-horizonte (L1/L2/L3)
# ─────────────────────────────────────────────────────────────────────────────

_MOMENTUM_BUNDLE: dict | None = None
_BUNDLE_MTIME: float = 0.0


def _load_bundle() -> dict | None:
    """Carrega bundle multi-horizonte (com cache em memória)."""
    global _MOMENTUM_BUNDLE, _BUNDLE_MTIME
    from pathlib import Path
    bundle_path = Path("/data/momentum_model.pkl") if Path("/data").exists() else Path("/tmp/momentum_model.pkl")
    if not bundle_path.exists():
        return None
    mtime = bundle_path.stat().st_mtime
    if _MOMENTUM_BUNDLE is None or mtime != _BUNDLE_MTIME:
        try:
            import joblib
            _MOMENTUM_BUNDLE = joblib.load(bundle_path)
            _BUNDLE_MTIME = mtime
            log.info("[momentum] Bundle ML carregado")
        except Exception as e:
            log.warning(f"[momentum] Erro ao carregar bundle: {e}")
            return None
    return _MOMENTUM_BUNDLE


def _ml_score_momentum(signals: dict) -> dict:
    """Aplica modelos L1/L2/L3 e devolve scores e durability tag.

    Retorna dict:
      ml_score_l1, ml_score_l2, ml_score_l3, ml_score_ensemble,
      ml_durability ("DURÁVEL" | "TÁCTICO" | "MODERADO"),
      ml_ready (bool)
    """
    bundle = _load_bundle()
    if bundle is None:
        return {"ml_ready": False}

    feat_cols = bundle.get("feature_cols", [])
    if not feat_cols:
        return {"ml_ready": False}

    # Construir vector de features
    feat_vec = np.array(
        [float(signals.get(c, 0.0) or 0.0) for c in feat_cols],
        dtype=np.float32
    ).reshape(1, -1)

    # Modelos multi-horizonte (bundle novo)
    models: dict = bundle.get("models", {})
    scores: dict[str, float] = {}

    if models:
        for label, info in models.items():
            m   = info.get("model")
            means = info.get("means", {})
            w   = info.get("weight", 1.0)
            if m is None:
                continue
            # Aplicar means individuais do horizonte
            fv = np.array(
                [float(signals.get(c, means.get(c, 0.0)) or means.get(c, 0.0)) for c in feat_cols],
                dtype=np.float32
            ).reshape(1, -1)
            try:
                pred = float(m.predict(fv)[0])
                scores[label] = round(pred, 4)
            except Exception:
                pass
    else:
        # Bundle antigo (single model): sem informação de horizonte.
        # Retornamos o score único como ensemble; durabilidade não é determinável.
        m = bundle.get("model")
        if m is None:
            return {"ml_ready": False}
        try:
            pred = float(m.predict(feat_vec)[0])
        except Exception:
            return {"ml_ready": False}
        return {
            "ml_ready":          True,
            "ml_score_l1":       round(pred, 4),
            "ml_score_l2":       round(pred, 4),
            "ml_score_l3":       round(pred, 4),
            "ml_score_ensemble": round(pred, 4),
            "ml_durability":     "MODERADO",  # indeterminável sem multi-horizonte
        }

    if not scores:
        return {"ml_ready": False}

    # Ensemble ponderado com os pesos do bundle (já normalizados)
    total_w = 0.0
    ensemble = 0.0
    for label, info in models.items():
        if label in scores:
            w = float(info.get("weight", 1.0))
            ensemble += scores[label] * w
            total_w  += w
    if total_w > 0:
        ensemble = ensemble / total_w
    else:
        ensemble = float(np.mean(list(scores.values())))

    # Durability: L3 (180d) vs L1 (30d) — só significativo com multi-horizonte
    l1 = scores.get("L1", ensemble)
    l3 = scores.get("L3", ensemble)
    has_multi = "L1" in scores and "L3" in scores
    if has_multi and l3 > l1 * 1.2:
        durability = "DURÁVEL"   # momentum esperado > 6 meses (ex: RKLB)
    elif has_multi and l1 > l3 * 1.3:
        durability = "TÁCTICO"   # pump de curto prazo
    else:
        durability = "MODERADO"

    return {
        "ml_ready":          True,
        "ml_score_l1":       round(scores.get("L1", 0.0), 4),
        "ml_score_l2":       round(scores.get("L2", 0.0), 4),
        "ml_score_l3":       round(scores.get("L3", 0.0), 4),
        "ml_score_ensemble": round(ensemble, 4),
        "ml_durability":     durability,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scanner principal
# ─────────────────────────────────────────────────────────────────────────────

def scan_momentum_universe(
    tickers: list[str] | None = None,
    min_score: float = 60.0,
    max_results: int = 10,
    sleep_between: float = 0.08,
) -> list[dict]:
    """Varre o universo à procura de momentum plays.

    Devolve lista de candidatos ordenados por score, cada um com:
      ticker, score, price, return_20d, volume_ratio_20d, rsi_14, reasons, fund
    """
    from universe import get_ml_universe
    from data_feed import get_eod_prices
    from market_client import get_fundamentals

    if tickers is None:
        try:
            tickers = get_ml_universe()
        except Exception as e:
            log.error(f"[momentum] get_ml_universe falhou: {e}")
            return []

    log.info(f"[momentum] A varrer {len(tickers)} tickers para momentum...")
    candidates: list[dict] = []
    scanned = 0

    for ticker in tickers:
        try:
            # Preços
            df = get_eod_prices(ticker, lookback_days=_LOOKBACK_DAYS)
            if df is None or df.empty or "Close" not in df.columns:
                continue
            if "date" in df.columns:
                df = df.set_index(pd.to_datetime(df["date"])).sort_index()

            # Normalizar nomes de colunas para lowercase (Tiingo vs yfinance diferem)
            df.columns = [c.capitalize() if c.lower() in ("open","high","low","close","volume","adj close") else c for c in df.columns]
            signals = _calc_momentum_signals(df)
            if not signals:
                continue

            # Filtros rígidos antes de buscar fundamentais (caro).
            # Sem filtro de drawdown: o scanner cobre TODOS os stocks em momentum,
            # incluindo os que nunca estiveram em dip (crescimento contínuo tipo Sandisk).
            if signals["return_20d"] < _MIN_RETURN_20D:
                continue
            if signals["volume_ratio_20d"] < _MIN_VOLUME_RATIO:
                continue
            if not (_RSI_MIN <= signals["rsi_14"] <= _RSI_MAX):
                continue

            # Fundamentais
            try:
                fund = get_fundamentals(ticker, min_market_cap=int(_MIN_MARKET_CAP_B * 1e9))
                if fund.get("skip"):
                    continue
            except Exception:
                fund = {}

            quality = float(fund.get("quality_score") or 0.0)
            if quality < _MIN_QUALITY:
                continue

            sector = fund.get("sector", "Unknown") or "Unknown"
            sect_ret = _sector_return_20d(sector)

            score, reasons = score_momentum(signals, fund, sector_ret)
            if score < min_score:
                continue

            ml = _ml_score_momentum(signals)

            candidates.append({
                "ticker":              ticker,
                "score":               score,
                "price":               signals["price"],
                "return_20d":          signals["return_20d"],
                "return_5d":           signals["return_5d"],
                "volume_ratio_20d":    signals["volume_ratio_20d"],
                "rsi_14":              signals["rsi_14"],
                "pct_from_52w_high":   signals["pct_from_52w_high"],
                "sector":              sector,
                "reasons":             reasons,
                "fund":                fund,
                **ml,
            })
            scanned += 1

            if sleep_between > 0:
                time.sleep(sleep_between)

        except Exception as e:
            log.debug(f"[momentum] {ticker}: {e}")

    log.info(f"[momentum] Scan concluído: {scanned} tickers passaram os filtros básicos, "
             f"{len(candidates)} com score >= {min_score}")

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:max_results]


# ─────────────────────────────────────────────────────────────────────────────
# Formato do alerta Telegram
# ─────────────────────────────────────────────────────────────────────────────

def format_momentum_alert(candidate: dict) -> str:
    """Formata um alerta de momentum para Telegram."""
    ticker   = candidate["ticker"]
    score    = candidate["score"]
    price    = candidate["price"]
    ret_20d  = candidate["return_20d"]
    vol      = candidate["volume_ratio_20d"]
    rsi      = candidate["rsi_14"]
    sector   = candidate["sector"]
    reasons  = candidate["reasons"]
    high_pct = candidate.get("pct_from_52w_high", 0.0)
    fund     = candidate.get("fund", {})

    high_str = f"+{abs(high_pct)*100:.1f}% acima do máx 52w" if high_pct >= 0 else f"{high_pct*100:.1f}% abaixo do máx 52w"

    lines = [
        f"🚀 *MOMENTUM — {ticker}*  [Score: {score:.0f}]",
        f"_{sector}_",
        "",
        f"💰 Preço: ${price:.2f}  |  +{ret_20d*100:.1f}% em 20d  |  {high_str}",
        f"📊 Volume: {vol:.1f}× médio  |  RSI: {rsi:.0f}",
    ]

    # ML multi-horizonte
    if candidate.get("ml_ready"):
        l1  = candidate.get("ml_score_l1",       0.0)
        l2  = candidate.get("ml_score_l2",       0.0)
        l3  = candidate.get("ml_score_l3",       0.0)
        ens = candidate.get("ml_score_ensemble", 0.0)
        dur = candidate.get("ml_durability",     "—")
        dur_emoji = {"DURÁVEL": "🏃", "TÁCTICO": "⚡", "MODERADO": "📈"}.get(dur, "")
        lines.append("")
        lines.append(
            f"🤖 *ML Multi-Horizonte* {dur_emoji} _{dur}_\n"
            f"  L1 30d: {l1:+.1%}  |  L2 90d: {l2:+.1%}  |  L3 180d: {l3:+.1%}\n"
            f"  Ensemble: {ens:+.1%}"
        )

    if reasons:
        lines.append("")
        lines.append("*Sinais:*")
        for r in reasons[:4]:
            lines.append(f"  • {r}")

    name = fund.get("name", "")
    if name:
        lines.append("")
        lines.append(f"_{name}_")

    # Estratégia adaptada à durabilidade
    dur = candidate.get("ml_durability", "MODERADO")
    if dur == "DURÁVEL":
        strategy = "Momentum duradouro (padrão RKLB/SanDisk). Hold 3-6 meses, trailing stop -15%."
    elif dur == "TÁCTICO":
        strategy = "Pump de curto prazo. Entrada rápida, stop apertado -8%, sair em 30d."
    else:
        strategy = "Momentum moderado. Trailing stop -12% do máximo."

    lines += [
        "",
        f"💡 *Estratégia:* {strategy}",
        f"_Diferente de um dip — o stock está a subir. Exit: trailing stop, não target fixo._",
    ]
    return "\n".join(lines)
