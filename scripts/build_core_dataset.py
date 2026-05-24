"""
scripts/build_core_dataset.py — Backfill histórico para CoreML.

Detecta episódios em que cada ticker estava perto da sua MA200w
e constrói rows de treino com as features do CoreRadar.

Universo de treino: core_universe.json (os mesmos tickers usados em produção).
O modelo aprende APENAS o padrão de quality compounders — incluir tickers
especulativos seria ruído porque o comportamento na MA200w é radicalmente
diferente.  Para adicionar mais tickers de treino (p.ex. NVDA, AMZN),
adiciona-os a core_universe.json ou usa --tickers.

Episódio: preço ≤5% ACIMA da MA200w (ratio_to_ma200w <= 1.05) — zona de suporte.
Target:  alpha_1y e alpha_2y (excesso de retorno vs SPY)
Features: price/MA/RSI/volume + sector ETF context — point-in-time sem look-ahead.
          3 features de setor (sem modelos separados por setor — um modelo unificado
          com setor como feature contínua; sem perda de dados):
            sector_etf_vs_ma200w:   ETF do setor relativo à sua própria MA200w
            stock_vs_sector_spread: stock dip relativo ao setor (stock-specific vs sector-wide)
            sector_return_20d:      momentum do setor nos últimos 20d
          Fundamentais (yfinance) marcados como não-PIT e opcionais.

Uso:
  python scripts/build_core_dataset.py                          # core_universe.json
  python scripts/build_core_dataset.py --tickers MSFT,AAPL     # subset/teste
  python scripts/build_core_dataset.py --no-fundamentals        # mais rápido
  python scripts/build_core_dataset.py --force-rebuild          # ignora parquet existente
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.regenerate_training_base import (
    _fetch_one_ticker,
    _fetch_batch,
    MACRO_TICKERS,
    SECTOR_ETFS,
    SECTOR_ETF,
)

log = logging.getLogger("build_core_dataset")

_DATA_DIR  = Path("/data") if Path("/data").exists() else Path("/tmp")
_CACHE_DIR = _DATA_DIR / "price_cache"
_OUTPUT    = _REPO_ROOT / "core_training_base.parquet"

# Episódio: perto ou abaixo da MA200w (dentro de 5% acima)
_MA200W_ENTRY_RATIO = 1.05   # price/MA200w <= 1.05 → episódio
_MA200W_EXIT_RATIO  = 1.12   # price/MA200w > 1.12 → episódio terminou (recuperação clara)
_MIN_WEEKS_FOR_MA   = 200    # mínimo de semanas para computar MA200w fiável
_MIN_BARS_AHEAD_1Y  = 252    # dias úteis necessários para target de 1 ano
_MIN_BARS_AHEAD_2Y  = 504    # dias úteis necessários para target de 2 anos


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento do universo core (core_universe.json)
# ─────────────────────────────────────────────────────────────────────────────

def _load_core_tickers() -> set[str]:
    """Devolve conjunto de tickers marcados como core_company."""
    universe_file = _REPO_ROOT / "core_universe.json"
    if not universe_file.exists():
        return set()
    try:
        import json
        data = json.loads(universe_file.read_text(encoding="utf-8"))
        return {e["ticker"].upper() for e in data.get("tickers", []) if e.get("ticker")}
    except Exception:
        return set()


def _load_core_yields() -> dict[str, float]:
    """Devolve mapa ticker → annual_yield_pct para os tickers core."""
    universe_file = _REPO_ROOT / "core_universe.json"
    if not universe_file.exists():
        return {}
    try:
        import json
        data = json.loads(universe_file.read_text(encoding="utf-8"))
        return {
            e["ticker"].upper(): float(e.get("annual_yield_pct") or 0.0)
            for e in data.get("tickers", []) if e.get("ticker")
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Cálculo da MA200w (resampled para semanal)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_weekly_ma200_pit(close_pit: pd.Series) -> Optional[float]:
    """MA200w point-in-time: resample para W-FRI, rolling 200 semanas, último valor."""
    try:
        weekly = close_pit.resample("W-FRI").last().dropna()
        if len(weekly) < _MIN_WEEKS_FOR_MA:
            return None
        ma200w = weekly.rolling(200).mean().iloc[-1]
        return float(ma200w) if math.isfinite(float(ma200w)) else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Cálculo do RSI14
# ─────────────────────────────────────────────────────────────────────────────

def _compute_rsi14(close_pit: pd.Series) -> float:
    try:
        delta = close_pit.diff()
        gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        last_loss = float(loss.iloc[-1])
        rs    = float(gain.iloc[-1]) / last_loss if last_loss > 0 else 100.0
        return round(100 - 100 / (1 + rs), 1)
    except Exception:
        return 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Sinais de volume / fluxo institucional (point-in-time)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_volume_signals_pit(
    close_pit: pd.Series, volume_pit: pd.Series
) -> dict:
    """Replica _compute_volume_signals de core_radar.py com dados point-in-time."""
    signals: dict = {}
    n = len(close_pit)
    if n < 20 or volume_pit is None or len(volume_pit) < 20:
        return {"obv_trend_20d": 0.0, "up_down_vol_ratio_20d": 1.0,
                "high_vol_up_days_20d": 0, "vol_spike_5d_vs_60d": 1.0}
    try:
        # OBV
        direction = np.sign(close_pit.diff().fillna(0).values)
        obv = np.cumsum(direction * volume_pit.values)
        obv_series = pd.Series(obv, index=close_pit.index)
        obv_20d_change = float(obv_series.iloc[-1] - obv_series.iloc[-21]) if n >= 21 else 0.0
        avg_vol = float(volume_pit.rolling(60).mean().iloc[-1]) or 1.0
        signals["obv_trend_20d"] = round(obv_20d_change / avg_vol, 2)
    except Exception:
        signals["obv_trend_20d"] = 0.0

    try:
        # Up/Down volume ratio 20d
        last_20_close  = close_pit.iloc[-20:]
        last_20_volume = volume_pit.iloc[-20:]
        daily_ret = last_20_close.pct_change().fillna(0)
        up_vol   = float(last_20_volume[daily_ret > 0].sum())
        down_vol = float(last_20_volume[daily_ret <= 0].sum())
        signals["up_down_vol_ratio_20d"] = round(up_vol / down_vol if down_vol > 0 else 1.0, 3)
    except Exception:
        signals["up_down_vol_ratio_20d"] = 1.0

    try:
        # High-volume up days
        avg_vol_20 = float(volume_pit.iloc[-20:].mean()) or 1.0
        last_20_close  = close_pit.iloc[-20:]
        last_20_volume = volume_pit.iloc[-20:]
        daily_ret = last_20_close.pct_change().fillna(0)
        hvup = int(((last_20_volume > avg_vol_20 * 1.5) & (daily_ret > 0)).sum())
        signals["high_vol_up_days_20d"] = hvup
    except Exception:
        signals["high_vol_up_days_20d"] = 0

    try:
        # Volume spike 5d vs 60d
        avg_5d  = float(volume_pit.iloc[-5:].mean())  or 1.0
        avg_60d = float(volume_pit.iloc[-60:].mean()) or 1.0
        signals["vol_spike_5d_vs_60d"] = round(avg_5d / avg_60d, 3)
    except Exception:
        signals["vol_spike_5d_vs_60d"] = 1.0

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Detecção de episódios MA200w
# ─────────────────────────────────────────────────────────────────────────────

def _find_ma200w_episodes(hist: pd.DataFrame) -> list[pd.Timestamp]:
    """Detecta datas em que o preço entrou na zona de suporte MA200w.

    Episódio inicia quando price/MA200w <= _MA200W_ENTRY_RATIO.
    Episódio termina quando price/MA200w > _MA200W_EXIT_RATIO.
    Só o primeiro dia qualificado de cada episódio é amostrado.
    Requer pelo menos _MIN_WEEKS_FOR_MA semanas de história acumulada.
    """
    close = hist["Close"]
    n = len(hist)
    episode_dates: list[pd.Timestamp] = []
    in_episode = False

    # Pré-computar MA200w em cada ponto — computação eficiente
    weekly_close = close.resample("W-FRI").last().dropna()

    # Mapeamento de data diária → ratio_to_ma200w calculado com dados até essa data
    # Para eficiência, percorre semanalmente e interpola para diário
    for i in range(_MIN_BARS_AHEAD_1Y, n):
        # Slice ponto-no-tempo: só dados até barra i
        close_pit = close.iloc[:i+1]
        price = float(close_pit.iloc[-1])

        # MA200w PIT (semanal resample até esta data)
        weekly_pit = close_pit.resample("W-FRI").last().dropna()
        if len(weekly_pit) < _MIN_WEEKS_FOR_MA:
            continue  # ainda sem história suficiente para MA200w

        ma200w_val = float(weekly_pit.rolling(200).mean().iloc[-1])
        if not math.isfinite(ma200w_val) or ma200w_val <= 0:
            continue

        ratio = price / ma200w_val

        if in_episode and ratio > _MA200W_EXIT_RATIO:
            in_episode = False  # recuperação clara — episódio terminou

        if in_episode:
            continue  # ainda no mesmo episódio — não re-amostrar

        if ratio <= _MA200W_ENTRY_RATIO:
            in_episode = True
            episode_dates.append(hist.index[i])

    return episode_dates


# ─────────────────────────────────────────────────────────────────────────────
# Targets: alpha_1y e alpha_2y (excesso vs SPY)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_alpha_targets(
    hist: pd.DataFrame,
    alert_date: pd.Timestamp,
    spy_hist: Optional[pd.DataFrame],
) -> dict:
    """alpha_1y e alpha_2y = log-return stock - log-return SPY nos horizontes."""
    entry_slice = hist[hist.index <= alert_date]
    if entry_slice.empty:
        return {"alpha_1y": float("nan"), "alpha_2y": float("nan")}
    entry_price = float(entry_slice["Close"].iloc[-1])
    if entry_price <= 0:
        return {"alpha_1y": float("nan"), "alpha_2y": float("nan")}

    results = {}
    for label, days in [("alpha_1y", 365), ("alpha_2y", 730)]:
        target_date = alert_date + pd.Timedelta(days=days)
        fwd = hist[(hist.index > alert_date) & (hist.index <= target_date + pd.Timedelta(days=30))]
        if len(fwd) < 50:
            results[label] = float("nan")
            continue
        exit_price = float(fwd["Close"].iloc[-1])
        stock_ret  = math.log(exit_price / entry_price) if exit_price > 0 else float("nan")
        if spy_hist is not None:
            spy_entry_sl = spy_hist[spy_hist.index <= alert_date]
            spy_fwd      = spy_hist[(spy_hist.index > alert_date) & (spy_hist.index <= target_date + pd.Timedelta(days=30))]
            if not spy_entry_sl.empty and len(spy_fwd) >= 50:
                spy_e = float(spy_entry_sl["Close"].iloc[-1])
                spy_x = float(spy_fwd["Close"].iloc[-1])
                spy_ret = math.log(spy_x / spy_e) if spy_e > 0 and spy_x > 0 else float("nan")
            else:
                spy_ret = float("nan")
        else:
            spy_ret = float("nan")

        if math.isfinite(stock_ret) and math.isfinite(spy_ret):
            results[label] = round(stock_ret - spy_ret, 6)
        else:
            results[label] = float("nan")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Features fundamentais (yfinance — NÃO PIT para dados históricos)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_fundamentals_yf(ticker: str) -> dict:
    """Busca fundamentais via yfinance.info — apenas para uso em episódios recentes (< 6 meses)."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}

        def _safe(key: str) -> Optional[float]:
            v = info.get(key)
            return float(v) if v is not None and math.isfinite(float(v)) else None

        revenue_growth = _safe("revenueGrowth")
        gross_margin   = _safe("grossMargins")
        roe            = _safe("returnOnEquity")
        free_cashflow  = _safe("freeCashflow")
        market_cap     = _safe("marketCap")
        fcf_yield      = (free_cashflow / market_cap) if (free_cashflow and market_cap and market_cap > 0) else None
        analyst_upside = _safe("targetMeanPrice")
        current_price  = _safe("currentPrice") or _safe("previousClose")
        if analyst_upside and current_price and current_price > 0:
            analyst_upside = (analyst_upside / current_price) - 1.0
        else:
            analyst_upside = None
        earnings_growth = _safe("earningsGrowth")

        return {
            "revenue_growth":  round(revenue_growth,  4) if revenue_growth  is not None else None,
            "gross_margin":    round(gross_margin,     4) if gross_margin    is not None else None,
            "roe":             round(roe,              4) if roe             is not None else None,
            "fcf_yield_fund":  round(fcf_yield,        4) if fcf_yield       is not None else None,
            "analyst_upside":  round(analyst_upside,   4) if analyst_upside  is not None else None,
            "earnings_growth": round(earnings_growth,  4) if earnings_growth is not None else None,
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder principal
# ─────────────────────────────────────────────────────────────────────────────

def _sector_etf_features_pit(
    alert_date: pd.Timestamp,
    sector_etf_hist: Optional[pd.DataFrame],
    price_vs_ma200w: float,
) -> dict:
    """Calcula features de contexto de setor point-in-time."""
    _default = {"sector_etf_vs_ma200w": 0.0, "stock_vs_sector_spread": price_vs_ma200w, "sector_return_20d": 0.0}
    if sector_etf_hist is None:
        return _default
    try:
        se = sector_etf_hist[sector_etf_hist.index <= alert_date]["Close"]
        if len(se) < _MIN_WEEKS_FOR_MA * 5:
            return _default
        se_ma200w = _compute_weekly_ma200_pit(se)
        if not se_ma200w or se_ma200w <= 0:
            return _default
        se_vs_ma200w = round(float(se.iloc[-1]) / se_ma200w - 1, 4)
        se_ret_20d   = round(float(se.iloc[-1] / se.iloc[-21] - 1), 4) if len(se) >= 21 else 0.0
        return {
            "sector_etf_vs_ma200w":   se_vs_ma200w,
            "stock_vs_sector_spread": round(price_vs_ma200w - se_vs_ma200w, 4),
            "sector_return_20d":      se_ret_20d,
        }
    except Exception:
        return _default


def build_core_dataset(
    tickers: list[str],
    price_cache: dict[str, pd.DataFrame],
    spy_hist: Optional[pd.DataFrame],
    sector_etf_cache: dict[str, pd.DataFrame],
    core_tickers: set[str],
    core_yields: dict[str, float],
    add_fundamentals: bool = True,
) -> pd.DataFrame:
    """Constrói dataset de treino para CoreML varrendo histórico de episódios MA200w."""
    import yfinance as _yf

    rows: list[dict] = []
    n_tickers = len(tickers)
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    six_months_ago = today - pd.Timedelta(days=182)

    for i, ticker in enumerate(tickers):
        hist = price_cache.get(ticker)
        if hist is None or len(hist) < _MIN_BARS_AHEAD_1Y + _MIN_WEEKS_FOR_MA * 5:
            continue

        # Normalizar índice para timezone-naive
        if hist.index.tz is not None:
            hist = hist.copy()
            hist.index = hist.index.tz_localize(None)

        is_core = int(ticker in core_tickers)
        ann_yield = core_yields.get(ticker, 0.0)

        # Sector ETF para este ticker (fetch uma vez, reutilizar em todos os episódios)
        sector = "Unknown"
        try:
            info = _yf.Ticker(ticker).info or {}
            sector = info.get("sector") or "Unknown"
        except Exception:
            pass
        etf_key = SECTOR_ETF.get(sector, "SPY")
        _se = sector_etf_cache.get(etf_key)
        sector_etf_hist = _se if _se is not None else sector_etf_cache.get("SPY")
        if sector_etf_hist is not None and sector_etf_hist.index.tz is not None:
            sector_etf_hist = sector_etf_hist.copy()
            sector_etf_hist.index = sector_etf_hist.index.tz_localize(None)

        episode_dates = _find_ma200w_episodes(hist)
        if not episode_dates:
            continue

        for alert_date in episode_dates:
            close_pit  = hist[hist.index <= alert_date]["Close"]
            volume_pit = hist[hist.index <= alert_date]["Volume"] if "Volume" in hist.columns else None
            if len(close_pit) < 21:
                continue

            price = float(close_pit.iloc[-1])

            # ── Preço e MA ──────────────────────────────────────────────────
            ma200w = _compute_weekly_ma200_pit(close_pit)
            if ma200w is None or ma200w <= 0:
                continue
            price_vs_ma200w = round(price / ma200w - 1, 4)

            ma_p   = min(200, len(close_pit) - 1)
            ma200d = float(close_pit.rolling(ma_p).mean().iloc[-1]) if ma_p >= 20 else None
            ma200d_trend = 0.0
            if ma200d and len(close_pit) >= ma_p + 60:
                ma200d_60d = float(close_pit.rolling(ma_p).mean().iloc[-61])
                ma200d_trend = round((ma200d / ma200d_60d - 1) if ma200d_60d > 0 else 0.0, 4)

            ret_20d = round(float(close_pit.iloc[-1] / close_pit.iloc[-21] - 1), 4) if len(close_pit) >= 21 else 0.0
            ret_5d  = round(float(close_pit.iloc[-1] / close_pit.iloc[-6]  - 1), 4) if len(close_pit) >= 6  else 0.0

            high_52w = float(close_pit.rolling(min(252, len(close_pit))).max().iloc[-1])
            dd_52w   = round(price / high_52w - 1, 4) if high_52w > 0 else 0.0

            rsi = _compute_rsi14(close_pit)

            # ── Volume / fluxo institucional ────────────────────────────────
            vol_sigs = _compute_volume_signals_pit(close_pit, volume_pit) if volume_pit is not None else {
                "obv_trend_20d": 0.0, "up_down_vol_ratio_20d": 1.0,
                "high_vol_up_days_20d": 0, "vol_spike_5d_vs_60d": 1.0,
            }

            # ── Setor ETF context ────────────────────────────────────────────
            se_feats = _sector_etf_features_pit(alert_date, sector_etf_hist, price_vs_ma200w)

            # ── Targets ─────────────────────────────────────────────────────
            targets = _compute_alpha_targets(hist, alert_date, spy_hist)

            # ── Fundamentais (só se recente e pedido) ───────────────────────
            fund: dict = {}
            if add_fundamentals and alert_date >= six_months_ago:
                try:
                    fund = _fetch_fundamentals_yf(ticker)
                except Exception:
                    pass

            row = {
                "alert_date":             alert_date,
                "ticker":                 ticker,
                "sector":                 sector,
                "is_core_company":        is_core,
                "annual_yield_pct":       ann_yield,
                "price":                  round(price, 4),
                "price_vs_ma200w":        price_vs_ma200w,
                "ma200d_trend_60d":       ma200d_trend,
                "return_20d":             ret_20d,
                "return_5d":              ret_5d,
                "dd_from_52w_high":       dd_52w,
                "rsi_14":                 rsi,
                "obv_trend_20d":          vol_sigs.get("obv_trend_20d",       0.0),
                "up_down_vol_ratio_20d":  vol_sigs.get("up_down_vol_ratio_20d", 1.0),
                "high_vol_up_days_20d":   vol_sigs.get("high_vol_up_days_20d",  0),
                "vol_spike_5d_vs_60d":    vol_sigs.get("vol_spike_5d_vs_60d",   1.0),
                "sector_etf_vs_ma200w":   se_feats["sector_etf_vs_ma200w"],
                "stock_vs_sector_spread": se_feats["stock_vs_sector_spread"],
                "sector_return_20d":      se_feats["sector_return_20d"],
                "revenue_growth":         fund.get("revenue_growth"),
                "gross_margin":           fund.get("gross_margin"),
                "roe":                    fund.get("roe"),
                "fcf_yield_fund":         fund.get("fcf_yield_fund"),
                "analyst_upside":         fund.get("analyst_upside"),
                "earnings_growth":        fund.get("earnings_growth"),
                "source":                 "backfill",
                **targets,
            }
            rows.append(row)

        if (i + 1) % 50 == 0 or (i + 1) == n_tickers:
            log.info(f"  [{i+1}/{n_tickers}] {ticker} | {len(rows)} episódios MA200w total")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Backfill histórico para treino do CoreML")
    p.add_argument("--tickers",         help="Tickers separados por vírgula (teste rápido)")
    p.add_argument("--start",           default="1993-01-01", help="Data de início (default: 1993 — igual ao DipRadar/MomentumRadar)")
    p.add_argument("--out",             default=str(_OUTPUT),  help="Output parquet")
    p.add_argument("--cache-dir",       default=str(_CACHE_DIR), help="Dir cache de preços")
    p.add_argument("--no-fundamentals", action="store_true", help="Saltar fetch de fundamentais recentes")
    p.add_argument("--force-rebuild",   action="store_true", help="Ignorar parquet existente")
    p.add_argument("-v", "--verbose",   action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
    )

    cache_dir = Path(args.cache_dir)
    out_path  = Path(args.out)

    # ── Universo de tickers ───────────────────────────────────────────────────
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
        log.info(f"Modo explícito: {len(tickers)} tickers")
    else:
        # Treinar APENAS em quality compounders — outros tickers são ruído para esta tese
        tickers = sorted(_load_core_tickers())
        if not tickers:
            log.error("core_universe.json vazio ou não encontrado — usa --tickers para especificar")
            return 1
        log.info(f"Universo core: {len(tickers)} tickers ({', '.join(tickers)})")

    core_tickers = _load_core_tickers()
    core_yields  = _load_core_yields()
    log.info(f"Core companies: {len(core_tickers)} tickers ({', '.join(sorted(core_tickers))})")

    # ── Modo incremental ──────────────────────────────────────────────────────
    existing_df: Optional[pd.DataFrame] = None
    start = args.start
    if out_path.exists() and not args.tickers and not args.force_rebuild:
        try:
            existing_df = pd.read_parquet(out_path)
            if "alert_date" in existing_df.columns and not existing_df.empty:
                last_date = pd.to_datetime(existing_df["alert_date"]).max()
                cutoff = (last_date - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
                start = cutoff
                log.info(f"Dataset existente: {len(existing_df)} rows até {last_date.date()}")
                log.info(f"Modo incremental: desde {start}")
        except Exception as e:
            log.warning(f"Parquet existente ilegível ({e}) — a reconstruir de raiz")
            existing_df = None

    # ── Fetch SPY, sector ETFs e preços ──────────────────────────────────────
    log.info(f"A descarregar SPY + sector ETFs desde {start}...")
    macro_etf_cache: dict[str, pd.DataFrame] = {}
    for t in MACRO_TICKERS + SECTOR_ETFS:
        df = _fetch_one_ticker(t, cache_dir, start=start)
        if df is not None:
            macro_etf_cache[t] = df
    spy_hist         = macro_etf_cache.get("SPY")
    sector_etf_cache = {k: v for k, v in macro_etf_cache.items() if k in SECTOR_ETFS}
    sector_etf_cache.setdefault("SPY", spy_hist)  # fallback para tickers sem setor mapeado
    log.info(f"Sector ETFs carregados: {list(sector_etf_cache.keys())}")

    log.info(f"A descarregar preços de {len(tickers)} tickers...")
    price_cache = _fetch_batch(tickers, cache_dir, start=start)
    log.info(f"Preços carregados: {len(price_cache)}/{len(tickers)} tickers")

    # ── Build dataset ─────────────────────────────────────────────────────────
    log.info(f"A detectar episódios MA200w (entry≤{_MA200W_ENTRY_RATIO:.0%}, exit>{_MA200W_EXIT_RATIO:.0%})...")
    new_df = build_core_dataset(
        tickers=list(price_cache.keys()),
        price_cache=price_cache,
        spy_hist=spy_hist,
        sector_etf_cache=sector_etf_cache,
        core_tickers=core_tickers,
        core_yields=core_yields,
        add_fundamentals=not args.no_fundamentals,
    )

    if new_df.empty:
        if existing_df is not None and not existing_df.empty:
            log.info("Sem novos episódios — dataset existente mantido")
            return 0
        log.error("Dataset vazio — nenhum episódio MA200w encontrado")
        return 1

    # ── Merge com dados existentes ────────────────────────────────────────────
    if existing_df is not None and not existing_df.empty:
        new_df["source"] = "backfill"
        if "source" not in existing_df.columns:
            existing_df["source"] = "real"
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined["_sort"] = (combined.get("source", "backfill") == "real").astype(int)
        combined = combined.sort_values("_sort", ascending=False).drop(columns=["_sort"])
        combined = combined.drop_duplicates(subset=["ticker", "alert_date"], keep="first")
        combined = combined.sort_values("alert_date").reset_index(drop=True)
        n_new = len(combined) - len(existing_df)
        log.info(f"Merge: {len(existing_df)} existentes + {n_new} novos → {len(combined)} total")
    else:
        combined = new_df.sort_values("alert_date").reset_index(drop=True)
        log.info(f"Dataset novo: {len(combined)} episódios")

    # ── Guardar ───────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)

    size_mb = out_path.stat().st_size / 1e6
    core_rows = int(combined.get("is_core_company", pd.Series([0])).sum()) if "is_core_company" in combined.columns else 0
    log.info(
        f"Guardado: {out_path} ({size_mb:.1f}MB) | "
        f"{len(combined)} rows | {combined['ticker'].nunique()} tickers | "
        f"{core_rows} rows de core companies"
    )
    for label in ("alpha_1y", "alpha_2y"):
        if label in combined.columns:
            v = combined[label].dropna()
            if len(v) > 0:
                log.info(f"{label}: mean={v.mean():.3f} std={v.std():.3f} ({len(v)} amostras)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
