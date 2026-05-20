"""regenerate_training_base.py — Reconstrói ml_training_base.parquet com dados correctos.

PROBLEMA QUE ESTE SCRIPT RESOLVE:
  O parquet original tinha look-ahead bias nos fundamentais (usava dados de
  hoje para avaliar dips de 2020/2022) e não tinha alpha_90d (target de 90 dias).
  Este script corrige ambos os problemas.

O que é adicionado/corrigido:
  FUNDAMENTAIS POINT-IN-TIME (novas/corrigidas):
    gross_margin, de_ratio, fcf_yield, revenue_growth, quality_score
    → via SEC EDGAR XBRL (US tickers) ou yfinance quarterly (não-US)
    → se TIINGO_API_KEY com Starter: Tiingo é usado (melhor qualidade)
    Sem look-ahead: usamos o filing mais recente ANTES de alert_date.

  TARGETS (novos):
    alpha_90d = log1p(close_90d/price) - log1p(spy_close_90d/spy_price)
    → target principal para o modelo (retorno em excesso sobre SPY em 90 dias)

  FEATURES TÉCNICAS (já existiam, mantidas):
    return_6m_pre, vol_of_vol, bb_width, vix_percentile_1y, spy_rsi_14,
    volume_zscore_20d, close_in_range_20d, up_days_pct_20d, true_range_pct_20d

Uso:
    python scripts/regenerate_training_base.py [--in PATH] [--out PATH] [--cache DIR]
    python scripts/regenerate_training_base.py --fundamentals-only  # só fundamentais
    python scripts/regenerate_training_base.py --targets-only       # só alpha_90d

Duração estimada:
  - Primeira execução: 45-90 min (download de preços + EDGAR para 700+ tickers)
  - Re-runs: 5-15 min (tudo em cache)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Garante que o package raiz do repo está no sys.path para importar
# `ml_features` (que está na raiz, não dentro de scripts/).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# yfinance é importado dentro de funções para permitir testes sem rede

log = logging.getLogger("regen")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_DATA_VOL = Path("/data") if Path("/data").exists() else Path("/tmp")

DEFAULT_IN  = _DATA_VOL / "ml_training_base.parquet"
DEFAULT_OUT = _DATA_VOL / "ml_training_base.parquet"
# Cache em /data/ (Railway Volume persistido) para não re-descarregar em cada restart.
# /tmp seria apagado em cada container restart, obrigando a re-download completo.
DEFAULT_CACHE_DIR = _DATA_VOL / "price_cache"

MACRO_TICKERS: list[str] = ["SPY", "^VIX", "^TNX"]
SECTOR_ETFS: list[str] = [
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK",
    "XLP", "XLRE", "XLU", "XLV", "XLY",
]

SECTOR_ETF: dict[str, str] = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
    "Unknown": "SPY",
}

NEW_FEATURE_COLS: list[str] = [
    "return_6m_pre",
    "return_12m_pre",
    "sector_relative_6m",
    "vol_of_vol",
    "bb_width",
    "vix_percentile_1y",
    "spy_rsi_14",
    "yield_10y_change_5d",
    "volume_zscore_20d",
    "close_in_range_20d",
    "up_days_pct_20d",
    "true_range_pct_20d",
]

# Colunas fundamentais corrigidas (PIT) — substituem valores constantes do bootstrap
FUNDAMENTAL_COLS: list[str] = [
    "gross_margin",
    "de_ratio",
    "fcf_yield",
    "revenue_growth",
    "quality_score",
]

# Targets novos que precisam de ser adicionados ao parquet
TARGET_COLS: list[str] = [
    "alpha_90d",
    "close_90d",
    "spy_close_90d",
]

# Features Fase 5 que são computadas point-in-time a partir de SEC EDGAR + preços.
# Nota: short_interest_trend, short_interest_pct, analyst_rating, earnings_beat_rate
# dependem de dados yfinance que NÃO são PIT → ficam em fallback no treino.
# Os EDGAR-based são totalmente PIT (filed_date <= alert_date).
FASE5_COLS: list[str] = [
    "consecutive_red_days",    # PIT via price_cache
    "ma_200d_ratio",           # PIT via price_cache
    "insider_buy_recent",      # PIT via SEC Form 4 (EDGAR)
    "insider_buy_amount_score",# PIT via SEC Form 4 (EDGAR)
    "recent_8k_score",         # PIT via SEC 8-K (EDGAR)
    "earnings_call_tone",      # PIT via SEC 8-K texto (EDGAR)
]


# ─────────────────────────────────────────────────────────────────────────────
# yfinance helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_one_ticker(
    ticker: str,
    cache_dir: Path,
    start: str = "1995-01-01",
    end: Optional[str] = None,
    force_refresh: bool = False,
) -> Optional[pd.DataFrame]:
    """Download price history de um ticker, com cache local em parquet."""
    import yfinance as yf

    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = ticker.replace("^", "_idx_").replace("/", "_")
    cache_file = cache_dir / f"{safe_name}.parquet"

    if not force_refresh and cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            log.warning(f"  cache corrupto {ticker} ({e}) — re-download")

    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    for attempt in range(3):
        try:
            df = yf.Ticker(ticker).history(
                start=start,
                end=end,
                auto_adjust=True,
            )
            if df is not None and not df.empty:
                # tz_convert(None) remove timezone de índices UTC-aware (yfinance >= 0.2)
                idx = pd.DatetimeIndex(df.index)
                df.index = idx.tz_convert(None) if idx.tz is not None else idx
                df = df[["Open", "High", "Low", "Close", "Volume"]]
                df.to_parquet(cache_file)
                return df
            return None
        except Exception as e:
            if attempt == 2:
                log.warning(f"  {ticker}: yfinance falhou ({e})")
                return None
            time.sleep(0.5 * (attempt + 1))
    return None


def _fetch_batch(
    tickers: list[str],
    cache_dir: Path,
    start: str = "1995-01-01",
    batch_size: int = 50,
) -> dict[str, pd.DataFrame]:
    """Fetch ticker histories with batch logging and cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, pd.DataFrame] = {}

    n = len(tickers)
    t0 = time.time()
    fetched = 0
    cached = 0
    failed = 0

    for i, ticker in enumerate(tickers):
        was_cached = (cache_dir / f"{ticker.replace('^','_idx_').replace('/','_')}.parquet").exists()
        df = _fetch_one_ticker(ticker, cache_dir, start=start)
        if df is None or df.empty:
            failed += 1
            continue
        result[ticker] = df
        if was_cached:
            cached += 1
        else:
            fetched += 1

        if (i + 1) % batch_size == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n - i - 1) / rate if rate > 0 else 0
            log.info(
                f"  [{i+1:>4}/{n}] fetched={fetched} cached={cached} "
                f"failed={failed} ({elapsed:.0f}s elapsed, ~{eta:.0f}s ETA)"
            )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation
# ─────────────────────────────────────────────────────────────────────────────

def _slice_history(
    hist: Optional[pd.DataFrame],
    alert_date: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """Slice price history to dates <= alert_date."""
    if hist is None or hist.empty:
        return None
    out = hist[hist.index <= alert_date]
    return out if len(out) >= 5 else None


def _compute_row_features(
    row: pd.Series,
    price_cache: dict[str, pd.DataFrame],
    sector_etf_cache: dict[str, pd.DataFrame],
    macro_cache: dict[str, pd.DataFrame],
) -> dict[str, float]:
    """Compute the new features (8 momentum/regime + 4 raw-OHLCV)."""
    from ml_features import (
        _FALLBACK,
        add_momentum_features,
        add_raw_ohlcv_features,
        add_regime_features,
    )

    ticker = str(row["ticker"])
    alert_date = pd.Timestamp(row["alert_date"]).tz_localize(None) if pd.Timestamp(row["alert_date"]).tz is not None else pd.Timestamp(row["alert_date"])
    sector = str(row.get("sector", "Unknown") or "Unknown")
    etf = SECTOR_ETF.get(sector, "SPY")

    hist = _slice_history(price_cache.get(ticker), alert_date)
    spy_slice = _slice_history(macro_cache.get("SPY"), alert_date)
    vix_slice = _slice_history(macro_cache.get("^VIX"), alert_date)
    tnx_slice = _slice_history(macro_cache.get("^TNX"), alert_date)
    sec_slice = _slice_history(sector_etf_cache.get(etf), alert_date)

    fv: dict[str, float] = {}

    # Momentum + bb_width + vol_of_vol
    if hist is not None:
        try:
            add_momentum_features(fv, hist, sec_slice, spy_slice)
        except Exception as e:
            log.debug(f"  momentum failed {ticker}@{alert_date.date()}: {e}")
    # Fallbacks para keys ausentes
    for k in ("return_6m_pre", "return_12m_pre", "sector_relative_6m", "vol_of_vol", "bb_width"):
        fv.setdefault(k, _FALLBACK.get(k, 0.0))

    # Regime
    try:
        add_regime_features(fv, spy_slice, tnx_slice, alert_date, vix_slice)
    except Exception as e:
        log.debug(f"  regime failed {ticker}@{alert_date.date()}: {e}")
    for k in ("vix_percentile_1y", "spy_rsi_14", "yield_10y_change_5d"):
        fv.setdefault(k, _FALLBACK.get(k, 0.0))

    # Raw OHLCV (PR #30) — usa o mesmo slice já fetchado, sem custo extra
    if hist is not None:
        try:
            add_raw_ohlcv_features(fv, hist)
        except Exception as e:
            log.debug(f"  raw_ohlcv failed {ticker}@{alert_date.date()}: {e}")
    for k in ("volume_zscore_20d", "close_in_range_20d", "up_days_pct_20d", "true_range_pct_20d"):
        fv.setdefault(k, _FALLBACK.get(k, 0.0))

    # Devolver só as colunas de interesse
    return {k: float(fv.get(k, _FALLBACK.get(k, 0.0))) for k in NEW_FEATURE_COLS}


def _compute_pit_fundamentals_row(
    row: pd.Series,
    cache_dir: Path,
) -> dict:
    """Busca fundamentais PIT para uma linha do parquet.

    Usa fundamental_history.get_pit_fundamentals() que tenta Tiingo →
    SEC EDGAR → yfinance quarterly, por esta ordem.
    """
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from fundamental_history import get_pit_fundamentals
    from ml_features import _FALLBACK

    ticker     = str(row["ticker"])
    alert_date = pd.Timestamp(row["alert_date"]).date()
    price      = float(row.get("price", 0) or 0)

    fund = get_pit_fundamentals(ticker, alert_date, cache_dir)

    result = {col: _FALLBACK.get(col, 0.0) for col in FUNDAMENTAL_COLS}

    if fund:
        for col in ("gross_margin", "de_ratio", "revenue_growth", "quality_score"):
            if col in fund and fund[col] is not None:
                result[col] = float(fund[col])

        # fcf_yield precisa do preço de mercado (fcf / market_cap)
        fcf_raw = fund.get("_fcf_raw") or fund.get("fcf_raw")
        if fcf_raw is not None and price > 0:
            # Estimativa grosseira: market_cap = preço × shares (não temos shares)
            # Usamos o fcf_yield directamente se disponível do Tiingo
            if "fcf_yield" in fund and fund["fcf_yield"] is not None:
                result["fcf_yield"] = float(fund["fcf_yield"])
            # EDGAR/yfinance: não temos shares outstanding, guardamos fcf_raw
            # para diagnóstico mas não calculamos yield sem market cap real

    return result


def _compute_alpha_targets_row(
    row: pd.Series,
    price_cache: dict[str, pd.DataFrame],
    spy_hist: Optional[pd.DataFrame],
    horizon_days: int = 90,
) -> dict:
    """Computa alpha_90d para uma linha do parquet.

    alpha_90d = log1p(stock_return_90d) - log1p(spy_return_90d)
    onde os retornos são calculados a partir do preço de entrada até T+90d.
    """
    import math

    result = {
        "alpha_90d":    float("nan"),
        "close_90d":    float("nan"),
        "spy_close_90d": float("nan"),
    }

    ticker     = str(row["ticker"])
    _ts        = pd.Timestamp(row["alert_date"])
    alert_date = _ts.tz_convert(None) if _ts.tz is not None else _ts

    hist = price_cache.get(ticker)
    if hist is None or hist.empty:
        return result

    # Entry price: closing price on or just before alert_date.
    # O parquet de treino não tem coluna "price" — calculamos directamente
    # da price history. Mais correcto (PIT) do que confiar no campo do parquet.
    entry_slice = hist[hist.index <= alert_date]
    if entry_slice.empty:
        return result
    price = float(entry_slice["Close"].iloc[-1])
    if price <= 0 or not math.isfinite(price):
        return result

    exit_date = alert_date + pd.Timedelta(days=horizon_days)

    # Preço do stock em T+90d (fecha mais próximo dentro da janela)
    fwd = hist[(hist.index > alert_date) & (hist.index <= exit_date)]
    if len(fwd) < 5:   # mínimo de 5 dias úteis na janela
        return result

    close_90d = float(fwd["Close"].iloc[-1])
    stock_ret = close_90d / price - 1.0
    if not math.isfinite(stock_ret) or abs(stock_ret) > 2.0:   # cap a ±200%
        return result

    # SPY no mesmo período
    if spy_hist is not None and not spy_hist.empty:
        spy_entry_slice = spy_hist[spy_hist.index <= alert_date]
        spy_exit_slice  = spy_hist[(spy_hist.index > alert_date) & (spy_hist.index <= exit_date)]
        if not spy_entry_slice.empty and not spy_exit_slice.empty:
            spy_price = float(spy_entry_slice["Close"].iloc[-1])
            spy_close = float(spy_exit_slice["Close"].iloc[-1])
            if spy_price > 0:
                spy_ret = spy_close / spy_price - 1.0
                if math.isfinite(spy_ret) and abs(spy_ret) <= 1.0:
                    result["alpha_90d"]     = round(math.log1p(stock_ret) - math.log1p(spy_ret), 6)
                    result["close_90d"]     = round(stock_ret, 6)
                    result["spy_close_90d"] = round(spy_ret, 6)
                    return result

    # Sem SPY: guarda só o retorno absoluto (alpha vs SPY não disponível)
    if math.isfinite(stock_ret):
        result["close_90d"] = round(stock_ret, 6)
    return result


def _compute_fase5_row(
    row: pd.Series,
    price_cache: dict[str, pd.DataFrame],
) -> dict[str, float]:
    """Computa sinais Fase 5 point-in-time para uma linha do parquet.

    Usa apenas EDGAR (Form 4 e 8-K) e preços — totalmente PIT.
    yfinance-live signals (short_interest, analyst_rating) não são computados
    aqui porque não são PIT para dados históricos.
    """
    from ml_features import _FALLBACK

    ticker     = str(row["ticker"])
    alert_date = pd.Timestamp(row["alert_date"])
    if alert_date.tz is not None:
        alert_date = alert_date.tz_localize(None)
    alert_dt   = alert_date.date()

    hist = _slice_history(price_cache.get(ticker), alert_date)
    fv: dict[str, float] = {k: _FALLBACK.get(k, 0.0) for k in FASE5_COLS}

    # Price-based: PIT via slice já existente
    if hist is not None and len(hist) >= 2:
        try:
            closes = hist["Close"]
            # consecutive_red_days: dias consecutivos em queda até alert_date
            rets = closes.pct_change().dropna()
            if len(rets) > 0:
                red = 0
                for r in reversed(rets.values.tolist()):
                    if r < 0:
                        red += 1
                    else:
                        break
                fv["consecutive_red_days"] = float(min(red, 30))
            # ma_200d_ratio: price / MA200 (quão abaixo da média longa)
            if len(closes) >= 20:
                ma200 = closes.rolling(200, min_periods=20).mean().iloc[-1]
                last  = float(closes.iloc[-1])
                if ma200 > 0:
                    fv["ma_200d_ratio"] = round(last / float(ma200), 4)
        except Exception as e:
            log.debug(f"  fase5 price {ticker}@{alert_dt}: {e}")

    # EDGAR-based: PIT via filed_date <= alert_date (caching por ticker)
    try:
        from fundamental_signals import (
            insider_buy_recent,
            insider_buy_amount_score,
            classify_recent_8k,
            earnings_call_tone as _earnings_tone,
        )
        # Form 4 insider buying
        try:
            fv["insider_buy_recent"] = float(insider_buy_recent(ticker, lookback_days=30, as_of_date=alert_dt) or 0.0)
        except Exception:
            pass
        try:
            score, _ = insider_buy_amount_score(ticker, lookback_days=30, as_of_date=alert_dt)
            if score is not None:
                fv["insider_buy_amount_score"] = float(score)
        except Exception:
            pass
        # 8-K classification
        try:
            score_8k = classify_recent_8k(ticker, lookback_days=60, as_of_date=alert_dt)
            if score_8k is not None:
                fv["recent_8k_score"] = float(score_8k)
        except Exception:
            pass
        # Earnings call tone
        try:
            tone = _earnings_tone(ticker, lookback_days=90, as_of_date=alert_dt)
            if tone is not None:
                fv["earnings_call_tone"] = float(tone)
        except Exception:
            pass
    except ImportError as e:
        log.debug(f"  fase5 EDGAR {ticker}: fundamental_signals não disponível ({e})")

    return {k: fv.get(k, _FALLBACK.get(k, 0.0)) for k in FASE5_COLS}


def regenerate(
    in_path: Path,
    out_path: Path,
    cache_dir: Path,
    start_date: str = "1995-01-01",
    add_fundamentals: bool = True,
    add_targets: bool = True,
    add_features: bool = True,
    add_fase5: bool = True,
) -> None:
    """Pipeline principal de regeneração."""
    if not in_path.exists():
        raise FileNotFoundError(f"input parquet não encontrado: {in_path}")

    log.info(f"[in] {in_path} ({in_path.stat().st_size/1e6:.1f} MB)")
    df = pd.read_parquet(in_path)
    df["alert_date"] = pd.to_datetime(df["alert_date"]).dt.tz_localize(None)
    log.info(f"[in] shape={df.shape}  unique tickers={df['ticker'].nunique()}")
    log.info(f"[in] date range: {df['alert_date'].min()} → {df['alert_date'].max()}")

    # ── Fetch macro + ETFs ──────────────────────────────────────────────────
    log.info("[fetch] macro + sector ETFs...")
    macro_etf_cache = _fetch_batch(MACRO_TICKERS + SECTOR_ETFS, cache_dir, start=start_date)
    macro_cache      = {k: v for k, v in macro_etf_cache.items() if k in MACRO_TICKERS}
    sector_etf_cache = {k: v for k, v in macro_etf_cache.items() if k in SECTOR_ETFS}
    if "SPY" in macro_cache:
        sector_etf_cache.setdefault("SPY", macro_cache["SPY"])
    spy_hist = macro_cache.get("SPY")
    log.info(f"[fetch] macro={len(macro_cache)} sector_etfs={len(sector_etf_cache)}")

    # ── Fetch ticker price histories ──────────────────────────────────────────
    unique_tickers = sorted(df["ticker"].astype(str).unique().tolist())
    log.info(f"[fetch] {len(unique_tickers)} ticker histories...")
    price_cache = _fetch_batch(unique_tickers, cache_dir, start=start_date)
    log.info(f"[fetch] {len(price_cache)}/{len(unique_tickers)} tickers OK")
    missing_tickers = [t for t in unique_tickers if t not in price_cache]
    if missing_tickers:
        log.warning(f"[fetch] {len(missing_tickers)} tickers sem history: {missing_tickers[:10]}...")

    t0 = time.time()

    # ── 1. Features técnicas ─────────────────────────────────────────────────
    if add_features:
        log.info("[compute] features técnicas por linha...")
        feat_results = []
        for i, (_, row) in enumerate(df.iterrows()):
            feat_results.append(_compute_row_features(row, price_cache, sector_etf_cache, macro_cache))
            if (i + 1) % 2000 == 0 or (i + 1) == len(df):
                log.info(f"  [features] [{i+1:>5}/{len(df)}] {time.time()-t0:.0f}s")
        feat_df = pd.DataFrame(feat_results, index=df.index)
        for col in NEW_FEATURE_COLS:
            df[col] = feat_df[col].values
        log.info(f"[compute] features técnicas: OK ({len(NEW_FEATURE_COLS)} colunas)")

    # ── 2. Fundamentais PIT ───────────────────────────────────────────────────
    if add_fundamentals:
        log.info("[compute] fundamentais point-in-time...")
        log.info("  Camadas: Tiingo → SEC EDGAR XBRL → yfinance quarterly")
        fund_results = []
        n_has_data = 0
        t1 = time.time()
        for i, (_, row) in enumerate(df.iterrows()):
            r = _compute_pit_fundamentals_row(row, cache_dir)
            fund_results.append(r)
            if any(v != 0.0 for v in r.values()):
                n_has_data += 1
            if (i + 1) % 1000 == 0 or (i + 1) == len(df):
                log.info(
                    f"  [fund] [{i+1:>5}/{len(df)}] com_dados={n_has_data} "
                    f"({n_has_data/(i+1):.0%}) {time.time()-t1:.0f}s"
                )
        fund_df = pd.DataFrame(fund_results, index=df.index)
        for col in FUNDAMENTAL_COLS:
            df[col] = fund_df[col].values
        log.info(
            f"[compute] fundamentais PIT: {n_has_data}/{len(df)} linhas com dados "
            f"({n_has_data/len(df):.0%})"
        )

    # ── 3. Targets alpha_90d ──────────────────────────────────────────────────
    if add_targets:
        log.info("[compute] targets alpha_90d...")
        target_results = []
        n_resolved = 0
        for i, (_, row) in enumerate(df.iterrows()):
            r = _compute_alpha_targets_row(row, price_cache, spy_hist, horizon_days=90)
            target_results.append(r)
            import math
            if math.isfinite(r.get("alpha_90d", float("nan"))):
                n_resolved += 1
            if (i + 1) % 2000 == 0 or (i + 1) == len(df):
                log.info(
                    f"  [targets] [{i+1:>5}/{len(df)}] resolved={n_resolved} "
                    f"({n_resolved/(i+1):.0%}) {time.time()-t0:.0f}s"
                )
        target_df = pd.DataFrame(target_results, index=df.index)
        for col in TARGET_COLS:
            df[col] = target_df[col].values
        log.info(
            f"[compute] alpha_90d: {n_resolved}/{len(df)} linhas resolvidas "
            f"({n_resolved/len(df):.0%})"
        )

    # ── 4. Sinais Fase 5 (EDGAR + preços) — PIT correcto ─────────────────────
    if add_fase5 and add_features:
        log.info("[compute] sinais Fase 5 (EDGAR Form4/8-K + preços)...")
        log.info("  EDGAR calls com cache 30d — rápido em re-runs, ~2-4h primeira vez")
        fase5_results = []
        n_edgar = 0
        t2 = time.time()
        for i, (_, row) in enumerate(df.iterrows()):
            r = _compute_fase5_row(row, price_cache)
            fase5_results.append(r)
            if r.get("insider_buy_recent", 0.0) != 0.0 or r.get("recent_8k_score", 0.0) != 0.0:
                n_edgar += 1
            if (i + 1) % 2000 == 0 or (i + 1) == len(df):
                log.info(
                    f"  [fase5] [{i+1:>5}/{len(df)}] edgar_signal={n_edgar} "
                    f"({n_edgar/(i+1):.0%}) {time.time()-t2:.0f}s"
                )
        fase5_df = pd.DataFrame(fase5_results, index=df.index)
        for col in FASE5_COLS:
            df[col] = fase5_df[col].values
        log.info(
            f"[compute] Fase5: {n_edgar}/{len(df)} linhas com sinal EDGAR "
            f"({n_edgar/len(df):.0%})"
        )

    # ── Limpar colunas — só ficam metadados + features actuais + targets ─────
    # Remove colunas legadas (alpha_60d, features antigas com schema diferente)
    # para garantir que o output é limpo e não contém lixo de versões anteriores.
    try:
        from ml_training.config import FEATURE_COLS, PRIMARY_TARGET
        _META_COLS   = ["ticker", "symbol", "alert_date", "sector", "price",
                        "market_cap", "market_cap_b", "_source"]
        _TARGET_COLS = [PRIMARY_TARGET, "alpha_60d", "close_90d", "spy_close_90d",
                        "max_drawdown_60d", "max_drawdown_20d", "max_return_60d",
                        "return_3m", "return_6m", "label_win", "label_further_drop"]
        _keep = set(_META_COLS) | set(FEATURE_COLS) | set(_TARGET_COLS)
        _drop = [c for c in df.columns if c not in _keep]
        if _drop:
            df = df.drop(columns=_drop)
            log.info(f"[clean] {len(_drop)} colunas legadas removidas: {_drop[:5]}{'...' if len(_drop)>5 else ''}")
    except Exception as e:
        log.warning(f"[clean] Falha ao limpar colunas ({e}) — a guardar tudo")

    # ── Save ──────────────────────────────────────────────────────────────────
    log.info(f"[out] a escrever {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    elapsed = time.time() - t0
    log.info(f"[out] CONCLUÍDO em {elapsed:.0f}s. shape={df.shape}  size={out_path.stat().st_size/1e6:.1f} MB")

    # Sanity stats finais
    log.info("[stats] resumo das colunas novas/corrigidas:")
    for col in NEW_FEATURE_COLS + FUNDAMENTAL_COLS + TARGET_COLS + FASE5_COLS:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            log.info(f"  {col:28s} — SEM DADOS")
        else:
            log.info(
                f"  {col:28s} n={len(s):>6} mean={s.mean():>9.4f} "
                f"std={s.std():>8.4f} nan={df[col].isna().sum()}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Regenera o parquet de treino com PIT fundamentais + alpha_90d")
    p.add_argument("--in",  dest="in_path",  type=Path, default=DEFAULT_IN)
    p.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cache", dest="cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--start", dest="start_date", default="1995-01-01")
    p.add_argument("--fundamentals-only", action="store_true",
                   help="Só actualiza fundamentais PIT, mantém features e targets")
    p.add_argument("--targets-only", action="store_true",
                   help="Só adiciona/actualiza alpha_90d")
    p.add_argument("--no-fundamentals", action="store_true",
                   help="Pula a fase de fundamentais PIT (mais rápido)")
    p.add_argument("--no-fase5", action="store_true",
                   help="Pula sinais Fase 5 EDGAR/preços (mais rápido, usa fallbacks)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    add_fundamentals = not args.no_fundamentals
    add_features     = not args.fundamentals_only and not args.targets_only
    add_targets      = not args.fundamentals_only
    add_fase5        = not getattr(args, "no_fase5", False)

    if args.fundamentals_only:
        add_fundamentals = True
        add_features     = False
        add_targets      = False
        add_fase5        = False
    if args.targets_only:
        add_fundamentals = False
        add_features     = False
        add_targets      = True
        add_fase5        = False

    log.info(
        f"Modo: features={add_features} fundamentais={add_fundamentals} "
        f"targets={add_targets} fase5={add_fase5}"
    )

    regenerate(
        args.in_path, args.out_path, args.cache_dir, args.start_date,
        add_fundamentals=add_fundamentals,
        add_targets=add_targets,
        add_features=add_features,
        add_fase5=add_fase5,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
