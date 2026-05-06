"""Construção do dataset v3.2 — extraído do notebook (cells 6, 13, 14)."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml_training.config import DEFAULT_ETF, HORIZON_DAYS, SECTOR_ETF

log = logging.getLogger(__name__)

# Tickers macro necessários para get_macro_context_historical.
# Descarrega uma vez no início do Colab e passa como macro_price_cache.
MACRO_TICKERS: list[str] = ["^VIX", "SPY", "^TNX", "^IRX", "HYG", "LQD", "IYT", "XLI"]


# ─────────────────────────────────────────────────────────────────────────────
# Carregar dataset base (cell 6)
# ─────────────────────────────────────────────────────────────────────────────

def load_base_dataset(parquet_path: Path) -> pd.DataFrame:
    """Carrega o parquet base (v1/v2) e normaliza colunas.

    Compatibilidade:
      - aceita ``symbol`` ou ``ticker`` (renomeia ``symbol`` → ``ticker``)
      - parsing de ``alert_date`` para datetime
      - sort por ``alert_date``
    """
    if not Path(parquet_path).exists():
        raise FileNotFoundError(f"parquet base não encontrado: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if "alert_date" not in df.columns:
        raise KeyError(f"parquet sem coluna alert_date: {parquet_path}")

    df["alert_date"] = pd.to_datetime(df["alert_date"])
    df = df.sort_values("alert_date").reset_index(drop=True)

    if "symbol" in df.columns and "ticker" not in df.columns:
        df = df.rename(columns={"symbol": "ticker"})

    if "ticker" not in df.columns:
        raise KeyError(f"parquet sem coluna ticker/symbol: {parquet_path}")

    log.info(
        f"[data] {parquet_path.name}: shape={df.shape} | "
        f"período {df['alert_date'].min().date()} → {df['alert_date'].max().date()} | "
        f"tickers={df['ticker'].nunique()}"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# sector_alert_count_7d (cell 13) — anti-leakage rolling
# ─────────────────────────────────────────────────────────────────────────────

def compute_sector_alert_count_7d(
    df: pd.DataFrame,
) -> dict[tuple[str, pd.Timestamp], int]:
    """Para cada alerta, conta quantos no mesmo sector ocorreram nos 7d ANTES
    (não inclui o próprio alerta — anti-leakage estrito).

    Devolve ``{(ticker, alert_date): count}``.
    """
    if df.empty:
        return {}

    seq = df[["alert_date", "sector", "ticker"]].copy()
    seq["alert_date"] = pd.to_datetime(seq["alert_date"])
    seq = seq.sort_values(["sector", "alert_date"]).reset_index(drop=True)
    seq["sector_alert_count_7d"] = 0

    for _sec, sub in seq.groupby("sector", sort=False):
        dates_arr = sub["alert_date"].to_numpy()  # np.datetime64[ns]
        counts = np.zeros(len(sub), dtype=np.int32)
        for i in range(len(sub)):
            win_start = dates_arr[i] - np.timedelta64(7, "D")
            prior = dates_arr[:i]
            counts[i] = int(((prior >= win_start) & (prior < dates_arr[i])).sum())
        seq.loc[sub.index, "sector_alert_count_7d"] = counts

    lookup: dict[tuple[str, pd.Timestamp], int] = {}
    for _, r in seq.iterrows():
        key = (r["ticker"], pd.Timestamp(r["alert_date"]))
        lookup[key] = int(r["sector_alert_count_7d"])
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# Forward-only helpers (cell 14)
# ─────────────────────────────────────────────────────────────────────────────

def spy_max_return_forward(
    spy_hist: Optional[pd.DataFrame],
    alert_date: pd.Timestamp,
    horizon: int = HORIZON_DAYS,
) -> float:
    """SPY max return em (alert_date, alert_date+horizon] — forward-only.

    Devolve NaN se < 5 candles forward ou entry inválida.
    """
    if spy_hist is None:
        return float("nan")
    entry_slice = spy_hist[spy_hist.index <= alert_date]
    if len(entry_slice) == 0:
        return float("nan")
    spy_entry = float(entry_slice["Close"].iloc[-1])
    if spy_entry <= 0:
        return float("nan")
    fwd = spy_hist[
        (spy_hist.index > alert_date)
        & (spy_hist.index <= alert_date + pd.Timedelta(days=horizon))
    ]
    if len(fwd) < 5:
        return float("nan")
    return float(fwd["Close"].max() / spy_entry - 1.0)


def days_since_52w_high(hist: pd.DataFrame, alert_date: pd.Timestamp) -> float:
    """Quantos dias desde o pico de 52 semanas até alert_date.

    Fallback (60.0) se janela < 20 candles. Replica o helper da cell 14.
    """
    from ml_features import _FALLBACK
    window = hist[
        (hist.index <= alert_date)
        & (hist.index > alert_date - pd.Timedelta(days=365))
    ]
    if len(window) < 20:
        return float(_FALLBACK.get("days_since_52w_high", 60.0))
    high_idx = window["High"].idxmax()
    return float((alert_date - high_idx).days)


# ─────────────────────────────────────────────────────────────────────────────
# Build dataset v3.2 (cell 14)
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset_v31(
    base_df: pd.DataFrame,
    price_cache: dict[str, pd.DataFrame],
    etf_cache: dict[str, pd.DataFrame],
    feature_cols_v31: list[str],
    horizon_days: int = HORIZON_DAYS,
    macro_price_cache: Optional[dict[str, pd.DataFrame]] = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Constrói dataset v3.2 linha-a-linha (replica cell 14 do notebook).

    Para cada alerta em ``base_df``:
      1. Computa macro point-in-time via get_macro_context_historical()
         (Stage 0: macro_score, vix, spy_drawdown_5d, sector_drawdown_5d)
      2. Carrega features v1/v2 do parquet + fallback _FALLBACK
         para features de qualidade (Stage 1: gross_margin, de_ratio, etc.)
      3. Adiciona derived features (add_derived_features)
      4. Adiciona v2 features price-based (build_v2_features)
      5. Adiciona momentum features (add_momentum_features)
      6. Adiciona context features v3.2 (add_context_features)
      7. Calcula targets max_return_60d, max_drawdown_60d,
         spy_max_return_60d, alpha_60d

    Parameters
    ----------
    base_df            : DataFrame   Alertas subsampled (output da Célula 3)
    price_cache        : dict        {ticker: OHLCV DataFrame}
    etf_cache          : dict        {etf_ticker: OHLCV DataFrame} — inclui SPY e sector ETFs
    feature_cols_v31   : list[str]   FEATURE_COLUMNS de ml_features.py (29 features)
    horizon_days       : int         Janela forward para targets (default: HORIZON_DAYS)
    macro_price_cache  : dict|None   {macro_ticker: OHLCV DataFrame} para macro histórico.
                                     Tickers necessários: MACRO_TICKERS (^VIX, SPY, ^TNX,
                                     ^IRX, HYG, LQD, IYT, XLI) + sector ETFs.
                                     Se None, as features macro ficam com fallback.

    Devolve (df_v32, skipped_dict).
    """
    from ml_features import (
        FEATURE_COLUMNS,
        _FALLBACK,
        add_derived_features,
        add_momentum_features,
        add_context_features,
    )
    from macro_data import get_macro_context_historical
    from experiments.ml_v2.pipeline import build_targets, build_v2_features

    sector_count_lookup = compute_sector_alert_count_7d(base_df)
    spy_hist = etf_cache.get(DEFAULT_ETF)

    # Merge macro_price_cache com etf_cache para ter sector ETFs disponíveis
    # no get_macro_context_historical sem duplicar memória.
    combined_macro_cache: dict[str, pd.DataFrame] = {}
    if macro_price_cache:
        combined_macro_cache.update(macro_price_cache)
    combined_macro_cache.update(etf_cache)  # sector ETFs sobrepõem se duplicados

    rows_v31: list[dict] = []
    skipped = {"no_price": 0, "short_history": 0, "no_target": 0, "no_spy_target": 0}

    for _, row in base_df.iterrows():
        ticker = row["ticker"]
        alert_date = pd.Timestamp(row["alert_date"])
        sector = row.get("sector", "Unknown") or "Unknown"
        etf = SECTOR_ETF.get(sector, DEFAULT_ETF)

        ohlcv = price_cache.get(ticker)
        if ohlcv is None:
            skipped["no_price"] += 1
            continue

        hist = ohlcv[ohlcv.index <= alert_date]
        if len(hist) < 25:
            skipped["short_history"] += 1
            continue

        # ── Stage 0: Macro point-in-time ─────────────────────────────────────
        macro_ctx = get_macro_context_historical(
            as_of_date=alert_date,
            sector=sector,
            macro_price_cache=combined_macro_cache if combined_macro_cache else None,
        )

        # ── Features: parquet → fv (com fallback para tudo) ──────────────────
        fv: dict[str, float] = {}
        for c in FEATURE_COLUMNS:
            v = row.get(c) if c in row.index else None
            fv[c] = float(v) if (v is not None and pd.notna(v)) else _FALLBACK.get(c, 0.0)

        # Sobrepõe Stage 0 com valores macro point-in-time reais
        fv["macro_score"]        = float(macro_ctx["macro_score"])
        fv["vix"]                = float(macro_ctx["vix"])
        fv["spy_drawdown_5d"]    = float(macro_ctx["spy_drawdown_5d"])
        fv["sector_drawdown_5d"] = float(macro_ctx["sector_drawdown_5d"])

        # ── Stage 1: Quality — parquet tem os valores históricos se disponíveis
        # (gross_margin, de_ratio, pe_vs_fair, analyst_upside, quality_score)
        # O loop já os leu do parquet acima via _FALLBACK se em falta.
        # Nada a fazer aqui — os fundamentals são point-in-time no parquet.

        add_derived_features(fv)

        # ── Features v2 extras (price-based) ─────────────────────────────────
        fv.update(build_v2_features(row, hist))

        # ── Stage 3b: Momentum ────────────────────────────────────────────────
        sec_hist = etf_cache.get(etf)
        sec_slice = sec_hist[sec_hist.index <= alert_date] if sec_hist is not None else None
        spy_slice = spy_hist[spy_hist.index <= alert_date] if spy_hist is not None else None
        add_momentum_features(fv, hist, sec_slice, spy_slice)

        # ── Stage 3d: Context features v3.2 ──────────────────────────────────
        add_context_features(
            fv,
            price_history=hist,
            sector_alert_count_7d=float(
                sector_count_lookup.get((ticker, alert_date), 0)
            ),
        )

        # ── Targets ───────────────────────────────────────────────────────────
        if "max_return_60d" in row.index and pd.notna(row.get("max_return_60d")):
            max_ret = float(row["max_return_60d"])
            max_draw = float(row.get("max_drawdown_60d", 0.0))
        else:
            entry_price = float(row.get("price", 0.0))
            if entry_price <= 0:
                skipped["no_target"] += 1
                continue
            future_close = ohlcv[
                (ohlcv.index > alert_date)
                & (ohlcv.index <= alert_date + pd.Timedelta(days=horizon_days))
            ]["Close"]
            if len(future_close) < 5:
                skipped["no_target"] += 1
                continue
            tgt = build_targets(alert_date, entry_price, future_close)
            if math.isnan(tgt["max_return_60d"]):
                skipped["no_target"] += 1
                continue
            max_ret = tgt["max_return_60d"]
            max_draw = tgt["max_drawdown_60d"]

        spy_max_ret = spy_max_return_forward(spy_hist, alert_date, horizon_days)
        if math.isnan(spy_max_ret):
            skipped["no_spy_target"] += 1
            continue

        alpha_60d = max_ret - spy_max_ret

        rec = {
            "ticker":     ticker,
            "alert_date": alert_date,
            "sector":     sector,
            **{c: fv[c] for c in feature_cols_v31 if c in fv},
            "max_return_60d":     max_ret,
            "max_drawdown_60d":   max_draw,
            "spy_max_return_60d": spy_max_ret,
            "alpha_60d":          alpha_60d,
        }
        rows_v31.append(rec)

    df_v31 = pd.DataFrame(rows_v31)
    log.info(f"[data] dataset v3.2: shape={df_v31.shape} | skipped={skipped}")
    return df_v31, skipped
