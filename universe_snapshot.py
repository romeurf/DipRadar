"""
universe_snapshot.py — Ingestão diária de features para todos os tickers do universo ML.

Para cada ticker em `get_ml_universe()` (~780), grava 1 linha por dia em
`/data/universe_snapshot.parquet` com:

  - Identificação            : symbol, snapshot_date, price
  - Técnicos (do dia)        : drop_pct_today, drawdown_52w, rsi_14, atr_ratio,
                               volume_spike
  - Macro                    : macro_score, vix, spy_drawdown_5d, sector_drawdown_5d
  - Fundamentais (cache 7d)  : fcf_yield, revenue_growth, gross_margin, de_ratio,
                               pe_vs_fair, analyst_upside, quality_score
  - Derived (calculadas)     : rsi_oversold_strength, vix_regime, pe_attractive,
                               drop_x_drawdown, vol_x_drop

Os labels (`label_win`, `outcome_label`, `return_3m`, `return_6m`) **não** são
preenchidos aqui — são back-filled mais tarde pelo `monthly_retrain_v2.py`
(quando passam ≥6m após snapshot_date) ou pelo job sunday-outcomes.

Idempotência:
  Cada chamada lê o parquet existente, descobre quais (symbol, snapshot_date)
  já existem hoje, e só processa os que faltam. Permite retomar após falha.

Storage:
  /data/universe_snapshot.parquet  (Railway Volume montado)
  /tmp/universe_snapshot.parquet   (fallback local/dev)

Cache de fundamentais:
  /data/universe_fund_cache.parquet  — uma linha por símbolo com last_refresh.
  Refresh: 7 dias. yfinance é lento, e fundamentais só mudam em earnings.

Schedule (em main.py setup_schedule):
  CronTrigger(hour=22, minute=30, day_of_week="mon-fri", timezone=LISBON_TZ)
  (após fecho US a 21:00 ET / 22:00 UTC inverno / 21:00 UTC verão)

Run manual (para testes):
  python universe_snapshot.py --tickers AAPL,MSFT,NVDA --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths (Railway Volume primeiro, /tmp fallback)
# ─────────────────────────────────────────────────────────────────────────────

_DATA_DIR = Path("/data") if Path("/data").exists() else Path("/tmp")
SNAPSHOT_PATH = _DATA_DIR / "universe_snapshot.parquet"
FUND_CACHE_PATH = _DATA_DIR / "universe_fund_cache.parquet"

# Quanto tempo (dias) o cache de fundamentais é considerado fresh.
# Fundamentais mudam só em earnings (≈trimestral) → 7d é seguro.
FUND_CACHE_DAYS = 7

# Universo a processar — importação lazy (evita ciclo + custo de Wikipedia)
def _load_universe() -> list[str]:
    from universe import get_ml_universe
    return get_ml_universe()


# ─────────────────────────────────────────────────────────────────────────────
# Schema do parquet — alinhado com FEATURE_COLUMNS de ml_features
# (com colunas extra symbol/snapshot_date/price para identificação)
# ─────────────────────────────────────────────────────────────────────────────

SNAPSHOT_COLUMNS = [
    # Identificação
    "symbol",
    "snapshot_date",  # ISO YYYY-MM-DD
    "price",
    # Técnicos
    "drop_pct_today",
    "drawdown_52w",
    "rsi_14",
    "atr_ratio",
    "volume_spike",
    # Macro
    "macro_score",
    "vix",
    "spy_drawdown_5d",
    "sector_drawdown_5d",
    # Fundamentais
    "fcf_yield",
    "revenue_growth",
    "gross_margin",
    "de_ratio",
    "pe_vs_fair",
    "analyst_upside",
    "quality_score",
    # Derived (computadas via add_derived_features)
    "rsi_oversold_strength",
    "vix_regime",
    "pe_attractive",
    "drop_x_drawdown",
    "vol_x_drop",
    # Telemetria
    "data_source",        # "yfinance" | "stooq" | "tiingo"
    "fund_age_days",      # idade (dias) dos fundamentais usados (NaN se fallback)
    "ingest_ts",          # ISO datetime UTC
]


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers (idempotency + recovery)
# ─────────────────────────────────────────────────────────────────────────────

def _load_existing_keys(snapshot_date: str) -> set[str]:
    """Devolve set de symbols com snapshot já existente para snapshot_date."""
    if not SNAPSHOT_PATH.exists():
        return set()
    try:
        df = pd.read_parquet(SNAPSHOT_PATH, columns=["symbol", "snapshot_date"])
        return set(df.loc[df["snapshot_date"] == snapshot_date, "symbol"].astype(str).tolist())
    except Exception as e:
        log.warning(f"[snapshot] Falha a ler keys existentes: {e}")
        return set()


def _append_rows(rows: list[dict]) -> None:
    """Concat com parquet existente e regrava (atomic via temp + rename)."""
    if not rows:
        return
    new_df = pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)
    if SNAPSHOT_PATH.exists():
        try:
            existing = pd.read_parquet(SNAPSHOT_PATH)
            combined = pd.concat([existing, new_df], ignore_index=True)
            # Final dedup defensivo (caso interrupção causou re-processamento)
            combined = combined.drop_duplicates(
                subset=["symbol", "snapshot_date"], keep="last"
            )
        except Exception as e:
            log.error(f"[snapshot] Falha a ler parquet existente, a recriar: {e}")
            combined = new_df
    else:
        combined = new_df

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SNAPSHOT_PATH.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp, index=False)
    os.replace(tmp, SNAPSHOT_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Cache de fundamentais (por símbolo, refresh 7d)
# ─────────────────────────────────────────────────────────────────────────────

_FUND_FALLBACK = {
    "fcf_yield":      0.04,
    "revenue_growth": 0.05,
    "gross_margin":   0.35,
    "de_ratio":       80.0,
    "pe_vs_fair":     1.0,
    "analyst_upside": 0.10,
    "quality_score":  0.50,
}


def _load_fund_cache() -> pd.DataFrame:
    if not FUND_CACHE_PATH.exists():
        return pd.DataFrame(columns=["symbol", "last_refresh"] + list(_FUND_FALLBACK.keys()))
    try:
        return pd.read_parquet(FUND_CACHE_PATH)
    except Exception as e:
        log.warning(f"[snapshot] fund cache corrompido: {e} — a recriar.")
        return pd.DataFrame(columns=["symbol", "last_refresh"] + list(_FUND_FALLBACK.keys()))


def _save_fund_cache(df: pd.DataFrame) -> None:
    FUND_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = FUND_CACHE_PATH.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, FUND_CACHE_PATH)


def _fetch_live_fundamentals(symbol: str) -> dict:
    """
    Tenta yfinance.Ticker(...).info — fallback gracioso a defaults se falhar.

    Não é point-in-time histórico (isso seria via Camada B do bootstrap_ml);
    é a foto actual dos fundamentais para alimentar o snapshot.
    """
    out = dict(_FUND_FALLBACK)
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info or {}
    except Exception as e:
        log.debug(f"[fund] {symbol}: yfinance indisponível ({e})")
        return out

    def _get(key, default):
        v = info.get(key)
        if v is None:
            return default
        try:
            f = float(v)
            return default if (np.isnan(f) or np.isinf(f)) else f
        except (TypeError, ValueError):
            return default

    # Fontes preferidas, com fallbacks. Os defaults são os mesmos do bootstrap.
    free_cf = _get("freeCashflow", None)
    market_cap = _get("marketCap", None)
    if free_cf and market_cap and market_cap > 0:
        out["fcf_yield"] = max(-1.0, min(1.0, free_cf / market_cap))

    out["revenue_growth"] = _get("revenueGrowth", _FUND_FALLBACK["revenue_growth"])
    out["gross_margin"]   = _get("grossMargins",  _FUND_FALLBACK["gross_margin"])
    out["de_ratio"]       = _get("debtToEquity",  _FUND_FALLBACK["de_ratio"])

    pe   = _get("trailingPE", None)
    fair_pe = _get("forwardPE", pe)  # fallback simples
    if pe and fair_pe and fair_pe > 0:
        out["pe_vs_fair"] = max(0.1, min(5.0, pe / fair_pe))

    target = _get("targetMeanPrice", None)
    cur    = _get("currentPrice", None)
    if target and cur and cur > 0:
        out["analyst_upside"] = max(-0.9, min(2.0, (target - cur) / cur))

    # quality_score: composto simples (gross_margin + revenue_growth + 1 - debt_proxy)
    gm  = out["gross_margin"]
    rg  = out["revenue_growth"]
    de  = out["de_ratio"]
    de_norm = max(0.0, min(1.0, 1.0 - de / 200.0))
    raw = (max(0.0, gm) + max(0.0, rg) + de_norm) / 3.0
    out["quality_score"] = round(max(0.0, min(1.0, raw)), 3)

    return out


def _get_fundamentals(symbol: str, today: date,
                      fund_cache: pd.DataFrame) -> tuple[dict, float, pd.DataFrame]:
    """
    Devolve (fund_dict, age_days, updated_cache_df).

    Se cache fresh (< FUND_CACHE_DAYS) → usa cache.
    Caso contrário → fetch live + actualiza cache.
    age_days=NaN se fundamentais são fallback (nunca conseguimos buscar).
    """
    cache_row = fund_cache.loc[fund_cache["symbol"] == symbol]
    if not cache_row.empty:
        last = pd.to_datetime(cache_row["last_refresh"].iloc[0]).date()
        age = (today - last).days
        if age <= FUND_CACHE_DAYS:
            cached = {k: float(cache_row[k].iloc[0]) for k in _FUND_FALLBACK.keys()}
            return cached, float(age), fund_cache

    # Refresh
    live = _fetch_live_fundamentals(symbol)
    age_days = 0.0
    new_row = {"symbol": symbol, "last_refresh": today.isoformat(), **live}
    if cache_row.empty:
        fund_cache = pd.concat([fund_cache, pd.DataFrame([new_row])], ignore_index=True)
    else:
        idx = cache_row.index[0]
        for k, v in new_row.items():
            fund_cache.at[idx, k] = v
    return live, age_days, fund_cache


# ─────────────────────────────────────────────────────────────────────────────
# Indicadores técnicos (mesmo código que bootstrap_ml.py — re-export)
# ─────────────────────────────────────────────────────────────────────────────

def _calc_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    """Adiciona rsi, atr, atr_ratio, ddp_52w, vol_ratio, ret_1d ao DataFrame."""
    from bootstrap_ml import calc_rsi, calc_atr, calc_volume_ratio
    if hist.empty or len(hist) < 20:
        return hist
    hist = hist.copy()
    hist["rsi"]       = calc_rsi(hist["Close"])
    hist["atr"]       = calc_atr(hist)
    hist["atr_ratio"] = hist["atr"] / (hist["Close"] + 1e-9)
    hist["vol_ratio"] = calc_volume_ratio(hist)
    hist["ret_1d"]    = hist["Close"].pct_change() * 100
    rolling_max = hist["Close"].rolling(window=252, min_periods=20).max()
    hist["ddp_52w"]   = (hist["Close"] - rolling_max) / rolling_max * 100
    return hist


def _fetch_history(symbol: str, lookback_days: int = 280) -> pd.DataFrame:
    """
    Tenta data_feed (Tiingo→yf→Stooq), fallback yfinance directo.
    280d ≈ 400d corridos → garante ≥252d trading days para 52w drawdown.
    Retorna DataFrame com DatetimeIndex e colunas Title-case (Open, High, Low, Close, Volume).
    """
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=lookback_days)

    try:
        from data_feed import get_eod_prices
        df = get_eod_prices(symbol, lookback_days=lookback_days)
        if df is not None and not df.empty:
            # data_feed devolve `date` como coluna — promover a index
            if "date" in df.columns:
                df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df["date"])))
                df = df.drop(columns=["date"])
            # Garante index DatetimeIndex sem timezone
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.DatetimeIndex(df.index)
            try:
                df.index = df.index.tz_localize(None)
            except (TypeError, AttributeError):
                pass  # já naive
            # Normalizar nomes Yahoo-style — preservar Volume e Adj Close
            return df
    except Exception as e:
        log.debug(f"[hist] {symbol}: data_feed falhou ({e}); a tentar yfinance directo.")

    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(start=start.isoformat(), end=end.isoformat(),
                                        auto_adjust=True, raise_errors=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        return df
    except Exception as e:
        log.debug(f"[hist] {symbol}: yfinance falhou ({e}).")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Construção da linha de snapshot
# ─────────────────────────────────────────────────────────────────────────────

def _build_row(
    symbol: str,
    snapshot_date: date,
    macro: dict,
    fund_cache: pd.DataFrame,
) -> tuple[Optional[dict], pd.DataFrame]:
    """
    Constroi 1 linha de snapshot para (symbol, snapshot_date).
    Devolve (row|None, fund_cache_actualizado).
    """
    hist = _fetch_history(symbol)
    if hist.empty or len(hist) < 20:
        return None, fund_cache

    hist = _calc_indicators(hist)
    last = hist.iloc[-1]
    snapshot_ts = hist.index[-1].date()

    # Se a última candle é mais antiga que snapshot_date - 7d, considera obsoleta
    if (snapshot_date - snapshot_ts).days > 7:
        log.debug(f"[snapshot] {symbol}: última candle {snapshot_ts} muito antiga, skip.")
        return None, fund_cache

    fund, age_days, fund_cache = _get_fundamentals(symbol, snapshot_date, fund_cache)

    from bootstrap_ml import safe_float
    row: dict = {
        "symbol":             symbol,
        "snapshot_date":      snapshot_date.isoformat(),
        "price":              round(float(last["Close"]), 4),
        # Técnicos
        "drop_pct_today":     round(safe_float(last.get("ret_1d"), 0.0), 3),
        "drawdown_52w":       round(safe_float(last.get("ddp_52w"), -15.0), 3),
        "rsi_14":             round(float(np.clip(safe_float(last.get("rsi"), 50.0), 0, 100)), 1),
        "atr_ratio":          round(safe_float(last.get("atr_ratio"), 0.02), 6),
        "volume_spike":       round(safe_float(last.get("vol_ratio"), 1.0), 4),
        # Macro
        "macro_score":        macro.get("macro_score", 2),
        "vix":                macro.get("vix", 20.0),
        "spy_drawdown_5d":    macro.get("spy_drawdown_5d", 0.0),
        "sector_drawdown_5d": macro.get("sector_drawdown_5d", 0.0),
        # Fundamentais (live ou cache 7d)
        **{k: float(fund[k]) for k in _FUND_FALLBACK.keys()},
        # Telemetria
        "data_source":        "yfinance",
        "fund_age_days":      float(age_days),
        "ingest_ts":          datetime.utcnow().isoformat(timespec="seconds"),
    }

    # Derived features (mesmo código de inferência)
    from ml_features import add_derived_features
    add_derived_features(row)

    # Garantir todas as colunas presentes
    for col in SNAPSHOT_COLUMNS:
        if col not in row:
            row[col] = None
    return {col: row[col] for col in SNAPSHOT_COLUMNS}, fund_cache


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point: snapshot diário do universo
# ─────────────────────────────────────────────────────────────────────────────

def run_daily_snapshot(
    tickers: Optional[list[str]] = None,
    snapshot_date: Optional[date] = None,
    sleep_between: float = 0.10,
    dry_run: bool = False,
) -> dict:
    """
    Itera o universo, computa features e grava em SNAPSHOT_PATH.
    Idempotente: skip de tickers já processados para snapshot_date.

    Returns dict com stats: {processed, skipped, failed, total, elapsed_s, path}.
    """
    snapshot_date = snapshot_date or date.today()
    if tickers is None:
        tickers = _load_universe()

    iso = snapshot_date.isoformat()
    existing = _load_existing_keys(iso)
    todo = [t for t in tickers if t not in existing]

    log.info(f"[snapshot] Universo: {len(tickers)} tickers | já feitos: {len(existing)} | "
             f"a processar: {len(todo)} | data: {iso}")

    if not todo:
        return {"processed": 0, "skipped": len(tickers), "failed": 0,
                "total": len(tickers), "path": str(SNAPSHOT_PATH), "elapsed_s": 0.0}

    # Macro (1× por run) — get_macro_context tem cache interno, mas force fresh aqui
    try:
        from macro_data import get_macro_context
        macro = get_macro_context(force_refresh=True)
        log.info(f"[snapshot] macro: VIX={macro.get('vix'):.1f} "
                 f"SPY5d={macro.get('spy_drawdown_5d'):+.2f}% "
                 f"score={macro.get('macro_score')}")
    except Exception as e:
        log.error(f"[snapshot] get_macro_context falhou ({e}); a usar defaults")
        macro = {"macro_score": 2, "vix": 20.0,
                 "spy_drawdown_5d": 0.0, "sector_drawdown_5d": 0.0}

    fund_cache = _load_fund_cache()

    processed = 0
    failed    = 0
    rows: list[dict] = []
    t_start = time.time()

    for i, symbol in enumerate(todo):
        try:
            row, fund_cache = _build_row(symbol, snapshot_date, macro, fund_cache)
            if row is None:
                failed += 1
                continue
            rows.append(row)
            processed += 1
        except Exception as e:
            log.warning(f"[snapshot] {symbol}: erro inesperado ({e})")
            failed += 1

        # Flush a cada 100 tickers para limitar perda em caso de crash
        if not dry_run and len(rows) >= 100:
            _append_rows(rows)
            rows = []

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            log.info(f"[snapshot] {i+1}/{len(todo)} | "
                     f"ok={processed} fail={failed} | {rate:.1f} tk/s")

        if sleep_between > 0:
            time.sleep(sleep_between)

    if not dry_run and rows:
        _append_rows(rows)
    if not dry_run:
        _save_fund_cache(fund_cache)

    elapsed = time.time() - t_start
    log.info(f"[snapshot] CONCLUÍDO: processed={processed} failed={failed} "
             f"skipped={len(existing)} elapsed={elapsed:.0f}s "
             f"→ {SNAPSHOT_PATH}")

    return {
        "processed": processed,
        "skipped":   len(existing),
        "failed":    failed,
        "total":     len(tickers),
        "elapsed_s": round(elapsed, 1),
        "path":      str(SNAPSHOT_PATH),
        "snapshot_date": iso,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily universe snapshot for ML training feed.")
    p.add_argument("--tickers", type=str, default=None,
                   help="Lista de tickers (CSV) — default: get_ml_universe().")
    p.add_argument("--date", type=str, default=None,
                   help="Override snapshot date (YYYY-MM-DD). Default: today.")
    p.add_argument("--sleep", type=float, default=0.10,
                   help="Segundos entre tickers (default 0.10).")
    p.add_argument("--dry-run", action="store_true",
                   help="Não escreve parquet — útil para debug.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    tickers: Optional[list[str]] = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    snap_date: Optional[date] = None
    if args.date:
        snap_date = date.fromisoformat(args.date)

    stats = run_daily_snapshot(
        tickers=tickers,
        snapshot_date=snap_date,
        sleep_between=args.sleep,
        dry_run=args.dry_run,
    )
    log.info(f"[snapshot] stats: {stats}")


if __name__ == "__main__":
    main()
