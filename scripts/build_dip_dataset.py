"""
scripts/build_dip_dataset.py — Constrói dataset histórico de treino para DipRadar.

Equivalente a momentum_radar/scripts/build_dataset.py mas para condições de dip.
Varre dados históricos de preço desde 1993 para cada ticker no universo,
detecta episódios de queda, e constrói rows completos com TODAS as features
do DipRadar — reutilizando as funções de regenerate_training_base.py para
garantir consistência absoluta com a produção.

Detecção de episódio (sem cooldown fixo):
  Episódio inicia quando return_20d < _EPISODE_ENTRY (default -5%).
  Termina quando return_20d recupera acima de _EPISODE_EXIT (default -2%).
  Só o primeiro dia qualificado de cada episódio é amostrado.
  Critérios propositadamente largos — modelo aprende os thresholds óptimos.

Modo incremental: se o parquet de output já existir, processa apenas
  datas novas (último registo - 200 dias) e faz merge com dedup.

Uso:
  python scripts/build_dip_dataset.py                    # universo completo, desde 1993
  python scripts/build_dip_dataset.py --tickers AAPL,MSFT  # teste rápido
  python scripts/build_dip_dataset.py --start 1985-01-01    # ir mais longe
  python scripts/build_dip_dataset.py --no-fundamentals     # mais rápido, sem EDGAR
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

# Importar infraestrutura partilhada com regenerate_training_base.py
# (feature computation, price fetching, sector maps, column lists)
from scripts.regenerate_training_base import (
    _fetch_one_ticker,
    _fetch_batch,
    _compute_row_features,
    _compute_pit_fundamentals_row,
    _compute_alpha_targets_row,
    _compute_fase5_row,
    SECTOR_ETF,
    MACRO_TICKERS,
    SECTOR_ETFS,
    NEW_FEATURE_COLS,
    FUNDAMENTAL_COLS,
    FASE5_COLS,
)

log = logging.getLogger("build_dip_dataset")

_DATA_DIR   = Path("/data") if Path("/data").exists() else Path("/tmp")
_CACHE_DIR  = _DATA_DIR / "price_cache"
_OUTPUT     = Path(__file__).parent.parent / "ml_training_base.parquet"

# ── Detecção de episódio ──────────────────────────────────────────────────────
# Propositadamente largos: o modelo aprende os thresholds óptimos.
# Histerese: impede re-triggers durante recuperações parciais do mesmo dip.
_EPISODE_ENTRY = -0.05   # return_20d < -5%  → início de episódio de dip
_EPISODE_EXIT  = -0.02   # return_20d > -2%  → episódio terminou (recuperação)

_MIN_MARKET_CAP_B = 1.0  # mínimo de liquidez (USD mil milhões)
_MIN_BARS_AHEAD   = 95   # dias úteis mínimos necessários para target de 90d


# ─────────────────────────────────────────────────────────────────────────────
# Target adicional: max_drawdown_60d (modelo de risco do DipRadar)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_max_drawdown_90d(
    hist: pd.DataFrame,
    alert_date: pd.Timestamp,
    horizon_days: int = 90,
) -> Optional[float]:
    """Drawdown máximo nos `horizon_days` dias após alert_date."""
    entry_slice = hist[hist.index <= alert_date]
    if entry_slice.empty:
        return None
    entry_price = float(entry_slice["Close"].iloc[-1])
    if entry_price <= 0:
        return None
    fwd = hist[
        (hist.index > alert_date) &
        (hist.index <= alert_date + pd.Timedelta(days=horizon_days))
    ]
    if len(fwd) < 5:
        return None
    lowest = float(fwd["Close"].min())
    dd = lowest / entry_price - 1.0
    return round(dd, 6) if math.isfinite(dd) else None


# ─────────────────────────────────────────────────────────────────────────────
# Deteção de episódios de dip por ticker
# ─────────────────────────────────────────────────────────────────────────────

def _find_dip_episodes(hist: pd.DataFrame) -> list[pd.Timestamp]:
    """Devolve lista de datas de início de cada episódio de dip.

    Percorre cada barra; quando return_20d cruza abaixo de _EPISODE_ENTRY
    e não estamos já num episódio activo, marca o início. O episódio termina
    quando return_20d recupera acima de _EPISODE_EXIT.
    """
    close = hist["Close"]
    n = len(hist)
    episode_dates: list[pd.Timestamp] = []
    in_episode = False

    for i in range(20, n - _MIN_BARS_AHEAD):
        if i < 20:
            continue
        ret_20d = float(close.iloc[i] / close.iloc[i - 20] - 1)

        if in_episode and ret_20d > _EPISODE_EXIT:
            in_episode = False  # recuperação suficiente — episódio terminou

        if in_episode:
            continue  # ainda no mesmo dip — não amostrar novamente

        if ret_20d < _EPISODE_ENTRY:
            in_episode = True
            episode_dates.append(hist.index[i])

    return episode_dates


# ─────────────────────────────────────────────────────────────────────────────
# Build dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_dip_dataset(
    tickers: list[str],
    price_cache: dict[str, pd.DataFrame],
    macro_cache: dict[str, pd.DataFrame],
    sector_etf_cache: dict[str, pd.DataFrame],
    cache_dir: Path,
    add_fundamentals: bool = True,
    add_fase5: bool = True,
) -> pd.DataFrame:
    """Constrói dataset de treino para DipRadar varrendo histórico de preços."""
    from ml_features import _FALLBACK

    spy_hist = macro_cache.get("SPY")
    rows: list[dict] = []
    n_tickers = len(tickers)

    for i, ticker in enumerate(tickers):
        hist = price_cache.get(ticker)
        if hist is None or len(hist) < 60:
            continue

        # Sector via yfinance info (com fallback)
        sector = "Unknown"
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info or {}
            sector = info.get("sector") or "Unknown"
            mktcap = float(info.get("marketCap") or 0) / 1e9
            if mktcap > 0 and mktcap < _MIN_MARKET_CAP_B:
                continue
        except Exception:
            pass

        episode_dates = _find_dip_episodes(hist)

        for alert_date in episode_dates:
            row_base = {
                "ticker":     ticker,
                "alert_date": alert_date,
                "price":      float(hist[hist.index <= alert_date]["Close"].iloc[-1]),
                "sector":     sector,
            }

            # ── Features técnicas (partilhadas com produção) ──────────────
            try:
                feat = _compute_row_features(
                    row_base, price_cache, macro_cache, sector_etf_cache
                )
            except Exception as e:
                log.debug(f"  features failed {ticker}@{alert_date.date()}: {e}")
                feat = {k: _FALLBACK.get(k, 0.0) for k in NEW_FEATURE_COLS}
            row_base.update(feat)

            # ── Target: alpha_90d ─────────────────────────────────────────
            try:
                targets = _compute_alpha_targets_row(
                    pd.Series(row_base), price_cache, spy_hist
                )
            except Exception:
                targets = {"alpha_90d": float("nan"), "close_90d": float("nan"), "spy_close_90d": float("nan")}
            row_base.update(targets)

            if not math.isfinite(row_base.get("alpha_90d", float("nan"))):
                continue  # sem target = inutilizável para treino

            # ── Target: max_drawdown_90d (modelo de risco) ────────────────
            dd = _compute_max_drawdown_90d(hist, alert_date)
            row_base["max_drawdown_90d"] = dd if dd is not None else 0.0

            # ── Fundamentais PIT (opcional — requer EDGAR/Tiingo) ─────────
            if add_fundamentals:
                try:
                    fund = _compute_pit_fundamentals_row(pd.Series(row_base), cache_dir)
                    row_base.update(fund)
                except Exception:
                    for c in FUNDAMENTAL_COLS:
                        row_base.setdefault(c, _FALLBACK.get(c, 0.0))
            else:
                for c in FUNDAMENTAL_COLS:
                    row_base.setdefault(c, _FALLBACK.get(c, 0.0))

            # ── Features Fase5 (EDGAR PIT) ────────────────────────────────
            if add_fase5:
                try:
                    f5 = _compute_fase5_row(pd.Series(row_base), price_cache)
                    row_base.update(f5)
                except Exception:
                    for c in FASE5_COLS:
                        row_base.setdefault(c, _FALLBACK.get(c, 0.0))
            else:
                for c in FASE5_COLS:
                    row_base.setdefault(c, _FALLBACK.get(c, 0.0))

            rows.append(row_base)

        if (i + 1) % 50 == 0 or (i + 1) == n_tickers:
            log.info(f"  [{i+1}/{n_tickers}] {ticker} | {len(rows)} episódios de dip até agora")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Constrói dataset histórico de treino para DipRadar")
    p.add_argument("--tickers",          help="Tickers separados por vírgula (teste rápido)")
    p.add_argument("--start",            default="1993-01-01", help="Data de início (default: 1993)")
    p.add_argument("--out",              default=str(_OUTPUT),  help="Output parquet")
    p.add_argument("--cache-dir",        default=str(_CACHE_DIR), help="Dir cache de preços")
    p.add_argument("--no-fundamentals",  action="store_true", help="Saltar EDGAR/Tiingo (mais rápido)")
    p.add_argument("--no-fase5",         action="store_true", help="Saltar features Fase5 EDGAR")
    p.add_argument("--force-rebuild",    action="store_true", help="Ignorar parquet existente e reconstruir desde --start")
    p.add_argument("-v", "--verbose",    action="store_true")
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
        log.info(f"Modo teste: {len(tickers)} tickers")
    else:
        from universe import get_ml_universe
        tickers = get_ml_universe()
        log.info(f"Universo completo: {len(tickers)} tickers")

    # ── Modo incremental ──────────────────────────────────────────────────────
    existing_df: Optional[pd.DataFrame] = None
    start = args.start
    if args.force_rebuild and out_path.exists():
        log.info(f"--force-rebuild: a ignorar parquet existente em {out_path}")
    if out_path.exists() and not args.tickers and not args.force_rebuild:
        try:
            existing_df = pd.read_parquet(out_path)
            if "alert_date" in existing_df.columns and not existing_df.empty:
                last_date = pd.to_datetime(existing_df["alert_date"]).max()
                # 200d lookback: 90d target + 110d buffer
                cutoff = (last_date - pd.Timedelta(days=200)).strftime("%Y-%m-%d")
                start = cutoff
                log.info(f"Dataset existente: {len(existing_df)} rows até {last_date.date()}")
                log.info(f"Modo incremental: a processar desde {start}")
        except Exception as e:
            log.warning(f"Parquet existente ilegível ({e}) — a reconstruir de raiz")
            existing_df = None

    # ── Fetch preços (partilha cache com regenerate_training_base.py) ─────────
    log.info(f"A descarregar macro (SPY, VIX, TNX) desde {start}...")
    macro_cache: dict[str, pd.DataFrame] = {}
    for t in MACRO_TICKERS:
        df = _fetch_one_ticker(t, cache_dir, start=start)
        if df is not None:
            macro_cache[t] = df

    log.info("A descarregar sector ETFs...")
    sector_etf_cache: dict[str, pd.DataFrame] = {}
    for t in SECTOR_ETFS:
        df = _fetch_one_ticker(t, cache_dir, start=start)
        if df is not None:
            sector_etf_cache[t] = df

    log.info(f"A descarregar preços de {len(tickers)} tickers...")
    price_cache = _fetch_batch(tickers, cache_dir, start=start)
    log.info(f"Preços carregados: {len(price_cache)}/{len(tickers)} tickers")

    # ── Build dataset ─────────────────────────────────────────────────────────
    log.info(
        f"A construir dataset de dips desde {start} "
        f"(entry<{_EPISODE_ENTRY:.0%}, exit>{_EPISODE_EXIT:.0%})..."
    )
    new_df = build_dip_dataset(
        tickers=list(price_cache.keys()),
        price_cache=price_cache,
        macro_cache=macro_cache,
        sector_etf_cache=sector_etf_cache,
        cache_dir=cache_dir,
        add_fundamentals=not args.no_fundamentals,
        add_fase5=not args.no_fase5,
    )

    if new_df.empty:
        if existing_df is not None and not existing_df.empty:
            log.info("Sem novos episódios — dataset existente mantido")
            return 0
        log.error("Dataset vazio — nenhum episódio de dip encontrado")
        return 1

    # ── Merge com dados existentes ────────────────────────────────────────────
    if existing_df is not None and not existing_df.empty:
        # Alertas reais do bot têm prioridade (keep='first' após ordenar real > sim)
        # Marcar simulados para desambiguação no merge
        if "source" not in existing_df.columns:
            existing_df["source"] = "real"
        new_df["source"] = "simulated"

        combined = pd.concat([existing_df, new_df], ignore_index=True)
        # Dedup: preferir "real" sobre "simulated" (real alerts têm melhor qualidade)
        combined["_sort"] = (combined.get("source", "simulated") == "real").astype(int)
        combined = combined.sort_values("_sort", ascending=False).drop(columns=["_sort"])
        combined = combined.drop_duplicates(subset=["ticker", "alert_date"], keep="first")
        combined = combined.sort_values("alert_date").reset_index(drop=True)
        n_new = len(combined) - len(existing_df)
        log.info(f"Merge: {len(existing_df)} existentes + {len(new_df)} novos → {len(combined)} total ({n_new} adicionados)")
    else:
        new_df["source"] = "simulated"
        combined = new_df.sort_values("alert_date").reset_index(drop=True)
        log.info(f"Dataset novo: {len(combined)} episódios")

    # ── Guardar ───────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)

    size_mb = out_path.stat().st_size / 1e6
    log.info(
        f"Guardado: {out_path} ({size_mb:.1f}MB) | "
        f"{len(combined)} rows | {combined['ticker'].nunique()} tickers | "
        f"{combined['sector'].nunique() if 'sector' in combined.columns else '?'} sectores"
    )
    if "alpha_90d" in combined.columns:
        v = combined["alpha_90d"].dropna()
        log.info(f"alpha_90d: mean={v.mean():.3f} std={v.std():.3f} ({len(v)} amostras)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
