"""
build_bootstrap_dataset.py — Gera o dataset histórico de treino desde 1995.

Simula o DipRadar retrospectivamente: para cada ticker e cada dia desde
1995, detecta dias de "dip" (queda significativa do máximo de 52 semanas)
e gera uma linha de treino com features + alpha_90d.

Captura todos os regimes de mercado importantes:
  - Rally dot-com 1995-2000
  - Crash dot-com 2000-2002
  - Recuperação 2003-2007
  - Crise financeira 2008-2009
  - Bull market 2010-2019
  - COVID crash e recuperação 2020
  - Subida de taxas 2022
  - AI rally 2023-2024

Critérios de dip (replicam o DipRadar em produção):
  - Drawdown do máximo 52w <= -15%
  - Queda no dia >= 2% (não é uma queda aleatória, é uma queda real)
  - RSI < 50 (momentum negativo)
  - Volume mínimo para liquidez

Amostragem:
  - Máximo 1 alerta por ticker por mês (evita autocorrelação temporal)
  - Só o dia com maior drawdown por mês é seleccionado

Output: /data/ml_training_base.parquet
  - Substitui o parquet existente (bootstrap é a fonte de verdade)
  - Features técnicas computadas PIT (só dados até alert_date)
  - alpha_90d computado onde disponível

Uso:
  python scripts/build_bootstrap_dataset.py [--start 1995] [--end 2026]
  python scripts/build_bootstrap_dataset.py --start 2010  # mais rápido
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("bootstrap")

_DATA_DIR  = Path("/data") if Path("/data").exists() else Path("/tmp")
_CACHE_DIR = _DATA_DIR / "price_cache"
_OUTPUT    = _DATA_DIR / "ml_training_base.parquet"

# Critérios de dip — alinhados com o scanner de produção
_MIN_DRAWDOWN_52W = -0.15   # >= 15% abaixo do máximo de 52 semanas
_MIN_DAY_DROP     = -0.02   # >= 2% de queda no dia
_MAX_RSI          = 50.0    # RSI < 50 (tendência negativa)
_MIN_VOLUME_RATIO = 0.5     # volume mínimo (evita dias sem liquidez)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de preços
# ─────────────────────────────────────────────────────────────────────────────

def _load_ticker(ticker: str, cache_dir: Path) -> Optional[pd.DataFrame]:
    """Lê histórico do cache local."""
    safe = ticker.replace("^", "_idx_").replace("/", "_").replace(".", "_")
    f = cache_dir / f"{safe}.parquet"
    if not f.exists():
        return None
    try:
        df = pd.read_parquet(f)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        return df.sort_index()
    except Exception as e:
        log.debug(f"  {ticker}: erro a ler cache — {e}")
        return None


def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


# ─────────────────────────────────────────────────────────────────────────────
# Detecção histórica de dips
# ─────────────────────────────────────────────────────────────────────────────

def find_historical_dips(
    ticker: str,
    hist: pd.DataFrame,
    spy_hist: pd.DataFrame,
    sector: str,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """Encontra todos os dias de dip para um ticker num período histórico.

    Retorna lista de rows prontos para o parquet de treino.
    Amostra 1 dip por mês (o maior drawdown do mês) para evitar autocorrelação.
    """
    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date) - pd.Timedelta(days=95)  # precisa de 90d forward

    close  = hist["Close"]
    volume = hist.get("Volume", pd.Series(dtype=float))

    # RSI para todo o histórico
    rsi_series = _calc_rsi(close)

    # Rolling max 52w (252 dias úteis)
    rolling_max = close.rolling(252, min_periods=60).max()

    # Filtrar período
    mask = (hist.index >= start) & (hist.index <= end)
    dates = hist.index[mask]
    if len(dates) < 10:
        return []

    # Volume médio móvel 20d
    vol_ma = volume.rolling(20, min_periods=5).mean()

    rows = []
    # Agrupar por mês — só o maior drawdown por mês
    monthly_best: dict[str, tuple] = {}  # "YYYY-MM" → (date, drawdown)

    for dt in dates:
        idx = close.index.get_loc(dt)
        if idx < 60:
            continue

        c_now  = float(close.iloc[idx])
        c_prev = float(close.iloc[idx - 1]) if idx >= 1 else c_now
        if c_prev <= 0 or c_now <= 0:
            continue

        day_change = (c_now - c_prev) / c_prev
        max_52w    = float(rolling_max.iloc[idx])
        if max_52w <= 0:
            continue
        drawdown = (c_now - max_52w) / max_52w

        # Filtros básicos
        if drawdown > _MIN_DRAWDOWN_52W:
            continue
        if day_change > _MIN_DAY_DROP:
            continue

        rsi = float(rsi_series.iloc[idx])
        if rsi >= _MAX_RSI:
            continue

        # Volume
        vol_now = float(volume.iloc[idx]) if not volume.empty else 0
        vol_avg = float(vol_ma.iloc[idx]) if not vol_ma.empty else 1
        if vol_avg > 0 and vol_now < vol_avg * _MIN_VOLUME_RATIO:
            continue

        month_key = dt.strftime("%Y-%m")
        if month_key not in monthly_best or drawdown < monthly_best[month_key][1]:
            monthly_best[month_key] = (dt, drawdown)

    # Construir rows para os melhores dips de cada mês
    for dt, drawdown in monthly_best.values():
        idx   = close.index.get_loc(dt)
        c_now = float(close.iloc[idx])

        # Features básicas PIT
        rsi   = float(rsi_series.iloc[idx])
        c_prev = float(close.iloc[idx - 1])
        day_ch = (c_now - c_prev) / c_prev

        # Momentum 1m/3m
        ret_1m  = (c_now / float(close.iloc[max(0, idx-21)]) - 1) if idx >= 21 else 0.0
        ret_3m  = (c_now / float(close.iloc[max(0, idx-63)]) - 1) if idx >= 63 else 0.0
        ret_6m  = (c_now / float(close.iloc[max(0, idx-126)]) - 1) if idx >= 126 else 0.0

        # BB width
        close_slice = close.iloc[max(0, idx-20):idx+1]
        bb_width = float(close_slice.std() / close_slice.mean()) if len(close_slice) >= 5 else 0.12

        # Volume ratio
        vol_slice = volume.iloc[max(0, idx-20):idx+1] if not volume.empty else pd.Series()
        vol_ratio = float(volume.iloc[idx] / vol_slice.mean()) if len(vol_slice) >= 5 and vol_slice.mean() > 0 else 1.0

        # Alpha_90d — só se tiver dados 90d à frente
        alpha_90d   = float("nan")
        close_90d   = float("nan")
        spy_close_90d = float("nan")
        try:
            future = close.iloc[idx+1:idx+95]
            if len(future) >= 88:
                c90  = float(future.iloc[-1])
                ret  = c90 / c_now - 1.0
                if abs(ret) <= 2.0:
                    close_90d = round(ret, 6)
                    # SPY no mesmo período
                    spy_at  = spy_hist[spy_hist.index <= dt]["Close"]
                    spy_fwd = spy_hist[(spy_hist.index > dt) &
                                       (spy_hist.index <= dt + pd.Timedelta(days=95))]["Close"]
                    if not spy_at.empty and not spy_fwd.empty:
                        spy_r = float(spy_fwd.iloc[-1]) / float(spy_at.iloc[-1]) - 1.0
                        if abs(spy_r) <= 1.0:
                            spy_close_90d = round(spy_r, 6)
                            alpha_90d = round(
                                np.log1p(ret) - np.log1p(spy_r), 6
                            )
        except Exception:
            pass

        rows.append({
            "ticker":         ticker,
            "alert_date":     dt.date().isoformat(),
            "sector":         sector,
            "price":          round(c_now, 4),
            # Features técnicas — computadas PIT
            "drop_pct_today": round(day_ch, 4),
            "drawdown_52w":   round(drawdown, 4),
            "rsi_14":         round(rsi, 2),
            "return_1m":      round(ret_1m, 4),
            "return_3m_pre":  round(ret_3m, 4),
            "return_6m_pre":  round(ret_6m, 4),
            "bb_width":       round(bb_width, 4),
            "volume_zscore_20d": round(vol_ratio - 1.0, 4),
            # Targets
            "alpha_90d":      alpha_90d,
            "close_90d":      close_90d,
            "spy_close_90d":  spy_close_90d,
            "max_drawdown_60d": float("nan"),  # a preencher pelo regen
        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="Constrói dataset histórico de treino desde o ano especificado"
    )
    p.add_argument("--start", default="1993-01-01",
                   help="Data de início (YYYY-MM-DD). Default: 1993-01-01 (máximo yfinance)")
    p.add_argument("--end",   default="today",
                   help="Data de fim. Default: hoje")
    p.add_argument("--out",   default=str(_OUTPUT))
    p.add_argument("--tickers",
                   help="Subset de tickers para teste (ex: AAPL,MSFT,NVDA)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
    )

    end_date = pd.Timestamp.today().strftime("%Y-%m-%d") if args.end == "today" else args.end
    log.info(f"Bootstrap histórico: {args.start} → {end_date}")

    # Tickers
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
        log.info(f"Modo teste: {len(tickers)} tickers")
    else:
        from universe import get_ml_universe
        tickers = get_ml_universe()
        log.info(f"Universo completo: {len(tickers)} tickers")

    # Carregar SPY
    spy_hist = _load_ticker("SPY", _CACHE_DIR)
    if spy_hist is None:
        log.error("SPY não disponível no cache. Corre /admin_regen_parquet primeiro.")
        return 1

    # Sector mapping
    sector_map: dict[str, str] = {}
    try:
        import yfinance as yf
        log.info("A obter sectores dos tickers...")
        for i, tk in enumerate(tickers):
            try:
                info = yf.Ticker(tk).info or {}
                sector_map[tk] = info.get("sector", "Unknown") or "Unknown"
            except Exception:
                sector_map[tk] = "Unknown"
            if (i + 1) % 100 == 0:
                log.info(f"  Sectores: {i+1}/{len(tickers)}")
            time.sleep(0.02)
    except Exception as e:
        log.warning(f"Erro a obter sectores: {e} — a usar 'Unknown'")

    # Notificações Telegram (via send_fn se disponível)
    _send_fn = None
    try:
        import importlib, os
        if os.getenv("TELEGRAM_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
            _bot = importlib.import_module("bot_commands")
            _send_fn = getattr(_bot, "_reply", None) or getattr(_bot, "_cb_send_telegram", None)
    except Exception:
        pass

    def _notify(msg: str) -> None:
        log.info(msg)
        if _send_fn:
            try:
                _send_fn(msg)
            except Exception:
                pass

    _notify(f"🔄 *Bootstrap {args.start[:4]}–{end_date[:4]}* iniciado — {len(tickers)} tickers")

    # Gerar dips históricos
    all_rows = []
    t0 = time.time()
    _milestone_pct = {25, 50, 75}

    for i, ticker in enumerate(tickers):
        hist = _load_ticker(ticker, _CACHE_DIR)
        if hist is None or len(hist) < 100:
            continue

        sector = sector_map.get(ticker, "Unknown")
        rows = find_historical_dips(
            ticker, hist, spy_hist, sector, args.start, end_date
        )
        all_rows.extend(rows)

        pct_done = int((i + 1) / len(tickers) * 100)
        if pct_done in _milestone_pct:
            _milestone_pct.discard(pct_done)
            elapsed = time.time() - t0
            _notify(
                f"⏳ Bootstrap {pct_done}% — {i+1}/{len(tickers)} tickers | "
                f"{len(all_rows):,} dips | {elapsed:.0f}s"
            )

        if (i + 1) % 50 == 0 or (i + 1) == len(tickers):
            elapsed = time.time() - t0
            log.info(
                f"  [{i+1:>3}/{len(tickers)}] {ticker} "
                f"| dips={len(rows)} | total={len(all_rows):,} | {elapsed:.0f}s"
            )

    if not all_rows:
        log.error("Nenhum dip encontrado. Verifica se o price cache está populado.")
        return 1

    df = pd.DataFrame(all_rows)
    df["alert_date"] = pd.to_datetime(df["alert_date"])
    df = df.sort_values("alert_date").reset_index(drop=True)

    n_with_alpha = int(df["alpha_90d"].notna().sum())
    pct = n_with_alpha / len(df)

    log.info(f"Dataset: {len(df):,} dips históricos | {df['ticker'].nunique()} tickers")
    log.info(f"Período: {df['alert_date'].min().date()} → {df['alert_date'].max().date()}")
    log.info(f"alpha_90d resolvido: {n_with_alpha:,}/{len(df):,} ({pct:.0%})")
    log.info(f"Por ano:")
    for yr, cnt in df.groupby(df["alert_date"].dt.year).size().items():
        log.info(f"  {yr}: {cnt:,} dips")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / 1e6
    log.info(f"Guardado em {out_path} ({size_mb:.1f}MB)")
    _notify(
        f"✅ *Bootstrap concluído!*\n"
        f"  {len(df):,} dips históricos | {df['ticker'].nunique()} tickers\n"
        f"  {df['alert_date'].min().date()} → {df['alert_date'].max().date()}\n"
        f"  alpha\\_90d: {n_with_alpha:,}/{len(df):,} ({pct:.0%})\n"
        f"  {size_mb:.1f}MB\n\n"
        f"_A iniciar enriquecimento de features..._"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
