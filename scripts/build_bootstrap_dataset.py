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

_REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ → repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("bootstrap")


def _ensure_price_cache(tickers: list[str], cache_dir: Path, start: str) -> None:
    """Garante que o price cache tem dados desde `start` para todos os tickers.

    Para cada ticker, verifica se o ficheiro de cache existe E se começa antes
    de `start`. Se não, apaga e re-descarrega via yfinance.
    Sem este passo, o bootstrap só encontra dips a partir da data do cache existente.
    """
    import yfinance as yf
    start_ts = pd.Timestamp(start)
    to_download = []

    for ticker in tickers:
        safe = ticker.replace("^", "_idx_").replace("/", "_").replace(".", "_")
        cache_file = cache_dir / f"{safe}.parquet"
        needs_download = True
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                idx_min = pd.to_datetime(df.index.min())
                if idx_min.tz is not None:
                    idx_min = idx_min.tz_convert(None)
                if idx_min <= start_ts + pd.Timedelta(days=30):  # 30d buffer
                    needs_download = False  # cache cobre o período
            except Exception:
                pass
        if needs_download:
            to_download.append(ticker)

    if not to_download:
        log.info(f"[cache] Price cache OK desde {start} — nenhum re-download necessário")
        return

    log.info(f"[cache] {len(to_download)}/{len(tickers)} tickers sem histórico desde {start} — a descarregar...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for ticker in to_download:
        safe = ticker.replace("^", "_idx_").replace("/", "_").replace(".", "_")
        cache_file = cache_dir / f"{safe}.parquet"
        if cache_file.exists():
            cache_file.unlink()  # remover cache incompleto
        try:
            df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
            if df is not None and not df.empty:
                idx = pd.DatetimeIndex(df.index)
                df.index = idx.tz_convert(None) if idx.tz is not None else idx
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                df.to_parquet(cache_file)
                done += 1
        except Exception as e:
            log.debug(f"  {ticker}: {e}")
        time.sleep(0.05)
        if (done + 1) % 100 == 0:
            log.info(f"  [cache] {done}/{len(to_download)} re-descarregados")
    log.info(f"[cache] Download completo: {done}/{len(to_download)} tickers actualizados")

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

def precompute_macro_series(macro_cache: dict) -> dict:
    """Pré-calcula TODAS as séries macro uma única vez.

    Converte cada série histórica em pd.Series indexed by date para lookup O(1).
    Sem isto, cada linha do loop recalcularia RSI e percentis em O(n) — levaria horas.
    """
    result: dict = {}

    # VIX: valor, percentile 1 ano, regime
    vix_hist = macro_cache.get("^VIX")
    if vix_hist is None:
        vix_hist = macro_cache.get("_idx_VIX")
    if vix_hist is not None and not vix_hist.empty:
        vix_close = vix_hist["Close"]
        result["vix"]       = vix_close
        # Rolling percentile 252d (1 ano)
        result["vix_pct"]   = vix_close.rolling(252, min_periods=60).apply(
            lambda x: (x[:-1] < x[-1]).mean(), raw=True
        )
        # VIX regime: 0=low(<15), 1=normal(15-25), 2=high(>25)
        result["vix_regime"] = vix_close.apply(
            lambda v: 0.0 if v < 15 else (1.0 if v < 25 else 2.0)
        )

    # SPY RSI 14d
    spy_hist = macro_cache.get("SPY")
    if spy_hist is not None and not spy_hist.empty:
        result["spy_rsi"] = _calc_rsi(spy_hist["Close"])

    # TNX yield — mudança em 5 dias
    tnx_hist = macro_cache.get("^TNX")
    if tnx_hist is None:
        tnx_hist = macro_cache.get("_idx_TNX")
    if tnx_hist is not None and not tnx_hist.empty:
        tnx_close = tnx_hist["Close"]
        result["tnx_5d"] = tnx_close - tnx_close.shift(5)

    # Sector ETFs — retorno 63d (≈3 meses) por sector
    from ml_training.config import SECTOR_ETF as _SE
    for sector, etf in _SE.items():
        if sector == "Unknown":
            continue
        etf_hist = macro_cache.get(etf)
        if etf_hist is not None and not etf_hist.empty:
            etf_close = etf_hist["Close"]
            result[f"sector_{sector}"] = etf_close.pct_change(63)  # retorno 3m

    log.info(f"[macro] Séries pré-calculadas: {list(result.keys())}")
    return result


def find_historical_dips(
    ticker: str,
    hist: pd.DataFrame,
    spy_hist: pd.DataFrame,
    sector: str,
    start_date: str,
    end_date: str,
    precomputed_macro: dict | None = None,  # séries pré-calculadas por data — lookup O(1)
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

        # Macro features PIT — lookup em séries pré-calculadas, tolerante a datas
        def _mlookup(series_key: str) -> float:
            """Lookup seguro: tenta data exacta, depois ±3 dias úteis."""
            if precomputed_macro is None or series_key not in precomputed_macro:
                return float("nan")
            s = precomputed_macro[series_key]
            try:
                # Data exacta
                if dt in s.index:
                    v = s.loc[dt]
                    return float(v) if pd.notna(v) else float("nan")
                # Data mais próxima dentro de 3 dias úteis
                idx_pos = s.index.searchsorted(dt)
                for off in [0, -1, 1, -2, 2, -3, 3]:
                    p = idx_pos + off
                    if 0 <= p < len(s):
                        v = s.iloc[p]
                        if pd.notna(v) and abs((s.index[p] - dt).days) <= 4:
                            return float(v)
            except Exception:
                pass
            return float("nan")

        vix_val        = _mlookup("vix")
        spy_rsi        = _mlookup("spy_rsi")
        vix_percentile = _mlookup("vix_pct")
        vix_regime_val = _mlookup("vix_regime")
        yield_5d_change= _mlookup("tnx_5d")
        etf_ret3m      = _mlookup(f"sector_{sector}")
        sector_relative= round(ret_3m - etf_ret3m, 4) if not np.isnan(etf_ret3m) else float("nan")

        rows.append({
            "ticker":         ticker,
            "alert_date":     dt.date().isoformat(),
            "sector":         sector,
            "price":          round(c_now, 4),
            # Features técnicas PIT
            "drop_pct_today":   round(day_ch, 4),
            "drawdown_52w":     round(drawdown, 4),
            "rsi_14":           round(rsi, 2),
            "return_1m":        round(ret_1m, 4),
            "return_3m_pre":    round(ret_3m, 4),
            "return_6m_pre":    round(ret_6m, 4),
            "bb_width":         round(bb_width, 4),
            "volume_zscore_20d":round(vol_ratio - 1.0, 4),
            # Macro features PIT — NaN se genuinamente indisponíveis
            "vix":              vix_val if not np.isnan(vix_val) else float("nan"),
            "spy_rsi_14":       spy_rsi if not np.isnan(spy_rsi) else float("nan"),
            "vix_percentile_1y":vix_percentile if not np.isnan(vix_percentile) else float("nan"),
            "vix_regime":       vix_regime_val if not np.isnan(vix_regime_val) else float("nan"),
            "sector_relative":  sector_relative if not np.isnan(sector_relative) else float("nan"),
            "yield_10y_change_5d": yield_5d_change if not np.isnan(yield_5d_change) else float("nan"),
            # Targets
            "alpha_90d":        alpha_90d,
            "close_90d":        close_90d,
            "spy_close_90d":    spy_close_90d,
            "max_drawdown_60d": float("nan"),
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

    # Notificações Telegram — definir ANTES de qualquer uso
    _send_fn = None
    try:
        import importlib, os as _os
        if _os.getenv("TELEGRAM_TOKEN") and _os.getenv("TELEGRAM_CHAT_ID"):
            _bot = importlib.import_module("bot_commands")
            _send_fn = getattr(_bot, "_reply", None) or getattr(_bot, "_cb_send_telegram", None)
    except Exception:
        pass

    def _notify(msg: str) -> None:
        log.info(msg)
        if _send_fn:
            try: _send_fn(msg)
            except Exception: pass

    # Garantir que o price cache tem dados desde start_date para todos os tickers.
    _MACRO_TICKERS = ["SPY", "^VIX", "^TNX", "XLK", "XLF", "XLV", "XLY",
                      "XLP", "XLI", "XLE", "XLU", "XLRE", "XLB", "XLC"]
    _ALL_NEEDED = tickers + _MACRO_TICKERS
    _notify(f"⏳ A verificar/actualizar price cache desde {args.start} ({len(_ALL_NEEDED)} séries)...")
    _ensure_price_cache(_ALL_NEEDED, _CACHE_DIR, start=args.start)

    # Carregar SPY e macro após garantir cache actualizado
    spy_hist = _load_ticker("SPY", _CACHE_DIR)
    if spy_hist is None:
        log.error("SPY não disponível. Verifica a rede.")
        return 1

    macro_cache: dict = {}
    for _mt in _MACRO_TICKERS:
        _mh = _load_ticker(_mt, _CACHE_DIR)
        if _mh is not None:
            macro_cache[_mt] = _mh
    log.info(f"Macro cache: {len(macro_cache)} séries históricas carregadas")

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

    # Pré-calcular séries macro uma vez — lookup O(1) no loop em vez de O(n) por linha
    log.info("A pré-calcular séries macro (VIX, SPY RSI, sector ETFs)...")
    precomputed_macro = precompute_macro_series(macro_cache)
    log.info(f"Séries macro prontas: {len(precomputed_macro)} séries")

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
            ticker, hist, spy_hist, sector, args.start, end_date,
            precomputed_macro=precomputed_macro,
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
