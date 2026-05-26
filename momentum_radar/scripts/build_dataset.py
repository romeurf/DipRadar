"""
build_momentum_dataset.py — Constrói dataset de treino para o Breakout ML.

Diferente do DipRadar (que usa alertas de dip como ponto de entrada),
o Breakout ML aprende a partir de dias em que um stock estava em momentum:
  - return_20d >= 10%  (breakout confirmado)
  - volume acima da média
  - RSI em zona de momentum (50-78)

Targets: alpha_30d / alpha_90d / alpha_180d — excess return over SPY per horizon.
Training on alpha removes market regime bias (raw returns conflate "market is up" with signal).
Entry criteria are deliberately wide: return_20d, volume_ratio_20d and rsi_14 are all model
features — the model learns optimal thresholds from the full gradient in training data.

Uso:
  python scripts/build_momentum_dataset.py
  python scripts/build_momentum_dataset.py --tickers AAPL,MSFT,NVDA  # teste rápido
  python scripts/build_momentum_dataset.py --start 2020-01-01

Output: /data/momentum_training.parquet
  Colunas: date, ticker, sector, [features], forward_return_30d, alpha_30d
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/ → momentum_radar/ → repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("momentum_dataset")

_DATA_DIR   = Path("/data") if Path("/data").exists() else Path("/tmp")
_CACHE_DIR  = _DATA_DIR / "price_cache"
_OUTPUT     = Path(__file__).parent.parent / "momentum_training.parquet"

# Entry criteria — deliberately wide so the model learns optimal thresholds from data.
# return_20d, volume_ratio_20d and rsi_14 are all model features: the model sees the
# full gradient (10% vs 15% vs 20% return) and discovers optimal boundaries itself.
# Pre-filtering to match the scanner would remove that gradient and reduce training data.
_MIN_RETURN_20D   = 0.10    # +10% in 20 days — minimum to identify a plausible breakout
_MIN_VOLUME_RATIO = 1.20    # 20% above average — confirms some buying pressure
_RSI_MIN          = 40.0    # liberal lower bound — model learns early vs confirmed momentum
_RSI_MAX          = 85.0    # extended upper bound — model learns overbought risk
_MIN_MARKET_CAP_B = 2.0     # minimum liquidity

# Episode exit threshold (hysteresis): below this the episode ends.
# Half of entry threshold to avoid re-triggering on short corrections.
_EPISODE_EXIT_RETURN = 0.05  # return_20d < 5% → episode ended (half of 10% entry)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de preços
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_history(ticker: str, start: str = "1993-01-01") -> Optional[pd.DataFrame]:
    cache = _CACHE_DIR / f"{ticker.replace('^','_idx_').replace('/','_').replace('.','_')}.parquet"
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 86400 * 7:
        try:
            df = pd.read_parquet(cache)
            df.index = pd.to_datetime(df.index)
            # Validar que o cache cobre a data pedida (com 30d de tolerância).
            # Se não cobre (e.g. cache antigo de 2019, pedido é 1993), invalidar.
            if not df.empty and df.index[0] <= pd.Timestamp(start) + pd.Timedelta(days=30):
                return df
            cache.unlink(missing_ok=True)  # cache desactualizado — re-fetch
        except Exception:
            pass
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
        if df is None or df.empty or len(df) < 30:
            return None
        idx = pd.DatetimeIndex(df.index)
        df.index = idx.tz_convert(None) if idx.tz is not None else idx
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        df.to_parquet(cache)
        return df
    except Exception as e:
        log.debug(f"  {ticker}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering para cada dia de momentum
# ─────────────────────────────────────────────────────────────────────────────

def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _build_momentum_features(hist: pd.DataFrame, idx: int) -> Optional[dict]:
    """Computa features de momentum para o dia hist.index[idx]."""
    if idx < 25:
        return None

    close  = hist["Close"]
    volume = hist["Volume"]
    high   = hist["High"]
    low    = hist["Low"]

    close_slice  = close.iloc[:idx+1]
    volume_slice = volume.iloc[:idx+1]

    # Retornos
    ret_20d = float(close.iloc[idx] / close.iloc[max(0, idx-20)] - 1) if idx >= 20 else None
    ret_5d  = float(close.iloc[idx] / close.iloc[max(0, idx-5)]  - 1) if idx >= 5  else None
    ret_60d = float(close.iloc[idx] / close.iloc[max(0, idx-60)] - 1) if idx >= 60 else None

    if ret_20d is None or ret_20d < _MIN_RETURN_20D:
        return None

    # Volume ratio
    vol_avg = float(volume_slice.iloc[-20:].mean()) if len(volume_slice) >= 20 else None
    vol_ratio = float(volume.iloc[idx] / vol_avg) if vol_avg and vol_avg > 0 else 1.0

    if vol_ratio < _MIN_VOLUME_RATIO:
        return None

    # RSI
    rsi_series = _compute_rsi(close_slice)
    rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

    if not (_RSI_MIN <= rsi <= _RSI_MAX):
        return None

    # Distância do máximo de 52 semanas
    high_52w = float(close_slice.iloc[-252:].max()) if len(close_slice) >= 20 else float(close_slice.max())
    pct_from_high = float(close.iloc[idx] / high_52w - 1)

    # ATR 14d
    h = high.iloc[max(0,idx-14):idx+1]
    l = low.iloc[max(0,idx-14):idx+1]
    c_prev = close.iloc[max(0,idx-15):idx].values
    if len(c_prev) > 0 and len(h) == len(l):
        tr_hl = h.values - l.values
        tr_hc = np.abs(h.values[1:] - c_prev[-len(h)+1:]) if len(c_prev) >= len(h)-1 else tr_hl[1:]
        tr_lc = np.abs(l.values[1:] - c_prev[-len(l)+1:]) if len(c_prev) >= len(l)-1 else tr_hl[1:]
        atr_pct = float(np.mean(tr_hl)) / float(close.iloc[idx]) if close.iloc[idx] > 0 else 0.02
    else:
        atr_pct = 0.02

    # Close in range 20d (fechamentos perto dos máximos = força)
    if len(high.iloc[max(0,idx-20):idx+1]) >= 5:
        h20 = high.iloc[max(0,idx-20):idx+1].values
        l20 = low.iloc[max(0,idx-20):idx+1].values
        c20 = close.iloc[max(0,idx-20):idx+1].values
        ranges = h20 - l20
        close_in_range = float(np.mean((c20 - l20) / np.where(ranges > 0, ranges, 1)))
    else:
        close_in_range = 0.5

    # Market Structure: VWAP ancorado no mínimo e Volume Profile POC
    hist_slice = hist.iloc[:idx + 1]
    try:
        from ml_features import _anchored_vwap_ratio, _volume_poc_ratio, _compute_mfi, _compute_volume_up_pct
        vwap_low_ratio = _anchored_vwap_ratio(hist_slice, use_high=False)
        poc_ratio      = _volume_poc_ratio(hist_slice, lookback=60)
        mfi_val        = _compute_mfi(hist_slice, period=14)
        vol_up_pct     = _compute_volume_up_pct(hist_slice, lookback=20)
    except Exception:
        vwap_low_ratio = 0.0
        poc_ratio      = 0.0
        mfi_val        = 50.0
        vol_up_pct     = 0.50

    return {
        "return_20d":          round(ret_20d, 4),
        "return_5d":           round(ret_5d or 0, 4),
        "return_60d_pre":      round(ret_60d or 0, 4),
        "volume_ratio_20d":    round(vol_ratio, 3),
        "rsi_14":              round(rsi, 2),
        "pct_from_52w_high":   round(pct_from_high, 4),
        "atr_pct":             round(atr_pct, 4),
        "close_in_range_20d":  round(close_in_range, 4),
        "vwap_from_low_ratio": round(vwap_low_ratio, 4),
        "volume_poc_ratio":    round(poc_ratio, 4),
        "mfi_14":              round(mfi_val, 2),
        "volume_up_pct_20d":   round(vol_up_pct, 4),
        "price":               round(float(close.iloc[idx]), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    tickers: list[str],
    spy_hist: pd.DataFrame,
    start: str = "1993-01-01",
) -> pd.DataFrame:
    """Para cada ticker, encontra dias com momentum e calcula forward_return_30d.

    Detecção de episódio por ticker: percorre cada dia de trading e amostra apenas
    o primeiro dia qualificado de cada episódio de momentum. Um episódio termina
    quando return_20d cai abaixo de _EPISODE_EXIT_RETURN — sem cooldown fixo,
    adapta-se automaticamente à duração real de cada rally.
    """
    rows = []
    n_tickers = len(tickers)

    for i, ticker in enumerate(tickers):
        hist = _fetch_history(ticker, start=start)
        if hist is None or len(hist) < 60:
            continue

        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info or {}
            sector = info.get("sector", "Unknown") or "Unknown"
            mktcap = float(info.get("marketCap") or 0) / 1e9
            if mktcap > 0 and mktcap < _MIN_MARKET_CAP_B:
                continue
        except Exception:
            sector = "Unknown"

        close = hist["Close"]
        n = len(hist)

        # Detecção de episódio: só se amostra o PRIMEIRO dia qualificado de cada episódio.
        # O episódio termina quando return_20d cai abaixo de _EPISODE_EXIT_RETURN (histerese).
        # Sem cooldown fixo — a duração adapta-se ao stock e ao contexto de mercado.
        in_episode = False

        for idx in range(25, n - 185):
            # Verificar saída de episódio com histerese
            ret_20d_raw = float(close.iloc[idx] / close.iloc[idx - 20] - 1)
            if in_episode and ret_20d_raw < _EPISODE_EXIT_RETURN:
                in_episode = False

            if in_episode:
                continue  # ainda no mesmo episódio — não duplicar

            feats = _build_momentum_features(hist, idx)
            if feats is None:
                continue

            # Novo episódio detectado — marcar e amostrar
            in_episode = True

            # Forward returns multi-horizon (L1=30d, L2=90d, L3=180d)
            future_30  = close.iloc[idx + 30]  if idx + 30  < n else None
            future_90  = close.iloc[idx + 90]  if idx + 90  < n else None
            future_180 = close.iloc[idx + 180] if idx + 180 < n else None

            if future_30 is None:
                continue
            fwd_30 = float(future_30 / close.iloc[idx] - 1)
            if not np.isfinite(fwd_30) or abs(fwd_30) > 2.0:
                continue

            fwd_90  = float(future_90  / close.iloc[idx] - 1) if future_90  is not None else None
            fwd_180 = float(future_180 / close.iloc[idx] - 1) if future_180 is not None else None
            if fwd_90  is not None and (not np.isfinite(fwd_90)  or abs(fwd_90)  > 3.0): fwd_90  = None
            if fwd_180 is not None and (not np.isfinite(fwd_180) or abs(fwd_180) > 5.0): fwd_180 = None

            # Alpha vs SPY por horizonte
            alert_date = hist.index[idx]

            def _spy_alpha(horizon_days: int, stock_ret: Optional[float]) -> Optional[float]:
                # Log-alpha for consistency with DIP and Core models.
                # Arithmetic differences diverge from log differences at long horizons
                # (e.g. 180d: stock +50%/SPY +10% gives arith=0.40, log=0.31 — a 30% gap).
                if stock_ret is None or stock_ret <= -1.0:
                    return None
                try:
                    spy_at  = spy_hist[spy_hist.index <= alert_date]["Close"]
                    spy_fwd = spy_hist[
                        (spy_hist.index > alert_date) &
                        (spy_hist.index <= alert_date + pd.Timedelta(days=horizon_days + 5))
                    ]["Close"]
                    if not spy_at.empty and not spy_fwd.empty:
                        spy_ret = float(spy_fwd.iloc[-1] / spy_at.iloc[-1] - 1)
                        if spy_ret <= -1.0:
                            return None
                        return round(math.log1p(stock_ret) - math.log1p(spy_ret), 4)
                except Exception:
                    pass
                return None

            rows.append({
                "ticker":              ticker,
                "date":                alert_date.date().isoformat(),
                "sector":              sector,
                **feats,
                "forward_return_30d":  round(fwd_30, 4),
                "forward_return_90d":  round(fwd_90, 4)  if fwd_90  is not None else None,
                "forward_return_180d": round(fwd_180, 4) if fwd_180 is not None else None,
                "alpha_30d":           _spy_alpha(30,  fwd_30),
                "alpha_90d":           _spy_alpha(90,  fwd_90),
                "alpha_180d":          _spy_alpha(180, fwd_180),
            })

        if (i + 1) % 50 == 0 or (i + 1) == n_tickers:
            log.info(f"  [{i+1}/{n_tickers}] {ticker} | {len(rows)} amostras momentum até agora")
        time.sleep(0.05)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Constrói dataset de treino para Breakout ML")
    p.add_argument("--tickers",       help="Lista de tickers separados por vírgula (teste rápido)")
    p.add_argument("--start",         default="1993-01-01", help="Data de início (YYYY-MM-DD). Default: 1993")
    p.add_argument("--out",           default=str(_OUTPUT),  help="Caminho de output do parquet")
    p.add_argument("--force-rebuild", action="store_true", help="Ignorar parquet existente e reconstruir desde --start")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
    )

    # Notificações Telegram
    _send_fn = None
    try:
        import importlib, os
        if os.getenv("TELEGRAM_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
            _bc = importlib.import_module("bot_commands")
            _send_fn = getattr(_bc, "_reply", None) or getattr(_bc, "_cb_send_telegram", None)
    except Exception:
        pass

    def _notify(msg: str) -> None:
        log.info(msg)
        if _send_fn:
            try: _send_fn(msg)
            except Exception: pass

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
        log.info(f"Modo teste: {len(tickers)} tickers")
    else:
        from universe import get_ml_universe
        tickers = get_ml_universe()
        log.info(f"Universo completo: {len(tickers)} tickers")

    log.info("A descarregar SPY como benchmark...")
    spy_hist = _fetch_history("SPY", start=args.start)
    if spy_hist is None:
        log.error("SPY não disponível — abortar")
        return 1

    out_path = Path(args.out)

    # Modo incremental: nunca apagar dados históricos.
    # Se o parquet já existe, só computar datas novas (após o último registo).
    existing_df = None
    incremental_start = args.start
    if args.force_rebuild and out_path.exists():
        log.info(f"--force-rebuild: a ignorar parquet existente em {out_path}")
    if out_path.exists() and not args.tickers and not args.force_rebuild:  # modo teste não é incremental
        try:
            existing_df = pd.read_parquet(out_path)
            if "date" in existing_df.columns and not existing_df.empty:
                last_date = existing_df["date"].max()
                # Recuar 220 dias (180d janela de targets + 40d buffer de sobreposição).
                # Com a janela de n-185 barras, precisamos de pelo menos 185 dias à frente
                # de cada ponto amostrado — este lookback garante capturar novas entradas.
                cutoff = (pd.Timestamp(last_date) - pd.Timedelta(days=220)).strftime("%Y-%m-%d")
                incremental_start = cutoff
                log.info(f"Dataset existente: {len(existing_df)} amostras até {last_date}")
                log.info(f"Modo incremental: a computar desde {incremental_start} (220d lookback para janela 180d)")
        except Exception as e:
            log.warning(f"Parquet existente ilegível ({e}) — a reconstruir de raiz")
            existing_df = None

    _notify(f"🚀 *Bootstrap Momentum {args.start[:4]}–hoje* iniciado — {len(tickers)} tickers")
    log.info(f"A construir dataset de momentum desde {incremental_start}...")
    new_df = build_dataset(tickers, spy_hist, start=incremental_start)

    if new_df.empty:
        if existing_df is not None and not existing_df.empty:
            log.info("Nenhum dado novo — dataset existente mantido sem alterações")
            return 0
        log.error("Dataset vazio — nenhum dia de momentum encontrado")
        return 1

    # Merge: dados existentes + novos, dedup por (ticker, date)
    # Colunas que podem estar ausentes em parquets de versões anteriores
    _BACKFILL_COLS = [
        "forward_return_90d", "forward_return_180d",
        "alpha_90d", "alpha_180d",
        "vwap_from_low_ratio", "volume_poc_ratio",
        "mfi_14", "volume_up_pct_20d",
    ]
    if existing_df is not None and not existing_df.empty:
        for col in _BACKFILL_COLS:
            if col not in existing_df.columns:
                existing_df[col] = np.nan
        for col in _BACKFILL_COLS:
            if col not in new_df.columns:
                new_df[col] = np.nan
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
        combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
        n_new = len(combined) - len(existing_df)
        log.info(f"Merge: {len(existing_df)} existentes + {len(new_df)} novos → {len(combined)} total ({n_new} adicionados)")
    else:
        combined = new_df
        log.info(f"Dataset novo: {len(combined)} amostras")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)

    log.info(f"Dataset: {len(combined)} amostras | {combined['ticker'].nunique()} tickers | {combined['sector'].nunique()} sectores")
    for col, label in [("forward_return_30d", "30d"), ("forward_return_90d", "90d"), ("forward_return_180d", "180d")]:
        if col in combined.columns:
            v = combined[col].dropna()
            log.info(f"Forward return {label}: mean={v.mean():.2%} std={v.std():.2%} ({len(v)} amostras)")

    # Per-column stats for Momentum model features
    from momentum_radar.trainer import FEATURE_COLS as _MOM_FEAT
    log.info(f"[stats] momentum feature summary ({len(_MOM_FEAT)} features):")
    for col in _MOM_FEAT:
        if col not in combined.columns:
            log.info(f"  {col:30s} — MISSING")
            continue
        s = combined[col].dropna()
        if len(s) == 0:
            log.info(f"  {col:30s} — NO DATA")
        else:
            log.info(
                f"  {col:30s} n={len(s):>6} mean={s.mean():>9.4f} "
                f"std={s.std():>8.4f} nan={combined[col].isna().sum()}"
            )
    size_mb = out_path.stat().st_size / 1e6
    log.info(f"Guardado em: {out_path} ({size_mb:.1f}MB)")
    _notify(
        f"✅ *Bootstrap Momentum concluído!*\n"
        f"  {len(combined):,} amostras | {combined['ticker'].nunique()} tickers\n"
        f"  {combined['date'].min()} → {combined['date'].max()}\n"
        f"  L1 30d: {combined['forward_return_30d'].mean():.1%} | "
        f"L2 90d: {combined['forward_return_90d'].dropna().mean():.1%} | "
        f"L3 180d: {combined['forward_return_180d'].dropna().mean():.1%}\n"
        f"  {size_mb:.1f}MB\n\n"
        f"_A iniciar treino do modelo de momentum..._"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
