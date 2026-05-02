"""
experiments/ml_v2/build_dataset.py

Constrói o dataset de treino para o modelo v2 a partir de:
  - alert_db.csv   (alertas históricos com labels resolvidas)
  - yfinance        (preços OHLCV para features v2 + targets)

Output:
  experiments/ml_v2/dataset_v2.parquet   (X + y_up + y_down + meta)

Anti-leakage garantido:
  - features  : só dados até à data do alerta (price_history[:alert_date])
  - targets   : só dados após a data do alerta (future_prices[alert_date+1:])

Uso:
  python experiments/ml_v2/build_dataset.py
  python experiments/ml_v2/build_dataset.py --db /data/alert_db.csv
  python experiments/ml_v2/build_dataset.py --min-samples 30 --horizon 60
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Adicionar root ao path para importar ml_features
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_features import FEATURE_COLUMNS, add_derived_features, _FALLBACK
from experiments.ml_v2.pipeline import (
    build_targets,
    build_v2_features,
    FEATURE_COLUMNS_V2,
)

logger = logging.getLogger(__name__)

_HERE       = Path(__file__).parent
_REPO_ROOT  = _HERE.parent.parent
_DATA_DIR   = Path("/data") if Path("/data").exists() else _REPO_ROOT
_ALERT_DB   = _DATA_DIR / "alert_db.csv"
_OUT_PATH   = _HERE / "dataset_v2.parquet"

_HORIZON_DAYS  = 60
_HISTORY_DAYS  = 120   # dias de historial para calcular MA50, zscore, etc.
_YF_SLEEP      = 0.5   # segundos entre requests yfinance


# ─────────────────────────────────────────────────────────────────────────────
# 1. Carregar e validar alert_db
# ─────────────────────────────────────────────────────────────────────────────

def load_alert_db(db_path: Path, min_samples: int = 10) -> pd.DataFrame:
    """
    Carrega o alert_db.csv e filtra:
      - só linhas com label_win resolvida (0 ou 1)
      - colunas obrigatórias presentes
    """
    if not db_path.exists():
        raise FileNotFoundError(f"alert_db não encontrado: {db_path}")

    df = pd.read_csv(db_path, dtype=str)
    logger.info(f"load_alert_db: {len(df)} linhas totais em {db_path}")

    # Normalizar coluna de data
    date_col = "date_iso" if "date_iso" in df.columns else "date"
    if date_col not in df.columns:
        raise ValueError("alert_db.csv precisa de coluna 'date_iso' ou 'date'")
    df["alert_date"] = pd.to_datetime(df[date_col], errors="coerce")

    # Normalizar ticker
    ticker_col = "symbol" if "symbol" in df.columns else "ticker"
    if ticker_col not in df.columns:
        raise ValueError("alert_db.csv precisa de coluna 'symbol' ou 'ticker'")
    df["ticker"] = df[ticker_col].str.strip().str.upper()

    # Normalizar preço
    price_col = next((c for c in ["price", "price_alert", "close"] if c in df.columns), None)
    if price_col is None:
        raise ValueError("alert_db.csv precisa de coluna 'price', 'price_alert' ou 'close'")
    df["entry_price"] = pd.to_numeric(df[price_col], errors="coerce")

    # Filtrar linhas com labels resolvidas
    resolved_mask = df["label_win"].isin(["0", "1", 0, 1])
    df = df[resolved_mask].copy()
    df["label_win"] = df["label_win"].astype(int)

    # Remover linhas sem data ou preço
    df = df.dropna(subset=["alert_date", "entry_price", "ticker"])
    df = df[df["entry_price"] > 0]

    logger.info(f"load_alert_db: {len(df)} linhas com labels resolvidas")

    if len(df) < min_samples:
        raise ValueError(
            f"Apenas {len(df)} amostras resolvidas (mínimo: {min_samples}). "
            f"O label_resolver precisa de mais tempo a correr."
        )

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fetch de precos por ticker (batch, 1 request por ticker)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_prices(df: pd.DataFrame, history_days: int = _HISTORY_DAYS) -> dict[str, pd.DataFrame]:
    """
    Faz 1 request yfinance por ticker, cobrindo:
      - history_days antes do alerta mais antigo (para features)
      - horizon_days depois do alerta mais recente (para targets)

    Devolve dict: ticker -> DataFrame OHLCV com DatetimeIndex.
    """
    price_cache: dict[str, pd.DataFrame] = {}

    ticker_groups = df.groupby("ticker")
    n_tickers = df["ticker"].nunique()
    logger.info(f"fetch_all_prices: a fazer fetch de {n_tickers} tickers...")

    for i, (ticker, group) in enumerate(ticker_groups, 1):
        earliest_alert = group["alert_date"].min()
        latest_alert   = group["alert_date"].max()

        fetch_start = (earliest_alert - pd.Timedelta(days=history_days)).strftime("%Y-%m-%d")
        fetch_end   = (latest_alert   + pd.Timedelta(days=_HORIZON_DAYS + 5)).strftime("%Y-%m-%d")

        logger.info(f"  [{i}/{n_tickers}] {ticker}: {fetch_start} → {fetch_end}")
        try:
            ohlcv = yf.download(
                ticker,
                start=fetch_start,
                end=fetch_end,
                progress=False,
                auto_adjust=True,
            )
            if ohlcv.empty:
                logger.warning(f"  {ticker}: sem dados yfinance")
            else:
                price_cache[ticker] = ohlcv
                logger.info(f"  {ticker}: {len(ohlcv)} dias")
        except Exception as e:
            logger.warning(f"  {ticker}: erro yfinance — {e}")

        time.sleep(_YF_SLEEP)

    return price_cache


# ─────────────────────────────────────────────────────────────────────────────
# 3. Construir dataset row-by-row
# ─────────────────────────────────────────────────────────────────────────────

def _get_v1_features(row: pd.Series) -> dict:
    """
    Extrai as features v1 (FEATURE_COLUMNS) da linha do alert_db.
    Usa os valores gravados no momento do alerta — point-in-time.
    Fallback para _FALLBACK se a coluna não existir.
    """
    fv = {}
    for col in FEATURE_COLUMNS:
        if col in row.index:
            try:
                fv[col] = float(row[col])
            except (TypeError, ValueError):
                fv[col] = _FALLBACK.get(col, 0.0)
        else:
            fv[col] = _FALLBACK.get(col, 0.0)
    # Recompute engineered features from base to ensure consistency
    add_derived_features(fv)
    return fv


def build_dataset(
    df_alerts: pd.DataFrame,
    price_cache: dict[str, pd.DataFrame],
    horizon_days: int = _HORIZON_DAYS,
    history_days: int = _HISTORY_DAYS,
) -> pd.DataFrame:
    """
    Constrói o dataset completo:
      - Para cada alerta: extrai features v1 (do CSV) + features v2 (dos preços)
        e calcula targets v2 (max_return_60d, max_drawdown_60d)

    Devolve DataFrame com colunas:
      ticker, alert_date, entry_price,
      [FEATURE_COLUMNS_V2],
      max_return_60d, max_drawdown_60d, label_win (para referencia)
    """
    rows = []
    skipped = 0

    for _, row in df_alerts.iterrows():
        ticker      = row["ticker"]
        alert_date  = pd.Timestamp(row["alert_date"])
        entry_price = float(row["entry_price"])

        ohlcv = price_cache.get(ticker)
        if ohlcv is None:
            logger.debug(f"build_dataset: sem preços para {ticker} — a saltar")
            skipped += 1
            continue

        # Normalizar index para DatetimeIndex sem timezone
        if ohlcv.index.tz is not None:
            ohlcv = ohlcv.copy()
            ohlcv.index = ohlcv.index.tz_localize(None)

        # — Features v1: valores gravados no alerta (point-in-time) —
        fv1 = _get_v1_features(row)

        # — Features v2: só preços ATÉ alert_date (anti-leakage) —
        hist = ohlcv[ohlcv.index <= alert_date].copy()
        if len(hist) < 20:
            logger.debug(f"build_dataset: historial insuficiente para {ticker} @ {alert_date.date()}")
            skipped += 1
            continue

        fv2 = build_v2_features(row, hist)

        # — Targets: só preços APÓS alert_date (anti-leakage) —
        future = ohlcv[
            (ohlcv.index > alert_date) &
            (ohlcv.index <= alert_date + pd.Timedelta(days=horizon_days))
        ]["Close"]

        if len(future) < 5:
            logger.debug(f"build_dataset: janela futura insuficiente para {ticker} @ {alert_date.date()}")
            skipped += 1
            continue

        targets = build_targets(
            alert_date=alert_date,
            entry_price=entry_price,
            future_prices=future,
            horizon_days=horizon_days,
        )

        if np.isnan(targets["max_return_60d"]) or np.isnan(targets["max_drawdown_60d"]):
            skipped += 1
            continue

        # Montar linha completa
        record = {
            "ticker":       ticker,
            "alert_date":   alert_date,
            "entry_price":  entry_price,
            "label_win":    int(row["label_win"]),
            **fv1,
            **fv2,
            "max_return_60d":   targets["max_return_60d"],
            "max_drawdown_60d": targets["max_drawdown_60d"],
            "min_idx_60d":      targets["min_idx"],
        }
        rows.append(record)

    logger.info(
        f"build_dataset: {len(rows)} amostras construídas, {skipped} saltadas"
    )

    if not rows:
        raise ValueError("Dataset vazio — verifica os preços e labels.")

    df_out = pd.DataFrame(rows)

    # Garantir colunas na ordem correcta
    meta_cols    = ["ticker", "alert_date", "entry_price", "label_win"]
    target_cols  = ["max_return_60d", "max_drawdown_60d", "min_idx_60d"]
    feature_cols = [c for c in FEATURE_COLUMNS_V2 if c in df_out.columns]
    missing      = [c for c in FEATURE_COLUMNS_V2 if c not in df_out.columns]

    if missing:
        logger.warning(f"build_dataset: features em falta (serão NaN): {missing}")
        for c in missing:
            df_out[c] = 0.0

    df_out = df_out[meta_cols + feature_cols + target_cols]
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main(db_path: Path, min_samples: int, out_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print(f"\n{'='*55}")
    print("  DipRadar v2 — Build Dataset")
    print(f"{'='*55}")
    print(f"  DB      : {db_path}")
    print(f"  Output  : {out_path}")
    print(f"  Horizon : {_HORIZON_DAYS}d")
    print()

    # 1. Carregar alertas
    df_alerts = load_alert_db(db_path, min_samples=min_samples)
    print(f"  Alertas resolvidos: {len(df_alerts)}")
    print(f"  Tickers únicos   : {df_alerts['ticker'].nunique()}")
    print(f"  Janela temporal  : {df_alerts['alert_date'].min().date()} → {df_alerts['alert_date'].max().date()}")
    print()

    # 2. Fetch de preços
    print("  A fazer fetch de preços (yfinance)...")
    price_cache = fetch_all_prices(df_alerts)
    print(f"  Tickers com preços: {len(price_cache)}/{df_alerts['ticker'].nunique()}")
    print()

    # 3. Construir dataset
    print("  A construir dataset...")
    df_out = build_dataset(df_alerts, price_cache)

    # 4. Guardar
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)

    print(f"\n{'='*55}")
    print(f"  Dataset guardado: {out_path}")
    print(f"  Shape           : {df_out.shape}")
    print(f"  Colunas         : {list(df_out.columns)}")
    print()
    print("  Distribuição dos targets:")
    print(f"    max_return_60d   : mean={df_out['max_return_60d'].mean():.3f}  std={df_out['max_return_60d'].std():.3f}")
    print(f"    max_drawdown_60d : mean={df_out['max_drawdown_60d'].mean():.3f}  std={df_out['max_drawdown_60d'].std():.3f}")
    print(f"    label_win rate   : {df_out['label_win'].mean():.1%}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DipRadar v2 — Build Dataset")
    parser.add_argument("--db",          default=str(_ALERT_DB),  help="Caminho para alert_db.csv")
    parser.add_argument("--out",         default=str(_OUT_PATH),  help="Ficheiro parquet de output")
    parser.add_argument("--min-samples", default=10, type=int,    help="Número mínimo de amostras resolvidas")
    args = parser.parse_args()
    main(Path(args.db), args.min_samples, Path(args.out))
