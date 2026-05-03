"""
bootstrap_ml.py — Merge helper para retreino mensal do XGB-v2.

Junta hist_backtest.csv (dados históricos sintéticos) com alert_db.csv
(alertas live de produção) num DataFrame limpo pronto para treino.

Garantias:
  1. Filtro estrito de FEATURE_COLUMNS em ambos os DataFrames antes do
     concat — evita colunas NaN por schema mismatch.
  2. Dedup por (symbol, semana ISO) — evita peso duplo em dips continuados
     (e.g. dois alertas do mesmo ticker com 2 dias de diferença).
  3. Trava de maturidade: só entram linhas com outcome_label preenchido
     E return_60d disponível (alinhado com max_return_60d do treino).

Uso:
    from bootstrap_ml import load_merged_dataset
    df = load_merged_dataset()
    # df tem FEATURE_COLUMNS + ['target', 'source', 'symbol', 'date_iso']

Target:
    'target' = return_60d (float, % retorno em 60d) — alinhado com
    max_return_60d usado para treinar o champion XGB-v2.

Para retreino com alpha (futuro):
    df['target'] = df['return_60d'] - df['spy_return_60d']
    (descomentar quando spy_return_60d tiver cobertura suficiente)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
_DATA_DIR    = Path("/data") if Path("/data").exists() else Path("/tmp")
_REPO_DIR    = Path(__file__).parent

_HIST_PATH   = next(
    (p for p in [_DATA_DIR / "hist_backtest.csv", _REPO_DIR / "hist_backtest.csv"] if p.exists()),
    _REPO_DIR / "hist_backtest.csv",
)
_ALERT_PATH  = _DATA_DIR / "alert_db.csv"

# ── Feature columns canónicas (17 base do XGB-v2 champion) ────────────────────
# Devem coincidir com ml_predictor._FEATURE_COLS.
# Usado como filtro rigoroso antes do concat para evitar NaN columns.
FEATURE_COLUMNS: list[str] = [
    "macro_score",
    "vix",
    "spy_drawdown_5d",
    "sector_drawdown_5d",
    "fcf_yield",
    "revenue_growth",
    "gross_margin",
    "de_ratio",
    "pe_vs_fair",
    "analyst_upside",
    "quality_score",
    "drop_pct_today",
    "drawdown_from_high",  # nome canónico (alert_db já usa este nome)
    "rsi_14",
    "atr_ratio",
    "volume_spike",
    "market_cap_b",
]

# Colunas de metadados a preservar (não entram no modelo, mas úteis para debug)
_META_COLS = ["symbol", "date_iso", "source"]

# Mapeamento de nomes do hist_backtest -> nomes canónicos do alert_db
# Apenas para colunas que divergem entre os dois CSVs.
_HIST_COL_MAP: dict[str, str] = {
    "rsi":              "rsi_14",
    "rsi_14":           "rsi_14",
    "drawdown_52w":     "drawdown_from_high",
    "drawdown_pct":     "drawdown_from_high",
    "drop_pct":         "drop_pct_today",
    "change_day_pct":   "drop_pct_today",
    "spy_change":       "spy_drawdown_5d",
    "sector_etf_change": "sector_drawdown_5d",
    "atr_pct":          "atr_ratio",
    "volume_ratio":     "volume_spike",
    "debt_equity":      "de_ratio",
    "market_cap":       "market_cap_b",
}

# Mapeamento de nomes do alert_db -> nomes canónicos (complementar)
_ALERT_COL_MAP: dict[str, str] = {
    "rsi":              "rsi_14",
    "volume_ratio":     "volume_spike",
    "debt_equity":      "de_ratio",
    "change_day_pct":   "drop_pct_today",
    "spy_change":       "spy_drawdown_5d",
    "sector_etf_change": "sector_drawdown_5d",
}


def _normalise_df(df: pd.DataFrame, col_map: dict[str, str]) -> pd.DataFrame:
    """Renomeia colunas e aplica filtro estrito para FEATURE_COLUMNS."""
    df = df.rename(columns=col_map)
    # Adiciona colunas em falta com NaN (serão imputadas pelo modelo)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def _add_week_key(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna 'week_key' = (symbol, ano-semanaISO) para dedup."""
    dates = pd.to_datetime(df["date_iso"], errors="coerce")
    df["week_key"] = (
        df["symbol"].astype(str)
        + "_"
        + dates.dt.isocalendar().year.astype(str)
        + "W"
        + dates.dt.isocalendar().week.astype(str).str.zfill(2)
    )
    return df


def load_hist_backtest(path: Path | None = None) -> pd.DataFrame:
    """
    Carrega hist_backtest.csv e normaliza para FEATURE_COLUMNS.
    Retorna DataFrame vazio se o ficheiro não existir.
    """
    p = path or _HIST_PATH
    if not p.exists():
        logging.warning(f"[bootstrap_ml] hist_backtest não encontrado: {p}")
        return pd.DataFrame()

    df = pd.read_csv(p, low_memory=False)
    df["source"] = "hist"
    df = _normalise_df(df, _HIST_COL_MAP)

    # Target: max_return_60d (nome do hist_backtest) -> target
    if "max_return_60d" in df.columns and "target" not in df.columns:
        df["target"] = pd.to_numeric(df["max_return_60d"], errors="coerce")
    elif "target" not in df.columns:
        df["target"] = np.nan

    logging.info(f"[bootstrap_ml] hist_backtest: {len(df)} linhas")
    return df


def load_alert_db(path: Path | None = None) -> pd.DataFrame:
    """
    Carrega alert_db.csv, aplica trava de maturidade (return_60d preenchido
    e outcome_label não vazio) e normaliza para FEATURE_COLUMNS.

    Retorna DataFrame vazio se o ficheiro não existir ou não tiver amostras
    maduras.
    """
    p = path or _ALERT_PATH
    if not p.exists():
        logging.info(f"[bootstrap_ml] alert_db não encontrado: {p}")
        return pd.DataFrame()

    df = pd.read_csv(p, low_memory=False)

    # Trava de maturidade: só amostras com return_60d e outcome_label
    before = len(df)
    df = df[
        df["return_60d"].notna()
        & (df["return_60d"] != "")
        & df["outcome_label"].notna()
        & (df["outcome_label"] != "")
    ].copy()
    mature = len(df)
    logging.info(
        f"[bootstrap_ml] alert_db: {before} total, {mature} maduros (return_60d + label)"
    )

    if df.empty:
        return pd.DataFrame()

    df["source"] = "live"
    df = _normalise_df(df, _ALERT_COL_MAP)

    # Target: return_60d -> target
    df["target"] = pd.to_numeric(df["return_60d"], errors="coerce")

    return df


def load_merged_dataset(
    hist_path: Path | None = None,
    alert_path: Path | None = None,
    dedup: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Merge de hist_backtest + alert_db num DataFrame pronto para treino.

    Passos:
      1. Carrega e normaliza ambos os DataFrames.
      2. Filtra estritamente para FEATURE_COLUMNS + target + meta.
      3. Concat (alert_db sobrepõe hist para a mesma semana/ticker).
      4. Dedup por (symbol, semana ISO) — mantém a linha live se existir.
      5. Drop de linhas com target NaN.

    Returns:
        pd.DataFrame com colunas FEATURE_COLUMNS + ['target', 'source',
        'symbol', 'date_iso']. Pronto para passar ao XGBoost.
    """
    df_hist  = load_hist_backtest(hist_path)
    df_live  = load_alert_db(alert_path)

    frames = []
    if not df_hist.empty:
        frames.append(df_hist)
    if not df_live.empty:
        frames.append(df_live)

    if not frames:
        logging.error("[bootstrap_ml] Nenhum dado disponível para treino.")
        return pd.DataFrame()

    # Filtro estrito: só FEATURE_COLUMNS + target + meta em ambos os frames
    # ANTES do concat — garante schema idêntico e zero colunas NaN espúrias.
    keep_cols = FEATURE_COLUMNS + ["target"] + _META_COLS
    clean_frames = []
    for fr in frames:
        # Adiciona meta cols em falta
        for col in _META_COLS:
            if col not in fr.columns:
                fr[col] = ""
        # Garante que só passam as colunas esperadas
        available = [c for c in keep_cols if c in fr.columns]
        clean_frames.append(fr[available].copy())

    df = pd.concat(clean_frames, ignore_index=True)

    # Conversão numérica das features
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Dedup por (symbol, semana ISO) — previne peso duplo de dips continuados.
    # Mantém a linha 'live' em caso de colisão (sort: live > hist).
    if dedup and "symbol" in df.columns and "date_iso" in df.columns:
        df = _add_week_key(df)
        # sort: live primeiro (source == 'live' ordena antes de 'hist' com ascending=False)
        df = df.sort_values("source", ascending=False)  # live > hist
        before_dedup = len(df)
        df = df.drop_duplicates(subset="week_key", keep="first")
        df = df.drop(columns=["week_key"])
        after_dedup = len(df)
        if verbose:
            logging.info(
                f"[bootstrap_ml] Dedup: {before_dedup} -> {after_dedup} "
                f"({before_dedup - after_dedup} duplicados removidos)"
            )

    # Drop de linhas sem target
    before_target = len(df)
    df = df.dropna(subset=["target"])
    if verbose and len(df) < before_target:
        logging.info(
            f"[bootstrap_ml] Drop NaN target: {before_target} -> {len(df)}"
        )

    if verbose:
        n_hist = (df["source"] == "hist").sum() if "source" in df.columns else 0
        n_live = (df["source"] == "live").sum() if "source" in df.columns else 0
        logging.info(
            f"[bootstrap_ml] Dataset final: {len(df)} linhas "
            f"(hist={n_hist}, live={n_live})"
        )

    return df
