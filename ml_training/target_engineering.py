"""
target_engineering.py — Neutralização sectorial do target + exits realistas.

Duas responsabilidades:
  1. neutralize_target_by_sector  — remove o efeito de sector da label,
     aplicada APENAS ao train set dentro do loop walk-forward.
  2. compute_realistic_target     — calcula o retorno até ao exit real
     (RSI recovery, trailing stop, etc.) em vez do dia 60 fixo.

Uso no loop walk-forward:
  from ml_training.target_engineering import neutralize_target_by_sector

  for fold_idx, (train_idx, test_idx) in enumerate(wf_splits):
      train = neutralize_target_by_sector(
          df.iloc[train_idx], target_col='alpha_60d_rank'
      )
      y_train = train['alpha_60d_rank_neutral_rank']
      # test set nunca é tocado — usa rank original para avaliação
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Neutralização sectorial
# ---------------------------------------------------------------------------

def neutralize_target_by_sector(
    df: pd.DataFrame,
    target_col: str,
    sector_col: str = "sector",
    date_col: str = "alert_date",
) -> pd.DataFrame:
    """
    Subtrai a mediana do sector em cada cross-section de datas.

    Evita que o modelo aprenda "compra tech" em vez de
    "compra dip real dentro do sector".

    Aplica DENTRO de cada fold do walk-forward no train set apenas.
    O test set usa sempre o rank original para avaliação.

    Retorna um novo DataFrame com duas colunas adicionadas:
      {target_col}_neutral        — valor residual (target - mediana sector)
      {target_col}_neutral_rank   — rank percentil [0, 1] do residual
    """
    df = df.copy()

    # Mediana por (data, sector)
    df["_sector_median"] = (
        df.groupby([date_col, sector_col])[target_col]
        .transform("median")
    )

    # Fallback: se sector for raro (< 3 empresas na data), usa mediana global da data
    global_median = df.groupby(date_col)[target_col].transform("median")
    n_per_group = df.groupby([date_col, sector_col])[target_col].transform("count")
    df["_sector_median"] = np.where(
        n_per_group >= 3,
        df["_sector_median"],
        global_median,
    )

    df[f"{target_col}_neutral"] = df[target_col] - df["_sector_median"]

    # Re-rank após neutralização (mantém distribuição uniforme [0, 1])
    df[f"{target_col}_neutral_rank"] = (
        df.groupby(date_col)[f"{target_col}_neutral"]
        .rank(pct=True)
    )

    df.drop(columns=["_sector_median"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# 2. Target com exit realista
# ---------------------------------------------------------------------------

def compute_realistic_target(
    df: pd.DataFrame,
    price_col: str = "close",
    rsi_col: str = "rsi_14",
    exit_strategy: str = "rsi_recovery",
    max_hold_days: int = 90,
    stop_loss: float = -0.08,
    take_profit: float = 0.15,
    rsi_exit_threshold: float = 55.0,
    min_hold_days: int = 5,
) -> pd.Series:
    """
    Calcula o retorno até ao exit real, não ao dia 60 fixo.

    O DataFrame deve estar ordenado por (ticker, date) e conter,
    para cada ticker, as linhas diárias a partir da data do alerta.

    exit_strategy opções
    --------------------
    'fixed_60d'      — baseline actual (retorno ao dia 60)
    'rsi_recovery'   — sai quando RSI volta a > rsi_exit_threshold
    'price_recovery' — sai quando preço volta ao nível pré-dip
    'trailing_stop'  — sai com stop_loss ou take_profit primeiro

    Retorna
    -------
    pd.Series com o retorno realizado por ticker (índice = ticker).
    """
    df = df.sort_values(["ticker", "date"])

    if exit_strategy == "rsi_recovery":
        def _exit(group: pd.DataFrame) -> float:
            entry = group[price_col].iloc[0]
            for i in range(min_hold_days, min(max_hold_days, len(group))):
                if group[rsi_col].iloc[i] > rsi_exit_threshold:
                    return group[price_col].iloc[i] / entry - 1
            if len(group) >= max_hold_days:
                return group[price_col].iloc[max_hold_days - 1] / entry - 1
            return np.nan

    elif exit_strategy == "price_recovery":
        def _exit(group: pd.DataFrame) -> float:
            entry = group[price_col].iloc[0]
            pre_dip_price = group[price_col].iloc[0]  # assumimos entrada no dip
            for i in range(min_hold_days, min(max_hold_days, len(group))):
                if group[price_col].iloc[i] >= pre_dip_price:
                    return group[price_col].iloc[i] / entry - 1
            if len(group) >= max_hold_days:
                return group[price_col].iloc[max_hold_days - 1] / entry - 1
            return np.nan

    elif exit_strategy == "trailing_stop":
        def _exit(group: pd.DataFrame) -> float:
            entry = group[price_col].iloc[0]
            for i in range(1, min(max_hold_days, len(group))):
                ret = group[price_col].iloc[i] / entry - 1
                if ret <= stop_loss or ret >= take_profit:
                    return ret
            if len(group) >= max_hold_days:
                return group[price_col].iloc[max_hold_days - 1] / entry - 1
            return np.nan

    else:  # fixed_60d
        def _exit(group: pd.DataFrame) -> float:
            if len(group) >= 60:
                return group[price_col].iloc[59] / group[price_col].iloc[0] - 1
            return np.nan

    return df.groupby("ticker").apply(_exit)
