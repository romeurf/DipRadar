"""
diagnostics.py — Health checks ao dataset e ao sinal do modelo.

Duas responsabilidades:
  1. dataset_health_check   — verifica volume, período, distribuição temporal
                              e calcula IC SR para decidir se o sinal é real.
  2. check_survivorship_bias — compara universo histórico com actual.

Uso:
  from ml_training.diagnostics import dataset_health_check, check_survivorship_bias

  dataset_health_check(df, fold_results, target_col='alpha_60d_rank')
  check_survivorship_bias(current_tickers, historical_tickers)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Dataset health check
# ---------------------------------------------------------------------------

def dataset_health_check(
    df: pd.DataFrame,
    fold_results: list[dict],
    target_col: str = "alpha_60d_rank",
    date_col: str = "alert_date",
    ticker_col: str = "ticker",
    min_alerts: int = 3000,
) -> dict:
    """
    Verifica a saúde do dataset e calcula as métricas fundamentais do sinal.

    Imprime:
      Dataset stats: total alertas, tickers únicos, período, alertas por ano
      IC Analysis:   mean, std, min, max, IC SR, % folds positivos
      Veredicto:     🔴 / 🟡 / 🟢 com descrição

    Retorna dict com ic_mean, ic_std, ic_sr, verdict para uso programático.
    """
    print("=" * 50)
    print("=== DATASET HEALTH ===")
    print(f"Total alertas:    {len(df):,}")
    print(f"Tickers únicos:   {df[ticker_col].nunique():,}")

    try:
        dates = pd.to_datetime(df[date_col])
        print(f"Período:          {dates.min().date()} → {dates.max().date()}")
        df_tmp = df.copy()
        df_tmp["year"] = dates.dt.year
        print("\nAlertas por ano:")
        print(df_tmp["year"].value_counts().sort_index().to_string())
    except Exception:
        pass

    # IC stats
    ics = np.array([f["ic_overall"] for f in fold_results
                    if f.get("ic_overall") is not None and not np.isnan(f.get("ic_overall", np.nan))])

    if len(ics) == 0:
        print("\n⚠️  Sem resultados de IC para analisar.")
        return {"ic_mean": np.nan, "ic_std": np.nan, "ic_sr": np.nan, "verdict": "no_data"}

    ic_mean = ics.mean()
    ic_std  = ics.std()
    ic_sr   = ic_mean / ic_std if ic_std > 0 else 0.0
    pct_pos = (ics > 0).mean()

    print("\n=== IC ANALYSIS ===")
    print(f"IC mean:  {ic_mean:.4f}")
    print(f"IC std:   {ic_std:.4f}")
    print(f"IC min:   {ics.min():.4f}")
    print(f"IC max:   {ics.max():.4f}")
    ic_sr_label = "✅ usável" if ic_sr > 0.5 else "⚠️  muito ruidoso"
    print(f"IC SR:    {ic_sr:.4f}  {ic_sr_label}")
    print(f"% folds positivos: {pct_pos:.0%}")
    print()

    # Veredicto
    if len(df) < min_alerts:
        verdict = "low_volume"
        print(f"🔴 Volume insuficiente ({len(df):,} < {min_alerts:,}) — IC não é estatisticamente fiável")
    elif ic_std > ic_mean * 2:
        verdict = "unstable"
        print("🟡 IC std muito alto vs mean — sinal existe mas é instável entre folds")
    elif ic_sr > 0.5 and ic_mean > 0.05:
        verdict = "production_ready"
        print("🟢 Sinal real e consistente — IC SR > 0.5 e IC mean > 0.05 — pode ir a produção")
    else:
        verdict = "marginal"
        print("🟡 Sinal marginal — avaliar se o edge é suficiente para o teu threshold de risco")

    print("=" * 50)
    return {
        "ic_mean":  ic_mean,
        "ic_std":   ic_std,
        "ic_sr":    ic_sr,
        "pct_pos":  pct_pos,
        "verdict":  verdict,
        "n_alerts": len(df),
        "n_folds":  len(ics),
    }


# ---------------------------------------------------------------------------
# 2. Survivorship bias check
# ---------------------------------------------------------------------------

def check_survivorship_bias(
    current_tickers: list[str],
    historical_universe_tickers: list[str],
    warn_threshold: float = 0.05,
) -> dict:
    """
    Compara o universo histórico com os tickers actuais.

    Se o overlap for > 95% (< 5% de tickers desaparecidos), o universo
    histórico provavelmente só contém sobreviventes — survivorship bias.

    Fix pragmático: adiciona tickers históricos do S&P 500 via Wikipedia
    (ver get_historical_sp500_constituents abaixo).

    Retorna dict com delisted, new, overlap_ratio.
    """
    current_set    = set(current_tickers)
    historical_set = set(historical_universe_tickers)

    only_historical = historical_set - current_set   # desapareceram (delisted/bankrupt)
    only_current    = current_set - historical_set   # novos (IPO recente)
    overlap         = current_set & historical_set
    overlap_ratio   = len(overlap) / len(historical_set) if historical_set else 0.0
    delisted_ratio  = len(only_historical) / len(historical_set) if historical_set else 0.0

    print("=" * 50)
    print("=== SURVIVORSHIP BIAS CHECK ===")
    print(f"Tickers históricos:  {len(historical_set):,}")
    print(f"Tickers actuais:     {len(current_set):,}")
    print(f"Desapareceram (delisted/bankrupt): {len(only_historical):,}")
    print(f"Novos (IPO recente): {len(only_current):,}")
    print(f"Overlap:             {overlap_ratio:.1%}")

    if delisted_ratio < warn_threshold:
        print(
            f"\n🔴 Provável survivorship bias — apenas {delisted_ratio:.1%} do universo "
            "histórico desapareceu.\n"
            "   Universo histórico = universo actual → backtest inflacionado."
        )
    else:
        print(f"\n✅ {delisted_ratio:.1%} de tickers históricos desapareceram — universo histórico parece válido.")

    print("=" * 50)
    return {
        "delisted":       list(only_historical),
        "new":            list(only_current),
        "overlap_ratio":  overlap_ratio,
        "delisted_ratio": delisted_ratio,
        "has_bias":       delisted_ratio < warn_threshold,
    }


def get_historical_sp500_constituents() -> pd.DataFrame:
    """
    Componentes históricos do S&P 500 via Wikipedia.

    Inclui empresas que saíram do índice — útil como proxy para
    um universo histórico sem survivorship bias.

    Retorna a tabela de mudanças (entradas e saídas) com colunas:
      Date, Added, Removed, Reason (variável por ano de edição Wikipedia)
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    # Tabela 0 = componentes actuais, Tabela 1 = mudanças históricas
    changes = tables[1] if len(tables) > 1 else tables[0]
    return changes
