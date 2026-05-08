"""
evaluation.py — IC por fold, sector e regime de VIX.

Métrica central: Spearman IC (Information Coefficient).
IC > 0.05 com std < IC*1.5 é sinal útil para ranking.
IC SR (IC_mean / IC_std) > 0.5 indica sinal consistente e accionável.

Uso no loop walk-forward:
  from ml_training.evaluation import (
      information_coefficient, ic_by_sector,
      ic_by_regime, full_fold_report
  )

  fold_results = []
  for fold_idx, (train_idx, test_idx) in enumerate(wf_splits):
      ...
      fold_results.append({
          'ic_overall': information_coefficient(y_test, preds),
          'ic_sector':  ic_by_sector(test_df, 'pred', 'alpha_60d_rank').to_dict('index'),
          'ic_regime':  ic_by_regime(test_df, 'pred', 'alpha_60d_rank').to_dict('index'),
      })

  report = full_fold_report(fold_results)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# IC base
# ---------------------------------------------------------------------------

def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman IC — métrica standard para modelos de ranking.

    Retorna nan se não houver variância suficiente.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Remove nans
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ic, _ = stats.spearmanr(y_true[mask], y_pred[mask])
    return float(ic) if not np.isnan(ic) else np.nan


# ---------------------------------------------------------------------------
# IC por sector
# ---------------------------------------------------------------------------

def ic_by_sector(
    df_fold: pd.DataFrame,
    pred_col: str,
    target_col: str,
    sector_col: str = "sector",
    min_samples: int = 10,
) -> pd.DataFrame:
    """
    IC por sector num único fold.

    Detecta se o modelo é sector-specific ou genuinamente geral.
    Um modelo saudável deve ter IC positivo em >= 60% dos sectores.

    Retorna DataFrame com colunas [ic, n] indexado por sector.
    """
    results: dict[str, dict] = {}
    for sector, grp in df_fold.groupby(sector_col):
        if len(grp) < min_samples:
            continue
        ic = information_coefficient(grp[target_col], grp[pred_col])
        results[str(sector)] = {"ic": ic, "n": len(grp)}

    if not results:
        return pd.DataFrame(columns=["ic", "n"])
    return pd.DataFrame(results).T.astype({"n": int})


# ---------------------------------------------------------------------------
# IC por regime de VIX
# ---------------------------------------------------------------------------

def ic_by_regime(
    df_fold: pd.DataFrame,
    pred_col: str,
    target_col: str,
    vix_col: str = "vix",
    min_samples: int = 10,
) -> pd.DataFrame:
    """
    IC separado por regime de VIX.

    Regimes: calm (<15), normal (15-25), stress (25-50), crisis (>50).

    Responde a: 'o modelo funciona bem em stress mas falha em rally?'
    Se IC_stress >> IC_calm em mais de 0.10, o modelo é regime-dependent
    e pode over-bet em períodos de alta volatilidade.

    Retorna DataFrame com colunas [ic, n] indexado por regime.
    """
    bins   = [0, 15, 25, 50, 999]
    labels = ["calm", "normal", "stress", "crisis"]

    df = df_fold.copy()
    df["_vix_regime_eval"] = pd.cut(
        df[vix_col], bins=bins, labels=labels
    )

    results: dict[str, dict] = {}
    for regime, grp in df.groupby("_vix_regime_eval", observed=True):
        if len(grp) < min_samples:
            continue
        ic = information_coefficient(grp[target_col], grp[pred_col])
        results[str(regime)] = {"ic": ic, "n": len(grp)}

    if not results:
        return pd.DataFrame(columns=["ic", "n"])
    return pd.DataFrame(results).T.astype({"n": int})


# ---------------------------------------------------------------------------
# Report agregado de todos os folds
# ---------------------------------------------------------------------------

def full_fold_report(fold_results: list[dict]) -> pd.DataFrame:
    """
    Agrega resultados de todos os folds e imprime diagnóstico.

    fold_results: lista de dicts com keys:
      ic_overall  — float
      ic_sector   — dict retornado por ic_by_sector(...).to_dict('index')
      ic_regime   — dict retornado por ic_by_regime(...).to_dict('index')

    Imprime:
      IC Summary — mean, std, min, max, IC SR, % folds positivos
      Alertas sobre regime-dependência e volume insuficiente

    Retorna DataFrame por fold para análise posterior.
    """
    records = []
    for i, fold in enumerate(fold_results):
        sec  = fold.get("ic_sector",  {})
        reg  = fold.get("ic_regime",  {})
        records.append({
            "fold":      i,
            "ic_overall": fold.get("ic_overall", np.nan),
            "ic_tech":    sec.get("Technology", {}).get("ic", np.nan)
                          if isinstance(sec.get("Technology"), dict)
                          else (sec.get("Technology", {}) or {}).get("ic", np.nan),
            "ic_health":  (sec.get("Healthcare") or {}).get("ic", np.nan),
            "ic_financials": (sec.get("Financials") or {}).get("ic", np.nan),
            "ic_calm":    (reg.get("calm")   or {}).get("ic", np.nan),
            "ic_normal":  (reg.get("normal") or {}).get("ic", np.nan),
            "ic_stress":  (reg.get("stress") or {}).get("ic", np.nan),
        })

    df = pd.DataFrame(records)

    ics      = df["ic_overall"].dropna()
    ic_mean  = ics.mean()
    ic_std   = ics.std()
    ic_sr    = ic_mean / ic_std if ic_std > 0 else 0.0
    pct_pos  = (ics > 0).mean()

    print("\n=== IC Summary across folds ===")
    print(df[["fold", "ic_overall", "ic_tech", "ic_health",
              "ic_calm", "ic_stress"]].to_string(index=False))
    print(f"\nIC mean:  {ic_mean:.4f}")
    print(f"IC std:   {ic_std:.4f}")
    print(f"IC min:   {ics.min():.4f}")
    print(f"IC max:   {ics.max():.4f}")
    ic_sr_label = "✅ usável" if ic_sr > 0.5 else "⚠️  muito ruidoso"
    print(f"IC SR:    {ic_sr:.4f}  {ic_sr_label}")
    print(f"% folds positivos: {pct_pos:.0%}")

    # Alerta sobre regime-dependência
    calm_mean   = df["ic_calm"].dropna().mean()
    stress_mean = df["ic_stress"].dropna().mean()
    if not np.isnan(calm_mean) and not np.isnan(stress_mean):
        ic_diff = stress_mean - calm_mean
        if abs(ic_diff) > 0.10:
            print(
                f"\n⚠️  IC stress vs calm diff = {ic_diff:.3f} — "
                "modelo regime-dependent (pode over-bet em alta vol)"
            )

    return df
