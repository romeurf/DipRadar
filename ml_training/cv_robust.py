"""
cv_robust.py — Funções robustas para treino e CV do DipRadar.

Mudanças vs versão anterior:
  - add_crosssection_rank(): target rank por data (neutraliza beta de mercado)
  - build_champion_ensemble(): selecciona e pondera modelos por IC
  - fit_ensemble_full(): treina ensemble no dataset completo
  - Calibrador isotónico P(top quartile)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler


# ─── Target Engineering ───────────────────────────────────────────────────────

def add_crosssection_rank(
    df: pd.DataFrame,
    raw_col: str = "alpha_60d",
    date_col: str = "alert_date",
) -> pd.DataFrame:
    """Rank percentílico cross-section por data.

    Para cada data de alerta, ranqueia os stocks pelo alpha bruto dentro
    dessa data. Resultado: coluna <raw_col>_rank ∈ [0, 1].

    Vantagens:
    - Elimina o beta de mercado (bull/bear market não afecta o rank relativo)
    - Target estacionário ao longo do tempo
    - IC muito mais estável entre folds do walk-forward CV
    """
    df = df.copy()
    df[f"{raw_col}_rank"] = (
        df.groupby(date_col)[raw_col]
          .rank(method="average", pct=True)
    )
    return df


# ─── Champion Ensemble Selection ─────────────────────────────────────────────

def build_champion_ensemble(
    cv_df: pd.DataFrame,
    n_folds: int,
    ic_threshold: float = 0.04,
    ic_min_floor: float = -0.01,
    top_n: int = 4,
) -> tuple[list[str], dict[str, float]]:
    """Selecciona modelos robustos e calcula pesos proporcionais ao IC_mean.

    Critérios de robustez:
      1. IC_mean >= ic_threshold (sinal mínimo aceitável)
      2. IC_min >= ic_min_floor (nenhum fold catastrófico)
      3. Pelo menos N_FOLDS-1 folds com resultado válido
      4. Apenas os top_n modelos passam (default 4) — diluição de
         sinal acima disto, já que modelos similares no zoo correlacionam
         predições e averaging não ajuda.

    Pesos: proporcionais a IC_mean^2 (favorece modelos top sobre medianos)
    clipado a 0 (modelos negativos = peso 0).
    Fallback: se nenhum passa, usa top 3 por IC_mean.
    """
    agg = (
        cv_df.groupby("model")
             .agg(
                 ic_mean=("ic", "mean"),
                 ic_std=("ic", "std"),
                 ic_min=("ic", "min"),
                 topk_mean=("topk_alpha", "mean"),
                 hit_mean=("hit_rate", "mean"),
                 n_folds_ok=("ic", "count"),
             )
             .sort_values("ic_mean", ascending=False)
    )

    robust = agg[
        (agg["ic_mean"] >= ic_threshold) &
        (agg["ic_min"] >= ic_min_floor) &
        (agg["n_folds_ok"] >= max(1, n_folds - 1))
    ].copy()

    if robust.empty:
        robust = agg.head(min(3, len(agg))).copy()

    # Limitar a top_n para evitar diluição por modelos similares.
    if top_n is not None and top_n > 0 and len(robust) > top_n:
        robust = robust.head(top_n).copy()

    # Pesos: ic_mean^2 (favorece top sobre mediano). Cap em 0 para excluir
    # modelos com IC negativo.
    pos = robust["ic_mean"].clip(lower=0.0)
    sq = pos ** 2
    total_w = sq.sum()
    if total_w > 0:
        robust["weight"] = sq / total_w
    else:
        robust["weight"] = pd.Series(1.0 / len(robust), index=robust.index)

    champion_models = robust.index.tolist()
    champion_weights = robust["weight"].to_dict()
    return champion_models, champion_weights


# ─── Full Training ────────────────────────────────────────────────────────────

def fit_ensemble_full(
    df_train: pd.DataFrame,
    model_configs: dict,
    champion_models: list[str],
    champion_weights: dict[str, float],
    target_col: str = "alpha_60d_rank",
    half_life_days: int = 365,
    date_col: str = "alert_date",
) -> tuple[dict, dict, dict, IsotonicRegression, np.ndarray]:
    """Treina ensemble completo no dataset inteiro.

    Returns
    -------
    trained_models : {name: fitted model}
    trained_scalers: {name: StandardScaler}
    trained_medians: {name: np.ndarray}
    calibrator     : IsotonicRegression (pred -> P(top quartile))
    ensemble_pred  : np.ndarray, predicoes in-sample (para diagnostico)
    """
    from ml_training.cv import temporal_weights

    dates_all = pd.to_datetime(df_train[date_col])
    max_date = dates_all.max()
    y_all = df_train[target_col].values.astype(float)
    w_all = temporal_weights(dates_all, max_date, half_life_days)

    trained_models: dict = {}
    trained_scalers: dict = {}
    trained_medians: dict = {}

    ensemble_pred = np.zeros(len(df_train))

    for name in champion_models:
        cfg = model_configs[name]
        feats = [f for f in cfg["feats"] if f in df_train.columns]

        Xi = df_train[feats].values.astype(float)
        med = np.nanmedian(Xi, axis=0)
        for col_i in range(Xi.shape[1]):
            nan_mask = ~np.isfinite(Xi[:, col_i])
            if nan_mask.any():
                Xi[nan_mask, col_i] = med[col_i]

        sc = StandardScaler()
        Xi_sc = sc.fit_transform(Xi)

        model = cfg["factory"]()
        try:
            model.fit(Xi_sc, y_all, sample_weight=w_all)
        except TypeError:
            model.fit(Xi_sc, y_all)

        trained_models[name] = model
        trained_scalers[name] = sc
        trained_medians[name] = med

        pred_i = model.predict(Xi_sc).astype(float)
        ensemble_pred += champion_weights[name] * pred_i

    # Rank final das predicoes ensemble
    ensemble_pred_rank = rankdata(ensemble_pred) / len(ensemble_pred)

    # Calibrador: P(top quartile) com IsotonicRegression
    y_top_quartile = (y_all >= 0.75).astype(float)
    calibrator = IsotonicRegression(out_of_bounds="clip", increasing=True)
    calibrator.fit(ensemble_pred_rank, y_top_quartile)

    return trained_models, trained_scalers, trained_medians, calibrator, ensemble_pred_rank


# ─── Inference Helper ────────────────────────────────────────────────────────

def predict_ensemble(
    df_new: pd.DataFrame,
    champion_models: list[str],
    champion_weights: dict[str, float],
    model_configs: dict,
    trained_models: dict,
    trained_scalers: dict,
    trained_medians: dict,
    calibrator: IsotonicRegression,
    reference_preds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Prediz score e probabilidade de top quartile para novos alertas.

    Se reference_preds e fornecido (predicoes do treino), rankeia os novos
    pontos RELATIVAMENTE ao historico (mais robusto).
    Caso contrario, usa o score bruto e aplica o calibrador directamente.
    """
    raw_preds = np.zeros(len(df_new))

    for name in champion_models:
        cfg = model_configs[name]
        feats = [f for f in cfg["feats"] if f in df_new.columns]
        med = trained_medians[name]
        sc = trained_scalers[name]
        model = trained_models[name]

        Xi = df_new[feats].values.astype(float)
        for col_i in range(Xi.shape[1]):
            nan_mask = ~np.isfinite(Xi[:, col_i])
            if nan_mask.any():
                Xi[nan_mask, col_i] = med[col_i]

        Xi_sc = sc.transform(Xi)
        pred_i = model.predict(Xi_sc).astype(float)
        raw_preds += champion_weights[name] * pred_i

    if reference_preds is not None:
        combined = np.concatenate([reference_preds, raw_preds])
        all_ranks = rankdata(combined) / len(combined)
        pred_rank = all_ranks[-len(raw_preds):]
    else:
        pred_rank = rankdata(raw_preds) / len(raw_preds)

    prob_top = calibrator.predict(pred_rank)
    score_100 = (pred_rank * 100).round(1)

    result = df_new[["ticker", "alert_date"]].copy() if "ticker" in df_new.columns else df_new.copy()
    result["score_rank"] = score_100
    result["prob_top_quartile"] = prob_top.round(4)
    return result
