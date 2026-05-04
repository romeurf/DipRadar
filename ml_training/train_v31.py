"""ml_training/train_v31.py — Orchestrator + CLI do treino v3.1.

Substitui o notebook ``experiments/ml_v2/DipRadar_v3_Training.ipynb`` por
módulos Python testáveis. Pipeline completo:

  1. Carregar dataset base (parquet com features v1/v2)
  2. (opcional) Fetch yfinance — pula se for fornecido ``price_cache``/``etf_cache``
  3. Construir dataset v3.1 (34 features + target alpha_60d)
  4. Walk-forward CV (10 folds expanding-window, purge 21d)
  5. Seleccionar champion (rho_alpha_mean máximo com PnL > 0)
  6. Treinar champion no dataset completo + calibrator isotónico
  7. Empacotar ``DipModelsV3`` + escrever ``ml_report_v3.json``

CLI:
    python -m ml_training.train_v31 \\
        --input  ml_training_merged.parquet \\
        --output dip_models_v3.pkl \\
        --report ml_report_v3.json \\
        [--n-folds 10] [--purge-days 21] [--max-tickers 30 --max-rows 1000]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from ml_training.bundle import (
    DipModelsV3,
    build_report,
    save_bundle,
    save_report,
)
from ml_training.config import (
    HALF_LIFE_DAYS,
    HORIZON_DAYS,
    MOMENTUM_FEATURES,
    NEW_FEATURES_V31,
    N_FOLDS,
    PURGE_DAYS,
)
from ml_training.data import build_dataset_v31, load_base_dataset
from ml_training.models import build_feature_lists, build_model_configs
from ml_training.train import (
    fit_isotonic_calibrator,
    run_walk_forward_cv,
    select_champion,
    summarize_results,
    train_full_champion,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline programático
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    input_parquet: Path,
    output_bundle: Optional[Path] = None,
    output_report: Optional[Path] = None,
    *,
    price_cache: Optional[dict[str, pd.DataFrame]] = None,
    etf_cache: Optional[dict[str, pd.DataFrame]] = None,
    n_folds: int = N_FOLDS,
    purge_days: int = PURGE_DAYS,
    horizon_days: int = HORIZON_DAYS,
    min_train: int = 100,
    min_test: int = 20,
    max_tickers: Optional[int] = None,
    max_rows: Optional[int] = None,
    skip_fetch: bool = False,
) -> dict:
    """Pipeline completo. Devolve dict com bundle, report e métricas.

    Parameters
    ----------
    input_parquet : Path
        Caminho do parquet base (e.g. ``ml_training_merged.parquet``).
    output_bundle : Path | None
        Onde escrever o ``dip_models_v3.pkl``. Se ``None``, não escreve.
    output_report : Path | None
        Onde escrever o ``ml_report_v3.json``. Se ``None``, não escreve.
    price_cache, etf_cache : dict | None
        Caches já fetchados (para testes ou re-treinos rápidos). Se ``None``
        e ``skip_fetch=False``, faz fetch yfinance.
    n_folds, purge_days, horizon_days : int
        Parâmetros do CV. Defaults match o notebook.
    max_tickers, max_rows : int | None
        Slicing para smoke tests. Se definido, usa apenas estes tickers/linhas.
    skip_fetch : bool
        Se True, exige que ``price_cache``/``etf_cache`` venham preenchidos
        (raise caso contrário). Útil em testes sem rede.

    Returns
    -------
    dict com chaves:
      - ``bundle``   : DipModelsV3
      - ``report``   : dict (conteúdo do ml_report_v3.json)
      - ``summary``  : pd.DataFrame com comparação dos modelos
      - ``df_v31``   : pd.DataFrame v3.1 (com features + target)
      - ``oof_pred`` : dict[name, np.ndarray]
    """
    base_df = load_base_dataset(Path(input_parquet))

    # Filtrar só linhas com targets resolvidos (cell 6 do notebook só conta
    # as que têm spy_return_ref, mas nós deixamos build_dataset_v31 filtrar).
    if "spy_return_ref" in base_df.columns:
        n_pre = len(base_df)
        base_df = base_df[base_df["spy_return_ref"].notna()].reset_index(drop=True)
        log.info(f"[orchestrator] Linhas com target resolvido: {len(base_df)}/{n_pre}")

    # Smoke test slicing (deterministic — sample por ticker)
    if max_tickers is not None and max_tickers < base_df["ticker"].nunique():
        kept = sorted(base_df["ticker"].unique())[:max_tickers]
        base_df = base_df[base_df["ticker"].isin(kept)].reset_index(drop=True)
        log.info(f"[orchestrator] max_tickers={max_tickers} → kept {len(base_df)} rows")

    if max_rows is not None and max_rows < len(base_df):
        base_df = base_df.head(max_rows).reset_index(drop=True)
        log.info(f"[orchestrator] max_rows={max_rows} → kept {len(base_df)} rows")

    # Fetch caches (se não fornecidos e fetch não foi desactivado)
    if price_cache is None or etf_cache is None:
        if skip_fetch:
            raise ValueError(
                "skip_fetch=True mas price_cache/etf_cache não foram fornecidos"
            )
        from ml_training.price_fetch import fetch_caches_for_dataset
        etf_cache, price_cache = fetch_caches_for_dataset(base_df, horizon_days)

    # Feature lists + model configs
    feats_v31, feats_baseline = build_feature_lists()
    log.info(
        f"[orchestrator] FEATURE_COLUMNS_V31={len(feats_v31)} | "
        f"baseline={len(feats_baseline)}"
    )

    # Construir dataset v3.1
    df_v31, skipped = build_dataset_v31(
        base_df=base_df,
        price_cache=price_cache,
        etf_cache=etf_cache,
        feature_cols_v31=feats_v31,
        horizon_days=horizon_days,
    )
    if df_v31.empty:
        raise RuntimeError(
            f"Dataset v3.1 vazio. Skipped: {skipped}. "
            f"Verifica se price_cache cobre os tickers."
        )

    # Walk-forward CV
    model_configs = build_model_configs(feats_v31, feats_baseline)
    results, oof_pred, fold_specs = run_walk_forward_cv(
        df_v31=df_v31,
        model_configs=model_configs,
        n_folds=n_folds,
        purge_days=purge_days,
        min_train=min_train,
        min_test=min_test,
    )

    # Sumarização + champion
    summary = summarize_results(results)
    if summary.empty:
        raise RuntimeError(
            f"Walk-forward CV não produziu resultados — todos os folds caíram "
            f"abaixo de min_train={min_train}/min_test={min_test}. "
            f"Reduz min_test ou alarga o dataset."
        )
    log.info("[orchestrator] Resumo:\n" + summary.round(4).to_string(index=False))
    champion_name, champion_row = select_champion(summary)
    log.info(f"[orchestrator] Champion: {champion_name}")

    # Calibrator OOF
    iso, brier, n_oof = fit_isotonic_calibrator(
        oof_pred[champion_name],
        df_v31["alpha_60d"].values,
        alpha_threshold=0.05,
    )
    log.info(
        f"[orchestrator] Calibrator: brier_oof={brier:.4f} (target<0.20) | "
        f"n_oof={n_oof}"
    )

    # Treino full do champion
    champ_alpha, champ_down, feats_used, n_train = train_full_champion(
        df_v31=df_v31,
        champion_cfg=model_configs[champion_name],
    )

    # Empacotar bundle
    bundle = DipModelsV3(
        model_up=champ_alpha,
        model_down=champ_down,
        feature_cols=feats_used,
        score_calibrator=iso,
        n_train_samples=int(n_train),
        train_date=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        champion_name=champion_name,
        schema_version=3,
        momentum_feats=list(MOMENTUM_FEATURES),
        rho_mean=float(champion_row["rho_alpha_mean"]),
        rho_alpha=float(champion_row["rho_alpha_mean"]),
        rho_down=float(champion_row["rho_down_mean"]),
        topk_pnl=float(champion_row["topk_pnl_mean"]),
        fold_metrics=results[champion_name],
    )

    # Report
    win_rate_alpha = float((df_v31["alpha_60d"] > 0.05).mean())
    report = build_report(
        bundle=bundle,
        summary_df=summary,
        brier_oof=brier,
        win_rate_alpha=win_rate_alpha,
        n_folds_used=len(fold_specs),
        purge_days=purge_days,
        horizon_days=horizon_days,
        new_features=NEW_FEATURES_V31,
    )

    # Persist (opcional)
    if output_bundle is not None:
        save_bundle(bundle, Path(output_bundle))
    if output_report is not None:
        save_report(report, Path(output_report))

    return {
        "bundle":   bundle,
        "report":   report,
        "summary":  summary,
        "df_v31":   df_v31,
        "oof_pred": oof_pred,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DipRadar v3.1 training pipeline")
    p.add_argument("--input", type=Path, required=True,
                   help="Parquet base (e.g. ml_training_merged.parquet)")
    p.add_argument("--output", type=Path, default=Path("dip_models_v3.pkl"),
                   help="Output bundle path (default dip_models_v3.pkl)")
    p.add_argument("--report", type=Path, default=Path("ml_report_v3.json"),
                   help="Output report path (default ml_report_v3.json)")
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    p.add_argument("--purge-days", type=int, default=PURGE_DAYS)
    p.add_argument("--horizon-days", type=int, default=HORIZON_DAYS)
    p.add_argument("--min-train", type=int, default=100,
                   help="Mínimo de samples train por fold (default 100)")
    p.add_argument("--min-test", type=int, default=20,
                   help="Mínimo de samples test por fold (default 20)")
    p.add_argument("--max-tickers", type=int, default=None,
                   help="Smoke test: usa apenas N tickers (slice determinista)")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Smoke test: usa apenas N linhas")
    p.add_argument("--skip-fetch", action="store_true",
                   help="Não faz yfinance fetch — exige caches injectados")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    log.info("=" * 70)
    log.info(f"DipRadar v3.1 — Training Pipeline ({datetime.utcnow().isoformat(timespec='seconds')}Z)")
    log.info("=" * 70)
    log.info(f"  input        = {args.input}")
    log.info(f"  output       = {args.output}")
    log.info(f"  report       = {args.report}")
    log.info(f"  n_folds      = {args.n_folds}")
    log.info(f"  purge_days   = {args.purge_days}")
    log.info(f"  horizon_days = {args.horizon_days}")
    log.info(f"  max_tickers  = {args.max_tickers}")
    log.info(f"  max_rows     = {args.max_rows}")
    log.info(f"  skip_fetch   = {args.skip_fetch}")

    out = run_training(
        input_parquet=args.input,
        output_bundle=args.output,
        output_report=args.report,
        n_folds=args.n_folds,
        purge_days=args.purge_days,
        horizon_days=args.horizon_days,
        min_train=args.min_train,
        min_test=args.min_test,
        max_tickers=args.max_tickers,
        max_rows=args.max_rows,
        skip_fetch=args.skip_fetch,
    )

    bundle = out["bundle"]
    log.info("=" * 70)
    log.info(
        f"Champion: {bundle.champion_name} | "
        f"rho_alpha={bundle.rho_alpha:.4f} | "
        f"rho_down={bundle.rho_down:.4f} | "
        f"topk_pnl={bundle.topk_pnl:.4f}"
    )
    log.info(f"n_train={bundle.n_train_samples} | n_features={len(bundle.feature_cols)}")
    log.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
