"""
monthly_retrain.py — Pipeline mensal de retreino v3.

Corre no dia 1 de cada mês via cron (main.run_monthly_retrain → schedule).
Refactor completo: substitui o pipeline Stage1/Stage2 (AUC-PR) pela arquitectura
v3.1 (regressor alpha + isotonic calibrator) já em produção desde PR #11/#13.

Fluxo:

  1. Build training input incremental:
     a. bootstrap historical (`ml_training_merged.parquet`)
     b. `alert_db.csv` (alertas reais com outcomes preenchidos)
     c. `universe_snapshot.parquet` (rows com ≥6m maturados, label resolvido)

  2. Treino candidate via `ml_training.train_v31.run_training`:
     - 10 folds expanding-window, purge gap 21 dias
     - 34 features, target alpha_60d (max_return − spy_max_return)
     - Champion = melhor ρ_α com PnL > 0
     - Isotonic calibrator em OOF predictions

  3. **Gating** baseado em ρ_α (rho_alpha_mean):
     - Promove só se ρ_α candidate ≥ ρ_α produção × `gating_ratio` (default 0.90)
     - Floor absoluto ρ_α ≥ `FLOOR_RHO_ALPHA` (default 0.20)
     - Se candidate cai > 10% mas ainda passa o floor → guarda como
       `dip_models_v3_pending.pkl` para revisão manual (não promove)

  4. **Atomic deploy** via shutil.move + archive em `/data/archive/`.

  5. Devolve dict com delta de métricas (ρ_α, brier_oof, top-k PnL)
     para o Telegram do main.py formatar.

CLI:
    python monthly_retrain.py --dry-run
    python monthly_retrain.py --gating-ratio 0.85
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths (Railway Volume primeiro, /tmp fallback)
# ─────────────────────────────────────────────────────────────────────────────

_DATA_DIR = Path("/data") if Path("/data").exists() else Path("/tmp")

PRODUCTION_DIR     = _DATA_DIR
CANDIDATE_DIR      = _DATA_DIR / "candidate"
ARCHIVE_DIR        = _DATA_DIR / "archive"
SNAPSHOT_PATH      = _DATA_DIR / "universe_snapshot.parquet"
BOOTSTRAP_PATH     = _DATA_DIR / "ml_training_merged.parquet"
ALERT_DB_PATH      = _DATA_DIR / "alert_db.csv"
TRAINING_INPUT     = _DATA_DIR / "ml_training_input.parquet"

PRODUCTION_BUNDLE  = PRODUCTION_DIR / "dip_models_v3.pkl"
PRODUCTION_REPORT  = PRODUCTION_DIR / "ml_report_v3.json"
CANDIDATE_BUNDLE   = CANDIDATE_DIR / "dip_models_v3.pkl"
CANDIDATE_REPORT   = CANDIDATE_DIR / "ml_report_v3.json"
PENDING_BUNDLE     = PRODUCTION_DIR / "dip_models_v3_pending.pkl"
PENDING_REPORT     = PRODUCTION_DIR / "ml_report_v3_pending.json"

# ─────────────────────────────────────────────────────────────────────────────
# Gating constants
# ─────────────────────────────────────────────────────────────────────────────

# Candidate só promove se rho_alpha >= prod * gating_ratio
DEFAULT_GATING_RATIO = 0.90

# Floor absoluto: candidate <0.20 nunca é promovido
FLOOR_RHO_ALPHA_DEFAULT = 0.20
FLOOR_PATH = _DATA_DIR / "ml_floor_rho_alpha.json"

LABEL_HORIZON_DAYS = 182  # 6 meses para resolver outcomes


# ─────────────────────────────────────────────────────────────────────────────
# Outcome resolution para snapshot data (preservado do v2)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_snapshot_outcomes(snap: pd.DataFrame,
                               horizon_days: int = LABEL_HORIZON_DAYS) -> pd.DataFrame:
    """Resolve outcomes maduros do universe_snapshot via merge interno
    (sem network — usa a própria snapshot table como benchmark de preços).

    Para cada (symbol, snapshot_date) com ≥horizon_days no passado: procura
    snapshots futuras a +90d e +180d para calcular return_3m, return_6m e
    alpha vs SPY.
    """
    if snap.empty:
        return pd.DataFrame()

    snap = snap.copy()
    snap["alert_date"] = pd.to_datetime(snap["snapshot_date"])
    today_ts = pd.Timestamp(date.today())

    cutoff = today_ts - pd.Timedelta(days=horizon_days)
    mature = snap[snap["alert_date"] <= cutoff].copy()
    if mature.empty:
        log.info(f"[outcome] Nenhum snapshot maduro (>={horizon_days}d). N={len(snap)}.")
        return pd.DataFrame()
    log.info(f"[outcome] Snapshots maduros: {len(mature)}/{len(snap)}")

    snap_idx = snap.set_index(["symbol", "alert_date"])["price"].sort_index()

    def _price_near(symbol: str, target: pd.Timestamp) -> Optional[float]:
        for delta in (0, 1, 2, 3, 4, 5):
            t = target + pd.Timedelta(days=delta)
            try:
                v = snap_idx.loc[(symbol, t)]
                if isinstance(v, pd.Series):
                    v = v.iloc[0]
                if v is not None and float(v) > 0:
                    return float(v)
            except KeyError:
                continue
        return None

    spy_close: dict[pd.Timestamp, float] = {}
    try:
        import yfinance as yf
        start = (mature["alert_date"].min() - pd.Timedelta(days=10)).date()
        end   = (mature["alert_date"].max() + pd.Timedelta(days=horizon_days + 10)).date()
        spy_hist = yf.Ticker("SPY").history(
            start=start.isoformat(), end=end.isoformat(),
            auto_adjust=True, raise_errors=False,
        )
        if spy_hist is not None and not spy_hist.empty:
            spy_close = {
                pd.Timestamp(idx.date()): float(row["Close"])
                for idx, row in spy_hist.iterrows()
            }
            log.info(f"[outcome] SPY benchmark: {len(spy_close)} candles ({start} → {end})")
    except Exception as e:  # pragma: no cover (network)
        log.warning(f"[outcome] Falha a buscar SPY benchmark ({e}); fallback absoluto.")

    def _spy_near(target: pd.Timestamp) -> Optional[float]:
        if not spy_close:
            return None
        for delta in range(0, 6):
            t = target + pd.Timedelta(days=delta)
            v = spy_close.get(t)
            if v is not None:
                return v
        return None

    out_rows = []
    for _, row in mature.iterrows():
        sym  = row["symbol"]
        d    = row["alert_date"]
        entry = float(row["price"])
        if entry <= 0 or not np.isfinite(entry):
            continue

        p3m = _price_near(sym, d + pd.Timedelta(days=91))
        p6m = _price_near(sym, d + pd.Timedelta(days=182))
        if p3m is None and p6m is None:
            continue

        r3m = (p3m - entry) / entry * 100 if p3m else None
        r6m = (p6m - entry) / entry * 100 if p6m else None

        spy_entry = _spy_near(d)
        spy_3m    = _spy_near(d + pd.Timedelta(days=91)) if r3m is not None else None
        spy_6m    = _spy_near(d + pd.Timedelta(days=182)) if r6m is not None else None
        spy_r3m   = (spy_3m - spy_entry) / spy_entry * 100 if (spy_entry and spy_3m) else None
        spy_r6m   = (spy_6m - spy_entry) / spy_entry * 100 if (spy_entry and spy_6m) else None

        ref      = r6m if r6m is not None else r3m
        spy_ref  = spy_r6m if r6m is not None else spy_r3m
        if ref is None:
            continue

        if spy_ref is not None:
            from bootstrap_ml import outcome_label_alpha, label_win_binary
            label = outcome_label_alpha(ref, spy_ref)
        else:
            from bootstrap_ml import outcome_label, label_win_binary
            label = outcome_label(ref)

        new_row = dict(row)
        new_row["return_3m"]      = round(r3m, 2) if r3m is not None else None
        new_row["return_6m"]      = round(r6m, 2) if r6m is not None else None
        new_row["spy_return_ref"] = round(spy_ref, 2) if spy_ref is not None else 0.0
        new_row["outcome_label"]  = label
        new_row["label_win"]      = label_win_binary(label)
        out_rows.append(new_row)

    if not out_rows:
        return pd.DataFrame()

    out_df = pd.DataFrame(out_rows)
    log.info(f"[outcome] Resolved {len(out_df)} snapshot outcomes "
             f"(WIN={int(out_df['label_win'].sum())}, "
             f"LOSE={int((out_df['label_win']==0).sum())})")
    return out_df


# ─────────────────────────────────────────────────────────────────────────────
# Alert-DB CSV → DataFrame in training format (preservado do v2)
# ─────────────────────────────────────────────────────────────────────────────

def _load_alert_db_as_training() -> pd.DataFrame:
    """Read alert_db.csv and convert to ml_training schema."""
    if not ALERT_DB_PATH.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(ALERT_DB_PATH)
    except Exception as e:
        log.warning(f"[alert_db] Falha a ler {ALERT_DB_PATH}: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df = df[df["outcome_label"].notna() & (df["outcome_label"] != "")].copy()
    if df.empty:
        return pd.DataFrame()

    rename_map = {
        "date_iso":          "alert_date",
        "drawdown_52w":      "drawdown_52w",
        "change_day_pct":    "drop_pct_today",
        "rsi":               "rsi_14",
        "fcf_yield":         "fcf_yield",
        "revenue_growth":    "revenue_growth",
        "gross_margin":      "gross_margin",
        "debt_equity":       "de_ratio",
        "spy_change":        "spy_drawdown_5d",
        "sector_etf_change": "sector_drawdown_5d",
    }
    df = df.rename(columns=rename_map)

    defaults = {
        "macro_score":     2,
        "vix":             20.0,
        "atr_ratio":       0.02,
        "volume_spike":    df.get("volume_ratio", 1.0),
        "pe_vs_fair":      1.0,
        "analyst_upside":  df.get("analyst_upside", 0.10),
        "quality_score":   0.5,
    }
    for k, v in defaults.items():
        if k not in df.columns:
            df[k] = v

    if "pe" in df.columns and "pe_fair" in df.columns:
        pe   = pd.to_numeric(df["pe"], errors="coerce")
        fair = pd.to_numeric(df["pe_fair"], errors="coerce")
        df["pe_vs_fair"] = (pe / fair).clip(0.1, 5.0).fillna(1.0)

    df["label_win"] = df["outcome_label"].isin(["WIN_40", "WIN_20"]).astype(int)
    df["label_further_drop"] = None
    df["return_3m"] = pd.to_numeric(df.get("return_3m"), errors="coerce")
    df["return_6m"] = pd.to_numeric(df.get("return_6m"), errors="coerce")
    df["spy_return_ref"] = pd.to_numeric(df.get("spy_change", 0.0), errors="coerce").fillna(0.0)

    if "symbol" not in df.columns:
        return pd.DataFrame()

    log.info(f"[alert_db] Loaded {len(df)} alertas com outcome resolvido")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Build training input from 3 sources
# ─────────────────────────────────────────────────────────────────────────────

def build_training_input(include_snapshot: bool = True,
                         include_alert_db: bool = True) -> Path:
    """Concat: bootstrap + alert_db + snapshot (matured).

    Dedup: (symbol, alert_date) preferindo snapshot > alert_db > bootstrap.
    """
    parts: list[pd.DataFrame] = []

    if BOOTSTRAP_PATH.exists():
        try:
            df_bs = pd.read_parquet(BOOTSTRAP_PATH)
            if "alert_date" in df_bs.columns:
                df_bs["alert_date"] = pd.to_datetime(df_bs["alert_date"])
            log.info(f"[input] bootstrap historical: {len(df_bs)} rows")
            parts.append(df_bs.assign(_source="bootstrap"))
        except Exception as e:
            log.error(f"[input] Falha a ler {BOOTSTRAP_PATH}: {e}")
    else:
        log.warning(f"[input] {BOOTSTRAP_PATH} não existe.")

    if include_alert_db:
        df_alerts = _load_alert_db_as_training()
        if not df_alerts.empty:
            df_alerts["alert_date"] = pd.to_datetime(df_alerts["alert_date"])
            parts.append(df_alerts.assign(_source="alert_db"))

    if include_snapshot and SNAPSHOT_PATH.exists():
        try:
            snap = pd.read_parquet(SNAPSHOT_PATH)
            df_snap = _resolve_snapshot_outcomes(snap)
            if not df_snap.empty:
                df_snap["alert_date"] = pd.to_datetime(df_snap["alert_date"])
                parts.append(df_snap.assign(_source="snapshot"))
        except Exception as e:
            log.error(f"[input] Snapshot processing falhou: {e}")

    if not parts:
        raise RuntimeError("Sem dados de treino — bootstrap, alert_db e snapshot todos vazios.")

    merged = pd.concat(parts, ignore_index=True, sort=False)
    log.info(f"[input] Concat total: {len(merged)} rows from "
             f"{merged['_source'].value_counts().to_dict()}")

    priority = {"bootstrap": 0, "alert_db": 1, "snapshot": 2}
    merged["_prio"] = merged["_source"].map(priority).fillna(0)
    merged = merged.sort_values("_prio").drop_duplicates(
        subset=["symbol", "alert_date"], keep="last"
    ).drop(columns=["_prio", "_source"])
    log.info(f"[input] After dedup (symbol, alert_date): {len(merged)} rows")

    TRAINING_INPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(TRAINING_INPUT, index=False)
    log.info(f"[input] Wrote {TRAINING_INPUT} ({len(merged)} rows)")
    return TRAINING_INPUT


# ─────────────────────────────────────────────────────────────────────────────
# v3 metrics readers (rho_alpha-based)
# ─────────────────────────────────────────────────────────────────────────────

def _read_v3_metrics(report_path: Path) -> dict[str, Optional[float]]:
    """Lê métricas v3 do report. Devolve dict com chaves canónicas."""
    from ml_training.bundle import metrics_from_report
    return metrics_from_report(report_path)


def _read_floor_rho_alpha() -> float:
    """Lê o floor absoluto (cria se não existe). Editável manualmente via JSON."""
    try:
        if FLOOR_PATH.exists():
            data = json.loads(FLOOR_PATH.read_text())
            return float(data.get("floor_rho_alpha", FLOOR_RHO_ALPHA_DEFAULT))
    except Exception as e:
        log.warning(f"[gating] Floor file corrupt at {FLOOR_PATH}: {e} — usando default")

    try:
        FLOOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        FLOOR_PATH.write_text(json.dumps({
            "floor_rho_alpha": FLOOR_RHO_ALPHA_DEFAULT,
            "set_at":          datetime.utcnow().isoformat() + "Z",
            "comment":         "Initial floor — baseline ρ_α 0.334 (PR #13 v3.1).",
        }, indent=2))
    except Exception as e:
        log.debug(f"[gating] Não foi possível persistir floor: {e}")
    return FLOOR_RHO_ALPHA_DEFAULT


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward gating + atomic deploy (v3)
# ─────────────────────────────────────────────────────────────────────────────

def gate_and_promote_v3(
    cand_metrics: dict[str, Optional[float]],
    prod_metrics: dict[str, Optional[float]],
    gating_ratio: float = DEFAULT_GATING_RATIO,
) -> dict:
    """Compara candidate vs production via ρ_α.

    Decisão:
      - FAILED: candidate sem ρ_α válido
      - KEPT_FLOOR: candidate < floor → recusado (mesmo sem produção)
      - PROMOTED (cold start): produção vazia, candidate ≥ floor
      - PROMOTED: candidate ≥ produção × gating_ratio
      - PENDING: candidate ≥ floor mas < produção × gating_ratio →
          guarda como `dip_models_v3_pending.pkl` para revisão manual
    """
    cand_rho  = cand_metrics.get("rho_alpha_mean")
    prod_rho  = prod_metrics.get("rho_alpha_mean")
    cand_brier = cand_metrics.get("brier_oof")
    prod_brier = prod_metrics.get("brier_oof")
    cand_pnl  = cand_metrics.get("topk_pnl_mean")
    prod_pnl  = prod_metrics.get("topk_pnl_mean")
    floor     = _read_floor_rho_alpha()

    log.info(
        f"[gating] candidate ρ_α={cand_rho} | production ρ_α={prod_rho} | "
        f"ratio_threshold={gating_ratio} | floor={floor:.4f}"
    )

    base_result = {
        "candidate_rho_alpha":  cand_rho,
        "production_rho_alpha": prod_rho,
        "candidate_brier":      cand_brier,
        "production_brier":     prod_brier,
        "candidate_topk_pnl":   cand_pnl,
        "production_topk_pnl":  prod_pnl,
        "gating_ratio":         gating_ratio,
        "floor_rho_alpha":      floor,
    }

    if cand_rho is None:
        return {
            **base_result,
            "decision": "FAILED",
            "reason":   "candidate report missing/invalid (rho_alpha_mean ausente)",
        }

    # Floor absoluto: rejeita sempre se candidate < floor
    if cand_rho < floor:
        return {
            **base_result,
            "decision": "KEPT_FLOOR",
            "reason":   f"candidate ρ_α {cand_rho:.4f} < floor {floor:.4f}",
        }

    # Cold start: sem produção, basta passar floor
    if prod_rho is None:
        return _do_promote({
            **base_result,
            "decision": "PROMOTED",
            "reason":   f"cold start; candidate ρ_α {cand_rho:.4f} ≥ floor {floor:.4f}",
        })

    threshold = prod_rho * gating_ratio
    if cand_rho >= threshold:
        delta_pct = (cand_rho - prod_rho) / prod_rho * 100 if prod_rho else 0
        return _do_promote({
            **base_result,
            "decision":  "PROMOTED",
            "reason":    (
                f"candidate ρ_α {cand_rho:.4f} ≥ {prod_rho:.4f} × {gating_ratio} "
                f"= {threshold:.4f} (Δ {delta_pct:+.1f}%)"
            ),
            "delta_pct": float(delta_pct),
        })

    # Não passa o gating mas passa o floor → guarda como pending
    return _save_pending({
        **base_result,
        "decision":  "PENDING",
        "reason":    (
            f"candidate ρ_α {cand_rho:.4f} < {prod_rho:.4f} × {gating_ratio} "
            f"= {threshold:.4f}; guardado como pending para revisão manual"
        ),
    })


def _do_promote(result: dict) -> dict:
    """Atomic move: candidate → production (com archive da produção anterior)."""
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Archive produção actual (se existir)
    if PRODUCTION_BUNDLE.exists():
        archived_pkl = ARCHIVE_DIR / f"dip_models_v3_{timestamp}.pkl"
        shutil.copy2(PRODUCTION_BUNDLE, archived_pkl)
        log.info(f"[gating] Archived production bundle → {archived_pkl.name}")
    if PRODUCTION_REPORT.exists():
        archived_json = ARCHIVE_DIR / f"ml_report_v3_{timestamp}.json"
        shutil.copy2(PRODUCTION_REPORT, archived_json)

    # Atomic copy candidate → production
    promoted: list[str] = []
    if CANDIDATE_BUNDLE.exists():
        tmp = PRODUCTION_BUNDLE.with_suffix(".pkl.tmp")
        shutil.copy2(CANDIDATE_BUNDLE, tmp)
        tmp.replace(PRODUCTION_BUNDLE)
        promoted.append(PRODUCTION_BUNDLE.name)
    if CANDIDATE_REPORT.exists():
        tmp = PRODUCTION_REPORT.with_suffix(".json.tmp")
        shutil.copy2(CANDIDATE_REPORT, tmp)
        tmp.replace(PRODUCTION_REPORT)

    log.info(f"[gating] PROMOTED — {promoted}")
    result["promoted_files"]    = promoted
    result["archive_timestamp"] = timestamp
    return result


def _save_pending(result: dict) -> dict:
    """Guarda candidate como `dip_models_v3_pending.pkl` (não promove)."""
    if CANDIDATE_BUNDLE.exists():
        shutil.copy2(CANDIDATE_BUNDLE, PENDING_BUNDLE)
        log.info(f"[gating] Pending bundle: {PENDING_BUNDLE}")
        result["pending_bundle"] = str(PENDING_BUNDLE)
    if CANDIDATE_REPORT.exists():
        shutil.copy2(CANDIDATE_REPORT, PENDING_REPORT)
        result["pending_report"] = str(PENDING_REPORT)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main retrain entrypoint v3 (chamado pelo cron de main.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_monthly_retrain_v3(
    gating_ratio: float = DEFAULT_GATING_RATIO,
    include_snapshot: bool = True,
    include_alert_db: bool = True,
    dry_run: bool = False,
    n_folds: Optional[int] = None,
    purge_days: Optional[int] = None,
) -> dict:
    """Pipeline completo de retreino mensal v3.

    Returns dict com tudo necessário para o Telegram do main.py:
      decision, reason, candidate_rho_alpha, production_rho_alpha,
      candidate_brier, production_brier, candidate_topk_pnl, production_topk_pnl,
      gating_ratio, elapsed_s, outcome_stats, ...
    """
    t0 = time.time()
    log.info("=" * 70)
    log.info(f"DipRadar — Monthly Retrain v3 ({datetime.utcnow().isoformat(timespec='seconds')}Z)")
    log.info("=" * 70)
    log.info(f"  gating_ratio     = {gating_ratio}")
    log.info(f"  include_snapshot = {include_snapshot}")
    log.info(f"  include_alert_db = {include_alert_db}")
    log.info(f"  dry_run          = {dry_run}")

    # 1. Outcomes back-fill (alert_db)
    try:
        from alert_db import fill_db_outcomes
        log.info("[retrain] A actualizar outcomes alert_db...")
        outcome_stats = fill_db_outcomes()
        log.info(f"[retrain] alert_db outcomes: {outcome_stats}")
    except Exception as e:  # pragma: no cover
        log.warning(f"[retrain] fill_db_outcomes falhou ({e}) — a continuar sem.")
        outcome_stats = {"updated": 0, "skipped": 0}

    # 2. Build training input
    try:
        train_path = build_training_input(
            include_snapshot=include_snapshot,
            include_alert_db=include_alert_db,
        )
    except Exception as e:
        log.error(f"[retrain] build_training_input falhou: {e}")
        return {
            "decision":  "FAILED",
            "reason":    f"input build failed: {e}",
            "elapsed_s": round(time.time() - t0, 1),
        }

    if dry_run:
        log.info(f"[retrain] DRY-RUN: training input = {train_path}; pulei treino.")
        return {
            "decision":       "DRY-RUN",
            "training_input": str(train_path),
            "elapsed_s":      round(time.time() - t0, 1),
        }

    # 3. Treinar candidate via ml_training.train_v31
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from ml_training.config import N_FOLDS, PURGE_DAYS
        from ml_training.train_v31 import run_training

        log.info(f"[retrain] A treinar candidate em {CANDIDATE_DIR}...")
        run_training(
            input_parquet=train_path,
            output_bundle=CANDIDATE_BUNDLE,
            output_report=CANDIDATE_REPORT,
            n_folds=n_folds if n_folds is not None else N_FOLDS,
            purge_days=purge_days if purge_days is not None else PURGE_DAYS,
        )
    except Exception as e:
        log.error(f"[retrain] Treino falhou: {e}", exc_info=True)
        return {
            "decision":      "FAILED",
            "reason":        f"train_v31 failed: {e}",
            "elapsed_s":     round(time.time() - t0, 1),
            "outcome_stats": outcome_stats,
        }

    # 4. Gating + atomic deploy
    cand_metrics = _read_v3_metrics(CANDIDATE_REPORT)
    prod_metrics = _read_v3_metrics(PRODUCTION_REPORT)
    gate = gate_and_promote_v3(
        cand_metrics=cand_metrics,
        prod_metrics=prod_metrics,
        gating_ratio=gating_ratio,
    )

    elapsed = time.time() - t0
    log.info(f"[retrain] CONCLUÍDO em {elapsed:.0f}s — decision={gate['decision']}")

    return {
        **gate,
        "elapsed_s":     round(elapsed, 1),
        "outcome_stats": outcome_stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat alias para o main.py existente
# ─────────────────────────────────────────────────────────────────────────────

def run_monthly_retrain_v2(*args, **kwargs) -> dict:
    """Alias retrocompatível — main.py ainda importa este nome.

    Aceita os mesmos kwargs que v3 (gating_ratio, include_snapshot,
    include_alert_db, dry_run). Argumentos legacy (algos, etc.) são ignorados.
    """
    legacy_keys = {"algos"}
    clean_kwargs = {k: v for k, v in kwargs.items() if k not in legacy_keys}
    return run_monthly_retrain_v3(*args, **clean_kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monthly retrain v3 with rho_alpha gating")
    p.add_argument("--gating-ratio", type=float, default=DEFAULT_GATING_RATIO,
                   help=f"Candidate só é promovido se ρ_α ≥ prod × ratio "
                        f"(default {DEFAULT_GATING_RATIO})")
    p.add_argument("--no-snapshot", action="store_true",
                   help="Não inclui universe_snapshot.parquet no input.")
    p.add_argument("--no-alert-db", action="store_true",
                   help="Não inclui alert_db.csv no input.")
    p.add_argument("--n-folds", type=int, default=None,
                   help="Override de n_folds (default: ml_training.config.N_FOLDS)")
    p.add_argument("--purge-days", type=int, default=None,
                   help="Override de purge_days")
    p.add_argument("--dry-run", action="store_true",
                   help="Só constrói o training input — não treina nem promove.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    result = run_monthly_retrain_v3(
        gating_ratio=args.gating_ratio,
        include_snapshot=not args.no_snapshot,
        include_alert_db=not args.no_alert_db,
        dry_run=args.dry_run,
        n_folds=args.n_folds,
        purge_days=args.purge_days,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
