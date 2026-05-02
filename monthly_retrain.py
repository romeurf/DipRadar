"""
monthly_retrain.py — Pipeline mensal de retreino com walk-forward gating.

Corre no dia 1 de cada mês (cron em main.py setup_schedule). Faz:

  1. Build training set incremental:
     a. bootstrap historical (`ml_training_merged.parquet`)
     b. `alert_db.csv` (alertas reais com outcomes preenchidos)
     c. `universe_snapshot.parquet` (rows com ≥6m maturados, label resolvido)

  2. Treina **candidate model** em `/data/candidate/`

  3. **Walk-forward gating**: compara candidate vs production via Stage1 AUC-PR.
     Promove candidate **só se** AUC-PR_candidate ≥ AUC-PR_production × 0.95.

  4. **Atomic deploy**: rename candidate.pkl → produção. Modelo antigo arquivado
     em `/data/archive/dip_model_stageX_<timestamp>.pkl`.

  5. Devolve dict com delta de métricas para Telegram.

Manual (debug):
  python monthly_retrain.py --dry-run
  python monthly_retrain.py --gating-ratio 0.90 --no-snapshot
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

# Paths (Railway Volume primeiro, /tmp fallback)
_DATA_DIR = Path("/data") if Path("/data").exists() else Path("/tmp")

PRODUCTION_DIR     = _DATA_DIR
CANDIDATE_DIR      = _DATA_DIR / "candidate"
ARCHIVE_DIR        = _DATA_DIR / "archive"
SNAPSHOT_PATH      = _DATA_DIR / "universe_snapshot.parquet"
BOOTSTRAP_PATH     = _DATA_DIR / "ml_training_merged.parquet"
ALERT_DB_PATH      = _DATA_DIR / "alert_db.csv"
TRAINING_INPUT     = _DATA_DIR / "ml_training_input.parquet"

PRODUCTION_REPORT  = PRODUCTION_DIR / "ml_report.json"
CANDIDATE_REPORT   = CANDIDATE_DIR / "ml_report.json"

DEFAULT_GATING_RATIO = 0.95     # candidate só substitui se AUC-PR ≥ prod × 0.95

# Floor absoluto: o gating nunca aceita candidate < FLOOR_AUC_PR mesmo que a
# produção tenha derivado para baixo. Inicializado a 0.18 (ligeiramente abaixo
# do baseline 0.192 do bootstrap inicial — Tier A+B+C). Persistido em /data/
# para sobreviver a redeploys e ser editável manualmente se preciso.
FLOOR_AUC_PR_DEFAULT = 0.18
FLOOR_PATH = _DATA_DIR / "ml_floor_auc_pr.json"
LABEL_HORIZON_DAYS   = 182      # 6m de price action necessários para resolver outcome


# ─────────────────────────────────────────────────────────────────────────────
# Outcome resolution para snapshot data (FREE — usa a própria snapshot table)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_snapshot_outcomes(snap: pd.DataFrame,
                               horizon_days: int = LABEL_HORIZON_DAYS) -> pd.DataFrame:
    """
    Para cada (symbol, snapshot_date) com snapshot ≥horizon_days no passado:
    procura snapshots futuras para o mesmo símbolo a +90d e +180d para calcular
    return_3m e return_6m. Faz lookup fuzzy (±3 dias trading).

    Alpha vs SPY é computado via merge com snapshot do SPY (se existir),
    senão fallback para outcome_label baseado em returns absolutos.

    Devolve subset de snap com colunas adicionais:
      label_win, outcome_label, return_3m, return_6m, alert_date.
    """
    if snap.empty:
        return pd.DataFrame()

    snap = snap.copy()
    snap["alert_date"] = pd.to_datetime(snap["snapshot_date"])
    today_ts = pd.Timestamp(date.today())

    # Mature: rows old enough to have 6m of forward data
    cutoff = today_ts - pd.Timedelta(days=horizon_days)
    mature = snap[snap["alert_date"] <= cutoff].copy()
    if mature.empty:
        log.info(f"[outcome] Nenhum snapshot maduro (>={horizon_days}d). N={len(snap)}.")
        return pd.DataFrame()
    log.info(f"[outcome] Snapshots maduros: {len(mature)}/{len(snap)}")

    # Index: (symbol, date) → price.
    snap_idx = snap.set_index(["symbol", "alert_date"])["price"].sort_index()

    def _price_near(symbol: str, target: pd.Timestamp) -> Optional[float]:
        """
        Procura preço para symbol em [target, target+5d] — forward-only.
        Evita look-ahead implicito que poderia introduzir leakage temporal
        em casos de earnings gaps ou halts próximos da data alvo.
        """
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

    # SPY benchmark (via yfinance one-shot — só para o intervalo necessário)
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
    except Exception as e:
        log.warning(f"[outcome] Falha a buscar SPY benchmark ({e}); fallback absoluto.")

    def _spy_near(target: pd.Timestamp) -> Optional[float]:
        """SPY price em [target, target+5d] — forward-only para consistência."""
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

        # Outcome label: alpha-based se SPY disponível, senão absolute
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
# Alert-DB CSV → DataFrame in training format
# ─────────────────────────────────────────────────────────────────────────────

def _load_alert_db_as_training() -> pd.DataFrame:
    """Read alert_db.csv and convert to ml_training schema (FEATURE_COLUMNS + label)."""
    if not ALERT_DB_PATH.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(ALERT_DB_PATH)
    except Exception as e:
        log.warning(f"[alert_db] Falha a ler {ALERT_DB_PATH}: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Filtra rows com label resolvido (outcome_label preenchido)
    df = df[df["outcome_label"].notna() & (df["outcome_label"] != "")].copy()
    if df.empty:
        return pd.DataFrame()

    # Map alert_db cols → training cols
    rename_map = {
        "date_iso":       "alert_date",
        "drawdown_52w":   "drawdown_52w",
        "change_day_pct": "drop_pct_today",
        "rsi":            "rsi_14",
        "fcf_yield":      "fcf_yield",
        "revenue_growth": "revenue_growth",
        "gross_margin":   "gross_margin",
        "debt_equity":    "de_ratio",
        "spy_change":     "spy_drawdown_5d",
        "sector_etf_change": "sector_drawdown_5d",
    }
    df = df.rename(columns=rename_map)

    # Defaults para colunas que alert_db não tem
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

    # PE vs fair: se temos pe e pe_fair, calcular
    if "pe" in df.columns and "pe_fair" in df.columns:
        pe   = pd.to_numeric(df["pe"], errors="coerce")
        fair = pd.to_numeric(df["pe_fair"], errors="coerce")
        df["pe_vs_fair"] = (pe / fair).clip(0.1, 5.0).fillna(1.0)

    # label_win
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
    """
    Concat: bootstrap historical + alert_db (with outcomes) + snapshot (matured).
    Dedup: (symbol, alert_date) — preferindo snapshot > alert_db > bootstrap.
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
                # Garante alert_date como datetime
                df_snap["alert_date"] = pd.to_datetime(df_snap["alert_date"])
                parts.append(df_snap.assign(_source="snapshot"))
        except Exception as e:
            log.error(f"[input] Snapshot processing falhou: {e}")

    if not parts:
        raise RuntimeError("Sem dados de treino — bootstrap, alert_db e snapshot todos vazios.")

    merged = pd.concat(parts, ignore_index=True, sort=False)
    log.info(f"[input] Concat total: {len(merged)} rows from "
             f"{merged['_source'].value_counts().to_dict()}")

    # Dedup: preferência snapshot > alert_db > bootstrap (último a ficar via keep="last")
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
# Walk-forward gating
# ─────────────────────────────────────────────────────────────────────────────

def _read_auc_pr(report_path: Path) -> Optional[float]:
    """Lê AUC-PR Stage1 de ml_report.json. Devolve None se não existir / corrupt."""
    if not report_path.exists():
        return None
    try:
        data = json.loads(report_path.read_text())
    except Exception as e:
        log.warning(f"[gating] Report corrupt at {report_path}: {e}")
        return None
    s1 = data.get("stage1") or {}
    val = s1.get("auc_pr_test") or s1.get("auc_pr")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _read_floor_auc_pr() -> float:
    """
    Lê o floor AUC-PR persistido. Cria com FLOOR_AUC_PR_DEFAULT se não existe.
    Editável manualmente via JSON ({"floor_auc_pr": 0.18, "set_at": "..."}).
    """
    try:
        if FLOOR_PATH.exists():
            data = json.loads(FLOOR_PATH.read_text())
            val = float(data.get("floor_auc_pr", FLOOR_AUC_PR_DEFAULT))
            return val
    except Exception as e:
        log.warning(f"[gating] Floor file corrupt at {FLOOR_PATH}: {e} — usando default")

    # Inicializa com default
    try:
        FLOOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        FLOOR_PATH.write_text(json.dumps({
            "floor_auc_pr": FLOOR_AUC_PR_DEFAULT,
            "set_at":       datetime.utcnow().isoformat() + "Z",
            "comment":      "Initial floor — bootstrap baseline ~0.192 (PR #3 Tier A+B+C).",
        }, indent=2))
    except Exception as e:
        log.debug(f"[gating] Não foi possível persistir floor: {e}")
    return FLOOR_AUC_PR_DEFAULT


def gate_and_promote(gating_ratio: float = DEFAULT_GATING_RATIO) -> dict:
    """
    Compara CANDIDATE_REPORT vs PRODUCTION_REPORT (Stage1 AUC-PR test).
    Promove candidate se:
      - candidate ≥ produção × gating_ratio   (gating relativo)
      - E candidate ≥ floor_auc_pr             (gating absoluto, anti-drift)
    Atomic via shutil.move + archive.
    """
    cand_auc = _read_auc_pr(CANDIDATE_REPORT)
    prod_auc = _read_auc_pr(PRODUCTION_REPORT)
    floor    = _read_floor_auc_pr()

    log.info(f"[gating] candidate AUC-PR={cand_auc} | production AUC-PR={prod_auc} "
             f"| ratio_threshold={gating_ratio} | floor={floor:.4f}")

    if cand_auc is None:
        return {"decision": "FAILED", "reason": "candidate report missing/invalid",
                "candidate_auc_pr": None, "production_auc_pr": prod_auc,
                "floor_auc_pr": floor}

    # Gate absoluto primeiro — recusa qualquer candidate abaixo do floor mínimo,
    # mesmo em cold start ou quando a produção derivou para baixo.
    if cand_auc < floor:
        return {
            "decision":          "KEPT",
            "reason":            f"candidate {cand_auc:.4f} < floor {floor:.4f} (absolute minimum)",
            "candidate_auc_pr":  cand_auc,
            "production_auc_pr": prod_auc,
            "gating_ratio":      gating_ratio,
            "floor_auc_pr":      floor,
        }

    if prod_auc is None:
        # Cold start — sem produção anterior, mas candidate ≥ floor → promove
        decision = "PROMOTED"
        reason   = f"cold start (no production model); candidate {cand_auc:.4f} ≥ floor {floor:.4f}"
    else:
        threshold = prod_auc * gating_ratio
        if cand_auc >= threshold:
            decision = "PROMOTED"
            reason   = f"candidate {cand_auc:.4f} ≥ {prod_auc:.4f} × {gating_ratio} = {threshold:.4f}"
        else:
            decision = "KEPT"
            reason   = f"candidate {cand_auc:.4f} < {prod_auc:.4f} × {gating_ratio} = {threshold:.4f}"

    result = {
        "decision":           decision,
        "reason":             reason,
        "candidate_auc_pr":   cand_auc,
        "production_auc_pr":  prod_auc,
        "gating_ratio":       gating_ratio,
        "floor_auc_pr":       floor,
    }

    if decision != "PROMOTED":
        return result

    # Archive prod first, then move candidate to prod
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for stage in (1, 2):
        src_prod = PRODUCTION_DIR / f"dip_model_stage{stage}.pkl"
        if src_prod.exists():
            dst = ARCHIVE_DIR / f"dip_model_stage{stage}_{timestamp}.pkl"
            shutil.copy2(src_prod, dst)
            log.info(f"[gating] Archived production → {dst.name}")

    promoted = []
    for stage in (1, 2):
        cand_pkl = CANDIDATE_DIR / f"dip_model_stage{stage}.pkl"
        if cand_pkl.exists():
            target  = PRODUCTION_DIR / f"dip_model_stage{stage}.pkl"
            tmp     = target.with_suffix(".pkl.tmp")
            shutil.copy2(cand_pkl, tmp)
            tmp.replace(target)
            promoted.append(target.name)

    # Replace report
    if CANDIDATE_REPORT.exists():
        tmp = PRODUCTION_REPORT.with_suffix(".json.tmp")
        shutil.copy2(CANDIDATE_REPORT, tmp)
        tmp.replace(PRODUCTION_REPORT)

    result["promoted_files"]    = promoted
    result["archive_timestamp"] = timestamp
    log.info(f"[gating] PROMOTED — {promoted}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main retrain entrypoint (chamado pelo cron de main.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_monthly_retrain_v2(
    gating_ratio: float = DEFAULT_GATING_RATIO,
    include_snapshot: bool = True,
    include_alert_db: bool = True,
    algos: Optional[list[str]] = None,
    dry_run: bool = False,
) -> dict:
    """
    Pipeline completo de retreino mensal.

    Returns dict com tudo o que precisas para um Telegram resumido:
      decision, reason, candidate_auc_pr, production_auc_pr, gating_ratio,
      n_train, n_cal, n_test, win_rate_test, elapsed_s.
    """
    t0 = time.time()
    algos = algos or ["rf", "xgb", "lgbm"]
    log.info("=" * 70)
    log.info(f"DipRadar — Monthly Retrain v2 ({datetime.utcnow().isoformat(timespec='seconds')}Z)")
    log.info("=" * 70)
    log.info(f"  gating_ratio       = {gating_ratio}")
    log.info(f"  include_snapshot   = {include_snapshot}")
    log.info(f"  include_alert_db   = {include_alert_db}")
    log.info(f"  algos              = {algos}")
    log.info(f"  dry_run            = {dry_run}")

    # 1. Outcomes back-fill (alert_db) — se a função existir
    try:
        from alert_db import fill_db_outcomes
        log.info("[retrain] A actualizar outcomes alert_db...")
        outcome_stats = fill_db_outcomes()
        log.info(f"[retrain] alert_db outcomes: {outcome_stats}")
    except Exception as e:
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
        return {"decision": "FAILED", "reason": f"input build failed: {e}",
                "elapsed_s": round(time.time() - t0, 1)}

    if dry_run:
        log.info(f"[retrain] DRY-RUN: training input = {train_path}; pulei treino.")
        return {"decision": "DRY-RUN", "training_input": str(train_path),
                "elapsed_s": round(time.time() - t0, 1)}

    # 3. Train candidate
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from train_model import train_all
        log.info(f"[retrain] A treinar candidate em {CANDIDATE_DIR}...")
        result = train_all(
            parquet_path=train_path,
            output_dir=CANDIDATE_DIR,
            algos=algos,
        )
    except Exception as e:
        log.error(f"[retrain] train_all falhou: {e}")
        return {"decision": "FAILED", "reason": f"train_all failed: {e}",
                "elapsed_s": round(time.time() - t0, 1),
                "outcome_stats": outcome_stats}

    s1 = result.get("stage1") or {}
    s2 = result.get("stage2") or {}

    # 4. Gating + atomic deploy
    gate = gate_and_promote(gating_ratio=gating_ratio)

    elapsed = time.time() - t0
    log.info(f"[retrain] CONCLUÍDO em {elapsed:.0f}s — decision={gate['decision']}")

    return {
        **gate,
        "elapsed_s":      round(elapsed, 1),
        "candidate_stage1": {
            "auc_pr_test":  s1.get("auc_pr_test"),
            "threshold":    s1.get("threshold"),
            "test_precision": s1.get("test_precision"),
            "test_recall":  s1.get("test_recall"),
            "n_train":      s1.get("n_train"),
            "n_cal":        s1.get("n_cal"),
            "n_test":       s1.get("n_test"),
        },
        "candidate_stage2": {
            "auc_pr_test":  s2.get("auc_pr_test"),
            "threshold":    s2.get("threshold"),
        } if s2 else None,
        "outcome_stats":  outcome_stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monthly retrain v2 with walk-forward gating")
    p.add_argument("--gating-ratio", type=float, default=DEFAULT_GATING_RATIO,
                   help=f"Candidate só é promovido se AUC-PR ≥ prod × ratio "
                        f"(default {DEFAULT_GATING_RATIO})")
    p.add_argument("--no-snapshot", action="store_true",
                   help="Não inclui universe_snapshot.parquet no input.")
    p.add_argument("--no-alert-db", action="store_true",
                   help="Não inclui alert_db.csv no input.")
    p.add_argument("--algos", type=str, default="rf,xgb,lgbm",
                   help="Lista CSV de algoritmos (default rf,xgb,lgbm)")
    p.add_argument("--dry-run", action="store_true",
                   help="Só constrói o training input — não treina nem promove.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    result = run_monthly_retrain_v2(
        gating_ratio=args.gating_ratio,
        include_snapshot=not args.no_snapshot,
        include_alert_db=not args.no_alert_db,
        algos=algos,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
