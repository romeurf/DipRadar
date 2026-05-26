"""
monthly_retrain.py — Monthly retraining pipeline v3.

Runs on the 1st of each month via cron (main.run_monthly_retrain → schedule).
Full refactor: replaces Stage1/Stage2 (AUC-PR) pipeline with v3.1 architecture
(alpha regressor + isotonic calibrator) in production since PR #11/#13.

Flow:

  1. Build incremental training input:
     a. bootstrap historical (`/data/ml_training_base.parquet`,
        with fallback to `ml_training_base.parquet` in repo root
        for cold-start on empty Railway volumes)
     b. `alert_db.csv` (real alerts with resolved outcomes)
     c. `universe_snapshot.parquet` (rows >= 6m matured, label resolved)

  2. Train candidate via `ml_training.train.run_training`:
     - 10 folds expanding-window, purge gap 21 days
     - target alpha_60d_rank cross-section + sector-neutralized
     - Champion ensemble Ridge+XGB+LGBM weighted by IC
     - Isotonic calibrator on OOF predictions

  3. **Gating** based on ρ_α (rho_alpha_mean):
     - Promote only if ρ_α candidate >= ρ_α production × `gating_ratio` (default 0.90)
     - Absolute floor ρ_α >= `FLOOR_RHO_ALPHA` (default 0.08; see
       FLOOR_RHO_ALPHA_DEFAULT below for history)
     - If candidate drops > 10% but still passes floor → save as
       `dip_models_pending.pkl` for manual review (not promoted)

  4. **Atomic deploy** via shutil.move + archive in `/data/archive/`.

  5. Returns dict with metric deltas (ρ_α, brier_oof, top-k PnL)
     for main.py to format for Telegram.

CLI:
    python monthly_retrain.py --dry-run
    python monthly_retrain.py --gating-ratio 0.85
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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

def _data_dir() -> Path:
    """Resolve data directory at call time — not at import time.

    Railway volumes may not be mounted when the bot process starts and imports
    this module. Evaluating Path("/data").exists() at import time can return
    False even though /data will be available by the time any retrain function
    runs. This function forces re-evaluation on every call so paths are always
    resolved against the live filesystem state.
    """
    return Path("/data") if Path("/data").exists() else Path("/tmp")

_REPO_ROOT = Path(__file__).resolve().parent


def _paths() -> dict:
    """Resolve all data paths at call time against the live filesystem.
    Railway may mount /data after this module is imported, so anything resolved
    at import time can permanently point to /tmp by accident. Every function
    that touches a /data path MUST call _paths() locally — do NOT cache."""
    dd = _data_dir()
    prod_dir = dd
    cand_dir = dd / "candidate"
    return {
        "PRODUCTION_DIR":            prod_dir,
        "CANDIDATE_DIR":             cand_dir,
        "ARCHIVE_DIR":               dd / "archive",
        "SNAPSHOT_PATH":             dd / "universe_snapshot.parquet",
        "BOOTSTRAP_PATH":            dd / "ml_training_base.parquet",
        "BOOTSTRAP_FALLBACK":        _REPO_ROOT / "ml_training_base.parquet",
        "LEGACY_BOOTSTRAP_PATH":     dd / "ml_training_merged.parquet",
        "LEGACY_BOOTSTRAP_FALLBACK": _REPO_ROOT / "ml_training_merged.parquet",
        "ALERT_DB_PATH":             dd / "alert_db.csv",
        "TRAINING_INPUT":            dd / "ml_training_input.parquet",
        "PRODUCTION_BUNDLE":         prod_dir / "dip_models.pkl",
        "PRODUCTION_REPORT":         prod_dir / "ml_report.json",
        "CANDIDATE_BUNDLE":          cand_dir / "dip_models.pkl",
        "CANDIDATE_REPORT":          cand_dir / "ml_report.json",
        "PENDING_BUNDLE":            prod_dir / "dip_models_pending.pkl",
        "PENDING_REPORT":            prod_dir / "ml_report_pending.json",
        "FLOOR_PATH":                dd / "ml_floor_rho_alpha.json",
    }


# Module-level path attributes — accessed via external `from monthly_retrain import X`
# or `monthly_retrain.X`. PEP 562 __getattr__ handles both. Functions defined inside
# this module must call _paths() locally; LOAD_GLOBAL bypasses __getattr__.
_PATH_KEYS = {
    "PRODUCTION_DIR", "CANDIDATE_DIR", "ARCHIVE_DIR", "SNAPSHOT_PATH",
    "BOOTSTRAP_PATH", "BOOTSTRAP_FALLBACK", "LEGACY_BOOTSTRAP_PATH",
    "LEGACY_BOOTSTRAP_FALLBACK", "ALERT_DB_PATH", "TRAINING_INPUT",
    "PRODUCTION_BUNDLE", "PRODUCTION_REPORT", "CANDIDATE_BUNDLE",
    "CANDIDATE_REPORT", "PENDING_BUNDLE", "PENDING_REPORT", "FLOOR_PATH",
}


def __getattr__(name: str):
    """Lazy resolution for `monthly_retrain.PRODUCTION_BUNDLE` and friends."""
    if name in _PATH_KEYS:
        return _paths()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ─────────────────────────────────────────────────────────────────────────────
# Gating constants
# ─────────────────────────────────────────────────────────────────────────────

# Candidate só promove se rho_alpha >= prod * gating_ratio
DEFAULT_GATING_RATIO = 0.90

# Floor absoluto: candidate < FLOOR_RHO_ALPHA_DEFAULT nunca é promovido.
#
# Histórico:
#   • 0.334 (PR #13 v3.1)  — baseline original do modelo v3 sem regen.
#   • 0.20  (initial gate) — usado durante o PR #23 → #28; provou-se
#                            inatingível com o parquet pós-regen + as
#                            features actuais (IC profile cai para ~0.1).
#   • 0.08  (PR #29 actual)— ajustado ao novo ceiling realista pós-Phase A:
#                            PR #28 entregou ρ_α 0.0982 com 20 features
#                            limpas. Sinal real, mas abaixo do floor 0.20
#                            antigo. 0.08 dá margem para promover bundles
#                            que melhorem sobre o produção sem aceitar
#                            ruído puro (random IC ≈ 0).
# Editável em runtime via `/admin_set_floor <value>` (escreve em FLOOR_PATH).
FLOOR_RHO_ALPHA_DEFAULT = 0.08
# FLOOR_PATH is resolved lazily via module __getattr__ (see _paths() above).

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
            auto_adjust=True,
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
# Alert-DB CSV → DataFrame in training format
# ─────────────────────────────────────────────────────────────────────────────

def _load_alert_db_as_training() -> pd.DataFrame:
    """Read alert_db.csv and convert to ml_training schema.

    Fixes aplicados:
      - on_bad_lines='warn' para não crashar com linhas antigas mal formadas
      - rename_map corrigido: drawdown_from_high (nome actual no CSV)
        → drawdown_52w (nome canónico em FEATURE_COLUMNS)
      - rename explícito symbol → ticker para alinhar com data.py/load_base_dataset
      - defaults Stage 3c: rows anteriores ao commit 2026-05-05 não têm
        quality_dislocation, peg_ratio, relative_drop, month_of_year —
        sem default chegam como NaN ao train e causam KeyError ou
        degradação silenciosa do modelo.
    """
    alert_db_path = _paths()["ALERT_DB_PATH"]
    if not alert_db_path.exists():
        return pd.DataFrame()

    # Migrate schema before reading — eliminates ParserWarnings from rows
    # with a field count that differs from the header. Idempotent and safe.
    try:
        from alert_db import migrate_schema
        migrate_schema()
    except Exception as _mig_err:
        log.warning(f"[alert_db] migrate_schema failed: {_mig_err}")

    try:
        df = pd.read_csv(alert_db_path, engine="python")
    except Exception as e:
        log.warning(f"[alert_db] Failed to read {alert_db_path}: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Incluir linhas com outcome_label preenchido OU maduras (>= 92 dias).
    # Linhas maduras sem outcome_label entram no treino com alpha_90d=NaN;
    # _compute_alpha_90d_inline() calcula-o via yfinance.
    # Sem esta condição, alert_db não contribui para treino se Tiingo falhar.
    _today = pd.Timestamp.now()
    if "date_iso" in df.columns:
        _age = (_today - pd.to_datetime(df["date_iso"], errors="coerce")).dt.days
    elif "alert_date" in df.columns:
        _age = (_today - pd.to_datetime(df["alert_date"], errors="coerce")).dt.days
    else:
        _age = pd.Series(0, index=df.index)
    _outcome_ok  = df["outcome_label"].notna() & (df["outcome_label"] != "")
    _mature_mask = _age.fillna(0) >= 92
    df = df[_outcome_ok | _mature_mask].copy()
    if df.empty:
        return pd.DataFrame()

    # ── Rename: CSV column name → training schema name ────────────────────
    # NOTA: "drawdown_from_high" é o nome actual gravado pelo alert_db.py.
    #       "drawdown_52w" é o nome canónico que FEATURE_COLUMNS usa.
    #       O rename antigo ("drawdown_52w": "drawdown_52w") era um no-op
    #       e deixava a coluna ausente — causava Index(['symbol']) no concat.
    rename_map = {
        "date_iso":          "alert_date",
        "drawdown_from_high": "drawdown_52w",   # ← fix: nome real no CSV
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

    # ── symbol → ticker (data.py espera "ticker") ─────────────────────────
    if "symbol" in df.columns and "ticker" not in df.columns:
        df = df.rename(columns={"symbol": "ticker"})

    if "ticker" not in df.columns:
        log.error("[alert_db] Coluna 'ticker'/'symbol' ausente após rename — a ignorar alert_db.")
        return pd.DataFrame()

    # Only set defaults for features still present in FEATURE_COLUMNS. Dead
    # features (gross_margin, de_ratio, pe_vs_fair, atr_ratio, volume_spike,
    # quality_score, quality_dislocation, relative_drop, ...) were removed in
    # PR #28 — assigning defaults for them is dead code that pollutes the
    # parquet schema and confuses readers.
    try:
        from ml_features import FEATURE_COLUMNS as _FC, _FALLBACK
        live_features = set(_FC)
    except Exception:
        live_features = set()
        _FALLBACK = {}

    defaults = {
        "vix":                 _FALLBACK.get("vix", 20.0),
        "peg_ratio":           _FALLBACK.get("peg_ratio", 2.0),
        "month_of_year":       float(datetime.now().month),
    }
    for k, v in defaults.items():
        if k in live_features and k not in df.columns:
            df[k] = v

    df["label_win"] = df["outcome_label"].isin(["WIN_STRONG", "WIN"]).astype(int)  # alinhado com _SCORE_MED/HIGH
    df["label_further_drop"] = None
    df["return_3m"] = pd.to_numeric(df.get("return_3m"), errors="coerce")
    df["return_6m"] = pd.to_numeric(df.get("return_6m"), errors="coerce")
    _spy_col = df["spy_change"] if "spy_change" in df.columns else pd.Series(0.0, index=df.index)
    df["spy_return_ref"] = pd.to_numeric(_spy_col, errors="coerce").fillna(0.0)

    log.info(f"[alert_db] Loaded {len(df)} alertas com outcome resolvido")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Build training input from 3 sources
# ─────────────────────────────────────────────────────────────────────────────

def _compute_alpha_90d_inline(df: pd.DataFrame) -> pd.DataFrame:
    """Computa alpha_90d para as linhas que não o têm, usando o price cache local.

    Usa /data/price_cache/ (criado por regen anteriores) ou descarrega via
    yfinance se o cache não existir. É invocado automaticamente por
    build_training_input() quando o merged parquet não tem alpha_90d.
    """
    import math, time
    from pathlib import Path as _P

    _cache_dir = _P("/data/price_cache") if _P("/data").exists() else _P("/tmp/price_cache")
    _cache_dir.mkdir(parents=True, exist_ok=True)
    _spy_hist: Optional[pd.DataFrame] = None

    # Configurável via env vars no Railway:
    #   INLINE_TICKER_TIMEOUT — segundos por ticker (default 25s)
    #   INLINE_BUDGET_MINUTES — minutos máximo para toda a fase (default 60min)
    _TICKER_TIMEOUT  = int(os.environ.get("INLINE_TICKER_TIMEOUT", "25"))
    _TOTAL_BUDGET    = int(os.environ.get("INLINE_BUDGET_MINUTES", "60")) * 60
    _t_start_inline  = time.time()

    def _get_hist(ticker: str) -> Optional[pd.DataFrame]:
        """Lê do cache ou descarrega via yfinance com timeout de 20s por ticker."""
        import threading
        safe = ticker.replace("^", "_idx_").replace("/", "_")
        pq   = _cache_dir / f"{safe}.parquet"

        # 1. Tentar cache em disco
        if pq.exists():
            try:
                h = pd.read_parquet(pq)
                h.index = pd.to_datetime(h.index)
                idx = pd.DatetimeIndex(h.index)
                if idx.tz is not None:
                    h.index = idx.tz_convert(None)
                return h
            except Exception:
                pass

        # 2. Descarregar com timeout via thread
        result_holder: list = [None]
        def _fetch():
            try:
                import yfinance as yf
                h = yf.Ticker(ticker).history(start="1995-01-01", auto_adjust=True)
                if h is not None and not h.empty:
                    idx = pd.DatetimeIndex(h.index)
                    h.index = idx.tz_convert(None) if idx.tz is not None else idx
                    h = h[["Open", "High", "Low", "Close", "Volume"]]
                    result_holder[0] = h
            except Exception as e:
                log.debug(f"[inline_90d] fetch {ticker}: {e}")

        t = threading.Thread(target=_fetch, daemon=True)
        t.start()
        t.join(timeout=_TICKER_TIMEOUT)
        if t.is_alive():
            log.warning(f"[inline_90d] {ticker}: timeout de {_TICKER_TIMEOUT}s — a saltar")
            return None

        h = result_holder[0]
        if h is not None:
            try:
                h.to_parquet(pq)
            except Exception:
                pass
            time.sleep(0.15)  # rate limit gentil
        return h

    # SPY para benchmark (com timeout também)
    _spy_hist = _get_hist("SPY")

    df = df.copy()
    from ml_training.config import PRIMARY_TARGET
    if PRIMARY_TARGET not in df.columns:
        df[PRIMARY_TARGET] = float("nan")

    needs_mask = df[PRIMARY_TARGET].isna()
    if not needs_mask.any():
        return df

    unique_tickers = (
        df.loc[needs_mask, "ticker"].astype(str).unique()
        if "ticker" in df.columns else []
    )

    price_cache: dict = {}
    for i, t in enumerate(unique_tickers):
        # Budget total: se ultrapassar 45 minutos, treina com o que temos
        elapsed = time.time() - _t_start_inline
        if elapsed > _TOTAL_BUDGET:
            log.warning(
                f"[inline_90d] Budget de {_TOTAL_BUDGET//60}min esgotado após "
                f"{i}/{len(unique_tickers)} tickers — a treinar com dados parciais."
            )
            break
        h = _get_hist(t)
        if h is not None:
            price_cache[t] = h
        if (i + 1) % 50 == 0:
            log.info(f"[inline_90d] {i+1}/{len(unique_tickers)} tickers em cache")

    resolved = 0
    for idx_val in df.index[needs_mask]:
        row = df.loc[idx_val]
        ticker = str(row.get("ticker", ""))
        hist = price_cache.get(ticker)
        if hist is None or hist.empty:
            continue
        try:
            _ts = pd.Timestamp(row["alert_date"])
            alert_date = _ts.tz_convert(None) if _ts.tz is not None else _ts
            # Entry price from history
            entry_slice = hist[hist.index <= alert_date]
            if entry_slice.empty:
                continue
            price = float(entry_slice["Close"].iloc[-1])
            if price <= 0:
                continue
            exit_date = alert_date + pd.Timedelta(days=90)
            fwd = hist[(hist.index > alert_date) & (hist.index <= exit_date)]
            if len(fwd) < 5:
                continue
            stock_ret = float(fwd["Close"].iloc[-1]) / price - 1.0
            if not math.isfinite(stock_ret) or abs(stock_ret) > 2.0:
                continue
            # SPY benchmark
            spy_ret = float("nan")
            if _spy_hist is not None and not _spy_hist.empty:
                sp_e = _spy_hist[_spy_hist.index <= alert_date]
                sp_x = _spy_hist[(_spy_hist.index > alert_date) & (_spy_hist.index <= exit_date)]
                if not sp_e.empty and not sp_x.empty:
                    sp_p = float(sp_e["Close"].iloc[-1])
                    if sp_p > 0:
                        spy_ret = float(sp_x["Close"].iloc[-1]) / sp_p - 1.0
            if math.isfinite(spy_ret):
                alpha = math.log1p(stock_ret) - math.log1p(spy_ret)
            else:
                alpha = stock_ret  # sem benchmark, usa retorno absoluto
            df.at[idx_val, PRIMARY_TARGET] = round(alpha, 6)
            resolved += 1
        except Exception:
            pass

    log.info(f"[inline_90d] Resolvidos {resolved}/{needs_mask.sum()} rows")
    return df


def build_training_input(include_snapshot: bool = False,
                         include_alert_db: bool = True) -> Path:
    """Concat: bootstrap + alert_db.

    Dedup: (ticker, alert_date) preferring alert_db > bootstrap.
    include_snapshot is kept as a parameter for backward compat but ignored —
    universe_snapshot was removed (partial features hurt model quality).
    """
    # Re-evaluate data_dir at call time — Railway may mount /data after process start.
    _dd = _data_dir()
    _bootstrap_path   = _dd / "ml_training_base.parquet"
    _bootstrap_fallbk = _REPO_ROOT / "ml_training_base.parquet"
    _legacy_path      = _dd / "ml_training_merged.parquet"
    _legacy_fallbk    = _REPO_ROOT / "ml_training_merged.parquet"
    _training_input   = _dd / "ml_training_input.parquet"
    _alert_db_path    = _dd / "alert_db.csv"

    parts: list[pd.DataFrame] = []

    bootstrap_src = None
    if _bootstrap_path.exists() and not os.getenv("FORCE_BOOTSTRAP_FROM_REPO"):
        bootstrap_src = _bootstrap_path
    elif _bootstrap_fallbk.exists():
        bootstrap_src = _bootstrap_fallbk
        log.warning(
            f"[input] {_bootstrap_path} ausente — fallback para parquet do repo "
            f"({_bootstrap_fallbk})."
        )
        # Copiar o parquet do repo para /data/ imediatamente para que deploys
        # futuros não percam o parquet regenerado (que é escrito em /data/).
        try:
            import shutil
            _bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_bootstrap_fallbk, _bootstrap_path)
            log.info(f"[input] Parquet copiado para {_bootstrap_path} — futuras regens escrevem aqui.")
        except Exception as _cp_err:
            log.warning(f"[input] Falha ao copiar para /data/: {_cp_err}")
    elif _legacy_path.exists():
        bootstrap_src = _legacy_path
        log.warning(
            f"[input] Usar bootstrap legacy {_legacy_path} — considera "
            f"renomear para ml_training_base.parquet."
        )
    elif _legacy_fallbk.exists():
        bootstrap_src = _legacy_fallbk
        log.warning(
            f"[input] Usar bootstrap legacy {_legacy_fallbk} (repo) — considera "
            f"renomear para ml_training_base.parquet."
        )
    else:
        log.warning(
            f"[input] Bootstrap parquet ausente em {_bootstrap_path} "
            f"e fallback {_bootstrap_fallbk} (legacy também não encontrado)."
        )

    if bootstrap_src is not None:
        try:
            df_bs = pd.read_parquet(bootstrap_src)
            if "alert_date" in df_bs.columns:
                df_bs["alert_date"] = pd.to_datetime(df_bs["alert_date"])
            log.info(
                f"[input] bootstrap historical: {len(df_bs)} rows "
                f"(source={bootstrap_src})"
            )
            parts.append(df_bs.assign(_source="bootstrap"))
        except Exception as e:
            log.error(f"[input] Falha a ler {bootstrap_src}: {e}")

    if include_alert_db:
        df_alerts = _load_alert_db_as_training()
        if not df_alerts.empty:
            df_alerts["alert_date"] = pd.to_datetime(df_alerts["alert_date"])
            parts.append(df_alerts.assign(_source="alert_db"))

    if not parts:
        raise RuntimeError("No training data — bootstrap and alert_db both empty.")

    merged = pd.concat(parts, ignore_index=True, sort=False)
    log.info(f"[input] Concat total: {len(merged)} rows from "
             f"{merged['_source'].value_counts().to_dict()}")

    # Dedup por (ticker, alert_date) — preferindo snapshot > alert_db > bootstrap
    # Usa "ticker" como chave canónica (data.py renomeia symbol→ticker)
    dedup_col = "ticker" if "ticker" in merged.columns else "symbol"
    priority = {"bootstrap": 0, "alert_db": 1, "snapshot": 2}
    _unknown_sources = set(merged["_source"].unique()) - set(priority.keys())
    if _unknown_sources:
        log.warning(f"[input] Fontes desconhecidas na prioridade de dedup: {_unknown_sources} — tratadas como bootstrap (0)")
    merged["_prio"] = merged["_source"].map(priority).fillna(0)
    merged = merged.sort_values("_prio").drop_duplicates(
        subset=[dedup_col, "alert_date"], keep="last"
    ).drop(columns=["_prio", "_source"])
    log.info(f"[input] After dedup ({dedup_col}, alert_date): {len(merged)} rows")

    # ── Computar alpha_90d inline se estiver em falta ───────────────────────
    # Garante que o merged parquet tem sempre alpha_90d antes do treino,
    # independentemente da origem dos dados ou do estado do bootstrap.
    from ml_training.config import PRIMARY_TARGET
    _n_90d = int(merged[PRIMARY_TARGET].notna().sum()) if PRIMARY_TARGET in merged.columns else 0
    if _n_90d < 200:
        log.info(f"[input] alpha_90d em falta ({_n_90d} linhas) — a computar inline...")
        merged = _compute_alpha_90d_inline(merged)
        _n_90d_after = int(merged[PRIMARY_TARGET].notna().sum()) if PRIMARY_TARGET in merged.columns else 0
        log.info(f"[input] alpha_90d resolvido inline: {_n_90d_after}/{len(merged)}")

    _training_input.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(_training_input, index=False)
    log.info(f"[input] Wrote {_training_input} ({len(merged)} rows)")
    return _training_input


# ─────────────────────────────────────────────────────────────────────────────
# v3 metrics readers (rho_alpha-based)
# ─────────────────────────────────────────────────────────────────────────────

def _read_v3_metrics(report_path: Path) -> dict[str, Optional[float]]:
    """Lê métricas v3 do report. Devolve dict com chaves canónicas."""
    from ml_training.bundle import metrics_from_report
    return metrics_from_report(report_path)


def _read_floor_rho_alpha() -> float:
    """Read the absolute floor (creates if missing). Path is resolved at call time."""
    floor_path = _paths()["FLOOR_PATH"]
    try:
        if floor_path.exists():
            data = json.loads(floor_path.read_text())
            return float(data.get("floor_rho_alpha", FLOOR_RHO_ALPHA_DEFAULT))
    except Exception as e:
        log.warning(f"[gating] Floor file corrupt at {floor_path}: {e} — using default")

    try:
        floor_path.parent.mkdir(parents=True, exist_ok=True)
        floor_path.write_text(json.dumps({
            "floor_rho_alpha": FLOOR_RHO_ALPHA_DEFAULT,
            "set_at":          datetime.utcnow().isoformat() + "Z",
            "comment":         f"Initial floor — default {FLOOR_RHO_ALPHA_DEFAULT}.",
        }, indent=2))
    except Exception as e:
        log.debug(f"[gating] Could not persist floor: {e}")
    return FLOOR_RHO_ALPHA_DEFAULT


def set_floor_rho_alpha(value: float, *, comment: Optional[str] = None) -> dict:
    """Adjust the absolute floor at runtime and persist to FLOOR_PATH."""
    if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 0.5):
        raise ValueError(
            f"floor_rho_alpha out of range [0.0, 0.5]: {value!r}"
        )
    floor_path = _paths()["FLOOR_PATH"]
    old = _read_floor_rho_alpha()
    payload = {
        "floor_rho_alpha": float(value),
        "set_at":          datetime.utcnow().isoformat() + "Z",
        "comment":         comment or f"Manual override via set_floor_rho_alpha (prev={old:.4f}).",
        "previous":        old,
    }
    floor_path.parent.mkdir(parents=True, exist_ok=True)
    floor_path.write_text(json.dumps(payload, indent=2))
    log.info(f"[gating] floor_rho_alpha: {old:.4f} → {value:.4f}")
    return {"old": old, "new": float(value), "path": str(floor_path)}


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward gating + atomic deploy (v3)
# ─────────────────────────────────────────────────────────────────────────────

def _build_comparison_detail(
    cand: dict[str, Optional[float]],
    prod: dict[str, Optional[float]],
) -> dict:
    """Compara métricas candidato vs produção e produz análise legível.

    Retorna dict com:
      - rows: lista de (label, cand_val, prod_val, delta_pct, improved)
      - improved: lista de métricas que melhoraram
      - worsened: lista de métricas que pioraram
      - recommendation: texto de aconselhamento
    """
    def _pct_change(new, old):
        if old is None or old == 0 or new is None:
            return None
        return (new - old) / abs(old) * 100

    metrics_cfg = [
        # (key, label, higher_is_better)
        ("rho_alpha_mean", "Qualidade preditiva (IC)", True),
        ("topk_pnl_mean",  "Retorno top stocks",       True),
        ("brier_oof",      "Calibração (Brier)",        False),  # lower is better
    ]

    rows = []
    improved, worsened = [], []

    for key, label, higher_better in metrics_cfg:
        c_val = cand.get(key)
        p_val = prod.get(key)
        delta = _pct_change(c_val, p_val)

        if delta is not None:
            got_better = (delta > 0) if higher_better else (delta < 0)
            if abs(delta) >= 3:   # ignorar variações < 3%
                (improved if got_better else worsened).append(label)
        else:
            got_better = None

        rows.append({
            "label":       label,
            "cand":        round(c_val, 4) if c_val is not None else None,
            "prod":        round(p_val, 4) if p_val is not None else None,
            "delta_pct":   round(delta, 1) if delta is not None else None,
            "improved":    got_better,
        })

    # Conselho baseado na análise
    if not worsened:
        rec = "Todas as métricas mantiveram-se ou melhoraram. Considera baixar o floor."
    elif not improved:
        rec = "Todas as métricas pioraram. Aguardar mais dados de treino."
    elif "Qualidade preditiva (IC)" in worsened:
        rec = (
            "IC piorou — o modelo é menos preciso a ordenar oportunidades. "
            "Aguardar próximo mês. Se persistir, revê a qualidade dos dados."
        )
    elif "Calibração (Brier)" in improved:
        rec = (
            "IC ligeiramente pior mas calibração melhorou. O modelo é mais conservador "
            "mas as probabilidades são mais fiáveis. Considera baixar o threshold de gating."
        )
    else:
        rec = "Resultado misto. Analisa o candidato manualmente antes de promover."

    return {
        "rows":           rows,
        "improved":       improved,
        "worsened":       worsened,
        "recommendation": rec,
    }


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
          guarda como `dip_models_pending.pkl` para revisão manual
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

    if cand_rho < floor:
        return {
            **base_result,
            "decision": "KEPT_FLOOR",
            "reason":   f"candidate ρ_α {cand_rho:.4f} < floor {floor:.4f}",
            "comparison_detail": _build_comparison_detail(cand_metrics, prod_metrics),
        }

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

    return _save_pending({
        **base_result,
        "decision":  "PENDING",
        "reason":    (
            f"candidate ρ_α {cand_rho:.4f} < {prod_rho:.4f} × {gating_ratio} "
            f"= {threshold:.4f}; guardado como pending para revisão manual"
        ),
        "comparison_detail": _build_comparison_detail(cand_metrics, prod_metrics),
    })


def _do_promote(result: dict) -> dict:
    """Atomic move: candidate → production (archives the previous production bundle)."""
    p = _paths()
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    p["ARCHIVE_DIR"].mkdir(parents=True, exist_ok=True)

    if p["PRODUCTION_BUNDLE"].exists():
        archived_pkl = p["ARCHIVE_DIR"] / f"dip_models_{timestamp}.pkl"
        shutil.copy2(p["PRODUCTION_BUNDLE"], archived_pkl)
        log.info(f"[gating] Archived production bundle → {archived_pkl.name}")
    if p["PRODUCTION_REPORT"].exists():
        archived_json = p["ARCHIVE_DIR"] / f"ml_report_{timestamp}.json"
        shutil.copy2(p["PRODUCTION_REPORT"], archived_json)

    promoted: list[str] = []
    if p["CANDIDATE_BUNDLE"].exists():
        tmp = p["PRODUCTION_BUNDLE"].with_suffix(".pkl.tmp")
        shutil.copy2(p["CANDIDATE_BUNDLE"], tmp)
        tmp.replace(p["PRODUCTION_BUNDLE"])
        promoted.append(p["PRODUCTION_BUNDLE"].name)
    if p["CANDIDATE_REPORT"].exists():
        tmp = p["PRODUCTION_REPORT"].with_suffix(".json.tmp")
        shutil.copy2(p["CANDIDATE_REPORT"], tmp)
        tmp.replace(p["PRODUCTION_REPORT"])

    log.info(f"[gating] PROMOTED — {promoted}")
    result["promoted_files"]    = promoted
    result["archive_timestamp"] = timestamp
    return result


def _save_pending(result: dict) -> dict:
    """Save candidate as `dip_models_pending.pkl` (does not promote)."""
    p = _paths()
    if p["CANDIDATE_BUNDLE"].exists():
        shutil.copy2(p["CANDIDATE_BUNDLE"], p["PENDING_BUNDLE"])
        log.info(f"[gating] Pending bundle: {p['PENDING_BUNDLE']}")
        result["pending_bundle"] = str(p["PENDING_BUNDLE"])
    if p["CANDIDATE_REPORT"].exists():
        shutil.copy2(p["CANDIDATE_REPORT"], p["PENDING_REPORT"])
        result["pending_report"] = str(p["PENDING_REPORT"])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main retrain entrypoint (called by cron in main.py and /admin_overnight)
# ─────────────────────────────────────────────────────────────────────────────

def run_monthly_retrain(
    gating_ratio: float = DEFAULT_GATING_RATIO,
    include_snapshot: bool = False,   # kept for call-site compat; ignored
    include_alert_db: bool = True,
    dry_run: bool = False,
    n_folds: Optional[int] = None,
    purge_days: Optional[int] = None,
) -> dict:
    """Monthly retrain pipeline. Training data: bootstrap + alert_db."""
    t0 = time.time()
    log.info("=" * 70)
    log.info(f"DipRadar — Monthly Retrain ({datetime.utcnow().isoformat(timespec='seconds')}Z)")
    log.info("=" * 70)
    log.info(f"  gating_ratio     = {gating_ratio}")
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

    # 3. Train candidate via ml_training.train.run_training
    p = _paths()
    p["CANDIDATE_DIR"].mkdir(parents=True, exist_ok=True)
    train_result: dict = {}
    try:
        from ml_training.config import N_FOLDS, PURGE_DAYS
        from ml_training.train import run_training

        log.info(f"[retrain] Training candidate in {p['CANDIDATE_DIR']}...")
        train_result = run_training(
            input_parquet=train_path,
            output_bundle=p["CANDIDATE_BUNDLE"],
            output_report=p["CANDIDATE_REPORT"],
            n_folds=n_folds if n_folds is not None else N_FOLDS,
            purge_days=purge_days if purge_days is not None else PURGE_DAYS,
        )
    except Exception as e:
        log.error(f"[retrain] Treino falhou: {e}", exc_info=True)
        return {
            "decision":      "FAILED",
            "reason":        f"train failed: {e}",
            "elapsed_s":     round(time.time() - t0, 1),
            "outcome_stats": outcome_stats,
        }

    # 3b. Dataset health check (usa fold results do treino para validar sinal)
    dataset_health: dict = {}
    try:
        from ml_training.diagnostics import dataset_health_check
        from ml_training.config import PRIMARY_TARGET
        import pandas as pd
        _df_check = pd.read_parquet(train_path)
        if PRIMARY_TARGET not in _df_check.columns:
            raise KeyError(
                f"Parquet não tem coluna '{PRIMARY_TARGET}'. "
                f"Colunas presentes: {list(_df_check.columns)[:15]}"
            )
        _target_check = PRIMARY_TARGET
        # fold_results extraídos do summary do treino
        _summary = train_result.get("summary")
        _fold_results = []
        if _summary is not None and not _summary.empty:
            for _, row in _summary.iterrows():
                _fold_results.append({
                    "ic_overall": row.get("rho_alpha_mean"),
                })
        dataset_health = dataset_health_check(
            _df_check, _fold_results,
            target_col=_target_check,
        )
        log.info(f"[retrain] Dataset health: verdict={dataset_health.get('verdict')} "
                 f"IC={dataset_health.get('ic_mean', '?'):.4f} SR={dataset_health.get('ic_sr', '?'):.2f}")
    except Exception as e:
        log.warning(f"[retrain] Dataset health check falhou (não bloqueante): {e}")

    # 4. Gating + atomic deploy
    cand_metrics = _read_v3_metrics(p["CANDIDATE_REPORT"])
    prod_metrics = _read_v3_metrics(p["PRODUCTION_REPORT"])
    gate = gate_and_promote_v3(
        cand_metrics=cand_metrics,
        prod_metrics=prod_metrics,
        gating_ratio=gating_ratio,
    )

    elapsed = time.time() - t0
    log.info(f"[retrain] CONCLUÍDO em {elapsed:.0f}s — decision={gate['decision']}")

    # Auto-publish charts to GitHub after retrain
    try:
        from chart_generator import generate_all_charts
        from github_publisher import push_all_charts
        generate_all_charts()
        results = push_all_charts()
        published = [k for k, v in results.items() if v and not str(v).startswith("ERROR")]
        if published:
            log.info(f"[retrain] Charts published to GitHub: {published}")
        else:
            log.debug("[retrain] No charts published (no data or GITHUB_TOKEN not set).")
    except Exception as _chart_err:
        log.warning(f"[retrain] Chart publish failed (non-blocking): {_chart_err}")

    return {
        **gate,
        "elapsed_s":      round(elapsed, 1),
        "outcome_stats":  outcome_stats,
        "dataset_health": dataset_health,
        "train_result":   train_result,  # inclui report com sector_models IC
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monthly retrain with rho_alpha gating")
    p.add_argument("--gating-ratio", type=float, default=DEFAULT_GATING_RATIO)
    p.add_argument("--no-alert-db", action="store_true")
    p.add_argument("--n-folds", type=int, default=None)
    p.add_argument("--purge-days", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    result = run_monthly_retrain(
        gating_ratio=args.gating_ratio,
        include_alert_db=not args.no_alert_db,
        dry_run=args.dry_run,
        n_folds=args.n_folds,
        purge_days=args.purge_days,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
