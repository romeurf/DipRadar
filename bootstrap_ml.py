"""
bootstrap_ml.py — Dual-Layer ML: Camada A (preço, 20 anos) + Camada B (fundamentais, 7 anos).

MODO AUTOMÁTICO (agendado pelo bot, corre às 02:00 UTC todos os dias):
    - Janela = [hoje - anos_config, ontem]
    - Registos fora da janela são eliminados automaticamente
    - Retreina ambas as camadas

MODO MANUAL (Railway CLI / Colab):
    python bootstrap_ml.py                          # tudo com defaults
    python bootstrap_ml.py --algo xgb              # XGBoost
    python bootstrap_ml.py --layer price           # só Camada A
    python bootstrap_ml.py --layer fund            # só Camada B
    python bootstrap_ml.py --years-price 20        # janela Camada A
    python bootstrap_ml.py --years-fund 7          # janela Camada B
    python bootstrap_ml.py --skip-backfill         # só treino
    python bootstrap_ml.py --force-full            # refaz backfill completo

OUTPUT:
    /data/dip_model_price.pkl    ← Camada A (só técnicas)
    /data/dip_model_stage1.pkl   ← Camada B stage 1 (win vs no-win)
    /data/dip_model_stage2.pkl   ← Camada B stage 2 (win40 vs win20)
    /data/ml_training_price.parquet   ← dados Camada A
    /data/ml_training_fund.parquet    ← dados Camada B
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [bootstrap_ml] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bootstrap_ml")

# ── Caminhos ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("/data") if Path("/data").exists() else Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

PKL_PRICE = DATA_DIR / "dip_model_price.pkl"   # Camada A
PKL_S1    = DATA_DIR / "dip_model_stage1.pkl"  # Camada B — win vs no-win
PKL_S2    = DATA_DIR / "dip_model_stage2.pkl"  # Camada B — win40 vs win20

PARQUET_PRICE = DATA_DIR / "ml_training_price.parquet"   # janela Camada A
PARQUET_FUND  = DATA_DIR / "ml_training_fund.parquet"    # janela Camada B

# ── Janelas deslizantes ────────────────────────────────────────────────────────
YEARS_PRICE = 20   # Camada A — preço puro
YEARS_FUND  = 7    # Camada B — fundamentais


def _window(years: int) -> tuple[date, date]:
    end   = date.today() - timedelta(days=1)
    start = end.replace(year=end.year - years)
    return start, end


# ── Features ──────────────────────────────────────────────────────────────────
# Camada A — só técnicas (disponíveis nos 20 anos de histórico)
FEATURES_PRICE: list[str] = [
    "rsi", "drawdown_pct", "change_day_pct",
    "beta", "spy_change", "sector_etf_change",
    "volume_ratio", "atr_pct",
]

# Camada B — técnicas + fundamentais
FEATURES_FUND: list[str] = [
    "rsi", "drawdown_pct", "change_day_pct",
    "pe_ratio", "pb_ratio", "fcf_yield", "analyst_upside",
    "revenue_growth", "gross_margin",
    "debt_to_equity", "beta", "short_pct",
    "spy_change", "sector_etf_change", "earnings_days",
    "market_cap_b", "dip_score",
]

# ── Universo ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    # Tech
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD","INTC","CRM",
    "ORCL","ADBE","QCOM","TXN","AVGO","MU","AMAT","LRCX","KLAC",
    "NOW","SNOW","PANW","CRWD","DDOG","NET","ZS","FTNT",
    # Financials
    "JPM","BAC","WFC","GS","MS","BLK","SCHW","AXP","V","MA",
    "C","USB","PNC","TFC","COF",
    # Healthcare
    "JNJ","UNH","PFE","ABBV","LLY","MRK","BMY","AMGN","GILD",
    "CVS","CI","HUM","ISRG","EW","BSX","MDT",
    # Consumer
    "WMT","COST","TGT","HD","LOW","MCD","SBUX","NKE","PG","KO",
    "PEP","PM","MO","MDLZ","GIS","CL",
    # Industrials
    "CAT","DE","HON","MMM","GE","RTX","LMT","NOC","BA","UPS",
    "FDX","CSX","UNP","EMR","ITW","ETN",
    # Energy
    "XOM","CVX","COP","EOG","SLB","MPC","VLO","OXY",
    # REITs / Utilities
    "AMT","PLD","EQIX","SPG","O","DLR","PSA",
    "NEE","DUK","SO","AEP","EXC","D","AWK",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def calc_atr_pct(hist: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR como % do preço de fecho."""
    high_low   = hist["High"] - hist["Low"]
    high_close = (hist["High"] - hist["Close"].shift()).abs()
    low_close  = (hist["Low"] - hist["Close"].shift()).abs()
    tr  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return (atr / hist["Close"]) * 100


def calc_volume_ratio(hist: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volume do dia / média dos últimos N dias."""
    avg = hist["Volume"].rolling(period).mean()
    return hist["Volume"] / (avg + 1e-9)


def outcome_label(ret: float) -> str:
    if   ret >= 40:  return "WIN_40"
    elif ret >= 20:  return "WIN_20"
    elif ret >= -15: return "NEUTRAL"
    else:            return "LOSS_15"


def get_price_near(hist: pd.DataFrame, target: date, window: int = 5) -> float | None:
    for d in range(-3, 6):
        check = target + timedelta(days=d)
        m = hist[hist.index.date == check]
        if not m.empty:
            return float(m["Close"].iloc[0])
    return None


def safe_float(val, default=None):
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


def simple_dip_score(r: dict) -> float:
    score = 50.0
    rsi = r.get("rsi") or 50
    if   rsi < 25: score += 20
    elif rsi < 35: score += 12
    elif rsi < 45: score += 5
    ddp = r.get("drawdown_pct") or 0
    if   ddp <= -40: score += 20
    elif ddp <= -25: score += 12
    elif ddp <= -15: score += 7
    elif ddp <= -10: score += 3
    chg = r.get("change_day_pct") or 0
    if   chg <= -8:  score += 15
    elif chg <= -5:  score += 9
    elif chg <= -3:  score += 4
    pe = r.get("pe_ratio") or 20
    if pe > 0:
        if   pe < 12: score += 10
        elif pe < 18: score += 5
        elif pe > 50: score -= 5
    up = r.get("analyst_upside") or 0
    if   up > 40: score += 10
    elif up > 20: score += 5
    return min(max(score, 0), 100)


# ── Backfill — Camada A (preço puro, 20 anos) ─────────────────────────────────

def backfill_price(
    start: date,
    end:   date,
    dip_thresh: float = 0.04,
    max_per_ticker: int = 10,
    existing_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Só features técnicas — sem chamadas a .info().
    Funciona com 20 anos de histórico no yfinance.
    """
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance não instalado")
        sys.exit(1)

    existing_keys: set[tuple] = set()
    if existing_df is not None and not existing_df.empty:
        existing_keys = set(
            zip(existing_df["symbol"].astype(str),
                existing_df["alert_date"].astype(str))
        )

    start_str  = start.isoformat()
    fetch_end  = min(end + timedelta(days=200), date.today()).isoformat()

    log.info(f"[CamadaA] Backfill preço: {start_str} → {end.isoformat()}")
    log.info(f"  Já existentes: {len(existing_keys)}")

    spy_hist = yf.Ticker("SPY").history(start=start_str, end=fetch_end, interval="1d")
    spy_hist["spy_ret"] = spy_hist["Close"].pct_change() * 100
    spy_map = {d.date(): float(r) for d, r in spy_hist["spy_ret"].items()}

    all_alerts: list[dict] = []

    for i, ticker in enumerate(UNIVERSE):
        try:
            hist = yf.Ticker(ticker).history(start=start_str, end=fetch_end, interval="1d")
            if hist.empty or len(hist) < 60:
                continue

            hist["rsi"]          = calc_rsi(hist["Close"])
            hist["ret_1d"]       = hist["Close"].pct_change() * 100
            hist["atr_pct"]      = calc_atr_pct(hist)
            hist["vol_ratio"]    = calc_volume_ratio(hist)
            roll_max             = hist["Close"].rolling(252, min_periods=30).max()
            hist["ddp"]          = (hist["Close"] - roll_max) / roll_max * 100

            # beta aproximado via correlação com SPY na janela de 252 dias
            spy_aligned = pd.Series(spy_map).reindex(
                [d.date() for d in hist.index], fill_value=np.nan
            )
            spy_aligned.index = hist.index
            cov = hist["ret_1d"].rolling(252).cov(spy_aligned)
            var = spy_aligned.rolling(252).var()
            hist["beta_roll"] = (cov / (var + 1e-9)).clip(-3, 5)

            mask = (
                (hist["ret_1d"] <= -(dip_thresh * 100)) &
                (hist["rsi"] < 55) &
                (hist.index >= pd.Timestamp(start_str)) &
                (hist.index <= pd.Timestamp(end.isoformat()))
            )
            dip_days = hist[mask]
            if dip_days.empty:
                continue

            selected = []
            last_dt  = None
            for dt, row in dip_days.iterrows():
                alert_date = dt.date()
                if (ticker, alert_date.isoformat()) in existing_keys:
                    continue
                if last_dt is None or (alert_date - last_dt).days >= 20:
                    selected.append((dt, row))
                    last_dt = alert_date
                if len(selected) >= max_per_ticker:
                    break

            for dt, row in selected:
                alert_date = dt.date()
                spy_chg    = spy_map.get(alert_date, 0.0)
                hist_after = hist[hist.index.date > alert_date]
                if hist_after.empty:
                    continue

                entry = float(row["Close"])
                p3m   = get_price_near(hist_after, alert_date + timedelta(days=91))
                p6m   = get_price_near(hist_after, alert_date + timedelta(days=182))
                if p3m is None and p6m is None:
                    continue

                r3m = (p3m - entry) / entry * 100 if p3m else None
                r6m = (p6m - entry) / entry * 100 if p6m else None
                ref = r6m if r6m is not None else r3m
                if ref is None:
                    continue

                all_alerts.append({
                    "symbol":         ticker,
                    "alert_date":     alert_date.isoformat(),
                    "price":          round(entry, 2),
                    "rsi":            round(safe_float(row["rsi"], 50), 1),
                    "drawdown_pct":   round(safe_float(row["ddp"], 0), 2),
                    "change_day_pct": round(float(row["ret_1d"]), 2),
                    "beta":           round(safe_float(row["beta_roll"], 1.0), 2),
                    "atr_pct":        round(safe_float(row["atr_pct"], 1.0), 2),
                    "volume_ratio":   round(safe_float(row["vol_ratio"], 1.0), 2),
                    "spy_change":     round(spy_chg, 2),
                    "sector_etf_change": round(spy_chg * 0.9, 2),
                    "return_3m":      round(r3m, 2) if r3m is not None else None,
                    "return_6m":      round(r6m, 2) if r6m is not None else None,
                    "outcome_label":  outcome_label(ref),
                })

            if (i + 1) % 20 == 0:
                log.info(f"  [{i+1}/{len(UNIVERSE)}] {len(all_alerts)} alertas")
            time.sleep(0.2)

        except Exception as e:
            log.warning(f"  ERRO {ticker}: {e}")

    log.info(f"[CamadaA] {len(all_alerts)} novos alertas")
    return pd.DataFrame(all_alerts) if all_alerts else pd.DataFrame()


# ── Backfill — Camada B (fundamentais, 7 anos) ───────────────────────────────

def backfill_fund(
    start: date,
    end:   date,
    dip_thresh: float = 0.04,
    max_per_ticker: int = 8,
    existing_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Features técnicas + fundamentais via yfinance .info().
    Janela mais curta (7 anos) onde os fundamentais são fiáveis.
    """
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance não instalado")
        sys.exit(1)

    existing_keys: set[tuple] = set()
    if existing_df is not None and not existing_df.empty:
        existing_keys = set(
            zip(existing_df["symbol"].astype(str),
                existing_df["alert_date"].astype(str))
        )

    start_str = start.isoformat()
    fetch_end = min(end + timedelta(days=200), date.today()).isoformat()

    log.info(f"[CamadaB] Backfill fundamentais: {start_str} → {end.isoformat()}")
    log.info(f"  Já existentes: {len(existing_keys)}")

    spy_hist = yf.Ticker("SPY").history(start=start_str, end=fetch_end, interval="1d")
    spy_hist["spy_ret"] = spy_hist["Close"].pct_change() * 100
    spy_map = {d.date(): float(r) for d, r in spy_hist["spy_ret"].items()}

    all_alerts: list[dict] = []

    for i, ticker in enumerate(UNIVERSE):
        try:
            tk   = yf.Ticker(ticker)
            hist = tk.history(start=start_str, end=fetch_end, interval="1d")
            if hist.empty or len(hist) < 60:
                continue

            info  = tk.info or {}
            hist["rsi"]    = calc_rsi(hist["Close"])
            hist["ret_1d"] = hist["Close"].pct_change() * 100
            roll_max       = hist["Close"].rolling(252, min_periods=30).max()
            hist["ddp"]    = (hist["Close"] - roll_max) / roll_max * 100

            pe    = safe_float(info.get("trailingPE") or info.get("forwardPE"))
            pb    = safe_float(info.get("priceToBook"))
            mcap  = safe_float(info.get("marketCap"), 0) / 1e9
            fcf   = safe_float(info.get("freeCashflow"))
            mc_raw = safe_float(info.get("marketCap"))
            fcfy  = (fcf / mc_raw * 100) if fcf and mc_raw else None
            revg  = safe_float(info.get("revenueGrowth"), 0) * 100
            gm    = safe_float(info.get("grossMargins"), 0) * 100
            de    = safe_float(info.get("debtToEquity"), 0) / 100
            beta  = safe_float(info.get("beta"), 1.0)
            short = safe_float(info.get("shortPercentOfFloat"), 0) * 100
            tgt   = safe_float(info.get("targetMeanPrice"))
            cur   = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"), 1)
            upside = ((tgt - cur) / cur * 100) if tgt and cur else 0.0

            mask = (
                (hist["ret_1d"] <= -(dip_thresh * 100)) &
                (hist["rsi"] < 55) &
                (hist.index >= pd.Timestamp(start_str)) &
                (hist.index <= pd.Timestamp(end.isoformat()))
            )
            dip_days = hist[mask]
            if dip_days.empty:
                continue

            selected = []
            last_dt  = None
            for dt, row in dip_days.iterrows():
                alert_date = dt.date()
                if (ticker, alert_date.isoformat()) in existing_keys:
                    continue
                if last_dt is None or (alert_date - last_dt).days >= 20:
                    selected.append((dt, row))
                    last_dt = alert_date
                if len(selected) >= max_per_ticker:
                    break

            for dt, row in selected:
                alert_date = dt.date()
                spy_chg    = spy_map.get(alert_date, 0.0)
                hist_after = hist[hist.index.date > alert_date]
                if hist_after.empty:
                    continue

                entry = float(row["Close"])
                p3m   = get_price_near(hist_after, alert_date + timedelta(days=91))
                p6m   = get_price_near(hist_after, alert_date + timedelta(days=182))
                if p3m is None and p6m is None:
                    continue

                r3m = (p3m - entry) / entry * 100 if p3m else None
                r6m = (p6m - entry) / entry * 100 if p6m else None
                ref = r6m if r6m is not None else r3m
                if ref is None:
                    continue

                feat: dict = {
                    "rsi":              round(safe_float(row["rsi"], 50), 1),
                    "drawdown_pct":     round(safe_float(row["ddp"], 0), 2),
                    "change_day_pct":   round(float(row["ret_1d"]), 2),
                    "pe_ratio":         round(pe, 1) if pe else None,
                    "pb_ratio":         round(pb, 2) if pb else None,
                    "fcf_yield":        round(fcfy, 4) if fcfy else None,
                    "analyst_upside":   round(upside, 1),
                    "revenue_growth":   round(revg, 2),
                    "gross_margin":     round(gm, 2),
                    "debt_to_equity":   round(de, 2),
                    "beta":             round(beta, 2),
                    "short_pct":        round(short, 2),
                    "spy_change":       round(spy_chg, 2),
                    "sector_etf_change": round(spy_chg * 0.9, 2),
                    "earnings_days":    90,
                    "market_cap_b":     round(mcap, 2),
                }
                feat["dip_score"]     = round(simple_dip_score(feat), 1)
                feat["symbol"]        = ticker
                feat["alert_date"]    = alert_date.isoformat()
                feat["price"]         = round(entry, 2)
                feat["return_3m"]     = round(r3m, 2) if r3m is not None else None
                feat["return_6m"]     = round(r6m, 2) if r6m is not None else None
                feat["outcome_label"] = outcome_label(ref)
                all_alerts.append(feat)

            if (i + 1) % 20 == 0:
                log.info(f"  [{i+1}/{len(UNIVERSE)}] {len(all_alerts)} alertas")
            time.sleep(0.3)

        except Exception as e:
            log.warning(f"  ERRO {ticker}: {e}")

    log.info(f"[CamadaB] {len(all_alerts)} novos alertas")
    return pd.DataFrame(all_alerts) if all_alerts else pd.DataFrame()


# ── Janela deslizante: carrega, purga e faz merge ─────────────────────────────

def load_and_slide(
    parquet: Path,
    start: date,
    new_df: pd.DataFrame,
) -> pd.DataFrame:
    start_str = start.isoformat()

    if parquet.exists():
        existing = pd.read_parquet(parquet)
        rows_before = len(existing)
        existing["alert_date"] = existing["alert_date"].astype(str)
        existing = existing[existing["alert_date"] >= start_str].copy()
        purged = rows_before - len(existing)
        if purged > 0:
            log.info(f"🗑  Purgados {purged} registos fora da janela (< {start_str})")
    else:
        existing = pd.DataFrame()

    if new_df.empty and existing.empty:
        log.error("Sem dados — Parquet vazio e backfill sem resultados.")
        sys.exit(1)

    if new_df.empty:
        combined = existing
    elif existing.empty:
        combined = new_df
    else:
        combined = pd.concat([existing, new_df], ignore_index=True)

    combined["alert_date"] = combined["alert_date"].astype(str)
    combined.drop_duplicates(subset=["symbol", "alert_date"], keep="last", inplace=True)
    combined.sort_values("alert_date", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    combined.to_parquet(parquet, index=False)
    log.info(f"📦 Parquet: {len(combined)} registos → {parquet}")
    return combined


# ── Pipeline de treino ────────────────────────────────────────────────────────

def _build_pipeline(algo: str = "rf"):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]
    if algo == "rf":
        clf = RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
    elif algo == "xgb":
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                eval_metric="logloss", verbosity=0,
            )
        except ImportError:
            log.warning("xgboost não instalado — a usar GradientBoosting")
            clf = GradientBoostingClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
    else:
        raise ValueError(f"Algoritmo desconhecido: {algo}")
    steps.append(("clf", clf))
    return Pipeline(steps)


def _train_layer(df: pd.DataFrame, features: list[str], pkl_s1: Path, pkl_s2: Path | None, algo: str, label: str) -> None:
    from sklearn.metrics import average_precision_score, classification_report

    df2 = df[df["outcome_label"].notna()].copy()
    df2["target_s1"] = df2["outcome_label"].apply(
        lambda x: 1 if x in ("WIN_40", "WIN_20") else 0
    )

    if len(df2) < 30 or df2["target_s1"].sum() < 10:
        log.error(f"[{label}] Dados insuficientes: {len(df2)} linhas, {int(df2['target_s1'].sum())} wins")
        return

    for col in features:
        if col not in df2.columns:
            df2[col] = np.nan

    df2 = df2.sort_values("alert_date").reset_index(drop=True)
    split    = int(len(df2) * 0.80)
    train_df = df2.iloc[:split]
    test_df  = df2.iloc[split:]

    X_tr = train_df[features].values.astype(np.float32)
    y_tr = train_df["target_s1"].values
    X_te = test_df[features].values.astype(np.float32)
    y_te = test_df["target_s1"].values

    log.info(f"[{label}] {algo.upper()} | train={len(X_tr)} test={len(X_te)} wins={y_tr.sum()}")

    pipe = _build_pipeline(algo)
    if algo == "xgb":
        ratio = max((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 1.0)
        pipe.named_steps["clf"].set_params(scale_pos_weight=ratio)
    pipe.fit(X_tr, y_tr)

    probs  = pipe.predict_proba(X_te)[:, 1]
    y_pred = (probs >= 0.50).astype(int)
    auc_pr = average_precision_score(y_te, probs)

    log.info(f"[{label}] AUC-PR: {auc_pr:.4f}")
    log.info("\n" + classification_report(y_te, y_pred, target_names=["NO_WIN", "WIN"], digits=3))

    bundle = {
        "model":           pipe,
        "feature_columns": features,
        "threshold":       0.50,
        "algorithm":       algo,
        "auc_pr":          round(auc_pr, 4),
        "n_samples":       int(len(X_tr)),
        "train_date":      datetime.now().strftime("%Y-%m-%d %H:%M"),
        "layer":           label,
    }
    with open(pkl_s1, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"✅ {pkl_s1}  ({pkl_s1.stat().st_size / 1024:.0f} KB)")

    # Stage 2
    if pkl_s2 is not None:
        wins_tr = train_df[train_df["outcome_label"].isin(["WIN_40", "WIN_20"])].copy()
        wins_tr["target_s2"] = (wins_tr["outcome_label"] == "WIN_40").astype(int)
        if len(wins_tr) >= 30:
            pipe2 = _build_pipeline(algo)
            pipe2.fit(
                wins_tr[features].values.astype(np.float32),
                wins_tr["target_s2"].values,
            )
            bundle2 = {
                "model":           pipe2,
                "feature_columns": features,
                "threshold":       0.55,
                "algorithm":       algo,
                "n_samples":       len(wins_tr),
                "train_date":      datetime.now().strftime("%Y-%m-%d %H:%M"),
                "layer":           label,
            }
            with open(pkl_s2, "wb") as f:
                pickle.dump(bundle2, f, protocol=pickle.HIGHEST_PROTOCOL)
            log.info(f"✅ {pkl_s2}  ({pkl_s2.stat().st_size / 1024:.0f} KB)")
        else:
            log.info(f"[{label}] Stage 2 saltado ({len(wins_tr)} wins < 30)")


# ── Ponto de entrada público (scheduler do bot) ───────────────────────────────

def run_auto() -> None:
    log.info("=" * 55)
    log.info("AUTO RUN — dual-layer ML")

    try:
        import sklearn  # noqa: F401
    except ImportError:
        log.error("scikit-learn não instalado")
        return

    # Camada A — preço 20 anos
    start_p, end_p = _window(YEARS_PRICE)
    existing_p = pd.read_parquet(PARQUET_PRICE) if PARQUET_PRICE.exists() else pd.DataFrame()
    new_p  = backfill_price(start=start_p, end=end_p, existing_df=existing_p)
    df_p   = load_and_slide(PARQUET_PRICE, start_p, new_p)
    _train_layer(df_p, FEATURES_PRICE, PKL_PRICE, None, "rf", "CamadaA")

    # Camada B — fundamentais 7 anos
    start_f, end_f = _window(YEARS_FUND)
    existing_f = pd.read_parquet(PARQUET_FUND) if PARQUET_FUND.exists() else pd.DataFrame()
    new_f  = backfill_fund(start=start_f, end=end_f, existing_df=existing_f)
    df_f   = load_and_slide(PARQUET_FUND, start_f, new_f)
    _train_layer(df_f, FEATURES_FUND, PKL_S1, PKL_S2, "rf", "CamadaB")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DipRadar — Dual-Layer ML (Camada A: preço 20a | Camada B: fundamentais 7a)"
    )
    p.add_argument("--algo",         choices=["rf", "xgb"], default="rf")
    p.add_argument("--layer",        choices=["all", "price", "fund"], default="all")
    p.add_argument("--years-price",  type=int, default=YEARS_PRICE)
    p.add_argument("--years-fund",   type=int, default=YEARS_FUND)
    p.add_argument("--dip-thresh",   type=float, default=0.04)
    p.add_argument("--skip-backfill",action="store_true")
    p.add_argument("--force-full",   action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        import sklearn  # noqa: F401
    except ImportError:
        log.error("scikit-learn não instalado")
        sys.exit(1)

    run_price = args.layer in ("all", "price")
    run_fund  = args.layer in ("all", "fund")

    # ── Camada A ──────────────────────────────────────────────────────────────
    if run_price:
        start_p, end_p = _window(args.years_price)
        log.info(f"[CamadaA] Janela: {start_p} → {end_p}")
        if args.skip_backfill:
            df_p = pd.read_parquet(PARQUET_PRICE) if PARQUET_PRICE.exists() else pd.DataFrame()
        else:
            existing_p = pd.DataFrame() if args.force_full else (
                pd.read_parquet(PARQUET_PRICE) if PARQUET_PRICE.exists() else pd.DataFrame()
            )
            new_p = backfill_price(
                start=start_p, end=end_p,
                dip_thresh=args.dip_thresh,
                existing_df=existing_p,
            )
            df_p = load_and_slide(PARQUET_PRICE, start_p, new_p)
        _train_layer(df_p, FEATURES_PRICE, PKL_PRICE, None, args.algo, "CamadaA")

    # ── Camada B ──────────────────────────────────────────────────────────────
    if run_fund:
        start_f, end_f = _window(args.years_fund)
        log.info(f"[CamadaB] Janela: {start_f} → {end_f}")
        if args.skip_backfill:
            df_f = pd.read_parquet(PARQUET_FUND) if PARQUET_FUND.exists() else pd.DataFrame()
        else:
            existing_f = pd.DataFrame() if args.force_full else (
                pd.read_parquet(PARQUET_FUND) if PARQUET_FUND.exists() else pd.DataFrame()
            )
            new_f = backfill_fund(
                start=start_f, end=end_f,
                dip_thresh=args.dip_thresh,
                existing_df=existing_f,
            )
            df_f = load_and_slide(PARQUET_FUND, start_f, new_f)
        _train_layer(df_f, FEATURES_FUND, PKL_S1, PKL_S2, args.algo, "CamadaB")

    log.info("=" * 55)
    log.info("TREINO CONCLUÍDO")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
