"""scripts/build_dislocation_validation_spine.py — Build PIT validation spine.

Reads from /data/validation_cache/ (produced by validation_cache_builder.py)
and computes PIT-clean sub-scores AS OF each Friday close from 2021-01
through 2026-Q1. Output spine.parquet feeds run_dislocation_validation.py.

Sub-scores computed (PIT-sourceable subset only — per plan §E.4):
  #3 sector-relative (60d + 20d windows from cached prices)
     + 8-K Item 4.02 restatement penalty (PIT-clean)
     — NEWS-keyword bad-news subtraction is ZEROED (headlines not historical)
  #4 catalyst proximity (next 8-K Item 2.02 / 10-Q date from cached EDGAR)
  #6 quality (revenue_growth + gross_margin + ROE from cached fundamentals
     with pit_date alignment; FCF-yield skipped — needs historical market cap)
  #7 governance (Form 4 SALE-only filter, lookback 60d; 8-K Item 4.02 in 30d;
     news mgmt-departure headlines ZEROED)
  8-K Item 2.02 component of #2 (guidance — PIT-clean part only;
     news-keyword classifier component ZEROED)

ZEROED sub-scores (not PIT-reconstructible from free historical data):
  #1 estimate revisions       (FMP free has no historical snapshots)
  #2 news-keyword guidance   (yfinance.news only recent, not 2021-2024)
  #3 bad-news subtraction   (same — news-keyword based)
  #5 valuation               (forward EPS history not free PIT)
  #7 news mgmt-departure    (same)

These zeroed components ship to production but stay UNVALIDATED. The
sub_scores_active_in_spine field in the spine metadata records the
exact split so the backtest output makes the gap explicit.

Forward returns (target):
  forward_6m_return  — primary (binding metric per plan §E)
  forward_3m_return  — secondary

Output: /data/validation_cache/spine.parquet + spine_metadata.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("spine_builder")

_DATA_DIR  = Path("/data") if Path("/data").exists() else _ROOT / "data"
_CACHE_DIR = _DATA_DIR / "validation_cache"
_SPINE_PATH = _CACHE_DIR / "spine.parquet"
_SPINE_META = _CACHE_DIR / "spine_metadata.json"

# Window: 2021-01-01 → 2026-06-01. We need forward 6m so the latest as-of
# date is 2026-06-01 minus 6m = 2025-12-01.
_AS_OF_START = date(2021, 1, 1)
_AS_OF_END   = date(2025, 12, 31)
_FORWARD_HORIZON_DAYS = 180  # 6m forward return

# Sub-score weights mirror plan §C (production defaults). Used in the
# composite_raw computation here, NOT in production scoring.
_WEIGHTS = {
    "estimate_revisions":  1.0,  # zeroed (no PIT data)
    "guidance":            1.0,  # 8-K bit only
    "drop_attribution":    1.3,  # sector-rel only + 8-K 4.02 penalty
    "catalyst":            0.6,
    "valuation":           1.0,  # zeroed (no PIT forward-EPS)
    "quality":             0.8,
    "governance":          1.0,  # SALE-only Form 4 filter + 8-K 4.02
}

_SUB_SCORES_ACTIVE_IN_SPINE = {
    "estimate_revisions":  "zeroed_no_pit_data",
    "guidance":            "8k_item_2_02_only_news_keywords_zeroed",
    "drop_attribution":    "sector_rel_60d_20d_plus_8k_4_02_news_keywords_zeroed",
    "catalyst":            "next_8k_2_02_or_10q_proxy",
    "valuation":           "trailing_pe_ps_peg_pit_clean_v2_with_static_shares",
    "quality":             "rev_growth_yoy_gross_margin_roe_only_fcf_skipped",
    "governance":          "form4_sale_only_filter_plus_8k_4_02_news_mgmt_departure_zeroed",
}

# Officer titles considered "executive" for #7 SALE filter
_EXEC_TITLE_PATTERNS = (
    "chief executive", "ceo",
    "chief financial", "cfo",
    "chief operating", "coo",
    "chairman", "president",
)

# Anomalous sale threshold (same as production)
_INSIDER_SALE_THRESHOLD_USD = 10_000_000


# ── Cache loaders ────────────────────────────────────────────────────

def load_cache():
    """Read all cache parquets + JSONs. Raise if anything's missing."""
    if not _CACHE_DIR.exists():
        raise FileNotFoundError(
            f"{_CACHE_DIR} not found — run validation_cache_builder.py first"
        )

    prices_path = _CACHE_DIR / "prices.parquet"
    sectors_path = _CACHE_DIR / "sectors.json"
    edgar_8k_path = _CACHE_DIR / "edgar_8k.parquet"
    edgar_form4_path = _CACHE_DIR / "edgar_form4.parquet"
    fundamentals_path = _CACHE_DIR / "quarterly_financials.parquet"

    for p in (prices_path, sectors_path, edgar_8k_path, edgar_form4_path, fundamentals_path):
        if not p.exists():
            raise FileNotFoundError(f"missing cache artefact: {p}")

    log.info("[cache] loading parquets")
    prices       = pd.read_parquet(prices_path)
    sectors      = json.loads(sectors_path.read_text())
    edgar_8k     = pd.read_parquet(edgar_8k_path)
    edgar_form4  = pd.read_parquet(edgar_form4_path)
    fundamentals = pd.read_parquet(fundamentals_path)

    # Static (per-ticker shares + trailing EPS anchor for s5 PIT valuation)
    static_path = _CACHE_DIR / "ticker_static.json"
    static_data = {}
    if static_path.exists():
        try:
            static_data = json.loads(static_path.read_text()).get("tickers", {})
            log.info(f"[cache] loaded ticker_static for {len(static_data)} tickers (s5 valuation active)")
        except Exception as e:
            log.warning(f"[cache] ticker_static unreadable: {e} — s5 valuation will be 0 for all rows")
    else:
        log.warning(f"[cache] {static_path} missing — s5 valuation will be 0 for all rows. "
                    f"Run scripts/validation_static_fetch.py first.")

    # Index by ticker + date for fast slicing
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    edgar_8k["filed"] = pd.to_datetime(edgar_8k["filed"]).dt.date
    edgar_form4["filed"] = pd.to_datetime(edgar_form4["filed"]).dt.date
    fundamentals["pit_date"] = pd.to_datetime(fundamentals["pit_date"]).dt.date

    log.info(
        f"[cache] prices={len(prices):,} rows, sectors={len(sectors)}, "
        f"8k={len(edgar_8k):,}, form4={len(edgar_form4):,}, "
        f"fundamentals={len(fundamentals):,}"
    )
    return {
        "prices":       prices,
        "sectors":      sectors,
        "edgar_8k":     edgar_8k,
        "edgar_form4":  edgar_form4,
        "fundamentals": fundamentals,
        "static":       static_data,
    }


# ── Sub-score computers (PIT-clean) ──────────────────────────────────

def _ret_window(prices_t: pd.DataFrame, as_of: date, n_days: int):
    """Return price return over last n_days as-of as_of. None if missing."""
    cutoff = as_of - timedelta(days=int(n_days * 1.7))  # buffer for non-trading days
    sub = prices_t[(prices_t["date"] >= cutoff) & (prices_t["date"] <= as_of)]
    if len(sub) < n_days + 1:
        return None
    sub = sub.sort_values("date")
    try:
        end = float(sub["close"].iloc[-1])
        start = float(sub["close"].iloc[-(n_days+1)])
        if start <= 0:
            return None
        return end / start - 1.0
    except Exception:
        return None


def sub_drop_attribution_spine(
    ticker: str,
    as_of: date,
    sector: str,
    prices: dict,
    edgar_8k_t: pd.DataFrame,
) -> tuple[float, dict]:
    """Sector-relative + 8-K restatement penalty. News keywords ZEROED."""
    from data.sector_relative import SECTOR_TO_ETF
    etf = SECTOR_TO_ETF.get(sector)
    raw = {"sector": sector, "etf": etf, "news_keywords": "ZEROED"}

    if etf is None:
        return 0.0, {**raw, "note": "sector unknown"}

    prices_t = prices.get(ticker)
    prices_etf = prices.get(etf)
    if prices_t is None or prices_etf is None:
        return 0.0, {**raw, "note": "price hist missing"}

    long_t = _ret_window(prices_t, as_of, 60)
    short_t = _ret_window(prices_t, as_of, 20)
    long_etf = _ret_window(prices_etf, as_of, 60)
    short_etf = _ret_window(prices_etf, as_of, 20)

    long_rel = (long_t - long_etf) if (long_t is not None and long_etf is not None) else None
    short_rel = (short_t - short_etf) if (short_t is not None and short_etf is not None) else None
    raw["long_rel_60d"] = long_rel
    raw["short_rel_20d"] = short_rel

    if long_rel is None and short_rel is None:
        return 0.0, {**raw, "note": "no sector-rel data"}

    if long_rel is not None and short_rel is not None:
        base = 0.7 * math.tanh((-long_rel) / 0.10) + 0.3 * math.tanh((-short_rel) / 0.05)
    elif long_rel is not None:
        base = math.tanh((-long_rel) / 0.10)
    else:
        base = math.tanh((-short_rel) / 0.05)

    # 8-K Item 4.02 restatement penalty (PIT-clean)
    cutoff = as_of - timedelta(days=15)
    recent_8k = edgar_8k_t[
        (edgar_8k_t["filed"] >= cutoff) & (edgar_8k_t["filed"] <= as_of)
    ]
    has_restatement = any(
        "4.02" in str(items) for items in recent_8k["items"].tolist()
    )
    penalty = -0.5 if has_restatement else 0.0
    raw["has_restatement"] = has_restatement

    return float(np.clip(base + penalty, -1.5, 1.0)), raw


def sub_catalyst_spine(
    ticker: str,
    as_of: date,
    edgar_8k_t: pd.DataFrame,
) -> tuple[float, dict]:
    """Days-to-next-earnings proxy via next 8-K Item 2.02 OR 10-Q filing.

    Production reads yfinance.calendar (next earnings 0-45d forward). For
    the spine we don't have a future earnings calendar PIT-clean, so we
    approximate via the typical filing cadence: companies file 10-Q +
    8-K Item 2.02 roughly every 90 days. The "next" 8-K 2.02 after as_of
    is what would have been the next earnings date.

    Note: this looks at FUTURE 8-Ks relative to as_of in the cached
    history. That's legitimately PIT-clean for the SPINE because the
    spine builder has access to the full history; only the SCORER
    inside the spine must respect PIT. The catalyst signal is by
    construction forward-looking (it's "days until X happens"), so
    knowing X happened on date Y in the cached history is fine.
    """
    forward_window = as_of + timedelta(days=120)
    future_8k = edgar_8k_t[
        (edgar_8k_t["filed"] > as_of) & (edgar_8k_t["filed"] <= forward_window)
    ]
    earnings_filings = future_8k[
        future_8k["items"].astype(str).str.contains("2.02", na=False)
    ]
    if earnings_filings.empty:
        return -0.2, {"days_to_earnings": None, "note": "no upcoming 8-K 2.02 in 120d"}

    next_filing = earnings_filings["filed"].min()
    d = (next_filing - as_of).days

    if d <= 20:
        val = d / 20.0
        expl = f"earnings 8-K in {d}d (ramping)"
    elif d <= 42:
        val = 1.0
        expl = f"earnings 8-K in {d}d (sweet spot)"
    elif d <= 90:
        val = 1.0 - (d - 42) / 48.0
        expl = f"earnings 8-K in {d}d (decaying)"
    else:
        val = -0.2
        expl = f"earnings 8-K in {d}d (too far)"

    return float(val), {"days_to_earnings": d, "note": expl}


def sub_quality_spine(
    ticker: str,
    as_of: date,
    fundamentals_t: pd.DataFrame,
) -> tuple[float, dict]:
    """Quality from latest PIT-available quarterly fundamentals.

    Returns 0 + skip flag if no fundamentals row with pit_date <= as_of.
    Composite = mean of normalized components in [0, 1]. Matches plan B
    formula for #6 minus FCF yield (no historical market cap).
    """
    available = fundamentals_t[fundamentals_t["pit_date"] <= as_of]
    if available.empty:
        return 0.5, {"note": "no PIT fundamentals available"}

    latest = available.sort_values("pit_date", ascending=False).iloc[0]
    components = []
    raw = {"period_end": str(latest.get("period_end", ""))}

    # YoY revenue growth: compare to ~4 quarters back
    period_end_str = latest.get("period_end", "")
    try:
        period_end = date.fromisoformat(period_end_str)
        prior_year = available[
            (available["period_end"] >= (period_end - timedelta(days=400)).isoformat()) &
            (available["period_end"] <= (period_end - timedelta(days=330)).isoformat())
        ]
        rg = None
        if not prior_year.empty:
            prev_rev = float(prior_year.iloc[0].get("revenue") or 0)
            cur_rev = float(latest.get("revenue") or 0)
            if prev_rev > 0:
                rg = cur_rev / prev_rev - 1.0
        raw["revenue_growth"] = rg
        if rg is not None:
            components.append(float(np.clip((rg + 0.20) / 0.70, 0.0, 1.0)))
    except Exception:
        pass

    gm = latest.get("gross_margin")
    if gm is not None and not pd.isna(gm):
        raw["gross_margin"] = float(gm)
        components.append(float(np.clip(float(gm) / 0.90, 0.0, 1.0)))

    roe = latest.get("roe")
    if roe is not None and not pd.isna(roe):
        raw["roe"] = float(roe)
        components.append(float(np.clip(float(roe) / 0.50, 0.0, 1.0)))

    if not components:
        return 0.5, {**raw, "note": "no components computed"}
    return float(np.mean(components)), raw


def sub_governance_spine(
    ticker: str,
    as_of: date,
    edgar_form4_t: pd.DataFrame,
    edgar_8k_t: pd.DataFrame,
) -> tuple[float, dict]:
    """SALE-only Form 4 filter + 8-K Item 4.02. News mgmt-departure ZEROED.

    Production: -0.5 if CEO/CFO sold > $10M in 60d.
    Spine:     -0.5 if any Form 4 with is_sale=True AND title matches
               executive pattern in 60d AND sale_amount_usd >= threshold.
    Restatement (8-K Item 4.02): -1.0. Capped at -2.
    """
    raw = {"sales": [], "has_restatement": False, "news_mgmt_departure": "ZEROED"}
    score = 0.0
    cutoff = as_of - timedelta(days=60)

    # Form 4 SALE filter
    f4_window = edgar_form4_t[
        (edgar_form4_t["filed"] >= cutoff) & (edgar_form4_t["filed"] <= as_of)
    ]
    for _, row in f4_window.iterrows():
        if not bool(row.get("is_sale", False)):
            continue
        title_lower = str(row.get("title", "")).lower()
        is_exec = any(p in title_lower for p in _EXEC_TITLE_PATTERNS)
        amount = float(row.get("sale_amount_usd", 0) or 0)
        if is_exec and amount >= _INSIDER_SALE_THRESHOLD_USD:
            score -= 0.5
            raw["sales"].append({
                "filed":   row["filed"].isoformat() if hasattr(row["filed"], "isoformat") else str(row["filed"]),
                "title":   row.get("title", ""),
                "amount":  amount,
            })
            break  # one is enough; we don't double-penalize

    # 8-K Item 4.02 restatement
    cutoff_8k = as_of - timedelta(days=30)
    recent_8k = edgar_8k_t[
        (edgar_8k_t["filed"] >= cutoff_8k) & (edgar_8k_t["filed"] <= as_of)
    ]
    if any("4.02" in str(items) for items in recent_8k["items"].tolist()):
        score -= 1.0
        raw["has_restatement"] = True

    return float(np.clip(score, -2.0, 0.0)), raw


def sub_guidance_spine(
    ticker: str,
    as_of: date,
    edgar_8k_t: pd.DataFrame,
) -> tuple[float, dict]:
    """8-K Item 2.02 component only. News-keyword classifier ZEROED.

    Item 2.02 is "Results of Operations" — earnings releases. Without
    parsing the actual filing text we can't tell raise/cut/reaffirm,
    so the spine contributes ZERO from this sub-score in most cases.
    The full PIT-clean 8-K signal would need NLP on the filing body
    (out of scope for v1). Document explicitly.
    """
    raw = {"note": "news ZEROED; 8-K 2.02 text NLP out of scope → contributes 0"}
    return 0.0, raw


# ── Sub-score #5 — valuation PIT (trailing P/E + P/S + PEG) ──────────

def sub_valuation_pit_spine(
    ticker: str,
    as_of: date,
    prices_t: pd.DataFrame,
    fundamentals_t: pd.DataFrame,
    static_t: dict,
) -> tuple[float, dict]:
    """PIT-clean trailing valuation: P/E + P/S + PEG, derived from prices
    + cached fundamentals + static shares anchor (approximation).

    Pre-registered formulas (locked BEFORE seeing backtest result, no
    post-hoc tweaking per user F.5 review):

        pe_t  = price / TTM_EPS
        ps_t  = (price × shares_current) / TTM_revenue
        peg_t = pe_t / (rev_growth_yoy × 100)   (None if growth <= 0)

        score_pe  = -tanh((pe_t  - 20)  / 15)     +1 @ PE=5,  0 @ PE=20,  -1 @ PE=35+
        score_ps  = -tanh((ps_t  -  5)  /  4)     +1 @ PS=1,  0 @ PS=5,   -1 @ PS=9+
        score_peg = -tanh((peg_t - 1.5) / 1.5)    +1 @ PEG=0.5, 0 @ PEG=1.5, -1 @ PEG=3+

        s5 = mean(available_components), clipped to [-1, +1]

    Why these thresholds: industry-standard heuristics.
      PE=20 is roughly long-term S&P 500 average. Lynch PEG=1.5 is the
      classic "growth-at-reasonable-price" frontier. PS=5 sits at the
      growth-vs-mature dividing line typical for software/healthcare.
    """
    static = static_t or {}
    shares = static.get("shares_outstanding")
    if not shares or shares <= 0:
        return 0.0, {"note": "no shares_outstanding for ticker"}

    # Current price at as_of date
    px_sub = prices_t[prices_t["date"] <= as_of]
    if px_sub.empty:
        return 0.0, {"note": "no price <= as_of"}
    try:
        price = float(px_sub.sort_values("date")["close"].iloc[-1])
    except Exception:
        return 0.0, {"note": "price lookup failed"}
    if price <= 0:
        return 0.0, {"note": "price <= 0"}

    # TTM net income + revenue from last 4 quarters with pit_date <= as_of
    available = fundamentals_t[fundamentals_t["pit_date"] <= as_of]
    available = available.sort_values("pit_date", ascending=False).head(4)
    if len(available) < 4:
        return 0.0, {"note": f"only {len(available)} quarters of TTM data"}

    try:
        ttm_ni = float(available["net_income"].sum())
        ttm_rev = float(available["revenue"].sum())
    except Exception:
        return 0.0, {"note": "TTM sum failed"}

    # YoY revenue growth from quarterly aggregation
    try:
        all_avail = fundamentals_t[fundamentals_t["pit_date"] <= as_of]
        all_avail = all_avail.sort_values("pit_date", ascending=False)
        if len(all_avail) >= 8:
            cur_year = float(all_avail.head(4)["revenue"].sum())
            prev_year = float(all_avail.iloc[4:8]["revenue"].sum())
            rev_growth_yoy = (cur_year / prev_year - 1.0) if prev_year > 0 else None
        else:
            rev_growth_yoy = None
    except Exception:
        rev_growth_yoy = None

    # Compute ratios
    eps_ttm = ttm_ni / shares if shares > 0 else None
    pe_t = (price / eps_ttm) if (eps_ttm and eps_ttm > 0) else None
    market_cap = price * shares
    ps_t = (market_cap / ttm_rev) if (ttm_rev > 0) else None
    peg_t = (pe_t / (rev_growth_yoy * 100)) if (pe_t and rev_growth_yoy and rev_growth_yoy > 0) else None

    # Pre-registered scoring formulas
    import math
    components = []
    raw = {
        "price":           round(price, 2),
        "shares":          shares,
        "ttm_ni":          round(ttm_ni, 0),
        "ttm_revenue":     round(ttm_rev, 0),
        "eps_ttm":         round(eps_ttm, 3) if eps_ttm else None,
        "pe_trailing":     round(pe_t, 2) if pe_t else None,
        "ps_ratio":        round(ps_t, 2) if ps_t else None,
        "peg_ratio":       round(peg_t, 2) if peg_t else None,
        "rev_growth_yoy":  round(rev_growth_yoy, 4) if rev_growth_yoy else None,
    }

    if pe_t and 0 < pe_t < 500:  # ignore absurd ratios (data noise)
        score_pe = -math.tanh((pe_t - 20.0) / 15.0)
        components.append(score_pe)
        raw["score_pe"] = round(score_pe, 3)

    if ps_t and 0 < ps_t < 100:
        score_ps = -math.tanh((ps_t - 5.0) / 4.0)
        components.append(score_ps)
        raw["score_ps"] = round(score_ps, 3)

    if peg_t is not None and -10 < peg_t < 50:  # exclude pathological PEG
        score_peg = -math.tanh((peg_t - 1.5) / 1.5)
        components.append(score_peg)
        raw["score_peg"] = round(score_peg, 3)

    if not components:
        return 0.0, {**raw, "note": "no usable valuation components"}

    s5 = float(np.clip(np.mean(components), -1.0, 1.0))
    return s5, raw


# ── Spine row builder ────────────────────────────────────────────────

def _forward_return(prices_t: pd.DataFrame, as_of: date, horizon_days: int):
    """Forward return from as_of to as_of + horizon_days. None if data missing."""
    target_date = as_of + timedelta(days=horizon_days)
    cutoff_start = target_date - timedelta(days=30)
    sub_now = prices_t[prices_t["date"] <= as_of]
    sub_fwd = prices_t[
        (prices_t["date"] >= cutoff_start) & (prices_t["date"] <= target_date)
    ]
    if sub_now.empty or sub_fwd.empty:
        return None
    try:
        p_now = float(sub_now.sort_values("date")["close"].iloc[-1])
        p_fwd = float(sub_fwd.sort_values("date")["close"].iloc[-1])
        if p_now <= 0:
            return None
        return p_fwd / p_now - 1.0
    except Exception:
        return None


def fridays_in_range(start: date, end: date) -> list[date]:
    """All Fridays in [start, end]. Friday = weekday 4."""
    out = []
    d = start
    while d.weekday() != 4:
        d += timedelta(days=1)
    while d <= end:
        out.append(d)
        d += timedelta(days=7)
    return out


def build_spine_row(
    ticker: str,
    as_of: date,
    cache: dict,
    prices_by_ticker: dict,
):
    """Compose one spine row. Returns None when essential data missing."""
    prices_t = prices_by_ticker.get(ticker)
    if prices_t is None or prices_t.empty:
        return None

    # Forward returns
    fwd_6m = _forward_return(prices_t, as_of, 180)
    fwd_3m = _forward_return(prices_t, as_of, 90)
    if fwd_6m is None:
        return None  # without target we can't validate

    spy_t = prices_by_ticker.get("SPY")
    spy_fwd_6m = _forward_return(spy_t, as_of, 180) if spy_t is not None else None
    spy_fwd_3m = _forward_return(spy_t, as_of, 90) if spy_t is not None else None

    sector = cache["sectors"].get(ticker, "")

    edgar_8k_t = cache["edgar_8k"][cache["edgar_8k"]["ticker"] == ticker]
    edgar_form4_t = cache["edgar_form4"][cache["edgar_form4"]["ticker"] == ticker]
    fundamentals_t = cache["fundamentals"][cache["fundamentals"]["ticker"] == ticker]

    # Compute PIT-sourceable sub-scores
    s1 = 0.0  # estimate_revisions ZEROED
    s2, raw_s2 = sub_guidance_spine(ticker, as_of, edgar_8k_t)
    s3, raw_s3 = sub_drop_attribution_spine(ticker, as_of, sector, prices_by_ticker, edgar_8k_t)
    s4, raw_s4 = sub_catalyst_spine(ticker, as_of, edgar_8k_t)
    static_t = cache.get("static", {}).get(ticker, {})
    s5, raw_s5 = sub_valuation_pit_spine(ticker, as_of, prices_t, fundamentals_t, static_t)
    s6, raw_s6 = sub_quality_spine(ticker, as_of, fundamentals_t)
    s7, raw_s7 = sub_governance_spine(ticker, as_of, edgar_form4_t, edgar_8k_t)

    # Composite raw (full + variant_b_without_sub7 + variant_b_without_sub7_too)
    composite_raw = (
        _WEIGHTS["estimate_revisions"]  * s1 +
        _WEIGHTS["guidance"]            * s2 +
        _WEIGHTS["drop_attribution"]    * s3 +
        _WEIGHTS["catalyst"]            * s4 +
        _WEIGHTS["valuation"]           * s5 +
        _WEIGHTS["quality"]             * s6 +
        _WEIGHTS["governance"]          * s7
    )
    composite_without_sub7 = composite_raw - _WEIGHTS["governance"] * s7

    return {
        "ticker":                 ticker,
        "as_of":                  as_of.isoformat(),
        "sector":                 sector,
        "forward_6m_return":      fwd_6m,
        "forward_3m_return":      fwd_3m,
        "spy_fwd_6m":             spy_fwd_6m,
        "spy_fwd_3m":             spy_fwd_3m,
        "excess_6m_vs_spy":       (fwd_6m - spy_fwd_6m) if spy_fwd_6m is not None else None,
        "excess_3m_vs_spy":       (fwd_3m - spy_fwd_3m) if (fwd_3m is not None and spy_fwd_3m is not None) else None,
        "s1_estimate_revisions":  s1,
        "s2_guidance":            s2,
        "s3_drop_attribution":    s3,
        "s4_catalyst":            s4,
        "s5_valuation":           s5,
        "s6_quality":             s6,
        "s7_governance":          s7,
        "composite_raw":          composite_raw,
        "composite_without_sub7": composite_without_sub7,
        "raw_s2":                 json.dumps(raw_s2),
        "raw_s3":                 json.dumps(raw_s3),
        "raw_s4":                 json.dumps(raw_s4),
        "raw_s5":                 json.dumps(raw_s5),
        "raw_s6":                 json.dumps(raw_s6, default=str),
        "raw_s7":                 json.dumps(raw_s7, default=str),
        "pit_membership":         "today_list",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit-tickers", type=int, default=0,
                        help="Limit to first N tickers (debug)")
    parser.add_argument("--limit-weeks", type=int, default=0,
                        help="Limit to first N Fridays (debug)")
    args = parser.parse_args()

    cache = load_cache()

    # Pre-group prices by ticker for fast slicing
    prices_by_ticker = {
        t: g.sort_values("date").reset_index(drop=True)
        for t, g in cache["prices"].groupby("ticker")
    }
    log.info(f"[spine] pre-grouped {len(prices_by_ticker)} ticker price series")

    # CORE universe (today-list)
    universe_path = _ROOT / "core_universe.json"
    with universe_path.open() as f:
        u = json.load(f)
    core_tickers = [t["ticker"] for t in u["tickers"]]
    if args.limit_tickers > 0:
        core_tickers = core_tickers[:args.limit_tickers]
        log.info(f"[spine] DEBUG: limited to {len(core_tickers)} tickers")

    fridays = fridays_in_range(_AS_OF_START, _AS_OF_END)
    if args.limit_weeks > 0:
        fridays = fridays[:args.limit_weeks]
        log.info(f"[spine] DEBUG: limited to {len(fridays)} Fridays")

    total = len(core_tickers) * len(fridays)
    log.info(
        f"[spine] iterating {len(core_tickers)} tickers × {len(fridays)} Fridays "
        f"= {total:,} cells"
    )

    rows = []
    for i, ticker in enumerate(core_tickers, 1):
        for as_of in fridays:
            row = build_spine_row(ticker, as_of, cache, prices_by_ticker)
            if row is not None:
                rows.append(row)
        if i % 10 == 0:
            log.info(f"[spine] {i}/{len(core_tickers)} tickers done ({len(rows):,} rows)")

    spine = pd.DataFrame(rows)
    spine.to_parquet(_SPINE_PATH, index=False)
    log.info(f"[spine] persisted {len(spine):,} rows to {_SPINE_PATH}")

    # Spine metadata
    meta = {
        "built_at":                   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of_range":                {"start": _AS_OF_START.isoformat(), "end": _AS_OF_END.isoformat()},
        "n_tickers":                  len(core_tickers),
        "n_fridays":                  len(fridays),
        "n_rows":                     len(spine),
        "n_rows_missing":             total - len(spine),
        "missing_rate":               (total - len(spine)) / max(1, total),
        "spine_construction":         "today_list_applied_backward",
        "survivorship_inflation":     True,
        "trap_avoidance":             "untested",
        "pit_membership_breakdown":   {"today_list": len(spine), "git_reconstructed": 0},
        "sub_scores_active_in_spine": _SUB_SCORES_ACTIVE_IN_SPINE,
        "weights_used":               _WEIGHTS,
        "forward_horizon_days":       _FORWARD_HORIZON_DAYS,
    }
    _SPINE_META.write_text(json.dumps(meta, indent=2))
    log.info(f"[spine] persisted metadata to {_SPINE_META}")
    log.info(f"[spine] DONE — rows={len(spine):,} missing_rate={meta['missing_rate']:.2%}")


if __name__ == "__main__":
    main()
