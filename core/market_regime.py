"""market_regime.py — Macro regime detector for the Manager Layer.

Computes a 0-100 regime score combining several macro indicators that
historically correlate with market tops and bottoms. NOT predictive of
exact timing — used to inform dynamic cash allocation:

  0-30  (FROTHY / EXPENSIVE)  → hold dry powder, target high cash %
  30-70 (NEUTRAL)             → deploy normally, ~15-20% cash buffer
  70-100 (PANIC / GENERATIONAL) → deploy aggressively, near-zero cash

Components (each 0-100, then combined):

  VIX percentile (weight 0.25)
    Where is current VIX in the 1y range? Low VIX (<15) = complacency,
    high VIX (>30) = fear. We normalize so high VIX maps to a HIGH
    regime score (panic = good buying time).

  SPY vs MA200d (weight 0.20)
    Distance from 200-day MA. -10%+ below = bearish regime (good for
    buying), +10%+ above = bullish regime (frothy).

  Quality compounders in deep drawdown (weight 0.30)
    The DENSITY signal: how many of our 230 quality compounders are
    currently in -25%+ drawdown? Many = "panic in quality" = the
    contrarian's dream. Few = market broad-based bullish.

  Sectoral divergence (weight 0.15)
    Are some sectors deeply down while others at highs? Big spread =
    "rotation regime" (selective opportunities). Small spread = uniform
    market (less selective edge).

  Rolling 3-month SPY return (weight 0.10)
    Recent direction. Strong negative = post-crash buying zone. Strong
    positive = momentum top approaching.

User context (2026-06-15): SPY near ATH but quality non-AI tech in
recession. This module quantifies that divergence — the FROTHY/PANIC
mix the user described.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Component weights (must sum to 1.0)
_WEIGHTS = {
    "vix":          0.25,
    "spy_ma200d":   0.20,
    "quality_dd":   0.30,
    "divergence":   0.15,
    "spy_3m":       0.10,
}


def _fetch_vix_history(lookback_days: int = 365):
    """Returns pandas Series of daily VIX close, or None on failure."""
    try:
        from data.data_feed import get_eod_prices
        df = get_eod_prices("^VIX", lookback_days=lookback_days)
        if df is None or df.empty:
            return None
        if "date" in df.columns:
            import pandas as pd
            df = df.set_index(pd.to_datetime(df["date"])).sort_index()
        if "close" in df.columns:
            return df["close"]
        if "Close" in df.columns:
            return df["Close"]
        return None
    except Exception as e:
        log.warning(f"[regime] VIX fetch failed: {e}")
        return None


def _fetch_spy_history(lookback_days: int = 400):
    """Returns pandas Series of daily SPY close, or None on failure."""
    try:
        from data.data_feed import get_eod_prices
        df = get_eod_prices("SPY", lookback_days=lookback_days)
        if df is None or df.empty:
            return None
        if "date" in df.columns:
            import pandas as pd
            df = df.set_index(pd.to_datetime(df["date"])).sort_index()
        if "close" in df.columns:
            return df["close"]
        if "Close" in df.columns:
            return df["Close"]
        return None
    except Exception as e:
        log.warning(f"[regime] SPY fetch failed: {e}")
        return None


def _component_vix(vix_history) -> tuple[float, dict]:
    """Score 0-100 based on VIX percentile (high VIX = high score = buy regime).

    VIX < 12  → 0   (extreme complacency, frothy)
    VIX 12-15 → 10-25
    VIX 15-20 → 25-50 (neutral)
    VIX 20-30 → 50-75
    VIX 30-40 → 75-90 (fear)
    VIX > 40  → 90-100 (panic, classic generational buy zone)
    """
    if vix_history is None or len(vix_history) < 20:
        return 50.0, {"vix": None, "vix_p1y": None, "reason": "vix_data_unavailable"}
    current = float(vix_history.iloc[-1])
    # Continuous mapping from VIX level to 0-100 score
    if current < 12:
        score = current * (10 / 12)  # 0 at VIX 0, 10 at VIX 12
    elif current < 15:
        score = 10 + (current - 12) * (15 / 3)  # 10 to 25
    elif current < 20:
        score = 25 + (current - 15) * (25 / 5)  # 25 to 50
    elif current < 30:
        score = 50 + (current - 20) * (25 / 10)  # 50 to 75
    elif current < 40:
        score = 75 + (current - 30) * (15 / 10)  # 75 to 90
    else:
        score = min(100.0, 90 + (current - 40) * 0.5)
    # 1y percentile (where is current VIX in 1y range)
    p1y = float((vix_history < current).sum()) / len(vix_history)
    return float(score), {
        "vix": round(current, 2),
        "vix_p1y": round(p1y, 3),
        "reason": f"VIX={current:.1f}, 1y percentile={p1y:.0%}",
    }


def _component_spy_ma200d(spy_history) -> tuple[float, dict]:
    """Score 0-100 based on SPY distance from 200d MA.

    SPY 20%+ above MA200 → 0   (extremely overbought, frothy)
    SPY at MA200         → 50  (neutral)
    SPY 20%+ below MA200 → 100 (oversold, buy zone)
    """
    if spy_history is None or len(spy_history) < 200:
        return 50.0, {"spy": None, "ma200d": None, "reason": "spy_data_unavailable"}
    current = float(spy_history.iloc[-1])
    ma200 = float(spy_history.tail(200).mean())
    if ma200 <= 0:
        return 50.0, {"reason": "ma200_zero"}
    diff_pct = (current - ma200) / ma200
    # Map diff_pct from [-0.2, +0.2] linearly to [100, 0]
    score = 50.0 - (diff_pct / 0.20) * 50.0
    score = max(0.0, min(100.0, score))
    return float(score), {
        "spy": round(current, 2),
        "ma200d": round(ma200, 2),
        "spy_vs_ma200_pct": round(diff_pct * 100, 2),
        "reason": f"SPY {current:.0f} {'above' if diff_pct > 0 else 'below'} MA200d {ma200:.0f} ({diff_pct:+.1%})",
    }


def _component_quality_dd() -> tuple[float, dict]:
    """Score based on density of quality compounders in deep drawdown.

    Many CORE picks at score 80+ (drawdown + quality) = panic-in-quality regime.
    """
    try:
        from core.core_radar import scan_core_universe
        opps_60 = scan_core_universe(min_score=60.0)
        opps_80 = [o for o in opps_60 if float(o.get("score", 0)) >= 80.0]
        n60 = len(opps_60)
        n80 = len(opps_80)
        # Universe is ~230 tickers. Calibrate score:
        #   n80 < 5  → 20 (broad bull, few opportunities)
        #   n80 = 10 → 50 (neutral)
        #   n80 = 20 → 75
        #   n80 > 30 → 90 (lots of quality on sale = high regime score)
        if n80 < 5:
            score = 20 * (n80 / 5)
        elif n80 < 10:
            score = 20 + (n80 - 5) * 6
        elif n80 < 20:
            score = 50 + (n80 - 10) * 2.5
        elif n80 < 30:
            score = 75 + (n80 - 20) * 1.5
        else:
            score = min(100.0, 90 + (n80 - 30) * 0.5)
        return float(score), {
            "n_picks_60plus": n60,
            "n_picks_80plus": n80,
            "reason": f"{n80} compounders at score 80+ ({n60} at 60+), of universe ~230",
        }
    except Exception as e:
        log.warning(f"[regime] quality_dd failed: {e}")
        return 50.0, {"reason": f"quality_dd_failed: {e}"}


def _component_divergence(spy_history) -> tuple[float, dict]:
    """Sectoral divergence: are sectors moving in lockstep or not?
    For v1, use a simple proxy — sample a few sector ETFs and measure spread.
    """
    try:
        from data.data_feed import get_eod_prices
        import pandas as pd
        # Sample sector ETFs
        sectors = ["XLK", "XLV", "XLF", "XLY", "XLP", "XLI", "XLE"]
        returns_1y = []
        for sym in sectors:
            try:
                df = get_eod_prices(sym, lookback_days=260)
                if df is None or df.empty:
                    continue
                if "date" in df.columns:
                    df = df.set_index(pd.to_datetime(df["date"])).sort_index()
                close_col = "close" if "close" in df.columns else "Close"
                if close_col not in df.columns:
                    continue
                close = df[close_col]
                if len(close) < 200:
                    continue
                ret = (float(close.iloc[-1]) - float(close.iloc[0])) / float(close.iloc[0])
                returns_1y.append(ret)
            except Exception:
                continue
        if len(returns_1y) < 4:
            return 50.0, {"reason": "insufficient_sector_data"}
        # Standard deviation of 1y returns across sectors — higher = more
        # divergence = more selective opportunities.
        import statistics as _stats
        spread = _stats.stdev(returns_1y)
        # 5% std = uniform market, 25% std = wide divergence
        score = min(100.0, max(0.0, (spread - 0.05) / 0.20 * 100))
        return float(score), {
            "sector_return_std_1y": round(spread, 4),
            "n_sectors": len(returns_1y),
            "reason": f"Sector 1y return spread σ={spread:.1%}",
        }
    except Exception as e:
        log.warning(f"[regime] divergence failed: {e}")
        return 50.0, {"reason": f"divergence_failed: {e}"}


def _component_spy_3m(spy_history) -> tuple[float, dict]:
    """Score based on SPY 3-month rolling return.

    SPY -15%+ in 3m → 100 (post-crash buy zone)
    SPY  flat in 3m → 50
    SPY +15%+ in 3m → 0   (momentum top approaching)
    """
    if spy_history is None or len(spy_history) < 63:
        return 50.0, {"reason": "spy_3m_data_unavailable"}
    current = float(spy_history.iloc[-1])
    three_m_ago = float(spy_history.iloc[-63])
    if three_m_ago <= 0:
        return 50.0, {"reason": "zero_3m_price"}
    ret_3m = (current - three_m_ago) / three_m_ago
    # Map [-0.15, +0.15] → [100, 0]
    score = 50.0 - (ret_3m / 0.15) * 50.0
    score = max(0.0, min(100.0, score))
    return float(score), {
        "spy_3m_return_pct": round(ret_3m * 100, 2),
        "reason": f"SPY 3m return {ret_3m:+.1%}",
    }


def compute_regime() -> dict:
    """Compute the market regime score and component breakdown.

    Returns a dict:
      {
        "regime_score":  float (0-100),
        "regime_label":  str ("FROTHY" | "NEUTRAL" | "PANIC"),
        "components":    {name: {score, weight, reason, ...metadata}},
        "computed_at":   ISO timestamp,
        "interpretation":str (one-line summary),
        "cash_target_pct": int (recommended portfolio cash %),
      }
    """
    vix_hist = _fetch_vix_history(lookback_days=365)
    spy_hist = _fetch_spy_history(lookback_days=400)

    vix_score, vix_meta             = _component_vix(vix_hist)
    spy_ma_score, spy_ma_meta       = _component_spy_ma200d(spy_hist)
    quality_score, quality_meta     = _component_quality_dd()
    divergence_score, divergence_meta = _component_divergence(spy_hist)
    spy3m_score, spy3m_meta         = _component_spy_3m(spy_hist)

    components = {
        "vix":         {"score": round(vix_score, 1),         "weight": _WEIGHTS["vix"],         **vix_meta},
        "spy_ma200d":  {"score": round(spy_ma_score, 1),      "weight": _WEIGHTS["spy_ma200d"],  **spy_ma_meta},
        "quality_dd":  {"score": round(quality_score, 1),     "weight": _WEIGHTS["quality_dd"],  **quality_meta},
        "divergence":  {"score": round(divergence_score, 1),  "weight": _WEIGHTS["divergence"],  **divergence_meta},
        "spy_3m":      {"score": round(spy3m_score, 1),       "weight": _WEIGHTS["spy_3m"],      **spy3m_meta},
    }

    regime_score = (
        vix_score        * _WEIGHTS["vix"]
        + spy_ma_score   * _WEIGHTS["spy_ma200d"]
        + quality_score  * _WEIGHTS["quality_dd"]
        + divergence_score * _WEIGHTS["divergence"]
        + spy3m_score    * _WEIGHTS["spy_3m"]
    )

    if regime_score < 30:
        label = "FROTHY"
        cash_target_pct = 50
        interpretation = (
            "Market expensive/complacent. Hold dry powder. Deploying full "
            "monthly inflow now likely overpays vs waiting 3-6 months."
        )
    elif regime_score < 70:
        label = "NEUTRAL"
        cash_target_pct = 20
        interpretation = (
            "Mixed signals. Deploy monthly inflow normally with concentration "
            "cap protection. Keep ~20% buffer for unexpected opportunities."
        )
    else:
        label = "PANIC"
        cash_target_pct = 5
        interpretation = (
            "Generational buy zone. Deploy aggressively — including deferred "
            "monthly inflows and any excess cash beyond emergency fund."
        )

    return {
        "regime_score":     round(regime_score, 1),
        "regime_label":     label,
        "components":       components,
        "computed_at":      datetime.utcnow().isoformat() + "Z",
        "interpretation":   interpretation,
        "cash_target_pct":  cash_target_pct,
    }


def format_regime_summary(regime: dict) -> str:
    """Telegram-friendly multi-line formatted summary."""
    score = regime.get("regime_score", 0)
    label = regime.get("regime_label", "?")
    interp = regime.get("interpretation", "")
    target = regime.get("cash_target_pct", 0)

    badge = {"FROTHY": "🟦", "NEUTRAL": "🟨", "PANIC": "🟥"}.get(label, "⚪")
    lines = [
        f"{badge} *Market Regime: {label} ({score:.0f}/100)*",
        "",
        f"_{interp}_",
        "",
        f"💰 Recommended cash target: *{target}%* of portfolio",
        "",
        "*Components:*",
    ]
    for name, comp in regime.get("components", {}).items():
        s = comp.get("score", 0)
        w = comp.get("weight", 0)
        reason = comp.get("reason", "")
        lines.append(f"  `{name}` ({w:.0%}): {s:.0f}/100 — {reason}")
    return "\n".join(lines)
