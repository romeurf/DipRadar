"""Sector dip / momentum / exit scoring engine.

Computes per-ETF signals for the radar's three jobs:

1. **BUY opportunity** — `dip_score` and `momentum_score`. Higher = stronger.
   `dip_score` flags beaten-up sectors primed for mean reversion;
   `momentum_score` flags sectors trending up sustainably. Final BUY
   ranking takes `max(dip, momentum) - 0.5 × secular_risk` so a strong
   momentum candidate can outrank a dip with secular trouble.

2. **Avoid value trap** — `secular_risk`. Higher = sector might be dying
   (XLE 2014-2020 style). Buy signals downweighted by this.

3. **SELL signal for existing holdings** — `exit_strength`. Higher =
   stronger sell. Hysteresis: alert fires when crossing >70 from below,
   silenced until normalizes to <50 then crosses again. Re-entry watch
   via `entry_strength`.

All scores are clipped to [0, 100]. Architecture is intentionally
linear / interpretable — each contribution to the score is a separate
term, so when an alert fires you can see exactly what triggered it. A
future ML overlay can replace the linear combiners without changing the
upstream feature pipeline (see compute_signals — the structure stays).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TickerSignals:
    """All scores + raw metrics for one ticker on one day."""
    ticker: str
    price: float

    # Composite scores (0-100)
    dip_score: float
    momentum_score: float
    secular_risk: float
    exit_strength: float
    entry_strength: float
    caution_score: float       # early-warning: deterioration starting

    # Raw metrics used in scoring (kept for display / debugging)
    drawdown_52w: float        # negative number, e.g. -0.18 = -18%
    rsi_14: float
    ma200_ratio: float          # price / MA200 (1.0 = at MA)
    ma50_ratio: float           # price / MA50
    return_7d: float
    return_7d_vs_spy: float     # relative strength 7d (early warning)
    return_30d: float
    return_30d_vs_spy: float    # relative strength 30d (this - SPY)
    return_ytd: float
    return_ytd_vs_spy: float    # excess return YTD vs SPY
    pullback_from_recent_peak: float  # negative — drop from last 30d high
    days_below_ma200: int       # consecutive days price < MA200
    dd_duration_months: float   # months since last 52w high

    # Derived state — for the FSM
    state: str = "NEUTRAL"      # "NEUTRAL" | "SELL_ZONE" | "BUY_ZONE"

    # Combined BUY score (used for ranking)
    combined_buy: float = 0.0

    def is_buy_signal(self, threshold: float = 70.0,
                      max_secular_risk: float = 60.0) -> bool:
        """True if either dip or momentum opportunity, and not value-trap."""
        return (
            self.combined_buy >= threshold
            and self.secular_risk < max_secular_risk
        )

    def is_sell_signal(self, threshold: float = 70.0) -> bool:
        return self.exit_strength >= threshold


# ─── Indicator primitives ────────────────────────────────────────────────────

def _rsi(closes: pd.Series, window: int = 14) -> float:
    """Standard Wilder-smoothed RSI. Returns NaN on insufficient data."""
    if len(closes) < window + 1:
        return float("nan")
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gain / loss.replace(0.0, float("nan"))
    rsi = 100.0 - 100.0 / (1.0 + rs)
    last = rsi.iloc[-1]
    return float(last) if pd.notna(last) else float("nan")


def _consecutive_days_below(closes: pd.Series, ma: pd.Series) -> int:
    """Count consecutive trailing days where close < ma. Capped at 252."""
    if closes.empty or ma.empty:
        return 0
    diff = (closes - ma).dropna()
    n = 0
    for v in reversed(diff.tolist()):
        if v < 0:
            n += 1
            if n >= 252:
                break
        else:
            break
    return n


def _drawdown_duration_months(closes: pd.Series) -> float:
    """Months since the last 52w high."""
    if len(closes) < 21:
        return 0.0
    lookback = min(len(closes), 252)
    recent = closes.iloc[-lookback:]
    high_idx = recent.idxmax()
    days_since_high = (closes.index[-1] - high_idx).days
    return days_since_high / 30.4375  # avg days/month


def _max_rally_pct_window(closes: pd.Series, window: int = 60,
                          run_length: int = 5) -> float:
    """Max trailing N-day cumulative return rolled over `window` days.

    Used to detect "is the sector having any bounce attempts?" — if the
    biggest rally in the last 60 days is <5%, it's a sign the dip is
    persistent (not consolidating for recovery).
    """
    if len(closes) < window + run_length:
        return 0.0
    tail = closes.iloc[-window:]
    rolling_ret = tail.pct_change().rolling(run_length).sum()
    m = rolling_ret.max()
    return float(m) if pd.notna(m) else 0.0


def _count_lower_lows(closes: pd.Series, window: int = 60,
                      segment: int = 20) -> int:
    """Count of consecutive 'lower lows' in N-day chunks.

    Splits last `window` days into segments of `segment` days, finds the
    low of each, and counts how many consecutive segments had a lower low
    than the previous. Persistent lower-lows = bearish structure.
    """
    if len(closes) < window:
        return 0
    tail = closes.iloc[-window:]
    n_segments = window // segment
    lows = []
    for i in range(n_segments):
        chunk = tail.iloc[i * segment:(i + 1) * segment]
        if len(chunk) > 0:
            lows.append(float(chunk.min()))
    if len(lows) < 2:
        return 0
    consecutive = 0
    for i in range(1, len(lows)):
        if lows[i] < lows[i - 1]:
            consecutive += 1
        else:
            consecutive = 0
    return consecutive


# ─── Score computers ─────────────────────────────────────────────────────────

def _compute_dip_score(
    drawdown_52w: float,
    rsi_14: float,
    ma200_ratio: float,
    return_ytd_vs_spy: float,
    return_30d: float,
) -> float:
    """0-100 BUY-the-dip opportunity score.

    Weights chosen to value depth over signals of bottoming. The 30d
    return contribution is a "stabilization bonus" — if the sector has
    stopped falling (return_30d > -2%), add 10. We don't want to catch
    the falling knife.
    """
    score = 0.0

    # Drawdown weight 40 — saturates at -25% (deeper isn't more bullish)
    if drawdown_52w < 0:
        score += 40.0 * min(1.0, abs(drawdown_52w) / 0.25)

    # RSI weight 25 — only contributes when oversold (<30)
    if rsi_14 < 30:
        score += 25.0 * (30.0 - rsi_14) / 30.0

    # MA200 weight 15 — only when below by >5%
    if ma200_ratio < 0.95:
        score += 15.0 * min(1.0, (0.95 - ma200_ratio) / 0.20)

    # Underperformance vs SPY weight 10 — only when behind
    if return_ytd_vs_spy < 0:
        score += 10.0 * min(1.0, abs(return_ytd_vs_spy) / 0.10)

    # Stabilization bonus weight 10 — recent 30d not still cratering
    if return_30d > -0.02:
        score += 10.0

    return max(0.0, min(100.0, score))


def _compute_momentum_score(
    rsi_14: float,
    ma200_ratio: float,
    ma50_ratio: float,
    return_30d: float,
    return_30d_vs_spy: float,
) -> float:
    """0-100 momentum opportunity score — RELATIVE strength required.

    A "momentum buy" only fires when the ticker is OUTPERFORMING the
    broader market. Just trending up alongside SPY is beta participation,
    not actionable signal — every sector grinds higher in a bull market
    but only some are leadership candidates.

    Gates (return 0 if any fails):
      - return_30d_vs_spy must be ≥ 2pp (true relative strength)
      - ma50_ratio must be ≥ 1.0 (trend confirmed)

    Score (after gates pass):
      - Relative strength magnitude  (40 max)
      - RSI in healthy range          (20 max)
      - Established trend             (20 max)
      - 30d return tier               (20 max)

    Penalties:
      - Parabolic exhaustion (RSI > 78 or return_30d > 20%): -25
    """
    # Hard gates — no momentum without these
    if return_30d_vs_spy < 0.02:
        return 0.0  # not outperforming
    if ma50_ratio < 1.0:
        return 0.0  # below MA50, no trend

    score = 0.0

    # Relative strength magnitude — saturate at +10pp
    score += 40.0 * min(1.0, return_30d_vs_spy / 0.10)

    # RSI healthy (50-70 ideal)
    if 50 <= rsi_14 <= 70:
        score += 20.0
    elif 45 <= rsi_14 < 50 or 70 < rsi_14 <= 75:
        score += 12.0

    # Established trend (above MA200 by ≥5%)
    if ma200_ratio > 1.05:
        score += 20.0
    elif ma200_ratio > 1.0:
        score += 10.0

    # Absolute return reasonable
    if 0.03 <= return_30d <= 0.12:
        score += 20.0
    elif return_30d > 0.12:
        score += 8.0

    # Parabolic exhaustion penalty — heavily penalize blowoff tops
    if rsi_14 > 78 or return_30d > 0.20:
        score -= 25.0

    return max(0.0, min(100.0, score))


def _compute_secular_risk(
    closes: pd.Series,
    drawdown_52w: float,
    ma200: pd.Series,
) -> float:
    """0-100 secular-risk score. Higher = sector more likely structurally broken.

    Goal: flag XLE 2014-2020-style traps where a heuristic dip score
    would scream BUY for years while the sector quietly dies. Five 20pt
    contributions; needs at least 3 to flag.
    """
    score = 0.0

    # 1. Drawdown duration > 12 months (in *current* drawdown)
    dd_months = _drawdown_duration_months(closes)
    if dd_months > 12:
        score += 20.0

    # 2. No rallies > 5% in last 60 days (no bounce attempts → trending down)
    max_rally = _max_rally_pct_window(closes, window=60, run_length=5)
    if max_rally < 0.05 and drawdown_52w < -0.10:
        score += 20.0

    # 3. Below MA200 for > 9 months consecutively
    days_below = _consecutive_days_below(closes, ma200)
    if days_below > 189:  # ~9 months of trading days
        score += 20.0

    # 4. Lower lows persistent in last 60d
    lower_lows = _count_lower_lows(closes, window=60, segment=20)
    if lower_lows >= 2:  # 2 consecutive lower lows out of 3 segments
        score += 20.0

    # 5. 10y underperformance vs theoretical 8%/yr sector growth (proxy:
    #    long-term CAGR way below market average). Need 10y of data;
    #    skip silently if we don't have it.
    if len(closes) >= 2520:
        ten_y_ago = closes.iloc[-2520]
        current = closes.iloc[-1]
        if ten_y_ago > 0:
            cagr_10y = (current / ten_y_ago) ** (1 / 10) - 1
            if cagr_10y < 0.04:  # less than 4%/yr = bad
                score += 20.0

    return max(0.0, min(100.0, score))


def _compute_exit_strength(
    closes: pd.Series,
    rsi_14: float,
    ma200_ratio: float,
    drawdown_52w: float,
    ma50: pd.Series,
    ma200: pd.Series,
) -> float:
    """0-100 SELL signal strength for current holdings.

    Detects the EARLY phase of a sector breakdown — before the big
    drawdown. Combines trend (MA cross), momentum (RSI rolling over),
    and position (drawdown context). Threshold of 70 needed to fire
    (with hysteresis at 50 for reset).
    """
    score = 0.0

    if len(closes) < 1 or ma50.empty or ma200.empty:
        return 0.0
    current = float(closes.iloc[-1])

    # Below MA50 — early warning
    if current < float(ma50.iloc[-1]):
        score += 20.0

    # MA50 < MA200 — confirmation (death cross territory)
    if float(ma50.iloc[-1]) < float(ma200.iloc[-1]):
        score += 25.0

    # RSI < 40 but not yet capitulated (RSI > 25)
    if 25 < rsi_14 < 40:
        score += 20.0

    # Drawdown > 10% already
    if drawdown_52w < -0.10:
        score += 15.0

    # MA200 slope negative over last 6 sessions
    if len(ma200) >= 7:
        slope_recent = float(ma200.iloc[-1]) - float(ma200.iloc[-7])
        if slope_recent < 0:
            score += 10.0

    # Volume spike on most recent down day (capitulation start)
    # Skipped here — would need Volume in the data. The 5 signals above
    # are enough to flag exit; volume would only refine threshold.

    return max(0.0, min(100.0, score))


def _compute_caution_score(
    return_7d: float,
    return_7d_vs_spy: float,
    return_30d: float,
    return_30d_peak_60d: float,
    return_30d_vs_spy: float,
    return_30d_vs_spy_peak_60d: float,
    pullback_from_recent_peak: float,
    rsi_14: float,
    rsi_peak_30d: float,
    rsi_5d_ago: float,
    price_now: float,
    price_5d_ago: float,
) -> float:
    """0-100 early-warning score. Mix of LEADING and LAGGING signals.

    Reactive caution (price already dropped 7%+) saves you from the next
    leg down but you've already lost. This version blends:

    LEADING signals — fire BEFORE significant drop:
      L1. Momentum deceleration: 30d return decayed > 50% from 60d peak
      L2. Relative strength decay: 30d vs SPY dropped > 50% from peak
      L3. RSI rollover from recent peak (was >65, now -10pts lower)
      L4. RSI bearish divergence: price up but RSI down over 5d

    LAGGING signals — confirm an underway pullback:
      L5. Pullback from recent peak ≥ 4%
      L6. 7-day return < -2%
      L7. 7-day relative strength loss > 1pp vs SPY

    Trade-off: leading signals catch tops earlier but have more false
    positives (sometimes momentum re-accelerates after deceleration). The
    multi-signal gate (≥2 conditions) reduces noise. Use with scaling
    exit policy — trim incrementally as caution rises, don't exit all at
    once on a single trigger.
    """
    score = 0.0
    n_signals = 0

    # ── LEADING signals ──────────────────────────────────────────────
    # L1. Momentum deceleration — was strong, halved
    if return_30d_peak_60d > 0.08 and return_30d < return_30d_peak_60d * 0.5:
        score += 20.0
        n_signals += 1

    # L2. Relative strength decaying — leadership fading
    if (return_30d_vs_spy_peak_60d > 0.05
            and return_30d_vs_spy < return_30d_vs_spy_peak_60d * 0.4):
        score += 20.0
        n_signals += 1

    # L3. RSI rollover from recent peak (was hot, cooling)
    if (rsi_peak_30d - rsi_14) >= 10.0 and rsi_14 < 65 and rsi_peak_30d > 65:
        score += 15.0
        n_signals += 1

    # L4. RSI bearish divergence over 5d — price up, RSI down
    if price_now > price_5d_ago and rsi_14 < rsi_5d_ago - 3:
        score += 15.0
        n_signals += 1

    # ── LAGGING signals (confirming) ─────────────────────────────────
    # L5. Pullback from recent peak ≥ 4% — drop starting in earnest
    if pullback_from_recent_peak < -0.04:
        score += 15.0
        n_signals += 1

    # L6. 7-day return < -2% — short-term weakness
    if return_7d < -0.02:
        score += 10.0
        n_signals += 1

    # L7. 7-day relative strength loss > 1pp — underperforming acutely
    if return_7d_vs_spy < -0.01:
        score += 15.0
        n_signals += 1

    # Multi-signal gate: ≥2 conditions to fire — single signals are noise
    if n_signals < 2:
        return 0.0

    return max(0.0, min(100.0, score))


def _compute_entry_strength(
    closes: pd.Series,
    rsi_14: float,
    return_30d: float,
    ma50: pd.Series,
    ma200: pd.Series,
) -> float:
    """0-100 RE-ENTRY signal strength.

    After we've exited a holding via exit_strength, this watches for the
    bottom: price crosses back above MA50, RSI bouncing from oversold,
    higher low formation, MA50 turning up.
    """
    score = 0.0
    if len(closes) < 50 or ma50.empty:
        return 0.0
    current = float(closes.iloc[-1])

    # Price cruzes back above MA50 — first sign of trend reversal
    if current > float(ma50.iloc[-1]):
        # Cross is only meaningful if recently was below
        if len(closes) >= 5:
            recent_below = any(
                float(closes.iloc[-i]) < float(ma50.iloc[-i])
                for i in range(2, 6)
                if i < len(ma50)
            )
            if recent_below:
                score += 25.0
            else:
                score += 10.0  # already established above

    # RSI bottoming from oversold
    if 35 <= rsi_14 <= 55 and len(closes) >= 30:
        # Was the RSI <30 in the last 30 days?
        recent_rsi = []
        for i in range(min(30, len(closes) - 14)):
            window = closes.iloc[-(i + 14 + 1):-i if i > 0 else None]
            if len(window) >= 15:
                recent_rsi.append(_rsi(window))
        if recent_rsi and min(recent_rsi) < 30:
            score += 25.0

    # 30d return turning positive
    if 0.02 <= return_30d <= 0.15:
        score += 20.0
    elif 0 < return_30d < 0.02:
        score += 10.0

    # Higher low formation (last low > prior low)
    if len(closes) >= 60:
        last_30_low = float(closes.iloc[-30:].min())
        prior_30_low = float(closes.iloc[-60:-30].min())
        if last_30_low > prior_30_low * 1.005:
            score += 15.0

    # MA50 slope starting to inflect up
    if len(ma50) >= 7:
        slope_recent = float(ma50.iloc[-1]) - float(ma50.iloc[-7])
        if slope_recent > 0:
            score += 15.0

    return max(0.0, min(100.0, score))


# ─── Public API ──────────────────────────────────────────────────────────────

def compute_signals(
    ticker: str,
    hist: pd.DataFrame,
    spy_hist: Optional[pd.DataFrame] = None,
) -> Optional[TickerSignals]:
    """Compute all scores + metrics for a ticker.

    Caller is responsible for fetching `hist` (OHLCV DataFrame with
    DateTimeIndex and a 'Close' column). Pre-fetching shared SPY history
    once and passing it in avoids 11+ duplicate yfinance calls per
    radar sweep.
    """
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    closes = hist["Close"].astype(float)
    if len(closes) < 200:
        # Not enough data for MA200 — sector ETF too new
        return None

    current = float(closes.iloc[-1])
    if current <= 0 or not math.isfinite(current):
        return None

    # Moving averages
    ma50 = closes.rolling(50).mean()
    ma200 = closes.rolling(200).mean()
    ma50_last = float(ma50.iloc[-1])
    ma200_last = float(ma200.iloc[-1])

    # Core metrics
    high_52w_window = min(252, len(closes))
    high_52w = float(closes.iloc[-high_52w_window:].max())
    drawdown_52w = current / high_52w - 1.0 if high_52w > 0 else 0.0
    rsi_14 = _rsi(closes)
    ma200_ratio = current / ma200_last if ma200_last > 0 else 1.0
    ma50_ratio = current / ma50_last if ma50_last > 0 else 1.0

    # Returns — short and medium term
    return_7d = current / float(closes.iloc[-7]) - 1.0 if len(closes) >= 7 else 0.0
    return_30d = current / float(closes.iloc[-30]) - 1.0 if len(closes) >= 30 else 0.0

    # Pullback from recent peak (last 30d high) — early warning that the
    # rally is fading, even if the 52w high is far away.
    if len(closes) >= 30:
        recent_peak = float(closes.iloc[-30:].max())
        pullback_from_recent_peak = current / recent_peak - 1.0 if recent_peak > 0 else 0.0
    else:
        pullback_from_recent_peak = 0.0

    # RSI and price snapshots — for divergence detection (5d ago)
    price_5d_ago = float(closes.iloc[-6]) if len(closes) >= 6 else current
    if len(closes) >= 20:
        rsi_5d_ago = _rsi(closes.iloc[:-5])
        if not math.isfinite(rsi_5d_ago):
            rsi_5d_ago = rsi_14
    else:
        rsi_5d_ago = rsi_14

    # Rolling RSI history over last 30 days — for rollover detection
    rsi_peak_30d = rsi_14
    if len(closes) >= 45:
        rsi_history = []
        for i in range(0, 30):
            window = closes.iloc[-(i + 14 + 1):len(closes) - i] if i > 0 else closes
            if len(window) >= 15:
                rsi_at_i = _rsi(window)
                if math.isfinite(rsi_at_i):
                    rsi_history.append(rsi_at_i)
        if rsi_history:
            rsi_peak_30d = max(rsi_history)

    # Rolling 30d returns over last 60 days — for momentum-deceleration
    # detection. The peak shows the strongest 30d window in recent memory;
    # if current 30d is much lower, momentum is decaying even if still
    # positive in absolute terms.
    return_30d_peak_60d = return_30d
    if len(closes) >= 90:
        rolling_30d_returns = []
        for i in range(0, 60):
            end = len(closes) - i
            if end >= 30:
                ret_i = float(closes.iloc[end - 1] / closes.iloc[end - 30] - 1.0)
                if math.isfinite(ret_i):
                    rolling_30d_returns.append(ret_i)
        if rolling_30d_returns:
            return_30d_peak_60d = max(rolling_30d_returns)

    # SPY-relative returns (7d + 30d + YTD) and peak relative strength
    spy_return_30d = 0.0
    spy_return_7d = 0.0
    return_7d_vs_spy = 0.0
    return_30d_vs_spy = 0.0
    return_30d_vs_spy_peak_60d = 0.0  # populated inside try if SPY available
    return_ytd = 0.0
    return_ytd_vs_spy = 0.0
    try:
        last_dt = pd.Timestamp(closes.index[-1])
        ytd_start = last_dt.replace(month=1, day=1, hour=0, minute=0, second=0)
        ytd_slice = closes[closes.index >= ytd_start]
        if len(ytd_slice) >= 2:
            return_ytd = float(ytd_slice.iloc[-1] / ytd_slice.iloc[0] - 1.0)
        if spy_hist is not None and not spy_hist.empty:
            spy_closes = spy_hist["Close"].astype(float)
            # SPY 7d + 30d returns (for relative strength at both horizons)
            if len(spy_closes) >= 7:
                spy_return_7d = float(spy_closes.iloc[-1] / spy_closes.iloc[-7] - 1.0)
                return_7d_vs_spy = return_7d - spy_return_7d
            if len(spy_closes) >= 30:
                spy_return_30d = float(spy_closes.iloc[-1] / spy_closes.iloc[-30] - 1.0)
                return_30d_vs_spy = return_30d - spy_return_30d
            # SPY YTD
            spy_ytd = spy_closes[spy_closes.index >= ytd_start]
            if len(spy_ytd) >= 2:
                spy_ret_ytd = float(spy_ytd.iloc[-1] / spy_ytd.iloc[0] - 1.0)
                return_ytd_vs_spy = return_ytd - spy_ret_ytd
            # Peak relative strength over last 60d — for leadership-decay
            # detection. If we were leading by +8pp and now only +3pp,
            # leadership is fading even though we're still outperforming.
            return_30d_vs_spy_peak_60d = return_30d_vs_spy
            if len(closes) >= 90 and len(spy_closes) >= 90:
                rs_history = []
                # Align lengths from the tail
                n = min(len(closes), len(spy_closes))
                tk_tail = closes.iloc[-n:].reset_index(drop=True)
                sp_tail = spy_closes.iloc[-n:].reset_index(drop=True)
                for i in range(0, 60):
                    end = n - i
                    if end >= 30:
                        tk_ret = float(tk_tail.iloc[end - 1] / tk_tail.iloc[end - 30] - 1.0)
                        sp_ret = float(sp_tail.iloc[end - 1] / sp_tail.iloc[end - 30] - 1.0)
                        if math.isfinite(tk_ret) and math.isfinite(sp_ret):
                            rs_history.append(tk_ret - sp_ret)
                if rs_history:
                    return_30d_vs_spy_peak_60d = max(rs_history)
    except Exception as e:
        logger.debug(f"[sector_radar] returns calc failed for {ticker}: {e}")

    # For SPY itself, "vs SPY" is identically 0 — that's correct, and the
    # momentum gate naturally rejects SPY as a momentum candidate. Dip
    # score still works for SPY (it's a sector-like for the radar purposes).

    # Duration metrics
    days_below_ma200 = _consecutive_days_below(closes, ma200)
    dd_duration_months = _drawdown_duration_months(closes)

    # Scores
    dip_score = _compute_dip_score(
        drawdown_52w, rsi_14, ma200_ratio, return_ytd_vs_spy, return_30d
    )
    momentum_score = _compute_momentum_score(
        rsi_14, ma200_ratio, ma50_ratio, return_30d, return_30d_vs_spy
    )
    secular_risk = _compute_secular_risk(closes, drawdown_52w, ma200)
    exit_strength = _compute_exit_strength(
        closes, rsi_14, ma200_ratio, drawdown_52w, ma50, ma200
    )
    entry_strength = _compute_entry_strength(closes, rsi_14, return_30d, ma50, ma200)
    caution_score = _compute_caution_score(
        return_7d=return_7d,
        return_7d_vs_spy=return_7d_vs_spy,
        return_30d=return_30d,
        return_30d_peak_60d=return_30d_peak_60d,
        return_30d_vs_spy=return_30d_vs_spy,
        return_30d_vs_spy_peak_60d=return_30d_vs_spy_peak_60d,
        pullback_from_recent_peak=pullback_from_recent_peak,
        rsi_14=rsi_14,
        rsi_peak_30d=rsi_peak_30d,
        rsi_5d_ago=rsi_5d_ago,
        price_now=current,
        price_5d_ago=price_5d_ago,
    )

    # Combined BUY score: stronger of dip/momentum, penalised by both
    # SECULAR risk (might be dying) and CAUTION (actively deteriorating).
    # The caution penalty ensures a sector in active pullback doesn't get
    # a "REFORÇAR" recommendation just because its 30d trend is still up —
    # buying into the drop is exactly what we want to avoid.
    combined_buy = (
        max(dip_score, momentum_score)
        - 0.5 * secular_risk
        - 0.5 * caution_score
    )
    combined_buy = max(0.0, min(100.0, combined_buy))

    return TickerSignals(
        ticker=ticker.upper(),
        price=current,
        dip_score=round(dip_score, 1),
        momentum_score=round(momentum_score, 1),
        secular_risk=round(secular_risk, 1),
        exit_strength=round(exit_strength, 1),
        entry_strength=round(entry_strength, 1),
        caution_score=round(caution_score, 1),
        drawdown_52w=round(drawdown_52w, 4),
        rsi_14=round(rsi_14, 1) if math.isfinite(rsi_14) else float("nan"),
        ma200_ratio=round(ma200_ratio, 4),
        ma50_ratio=round(ma50_ratio, 4),
        return_7d=round(return_7d, 4),
        return_7d_vs_spy=round(return_7d_vs_spy, 4),
        return_30d=round(return_30d, 4),
        return_30d_vs_spy=round(return_30d_vs_spy, 4),
        return_ytd=round(return_ytd, 4),
        return_ytd_vs_spy=round(return_ytd_vs_spy, 4),
        pullback_from_recent_peak=round(pullback_from_recent_peak, 4),
        days_below_ma200=int(days_below_ma200),
        dd_duration_months=round(dd_duration_months, 1),
        combined_buy=round(combined_buy, 1),
    )


def fetch_spy_history(period: str = "3y") -> Optional[pd.DataFrame]:
    """Fetch SPY OHLCV — needed for relative strength computation.

    Callers computing a single ticker (e.g., `/sectors XLK` deep-dive)
    should fetch SPY first and pass to `fetch_and_compute`. Otherwise
    `return_30d_vs_spy` falls back to 0.0, the momentum gate fails for
    any ticker, and the FSM silently overwrites legitimate BUY_ZONE
    states with NEUTRAL. The bug manifested as an oscillation loop:
    snapshot fires BUY alert → deep-dive silently resets to NEUTRAL →
    next snapshot fires BUY alert again.
    """
    try:
        import yfinance as yf
        return yf.Ticker("SPY").history(period=period, auto_adjust=True)
    except Exception as e:
        logger.warning(f"[sector_radar] SPY fetch failed: {e}")
        return None


def fetch_and_compute(
    ticker: str,
    spy_hist: Optional[pd.DataFrame] = None,
    period: str = "3y",
    auto_fetch_spy: bool = True,
) -> Optional[TickerSignals]:
    """Convenience wrapper: fetch OHLCV via yfinance then compute signals.

    For batch computation across many tickers, fetch SPY once and reuse
    (`rank_universe` does this). For one-off calls (admin commands,
    diagnostics), leave `auto_fetch_spy=True` so the function fetches
    SPY itself — otherwise momentum gating fails and FSM state writes
    are corrupted (see fetch_spy_history docstring).
    """
    if spy_hist is None and auto_fetch_spy and ticker.upper() != "SPY":
        spy_hist = fetch_spy_history(period=period)
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    except Exception as e:
        logger.warning(f"[sector_radar] Fetch failed for {ticker}: {e}")
        return None
    return compute_signals(ticker, hist, spy_hist=spy_hist)


def rank_universe(tickers: list[str]) -> list[TickerSignals]:
    """Compute signals for many tickers and return sorted by combined_buy desc.

    Fetches SPY once and reuses across all signal computations. Tickers
    that fail to fetch / have insufficient history are silently skipped.
    """
    spy_hist = None
    try:
        import yfinance as yf
        spy_hist = yf.Ticker("SPY").history(period="3y", auto_adjust=True)
    except Exception as e:
        logger.warning(f"[sector_radar] SPY fetch failed: {e} — YTD vs SPY = 0")

    results: list[TickerSignals] = []
    for t in tickers:
        sig = fetch_and_compute(t, spy_hist=spy_hist)
        if sig is not None:
            results.append(sig)

    results.sort(key=lambda s: s.combined_buy, reverse=True)
    return results
