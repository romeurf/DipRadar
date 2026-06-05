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

    # Raw metrics used in scoring (kept for display / debugging)
    drawdown_52w: float        # negative number, e.g. -0.18 = -18%
    rsi_14: float
    ma200_ratio: float          # price / MA200 (1.0 = at MA)
    ma50_ratio: float           # price / MA50
    return_30d: float
    return_30d_vs_spy: float    # relative strength 30d (this - SPY)
    return_ytd: float
    return_ytd_vs_spy: float    # excess return YTD vs SPY
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

    # Returns
    return_30d = current / float(closes.iloc[-30]) - 1.0 if len(closes) >= 30 else 0.0

    # SPY 30d return + YTD for relative strength calcs
    spy_return_30d = 0.0
    return_30d_vs_spy = 0.0
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
            # SPY 30d return (matches ticker's 30d window)
            if len(spy_closes) >= 30:
                spy_return_30d = float(spy_closes.iloc[-1] / spy_closes.iloc[-30] - 1.0)
                return_30d_vs_spy = return_30d - spy_return_30d
            # SPY YTD
            spy_ytd = spy_closes[spy_closes.index >= ytd_start]
            if len(spy_ytd) >= 2:
                spy_ret_ytd = float(spy_ytd.iloc[-1] / spy_ytd.iloc[0] - 1.0)
                return_ytd_vs_spy = return_ytd - spy_ret_ytd
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

    # Combined BUY score: stronger of dip/momentum, downweighted by secular risk.
    # Half-weight on secular ensures that a strong dip in a moderately-risky
    # sector (e.g., 80 dip + 40 secular = 60) still beats a weak signal in a
    # clean sector (e.g., 60 dip + 0 secular = 60), with edge to clean.
    combined_buy = max(dip_score, momentum_score) - 0.5 * secular_risk
    combined_buy = max(0.0, min(100.0, combined_buy))

    return TickerSignals(
        ticker=ticker.upper(),
        price=current,
        dip_score=round(dip_score, 1),
        momentum_score=round(momentum_score, 1),
        secular_risk=round(secular_risk, 1),
        exit_strength=round(exit_strength, 1),
        entry_strength=round(entry_strength, 1),
        drawdown_52w=round(drawdown_52w, 4),
        rsi_14=round(rsi_14, 1) if math.isfinite(rsi_14) else float("nan"),
        ma200_ratio=round(ma200_ratio, 4),
        ma50_ratio=round(ma50_ratio, 4),
        return_30d=round(return_30d, 4),
        return_30d_vs_spy=round(return_30d_vs_spy, 4),
        return_ytd=round(return_ytd, 4),
        return_ytd_vs_spy=round(return_ytd_vs_spy, 4),
        days_below_ma200=int(days_below_ma200),
        dd_duration_months=round(dd_duration_months, 1),
        combined_buy=round(combined_buy, 1),
    )


def fetch_and_compute(
    ticker: str,
    spy_hist: Optional[pd.DataFrame] = None,
    period: str = "3y",
) -> Optional[TickerSignals]:
    """Convenience wrapper: fetch OHLCV via yfinance then compute signals.

    Used by the daily radar sweep. For batch computation across many
    tickers, fetch SPY once and reuse.
    """
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
