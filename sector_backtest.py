"""Backtest harness for the sector_radar scaling-exit policy.

Replays the radar daily over historical OHLCV and simulates a position
that scales between 0% and 100% invested based on caution_score. Compares
total return + max drawdown to a passive buy-and-hold benchmark.

The question this answers honestly: does cutting at the first leading
signals net you ahead of just holding through pullbacks? Leading signals
catch tops earlier but produce false positives. Buy-and-hold captures
the full bull-market climb but eats the full drawdowns.

Policy under test (matches /holdings_etf_status logic):
  caution 0-29  → target 100%
  caution 30-49 → target 100% (just VIGIAR — don't trim yet, too noisy)
  caution 50-69 → target 75%  (trim 25%)
  caution 70-89 → target 50%  (trim 50%)
  caution 90+   → target 25%  (trim 75%)
  exit_strength > 70 → target 0%  (full exit on structural breakdown)
  Re-entry: target 100% when caution drops < 30 AND exit_strength < 30.

Costs: 10 bps per traded fraction (round-trip ~20bps). Realistic for a
EU broker like Trading 212 or IBKR.

Cooldown: 5 trading days between position changes to prevent flip-flop
on borderline scores.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_INITIAL_LOOKBACK_DAYS = 250    # need ≥ MA200 + buffer to start simulating
_COST_BPS = 0.001               # 10 bps per traded fraction
_COOLDOWN_DAYS = 5
_MIN_TRADE_THRESHOLD = 0.05     # ignore changes < 5% to avoid noise


@dataclass
class BacktestResult:
    """Summary metrics of one ETF backtest."""
    ticker: str
    n_days: int
    n_trades: int

    # Returns
    strategy_total_return: float
    buyhold_total_return: float
    alpha: float                # strategy - buyhold
    strategy_cagr: float
    buyhold_cagr: float

    # Risk
    strategy_max_drawdown: float    # negative number
    buyhold_max_drawdown: float
    drawdown_reduction: float       # bh - strategy (positive = strategy better)

    # Risk-adjusted
    strategy_sharpe: float
    buyhold_sharpe: float

    # Cost overhead
    total_cost_pct: float           # total cost paid as % of initial capital

    # Coverage
    pct_time_invested: float        # % of days strategy was > 90% invested
    final_position: float


def _decide_target_position(
    caution_score: float,
    exit_strength: float,
    current_position: float,
) -> float:
    """Pure function: return target position based on signals.

    Mirrors /holdings_etf_status logic. The "no trim until 50" floor
    prevents the strategy from trading on isolated leading signals —
    only acts on confluence.
    """
    if exit_strength > 70:
        return 0.0
    if caution_score >= 90:
        return 0.25
    if caution_score >= 70:
        return 0.50
    if caution_score >= 50:
        return 0.75
    # Re-entry: only when caution AND exit are both low
    if caution_score < 30 and exit_strength < 30:
        return 1.0
    return current_position  # hold


def _max_drawdown(values: list[float]) -> float:
    """Return the worst drawdown of a value series (negative number)."""
    if not values:
        return 0.0
    peak = values[0]
    mdd = 0.0
    for v in values:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (v - peak) / peak
            if dd < mdd:
                mdd = dd
    return mdd


def _annualized_sharpe(returns: list[float]) -> float:
    """Annualized Sharpe from a list of daily returns."""
    if len(returns) < 30:
        return 0.0
    arr = np.array(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 30:
        return 0.0
    mu  = float(np.mean(arr)) * 252.0
    sd  = float(np.std(arr))  * math.sqrt(252.0)
    return mu / sd if sd > 1e-9 else 0.0


def backtest_sector(
    ticker: str,
    n_years: int = 3,
    spy_hist: Optional[pd.DataFrame] = None,
) -> Optional[BacktestResult]:
    """Run the scaling-exit policy on `ticker` over the last n_years.

    Returns None if data is insufficient. Slow path: re-computes signals
    every trading day (~50-100 ms per call). For a 3-year backtest that's
    ~750 calls = ~1 minute per ticker. For the full 17-ETF universe,
    budget ~15-20 minutes — run in background.
    """
    from sector_radar import compute_signals

    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=f"{n_years}y", auto_adjust=True)
    except Exception as e:
        logger.warning(f"[backtest] {ticker} fetch failed: {e}")
        return None

    if hist is None or len(hist) < _INITIAL_LOOKBACK_DAYS:
        logger.info(f"[backtest] {ticker}: insufficient history ({len(hist) if hist is not None else 0}d)")
        return None

    # Fetch SPY once if not provided
    if spy_hist is None and ticker.upper() != "SPY":
        try:
            import yfinance as yf
            spy_hist = yf.Ticker("SPY").history(period=f"{n_years}y", auto_adjust=True)
        except Exception:
            spy_hist = None

    closes = hist["Close"].astype(float)

    # Start simulation after lookback fills in
    start_idx = _INITIAL_LOOKBACK_DAYS

    # State variables
    position = 1.0  # fully invested at start
    cash = 0.0
    n_trades = 0
    total_cost = 0.0
    days_since_change = _COOLDOWN_DAYS  # ready to trade from day 1
    n_days_fully_invested = 0

    # For metrics
    strategy_values: list[float] = []
    buyhold_values: list[float] = []
    strategy_daily_returns: list[float] = []
    buyhold_daily_returns: list[float] = []

    initial_price = float(closes.iloc[start_idx])

    for i in range(start_idx, len(hist)):
        # Slice history to "today" — no future leakage
        hist_window = hist.iloc[:i + 1]
        spy_window = spy_hist.iloc[:i + 1] if spy_hist is not None else None

        sig = compute_signals(ticker, hist_window, spy_hist=spy_window)
        if sig is None:
            continue

        price = sig.price

        # Decide target position
        target = _decide_target_position(
            sig.caution_score, sig.exit_strength, position
        )

        # Apply cooldown
        if abs(target - position) >= _MIN_TRADE_THRESHOLD and days_since_change >= _COOLDOWN_DAYS:
            delta = target - position
            trade_value = abs(delta) * price
            cost = trade_value * _COST_BPS
            # When we sell (delta < 0): cash increases by delta * price (-delta * price)
            # minus the cost.
            cash += -delta * price - cost
            position = target
            total_cost += cost
            n_trades += 1
            days_since_change = 0
        days_since_change += 1

        # Portfolio value (in price units, relative to 1 initial share)
        strategy_value = position * price + cash
        buyhold_value = price  # always 1 share

        strategy_values.append(strategy_value)
        buyhold_values.append(buyhold_value)

        if position >= 0.90:
            n_days_fully_invested += 1

        # Daily returns (after first day in sim)
        if len(strategy_values) >= 2:
            prev_s = strategy_values[-2]
            prev_b = buyhold_values[-2]
            if prev_s > 0:
                strategy_daily_returns.append(strategy_value / prev_s - 1.0)
            if prev_b > 0:
                buyhold_daily_returns.append(buyhold_value / prev_b - 1.0)

    if not strategy_values:
        return None

    n_days = len(strategy_values)
    n_years_actual = n_days / 252.0

    final_strategy = strategy_values[-1]
    final_buyhold = buyhold_values[-1]

    strategy_total_return = final_strategy / initial_price - 1.0
    buyhold_total_return = final_buyhold / initial_price - 1.0

    strategy_cagr = (final_strategy / initial_price) ** (1.0 / n_years_actual) - 1.0 if n_years_actual > 0 else 0.0
    buyhold_cagr = (final_buyhold / initial_price) ** (1.0 / n_years_actual) - 1.0 if n_years_actual > 0 else 0.0

    return BacktestResult(
        ticker=ticker.upper(),
        n_days=n_days,
        n_trades=n_trades,
        strategy_total_return=round(strategy_total_return, 4),
        buyhold_total_return=round(buyhold_total_return, 4),
        alpha=round(strategy_total_return - buyhold_total_return, 4),
        strategy_cagr=round(strategy_cagr, 4),
        buyhold_cagr=round(buyhold_cagr, 4),
        strategy_max_drawdown=round(_max_drawdown(strategy_values), 4),
        buyhold_max_drawdown=round(_max_drawdown(buyhold_values), 4),
        drawdown_reduction=round(
            _max_drawdown(buyhold_values) - _max_drawdown(strategy_values), 4
        ),
        strategy_sharpe=round(_annualized_sharpe(strategy_daily_returns), 3),
        buyhold_sharpe=round(_annualized_sharpe(buyhold_daily_returns), 3),
        total_cost_pct=round(total_cost / initial_price, 4),
        pct_time_invested=round(n_days_fully_invested / n_days, 3),
        final_position=round(position, 3),
    )


def backtest_universe(
    tickers: list[str],
    n_years: int = 3,
    progress_callback=None,
) -> list[BacktestResult]:
    """Run backtests for many tickers. Reuses SPY history across all.

    `progress_callback(ticker, i, n)` is called before each ticker, so
    the bot can post incremental updates to Telegram while the long
    background run is in progress.
    """
    try:
        import yfinance as yf
        spy_hist = yf.Ticker("SPY").history(period=f"{n_years}y", auto_adjust=True)
    except Exception as e:
        logger.warning(f"[backtest] SPY fetch failed: {e}")
        spy_hist = None

    results: list[BacktestResult] = []
    for i, t in enumerate(tickers):
        if progress_callback is not None:
            try:
                progress_callback(t, i, len(tickers))
            except Exception:
                pass
        res = backtest_sector(t, n_years=n_years, spy_hist=spy_hist)
        if res is not None:
            results.append(res)
    return results
