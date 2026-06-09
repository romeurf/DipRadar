"""Sector rotation backtest — top-N equal-weight rebalancing.

Strategy under test: at each monthly checkpoint (plus forced rebalance
when top-N membership changes between checkpoints), allocate the
portfolio equally across the top-N ETFs by `rotation_score`. The
rotation score is cross-sectional (rank-based) so we always pick "the
best of what's available" — even in a unanimous bear market the top-3
relative outperformers get the capital.

The thesis: trying to time tops by trimming on caution destroys value
(empirically proven by the prior backtest, 0/20 wins). But ROTATING
capital toward whichever sectors are currently the strongest may capture
alpha — the AQR/Asness "momentum" literature documents this generates
~3-5%/year on monthly-rebalanced equity factor strategies.

The honest test: does rotation beat:
  1. Buy-and-hold SPY (the lazy benchmark)?
  2. Equal-weight basket of the whole universe (true diversification)?

If yes for both → strategy has alpha worth pursuing.
If not → rotation at retail scale doesn't work either, accept BH.

Mechanics:
  - Monthly checkpoint (1st trading day of month) → re-rank, rebalance
  - Forced rebalance when current holding falls out of top-N
  - Equal weight 1/N each in top-N
  - 10 bps cost per traded fraction
  - All-cash fallback only if every top-N candidate has rotation_score
    below `score_floor` (rare — usually something is always relatively
    less bad)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_INITIAL_LOOKBACK_DAYS = 250
# Trading 212 EUR ETFs have zero commission — only bid-ask spread is real
# cost. For liquid UCITS (SXR8, IS3N, IUIT) spread is ~1-3 bps. For
# thematic (SEC0) ~5-10 bps. 5 bps is a conservative average for the
# universe; reality is likely lower for the broad-market positions.
_COST_BPS = 0.0005
_DRIFT_TRIGGER = 0.10  # rebalance if any position's weight drifts >10% from target


@dataclass
class RotationResult:
    """Summary of a rotation backtest run."""
    n_days: int
    n_years: float
    top_n: int

    # Strategy
    strategy_total_return: float
    strategy_cagr: float
    strategy_max_drawdown: float
    strategy_sharpe: float

    # Benchmark: SPY buy-and-hold
    spy_total_return: float
    spy_cagr: float
    spy_max_drawdown: float
    spy_sharpe: float

    # Benchmark: equal-weight universe basket buy-and-hold
    basket_total_return: float
    basket_cagr: float
    basket_max_drawdown: float
    basket_sharpe: float

    # Alphas
    alpha_vs_spy: float
    alpha_vs_basket: float

    # Trade activity
    n_rebalances: int
    n_trades: int
    total_cost_pct: float

    # Holdings statistics
    avg_holding_period_days: float
    top_tickers: list[str] = field(default_factory=list)  # most often held
    ticker_appearances: dict = field(default_factory=dict)


def _max_drawdown(values: list[float]) -> float:
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


def _annualized_sharpe(daily_values: list[float]) -> float:
    if len(daily_values) < 30:
        return 0.0
    returns = []
    for i in range(1, len(daily_values)):
        if daily_values[i - 1] > 0:
            returns.append(daily_values[i] / daily_values[i - 1] - 1.0)
    if len(returns) < 30:
        return 0.0
    arr = np.array(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 30:
        return 0.0
    mu = float(np.mean(arr)) * 252.0
    sd = float(np.std(arr)) * math.sqrt(252.0)
    return mu / sd if sd > 1e-9 else 0.0


def backtest_rotation(
    tickers: list[str],
    n_years: int = 3,
    top_n: int = 3,
    score_floor: float = 0.0,
    allocation_mode: str = "new_only",
    progress_callback=None,
) -> Optional[RotationResult]:
    """Run the top-N rotation backtest over n_years of history.

    `allocation_mode` controls how cash from sold positions is
    distributed at each monthly rebalance:

      "new_only"      → all cash goes to NEW entries (added) only;
                        kept holdings stay untouched. If no new entries,
                        cash distributes equally to kept (rare edge
                        case from dedup-collapse).
      "equal_top_n"   → cash from sells splits equally among ALL
                        members of new top-N (added + kept). Kept
                        positions get topped up.
      "score_weighted"→ same targets as equal_top_n but weighted by
                        rotation_score. Higher-conviction picks get
                        more cash.

    NO target weight enforcement — positions drift freely with
    performance. The user explicitly accepts concentrated exposure
    in winners to maximise alpha.

    Heavy lifting is upfront — fetch all histories once, align dates,
    then walk forward day by day. Signal computation only happens at
    checkpoint days (monthly), so even a 25-ETF × 5-year run is ~60
    checkpoints × 25 ETFs = 1500 signal computations ≈ 2-3 minutes.
    """
    from core.sector_radar import compute_signals, compute_rotation_scores

    # Step 1: fetch all histories
    histories: dict[str, pd.DataFrame] = {}
    try:
        import yfinance as yf
        for i, t in enumerate(tickers):
            if progress_callback is not None:
                try:
                    progress_callback(f"Fetching {t}", i, len(tickers))
                except Exception:
                    pass
            try:
                h = yf.Ticker(t).history(period=f"{n_years}y", auto_adjust=True)
                if h is not None and len(h) >= _INITIAL_LOOKBACK_DAYS:
                    histories[t] = h
                else:
                    logger.info(f"[rotation] {t}: insufficient history, skipping")
            except Exception as e:
                logger.warning(f"[rotation] {t} fetch failed: {e}")
    except ImportError:
        logger.error("[rotation] yfinance not available")
        return None

    if len(histories) < top_n:
        logger.error(f"[rotation] only {len(histories)} tickers usable, need ≥{top_n}")
        return None

    spy_hist = histories.get("SPY")
    if spy_hist is None:
        try:
            spy_hist = yf.Ticker("SPY").history(period=f"{n_years}y", auto_adjust=True)
        except Exception:
            spy_hist = None

    # Step 2: align dates — intersection of all histories
    common_dates = None
    for h in histories.values():
        dates_set = set(h.index)
        common_dates = dates_set if common_dates is None else (common_dates & dates_set)
    if not common_dates or len(common_dates) < _INITIAL_LOOKBACK_DAYS + 100:
        logger.error(f"[rotation] only {len(common_dates) if common_dates else 0} common dates")
        return None
    all_dates = sorted(common_dates)

    # Step 3: identify monthly checkpoints (first trading day of each month)
    checkpoints: list = []
    prev_month = None
    for d in all_dates:
        ts = pd.Timestamp(d)
        ym = (ts.year, ts.month)
        if ym != prev_month:
            checkpoints.append(d)
            prev_month = ym

    # Step 4: simulation
    start_idx = _INITIAL_LOOKBACK_DAYS  # need 200d for MA200 + buffer
    if start_idx >= len(all_dates):
        return None

    capital_cash = 1.0       # uninvested fraction
    positions: dict[str, float] = {}  # ticker → fraction of initial capital
    entry_prices: dict[str, float] = {}  # ticker → entry price (for tracking)
    daily_values: list[float] = []
    spy_values: list[float] = []
    basket_values: list[float] = []
    n_rebalances = 0
    n_trades = 0
    total_cost = 0.0
    current_top: set[str] = set()
    ticker_holdings_days: dict[str, int] = {}

    # SPY benchmark: track value if we'd bought $1 of SPY on start day
    spy_start_price = None
    if spy_hist is not None and len(spy_hist) > start_idx:
        for d in all_dates[start_idx:]:
            if d in spy_hist.index:
                spy_start_price = float(spy_hist.loc[d, "Close"])
                break

    # Basket benchmark: $1 split equally across all available tickers on start
    basket_shares: dict[str, float] = {}
    start_date = all_dates[start_idx]
    n_basket = sum(1 for h in histories.values() if start_date in h.index)
    if n_basket > 0:
        per_ticker = 1.0 / n_basket
        for t, h in histories.items():
            if start_date in h.index:
                p = float(h.loc[start_date, "Close"])
                if p > 0:
                    basket_shares[t] = per_ticker / p

    for i, date in enumerate(all_dates):
        if i < start_idx:
            continue

        # Compute portfolio value (positions + cash)
        portfolio_value = capital_cash
        for t, frac in positions.items():
            if t in histories and date in histories[t].index:
                if entry_prices.get(t):
                    current_price = float(histories[t].loc[date, "Close"])
                    portfolio_value += frac * (current_price / entry_prices[t])

        daily_values.append(portfolio_value)

        # SPY value
        if spy_hist is not None and date in spy_hist.index and spy_start_price:
            spy_p = float(spy_hist.loc[date, "Close"])
            spy_values.append(spy_p / spy_start_price)
        elif spy_values:
            spy_values.append(spy_values[-1])
        else:
            spy_values.append(1.0)

        # Basket value
        basket_value = 0.0
        for t, shares in basket_shares.items():
            if t in histories and date in histories[t].index:
                p = float(histories[t].loc[date, "Close"])
                basket_value += shares * p
        if basket_value <= 0 and basket_values:
            basket_value = basket_values[-1]
        basket_values.append(basket_value if basket_value > 0 else 1.0)

        # Track holding days for stats
        for t in positions:
            ticker_holdings_days[t] = ticker_holdings_days.get(t, 0) + 1

        # Rebalance check — monthly checkpoint ONLY.
        #
        # The original design had a weekly forced-rebalance trigger that
        # caught between-checkpoint regime changes, but the 1.4y test
        # showed it produced 363 trades — 44-day average holding period
        # vs. the ~30-day expected from pure monthly. That's nearly weekly
        # turnover, which inflates cost and starts to look like noise
        # trading. Pure monthly aligns with the AQR/Asness momentum
        # literature (which validates monthly cadence) and dramatically
        # reduces cost drag. If a sector falls hard between checkpoints,
        # the next monthly will rotate out — at most one month of stale
        # holding.
        is_checkpoint = date in checkpoints
        force_rebal = False

        if is_checkpoint:
            # Compute signals for all available tickers as of this date
            ticker_signals = {}
            for t, h in histories.items():
                hist_slice = h[h.index <= date]
                if len(hist_slice) < 200:
                    continue
                spy_slice = (spy_hist[spy_hist.index <= date] if spy_hist is not None else None)
                sig = compute_signals(t, hist_slice, spy_hist=spy_slice)
                if sig is not None:
                    ticker_signals[t] = sig

            if not ticker_signals:
                continue

            # Cross-sectional rotation scores
            scores = compute_rotation_scores(ticker_signals)

            # Sort high→low. Then pick top-N with group de-duplication
            # (SOXX/SMH are both semis — never both in top-N at once).
            sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
            # Apply score floor before dedup
            sorted_filtered = [(t, s) for t, s in sorted_scores if s >= score_floor]
            from core.etf_universe import select_top_n_deduped
            new_top = select_top_n_deduped(sorted_filtered, top_n)
            new_top_set = set(new_top)

            if new_top:
                # Rotation rebalance: trade only transitions, NO drift
                # correction (winners are allowed to keep growing).
                kept = current_top & new_top_set
                dropped = current_top - new_top_set
                added = new_top_set - current_top

                # 1) Sell dropped positions → cash
                for t in dropped:
                    if t in positions and t in histories and date in histories[t].index and entry_prices.get(t):
                        current_price = float(histories[t].loc[date, "Close"])
                        position_value = positions[t] * (current_price / entry_prices[t])
                        cost = position_value * _COST_BPS
                        total_cost += cost
                        capital_cash += position_value - cost
                        n_trades += 1
                    if t in positions:
                        del positions[t]
                    if t in entry_prices:
                        del entry_prices[t]

                # 2) Determine where the freed cash goes, per mode
                cash_allocation: dict[str, float] = {}
                if allocation_mode == "new_only":
                    # Cash → only new entries. Kept untouched.
                    if added:
                        per = capital_cash / len(added)
                        cash_allocation = {t: per for t in added}
                    elif kept:
                        # Fallback when dedup collapsed top-N and there
                        # are no new entries: distribute cash to kept.
                        per = capital_cash / len(kept)
                        cash_allocation = {t: per for t in kept}
                elif allocation_mode == "equal_top_n":
                    # Cash → split equally among ALL members of new top-N
                    # (both added and kept). Kept positions get topped up.
                    targets = added | kept
                    if targets:
                        per = capital_cash / len(targets)
                        cash_allocation = {t: per for t in targets}
                elif allocation_mode == "score_weighted":
                    # Cash → distributed by rotation_score weight across
                    # all new top-N. Higher-conviction picks get more.
                    targets = added | kept
                    if targets:
                        weights = {t: max(1.0, scores.get(t, 0.0)) for t in targets}
                        total_w = sum(weights.values())
                        for t in targets:
                            cash_allocation[t] = capital_cash * (weights[t] / total_w)
                else:
                    # Unknown mode — fail to new_only
                    if added:
                        per = capital_cash / len(added)
                        cash_allocation = {t: per for t in added}

                # 3) Execute allocations
                for t, cash_to_alloc in cash_allocation.items():
                    if cash_to_alloc <= 0:
                        continue
                    if t not in histories or date not in histories[t].index:
                        continue
                    price = float(histories[t].loc[date, "Close"])
                    if price <= 0:
                        continue
                    cost = cash_to_alloc * _COST_BPS
                    total_cost += cost
                    net_invested = cash_to_alloc - cost

                    if t in added:
                        # Brand new position
                        positions[t] = net_invested
                        entry_prices[t] = price
                    else:
                        # Top up existing kept position. Re-anchor the
                        # `positions[t]` scalar so that
                        # positions[t] * (current_price / entry_price)
                        # equals the new total value at the current price.
                        current_value = positions[t] * (price / entry_prices[t])
                        new_total_value = current_value + net_invested
                        positions[t] = new_total_value * (entry_prices[t] / price)

                    capital_cash -= cash_to_alloc
                    n_trades += 1

                current_top = new_top_set
                n_rebalances += 1

    # Compute metrics
    if not daily_values or not spy_values or not basket_values:
        return None

    n_days = len(daily_values)
    n_years_actual = n_days / 252.0

    strategy_total = (daily_values[-1] / daily_values[0]) - 1.0 if daily_values[0] > 0 else 0.0
    spy_total = (spy_values[-1] / spy_values[0]) - 1.0 if spy_values[0] > 0 else 0.0
    basket_total = (basket_values[-1] / basket_values[0]) - 1.0 if basket_values[0] > 0 else 0.0

    strategy_cagr = (daily_values[-1] / daily_values[0]) ** (1 / n_years_actual) - 1.0 if n_years_actual > 0 and daily_values[0] > 0 else 0.0
    spy_cagr = (spy_values[-1] / spy_values[0]) ** (1 / n_years_actual) - 1.0 if n_years_actual > 0 and spy_values[0] > 0 else 0.0
    basket_cagr = (basket_values[-1] / basket_values[0]) ** (1 / n_years_actual) - 1.0 if n_years_actual > 0 and basket_values[0] > 0 else 0.0

    # Most-held tickers
    sorted_holdings = sorted(ticker_holdings_days.items(), key=lambda x: -x[1])
    top_held = [t for t, _ in sorted_holdings[:5]]

    avg_holding_period = (
        sum(ticker_holdings_days.values()) / max(1, len(ticker_holdings_days))
    )

    return RotationResult(
        n_days=n_days,
        n_years=round(n_years_actual, 2),
        top_n=top_n,
        strategy_total_return=round(strategy_total, 4),
        strategy_cagr=round(strategy_cagr, 4),
        strategy_max_drawdown=round(_max_drawdown(daily_values), 4),
        strategy_sharpe=round(_annualized_sharpe(daily_values), 3),
        spy_total_return=round(spy_total, 4),
        spy_cagr=round(spy_cagr, 4),
        spy_max_drawdown=round(_max_drawdown(spy_values), 4),
        spy_sharpe=round(_annualized_sharpe(spy_values), 3),
        basket_total_return=round(basket_total, 4),
        basket_cagr=round(basket_cagr, 4),
        basket_max_drawdown=round(_max_drawdown(basket_values), 4),
        basket_sharpe=round(_annualized_sharpe(basket_values), 3),
        alpha_vs_spy=round(strategy_total - spy_total, 4),
        alpha_vs_basket=round(strategy_total - basket_total, 4),
        n_rebalances=n_rebalances,
        n_trades=n_trades,
        total_cost_pct=round(total_cost, 4),
        avg_holding_period_days=round(avg_holding_period, 1),
        top_tickers=top_held,
        ticker_appearances=dict(sorted_holdings[:15]),
    )
