"""
chart_generator.py — Performance chart generation for public transparency.

Generates 4 PNG charts + 1 Markdown audit report from paper_trading data:
  1. equity_curve.png    — cumulative alpha over time (DIP + MOMENTUM closed trades)
  2. monthly_alpha.png   — monthly alpha bar chart with win rate
  3. model_breakdown.png — win rate + avg alpha per strategy
  4. core_holdings.png   — CORE open positions: unrealized P&L + yield income
  5. drip_ledger.md      — full DRIP audit trail with every dividend event and tax disclaimer

Charts and report saved to /data/charts/ (Railway Volume).
Call generate_all_charts() to produce all four charts + the markdown ledger.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_DATA_DIR   = Path("/data") if Path("/data").exists() else Path("/tmp")
_CHARTS_DIR = _DATA_DIR / "charts"

# Palette
_C_DIP      = "#2563EB"
_C_MOMENTUM = "#16A34A"
_C_CORE     = "#7C3AED"
_C_SPY      = "#9CA3AF"
_C_POS      = "#16A34A"
_C_NEG      = "#DC2626"
_C_BG       = "#FFFFFF"
_C_GRID     = "#F3F4F6"
_C_TEXT     = "#111827"
_C_SUBTEXT  = "#6B7280"

_WATERMARK  = "DipRadar — live signals, not backtest"


def _setup_matplotlib() -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "figure.facecolor": _C_BG,
            "axes.facecolor":   _C_BG,
            "axes.edgecolor":   "#E5E7EB",
            "axes.labelcolor":  _C_TEXT,
            "axes.grid":        True,
            "grid.color":       _C_GRID,
            "grid.linestyle":   "-",
            "grid.linewidth":   0.7,
            "text.color":       _C_TEXT,
            "xtick.color":      _C_SUBTEXT,
            "ytick.color":      _C_SUBTEXT,
            "font.family":      "sans-serif",
            "font.size":        10,
        })
        return True
    except ImportError:
        log.warning("[charts] matplotlib not installed — chart generation skipped")
        return False


def _save(fig, path: Path) -> Path:
    _CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[charts] saved: {path.name}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# SPY benchmark fetch
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_spy_returns(start_date: str, end_date: str) -> tuple[list, list]:
    """Fetch SPY daily closes and return (dates, cumulative_pct) starting from 0%."""
    try:
        import yfinance as yf
        import pandas as pd
        h = yf.Ticker("SPY").history(start=start_date, end=end_date, auto_adjust=True)
        if h is None or h.empty:
            return [], []
        if h.index.tz is not None:
            h.index = h.index.tz_convert(None)
        closes = h["Close"].dropna()
        if closes.empty:
            return [], []
        base = float(closes.iloc[0])
        cum_pct = ((closes / base) - 1.0) * 100.0
        dates = [d.to_pydatetime() for d in cum_pct.index]
        return dates, list(cum_pct.values)
    except Exception as e:
        log.debug(f"[charts] SPY fetch failed: {e}")
        return [], []


# ─────────────────────────────────────────────────────────────────────────────
# 1. Equity curve — portfolio % return vs SPY
# ─────────────────────────────────────────────────────────────────────────────

def generate_equity_curve(trades: list[dict]) -> Optional[Path]:
    """Portfolio cumulative % return vs SPY buy-and-hold, from first trade date.

    Both lines start at 0%. Portfolio curve compounds closed P&L relative to the
    starting pool (PAPER_PORTFOLIO_EUR). SPY is fetched live for the same period.
    """
    if not _setup_matplotlib():
        return None
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from paper_trading import PAPER_PORTFOLIO_EUR

    closed = [
        t for t in trades
        if t.get("status", "").startswith("CLOSED")
        and t.get("close_date")
        and t.get("return_pct") is not None
        and t.get("amount_eur") is not None
        and t.get("trade_type") in ("DIP", "MOMENTUM")
    ]
    if len(closed) < 2:
        return None

    closed.sort(key=lambda t: t["close_date"])

    # Build portfolio equity curve as cumulative % of starting pool
    first_open = min(
        (t.get("open_date") or t["close_date"]) for t in closed
    )
    port_dates, port_pcts = [], []
    running_pnl = 0.0
    for t in closed:
        pnl = float(t["amount_eur"]) * float(t["return_pct"]) / 100.0
        running_pnl += pnl
        port_dates.append(datetime.fromisoformat(t["close_date"]))
        port_pcts.append(round(running_pnl / max(PAPER_PORTFOLIO_EUR, 1.0) * 100.0, 3))

    # Prepend starting point (0%)
    start_dt = datetime.fromisoformat(first_open)
    port_dates = [start_dt] + port_dates
    port_pcts  = [0.0] + port_pcts

    # SPY fetched for the exact same period as the portfolio (PIT comparison).
    # Using last close date rather than today avoids inflating/deflating alpha
    # when the report is generated weeks after the last trade closed.
    last_close = max(t["close_date"] for t in closed)
    spy_dates, spy_pcts = _fetch_spy_returns(first_open, last_close)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Portfolio line
    final_pct = port_pcts[-1]
    port_color = _C_POS if final_pct >= 0 else _C_NEG
    ax.plot(port_dates, port_pcts, color=port_color, linewidth=2.4,
            label=f"DipRadar portfolio ({final_pct:+.1f}%)", zorder=4)
    ax.fill_between(port_dates, 0, port_pcts,
                    alpha=0.09, color=port_color, zorder=2)
    ax.annotate(
        f"{final_pct:+.1f}%",
        xy=(port_dates[-1], port_pcts[-1]),
        xytext=(6, 0), textcoords="offset points",
        color=port_color, fontsize=10, fontweight="bold", va="center",
    )

    # SPY line
    if spy_dates and spy_pcts:
        spy_final = spy_pcts[-1]
        ax.plot(spy_dates, spy_pcts, color=_C_SPY, linewidth=1.8, linestyle="--",
                label=f"SPY buy & hold ({spy_final:+.1f}%)", zorder=3)
        ax.annotate(
            f"{spy_final:+.1f}%",
            xy=(spy_dates[-1], spy_pcts[-1]),
            xytext=(6, 0), textcoords="offset points",
            color=_C_SPY, fontsize=9, fontweight="bold", va="center",
        )

    ax.axhline(0, color="#D1D5DB", linewidth=1, linestyle="-", zorder=1)

    n_total = len(closed)
    n_missed = sum(1 for t in trades if t.get("status") == "MISSED_CAPITAL")
    missed_note = f" | {n_missed} signals missed (capital deployed)" if n_missed else ""
    ax.set_title(
        f"Portfolio Return vs SPY — {n_total} closed trades "
        f"({first_open[:7]} → {last_close[:7]}){missed_note}",
        fontsize=13, fontweight="bold", color=_C_TEXT, pad=14,
    )
    ax.set_ylabel("Cumulative return (%)", color=_C_SUBTEXT)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.1f}%"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    plt.xticks(rotation=30, ha="right")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.99, 0.02, _WATERMARK, transform=ax.transAxes,
            ha="right", va="bottom", color=_C_SUBTEXT, fontsize=8, style="italic")

    return _save(fig, _CHARTS_DIR / "equity_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Monthly alpha bar chart + win rate sparkline
# ─────────────────────────────────────────────────────────────────────────────

def generate_monthly_alpha_chart(trades: list[dict]) -> Optional[Path]:
    """Bar chart of capital-weighted alpha by month + win rate subplot."""
    if not _setup_matplotlib():
        return None
    import matplotlib.pyplot as plt

    closed = [
        t for t in trades
        if t.get("status", "").startswith("CLOSED")
        and t.get("close_date")
        and t.get("return_pct") is not None
    ]
    if not closed:
        return None

    # Group by month (last 12)
    monthly: dict[str, list] = {}
    for t in closed:
        monthly.setdefault(t["close_date"][:7], []).append(t)
    months = sorted(monthly)[-12:]

    labels, alphas, win_rates, ns = [], [], [], []
    for m in months:
        group = monthly[m]
        total_amt = sum(float(t.get("amount_eur", 1)) for t in group)
        w_alpha = (
            sum(float(t.get("alpha_pct") or 0) * float(t.get("amount_eur", 1)) for t in group)
            / max(total_amt, 1)
        )
        wr = sum(1 for t in group if float(t.get("return_pct", 0) or 0) > 0) / len(group)
        labels.append(m)
        alphas.append(round(w_alpha, 2))
        win_rates.append(round(wr * 100, 1))
        ns.append(len(group))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7),
                                   gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    bar_colors = [_C_POS if a >= 0 else _C_NEG for a in alphas]
    bars = ax1.bar(range(len(labels)), alphas, color=bar_colors, alpha=0.82, width=0.6)

    for i, (bar, a) in enumerate(zip(bars, alphas)):
        ypos = a + 0.15 if a >= 0 else a - 0.15
        va = "bottom" if a >= 0 else "top"
        ax1.text(i, ypos, f"{a:+.1f}pp", ha="center", va=va, fontsize=8, color=_C_TEXT)

    ax1.axhline(0, color=_C_SPY, linewidth=1, linestyle="--")
    ax1.set_title("Monthly Return vs SPY (capital-weighted) — last 12 months",
                  fontsize=13, fontweight="bold", color=_C_TEXT, pad=12)
    ax1.set_ylabel("Alpha (%)", color=_C_SUBTEXT)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.1f}%"))
    ax1.text(0.99, 0.04, _WATERMARK, transform=ax1.transAxes,
             ha="right", va="bottom", color=_C_SUBTEXT, fontsize=8, style="italic")

    ax2.bar(range(len(labels)), win_rates, color=_C_DIP, alpha=0.55, width=0.6)
    ax2.axhline(50, color=_C_SPY, linewidth=1, linestyle="--")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Win %", color=_C_SUBTEXT, fontsize=9)
    for i, (n, wr) in enumerate(zip(ns, win_rates)):
        ax2.text(i, 4, f"n={n}", ha="center", fontsize=7, color=_C_SUBTEXT)

    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)

    return _save(fig, _CHARTS_DIR / "monthly_alpha.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Strategy breakdown — win rate + avg alpha per model/type
# ─────────────────────────────────────────────────────────────────────────────

def generate_model_breakdown(trades: list[dict]) -> Optional[Path]:
    """Win rate and avg alpha per strategy (DIP / MOMENTUM / CORE closed)."""
    if not _setup_matplotlib():
        return None
    import matplotlib.pyplot as plt
    import numpy as np

    closed = [
        t for t in trades
        if t.get("status", "").startswith("CLOSED")
        and t.get("return_pct") is not None
    ]
    if not closed:
        return None

    # Group by model_id if stored, else trade_type
    groups: dict[str, list] = {}
    for t in closed:
        key = t.get("model_id") or t.get("trade_type", "DIP")
        groups.setdefault(key, []).append(t)

    if not groups:
        return None

    _labels_order = ["dip_us", "DIP", "momentum_us", "MOMENTUM", "core_us", "CORE"]
    labels = [l for l in _labels_order if l in groups] + \
             [l for l in sorted(groups) if l not in _labels_order]

    _color_map = {
        "DIP": _C_DIP, "dip_us": _C_DIP,
        "MOMENTUM": _C_MOMENTUM, "momentum_us": _C_MOMENTUM,
        "CORE": _C_CORE, "core_us": _C_CORE,
    }

    win_rates, avg_alphas, ns = [], [], []
    for g in labels:
        group = groups[g]
        wr = sum(1 for t in group if float(t.get("return_pct", 0) or 0) > 0) / max(len(group), 1)
        aa = sum(float(t.get("alpha_pct", 0) or 0) for t in group) / max(len(group), 1)
        win_rates.append(round(wr * 100, 1))
        avg_alphas.append(round(aa, 2))
        ns.append(len(group))

    x = np.arange(len(labels))
    bar_colors = [_color_map.get(l, _C_DIP) for l in labels]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    w = 0.35
    ax1.bar(x - w / 2, win_rates, w, color=bar_colors, alpha=0.85, label="Win rate %")
    ax2.bar(x + w / 2, avg_alphas, w, color=bar_colors, alpha=0.40,
            hatch="//", label="Avg alpha (pp)")

    ax1.axhline(50, color=_C_SPY, linewidth=1, linestyle="--", alpha=0.6)
    ax2.axhline(0,  color=_C_SPY, linewidth=1, linestyle="-",  alpha=0.3)

    ax1.set_ylabel("Win rate (%)", color=_C_TEXT)
    ax2.set_ylabel("Avg alpha (%)", color=_C_TEXT)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.1f}%"))
    ax1.set_title("Performance by Strategy", fontsize=13, fontweight="bold",
                  color=_C_TEXT, pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, ns)])

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right")
    ax1.text(0.99, 0.02, _WATERMARK, transform=ax1.transAxes,
             ha="right", va="bottom", color=_C_SUBTEXT, fontsize=8, style="italic")

    return _save(fig, _CHARTS_DIR / "model_breakdown.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. CORE holdings — unrealized P&L + yield income (open positions)
# ─────────────────────────────────────────────────────────────────────────────

def generate_core_holdings_chart(trades: list[dict]) -> Optional[Path]:
    """Horizontal bar chart: CORE open positions showing P&L and accumulated yield."""
    if not _setup_matplotlib():
        return None
    import matplotlib.pyplot as plt
    import numpy as np

    from datetime import date as _date

    core = [
        t for t in trades
        if t.get("trade_type") == "CORE" and t.get("status") == "OPEN"
        and t.get("open_price") and t.get("amount_eur")
    ]
    if not core:
        return None

    today = _date.today()
    tickers, pnl_pcts, yield_pcts, days_held_list = [], [], [], []

    for t in core:
        ticker = t["ticker"]
        open_price = float(t["open_price"])
        amount_eur = float(t.get("amount_eur", 0))
        open_date  = _date.fromisoformat(t["open_date"])
        days = (today - open_date).days

        # Try to use stored return if recently updated
        ret_pct = float(t.get("ret_pct") or 0)  # from get_core_holdings_summary

        # Yield income (pro-rata)
        annual_yield = float(t.get("annual_yield_pct") or 0)
        yield_pct_accum = annual_yield * (days / 365) if days > 0 else 0.0

        tickers.append(f"{ticker} ({days}d)")
        pnl_pcts.append(round(ret_pct, 2))
        yield_pcts.append(round(yield_pct_accum, 2))
        days_held_list.append(days)

    if not tickers:
        return None

    y = np.arange(len(tickers))
    fig, ax = plt.subplots(figsize=(10, max(4, len(tickers) * 0.9 + 1.5)))

    bar_colors = [_C_POS if p >= 0 else _C_NEG for p in pnl_pcts]
    ax.barh(y, pnl_pcts, 0.5, color=bar_colors, alpha=0.85, label="Price P&L (%)")
    ax.barh(y, yield_pcts, 0.5, left=pnl_pcts, color=_C_CORE, alpha=0.55,
            label="Accumulated yield (%)")

    total = [p + yld for p, yld in zip(pnl_pcts, yield_pcts)]
    for i, (p, yld, tot) in enumerate(zip(pnl_pcts, yield_pcts, total)):
        ax.text(tot + 0.3, i, f"{tot:+.1f}%", va="center", fontsize=9,
                color=_C_POS if tot >= 0 else _C_NEG, fontweight="bold")

    ax.axvline(0, color=_C_SPY, linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(tickers)
    ax.set_xlabel("Return (%)", color=_C_SUBTEXT)
    ax.set_title("CORE Holdings — Unrealized P&L + Yield Income",
                 fontsize=13, fontweight="bold", color=_C_TEXT, pad=12)
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.99, 0.02, _WATERMARK, transform=ax.transAxes,
            ha="right", va="bottom", color=_C_SUBTEXT, fontsize=8, style="italic")

    return _save(fig, _CHARTS_DIR / "core_holdings.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Per-trade card — stock vs SPY for the exact holding period
# ─────────────────────────────────────────────────────────────────────────────

def generate_trade_card(trade: dict) -> Optional[Path]:
    """Mini performance card for a single closed trade.

    Shows the stock vs SPY normalised to 0% at entry, with vertical markers
    at entry and exit dates, and a stats box showing return %, SPY %, and alpha.
    Saved to charts/trade_cards/{TICKER}_{open_date}.png.
    """
    if not _setup_matplotlib():
        return None

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd

    ticker      = str(trade.get("ticker", ""))
    open_date   = str(trade.get("open_date", ""))
    close_date  = str(trade.get("close_date", ""))
    open_price  = float(trade.get("open_price") or 0)
    ret_pct     = trade.get("return_pct")
    alpha_pct   = trade.get("alpha_pct")
    spy_ret     = trade.get("spy_return_pct")
    trade_type  = trade.get("trade_type", "DIP")
    status      = str(trade.get("status", "CLOSED"))

    if not ticker or not open_date or not close_date or open_price <= 0:
        return None

    try:
        import yfinance as yf
        _open_dt  = datetime.fromisoformat(open_date)
        _close_dt = datetime.fromisoformat(close_date)
        # 5-day buffer before entry and 3 days after exit for visual context
        _start = (_open_dt  - timedelta(days=7)).strftime("%Y-%m-%d")
        _end   = (_close_dt + timedelta(days=5)).strftime("%Y-%m-%d")

        tick_h = yf.Ticker(ticker).history(start=_start, end=_end, auto_adjust=True)
        spy_h  = yf.Ticker("SPY").history(start=_start, end=_end, auto_adjust=True)

        if tick_h is None or tick_h.empty or spy_h is None or spy_h.empty:
            return None

        for _df in (tick_h, spy_h):
            if _df.index.tz is not None:
                _df.index = _df.index.tz_convert(None)

    except Exception as e:
        log.debug(f"[trade_card] fetch failed {ticker}: {e}")
        return None

    # Reference point: first close on or after open_date (entry day)
    _open_ts  = pd.Timestamp(open_date)
    _close_ts = pd.Timestamp(close_date)

    tick_from_entry = tick_h[tick_h.index >= _open_ts]
    spy_from_entry  = spy_h[spy_h.index  >= _open_ts]
    if tick_from_entry.empty or spy_from_entry.empty:
        return None

    tick_ref = float(tick_from_entry["Close"].iloc[0])
    spy_ref  = float(spy_from_entry["Close"].iloc[0])
    if tick_ref <= 0 or spy_ref <= 0:
        return None

    # Normalise both series to 0% at entry
    tick_norm = ((tick_h["Close"] / tick_ref) - 1.0) * 100.0
    spy_norm  = ((spy_h["Close"] / spy_ref)  - 1.0) * 100.0

    # ── Chart ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4.5))

    is_win     = (ret_pct or 0) > 0
    t_color    = _C_POS if is_win else _C_NEG
    type_color = {
        "DIP":      _C_DIP,
        "MOMENTUM": _C_MOMENTUM,
        "CORE":     _C_CORE,
    }.get(trade_type, _C_DIP)

    ax.plot(tick_norm.index, tick_norm.values,
            color=t_color, linewidth=2.2, label=ticker, zorder=4)
    ax.plot(spy_norm.index, spy_norm.values,
            color=_C_SPY, linewidth=1.6, linestyle="--", label="SPY", zorder=3)

    # Entry / exit vertical markers
    ax.axvline(_open_ts,  color=type_color, linewidth=1.4, linestyle=":", alpha=0.9, zorder=2)
    ax.axvline(_close_ts, color="#374151",  linewidth=1.4, linestyle=":", alpha=0.9, zorder=2)

    # Entry dot at 0%
    ax.scatter([_open_ts], [0.0], color=type_color, s=55, zorder=5)

    # Exit dot on the ticker line
    tick_at_exit = tick_norm[tick_norm.index <= _close_ts]
    if not tick_at_exit.empty:
        exit_val = float(tick_at_exit.iloc[-1])
        ax.scatter([_close_ts], [exit_val],
                   color=t_color, s=55, marker="D", zorder=5)

    ax.axhline(0, color="#D1D5DB", linewidth=0.9, linestyle="-", zorder=1)

    # ── Stats box (top-left) ─────────────────────────────────────────────────
    days_held = (_close_dt - _open_dt).days
    ret_s   = f"{ret_pct:+.1f}%"   if ret_pct   is not None else "—"
    spy_s   = f"{spy_ret:+.1f}%"   if spy_ret   is not None else "—"
    alpha_s = f"{alpha_pct:+.1f}%" if alpha_pct is not None else "—"
    status_label = status.replace("CLOSED_", "").replace("_", " ").title()

    box_text = (
        f"{ticker}  ·  {trade_type}\n"
        f"Entry {open_date}  →  Exit {close_date}  ({days_held}d)\n"
        f"Return: {ret_s}   SPY: {spy_s}   Alpha: {alpha_s}"
    )
    ax.text(
        0.01, 0.97, box_text,
        transform=ax.transAxes, va="top", ha="left", fontsize=9, color=_C_TEXT,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F9FAFB",
                  edgecolor="#E5E7EB", alpha=0.95),
    )

    # Status badge (top-right)
    ax.text(
        0.99, 0.97, status_label,
        transform=ax.transAxes, va="top", ha="right",
        fontsize=9, color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35",
                  facecolor=_C_POS if is_win else _C_NEG, edgecolor="none"),
    )

    ax.set_ylabel("Return since entry (%)", color=_C_SUBTEXT)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.1f}%"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b '%y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
    plt.xticks(rotation=25, ha="right")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.text(0.99, 0.02, _WATERMARK, transform=ax.transAxes,
            ha="right", va="bottom", color=_C_SUBTEXT, fontsize=7.5, style="italic")

    plt.tight_layout()

    out_dir  = _CHARTS_DIR / "trade_cards"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}_{open_date}.png"
    return _save(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_charts() -> list[Path]:
    """Generate all 4 performance charts. Returns list of created file paths.

    Also tries to enrich CORE holdings with live P&L from paper_trading before
    passing to generate_core_holdings_chart.
    """
    try:
        from paper_trading import _load, get_core_holdings_summary
        trades = _load()
    except Exception as e:
        log.error(f"[charts] Could not load trades: {e}")
        return []

    # Enrich CORE open positions with live P&L from get_core_holdings_summary
    try:
        core_live = {h["ticker"]: h for h in get_core_holdings_summary()}
        for t in trades:
            if t.get("trade_type") == "CORE" and t.get("status") == "OPEN":
                live = core_live.get(t["ticker"], {})
                if live:
                    t["ret_pct"] = live.get("ret_pct", 0)
                    t["annual_yield_pct"] = live.get("annual_yield_pct", t.get("annual_yield_pct", 0))
    except Exception as e:
        log.debug(f"[charts] CORE live enrichment: {e}")

    generators = [
        ("equity_curve",    generate_equity_curve),
        ("monthly_alpha",   generate_monthly_alpha_chart),
        ("model_breakdown", generate_model_breakdown),
        ("core_holdings",   generate_core_holdings_chart),
    ]

    charts: list[Path] = []
    for name, fn in generators:
        try:
            path = fn(trades)
            if path and path.exists():
                charts.append(path)
                log.info(f"[charts] OK: {name}")
            else:
                log.info(f"[charts] SKIP: {name} (insufficient data)")
        except Exception as e:
            log.warning(f"[charts] {name} failed: {e}")

    # Generate DRIP ledger markdown — always (even with zero events, for transparency)
    drip_path = generate_drip_report()
    if drip_path:
        charts.append(drip_path)
        log.info("[charts] OK: drip_ledger")

    return charts


def generate_drip_report() -> Optional[Path]:
    """Generate drip_ledger.md — full DRIP audit trail for public transparency.

    Saved to /data/charts/drip_ledger.md alongside the PNG charts.
    Published to GitHub via push_all_charts() which handles both .png and .md files.

    Returns the path to the written file, or None on error.
    """
    try:
        from paper_trading import format_drip_report_md
        md_content = format_drip_report_md()
    except Exception as e:
        log.warning(f"[drip_report] Could not generate DRIP markdown: {e}")
        return None

    try:
        _CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        out = _CHARTS_DIR / "drip_ledger.md"
        out.write_text(md_content, encoding="utf-8")
        log.info(f"[drip_report] Written {len(md_content)} chars → {out}")
        return out
    except Exception as e:
        log.warning(f"[drip_report] Could not write drip_ledger.md: {e}")
        return None
