"""core/honest_watchlist.py — Quality compounders in drawdown with near catalyst.

NOT a ranker. NOT alpha. A filter + display, honest about its scope.

Per F.5 backtest verdict (REVERT): the rule-based dislocation_z scorer
failed its binding gate (top-decile -1.09pp vs SPY, worst-fold -22.48pp).
The signals that distinguish "fear" from "deterioration" — estimate
revisions, news guidance, forward valuation — need PIT historical data
we don't have free. Without them, ranking by composite z confuses
dislocations with value traps.

The honest replacement: don't claim ranking. Filter for the names that
combine quality + drawdown + near catalyst. Show sector-relative as a
DISPLAY column (lets the user visually triage "fell with market" from
"fell alone") but do NOT filter by it — that would re-shut the door on
ADBE-style absolute drops in a market correction.

Criteria (ALL must hold):
  1. quality_passes_gates:
       revenue_growth_yoy >= -10%   (not collapsing)
       gross_margin       >= 25%    (real moat)
       fcf_yield          >= 0      (or in documented investment phase)
  2. dd_from_52w_high <= -25%       (drawdown material)
  3. days_to_catalyst <= 30         (concrete near-term event;
                                     90d was decorative — quarterlies
                                     cycle ~every 90d so it caught
                                     everyone. 30d is meaningful.)

Output sorted by dd_from_52w_high ascending (deepest first). The user
reads the list top-to-bottom and decides which names have a story.
No "this is the best alpha pick" claim.

Consumers: /scan_now, /analisar appendix, watchdog cron, /cash (via
dispatcher when DIPRADAR_SCORING_MODE=watchlist, which is the default).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Optional

log = logging.getLogger("watchlist")

# Default thresholds — exposed so the bot admin can tighten/loosen via
# env or CLI if needed. Picked once based on user F.5 review:
#   DD 25% = "material drawdown" (not noise; not catastrophic-only)
#   Catalyst 30d = "concrete near-term event" (90d included everyone)
DEFAULT_MIN_DD_PCT          = -0.25
DEFAULT_CATALYST_MAX_DAYS   = 30
DEFAULT_CATALYST_MIN_DAYS   = -10   # accept "just reported" too (post-earnings revaluation window)

# Last get_watchlist() run breakdown — exposed for /cash and /scan_now
# to surface "why watchlist is empty" honestly. Reset on each call.
_last_run_stats: dict = {}


def get_last_run_stats() -> dict:
    """Per-gate breakdown of the most recent get_watchlist() call.

    Used by callers (/cash, /scan_now) to render an honest diagnostic
    when the watchlist is empty — answers "why?" with concrete numbers
    instead of leaving the user guessing if it's the filter or a bug.

    Schema: {n_universe, n_dd_fail, n_quality_fail, n_catalyst_fail,
             n_qualified, fetched_at_iso}
    """
    return dict(_last_run_stats)


@dataclass
class WatchlistEntry:
    ticker:               str
    qualifies:            bool
    # Drawdown
    dd_from_52w_high:     Optional[float]   # negative; -0.44 = -44%
    # Sector context (DISPLAY ONLY — not used to filter)
    sector:               str
    sector_relative_60d:  Optional[float]   # ticker_60d - sector_etf_60d; negative = underperformed
    # Catalyst
    days_to_catalyst:     Optional[int]
    catalyst_source:      str               # "earnings_calendar" | "missing"
    # Quality (gate floor + display)
    quality_passes:       bool
    quality_reason:       str               # exclude reason when gate fails
    roe:                  Optional[float]
    gross_margin:         Optional[float]
    revenue_growth:       Optional[float]
    fcf_yield:            Optional[float]
    # Misc
    current_price:        Optional[float]
    core_score:           Optional[float]   # CoreRadar score (NOT used for ranking; for cross-reference)
    summary_line:         str               # pre-rendered display line

    def to_dict(self) -> dict:
        return asdict(self)


def _quality_passes(signals: dict) -> tuple[bool, str]:
    """Hard quality gate. Matches core_radar criteria.

    Missing fields default to PASS (we don't exclude on missing data).
    """
    rg = signals.get("revenue_growth")
    if rg is not None and rg < -0.10:
        return False, f"revenue_growth {rg*100:+.0f}% < -10%"
    gm = signals.get("gross_margin")
    if gm is not None and gm < 0.25:
        return False, f"gross_margin {gm*100:.0f}% < 25%"
    fcf = signals.get("fcf_yield_fund") or signals.get("fcf_yield")
    if fcf is not None and fcf < 0 and not signals.get("is_investment_phase"):
        return False, f"fcf_yield {fcf*100:+.1f}% < 0 (not investment phase)"
    return True, ""


def _fetch_sector_for(ticker: str) -> str:
    """Lazy yfinance.info sector lookup. Returns '' on failure."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        return info.get("sector", "") or ""
    except Exception as e:
        log.debug(f"[watchlist] sector fetch {ticker}: {e}")
        return ""


def _fetch_catalyst_days(ticker: str) -> Optional[int]:
    """Days to next earnings. None if not available within yfinance's window."""
    try:
        from data.market_client import get_earnings_days
        return get_earnings_days(ticker)
    except Exception as e:
        log.debug(f"[watchlist] catalyst fetch {ticker}: {e}")
        return None


def _fetch_sector_relative_60d(ticker: str, sector: str) -> Optional[float]:
    """Sector-relative 60d return. Returns None when sector unknown
    or price hist missing."""
    if not sector:
        return None
    try:
        from data.sector_relative import sector_relative_return
        return sector_relative_return(ticker, sector, window_days=60)
    except Exception as e:
        log.debug(f"[watchlist] sector_rel fetch {ticker}: {e}")
        return None


def _format_summary_line(entry: WatchlistEntry) -> str:
    """Format the one-line display per the agreed output format:

    'NVO  -44% from 52w | vs sector -31% | catalyst 24d | ROE 71%, GM 84%'

    When fields are missing the placeholder is honest (n/a), not zero.
    """
    dd = entry.dd_from_52w_high
    dd_str = f"{dd*100:+.0f}% from 52w" if dd is not None else "DD n/a"

    sr = entry.sector_relative_60d
    sr_str = f"vs sector {sr*100:+.0f}%" if sr is not None else "vs sector n/a"

    cd = entry.days_to_catalyst
    if cd is None:
        cat_str = "catalyst n/a"
    elif cd < 0:
        cat_str = f"catalyst {-cd}d ago"
    else:
        cat_str = f"catalyst {cd}d"

    quals = []
    if entry.roe is not None:
        quals.append(f"ROE {entry.roe*100:.0f}%")
    if entry.gross_margin is not None:
        quals.append(f"GM {entry.gross_margin*100:.0f}%")
    qual_str = ", ".join(quals) if quals else "quality n/a"

    return f"{entry.ticker:6s} {dd_str:>16s}  |  {sr_str:>16s}  |  {cat_str:>14s}  |  {qual_str}"


def evaluate_ticker(
    ticker: str,
    signals: dict,
    *,
    min_dd_pct: float = DEFAULT_MIN_DD_PCT,
    catalyst_max_days: int = DEFAULT_CATALYST_MAX_DAYS,
    catalyst_min_days: int = DEFAULT_CATALYST_MIN_DAYS,
) -> WatchlistEntry:
    """Build a WatchlistEntry from a CoreRadar signals dict.

    `signals` should be the dict produced by core_radar._score_core_dip
    (has roe, gross_margin, dd_from_52w_high, etc.).

    Always returns an entry; `entry.qualifies` indicates whether all
    criteria passed. Use this for both "should I add to watchlist?" and
    "show me why this candidate did/didn't qualify" (e.g. /analisar).
    """
    dd = signals.get("dd_from_52w_high")
    dd_ok = (dd is not None) and (dd <= min_dd_pct)

    quality_passes, quality_reason = _quality_passes(signals)

    sector = signals.get("sector") or _fetch_sector_for(ticker)
    sector_rel = _fetch_sector_relative_60d(ticker, sector)

    catalyst_days = _fetch_catalyst_days(ticker)
    catalyst_ok = (
        catalyst_days is not None
        and catalyst_min_days <= catalyst_days <= catalyst_max_days
    )

    qualifies = bool(quality_passes and dd_ok and catalyst_ok)

    entry = WatchlistEntry(
        ticker=ticker,
        qualifies=qualifies,
        dd_from_52w_high=dd,
        sector=sector,
        sector_relative_60d=sector_rel,
        days_to_catalyst=catalyst_days,
        catalyst_source="earnings_calendar" if catalyst_days is not None else "missing",
        quality_passes=quality_passes,
        quality_reason=quality_reason,
        roe=signals.get("roe"),
        gross_margin=signals.get("gross_margin"),
        revenue_growth=signals.get("revenue_growth"),
        fcf_yield=signals.get("fcf_yield_fund") or signals.get("fcf_yield"),
        current_price=signals.get("current_price"),
        core_score=signals.get("core_score") or signals.get("score"),
        summary_line="",
    )
    entry.summary_line = _format_summary_line(entry)
    return entry


def get_watchlist(
    *,
    min_dd_pct: float = DEFAULT_MIN_DD_PCT,
    catalyst_max_days: int = DEFAULT_CATALYST_MAX_DAYS,
    catalyst_min_days: int = DEFAULT_CATALYST_MIN_DAYS,
) -> list[WatchlistEntry]:
    """Walk CORE universe, evaluate each, return qualifying entries.

    Sorted by dd_from_52w_high ascending (deepest drop first).
    Names that don't qualify are EXCLUDED from the returned list.
    Use evaluate_ticker() directly for the pass/fail breakdown of a
    single ticker (e.g. /analisar appendix).

    Performance: pre-filters by DD + quality gate (CHEAP — uses signals
    already produced by scan_core_universe) BEFORE calling
    evaluate_ticker (which does 3 expensive yfinance fetches per
    candidate: sector lookup, sector_relative price hist pair, earnings
    calendar). Without pre-filter, a 230-ticker universe took ~20min;
    with pre-filter, ~1-2min because only the 10-30 candidates that
    pass the cheap gates incur the network cost.

    Catalyst proximity stays in the expensive bucket — there's no PIT
    proxy for it in signals — so the worst-case is still bound by the
    pre-filter size, not the full universe.
    """
    try:
        from core.core_radar import scan_core_universe
        opps = scan_core_universe(min_score=0.0)
    except Exception as e:
        log.warning(f"[watchlist] universe scan failed: {e}")
        return []

    if not opps:
        log.info("[watchlist] universe scan returned 0 opps")
        return []

    # ── Pre-filter: cheap signals already in opp dict ─────────────────
    # DD threshold + quality gate use fields scan_core_universe already
    # populated (dd_from_52w_high, revenue_growth, gross_margin,
    # fcf_yield_fund). No yfinance calls here — pure dict lookups.
    candidates = []
    n_dd_fail = 0
    n_quality_fail = 0
    for opp in opps:
        dd = opp.get("dd_from_52w_high")
        if dd is None or dd > min_dd_pct:
            n_dd_fail += 1
            continue
        ok, _reason = _quality_passes(opp)
        if not ok:
            n_quality_fail += 1
            continue
        candidates.append(opp)

    log.info(
        f"[watchlist] pre-filter: {len(candidates)} candidates from {len(opps)} universe "
        f"(DD-fail {n_dd_fail}, quality-fail {n_quality_fail}). "
        f"Now running expensive sector+catalyst fetches on candidates only."
    )

    # ── Expensive fetches only on pre-filtered candidates ─────────────
    entries = []
    n_catalyst_fail = 0
    for opp in candidates:
        try:
            entry = evaluate_ticker(
                opp["ticker"], opp,
                min_dd_pct=min_dd_pct,
                catalyst_max_days=catalyst_max_days,
                catalyst_min_days=catalyst_min_days,
            )
            if entry.qualifies:
                entries.append(entry)
            else:
                # Pre-filter already passed quality + DD, so failure here
                # is catalyst (the only remaining filter at this stage)
                n_catalyst_fail += 1
        except Exception as e:
            log.debug(f"[watchlist] {opp.get('ticker','?')}: {e}")
            continue

    # Sort by drawdown depth (most negative first = deepest drop first)
    entries.sort(key=lambda e: e.dd_from_52w_high if e.dd_from_52w_high is not None else 0.0)
    log.info(
        f"[watchlist] {len(entries)} names qualified "
        f"({n_catalyst_fail} dropped on catalyst proximity)"
    )

    # Persist run stats for callers to render diagnostics
    from datetime import datetime as _dt
    global _last_run_stats
    _last_run_stats = {
        "n_universe":       len(opps),
        "n_dd_fail":        n_dd_fail,
        "n_quality_fail":   n_quality_fail,
        "n_catalyst_fail":  n_catalyst_fail,
        "n_qualified":      len(entries),
        "fetched_at_iso":   _dt.now().isoformat(timespec="seconds"),
    }
    return entries


def format_watchlist_telegram(
    entries: list[WatchlistEntry],
    *,
    header: str = "📋 *Honest Watchlist*",
    include_disclaimer: bool = True,
) -> str:
    """Render the full watchlist as a Telegram message.

    NOT ranked by predicted return. Top of list = deepest drawdown,
    not "best alpha". Disclaimer reinforces this honestly.
    """
    if not entries:
        # Surface the per-gate breakdown so the user sees WHY it's empty
        stats = get_last_run_stats()
        lines = [
            header,
            "",
            "⚪ *No qualifiers — empty watchlist hoje.*",
            "",
            "_Critérios: quality + DD >= 25% + catalyst <= 30d._",
        ]
        if stats:
            lines.append("")
            lines.append("📊 *Breakdown* (porquê está vazia):")
            lines.append(f"  • Universo CORE scanado: *{stats.get('n_universe', '?')}* tickers")
            lines.append(f"  • Falharam DD ≥ 25%: *{stats.get('n_dd_fail', '?')}*")
            lines.append(f"  • Falharam quality gate: *{stats.get('n_quality_fail', '?')}*")
            lines.append(f"  • Passaram DD + quality mas falharam catalyst: *{stats.get('n_catalyst_fail', '?')}*")
            lines.append(f"  • Qualificaram (todos os 3): *{stats.get('n_qualified', 0)}*")
            n_dd_pass = stats.get("n_universe", 0) - stats.get("n_dd_fail", 0) - stats.get("n_quality_fail", 0)
            if n_dd_pass > 0 and stats.get("n_catalyst_fail", 0) == n_dd_pass:
                lines.append("")
                lines.append("_Diagnóstico: candidatos passam DD+quality mas nenhum tem earnings nos próximos 30d (ou nos últimos 10d). Catalyst filter é o gate restritivo hoje._")
            elif n_dd_pass == 0:
                lines.append("")
                lines.append("_Diagnóstico: poucos nomes com DD ≥ 25% — mercado provavelmente em regime caro/lateral._")
        return "\n".join(lines)

    lines = [
        header,
        "",
        f"_Quality compounders, DD >= 25% from 52w, catalyst <= 30d._",
        f"_NÃO ranked by predicted return — apenas listed, ordered by DD depth._",
        "",
    ]
    for e in entries:
        lines.append(f"`{e.summary_line}`")
    if include_disclaimer:
        lines.append("")
        lines.append("_Decide tu se há story para cada um._")
        lines.append("_`vs sector ≈ 0` = caiu com mercado, descarta. `vs sector` muito negativo = caiu sozinho, investiga._")
    return "\n".join(lines)
