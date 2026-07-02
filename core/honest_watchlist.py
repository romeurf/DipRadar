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
from dataclasses import dataclass, asdict, field
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


def _scan_extra_tickers(tickers: list[str]) -> list[dict]:
    """Run scan_core_universe-equivalent path for tickers OUTSIDE the CORE
    universe (DIP alerts or full universe). Reuses core_radar's
    _fetch_hist + _score_core_dip + _fetch_quality_metrics so the
    returned dicts have the same shape as scan_core_universe opps.

    Slow (one yfinance hist + info per ticker). The pre-filter in
    get_watchlist gates the EVEN MORE expensive evaluate_ticker fetches.
    """
    if not tickers:
        return []
    try:
        from core.core_radar import _fetch_hist, _score_core_dip
    except Exception as e:
        log.warning(f"[watchlist] core_radar imports failed: {e}")
        return []

    extras = []
    for t in tickers:
        try:
            hist = _fetch_hist(t)
            if hist is None or hist.empty:
                continue
            close = hist["Close"]
            current_price = float(close.iloc[-1])
            volume = hist["Volume"] if "Volume" in hist.columns else None
            score, signals = _score_core_dip(
                close, current_price, 0.0,
                ticker=t, volume=volume,
            )
            extras.append({
                "ticker": t,
                "score": score,
                "current_price": current_price,
                **signals,
            })
        except Exception as e:
            log.debug(f"[watchlist] _scan_extra {t}: {e}")
            continue
    return extras


def get_last_run_stats() -> dict:
    """Per-gate breakdown of the most recent get_watchlist() call.

    Used by callers (/cash, /scan_now) to render an honest diagnostic
    when the watchlist is empty — answers "why?" with concrete numbers
    instead of leaving the user guessing if it's the filter or a bug.

    Schema: {n_universe, n_dd_fail, n_quality_fail, n_catalyst_fail,
             n_qualified, fetched_at_iso}
    """
    return dict(_last_run_stats)


# ── Triage layer (CLEAN / BORDERLINE / EXCLUDED) ─────────────────────
#
# Automates the manual judgment the user applied to the first watchlist
# /cash proposal (2026-06-29): NVO at 30.3% got +€84 proposed despite
# being over the 30% cap; HOOD at vs_sector +34pp was listed despite
# outperforming its sector (not a dislocation); MKC score 100 hid a
# $44.8B merger; the whole list clustered in software-AI. The machine
# excludes only the clear-cut cases; borderlines are generous and left
# to the human, with reasons shown ("mais free nas borderlines para que
# possa ser um humano a tomar a decisão final").

# Event-risk keywords for headline scan. merger/acquisition family is
# NOT in dislocation_score._BAD_NEWS_KW (that set targets deterioration
# signals); "cut"/"lowered" overlap deliberately — a guidance cut is
# both bad news AND an event that changes the thesis.
_EVENT_RISK_KW = frozenset({
    "merger", "acquisition", "acquires", "takeover", "buyout",
    "spin-off", "spinoff", "divestiture", "cut", "lowered",
})

# Cluster threshold: candidate's sector already above this share of
# portfolio cost basis → BORDERLINE "adensa" flag.
_CLUSTER_PCT_THRESHOLD = 0.35


def triage_watchlist(
    entries: "list[WatchlistEntry]",
    portfolio_context: Optional[dict] = None,
    *,
    fetch_headlines_fn=None,
) -> "list[WatchlistEntry]":
    """Classify each qualifying entry into CLEAN / BORDERLINE / EXCLUDED.

    Mutates entries in place (sets triage_tier + triage_reasons) and
    returns the same list. Reasons accumulate — an entry can be
    BORDERLINE for near-cap AND event-risk AND cluster simultaneously,
    and the human sees all three.

    portfolio_context schema:
        {"position_pct":   {SYM: float},    # live position % of wealth, 0..1
         "sector_cost_pct": {sector: float}} # sector share of cost basis, 0..1

    EXCLUDED (machine-safe, clear-cut only):
      - position_pct >= 30%  — at/over hard cap, headroom €0 (NVO case)
      - sector_relative_60d >= +10pp — outperformed sector; definitionally
        not a dislocation (HOOD case, +34pp)
    BORDERLINE (human decides):
      - position_pct in [25%, 30%) — near cap
      - sector_relative_60d in [-5pp, +10pp) — "fell with market". In a
        market-wide selloff most names legitimately sit here and WILL
        flood borderline — correct behavior, not a bug: they aren't
        specific dislocations and the human sees them flagged as such.
      - event-risk keyword in recent headlines — reason carries the
        HEADLINE SNIPPET, not just the keyword (an MKC flag is only
        useful if it shows "$44.8B merger...", never just "merger")
      - candidate sector > 35% of portfolio cost basis — adensa check
    CLEAN: everything else (fell alone < -5pp strictly, no flags).

    With portfolio_context=None all entries stay CLEAN with an explicit
    "triage skipped" reason — legacy callers see no behavior change.
    Headline fetch is SKIPPED for names already EXCLUDED (saves 1-2s each).
    """
    if fetch_headlines_fn is None:
        from utils.news_headlines import fetch_headlines as fetch_headlines_fn

    if portfolio_context is None:
        for e in entries:
            e.triage_tier = "CLEAN"
            e.triage_reasons = ["triage skipped (sem contexto de portfólio)"]
        return entries

    position_pct    = portfolio_context.get("position_pct") or {}
    sector_cost_pct = portfolio_context.get("sector_cost_pct") or {}

    for e in entries:
        reasons: list[str] = []
        tier = "CLEAN"

        # ── EXCLUDED rules (checked first; skip expensive work after) ──
        pos_pct = position_pct.get(e.ticker.upper())
        if pos_pct is not None and pos_pct >= 0.30:
            tier = "EXCLUDED"
            reasons.append(
                f"posição em {pos_pct*100:.1f}% — no/acima do hard cap 30%, headroom €0"
            )

        sr = e.sector_relative_60d
        # sector_relative None does NOT vote: without sector resolution we
        # can't assert it's a dislocation NOR that it isn't — abstaining
        # is the only honest position (not merely "missing data").
        if sr is not None and sr >= 0.10:
            tier = "EXCLUDED"
            reasons.append(
                f"vs sector {sr*100:+.0f}pp — OUTPERFORMOU o sector; não é dislocation"
            )

        if tier == "EXCLUDED":
            e.triage_tier = tier
            e.triage_reasons = reasons
            continue  # skip headline fetch — already out

        # ── BORDERLINE rules (accumulate) ──────────────────────────────
        if pos_pct is not None and 0.25 <= pos_pct < 0.30:
            tier = "BORDERLINE"
            reasons.append(
                f"perto do cap: posição em {pos_pct*100:.1f}% (soft cap 25%)"
            )

        if sr is not None and -0.05 <= sr < 0.10:
            tier = "BORDERLINE"
            reasons.append(
                f"vs sector {sr*100:+.0f}pp — caiu COM o mercado, não é dislocation específica"
            )

        # Event risk in recent headlines — reason must carry the snippet
        try:
            headlines = fetch_headlines_fn(e.ticker, max_results=8)
        except Exception as ex:
            log.debug(f"[triage] headlines {e.ticker}: {ex}")
            headlines = []
        for h in headlines or []:
            title = getattr(h, "title", "") or ""
            title_lower = title.lower()
            matched = next((kw for kw in _EVENT_RISK_KW if kw in title_lower), None)
            if matched:
                tier = "BORDERLINE"
                snippet = title[:90] + ("…" if len(title) > 90 else "")
                reasons.append(f'event risk: "{snippet}" (matched: {matched})')
                break  # one event flag is enough

        # Cluster concentration — the "adensa vs diversifica" check
        cand_sector = e.sector or ""
        if cand_sector:
            cluster_pct = sector_cost_pct.get(cand_sector)
            if cluster_pct is not None and cluster_pct > _CLUSTER_PCT_THRESHOLD:
                tier = "BORDERLINE"
                reasons.append(
                    f"adensa cluster: {cand_sector} já é {cluster_pct*100:.0f}% do portfólio"
                )

        e.triage_tier = tier
        e.triage_reasons = reasons
    return entries


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
    # Triage (set by triage_watchlist; defaults keep legacy callers intact)
    triage_tier:          str = "CLEAN"     # CLEAN | BORDERLINE | EXCLUDED
    triage_reasons:       list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _quality_passes(signals: dict, thresholds: Optional[dict] = None) -> tuple[bool, str]:
    """RIGOROUS quality gate — DATA-DRIVEN thresholds, partial data OK.

    Per user F.5 followup, two principles:
      1. "Cortar com data, não com acho que" — thresholds derive from
         the universe's own distribution (median), not my judgment.
      2. "Requerer todos os campos pode ser limitador, sem aceitar 0
         nem pedir tudo" — partial data is OK (at least 2 of 3 signals
         present), but missing data does NOT count as 0 / pass; it
         simply doesn't vote.

    Rules:
      - At least 2 of {revenue_growth, gross_margin, roe} must be PRESENT
      - Each PRESENT signal must pass its universe-derived threshold
      - Missing signals are NEUTRAL (don't vote for or against)
      - FCF: if present and negative + not investment-phase → fail
              (FCF is graded so don't count for the 2-of-3 sufficiency)

    `thresholds` dict (from compute_quality_thresholds) carries
    universe-derived medians:
      {"roe": 0.XX, "gross_margin": 0.XX, "revenue_growth": 0.XX}
    A name qualifies when its PRESENT signals all sit above the
    CORE universe median for that metric.

    When thresholds is None (e.g. /analisar single-ticker), falls back
    to data-sufficiency only (2 of 3 fields present).
    """
    rg  = signals.get("revenue_growth")
    gm  = signals.get("gross_margin")
    roe = signals.get("roe")

    present = sum(1 for v in (rg, gm, roe) if v is not None)
    if present < 2:
        missing = []
        if rg is None:  missing.append("revenue_growth")
        if gm is None:  missing.append("gross_margin")
        if roe is None: missing.append("roe")
        return False, f"only {present}/3 quality signals present (missing: {', '.join(missing)})"

    # Data-driven threshold gates: each PRESENT signal must pass universe
    # median. Missing signals are neutral (skipped, neither vote).
    if thresholds:
        roe_floor = thresholds.get("roe")
        gm_floor  = thresholds.get("gross_margin")
        rg_floor  = thresholds.get("revenue_growth")
        if roe is not None and roe_floor is not None and roe < roe_floor:
            return False, f"roe {roe*100:.0f}% < universe median {roe_floor*100:.0f}%"
        if gm is not None and gm_floor is not None and gm < gm_floor:
            return False, f"gross_margin {gm*100:.0f}% < universe median {gm_floor*100:.0f}%"
        if rg is not None and rg_floor is not None and rg < rg_floor:
            return False, f"revenue_growth {rg*100:+.0f}% < universe median {rg_floor*100:+.0f}%"

    # FCF graded check (allows investment-phase compounders like AMZN-style).
    # Doesn't count toward the 2-of-3 sufficiency — purely a veto check
    # for clearly broken cash flow without investment-phase justification.
    fcf = signals.get("fcf_yield_fund") or signals.get("fcf_yield")
    if fcf is not None and fcf < 0 and not signals.get("is_investment_phase"):
        return False, f"fcf_yield {fcf*100:+.1f}% < 0 (not investment phase)"

    return True, ""


def compute_quality_thresholds(opps: list[dict]) -> dict:
    """Derive quality thresholds from the universe distribution.

    Computes the median of ROE, gross_margin, revenue_growth across
    all opps with data present. Returns dict with floors per metric.

    Used by get_watchlist to set _quality_passes() thresholds — names
    must be ABOVE the CORE universe median on all three signals.
    Curated CORE is all quality compounders, so above-median there
    is a high bar.
    """
    import numpy as np
    def _median_of(key: str) -> Optional[float]:
        vals = [o.get(key) for o in opps if o.get(key) is not None]
        if not vals:
            return None
        return float(np.median(vals))

    return {
        "roe":            _median_of("roe"),
        "gross_margin":   _median_of("gross_margin"),
        "revenue_growth": _median_of("revenue_growth"),
    }


def _fetch_sector_for(ticker: str) -> str:
    """Sector lookup via the shared persistent cache (utils/sector_cache).

    The cache guarantees candidates AND portfolio holdings resolve
    sectors through the same source, so triage cluster math compares
    like-for-like strings. 90-day TTL — sectors virtually never change.
    """
    try:
        from utils.sector_cache import get_sector
        return get_sector(ticker)
    except Exception as e:
        log.debug(f"[watchlist] sector fetch {ticker}: {e}")
        return ""


def _fetch_catalyst_days(ticker: str) -> Optional[int]:
    """Days to next earnings. None if not available within 120d.

    120d window (vs the legacy 45d default) so the display can honestly
    say "catalyst far (70d)" instead of pretending the date is unknown —
    catalyst is a SIGNAL not a filter, so far dates are information too.
    """
    try:
        from data.market_client import get_earnings_days
        return get_earnings_days(ticker, max_days=120)
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

    # Catalyst as SIGNAL not FILTER: badge 🎯 when within 30d window
    # (recent post-earnings or upcoming), "—" otherwise. Never excludes.
    cd = entry.days_to_catalyst
    if cd is None:
        cat_str = "catalyst —"
    elif cd < -10:
        cat_str = f"catalyst {-cd}d ago"
    elif cd < 0:
        cat_str = f"🎯 catalyst {-cd}d ago"  # post-earnings revaluation window
    elif cd <= 30:
        cat_str = f"🎯 catalyst {cd}d"        # near catalyst — bonus
    elif cd <= 60:
        cat_str = f"catalyst {cd}d"            # visible but no badge
    else:
        cat_str = f"catalyst far ({cd}d)"      # honest: too far to matter

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
    HARD criteria passed (quality gate + DD threshold). Catalyst is
    a SIGNAL not a filter: computed and shown for prioritization but
    does NOT exclude. A great quality dislocation 60d before earnings
    shouldn't be missed just because the catalyst is far away.

    Use this for both "should I add to watchlist?" and "show me why
    this candidate did/didn't qualify" (e.g. /analisar appendix).
    """
    dd = signals.get("dd_from_52w_high")
    dd_ok = (dd is not None) and (dd <= min_dd_pct)

    quality_passes, quality_reason = _quality_passes(signals)

    sector = signals.get("sector") or _fetch_sector_for(ticker)
    sector_rel = _fetch_sector_relative_60d(ticker, sector)

    catalyst_days = _fetch_catalyst_days(ticker)
    catalyst_near = (
        catalyst_days is not None
        and catalyst_min_days <= catalyst_days <= catalyst_max_days
    )

    # HARD criteria: quality + DD. Catalyst is a bonus signal, NOT a
    # filter — per user F.5 review followup: "catalyst should be an
    # advantage, not an impediment; a genuinely good dislocation
    # shouldn't be missed just because it's far from a catalyst."
    qualifies = bool(quality_passes and dd_ok)

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
    source: str = "default",
    dip_lookback_days: int = 14,
    portfolio_context: Optional[dict] = None,
) -> list[WatchlistEntry]:
    """Walk the chosen universe, evaluate each, return qualifying entries.

    Sources (per F.5 followup: "why only CORE? what about the 1000+ universe?"):
      "default" — CORE (230 curated) ∪ DIP scanner alerts last 14d.
                  Fast: only re-scans names the DIP scanner already flagged
                  from the broader universe, adds them to CORE. Typical
                  runtime ~1-3 min.
      "full"    — Full 1518-ticker universe from us_universe_static.json.
                  Slow (~30-60min cold; subsequent fast via /data caches).
                  Use when you suspect interesting names outside CORE.

    Sorted: names with near catalyst (~30d) first as tiebreaker, then by
    drawdown depth (deepest first). Catalyst is a SIGNAL not a FILTER
    (does not exclude — names with no near catalyst still qualify).

    Pre-filter: cheap signals (DD + quality from signals dict) gate the
    expensive evaluate_ticker fetches (sector, sector_relative, calendar).
    """
    # Source: CORE base, optionally extended
    try:
        from core.core_radar import scan_core_universe
        opps = scan_core_universe(min_score=0.0)
    except Exception as e:
        log.warning(f"[watchlist] CORE scan failed: {e}")
        opps = []

    if source == "full":
        # Full universe: include all 1518 from us_universe_static.json
        try:
            from core.universe import get_full_universe
            full = get_full_universe()
            core_set = {o["ticker"] for o in opps}
            extra = [t for t in full if t not in core_set]
            log.info(f"[watchlist] source=full: CORE={len(opps)} + {len(extra)} broader = {len(opps) + len(extra)} to scan")
            # Scan the extras through scan_core_universe equivalent path
            opps.extend(_scan_extra_tickers(extra))
        except Exception as e:
            log.warning(f"[watchlist] full universe expansion failed: {e}")
    elif source == "default":
        # Default: union CORE + recent DIP alert_db tickers (cheap — names
        # the DIP scanner already flagged from broader universe over 14d).
        try:
            from scanners.alert_db import get_recent_alert_tickers
            dip_tickers = get_recent_alert_tickers(lookback_days=dip_lookback_days)
            core_set = {o["ticker"] for o in opps}
            extra = [t for t in dip_tickers if t not in core_set]
            if extra:
                log.info(f"[watchlist] source=default: CORE={len(opps)} + {len(extra)} from DIP alerts ({dip_lookback_days}d)")
                opps.extend(_scan_extra_tickers(extra))
        except Exception as e:
            log.debug(f"[watchlist] DIP alert_db union skipped: {e}")
    # source == "core" → no expansion, just CORE

    if not opps:
        log.info("[watchlist] universe scan returned 0 opps")
        return []

    # Derive quality thresholds from CORE universe distribution.
    # Per user F.5 followup: cut by data, not "acho que". Thresholds
    # are the universe median (P50) of ROE / gross_margin / revenue_growth
    # across the CURATED CORE baseline — names then need to be ABOVE
    # the median of their peer group.
    quality_thresholds = compute_quality_thresholds(opps)
    log.info(
        f"[watchlist] data-driven quality thresholds (universe median): "
        f"ROE >= {(quality_thresholds.get('roe') or 0)*100:.0f}%, "
        f"GM >= {(quality_thresholds.get('gross_margin') or 0)*100:.0f}%, "
        f"rev_growth >= {(quality_thresholds.get('revenue_growth') or 0)*100:+.0f}%"
    )

    # ── Pre-filter: cheap signals already in opp dict ─────────────────
    # DD threshold + quality gate use fields scan_core_universe already
    # populated. Quality now data-driven via universe-median thresholds.
    candidates = []
    n_dd_fail = 0
    n_quality_fail = 0
    for opp in opps:
        dd = opp.get("dd_from_52w_high")
        if dd is None or dd > min_dd_pct:
            n_dd_fail += 1
            continue
        ok, _reason = _quality_passes(opp, thresholds=quality_thresholds)
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

    # Sort: (1) names with near catalyst (~30d window) come first as
    # tiebreaker — catalyst is now a SIGNAL not FILTER (per F.5
    # followup) so it sorts but doesn't exclude; (2) within each group,
    # deepest drawdown first.
    def _sort_key(e: WatchlistEntry):
        cd = e.days_to_catalyst
        has_near_catalyst = cd is not None and -10 <= cd <= 30
        dd = e.dd_from_52w_high if e.dd_from_52w_high is not None else 0.0
        # Primary: near-catalyst first (False sorts before True; invert)
        return (not has_near_catalyst, dd)
    entries.sort(key=_sort_key)

    n_with_catalyst = sum(
        1 for e in entries
        if e.days_to_catalyst is not None and -10 <= e.days_to_catalyst <= 30
    )
    log.info(
        f"[watchlist] {len(entries)} names qualified "
        f"({n_with_catalyst} with near catalyst, {len(entries) - n_with_catalyst} without)"
    )

    # Triage layer: classify qualifiers into CLEAN / BORDERLINE / EXCLUDED
    # against the caller-supplied portfolio context. None → all CLEAN with
    # explicit "triage skipped" reason (legacy callers unchanged).
    entries = triage_watchlist(entries, portfolio_context)
    n_clean      = sum(1 for e in entries if e.triage_tier == "CLEAN")
    n_borderline = sum(1 for e in entries if e.triage_tier == "BORDERLINE")
    n_excluded   = sum(1 for e in entries if e.triage_tier == "EXCLUDED")
    if portfolio_context is not None:
        log.info(
            f"[watchlist] triage: {n_clean} CLEAN, {n_borderline} BORDERLINE, "
            f"{n_excluded} EXCLUDED"
        )

    # Persist run stats + the data-driven thresholds for callers to
    # render in /cash and /scan_now output (transparency: user sees
    # WHICH median ROE/GM/rev_growth was used to cut).
    from datetime import datetime as _dt
    global _last_run_stats
    _last_run_stats = {
        "n_universe":           len(opps),
        "n_dd_fail":            n_dd_fail,
        "n_quality_fail":       n_quality_fail,
        "n_qualified":          len(entries),
        "n_with_near_catalyst": n_with_catalyst,
        "n_clean":              n_clean,
        "n_borderline":         n_borderline,
        "n_excluded":           n_excluded,
        "quality_thresholds":   quality_thresholds,
        "fetched_at_iso":       _dt.now().isoformat(timespec="seconds"),
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
        stats = get_last_run_stats()
        lines = [
            header,
            "",
            "⚪ *No qualifiers — empty watchlist hoje.*",
            "",
            "_Critérios hard: quality gate + DD ≥ 25%. Catalyst é sinal, não filtro._",
        ]
        if stats:
            lines.append("")
            lines.append("📊 *Breakdown* (porquê está vazia):")
            lines.append(f"  • Universo scanado: *{stats.get('n_universe', '?')}* tickers")
            lines.append(f"  • Falharam DD ≥ 25%: *{stats.get('n_dd_fail', '?')}*")
            lines.append(f"  • Falharam quality gate: *{stats.get('n_quality_fail', '?')}*")
            lines.append(f"  • Qualificaram (quality + DD): *{stats.get('n_qualified', 0)}*")
            n_dd_pass = stats.get("n_universe", 0) - stats.get("n_dd_fail", 0) - stats.get("n_quality_fail", 0)
            if n_dd_pass == 0:
                lines.append("")
                lines.append("_Diagnóstico: nenhum nome com DD ≥ 25% — mercado provavelmente em regime caro/lateral. Considera correr `/scan_now --full` para escanar universo broader (1500+ tickers)._")
            else:
                lines.append("")
                lines.append("_Inesperado: candidatos passaram DD+quality mas 0 entraram. Pode haver erro silencioso — verifica logs._")
        return "\n".join(lines)

    lines = [
        header,
        "",
        f"_Quality compounders, DD >= 25% from 52w. Catalyst é sinal, não filtro._",
        f"_NÃO ranked by predicted return — apenas listed, ordered by DD depth._",
        "",
    ]

    # Group by triage tier when triage ran (any entry with a tier other
    # than the untriaged default set). Untriaged lists render flat.
    _triaged = any(
        e.triage_reasons and e.triage_reasons != ["triage skipped (sem contexto de portfólio)"]
        for e in entries
    ) or any(e.triage_tier != "CLEAN" for e in entries)

    if _triaged:
        _clean      = [e for e in entries if e.triage_tier == "CLEAN"]
        _borderline = [e for e in entries if e.triage_tier == "BORDERLINE"]
        _excluded   = [e for e in entries if e.triage_tier == "EXCLUDED"]

        if _clean:
            lines.append(f"*✅ CLEAN ({len(_clean)}):*")
            for e in _clean:
                lines.append(f"`{e.summary_line}`")
            lines.append("")
        if _borderline:
            lines.append(f"*🟡 BORDERLINE — decide tu ({len(_borderline)}):*")
            for e in _borderline:
                lines.append(f"`{e.summary_line}`")
                for r in e.triage_reasons:
                    lines.append(f"    · _{r}_")
            lines.append("")
        if _excluded:
            lines.append(f"*⛔ EXCLUÍDOS ({len(_excluded)}):*")
            for e in _excluded:
                _r0 = e.triage_reasons[0] if e.triage_reasons else "?"
                lines.append(f"  ⛔ `{e.ticker}` — _{_r0}_")
            lines.append("")
    else:
        for e in entries:
            lines.append(f"`{e.summary_line}`")

    if include_disclaimer:
        lines.append("")
        lines.append("_Decide tu se há story para cada um._")
        lines.append("_`vs sector ≈ 0` = caiu com mercado, descarta. `vs sector` muito negativo = caiu sozinho, investiga._")
    return "\n".join(lines)
