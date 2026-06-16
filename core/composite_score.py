"""core/composite_score.py — Unified screener z-score for CORE + DIP picks.

Replaces ML-derived alpha predictions with a transparent heuristic
composite that ranks candidates against the CORE universe distribution.
No alpha promises — just "how compelling is this candidate vs the
universe right now" expressed as a z-score across three signals:

    composite_z = quality_z + dd_depth_z + valuation_z

Horizon-aware:
  - CORE horizon → drawdown from 52w high, valuation vs MA200w
  - DIP  horizon → return_20d (tactical drop), same valuation anchor

The quality and valuation z-scores are shared across horizons; only the
drawdown signal differs (52w for CORE compounders, 20d for tactical DIP
rotations).

Universe stats cached daily to /data/universe_stats.json. Walks the CORE
universe (~230 quality compounders) once per day, computes mean/std of
each component for z-scoring. First call after midnight rebuilds; rest
of the day reuses the cache.

Consumers:
  - /cash unified queue (CORE + DIP into one ranked deploy queue)
  - /scan_now (manual scan + qualifiers list)
  - /analisar appendix (single-ticker allocation context)
  - daily watchdog cron (proactive alert when held cash has new picks)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger("composite")

_DATA_DIR = (
    Path("/data") if Path("/data").exists()
    else Path(__file__).resolve().parent.parent / "data"
)
_STATS_PATH = _DATA_DIR / "universe_stats.json"
_STATS_TTL_HOURS = 24


@dataclass
class CompositeBreakdown:
    ticker: str
    horizon: str
    composite_z: float
    quality_z: float
    dd_z: float
    valuation_z: float
    quality_raw: float
    dd_raw: float
    valuation_raw: float

    def to_dict(self) -> dict:
        return asdict(self)


def _quality_composite(signals: dict) -> float:
    """Single quality scalar from ROE + gross_margin + revenue_growth + fcf_yield.

    Each metric clipped to a sensible range and equally weighted. Returns
    mean of normalized components in [0, 1]. Neutral 0.5 when no quality
    data available (e.g. DIP candidate outside CORE universe).
    """
    components: list[float] = []

    roe = signals.get("roe")
    if roe is not None:
        components.append(float(np.clip(float(roe) / 0.50, 0.0, 1.0)))

    gm = signals.get("gross_margin")
    if gm is not None:
        components.append(float(np.clip(float(gm) / 0.90, 0.0, 1.0)))

    rg = signals.get("revenue_growth")
    if rg is not None:
        components.append(float(np.clip((float(rg) + 0.20) / 0.70, 0.0, 1.0)))

    fcf = signals.get("fcf_yield_fund") or signals.get("fcf_yield")
    if fcf is not None:
        components.append(float(np.clip((float(fcf) + 0.10) / 0.25, 0.0, 1.0)))

    if not components:
        return 0.5
    return float(np.mean(components))


def _dd_depth(signals: dict, horizon: str) -> float:
    """Drawdown depth — positive number, larger = deeper drop.

    CORE: |min(dd_from_52w_high, 0)| → 0.40 means −40% from 52w peak.
    DIP:  |min(return_20d, 0)|       → 0.15 means −15% in last 20 days.
    """
    if horizon == "CORE":
        dd = signals.get("dd_from_52w_high")
    else:
        dd = signals.get("return_20d")
    if dd is None:
        return 0.0
    return abs(min(float(dd), 0.0))


def _valuation_proxy(signals: dict) -> float:
    """Valuation z-input via price_vs_ma200w.

    Negative price_vs_ma200w means trading below the long-term trend
    (cheap). We flip the sign so deeper-below-trend = bigger positive
    valuation_raw. Capped at +0.5 (50% below MA200w) to avoid outliers
    dominating the distribution.
    """
    pv = signals.get("price_vs_ma200w")
    if pv is None:
        return 0.0
    return float(np.clip(-float(pv), -0.2, 0.5))


def build_universe_stats(force_refresh: bool = False) -> dict:
    """Walk CORE universe and return mean/std per component.

    Cached daily. Cache hit when /data/universe_stats.json is < 24h old.
    Force refresh by passing force_refresh=True (used by /admin_recompute_stats
    if added later, or after universe edits).
    """
    if not force_refresh and _STATS_PATH.exists():
        try:
            cached = json.loads(_STATS_PATH.read_text())
            computed_at = cached.get("computed_at")
            if computed_at:
                age_h = (
                    datetime.now() - datetime.fromisoformat(computed_at)
                ).total_seconds() / 3600
                if age_h < _STATS_TTL_HOURS:
                    log.info(
                        f"[composite] universe_stats cached "
                        f"({age_h:.1f}h old, n={cached.get('n_universe')})"
                    )
                    return cached
        except Exception as e:
            log.warning(f"[composite] cached stats unreadable: {e}")

    log.info("[composite] rebuilding universe_stats (2-5 min)")
    # Importing inside the function avoids a circular import at module load.
    from core.core_radar import scan_core_universe
    opps = scan_core_universe(min_score=0.0)
    if not opps:
        log.warning("[composite] universe scan returned 0 opps — empty stats")
        return {}

    quality_raws  = [_quality_composite(o)         for o in opps]
    dd_core_raws  = [_dd_depth(o, "CORE")          for o in opps]
    dd_dip_raws   = [_dd_depth(o, "DIP")           for o in opps]
    valuation_raws = [_valuation_proxy(o)           for o in opps]

    def _stats(arr: list[float]) -> dict:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if len(a) == 0:
            return {"mean": 0.0, "std": 0.01, "n": 0}
        return {
            "mean": float(a.mean()),
            "std":  float(a.std()) or 0.01,
            "n":    int(len(a)),
            "p25":  float(np.quantile(a, 0.25)),
            "p75":  float(np.quantile(a, 0.75)),
        }

    stats = {
        "computed_at": datetime.now().isoformat(timespec="seconds"),
        "n_universe":  len(opps),
        "quality":     _stats(quality_raws),
        "dd_core":     _stats(dd_core_raws),
        "dd_dip":      _stats(dd_dip_raws),
        "valuation":   _stats(valuation_raws),
    }

    try:
        _STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATS_PATH.write_text(json.dumps(stats, indent=2))
        log.info(f"[composite] persisted universe_stats to {_STATS_PATH}")
    except Exception as e:
        log.warning(f"[composite] persist failed: {e}")
    return stats


def composite_from_signals(
    ticker: str,
    signals: dict,
    *,
    horizon: str = "CORE",
    stats: dict | None = None,
) -> CompositeBreakdown:
    """Compute composite z-score from a pre-fetched signals dict.

    `signals` must contain the fields produced by
    core_radar.scan_core_universe (roe, gross_margin, revenue_growth,
    fcf_yield_fund, dd_from_52w_high, return_20d, price_vs_ma200w).
    Missing fields degrade gracefully toward 0-contribution.
    """
    if stats is None:
        stats = build_universe_stats()

    horizon = "DIP" if str(horizon).upper() == "DIP" else "CORE"

    q_raw   = _quality_composite(signals)
    dd_raw  = _dd_depth(signals, horizon)
    val_raw = _valuation_proxy(signals)

    q_stats   = stats.get("quality")  or {"mean": 0.5,  "std": 0.15}
    dd_key    = "dd_core" if horizon == "CORE" else "dd_dip"
    dd_stats  = stats.get(dd_key)     or {"mean": 0.15, "std": 0.10}
    val_stats = stats.get("valuation") or {"mean": 0.0,  "std": 0.10}

    q_z   = (q_raw   - q_stats["mean"])   / max(q_stats["std"],   0.01)
    dd_z  = (dd_raw  - dd_stats["mean"])  / max(dd_stats["std"],  0.01)
    val_z = (val_raw - val_stats["mean"]) / max(val_stats["std"], 0.01)

    composite_z = q_z + dd_z + val_z

    return CompositeBreakdown(
        ticker=ticker,
        horizon=horizon,
        composite_z=composite_z,
        quality_z=q_z,
        dd_z=dd_z,
        valuation_z=val_z,
        quality_raw=q_raw,
        dd_raw=dd_raw,
        valuation_raw=val_raw,
    )


def composite_for_ticker(
    ticker: str,
    *,
    horizon: str = "CORE",
    stats: dict | None = None,
) -> CompositeBreakdown | None:
    """Convenience: fetch signals + compute composite z for a single ticker.

    Used by /analisar appendix and /check (if added later). Returns None
    when the ticker can't be fetched (delisted, network error, etc.).
    """
    try:
        from core.core_radar import (
            _fetch_hist, _fetch_quality_metrics, _score_core_dip,
        )
        hist = _fetch_hist(ticker)
        if hist is None or hist.empty:
            return None
        close = hist["Close"]
        volume = hist["Volume"] if "Volume" in hist.columns else None
        current_price = float(close.iloc[-1])
        _score, signals = _score_core_dip(
            close, current_price, 0.0, ticker=ticker, volume=volume,
        )
        return composite_from_signals(
            ticker, signals, horizon=horizon, stats=stats,
        )
    except Exception as e:
        log.warning(f"[composite] composite_for_ticker({ticker}): {e}")
        return None


def softmax_sizing(
    candidates: list[tuple[str, float]],
    *,
    budget_eur: float,
    temperature: float = 2.0,
    threshold_z: float = 0.3,
    min_alloc_eur: float = 100.0,
    concentration_cap_pct: float = 0.30,
    portfolio_value_eur: float = 0.0,
    current_positions_eur: dict[str, float] | None = None,
) -> tuple[dict[str, float], list[tuple[str, str]]]:
    """Convert (ticker, composite_z) pairs into € allocations.

    Pipeline:
      1. Threshold gate — drop any z < threshold_z
      2. Softmax weights with temperature T (higher T → more concentrated)
      3. Cap each allocation at concentration headroom (portfolio_value × cap_pct − current)
      4. Redistribute capped overflow proportionally among non-capped, respecting their caps
      5. Drop allocations below min_alloc_eur (T212 fee floor)

    Returns:
      allocations: {ticker: eur} for qualifying picks
      skipped:     [(ticker, reason), ...] for everything dropped + why
    """
    current_positions_eur = current_positions_eur or {}

    qualified: list[tuple[str, float]] = []
    skipped: list[tuple[str, str]] = []

    for ticker, z in candidates:
        if z < threshold_z:
            skipped.append((ticker, f"below_threshold z={z:+.2f}"))
            continue
        qualified.append((ticker, float(z)))

    if not qualified:
        return {}, skipped

    headrooms: dict[str, float] = {}
    cap_active = portfolio_value_eur > 0
    for ticker, _ in qualified:
        if cap_active:
            cap_eur = portfolio_value_eur * concentration_cap_pct
            current = float(current_positions_eur.get(ticker, 0.0))
            headrooms[ticker] = max(0.0, cap_eur - current)
        else:
            headrooms[ticker] = float("inf")

    z_arr = np.array([z for _, z in qualified], dtype=float)
    exp_arr = np.exp(z_arr * float(temperature))
    weights = exp_arr / exp_arr.sum()

    allocs: dict[str, float] = {}
    capped: set[str] = set()
    overflow = 0.0

    for i, (ticker, _) in enumerate(qualified):
        raw_alloc = float(weights[i] * budget_eur)
        head = headrooms[ticker]
        if raw_alloc > head:
            overflow += raw_alloc - head
            allocs[ticker] = head
            capped.add(ticker)
        else:
            allocs[ticker] = raw_alloc

    if overflow > 0.01:
        non_capped = [(t, z) for t, z in qualified if t not in capped]
        if non_capped:
            nc_z = np.array([z for _, z in non_capped], dtype=float)
            nc_exp = np.exp(nc_z * float(temperature))
            nc_weights = nc_exp / nc_exp.sum()
            for i, (t, _) in enumerate(non_capped):
                addition = float(nc_weights[i] * overflow)
                remaining_head = headrooms[t] - allocs[t]
                if addition > remaining_head:
                    addition = remaining_head
                allocs[t] = allocs[t] + addition

    final: dict[str, float] = {}
    for ticker, eur in allocs.items():
        if eur < min_alloc_eur:
            skipped.append(
                (ticker, f"below_min_alloc €{eur:.0f} < €{min_alloc_eur:.0f}")
            )
            continue
        final[ticker] = round(eur, 0)

    return final, skipped
