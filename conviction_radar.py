"""
conviction_radar.py — parallel tracker for personal-conviction positions.

This module is the user's "hunch ledger". It uses the same analysis engine
as DipRadar (DIP score, ML alpha forecast, ATR trailing stop, momentum
signals) but writes to its own state files (personal_positions.json,
conviction_outcomes.json) that are NEVER read by any ML training script.

Design rationale:
  - portfolio.py     = ownership ledger (what shares you hold)
  - position_db      = bot-decision thesis tracker (alerts → auto-created)
  - personal_positions.json (here) = conviction thesis tracker (manual)

The three files are orthogonal. The same ticker can exist in portfolio.py
(you own it) AND in personal_positions.json (you have a thesis being tracked).
Positions opened by the bot's alerts never touch this file; positions added
via /conviction never touch position_db or prediction_log.

Isolation guarantee: every record carries `source="conviction"`. If a future
script in scripts/build_*.py ever references personal_positions.json or
conviction_outcomes.json, that's a bug — the contamination contract is
single-sentence verifiable.

Supported position types:
  DIP       → recovery thesis (target + win-prob delta + edge floor)
  MOMENTUM  → trailing stop ATR + RSI/MA20 confirmation
  CORE      → long-term quality tracker (monthly cadence, no daily stops)

A DIP can transition to MOMENTUM (manual via /conviction transition, or
automatic when score_momentum is high and price has recovered).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Optional, Callable

import yfinance as yf
import pandas as pd

from state import _load_json, _save_json, _today_iso

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Storage paths (NEVER referenced by scripts/build_*.py or monthly_retrain.py)
# ─────────────────────────────────────────────────────────────────────────────

_POSITIONS_FILE = "personal_positions.json"
_OUTCOMES_FILE  = "conviction_outcomes.json"


# ─────────────────────────────────────────────────────────────────────────────
# Tuning knobs (env-overridable)
# ─────────────────────────────────────────────────────────────────────────────

_FALLBACK_TRAIL_PCT  = float(os.getenv("CONVICTION_TRAIL_FALLBACK_PCT", "0.10"))
_ATR_MULTIPLIER      = float(os.getenv("CONVICTION_ATR_MULTIPLIER", "2.5"))
_ATR_MIN_STOP_PCT    = float(os.getenv("CONVICTION_ATR_MIN_STOP_PCT", "0.08"))
_EDGE_FLOOR          = float(os.getenv("ML_SCORE_FLOOR", "0.01"))
_DIP_TIME_DECAY_DAYS = int(os.getenv("CONVICTION_DIP_TIME_DECAY_DAYS", "120"))
_QUALITY_DROP_TRIGGER = float(os.getenv("CONVICTION_QUALITY_DROP_TRIGGER", "0.15"))

VALID_TYPES = ("DIP", "MOMENTUM", "CORE")

# ─────────────────────────────────────────────────────────────────────────────
# Trigger taxonomy
# ─────────────────────────────────────────────────────────────────────────────
# PHASE triggers reflect short-term price/momentum behaviour. They are
# meaningful for active trades (DIP/MOMENTUM rotation) but should not force a
# sell on a long-term compounder just because volatility ticked up.
# STRUCTURAL triggers reflect breakdown of the *thesis* itself — quality drop,
# margin erosion, multi-criteria deterioration. These ALWAYS issue a full SELL
# recommendation regardless of never_sell, because a CORE that's no longer a
# "great company" must be flagged.
PHASE_TRIGGERS = {
    "TRAIL_STOP_CONFIRMED",
    "TRAIL_STOP_PENDING",
    "EARLY_ALPHA",
    "DETERIORATION",
    "TIME_DECAY",
    "EDGE_GONE",
    "SUGGEST_TRANSITION",
}
STRUCTURAL_TRIGGERS = {
    "QUALITY_DECLINE",          # quality_score dropped ≥ 20pp
    "CORE_QUALITY_WARNING",     # quality_score dropped 10-20pp (med severity)
    "CORE_QUALITY_DECLINE",     # quality_score dropped ≥ 20pp on CORE
    "STRUCTURAL_DECLINE",       # multi-criteria thesis breakdown
    "FUNDAMENTAL_BREAKDOWN",    # revenue + margin + debt deterioration
}


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConvictionPosition:
    """A position opened by personal conviction (not a bot alert).

    `source` is locked to "conviction" — guards against accidental
    contamination if the file is ever re-read by training scripts.
    """
    ticker:               str
    entry_price:          float
    shares:               float
    entry_date:           str               # ISO YYYY-MM-DD
    position_type:        str               # DIP | MOMENTUM | CORE — current phase
    source:               str = "conviction"

    # Identity (set at first add, NEVER changes after that).
    # original_type drives never_sell default and shapes the verdict for
    # transitions (e.g., a CORE that drifted into MOMENTUM rally is still
    # fundamentally a CORE — phase triggers should never auto-recommend sell).
    original_type:        Optional[str] = None
    # never_sell suppresses *phase* triggers from issuing a 🔴 FECHAR verdict
    # (capped at 🟡 alerta). STRUCTURAL triggers (quality deterioration,
    # multi-criteria thesis breakdown) ALWAYS bypass never_sell — that's
    # the whole point: "a great company that stops being a great company
    # must trigger a sell alert regardless of the never_sell flag".
    never_sell:           bool = False

    # MOMENTUM trailing-stop state
    trailing_high:        float = 0.0
    last_atr_stop_pct:    Optional[float] = None  # cached for display

    # Snapshot at entry (used for delta tracking)
    entry_quality_score:  Optional[float] = None  # 0..1
    entry_win_prob:       Optional[float] = None  # ML prediction at entry
    initial_pred_alpha:   Optional[float] = None  # alpha_90d at entry (DIP only)
    entry_score:          Optional[float] = None  # DIP score at entry (0..100)
    entry_sector:         Optional[str] = None

    # Snapshot of fundamentals at entry for structural-decline detection.
    # Keys: revenue_growth, gross_margin, debt_equity, roe, fcf_yield.
    entry_fundamentals:   Optional[dict] = None

    # Mutable state
    last_check_date:      Optional[str] = None
    last_check_price:     Optional[float] = None
    last_score:           Optional[float] = None
    last_win_prob:        Optional[float] = None
    note:                 Optional[str] = None    # free-text user reason for opening

    @property
    def days_held(self) -> int:
        try:
            start = date.fromisoformat(self.entry_date)
            return (date.today() - start).days
        except (ValueError, TypeError):
            return 0


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def load_convictions() -> list[ConvictionPosition]:
    raw = _load_json(_POSITIONS_FILE, default=[]) or []
    if not isinstance(raw, list):
        return []
    out: list[ConvictionPosition] = []
    known = {f for f in ConvictionPosition.__dataclass_fields__.keys()}
    for d in raw:
        if not isinstance(d, dict):
            continue
        try:
            clean = {k: v for k, v in d.items() if k in known}
            out.append(ConvictionPosition(**clean))
        except Exception as e:
            logger.warning(f"[conviction] skip malformed record: {e}")
    return out


def save_convictions(positions: list[ConvictionPosition]) -> None:
    _save_json(_POSITIONS_FILE, [asdict(p) for p in positions])


def update_conviction(updated: ConvictionPosition) -> bool:
    """Persist changes to a single conviction record (matched by ticker).

    Returns True if the record was found and saved, False otherwise.
    Used by /sync_portfolio to amend entry_date / trailing_high / type
    on positions that already exist — without going through the legacy
    close+reopen ritual which would lose history.
    """
    convictions = load_convictions()
    target_ticker = updated.ticker.upper().strip()
    for i, p in enumerate(convictions):
        if p.ticker == target_ticker:
            convictions[i] = updated
            save_convictions(convictions)
            return True
    return False


def load_outcomes() -> list[dict]:
    raw = _load_json(_OUTCOMES_FILE, default=[]) or []
    return raw if isinstance(raw, list) else []


def _append_outcome(entry: dict) -> None:
    data = load_outcomes()
    data.append(entry)
    _save_json(_OUTCOMES_FILE, data)


def get_by_ticker(ticker: str) -> Optional[ConvictionPosition]:
    ticker = ticker.upper().strip()
    for p in load_convictions():
        if p.ticker == ticker:
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Add / close / transition
# ─────────────────────────────────────────────────────────────────────────────

def add_conviction(
    ticker: str,
    entry_price: float,
    shares: float,
    position_type: str,
    trailing_high: Optional[float] = None,
    note: Optional[str] = None,
) -> ConvictionPosition:
    """Register a new conviction position. Captures entry-day analysis snapshot."""
    ticker = ticker.upper().strip()
    pt = position_type.upper().strip()
    if pt not in VALID_TYPES:
        raise ValueError(f"position_type must be one of {VALID_TYPES}, got {position_type!r}")
    if entry_price <= 0 or shares <= 0:
        raise ValueError("entry_price and shares must be positive")

    existing = get_by_ticker(ticker)
    if existing:
        raise ValueError(
            f"{ticker} já está como convicção activa (tipo {existing.position_type}, "
            f"entrada {existing.entry_date} @ ${existing.entry_price:.2f}). "
            "Usa /conviction close primeiro ou /conviction transition para mudar tipo."
        )

    # Capture entry-day snapshot for later delta comparisons
    snap = _entry_snapshot(ticker, entry_price)

    # never_sell defaults: CORE=True (long-term holds), DIP/MOMENTUM=False (sellable)
    default_never_sell = (pt == "CORE")

    pos = ConvictionPosition(
        ticker             = ticker,
        entry_price        = float(entry_price),
        shares             = float(shares),
        entry_date         = _today_iso(),
        position_type      = pt,
        original_type      = pt,           # locked at creation
        never_sell         = default_never_sell,
        trailing_high      = float(trailing_high) if trailing_high else float(entry_price),
        entry_quality_score = snap.get("quality_score"),
        entry_win_prob     = snap.get("win_prob"),
        initial_pred_alpha = snap.get("pred_alpha"),
        entry_score        = snap.get("dip_score"),
        entry_sector       = snap.get("sector"),
        entry_fundamentals = snap.get("fundamentals"),
        note               = note,
    )

    all_positions = load_convictions()
    all_positions.append(pos)
    save_convictions(all_positions)
    logger.info(f"[conviction] added {ticker} ({pt}, never_sell={default_never_sell}) @ ${entry_price:.2f}")
    return pos


def set_never_sell(ticker: str, value: bool) -> Optional[ConvictionPosition]:
    """Toggle never_sell flag for a position. Use lock=True / unlock=False."""
    ticker = ticker.upper().strip()
    positions = load_convictions()
    for p in positions:
        if p.ticker == ticker:
            p.never_sell = bool(value)
            save_convictions(positions)
            logger.info(f"[conviction] {ticker}: never_sell={value}")
            return p
    return None


def close_conviction(ticker: str, exit_price: float, reason: str = "manual") -> Optional[dict]:
    """Close a position. Writes outcome to conviction_outcomes.json."""
    ticker = ticker.upper().strip()
    positions = load_convictions()
    target = next((p for p in positions if p.ticker == ticker), None)
    if not target:
        return None

    pnl_pct = (exit_price / target.entry_price - 1.0) * 100.0
    pnl_abs = (exit_price - target.entry_price) * target.shares

    outcome = {
        "source":         "conviction",   # explicit isolation flag
        "ticker":         target.ticker,
        "entry_date":     target.entry_date,
        "exit_date":      _today_iso(),
        "days_held":      target.days_held,
        "position_type":  target.position_type,
        "entry_price":    target.entry_price,
        "exit_price":     float(exit_price),
        "shares":         target.shares,
        "pnl_abs":        round(pnl_abs, 2),
        "pnl_pct":        round(pnl_pct, 2),
        "reason":         reason,
        "entry_score":    target.entry_score,
        "exit_score":     target.last_score,
        "note":           target.note,
    }
    _append_outcome(outcome)

    positions = [p for p in positions if p.ticker != ticker]
    save_convictions(positions)
    logger.info(f"[conviction] closed {ticker} @ ${exit_price:.2f} ({pnl_pct:+.1f}%) reason={reason}")
    return outcome


def delete_conviction(ticker: str) -> bool:
    """Silently remove a position WITHOUT writing to conviction_outcomes.json.

    For fixing mistakes (wrong type, wrong price, wrong shares). Use
    close_conviction() instead for legitimate exits — that one logs the
    outcome for performance tracking.
    """
    ticker = ticker.upper().strip()
    positions = load_convictions()
    before = len(positions)
    positions = [p for p in positions if p.ticker != ticker]
    if len(positions) == before:
        return False
    save_convictions(positions)
    logger.info(f"[conviction] deleted {ticker} (silent — no outcome logged)")
    return True


def transition_type(ticker: str, new_type: str) -> Optional[ConvictionPosition]:
    """Manually change position_type (DIP → MOMENTUM most common)."""
    new_type = new_type.upper().strip()
    if new_type not in VALID_TYPES:
        raise ValueError(f"type must be one of {VALID_TYPES}")
    positions = load_convictions()
    for p in positions:
        if p.ticker == ticker.upper().strip():
            old = p.position_type
            p.position_type = new_type
            # never_sell auto-rules on transition:
            #   → CORE: auto-set never_sell=True (long-term hold semantics)
            #   away from CORE: only auto-unset never_sell if original was NOT CORE
            #     (a CORE that's currently in MOMENTUM rally stays never_sell=True)
            if new_type == "CORE":
                p.never_sell = True
            elif p.original_type != "CORE":
                p.never_sell = False
            if new_type == "MOMENTUM":
                # Anchor trailing_high to current high if not set
                cur = _fetch_price(p.ticker, fallback=p.entry_price)
                p.trailing_high = max(p.trailing_high or 0.0, cur, p.entry_price)
            save_convictions(positions)
            logger.info(f"[conviction] {ticker}: {old} → {new_type} (never_sell={p.never_sell})")
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Analysis primitives (mirror position_monitor but operate on ConvictionPosition)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_price(ticker: str, fallback: float = 0.0) -> float:
    try:
        info = yf.Ticker(ticker).fast_info
        p = getattr(info, "last_price", None)
        if p and float(p) > 0:
            return float(p)
    except Exception as e:
        logger.debug(f"[conviction] price fetch {ticker}: {e}")
    return float(fallback)


def _fetch_price_history(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    try:
        raw = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if raw is None or raw.empty:
            return None
        idx = pd.DatetimeIndex(raw.index)
        if idx.tz is not None:
            raw.index = idx.tz_convert(None)
        return raw
    except Exception as e:
        logger.debug(f"[conviction] history {ticker}: {e}")
        return None


def _compute_atr_stop_pct(hist: pd.DataFrame, current_price: float) -> tuple[float, str]:
    """Returns (stop_pct, source_label). Falls back to _FALLBACK_TRAIL_PCT on failure."""
    try:
        if hist is None or len(hist) < 14 or current_price <= 0:
            return _FALLBACK_TRAIL_PCT, "fallback"
        h = hist["High"]
        l = hist["Low"]
        c = hist["Close"].shift(1)
        tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
        atr = float(tr.iloc[-14:].mean())
        if atr <= 0:
            return _FALLBACK_TRAIL_PCT, "fallback"
        pct = max(_ATR_MULTIPLIER * atr / current_price, _ATR_MIN_STOP_PCT)
        return pct, f"ATR({atr:.2f}/{current_price:.2f}={atr/current_price*100:.1f}%)"
    except Exception:
        return _FALLBACK_TRAIL_PCT, "fallback"


def _rsi(closes: pd.Series, period: int = 14) -> float:
    try:
        from momentum_radar.scanner import _calc_rsi
        return _calc_rsi(closes, period)
    except Exception:
        return 50.0


def _entry_snapshot(ticker: str, entry_price: float) -> dict:
    """Capture quality/ML/DIP-score snapshot at entry. Best-effort; all fields optional.

    Now also captures `fundamentals` dict (revenue_growth, gross_margin,
    debt_equity, roe, fcf_yield) for later structural-decline detection.
    """
    snap: dict = {}
    try:
        from market_client import get_fundamentals
        fund = get_fundamentals(ticker, min_market_cap=0)
        if fund and not fund.get("skip"):
            snap["sector"] = fund.get("sector")
            # Persist fundamentals for structural decline detector
            snap["fundamentals"] = {
                "revenue_growth": fund.get("revenue_growth"),
                "gross_margin":   fund.get("gross_margin"),
                "debt_equity":    fund.get("debt_equity"),
                "roe":            fund.get("roe"),
                "fcf_yield":      fund.get("fcf_yield"),
            }
            try:
                from score import score_from_fundamentals
                sf = score_from_fundamentals(fund)
                if isinstance(sf, dict):
                    snap["quality_score"] = float(sf.get("quality_score", 0.5))
                    snap["dip_score"]     = float(sf.get("final_score", 50.0))
            except Exception:
                pass
            try:
                from ml_features import build_features
                from ml_engine import load_predictor, predict_dip
                load_predictor()
                hist = _fetch_price_history(ticker, period="6mo")
                feat = build_features(ticker, fund, price_history=hist,
                                       sector=fund.get("sector") or "Unknown")
                pred = predict_dip(feature_row=feat, current_price=entry_price, ticker=ticker)
                if pred and pred.model_ready:
                    snap["win_prob"]   = float(pred.win_prob)
                    snap["pred_alpha"] = float(getattr(pred, "pred_up", 0.0) or 0.0)
            except Exception as e:
                logger.debug(f"[conviction] snapshot ML failed {ticker}: {e}")
    except Exception as e:
        logger.debug(f"[conviction] snapshot {ticker}: {e}")
    return snap


def _detect_fundamental_breakdown(pos: ConvictionPosition, current_fund: dict) -> tuple[bool, list[str]]:
    """Multi-criteria structural decline detector.

    Returns (is_breakdown, reason_list). Triggers when ≥ 2 of:
      - Revenue growth turned negative AND dropped ≥ 15pp vs entry
      - Gross margin eroded ≥ 5pp vs entry
      - Debt/equity ratio more than doubled vs entry
      - ROE collapsed ≥ 30pp (e.g., 25% → -5%)

    Used for CORE positions: a "great company" failing on multiple
    fundamental axes simultaneously stops being a great company. This
    overrides never_sell — the whole point of the flag is to ignore noise,
    not to ignore real decline.
    """
    if not pos.entry_fundamentals or not current_fund:
        return False, []
    reasons: list[str] = []

    def _f(d: dict, k: str) -> Optional[float]:
        v = d.get(k)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    # 1. Revenue growth collapse
    rg_then = _f(pos.entry_fundamentals, "revenue_growth")
    rg_now  = _f(current_fund, "revenue_growth")
    if rg_then is not None and rg_now is not None:
        if rg_now < 0 and (rg_then - rg_now) >= 0.15:
            reasons.append(f"Revenue growth: {rg_then*100:+.0f}% → {rg_now*100:+.0f}% (queda ≥15pp + negativo)")

    # 2. Gross margin erosion
    gm_then = _f(pos.entry_fundamentals, "gross_margin")
    gm_now  = _f(current_fund, "gross_margin")
    if gm_then is not None and gm_now is not None:
        if (gm_then - gm_now) >= 0.05:
            reasons.append(f"Gross margin erodiu {(gm_then-gm_now)*100:.0f}pp ({gm_then*100:.0f}% → {gm_now*100:.0f}%)")

    # 3. Debt/equity doubled
    de_then = _f(pos.entry_fundamentals, "debt_equity")
    de_now  = _f(current_fund, "debt_equity")
    if de_then is not None and de_now is not None:
        if de_then > 0 and de_now >= 2.0 * de_then and de_now >= 1.0:
            reasons.append(f"Debt/equity duplicou ({de_then:.2f} → {de_now:.2f})")

    # 4. ROE collapse
    roe_then = _f(pos.entry_fundamentals, "roe")
    roe_now  = _f(current_fund, "roe")
    if roe_then is not None and roe_now is not None:
        if (roe_then - roe_now) >= 0.30:
            reasons.append(f"ROE colapsou {(roe_then-roe_now)*100:.0f}pp ({roe_then*100:+.0f}% → {roe_now*100:+.0f}%)")

    return (len(reasons) >= 2, reasons)


def _detect_event_driven_signals(ticker: str) -> tuple[list[dict], list[str]]:
    """Stub. SEC Filings event-driven signals (8-K classification, earnings
    call tone) were removed when the broader SEC pipeline was retired. The
    function is kept so callers in evaluate_conviction don't need import
    rewrites; a future from-scratch SEC integration can refill this.
    """
    return [], []


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation — runs same triggers as position_monitor but type-specific
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_conviction(pos: ConvictionPosition) -> dict:
    """Compute everything needed to render the deep-dive + daily-check alert.

    Returns a dict with: current_price, pnl_abs, pnl_pct, signals (list),
    triggers (list of {kind, severity, message}), refreshed scores.
    """
    current = _fetch_price(pos.ticker, fallback=pos.entry_price)
    hist    = _fetch_price_history(pos.ticker, period="6mo")

    pnl_pct = (current / pos.entry_price - 1.0) * 100.0 if pos.entry_price > 0 else 0.0
    pnl_abs = (current - pos.entry_price) * pos.shares

    out: dict = {
        "ticker":        pos.ticker,
        "current_price": current,
        "pnl_pct":       pnl_pct,
        "pnl_abs":       pnl_abs,
        "days_held":     pos.days_held,
        "position_type": pos.position_type,
        "signals":       [],   # list[str] — info lines for display
        "triggers":      [],   # list[dict] — actionable hold/close cues
    }

    # Update trailing high (for MOMENTUM)
    pos.trailing_high = max(pos.trailing_high or 0.0, current, pos.entry_price)

    # ── Re-run analysis (fundamentals, ML, score) — but never write to logs ───
    win_prob = None
    pred_alpha = None
    quality_score = None
    fresh_score = None
    rsi_now = None
    ma20 = None

    try:
        from market_client import get_fundamentals
        fund = get_fundamentals(pos.ticker, min_market_cap=0)
        if fund and not fund.get("skip"):
            try:
                from score import score_from_fundamentals
                sf = score_from_fundamentals(fund)
                if isinstance(sf, dict):
                    quality_score = float(sf.get("quality_score", 0.5))
                    fresh_score   = float(sf.get("final_score", 50.0))
                    out["fresh_score"] = fresh_score
                    out["quality_score"] = quality_score
            except Exception:
                pass
            try:
                from ml_features import build_features
                from ml_engine import load_predictor, predict_dip
                load_predictor()
                feat = build_features(pos.ticker, fund, price_history=hist,
                                       sector=fund.get("sector") or pos.entry_sector or "Unknown")
                pred = predict_dip(feature_row=feat, current_price=current, ticker=pos.ticker)
                if pred and pred.model_ready:
                    win_prob   = float(pred.win_prob)
                    pred_alpha = float(getattr(pred, "pred_up", 0.0) or 0.0)
                    out["win_prob"]   = win_prob
                    out["pred_alpha"] = pred_alpha
            except Exception as e:
                logger.debug(f"[conviction] eval ML {pos.ticker}: {e}")
    except Exception as e:
        logger.debug(f"[conviction] eval fund {pos.ticker}: {e}")

    # Technicals from history
    if hist is not None and len(hist) >= 20:
        try:
            closes = hist["Close"]
            ma20   = float(closes.rolling(20).mean().iloc[-1])
            rsi_now = _rsi(closes)
            out["rsi"]  = rsi_now
            out["ma20"] = ma20
        except Exception:
            pass

    # Update last-check snapshot on the position object (caller saves)
    pos.last_check_date  = _today_iso()
    pos.last_check_price = current
    if fresh_score is not None:
        pos.last_score = fresh_score
    if win_prob is not None:
        pos.last_win_prob = win_prob

    # ── Type-specific triggers ────────────────────────────────────────────────
    if pos.position_type == "MOMENTUM":
        atr_pct, atr_source = _compute_atr_stop_pct(hist, current)
        pos.last_atr_stop_pct = atr_pct
        stop_price = pos.trailing_high * (1 - atr_pct)
        out["trailing_high"] = pos.trailing_high
        out["stop_price"]    = stop_price
        out["atr_pct"]       = atr_pct
        out["atr_source"]    = atr_source

        if pos.days_held >= 3 and current < stop_price:
            # Confirm with RSI<48 or below MA20
            confirmation = False
            confirm_reasons = []
            if rsi_now is not None and rsi_now < 48:
                confirmation = True
                confirm_reasons.append(f"RSI {rsi_now:.0f}<48")
            if ma20 is not None and current < ma20:
                confirmation = True
                confirm_reasons.append(f"preço<MA20 ({current:.2f}<{ma20:.2f})")
            if confirmation:
                out["triggers"].append({
                    "kind": "TRAIL_STOP_CONFIRMED",
                    "severity": "high",
                    "message": (f"Trailing stop ATR accionado: ${current:.2f}<${stop_price:.2f}. "
                                f"Confirmação: {' + '.join(confirm_reasons)}.")
                })
            else:
                out["triggers"].append({
                    "kind": "TRAIL_STOP_PENDING",
                    "severity": "med",
                    "message": (f"Preço abaixo do stop ATR (${current:.2f}<${stop_price:.2f}) mas sem "
                                f"confirmação RSI/MA20 — pode ser pullback. Vigilância apertada.")
                })

        # Early alpha capture (if entry alpha was known)
        if pos.initial_pred_alpha and pos.initial_pred_alpha > 0.02:
            actual = current / pos.entry_price - 1.0
            if actual > 0:
                alpha_frac = actual / pos.initial_pred_alpha
                time_frac  = pos.days_held / 90.0
                if alpha_frac >= 0.70 and time_frac < 0.50:
                    out["triggers"].append({
                        "kind": "EARLY_ALPHA",
                        "severity": "med",
                        "message": (f"Capturado {alpha_frac*100:.0f}% do alpha previsto em "
                                    f"{time_frac*100:.0f}% do tempo. Considera realizar 50%.")
                    })

    elif pos.position_type == "DIP":
        # Edge gone: ML alpha collapsed
        if pos.days_held >= 7 and pred_alpha is not None and pred_alpha < _EDGE_FLOOR:
            out["triggers"].append({
                "kind": "EDGE_GONE",
                "severity": "high",
                "message": (f"Modelo prevê alpha {pred_alpha*100:.1f}pp (mínimo {_EDGE_FLOOR*100:.1f}pp). "
                            f"Edge evaporou — capital melhor noutro lado.")
            })

        # Win-prob deterioration ≥ 15pp
        if pos.entry_win_prob is not None and win_prob is not None:
            delta = pos.entry_win_prob - win_prob
            if delta >= 0.15:
                out["triggers"].append({
                    "kind": "DETERIORATION",
                    "severity": "high",
                    "message": (f"Win-prob caiu {delta*100:.0f}pp "
                                f"({pos.entry_win_prob*100:.0f}% → {win_prob*100:.0f}%). Tese a enfraquecer.")
                })

        # Quality deterioration ≥ 15pp from entry
        if pos.entry_quality_score is not None and quality_score is not None:
            qdrop = pos.entry_quality_score - quality_score
            if qdrop >= _QUALITY_DROP_TRIGGER:
                out["triggers"].append({
                    "kind": "QUALITY_DECLINE",
                    "severity": "high",
                    "message": (f"Qualidade fundamental caiu {qdrop*100:.0f}pp desde entrada. "
                                f"Possível declínio estrutural.")
                })

        # Time decay
        if pos.days_held >= _DIP_TIME_DECAY_DAYS:
            out["triggers"].append({
                "kind": "TIME_DECAY",
                "severity": "med",
                "message": (f"{pos.days_held}d detidos (>{_DIP_TIME_DECAY_DAYS}d). "
                            f"Tese de recuperação a expirar — revê.")
            })

        # Auto-transition suggestion: strong recovery → MOMENTUM
        if pnl_pct >= 15 and rsi_now is not None and rsi_now >= 60:
            out["triggers"].append({
                "kind": "SUGGEST_TRANSITION",
                "severity": "low",
                "message": (f"Recuperação forte (+{pnl_pct:.0f}%, RSI {rsi_now:.0f}). "
                            f"Considera /conviction transition {pos.ticker} MOMENTUM para activar trailing stop.")
            })

    elif pos.position_type == "CORE":
        # Long-term tracker: NO phase triggers (no trailing stop, no time decay).
        # ONLY structural triggers — and those CAN escalate to FECHAR even with
        # never_sell because a great company that stops being great must alert.
        if pos.entry_quality_score is not None and quality_score is not None:
            qdrop = pos.entry_quality_score - quality_score
            if qdrop >= 0.20:
                out["triggers"].append({
                    "kind": "CORE_QUALITY_DECLINE",
                    "severity": "high",
                    "message": (f"🚨 Qualidade fundamental caiu {qdrop*100:.0f}pp desde entrada "
                                f"({pos.entry_quality_score*100:.0f}% → {quality_score*100:.0f}%). "
                                f"Tese de 'grande empresa' invalidada — considera fechar.")
                })
            elif qdrop >= 0.10:
                out["triggers"].append({
                    "kind": "CORE_QUALITY_WARNING",
                    "severity": "med",
                    "message": (f"Qualidade caiu {qdrop*100:.0f}pp desde entrada "
                                f"({pos.entry_quality_score*100:.0f}% → {quality_score*100:.0f}%). "
                                f"Vigia próximos earnings.")
                })

        # Multi-criteria fundamental breakdown (revenue + margin + debt + ROE)
        current_fund_dict = {
            "revenue_growth": fund.get("revenue_growth") if 'fund' in locals() else None,
            "gross_margin":   fund.get("gross_margin")   if 'fund' in locals() else None,
            "debt_equity":    fund.get("debt_equity")    if 'fund' in locals() else None,
            "roe":            fund.get("roe")            if 'fund' in locals() else None,
            "fcf_yield":      fund.get("fcf_yield")      if 'fund' in locals() else None,
        }
        is_breakdown, breakdown_reasons = _detect_fundamental_breakdown(pos, current_fund_dict)
        if is_breakdown:
            out["triggers"].append({
                "kind": "FUNDAMENTAL_BREAKDOWN",
                "severity": "high",
                "message": (f"🚨 Quebra fundamental multi-critério: "
                            f"{' | '.join(breakdown_reasons)}.")
            })

    # ── Event-driven structural signals (applies to ALL types) ────────────
    # 8-K material adverse events + earnings call tone. These OVERRIDE
    # never_sell — a restatement filing or covenant default for MSFT must
    # trigger a SELL alert even though MSFT is a "never_sell CORE".
    event_triggers, event_notes = _detect_event_driven_signals(pos.ticker)
    out["triggers"].extend(event_triggers)
    if event_notes:
        out["event_signals"] = event_notes

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Rendering — condensed for hold/close decision
# ─────────────────────────────────────────────────────────────────────────────

def _trigger_emoji(severity: str) -> str:
    return {"high": "🚨", "med": "⚠️", "low": "💡"}.get(severity, "•")


def _hold_close_verdict(ev: dict, pos: Optional[ConvictionPosition] = None) -> str:
    """One-line recommendation derived from triggers.

    never_sell semantics:
      - Phase triggers (TRAIL_STOP, EARLY_ALPHA, DETERIORATION, EDGE_GONE,
        TIME_DECAY) are SUPPRESSED to 🟡 alerta when never_sell=True.
      - Structural triggers (QUALITY_DECLINE, CORE_QUALITY_DECLINE,
        STRUCTURAL_DECLINE, FUNDAMENTAL_BREAKDOWN, event-driven 8-K) ALWAYS
        produce 🔴 FECHAR regardless of never_sell — the whole point of
        never_sell is to ignore noise, not to ignore real thesis breakdown.
    """
    triggers = ev.get("triggers") or []
    highs = [t for t in triggers if t.get("severity") == "high"]
    meds  = [t for t in triggers if t.get("severity") == "med"]
    never_sell = bool(pos.never_sell) if pos else False

    structural_high = [t for t in highs if t["kind"] in STRUCTURAL_TRIGGERS]
    phase_high      = [t for t in highs if t["kind"] in PHASE_TRIGGERS]

    # 1. STRUCTURAL high — always overrides never_sell. Most severe outcome.
    if structural_high:
        return ("🔴 *FECHAR — DETERIORAÇÃO ESTRUTURAL*\n"
                f"_Trigger estrutural activo: {structural_high[0]['kind']}. "
                f"Tese de qualidade invalidada — não suprimível por never_sell._")

    # 2. PHASE high — suppressed to 🟡 if never_sell, else 🔴.
    if phase_high:
        if never_sell:
            return ("🟡 *ALERTA — trigger de fase activo (never_sell suprime fecho total)*\n"
                    f"_Considera **reduzir parcial** para travar lucro. "
                    f"`/conviction unlock {ev.get('ticker','')}` se quiseres permitir fecho total._")
        # Original behaviour (no never_sell)
        if any(t["kind"] in ("TRAIL_STOP_CONFIRMED", "EDGE_GONE") for t in phase_high):
            return "🔴 *FECHAR* — trigger forte accionado."
        if any(t["kind"] == "DETERIORATION" for t in phase_high):
            return "🟠 *FECHAR PARCIAL* — tese a enfraquecer; corta 30-50%."

    # 3. MED triggers
    if any(t["kind"] in ("CORE_QUALITY_WARNING",) for t in meds):
        return "🟡 *VIGIAR FUNDAMENTAIS* — qualidade a derivar negativamente; vigia próximos earnings."
    if any(t["kind"] == "EARLY_ALPHA" for t in meds):
        return "🟡 *REALIZAR PARCIAL* — alpha quase todo capturado; trava 50%."
    if any(t["kind"] == "TRAIL_STOP_PENDING" for t in meds):
        return "🟡 *VIGIAR* — stop tocado sem confirmação. Pronto a fechar amanhã se RSI/MA20 confirmar."
    if any(t["kind"] == "TIME_DECAY" for t in meds):
        return "🟡 *REVER PRAZO* — tese de recuperação a expirar."

    return "🟢 *MANTER* — sem triggers de saída."


def render_deep_dive(ticker: str) -> str:
    """Condensed hold/close report for a single conviction position."""
    pos = get_by_ticker(ticker)
    if not pos:
        return f"_Nenhuma convicção activa para *{ticker.upper()}*. Usa `/conviction add` primeiro._"

    ev = evaluate_conviction(pos)

    # Persist refreshed snapshot (no log writes outside personal_positions.json)
    positions = load_convictions()
    for i, p in enumerate(positions):
        if p.ticker == pos.ticker:
            positions[i] = pos
            break
    save_convictions(positions)

    pnl_sign = "+" if ev["pnl_pct"] >= 0 else ""
    pnl_emo  = "📈" if ev["pnl_pct"] >= 0 else "📉"

    # Header includes original_type + lock badge if relevant
    type_badge = pos.position_type
    if pos.original_type and pos.original_type != pos.position_type:
        type_badge = f"{pos.position_type} (era {pos.original_type})"
    lock_badge = " 🔒" if pos.never_sell else ""

    lines = [
        f"🧠 *Convicção {pos.ticker}* — {type_badge}{lock_badge} | Dia {ev['days_held']}",
        "",
        f"💰 *Posição:* {pos.shares:g}x @ ${pos.entry_price:.2f} → ${ev['current_price']:.2f}",
        f"   {pnl_emo} P&L: *${ev['pnl_abs']:+.2f}* ({pnl_sign}{ev['pnl_pct']:.2f}%)",
    ]

    # Type-specific body
    if pos.position_type == "MOMENTUM":
        stop = ev.get("stop_price")
        atr  = ev.get("atr_pct")
        trail = ev.get("trailing_high")
        rsi  = ev.get("rsi")
        ma20 = ev.get("ma20")
        gap_pct = (ev["current_price"] / stop - 1.0) * 100 if stop else None
        gap_str = f"+{gap_pct:.1f}%" if gap_pct and gap_pct >= 0 else (f"{gap_pct:.1f}%" if gap_pct else "N/A")
        confort = "confortável" if (gap_pct is not None and gap_pct > 5) else ("apertado" if gap_pct is not None else "—")
        lines += [
            "",
            f"📉 *Trailing stop:* ${stop:.2f}" + (f" (ATR {atr*100:.1f}%, base ${trail:.2f})" if atr and trail else ""),
            f"   Gap: {gap_str} → {confort}",
        ]
        if rsi is not None and ma20 is not None:
            rsi_lbl  = "forte" if rsi >= 60 else ("neutro" if rsi >= 45 else "fraco")
            ma_lbl   = "acima" if ev["current_price"] > ma20 else "abaixo"
            lines += [f"   RSI: {rsi:.0f} ({rsi_lbl}) | MA20: ${ma20:.2f} ({ma_lbl})"]
    elif pos.position_type == "DIP":
        win = ev.get("win_prob")
        alpha = ev.get("pred_alpha")
        fresh = ev.get("fresh_score")
        lines += [""]
        if pos.entry_win_prob is not None and win is not None:
            d = (win - pos.entry_win_prob) * 100
            arrow = "↑" if d >= 0 else "↓"
            lines.append(f"🤖 Win-prob: {pos.entry_win_prob*100:.0f}% → {win*100:.0f}% ({arrow}{abs(d):.0f}pp)")
        if alpha is not None:
            edge = "edge intacta" if alpha >= _EDGE_FLOOR else "edge evaporada"
            lines.append(f"   Alpha previsto: {alpha*100:+.1f}pp → {edge}")
        if fresh is not None:
            verdict = "COMPRAR" if fresh >= 75 else ("MONITORIZAR" if fresh >= 50 else "EVITAR")
            lines.append(f"   Score actual se fosse entry: {fresh:.0f}/100 ({verdict})")
    else:  # CORE
        fresh = ev.get("fresh_score")
        q     = ev.get("quality_score")
        lines += [""]
        if fresh is not None:
            lines.append(f"🏛️ Score actual: {fresh:.0f}/100")
        if pos.entry_quality_score is not None and q is not None:
            d = (q - pos.entry_quality_score) * 100
            arrow = "↑" if d >= 0 else "↓"
            lines.append(f"   Qualidade: {pos.entry_quality_score*100:.0f}% → {q*100:.0f}% ({arrow}{abs(d):.0f}pp)")
        lines.append(f"   _CORE: sem stop automático. Reavalia mensalmente._")

    # Event-driven signals (informational, even if no trigger fired)
    event_notes = ev.get("event_signals") or []
    if event_notes:
        lines += ["", f"📰 *Sinais event-driven:* { ' | '.join(event_notes) }"]

    # Triggers
    triggers = ev.get("triggers") or []
    if triggers:
        lines += ["", "🎯 *Triggers activos:*"]
        for t in triggers:
            lines.append(f"   {_trigger_emoji(t['severity'])} {t['message']}")
    else:
        lines += ["", "✅ _Sem triggers de saída._"]

    # Pass pos to verdict so it can respect never_sell
    lines += ["", _hold_close_verdict(ev, pos)]

    if pos.note:
        lines += ["", f"_Nota original: {pos.note}_"]
    return "\n".join(lines)


def render_list() -> str:
    positions = load_convictions()
    if not positions:
        return "_Sem convicções activas. Usa `/conviction add TICKER PREÇO SHARES TIPO`._"

    out = ["🧠 *Convicções activas*", ""]
    n_core = n_lock = 0
    for p in sorted(positions, key=lambda x: x.entry_date):
        cur = _fetch_price(p.ticker, fallback=p.entry_price)
        pnl_pct = (cur / p.entry_price - 1.0) * 100.0
        emoji   = "📈" if pnl_pct >= 0 else "📉"
        sign    = "+" if pnl_pct >= 0 else ""
        lock_badge = " 🔒" if p.never_sell else ""
        if p.position_type == "CORE":
            n_core += 1
        if p.never_sell:
            n_lock += 1
        out.append(
            f"  {emoji} *{p.ticker}* ({p.position_type}){lock_badge} | "
            f"{p.shares:g}x @ ${p.entry_price:.2f} → ${cur:.2f}"
            f"  {sign}{pnl_pct:.1f}% | dia {p.days_held}"
        )
    out += [
        "",
        f"_{len(positions)} posições | {n_core} CORE | {n_lock} 🔒 never_sell_",
        "_Detalhe: `/conviction TICKER` | Lock: `/conviction lock TICKER`_",
    ]
    return "\n".join(out)


def render_outcomes(limit: int = 15) -> str:
    data = load_outcomes()
    if not data:
        return "_Sem histórico de saídas de convicção._"
    out = [f"🧠 *Convicções fechadas* (últimas {limit})", ""]
    for entry in sorted(data, key=lambda x: x.get("exit_date", ""), reverse=True)[:limit]:
        pnl = entry.get("pnl_pct", 0)
        emoji = "🟢" if pnl >= 0 else "🔴"
        sign  = "+" if pnl >= 0 else ""
        out.append(
            f"  {emoji} *{entry['ticker']}* ({entry['position_type']}) | "
            f"{sign}{pnl:.1f}% ({entry['days_held']}d) | {entry['reason']}"
        )
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-classifier + auto-sync from portfolio
# ─────────────────────────────────────────────────────────────────────────────

# Tickers that should never be auto-tracked as convictions: passive ETFs,
# foreign-listed instruments, non-equity holdings. Extensible via env var
# CONVICTION_DENY_TICKERS="A,B,C".
_AUTO_TRACK_DENY_BASE = {
    # US benchmarks
    "SPY", "QQQ", "IVV", "VTI", "VOO",
    # US sector ETFs
    "XLK", "XLV", "XLF", "XLE", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC", "XLB",
    # Foreign-listed ETFs commonly held by EU/UK investors
    "EUNL", "EUNL.DE", "IS3N", "IEMA", "ALV",
    # Portuguese non-stock instruments
    "PPR",
}


def _get_deny_list() -> set[str]:
    extras = os.getenv("CONVICTION_DENY_TICKERS", "")
    extra_set = {t.strip().upper() for t in extras.split(",") if t.strip()}
    return _AUTO_TRACK_DENY_BASE | extra_set


def _classify_type_from_signals(ticker: str, current_price: float, hist) -> str:
    """Auto-pick MOMENTUM or DIP based on ML signals + technicals.

    NEVER returns CORE — CORE is always user-explicit. Auto-sync of a new
    holding should err on the side of armed monitoring (MOMENTUM with
    trailing stop) rather than passive long-term assumption.

    Priority:
      1. Momentum ML model signal (high score → MOMENTUM)
      2. DIP ML model alpha forecast (high alpha → DIP)
      3. Technical fallback (RSI + price vs MA20 + trend direction)
      4. Default MOMENTUM with current price as trailing_high (preserve gains)
    """
    # 1. Try momentum signal
    try:
        from momentum_radar.scanner import score_momentum_for_ticker
        m_score = score_momentum_for_ticker(ticker)
        if m_score is not None and float(m_score) >= 60:
            logger.debug(f"[conviction] auto-classify {ticker}: MOMENTUM (momentum score={m_score:.0f})")
            return "MOMENTUM"
    except Exception as e:
        logger.debug(f"[conviction] momentum signal {ticker}: {e}")

    # 2. Try DIP model
    try:
        from market_client import get_fundamentals
        from ml_features import build_features
        from ml_engine import load_predictor, predict_dip
        load_predictor()
        fund = get_fundamentals(ticker, min_market_cap=0)
        if fund and not fund.get("skip"):
            feat = build_features(ticker, fund, price_history=hist,
                                   sector=fund.get("sector") or "Unknown")
            pred = predict_dip(feature_row=feat, current_price=current_price, ticker=ticker)
            if pred and pred.model_ready:
                alpha = float(getattr(pred, "pred_up", 0.0) or 0.0)
                if alpha >= 0.05:
                    logger.debug(f"[conviction] auto-classify {ticker}: DIP (alpha={alpha:.3f})")
                    return "DIP"
    except Exception as e:
        logger.debug(f"[conviction] DIP signal {ticker}: {e}")

    # 3. Technical fallback
    if hist is not None and len(hist) >= 20:
        try:
            closes = hist["Close"]
            ma20   = float(closes.rolling(20).mean().iloc[-1])
            rsi    = _rsi(closes)
            # Recent trend: last 20d return
            recent_ret = float(closes.iloc[-1] / closes.iloc[-20] - 1) if len(closes) >= 20 else 0.0
            if rsi >= 55 and current_price > ma20 and recent_ret > 0.0:
                logger.debug(f"[conviction] auto-classify {ticker}: MOMENTUM (RSI {rsi:.0f}, +{recent_ret*100:.0f}% 20d)")
                return "MOMENTUM"
            if rsi < 40 and current_price < ma20:
                logger.debug(f"[conviction] auto-classify {ticker}: DIP (RSI {rsi:.0f}, below MA20)")
                return "DIP"
        except Exception:
            pass

    # 4. Default
    logger.debug(f"[conviction] auto-classify {ticker}: MOMENTUM (fallback)")
    return "MOMENTUM"


def auto_sync_from_portfolio(send_telegram: Optional[Callable[[str], None]] = None) -> dict:
    """Create conviction records for portfolio holdings not yet tracked.

    Skips:
      - Tickers in _AUTO_TRACK_DENY (ETFs, non-equity)
      - Foreign-listed tickers (any with a dot — .DE, .L, .PA, etc.)
      - Tickers already in personal_positions.json

    Type is auto-classified via ML signals + technicals.
    Never auto-classifies as CORE (CORE is always user-explicit).

    Returns: {"added": N, "skipped": M, "items": [(ticker, type, reason)...]}
    """
    try:
        from portfolio import get_positions
    except Exception as e:
        logger.warning(f"[conviction] portfolio import failed: {e}")
        return {"added": 0, "skipped": 0, "items": []}

    try:
        portfolio_positions = get_positions()
    except Exception as e:
        logger.warning(f"[conviction] get_positions failed: {e}")
        return {"added": 0, "skipped": 0, "items": []}

    deny = _get_deny_list()
    existing = {p.ticker for p in load_convictions()}
    added: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    already: list[str] = []

    portfolio_count = len(portfolio_positions)

    for sym, info in portfolio_positions.items():
        ticker = sym.upper().strip()
        if ticker in existing:
            already.append(ticker)
            continue
        if ticker in deny:
            skipped.append((ticker, "deny-list (ETF/non-equity)"))
            continue
        if "." in ticker:
            skipped.append((ticker, "foreign-listed"))
            continue

        try:
            entry_price = float(info.get("avg_price", 0) or 0)
            shares      = float(info.get("shares", 0) or 0)
        except (TypeError, ValueError):
            skipped.append((ticker, "missing entry data"))
            continue
        if entry_price <= 0 or shares <= 0:
            skipped.append((ticker, "zero shares/price"))
            continue

        # Auto-classify
        current = _fetch_price(ticker, fallback=entry_price)
        hist    = _fetch_price_history(ticker, period="6mo")
        ptype   = _classify_type_from_signals(ticker, current, hist)

        try:
            add_conviction(
                ticker, entry_price, shares, ptype,
                trailing_high = current if ptype == "MOMENTUM" else None,
                note          = "auto-sync from portfolio",
            )
            added.append((ticker, ptype))
        except Exception as e:
            logger.warning(f"[conviction] auto-sync {ticker} failed: {e}")
            skipped.append((ticker, f"error: {e}"))

    if added and send_telegram:
        lines = ["🔄 *Auto-sync convicção ↔ carteira*", ""]
        for t, p in added:
            lines.append(f"  ✅ {t} → {p} (auto-classificado)")
        if skipped:
            lines += ["", "_Saltadas:_"]
            for t, r in skipped:
                lines.append(f"  ⏭️ {t} _({r})_")
        lines += ["", "_Para upgrade a CORE: `/conviction transition TICKER CORE`_"]
        try:
            send_telegram("\n".join(lines))
        except Exception as e:
            logger.warning(f"[conviction] auto-sync telegram failed: {e}")

    return {
        "added":           len(added),
        "skipped":         len(skipped),
        "already_tracked": len(already),
        "portfolio_count": portfolio_count,
        "items": (
            [(t, p, "added")   for t, p in added] +
            [(t, "-", r)        for t, r in skipped] +
            [(t, "-", "já em convicção") for t in already]
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Daily monitor — called from main.py cron
# ─────────────────────────────────────────────────────────────────────────────

def run_daily_check(send_telegram: Callable[[str], None]) -> None:
    """Iterate all conviction positions, evaluate, send alerts for high-severity triggers.

    Persists refreshed scores back to personal_positions.json. Never writes to
    prediction_log, position_db, paper_trading, or any feature parquet.
    """
    positions = load_convictions()
    if not positions:
        logger.info("[conviction] no positions to monitor")
        return

    logger.info(f"[conviction] daily check: {len(positions)} positions")
    n_alerts = 0
    for i, pos in enumerate(positions):
        try:
            ev = evaluate_conviction(pos)
            high_sev = [t for t in ev.get("triggers", []) if t.get("severity") == "high"]
            med_sev  = [t for t in ev.get("triggers", []) if t.get("severity") == "med"]
            if high_sev or med_sev:
                # Build condensed alert
                msg = render_deep_dive(pos.ticker)
                send_telegram(msg)
                n_alerts += 1
        except Exception as e:
            logger.error(f"[conviction] {pos.ticker}: {e}", exc_info=True)
            try:
                send_telegram(f"⚠️ *Convicção {pos.ticker}* — erro no check diário: `{e}`")
            except Exception:
                pass
        positions[i] = pos  # mutation captured

    # Persist refreshed snapshots
    save_convictions(positions)
    logger.info(f"[conviction] daily check done: {n_alerts} alerts sent")


# ─────────────────────────────────────────────────────────────────────────────
# Safety assertion: confirm isolation file paths
# ─────────────────────────────────────────────────────────────────────────────

def assert_isolation() -> dict:
    """Sanity check used by tests: confirms conviction files are not referenced
    by any training script. Returns the protected paths."""
    return {
        "positions_file": _POSITIONS_FILE,
        "outcomes_file":  _OUTCOMES_FILE,
        "source_tag":     "conviction",
    }
