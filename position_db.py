"""
position_db.py — Persistent storage for active DipRadar positions.

Design constraints:
  - No external database. Storage is a single JSON file on a Railway
    persistent volume (mounted at $DIPR_POSITIONS_PATH, default:
    data/positions.json).
  - File is written atomically via a temp file + os.replace() to prevent
    corruption on Railway container restarts mid-write.
  - fcntl advisory lock prevents concurrent writes from scheduler + bot
    command handler running in the same process (APScheduler threads).
  - All datetimes stored as ISO-8601 strings (timezone-naive UTC).

Public API:
  add_position(record)          → saves a new PositionRecord
  close_position(ticker, reason)→ sets status="CLOSED", stamps closed_at
  get_active()                  → list[PositionRecord] where status=="ACTIVE"
  get_all()                     → list[PositionRecord] (all statuses)
  update_record(record)         → overwrites existing record by ticker
  get_by_ticker(ticker)         → PositionRecord | None

PositionRecord fields:
  Core identity:
    ticker          str
    status          "ACTIVE" | "CLOSED" | "TAKE_PROFIT" | "STOP_LOSS"
    alert_date      str (ISO date of original DIP ALERT)

  Alert-day snapshot (frozen — never updated after creation):
    alert_price         float   price at moment of alert
    alert_win_prob      float   win_prob from ml_engine on alert day
    alert_feature_row   list[float]  16-feature vector on alert day
    initial_buy_target  float
    initial_sell_target float
    initial_hold_days   int
    dip_score           float
    fundamentals_snap   dict    snapshot of fundamentals at alert time

  Live targets (updated daily by position_monitor):
    current_sell_target float
    current_hold_days   int
    last_win_prob       float   most recent win_prob from monitoring
    last_checked_date   str
    thesis_health       "STRONG" | "WEAKENING" | "DETERIORATING" | "IMPROVING"

  Outcome:
    closed_at   str | None
    close_reason str | None   e.g. "TAKE_PROFIT", "TIME_DECAY", "MANUAL"
    close_price float | None

  Audit trail:
    history  list[dict]   daily monitoring snapshots
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("DIPR_POSITIONS_PATH", "data/positions.json"))


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PositionRecord:
    # Identity
    ticker:               str
    status:               str                    # ACTIVE | CLOSED | TAKE_PROFIT | STOP_LOSS
    alert_date:           str                    # ISO date string

    # Alert-day frozen snapshot
    alert_price:          float
    alert_win_prob:       float
    alert_feature_row:    list                   # list[float], len == N_FEATURES
    initial_buy_target:   float
    initial_sell_target:  float
    initial_hold_days:    int
    dip_score:            float
    fundamentals_snap:    dict

    # Live targets (mutated by monitor)
    current_sell_target:  float
    current_hold_days:    int
    last_win_prob:        float
    last_checked_date:    Optional[str]          = None
    thesis_health:        str                    = "STRONG"

    # Alpha inicial previsto pelo ML no momento do alerta (pred_up / alpha_90d).
    # Usado pelo Early Alpha Capture: quando o retorno actual já representa
    # >70% do alpha previsto em menos de metade do tempo, sugere saída parcial.
    initial_pred_alpha:   Optional[float]        = None   # log-return excess SPY

    # Outcome
    closed_at:            Optional[str]          = None
    close_reason:         Optional[str]          = None
    close_price:          Optional[float]        = None

    # Audit trail — list of daily monitoring snapshots
    history:              list                   = field(default_factory=list)

    @property
    def days_held(self) -> int:
        """Calendar days since the alert date."""
        try:
            start = date.fromisoformat(self.alert_date)
            return (date.today() - start).days
        except (ValueError, TypeError):
            return 0

    @property
    def days_remaining(self) -> int:
        """Estimated days left before hold period expires."""
        return max(0, self.current_hold_days - self.days_held)

    @property
    def win_prob_delta(self) -> float:
        """Change in win_prob since alert day (positive = improved)."""
        return self.last_win_prob - self.alert_win_prob


def _record_from_dict(d: dict) -> PositionRecord:
    """Reconstruct a PositionRecord from a JSON-deserialised dict."""
    # Strip unknown keys for forward-compatibility
    known = {f.name for f in PositionRecord.__dataclass_fields__.values()}
    clean = {k: v for k, v in d.items() if k in known}
    return PositionRecord(**clean)


# ─────────────────────────────────────────────────────────────────────────────
# File I/O  (atomic write + advisory lock)
# ─────────────────────────────────────────────────────────────────────────────

def _load_raw() -> list[dict]:
    """Read the JSON file. Returns [] if missing or malformed."""
    if not DB_PATH.exists():
        return []
    try:
        with DB_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"[position_db] Failed to read {DB_PATH}: {e}")
        return []


def _save_raw(records: list[dict]) -> None:
    """Atomically write records to disk with advisory lock."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_path = DB_PATH.with_suffix(".lock")
    with open(lock_path, "w") as lock_fh:
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
            # Write to temp file in same directory, then atomic replace
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=DB_PATH.parent, prefix=".positions_", suffix=".tmp"
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_fh:
                    json.dump(records, tmp_fh, indent=2, ensure_ascii=False)
                os.replace(tmp_path, DB_PATH)
            except Exception:
                os.unlink(tmp_path)
                raise
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
    logger.debug(f"[position_db] Saved {len(records)} records → {DB_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_all() -> list[PositionRecord]:
    """Return all positions (any status)."""
    return [_record_from_dict(d) for d in _load_raw()]


def save_all(records: list[PositionRecord]) -> None:
    """Overwrite the full positions file."""
    _save_raw([asdict(r) for r in records])


def get_all() -> list[PositionRecord]:
    return load_all()


def get_active() -> list[PositionRecord]:
    """Return only ACTIVE positions."""
    return [r for r in load_all() if r.status == "ACTIVE"]


def get_by_ticker(ticker: str) -> Optional[PositionRecord]:
    """Return the most recent position for ticker, or None."""
    matches = [r for r in load_all() if r.ticker == ticker]
    if not matches:
        return None
    # Most recent alert date first
    return sorted(matches, key=lambda r: r.alert_date, reverse=True)[0]


def add_position(record: PositionRecord) -> None:
    """
    Persist a new position. Replaces any existing ACTIVE record for the
    same ticker (avoids duplicate active positions on re-alert).
    """
    all_records = load_all()
    # Archive any existing active record for same ticker
    for r in all_records:
        if r.ticker == record.ticker and r.status == "ACTIVE":
            r.status = "CLOSED"
            r.close_reason = "SUPERSEDED"
            r.closed_at = datetime.utcnow().isoformat()
            logger.info(f"[position_db] Superseded existing ACTIVE position for {record.ticker}")
    all_records.append(record)
    save_all(all_records)
    logger.info(f"[position_db] Added position: {record.ticker} (alert_date={record.alert_date})")


def update_record(record: PositionRecord) -> None:
    """
    Update an existing record in-place (matched by ticker + alert_date).
    Used by position_monitor to write revised targets and health each day.
    """
    all_records = load_all()
    replaced = False
    for i, r in enumerate(all_records):
        if r.ticker == record.ticker and r.alert_date == record.alert_date:
            all_records[i] = record
            replaced = True
            break
    if not replaced:
        logger.warning(
            f"[position_db] update_record: no existing record for "
            f"{record.ticker} / {record.alert_date} — appending as new"
        )
        all_records.append(record)
    save_all(all_records)


def close_position(
    ticker: str,
    reason: str,
    close_price: Optional[float] = None,
    alert_date: Optional[str] = None,
) -> bool:
    """
    Mark a position as closed. Returns True if a matching ACTIVE record
    was found and updated, False otherwise.
    close_reason should be one of: TAKE_PROFIT, TIME_DECAY, DETERIORATION,
    MANUAL, STOP_LOSS, SUPERSEDED.
    """
    all_records = load_all()
    closed = False
    for r in all_records:
        if r.ticker == ticker and r.status == "ACTIVE":
            if alert_date and r.alert_date != alert_date:
                continue
            r.status = "CLOSED"
            r.close_reason = reason
            r.closed_at = datetime.utcnow().isoformat()
            r.close_price = close_price
            closed = True
            logger.info(f"[position_db] Closed {ticker} — reason={reason} price={close_price}")
            break
    if closed:
        save_all(all_records)
    else:
        logger.warning(f"[position_db] close_position: no ACTIVE record found for {ticker}")
    return closed


def summary_text() -> str:
    """One-line summary for /posicoes Telegram command."""
    active = get_active()
    if not active:
        return "📭 Sem posições activas."
    lines = [f"📋 *Posições activas* ({len(active)})", ""]
    for r in sorted(active, key=lambda x: x.alert_date):
        health_emoji = {
            "STRONG":             "🟢",
            "IMPROVING":          "📈",
            "WEAKENING":          "🟡",
            "DETERIORATING":      "🔴",
            "STRUCTURAL_DECLINE": "🚨",
        }.get(r.thesis_health, "⚪")
        pnl_pct = ((r.alert_price / r.initial_buy_target) - 1) * 100 if r.initial_buy_target else 0
        lines.append(
            f"{health_emoji} *{r.ticker}*  Dia {r.days_held}/{r.current_hold_days}  "
            f"| Alvo venda: ${r.current_sell_target:.2f}  "
            f"| Conf: {r.last_win_prob*100:.0f}%"
        )
    return "\n".join(lines)
