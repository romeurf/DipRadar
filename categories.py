"""categories.py — LEGACY position category constants.

DEPRECATION NOTE:
  These constants (CAT_HIGH_CONVICTION, CAT_GROWTH) don't map to the user's
  current mental model. The actual position buckets are managed via
  /sync_portfolio: CORE / ROTATION / MOMENTUM / DIP. The CAT_* values here
  are a separate "scoring/signal classification" dimension that has been
  effectively abandoned. They're kept ONLY to avoid breaking 5 legacy
  callers (score.py:classify_dip_category, main.py:multiple, watchlist.py:
  curated tickers, alert_db.py:filter, backtest.py).

  A future cleanup will:
    - Replace classify_dip_category with direct gating (is_bluechip + score)
    - Tag watchlist.py curated tickers with user buckets (CORE / MOMENTUM)
    - Remove all CAT_* imports
    - Delete this file

CAT_FLIP was already removed (Phase 6) — Flip Fund concept replaced by /cash.
CAT_CORE / CAT_PASS were used by the deleted allocation_engine.py — also
removed since they're not imported anywhere else.
"""

CAT_HIGH_CONVICTION = "HIGH_CONVICTION"
CAT_GROWTH          = "GROWTH"
