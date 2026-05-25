"""
config.py — Single source of truth for market constants, model IDs, and data paths.

Import from here instead of hardcoding values across the codebase.
"""
from __future__ import annotations

from pathlib import Path

# ── Market universe ───────────────────────────────────────────────────────────
MIN_MARKET_CAP_USD: int   = 2_000_000_000   # $2B floor — mid-cap and above
MAX_TICKERS_PER_SCAN: int = 3_500           # hard cap on live scan universe size
US_EXCHANGES: list[str]   = ["NYSE", "NASDAQ", "AMEX", "NYSEARCA", "BATS"]

# Exchanges where SEC filings (Form 4, 8-K) are expected
SEC_FILING_EXCHANGES: set[str] = {"NYSE", "NASDAQ", "AMEX"}

# Countries whose companies are required to file with the SEC.
# Empty string = yfinance returned no country — treated as possibly US (fail-open).
SEC_DOMICILE_COUNTRIES: set[str] = {"US", "USA", ""}


# ── Model registry IDs ────────────────────────────────────────────────────────
class ModelID:
    # Phase 1 — US markets (implemented)
    DIP_US       = "dip_us"
    CORE_US      = "core_us"
    MOMENTUM_US  = "momentum_us"

    # Phase 2 — European markets (not yet implemented)
    DIP_EU       = "dip_eu"
    CORE_EU      = "core_eu"
    MOMENTUM_EU  = "momentum_eu"


IMPLEMENTED_MODELS: list[str] = [
    ModelID.DIP_US,
    ModelID.CORE_US,
    ModelID.MOMENTUM_US,
]


# ── Language ──────────────────────────────────────────────────────────────────
DEFAULT_LANGUAGE: str        = "pt"
SUPPORTED_LANGUAGES: list[str] = ["en", "pt"]


# ── Data paths (Railway volume first, /tmp fallback) ──────────────────────────
_DATA_DIR          = Path("/data") if Path("/data").exists() else Path("/tmp")

BOOTSTRAP_PATH     = _DATA_DIR / "ml_training_base.parquet"
SNAPSHOT_PATH      = _DATA_DIR / "universe_snapshot.parquet"
ALERT_DB_PATH      = _DATA_DIR / "alert_db.csv"
TRAINING_INPUT     = _DATA_DIR / "ml_training_input.parquet"
PRODUCTION_BUNDLE  = _DATA_DIR / "dip_models.pkl"
