"""scripts/validation_static_fetch.py — Pull static fundamentals for trailing valuation.

For each CORE ticker, fetch a small set of CURRENT values from yfinance
that the spine builder needs to compute trailing-P/E, P/S, and PEG:
  shares_outstanding  : current shares (assumed roughly stable over the
                        backtest window — small buyback drift OK)
  trailing_eps        : current TTM EPS (used as anchor; spine recomputes
                        per-period TTM EPS from quarterly_financials cache
                        + this static shares value)

Persisted to /data/validation_cache/ticker_static.json (small JSON).
Used by build_dislocation_validation_spine.py to activate sub-score #5
(valuation_pit_clean) — previously zeroed because forward EPS lacked
free PIT history. Trailing variants are reconstructible from prices +
cached fundamentals + this static shares anchor.

Caveat documented in spine_metadata.sub_scores_active_in_spine:
shares_outstanding is "current" not "as-of-Friday", so capital actions
(buybacks/splits/issuance) between past Fridays and today introduce a
small approximation. For 5y backtest window, typical shares drift is
<5% for mature compounders, <15% for growers. Material for individual
year-by-year accuracy, immaterial for top-decile signal direction.

Runtime: ~3-4 min for 230 tickers (sequential .info fetches).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("static_fetch")

_DATA_DIR  = Path("/data") if Path("/data").exists() else _ROOT / "data"
_CACHE_DIR = _DATA_DIR / "validation_cache"
_OUT_PATH  = _CACHE_DIR / "ticker_static.json"


def _load_core() -> list[str]:
    u = json.loads((_ROOT / "core_universe.json").read_text())
    return [t["ticker"] for t in u.get("tickers", [])]


def main():
    if not _CACHE_DIR.exists():
        log.error(f"{_CACHE_DIR} missing — run validation_cache_builder.py first")
        sys.exit(1)

    if _OUT_PATH.exists():
        log.info(f"[static] cache hit: {_OUT_PATH} — use rm to rebuild")
        return

    import yfinance as yf

    core = _load_core()
    log.info(f"[static] fetching shares + trailing_eps for {len(core)} tickers")

    out = {}
    skipped = []
    for i, t in enumerate(core, 1):
        try:
            info = yf.Ticker(t).info or {}
            shares = info.get("sharesOutstanding")
            eps_ttm = info.get("trailingEps")
            market_cap = info.get("marketCap")
            entry = {
                "shares_outstanding": float(shares) if shares else None,
                "trailing_eps":       float(eps_ttm) if eps_ttm else None,
                "market_cap_current": float(market_cap) if market_cap else None,
            }
            # Need at least shares to compute anything PIT
            if entry["shares_outstanding"] is None:
                skipped.append({"ticker": t, "reason": "no sharesOutstanding"})
                continue
            out[t] = entry
        except Exception as e:
            log.warning(f"[static] {t}: {e}")
            skipped.append({"ticker": t, "reason": f"yfinance error: {e}"})
        if i % 25 == 0:
            log.info(f"[static] {i}/{len(core)} (skipped so far: {len(skipped)})")

    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_tickers":  len(out),
        "n_skipped":  len(skipped),
        "skipped":    skipped,
        "tickers":    out,
    }
    _OUT_PATH.write_text(json.dumps(payload, indent=2))
    log.info(f"[static] persisted {len(out)} tickers to {_OUT_PATH} (skipped: {len(skipped)})")


if __name__ == "__main__":
    main()
