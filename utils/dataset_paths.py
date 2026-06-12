"""
dataset_paths.py — Single source of truth for training-dataset paths.

Why this module exists
======================
Before 2026-06-12, every pipeline orchestrator in bot_commands.py computed
its own `out_path = data_dir / "<dataset>.parquet"` and then appended
`--tickers <T1,T2>` when the user asked for `--test`. The build scripts
respected the path they were given, so a smoke test wrote 211k → 381 rows
into the SAME production file. Three datasets had this bug
(ml_training_base, core_training_base, momentum_training) and one test run
corrupted all of them.

The fix is structural: production and test must NEVER share a path.

  production:  /data/<dataset>.parquet
  test mode:   /data/test/<dataset>.parquet

The test directory is created lazily. Test parquets are NOT seeded from the
repo and NOT copied to the repo or GitHub — they are throw-away artefacts
that only need to verify the pipeline finishes without crashing.

Usage
-----
    from utils.dataset_paths import resolve_dataset_path, is_test_mode

    is_test = is_test_mode(ticker_val)
    out_path = resolve_dataset_path("ml_training_base.parquet", is_test=is_test)
    cmd = [sys.executable, str(script), "--out", str(out_path), ...]
    if is_test:
        cmd += ["--tickers", ticker_val]

Whoever adds a fourth dataset later: route it through here too.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


# /data/ is the Railway persistent volume; on local dev it doesn't exist
# and we fall back to /tmp/. The two paths are siblings of each other in
# their test/prod split — never mixed.
def _data_root() -> Path:
    return Path("/data") if Path("/data").exists() else Path("/tmp")


def is_test_mode(ticker_val: Optional[str]) -> bool:
    """True when the orchestrator received a --test/--tickers value.

    A non-empty ticker_val is the canonical signal that the user wants a
    smoke test on a subset (e.g. "AAPL,MSFT") rather than a full universe
    rebuild. Treat empty string as production to be defensive against
    callers that pass "" for "no value".
    """
    return bool(ticker_val and ticker_val.strip())


def resolve_dataset_path(filename: str, *, is_test: bool) -> Path:
    """Resolve the parquet path for a dataset build.

    is_test=False → /data/<filename>     (production — overwrite OK)
    is_test=True  → /data/test/<filename> (sandboxed — never touches prod)

    Creates the parent directory on demand so callers don't have to.
    """
    root = _data_root()
    if is_test:
        out = root / "test" / filename
    else:
        out = root / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def production_dataset_path(filename: str) -> Path:
    """Explicit production path — used by readers (calibrators, retrainers)
    that must NEVER pick up a test parquet by accident."""
    return _data_root() / filename
