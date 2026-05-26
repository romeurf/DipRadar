"""
scripts/collect_news_sentiment.py — News headline sentiment at alert time.

Scores recent headlines for a ticker using VADER (free, offline, no API key).
Called only when a real DIP/Core/Momentum alert fires — never in batch.

VADER (Valence Aware Dictionary and sEntiment Reasoner) is purpose-built
for short social/news text. compound score: +1 = very positive, -1 = very negative.
Thresholds: compound >= 0.05 → POSITIVE (+1), <= -0.05 → NEGATIVE (-1), else NEUTRAL (0).

Accumulates in /data/news_sentiment.parquet for future training features:
  news_sentiment_7d  = mean score over last 7 days before alert date
  news_sentiment_30d = mean score over last 30 days

No API key required. Install: vaderSentiment (in requirements.txt).
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR       = Path("/data") if Path("/data").exists() else Path("/tmp")
_SENTIMENT_PATH = _DATA_DIR / "news_sentiment.parquet"
_MAX_HEADLINES  = 3      # per ticker per alert
_MAX_AGE_HOURS  = 48     # only headlines from last 48h
_SLEEP_BETWEEN  = 0.15   # sec between tickers (yfinance rate limit)

log = logging.getLogger("collect_sentiment")


# ── VADER scorer ─────────────────────────────────────────────────────────────

def _get_vader():
    """Return a SentimentIntensityAnalyzer, or None if vaderSentiment not installed."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        log.debug("[sentiment] vaderSentiment not installed — pip install vaderSentiment")
        return None


def _score_headline_vader(headline: str, analyzer) -> float:
    """Score a headline with VADER. Returns 1.0 (positive), -1.0 (negative), 0.0 (neutral)."""
    compound = analyzer.polarity_scores(headline)["compound"]
    if compound >= 0.05:
        return 1.0
    if compound <= -0.05:
        return -1.0
    return 0.0


# ── Fetch headlines ───────────────────────────────────────────────────────────

def _headline_hash(headline: str) -> str:
    return hashlib.md5(headline.encode()).hexdigest()[:12]


def _fetch_recent_headlines(ticker: str, max_age_h: int = _MAX_AGE_HOURS) -> list[dict]:
    """Return list of {headline, published_date} from the last max_age_h hours."""
    try:
        import yfinance as yf
        news = yf.Ticker(ticker).news or []
    except Exception:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_h)
    results = []
    for item in news[:_MAX_HEADLINES * 2]:
        ts = item.get("providerPublishTime") or item.get("content", {}).get("pubDate")
        if not ts:
            continue
        try:
            pub = (datetime.fromtimestamp(int(ts), tz=timezone.utc)
                   if isinstance(ts, (int, float))
                   else datetime.fromisoformat(str(ts).replace("Z", "+00:00")))
        except Exception:
            continue
        if pub < cutoff:
            continue
        title = (item.get("title") or item.get("content", {}).get("title") or "").strip()
        if not title:
            continue
        results.append({"headline": title, "published_date": pub.strftime("%Y-%m-%d")})
        if len(results) >= _MAX_HEADLINES:
            break
    return results


# ── On-demand scorer (called at alert time) ───────────────────────────────────

def score_ticker_now(ticker: str) -> str:
    """Score current headlines for a single ticker at alert time.

    Returns a one-line summary for the alert message, e.g.:
      "📰 2 positive · 1 neutral"
    Returns '' if no headlines, VADER not installed, or any error.
    Also stores new rows in the sentiment parquet for future training.
    """
    analyzer = _get_vader()
    if analyzer is None:
        return ""

    headlines = _fetch_recent_headlines(ticker)
    if not headlines:
        return ""

    # Load existing hashes to avoid re-scoring
    existing_hashes: set[str] = set()
    existing_rows: list[dict] = []
    if _SENTIMENT_PATH.exists():
        try:
            ex = pd.read_parquet(_SENTIMENT_PATH)
            if "hash" in ex.columns:
                existing_hashes = set(ex["hash"].dropna().tolist())
            existing_rows = ex.to_dict("records")
        except Exception:
            pass

    new_rows: list[dict] = []
    scores: list[float] = []
    scored_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    for h in headlines:
        h_hash = _headline_hash(h["headline"])
        if h_hash in existing_hashes:
            continue
        s = _score_headline_vader(h["headline"], analyzer)
        scores.append(s)
        new_rows.append({
            "ticker":         ticker,
            "published_date": h["published_date"],
            "headline":       h["headline"],
            "sentiment":      s,
            "hash":           h_hash,
            "scored_at":      scored_at,
        })
        existing_hashes.add(h_hash)

    if new_rows:
        try:
            all_rows = existing_rows + new_rows
            df = pd.DataFrame(all_rows)
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(_SENTIMENT_PATH, index=False)
        except Exception as e:
            log.warning(f"[sentiment] Failed to save for {ticker}: {e}")

    if not scores:
        return ""

    pos = sum(1 for s in scores if s > 0)
    neg = sum(1 for s in scores if s < 0)
    neu = sum(1 for s in scores if s == 0)
    parts = []
    if pos: parts.append(f"{pos} positive")
    if neg: parts.append(f"{neg} negative")
    if neu: parts.append(f"{neu} neutral")
    return "📰 " + " · ".join(parts) if parts else ""


# ── Training feature helper ───────────────────────────────────────────────────

def get_sentiment_features(
    ticker: str,
    as_of_date: str,
    lookbacks: tuple[int, ...] = (7, 30),
) -> dict[str, float]:
    """Return {news_sentiment_7d, news_sentiment_30d} for (ticker, date).

    Used in dataset builders when enough sentiment data has accumulated.
    Returns 0.0 (neutral) for both if no data exists.
    """
    if not _SENTIMENT_PATH.exists():
        return {f"news_sentiment_{d}d": 0.0 for d in lookbacks}
    try:
        df = pd.read_parquet(_SENTIMENT_PATH)
        df = df[(df["ticker"] == ticker) & df["sentiment"].notna()]
        if df.empty:
            return {f"news_sentiment_{d}d": 0.0 for d in lookbacks}
        df["published_date"] = pd.to_datetime(df["published_date"])
        ref = pd.Timestamp(as_of_date)
        result = {}
        for d in lookbacks:
            window = df[
                (df["published_date"] <= ref) &
                (df["published_date"] >= ref - pd.Timedelta(days=d))
            ]
            result[f"news_sentiment_{d}d"] = (
                round(float(window["sentiment"].mean()), 4) if not window.empty else 0.0
            )
        return result
    except Exception:
        return {f"news_sentiment_{d}d": 0.0 for d in lookbacks}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(score_ticker_now(ticker) or f"No headlines scored for {ticker}")
