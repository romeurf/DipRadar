"""utils/news_headlines.py — Recent news headlines for a ticker via yfinance.

Used by /cash output, /scan_now, /analisar appendix, DIP alerts and the
cash watchdog notification to attach 1-5 line news context to a pick.

Why: the FOX dip in 2026-06-14 was a textbook example — the screener
correctly fired the alert, but the user had to look up the dilution-from-
acquisition story separately. With headlines attached, the same alert
shows "Fox to issue 50M shares for Tubi acquisition" inline → 5-second
decision instead of 30-minute investigation.

Cache: in-memory only (per-process), 15-minute TTL. Most yfinance .news
calls hit local cache after the first fetch for the same ticker in the
same session, so we don't need a persistent disk cache.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

log = logging.getLogger("news")

_CACHE_TTL_SEC = 900  # 15 min
_cache: dict[str, tuple[float, list]] = {}


@dataclass
class Headline:
    title: str
    publisher: str
    published_at: str  # ISO date, or "" if unknown
    url: str

    def short(self, max_len: int = 80) -> str:
        t = self.title or ""
        if len(t) > max_len:
            t = t[: max_len - 1] + "…"
        return t


def fetch_headlines(
    ticker: str,
    max_results: int = 3,
    *,
    as_of_date=None,
) -> list[Headline]:
    """Fetch most recent headlines for a ticker. Empty list on failure.

    yfinance .news returns a list of dicts with keys: title, publisher,
    providerPublishTime (unix), link, type. We normalize, sort by date
    desc, optionally filter to `published_at <= as_of_date`, then trim
    to `max_results`.

    `as_of_date` (date | None): when set, headlines published AFTER
    this date are excluded. Used by the backtest harness for PIT
    correctness — without this filter, a backtest at as_of=2024-03-15
    would otherwise see headlines from 2024-03-20 as if they were
    knowable on the as_of date, which is look-ahead. In production
    callers pass None (or omit) and get all available headlines.

    Cache stores the RAW unfiltered list; the as_of_date filter is
    applied per-call so one cache entry serves both production and
    backtest queries for the same ticker.
    """
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return []

    now = time.time()
    cached = _cache.get(ticker)
    if cached and (now - cached[0]) < _CACHE_TTL_SEC:
        return _filter_and_trim(cached[1], as_of_date, max_results)

    try:
        import yfinance as yf
        raw = yf.Ticker(ticker).news or []
    except Exception as e:
        log.debug(f"[news] yfinance.news({ticker}) failed: {e}")
        _cache[ticker] = (now, [])
        return []

    parsed: list[Headline] = []
    for item in raw:
        try:
            # yfinance schema drifts: some versions return flat dicts,
            # newer versions wrap each item under "content" with nested keys.
            content = item.get("content") if isinstance(item, dict) else None
            if isinstance(content, dict):
                title = content.get("title") or ""
                publisher = (
                    (content.get("provider") or {}).get("displayName")
                    or content.get("publisher")
                    or ""
                )
                pub_time = (
                    content.get("pubDate")
                    or content.get("displayTime")
                    or ""
                )
                url = (
                    (content.get("canonicalUrl") or {}).get("url")
                    or (content.get("clickThroughUrl") or {}).get("url")
                    or content.get("link")
                    or ""
                )
            else:
                title = item.get("title") or ""
                publisher = item.get("publisher") or ""
                pub_unix = item.get("providerPublishTime")
                if pub_unix:
                    from datetime import datetime
                    pub_time = datetime.fromtimestamp(int(pub_unix)).isoformat(
                        timespec="seconds"
                    )
                else:
                    pub_time = ""
                url = item.get("link") or ""

            if not title:
                continue
            parsed.append(Headline(
                title=str(title).strip(),
                publisher=str(publisher).strip(),
                published_at=str(pub_time).strip(),
                url=str(url).strip(),
            ))
        except Exception:
            continue

    # Sort descending by published_at (lexicographic on ISO works) — newest first
    parsed.sort(key=lambda h: h.published_at, reverse=True)
    _cache[ticker] = (now, parsed)
    return _filter_and_trim(parsed, as_of_date, max_results)


def _filter_and_trim(headlines: list, as_of_date, max_results: int) -> list:
    """Apply optional as_of_date filter then trim to max_results.

    `as_of_date` semantics: only return headlines whose published_at
    date (parsed from ISO prefix YYYY-MM-DD) is <= as_of_date. None
    means no filter (production default). When a headline has no
    parseable date AND a filter is active, the headline is dropped
    (safer to omit than risk including a future headline by accident).
    """
    if as_of_date is None:
        return headlines[:max_results]
    from datetime import date as _date
    out = []
    for h in headlines:
        date_str = (h.published_at or "")[:10]
        if not date_str:
            continue  # safer drop when filter active
        try:
            pub_date = _date.fromisoformat(date_str)
        except ValueError:
            continue
        if pub_date <= as_of_date:
            out.append(h)
        if len(out) >= max_results:
            break
    return out


def format_headlines_for_telegram(
    headlines: list[Headline],
    max_results: int = 3,
    indent: str = "  ",
) -> str:
    """Render headlines as compact Telegram-friendly Markdown lines.

    Format per line: "  • YYYY-MM-DD  Title (publisher)"
    Returns "" when headlines empty so callers can conditionally append.
    """
    if not headlines:
        return ""
    lines = []
    for h in headlines[:max_results]:
        date_str = (h.published_at or "")[:10]
        line = f"{indent}• "
        if date_str:
            line += f"`{date_str}`  "
        line += h.short(80)
        if h.publisher:
            line += f"  _({h.publisher})_"
        lines.append(line)
    return "\n".join(lines)


def headlines_with_fallback_url(ticker: str, max_results: int = 3) -> str:
    """Convenience: fetch + format, with Yahoo Finance fallback when empty.

    When yfinance returns no news (obscure tickers, rate limit), embed a
    link to the Yahoo news page so the user can click through manually.
    """
    headlines = fetch_headlines(ticker, max_results=max_results)
    if headlines:
        return format_headlines_for_telegram(headlines, max_results=max_results)
    return (
        f"  _Sem headlines via yfinance — "
        f"[ver no Yahoo](https://finance.yahoo.com/quote/{ticker}/news)_"
    )
