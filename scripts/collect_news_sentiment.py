"""
scripts/collect_news_sentiment.py — Recolha diária de sentimento de notícias.

Para cada ticker no universo ML:
  1. Busca headlines recentes via yfinance (últimas 48h)
  2. Filtra já processadas (hash da headline já no parquet)
  3. Classifica com Claude Haiku: POSITIVE(+1) / NEUTRAL(0) / NEGATIVE(-1)
  4. Acumula em /data/news_sentiment.parquet

Feature derivada para treino (após ~6 meses de acumulação):
  news_sentiment_7d  = média dos scores nos últimos 7 dias antes da data alvo
  news_sentiment_30d = média dos scores nos últimos 30 dias

Custo estimado: ~$7/ano com Claude Haiku a 500 tickers × 3 headlines/dia.
Requer env var: ANTHROPIC_API_KEY
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("collect_sentiment")

_DATA_DIR       = Path("/data") if Path("/data").exists() else Path("/tmp")
_SENTIMENT_PATH = _DATA_DIR / "news_sentiment.parquet"
_MAX_HEADLINES  = 3      # per ticker per run (was 5 — reduce API calls)
_MAX_AGE_HOURS  = 48     # only headlines from last 48h
_SLEEP_BETWEEN  = 0.15   # sec between tickers (yfinance rate limit)
_SLEEP_API      = 1.5    # sec between Haiku calls — keeps us at ~40 req/min (tier limit: 50)


# ─────────────────────────────────────────────────────────────────────────────
# LLM scoring
# ─────────────────────────────────────────────────────────────────────────────

def _build_client():
    """Cria cliente Anthropic. Devolve None se API key não configurada."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.warning("[sentiment] ANTHROPIC_API_KEY não configurada — sentimento não será classificado")
        return None
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        log.error("[sentiment] anthropic SDK não instalado — adicionar ao requirements.txt")
        return None


def _score_headline(ticker: str, headline: str, client) -> Optional[float]:
    """Score headline with Claude Haiku. Returns 1.0, 0.0, or -1.0."""
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5,
            messages=[{
                "role": "user",
                "content": (
                    f"Financial headline about {ticker}:\n"
                    f"\"{headline}\"\n\n"
                    f"Short-term stock price impact? Reply with one word only: "
                    f"POSITIVE, NEGATIVE, or NEUTRAL."
                ),
            }],
        )
        time.sleep(_SLEEP_API)  # 1.5s → ~40 req/min, safely under 50 req/min tier limit
        answer = msg.content[0].text.strip().upper()
        if "POSITIVE" in answer:
            return 1.0
        if "NEGATIVE" in answer:
            return -1.0
        return 0.0
    except Exception as e:
        log.debug(f"[sentiment] score failed ({ticker}): {e}")
        time.sleep(_SLEEP_API)  # also sleep on error to back off
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Fetch + dedup
# ─────────────────────────────────────────────────────────────────────────────

def _headline_hash(headline: str) -> str:
    return hashlib.md5(headline.encode()).hexdigest()[:12]


def _fetch_recent_headlines(ticker: str, max_age_h: int = _MAX_AGE_HOURS) -> list[dict]:
    """Devolve lista de {headline, published_date} das últimas max_age_h horas."""
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
            pub = datetime.fromtimestamp(int(ts), tz=timezone.utc) if isinstance(ts, (int, float)) else datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:
            continue
        if pub < cutoff:
            continue
        title = (item.get("title") or item.get("content", {}).get("title") or "").strip()
        if not title:
            continue
        results.append({
            "headline":       title,
            "published_date": pub.strftime("%Y-%m-%d"),
        })
        if len(results) >= _MAX_HEADLINES:
            break
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def collect_sentiment(tickers: Optional[list[str]] = None) -> dict:
    """Recolhe e classifica headlines recentes para todos os tickers.

    Devolve dict com: n_new, n_skipped, n_errors, n_tickers, elapsed_s
    """
    t0 = time.time()

    if tickers is None:
        try:
            from universe import get_ml_universe
            tickers = get_ml_universe()
        except Exception as e:
            log.error(f"[sentiment] Não foi possível carregar universo: {e}")
            return {"n_new": 0, "n_skipped": 0, "n_errors": 1, "n_tickers": 0}

    # Carregar parquet existente para dedup
    existing_hashes: set[str] = set()
    existing_rows: list[dict] = []
    if _SENTIMENT_PATH.exists():
        try:
            ex = pd.read_parquet(_SENTIMENT_PATH)
            if "hash" in ex.columns:
                existing_hashes = set(ex["hash"].dropna().tolist())
            existing_rows = ex.to_dict("records")
        except Exception as e:
            log.warning(f"[sentiment] Erro a ler parquet existente: {e}")

    client = _build_client()
    new_rows: list[dict] = []
    n_skipped = n_errors = 0
    scored_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    for i, ticker in enumerate(tickers):
        headlines = _fetch_recent_headlines(ticker)
        for h in headlines:
            h_hash = _headline_hash(h["headline"])
            if h_hash in existing_hashes:
                n_skipped += 1
                continue

            sentiment: Optional[float] = None
            if client is not None:
                sentiment = _score_headline(ticker, h["headline"], client)
                if sentiment is None:
                    n_errors += 1

            new_rows.append({
                "ticker":         ticker,
                "published_date": h["published_date"],
                "headline":       h["headline"],
                "sentiment":      sentiment,   # None = não classificado
                "hash":           h_hash,
                "scored_at":      scored_at,
            })
            existing_hashes.add(h_hash)

        if (i + 1) % 100 == 0:
            log.info(f"[sentiment] {i+1}/{len(tickers)} tickers | {len(new_rows)} novas headlines")
        time.sleep(_SLEEP_BETWEEN)

    if not new_rows:
        log.info("[sentiment] Sem headlines novas — parquet mantido")
        return {
            "n_new":     0,
            "n_skipped": n_skipped,
            "n_errors":  n_errors,
            "n_tickers": len(tickers),
            "elapsed_s": round(time.time() - t0, 1),
        }

    # Merge e guardar
    all_rows = existing_rows + new_rows
    df = pd.DataFrame(all_rows)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_SENTIMENT_PATH, index=False)

    n_scored = sum(1 for r in new_rows if r["sentiment"] is not None)
    log.info(
        f"[sentiment] {len(new_rows)} novas | {n_scored} classificadas | "
        f"{n_skipped} duplicadas | {len(df)} total | {round(time.time()-t0,1)}s"
    )
    return {
        "n_new":     len(new_rows),
        "n_scored":  n_scored,
        "n_skipped": n_skipped,
        "n_errors":  n_errors,
        "n_tickers": len(tickers),
        "elapsed_s": round(time.time() - t0, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature helper (usado em dataset build para joins por ticker+data)
# ─────────────────────────────────────────────────────────────────────────────

def get_sentiment_features(
    ticker: str,
    as_of_date: str,
    lookbacks: tuple[int, ...] = (7, 30),
) -> dict[str, float]:
    """Devolve {news_sentiment_7d, news_sentiment_30d} para (ticker, data).

    Usado em build_dataset.py quando o parquet de sentimento tiver dados
    suficientes. Se não houver dados, devolve 0.0 (neutro) para ambos.
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
            window = df[(df["published_date"] <= ref) & (df["published_date"] >= ref - pd.Timedelta(days=d))]
            result[f"news_sentiment_{d}d"] = round(float(window["sentiment"].mean()), 4) if not window.empty else 0.0
        return result
    except Exception:
        return {f"news_sentiment_{d}d": 0.0 for d in lookbacks}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    result = collect_sentiment()
    print(result)
