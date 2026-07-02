"""core/llm_attribution.py — LLM drop-attribution classifier for triage.

Upgrades the keyword-based event-risk scan in triage_watchlist: instead
of matching "merger" as a substring in a headline title, an LLM reads
the recent headlines and classifies WHY the stock dropped, with a
one-line explanation in Portuguese and the decisive headline as
evidence. This automates the judgment the user applied manually to the
2026-06-29 /cash proposal (MKC score 100 hiding a $44.8B merger) and
kills keyword false-positives (a Datadog "price target cut" headline
in the ESTC feed used to flag ESTC as event-risk).

Categories (fixed enum, enforced by json_schema):
  corporate_event   merger/acquisition/spin-off — the analyzed company
                    stops existing in that form; thesis changes
  deterioration     guidance cut, structural damage, restatement —
                    headlines show real damage to fundamentals
  specific_fear     company-specific fear WITHOUT fundamental damage —
                    the classic fear-dislocation profile (the thesis)
  sector_or_market  fell with sector/macro, not company-specific
  unclear           headlines don't explain the drop

Honest-by-construction (enforced by the caller, triage_watchlist):
  - the classifier can only move CLEAN → BORDERLINE, never EXCLUDE,
    never promote, never predicts alpha — it explains, it doesn't score
  - the triage reason always carries the explanation + headline snippet
    so the human can dismiss a false flag in seconds
  - ANY failure (no API key, SDK missing, API error, refusal, bad JSON)
    → return None → caller runs the keyword scan, byte-identical to the
    pre-LLM behavior

Config (env vars):
  ANTHROPIC_API_KEY    required; absent → classifier disabled, keyword
                       fallback everywhere (bot works without it)
  DIPRADAR_LLM_TRIAGE  "0" to force-disable even with a key present
  DIPRADAR_LLM_MODEL   default "claude-opus-4-8"; e.g. "claude-haiku-4-5"
                       for ~20x cheaper classification

Cost note: ~750 input + ~120 output tokens per name → ~$0.007/name on
Opus 4.8. Watchdog (1x/day) + interactive /cash runs classify only
non-EXCLUDED names with headlines; the 6h result cache means the same
ticker+headlines pair is never billed twice in a day.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("llm_attribution")

_DEFAULT_MODEL = "claude-opus-4-8"

CATEGORIES = (
    "corporate_event",
    "deterioration",
    "specific_fear",
    "sector_or_market",
    "unclear",
)

# Categories that flag BORDERLINE in triage (thesis-changing).
FLAG_CATEGORIES = ("corporate_event", "deterioration")

# Result cache: same ticker + same headline set → same answer. 6h TTL
# bounds cost when /cash, /scan_now and the watchdog run on the same day.
_CACHE_TTL_SEC = 6 * 3600
_result_cache: dict[str, tuple[float, "DropAttribution"]] = {}

# Circuit breaker: one API failure disables the classifier for 10 min so
# a down API costs ONE ~20s timeout, not 20s × every name in the scan
# (the /cash-hangs-8-minutes lesson).
_BREAKER_SEC = 600
_disabled_until: float = 0.0

_client = None  # lazy anthropic.Anthropic singleton


@dataclass
class DropAttribution:
    category: str       # one of CATEGORIES
    explanation: str    # 1 concrete sentence, Portuguese
    key_headline: str   # exact title of the decisive headline, "" if none
    confidence: str     # "high" | "medium" | "low"


_SYSTEM_PROMPT = """\
És um analista de equity que classifica PORQUE é que uma ação de qualidade caiu, \
com base nas headlines recentes. O utilizador é um investidor DCA de longo prazo que \
compra quality compounders em drawdown — o teu papel é distinguir quedas que mudam a \
tese (evento corporativo, deterioração real) de quedas que são medo (a oportunidade).

Classifica numa de 5 categorias:
- corporate_event: fusão, aquisição, takeover, spin-off, venda de divisão relevante — \
a empresa analisada deixa de existir nessa forma; a tese muda independentemente do preço.
- deterioration: corte de guidance, perda estrutural de clientes ou quota, problemas \
contabilísticos, restatement — as headlines mostram dano real aos fundamentais.
- specific_fear: medo específico da empresa (processo judicial, medo regulatório, \
downgrade de analista, competição anunciada) SEM evidência nas headlines de dano aos \
fundamentais — o perfil clássico de dislocation por medo.
- sector_or_market: a queda parece ser do sector ou do mercado, não específica da empresa.
- unclear: as headlines não explicam a queda.

Regras:
- IGNORA headlines que são sobre OUTRA empresa (aparecem no feed por engano — ex.: uma \
headline da Datadog no feed da Elastic). Se todas as headlines relevantes forem sobre \
outras empresas, usa "unclear".
- IGNORA listas de opinião genéricas ("3 stocks to buy now", "best dividend stocks").
- Na dúvida entre categorias, escolhe a MENOS alarmante — o humano revê os borderlines \
e falsos alarmes têm custo. Mas um corporate_event ou deterioration claro NUNCA deve \
passar despercebido.
- explanation: 1 frase em português, concreta, citando o facto relevante (valores, nomes).
- key_headline: copia EXATAMENTE o título da headline decisiva, ou "" se nenhuma."""


_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "explanation": {"type": "string"},
        "key_headline": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["category", "explanation", "key_headline", "confidence"],
    "additionalProperties": False,
}


def is_enabled() -> bool:
    """True when the classifier can run: key present, not force-disabled,
    SDK installed, circuit breaker not tripped."""
    if (os.getenv("DIPRADAR_LLM_TRIAGE") or "").strip() == "0":
        return False
    if not os.getenv("ANTHROPIC_API_KEY"):
        return False
    if time.time() < _disabled_until:
        return False
    try:
        import anthropic  # noqa: F401 — availability probe only
    except ImportError:
        return False
    return True


def classify_drop(
    ticker: str,
    headlines: list,
    *,
    dd_pct: Optional[float] = None,        # e.g. -0.44 (fraction, negative)
    sector_rel_pp: Optional[float] = None,  # e.g. -0.12 (fraction)
    sector: str = "",
) -> Optional[DropAttribution]:
    """Classify why `ticker` dropped, from its recent headlines.

    Returns None on any failure — the caller MUST treat None as "run the
    keyword fallback", never as "no event risk".
    """
    global _disabled_until

    if not headlines:
        return None
    if time.time() < _disabled_until:
        return None

    titles = tuple((getattr(h, "title", "") or "") for h in headlines[:10])
    cache_key = f"{(ticker or '').upper()}|{hash(titles)}"
    cached = _result_cache.get(cache_key)
    if cached and (time.time() - cached[0]) < _CACHE_TTL_SEC:
        return cached[1]

    try:
        raw = _call_api(ticker, headlines, dd_pct, sector_rel_pp, sector)
    except Exception as e:
        # Trip the breaker: one failure → keyword fallback for 10 min,
        # instead of a timeout per remaining name in the scan.
        _disabled_until = time.time() + _BREAKER_SEC
        log.warning(
            f"[llm-attribution] {ticker}: API falhou ({e}) — "
            f"keyword fallback pelos próximos {_BREAKER_SEC // 60} min"
        )
        return None

    result = _parse_response(raw)
    if result is None:
        log.warning(f"[llm-attribution] {ticker}: resposta não parseável — keyword fallback")
        return None
    _result_cache[cache_key] = (time.time(), result)
    log.info(
        f"[llm-attribution] {ticker}: {result.category} "
        f"({result.confidence}) — {result.explanation[:80]}"
    )
    return result


def _get_client():
    global _client
    if _client is None:
        import anthropic
        # 20s timeout + 1 retry: triage runs sequentially over ~5-15
        # names; a slow API must degrade to keywords, not stall /cash.
        _client = anthropic.Anthropic(timeout=20.0, max_retries=1)
    return _client


def _build_user_msg(ticker, headlines, dd_pct, sector_rel_pp, sector) -> str:
    lines = [f"Ticker: {ticker}" + (f" ({sector})" if sector else "")]
    if dd_pct is not None:
        lines.append(f"Drawdown do máximo de 52 semanas: {dd_pct * 100:.0f}%")
    if sector_rel_pp is not None:
        lines.append(f"vs sector (60 dias): {sector_rel_pp * 100:+.0f}pp")
    lines.append("")
    lines.append("Headlines recentes (mais recente primeiro):")
    for i, h in enumerate(headlines[:10], 1):
        title = (getattr(h, "title", "") or "").strip()
        if not title:
            continue
        pub = (getattr(h, "publisher", "") or "").strip()
        date = (getattr(h, "published_at", "") or "")[:10]
        line = f"{i}. "
        if date:
            line += f"[{date}] "
        line += title
        if pub:
            line += f" ({pub})"
        lines.append(line)
    lines.append("")
    lines.append("Porque caiu esta ação? Responde no formato JSON pedido.")
    return "\n".join(lines)


def _call_api(ticker, headlines, dd_pct, sector_rel_pp, sector) -> str:
    client = _get_client()
    response = client.messages.create(
        model=os.getenv("DIPRADAR_LLM_MODEL", _DEFAULT_MODEL),
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        messages=[{
            "role": "user",
            "content": _build_user_msg(ticker, headlines, dd_pct, sector_rel_pp, sector),
        }],
    )
    if response.stop_reason == "refusal":
        return ""
    return next((b.text for b in response.content if b.type == "text"), "")


def _parse_response(raw: str) -> Optional[DropAttribution]:
    """Defensive parse: json_schema output_config guarantees valid JSON on
    success, but refusals/max_tokens/older-SDK paths can produce anything."""
    if not raw or not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    cat = data.get("category")
    if cat not in CATEGORIES:
        return None
    return DropAttribution(
        category=cat,
        explanation=str(data.get("explanation") or "").strip(),
        key_headline=str(data.get("key_headline") or "").strip(),
        confidence=str(data.get("confidence") or "low").strip(),
    )
