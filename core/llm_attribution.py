"""core/llm_attribution.py — LLM drop-attribution classifier for triage.

Upgrades the keyword-based event-risk scan in triage_watchlist: instead
of matching "merger" as a substring in a headline title, an LLM reads
the recent headlines and classifies WHY the stock dropped, with a
one-line explanation in Portuguese and the decisive headline as
evidence. This automates the judgment the user applied manually to the
2026-06-29 /cash proposal (MKC score 100 hiding a $44.8B merger) and
kills keyword false-positives (a Datadog "price target cut" headline
in the ESTC feed used to flag ESTC as event-risk).

Categories (fixed enum, enforced by response schema):
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
  - ANY failure (no API key, API error, refusal/safety block, bad JSON)
    → return None → caller runs the keyword scan, byte-identical to the
    pre-LLM behavior

Providers (auto-detected, free-first):
  gemini     GEMINI_API_KEY — Google AI Studio (aistudio.google.com/apikey).
             FREE tier (~250 req/day on flash — the watchdog + /cash use
             ~15/day). Plain REST via `requests`, no extra dependency.
             Default model: gemini-2.5-flash. PREFERRED when key present.
  anthropic  ANTHROPIC_API_KEY — paid credits; needs `pip install anthropic`
             (deliberately NOT in requirements.txt — the shipped path is
             the free Gemini tier). Default model: claude-opus-4-8.

Config (env vars):
  GEMINI_API_KEY        enables the free Gemini backend (preferred)
  ANTHROPIC_API_KEY     enables the Anthropic backend (paid, optional)
  DIPRADAR_LLM_PROVIDER "gemini" | "anthropic" — force one; default auto
                        (gemini first, then anthropic)
  DIPRADAR_LLM_TRIAGE   "0" to force-disable everything → keyword fallback
  DIPRADAR_LLM_MODEL    override the ACTIVE provider's default model
                        (Gemini: must be a 2.5-family model — 2.0 rejects
                        the thinkingConfig field we send)
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("llm_attribution")

_DEFAULT_MODELS = {
    "gemini":    "gemini-2.5-flash",
    "anthropic": "claude-opus-4-8",
}

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
# bounds API usage when /cash, /scan_now and the watchdog run on the
# same day (also keeps us far inside the Gemini free-tier daily quota).
_CACHE_TTL_SEC = 6 * 3600
_result_cache: dict[str, tuple[float, "DropAttribution"]] = {}

# Circuit breaker: one API failure disables the classifier for 10 min so
# a down API costs ONE ~20s timeout, not 20s × every name in the scan
# (the /cash-hangs-8-minutes lesson).
_BREAKER_SEC = 600
_disabled_until: float = 0.0

_client = None            # lazy anthropic.Anthropic singleton
_logged_provider = False  # one INFO line per process for Railway logs


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
- key_headline: copia EXATAMENTE o título da headline decisiva, ou "" se nenhuma.

Responde APENAS com o JSON pedido."""


# Anthropic structured-outputs schema (JSON Schema draft subset).
_ANTHROPIC_SCHEMA = {
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

# Gemini responseSchema (proto Schema — UPPERCASE type enum names).
_GEMINI_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "category":     {"type": "STRING", "enum": list(CATEGORIES)},
        "explanation":  {"type": "STRING"},
        "key_headline": {"type": "STRING"},
        "confidence":   {"type": "STRING", "enum": ["high", "medium", "low"]},
    },
    "required": ["category", "explanation", "key_headline", "confidence"],
}

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


def _anthropic_sdk_ok() -> bool:
    try:
        import anthropic  # noqa: F401 — availability probe only
        return True
    except ImportError:
        return False


def _active_provider() -> Optional[str]:
    """Which backend can run right now. Free-first: gemini preferred when
    its key is present; DIPRADAR_LLM_PROVIDER forces one (still requires
    the matching key). None → classifier disabled → keyword fallback."""
    forced = (os.getenv("DIPRADAR_LLM_PROVIDER") or "").strip().lower()
    order = [forced] if forced in ("gemini", "anthropic") else ["gemini", "anthropic"]
    for prov in order:
        if prov == "gemini" and os.getenv("GEMINI_API_KEY"):
            return "gemini"
        if prov == "anthropic" and os.getenv("ANTHROPIC_API_KEY") and _anthropic_sdk_ok():
            return "anthropic"
    return None


def _model_for(provider: str) -> str:
    return os.getenv("DIPRADAR_LLM_MODEL") or _DEFAULT_MODELS[provider]


def is_enabled() -> bool:
    """True when the classifier can run: not force-disabled, breaker not
    tripped, and at least one provider has a usable key."""
    if (os.getenv("DIPRADAR_LLM_TRIAGE") or "").strip() == "0":
        return False
    if time.time() < _disabled_until:
        return False
    return _active_provider() is not None


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
    global _disabled_until, _logged_provider

    if not headlines:
        return None
    if time.time() < _disabled_until:
        return None
    provider = _active_provider()
    if provider is None:
        return None

    titles = tuple((getattr(h, "title", "") or "") for h in headlines[:10])
    cache_key = f"{(ticker or '').upper()}|{hash(titles)}"
    cached = _result_cache.get(cache_key)
    if cached and (time.time() - cached[0]) < _CACHE_TTL_SEC:
        return cached[1]

    if not _logged_provider:
        log.info(f"[llm-attribution] provider={provider} model={_model_for(provider)}")
        _logged_provider = True

    try:
        raw = _call_api(provider, ticker, headlines, dd_pct, sector_rel_pp, sector)
    except Exception as e:
        # Trip the breaker: one failure → keyword fallback for 10 min,
        # instead of a timeout per remaining name in the scan.
        _disabled_until = time.time() + _BREAKER_SEC
        log.warning(
            f"[llm-attribution] {ticker}: {provider} falhou ({e}) — "
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


def _call_api(provider, ticker, headlines, dd_pct, sector_rel_pp, sector) -> str:
    if provider == "gemini":
        return _call_gemini(ticker, headlines, dd_pct, sector_rel_pp, sector)
    return _call_anthropic(ticker, headlines, dd_pct, sector_rel_pp, sector)


# ── Gemini backend (free tier, plain REST) ───────────────────────────

def _call_gemini(ticker, headlines, dd_pct, sector_rel_pp, sector) -> str:
    import requests
    model = _model_for("gemini")
    body = {
        "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
        "contents": [{
            "role": "user",
            "parts": [{"text": _build_user_msg(
                ticker, headlines, dd_pct, sector_rel_pp, sector)}],
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": _GEMINI_SCHEMA,
            "maxOutputTokens": 1024,
            "temperature": 0.0,
            # 2.5-flash "thinks" by default and thoughts bill against
            # maxOutputTokens — an unbounded think can eat the whole
            # budget and return empty text. Disable it: classification
            # of 8 headlines doesn't need it. (2.5-family only; 2.0
            # models reject this field → breaker → keyword fallback.)
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    resp = requests.post(
        _GEMINI_URL.format(model=model),
        headers={
            "x-goog-api-key": os.environ["GEMINI_API_KEY"],
            "Content-Type": "application/json",
        },
        json=body,
        timeout=20,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini HTTP {resp.status_code}: {resp.text[:200]}")
    return _extract_gemini_text(resp.json())


def _extract_gemini_text(payload) -> str:
    """Pull the text out of a generateContent response. Defensive: safety
    blocks / empty candidates / schema drift all collapse to "" (→ parse
    None → keyword fallback), never an exception."""
    try:
        cands = payload.get("candidates") or []
        content = (cands[0].get("content") or {}) if cands else {}
        parts = content.get("parts") or []
        return "".join(
            p.get("text", "") for p in parts if isinstance(p, dict)
        ).strip()
    except (AttributeError, IndexError, TypeError):
        return ""


# ── Anthropic backend (paid credits, optional) ───────────────────────

def _get_client():
    global _client
    if _client is None:
        import anthropic
        # 20s timeout + 1 retry: triage runs sequentially over ~5-15
        # names; a slow API must degrade to keywords, not stall /cash.
        _client = anthropic.Anthropic(timeout=20.0, max_retries=1)
    return _client


def _call_anthropic(ticker, headlines, dd_pct, sector_rel_pp, sector) -> str:
    client = _get_client()
    response = client.messages.create(
        model=_model_for("anthropic"),
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": _ANTHROPIC_SCHEMA}},
        messages=[{
            "role": "user",
            "content": _build_user_msg(ticker, headlines, dd_pct, sector_rel_pp, sector),
        }],
    )
    if response.stop_reason == "refusal":
        return ""
    return next((b.text for b in response.content if b.type == "text"), "")


# ── Shared response parsing ──────────────────────────────────────────

def _parse_response(raw: str) -> Optional[DropAttribution]:
    """Defensive parse: both backends enforce a schema server-side, but
    refusals/safety blocks/max_tokens/HTTP oddities can produce anything."""
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
