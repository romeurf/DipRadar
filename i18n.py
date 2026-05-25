"""
i18n.py — Per-chat language preferences and string translations for DipRadar.

Public API:
    t(key, lang, **kwargs)           — translate key into lang, interpolating kwargs
    set_language(chat_id, lang)      — store language preference for a chat
    get_language(chat_id)            — retrieve stored language (default: "pt")
"""
from __future__ import annotations

from config import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES

_user_prefs: dict[int, str] = {}

STRINGS: dict[str, dict[str, str]] = {
    # ── Language commands ─────────────────────────────────────────────────────
    "language_set": {
        "pt": "✅ Idioma definido para *Português*.",
        "en": "✅ Language set to *English*.",
    },
    "language_invalid": {
        "pt": "⚠️ Idioma não suportado: `{lang}`.\nIdiomas disponíveis: {available}",
        "en": "⚠️ Unsupported language: `{lang}`.\nAvailable languages: {available}",
    },
    # ── EU Phase 2 stub ───────────────────────────────────────────────────────
    "not_implemented_eu": {
        "pt": "⚠️ Mercados europeus ainda não implementados (Fase 2).",
        "en": "⚠️ European markets not yet implemented (Phase 2).",
    },
    # ── Scan messages ─────────────────────────────────────────────────────────
    "scan_started": {
        "pt": "🔍 Scan iniciado...",
        "en": "🔍 Scan started...",
    },
    "scan_complete": {
        "pt": "✅ Scan concluído: {count} candidato(s) encontrado(s).",
        "en": "✅ Scan complete: {count} candidate(s) found.",
    },
    "scan_no_candidates": {
        "pt": "📭 Sem candidatos encontrados neste scan.",
        "en": "📭 No candidates found in this scan.",
    },
    # ── Portfolio messages ────────────────────────────────────────────────────
    "portfolio_empty": {
        "pt": "📭 Sem posições activas no portfólio.",
        "en": "📭 No active positions in portfolio.",
    },
    "position_opened": {
        "pt": "📥 Posição aberta: *{ticker}* @ ${price:.2f}",
        "en": "📥 Position opened: *{ticker}* @ ${price:.2f}",
    },
    "position_closed": {
        "pt": "📤 Posição fechada: *{ticker}* — P&L: {pnl:+.1f}%",
        "en": "📤 Position closed: *{ticker}* — P&L: {pnl:+.1f}%",
    },
    "stop_hit": {
        "pt": "🛑 Stop atingido: *{ticker}* — saída a ${price:.2f}",
        "en": "🛑 Stop hit: *{ticker}* — exit at ${price:.2f}",
    },
    "target_hit": {
        "pt": "🎯 Target atingido: *{ticker}* — saída a ${price:.2f} (+{pnl:.1f}%)",
        "en": "🎯 Target hit: *{ticker}* — exit at ${price:.2f} (+{pnl:.1f}%)",
    },
    # ── Retrain messages ──────────────────────────────────────────────────────
    "retrain_started": {
        "pt": "🤖 Retreino iniciado. Notificação quando terminar.",
        "en": "🤖 Retrain started. You will be notified when complete.",
    },
    "retrain_promoted": {
        "pt": "🚀 Modelo promovido a produção. ρ_α: {rho:.4f}",
        "en": "🚀 Model promoted to production. ρ_α: {rho:.4f}",
    },
    "retrain_pending": {
        "pt": "⚠️ Modelo pendente — não superou produção. Guardado como `dip_models_pending.pkl`.",
        "en": "⚠️ Model pending — did not beat production. Saved as `dip_models_pending.pkl`.",
    },
    "retrain_failed": {
        "pt": "❌ Retreino falhou: {error}",
        "en": "❌ Retrain failed: {error}",
    },
    # ── Alert messages ────────────────────────────────────────────────────────
    "alert_dip": {
        "pt": "📉 Dip detectado: *{ticker}* ({change:+.1f}%) — Score {score:.0f}/100",
        "en": "📉 Dip detected: *{ticker}* ({change:+.1f}%) — Score {score:.0f}/100",
    },
    "alert_skipped_foreign": {
        "pt": "⏭️ *{ticker}* ignorado — ticker estrangeiro (mercado US apenas).",
        "en": "⏭️ *{ticker}* skipped — foreign ticker (US market only).",
    },
    # ── Error messages ────────────────────────────────────────────────────────
    "error_generic": {
        "pt": "❌ Erro inesperado: {error}",
        "en": "❌ Unexpected error: {error}",
    },
    "error_feature_schema": {
        "pt": (
            "❌ Esquema de features inválido: {missing} colunas em falta no parquet.\n"
            "Re-executa o bootstrap para regenerar o parquet."
        ),
        "en": (
            "❌ Feature schema violation: {missing} columns missing from parquet.\n"
            "Re-run bootstrap to regenerate the parquet."
        ),
    },
}


def t(key: str, lang: str | None = None, **kwargs: object) -> str:
    """Return the translation for key in lang, interpolating kwargs.

    Falls back to DEFAULT_LANGUAGE if lang is None or not in SUPPORTED_LANGUAGES.
    Falls back to the key itself if no translation exists (never raises).
    """
    resolved_lang = lang if lang in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
    translations = STRINGS.get(key)
    if translations is None:
        return key
    text = translations.get(resolved_lang) or translations.get(DEFAULT_LANGUAGE) or key
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass
    return text


def set_language(chat_id: int, lang: str) -> bool:
    """Store language preference for chat_id. Returns True if lang is valid."""
    if lang not in SUPPORTED_LANGUAGES:
        return False
    _user_prefs[chat_id] = lang
    return True


def get_language(chat_id: int) -> str:
    """Return stored language for chat_id, defaulting to DEFAULT_LANGUAGE."""
    return _user_prefs.get(chat_id, DEFAULT_LANGUAGE)
