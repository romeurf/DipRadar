"""commands/themes.py — /themes, /add_theme, /remove_theme handlers.

These manage the trend themes table (themes are stored in scanners/themes.py
via the get_*, add_theme, remove_theme functions). The handlers below are
the Telegram-facing UX layer.

Shared utilities (`_reply`, `_check_rate`) are accessed via lazy import of
bot_commands at function call time to avoid circular imports at module load.
"""
from __future__ import annotations


def _reply(text: str) -> None:
    """Lazy proxy to bot_commands._reply."""
    import bot_commands
    bot_commands._reply(text)


def handle_themes() -> None:
    """Lista todos os temas activos com rationale e bonus de sizing."""
    try:
        from scanners.themes import format_themes_list
        _reply(format_themes_list())
    except Exception as e:
        _reply(f"_Erro ao carregar temas: {e}_")


def handle_add_theme(parts: list[str]) -> None:
    """/add_theme <key> <label (multi-word)> <TICK1,TICK2,...> [confiança 0-1]

    Exemplos:
      /add_theme robotics Robótica IRBT,ABB,FANUY 0.75
      /add_theme biotech2 Biotech Oncologia MRNA,BEAM,CRSP
    """
    if len(parts) < 4:
        _reply(
            "⚠️ Uso: `/add\\_theme <key> <label> <TICK1,TICK2,...> [confiança]`\n\n"
            "Exemplos:\n"
            "`/add_theme robotics Robótica IRBT,ABB,FANUY 0.75`\n"
            "`/add_theme biotech2 Biotech Oncologia MRNA,BEAM,CRSP`"
        )
        return

    key     = parts[1].lower().strip()
    # Último token numérico = confiança (opcional)
    confidence = 0.75
    ticker_idx = 3
    if len(parts) >= 5:
        try:
            conf_candidate = float(parts[-1])
            if 0.0 <= conf_candidate <= 1.0:
                confidence = conf_candidate
                ticker_idx = 3
                # label = todos os tokens entre key e tickers, excluindo o último (confiança)
        except ValueError:
            pass

    label   = parts[2]
    tickers = [t.upper().strip() for t in parts[ticker_idx].split(",") if t.strip()]
    if not tickers:
        _reply("_Nenhum ticker válido. Usa formato: TICK1,TICK2,TICK3_")
        return

    try:
        from scanners.themes import add_theme
        t = add_theme(key, label, tickers, confidence=confidence, added_by="user_bot")
        _reply(
            f"✅ *Tema adicionado:* `{key}`\n"
            f"  Label: {t['label']}\n"
            f"  Tickers: {', '.join(t['tickers'])}\n"
            f"  Confiança: {t['confidence']:.0%}  |  Bonus: ver /themes"
        )
    except Exception as e:
        _reply(f"_Erro ao adicionar tema: {e}_")


def handle_remove_theme(parts: list[str]) -> None:
    """/remove_theme <key>"""
    if len(parts) < 2:
        _reply("⚠️ Uso: `/remove\\_theme <key>`  (ver keys em /themes)")
        return
    key = parts[1].lower().strip()
    try:
        from scanners.themes import remove_theme
        removed = remove_theme(key)
        if removed:
            _reply(f"✅ Tema `{key}` removido.")
        else:
            _reply(f"⚠️ Tema `{key}` não encontrado. Usa /themes para ver os keys activos.")
    except Exception as e:
        _reply(f"_Erro: {e}_")
