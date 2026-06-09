"""commands/core_cmds.py — /core handlers: tracking + universe management.

Phase 8.6 of the bot_commands.py split. Uses `import bot_commands as _bc`.
The 'core' here refers to the CORE conviction bucket (hold-for-life
compounders), not to be confused with the core/ package.
"""
from __future__ import annotations

import logging
import threading

import bot_commands as _bc



def handle_core(parts: list[str]) -> None:
    """/core — mostra todas as posições CORE com P&L e yield acumulado."""
    try:
        from portfolio.paper_trading import format_core_holdings_report
        _bc._reply(format_core_holdings_report())
    except Exception as e:
        _bc._reply(f"❌ Erro ao ler posições CORE: `{e}`")




def handle_core_add(parts: list[str]) -> None:
    """/core_add TICKER PRICE AMOUNT_EUR [--yield YLD] [--note "texto"]

    Exemplos:
      /core_add MSFT 359 2000
      /core_add NVO 38 1500 --yield 5.0 --note "alta yield, comprado no dip"
    """
    if len(parts) < 4:
        _bc._reply(
            "*Uso:* `/core_add TICKER PRICE AMOUNT_EUR [--yield YLD%] [--note texto]`\n\n"
            "Exemplo: `/core_add MSFT 359 2000`\n"
            "Exemplo: `/core_add NVO 38 1500 --yield 5.0 --note \"comprado no dip\"`"
        )
        return

    try:
        ticker     = parts[1].upper().strip()
        open_price = float(parts[2])
        amount_eur = float(parts[3])
    except (ValueError, IndexError):
        _bc._reply("❌ Preço e montante têm de ser números. Ex: `/core_add MSFT 359 2000`")
        return

    # Flags opcionais
    annual_yield = 0.0
    note = ""
    i = 4
    while i < len(parts):
        if parts[i] == "--yield" and i + 1 < len(parts):
            try:
                annual_yield = float(parts[i + 1])
                i += 2
            except ValueError:
                i += 1
        elif parts[i] == "--note" and i + 1 < len(parts):
            note = " ".join(parts[i + 1:])
            break
        else:
            i += 1

    try:
        from portfolio.paper_trading import record_core_holding
        trade_id = record_core_holding(
            ticker=ticker, open_price=open_price, amount_eur=amount_eur,
            annual_yield_pct=annual_yield, note=note,
            model_id="core_us", sector="",
        )
        if trade_id:
            yield_str = f" | Yield: {annual_yield:.1f}%/ano" if annual_yield > 0 else ""
            note_str  = f"\n_{note}_" if note else ""
            _bc._reply(
                f"✅ *CORE {ticker}* registado `[{trade_id}]`\n"
                f"  €{amount_eur:.0f} @ ${open_price:.2f}{yield_str}{note_str}\n"
                f"_Posição de longo prazo — sem gestão automática._"
            )
        else:
            _bc._reply("❌ Não foi possível registar a posição CORE.")
    except Exception as e:
        _bc._reply(f"❌ Erro: `{e}`")




def handle_core_close(parts: list[str]) -> None:
    """/core_close TICKER PRICE — fecha manualmente uma posição CORE."""
    if len(parts) < 3:
        _bc._reply("*Uso:* `/core_close TICKER PRICE`\nEx: `/core_close MSFT 420`")
        return
    try:
        ticker      = parts[1].upper().strip()
        close_price = float(parts[2])
    except (ValueError, IndexError):
        _bc._reply("❌ Preço inválido.")
        return
    try:
        from portfolio.paper_trading import close_core_holding
        closed_trade = close_core_holding(ticker, close_price)
        if closed_trade:
            ret_s   = f"{closed_trade.get('return_pct', 0):+.1f}%"
            alpha_s = f"{closed_trade.get('alpha_pct', 0):+.1f}%" if closed_trade.get("alpha_pct") is not None else "N/A"
            _bc._reply(
                f"✅ <b>CORE {ticker}</b> closed manually @ ${close_price:.2f}\n"
                f"Return: {ret_s}  Alpha vs SPY: {alpha_s}"
            )
            # Generate and send trade card (CORE cards include yield income line)
            try:
                from utils.chart_generator import generate_trade_card
                card = generate_trade_card(closed_trade)
                if card and card.exists():
                    _send_photo(card.read_bytes(), caption=f"CORE {ticker} — trade card")
            except Exception as _ce:
                logging.debug(f"[trade_card] core close card: {_ce}")
        else:
            _bc._reply(f"⚠️ No open CORE position found for {ticker}.")
    except Exception as e:
        _bc._reply(f"❌ Error: <code>{e}</code>")




# ── /core_universe handlers ───────────────────────────────────────────────────────


def handle_core_universe(parts: list[str]) -> None:
    """/core_universe — lista o universo CoreRadar."""
    try:
        from core.core_radar import get_universe
        entries = get_universe()
        if not entries:
            _bc._reply(
                "_Universo CORE vazio._\n\n"
                "Adiciona tickers com:\n"
                "`/core_universe_add TICKER [yield%] [nota]`\n\n"
                "Exemplo: `/core_universe_add MSFT 0.7 compounder secular`"
            )
            return
        lines = [f"💎 *Universo CoreRadar — {len(entries)} tickers*", ""]
        for e in entries:
            ticker   = e["ticker"]
            yld      = e.get("annual_yield_pct", 0)
            note     = e.get("note", "")
            yld_str  = f" | *{yld:.1f}%/ano*" if yld > 0 else ""
            note_str = f" — _{note}_" if note else ""
            lines.append(f"  • *{ticker}*{yld_str}{note_str}")
        lines += [
            "",
            "_/core_universe_add TICK [yield] [nota] — adicionar_",
            "_/core_universe_remove TICK — remover_",
        ]
        _bc._reply("\n".join(lines))
    except Exception as e:
        _bc._reply(f"❌ Erro: `{e}`")




def handle_core_universe_add(parts: list[str]) -> None:
    """/core_universe_add TICKER [yield_pct] [nota]"""
    if len(parts) < 2:
        _bc._reply(
            "*Uso:* `/core_universe_add TICKER [yield%] [nota]`\n\n"
            "Exemplos:\n"
            "`/core_universe_add MSFT 0.7 compounder secular cloud+AI`\n"
            "`/core_universe_add ABBV 3.8 pharma high-yield Humira+Skyrizi`"
        )
        return
    ticker       = parts[1].upper().strip()
    annual_yield = 0.0
    note         = ""
    if len(parts) >= 3:
        try:
            annual_yield = float(parts[2])
            note = " ".join(parts[3:]) if len(parts) > 3 else ""
        except ValueError:
            note = " ".join(parts[2:])
    try:
        from core.core_radar import add_to_universe
        added = add_to_universe(ticker, annual_yield_pct=annual_yield, note=note)
        if added:
            yld_str  = f" | Yield: {annual_yield:.1f}%/ano" if annual_yield > 0 else ""
            note_str = f"\n_{note}_" if note else ""
            _bc._reply(
                f"✅ *{ticker}* adicionado ao universo CoreRadar{yld_str}{note_str}\n"
                f"_Será escaneado no próximo CoreRadar scan (22h Lisboa)._"
            )
        else:
            _bc._reply(f"⚠️ *{ticker}* já está no universo CoreRadar.")
    except Exception as e:
        _bc._reply(f"❌ Erro: `{e}`")




def handle_core_universe_remove(parts: list[str]) -> None:
    """/core_universe_remove TICKER"""
    if len(parts) < 2:
        _bc._reply("*Uso:* `/core_universe_remove TICKER`\nEx: `/core_universe_remove NVO`")
        return
    ticker = parts[1].upper().strip()
    try:
        from core.core_radar import remove_from_universe
        removed = remove_from_universe(ticker)
        if removed:
            _bc._reply(f"✅ *{ticker}* removido do universo CoreRadar.")
        else:
            _bc._reply(f"⚠️ *{ticker}* não estava no universo CoreRadar.")
    except Exception as e:
        _bc._reply(f"❌ Erro: `{e}`")


