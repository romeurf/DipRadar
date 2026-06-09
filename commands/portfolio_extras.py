"""commands/portfolio_extras.py — /buy, /sell, /ppr, /etf, /liquidez handlers.

Portfolio management commands separated from the main scanner logic.
Uses 'import bot_commands as _bc' lazy-proxy for shared state.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime

import bot_commands as _bc



# /allocate removed in Phase 6 — replaced by /cash which handles all
# allocation logic in a unified, tax-aware, ROI-ranked queue.



def handle_buy(parts: list[str]) -> None:
    if len(parts) < 4:
        _bc._reply(
            "⚠️ Uso: `/buy <TICKER> <PREÇO> <SHARES> [SCORE]`\n"
            "_Exemplo: `/buy CRWD 245.50 3 82`_\n"
            "_Score é opcional (0-100). ETFs não precisam de score._"
        )
        return

    symbol = parts[1].upper().strip()
    try:
        price  = float(parts[2])
        shares = float(parts[3])
    except ValueError:
        _bc._reply("⚠️ Preço e shares têm de ser números.")
        return

    if price <= 0 or shares <= 0:
        _bc._reply("⚠️ Preço e shares têm de ser positivos.")
        return

    score = None
    if len(parts) >= 5:
        try:
            score = int(float(parts[4]))
            if not (0 <= score <= 100):
                _bc._reply("⚠️ Score tem de ser entre 0 e 100.")
                return
        except ValueError:
            _bc._reply("⚠️ Score tem de ser um número inteiro.")
            return

    try:
        from portfolio.portfolio import buy as portfolio_buy, get_liquidity
        result = portfolio_buy(symbol, price, shares, entry_score=score)
    except Exception as e:
        _bc._reply(f"❌ Erro ao registar compra: `{e}`")
        return

    action_str = "📈 *DCA efectuado*" if result["action"] == "avg_down" else "🆕 *Nova posição aberta*"
    pos        = result["position"]
    liq        = result["liquidity"]
    liq_warn   = "\n⚠️ _Liquidez ficou negativa — faz `/liquidez +VALOR` para corrigir._" if result["liq_warning"] else ""

    # Auto-create conviction if not already tracked AND not in deny-list.
    # Rationale: "if you /buy, you have a thesis — let the radar watch it".
    # Skips ETFs/foreign-listed/PPR via _AUTO_TRACK_DENY.
    # If conviction already exists, surface the DCA hint instead of duplicating.
    conviction_tail = ""
    try:
        from core import conviction_radar as cr
        existing_conv = cr.get_by_ticker(symbol)
        if existing_conv:
            conviction_tail = (
                f"\n\n🧠 _Convicção {symbol} já activa "
                f"({existing_conv.position_type} @ ${existing_conv.entry_price:.2f}). "
                f"Preço médio mudou para ${pos['avg_price']:.2f} — se quiseres "
                f"re-ancorar a tese ao novo preço médio:_\n"
                f"  `/conviction close {symbol} {price:.2f} dca`\n"
                f"  `/conviction add {symbol} {pos['avg_price']:.2f} {pos['shares']} {existing_conv.position_type}`"
            )
        else:
            deny = cr._get_deny_list()
            if symbol in deny or "." in symbol:
                conviction_tail = (
                    f"\n\n🧠 _Carteira actualizada. {symbol} é ETF/foreign-listed — "
                    f"não auto-criada como convicção (deny-list)._"
                )
            else:
                # Auto-classify and create
                try:
                    current = cr._fetch_price(symbol, fallback=price)
                    hist    = cr._fetch_price_history(symbol, period="6mo")
                    auto_type = cr._classify_type_from_signals(symbol, current, hist)
                    cr.add_conviction(
                        symbol, pos["avg_price"], pos["shares"], auto_type,
                        trailing_high=current if auto_type == "MOMENTUM" else None,
                        note="auto-created from /buy",
                    )
                    conviction_tail = (
                        f"\n\n🧠 _Convicção {symbol} criada auto como **{auto_type}** "
                        f"(classificação por ML + technicals). "
                        f"`/conviction transition {symbol} CORE` se queres long-term hold._"
                    )
                except Exception as e:
                    conviction_tail = (
                        f"\n\n🧠 _Convicção {symbol} não criada (auto-classifier falhou: {e}). "
                        f"Faz `/conviction add {symbol} {pos['avg_price']:.2f} {pos['shares']} TIPO` manual._"
                    )
    except Exception as e:
        logging.debug(f"[buy→conviction auto-create] {symbol}: {e}")

    _bc._reply(
        f"✅ {action_str} — *{symbol}*\n\n"
        f"  💰 Compra: *{shares}x* @ *${price:.2f}* = *${result['cost']:.2f}*\n"
        f"  📊 Preço médio: *${pos['avg_price']:.2f}* | Total: *{pos['shares']}x*\n"
        f"  🏷️ Categoria: _{pos.get('category', 'N/D')}_"
        + (f"\n  🎯 Score entrada: *{score}/100*" if score is not None else "")
        + f"\n\n  💵 Liquidez restante: *€{liq:.2f}*"
        + liq_warn
        + conviction_tail
    )


# ── /sell handler ─────────────────────────────────────────────────────────────────

def handle_sell(parts: list[str]) -> None:
    if len(parts) < 3:
        _bc._reply(
            "⚠️ Uso: `/sell <TICKER> <PREÇO> [SHARES]`\n"
            "_Omite SHARES para fechar toda a posição._\n"
            "_Exemplo: `/sell CRWD 280.00 2`_"
        )
        return

    symbol = parts[1].upper().strip()
    try:
        price = float(parts[2])
    except ValueError:
        _bc._reply("⚠️ Preço tem de ser um número.")
        return

    if price <= 0:
        _bc._reply("⚠️ Preço tem de ser positivo.")
        return

    shares = None
    if len(parts) >= 4:
        try:
            shares = float(parts[3])
            if shares <= 0:
                _bc._reply("⚠️ Shares têm de ser positivas.")
                return
        except ValueError:
            _bc._reply("⚠️ Shares têm de ser um número.")
            return

    try:
        from portfolio.portfolio import sell as portfolio_sell
        result = portfolio_sell(symbol, price, shares)
    except Exception as e:
        _bc._reply(f"❌ Erro ao registar venda: `{e}`")
        return

    if result is None:
        _bc._reply(f"⚠️ Posição *{symbol}* não encontrada na carteira.\n_Usa `/portfolio` para ver as posições activas._")
        return

    pnl    = result["pnl"]
    pct    = result["pnl_pct"]
    em     = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
    action = "🏁 *Posição fechada*" if result["action"] == "closed" else "✂️ *Venda parcial*"

    remaining_str = ""
    if result["action"] == "partial":
        remaining_str = f"\n  📦 Restam: *{result['remaining']}x*"

    # Auto-sync conviction tracker: full close → auto-close conviction;
    # partial → leave a hint for the user.
    conviction_tail = ""
    try:
        from core import conviction_radar as cr
        existing_conv = cr.get_by_ticker(symbol)
        if existing_conv:
            if result["action"] == "closed":
                outcome = cr.close_conviction(symbol, price, reason="sold-via-/sell")
                if outcome:
                    conviction_tail = (
                        f"\n\n🧠 _Convicção {symbol} fechada automaticamente "
                        f"({outcome['pnl_pct']:+.1f}% / {outcome['days_held']}d). "
                        f"Gravado em conviction_outcomes.json._"
                    )
            else:
                conviction_tail = (
                    f"\n\n🧠 _Convicção {symbol} continua activa (venda parcial). "
                    f"Usa `/conviction close {symbol} {price:.2f}` se também quiseres "
                    f"encerrar a tese._"
                )
    except Exception as e:
        logging.debug(f"[sell→conviction sync] {symbol}: {e}")

    _bc._reply(
        f"{em} {action} — *{symbol}*\n\n"
        f"  💰 Vendido: *{result['shares_sold']}x* @ *${price:.2f}* = *${result['proceeds']:.2f}*\n"
        f"  {'📈' if pnl >= 0 else '📉'} P&L: *${pnl:+.2f}* ({pct:+.1f}%)"
        + remaining_str
        + f"\n\n  💵 Liquidez: *€{result['liquidity']:.2f}*"
        + conviction_tail
    )


# ── /conviction handler ──────────────────────────────────────────────────────────
#
# Parallel tracker for positions opened by personal conviction (outside the bot's
# alert flow). Uses the same DIP/Momentum/Core analysis engine but writes to
# personal_positions.json + conviction_outcomes.json — files NEVER touched by
# any ML training script. See conviction_radar.py for the isolation contract.

# /conviction handler moved to commands/conviction.py (Phase 8.3)
from commands.conviction import handle_conviction as _handle_conviction_impl


def handle_conviction(parts: list[str]) -> None:
    _handle_conviction_impl(parts)


# ── /ppr handler ─────────────────────────────────────────────────────────────────
#
# PPR (Plano Poupança Reforma) — Telegram-managed, replaces the deprecated
# PPR_SHARES / PPR_AVG_COST env vars which caused a €19B snapshot bug when
# set wrong. State lives in /data/ppr.json (state.get_ppr / set_ppr).

def handle_ppr(parts: list[str]) -> None:
    from state import get_ppr, set_ppr, set_ppr_nav, clear_ppr

    sub = parts[1].lower() if len(parts) > 1 else "show"

    if sub in ("show", ""):
        s = get_ppr()
        shares = s.get("shares", 0)
        avg    = s.get("avg_cost", 0)
        nav    = s.get("current_nav")
        nav_dt = s.get("nav_updated")
        if shares <= 0 or avg <= 0:
            _bc._reply(
                "💰 *PPR* — sem valores configurados.\n\n"
                "*Setup inicial (custo base):*\n"
                "  `/ppr set SHARES PREÇO_MÉDIO`\n"
                "  _Ex: `/ppr set 100 50.25` → 100 unidades a €50.25_\n\n"
                "*Update cotação (NAV actual do banco):*\n"
                "  `/ppr nav NOVO_NAV`\n"
                "  _Ex: `/ppr nav 52.80` → cotação actual €52.80/unidade_\n\n"
                "_Sem NAV: valor = custo (não há live tracking automático para PPRs)._\n"
                "_Para limpar: `/ppr clear`_"
            )
            return
        cost = shares * avg
        if nav and nav > 0:
            value = shares * nav
            pnl   = value - cost
            pnl_pct = (nav / avg - 1) * 100 if avg > 0 else 0
            arrow = "📈" if pnl >= 0 else "📉"
            stale_note = ""
            if nav_dt:
                from datetime import date as _date
                try:
                    age = (_date.today() - _date.fromisoformat(nav_dt)).days
                    if age > 30:
                        stale_note = f" ⚠️ NAV com {age} dias (desactualizada — corre `/ppr nav` se possível)"
                    else:
                        stale_note = f" _(NAV de há {age} dias)_"
                except Exception:
                    pass
            _bc._reply(
                f"💰 *PPR* — configurada com NAV live\n\n"
                f"  Unidades: *{shares:g}*\n"
                f"  Custo médio: *€{avg:.4f}*\n"
                f"  NAV actual: *€{nav:.4f}*{stale_note}\n"
                f"  Custo total: *€{cost:,.2f}*\n"
                f"  Valor actual: *€{value:,.2f}*\n"
                f"  {arrow} P&L: *€{pnl:+,.2f}* ({pnl_pct:+.2f}%)\n\n"
                f"_Update NAV: `/ppr nav NOVO_VALOR`_"
            )
        else:
            _bc._reply(
                f"💰 *PPR* — sem NAV (valor frozen no custo)\n\n"
                f"  Unidades: *{shares:g}*\n"
                f"  Preço médio: *€{avg:.4f}*\n"
                f"  Valor (= custo): *€{cost:,.2f}*\n\n"
                f"⚠️ _Sem NAV definida, o snapshot trata a PPR como sem variação._\n"
                f"_Para tracking de P&L: lê a cotação actual no teu banco e corre:_\n"
                f"  `/ppr nav NOVO_NAV`"
            )
        return

    if sub == "set":
        if len(parts) < 4:
            _bc._reply("⚠️ Uso: `/ppr set SHARES PREÇO_MÉDIO`\n_Exemplo: `/ppr set 100 50.25`_")
            return
        try:
            shares = float(parts[2])
            avg    = float(parts[3])
        except ValueError:
            _bc._reply("⚠️ Shares e preço têm de ser números.")
            return
        if shares < 0 or avg < 0:
            _bc._reply("⚠️ Shares e preço não podem ser negativos.")
            return
        value = shares * avg
        if value > 1_000_000:
            _bc._reply(
                f"⚠️ *Valor suspeito:* {shares} × €{avg:.4f} = *€{value:,.2f}*\n\n"
                f"_Mais de €1M — provável typo. Confirma com `--force` se for mesmo:_\n"
                f"  `/ppr set {parts[2]} {parts[3]} --force`"
            )
            if "--force" not in parts:
                return
        set_ppr(shares, avg)
        existing_nav = get_ppr().get("current_nav")
        nav_hint = ""
        if not existing_nav:
            nav_hint = (
                f"\n\n_Para ver P&L real, faz `/ppr nav <cotação_actual>` "
                f"quando souberes a NAV do dia (lê no teu banco)._"
            )
        _bc._reply(
            f"✅ *PPR actualizada* (custo base)\n\n"
            f"  Unidades: *{shares:g}*\n"
            f"  Preço médio: *€{avg:.4f}*\n"
            f"  Custo total: *€{value:,.2f}*"
            f"{nav_hint}"
        )
        return

    if sub == "nav":
        if len(parts) < 3:
            _bc._reply(
                "⚠️ Uso: `/ppr nav NOVA_COTAÇÃO`\n\n"
                "_Lê a cotação actual da PPR no teu banco (extracto/app) e "
                "introduz aqui. Snapshot passa a usar este valor para calcular "
                "P&L real até actualizares de novo._\n\n"
                "_Exemplo: `/ppr nav 52.80`_"
            )
            return
        try:
            nav = float(parts[2])
        except ValueError:
            _bc._reply("⚠️ NAV tem de ser um número.")
            return
        if nav <= 0:
            _bc._reply("⚠️ NAV tem de ser positiva.")
            return
        existing = get_ppr()
        shares = existing.get("shares", 0)
        avg    = existing.get("avg_cost", 0)
        if shares <= 0 or avg <= 0:
            _bc._reply("⚠️ Define primeiro o custo base com `/ppr set SHARES PREÇO_MÉDIO`.")
            return
        # Sanity: refuse NAVs absurdly far from avg_cost (likely typo)
        if avg > 0 and (nav > avg * 10 or nav < avg / 10):
            _bc._reply(
                f"⚠️ *NAV suspeita:* €{nav:.4f} vs preço médio €{avg:.4f} "
                f"(diferença >10×). Provável typo. Confirma o valor antes de inserir."
            )
            return
        set_ppr_nav(nav)
        value = shares * nav
        cost  = shares * avg
        pnl   = value - cost
        pnl_pct = (nav / avg - 1) * 100
        arrow = "📈" if pnl >= 0 else "📉"
        _bc._reply(
            f"✅ *NAV da PPR actualizada*\n\n"
            f"  NAV: *€{nav:.4f}* (era €{avg:.4f} na entrada)\n"
            f"  Valor actual: *€{value:,.2f}*\n"
            f"  {arrow} P&L: *€{pnl:+,.2f}* ({pnl_pct:+.2f}%)\n\n"
            f"_Snapshot passa a usar este NAV até nova actualização._\n"
            f"_Recomendado: actualizar pelo menos 1× por mês._"
        )
        return

    if sub == "scrape":
        # /ppr scrape — auto-fetch NAV from investing.com using stored ISIN
        from state import set_ppr_nav
        existing = get_ppr()
        if not existing or existing.get("shares", 0) <= 0:
            _bc._reply("⚠️ Define primeiro o custo base com `/ppr set SHARES PREÇO_MÉDIO`.")
            return
        isin = existing.get("isin", "PTARMJHM0003")
        _bc._reply(f"_🔍 A consultar cotação de `{isin}` no investing.com..._")
        try:
            import ppr_scraper
            result = ppr_scraper.fetch_ppr_nav(isin)
        except Exception as e:
            _bc._reply(f"❌ Erro no scraper: `{e}`")
            return
        if not result or not result.get("nav"):
            _bc._reply(
                f"⚠️ Não foi possível obter NAV de `{isin}` via investing.com.\n\n"
                f"_Causa provável: alteração na estrutura da página ou rate limit._\n"
                f"_Faz `/ppr nav <valor>` manual com a cotação do banco._"
            )
            return
        nav = result["nav"]
        avg = existing.get("avg_cost", 0)
        if avg > 0 and (nav > avg * 10 or nav < avg / 10):
            _bc._reply(
                f"⚠️ Cotação obtida (€{nav:.4f}) suspeita vs preço médio "
                f"(€{avg:.4f}). Não aplicada. Verifica manualmente."
            )
            return
        set_ppr_nav(nav, source=result.get("source", "investing.com"))

        shares = existing.get("shares", 0)
        cost   = shares * avg
        value  = shares * nav
        pnl    = value - cost
        pnl_pct = (nav / avg - 1) * 100 if avg > 0 else 0
        arrow = "📈" if pnl >= 0 else "📉"
        extras = []
        if result.get("change_pct") is not None:
            extras.append(f"variação dia: {result['change_pct']:+.2f}%")
        if result.get("status"):
            extras.append(result["status"])
        extras_str = " | ".join(extras) if extras else ""
        _bc._reply(
            f"✅ *NAV actualizada via scraper*\n\n"
            f"  ISIN: `{isin}`\n"
            f"  NAV: *€{nav:.4f}*\n"
            f"  Fonte: _{result.get('source', 'investing.com')}_"
            + (f"\n  Estado: _{extras_str}_" if extras_str else "")
            + f"\n\n"
            f"  Valor actual: *€{value:,.2f}*\n"
            f"  {arrow} P&L: *€{pnl:+,.2f}* ({pnl_pct:+.2f}%)\n\n"
            f"_Auto-scrape diário 22h30 Lisboa._"
        )
        return

    if sub == "isin":
        # /ppr isin <NEW_ISIN> — change which fund the scraper targets
        from state import set_ppr_isin
        if len(parts) < 3:
            existing = get_ppr()
            cur_isin = existing.get("isin", "PTARMJHM0003")
            _bc._reply(
                f"💰 *PPR ISIN actual:* `{cur_isin}`\n\n"
                f"_Para alterar (caso tenhas outro PPR):_\n"
                f"  `/ppr isin NOVO_ISIN`\n\n"
                f"_Exemplo: `/ppr isin PTBPPHM0008`_\n"
                f"_Confirma no investing.com que o fundo aparece em "
                f"pt.investing.com/funds/<ISIN_minusculo>_"
            )
            return
        new_isin = parts[2].upper().strip()
        if len(new_isin) != 12:
            _bc._reply(f"⚠️ ISIN inválido (deve ter 12 caracteres): `{new_isin}`")
            return
        set_ppr_isin(new_isin)
        _bc._reply(
            f"✅ ISIN actualizado: `{new_isin}`\n"
            f"_Próximo `/ppr scrape` vai usar este ISIN._"
        )
        return

    if sub == "clear":
        clear_ppr()
        _bc._reply("✅ *PPR limpa.* Snapshot da carteira já não a inclui.")
        return

    _bc._reply(f"⚠️ Sub-comando `{sub}` desconhecido. Usa: `show`, `set`, `nav`, `scrape`, `isin`, `clear`.")


# ── /etf handler ─────────────────────────────────────────────────────────────────
#
# Passive ETF tracker. Same pattern as /ppr — Telegram-managed state, lives
# outside the active scan / conviction / ML pipeline. Live prices fetched
# via yfinance when ticker resolves (use exchange suffix: SXR8.DE, IS3N.DE,
# CSPX.L, etc.), falls back to avg_cost if the ticker doesn't resolve.

def handle_etf(parts: list[str]) -> None:
    from state import get_etfs, set_etf, remove_etf, clear_etfs
    import yfinance as _yf

    sub = parts[1].lower() if len(parts) > 1 else "show"

    if sub in ("show", "list", ""):
        etfs = get_etfs()
        if not etfs:
            _bc._reply(
                "📊 *ETFs* — nenhum configurado.\n\n"
                "Para adicionar (com sufixo do mercado: `.DE` Xetra, `.L` Londres):\n"
                "  `/etf set SXR8.DE 5 450.50`\n"
                "  `/etf set IS3N.DE 10 32.40`\n\n"
                "_Tickers sem ponto assumem USD (NYSE/Nasdaq)._"
            )
            return
        lines = [f"📊 *ETFs* ({len(etfs)} posições)"]
        total_value = 0.0
        total_cost  = 0.0
        for tkr in sorted(etfs.keys()):
            pos = etfs[tkr]
            shares = pos["shares"]
            avg    = pos["avg_cost"]
            # Live price attempt
            live = None
            try:
                _info = _yf.Ticker(tkr).fast_info
                _p = getattr(_info, "last_price", None)
                if _p and float(_p) > 0:
                    live = float(_p)
            except Exception:
                pass
            cost   = shares * avg
            if live:
                value = shares * live
                pnl   = value - cost
                pnl_pct = (live / avg - 1) * 100 if avg > 0 else 0
                arrow = "📈" if pnl >= 0 else "📉"
                lines.append(
                    f"  {arrow} *{tkr}* | {shares:g} @ €{avg:.2f} → €{live:.2f}"
                    f"  ({pnl:+,.2f} / {pnl_pct:+.1f}%)"
                )
                total_value += value
            else:
                value = cost  # fallback
                lines.append(
                    f"  ⚪ *{tkr}* | {shares:g} @ €{avg:.2f} _(sem live price)_"
                    f"  €{value:,.2f}"
                )
                total_value += value
            total_cost += cost
        total_pnl = total_value - total_cost
        pnl_pct_total = (total_value / total_cost - 1) * 100 if total_cost > 0 else 0
        lines += [
            "",
            f"💼 *Total ETFs:* €{total_value:,.2f}",
            f"   Custo: €{total_cost:,.2f} | P&L: €{total_pnl:+,.2f} ({pnl_pct_total:+.1f}%)",
            "",
            "_Adicionar: `/etf set TICKER SHARES PREÇO`_",
            "_Remover:   `/etf remove TICKER`_",
        ]
        _bc._reply("\n".join(lines))
        return

    if sub == "set":
        if len(parts) < 5:
            _bc._reply(
                "⚠️ Uso: `/etf set TICKER SHARES PREÇO_MÉDIO`\n"
                "_Exemplo: `/etf set SXR8.DE 5 450.50`_\n"
                "_Usa o sufixo do mercado (.DE Xetra, .L Londres) para live prices._"
            )
            return
        ticker = parts[2].upper().strip()
        try:
            shares = float(parts[3])
            avg    = float(parts[4])
        except ValueError:
            _bc._reply("⚠️ Shares e preço têm de ser números.")
            return
        if shares < 0 or avg < 0:
            _bc._reply("⚠️ Shares e preço não podem ser negativos.")
            return
        # Sanity check
        cost = shares * avg
        if cost > 500_000:
            _bc._reply(
                f"⚠️ *Valor suspeito:* {shares} × €{avg:.2f} = *€{cost:,.2f}*\n"
                f"_Mais de €500k num único ETF — confirma os valores._"
            )
            return
        # Try live price for confirmation
        live_str = ""
        try:
            _info = _yf.Ticker(ticker).fast_info
            _p = getattr(_info, "last_price", None)
            if _p and float(_p) > 0:
                live_val = shares * float(_p)
                live_str = f"\n  📡 Preço actual: €{float(_p):.2f} → valor live: *€{live_val:,.2f}*"
            else:
                live_str = f"\n  ⚠️ _yfinance sem dados para `{ticker}` — vai usar custo médio no snapshot._"
        except Exception:
            live_str = f"\n  ⚠️ _yfinance falhou para `{ticker}` — usa fallback custo médio._"

        set_etf(ticker, shares, avg)
        _bc._reply(
            f"✅ *ETF actualizado — {ticker}*\n\n"
            f"  Unidades: *{shares:g}*\n"
            f"  Custo médio: *€{avg:.4f}*\n"
            f"  Custo total: *€{cost:,.2f}*"
            f"{live_str}\n\n"
            f"_Vê todos: `/etf`_"
        )
        return

    if sub == "remove":
        if len(parts) < 3:
            _bc._reply("⚠️ Uso: `/etf remove TICKER`")
            return
        ticker = parts[2].upper().strip()
        if remove_etf(ticker):
            _bc._reply(f"🗑️ *{ticker}* removida do tracker de ETFs.")
        else:
            _bc._reply(f"⚠️ *{ticker}* não está no tracker.")
        return

    if sub == "clear":
        clear_etfs()
        _bc._reply("✅ *Tracker de ETFs limpo.* Snapshot já não os inclui.")
        return

    _bc._reply(f"⚠️ Sub-comando `{sub}` desconhecido. Usa: `show`, `set`, `remove`, `clear`.")


# ── /liquidez handler ─────────────────────────────────────────────────────────────

def handle_liquidez(parts: list[str]) -> None:
    try:
        from portfolio.portfolio import get_liquidity, add_liquidity, set_liquidity
    except Exception as e:
        _bc._reply(f"❌ Módulo portfolio não disponível: `{e}`")
        return

    if len(parts) < 2:
        liq = get_liquidity()
        em  = "🟢" if liq >= 0 else "🔴"
        _bc._reply(
            f"💵 *Liquidez disponível:* {em} *€{liq:.2f}*\n\n"
            "_Comandos:_\n"
            "  `/liquidez +500`  → adicionar €500\n"
            "  `/liquidez -100`  → retirar €100\n"
            "  `/liquidez =1500` → definir saldo exacto"
        )
        return

    raw = parts[1].strip()

    if raw.startswith("="):
        try:
            amount = float(raw[1:])
        except ValueError:
            _bc._reply("⚠️ Valor inválido. Exemplo: `/liquidez =1500`")
            return
        new_liq = set_liquidity(amount)
        _bc._reply(f"✅ *Liquidez definida para €{new_liq:.2f}*")
        return

    if raw.startswith("+") or raw.startswith("-"):
        try:
            amount = float(raw)
        except ValueError:
            _bc._reply("⚠️ Valor inválido. Exemplo: `/liquidez +500` ou `/liquidez -100`")
            return
        new_liq = add_liquidity(amount, note="ajuste manual Telegram")
        old_liq = new_liq - amount
        direction = "adicionado" if amount >= 0 else "retirado"
        em = "🟢" if new_liq >= 0 else "🔴"
        _bc._reply(
            f"✅ *€{abs(amount):.2f} {direction}*\n"
            f"  €{old_liq:.2f} → {em} *€{new_liq:.2f}*"
        )
        return

    try:
        amount  = float(raw)
        new_liq = set_liquidity(amount)
        _bc._reply(f"✅ *Liquidez definida para €{new_liq:.2f}*")
    except ValueError:
        _bc._reply(
            "⚠️ Formato inválido.\n"
            "  `/liquidez`       → ver saldo\n"
            "  `/liquidez +500`  → adicionar\n"
            "  `/liquidez =1500` → definir"
        )


# ── helpers ───────────────────────────────────────────────────────────────────────

def _fetch_eur_usd() -> float:
    """Devolve taxa EUR/USD via yfinance. Fallback para 1.0 em caso de erro."""
    try:
        import yfinance as yf
        info = yf.Ticker("EURUSD=X").fast_info
        rate = float(getattr(info, "last_price", None) or info.get("lastPrice") or 0)
        if rate > 0:
            return rate
    except Exception:
        pass
    return 1.0


# ── /core handlers ────────────────────────────────────────────────────────────────


