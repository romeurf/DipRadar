"""
paper_trading.py — Paper trading automático do DipRadar.

Cada alerta COMPRAR cria automaticamente uma posição simulada. O bot
acompanha o resultado real 90 dias depois e reporta mensalmente se
gerou alpha vs SPY — sem intervenção humana.

Filosofia: o bot deve ser o seu próprio investidor. Se as suas recomendações
não batem o mercado em simulação, não devem ser seguidas na realidade.

Persistência: /data/paper_trades.json (Railway Volume)

Campos por posição:
  id          str        — UUID da posição
  ticker      str        — symbol
  open_date   str        — ISO date do alerta
  open_price  float      — preço no dia do alerta
  amount_eur  float      — capital simulado (do allocation engine)
  shares      float      — amount_eur / (open_price * usd_eur)
  sell_target float      — preço alvo sugerido pelo bot
  hold_days   int        — dias de holding máximo
  status      str        — OPEN | CLOSED_TARGET | CLOSED_TIME | CLOSED_STOP
  close_date  str | None — ISO date de fecho
  close_price float | None
  return_pct  float | None — retorno da posição (%)
  spy_return_pct float | None — SPY no mesmo período (%)
  alpha_pct   float | None — return_pct - spy_return_pct
  score_v2    float | None — score V2 no momento do alerta
  ml_win_prob float | None — P(win) ML no momento do alerta
"""

from __future__ import annotations

import json
import logging
import math
import os
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_DATA_DIR   = Path("/data") if Path("/data").exists() else Path("/tmp")
_TRADES_FILE = _DATA_DIR / "paper_trades.json"
_USD_EUR_DEFAULT = 0.92

# ── Rotating paper portfolio ──────────────────────────────────────────────────
# The paper portfolio starts with PAPER_PORTFOLIO_EUR (default €10k) and
# behaves like a real rotating fund:
#   - Each open position draws capital from the pool.
#   - When a position closes, the capital (+ P&L) rotates back into the pool.
#   - New signals can only open if enough available capital remains.
#   - No per-position cap: the allocator decides sizing; the only limit is
#     what's actually available. If the portfolio grows to €12k, you can
#     deploy €12k (or any fraction) on the next trade.
#   - When a signal fires but available < MIN_POSITION_EUR, the trade is
#     recorded as "MISSED_CAPITAL" so the ML can learn from capital constraints.
PAPER_PORTFOLIO_EUR: float = float(os.environ.get("PAPER_PORTFOLIO_EUR", "10000"))
MIN_POSITION_EUR:    float = 100.0   # below €100 a position isn't worth tracking


# ── I/O ───────────────────────────────────────────────────────────────────────

def _load() -> list[dict]:
    if not _TRADES_FILE.exists():
        return []
    try:
        return json.loads(_TRADES_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        log.error(f"[paper] Erro a ler {_TRADES_FILE}: {e}")
        return []


def _save(trades: list[dict]) -> None:
    try:
        _TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _TRADES_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(trades, indent=2, default=str), encoding="utf-8")
        tmp.replace(_TRADES_FILE)
    except Exception as e:
        log.error(f"[paper] Erro a guardar: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fetch_close(ticker: str, target_date: date) -> Optional[float]:
    """Fetch closing price para um ticker numa data específica (± 3 dias)."""
    try:
        import yfinance as yf
        import pandas as pd
        start = (target_date - timedelta(days=5)).isoformat()
        end   = (target_date + timedelta(days=4)).isoformat()
        h = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if h is None or h.empty:
            return None
        idx = pd.DatetimeIndex(h.index)
        if idx.tz is not None:
            h.index = idx.tz_convert(None)
        target_ts = pd.Timestamp(target_date)
        valid = h[h.index <= target_ts + pd.Timedelta(days=3)]
        if valid.empty:
            return None
        return float(valid["Close"].iloc[-1])
    except Exception as e:
        log.debug(f"[paper] fetch_close {ticker} {target_date}: {e}")
        return None


def _fetch_spy_return(open_date: date, close_date: date) -> Optional[float]:
    """SPY return em % entre open_date e close_date."""
    try:
        spy_open  = _fetch_close("SPY", open_date)
        spy_close = _fetch_close("SPY", close_date)
        if spy_open and spy_close and spy_open > 0:
            return round((spy_close / spy_open - 1) * 100, 3)
    except Exception:
        pass
    return None


# ── API pública ───────────────────────────────────────────────────────────────

def _get_monthly_budget() -> float:
    """Orçamento mensal disponível para dip hunting (exclui DCA em ETFs).

    Usa PAPER_BUDGET_EUR se definido explicitamente.
    Caso contrário, usa MONTHLY_BUDGET_EUR menos os ETFs DCA mensais:
      PAPER_BUDGET_EUR = MONTHLY_BUDGET_EUR - ETF_DCA_EUR
    Define PAPER_BUDGET_EUR=850 no Railway se investires €200/mês em ETFs.
    """
    explicit = os.environ.get("PAPER_BUDGET_EUR")
    if explicit:
        return float(explicit)
    total    = float(os.environ["MONTHLY_BUDGET_EUR"])
    etf_dca  = float(os.environ.get("ETF_DCA_EUR", "0"))  # ex: 200 (EUNL + IS3N)
    return max(0.0, total - etf_dca)


def _capital_used_this_month() -> float:
    """Capital allocated this month across all paper positions."""
    trades     = _load()
    this_month = date.today().strftime("%Y-%m")
    return sum(
        t.get("amount_eur", 0)
        for t in trades
        if t.get("open_date", "")[:7] == this_month
    )


# ── Rotating portfolio accounting ─────────────────────────────────────────────

def get_paper_portfolio_value(trades: Optional[list] = None) -> float:
    """Current portfolio value = starting capital + sum of all closed position P&L."""
    if trades is None:
        trades = _load()
    closed_pnl = sum(
        t["amount_eur"] * (t["return_pct"] / 100.0)
        for t in trades
        if t.get("return_pct") is not None and t.get("amount_eur") is not None
    )
    return PAPER_PORTFOLIO_EUR + closed_pnl


def get_paper_allocated(trades: Optional[list] = None) -> float:
    """Capital currently locked in ALL open positions (DIP + MOMENTUM + CORE).

    The €10k paper pool is shared across all three models — a CORE position
    consumes the same capital as a DIP position and reduces what's available
    for the next signal, regardless of strategy.
    """
    if trades is None:
        trades = _load()
    return sum(
        t.get("amount_eur", 0)
        for t in trades
        if t.get("status") == "OPEN" and t.get("trade_type") in ("DIP", "MOMENTUM", "CORE")
    )


def get_paper_available(trades: Optional[list] = None) -> float:
    """Capital available to open new positions (portfolio value minus allocated)."""
    if trades is None:
        trades = _load()
    return max(0.0, get_paper_portfolio_value(trades) - get_paper_allocated(trades))


def get_missed_trades_stats(trades: Optional[list] = None) -> dict:
    """Stats on signals missed due to capital constraints (status=MISSED_CAPITAL).

    Returns counts and total requested EUR to show the cost of being fully deployed.
    Useful for ML feedback: high missed_alpha_est = strategy should be less aggressive.
    """
    if trades is None:
        trades = _load()
    missed = [t for t in trades if t.get("status") == "MISSED_CAPITAL"]
    if not missed:
        return {"n_missed": 0, "total_requested_eur": 0.0, "avg_ml_prob": None}
    total_req  = sum(t.get("requested_eur", 0) for t in missed)
    probs      = [t["ml_win_prob"] for t in missed if t.get("ml_win_prob") is not None]
    m_probs    = [t["ml_score_ensemble"] for t in missed if t.get("ml_score_ensemble") is not None]
    all_probs  = probs + m_probs
    avg_prob   = round(sum(all_probs) / len(all_probs), 3) if all_probs else None
    resolved   = [t for t in missed if t.get("actual_return_90d") is not None]
    avg_missed_alpha = (
        round(sum(t.get("actual_alpha_90d", 0) for t in resolved) / len(resolved), 2)
        if resolved else None
    )
    return {
        "n_missed":            len(missed),
        "n_resolved":          len(resolved),
        "total_requested_eur": round(total_req, 2),
        "avg_ml_prob":         avg_prob,
        "avg_missed_alpha":    avg_missed_alpha,   # mean alpha we left on the table
        "by_model": {
            "dip_us":      sum(1 for t in missed if t.get("model_id") == "dip_us"),
            "momentum_us": sum(1 for t in missed if t.get("model_id") == "momentum_us"),
        },
    }


def update_missed_trade_outcomes(horizon_days: int = 90) -> dict:
    """Resolve actual forward returns for MISSED_CAPITAL trades after horizon_days.

    For each unresolved missed trade older than horizon_days, fetches:
      - actual_return_90d   — what the stock did in the next 90 days
      - spy_return_90d      — what SPY did in the same window
      - actual_alpha_90d    — actual_return_90d - spy_return_90d

    This data lets the model learn: "when I was capital-constrained and skipped
    a high-probability signal, how much alpha did I actually leave on the table?"
    Over many cycles, patterns emerge — e.g. high VIX + high win_prob missed trades
    during crowded portfolios tend to have above-average alpha.

    Returns {"resolved": N, "skipped": N, "errors": N}.
    """
    trades  = _load()
    today   = date.today()
    cutoff  = today - timedelta(days=horizon_days)

    resolved_count = 0
    skipped_count  = 0
    error_count    = 0
    changed        = False

    for t in trades:
        if t.get("status") != "MISSED_CAPITAL":
            continue
        if t.get("actual_return_90d") is not None:
            skipped_count += 1
            continue  # already resolved

        signal_date_str = t.get("signal_date", "")
        if not signal_date_str:
            skipped_count += 1
            continue

        signal_date = date.fromisoformat(signal_date_str)
        if signal_date > cutoff:
            skipped_count += 1
            continue  # too recent — 90-day window not elapsed

        target_date = signal_date + timedelta(days=horizon_days)
        signal_price = float(t.get("signal_price") or 0)
        ticker = t.get("ticker", "")

        if not ticker or signal_price <= 0:
            skipped_count += 1
            continue

        try:
            close_90d = _fetch_close(ticker, target_date)
            if close_90d is None:
                error_count += 1
                continue

            ret_pct = round((close_90d / signal_price - 1) * 100, 3)
            spy_ret = _fetch_spy_return(signal_date, target_date)
            alpha   = round(ret_pct - (spy_ret or 0), 3) if spy_ret is not None else None

            t["actual_return_90d"] = ret_pct
            t["spy_return_90d"]    = spy_ret
            t["actual_alpha_90d"]  = alpha
            t["resolved_date"]     = today.isoformat()
            changed = True
            resolved_count += 1
            log.info(
                f"[missed] Resolved {ticker} (missed {signal_date_str}): "
                f"ret={ret_pct:+.1f}% alpha={alpha:+.1f}pp"
                if alpha is not None else
                f"[missed] Resolved {ticker}: ret={ret_pct:+.1f}%"
            )
        except Exception as exc:
            log.debug(f"[missed] Error resolving {ticker}: {exc}")
            error_count += 1

    if changed:
        _save(trades)

    return {
        "resolved": resolved_count,
        "skipped":  skipped_count,
        "errors":   error_count,
    }


def get_open_trade(ticker: str) -> Optional[dict]:
    """Devolve o primeiro trade OPEN para ticker, ou None."""
    ticker = ticker.upper()
    for t in _load():
        if t.get("ticker") == ticker and t.get("status") == "OPEN":
            return t
    return None


def has_open_momentum(ticker: str) -> bool:
    """True se já existe um trade MOMENTUM aberto para ticker."""
    ticker = ticker.upper()
    return any(
        t.get("ticker") == ticker
        and t.get("status") == "OPEN"
        and t.get("trade_type") == "MOMENTUM"
        for t in _load()
    )


def upgrade_paper_dip_to_momentum(
    ticker:            str,
    current_price:     float,
    trailing_stop_pct: float = 0.12,
    hold_days:         int   = 60,
    momentum_score:    float = 0.0,
    ml_durability:     str   = "MODERADO",
    ml_score_l1:       float = 0.0,
    ml_score_l2:       float = 0.0,
    ml_score_l3:       float = 0.0,
    ml_score_ensemble: float = 0.0,
) -> bool:
    """Converte o paper trade DIP aberto de ticker para modo MOMENTUM puro.

    A partir deste momento o trade é tratado como um alerta MOMENTUM fresco:
    - hold_days reseta para a duração recomendada pela durabilidade ML (não acumula)
    - sell_target zerado — a saída é 100% por trailing stop
    - trailing_high resetado para current_price (pico conhecido no momento do upgrade)
    - Campos ML do MOMENTUM populados tal como record_momentum_paper_buy faria

    update_open_positions ignora o trade (trade_type=MOMENTUM).
    update_momentum_positions passa a geri-lo com re-score diário.

    Devolve True se encontrou e converteu, False se não havia DIP aberto.
    """
    ticker    = ticker.upper()
    today_iso = date.today().isoformat()
    trades    = _load()
    changed   = False
    for t in trades:
        if t.get("ticker") != ticker or t.get("status") != "OPEN":
            continue
        if t.get("trade_type") == "MOMENTUM":
            continue  # já convertido

        t["prev_trade_type"]      = "DIP"
        t["dip_hold_days"]        = t.get("hold_days", 90)     # preservar para auditoria
        t["dip_sell_target"]      = float(t.get("sell_target") or 0.0)  # target DIP original
        t["upgrade_date"]         = today_iso
        t["momentum_start_date"]  = today_iso  # days_held para MOMENTUM conta daqui
        t["trade_type"]           = "MOMENTUM"
        t["sell_target"]          = 0.0
        t["trailing_high"]        = round(current_price, 4)
        t["trailing_stop_pct"]    = round(trailing_stop_pct, 4)
        # hold_days reseta: conta a partir de momentum_start_date
        t["hold_days"]            = hold_days
        # Campos de MOMENTUM (como se fosse alerta novo)
        t["momentum_score"]      = round(momentum_score, 1)
        t["ml_score_l1"]         = round(ml_score_l1, 4)
        t["ml_score_l2"]         = round(ml_score_l2, 4)
        t["ml_score_l3"]         = round(ml_score_l3, 4)
        t["ml_score_ensemble"]   = round(ml_score_ensemble, 4)
        t["ml_durability_open"]  = ml_durability
        t["ml_durability_cur"]   = ml_durability
        t["durability_history"]  = [{
            "date":       today_iso,
            "durability": ml_durability,
            "l1":         round(ml_score_l1, 4),
            "l2":         round(ml_score_l2, 4),
            "l3":         round(ml_score_l3, 4),
        }]
        # Limpar flags de DIP que já não se aplicam
        t.pop("timeout_trailing_date", None)
        t.pop("past_hold_tightened", None)
        changed = True
        log.info(
            f"[paper] UPGRADE {ticker} DIP→MOMENTUM "
            f"dur={ml_durability} stop={trailing_stop_pct:.0%} "
            f"hold={hold_days}d score={momentum_score:.0f}"
        )
        break  # só um trade aberto por ticker
    if changed:
        _save(trades)
    return changed


def record_paper_buy(
    ticker:      str,
    open_price:  float,
    amount_eur:  float,
    sell_target: float,
    hold_days:   int,
    score_v2:    float = 0.0,
    ml_win_prob: float = 0.0,
    usd_eur:     float = _USD_EUR_DEFAULT,
    model_id:    str   = "dip_us",
    sector:      str   = "",
) -> str:
    """Record a simulated DIP position when the bot fires a BUY alert.

    Every signal is always recorded — no budget exhaustion guard.
    Positions are expressed as weight_pct = amount_eur / PAPER_PORTFOLIO_EUR * 100,
    treating the portfolio as a hypothetical €10k (configurable via PAPER_PORTFOLIO_EUR env var).

    Returns the trade ID, or "" if the ticker already has an open DIP position.
    """
    if open_price <= 0:
        return ""

    # Dedup: don't open a second DIP on the same ticker
    if get_open_trade(ticker):
        log.info(f"[paper] SKIP {ticker} — position already open")
        return ""

    # Rotating pool: clamp to whatever is actually available (no per-position % cap)
    trades     = _load()
    port_value = get_paper_portfolio_value(trades)
    available  = get_paper_available(trades)
    requested  = float(amount_eur)
    amount_eur = min(requested, available)

    if amount_eur < MIN_POSITION_EUR:
        # Signal fired but capital is tied up — record as MISSED so ML can learn
        missed = {
            "id":               str(uuid.uuid4())[:8],
            "ticker":           ticker.upper(),
            "trade_type":       "DIP",
            "status":           "MISSED_CAPITAL",
            "signal_date":      date.today().isoformat(),
            "signal_price":     round(open_price, 4),
            "requested_eur":    round(requested, 2),
            "available_eur":    round(available, 2),
            "portfolio_value":  round(port_value, 2),
            "score_v2":         round(score_v2, 2),
            "ml_win_prob":      round(ml_win_prob, 3),
            "model_id":         model_id,
            "sector":           sector,
        }
        trades.append(missed)
        _save(trades)
        log.info(
            f"[paper] MISSED {ticker} — available €{available:.0f} < €{MIN_POSITION_EUR:.0f} "
            f"(requested €{requested:.0f}, portfolio €{port_value:.0f})"
        )
        return ""

    trade_id   = str(uuid.uuid4())[:8]
    shares     = amount_eur / max(open_price * usd_eur, 0.01)
    weight_pct = round(amount_eur / max(port_value, 1.0) * 100, 2)

    trade = {
        "id":            trade_id,
        "ticker":        ticker.upper(),
        "trade_type":    "DIP",
        "open_date":     date.today().isoformat(),
        "open_price":    round(open_price, 4),
        "amount_eur":    round(amount_eur, 2),
        "weight_pct":    weight_pct,
        "shares":        round(shares, 6),
        "sell_target":   round(sell_target, 4),
        "hold_days":     hold_days,
        "status":        "OPEN",
        "close_date":    None,
        "close_price":   None,
        "return_pct":    None,
        "spy_return_pct": None,
        "alpha_pct":     None,
        "score_v2":      round(score_v2, 2),
        "ml_win_prob":   round(ml_win_prob, 3),
        "model_id":      model_id,
        "sector":        sector,
        "trailing_high":           round(open_price, 4),
        "trailing_stop_pct":       0.22,
        "portfolio_value_at_open": round(port_value, 2),
        "available_at_open":       round(available, 2),
        # PIT SPY price stored at open so alpha is not recalculated from historical yfinance later
        "spy_price_open": _fetch_close("SPY", date.today()),
    }
    trades.append(trade)
    _save(trades)
    log.info(
        f"[paper] OPEN {ticker} €{amount_eur:.0f} ({weight_pct:.1f}% of €{port_value:.0f}) "
        f"@ {open_price:.2f} target={sell_target:.2f} hold={hold_days}d score={score_v2:.0f} "
        f"[available €{available:.0f} → €{available - amount_eur:.0f}]"
    )
    return trade_id


def update_open_positions() -> dict:
    """Verifica todas as posições abertas e fecha as que atingiram o target ou timeout.

    Chamado diariamente pelo scheduler. Retorna estatísticas do update.
    """
    trades     = _load()
    today      = date.today()
    updated    = 0
    still_open = 0
    just_closed: list[dict] = []

    for t in trades:
        if t.get("status") != "OPEN":
            continue
        if t.get("trade_type") == "MOMENTUM":
            continue  # geridas por update_momentum_positions
        if t.get("trade_type") == "CORE":
            continue  # nunca fechadas automaticamente

        open_date = date.fromisoformat(t["open_date"])
        # Para posições revertidas de MOMENTUM→DIP, o prazo DIP conta da data de revert.
        _dip_ref_raw = t.get("revert_to_dip_date")
        dip_ref_date = date.fromisoformat(_dip_ref_raw) if _dip_ref_raw else open_date
        days_held    = (today - dip_ref_date).days

        # Ainda dentro do período de holding?
        if days_held < t["hold_days"] and days_held < 5:
            still_open += 1
            continue

        # Fetch preço actual
        current_price = _fetch_close(t["ticker"], today)
        if current_price is None:
            still_open += 1
            continue

        # ── Trailing stop (DIP que expirou ou foi activado manualmente) ──────────
        _t_high = t.get("trailing_high")
        _t_stop = t.get("trailing_stop_pct")
        if _t_high is not None:
            t["trailing_high"] = round(max(float(_t_high), current_price), 4)
        hit_trailing = (
            _t_high is not None and _t_stop is not None
            and current_price < float(_t_high) * (1.0 - float(_t_stop))
        )

        # ── Saída antecipada (Early Alpha Capture) ─────────────────────────────
        ret_actual    = (current_price / t["open_price"] - 1) if t["open_price"] > 0 else 0
        sell_tgt      = float(t.get("sell_target") or 0)
        target_return = (sell_tgt / t["open_price"] - 1) if sell_tgt > 0 and t["open_price"] > 0 else 0
        time_fraction  = days_held / max(t["hold_days"], 1)
        alpha_fraction = ret_actual / max(target_return, 0.001) if target_return > 0 else 0

        early_exit = (
            alpha_fraction >= 0.70
            and time_fraction < 0.50
            and ret_actual > 0.05
        )

        # Decidir se fechar
        hit_target = sell_tgt > 0 and current_price >= sell_tgt
        time_up    = days_held >= t["hold_days"]

        # Prazo expirou sem atingir target: apertar trailing stop em vez de fechar.
        # Fechamos quando o upside se esgota, não pelo calendário.
        if time_up and not hit_target and not hit_trailing:
            if not t.get("timeout_trailing_date"):  # primeira vez que expira
                old_stop = float(t.get("trailing_stop_pct") or 0.22)
                t["trailing_stop_pct"]       = round(min(old_stop, 0.12), 4)
                t["timeout_trailing_date"]   = today.isoformat()
                log.info(
                    f"[paper] TIMEOUT→TRAILING {t['ticker']} "
                    f"stop apertado {old_stop:.0%}→{t['trailing_stop_pct']:.0%} "
                    f"high={t['trailing_high']:.2f}"
                )
                # Recalcular hit_trailing com stop apertado
                hit_trailing = (
                    current_price < float(t["trailing_high"]) * (1.0 - float(t["trailing_stop_pct"]))
                )
                if hit_trailing:
                    pass  # cai para o bloco de fecho abaixo
                else:
                    still_open += 1
                    continue
            else:
                still_open += 1
                continue

        if not hit_target and not hit_trailing and not early_exit:
            still_open += 1
            continue

        # Close position — compute alpha using PIT SPY price stored at open
        ret_pct = round((current_price / t["open_price"] - 1) * 100, 3)
        _spy_open_stored = t.get("spy_price_open")
        _spy_close_live  = _fetch_close("SPY", today)
        if _spy_open_stored and _spy_close_live and _spy_open_stored > 0:
            spy_ret = round((_spy_close_live / _spy_open_stored - 1) * 100, 3)
            t["spy_price_close"] = _spy_close_live
        else:
            spy_ret = _fetch_spy_return(open_date, today)
        alpha = round(ret_pct - (spy_ret or 0), 3) if spy_ret is not None else None

        if hit_target:
            t["status"] = "CLOSED_TARGET"
        elif early_exit:
            t["status"] = "CLOSED_EARLY_ALPHA"
            log.info(
                f"[paper] EARLY EXIT {t['ticker']} — "
                f"{alpha_fraction*100:.0f}% do alpha em {time_fraction*100:.0f}% do tempo"
            )
        elif hit_trailing:
            t["status"] = "CLOSED_TRAILING_STOP"
            log.info(
                f"[paper] TRAILING STOP {t['ticker']} "
                f"high={_t_high:.2f} price={current_price:.2f}"
            )
        t["close_date"]    = today.isoformat()
        t["close_price"]   = round(current_price, 4)
        t["return_pct"]    = ret_pct
        t["spy_return_pct"] = spy_ret
        t["alpha_pct"]     = alpha

        updated += 1
        just_closed.append(t)
        spy_s   = f"{spy_ret:.1f}%" if spy_ret is not None else "N/A"
        alpha_s = f"{alpha:.1f}%" if alpha is not None else "N/A"
        log.info(
            f"[paper] CLOSED {t['ticker']} {t['status']} "
            f"ret={ret_pct:.1f}% spy={spy_s} alpha={alpha_s}"
        )

    _save(trades)
    return {"updated": updated, "still_open": still_open, "total": len(trades), "just_closed": just_closed}


def get_monthly_performance(months_back: int = 3) -> dict:
    """Calcula a performance do paper portfolio vs SPY nos últimos N meses.

    Retorna métricas mensais e totais para o report automático.
    """
    trades   = _load()
    today    = date.today()
    cutoff   = today - timedelta(days=months_back * 30)

    closed = [
        t for t in trades
        if t.get("status", "").startswith("CLOSED")
        and t.get("close_date")
        and date.fromisoformat(t["close_date"]) >= cutoff
        and t.get("return_pct") is not None
    ]
    early_exits = sum(1 for t in closed if t.get("status") == "CLOSED_EARLY_ALPHA")

    if not closed:
        open_count  = sum(1 for t in trades if t.get("status") == "OPEN")
        port_value  = get_paper_portfolio_value(trades)
        allocated   = get_paper_allocated(trades)
        available   = get_paper_available(trades)
        return {
            "n_closed":             0,
            "n_open":               open_count,
            "n_early_exits":        0,
            "total_return":         0.0,
            "spy_return":           0.0,
            "alpha":                0.0,
            "win_rate":             0.0,
            "total_capital":        0.0,
            "portfolio_start":      round(PAPER_PORTFOLIO_EUR, 2),
            "portfolio_value":      round(port_value, 2),
            "portfolio_return_pct": 0.0,
            "allocated_eur":        round(allocated, 2),
            "available_eur":        round(available, 2),
            "best":                 None,
            "worst":                None,
            "monthly":              {},
            "months_back":          months_back,
        }

    # Métricas totais
    rets    = [t["return_pct"] for t in closed]
    spys    = [t["spy_return_pct"] for t in closed if t.get("spy_return_pct") is not None]
    alphas  = [t["alpha_pct"] for t in closed if t.get("alpha_pct") is not None]
    amounts = [t.get("amount_eur", 1) for t in closed]

    # Retorno ponderado pelo capital
    total_capital = sum(amounts)
    weighted_ret  = sum(r * a for r, a in zip(rets, amounts)) / max(total_capital, 1)
    weighted_spy  = sum(s * a for s, a in zip(spys, [t.get("amount_eur", 1) for t in closed if t.get("spy_return_pct") is not None])) / max(total_capital, 1) if spys else 0.0
    avg_alpha     = sum(alphas) / len(alphas) if alphas else 0.0

    wins = [r for r in rets if r > 0]
    win_rate = len(wins) / len(rets)

    best  = max(closed, key=lambda t: t["return_pct"])
    worst = min(closed, key=lambda t: t["return_pct"])

    # Performance por mês
    monthly: dict = {}
    for t in closed:
        month = t["close_date"][:7]  # YYYY-MM
        if month not in monthly:
            monthly[month] = {"n": 0, "ret": 0.0, "spy": 0.0, "capital": 0.0}
        amt = t.get("amount_eur", 1)
        monthly[month]["n"]       += 1
        monthly[month]["ret"]     += t["return_pct"] * amt
        monthly[month]["spy"]     += (t.get("spy_return_pct") or 0) * amt
        monthly[month]["capital"] += amt

    for m in monthly.values():
        cap = max(m["capital"], 1)
        m["ret"] /= cap
        m["spy"] /= cap
        m["alpha"] = round(m["ret"] - m["spy"], 2)
        m["ret"]   = round(m["ret"], 2)
        m["spy"]   = round(m["spy"], 2)

    open_count       = sum(1 for t in trades if t.get("status") == "OPEN")
    total_capital    = sum(t.get("amount_eur", 0) for t in closed)
    port_value       = get_paper_portfolio_value(trades)
    allocated        = get_paper_allocated(trades)
    available        = get_paper_available(trades)
    port_return_pct  = round((port_value - PAPER_PORTFOLIO_EUR) / max(PAPER_PORTFOLIO_EUR, 1) * 100, 2)

    return {
        "n_closed":             len(closed),
        "n_open":               open_count,
        "n_early_exits":        early_exits,
        "total_return":         round(weighted_ret, 2),
        "spy_return":           round(weighted_spy, 2),
        "alpha":                round(avg_alpha, 2),
        "win_rate":             round(win_rate, 3),
        "total_capital":        round(total_capital, 2),
        # Rotating portfolio stats
        "portfolio_start":      round(PAPER_PORTFOLIO_EUR, 2),
        "portfolio_value":      round(port_value, 2),
        "portfolio_return_pct": port_return_pct,
        "allocated_eur":        round(allocated, 2),
        "available_eur":        round(available, 2),
        "best":                 {"ticker": best["ticker"], "ret": best["return_pct"], "date": best["close_date"]},
        "worst":                {"ticker": worst["ticker"], "ret": worst["return_pct"], "date": worst["close_date"]},
        "monthly":              dict(sorted(monthly.items())),
        "months_back":          months_back,
    }


def get_trades_by_month(months_back: int = 3) -> dict[str, list[dict]]:
    """Devolve os trades fechados agrupados por mês de fecho."""
    trades  = _load()
    cutoff  = date.today() - timedelta(days=months_back * 30)
    by_month: dict[str, list[dict]] = {}
    for t in trades:
        if not t.get("status", "").startswith("CLOSED"):
            continue
        close_date = t.get("close_date")
        if not close_date:
            continue
        if date.fromisoformat(close_date) < cutoff:
            continue
        month = close_date[:7]
        by_month.setdefault(month, []).append(t)
    return dict(sorted(by_month.items()))


def format_monthly_trade_list(months_back: int = 3) -> str:
    """Lista detalhada de todos os trades por mês com lucro/perda por trade."""
    by_month = get_trades_by_month(months_back)
    if not by_month:
        return "Sem trades fechados no periodo."

    lines = []
    grand_total_pnl = 0.0

    for month, trades in sorted(by_month.items()):
        month_pnl   = 0.0
        month_lines = []
        for t in sorted(trades, key=lambda x: x.get("close_date", "")):
            ret     = t.get("return_pct", 0) or 0
            amount  = t.get("amount_eur", 0)
            pnl_eur = round(amount * ret / 100, 2)
            status  = t.get("status", "")
            badge   = "(early)" if "EARLY" in status else ("(target)" if "TARGET" in status else "(time)")
            sign    = "+" if pnl_eur >= 0 else ""
            month_lines.append(
                f"  {'UP' if ret >= 0 else 'DN'}  {t['ticker']:6s} "
                f"EUR{amount:.0f} @ {t['open_price']:.2f} -> {t['close_price']:.2f}  "
                f"{ret:+.1f}%  {sign}EUR{pnl_eur:.0f} {badge}"
            )
            month_pnl += pnl_eur

        grand_total_pnl += month_pnl
        sign = "+" if month_pnl >= 0 else ""
        lines.append(f"\n{month}  [{sign}EUR{month_pnl:.0f}]")
        lines.extend(month_lines)

    sign_total = "+" if grand_total_pnl >= 0 else ""
    lines.append(f"\nTotal {months_back}m: {sign_total}EUR{grand_total_pnl:.0f}")
    return "\n".join(lines)


def format_performance_report(perf: dict) -> str:
    """Format the paper trading performance report for Telegram."""
    port_val  = perf.get("portfolio_value", PAPER_PORTFOLIO_EUR)
    port_ret  = perf.get("portfolio_return_pct", 0.0)
    allocated = perf.get("allocated_eur", 0.0)
    available = perf.get("available_eur", PAPER_PORTFOLIO_EUR)

    if perf["n_closed"] == 0:
        n_open = perf.get("n_open", 0)
        return (
            f"Paper portfolio active — {n_open} open positions.\n"
            f"Portfolio: EUR{port_val:.0f} | Available: EUR{available:.0f}\n"
            f"No closed positions yet — performance will appear after first exit."
        )

    months = perf.get("months_back", 3)
    ret    = perf["total_return"]
    spy    = perf["spy_return"]
    alpha  = perf["alpha"]
    wr     = perf["win_rate"]
    early  = perf.get("n_early_exits", 0)
    n_open = perf.get("n_open", 0)

    alpha_verdict = "BEATS MARKET" if alpha > 0 else "BELOW MARKET"
    lines = [
        f"Paper portfolio — rotating EUR{PAPER_PORTFOLIO_EUR:.0f} fund, last {months} months:",
        f"Portfolio value: EUR{port_val:.0f} ({port_ret:+.1f}% total) | "
        f"Allocated: EUR{allocated:.0f} | Available: EUR{available:.0f}",
        "",
        f"{perf['n_closed']} closed trades:",
        f"  Return: {ret:+.1f}% | SPY: {spy:+.1f}% | Alpha: {alpha:+.1f}pp  [{alpha_verdict}]",
        f"  Win rate: {wr:.0%}  ({int(perf['n_closed']*wr)}/{perf['n_closed']} positive)",
    ]
    if early > 0:
        lines.append(f"  Early exits: {early} trades")
    lines.append("")

    if perf.get("best"):
        b = perf["best"]
        lines.append(f"Best: {b['ticker']} {b['ret']:+.1f}% ({b['date']})")
    if perf.get("worst"):
        w = perf["worst"]
        lines.append(f"Worst: {w['ticker']} {w['ret']:+.1f}% ({w['date']})")

    monthly = perf.get("monthly", {})
    if monthly:
        lines.append("")
        lines.append("By month:")
        for month, m in sorted(monthly.items())[-6:]:
            sign = "+" if m["alpha"] >= 0 else ""
            lines.append(
                f"  {month}: {m['ret']:+.1f}% vs SPY {m['spy']:+.1f}% "
                f"(alpha {sign}{m['alpha']:.1f}pp) | {m['n']} trades"
            )

    if n_open:
        lines.append(f"\n{n_open} positions still open.")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Momentum Paper Trading
# ─────────────────────────────────────────────────────────────────────────────
# Usa o mesmo _TRADES_FILE mas com trade_type="MOMENTUM".
# Exit por trailing stop (não target fixo).

_MOMENTUM_TRAILING_STOP = float(os.environ.get("MOMENTUM_TRAILING_STOP_PCT", "0.12"))
_MOMENTUM_HOLD_DAYS     = 60   # default; sobreposto pela durability tag


def _trailing_stop_for_durability(durability: str) -> float:
    """Stop adaptado à durabilidade prevista: DURÁVEL tolera mais pullback."""
    return {"DURÁVEL": 0.15, "MODERADO": 0.12, "TÁCTICO": 0.08}.get(durability, _MOMENTUM_TRAILING_STOP)


def _hold_days_for_durability(durability: str) -> int:
    """Hold máximo adaptado: DURÁVEL pode durar 6 meses, TÁCTICO sai em 30d."""
    return {"DURÁVEL": 180, "MODERADO": 60, "TÁCTICO": 30}.get(durability, _MOMENTUM_HOLD_DAYS)


def _get_current_momentum_signals(ticker: str) -> dict:
    """Obtém sinais de momentum actuais para re-scoring diário.

    Usa data_feed para preços EOD e o cálculo do scanner.
    Retorna {} em caso de falha para que o caller salte sem erros.
    """
    try:
        from data.data_feed import get_eod_prices
        import pandas as pd
        df = get_eod_prices(ticker, lookback_days=280)  # 280d para 52-week high e VWAP
        if df is None or df.empty or "Close" not in df.columns:
            return {}
        if "date" in df.columns:
            df = df.set_index(pd.to_datetime(df["date"])).sort_index()
        df.columns = [c.capitalize() if c.lower() in ("open","high","low","close","volume") else c for c in df.columns]
        from momentum_radar.scanner import _calc_momentum_signals
        return _calc_momentum_signals(df)
    except Exception as e:
        log.debug(f"[paper_m] _get_current_signals {ticker}: {e}")
        return {}


def record_momentum_paper_buy(
    ticker:            str,
    open_price:        float,
    amount_eur:        float,
    momentum_score:    float = 0.0,
    ml_score_l1:       float = 0.0,
    ml_score_l2:       float = 0.0,
    ml_score_l3:       float = 0.0,
    ml_score_ensemble: float = 0.0,
    ml_durability:     str   = "MODERADO",
    usd_eur:           float = _USD_EUR_DEFAULT,
    model_id:          str   = "momentum_us",
    sector:            str   = "",
) -> str:
    """Regista posição simulada de momentum quando o scanner gera alerta.

    A saída é por trailing stop adaptado à durabilidade (DURÁVEL=-15%,
    MODERADO=-12%, TÁCTICO=-8%), não por target fixo.
    """
    if open_price <= 0:
        return ""

    # Dedup: don't open MOMENTUM on top of an existing MOMENTUM position.
    # DIP + MOMENTUM on same ticker is allowed (independent positions, exit separately).
    if has_open_momentum(ticker):
        log.info(f"[paper_m] SKIP {ticker} — MOMENTUM already open")
        return ""

    # Rotating pool: clamp to available capital (no per-position % cap)
    trades     = _load()
    port_value = get_paper_portfolio_value(trades)
    available  = get_paper_available(trades)
    requested  = float(amount_eur)
    amount_eur = min(requested, available)

    if amount_eur < MIN_POSITION_EUR:
        missed = {
            "id":               str(uuid.uuid4())[:8],
            "ticker":           ticker.upper(),
            "trade_type":       "MOMENTUM",
            "status":           "MISSED_CAPITAL",
            "signal_date":      date.today().isoformat(),
            "signal_price":     round(open_price, 4),
            "requested_eur":    round(requested, 2),
            "available_eur":    round(available, 2),
            "portfolio_value":  round(port_value, 2),
            "momentum_score":   round(momentum_score, 1),
            "ml_score_ensemble": round(ml_score_ensemble, 4),
            "ml_durability":    ml_durability,
            "model_id":         model_id,
            "sector":           sector,
        }
        trades.append(missed)
        _save(trades)
        log.info(
            f"[paper_m] MISSED {ticker} — available €{available:.0f} < €{MIN_POSITION_EUR:.0f} "
            f"(requested €{requested:.0f}, portfolio €{port_value:.0f})"
        )
        return ""

    trade_id      = str(uuid.uuid4())[:8]
    shares        = amount_eur / max(open_price * usd_eur, 0.01)
    trailing_stop = _trailing_stop_for_durability(ml_durability)
    hold_days     = _hold_days_for_durability(ml_durability)
    today_iso     = date.today().isoformat()
    weight_pct    = round(amount_eur / max(port_value, 1.0) * 100, 2)

    trade = {
        "id":                  trade_id,
        "trade_type":          "MOMENTUM",
        "ticker":              ticker.upper(),
        "open_date":           today_iso,
        "open_price":          round(open_price, 4),
        "amount_eur":          round(amount_eur, 2),
        "weight_pct":          weight_pct,
        "shares":              round(shares, 6),
        "sell_target":         0.0,
        "trailing_high":       round(open_price, 4),
        "trailing_stop_pct":   trailing_stop,
        "hold_days":           hold_days,
        "status":              "OPEN",
        "close_date":          None,
        "close_price":         None,
        "return_pct":          None,
        "spy_return_pct":      None,
        "alpha_pct":           None,
        "momentum_score":      round(momentum_score, 1),
        "ml_score_l1":         round(ml_score_l1, 4),
        "ml_score_l2":         round(ml_score_l2, 4),
        "ml_score_l3":         round(ml_score_l3, 4),
        "ml_score_ensemble":   round(ml_score_ensemble, 4),
        "ml_durability_open":  ml_durability,
        "ml_durability_cur":   ml_durability,
        "durability_history":  [{
            "date":       today_iso,
            "durability": ml_durability,
            "l1":         round(ml_score_l1, 4),
            "l2":         round(ml_score_l2, 4),
            "l3":         round(ml_score_l3, 4),
        }],
        "model_id":            model_id,
        "sector":              sector,
        "portfolio_value_at_open": round(port_value, 2),
        "available_at_open":       round(available, 2),
        "spy_price_open": _fetch_close("SPY", date.today()),
    }
    trades.append(trade)
    _save(trades)
    log.info(
        f"[paper_m] OPEN {ticker} MOMENTUM €{amount_eur:.0f} ({weight_pct:.1f}% of €{port_value:.0f}) "
        f"@ {open_price:.2f} score={momentum_score:.0f} dur={ml_durability} "
        f"stop={trailing_stop:.0%} hold={hold_days}d "
        f"[available €{available:.0f} → €{available - amount_eur:.0f}]"
    )
    return trade_id


# Upside mínimo ao target DIP original para reverter MOMENTUM→DIP em vez de fechar
_DIP_REVERT_MIN_UPSIDE = 0.08   # 8% para target original ainda por capturar
_DIP_REVERT_MAX_DAYS   = 270    # após 270d de hold total a tese DIP expirou


def _revert_momentum_to_dip(t: dict, current_price: float, today_iso: str) -> None:
    """Converte posição MOMENTUM (vinda de DIP upgrade) de volta a gestão DIP.

    Chamado quando o momentum arrefece mas o target DIP original ainda tem
    upside suficiente. A tese DIP retoma — update_open_positions volta a gerir.
    """
    dip_target = float(t.get("dip_sell_target") or 0.0)
    t["trade_type"]          = "DIP"
    t["sell_target"]         = round(dip_target, 4)
    # Trailing stop largo para dar espaço à recuperação DIP
    t["trailing_high"]       = round(current_price, 4)
    t["trailing_stop_pct"]   = 0.22
    # hold_days: mais 60d a partir de hoje para a tese DIP se confirmar
    t["hold_days"]           = 60
    t["revert_to_dip_date"]  = today_iso
    # Limpar flags MOMENTUM que já não se aplicam
    t.pop("momentum_start_date",  None)
    t.pop("past_hold_tightened",  None)
    t.pop("timeout_trailing_date", None)
    log.info(
        f"[paper] REVERT {t['ticker']} MOMENTUM→DIP "
        f"dip_target={dip_target:.2f} price={current_price:.2f} "
        f"upside_restante={dip_target/current_price-1:+.1%}"
    )


def update_momentum_positions(send_fn=None) -> dict:
    """Actualiza posições MOMENTUM abertas: drift de durabilidade + trailing stop.

    Algoritmo diário:
      1. Re-score ML multi-horizonte com sinais actuais
      2. Detectar drift: TÁCTICO→DURÁVEL (alargar stop) ou DURÁVEL→TÁCTICO (apertar)
      3. Enviar alerta se mudança significativa
      4. Actualizar trailing stop adaptado
      5. Fechar APENAS se trailing stop atingido — sem janela fixa de fecho.
         Quando o hold_days esperado é ultrapassado, o stop aperta para reduzir
         tolerância a giveback, mas a posição continua até o stop ser tocado.

    Returns dict com chaves: updated, drifted, closed.
    """
    trades      = _load()
    today       = date.today()
    today_iso   = today.isoformat()
    updated     = 0
    drifted     = 0
    closed      = 0
    just_closed: list[dict] = []

    for t in trades:
        if t.get("status") != "OPEN" or t.get("trade_type") != "MOMENTUM":
            continue

        open_date = date.fromisoformat(t["open_date"])
        # Para posições vindas de upgrade DIP→MOMENTUM, a duração de MOMENTUM
        # conta a partir do momentum_start_date (não do open_date do DIP).
        _mom_start_raw = t.get("momentum_start_date")
        momentum_start = date.fromisoformat(_mom_start_raw) if _mom_start_raw else open_date
        days_held = (today - momentum_start).days

        if days_held < 3:
            continue

        ticker = t["ticker"]
        current_price = _fetch_close(ticker, today)
        if current_price is None:
            continue

        # ── 1. Re-score ML com sinais actuais ────────────────────────────────
        prev_durability = t.get("ml_durability_cur", t.get("ml_durability_open", "MODERADO"))
        new_durability  = prev_durability  # fallback se ML falhar
        new_l1 = new_l2 = new_l3 = 0.0

        signals = _get_current_momentum_signals(ticker)
        if signals:
            try:
                from momentum_radar.scanner import _ml_score_momentum
                ml = _ml_score_momentum(signals)
                if ml.get("ml_ready"):
                    new_l1         = ml.get("ml_score_l1", 0.0)
                    new_l2         = ml.get("ml_score_l2", 0.0)
                    new_l3         = ml.get("ml_score_l3", 0.0)
                    new_durability = ml.get("ml_durability", prev_durability)

                    # Guardar no histórico (1 entrada por dia, não duplicar)
                    hist = t.setdefault("durability_history", [])
                    if not hist or hist[-1].get("date") != today_iso:
                        hist.append({
                            "date": today_iso, "durability": new_durability,
                            "l1": round(new_l1, 4), "l2": round(new_l2, 4), "l3": round(new_l3, 4),
                        })
            except Exception as e:
                log.debug(f"[paper_m] Re-score {ticker}: {e}")

        # ── 2. Detectar drift de durabilidade ────────────────────────────────
        drift_alert = ""
        if new_durability != prev_durability:
            t["ml_durability_cur"] = new_durability

            # Adaptar trailing stop e hold_days
            new_stop = _trailing_stop_for_durability(new_durability)
            new_hold = _hold_days_for_durability(new_durability)
            old_stop = float(t.get("trailing_stop_pct") or _MOMENTUM_TRAILING_STOP)
            t["trailing_stop_pct"] = new_stop
            t["hold_days"]         = new_hold

            cur_ret = (current_price / float(t["open_price"]) - 1)
            if prev_durability == "TÁCTICO" and new_durability == "DURÁVEL":
                drift_alert = (
                    f"📈 *Drift positivo — {ticker}*\n"
                    f"TÁCTICO → DURÁVEL ✅\n"
                    f"Momentum a ganhar força: L3 {new_l3:+.1%} > L1 {new_l1:+.1%}\n"
                    f"Stop alargado: {old_stop:.0%} → {new_stop:.0%} | Hold: {new_hold}d\n"
                    f"Retorno actual: {cur_ret:+.1%}"
                )
            elif prev_durability == "DURÁVEL" and new_durability in ("TÁCTICO", "MODERADO"):
                drift_alert = (
                    f"⚠️ *Drift negativo — {ticker}*\n"
                    f"DURÁVEL → {new_durability}\n"
                    f"Momentum a perder força: L3 {new_l3:+.1%} vs L1 {new_l1:+.1%}\n"
                    f"Stop apertado: {old_stop:.0%} → {new_stop:.0%} | Hold: {new_hold}d\n"
                    f"Retorno actual: {cur_ret:+.1%}\n"
                    f"_Considera saída se stop for atingido antes da tese se confirmar._"
                )
            else:
                drift_alert = (
                    f"🔄 *Drift — {ticker}*: {prev_durability} → {new_durability} "
                    f"(L1={new_l1:+.1%} L3={new_l3:+.1%})"
                )

            drifted += 1
            log.info(f"[paper_m] DRIFT {ticker}: {prev_durability} → {new_durability}")

            if send_fn and drift_alert:
                try:
                    send_fn(drift_alert)
                except Exception:
                    pass

        # ── 3. Trailing stop adaptativo (ATR-based) ──────────────────────────
        trailing_high = max(float(t.get("trailing_high") or t["open_price"]), current_price)
        t["trailing_high"] = round(trailing_high, 4)

        # stop_pct base: vem da durabilidade e é o máximo permitido
        stop_pct = float(t.get("trailing_stop_pct") or _MOMENTUM_TRAILING_STOP)

        # Quando passamos o hold_days esperado, aperta o stop_pct base (uma vez).
        time_up = days_held >= int(t.get("hold_days") or _MOMENTUM_HOLD_DAYS)
        if time_up and not t.get("past_hold_tightened"):
            tightened_stop = round(stop_pct * 0.75, 4)  # sem floor — ATR é o protector
            if tightened_stop < stop_pct:
                log.info(
                    f"[paper_m] PAST_HOLD {ticker} day={days_held} "
                    f"stop_base {stop_pct:.1%}→{tightened_stop:.1%}"
                )
                t["trailing_stop_pct"]   = tightened_stop
                t["past_hold_tightened"] = True
                stop_pct                 = tightened_stop

        # ATR-based refinement: em baixa volatilidade, sair mais perto do pico.
        # Em alta volatilidade, o ATR fica largo e usa-se o stop_pct base.
        # Multiplicadores: DURÁVEL mantém mais espaço, TÁCTICO aperta mais.
        _cur_dur  = t.get("ml_durability_cur") or "MODERADO"
        _atr_mult = {"DURÁVEL": 5.0, "MODERADO": 4.0, "TÁCTICO": 3.0}.get(_cur_dur, 4.0)
        _atr_pct  = float(signals.get("atr_pct", 0.02)) if signals else 0.02
        _atr_stop = _atr_pct * _atr_mult
        # Nunca usar stop_effectivo < 2% (evita disparos por micro-flutuações EOD)
        effective_stop_pct = min(stop_pct, max(_atr_stop, 0.02))

        trailing_stop_price = trailing_high * (1 - effective_stop_pct)
        hit_trailing = current_price < trailing_stop_price

        updated += 1

        if not hit_trailing:
            continue

        # ── 4. Trailing stop atingido: verificar se tese DIP original ainda válida ──
        # Se a posição veio de upgrade DIP→MOMENTUM e o target DIP original ainda
        # tem upside suficiente, revertemos para gestão DIP em vez de fechar.
        total_days = (today - open_date).days
        if (
            t.get("prev_trade_type") == "DIP"
            and total_days < _DIP_REVERT_MAX_DAYS
            and not t.get("revert_to_dip_date")   # só reverter uma vez
        ):
            dip_target = float(t.get("dip_sell_target") or 0.0)
            remaining_upside = (dip_target / current_price - 1) if dip_target > current_price else 0.0
            if remaining_upside >= _DIP_REVERT_MIN_UPSIDE:
                _revert_momentum_to_dip(t, current_price, today_iso)
                if send_fn:
                    try:
                        send_fn(
                            f"🔄 *MOMENTUM→DIP — {ticker}*\n"
                            f"Momentum arrefeceu mas a tese DIP ainda tem "
                            f"{remaining_upside:+.1%} de upside para o target {dip_target:.2f}\n"
                            f"Stop alargado de volta a 22% | Hold +60d\n"
                            f"Retorno actual: {(current_price/float(t['open_price'])-1):+.1%}"
                        )
                    except Exception:
                        pass
                continue  # posição fica aberta, update_open_positions() retoma

        # ── 5. Close position — PIT SPY comparison using stored open price ──────
        ret_pct = round((current_price / float(t["open_price"]) - 1) * 100, 3)
        _spy_open_stored = t.get("spy_price_open")
        _spy_close_live  = _fetch_close("SPY", today)
        if _spy_open_stored and _spy_close_live and _spy_open_stored > 0:
            spy_ret = round((_spy_close_live / _spy_open_stored - 1) * 100, 3)
            t["spy_price_close"] = _spy_close_live
        else:
            spy_ret = _fetch_spy_return(open_date, today)
        alpha = round(ret_pct - (spy_ret or 0), 3) if spy_ret is not None else None

        t["status"]         = "CLOSED_TRAILING_STOP"
        t["close_date"]     = today_iso
        t["close_price"]    = round(current_price, 4)
        t["return_pct"]     = ret_pct
        t["spy_return_pct"] = spy_ret
        t["alpha_pct"]      = alpha
        closed += 1
        just_closed.append(t)
        log.info(
            f"[paper_m] CLOSED {ticker} CLOSED_TRAILING_STOP dur={t.get('ml_durability_cur','?')} "
            f"ret={ret_pct:.1f}% high={trailing_high:.2f} stop={trailing_stop_price:.2f} "
            f"eff_stop={effective_stop_pct:.1%} atr={_atr_pct:.1%}"
        )

    _save(trades)
    return {"updated": updated, "drifted": drifted, "closed": closed, "just_closed": just_closed}


# ─────────────────────────────────────────────────────────────────────────────
# Posições CORE — Hold de longo prazo (compounder / alto yield)
# ─────────────────────────────────────────────────────────────────────────────
# Filosofia: CORE nunca fecha automaticamente. O sistema rastreia o P&L
# unrealizado e o dividend income acumulado, mas a decisão de saída é sempre
# manual. Adequado para: compounders de qualidade (MSFT, GOOGL…), high-yield
# defensivos (NVO, ABBV…), posições de very-long-term conviction.


def record_core_holding(
    ticker:           str,
    open_price:       float,
    amount_eur:       float,
    annual_yield_pct: float = 0.0,
    note:             str   = "",
    usd_eur:          float = _USD_EUR_DEFAULT,
    model_id:         str   = "core_us",
    sector:           str   = "",
) -> str:
    """Record a CORE long-term holding position.

    CORE positions share the same €10k paper pool as DIP and MOMENTUM — opening
    a CORE position reduces the capital available for tactical signals and vice versa.
    annual_yield_pct: expected annual dividend yield (%), e.g. 5.0 for NVO.
    """
    if open_price <= 0 or amount_eur <= 0:
        return ""

    # Rotating pool: shared across all models
    trades     = _load()
    port_value = get_paper_portfolio_value(trades)
    available  = get_paper_available(trades)
    requested  = float(amount_eur)
    amount_eur = min(requested, available)

    if amount_eur < MIN_POSITION_EUR:
        missed = {
            "id":               str(uuid.uuid4())[:8],
            "ticker":           ticker.upper(),
            "trade_type":       "CORE",
            "status":           "MISSED_CAPITAL",
            "signal_date":      date.today().isoformat(),
            "signal_price":     round(open_price, 4),
            "requested_eur":    round(requested, 2),
            "available_eur":    round(available, 2),
            "portfolio_value":  round(port_value, 2),
            "model_id":         model_id,
            "sector":           sector,
            "note":             note,
        }
        trades.append(missed)
        _save(trades)
        log.info(
            f"[core] MISSED {ticker.upper()} — available €{available:.0f} < €{MIN_POSITION_EUR:.0f} "
            f"(requested €{requested:.0f}, portfolio €{port_value:.0f})"
        )
        return ""

    trade_id   = str(uuid.uuid4())[:8]
    shares     = amount_eur / max(open_price * usd_eur, 0.01)
    today_iso  = date.today().isoformat()
    weight_pct = round(amount_eur / max(port_value, 1.0) * 100, 2)
    weight_pct = round(amount_eur / max(PAPER_PORTFOLIO_EUR, 1.0) * 100, 2)

    trade = {
        "id":               trade_id,
        "trade_type":       "CORE",
        "ticker":           ticker.upper(),
        "open_date":        today_iso,
        "open_price":       round(open_price, 4),
        "amount_eur":       round(amount_eur, 2),
        "weight_pct":       weight_pct,
        "shares":           round(shares, 6),
        "annual_yield_pct": round(annual_yield_pct, 2),
        "note":             note,
        "model_id":         model_id,
        "sector":           sector,
        "status":           "OPEN",
        "close_date":       None,
        "close_price":      None,
        "return_pct":       None,
        "spy_return_pct":   None,
        "alpha_pct":        None,
        "sell_target":             0.0,
        "hold_days":               0,
        "portfolio_value_at_open": round(port_value, 2),
        "available_at_open":       round(available, 2),
        "spy_price_open": _fetch_close("SPY", date.today()),
        # Dividend reinvestment tracking (DRIP — gross, pre-tax)
        "dividends_reinvested_eur": 0.0,   # cumulative gross dividends reinvested
        "last_dividend_check":      today_iso,
        "dividend_history":         [],    # [{date, per_share, shares_added, price, gross_eur}]
    }
    # `trades` was already loaded by the pool check above — reuse it
    trades.append(trade)
    _save(trades)
    log.info(
        f"[core] OPEN {ticker.upper()} €{amount_eur:.0f} ({weight_pct:.1f}% of €{port_value:.0f}) "
        f"@ {open_price:.2f} yield={annual_yield_pct:.1f}% "
        f"[available €{available:.0f} → €{available - amount_eur:.0f}]"
    )
    return trade_id


def close_core_holding(ticker: str, close_price: float) -> Optional[dict]:
    """Close a CORE position manually. Returns the closed trade dict, or None if not found."""
    ticker    = ticker.upper()
    today     = date.today()
    trades    = _load()
    closed_t: Optional[dict] = None
    for t in trades:
        if t.get("ticker") != ticker or t.get("status") != "OPEN":
            continue
        if t.get("trade_type") != "CORE":
            continue
        open_date = date.fromisoformat(t["open_date"])
        ret_pct   = round((close_price / float(t["open_price"]) - 1) * 100, 3)
        _spy_open_stored = t.get("spy_price_open")
        _spy_close_live  = _fetch_close("SPY", today)
        if _spy_open_stored and _spy_close_live and _spy_open_stored > 0:
            spy_ret = round((_spy_close_live / _spy_open_stored - 1) * 100, 3)
            t["spy_price_close"] = _spy_close_live
        else:
            spy_ret = _fetch_spy_return(open_date, today)
        alpha = round(ret_pct - (spy_ret or 0), 3) if spy_ret is not None else None
        t["status"]         = "CLOSED_MANUAL"
        t["close_date"]     = today.isoformat()
        t["close_price"]    = round(close_price, 4)
        t["return_pct"]     = ret_pct
        t["spy_return_pct"] = spy_ret
        t["alpha_pct"]      = alpha
        closed_t = t
        log.info(f"[core] CLOSE {ticker} MANUAL @ {close_price:.2f} ret={ret_pct:.1f}%")
        break  # close only the oldest open position
    if closed_t is not None:
        _save(trades)
    return closed_t


_TAX_DISCLAIMER = (
    "GROSS dividends reinvested (pre-tax). "
    "No withholding tax deducted — actual after-tax returns will be lower. "
    "PT residents: dividends taxed at 28% IRS + possible foreign withholding (e.g. 15% US, 15% DK). "
    "Tax is owed but not modelled here."
)


def _fetch_dividends_since(ticker: str, since_date: date) -> list[dict]:
    """Return list of {date, per_share} dividend events since since_date (exclusive)."""
    try:
        import yfinance as yf
        import pandas as pd
        hist = yf.Ticker(ticker).dividends
        if hist is None or hist.empty:
            return []
        if hist.index.tz is not None:
            hist.index = hist.index.tz_convert(None)
        # Keep only dividends after since_date
        since_ts = pd.Timestamp(since_date)
        recent   = hist[hist.index > since_ts]
        return [
            {"date": str(idx.date()), "per_share": round(float(val), 6)}
            for idx, val in recent.items()
            if float(val) > 0
        ]
    except Exception as e:
        log.debug(f"[core_drip] dividend fetch {ticker}: {e}")
        return []


def update_core_dividends(usd_eur: float = _USD_EUR_DEFAULT) -> dict:
    """Reinvest dividends for all open CORE positions (DRIP — gross, pre-tax).

    For each CORE position, fetches actual ex-dividend events since the last
    check date and buys additional shares at the closing price of that day.
    Shares accumulate in the position — no cash leaves the portfolio.

    IMPORTANT: dividends are reinvested GROSS (before any tax).
    See _TAX_DISCLAIMER for the tax caveat.

    Returns: {"processed": N, "events": N, "total_reinvested_eur": X, "errors": N}
    """
    trades   = _load()
    today    = date.today()
    changed  = False
    total_events       = 0
    total_reinvested   = 0.0
    processed_tickers  = 0
    errors             = 0

    for t in trades:
        if t.get("trade_type") != "CORE" or t.get("status") != "OPEN":
            continue

        ticker     = t["ticker"]
        shares     = float(t.get("shares", 0))
        last_check = date.fromisoformat(t.get("last_dividend_check") or t.get("open_date", "2000-01-01"))

        if shares <= 0:
            continue

        try:
            events = _fetch_dividends_since(ticker, last_check)
        except Exception as e:
            log.warning(f"[core_drip] {ticker} fetch error: {e}")
            errors += 1
            continue

        processed_tickers += 1

        for ev in events:
            ex_date    = date.fromisoformat(ev["date"])
            per_share  = float(ev["per_share"])
            gross_usd  = shares * per_share
            gross_eur  = round(gross_usd * usd_eur, 4)

            if gross_eur < 0.01:
                continue  # negligible

            # Fetch closing price on the ex-date to compute reinvestment price
            reinvest_price_usd = _fetch_close(ticker, ex_date)
            if reinvest_price_usd is None or reinvest_price_usd <= 0:
                log.debug(f"[core_drip] {ticker} no price on {ex_date} — skipping dividend")
                continue

            reinvest_price_eur = reinvest_price_usd * usd_eur
            new_shares = round(gross_eur / reinvest_price_eur, 6)

            # Reinvest: add shares to position
            t["shares"]                    = round(shares + new_shares, 6)
            t["dividends_reinvested_eur"]  = round(
                float(t.get("dividends_reinvested_eur", 0)) + gross_eur, 4
            )
            shares += new_shares  # update local var for next event in same loop

            history_entry = {
                "date":        ev["date"],
                "per_share":   per_share,
                "shares_before": round(shares - new_shares, 6),
                "shares_added":  new_shares,
                "reinvest_price_usd": round(reinvest_price_usd, 4),
                "gross_eur":   gross_eur,
                "tax_note":    "gross — no tax deducted",
            }
            if "dividend_history" not in t or not isinstance(t["dividend_history"], list):
                t["dividend_history"] = []
            t["dividend_history"].append(history_entry)

            total_events      += 1
            total_reinvested  += gross_eur
            changed            = True
            log.info(
                f"[core_drip] {ticker} DRIP {ev['date']}: "
                f"{per_share:.4f}/share × {shares - new_shares:.4f}sh "
                f"= €{gross_eur:.2f} → +{new_shares:.4f} shares @ {reinvest_price_usd:.2f}. "
                f"[{_TAX_DISCLAIMER[:60]}...]"
            )

        # Update last_dividend_check regardless (even if no events — avoids re-fetching old history)
        if str(today) != t.get("last_dividend_check"):
            t["last_dividend_check"] = today.isoformat()
            changed = True

    if changed:
        _save(trades)

    return {
        "processed":            processed_tickers,
        "events":               total_events,
        "total_reinvested_eur": round(total_reinvested, 2),
        "errors":               errors,
        "tax_disclaimer":       _TAX_DISCLAIMER,
    }


def get_core_holdings_summary() -> list[dict]:
    """Devolve todas as posições CORE abertas com P&L e yield acumulado."""
    trades = _load()
    today  = date.today()
    result = []

    for t in trades:
        if t.get("status") != "OPEN" or t.get("trade_type") != "CORE":
            continue

        ticker     = t["ticker"]
        open_date  = date.fromisoformat(t["open_date"])
        days_held  = (today - open_date).days
        open_price = float(t["open_price"])
        amount_eur = float(t["amount_eur"])

        current_price = _fetch_close(ticker, today)
        if current_price is None:
            current_price = open_price  # fallback para não crashar

        # P&L unrealizado
        ret_pct    = round((current_price / open_price - 1) * 100, 3)
        unrealized = round(amount_eur * ret_pct / 100, 2)

        # Actual reinvested dividends (from DRIP history — more accurate than estimated yield)
        drip_eur    = float(t.get("dividends_reinvested_eur", 0.0))
        drip_events = len(t.get("dividend_history", []))
        shares_now  = float(t.get("shares", 0))

        # Estimated yield income for display (pro-rata, for positions without DRIP data yet)
        annual_yield = float(t.get("annual_yield_pct") or 0.0)
        yield_income_est = round(amount_eur * (annual_yield / 100) * (days_held / 365), 2)
        # Prefer actual DRIP amount if available; fall back to estimate
        yield_income = drip_eur if drip_eur > 0 else yield_income_est

        # Total return: price P&L + actual dividend income
        # Dividends already reinvested → reflected in shares_now > shares_original
        # We express total return as (current_value - original_investment) / original_investment
        original_shares = float(amount_eur) / max(open_price, 0.01)  # approximate (ignores usd_eur drift)
        current_value_eur = shares_now * current_price  # in USD, approximate
        drip_adjusted_ret_pct = round(
            (current_value_eur / max(amount_eur / (current_price / open_price if open_price else 1), 0.01) - 1) * 100, 3
        ) if shares_now > 0 else ret_pct
        total_return_pct = round(ret_pct + (drip_eur / max(amount_eur, 1)) * 100, 3)

        # SPY in the same period
        spy_ret = _fetch_spy_return(open_date, today)

        result.append({
            "id":                       t["id"],
            "ticker":                   ticker,
            "open_date":                t["open_date"],
            "open_price":               open_price,
            "current_price":            round(current_price, 4),
            "amount_eur":               amount_eur,
            "shares":                   shares_now,
            "days_held":                days_held,
            "ret_pct":                  ret_pct,
            "unrealized_eur":           unrealized,
            "annual_yield_pct":         annual_yield,
            "yield_income_eur":         yield_income,
            "yield_income_is_actual":   drip_eur > 0,  # True = from real DRIP, False = estimate
            "drip_events":              drip_events,
            "dividends_reinvested_eur": drip_eur,
            "total_return_pct":         total_return_pct,
            "spy_ret_pct":              spy_ret,
            "alpha_pct":                round(total_return_pct - (spy_ret or 0), 3) if spy_ret is not None else None,
            "note":                     t.get("note", ""),
            "tax_disclaimer":           _TAX_DISCLAIMER,
        })

    result.sort(key=lambda x: x["open_date"])
    return result


def get_drip_history() -> dict:
    """Return a full, auditable DRIP ledger for all CORE positions.

    Structure:
    {
      "generated":      ISO datetime,
      "portfolio_start_eur": float,
      "tax_disclaimer": str,
      "note_gross":     str,
      "positions": [
        {
          "ticker":         str,
          "open_date":      str,
          "amount_eur":     float,
          "shares_original": float,
          "shares_current":  float,
          "shares_added_via_drip": float,
          "total_reinvested_eur":  float,
          "drip_events": [
            {
              "date":            str,       # ex-dividend date
              "per_share_usd":   float,     # gross dividend per original share (USD)
              "shares_held":     float,     # shares held at that date
              "gross_usd":       float,     # total gross dividend (USD)
              "gross_eur":       float,     # converted at rate used
              "reinvest_price_usd": float,  # closing price on that day
              "new_shares":      float,     # shares purchased with this dividend
              "tax_note":        str,       # always "gross — no tax deducted"
            }
          ],
          "note": str,
        }
      ],
      "totals": {
        "total_reinvested_eur": float,
        "total_drip_events":    int,
        "total_positions":      int,
      }
    }
    """
    trades   = _load()
    today    = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    positions = []

    for t in trades:
        if t.get("trade_type") != "CORE" or t.get("status") != "OPEN":
            continue

        raw_history = t.get("dividend_history") or []
        shares_original = float(t.get("amount_eur", 0)) / max(float(t.get("open_price", 1)), 1e-9)
        shares_current  = float(t.get("shares", shares_original))
        drip_eur        = float(t.get("dividends_reinvested_eur", 0.0))
        shares_added    = round(shares_current - shares_original, 6)

        events_clean = []
        for ev in raw_history:
            events_clean.append({
                "date":               ev.get("date", ""),
                "per_share_usd":      round(float(ev.get("per_share", ev.get("per_share_usd", 0))), 6),
                "shares_held":        round(float(ev.get("shares_held", 0)), 4),
                "gross_usd":          round(float(ev.get("gross_usd", 0)), 4),
                "gross_eur":          round(float(ev.get("gross_eur", 0)), 4),
                "reinvest_price_usd": round(float(ev.get("reinvest_price_usd", 0)), 4),
                "new_shares":         round(float(ev.get("new_shares", ev.get("shares_added", 0))), 6),
                "tax_note":           "gross — no tax deducted",
            })
        events_clean.sort(key=lambda e: e["date"])

        positions.append({
            "ticker":                t["ticker"],
            "open_date":             t["open_date"],
            "amount_eur":            float(t.get("amount_eur", 0)),
            "shares_original":       round(shares_original, 6),
            "shares_current":        round(shares_current, 6),
            "shares_added_via_drip": round(shares_added, 6),
            "total_reinvested_eur":  round(drip_eur, 4),
            "drip_events":           events_clean,
            "note":                  t.get("note", ""),
        })

    positions.sort(key=lambda p: p["open_date"])
    total_reinvested = sum(p["total_reinvested_eur"] for p in positions)
    total_events     = sum(len(p["drip_events"]) for p in positions)

    return {
        "generated":          today,
        "portfolio_start_eur": PAPER_PORTFOLIO_EUR,
        "tax_disclaimer":     _TAX_DISCLAIMER,
        "note_gross": (
            "All dividends are recorded and reinvested GROSS (before any withholding tax). "
            "This ledger is published for transparency — not as tax advice. "
            "Your actual after-tax return will be lower. See 'tax_disclaimer' field for details."
        ),
        "positions": positions,
        "totals": {
            "total_reinvested_eur": round(total_reinvested, 4),
            "total_drip_events":    total_events,
            "total_positions":      len(positions),
        },
    }


def format_drip_report_md() -> str:
    """Generate a human-readable Markdown DRIP ledger for publishing.

    This is the public-facing transparency document — every dividend event,
    every reinvestment, explicit tax disclaimer.
    """
    data     = get_drip_history()
    today    = data["generated"]
    positions = data["positions"]
    totals   = data["totals"]

    lines = [
        "# DipRadar — CORE DRIP Ledger (Dividend Reinvestment Audit Trail)",
        "",
        f"**Generated:** {today}  ",
        f"**Portfolio base:** €{data['portfolio_start_eur']:,.0f}  ",
        f"**Total positions:** {totals['total_positions']}  ",
        f"**Total DRIP events:** {totals['total_drip_events']}  ",
        f"**Total reinvested (gross):** €{totals['total_reinvested_eur']:,.2f}  ",
        "",
        "---",
        "",
        "## ⚠️ Tax Disclaimer",
        "",
        f"> {data['tax_disclaimer']}",
        "",
        f"> {data['note_gross']}",
        "",
        "---",
        "",
    ]

    if not positions:
        lines += [
            "## No CORE positions with dividend history yet.",
            "",
            "_DRIP tracking activates automatically once CORE positions receive their first dividend._",
        ]
        return "\n".join(lines)

    lines += ["## Position Summary", ""]
    lines += ["| Ticker | Open Date | Invested (€) | Shares (original) | Shares (current) | Via DRIP | Reinvested (€) | Events |"]
    lines += ["|--------|-----------|-------------|-------------------|-----------------|----------|----------------|--------|"]
    for p in positions:
        lines.append(
            f"| {p['ticker']} | {p['open_date']} | €{p['amount_eur']:,.2f} "
            f"| {p['shares_original']:.4f} | {p['shares_current']:.4f} "
            f"| +{p['shares_added_via_drip']:.4f} | €{p['total_reinvested_eur']:,.4f} "
            f"| {len(p['drip_events'])} |"
        )

    lines += ["", "---", ""]

    for p in positions:
        ticker = p["ticker"]
        lines += [
            f"## {ticker} — DRIP Event Log",
            "",
            f"- **Position opened:** {p['open_date']}",
            f"- **Capital deployed:** €{p['amount_eur']:,.2f}",
            f"- **Original shares:** {p['shares_original']:.4f}",
            f"- **Current shares:** {p['shares_current']:.4f} "
            f"(+{p['shares_added_via_drip']:.4f} via DRIP = "
            f"+{(p['shares_added_via_drip'] / max(p['shares_original'], 1e-9) * 100):.2f}%)",
            f"- **Total reinvested:** €{p['total_reinvested_eur']:,.4f} (gross, pre-tax)",
            "",
        ]
        if p.get("note"):
            lines += [f"*Note: {p['note']}*", ""]

        if not p["drip_events"]:
            lines += ["_No dividend events recorded yet._", "", ""]
            continue

        lines += [
            "| Date | Per Share (USD) | Shares Held | Gross (USD) | Gross (EUR) | Reinvest Price | New Shares | Tax Note |",
            "|------|----------------|-------------|-------------|-------------|----------------|------------|----------|",
        ]
        for ev in p["drip_events"]:
            lines.append(
                f"| {ev['date']} | ${ev['per_share_usd']:.4f} | {ev['shares_held']:.4f} "
                f"| ${ev['gross_usd']:.4f} | €{ev['gross_eur']:.4f} "
                f"| ${ev['reinvest_price_usd']:.2f} | +{ev['new_shares']:.6f} "
                f"| {ev['tax_note']} |"
            )
        lines += [""]

    lines += [
        "---",
        "",
        "_This report is auto-generated by DipRadar. "
        "All figures are simulated paper-trading — not real money. "
        "Past performance is not indicative of future results. "
        "No investment advice is implied._",
    ]

    return "\n".join(lines)


def update_core_positions(send_fn=None) -> dict:
    """Monitoramento diário de posições CORE: deteção de deterioração estrutural.

    NÃO fecha posições automaticamente. Usa lógica multi-fator para distinguir
    "compounder de qualidade em dip temporário" (NVO, MSFT) de "colapso real"
    (CISCO 2001, GE 2018).

    Níveis de alerta:
      🚨 RED   — ≥3 sinais negativos OU (≥2 negativos + fundamentais a deteriorar)
      ⚠️ ORANGE — ≥2 sinais de preço mas fundamentais ainda intactos
      💡 YELLOW — drawdown profundo com negócio sólido (potencial acumulação)
      📉 INFO   — MA200 trend break isolado ou declínio secular (sem acção imediata)

    Retorna dict com chaves: checked, alerts_sent.
    """
    trades    = _load()
    today     = date.today()
    today_iso = today.isoformat()
    checked   = 0
    alerts_sent = 0

    for t in trades:
        if t.get("status") != "OPEN" or t.get("trade_type") != "CORE":
            continue

        ticker = t["ticker"]
        checked += 1

        # Fetch histórico longo para MA200 e 52w high
        try:
            from data.data_feed import get_eod_prices
            import pandas as _pd
            df = get_eod_prices(ticker, lookback_days=280)
            if df is None or df.empty or len(df) < 50:
                continue
            if "date" in df.columns:
                df = df.set_index(_pd.to_datetime(df["date"])).sort_index()
            df.columns = [
                c.capitalize() if c.lower() in ("open","high","low","close","volume") else c
                for c in df.columns
            ]
            close = df["Close"]
        except Exception as e:
            log.debug(f"[core] fetch {ticker}: {e}")
            continue

        current_price = float(close.iloc[-1])
        open_price    = float(t["open_price"])

        # ── Indicadores de preço ──────────────────────────────────────────────
        ma_period = min(200, len(close) - 1)
        ma200     = float(close.rolling(ma_period).mean().iloc[-1]) if ma_period >= 20 else None

        high_52w         = float(close.rolling(min(252, len(close))).max().iloc[-1])
        dd_from_52w_high = (current_price / high_52w - 1) if high_52w > 0 else 0.0

        days_below_ma200 = 0
        if ma200 is not None:
            ma_series = close.rolling(ma_period).mean()
            for j in range(len(close) - 1, max(len(close) - 60, -1), -1):
                if close.iloc[j] < ma_series.iloc[j]:
                    days_below_ma200 += 1
                else:
                    break

        ma200_trend = 0.0
        if ma200 is not None and len(close) >= ma_period + 60:
            ma200_60d_ago = float(close.rolling(ma_period).mean().iloc[-61])
            ma200_trend   = (ma200 / ma200_60d_ago - 1) if ma200_60d_ago > 0 else 0.0

        ret_vs_cost = (current_price / open_price - 1) if open_price > 0 else 0.0

        already_alerted_today = (t.get("last_structural_alert_date", "") == today_iso)

        # ── Avaliação multi-fator ─────────────────────────────────────────────
        decline_signals: list[str] = []
        quality_signals: list[str] = []

        # Sinais de preço negativos
        if dd_from_52w_high < -0.35:
            decline_signals.append(f"queda {dd_from_52w_high:+.1%} do pico 52s")
        if ma200_trend < -0.05:
            decline_signals.append(f"MA200 em declínio {ma200_trend:+.1%} (60d)")
        if days_below_ma200 >= 30:
            decline_signals.append(f"{days_below_ma200}d consecutivos abaixo da MA200")

        # Fundamentais — lazy fetch só quando há pelo menos 1 sinal de preço negativo
        # (evita chamadas de API desnecessárias em dias normais)
        if len(decline_signals) >= 1:
            try:
                import yfinance as _yf
                _info        = _yf.Ticker(ticker).info
                _rev         = _info.get("revenueGrowth")       # decimal y/y
                _fcf         = float(_info.get("freeCashflow") or 0)
                _mktcap      = float(_info.get("marketCap") or 1)
                _margin      = _info.get("grossMargins")
                _earn        = _info.get("earningsGrowth")
                _fcf_yield   = (_fcf / _mktcap) if _mktcap > 0 else None
                _roe         = _info.get("returnOnEquity")

                # Sinais fundamentais negativos
                if _rev is not None and _rev < -0.05:
                    decline_signals.append(f"revenue em queda {_rev:+.1%} y/y")
                if _fcf_yield is not None and _fcf_yield < 0:
                    decline_signals.append("FCF negativo (burning cash)")
                if _earn is not None and _earn < -0.15:
                    decline_signals.append(f"earnings -{ abs(_earn):.0%} y/y")

                # Sinais fundamentais positivos (qualidade intacta)
                if _rev is not None and _rev > 0.05:
                    quality_signals.append(f"revenue +{_rev:.0%}")
                if _fcf_yield is not None and _fcf_yield > 0.02:
                    quality_signals.append(f"FCF yield {_fcf_yield:.1%}")
                if _margin is not None and _margin > 0.35:
                    quality_signals.append(f"margem bruta {_margin:.0%}")
                if _roe is not None and _roe > 0.15:
                    quality_signals.append(f"ROE {_roe:.0%}")
                if _earn is not None and _earn > 0.05:
                    quality_signals.append(f"earnings +{_earn:.0%}")

            except Exception as _e:
                log.debug(f"[core] fundamentals {ticker}: {_e}")

        n_decline = len(decline_signals)
        n_quality = len(quality_signals)

        # ── Construir alertas por nível ───────────────────────────────────────
        alerts: list[str] = []

        # 🚨 RED: ≥3 sinais negativos OU (≥2 negativos E nenhum sinal de qualidade)
        # → CISCO 2001: drawdown + MA200 falling + revenue declining → 3-4 sinais negativos
        # → NVO 2025: só drawdown, MA200 still rising, revenue +30% → 1 sinal → NOT RED
        if n_decline >= 3 or (n_decline >= 2 and n_quality == 0):
            if not already_alerted_today:
                qual_str = f"\nFundamentais intactos: {', '.join(quality_signals)}" if quality_signals else ""
                alerts.append(
                    f"🚨 *COLAPSO ESTRUTURAL — {ticker}*\n"
                    f"*{n_decline} sinais negativos simultâneos:*\n"
                    + "\n".join(f"  • {s}" for s in decline_signals)
                    + qual_str + "\n\n"
                    f"*Rever a tese de investimento IMEDIATAMENTE.*\n"
                    f"_Usa `/core_close {ticker} {current_price:.2f}` se decidires sair._"
                )

        # ⚠️ ORANGE: ≥2 sinais de preço mas fundamentais ainda OK
        # → Ex: queda 40% + MA200 declining, mas revenue ainda cresce
        elif n_decline >= 2 and n_quality >= 1:
            if not already_alerted_today and not t.get("orange_alert_date"):
                t["orange_alert_date"] = today_iso
                alerts.append(
                    f"⚠️ *Deterioração CORE — {ticker}*\n"
                    f"Sinais preocupantes: {' | '.join(decline_signals)}\n"
                    f"Fundamentos ainda intactos: {', '.join(quality_signals)}\n"
                    f"P&L vs custo: {ret_vs_cost:+.1%} | Preço: ${current_price:.2f}\n"
                    f"_Monitorizar. Sem acção imediata necessária._"
                )
            elif n_decline < 2 and t.get("orange_alert_date"):
                t.pop("orange_alert_date", None)

        # 💡 YELLOW: drawdown profundo mas negócio intacto
        # → Ex: NVO a -36% 52w com GLP-1 a crescer 30% → potencial acumulação
        elif dd_from_52w_high < -0.30 and n_quality >= 2 and not t.get("deep_dip_alerted"):
            t["deep_dip_alerted"] = True
            qual_str = f" ({', '.join(quality_signals)})" if quality_signals else ""
            alerts.append(
                f"💡 *Dip Profundo em Compounder de Qualidade — {ticker}*\n"
                f"Queda {dd_from_52w_high:+.1%} do pico de 52 semanas\n"
                f"Fundamentais sólidos{qual_str}\n"
                f"Preço: ${current_price:.2f} | Custo de entrada: ${open_price:.2f} ({ret_vs_cost:+.1%})\n"
                f"_Potencial oportunidade de acumulação. Verifica a MA200 semanal._"
            )
        elif dd_from_52w_high > -0.15 and t.get("deep_dip_alerted"):
            t.pop("deep_dip_alerted", None)

        # 📉 INFO: Trend break isolado (sem outros sinais negativos)
        if days_below_ma200 >= 10 and not t.get("trend_break_alerted") and n_decline < 2:
            t["trend_break_alerted"] = True
            if not already_alerted_today:
                alerts.append(
                    f"📉 *Trend Break — {ticker}*\n"
                    f"{days_below_ma200} dias consecutivos abaixo da MA200"
                    + (f" (${ma200:.2f})" if ma200 else "") + "\n"
                    f"Preço: ${current_price:.2f} ({dd_from_52w_high:+.1%} do pico 52s)\n"
                    + (f"Fundamentais intactos: {', '.join(quality_signals)}\n" if quality_signals else "")
                    + f"_Sinal de atenção. Não é trigger de saída por si só._"
                )
        elif days_below_ma200 < 5 and t.get("trend_break_alerted"):
            t["trend_break_alerted"] = False

        # 📉 INFO: MA200 em queda secular (independente)
        if ma200_trend < -0.05 and not t.get("secular_decline_alerted") and n_decline < 2:
            t["secular_decline_alerted"] = True
            if not already_alerted_today:
                alerts.append(
                    f"📉 *Declínio Secular — {ticker}*\n"
                    f"MA200 caiu {ma200_trend:+.1%} em 60 dias\n"
                    + (f"Fundamentais intactos: {', '.join(quality_signals)}\n" if quality_signals else "")
                    + f"_Verificar se o negócio fundamental está intacto._"
                )
        elif ma200_trend > 0 and t.get("secular_decline_alerted"):
            t["secular_decline_alerted"] = False

        # 💡 Acumulação: preço >8% abaixo do custo de entrada
        if ret_vs_cost < -0.08 and not t.get("accumulation_alerted"):
            t["accumulation_alerted"] = True
            if not already_alerted_today:
                qual_str = f" (fundamentos: {', '.join(quality_signals)})" if quality_signals else ""
                alerts.append(
                    f"💡 *Oportunidade de Acumulação — {ticker}*\n"
                    f"Preço {ret_vs_cost:+.1%} abaixo do teu custo de entrada (${open_price:.2f})\n"
                    f"Actual: ${current_price:.2f}{qual_str}\n"
                    f"_Se a tese se mantém, considera reforçar a posição CORE._"
                )
        elif ret_vs_cost > -0.03 and t.get("accumulation_alerted"):
            t["accumulation_alerted"] = False

        # ── Enviar alertas ────────────────────────────────────────────────────
        if alerts:
            t["last_structural_alert_date"] = today_iso
            for alert_msg in alerts:
                if send_fn:
                    try:
                        send_fn(alert_msg)
                        alerts_sent += 1
                    except Exception:
                        pass
                log.info(f"[core] ALERT {ticker}: {alert_msg[:80]}")

    _save(trades)
    return {"checked": checked, "alerts_sent": alerts_sent}


def format_core_holdings_report() -> str:
    """Formata o relatório de posições CORE para Telegram."""
    holdings = get_core_holdings_summary()
    if not holdings:
        return "Sem posições CORE registadas."

    lines = ["*Posições CORE — Hold de Longo Prazo*\n"]
    total_invested = 0.0
    total_unrealized = 0.0
    total_yield = 0.0

    for h in holdings:
        ret_icon  = "" if h["ret_pct"] >= 0 else ""
        spy_line  = f" vs SPY {h['spy_ret_pct']:+.1f}%" if h["spy_ret_pct"] is not None else ""
        note_line = f" _{h['note']}_" if h["note"] else ""
        yld_line  = f" | Yield: €{h['yield_income_eur']:.0f} ({h['annual_yield_pct']:.1f}%/ano)" if h["annual_yield_pct"] > 0 else ""

        lines.append(
            f"{ret_icon} *{h['ticker']}* — {h['days_held']}d\n"
            f"  €{h['amount_eur']:.0f} @ ${h['open_price']:.2f} → ${h['current_price']:.2f}\n"
            f"  P&L: {h['ret_pct']:+.1f}% (€{h['unrealized_eur']:+.0f}){yld_line}\n"
            f"  Total return: {h['total_return_pct']:+.1f}%{spy_line}{note_line}\n"
        )
        total_invested   += h["amount_eur"]
        total_unrealized += h["unrealized_eur"]
        total_yield      += h["yield_income_eur"]

    avg_ret = total_unrealized / total_invested * 100 if total_invested > 0 else 0
    lines.append(
        f"*Total CORE* | Investido: €{total_invested:.0f} | "
        f"P&L: €{total_unrealized:+.0f} ({avg_ret:+.1f}%) | "
        f"Yield acumulado: €{total_yield:.0f}"
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Relatório Unificado (DipRadar vs MomentumRadar)
# ─────────────────────────────────────────────────────────────────────────────

_REPORTS_FILE = _DATA_DIR / "monthly_reports.json"


def _save_monthly_report_snapshot(report: dict) -> None:
    """Persiste snapshot mensal para consulta histórica comparativa."""
    try:
        existing = []
        if _REPORTS_FILE.exists():
            existing = json.loads(_REPORTS_FILE.read_text(encoding="utf-8"))
        month_key = date.today().strftime("%Y-%m")
        existing = [r for r in existing if r.get("month") != month_key]
        existing.append({"month": month_key, **report})
        tmp = _REPORTS_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
        tmp.replace(_REPORTS_FILE)
    except Exception as e:
        log.error(f"[paper] Erro ao guardar report mensal: {e}")


def get_unified_monthly_report(months_back: int = 3) -> dict:
    """Relatório unificado: DipRadar vs MomentumRadar vs SPY.

    Compara alpha gerado por cada módulo para validação contínua.
    Persiste automaticamente para histórico consultável.
    """
    trades  = _load()
    today   = date.today()
    cutoff  = today - timedelta(days=months_back * 30)

    closed = [
        t for t in trades
        if t.get("status", "").startswith("CLOSED")
        and t.get("close_date")
        and date.fromisoformat(t["close_date"]) >= cutoff
        and t.get("return_pct") is not None
    ]

    def _stats(subset: list[dict]) -> dict:
        if not subset:
            return {"n": 0, "avg_return": 0.0, "avg_spy": 0.0, "avg_alpha": 0.0, "win_rate": 0.0}
        rets   = [t["return_pct"] for t in subset]
        spys   = [t["spy_return_pct"] or 0 for t in subset]
        alphas = [t["alpha_pct"] or 0 for t in subset]
        wins   = sum(1 for r in rets if r > 0)
        return {
            "n":           len(subset),
            "avg_return":  round(sum(rets) / len(rets), 2),
            "avg_spy":     round(sum(spys) / len(spys), 2),
            "avg_alpha":   round(sum(alphas) / len(alphas), 2),
            "win_rate":    round(wins / len(subset), 3),
        }

    dip_trades      = [t for t in closed if t.get("trade_type", "DIP") == "DIP"]
    momentum_trades = [t for t in closed if t.get("trade_type") == "MOMENTUM"]

    report = {
        "generated_at":  today.isoformat(),
        "months_back":   months_back,
        "dip":           _stats(dip_trades),
        "momentum":      _stats(momentum_trades),
        "combined":      _stats(closed),
        "open_dip":      sum(1 for t in trades if t.get("status") == "OPEN" and t.get("trade_type", "DIP") == "DIP"),
        "open_momentum": sum(1 for t in trades if t.get("status") == "OPEN" and t.get("trade_type") == "MOMENTUM"),
    }
    _save_monthly_report_snapshot(report)
    return report


def format_unified_report(months_back: int = 3) -> str:
    """Formata o relatório comparativo DipRadar vs MomentumRadar para Telegram."""
    r = get_unified_monthly_report(months_back)
    dip = r["dip"]
    mom = r["momentum"]
    comb = r["combined"]

    def _verdict(alpha: float) -> str:
        if alpha > 3:   return "BATE O MERCADO"
        if alpha > 0:   return "ligeiramente acima"
        if alpha > -3:  return "abaixo do mercado"
        return "ABAIXO DO MERCADO"

    lines = [
        f"*Relatório Unificado — últimos {months_back} meses*",
        "",
        f"*DipRadar* ({dip['n']} trades | {r['open_dip']} abertas)",
        f"  Retorno médio : {dip['avg_return']:+.1f}%",
        f"  SPY período   : {dip['avg_spy']:+.1f}%",
        f"  Alpha         : {dip['avg_alpha']:+.1f}pp  [{_verdict(dip['avg_alpha'])}]",
        f"  Win rate      : {dip['win_rate']:.0%}",
        "",
        f"*MomentumRadar* ({mom['n']} trades | {r['open_momentum']} abertas)",
        f"  Retorno médio : {mom['avg_return']:+.1f}%",
        f"  SPY período   : {mom['avg_spy']:+.1f}%",
        f"  Alpha         : {mom['avg_alpha']:+.1f}pp  [{_verdict(mom['avg_alpha'])}]",
        f"  Win rate      : {mom['win_rate']:.0%}",
        "",
        f"*Combinado* ({comb['n']} trades)",
        f"  Alpha total   : {comb['avg_alpha']:+.1f}pp",
        f"  Win rate      : {comb['win_rate']:.0%}",
    ]

    # Histórico de reports mensais
    try:
        if _REPORTS_FILE.exists():
            history = json.loads(_REPORTS_FILE.read_text(encoding="utf-8"))
            if len(history) > 1:
                lines.append("")
                lines.append("*Histórico mensal (alpha):*")
                for snap in sorted(history, key=lambda x: x.get("month", ""))[-6:]:
                    d = snap.get("dip", {})
                    m_snap = snap.get("momentum", {})
                    lines.append(
                        f"  {snap.get('month','')}:  Dip {d.get('avg_alpha', 0):+.1f}pp  "
                        f"Mom {m_snap.get('avg_alpha', 0):+.1f}pp"
                    )
    except Exception:
        pass

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Capital Arbiter — ranking unificado por alpha_per_day
# ─────────────────────────────────────────────────────────────────────────────

def score_capital_efficiency(
    trade_type:   str,     # "DIP" | "MOMENTUM"
    expected_ret: float,   # pred_up para DIP, estimate para MOMENTUM
    horizon_days: int,     # 90 para DIP, 30 para MOMENTUM
    win_prob:     float,   # P(win) do modelo ou estimado do score
) -> float:
    """Calcula alpha_per_day esperado para comparar oportunidades DIP vs MOMENTUM.

    Fórmula: win_prob × expected_ret / horizon_days
    Esta métrica responde: "Quanto alpha esperado por dia de capital alocado?"

    Exemplo:
      DIP NVDA:  win_prob=0.65, pred_up=0.22, horizon=90 → 0.65×0.22/90 = 0.00159/dia
      MOM NOW:   win_prob=0.65, expected=0.20, horizon=30 → 0.65×0.20/30 = 0.00433/dia
      → MOM NOW tem prioridade
    """
    if horizon_days <= 0 or expected_ret <= 0:
        return 0.0
    return round(win_prob * expected_ret / horizon_days, 6)


def rank_opportunities(opportunities: list[dict]) -> list[dict]:
    """Ordena oportunidades por capital_efficiency (alpha_per_day) descendente.

    Cada oportunidade deve ter:
      trade_type:   "DIP" | "MOMENTUM"
      expected_ret: float (pred_up ou estimado do momentum score)
      horizon_days: int
      win_prob:     float
      amount_eur:   float (capital solicitado)
      ticker:       str
      + outros campos específicos do tipo

    Retorna a lista ordenada — quem vem primeiro ganha o capital disponível.
    """
    for opp in opportunities:
        opp["alpha_per_day"] = score_capital_efficiency(
            trade_type   = opp.get("trade_type", "DIP"),
            expected_ret = float(opp.get("expected_ret", 0)),
            horizon_days = int(opp.get("horizon_days", 90)),
            win_prob     = float(opp.get("win_prob", 0.5)),
        )
    return sorted(opportunities, key=lambda x: x["alpha_per_day"], reverse=True)


def allocate_capital_unified(
    opportunities: list[dict],
    available_eur: float,
    min_position_eur: float = 50.0,
) -> list[dict]:
    """Dado um conjunto de oportunidades (DIP + MOMENTUM), aloca capital na ordem óptima.

    Garante que o capital não excede o budget disponível e que posições
    pequenas demais não são abertas (min_position_eur).

    Retorna lista das oportunidades aprovadas com amount_eur final.
    """
    ranked   = rank_opportunities(opportunities)
    approved = []
    remaining = available_eur

    for opp in ranked:
        requested = float(opp.get("amount_eur", 0))
        if requested <= 0 or remaining < min_position_eur:
            break
        allocated = min(requested, remaining)
        if allocated < min_position_eur:
            break
        approved.append({**opp, "amount_eur": round(allocated, 2)})
        remaining -= allocated
        log.info(
            f"[arbiter] {opp['trade_type']} {opp['ticker']} "
            f"alpha/day={opp['alpha_per_day']:.5f} "
            f"→ €{allocated:.0f} alocado (restam €{remaining:.0f})"
        )

    return approved
