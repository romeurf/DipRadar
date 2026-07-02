"""
DipRadar — Stock Alert Bot v2.0
Scheduler: APScheduler (US-only market windows: 17h45 intraday, 21h15 EOD)
Data feed: Tiingo EOD + yfinance fallback

Variáveis Railway obrigatórias:
  TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
  TZ=Europe/Lisbon

Variáveis opcionais:
  TIINGO_API_KEY               (Chunk 3a — data feed principal)
  DROP_THRESHOLD=8
  MIN_MARKET_CAP=2000000000
  SCAN_EVERY_MINUTES=30
  MIN_DIP_SCORE=50
  TAVILY_API_KEY
  PORTFOLIO_STRESS_PCT=5
  RECOVERY_TARGET_PCT=15
  DEGRADATION_DROP_THRESHOLD=15
"""

import os
import re
import time
import logging
import requests
from pathlib import Path
from datetime import datetime, timedelta

# APScheduler — substitui `schedule` para cron jobs precisos
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron       import CronTrigger
import pytz

from data.market_client import (
    screen_global_dips, screen_structural_dips, screen_period_dips,
    get_fundamentals, get_news, get_historical_pe,
    get_52w_drawdown, get_earnings_date, get_earnings_days,
    get_catalyst, get_spy_change, is_market_open,
    get_usdeur, get_portfolio_snapshot,
    get_macro_context,
)
from portfolio.portfolio import (
    get_holdings_tuples, PPR_SHARES, PPR_AVG_COST,
    DIRECT_TICKERS, FLIP_FUND_EUR, suggest_position_size,
    get_positions, update_position_data,
    mark_degradation_alerted, reset_degradation_flag,
)
from core.sectors import get_sector_config, score_fundamentals
from core.valuation import format_valuation_block
from scanners.score import (
    calculate_dip_score, build_score_breakdown, classify_dip_category,
    is_bluechip,
    score_from_fundamentals, format_score_v2_breakdown,
)
from categories import CAT_HIGH_CONVICTION, CAT_GROWTH

# Display labels para as categorias canónicas (emojis só aqui, nunca em lógica)
_CAT_DISPLAY = {
    "CORE":             "📦 Core / ETF",
    "HIGH_CONVICTION":  "💎 Blue Chip",
    "GROWTH":           "🌱 Crescimento",
    "FLIP":             "⚡ Flip",
    "PASS":             "⏸ Aguardar",
}
from state import (
    load_alerts, save_alerts, clear_alerts,
    load_weekly_log, save_weekly_log, append_weekly_log,
    load_rejected_log, append_rejected_log,
    append_backtest_entry,
    load_recovery_watch, add_recovery_position,
    mark_recovery_alerted, remove_recovery_position,
    get_stale_recovery_positions, mark_stale_alerted,
    # ── Feature 8: Dip Persistente ──────────────────────────────────
    record_dip_day, mark_persistent_alerted, expire_missing_streaks,
)
from scanners.backtest import backtest_runner, build_backtest_summary
from scanners.alert_db import log_alert_snapshot, get_db_stats, fill_db_outcomes
import bot_commands
from portfolio.position_monitor import run_daily_check
from momentum_radar.scanner import scan_momentum_universe, format_momentum_alert, clear_sector_cache

# Chunk 3a — data feed
from data.data_feed import get_eod_prices, get_latest_price, is_tiingo_available

# Chunk 3 — universe
from core.universe import get_full_universe, get_ml_universe, ETF_TICKERS, is_etf

# Chunk 6 — ML predictor
from ml.ml_predictor import ml_score, ml_badge, is_model_ready, MLResult

# Feature 8 — Dip Persistente
from scanners.persistent_dip import check_and_alert_streak

# ── Label Resolver — Flywheel de ML ──────────────────────────────────────────
from data.label_resolver import run_label_resolver_job

# ── Backup diário ──────────────────────────────────────────────────────────────
from utils.backup_data import run_backup, build_telegram_summary as backup_telegram_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Suppress known-harmless yfinance deprecation noise. Ticker.earnings has
# been deprecated for years but several internal yfinance scrapers still
# call it — every alert/scan/conviction-eval emits the warning. Filtering
# at module level keeps logs readable. Re-enable per-call if needed via
# warnings.catch_warnings().
import warnings as _w
_w.filterwarnings(
    "ignore",
    message=r".*Ticker\.earnings.*deprecated.*",
    category=DeprecationWarning,
)

TELEGRAM_TOKEN    = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID  = os.environ.get("TELEGRAM_CHAT_ID", "")
DROP_THRESHOLD    = float(os.environ.get("DROP_THRESHOLD", "8"))
MIN_MARKET_CAP    = int(os.environ.get("MIN_MARKET_CAP", "2000000000"))
SCAN_MINUTES      = int(os.environ.get("SCAN_EVERY_MINUTES", "30"))
MIN_DIP_SCORE     = int(os.environ.get("MIN_DIP_SCORE", "50"))
STRESS_PCT        = float(os.environ.get("PORTFOLIO_STRESS_PCT", "5"))
RECOVERY_PCT      = float(os.environ.get("RECOVERY_TARGET_PCT", "15"))
DEGRADATION_DROP_THRESHOLD = int(os.environ.get("DEGRADATION_DROP_THRESHOLD", "15"))

LISBON_TZ = pytz.timezone("Europe/Lisbon")

_alerted_today:  set  = load_alerts()
_scan_running:   bool = False
_stress_alerted: set  = set()
_last_tier3:     list = []


# ── Sector ETF map ────────────────────────────────────────────────────────────

_SECTOR_ETF = {
    "Technology":             "XLK",
    "Healthcare":             "XLV",
    "Communication Services": "XLC",
    "Financial Services":     "XLF",
    "Consumer Cyclical":      "XLY",
    "Consumer Defensive":     "XLP",
    "Industrials":            "XLI",
    "Energy":                 "XLE",
    "Utilities":              "XLU",
    "Real Estate":            "XLRE",
    "Basic Materials":        "XLB",
}
_sector_etf_cache: dict = {}

def get_sector_change(sector: str) -> float | None:
    etf = _SECTOR_ETF.get(sector)
    if not etf:
        return None
    if etf in _sector_etf_cache:
        return _sector_etf_cache[etf]
    try:
        import yfinance as yf
        info = yf.Ticker(etf).fast_info
        prev = getattr(info, "previous_close", None)
        last = getattr(info, "last_price", None)
        if prev and last and prev > 0:
            chg = (last - prev) / prev * 100
            _sector_etf_cache[etf] = chg
            return chg
    except Exception as e:
        logging.warning(f"Sector ETF {etf}: {e}")
    return None


# ── Helpers de badge ──────────────────────────────────────────────────────────

def score_badge(score: float) -> str:
    if score >= 80:  return "🔥"
    if score >= 55:  return "⭐"
    return "📊"


# ── Insider buying & short interest flags ────────────────────────────────────

def get_insider_buy_flag(symbol: str) -> str:
    try:
        import yfinance as yf
        transactions = yf.Ticker(symbol).insider_transactions
        if transactions is None or transactions.empty:
            return ""
        cutoff = datetime.now(LISBON_TZ) - timedelta(days=90)
        recent = transactions[
            (transactions.index >= cutoff) &
            (transactions["Shares"].fillna(0) > 0)
        ]
        if not recent.empty:
            total_shares = int(recent["Shares"].sum())
            return f" | 👤 Insider buy: +{total_shares:,} acções (90d)"
    except Exception as e:
        logging.debug(f"Insider flag {symbol}: {e}")
    return ""

def get_short_interest_flag(fundamentals: dict) -> str:
    short_pct = fundamentals.get("short_percent_of_float") or 0
    if short_pct >= 0.20:
        return f" | ⚠️ Short {short_pct*100:.0f}% float (risco/squeeze)"
    if short_pct >= 0.10:
        return f" | 🔻 Short {short_pct*100:.0f}% float"
    return ""


# ── Telegram ──────────────────────────────────────────────────────────────────

def _md_to_html(text: str) -> str:
    """Convert Telegram Markdown v1 syntax to HTML tags.

    Handles: ```pre```, `code`, *bold*, _italic_.
    Special HTML chars (&, <, >) are escaped before tag insertion so that
    dynamic content (ticker names, error messages) never breaks the parser.
    """
    # 1. Extract pre-blocks and inline code before escaping
    pre_placeholders:  list[str] = []
    code_placeholders: list[str] = []

    def _stash_pre(m: re.Match) -> str:
        pre_placeholders.append(m.group(1))
        return f"\x00PRE{len(pre_placeholders)-1}\x00"

    def _stash_code(m: re.Match) -> str:
        code_placeholders.append(m.group(1))
        return f"\x00CODE{len(code_placeholders)-1}\x00"

    text = re.sub(r"```(.*?)```", _stash_pre,  text, flags=re.DOTALL)
    text = re.sub(r"`([^`\n]+)`",  _stash_code, text)

    # 2. Escape HTML special chars in the remaining plain text
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 3. Convert Markdown to HTML tags
    text = re.sub(r"\*([^*\n]+)\*", r"<b>\1</b>", text)
    # Italic: only match when underscore is not inside a word (avoids alpha_90d)
    text = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"<i>\1</i>", text)

    # 4. Restore code/pre blocks (content already HTML-escaped implicitly via placeholder)
    for i, content in enumerate(code_placeholders):
        safe = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00CODE{i}\x00", f"<code>{safe}</code>")
    for i, content in enumerate(pre_placeholders):
        safe = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00PRE{i}\x00", f"<pre>{safe}</pre>")

    return text


def send_telegram(message: str, retries: int = 2) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(message)
        return True
    chunks = []
    while len(message) > 4000:
        split_at = message.rfind("\n", 0, 4000)
        if split_at == -1: split_at = 4000
        chunks.append(message[:split_at])
        message = message[split_at:].lstrip("\n")
    chunks.append(message)
    url    = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    all_ok = True
    for chunk in chunks:
        html_chunk = _md_to_html(chunk)
        sent = False
        for attempt in range(retries + 1):
            try:
                r = requests.post(url, json={
                    "chat_id": TELEGRAM_CHAT_ID, "text": html_chunk,
                    "parse_mode": "HTML", "disable_web_page_preview": True,
                }, timeout=10)
                r.raise_for_status()
                sent = True
                break
            except requests.exceptions.HTTPError as e:
                # 400 = malformed HTML — fall back to plain text so the message always arrives
                if e.response is not None and e.response.status_code == 400:
                    logging.warning(
                        f"Telegram HTML 400 (attempt {attempt+1}) — "
                        f"retrying as plain text. First 120 chars: {chunk[:120]!r}"
                    )
                    try:
                        r2 = requests.post(url, json={
                            "chat_id": TELEGRAM_CHAT_ID, "text": chunk,
                            "disable_web_page_preview": True,
                        }, timeout=10)
                        r2.raise_for_status()
                        sent = True
                        break
                    except Exception as e2:
                        logging.error(f"Telegram plain-text fallback failed: {e2}")
                else:
                    logging.error(f"Telegram (attempt {attempt+1}/{retries+1}): {e}")
                if attempt < retries:
                    time.sleep(3)
            except Exception as e:
                logging.error(f"Telegram (attempt {attempt+1}/{retries+1}): {e}")
                if attempt < retries: time.sleep(3)
        if not sent: all_ok = False
        elif len(chunks) > 1: time.sleep(1)
    return all_ok


def send_telegram_photo(photo_path, caption: str = "") -> bool:
    """Send a photo file to the Telegram chat."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        from pathlib import Path as _P
        _path = _P(photo_path)
        if not _path.exists():
            return False
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        with open(_path, "rb") as _f:
            r = requests.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1024]},
                files={"photo": (_path.name, _f, "image/png")},
                timeout=30,
            )
        return r.ok
    except Exception as e:
        logging.debug(f"[send_photo] {e}")
        return False


# ── Heartbeat das 9h ──────────────────────────────────────────────────────────

def _pnl_emoji(v: float) -> str:
    return "🟢" if v > 0 else ("🔴" if v < 0 else "⚪")

def _get_snapshot() -> dict:
    usd_eur = get_usdeur()
    # PPR + ETFs are Telegram-managed (state-backed) — refresh each call
    # so /ppr set / /etf set changes take effect immediately, no restart.
    from portfolio.portfolio import _get_ppr_values
    from state import get_etfs, get_ppr
    ppr_shares, ppr_avg_cost = _get_ppr_values()
    ppr_state = get_ppr()
    ppr_current_nav = ppr_state.get("current_nav") if ppr_state else None
    etfs = get_etfs()
    return get_portfolio_snapshot(
        get_holdings_tuples(), ppr_shares, ppr_avg_cost, usd_eur,
        etfs=etfs, ppr_current_nav=ppr_current_nav,
    )

# ────────────────────────────────────────────────────────────────────────────
# Daily Digest infrastructure (Fase E — 2026-06-12)
# ────────────────────────────────────────────────────────────────────────────
# User feedback: with 11-14 alerts/day vs only 3-7 positions/month deployable,
# 98% of alerts become noise. The digest restructures delivery:
#
#   Real-time push  →  TRAILING_STOP (always)
#                      DIP/CORE alerts with score ≥ 75 AND ml_win_prob ≥ 0.65
#
#   Daily digest    →  everything else (DIP 50-74, STRUCTURAL, CORE warnings,
#                      ROTATION suggestions) buffered to disk, consolidated
#                      and emitted as one ranked message at 21:30 Lisboa.
#
# /cash is unchanged — it reads alert_db.csv directly. The digest is a
# focused daily summary so the user knows the top 5 opportunities to
# consider in /cash without scrolling through individual Telegram messages.

_DIGEST_BUFFER_FILE = "digest_buffer.json"

# Severity tiers in priority order
_REAL_TIME_ALERT_TYPES = ("TRAILING_STOP",)
_DIGEST_SCORE_GATE     = 75.0
_DIGEST_WIN_PROB_GATE  = 0.65


def _alert_severity_filter(
    alert_type: str,
    score: float | None = None,
    win_prob: float | None = None,
) -> str:
    """Return 'real_time' or 'digest' for a given alert.

    User feedback (2026-06-12): zero discrimination is the goal — everything
    that's NOT a time-critical sell signal goes to the daily digest. The
    user explicitly wants ONE consolidated notification per day instead of
    a stream of pushes throughout the trading session.

    Only TRAILING_STOP stays real-time because waiting 8 hours for the
    digest after a stop confirms means missing additional drawdown. All
    other alert types (DIP, CORE_WARNING, STRUCTURAL_DIP, future Rotation)
    funnel into run_daily_digest at 21:30.

    The previous 'high-conviction exception' (score≥75 AND ML≥65% → push
    real-time) was removed — the user prefers reading the full daily
    digest and deciding from there, even for genuinely strong signals.
    """
    if alert_type in _REAL_TIME_ALERT_TYPES:
        return "real_time"
    # score and win_prob kept in the signature for future tuning + so the
    # buffer entries still carry them for ranking inside the digest.
    _ = score, win_prob  # explicit "no longer gates"
    return "digest"


def _append_to_digest_buffer(entry: dict) -> None:
    """Atomic-append an entry to the digest buffer JSON file (/data/...)."""
    try:
        from state import _load_json, _save_json
        buf = _load_json(_DIGEST_BUFFER_FILE, default=[]) or []
        if not isinstance(buf, list):
            buf = []
        buf.append(entry)
        _save_json(_DIGEST_BUFFER_FILE, buf)
    except Exception as e:
        logging.warning(f"[digest] buffer append failed: {e}")


def _clear_digest_buffer() -> None:
    """Empty the digest buffer (called after emission)."""
    try:
        from state import _save_json
        _save_json(_DIGEST_BUFFER_FILE, [])
    except Exception as e:
        logging.warning(f"[digest] buffer clear failed: {e}")


def _emit_alert(
    text: str,
    alert_type: str,
    ticker: str,
    score: float | None = None,
    win_prob: float | None = None,
    pred_alpha_90d: float | None = None,
    in_portfolio: bool = False,
    summary: str = "",
) -> str:
    """Route an alert through the severity filter.

    Real-time alerts go straight to Telegram. Digest-bound alerts are
    buffered with metadata for ranking and final emission at 21:30.

    Returns the actual severity decision ('real_time' or 'digest') so
    callers can adjust state (e.g. mark _alerted_today differently).
    """
    severity = _alert_severity_filter(alert_type, score, win_prob)
    if severity == "real_time":
        send_telegram(text)
        return "real_time"

    # Buffer for end-of-day digest
    entry = {
        "timestamp":      datetime.now(LISBON_TZ).isoformat(timespec="seconds"),
        "ticker":         (ticker or "").upper(),
        "alert_type":     alert_type,
        "score":          float(score) if score is not None else None,
        "win_prob":       float(win_prob) if win_prob is not None else None,
        "pred_alpha_90d": float(pred_alpha_90d) if pred_alpha_90d is not None else None,
        "in_portfolio":   bool(in_portfolio),
        "summary":        summary[:200] if summary else "",
        "full_text":      text[:1000] if text else "",
    }
    _append_to_digest_buffer(entry)
    return "digest"


def run_daily_digest(clear_after: bool = True, label: str = "Digest") -> int:
    """Consolidate the digest-bound alerts.

    Parameters
    ----------
    clear_after : bool
        True for the 21:30 evening run (clear buffer after emit).
        False for the 17:45 midday run and the /digest on-demand command
        (peek, don't clear — keep accumulating for the evening run).
    label : str
        Title prefix in the message ("Digest", "Midday digest", ...).

    Reads the digest buffer, dedups by ticker (keeping the highest-α entry
    per ticker), sorts by pred_alpha_90d desc, and emits one Telegram
    message with the top 3 ranked plus a compact list of the rest.

    The /cash command stays as the primary deployment tool — this digest
    just surfaces the top opportunities.

    Returns the number of unique tickers in the digest.
    """
    try:
        from state import _load_json
        buf = _load_json(_DIGEST_BUFFER_FILE, default=[]) or []
    except Exception as e:
        logging.warning(f"[digest] load buffer failed: {e}")
        return 0

    if not buf:
        logging.info("[digest] empty buffer — nothing to emit")
        return 0

    # Dedup by ticker: keep the entry with highest pred_alpha_90d
    by_ticker: dict[str, dict] = {}
    for entry in buf:
        t = (entry.get("ticker") or "").upper()
        if not t:
            continue
        if t not in by_ticker:
            by_ticker[t] = entry
            continue
        prev = by_ticker[t]
        prev_a = prev.get("pred_alpha_90d") or -999
        curr_a = entry.get("pred_alpha_90d") or -999
        if curr_a > prev_a:
            by_ticker[t] = entry

    # Sort by alpha desc (None / negative pushed to the bottom)
    def _alpha_key(e: dict) -> float:
        v = e.get("pred_alpha_90d")
        return float(v) if v is not None else -999.0

    # Top-3 (not top-5) to match the per-bucket cap applied in /cash (201998f).
    # User reasoning: with 5 radars × top-3 cap each, the €1050 monthly budget
    # already constrains execution to ~3 real moves. Showing 5 here would be
    # inconsistent and create false sense of capacity. Top-3 = 1 action per
    # radar max, matches the budget reality.
    sorted_alerts = sorted(by_ticker.values(), key=_alpha_key, reverse=True)
    top_3         = sorted_alerts[:3]
    rest          = sorted_alerts[3:]

    try:
        from portfolio.portfolio import get_liquidity
        liquidity = get_liquidity()
    except Exception:
        liquidity = None

    # Build message
    date_str = datetime.now(LISBON_TZ).strftime("%d/%m")
    time_str = datetime.now(LISBON_TZ).strftime("%H:%M")
    header_note = "" if clear_after else f" _(snapshot às {time_str}, buffer mantido)_"
    lines = [
        f"📅 *{label} {date_str}* — {len(sorted_alerts)} oportunidades{header_note}",
        "",
    ]

    # Digest format is mode-scoped: in watchlist mode we do NOT show
    # "α esperado" / win_prob — those come from the ML layer that failed
    # the F.5 validation gate ("the numbers you learned not to trust").
    # In composite/dislocation legacy modes the OLD format is UNCHANGED.
    try:
        from core.scoring_dispatch import get_active_scorer
        _watchlist_mode = get_active_scorer() == "watchlist"
    except Exception:
        _watchlist_mode = True  # safe default: honest format

    if top_3 and _watchlist_mode:
        lines.append("🏆 *Top 3 por score*:")
        for i, e in enumerate(top_3, 1):
            ticker = e.get("ticker", "?")
            score  = e.get("score")
            score_s = f"score {score:.0f}" if score is not None else "score n/d"
            atype  = e.get("alert_type", "?")
            tag    = " 📦" if e.get("in_portfolio") else ""
            lines.append(f"  {i}. *{ticker}*{tag} [{atype}] {score_s}")
            if e.get("summary"):
                lines.append(f"     _{e['summary']}_")
        lines.append("")
    elif top_3:
        # LEGACY modes (composite/dislocation): old α-ranked format as-is
        lines.append("🏆 *Top 3 ranked por α esperado*:")
        for i, e in enumerate(top_3, 1):
            ticker = e.get("ticker", "?")
            score  = e.get("score")
            score_s = f"score {score:.0f}" if score is not None else "score n/d"
            alpha  = e.get("pred_alpha_90d")
            ann    = (float(alpha) * (365.0/90.0) * 100.0) if alpha is not None else None
            ann_s  = f"α={ann:+.1f}%/y" if ann is not None else "α n/d"
            atype  = e.get("alert_type", "?")
            tag    = " 📦" if e.get("in_portfolio") else ""
            lines.append(f"  {i}. *{ticker}*{tag} [{atype}] {score_s} {ann_s}")
            if e.get("summary"):
                lines.append(f"     _{e['summary']}_")
        lines.append("")

    if rest:
        compact = ", ".join(
            f"{e['ticker']}({(e.get('score') or 0):.0f})"
            for e in rest
        )
        lines.append(f"🔻 *Outros (não-top)*: {compact}")
        lines.append("")

    if liquidity is not None:
        if liquidity > 0:
            lines.append(f"💰 Liquidez actual: *€{liquidity:.2f}*")
            lines.append(f"   Próximo passo: `/cash {liquidity:.0f}` para proposta de allocation.")
        else:
            lines.append("💰 Sem liquidez — lista informativa apenas (sem sizing).")
        lines.append("")

    lines.append(
        "_Real-time só para sells (trailing stops) e high-conviction "
        f"(score≥{_DIGEST_SCORE_GATE:.0f}+ML≥{_DIGEST_WIN_PROB_GATE*100:.0f}%). "
        "Para detalhe: `/analisar TICKER`._"
    )

    try:
        send_telegram("\n".join(lines))
    except Exception as e:
        logging.error(f"[digest] send failed: {e}")
        return 0

    n = len(sorted_alerts)
    if clear_after:
        _clear_digest_buffer()
        logging.info(f"[digest] emitted (cleared): {len(top_3)} top + {len(rest)} rest ({n} tickers)")
    else:
        logging.info(f"[digest] emitted (peek, buffer kept): {len(top_3)} top + {len(rest)} rest ({n} tickers)")
    return n


def send_heartbeat() -> None:
    logging.info("A gerar heartbeat de carteira...")
    now_lisbon = datetime.now(LISBON_TZ)
    try:
        snapshot = _get_snapshot()
    except Exception as e:
        logging.error(f"Heartbeat snapshot: {e}")
        send_telegram(
            f"🤖 *DipRadar* ativo — {now_lisbon.strftime('%d/%m %H:%M')}\n"
            f"_Erro ao calcular carteira: {e}_"
        )
        return

    total      = snapshot["total_eur"]
    total_cost = snapshot.get("total_cost", 0)
    if total == 0:
        send_telegram(
            f"🤖 *DipRadar — Bom dia!* {now_lisbon.strftime('%d/%m/%Y')}\n"
            f"⚠️ *Variáveis da carteira em falta no Railway.*\n"
            f"_Adiciona HOLDING_*, PPR_SHARES, PPR_AVG_COST nas env vars._"
        )
        return

    pnl_d   = snapshot["pnl_day"]
    pnl_w   = snapshot["pnl_week"]
    pnl_m   = snapshot["pnl_month"]
    pnl_tot = snapshot["pnl_total"]
    fx      = snapshot["usd_eur"]

    def _pct(pnl, base):
        if base and base != 0:
            return f" ({'+' if pnl >= 0 else ''}{pnl / base * 100:.2f}%)"
        return ""

    yesterday = (now_lisbon - timedelta(days=1)).strftime("%d/%m")
    lines = [
        f"🤖 *DipRadar — Bom dia!* {now_lisbon.strftime('%d/%m/%Y')}",
        f"_USD/EUR: {fx:.4f}_",
        "",
        f"*📦 Carteira total: €{total:,.2f}*",
        "",
        f"  {_pnl_emoji(pnl_d)}  *Ontem ({yesterday}):*  €{pnl_d:+,.2f}{_pct(pnl_d, total - pnl_d)}",
        f"  {_pnl_emoji(pnl_w)}  *Semana:*            €{pnl_w:+,.2f}{_pct(pnl_w, total - pnl_w)}",
        f"  {_pnl_emoji(pnl_m)}  *Mês:*               €{pnl_m:+,.2f}{_pct(pnl_m, total - pnl_m)}",
    ]
    # FIX Bug 2: só mostra % se a base (total_cost) for fiável (> 100 €)
    if pnl_tot is not None:
        pnl_pct_str = f" ({pnl_tot / total_cost * 100:+.1f}%)" if total_cost and total_cost > 100 else ""
        lines.append(f"  {_pnl_emoji(pnl_tot)}  *P&L acumulado:*     €{pnl_tot:+,.2f}{pnl_pct_str}")

    movers = sorted(
        [p for p in snapshot["positions"] if p["pnl_day"] is not None],
        key=lambda x: x["pnl_day"], reverse=True,
    )
    if movers:
        lines += ["", f"*📈 Top movers {yesterday}:*"]
        for p in movers[:3]:
            base = p["value_eur"] - p["pnl_day"]
            pct  = p["pnl_day"] / base * 100 if base else 0
            lines.append(f"  {_pnl_emoji(p['pnl_day'])} *{p['symbol']}*: €{p['pnl_day']:+,.2f} ({pct:+.1f}%)")
        worst = movers[-1]
        if worst["pnl_day"] < 0 and worst["symbol"] != movers[0]["symbol"]:
            base = worst["value_eur"] - worst["pnl_day"]
            pct  = worst["pnl_day"] / base * 100 if base else 0
            lines.append(f"  {_pnl_emoji(worst['pnl_day'])} *{worst['symbol']}*: €{worst['pnl_day']:+,.2f} ({pct:+.1f}%) ← pior")

    lines += [
        "",
        f"  📊 PPR: €{snapshot['ppr_value']:,.2f}",
    ]

    # 'Orçamento' line removed (2026-06-12) — FLIP_FUND_EUR is the env var
    # name kept for backward compat as the monthly budget cap used inside
    # suggest_position_size, but showing it here as "disponíveis" was
    # misleading: the real available capital is dynamic (current liquidity
    # + freed cash from /cash sells). The heartbeat now leaves the deploy
    # math to /cash, which has all the live inputs.

    # Concentration denominator: stocks + liquidity, WITHOUT PPR (user
    # decision 2026-07-02, option A). The PPR is real wealth but illiquid
    # and outside the actively-managed decision space the cap protects —
    # including it made NVO read 21.2% while the triage's buy-gate
    # (correctly) read ~30%. Same basis everywhere now; the headline
    # "Carteira total" above deliberately KEEPS PPR (also user decision).
    _stocks_total = sum(
        (p.get("value_eur") or 0) for p in snapshot["positions"]
    )
    try:
        from portfolio.portfolio import get_liquidity as _hb_liq
        _stocks_total += float(_hb_liq() or 0)
    except Exception:
        pass

    if _stocks_total > 0:
        # Generic 10-step monitoring ladder (user decision 2026-07-02:
        # "de 10 em 10 — 30, 20 e 10"). Replaces the old per-ticker
        # hardcodes (NVO >= 28, ADBE "limite é 10-12%") with one policy
        # applied uniformly. NOTE: this ladder governs MONITORING alerts
        # on existing positions; the separate 25% soft / 30% hard caps
        # govern BUY eligibility in /cash + watchlist triage. Two scales,
        # two purposes — both deliberate. Weights here are LIVE values on
        # the stocks+liquidity base; triage uses COST basis on the same
        # base — small residual differences are the documented
        # cost-vs-live design note, no longer a denominator mismatch.
        concentration_warnings = []
        for pos in snapshot["positions"]:
            weight = pos["value_eur"] / _stocks_total * 100
            sym = pos["symbol"]
            if weight >= 30:
                concentration_warnings.append(
                    f"  🔴 *{sym}*: {weight:.1f}% — acima do hard cap 30%, não reforçar"
                )
            elif weight >= 20:
                concentration_warnings.append(
                    f"  🟠 *{sym}*: {weight:.1f}% — concentração elevada (≥20%)"
                )
            elif weight >= 10:
                concentration_warnings.append(
                    f"  🟡 *{sym}*: {weight:.1f}% — posição relevante (≥10%), monitorizar"
                )
        # 'Core (EUNL+PPR)' warning removed (2026-06-12). The 35% minimum
        # was hardcoded against an old asset allocation model. The user's
        # actual CORE bucket is 8 compounders (NVO, MSFT, ADBE, ADP, ...),
        # not EUNL+PPR — different concepts using the same word. The per-
        # ticker concentration checks above still cover real over-exposure.
        if concentration_warnings:
            lines += ["", "*🚨 Alertas de concentração:*"] + concentration_warnings

    earnings_alerts = []
    for sym, shares, _ in get_holdings_tuples():
        if not shares:
            continue
        # FIX Bug 1: ETFs não têm earnings — evita 404 no yfinance (ex: EUNL.DE)
        if is_etf(sym):
            continue
        clean_sym = sym.split(".")[0]
        try:
            days = get_earnings_days(clean_sym)
            if days is not None:
                dt_str = (now_lisbon + timedelta(days=days)).strftime("%d/%m")
                urgency = "🔴" if days <= 7 else "🟡" if days <= 21 else "📅"
                earnings_alerts.append(f"  {urgency} *{clean_sym}*: earnings em {days}d ({dt_str})")
        except Exception:
            pass

    if earnings_alerts:
        lines += ["", "*📅 Earnings próximos (carteira):*"] + sorted(earnings_alerts)

    # Semáforo macro
    try:
        mc_ctx = get_macro_context()
        if mc_ctx.get("summary"):
            lines += ["", mc_ctx["summary"]]
            if mc_ctx.get("alert"):
                lines.append("_⚠️ Condições macro adversas — redobrar cautela nas entradas_")
    except Exception as e:
        logging.warning(f"Heartbeat macro context: {e}")

    # Indicador Tiingo + ML
    tiingo_ok = is_tiingo_available()
    ml_ok     = is_model_ready()
    lines += [
        "",
        f"_{'🟢 Tiingo EOD activo' if tiingo_ok else '🟡 yfinance (sem chave Tiingo)'}_",
        f"_{'🤖 Modelo ML pronto' if ml_ok else '🤖 ML: modelo não treinado'}_",
        "_Mercado abre às 14h30 Lisboa_",
    ]

    send_telegram("\n".join(lines))
    logging.info("Heartbeat enviado.")


# ── Portfolio Stress Alert ────────────────────────────────────────────────────

def check_portfolio_stress() -> None:
    global _stress_alerted
    try:
        snapshot = _get_snapshot()
    except Exception as e:
        logging.warning(f"Portfolio stress snapshot: {e}")
        return

    positions = snapshot.get("positions", [])
    total_eur = snapshot.get("total_eur", 0)
    pnl_day   = snapshot.get("pnl_day", 0)

    for p in positions:
        sym   = p["symbol"]
        pnl_d = p.get("pnl_day")
        val   = p.get("value_eur", 0)
        if pnl_d is None or val == 0 or sym in _stress_alerted:
            continue
        base = val - pnl_d
        if base <= 0:
            continue
        pct_d = pnl_d / base * 100
        if pct_d <= -STRESS_PCT:
            in_portfolio = " 📦" if sym in DIRECT_TICKERS else ""
            send_telegram(
                f"🚨 *Alerta de stress: {sym}*{in_portfolio}\n"
                f"Queda de *{pct_d:.1f}%* no dia de hoje\n"
                f"P&L dia: €{pnl_d:+,.2f} | Valor actual: €{val:,.2f}\n"
                f"_⏰ {datetime.now(LISBON_TZ).strftime('%d/%m %H:%M')}_"
            )
            _stress_alerted.add(sym)
            logging.warning(f"Portfolio stress: {sym} {pct_d:.1f}%")

    if total_eur > 0 and pnl_day is not None:
        base_total = total_eur - pnl_day
        if base_total > 0:
            pct_total = pnl_day / base_total * 100
            if pct_total <= -3.0 and "_TOTAL_" not in _stress_alerted:
                _stress_alerted.add("_TOTAL_")
                spy_change = get_spy_change()
                spy_str    = f" | SPY: {spy_change:+.1f}%" if spy_change is not None else ""
                send_telegram(
                    f"🚨 *Carteira em stress macro*\n"
                    f"Queda de *{pct_total:.1f}%* no dia de hoje{spy_str}\n"
                    f"P&L dia: €{pnl_day:+,.2f} | Total: €{total_eur:,.2f}\n"
                    f"_⏰ {datetime.now(LISBON_TZ).strftime('%d/%m %H:%M')}_"
                )
                logging.warning(f"Portfolio stress MACRO: {pct_total:.1f}%")


# ── Thesis Degradation Check (EOD-aware) ──────────────────────────────────────

def check_thesis_degradation() -> None:
    """
    Check thesis degradation using clean EOD prices (Tiingo/yfinance).
    Runs alongside each US scan window (17h45 intraday, 21h15 EOD).
    US-only universe — no EU/exchange-suffix filtering needed.
    """
    positions = get_positions()
    if not positions:
        logging.info("[degradation] No active positions.")
        return

    to_check = {
        sym: pos for sym, pos in positions.items()
        if pos.get("entry_score") is not None and pos.get("category", "") != "ETF"
    }

    if not to_check:
        logging.info("[degradation] No positions to check.")
        return

    logging.info(f"[degradation] Checking {len(to_check)} positions...")

    for sym, pos in to_check.items():
        entry_score = pos["entry_score"]
        try:
            time.sleep(2)
            price = get_latest_price(sym) or 0

            fund = get_fundamentals(sym, min_market_cap=0)
            if fund.get("skip"):
                continue
            if price:
                fund["price"] = price

            earnings_days  = get_earnings_days(sym)
            sector_chg     = get_sector_change(fund.get("sector", ""))
            _ml_feat_deg   = {
                **fund,
                "spy_change":        get_spy_change(),
                "sector_etf_change": sector_chg,
                "earnings_days":     earnings_days,
            }
            _ml_res_deg  = ml_score(_ml_feat_deg, symbol=sym)
            _ml_prob_deg = _ml_res_deg.win_prob if _ml_res_deg.model_ready else None
            score, _      = calculate_dip_score(fund, sym, earnings_days, sector_change=sector_chg, ml_prob=_ml_prob_deg)
            category      = pos.get("category", "")

            update_position_data(sym, price=price, score=score, category=category)

            drop            = entry_score - score
            already_alerted = pos.get("degradation_alerted", False)

            if already_alerted and score >= entry_score - 5:
                reset_degradation_flag(sym)
                send_telegram(
                    f"♻️ *Tese recuperada: {sym}*\n"
                    f"Score actual: *{score:.0f}/100* — voltou perto do entry score ({entry_score}/100)\n"
                    f"_Flag de degradação limpa — posição estabilizada_\n"
                    f"_⏰ {datetime.now(LISBON_TZ).strftime('%d/%m %H:%M')}_"
                )
                continue

            if drop >= DEGRADATION_DROP_THRESHOLD and not already_alerted:
                entry_date = pos.get("entry_date", "N/D")
                avg_price  = pos.get("avg_price", 0)
                pnl_pct    = (price - avg_price) / avg_price * 100 if avg_price else 0
                sector_cfg = get_sector_config(fund.get("sector", ""))
                score_breakdown = build_score_breakdown(fund, sym, earnings_days, sector_change=sector_chg)

                lines = [
                    f"🧨 *Degradação de Tese: {sym}*",
                    f"_Posição em carteira desde {entry_date}_",
                    f"_Dados EOD — mercado já fechado_",
                    "",
                    f"  📉 Score entry: *{entry_score}/100*",
                    f"  📊 Score actual: *{score:.0f}/100*",
                    f"  🔻 Drop: *-{drop:.0f} pts*",
                    "",
                    f"  💰 Preço médio: ${avg_price:.2f} | Fecho EOD: ${price:.2f}",
                    f"  {_pnl_emoji(pnl_pct)} P&L posição: {pnl_pct:+.1f}%",
                    f"  🏷️ Categoria: {category} | {sector_cfg.get('label', '')}",
                    "",
                    score_breakdown,
                    "",
                    f"*⚠️ Reavalia a tese de investimento.*",
                    f"_Preço de fecho limpo — considera a tua ordem para amanhã_",
                    f"_⏰ {datetime.now(LISBON_TZ).strftime('%d/%m %H:%M')}_",
                ]
                send_telegram("\n".join(lines))
                mark_degradation_alerted(sym)
                logging.warning(f"[degradation] {sym}: drop {drop:.0f}pts (entry={entry_score} actual={score:.0f})")

        except Exception as e:
            logging.error(f"[degradation] Erro em {sym}: {e}", exc_info=True)

    logging.info("[degradation] Check complete.")


# ── Recovery Alert ────────────────────────────────────────────────────────────

def check_recovery_alerts() -> None:
    import yfinance as yf
    positions = load_recovery_watch()
    if not positions:
        return

    try:
        stale = get_stale_recovery_positions(days=60)
        for p in stale:
            sym          = p["symbol"]
            in_portfolio = " 📦" if sym in DIRECT_TICKERS else ""
            days_held    = (datetime.now(LISBON_TZ) - datetime.fromisoformat(p["date_iso"]).replace(tzinfo=LISBON_TZ)).days
            send_telegram(
                f"⏹️ *Recovery Watch — Stop Temporal: {sym}*{in_portfolio}\n"
                f"*{days_held} dias* sem atingir o target de recuperação.\n"
                f"Entrada: {p['date']} @ *${p['price_alert']:.2f}*\n"
                f"Target: *+{p['target_pct']}%* (${p['target_price']:.2f})\n"
                f"Score original: {p.get('score', 'N/A')}/100\n"
                f"_Considera fechar a posição ou manter manualmente._\n"
                f"_⏰ {datetime.now(LISBON_TZ).strftime('%d/%m %H:%M')}_"
            )
            mark_stale_alerted(sym)
    except Exception as e:
        logging.warning(f"[recovery] Stale check: {e}")

    for pos in positions:
        if pos.get("alerted"):
            continue
        sym          = pos["symbol"]
        target_price = pos.get("target_price", 0)
        price_alert  = pos.get("price_alert", 0)
        try:
            time.sleep(2)
            current = yf.Ticker(sym).info.get("regularMarketPrice") or 0
            if current and current >= target_price:
                pct_recovery = (current - price_alert) / price_alert * 100
                in_portfolio = " 📦" if sym in DIRECT_TICKERS else ""
                send_telegram(
                    f"🔔 *Recovery Alert: {sym}*{in_portfolio}\n"
                    f"Preço actual: *${current:.2f}* | Alerta foi a *${price_alert:.2f}*\n"
                    f"Recuperação: *+{pct_recovery:.1f}%* ✅\n"
                    f"_Score original: {pos.get('score', 'N/A')}/100 | Alerta em {pos.get('date', '')}_\n"
                    f"_⏰ {datetime.now(LISBON_TZ).strftime('%d/%m %H:%M')}_"
                )
                mark_recovery_alerted(sym)
        except Exception as e:
            logging.warning(f"Recovery check {sym}: {e}")


# ── Weekly structural dip scan ────────────────────────────────────────────────

def send_weekly_dip_scan() -> None:
    if datetime.now(LISBON_TZ).weekday() != 0:
        return
    logging.info("A correr weekly structural dip scan...")
    candidates = screen_structural_dips(min_drawdown_pct=25.0, min_market_cap=MIN_MARKET_CAP)
    if not candidates:
        return
    scored = []
    for stock in candidates[:40]:
        sym = stock["symbol"]
        try:
            fund = get_fundamentals(sym, min_market_cap=MIN_MARKET_CAP)
            if fund.get("skip"): continue
            earnings_days  = get_earnings_days(sym)
            sector_chg     = get_sector_change(fund.get("sector", ""))
            _ml_feat_wk    = {
                **fund,
                "spy_change":        None,
                "sector_etf_change": sector_chg,
                "earnings_days":     earnings_days,
            }
            _ml_res_wk   = ml_score(_ml_feat_wk, symbol=sym)
            _ml_prob_wk  = _ml_res_wk.win_prob if _ml_res_wk.model_ready else None
            score, rsi_str = calculate_dip_score(fund, sym, earnings_days, sector_change=sector_chg, ml_prob=_ml_prob_wk)
            if score >= 70:
                bc_flag  = is_bluechip(fund)
                category = classify_dip_category(fund, score, bc_flag)
                scored.append({
                    "symbol":         sym,
                    "score":          score,
                    "drawdown":       stock["drawdown_52w"],
                    "price":          fund.get("price") or stock["price"],
                    "mc_b":           (fund.get("market_cap") or 0) / 1e9,
                    "sector":         get_sector_config(fund.get("sector", "")).get("label", "") or fund.get("sector", ""),
                    "rsi":            rsi_str,
                    "bluechip":       bc_flag,
                    "category":       category,
                    "earnings_days":  earnings_days,
                    "analyst_upside": fund.get("analyst_upside") or 0,
                })
            time.sleep(5)
        except Exception as e:
            logging.warning(f"Weekly scan {sym}: {e}")
    if not scored:
        return
    scored.sort(key=lambda x: x["score"], reverse=True)
    lines = [
        f"*📶 Weekly Structural Dip Scan — {datetime.now(LISBON_TZ).strftime('%d/%m/%Y')}*",
        f"_Stocks ≥25% abaixo dos máximos de 52 semanas com score ≥70/100_",
        "",
    ]
    for s in scored[:12]:
        badge      = score_badge(s["score"])
        bc_tag     = " 💎" if s["bluechip"] else ""
        rsi_tag    = f" | RSI {s['rsi']}" if s["rsi"] else ""
        earn_tag   = f" | 📅 Earnings em {s['earnings_days']}d" if s["earnings_days"] is not None else ""
        upside_tag = f" | 📡 +{s['analyst_upside']:.0f}% analistas" if s["analyst_upside"] > 20 else ""
        lines.append(
            f"{badge} *{s['symbol']}*{bc_tag} — Score {s['score']:.0f}/100 | "
            f"${s['price']:.2f} | ${s['mc_b']:.1f}B | {s['drawdown']:.0f}% do topo"
        )
        lines.append(f"   _{s['category']} | {s['sector']}{rsi_tag}{earn_tag}{upside_tag}_")
        lines.append("")
    send_telegram("\n".join(lines))


# ── Structural 52w intraday DIP scan (Fase 4 — D) ────────────────────────────
#
# Companion to:
#   - screen_global_dips (intraday -8%, hourly, via run_scan)
#   - send_weekly_dip_scan (Monday-only, drawdown 25%, score 70 — intocável)
#
# Catches the gap between them: tickers that drifted -20%+ over weeks
# without any single intraday move big enough to enter Yahoo's day_losers
# list. This is the HUBS-at-52w-low case where the position is in deep
# drawdown but the daily change today is small.
#
# Anti-spam discipline:
#   - drawdown ≥ 20% (looser than weekly's 25% so it catches earlier)
#   - dip score ≥ 55 (above MIN_DIP_SCORE=50, below COMPRAR=75)
#   - ranking: score × |drawdown| desc → bigger structural signals first
#   - cap: top 3 fresh per scan run
#   - dedup: _alerted_today[f"{sym}_dip52w"] → 1 alert per ticker per day
#     even if the hourly sweep sees it five times
# Expected: 0-2/day in normal conditions; 1-3 on volatile days.

def run_structural_dip_intraday_scan() -> int:
    """Fase 4 — emit alerts for tickers in deep 52w drawdown that the
    intraday DIP screener misses. Returns count of fresh alerts sent.
    """
    try:
        candidates = screen_structural_dips(
            min_drawdown_pct=20.0,
            min_market_cap=MIN_MARKET_CAP,
            max_results=15,
        )
    except Exception as e:
        logging.warning(f"[dip52w] screener falhou: {e}")
        return 0
    if not candidates:
        return 0

    fresh = [
        c for c in candidates
        if f"{c['symbol']}_dip52w" not in _alerted_today
    ]
    if not fresh:
        return 0

    scored: list[dict] = []
    for stock in fresh:
        sym = stock.get("symbol", "")
        if not sym:
            continue
        try:
            # No explicit sleep here — _yf_info() inside get_fundamentals
            # already sleeps 10s before the first attempt + 30/50/70s on
            # retries. Adding our own would just double the per-ticker cost.
            fund = get_fundamentals(sym, min_market_cap=MIN_MARKET_CAP)
            if fund.get("skip"):
                continue
            earnings_days = get_earnings_days(sym)
            sector_chg    = get_sector_change(fund.get("sector", ""))
            score, rsi_str = calculate_dip_score(
                fund, sym, earnings_days, sector_change=sector_chg,
            )
            if score < 55:
                continue
            drawdown   = float(stock.get("drawdown_52w") or 0.0)
            rank_score = float(score) * abs(drawdown)
            bc_flag    = is_bluechip(fund)
            category   = classify_dip_category(fund, score, bc_flag)
            scored.append({
                "stock":      stock,
                "fund":       fund,
                "score":      score,
                "drawdown":   drawdown,
                "rank_score": rank_score,
                "category":   category,
                "rsi_str":    rsi_str,
            })
        except Exception as e:
            logging.debug(f"[dip52w] {sym}: {e}")
            continue

    scored.sort(key=lambda x: -x["rank_score"])
    top = scored[:3]
    if not top:
        return 0

    n_sent = 0
    for item in top:
        stock    = item["stock"]
        fund     = item["fund"]
        sym      = stock["symbol"]
        score    = item["score"]
        drawdown = item["drawdown"]
        category = item["category"]
        rsi_str  = item["rsi_str"]
        try:
            _name      = fund.get("name") or sym
            _price     = fund.get("price")
            _price_s   = f"${_price:.2f}" if _price else "?"
            _change_p  = float(stock.get("change_pct") or 0.0)
            _rsi_s     = f" | RSI: {rsi_str}" if rsi_str else ""
            _sector_s  = f" _{fund.get('sector') or ''}_" if fund.get("sector") else ""
            badge      = "🔥" if score >= 75 else ("⭐" if score >= 65 else "📉")
            msg = (
                f"📉 *STRUCTURAL DIP — {_name} ({sym})*{_sector_s}\n"
                f"_Drift 52w fora do screener intraday._\n\n"
                f"Pico 52w: *{drawdown:+.0f}%* | Hoje: *{_change_p:+.1f}%* — {_price_s}{_rsi_s}\n"
                f"{badge} DIP score: *{score:.0f}/100* | _{category}_\n\n"
                f"_Dedup: {sym}_dip52w · próxima janela 24h._"
            )
            # Structural DIPs go to the digest buffer — they're never
            # individually time-critical. The digest at 21:30 ranks them.
            _emit_alert(
                msg,
                alert_type="STRUCTURAL_DIP",
                ticker=sym,
                score=score,
                pred_alpha_90d=None,  # structural scan doesn't run ML yet
                in_portfolio=sym in DIRECT_TICKERS,
                summary=f"Pico 52w {drawdown:+.0f}%, hoje {_change_p:+.1f}%, RSI {rsi_str or '?'}",
            )
            _alerted_today.add(f"{sym}_dip52w")
            n_sent += 1
        except Exception as e:
            logging.warning(f"[dip52w] {sym}: emit failed: {e}")
            continue

    if n_sent > 0:
        save_alerts(_alerted_today)
        logging.info(f"[dip52w] {n_sent} structural alerts sent (from {len(top)} ranked)")
    return n_sent


# ── Saturday Weekly Report ────────────────────────────────────────────────────

def send_saturday_report() -> None:
    if datetime.now(LISBON_TZ).weekday() != 5:
        return
    logging.info("A gerar Saturday Weekly Report...")
    entries    = load_weekly_log()
    now        = datetime.now(LISBON_TZ)
    week_start_dt = (now - timedelta(days=now.weekday())).date()   # Segunda-feira desta semana ISO
    week_start_iso = week_start_dt.isoformat()
    week_start = week_start_dt.strftime("%d/%m")
    week_end   = (now - timedelta(days=1)).strftime("%d/%m")

    # Filtra entradas da semana actual (exclui acumulações de semanas anteriores)
    def _entry_date_iso(e: dict) -> str:
        if "date_iso" in e:
            return e["date_iso"]
        try:
            day, month = e["date"].split("/")
            return now.date().replace(month=int(month), day=int(day)).isoformat()
        except Exception:
            return "9999-99-99"

    entries = [e for e in entries if _entry_date_iso(e) >= week_start_iso]
    lines      = [f"*📊 Weekly Report — {week_start} a {week_end}*", ""]

    if not entries:
        lines.append("_Sem alertas diários esta semana._")
    else:
        by_date: dict[str, list] = {}
        for e in entries:
            by_date.setdefault(e["date"], []).append(e)
        total_alerts  = len(entries)
        comprar_count = sum(1 for e in entries if e["verdict"] == "COMPRAR")
        avg_score     = sum(e["score"] for e in entries) / total_alerts
        best          = max(entries, key=lambda x: x["score"])
        lines += [
            "*📋 Alertas da semana:*",
            f"  Total: *{total_alerts}* | COMPRAR: *{comprar_count}* | MONITORIZAR: *{total_alerts - comprar_count}*",
            f"  Score médio: *{avg_score:.1f}/100* | Melhor: *{best['symbol']}* (score {best['score']:.0f}, {best['date']})",
            "", "*🗓️ Por dia:*",
        ]
        for date_str in sorted(by_date.keys()):
            day_entries = by_date[date_str]
            lines.append(f"  *{date_str}* — {len(day_entries)} alerta(s)")
            for e in sorted(day_entries, key=lambda x: x["score"], reverse=True):
                badge = "🟢" if e["verdict"] == "COMPRAR" else "🟡"
                lines.append(
                    f"    {badge} *{e['symbol']}* Score {e['score']:.0f}/100 | "
                    f"{e['change']:+.1f}% | _{e['sector']}_ | {e.get('time','')}"
                )
        lines.append("")
        top3 = sorted(entries, key=lambda x: x["score"], reverse=True)[:3]
        lines.append("*🏆 Top 3 da semana:*")
        for i, e in enumerate(top3, 1):
            lines.append(f"  {i}. *{e['symbol']}* — Score {e['score']:.0f}/100 | {e['verdict']} | {e['date']}")
        lines.append("")

    # ── Scan heartbeat — make silence legible ─────────────────────────────
    # Aggregates per-scan diagnostics from state.scan_heartbeat for the last
    # 7 days. A week with 0 alerts is informative IF the scanner ran and
    # found scores near the threshold; suspicious if the scanner didn't run
    # or scored everything to zero. The historical baseline (alerts/month
    # over the 12 weeks of heartbeat history) tells the user whether silence
    # is normal or a regime change.
    try:
        from state import load_scan_heartbeats
        hbs = load_scan_heartbeats(days=7)
        hbs_hist = load_scan_heartbeats(days=84)  # ~12 weeks for baseline
        if hbs:
            n_runs = len(hbs)
            tot_screened = sum(int(h.get("n_screened", 0)) for h in hbs)
            tot_scored   = sum(int(h.get("n_scored", 0))   for h in hbs)
            tot_alerted  = sum(int(h.get("n_alerted", 0))  for h in hbs)
            tops   = [h["top_score"] for h in hbs if h.get("top_score") is not None]
            top_overall = max(tops) if tops else None
            thresh = next((h.get("threshold") for h in hbs if h.get("threshold") is not None), None)

            # Historical alert rate: alerts per scan-day over the rolling window
            hist_days = max(1, len({str(h.get("timestamp"))[:10] for h in hbs_hist}))
            hist_alerted = sum(int(h.get("n_alerted", 0)) for h in hbs_hist)
            alerts_per_day = hist_alerted / hist_days if hist_days else 0.0
            alerts_per_week = alerts_per_day * 5  # trading days

            lines += [
                "─" * 30, "",
                "*📡 Scan health (heartbeat):*",
                f"  Esta semana: *{n_runs}* scans corridos | "
                f"*{tot_screened}* tickers triados | "
                f"*{tot_scored}* pontuados",
                f"  Total alertas semana: *{tot_alerted}* | "
                f"Threshold: {thresh:.0f}" if thresh is not None else f"  Total alertas semana: *{tot_alerted}*",
            ]
            if top_overall is not None:
                gap = (top_overall - thresh) if thresh is not None else None
                gap_str = f" (gap {gap:+.0f} vs threshold)" if gap is not None else ""
                lines.append(f"  Top score atingido: *{top_overall:.0f}*/100{gap_str}")
            # Baseline rate — calmer reading when alerts are absent
            if hist_days >= 5:
                lines.append(
                    f"  Baseline {len({str(h.get('timestamp'))[:10] for h in hbs_hist})}d: "
                    f"≈*{alerts_per_week:.1f}* alertas/semana esperados "
                    f"(_{hist_alerted} sobre {hist_days} dias de scan_)"
                )
            lines.append("")
        else:
            lines += [
                "─" * 30, "",
                "*📡 Scan health (heartbeat):*  _Sem registos esta semana — verificar se o scheduler está activo._",
                "",
            ]
    except Exception as e:
        logging.warning(f"Heartbeat block: {e}")

    try:
        bt_block = build_backtest_summary()
        lines += ["─" * 30, "", bt_block, ""]
    except Exception as e:
        logging.warning(f"Backtest block: {e}")

    rejected = load_rejected_log()
    if rejected:
        lines += [
            "*🗑️ Rejeitados hoje:*",
            "_Stocks analisados que não cumpriram os critérios_",
            "",
        ]
        for r in sorted(rejected, key=lambda x: x.get("score") or 0, reverse=True)[:10]:
            score_str   = f" | Score {r['score']:.0f}/100" if r.get("score") is not None else ""
            verdict_str = f" | {r['verdict']}" if r.get("verdict") else ""
            lines.append(
                f"  ⛔ *{r['symbol']}* {r['change']:+.1f}% | "
                f"_{r['reason']}{score_str}{verdict_str}_"
            )
        lines.append("")

    lines += ["─" * 30, "",
              "*📉 Weekly Dips — quedas ≥10% nos últimos 7 dias:*",
              "_Top 5 com score; restantes só listados_", ""]
    try:
        period_dips = screen_period_dips(
            min_market_cap=MIN_MARKET_CAP, week_threshold=10.0, month_threshold=15.0,
        )
        weekly  = period_dips.get("weekly", [])
        monthly = period_dips.get("monthly", [])

        def _score_list(stocks, label):
            for i, s in enumerate(stocks[:10]):
                sym          = s["symbol"]
                mc_b         = (s.get("market_cap") or 0) / 1e9
                in_portfolio = " 📦" if sym in DIRECT_TICKERS else ""
                both_badge   = " 📅" if s.get("also_monthly") else ""
                if i < 5:
                    try:
                        fund = get_fundamentals(sym, min_market_cap=MIN_MARKET_CAP)
                        if not fund.get("skip"):
                            ed        = get_earnings_days(sym)
                            sec_chg   = get_sector_change(fund.get("sector", ""))
                            score, _  = calculate_dip_score(fund, sym, ed, sector_change=sec_chg)
                            badge     = score_badge(score)
                            lines.append(
                                f"  {badge} *{sym}*{in_portfolio}{both_badge}: *{s['change_pct']:.1f}%* ({label}) "
                                f"| Score {score:.0f}/100 | ${s['price']} | ${mc_b:.1f}B"
                            )
                        else:
                            lines.append(f"  📉 *{sym}*{in_portfolio}{both_badge}: *{s['change_pct']:.1f}%* ({label}) | ${s['price']} | ${mc_b:.1f}B")
                        time.sleep(4)
                    except Exception as ex:
                        logging.warning(f"Period score {sym}: {ex}")
                        lines.append(f"  📉 *{sym}*{in_portfolio}{both_badge}: *{s['change_pct']:.1f}%* ({label}) | ${s['price']} | ${mc_b:.1f}B")
                else:
                    lines.append(f"  • *{sym}*{in_portfolio}{both_badge}: {s['change_pct']:.1f}% | ${s['price']} | ${mc_b:.1f}B")

        if weekly:  _score_list(weekly, "7d")
        else:       lines.append("  _Nenhum stock com queda ≥10% na semana._")

        lines += ["", "*📉 Monthly Dips — quedas ≥15% no último mês:*",
                  "_📅 em weekly = queda ≥10% na semana E ≥15% no mês; excluídos do monthly_", ""]
        if monthly: _score_list(monthly, "30d")
        else:       lines.append("  _Nenhum stock com queda ≥15% no mês._")

    except Exception as e:
        logging.error(f"Saturday period dips: {e}")
        lines.append(f"  _Erro: {e}_")

    save_weekly_log([])
    lines += ["", "_⏰ Próximo report: sábado às 10h_"]
    send_telegram("\n".join(lines))


# ── Target Sell / Flip ────────────────────────────────────────────────────────

_SECTOR_FLIP_CAP = {
    "Technology":             0.55,
    "Communication Services": 0.45,
    "Consumer Cyclical":      0.40,
    "Healthcare":             0.35,
    "Industrials":            0.30,
    "Basic Materials":        0.30,
    "Energy":                 0.28,
    "Real Estate":            0.25,
    "Financial Services":     0.22,
    "Consumer Defensive":     0.18,
    "Utilities":              0.15,
}

def calculate_flip_target(
    fundamentals: dict,
    dip_score: float,
    earnings_date: str | None = None,
    catalyst: dict | None = None,
    spy_change: float | None = None,
    category: str | None = None,
) -> tuple[str, str]:
    price = fundamentals.get("price") or 0
    if not price or price <= 0:
        return "N/D", "SEM DADOS"

    if category == CAT_HIGH_CONVICTION or (is_bluechip(fundamentals) and dip_score >= 60):
        return "HOLD ETERNO", "💎 Blue Chip — Acumular em dips, horizonte longo"

    sector         = fundamentals.get("sector", "")
    sector_cfg     = get_sector_config(sector)
    pe_current     = fundamentals.get("pe") or 0
    pe_fair        = sector_cfg.get("pe_fair", 22)
    analyst_target = fundamentals.get("analyst_target")
    analyst_upside = fundamentals.get("analyst_upside") or 0
    beta           = fundamentals.get("beta") or 1.0
    anchors = []
    if pe_current and pe_current > 0 and pe_fair and pe_current < pe_fair:
        anchors.append(("PE rerating", price * (pe_fair / pe_current)))
    if analyst_target and analyst_target > price:
        anchors.append(("Analistas", float(analyst_target)))
    anchors.append(("Beta recovery", price * (1 + (beta or 1.0) * 0.12)))
    weights         = {"PE rerating": 2, "Analistas": 2, "Beta recovery": 1}
    total_w         = sum(weights[n] for n, _ in anchors)
    weighted_price  = sum(weights[n] * t for n, t in anchors) / total_w
    weighted_upside = (weighted_price / price) - 1
    sector_cap      = _SECTOR_FLIP_CAP.get(sector, 0.30)
    if dip_score >= 90:   sector_cap *= 1.20
    elif dip_score >= 75: sector_cap *= 1.10
    final_upside = min(weighted_upside, sector_cap)
    final_target = price * (1 + final_upside)
    macro_flag   = f" | 🌍 Queda macro SPY {spy_change:.1f}% — timeline incerta" if spy_change and spy_change <= -2.0 else ""
    if catalyst and catalyst.get("found"):
        snippet  = f": _{catalyst['snippet']}_" if catalyst.get("snippet") else ""
        cat_flag = f" | {catalyst['label']}{snippet}"
    elif earnings_date:
        cat_flag = f" | ✅ Earnings {earnings_date}"
    elif analyst_upside and analyst_upside > 20:
        cat_flag = " | 📡 Analistas veem upside forte"
    else:
        cat_flag = " | ⚠️ Sem catalisador identificado"

    cat_label = _CAT_DISPLAY.get(category, category)
    dividend_yield = fundamentals.get("dividend_yield") or 0
    dy_str = f" | 💰 Yield {dividend_yield*100:.1f}%/ano" if (category == CAT_HIGH_CONVICTION and dividend_yield > 0) else ""
    strategy = f"{cat_label}: ${final_target:.1f} (+{final_upside*100:.0f}%){dy_str}{cat_flag}{macro_flag}"

    return f"${final_target:.1f} (+{final_upside*100:.0f}%)", strategy


# ── Ranking Flip ──────────────────────────────────────────────────────────────

def build_flip_ranking(ranked_entries: list[dict], spy_change: float | None, exclude_syms: set | None = None) -> str:
    if not ranked_entries:
        return ""
    entries = [e for e in ranked_entries if not (exclude_syms and e["symbol"] in exclude_syms)]
    if not entries:
        return ""
    lines = ["", "*🏆 RANKING FLIP — Top compras de hoje*"]
    if spy_change is not None:
        sign = "+" if spy_change >= 0 else ""
        lines.append(f"_SPY hoje: {sign}{spy_change:.1f}%_")
    lines.append("")
    top = sorted(entries, key=lambda x: x["dip_score"], reverse=True)[:8]
    for i, entry in enumerate(top, 1):
        sym           = entry["symbol"]
        s             = entry["dip_score"]
        tier          = entry["tier"]
        f             = entry["fundamentals"]
        earnings      = entry.get("earnings_date")
        earnings_days = entry.get("earnings_days")
        catalyst      = entry.get("catalyst")
        category      = entry.get("category", CAT_GROWTH)
        price         = f.get("price", 0)
        mc_b          = (f.get("market_cap") or 0) / 1e9
        beta          = f.get("beta")
        in_portfolio  = " 📦" if sym in DIRECT_TICKERS else ""
        _, strategy   = calculate_flip_target(f, s, earnings, catalyst, spy_change, category=category)
        badge         = score_badge(s)
        tier_badge    = {1: "🔴T1", 2: "🟡T2", 3: "🔵T3"}.get(tier, "")
        # Sizing line is mode-scoped + liquidity-gated (per user report:
        # "€42 (40% do orçamento)" was rendered with €0 liquidity because
        # suggest_position_size never reads liquidity — it derives from
        # FLIP_FUND_EUR × score multipliers).
        #   watchlist mode: show sizing ONLY when there is actual liquidity;
        #                   otherwise an honest "sem liquidez" note.
        #   legacy modes (composite/dislocation): old behavior unchanged.
        sizing_line = None
        try:
            from core.scoring_dispatch import get_active_scorer as _gas
            _wl_mode = _gas() == "watchlist"
        except Exception:
            _wl_mode = True
        if _wl_mode:
            try:
                from portfolio.portfolio import get_liquidity as _get_liq
                _liq = _get_liq()
            except Exception:
                _liq = 0.0
            if _liq and _liq > 0:
                _, sizing_str = suggest_position_size(
                    score=s, beta=beta, earnings_days=earnings_days, spy_change=spy_change,
                )
                if sizing_str:
                    sizing_line = f"   💶 *Sizing:* {sizing_str}"
            else:
                sizing_line = "   💶 _Sem liquidez — sizing via /cash quando houver._"
        else:
            _, sizing_str = suggest_position_size(
                score=s, beta=beta, earnings_days=earnings_days, spy_change=spy_change,
            )
            if sizing_str:
                sizing_line = f"   💶 *Sizing:* {sizing_str}"

        lines.append(f"*{i}. {sym}*{in_portfolio} {tier_badge} | Score {s:.0f}/100 {badge} | {category}")
        lines.append(f"   💰 ${price} | 🏦 ${mc_b:.1f}B")
        lines.append(f"   {strategy}")
        if sizing_line:
            lines.append(sizing_line)
        lines.append("")
    return "\n".join(lines)


# ── Alerta individual ─────────────────────────────────────────────────────────

def _build_plain_summary(
    name: str,
    symbol: str,
    change: float,
    verdict: str,
    dip_score: float,
    ml_result: "MLResult | None",
    category: str,
) -> str:
    """Gera 1-2 frases em linguagem simples para o topo do alerta.

    Objectivo: qualquer pessoa percebe imediatamente o que está a acontecer
    e o que deve fazer, sem ter de decifrar os números abaixo.
    """
    # Frase 1 — o que aconteceu (sign-aware: ORCL +1.8% should NOT say
    # "caiu 1.8%"; the verb has to match the direction).
    abs_change = abs(change)
    if change < 0:
        if abs_change >= 10:
            move = f"desabou {abs_change:.0f}%"
        elif abs_change >= 5:
            move = f"caiu {abs_change:.0f}%"
        else:
            move = f"caiu {abs_change:.1f}%"
    else:
        if abs_change >= 10:
            move = f"disparou {abs_change:.0f}%"
        elif abs_change >= 5:
            move = f"subiu {abs_change:.0f}%"
        else:
            move = f"subiu {abs_change:.1f}%"

    # Sem italic _..._ — o em-dash (—) dentro de italic causa Markdown 400 em alguns stocks.
    # Plain text para a frase 2, bold apenas onde seguro.
    frase1 = f"*{name}* {move} hoje."

    # Frase 2 — o que fazer, em linguagem simples, sem repetir o score
    if verdict == "COMPRAR":
        if ml_result is not None and ml_result.model_ready and ml_result.win_prob >= 0.60:
            frase2 = f"Oportunidade de entrada — modelo confiante ({ml_result.win_prob:.0%})."
        else:
            frase2 = "Fundamentos suportam a queda. Momento de analisar entrada."

    elif verdict == "MONITORIZAR":
        if ml_result is not None and ml_result.model_ready and ml_result.win_prob > 0.55:
            frase2 = "Na watchlist — modelo optimista mas aguardar confirmacao."
        else:
            frase2 = "Ainda nao reune todos os criterios. Aguardar clareza."

    else:
        if ml_result is not None and ml_result.model_ready:
            frase2 = "O modelo nao ve vantagem clara neste momento."
        else:
            frase2 = "Sem suporte suficiente para entrada agora."

    return f"{frase1}\n{frase2}"


def build_alert(
    stock, fundamentals, historical_pe, news,
    verdict, emoji, reasons, dip_score, rsi_str,
    category: str = CAT_GROWTH,
    ml_result: MLResult | None = None,
    no_liquidity: bool = False,
) -> str:
    sector       = fundamentals.get("sector", "")
    sector_cfg   = get_sector_config(sector)
    name         = fundamentals.get("name", stock.get("name", stock.get("symbol", "N/A")))
    symbol       = stock["symbol"]
    change       = stock["change_pct"]
    price_raw    = fundamentals.get("price") or stock.get("price")
    try:
        price = f"{float(price_raw):.2f}" if price_raw not in (None, "") else "N/D"
    except (TypeError, ValueError):
        price = "N/D"
    mc_b         = (fundamentals.get("market_cap") or 0) / 1e9
    drawdown     = fundamentals.get("drawdown_from_high")
    drawdown_str = f" | 52w: {drawdown:.0f}%" if drawdown is not None else ""
    region_part  = f" ({stock['region']})" if stock.get("region") else ""
    in_portfolio = " 📦 *Já em carteira*" if symbol in DIRECT_TICKERS else ""

    # Live-price timestamp — when set by run_scan's refresh, show the HH:MM
    # at which the displayed price was sampled. Makes staleness visible: if
    # the alert fires at 19:30 but the price was sampled at 19:15, that's
    # the ~15min yfinance free-tier delay (acceptable). If sampled at 14:30
    # and alert at 16:00, something's wrong with the scan loop pacing.
    _ps = fundamentals.get("price_sampled_at")
    price_ts_str = ""
    if _ps:
        try:
            price_ts_str = f" _(às {_ps[11:16]})_"
        except Exception:
            pass

    # Gap-aware intraday change: "Hoje" is vs previous close (Yahoo standard)
    # but on a gap-and-fade day it hides the real session move. When the
    # intraday-from-open metric diverges >2pp from vs-previous, surface both
    # so the user sees the gap context. Both lines show absolute $ delta
    # alongside the % for direct mental comparison.
    _intraday    = stock.get("intraday_change_pct")
    _prev_close  = stock.get("previous_close")
    _open_price  = stock.get("open_price")
    _curr_price  = None
    try:
        _curr_price = float(price_raw) if price_raw not in (None, "") else None
    except (TypeError, ValueError):
        _curr_price = None

    # Append $ delta to the main "Hoje:" line (vs previous close)
    change_dollar_str = ""
    try:
        if _prev_close and _curr_price is not None:
            _d_close = _curr_price - float(_prev_close)
            change_dollar_str = f" (${_d_close:+.2f})"
    except (TypeError, ValueError):
        pass

    intraday_line = ""
    try:
        if _intraday is not None and abs(float(_intraday) - float(change)) > 2.0:
            _intra_dollar = ""
            if _open_price and _curr_price is not None:
                _d_open = _curr_price - float(_open_price)
                _intra_dollar = f" (${_d_open:+.2f})"
            intraday_line = f"vs abertura: *{float(_intraday):+.1f}%*{_intra_dollar}\n"
    except (TypeError, ValueError):
        pass

    badge_str  = score_badge(dip_score)
    score_line = f"{badge_str} Score: {dip_score:.0f}/100"

    vol          = fundamentals.get("volume") or 0
    avg_vol      = fundamentals.get("average_volume") or 0
    vol_flag     = " | 📈 Volume spike" if vol and avg_vol and vol > avg_vol * 1.5 else ""
    insider_flag = get_insider_buy_flag(symbol)
    short_flag   = get_short_interest_flag(fundamentals)

    sector_chg  = get_sector_change(sector)
    sector_warn = f" | 🟠 Sector {sector_chg:+.1f}% (rotation?)" if sector_chg is not None and sector_chg <= -2.0 else ""

    rsi_part = f" | RSI: {rsi_str}" if rsi_str else ""

    # Aviso de payout ratio insustentável (quando empresa paga dividendos)
    # REITs e Utilities: payout GAAP > 100% é NORMAL (usam FFO/AFFO). Não avisar.
    _div_yield    = float(fundamentals.get("dividend_yield") or 0)
    _payout_raw   = float(fundamentals.get("payoutRatio") or fundamentals.get("payout_ratio") or 0)
    _sector_lower = (sector or "").lower()
    _is_reit_like = any(s in _sector_lower for s in ("real estate", "mortgage", "utilities"))
    payout_warn = ""
    if _div_yield > 0 and _payout_raw > 1.0 and not _is_reit_like:
        _pr_pct = _payout_raw * 100
        if _payout_raw > 1.5:
            payout_warn = (
                f"\n⛔ *Payout ratio {_pr_pct:.0f}%* — paga mais em dividendos do que ganha. "
                f"Dividendo em risco de corte."
            )
        else:
            payout_warn = (
                f"\n⚠️ *Payout ratio {_pr_pct:.0f}%* — acima de 100% dos lucros GAAP. "
                f"Verifica se o FCF cobre o dividendo."
            )

    # FIX: pass sector as third argument (required by format_valuation_block)
    valuation_block = format_valuation_block(fundamentals, historical_pe, sector)

    spy_change   = get_spy_change()
    earnings_d   = get_earnings_days(symbol)
    earn_warn    = f"\n⚠️ *Earnings em {earnings_d} dias!* — Cautela" if earnings_d is not None and earnings_d < 14 else ""
    catalyst_data = None
    try:
        catalyst_data = get_catalyst(symbol, name)
    except Exception:
        pass
    _, strategy = calculate_flip_target(
        fundamentals, dip_score,
        earnings_date=get_earnings_date(symbol),
        catalyst=catalyst_data,
        spy_change=spy_change,
        category=category,
    )

    macro_ctx = ""
    try:
        mc = get_macro_context()
        if mc.get("alert"):
            macro_ctx = f"\n🌎 *Macro:* {mc['summary']}"
    except Exception:
        pass

    ml_label = ml_result.label if (ml_result is not None) else None
    # Pass ml_prob so the displayed `Score: X/100` actually reflects the ML
    # weight. Without it, build_score_breakdown internally hits
    # score_from_fundamentals with ml_prob=None, ml_weight=1.0, and the
    # score collapses to the pure rule-based number — ignoring the ML badge
    # rendered just below. User reported: "os scores parecem-me iguais com
    # ou sem modelo" (2026-06-12) — exactly because of this.
    ml_prob_for_score = (
        ml_result.win_prob
        if (ml_result is not None and ml_result.model_ready and ml_result.win_prob is not None)
        else None
    )
    score_breakdown = build_score_breakdown(
        fundamentals, symbol, earnings_d,
        sector_change=sector_chg,
        ml_prob=ml_prob_for_score,
        ml_label=ml_label,
    )

    _, sizing_str = suggest_position_size(
        score=dip_score,
        beta=fundamentals.get("beta"),
        earnings_days=earnings_d,
        spy_change=spy_change,
    )

    # ML badge (Chunk 6)
    ml_line = ""
    if ml_result is not None:
        ml_line = f"\n{ml_badge(ml_result)}"

    # Narrativa de sinais (Chunk 6b) — explica o PORQUÊ em linguagem natural
    signal_narrative_text = ""
    try:
        from core.signal_narrative import generate_signal_narrative, generate_8k_veto
        from ml.ml_features import _FALLBACK as _mf_fallback
        _feats = {**fundamentals} if fundamentals else {}
        _eight_k = _feats.get("recent_8k_score") or _mf_fallback.get("recent_8k_score")
        # Detalhes insider e 8-K (nome, cargo, descrição exacta)
        try:
            from data.fundamental_signals import get_insider_details, get_8k_details
            _ins_det = get_insider_details(symbol)
            _8k_det  = get_8k_details(symbol)
        except Exception:
            _ins_det = {}
            _8k_det  = {}

        _narrative = generate_signal_narrative(
            pred_alpha_90d        = ml_result.pred_up     if ml_result and ml_result.model_ready else None,
            win_prob              = ml_result.win_prob     if ml_result and ml_result.model_ready else None,
            pred_drawdown         = ml_result.pred_down    if ml_result and ml_result.model_ready else None,
            insider_recent        = _feats.get("insider_buy_recent"),
            insider_amount_score  = _feats.get("insider_buy_amount_score"),
            insider_buy_amount_usd = _feats.get("insider_buy_amount_usd") or _ins_det.get("amount"),
            insider_name          = _ins_det.get("name"),
            insider_title         = _ins_det.get("title"),
            eight_k_score         = _eight_k,
            eight_k_description   = _8k_det.get("description"),
            short_trend           = _feats.get("short_interest_trend"),
            consecutive_red       = _feats.get("consecutive_red_days"),
            ma_200d_ratio         = _feats.get("ma_200d_ratio"),
            stock_type            = ml_result.stock_type if ml_result and ml_result.model_ready else "SPECULATIVE",
            ticker                = symbol,
        )
        if _narrative:
            signal_narrative_text = f"\n\n{_narrative}"
    except Exception:
        pass

    # Tema/trend (Chunk 7) — identifica se o stock está num sector em trend
    theme_line = ""
    try:
        from scanners.themes import get_stock_themes, format_theme_tag
        matched_themes = get_stock_themes(
            ticker=symbol,
            sector=sector,
            company_name=name,
        )
        if matched_themes:
            theme_line = f"\n{format_theme_tag(matched_themes)}"
    except Exception:
        pass

    # ── Resumo em linguagem simples (1-2 frases no topo) ─────────────────────
    plain_summary = _build_plain_summary(
        name=name, symbol=symbol, change=change,
        verdict=verdict, dip_score=dip_score,
        ml_result=ml_result, category=category,
    )

    body = (
        f"{emoji} *{name} ({symbol})*{region_part}{in_portfolio}\n"
        f"{plain_summary}\n"
        f"{sector_cfg.get('label', sector)} | ≈{mc_b:.1f}B\n"
        f"Hoje: *{change:+.1f}%*{change_dollar_str} — ${price}{price_ts_str}{drawdown_str}{vol_flag}{insider_flag}{short_flag}{rsi_part}{sector_warn}\n"
        f"{intraday_line}"
        f"{payout_warn}"
        f"{earn_warn}"
        f"{valuation_block}"
        f"{theme_line}"
        f"\n{score_breakdown}"   # Score block immediately after fundamentals
        f"{ml_line}"             # ML logo após o score
        f"{signal_narrative_text}\n"
        f"\n*Estratégia:* {strategy}"
        f"{macro_ctx}"
    )
    # In watchlist mode the sizing gate consults the REAL liquidity ledger
    # (portfolio.get_liquidity), not the monthly-budget arithmetic that
    # `no_liquidity` was derived from — the budget resets on day 1 and made
    # sizing "€42 (40% do orçamento)" render with €0 actual cash (user
    # report 2026-07-02). Legacy modes keep the old no_liquidity semantics.
    _real_liq_ok = True
    try:
        from core.scoring_dispatch import get_active_scorer as _gas_alert
        if _gas_alert() == "watchlist":
            from portfolio.portfolio import get_liquidity as _gl_alert
            _real_liq_ok = (_gl_alert() or 0) > 0
    except Exception:
        pass

    if no_liquidity or not _real_liq_ok:
        body += "\n⚠️ _Sem liquidez disponível — alerta para referência e monitorização. Sem compra executada._"
    elif sizing_str and verdict not in ("EVITAR", "REJEITAR"):
        # Suppress sizing when the verdict is reject — showing "💶 Sizing: €20"
        # right after "Rejeitado: sem suporte técnico" is contradictory and
        # confused users into thinking the bot was recommending a buy.
        body += f"\n💶 *Sizing:* {sizing_str}"

    # Cash reserve hint: when /cash recommended holding €X liquid (no signals
    # at that time) and now a high-conviction COMPRAR fires, remind the user
    # that the capital is available. The system NEVER auto-buys — just shows
    # the reserve so the user can decide manually.
    if verdict == "COMPRAR":
        try:
            from state import get_cash_reserve_eur
            reserve = get_cash_reserve_eur()
            if reserve >= 50:
                body += (
                    f"\n📥 _Reserva disponível: €{reserve:.0f} "
                    f"(definida pelo último /cash). Considera deploy parcial._"
                )
        except Exception:
            pass

    return body


# ── On-demand ticker analysis (usado pelo /analisar command) ──────────────────

def analyze_ticker(symbol: str) -> str:
    symbol = symbol.upper().strip()
    try:
        fund = get_fundamentals(symbol, min_market_cap=0)
        if fund.get("skip"):
            skip_reason = fund.get("skip_reason", "Ticker sem dados disponíveis")
            # Diagnóstico extra: testar se yfinance responde para este ticker
            yf_ok = False
            try:
                import yfinance as yf
                fi = yf.Ticker(symbol).fast_info
                yf_ok = bool(getattr(fi, "last_price", None))
            except Exception:
                pass
            diag = "_yfinance: OK_" if yf_ok else "_yfinance: sem dados — possível ticker errado ou problema de rede_"
            return (
                f"⚠️ *{symbol}* — dados insuficientes\n"
                f"_{skip_reason}_\n\n"
                f"Diagnóstico: {diag}\n"
                f"_Verifica se o ticker está correcto (ex: `AAPL`, `NVO`, `EUNL.DE`)_"
            )

        hist_pe       = get_historical_pe(symbol)
        news          = get_news(symbol)
        earnings_days = get_earnings_days(symbol)
        sector_chg    = get_sector_change(fund.get("sector", ""))
        spy_change    = get_spy_change()
        score, rsi_str = calculate_dip_score(fund, symbol, earnings_days, sector_change=sector_chg)

        bc_flag  = is_bluechip(fund)
        category = classify_dip_category(fund, score, bc_flag)

        price      = fund.get("price") or 0
        change_pct = 0.0
        intraday_change_pct = None
        try:
            from data.data_feed import get_live_price
            _live = get_live_price(symbol)
            if _live and _live.source != "eod_fallback":
                price = _live.price
                fund["price"]            = _live.price
                fund["price_sampled_at"] = _live.sampled_at
                if _live.change_pct is not None:
                    change_pct = _live.change_pct
                # Capture intraday-from-open (gap-aware): when the session
                # opens with a big gap vs previous close, "vs ontem" hides
                # the real intraday move. Show both metrics when they
                # diverge meaningfully (>2pp).
                if _live.intraday_change_pct is not None:
                    intraday_change_pct = _live.intraday_change_pct
        except Exception:
            pass

        # ML score (Chunk 6)
        ml_features = {
            **fund,
            "score":            score,
            "spy_change":       spy_change,
            "sector_etf_change": sector_chg,
            "earnings_days":    earnings_days,
            "change_day_pct":   change_pct,
        }
        ml_result = ml_score(ml_features, symbol=symbol)

        # ── Score engine ─────────────────────────────────────────
        ml_prob_score = ml_result.win_prob if ml_result.model_ready else None
        score_data = score_from_fundamentals(fund, ml_prob=ml_prob_score)
        score      = score_data["final_score"]
        logging.info(
            f"[score/analisar] {symbol}: {score:.1f} | "
            f"skip={score_data.get('skip_recommended')} "
            f"trap={score_data.get('is_value_trap')}"
        )

        # Pass previous_close + today's open through to build_alert so it
        # can show both `Hoje: -1.4% (-$2.60)` and
        # `vs abertura: -10.8% (-$21.68)` for direct comparison.
        _prev_close_for_display = None
        _open_for_display       = None
        try:
            if _live is not None:
                _prev_close_for_display = _live.previous_close
                _open_for_display       = _live.open_price
        except NameError:
            pass

        stock_dict = {
            "symbol":   symbol,
            "change_pct": change_pct,
            "intraday_change_pct": intraday_change_pct,
            "previous_close": _prev_close_for_display,
            "open_price":     _open_for_display,
            "region":   "",
        }

        if score >= 75:
            verdict, emoji_str = "COMPRAR", "🟢"
        elif score >= 60:
            verdict, emoji_str = "MONITORIZAR", "🟡"
        elif score >= MIN_DIP_SCORE:
            verdict, emoji_str = "MONITORIZAR", "🔵"
        else:
            verdict, emoji_str = "EVITAR", "🔴"

        alert_text = build_alert(
            stock_dict, fund, hist_pe, news,
            verdict, emoji_str, [], score, rsi_str,
            category=category,
            ml_result=ml_result,
        )

        header = f"🔍 *Análise on-demand — {symbol}*\n_Pedido via /analisar — {datetime.now(LISBON_TZ).strftime('%d/%m %H:%M')}_\n\n"
        appendix = _build_allocation_appendix(symbol)
        return header + alert_text + appendix

    except Exception as e:
        logging.error(f"analyze_ticker({symbol}): {e}", exc_info=True)
        return f"❌ *Erro ao analisar {symbol}*\n_`{e}`_"


def _build_allocation_appendix(symbol: str) -> str:
    """Compose a screener allocation-context section appended to /analisar.

    Shows composite z-score breakdown, held-cash from last /cash, current
    portfolio position + concentration headroom, ranking vs the current
    queue (universe scan), and suggested sizing if user forced a deploy
    right now. Returns empty string on any failure so /analisar still
    works if this is broken.
    """
    try:
        import json as _J
        from pathlib import Path as _P
        from core.composite_score import softmax_sizing
        from core.scoring_dispatch import ScoringSession

        # Per-call session: one flag read, one engine for the appendix +
        # ranking. The engine tag is shown in the output so the user
        # knows which scorer's verdict they're looking at.
        session = ScoringSession()

        # Fetch signals for this ticker so the scorer can run. We reuse
        # the existing core_radar plumbing (fetch hist + quality + scoring)
        # rather than duplicating the data path.
        try:
            from core.core_radar import (
                _fetch_hist, _score_core_dip,
            )
            hist = _fetch_hist(symbol)
            if hist is None or hist.empty:
                return (
                    "\n\n─────────────────────────\n"
                    "💰 *Allocation context*\n"
                    "_Score indisponível — sem fundamentals ou price hist._"
                )
            close = hist["Close"]
            volume = hist["Volume"] if "Volume" in hist.columns else None
            current_price = float(close.iloc[-1])
            _score_x, signals = _score_core_dip(
                close, current_price, 0.0,
                ticker=symbol, volume=volume,
            )
            # _score_core_dip early-returns a sparse signals dict when the
            # ticker doesn't qualify as a dip (its quality fetch happens
            # AFTER the dip gate). /analisar is a diagnostic command — the
            # criteria check must show REAL numbers ("GM 10% < median"),
            # never "missing data" for data that exists. Top-up quality
            # fields with a direct fetch when the gate skipped them
            # (user report 2026-07-02, LMT case).
            if signals.get("roe") is None and signals.get("gross_margin") is None:
                try:
                    from core.core_radar import _fetch_quality_metrics
                    _q = _fetch_quality_metrics(symbol) or {}
                    for _k_src, _k_dst in (
                        ("roe", "roe"),
                        ("gross_margin", "gross_margin"),
                        ("revenue_growth", "revenue_growth"),
                        ("fcf_yield", "fcf_yield_fund"),
                    ):
                        if signals.get(_k_dst) is None and _q.get(_k_src) is not None:
                            signals[_k_dst] = _q[_k_src]
                except Exception as _qe:
                    logging.debug(f"_build_allocation_appendix({symbol}) quality top-up: {_qe}")
        except Exception as e:
            logging.warning(f"_build_allocation_appendix({symbol}) signals: {e}")
            return ""

        result = session.score(symbol, signals)

        # Held cash from last /cash run
        _dd = _P("/data") if _P("/data").exists() else _P(__file__).parent / "data"
        _watchdog_state_path = _dd / "watchdog_state.json"
        held_cash = 0.0
        last_cash_ts = None
        if _watchdog_state_path.exists():
            try:
                ws = _J.loads(_watchdog_state_path.read_text())
                held_cash = float(ws.get("held_cash_eur") or 0.0)
                last_cash_ts = ws.get("timestamp")
            except Exception:
                pass

        # Portfolio context
        try:
            from portfolio.portfolio import get_positions
            positions = get_positions() or {}
        except Exception:
            positions = {}
        # /sync_portfolio stores `total_cost` in USD (shares × USD avg_price);
        # value_eur/cost_eur are legacy/alternate fields. Convert to EUR so
        # display labels are honest and headroom math doesn't mix currencies
        # with EUR budgets. get_usdeur() has a built-in 0.92 fallback.
        from data.market_client import get_usdeur
        _usdeur = get_usdeur()
        portfolio_value = sum(
            float(p.get("total_cost") or p.get("value_eur") or p.get("cost_eur") or 0)
            for p in positions.values()
        ) * _usdeur
        ticker_pos = positions.get(symbol, {}) if isinstance(positions, dict) else {}
        current_pos_eur = float(
            ticker_pos.get("total_cost")
            or ticker_pos.get("value_eur")
            or ticker_pos.get("cost_eur")
            or 0
        ) * _usdeur
        current_pct = (current_pos_eur / portfolio_value) if portfolio_value > 0 else 0.0
        cap_eur = portfolio_value * 0.30 if portfolio_value > 0 else 0.0
        headroom = max(0.0, cap_eur - current_pos_eur) if portfolio_value > 0 else float("inf")

        # ── Watchlist mode: criteria pass/fail, no sizing ──────────────
        # Honest mode doesn't suggest allocations; it just tells you if
        # this name passes the watchlist filter and how it compares on
        # the criteria. Sizing is for legacy modes only.
        if session.scorer == "watchlist":
            entry = result.breakdown  # WatchlistEntry
            lines = ["\n\n─────────────────────────"]
            lines.append("📋 *Watchlist context*  ·  engine: watchlist (honest filter)")
            lines.append("")

            # Pass/fail per criterion
            quality_ok = "✅" if entry.quality_passes else "❌"
            dd_val = entry.dd_from_52w_high
            dd_ok = "✅" if (dd_val is not None and dd_val <= -0.25) else "❌"
            cd_val = entry.days_to_catalyst
            cat_ok = "✅" if (cd_val is not None and -10 <= cd_val <= 30) else "❌"

            lines.append(f"{quality_ok} Quality gate: {entry.quality_reason or 'pass'}")
            lines.append(f"{dd_ok} DD from 52w high: {dd_val*100:+.0f}% (need <= -25%)"
                         if dd_val is not None else f"{dd_ok} DD n/a")
            if cd_val is None:
                lines.append(f"{cat_ok} Catalyst: n/a (need within -10d to +30d)")
            elif cd_val < 0:
                lines.append(f"{cat_ok} Catalyst: reported {-cd_val}d ago (need within -10d to +30d)")
            else:
                lines.append(f"{cat_ok} Catalyst: {cd_val}d ahead (need within -10d to +30d)")

            if entry.sector_relative_60d is not None:
                sr_pp = entry.sector_relative_60d * 100
                sr_note = (
                    "fell with market — not a specific dislocation"
                    if abs(entry.sector_relative_60d) < 0.05
                    else "specific underperformance vs sector — investigate why"
                    if entry.sector_relative_60d < -0.05
                    else "outperformed sector — not a dislocation"
                )
                lines.append(f"ℹ️ vs sector 60d: {sr_pp:+.0f}pp  _({sr_note})_")

            lines.append("")
            qualifies_str = ("✅ *QUALIFIES* for watchlist" if entry.qualifies
                             else "⚪ *Does not qualify* — at least one criterion fails")
            lines.append(qualifies_str)

            # Portfolio + held cash context (informational only — no sizing)
            if portfolio_value > 0:
                lines.append("")
                lines.append(f"Posição actual: €{current_pos_eur:,.0f} ({current_pct*100:.1f}% portfolio)")
                if headroom != float("inf"):
                    lines.append(f"Headroom até cap 30%: €{headroom:,.0f}")
            if held_cash > 0:
                _ts_short = (last_cash_ts or "?")[:10]
                lines.append(f"Cash retido (último /cash {_ts_short}): *€{held_cash:,.0f}*")
                lines.append("_Watchlist mode não sugere alocação — decisão é tua._")

            return "\n".join(lines)

        # Ranking vs current universe via the SAME session (consistent engine)
        try:
            from core.core_radar import scan_core_universe
            opps = scan_core_universe(min_score=0.0)
        except Exception:
            opps = []

        ranked = []  # list of (ticker, z)
        if opps:
            try:
                for o in opps:
                    r = session.score(o["ticker"], o)
                    if r.hard_excluded or r.z is None:
                        continue
                    ranked.append((o["ticker"], r.z))
                ranked.sort(key=lambda x: -x[1])
            except Exception as e:
                logging.debug(f"_build_allocation_appendix ranking: {e}")
                ranked = []

        # Sizing if forced deploy with held_cash (or hypothetical €1000)
        budget = held_cash if held_cash >= 100 else 1000.0
        rank_position = None
        suggested_eur = None
        if ranked:
            for i, (t, _z) in enumerate(ranked):
                if t == symbol:
                    rank_position = i + 1
                    break
            allocs, _skipped = softmax_sizing(
                ranked,
                budget_eur=budget,
                threshold_z=0.3,
                min_alloc_eur=100.0,
                concentration_cap_pct=0.30,
                portfolio_value_eur=portfolio_value,
                current_positions_eur={
                    t: float((p.get("total_cost") or p.get("value_eur") or p.get("cost_eur") or 0)) * _usdeur
                    for t, p in positions.items() if isinstance(p, dict)
                },
            )
            suggested_eur = allocs.get(symbol)

        engine_tag = {
            "watchlist":   "📋 watchlist",
            "dislocation": "🧠 dislocation_z (experimental)",
            "composite":   "📐 composite_z (legacy)",
        }.get(session.scorer, "?")

        lines = ["\n\n─────────────────────────"]
        lines.append(f"💰 *Allocation context (screener)*  ·  engine: {engine_tag}")
        lines.append("")

        if result.hard_excluded:
            lines.append(f"🚫 *HARD-EXCLUDED*: {result.exclude_reason}")
            lines.append("Sistema NÃO sugere deploy mesmo com cash retido.")
            return "\n".join(lines)

        # Per-scorer breakdown via the unified explanation_lines helper
        lines.append(result.summary_line)
        for expl_line in result.explanation_lines(threshold_abs=0.3)[1:]:
            lines.append(expl_line)

        if rank_position is not None:
            lines.append(
                f"Ranking actual: *#{rank_position}* de {len(ranked)} (CORE universe)"
            )

        # Portfolio + held cash + headroom
        if portfolio_value > 0:
            lines.append(
                f"Posição actual: €{current_pos_eur:,.0f} "
                f"({current_pct*100:.1f}% portfolio)"
            )
            if headroom == float("inf"):
                lines.append("Headroom cap 30%: ilimitado (sem portfolio_value)")
            else:
                lines.append(f"Headroom até cap 30%: €{headroom:,.0f}")
        if held_cash > 0:
            _ts_short = (last_cash_ts or "?")[:10]
            lines.append(f"Cash retido (último /cash {_ts_short}): *€{held_cash:,.0f}*")

        # Suggested sizing
        if suggested_eur is not None:
            lines.append("")
            lines.append(f"💡 *Se forças deploy agora*: €{suggested_eur:.0f}")
            lines.append(f"   `/deployed {symbol} {suggested_eur:.0f} livre vontade`")
        elif result.z is not None and result.z < 0.3:
            lines.append("")
            lines.append(f"⚠️ Below threshold (z<0.3) — sistema sugere *NÃO deploy* agora.")
        elif portfolio_value > 0 and current_pct >= 0.30:
            lines.append("")
            lines.append(f"🚫 Concentration cap 30% atingida — sem headroom para top-up.")

        return "\n".join(lines)
    except Exception as e:
        logging.warning(f"_build_allocation_appendix({symbol}): {e}")
        return ""


# ── Cash watchdog ─────────────────────────────────────────────────────────────

def run_watchdog_scan() -> None:
    """Daily watchdog: notify when held cash has new screener qualifiers.

    Triggered by APScheduler cron at 18:00 Lisbon (after US open settles).
    Pipeline:
      1. Read /data/watchdog_state.json. If held_cash_eur <= 100€, exit silently.
      2. Walk CORE universe, compute composite z-score per ticker.
      3. Find qualifiers (z > 0.3) NOT in deployed_tickers and NOT in notified_tickers.
      4. If found, send Telegram push: top 3 with breakdown + headlines + suggested
         allocation softmaxed over the held cash.
      5. Append notified tickers to state so we don't spam the same name twice.
         Reset notified_tickers when held_cash_eur drops to 0 (new /cash run).

    Spam control: a previously-notified ticker is re-notified only if its
    composite_z has gained ≥+0.5 since the last notification (significant
    deepening of the opportunity).
    """
    try:
        from pathlib import Path
        import json
        from core.composite_score import softmax_sizing
        from core.core_radar import scan_core_universe
        from core.scoring_dispatch import ScoringSession
        from utils.news_headlines import fetch_headlines, format_headlines_for_telegram

        _data_dir = Path("/data") if Path("/data").exists() else Path(__file__).resolve().parent / "data"
        ws_path = _data_dir / "watchdog_state.json"

        if not ws_path.exists():
            logging.info("[watchdog] no state file — skipping (no /cash run yet)")
            return
        try:
            state = json.loads(ws_path.read_text())
        except Exception as e:
            logging.warning(f"[watchdog] state unreadable: {e}")
            return

        held_cash = float(state.get("held_cash_eur") or 0.0)
        if held_cash < 100:
            logging.info(f"[watchdog] held_cash €{held_cash:.0f} < €100 — skipping")
            return

        deployed_tickers = set(state.get("deployed_tickers") or [])
        notified = state.get("notified_tickers") or []
        # notified_tickers is stored as either list of strings (legacy) or
        # list of {"ticker": str, "z_at_notify": float} dicts (current).
        # Normalize.
        notified_map = {}
        for item in notified:
            if isinstance(item, str):
                notified_map[item] = -10.0  # unknown — treat as "always notify on movement"
            elif isinstance(item, dict):
                notified_map[item.get("ticker", "")] = float(item.get("z_at_notify") or -10.0)

        # ScoringSession reads DIPRADAR_SCORING_MODE flag ONCE so the
        # watchdog uses the same engine as /scan_now interactive scans
        # and /cash. One flag, all paths — never alert with one logic
        # and respond with another.
        session = ScoringSession()

        # ── Watchlist mode (default after F.5 REVERT) ─────────────────
        # Direct path through honest_watchlist for clarity. No z-based
        # ranking. "New qualifier" = ticker in current watchlist AND
        # not in notified set AND not already deployed AND triage tier
        # CLEAN or BORDERLINE (user decision 2026-07-02: borderlines
        # notify too — they exist FOR the human — tier-tagged so both
        # are distinguishable at a glance; EXCLUDED never notifies).
        if session.scorer == "watchlist":
            from core.honest_watchlist import get_watchlist
            try:
                from portfolio.portfolio import build_portfolio_context
                from data.market_client import get_usdeur as _gu_wd
                _pf_ctx = build_portfolio_context(usd_eur=_gu_wd())
            except Exception as _ctx_err:
                logging.warning(f"[watchdog] portfolio_context failed: {_ctx_err}")
                _pf_ctx = None
            entries = get_watchlist(portfolio_context=_pf_ctx)
            previously_notified = set(notified_map.keys())
            new_qualifiers_wl = [
                e for e in entries
                if e.ticker not in deployed_tickers
                and e.ticker not in previously_notified
                and e.triage_tier in ("CLEAN", "BORDERLINE")
            ]
            if not new_qualifiers_wl:
                logging.info(
                    f"[watchdog] watchlist: 0 new qualifiers "
                    f"(held_cash €{held_cash:.0f}, total qualifiers={len(entries)}, "
                    f"already notified/deployed/excluded)"
                )
                return

            from datetime import datetime
            ts = datetime.now(LISBON_TZ).strftime("%Y-%m-%d %H:%M")
            top3 = new_qualifiers_wl[:3]  # already sorted by DD depth in get_watchlist
            lines = [
                f"📋 *Watchlist Watchdog* — `{ts}`  ·  engine: 📋 watchlist",
                f"_Cash retido: *€{held_cash:,.0f}* (do último /cash)_",
                "",
                f"🚨 *{len(new_qualifiers_wl)} novo(s) qualifier(s)* (não-deployed, não-notificado):",
                "",
            ]
            for e in top3:
                _tier_tag = "✅ CLEAN" if e.triage_tier == "CLEAN" else "🟡 BORDERLINE"
                lines.append(f"{_tier_tag}  `{e.summary_line}`")
                if e.triage_tier == "BORDERLINE":
                    for r in e.triage_reasons:
                        lines.append(f"    · _{r}_")
                hl = fetch_headlines(e.ticker, max_results=2)
                if hl:
                    lines.append(format_headlines_for_telegram(hl, max_results=2, indent="  "))
                lines.append("")
            if len(new_qualifiers_wl) > 3:
                lines.append(
                    f"_+ {len(new_qualifiers_wl) - 3} mais qualifiers — corre `/scan_now` para ver todos_"
                )
                lines.append("")
            lines.append(
                "_Watchlist mode: filter only, no edge ranking._ "
                "_Deploy: `/deployed TICKER AMOUNT reason`. Investigate: `/analisar TICKER`._"
            )
            send_telegram("\n".join(lines))
            logging.info(
                f"[watchdog] watchlist pushed {len(new_qualifiers_wl)} qualifiers, "
                f"deepest DD: {top3[0].ticker} {top3[0].dd_from_52w_high*100:+.0f}%"
            )
            # Update notified state — store ticker only (z_at_notify=None for watchlist)
            new_notified = list(notified_map.items())
            for e in top3:
                existing_idx = next(
                    (i for i, (t, _z) in enumerate(new_notified) if t == e.ticker), None
                )
                if existing_idx is not None:
                    new_notified[existing_idx] = (e.ticker, None)
                else:
                    new_notified.append((e.ticker, None))
            state["notified_tickers"] = [
                {"ticker": t, "z_at_notify": z} for t, z in new_notified
            ]
            state["last_watchdog_run"] = datetime.now().isoformat(timespec="seconds")
            state["last_watchdog_scorer"] = "watchlist"
            try:
                ws_path.write_text(json.dumps(state, indent=2))
            except Exception as e:
                logging.warning(f"[watchdog] state write failed: {e}")
            return

        # ── Legacy scorer paths (composite / dislocation) ─────────────
        opps = scan_core_universe(min_score=0.0)
        if not opps:
            logging.warning("[watchdog] universe scan returned 0 opps — abort")
            return

        # Score the universe via the session; track each result alongside the opp
        ranked = []  # list of (opp, ScoreResult)
        for o in opps:
            r = session.score(o["ticker"], o)
            ranked.append((o, r))
        # Sort by z desc; exclude hard-excluded from ranking comparison
        ranked.sort(key=lambda x: -(x[1].z if (x[1].z is not None and not x[1].hard_excluded) else -99.0))

        # Filter: qualifiers not deployed and not hard-excluded; new OR moved ≥+0.5 since notify
        new_qualifiers = []  # list of (opp, ScoreResult)
        for o, r in ranked:
            ticker = o["ticker"]
            if ticker in deployed_tickers:
                continue
            if r.hard_excluded or r.z is None or r.z < 0.3:
                continue
            last_z = notified_map.get(ticker)
            if last_z is None or (r.z - last_z) >= 0.5:
                new_qualifiers.append((o, r))

        if not new_qualifiers:
            top_visible = next(
                ((o, r) for o, r in ranked if not r.hard_excluded and r.z is not None),
                None,
            )
            if top_visible:
                logging.info(
                    f"[watchdog] no new qualifiers (held_cash €{held_cash:.0f}, "
                    f"engine={session.scorer}, "
                    f"top z={top_visible[1].z:+.2f} {top_visible[0]['ticker']})"
                )
            else:
                logging.info(f"[watchdog] no new qualifiers (engine={session.scorer})")
            return

        # Suggest sizing for held_cash via softmax across NEW qualifiers
        allocs, _skipped = softmax_sizing(
            [(o["ticker"], r.z) for o, r in new_qualifiers],
            budget_eur=held_cash,
            threshold_z=0.3,
            min_alloc_eur=100.0,
        )

        # Compose Telegram message — top 3 to keep it scannable
        top3 = new_qualifiers[:3]
        from datetime import datetime
        ts = datetime.now(LISBON_TZ).strftime("%Y-%m-%d %H:%M")
        engine_tag = {
            "watchlist":   "📋 watchlist",
            "dislocation": "🧠 dislocation_z (experimental)",
            "composite":   "📐 composite_z (legacy)",
        }.get(session.scorer, "?")
        lines = [
            f"💰 *Cash Watchdog* — `{ts}`  ·  engine: {engine_tag}",
            f"_Cash retido: *€{held_cash:,.0f}* (do último /cash)_",
            "",
            f"🚨 *{len(new_qualifiers)} novo(s) qualifier(s)* (z > 0.3 não-deployed):",
            "",
        ]
        for o, r in top3:
            ticker = o["ticker"]
            eur_sug = allocs.get(ticker)
            dd_52w = float(o.get("dd_from_52w_high") or 0.0)
            head = f"*{ticker}*  z=*{r.z:+.2f}*"
            if eur_sug:
                head += f"  → €{eur_sug:.0f}"
            lines.append(head)
            # Use the unified explanation_lines so both engines render coherently
            for expl_line in r.explanation_lines(threshold_abs=0.3)[1:]:
                lines.append(expl_line)
            lines.append(
                f"  DD 52w: {dd_52w*100:+.0f}%  |  score CORE: {o.get('score', 0):.0f}"
            )
            hl = fetch_headlines(ticker, max_results=2)
            if hl:
                lines.append(format_headlines_for_telegram(hl, max_results=2, indent="  "))
            lines.append("")

        if len(new_qualifiers) > 3:
            lines.append(
                f"_+ {len(new_qualifiers) - 3} mais qualifiers — corre `/scan_now` para ver todos_"
            )
            lines.append("")
        lines.append(
            f"_Para deploy: `/deployed TICKER AMOUNT reason`  ·  "
            f"para inspeção: `/analisar TICKER`_"
        )

        send_telegram("\n".join(lines))
        logging.info(
            f"[watchdog] pushed {len(new_qualifiers)} qualifiers via {session.scorer}, "
            f"top {top3[0][0]['ticker']} z={top3[0][1].z:+.2f}"
        )

        # Update notified state — only mark TOP 3 as notified (the ones the
        # user saw). Lower-ranked qualifiers stay un-notified so they can
        # surface in a future scan if they overtake the leaders.
        new_notified = list(notified_map.items())
        for o, r in top3:
            ticker = o["ticker"]
            cb_z = r.z if r.z is not None else 0.0
            existing_idx = next(
                (i for i, (t, _z) in enumerate(new_notified) if t == ticker), None
            )
            if existing_idx is not None:
                new_notified[existing_idx] = (ticker, cb_z)
            else:
                new_notified.append((ticker, cb_z))
        state["notified_tickers"] = [
            {"ticker": t, "z_at_notify": z} for t, z in new_notified
        ]
        state["last_watchdog_run"] = datetime.now().isoformat(timespec="seconds")
        try:
            ws_path.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logging.warning(f"[watchdog] state write failed: {e}")

    except Exception as e:
        logging.error(f"[watchdog] failed: {e}", exc_info=True)


# ── Scan principal ────────────────────────────────────────────────────────────

def run_scan() -> None:
    global _scan_running, _alerted_today
    if _scan_running:
        logging.warning("Scan já a correr, a ignorar...")
        return
    _scan_running = True
    # ── Feature 8: conjunto de symbols que passaram score >= MIN_DIP_SCORE ──
    _scan_syms_scored: set[str] = set()
    # Heartbeat counters — written to state.scan_heartbeat at end of scan so
    # the weekly report can tell "scanner is healthy but silent" from "scanner
    # is broken". Updated inline as the scan progresses.
    _hb_scores: list[float] = []
    _hb_n_alerted = 0
    _hb_n_rejected = 0
    _hb_threshold = float(MIN_DIP_SCORE)
    _hb_n_screened = 0
    _hb_started = datetime.now(LISBON_TZ).isoformat(timespec="seconds")

    try:
        if not is_market_open():
            logging.info("Mercado fechado, scan ignorado.")
            return

        logging.info("A iniciar scan de dips...")
        spy_change = get_spy_change()
        stocks     = screen_global_dips(DROP_THRESHOLD, MIN_MARKET_CAP)
        _hb_n_screened = len(stocks) if stocks else 0
        if not stocks:
            logging.info("Sem stocks na triagem.")
            return

        # ── Macro-overlay: threshold dinâmico baseado em stress sectorial ──────
        # Se ≥ 3 sectores caírem >2% em simultâneo, é rotação agressiva ou
        # bear market — aumentamos o score mínimo para evitar "apanhar facas".
        try:
            from data.macro_data import get_dynamic_min_score
            effective_min_score = get_dynamic_min_score(base_min_score=MIN_DIP_SCORE)
            if effective_min_score > MIN_DIP_SCORE:
                logging.info(
                    f"[macro_overlay] Score mínimo elevado: "
                    f"{MIN_DIP_SCORE} → {effective_min_score:.0f} "
                    f"(stress sectorial detectado)"
                )
        except Exception:
            effective_min_score = float(MIN_DIP_SCORE)
        _hb_threshold = float(effective_min_score)

        ranked_entries = []
        comprar_syms   = set()

        for stock in stocks:
            sym = stock["symbol"]
            if sym in _alerted_today:
                continue
            try:
                time.sleep(3)
                fund = get_fundamentals(sym, min_market_cap=MIN_MARKET_CAP)
                if fund.get("skip"):
                    append_rejected_log({
                        "symbol": sym, "change": stock["change_pct"],
                        "reason": fund.get("skip_reason", "skip"),
                        "score": None, "verdict": None,
                        "radar": "DIP", "price": stock.get("price"),
                    })
                    continue

                hist_pe       = get_historical_pe(sym)
                news          = get_news(sym)
                earnings_days = get_earnings_days(sym)
                sector_chg    = get_sector_change(fund.get("sector", ""))
                score, rsi_str = calculate_dip_score(
                    fund, sym, earnings_days, sector_change=sector_chg
                )

                bc_flag  = is_bluechip(fund)
                category = classify_dip_category(fund, score, bc_flag)

                rsi_val = None
                try:
                    rsi_val = float(str(rsi_str).split()[0])
                except Exception:
                    pass

                # Record this score in the heartbeat — keeps the full distribution,
                # not just alerts, so a "0 alerts" week can still report top score
                # and median.
                try:
                    _hb_scores.append(float(score))
                except Exception:
                    pass

                # ── CORE intraday warning (Fase B) ────────────────────────
                # Fires INDEPENDENTLY of the DIP score gate when:
                #   - ticker is one the user actively tracks (DIRECT_TICKERS)
                #     OR the DIP scorer classified it as HIGH_CONVICTION
                #   - intraday drop is ≥ 7% (looser than the DIP -8% screener
                #     gate; -7% on a name the user already cares about is
                #     worth a heads-up before it gets worse)
                # Dedup key f"{sym}_core_intraday" is distinct from the plain
                # `sym` key the DIP path uses, so a single CORE warning + a
                # later DIP alert for the same ticker still both surface.
                # The standard DIP-score flow continues below regardless; if
                # score >= MIN_DIP_SCORE the user gets both signals.
                _core_dedup_key = f"{sym}_core_intraday"
                _intraday_drop  = float(stock.get("change_pct") or 0.0)
                _is_user_core   = (sym in DIRECT_TICKERS) or (category == CAT_HIGH_CONVICTION)
                if (_is_user_core
                        and _intraday_drop <= -7.0
                        and _core_dedup_key not in _alerted_today):
                    try:
                        _hold_tag = " 📦 _em carteira_" if sym in DIRECT_TICKERS else ""
                        _cat_tag  = " 💎 _high-conviction_" if category == CAT_HIGH_CONVICTION else ""
                        _name     = fund.get("name") or sym
                        _price    = fund.get("price")
                        _price_s  = f"${_price:.2f}" if _price else "?"
                        _drawdown = fund.get("drawdown_from_high")
                        _dd_s     = f" | 52w: {_drawdown:.0f}%" if _drawdown is not None else ""
                        _core_msg = (
                            f"💎 *CORE WARNING — {_name} ({sym})*{_hold_tag}{_cat_tag}\n"
                            f"_Queda intraday significativa num ticker do teu radar CORE._\n\n"
                            f"Hoje: *{_intraday_drop:+.1f}%* — {_price_s}{_dd_s}\n"
                            f"DIP score: *{score:.0f}/100* (gate normal: {effective_min_score:.0f})\n\n"
                            f"_Aviso emitido independentemente do DIP gate — o drop vale revisão._"
                        )
                        # CORE intraday warnings route through severity filter.
                        # High-conviction (score≥75 + ML≥65%) push real-time;
                        # rest accumulate in digest.
                        _emit_alert(
                            _core_msg,
                            alert_type="CORE_WARNING",
                            ticker=sym,
                            score=float(score),
                            win_prob=None,  # CORE warning fires pre-ML in scan
                            pred_alpha_90d=None,
                            in_portfolio=sym in DIRECT_TICKERS,
                            summary=f"Drop intraday {_intraday_drop:+.1f}% em ticker CORE",
                        )
                        _alerted_today.add(_core_dedup_key)
                        save_alerts(_alerted_today)
                        logging.info(
                            f"[core_intraday] {sym}: warning emitted "
                            f"(drop={_intraday_drop:.1f}% direct={sym in DIRECT_TICKERS} "
                            f"high_conv={category == CAT_HIGH_CONVICTION})"
                        )
                    except Exception as _cw_err:
                        logging.warning(f"[core_intraday] {sym}: emit failed: {_cw_err}")

                if score < effective_min_score:
                    _hb_n_rejected += 1
                    append_rejected_log({
                        "symbol": sym, "change": stock["change_pct"],
                        "reason": f"score baixo ({score:.0f} < {effective_min_score:.0f})",
                        "score": score, "verdict": "EVITAR",
                        "radar": "DIP", "price": fund.get("price"),
                    })
                    continue
                _hb_n_alerted += 1

                if score >= 75:
                    verdict, emoji_str, tier = "COMPRAR", "🟢", 1
                elif score >= 60:
                    verdict, emoji_str, tier = "MONITORIZAR", "🟡", 2
                else:
                    verdict, emoji_str, tier = "MONITORIZAR", "🔵", 3

                # Chunk 6 — ML score em tempo real
                ml_features = {
                    **fund,
                    "score":             score,
                    "spy_change":        spy_change,
                    "sector_etf_change": sector_chg,
                    "earnings_days":     earnings_days,
                    "change_day_pct":    stock["change_pct"],
                }
                ml_result = ml_score(ml_features, symbol=sym)
                if ml_result.model_ready:
                    logging.info(
                        f"[ml] {sym}: label={ml_result.label} "
                        f"prob={ml_result.win_prob:.2f} conf={ml_result.confidence_label}"
                    )

                # ── Score engine ─────────────────────────────────────────
                ml_prob_score = ml_result.win_prob if ml_result.model_ready else None
                score_data = score_from_fundamentals(fund, ml_prob=ml_prob_score)

                if score_data.get("skip_recommended"):
                    append_rejected_log({
                        "symbol": sym, "change": stock["change_pct"],
                        "reason": f"v2_skip: conf={score_data['confidence']:.2f}",
                        "score": score_data["final_score"], "verdict": "SKIP",
                        "radar": "DIP", "price": fund.get("price"),
                    })
                    logging.info(f"[v2] {sym} rejeitado — confidence insuficiente")
                    continue

                # ── Confirmação de entrada: RSI crossover + volume capitulação ──
                # RSI crossover: só entrar se RSI estava <40 (oversold) recentemente.
                # Evita entrar em stocks que simplesmente estão a cair sem parar.
                # Volume capitulação: dip em alto volume = vendas de pânico esgotadas.
                rsi_val_f = fund.get("rsi") or fund.get("rsi_14") or 50.0
                vol_zscore = fund.get("volume_zscore_20d") or 0.0
                rsi_ok     = float(rsi_val_f) < 42.0   # oversold ou próximo
                vol_ok     = float(vol_zscore) > 0.8   # volume acima da média
                if verdict == "COMPRAR" and not rsi_ok:
                    # RSI não está oversold — rebaixar para monitorizar
                    verdict   = "MONITORIZAR"
                    emoji_str = "🟡"
                    tier      = 2
                    logging.info(f"[rsi_filter] {sym}: RSI={rsi_val_f:.0f} não oversold → MONITORIZAR")

                # ── 8-K Veto: evento estruturalmente negativo detectado ────────
                # Um restatement ou default é um red flag absoluto — o Shield
                # eleva o alerta para MONITORIZAR independentemente do ML score.
                _8k_val = fund.get("recent_8k_score")
                try:
                    from core.signal_narrative import generate_8k_veto
                    _veto = generate_8k_veto(_8k_val) if _8k_val is not None else None
                    if _veto and verdict == "COMPRAR":
                        verdict    = "MONITORIZAR"
                        emoji_str  = "🟡"
                        tier       = 2
                        logging.warning(f"[8k_veto] {sym}: {_veto}")
                except Exception:
                    pass

                score = score_data["final_score"]
                logging.info(
                    f"[v2] {sym}: final={score:.1f} | "
                    f"Q={score_data['quality_score']:.2f} "
                    f"V={score_data['value_score']:.2f} "
                    f"T={score_data['timing_score']:.2f} "
                    f"conf={score_data['confidence']:.2f} "
                    f"trap={score_data['is_value_trap']}"
                )

                # Filtrar com score V2: stocks com score muito baixo não geram alertas.
                # ML a predizer P(win)=0% → score V2=0 → Shield diz NÃO.
                if score < MIN_DIP_SCORE:
                    append_rejected_log({
                        "symbol": sym, "change": stock["change_pct"],
                        "reason": f"v2_score baixo ({score:.0f})",
                        "score": score, "verdict": "EVITAR",
                        "radar": "DIP", "price": fund.get("price"),
                    })
                    logging.info(f"[v2] {sym} rejeitado — score V2 {score:.0f} < MIN ({MIN_DIP_SCORE})")
                    continue

                # Recalcular verdict a partir do score V2 (coerência visual)
                if score >= 75:
                    verdict, emoji_str, tier = "COMPRAR", "🟢", 1
                elif score >= 60:
                    verdict, emoji_str, tier = "MONITORIZAR", "🟡", 2
                else:
                    verdict, emoji_str, tier = "MONITORIZAR", "🔵", 3

                # ── Feature 8: Dip Persistente ────────────────────────────
                dip_state = record_dip_day(
                    sym,
                    score,
                    fund.get("price", 0),
                    stock["change_pct"],
                    verdict,
                )
                _scan_syms_scored.add(sym)
                check_and_alert_streak(
                    sym, dip_state, score, category,
                    fund, send_telegram, DIRECT_TICKERS, LISBON_TZ,
                )
                # ──────────────────────────────────────────────────────────

                # Compute portfolio context at signal time for ML training
                _cap_util_pct = None
                _n_open       = None
                try:
                    from portfolio.paper_trading import (
                        get_paper_portfolio_value, get_paper_allocated,
                        get_paper_available, _load as _pt_load,
                    )
                    _pt = _pt_load()
                    _pv = get_paper_portfolio_value(_pt)
                    _al = get_paper_allocated(_pt)
                    _cap_util_pct = round(_al / max(_pv, 1.0) * 100, 1)
                    _n_open = sum(1 for t in _pt if t.get("status") == "OPEN")
                except Exception:
                    pass

                log_alert_snapshot(
                    symbol=sym, fundamentals=fund, score=score,
                    verdict=verdict, category=category,
                    change_day_pct=stock["change_pct"], rsi_val=rsi_val,
                    historical_pe=hist_pe, spy_change=spy_change,
                    sector_etf_change=sector_chg,
                    capital_utilization_pct=_cap_util_pct,
                    n_open_positions=_n_open,
                    # ml.prediction_log.csv resets on FEATURE_COLUMNS drift —
                    # log the per-alert α into alert_db (durable migrate_schema)
                    # so /cash always has the real model output and never
                    # falls back to the hardcoded 0.05.
                    pred_alpha_90d=(
                        float(ml_result.pred_up)
                        if ml_result.model_ready and ml_result.pred_up is not None
                        else None
                    ),
                    ml_win_prob=(
                        float(ml_result.win_prob)
                        if ml_result.model_ready and ml_result.win_prob is not None
                        else None
                    ),
                )

                # Verificar liquidez disponível — o alerta dispara sempre,
                # mas se não há budget o utilizador é avisado explicitamente.
                _monthly_budget = float(os.environ.get("MONTHLY_BUDGET_EUR", "0") or 0)
                _capital_used   = 0.0
                try:
                    from portfolio.paper_trading import _capital_used_this_month as _cup
                    _capital_used = _cup()
                except Exception:
                    pass
                _no_liquidity = (verdict == "COMPRAR") and (_monthly_budget - _capital_used < 20)

                # ── Live-price refresh at alert moment ─────────────────────────
                # The scan path uses fund["price"] from yfinance .info (often
                # stale = previous close when US market is still closed). At
                # the moment we send the alert we re-fetch a fresh intraday
                # tick so the user sees a current price + recompute change_pct
                # against the genuine previous close. Falls back gracefully
                # to the existing stale price if the live fetch fails.
                try:
                    from data.data_feed import get_live_price
                    _live = get_live_price(sym)
                    if _live and _live.source != "eod_fallback":
                        fund["price"]            = _live.price
                        fund["price_sampled_at"] = _live.sampled_at
                        fund["price_source"]     = _live.source
                        if _live.change_pct is not None:
                            stock["change_pct"] = _live.change_pct
                except Exception as _lp_err:
                    logging.debug(f"[live_price] {sym}: {_lp_err}")

                alert_text = build_alert(
                    stock, fund, hist_pe, news,
                    verdict, emoji_str, [], score, rsi_str,
                    category=category,
                    ml_result=ml_result,
                    no_liquidity=_no_liquidity,
                )

                sentiment_line = _get_news_sentiment_line(sym)
                if sentiment_line:
                    alert_text = alert_text + f"\n{sentiment_line}"

                # ── Option C: enrich DIP alert with CORE block when ticker
                # is in the CORE universe AND the intraday drop matches the
                # DIP scanner threshold. Single consolidated Telegram message
                # in place of separate DIP@16h + CORE@22h notifications for
                # the same event. ~0-2 hits per day in practice.
                try:
                    from core.core_radar import (
                        get_core_universe_set,
                        score_dip_for_inline_alert,
                        build_core_dip_block,
                    )
                    if sym in get_core_universe_set() and abs(stock.get("change_pct", 0.0)) >= DROP_THRESHOLD:
                        _core = score_dip_for_inline_alert(sym)
                        if _core is not None:
                            _core_score, _core_signals = _core
                            alert_text += build_core_dip_block(_core_score, _core_signals)
                except Exception as _ce:
                    logging.debug(f"[core_inline] {sym}: {_ce}")

                ranked_entries.append({
                    "symbol": sym, "dip_score": score, "tier": tier,
                    "fundamentals": fund, "earnings_date": get_earnings_date(sym),
                    "earnings_days": earnings_days, "catalyst": None, "category": category,
                })

                # Route DIP alerts through severity filter. Real-time when
                # the ML model agrees with the rule-based score (>=75 AND
                # win_prob>=65%); otherwise buffered to the 21:30 digest.
                _ml_wp = (
                    float(ml_result.win_prob)
                    if (ml_result is not None and ml_result.model_ready
                        and ml_result.win_prob is not None)
                    else None
                )
                _ml_pa = (
                    float(ml_result.pred_up)
                    if (ml_result is not None and ml_result.model_ready
                        and ml_result.pred_up is not None)
                    else None
                )
                _emit_alert(
                    alert_text,
                    alert_type="DIP",
                    ticker=sym,
                    score=float(score),
                    win_prob=_ml_wp,
                    pred_alpha_90d=_ml_pa,
                    in_portfolio=sym in DIRECT_TICKERS,
                    summary=f"DIP score {score:.0f}, ML win_prob {(_ml_wp or 0)*100:.0f}%",
                )
                _alerted_today.add(sym)
                save_alerts(_alerted_today)

                # ── Paper trading: regista sempre para COMPRAR (dados de treino) ──
                # Mesmo sem liquidez real, o paper trade é criado com amount mínimo.
                # Isto garante que o modelo aprende com TODOS os alertas, não só
                # aqueles em que havia dinheiro disponível (selection bias).
                if verdict == "COMPRAR":
                    try:
                        from portfolio.paper_trading import record_paper_buy
                        from portfolio.portfolio import FLIP_FUND_EUR
                        paper_amount = float(
                            decision.amount_eur
                            if "decision" in dir() and hasattr(decision, "amount_eur") and decision.amount_eur > 0
                            else max(50.0, FLIP_FUND_EUR * (score / 100.0) * 0.5)
                        )
                        if paper_amount < 5:
                            paper_amount = 50.0  # mínimo para tracking
                        if True:  # sempre criar, independentemente do budget
                            _, strat = calculate_flip_target(
                                fund, score,
                                earnings_date=get_earnings_date(sym),
                                catalyst=None,
                                spy_change=spy_change,
                                category=category,
                            )
                            # Extrair preço target da estratégia
                            import re as _re
                            _m = _re.search(r'\$([0-9.]+)', strat or "")
                            sell_tgt = float(_m.group(1)) if _m else (fund.get("price", 0) * 1.15 or 0)
                            _price   = float(fund.get("price", 0) or 0)
                            _wp      = ml_result.win_prob if ml_result.model_ready else 0.0
                            if sell_tgt > _price > 0:
                                record_paper_buy(
                                    ticker=sym,
                                    open_price=_price,
                                    amount_eur=paper_amount,
                                    sell_target=sell_tgt,
                                    hold_days=90,
                                    score_v2=score,
                                    ml_win_prob=_wp,
                                    usd_eur=get_usdeur(),
                                    model_id="dip_us",
                                    sector=fund.get("sector", ""),
                                )
                    except Exception as _pe:
                        logging.debug(f"[paper] {sym}: {_pe}")

                # Auto-track no position_db quando COMPRAR.
                # O monitor diário acompanha stop-loss, target e deterioração.
                # Não requer intervenção manual — nunca perdes o tracking de uma posição.
                if verdict == "COMPRAR":
                    try:
                        from portfolio import position_db as _pdb
                        from ml.ml_engine import predict_dip as _predict
                        _price_now  = float(fund.get("price") or stock.get("price") or 0)
                        _feat_row   = {**fund, "score": score, "sector": fund.get("sector",""), "change_day_pct": stock["change_pct"]}
                        _pred       = _predict(_feat_row, current_price=_price_now, ticker=sym, bundle=None)
                        _target     = _pred.sell_target if _pred and _pred.sell_target > _price_now > 0 else _price_now * 1.15
                        _existing   = _pdb.get_by_ticker(sym)
                        if not _existing or _existing.status != "ACTIVE":
                            _pdb.add_position(_pdb.PositionRecord(
                                ticker             = sym,
                                status             = "ACTIVE",
                                alert_date         = datetime.now(LISBON_TZ).date().isoformat(),
                                alert_price        = _price_now,
                                alert_win_prob     = float(ml_result.win_prob if ml_result.model_ready else 0.5),
                                alert_feature_row  = [],
                                initial_buy_target = _price_now * 0.97,
                                initial_sell_target= round(_target, 2),
                                initial_hold_days  = 90,
                                dip_score          = float(score),
                                fundamentals_snap  = {k: fund.get(k) for k in ("sector","quality_score","analyst_upside","fcf_yield")},
                                current_sell_target= round(_target, 2),
                                current_hold_days  = 90,
                                last_win_prob      = float(ml_result.win_prob if ml_result.model_ready else 0.5),
                                initial_pred_alpha = float(ml_result.pred_up or 0) if ml_result.model_ready else None,
                            ))
                            logging.info(f"[portfolio] {sym}: posição criada automaticamente (target=${_target:.2f})")
                    except Exception as _ae:
                        logging.debug(f"[portfolio] auto-add {sym}: {_ae}")

                if verdict == "COMPRAR":
                    comprar_syms.add(sym)
                    add_recovery_position(
                        sym,
                        price_alert=fund.get("price") or stock.get("price", 0),
                        score=score,
                        target_pct=RECOVERY_PCT,
                        verdict=verdict,
                        category=category,
                    )

                append_weekly_log({
                    "symbol": sym, "score": score, "change": stock["change_pct"],
                    "verdict": verdict, "sector": fund.get("sector", ""),
                    "date": datetime.now(LISBON_TZ).strftime("%d/%m"),
                    "time": datetime.now(LISBON_TZ).strftime("%H:%M"),
                })
                append_backtest_entry({
                    "symbol": sym, "date": datetime.now(LISBON_TZ).strftime("%Y-%m-%d"),
                    "price": fund.get("price") or stock.get("price", 0),
                    "score": score, "verdict": verdict,
                })

            except Exception as e:
                logging.error(f"Erro no scan de {sym}: {e}", exc_info=True)

        if ranked_entries:
            ranking_text = build_flip_ranking(ranked_entries, spy_change, exclude_syms=comprar_syms)
            if ranking_text:
                send_telegram(ranking_text)

    finally:
        expire_missing_streaks(_scan_syms_scored)
        _scan_running = False
        # ── Scan heartbeat: telemetry that makes silence legible ────────
        # Even a "0 alerts" scan now leaves a row showing universe size,
        # score distribution and threshold. The weekly report aggregates
        # these so a quiet week can be classified as "scanner healthy,
        # market boring" vs "scanner broken".
        try:
            import state as _st_hb
            _top = max(_hb_scores) if _hb_scores else None
            _med = (
                sorted(_hb_scores)[len(_hb_scores) // 2] if _hb_scores else None
            )
            _st_hb.append_scan_heartbeat({
                "timestamp":   _hb_started,
                "n_screened":  int(_hb_n_screened),
                "n_scored":    len(_hb_scores),
                "n_alerted":   int(_hb_n_alerted),
                "n_rejected":  int(_hb_n_rejected),
                "top_score":   round(_top, 1) if _top is not None else None,
                "median_score": round(_med, 1) if _med is not None else None,
                "threshold":   round(_hb_threshold, 1),
            })
        except Exception as _hb_err:
            logging.debug(f"[scan_heartbeat] failed: {_hb_err}")
        logging.info("Scan concluído.")


# ── ML Outcomes Job (domingo 08:00) ───────────────────────────────────────────

def run_ml_outcomes_job() -> None:
    logging.info("[ml_outcomes] A actualizar outcomes da alert_db...")
    try:
        stats = fill_db_outcomes()
    except Exception as e:
        logging.error(f"[ml_outcomes] Erro: {e}")
        send_telegram(f"🤖 *ML Outcomes — Erro*\n`{e}`")
        return

    if stats.get("updated", 0) == 0 and stats.get("total", 0) == 0:
        return

    db_stats = get_db_stats()
    outcomes = db_stats.get("outcomes", {})
    labeled  = db_stats.get("labeled", 0)
    total    = db_stats.get("total", 0)

    outcome_lines = []
    for label, count in sorted(outcomes.items(), key=lambda x: x[1], reverse=True):
        emoji_map = {"WIN_STRONG": "🟢", "WIN": "✅", "NEUTRAL": "🟡", "LOSS": "🔴"}
        e = emoji_map.get(label, "📊")
        pct = count / labeled * 100 if labeled > 0 else 0
        outcome_lines.append(f"  {e} {label}: {count} ({pct:.0f}%)")

    lines = [
        f"🤖 *ML Outcomes — Update semanal*",
        f"_{datetime.now(LISBON_TZ).strftime('%d/%m/%Y %H:%M')}_",
        "",
        f"*🗓️ Alertas na base de dados:* {total}",
        f"*📊 Classificados:* {labeled} ({labeled/total*100:.0f}% do total)" if total > 0 else "*📊 Classificados:* 0",
        f"*✅ Actualizados agora:* {stats.get('updated', 0)}",
        f"*⏩ Ignorados (dados insuficientes):* {stats.get('skipped', 0)}",
        "",
    ]
    if outcome_lines:
        lines.append("*🎯 Distribuição de outcomes:*")
        lines.extend(outcome_lines)
        lines.append("")
    if labeled >= 50:
        lines.append("🚦 *Podes considerar treinar o primeiro modelo ML!*")
    elif labeled > 0:
        lines.append(f"_🕐 {50 - labeled} alertas até poder treinar o modelo ML_")

    send_telegram("\n".join(lines))


# ── Monthly ML Retrain Job (dia 1 de cada mês, 06:00) ─────────────────────────

def run_monthly_retrain() -> None:
    """
    Corre no dia 1 de cada mês às 06:00 Lisboa.

    Delega para `monthly_retrain.run_monthly_retrain` que faz:
      1. Build training input (bootstrap + alert_db + universe_snapshot)
      2. Treina candidate via `ml_training.train.run_training`
         (regressor alpha_90d + isotonic, 10 folds purgados)
      3. Gating ρ_α: candidate só substitui produção se
         ρ_α(cand) ≥ ρ_α(prod) × 0.90 e ρ_α(cand) ≥ floor (default 0.20).
         Caso contrário guardado como `dip_models_pending.pkl`.
      4. Backup automático em /data/archive/.

    Telegram: PROMOTED / PENDING / KEPT_FLOOR / FAILED + delta ρ_α.
    """
    logging.info("[monthly_retrain] A iniciar retreino mensal v3...")
    now_str = datetime.now(LISBON_TZ).strftime("%d/%m/%Y %H:%M")

    # Notificar início (era o que faltava — sem isto o utilizador não sabe que correu)
    send_telegram(
        f"Retreino mensal iniciado ({now_str}).\n"
        f"Pode demorar 1-2h. Sera enviada notificacao quando terminar."
    )

    # Watchdog: timeout total de 3 horas para o retrain completo.
    # Sem isto, um hang silencioso ficava 2 dias sem notificacao.
    import threading
    _MAX_RETRAIN_HOURS = int(os.environ.get("RETRAIN_MAX_HOURS", "3"))
    _result_holder: list = [None]
    _error_holder:  list = [None]

    # Redirigir logging do retrain para ficheiro — permite /admin_retrain_status
    _retrain_log = (Path("/data") if Path("/data").exists() else Path("/tmp")) / "retrain_progress.log"
    _retrain_log_handler = None
    try:
        import logging as _lg
        _retrain_log_handler = _lg.FileHandler(str(_retrain_log), mode="w", encoding="utf-8")
        _retrain_log_handler.setFormatter(_lg.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
        _lg.getLogger().addHandler(_retrain_log_handler)
    except Exception:
        pass

    def _run_retrain():
        try:
            # Fase 0: refresh INCREMENTAL do parquet base — adiciona episódios
            # do último mês ao ml_training_base.parquet sem rebuild completo.
            # Sem --force-rebuild, o builder detecta o parquet existente e só
            # processa datas novas (lookback 200d).
            try:
                import subprocess as _sp, sys as _sys
                from pathlib import Path as _P
                _data_dir = _P("/data") if _P("/data").exists() else _P("/tmp")
                _build_script = _P(__file__).parent / "scripts" / "build_dip_dataset.py"
                _bproc = _sp.run(
                    [_sys.executable, str(_build_script),
                     "--out", str(_data_dir / "ml_training_base.parquet")],
                    capture_output=True, text=True, timeout=2 * 3600
                )
                if _bproc.returncode == 0:
                    logging.info("[monthly_retrain] DIP dataset refresh incremental OK")
                else:
                    logging.warning(
                        f"[monthly_retrain] DIP incremental build exit "
                        f"{_bproc.returncode} — train continua com parquet antigo"
                    )
            except Exception as _be:
                logging.warning(f"[monthly_retrain] DIP incremental skipped: {_be}")

            from ml.monthly_retrain import run_monthly_retrain
            _result_holder[0] = run_monthly_retrain()
        except Exception as e:
            _error_holder[0] = e
        finally:
            if _retrain_log_handler:
                try:
                    logging.getLogger().removeHandler(_retrain_log_handler)
                    _retrain_log_handler.close()
                except Exception:
                    pass

    t = threading.Thread(target=_run_retrain, daemon=True)
    t.start()
    t.join(timeout=_MAX_RETRAIN_HOURS * 3600)

    if t.is_alive():
        logging.error(f"[monthly_retrain] Timeout de {_MAX_RETRAIN_HOURS}h — a abortar")
        send_telegram(
            f"Retreino mensal excedeu {_MAX_RETRAIN_HOURS}h e foi abortado.\n"
            f"O modelo anterior continua em producao.\n"
            f"Verifica os logs do Railway para perceber o que pendurou."
        )
        return

    if _error_holder[0] is not None:
        e = _error_holder[0]
        logging.error(f"[monthly_retrain] v3 falhou: {e}", exc_info=True)
        send_telegram(f"Retreino Mensal falhou:\n{e}\n{now_str}")
        return

    result = _result_holder[0] or {}

    decision   = result.get("decision", "?")
    cand_rho   = result.get("candidate_rho_alpha")
    prod_rho   = result.get("production_rho_alpha")
    cand_brier = result.get("candidate_brier")
    prod_brier = result.get("production_brier")
    cand_pnl   = result.get("candidate_topk_pnl")
    prod_pnl   = result.get("production_topk_pnl")
    reason     = result.get("reason", "")
    elapsed    = result.get("elapsed_s", 0)
    floor      = result.get("floor_rho_alpha")
    ratio      = result.get("gating_ratio")
    outcomes   = result.get("outcome_stats") or {}

    icon = {
        "PROMOTED":   "🚀",
        "PENDING":    "⚠️",
        "KEPT_FLOOR": "🛑",
        "FAILED":     "❌",
        "DRY-RUN":    "🧪",
    }.get(decision, "⚙️")

    def _fmt(v) -> str:
        if v is None:
            return "N/A"
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return str(v)

    def _delta_line(cand, prod) -> str:
        try:
            cand_f = float(cand)
            prod_f = float(prod)
            if prod_f == 0:
                return ""
            d = (cand_f - prod_f) / prod_f * 100
            sign = "+" if d >= 0 else ""
            return f"  delta     : *{sign}{d:.1f}%*"
        except (TypeError, ValueError):
            return ""

    lines = [
        f"{icon} *Retreino Mensal — {decision}*",
        f"_{now_str} ({elapsed:.0f}s)_",
        "",
    ]
    if reason:
        lines.append(f"_{reason}_")
        lines.append("")

    if cand_rho is not None or prod_rho is not None:
        lines += [
            "*ρ_α (Spearman alpha_90d):*",
            f"  candidate : *{_fmt(cand_rho)}*",
            f"  produção  : *{_fmt(prod_rho)}*",
        ]
        d_line = _delta_line(cand_rho, prod_rho)
        if d_line:
            lines.append(d_line)
        lines.append("")

    if cand_brier is not None or prod_brier is not None:
        lines += [
            "*Brier OOF (calibração):*",
            f"  candidate : *{_fmt(cand_brier)}*",
            f"  produção  : *{_fmt(prod_brier)}*",
            "",
        ]

    if cand_pnl is not None or prod_pnl is not None:
        lines += [
            "*Top-K PnL (mean):*",
            f"  candidate : *{_fmt(cand_pnl)}*",
            f"  produção  : *{_fmt(prod_pnl)}*",
            "",
        ]

    if floor is not None and ratio is not None:
        lines.append(f"_Gating: ratio={ratio} | floor ρ_α={_fmt(floor)}_")
        lines.append("")

    if outcomes:
        lines += [
            f"*alert_db outcomes preenchidos:* {outcomes.get('updated', 0)}",
            f"*ignorados:* {outcomes.get('skipped', 0)}",
            "",
        ]

    if decision == "PENDING":
        lines.append("_Bundle guardado como `dip_models_pending.pkl`._")
        lines.append("_Usa `/admin\\_promote` para promover manualmente._")
        lines.append("")

    # Sector models — sempre mostrar, mesmo se rejeitados
    _train_result = result.get("train_result") or {}
    _sreport = (_train_result.get("report") or {}).get("sector_models") or {}
    lines.append("*Modelos sectoriais:*")
    if _sreport:
        for _sname, _sinfo in sorted(_sreport.items()):
            _ic = _sinfo.get("ic_holdout")
            _n  = _sinfo.get("n_train", 0)
            _ic_str = f"{_ic:.4f}" if isinstance(_ic, (int, float)) else "sem dados"
            lines.append(f"  {_sname}: IC={_ic_str} (n={_n})")
    else:
        lines.append("  Nenhum sector model aceite (IC < -0.02 em todos).")
        lines.append("  O modelo global cobre todos os sectores.")
    lines.append("")

    lines.append("_Próximo retreino: dia 1 do mês seguinte às 06h_")
    send_telegram("\n".join(lines))
    logging.info(
        f"[monthly_retrain] decision={decision} cand_rho={cand_rho} prod_rho={prod_rho}"
    )

    # Push automático ao GitHub quando promovido — elimina dependência do Colab.
    # O bundle, parquet e report ficam versionados no repo para auditoria e
    # persistência entre deploys do Railway.
    if decision == "PROMOTED":
        try:
            from utils.github_sync import sync_training_artifacts
            _data_dir = Path("/data") if Path("/data").exists() else Path("/tmp")
            sync_results = sync_training_artifacts(_data_dir)
            _ok = sum(1 for v in sync_results.values() if v == "ok")
            _total = len(sync_results)
            if _ok == _total and _total > 0:
                send_telegram(f"_GitHub sync: {_ok}/{_total} artefactos guardados no repo ✓_")
                logging.info(f"[github_sync] sync completo: {sync_results}")
            elif _total > 0:
                logging.warning(f"[github_sync] sync parcial: {sync_results}")
        except Exception as _gs_e:
            logging.warning(f"[github_sync] falhou (não crítico): {_gs_e}")


# ── Momentum Scan (breakouts e upside plays) ──────────────────────────────────

def run_momentum_scan() -> None:
    """Corre o scanner de momentum e envia alertas ao Telegram.

    Separado do scan de dips: procura stocks em subida forte com confirmação
    de volume (Micron, NOW, SanDisk, etc.).

    Schedule (registado no APScheduler):
      17h50 Lisboa — janela EU (após fecho europeu 17h30)
      21h30 Lisboa — janela US (após fecho americano ~22h inverno/21h verão)
    """
    try:
        from portfolio.paper_trading import (
            record_momentum_paper_buy, allocate_capital_unified,
            _get_monthly_budget, _capital_used_this_month,
        )
        clear_sector_cache()  # reset entre scans EU e US para evitar dados stale
        logging.info("[momentum] A iniciar scan de momentum...")
        candidates = scan_momentum_universe(min_score=62.0, max_results=5)
        if not candidates:
            logging.info("[momentum] Nenhum candidato de momentum hoje.")
            return

        header = f"🚀 *Momentum Scanner — {len(candidates)} oportunidade{'s' if len(candidates) > 1 else ''}*\n"
        send_telegram(header)

        # Capital arbiter: dimensionar cada candidato e verificar budget disponível
        available = _get_monthly_budget() - _capital_used_this_month()
        opps = []
        for c in candidates:
            dur = c.get("ml_durability", "MODERADO")
            # Usar scores ML se disponíveis; fallback heurístico baseado no score rule-based
            if c.get("ml_ready") and c.get("ml_score_ensemble", 0.0) != 0.0:
                _exp_ret  = max(float(c.get("ml_score_ensemble", 0.0)), 0.03)
                # P(win) via ML: ensemble positivo → win_prob ~55-75%
                ens = float(c.get("ml_score_ensemble", 0.0))
                _win_prob = min(0.50 + ens * 2.0, 0.78) if ens > 0 else 0.40
            else:
                _exp_ret  = min(c["score"] / 100 * 0.30, 0.50)
                _win_prob = min(0.40 + c["score"] / 100 * 0.35, 0.80)
            # Horizonte adaptado à durabilidade: DURÁVEL usa L2 (90d), TÁCTICO usa L1 (30d)
            _horizon = {"DURÁVEL": 90, "MODERADO": 60, "TÁCTICO": 30}.get(dur, 60)
            opps.append({
                "trade_type":   "MOMENTUM",
                "ticker":       c["ticker"],
                "expected_ret": _exp_ret,
                "horizon_days": _horizon,
                "win_prob":     _win_prob,
                "amount_eur":   available / max(len(candidates), 1),
                "_candidate":   c,
            })

        approved = allocate_capital_unified(opps, available)
        approved_tickers = {a["ticker"] for a in approved}

        for c in candidates:
            try:
                # Live-price refresh at alert moment (same pattern as DIP path).
                # Momentum scan uses EOD bars to compute reasons but the user
                # sees the alert at HH:MM with the most recent intraday price.
                try:
                    from data.data_feed import get_live_price
                    _live = get_live_price(c["ticker"])
                    if _live and _live.source != "eod_fallback":
                        c["price"]            = _live.price
                        c["price_sampled_at"] = _live.sampled_at
                except Exception:
                    pass
                msg = format_momentum_alert(c)
                # Indicar se foi aprovado pelo arbiter
                if c["ticker"] in approved_tickers:
                    msg += "\n\n_Capital alocado pelo arbiter._"
                send_telegram(msg)
            except Exception as e:
                logging.warning(f"[momentum] Erro ao enviar alerta {c.get('ticker')}: {e}")

        # Paper trading para aprovados — inclui scores ML multi-horizonte
        usd_eur = get_usdeur() if "get_usdeur" in dir() else 0.92
        for a in approved:
            c = a["_candidate"]
            try:
                record_momentum_paper_buy(
                    ticker             = c["ticker"],
                    open_price         = c["price"],
                    amount_eur         = a["amount_eur"],
                    momentum_score     = c["score"],
                    ml_score_l1        = c.get("ml_score_l1", 0.0),
                    ml_score_l2        = c.get("ml_score_l2", 0.0),
                    ml_score_l3        = c.get("ml_score_l3", 0.0),
                    ml_score_ensemble  = c.get("ml_score_ensemble", 0.0),
                    ml_durability      = c.get("ml_durability", "MODERADO"),
                    usd_eur            = usd_eur,
                    model_id           = "momentum_us",
                    sector             = c.get("sector", ""),
                )
            except Exception as e:
                logging.warning(f"[momentum] Paper trade {c['ticker']}: {e}")

        logging.info(f"[momentum] {len(candidates)} alertas enviados, {len(approved)} com capital alocado.")
    except Exception as e:
        logging.error(f"[momentum] Scan falhou: {e}", exc_info=True)


# ── CoreRadar Scan (compounders de qualidade em dip) ─────────────────────────

def run_core_scan() -> None:
    """Varre o universo CORE e envia alertas de dip em compounders de qualidade.

    Foco no toque/cruzamento da MA200 semanal — o sinal histórico de entrada
    primário para compounders de alta convicção (MSFT, NVO, GOOGL, etc.).

    Schedule: 22:00 Lisboa (dados EOD US limpos, uma vez por dia)
    """
    try:
        from core.core_radar import scan_core_universe, format_core_alert, is_model_ready
        from portfolio.paper_trading import record_core_holding, get_core_holdings_summary

        if not is_model_ready():
            logging.warning(
                "[core_radar] ML model not trained — nightly scan disabled. "
                "Run /admin_train_core_model after rebuilding core_training_base.parquet."
            )
            send_telegram(
                "⚠️ <b>CoreRadar</b> — ML model not ready.\n"
                "Use <code>/admin_build_core_dataset</code> then <code>/admin_train_core_model</code> to enable production scanning."
            )
            return

        logging.info("[core_radar] Starting CoreRadar scan (ML model active)...")
        candidates = scan_core_universe(min_score=60.0)

        if not candidates:
            logging.info("[core_radar] No CORE opportunities today.")
            return

        # Cross-radar dedup: when the DIP scanner already alerted on a ticker
        # earlier today (and Option C appended the inline CORE block), the
        # standalone CORE scan must NOT re-alert the same ticker — that would
        # produce two Telegram messages for the same event, defeating the
        # whole point of the inline enrichment. _alerted_today is a set of
        # plain tickers (no scan_type qualifier) so membership testing here
        # catches DIP, inline-CORE-enriched-DIP, momentum, etc.
        global _alerted_today
        candidates_pre = candidates
        candidates = [c for c in candidates if c["ticker"] not in _alerted_today]
        skipped = len(candidates_pre) - len(candidates)
        if skipped:
            logging.info(
                f"[core_radar] Skipping {skipped} CORE candidate(s) already alerted "
                f"today (deduped against _alerted_today)."
            )
        if not candidates:
            logging.info("[core_radar] All CORE candidates already alerted earlier today — nothing to send.")
            return

        # Hard cap: send at most 5 alerts (sorted by score descending)
        _MAX_ALERTS    = 5
        _MAX_TRADES    = 3   # max auto-trades per day regardless of score
        _AUTO_THRESHOLD = 80  # score threshold for auto-trade (stricter than alert)

        top = candidates[:_MAX_ALERTS]
        hidden = len(candidates) - len(top)
        header = (
            f"💎 *CoreRadar — {len(top)} oportunidade{'s' if len(top) > 1 else ''} de dip*"
            + (f" _(+{hidden} com score mais baixo omitidos)_" if hidden > 0 else "")
            + "\n"
        )
        send_telegram(header)

        # Tickers CORE já com posição aberta (evitar duplicados)
        try:
            existing_core = {
                h["ticker"] for h in get_core_holdings_summary()
                if h.get("status") == "open"
            }
        except Exception:
            existing_core = set()

        usd_eur    = get_usdeur()
        registered = 0

        for c in top:
            try:
                msg = format_core_alert(c)
                sentiment_line = _get_news_sentiment_line(c["ticker"])
                if sentiment_line:
                    msg += f"\n{sentiment_line}"
                if c["ticker"] in existing_core:
                    msg += "\n\n_⚠️ Já tens posição CORE aberta neste ticker._"
                send_telegram(msg)
                # Mark in _alerted_today so manual /admin_run_core retries
                # don't re-fire the same alert within the day.
                _alerted_today.add(c["ticker"])
            except Exception as e:
                logging.warning(f"[core_radar] Alert {c.get('ticker')}: {e}")

        save_alerts(_alerted_today)

        # Auto-registration: score >= 80 AND daily trade cap not reached
        for c in top:
            if registered >= _MAX_TRADES:
                logging.info(f"[core_radar] Daily auto-trade cap ({_MAX_TRADES}) reached — skipping remaining.")
                break
            ticker = c["ticker"]
            if ticker in existing_core or c["score"] < _AUTO_THRESHOLD:
                continue
            try:
                trade_id = record_core_holding(
                    ticker          = ticker,
                    open_price      = c["current_price"],
                    amount_eur      = c.get("suggested_eur", 500.0),
                    annual_yield_pct= c.get("annual_yield_pct", 0.0),
                    note            = c.get("note", "CoreRadar auto-scan"),
                    usd_eur         = usd_eur,
                    model_id        = "core_us",
                    sector          = c.get("sector", ""),
                )
                if trade_id:
                    registered += 1
                    logging.info(
                        f"[core_radar] Paper CORE {ticker} [{trade_id}] "
                        f"@ ${c['current_price']:.2f} €{c.get('suggested_eur', 500):.0f}"
                    )
            except Exception as e:
                logging.warning(f"[core_radar] Paper trade {ticker}: {e}")

        if registered > 0:
            send_telegram(
                f"_✅ {registered} paper trade{'s' if registered > 1 else ''} CORE "
                f"registado{'s' if registered > 1 else ''} automaticamente._"
            )

        logging.info(f"[core_radar] {len(candidates)} alertas, {registered} registados.")
    except Exception as e:
        logging.error(f"[core_radar] Scan falhou: {e}", exc_info=True)




def _get_news_sentiment_line(ticker: str) -> str:
    """Score current headlines for ticker at alert time. Returns one-line summary or ''.

    Called once per real alert — never in batch for the full universe.
    Silently returns '' on any error (never blocks an alert).
    """
    try:
        from scripts.collect_news_sentiment import score_ticker_now
        return score_ticker_now(ticker)
    except Exception as e:
        logging.debug(f"[sentiment] {ticker}: {e}")
        return ""


# ── Daily Backup Job (23:30 Lisboa, seg-sex) ──────────────────────────────────

def run_daily_backup() -> None:
    """
    Backup diário dos ficheiros críticos: alert_db, state JSONs,
    parquet datasets, dip_models.pkl e ml_report.json.
    Arquiva em /data/backups/YYYY-MM-DD_HH-MM.zip e mantém os últimos 30.
    """
    logging.info("[backup] A iniciar backup diário...")
    try:
        result = run_backup()
        msg    = backup_telegram_summary(result)
        send_telegram(msg)
        logging.info(
            f"[backup] Concluído: {result.get('file_count')} ficheiros, "
            f"{result.get('size_mb')} MB"
        )
    except Exception as e:
        logging.error(f"[backup] Erro inesperado: {e}", exc_info=True)
        send_telegram(f"❌ *Backup Diário — Erro*\n`{e}`")


# ── Scheduler ─────────────────────────────────────────────────────────────────

def main() -> None:
    global _alerted_today

    # Migrar alert_db.csv para o schema actual no arranque.
    try:
        from scanners.alert_db import migrate_schema
        if migrate_schema():
            logging.info("[main] alert_db.csv migrado para schema actual.")
    except Exception as _e:
        logging.warning(f"[main] alert_db migrate_schema: {_e}")

    # ── Verificação de configuração crítica ──────────────────────────────────
    _missing_config: list[str] = []
    # FLIP_FUND_EUR é opcional — deriva automaticamente de MONTHLY_BUDGET_EUR × 10%.
    # Só alertar se AMBOS estiverem a 0 (Tesoureiro completamente sem capital).
    _flip_eur    = float(os.environ.get("FLIP_FUND_EUR", "0") or "0")
    _monthly_eur = float(os.environ.get("MONTHLY_BUDGET_EUR", "0") or "0")
    if _flip_eur <= 0 and _monthly_eur <= 0:
        _missing_config.append(
            "MONTHLY_BUDGET_EUR — orcamento mensal em euros "
            "(ex: MONTHLY_BUDGET_EUR=850). O Tesoureiro precisa disto para sizing."
        )
    if _missing_config:
        logging.warning(
            f"[config] {len(_missing_config)} env var(s) em falta que afectam "
            f"funcionalidade: {', '.join(v.split(' — ')[0] for v in _missing_config)}"
        )

    bot_commands.setup(
        send_fn=send_telegram,
        analyze_fn=analyze_ticker,
        clear_alerts_fn=lambda: (
            clear_alerts(),
            globals().update(_alerted_today=set()),
        ),
        # On-demand digest peek (no clear) for /digest command
        run_digest_fn=lambda: run_daily_digest(clear_after=False, label="Digest on-demand"),
        # Callbacks necessários para comandos do bot
        get_db_stats=get_db_stats,
        fill_db_outcomes=fill_db_outcomes,
        run_scan=run_scan,
        is_market_open=is_market_open,
        get_fundamentals=get_fundamentals,
        get_earnings_days=get_earnings_days,
        load_rejected_log=load_rejected_log,
        get_snapshot=_get_snapshot,
        build_backtest_summary=build_backtest_summary,
    )

    scheduler = BlockingScheduler(timezone=LISBON_TZ)

    # Intraday MOMENTUM stop sweep — defined here so all three scan lambdas
    # below resolve it cleanly (Python closures defer name lookup, but having
    # the def first keeps the call sites readable).
    def _run_active_stops_check():
        try:
            from core import conviction_radar
            n = conviction_radar.check_active_stops(send_telegram, _alerted_today)
            if n > 0:
                save_alerts(_alerted_today)
        except Exception as e:
            logging.error(f"[conviction] active stops check failed: {e}", exc_info=True)

    # ── Daily alert dedup reset: 06:00 Lisboa (before any scan) ───────────────
    # /data/alerted_today.json had no automatic clear job — clear_alerts was
    # registered as a callback for bot_commands but nothing scheduled it.
    # Result: tickers accumulated indefinitely across days/weeks. /cash's
    # DIP queue picked up stale entries from sessions ago (matching the
    # 'no log entry — fallback' pattern for alerts that predated the
    # pred_alpha_90d schema fix). Clearing daily at 06:00 means every
    # trading day starts with a fresh dedup set.
    def _run_daily_alert_reset():
        global _alerted_today
        try:
            clear_alerts()
            _alerted_today = set()
            logging.info("[scheduler] daily alert dedup reset complete")
        except Exception as e:
            logging.error(f"[scheduler] alert reset failed: {e}", exc_info=True)

    scheduler.add_job(
        _run_daily_alert_reset,
        CronTrigger(hour=6, minute=0, timezone=LISBON_TZ),
        id="alerted_today_reset", name="Alerted-today reset 06h00",
    )

    # ── Heartbeat: 09:00 Lisboa, seg-sex ──────────────────────────────────────
    scheduler.add_job(
        send_heartbeat,
        CronTrigger(day_of_week="mon-fri", hour=9, minute=0, timezone=LISBON_TZ),
        id="heartbeat", name="Heartbeat 09h",
    )

    # ── Scan US Intraday: 17:45 Lisboa, seg-sex ──────────────────────────────
    # Phase 2: European markets — not yet implemented.
    # See ModelID.DIP_EU, ModelID.CORE_EU, ModelID.MOMENTUM_EU in model_registry.py.
    scheduler.add_job(
        lambda: (
            run_scan(),
            check_thesis_degradation(),
            check_recovery_alerts(),
            _run_active_stops_check(),
            run_structural_dip_intraday_scan(),
        ),
        CronTrigger(day_of_week="mon-fri", hour=17, minute=45, timezone=LISBON_TZ),
        id="scan_intraday", name="Scan US Intraday 17h45",
    )

    # ── Hourly intraday scans: 15h, 16h, 18h, 19h, 20h Lisboa, seg-sex ───────
    # Why hourly: a dip happens at 10:30 ET; by the 21:15 Lisboa EOD scan,
    # the stock may have already recovered 5%+. Volatile names move too fast
    # to wait for end-of-day. Hourly granularity catches fresh dips while
    # they're still cheap.
    # Why these specific hours: US market is 14:30-21:00 Lisboa (summer/DST).
    # 15h-20h covers the bulk of the trading day. 17:00 + 21:00 omitted
    # because the existing 17:45 and 21:15 scans cover those windows already
    # (with extra checks like thesis_degradation, recovery_alerts, watchlist).
    # Each hourly scan respects the _alerted_today dedup, so a stock alerted
    # at 15h won't re-alert at 16h. No spam.
    scheduler.add_job(
        lambda: (
            run_scan(),
            check_thesis_degradation(),
            check_recovery_alerts(),
            _run_active_stops_check(),
            run_structural_dip_intraday_scan(),
        ),
        CronTrigger(day_of_week="mon-fri", hour="15,16,18,19,20", minute=0, timezone=LISBON_TZ),
        id="scan_hourly_intraday", name="Scan Hourly Intraday (15h,16h,18h,19h,20h)",
    )

    # ── Scan US EOD: 21:15 Lisboa, seg-sex ───────────────────────────────────
    scheduler.add_job(
        lambda: (
            run_scan(),
            check_thesis_degradation(),
            check_recovery_alerts(),
            _run_active_stops_check(),
            run_structural_dip_intraday_scan(),
            send_weekly_dip_scan(),
        ),
        CronTrigger(day_of_week="mon-fri", hour=21, minute=15, timezone=LISBON_TZ),
        id="scan_us", name="Scan US 21h15",
    )

    # ── Momentum scan US Intraday: 17h50 Lisboa ──────────────────────────────
    # Phase 2: European markets — not yet implemented.
    # See ModelID.DIP_EU, ModelID.CORE_EU, ModelID.MOMENTUM_EU in model_registry.py.
    scheduler.add_job(
        run_momentum_scan,
        CronTrigger(day_of_week="mon-fri", hour=17, minute=50, timezone=LISBON_TZ),
        id="momentum_scan_intraday", name="Momentum Scan US Intraday 17h50",
    )

    # ── Cash Watchdog: 18h00 Lisboa, seg-sex ─────────────────────────────────
    # When the last /cash run left held_cash_eur > 0, this scan walks the
    # CORE universe daily and pushes a Telegram notification when a new
    # qualifier (composite z > 0.3) emerges that wasn't deployed and hasn't
    # been notified yet. Solves the "being out of market is expensive" gap:
    # held cash earns 2.2% at T212; a real opportunity needs to be flagged
    # in hours, not weeks.
    scheduler.add_job(
        run_watchdog_scan,
        CronTrigger(day_of_week="mon-fri", hour=18, minute=0, timezone=LISBON_TZ),
        id="cash_watchdog", name="Cash Watchdog 18h",
    )
    # ── Momentum scan US: 21h30 Lisboa (após fecho americano) ────────────────
    scheduler.add_job(
        run_momentum_scan,
        CronTrigger(day_of_week="mon-fri", hour=21, minute=30, timezone=LISBON_TZ),
        id="momentum_scan_us", name="Momentum Scan US 21h30",
    )

    # ── Midday Digest: 17h45 Lisboa, seg-sex (peek, no clear) ─────────────────
    # Emits a snapshot of what's buffered so far — about 3h into the US
    # session, giving the user a chance to see DIPs/STRUCTURAL/CORE alerts
    # that fired in the first half of the day. Buffer is NOT cleared so
    # the 21:30 evening digest still consolidates the whole day.
    scheduler.add_job(
        lambda: run_daily_digest(clear_after=False, label="Midday digest"),
        CronTrigger(day_of_week="mon-fri", hour=17, minute=45, timezone=LISBON_TZ),
        id="midday_digest", name="Midday Digest 17h45",
    )

    # ── Evening Digest: 21h30 Lisboa, seg-sex (clear after) ───────────────────
    # Final consolidation of the day. Clears the buffer ready for tomorrow.
    scheduler.add_job(
        lambda: run_daily_digest(clear_after=True, label="Digest"),
        CronTrigger(day_of_week="mon-fri", hour=21, minute=30, timezone=LISBON_TZ),
        id="daily_digest", name="Daily Digest 21h30",
    )

    # ── CoreRadar scan: 22h00 Lisboa (dados EOD US limpos) ────────────────────
    scheduler.add_job(
        run_core_scan,
        CronTrigger(day_of_week="mon-fri", hour=22, minute=0, timezone=LISBON_TZ),
        id="core_scan", name="CoreRadar Scan 22h00",
    )

    # ── Stress check: cada 30 min, 14h30-22h30 Lisboa, seg-sex ────────────────
    scheduler.add_job(
        check_portfolio_stress,
        CronTrigger(
            day_of_week="mon-fri",
            hour="14-22", minute=f"*/{SCAN_MINUTES}",
            timezone=LISBON_TZ,
        ),
        id="stress", name="Portfolio Stress",
    )

    # ── Daily Backup: 23:30 Lisboa, seg-sex ───────────────────────────────────
    scheduler.add_job(
        run_daily_backup,
        CronTrigger(day_of_week="mon-fri", hour=23, minute=30, timezone=LISBON_TZ),
        id="daily_backup", name="Daily Backup 23h30",
    )

    # ── Label Resolver: 22:45 Lisboa, todos os dias ───────────────────────────
    scheduler.add_job(
        lambda: run_label_resolver_job(send_telegram),
        CronTrigger(hour=22, minute=45, timezone=LISBON_TZ),
        id="label_resolver", name="Label Resolver 22h45",
    )

    # ── Position monitor: 22:00 Lisboa, seg-sex ───────────────────────────────
    scheduler.add_job(
        lambda: run_daily_check(send_telegram),
        CronTrigger(day_of_week="mon-fri", hour=22, minute=0, timezone=LISBON_TZ),
        id="position_monitor", name="Position Monitor 22h00",
    )

    # ── Conviction monitor: 22:05 Lisboa, seg-sex (5min após position_monitor) ─
    # Parallel track for user's hunch-driven positions. Same engine, isolated
    # state (personal_positions.json + conviction_outcomes.json).
    def _run_conviction_monitor():
        try:
            from core import conviction_radar
            conviction_radar.run_daily_check(send_telegram)
        except Exception as e:
            logging.error(f"[conviction] daily check failed: {e}", exc_info=True)

    scheduler.add_job(
        _run_conviction_monitor,
        CronTrigger(day_of_week="mon-fri", hour=22, minute=5, timezone=LISBON_TZ),
        id="conviction_monitor", name="Conviction Monitor 22h05",
    )

    # ── PPR auto-scrape: 22:30 Lisboa, seg-sex ────────────────────────────────
    # Tries investing.com first (daily granularity), falls back to Banco
    # Invest monthly PDF when CF-blocked. To avoid hammering Banco Invest's
    # CDN with a 245KB PDF every weekday for the same monthly value, the
    # job skips when NAV is < 7 days old AND was last fetched from the PDF
    # source (i.e., investing.com isn't working anyway).
    def _run_ppr_auto_scrape():
        try:
            from state import get_ppr, set_ppr_nav
            from datetime import date as _date
            import ppr_scraper
            existing = get_ppr()
            if not existing or existing.get("shares", 0) <= 0:
                return  # no PPR configured

            # Politeness gate: skip if the last PDF-sourced NAV is fresh.
            # Investing.com still gets attempted because its data is daily,
            # so we only suppress when the PDF was the last source AND the
            # NAV is < 7 days old (PDFs update monthly anyway).
            nav_dt_str   = existing.get("nav_updated")
            last_source  = (existing.get("nav_source") or "").lower()
            if nav_dt_str and "pdf" in last_source:
                try:
                    age = (_date.today() - _date.fromisoformat(nav_dt_str)).days
                    if age < 7:
                        logging.debug(
                            f"[ppr] auto-scrape skipped: PDF-sourced NAV is {age}d old"
                        )
                        return
                except Exception:
                    pass

            isin = existing.get("isin", "PTARMJHM0003")
            result = ppr_scraper.fetch_ppr_nav(isin)
            if not result or not result.get("nav"):
                logging.warning(f"[ppr] auto-scrape failed for {isin} (no nav)")
                return
            nav = result["nav"]
            avg = existing.get("avg_cost", 0)
            if avg > 0 and (nav > avg * 10 or nav < avg / 10):
                logging.warning(
                    f"[ppr] auto-scrape: NAV {nav} suspicious vs avg {avg} — skipped"
                )
                return
            set_ppr_nav(nav, source=result.get("source", "investing.com"))
            logging.info(
                f"[ppr] auto-scraped NAV: €{nav:.4f} ({isin}) "
                f"from {result.get('source', '?')}"
            )
        except Exception as e:
            logging.error(f"[ppr] auto-scrape exception: {e}", exc_info=True)
    scheduler.add_job(
        _run_ppr_auto_scrape,
        CronTrigger(day_of_week="mon-fri", hour=22, minute=30, timezone=LISBON_TZ),
        id="ppr_auto_scrape", name="PPR Auto-Scrape 22h30",
    )

    # ── Conviction auto-sync: 06:00 Lisboa, mon-sun ───────────────────────────
    # Walks portfolio.get_positions() and creates conviction records for any
    # holding not yet tracked AND not in deny-list (ETFs, foreign-listed, PPR).
    # Type is auto-classified via ML signals + technicals. Never auto-creates
    # CORE — CORE is always user-explicit upgrade.
    def _run_conviction_auto_sync():
        try:
            from core import conviction_radar
            conviction_radar.auto_sync_from_portfolio(send_telegram)
        except Exception as e:
            logging.error(f"[conviction] auto-sync failed: {e}", exc_info=True)
    scheduler.add_job(
        _run_conviction_auto_sync,
        CronTrigger(hour=6, minute=0, timezone=LISBON_TZ),
        id="conviction_auto_sync", name="Conviction Auto-Sync 06h00",
    )

    # ── Saturday Weekly Report: sábado 10:00 Lisboa ───────────────────────────
    scheduler.add_job(
        send_saturday_report,
        CronTrigger(day_of_week="sat", hour=10, minute=0, timezone=LISBON_TZ),
        id="saturday_report", name="Saturday Weekly Report",
    )

    # ── ML Outcomes: domingo 08:00 Lisboa ─────────────────────────────────────
    scheduler.add_job(
        run_ml_outcomes_job,
        CronTrigger(day_of_week="sun", hour=8, minute=0, timezone=LISBON_TZ),
        id="ml_outcomes", name="ML Outcomes domingo",
    )

    # ── Shadow Portfolio (rejeições): domingo 07:30 Lisboa ────────────────────
    def _run_rejected_outcomes():
        try:
            from state import fill_rejected_outcomes, get_rejected_stats
            stats = fill_rejected_outcomes()
            if stats["filled"] == 0 and stats["total"] == 0:
                return
            rej_stats = get_rejected_stats(lookback_days=90)
            lines = [
                "🔍 *Shadow Portfolio — Rejeições*",
                f"_{datetime.now(LISBON_TZ).strftime('%d/%m/%Y %H:%M')}_",
                "",
                f"Rejeições com outcome: actualizadas +{stats['filled']} | pendentes {stats['pending']} | total {stats['total']}",
                "",
            ]
            for h_label, h_key in [("30d", "h30d"), ("60d", "h60d"), ("90d", "h90d")]:
                h = rej_stats.get(h_key)
                if not h:
                    continue
                lines.append(f"*Horizonte {h_label}* ({h['n']} amostras)")
                lines.append(f"  Win rate: *{h['win_rate']:.0%}* | Retorno médio: *{h['avg_return']:+.1f}%*" + (f" | Alpha: *{h['avg_alpha']:+.1f}pp*" if h.get('avg_alpha') is not None else ""))
                if h.get("by_reason"):
                    top = sorted(h["by_reason"].items(), key=lambda x: -abs(x[1]))[:3]
                    for reason, avg in top:
                        lines.append(f"    `{reason[:35]}`: {avg:+.1f}%")
                lines.append("")
            send_telegram("\n".join(lines))

            # ── Filter quality alert ───────────────────────────────────────
            _ALPHA_THRESHOLD = 2.0   # pp above SPY to flag a filter
            _MIN_SAMPLES     = 15    # minimum rejected entries to trust the stat
            flagged = []
            for h_label, h_key in [("30d", "h30d"), ("60d", "h60d"), ("90d", "h90d")]:
                h = rej_stats.get(h_key, {})
                for reason, detail in h.get("by_reason_detail", {}).items():
                    if (
                        detail["n"] >= _MIN_SAMPLES
                        and detail.get("avg_alpha") is not None
                        and detail["avg_alpha"] > _ALPHA_THRESHOLD
                    ):
                        flagged.append((h_label, reason, detail["n"], detail["avg_return"], detail["avg_alpha"]))
            if flagged:
                flagged.sort(key=lambda x: -x[4])
                warn_lines = [
                    "⚠️ *Filtros possivelmente demasiado agressivos*",
                    "_Rejeições com alpha médio acima de SPY — considera rever os thresholds:_",
                    "",
                ]
                for h_label, reason, n, avg_ret, avg_alpha in flagged:
                    warn_lines.append(f"• `{reason[:45]}` ({h_label})")
                    warn_lines.append(f"  {n} amostras | retorno médio *{avg_ret:+.1f}%* | alpha *{avg_alpha:+.1f}pp*")
                    warn_lines.append("")
                send_telegram("\n".join(warn_lines))

        except Exception as e:
            logging.warning(f"[rejected_outcomes] {e}")

    scheduler.add_job(
        _run_rejected_outcomes,
        CronTrigger(day_of_week="sun", hour=7, minute=30, timezone=LISBON_TZ),
        id="rejected_outcomes", name="Shadow Portfolio Rejeições",
    )

    # ── Monthly ML Retrain — REMOVED (2026-07-02) ─────────────────────────────
    # The DIP retrain cron kept promoting models monthly (2400s compute) that
    # nothing consumes: ML scoring is inert (DIPRADAR_USE_ML_SCORE=0 default)
    # and production scoring is DIPRADAR_SCORING_MODE=watchlist after the
    # F.5 validation REVERT. run_monthly_retrain stays available for manual
    # invocation via /admin_retrain if ML is ever revisited.

    # ── Monthly Retrain — MomentumRadar: dia 1, 08:00 Lisboa ──────────────────
    # Corre 2h após o DipRadar retrain para não competir por recursos.
    # Dataset incremental: nunca apaga dados históricos.
    def _run_momentum_monthly():
        import subprocess as _sp, sys as _sys
        from pathlib import Path as _Path
        t0_mom = time.time()
        now_str_mom = datetime.now(LISBON_TZ).strftime("%d/%m/%Y %H:%M")

        send_telegram(
            f"Retreino Momentum iniciado ({now_str_mom}).\n"
            f"Fase 1: dataset incremental (eventos do ultimo mes).\n"
            f"Fase 2: treino + gating. Notificacao quando terminar."
        )

        try:
            from momentum_radar.trainer import run_momentum_training

            # Fase 1: dataset incremental — build_dataset.py auto-detecta a
            # ultima data do parquet e so processa dados novos (220d lookback).
            # 12h timeout — momentum dataset now queries EDGAR per episode for
            # the 4 insider features, which made the 2h timeout too tight.
            script = _Path(__file__).parent / "momentum_radar" / "scripts" / "build_dataset.py"
            proc = _sp.run(
                [_sys.executable, str(script)],
                capture_output=True, text=True, timeout=12 * 3600
            )
            if proc.returncode != 0:
                stderr_tail = (proc.stderr or "")[-300:]
                logging.warning(f"[momentum_retrain] build_dataset exit {proc.returncode}")
                send_telegram(
                    f"⚠️ Dataset incremental falhou (exit {proc.returncode}) — "
                    f"a treinar no dataset anterior.\n`{stderr_tail}`"
                )
            else:
                logging.info("[momentum_retrain] Dataset actualizado com sucesso")

            # Fase 2: treinar
            result   = run_momentum_training()
            elapsed  = time.time() - t0_mom
            dec      = result.get("decision", "?")
            ic_new   = result.get("ic_ensemble")
            ic_prod  = result.get("ic_prod")
            n_train  = result.get("n_train", 0)
            n_sec    = result.get("n_sectors", 0)
            reason   = result.get("reason", "")

            def _f(v):
                return f"{float(v):.4f}" if isinstance(v, (int, float)) else "N/A"

            icon = {"PROMOTED": "🚀", "KEPT": "⚠️", "FAILED": "❌"}.get(dec, "⚙️")
            lines = [
                f"{icon} *MomentumRadar Retrain — {dec}*",
                f"_{now_str_mom} ({elapsed:.0f}s)_",
                "",
            ]
            if reason:
                lines += [f"_{reason}_", ""]

            lines += [
                "*IC ensemble (Spearman):*",
                f"  candidate : *{_f(ic_new)}*",
                f"  producao  : *{_f(ic_prod)}*",
            ]
            if isinstance(ic_new, float) and isinstance(ic_prod, float) and ic_prod != 0:
                delta = (ic_new - ic_prod) / abs(ic_prod) * 100
                sign  = "+" if delta >= 0 else ""
                lines.append(f"  delta     : *{sign}{delta:.1f}%*")
            lines += ["", f"  {n_train:,} amostras | {n_sec} sector models"]

            logging.info(f"[momentum_retrain] {dec} | IC={_f(ic_new)} vs prod={_f(ic_prod)}")
            send_telegram("\n".join(lines))

        except Exception as e:
            logging.error(f"[momentum_retrain] {e}", exc_info=True)
            send_telegram(f"❌ MomentumRadar retrain falhou: {e}")

    # Momentum retrain cron REMOVED (2026-07-02): the trainer is a RETIRED
    # stub (anti-predictive, CV IC = -0.0042) — the cron fired monthly, hit
    # the stub's RuntimeError and spammed the traceback to Telegram.
    # _run_momentum_monthly stays defined for the admin command path, which
    # surfaces the retirement message deliberately when invoked by hand.

    # ── CoreRadar Retrain — dia 1, 07h00 Lisboa (entre DIP e Momentum) ────────
    # Previously missing! Core stayed with whatever model was trained manually
    # via /admin_overnight, drifting further from market reality every month.
    # Same pattern as momentum: incremental dataset build + train + reload.
    def _run_core_monthly():
        import subprocess as _sp, sys as _sys
        from pathlib import Path as _Path
        t0_core = time.time()
        now_str_core = datetime.now(LISBON_TZ).strftime("%d/%m/%Y %H:%M")
        send_telegram(
            f"Retreino Core iniciado ({now_str_core}).\n"
            f"Fase 1: dataset incremental (eventos do ultimo mes).\n"
            f"Fase 2: treino. Notificacao quando terminar."
        )
        data_dir = Path("/data") if Path("/data").exists() else Path("/tmp")
        try:
            # Fase 1: dataset incremental — sem --force-rebuild, só adiciona
            # episódios novos ao parquet existente (lookback de 200d).
            build_script = _Path(__file__).parent / "scripts" / "build_core_dataset.py"
            build_proc = _sp.run(
                [_sys.executable, str(build_script),
                 "--out", str(data_dir / "core_training_base.parquet")],
                capture_output=True, text=True, timeout=4 * 3600
            )
            if build_proc.returncode != 0:
                stderr_tail = (build_proc.stderr or "")[-300:]
                logging.warning(f"[core_retrain] build_dataset exit {build_proc.returncode}")
                send_telegram(
                    f"⚠️ Core dataset incremental falhou (exit {build_proc.returncode}) — "
                    f"a treinar no dataset anterior.\n`{stderr_tail}`"
                )
            else:
                logging.info("[core_retrain] Dataset Core actualizado")

            # Fase 2: train
            train_script = _Path(__file__).parent / "scripts" / "train_core_model.py"
            train_proc = _sp.run(
                [_sys.executable, str(train_script),
                 "--parquet", str(data_dir / "core_training_base.parquet")],
                capture_output=True, text=True, timeout=1 * 3600
            )
            elapsed = time.time() - t0_core
            if train_proc.returncode == 0:
                # Reload in-process so the new model is active immediately
                try:
                    from core import core_radar as _cr
                    _cr._core_model_checked = False
                    _cr._core_model_bundle  = None
                    _cr._load_core_model()
                except Exception as _re:
                    logging.warning(f"[core_retrain] model reload: {_re}")
                send_telegram(
                    f"✅ *CoreRadar Retrain — OK*\n"
                    f"_{now_str_core} ({elapsed:.0f}s)_\n"
                    f"ML scoring active. /admin_check_config para CV_IC."
                )
            else:
                stderr_tail = (train_proc.stderr or "")[-300:]
                logging.error(f"[core_retrain] train exit {train_proc.returncode}: {stderr_tail}")
                send_telegram(
                    f"❌ Core train falhou (exit {train_proc.returncode}).\n"
                    f"`{stderr_tail}`"
                )
        except Exception as e:
            logging.error(f"[core_retrain] {e}", exc_info=True)
            send_telegram(f"❌ Core retrain falhou: {e}")

    # Core retrain cron REMOVED (2026-07-02): CoreRadar runs rule-based in
    # watchlist mode; the ML bundle it retrained is gated off by default
    # (DIPRADAR_USE_ML_SCORE=0). _run_core_monthly stays defined for manual
    # invocation if the ML path is ever revalidated.

    # ── Paper Trading: update diário de posições abertas (22:50 Lisboa) ───────
    # Fecha posições que atingiram o target ou esgotaram o hold_days.
    def _run_paper_update():
        try:
            from portfolio.paper_trading import (
                update_open_positions,
                update_momentum_positions,
                update_core_positions,
                update_missed_trade_outcomes,
                update_core_dividends,
            )
            stats  = update_open_positions()
            stats2 = update_momentum_positions(send_fn=send_telegram)
            stats3 = update_core_positions(send_fn=send_telegram)

            # Send trade card PNG for every trade that just closed
            _just_closed = stats.get("just_closed", []) + stats2.get("just_closed", [])
            if _just_closed:
                try:
                    from utils.chart_generator import generate_trade_card
                    for _trade in _just_closed:
                        _card = generate_trade_card(_trade)
                        if _card and _card.exists():
                            _ticker   = _trade.get("ticker", "")
                            _ret      = _trade.get("return_pct")
                            _alpha    = _trade.get("alpha_pct")
                            _ret_s    = f"{_ret:+.1f}%" if _ret is not None else "—"
                            _alpha_s  = f"{_alpha:+.1f}%" if _alpha is not None else "—"
                            _days     = ""
                            if _trade.get("open_date") and _trade.get("close_date"):
                                from datetime import date as _d
                                _days = f" · {(_d.fromisoformat(_trade['close_date']) - _d.fromisoformat(_trade['open_date'])).days}d"
                            _caption  = (
                                f"Trade closed: {_ticker} [{_trade.get('trade_type','?')}]{_days}\n"
                                f"Return: {_ret_s}  Alpha vs SPY: {_alpha_s}"
                            )
                            send_telegram_photo(_card, caption=_caption)
                except Exception as _ce:
                    logging.debug(f"[trade_card] failed: {_ce}")
            # DRIP: reinvest any new dividends in CORE positions (non-blocking)
            try:
                _ue = get_usdeur() if "get_usdeur" in dir() else 0.92
                drip = update_core_dividends(usd_eur=_ue)
                if drip["events"] > 0:
                    logging.info(
                        f"[core_drip] Reinvested €{drip['total_reinvested_eur']:.2f} "
                        f"across {drip['events']} dividend events. "
                        f"TAX NOTE: {drip['tax_disclaimer'][:80]}"
                    )
            except Exception as _de:
                logging.debug(f"[core_drip] update failed: {_de}")
            # Resolve 90-day outcomes for MISSED_CAPITAL trades (non-blocking)
            try:
                missed_stats = update_missed_trade_outcomes()
                if missed_stats["resolved"] > 0:
                    logging.info(f"[paper] Missed outcomes resolved: {missed_stats}")
            except Exception as _me:
                logging.debug(f"[paper] missed outcome update: {_me}")
            total  = stats.get("updated", 0) + stats2.get("updated", 0)
            drift  = stats2.get("drifted", 0)
            closed = stats2.get("closed", 0)
            if total > 0 or drift > 0 or stats3.get("alerts_sent", 0) > 0:
                logging.info(
                    f"[paper] Update dip={stats} "
                    f"momentum=updated:{stats2.get('updated',0)} "
                    f"drift:{drift} closed:{closed} "
                    f"core=checked:{stats3.get('checked',0)} alerts:{stats3.get('alerts_sent',0)}"
                )
        except Exception as e:
            logging.warning(f"[paper] Update falhou: {e}")

    scheduler.add_job(
        _run_paper_update,
        CronTrigger(day_of_week="mon-fri", hour=22, minute=50, timezone=LISBON_TZ),
        id="paper_update", name="Paper Trading Update",
    )

    # ── Paper Performance: relatório mensal (dia 1, 07:00 Lisboa) ─────────────
    def _run_paper_report():
        try:
            from portfolio.paper_trading import get_monthly_performance, format_performance_report
            perf = get_monthly_performance(months_back=3)
            send_telegram(f"📊 Paper Trading — Relatório Mensal\n\n{format_performance_report(perf)}")
        except Exception as e:
            logging.warning(f"[paper] Report falhou: {e}")

    scheduler.add_job(
        _run_paper_report,
        CronTrigger(day=1, hour=7, minute=0, timezone=LISBON_TZ),
        id="paper_report", name="Paper Trading Report Mensal",
    )

    logging.info("Scheduler iniciado. Jobs activos:")
    for job in scheduler.get_jobs():
        next_time = getattr(job, "next_run_time", "Pendente (aguarda arranque do scheduler)")
        logging.info(f"  • {job.name} — próxima execução: {next_time}")

    _now = datetime.now(LISBON_TZ).strftime("%d/%m/%Y %H:%M")
    send_telegram(
        f"\U0001F916 DipRadar iniciado\n"
        f"{_now}\n"
        f"Scans: Intraday 17h45 | EOD 21h15 (US only)"
    )

    scheduler.start()


if __name__ == "__main__":
    main()
