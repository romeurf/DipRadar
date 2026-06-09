"""
portfolio_simulator.py — Simulação de portfolio para medir o retorno real do DipRadar.

Responde à pergunta: "Se tivesse seguido todas as recomendações do bot,
qual seria o meu retorno anual e qual o risco que corri?"

Usa o alert_db.csv como fonte — alertas históricos com outcomes reais já preenchidos
pelo fill_db_outcomes(). Simula um portfolio seguindo as recomendações do bot:
  - COMPRAR com score ≥ threshold → abre posição
  - Exit em T+91 dias (retorno real de return_3m ou return_60d)
  - Sizing proporcional ao score
  - Transaction costs realistas (0.10% por trade — T212 fractional)
  - Compara vs SPY buy-and-hold no mesmo período

Output:
  PortfolioResult dataclass com métricas completas + equity curve + lista de trades.

Usado pelo comando /performance no Telegram.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_DATA_DIR = Path("/data") if Path("/data").exists() else Path("/tmp")
_ALERT_DB = _DATA_DIR / "alert_db.csv"
_REPO_ALERT_DB = Path(__file__).parent / "data" / "alert_db.csv"

_RISK_FREE_RATE = 0.04   # 4% risk-free (aproximação Euribor 2025)
_TRANSACTION_COST = 0.0010   # 0.10% por trade (T212 fractional shares)


@dataclass
class Trade:
    symbol:       str
    alert_date:   date
    exit_date:    date
    score:        float
    verdict:      str
    category:     str
    entry_amount: float
    return_pct:   float      # retorno real do trade (% ex: 18.5 = +18.5%)
    spy_return_pct: float    # SPY no mesmo período (para calcular alpha)
    pnl_eur:      float      # P&L em euros (líquido de transaction cost)
    alpha_pct:    float      # return_pct - spy_return_pct (excesso vs benchmark)


@dataclass
class PortfolioResult:
    # ── Parâmetros da simulação ─────────────────────────────────────────────
    period_start:    date
    period_end:      date
    period_years:    float
    initial_capital: float
    final_capital:   float
    score_threshold: float

    # ── Retornos ─────────────────────────────────────────────────────────────
    total_return_pct:  float    # retorno total do portfolio (%)
    annual_return_pct: float    # CAGR anualizado (%)
    benchmark_spy_pct: float    # retorno do SPY buy-and-hold no mesmo período (%)
    alpha_pct:         float    # annual_return - benchmark (alpha gerado)

    # ── Risco ────────────────────────────────────────────────────────────────
    sharpe_ratio:      float    # (annual_return - risk_free) / annual_vol
    max_drawdown_pct:  float    # pior pico-a-vale do portfolio (%)
    annual_vol_pct:    float    # volatilidade anualizada dos trade returns (%)

    # ── Trades ───────────────────────────────────────────────────────────────
    n_trades:          int
    n_wins:            int
    n_losses:          int
    win_rate_pct:      float
    best_trade:        Optional[Trade] = None
    worst_trade:       Optional[Trade] = None

    # ── Detalhes ─────────────────────────────────────────────────────────────
    trades:            list[Trade] = field(default_factory=list)
    monthly_returns:   dict[str, float] = field(default_factory=dict)
    n_skipped_no_outcome: int = 0
    n_skipped_low_score:  int = 0
    warnings:          list[str] = field(default_factory=list)


# ── Carregamento de dados ─────────────────────────────────────────────────────

def _load_alert_db() -> pd.DataFrame:
    """Carrega alert_db.csv. Tenta /data/ primeiro, depois repo/data/."""
    for p in (_ALERT_DB, _REPO_ALERT_DB):
        if p.exists():
            try:
                df = pd.read_csv(p, on_bad_lines="warn", engine="python")
                log.info(f"[portfolio] Loaded {len(df)} alerts from {p}")
                return df
            except Exception as e:
                log.warning(f"[portfolio] Falha a ler {p}: {e}")
    return pd.DataFrame()


def _safe_float(v, default: float = float("nan")) -> float:
    if v is None:
        return default
    try:
        f = float(str(v).replace(",", ".").strip())
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


# ── Sizing ────────────────────────────────────────────────────────────────────

def _position_size(
    score: float,
    portfolio: float,
    score_threshold: float,
    max_position_pct: float = 0.08,
) -> float:
    """Sizing proporcional ao score, entre 3% e 8% do portfolio.

    score ≥ 85 → 8% (máximo — alta convicção)
    score ≥ 75 → 6%
    score ≥ threshold → 4%
    """
    if score >= 85:
        frac = 0.08
    elif score >= 75:
        frac = 0.06
    elif score >= score_threshold:
        frac = 0.04
    else:
        return 0.0
    frac = min(frac, max_position_pct)
    return portfolio * frac


# ── Simulação ──────────────────────────────────────────────────────────────────

def run_portfolio_backtest(
    period_start: str | None = None,
    period_end:   str | None = None,
    initial_capital: float = 10_000.0,
    score_threshold: float = 60.0,
    exit_horizon_days: int = 91,   # ≈ 3 meses
    max_position_pct: float = 0.08,
    transaction_cost_pct: float = _TRANSACTION_COST,
    min_trade_eur: float = 20.0,
    verdict_filter: tuple[str, ...] = ("COMPRAR",),
) -> PortfolioResult:
    """Simula o portfolio DipRadar no período indicado.

    Parameters
    ----------
    period_start : YYYY-MM-DD ou None (usa o alerta mais antigo)
    period_end   : YYYY-MM-DD ou None (usa hoje)
    initial_capital : capital inicial simulado
    score_threshold : score mínimo para abrir posição (default 60)
    exit_horizon_days : dias até à saída (usa return_3m ≈ 91 dias)
    max_position_pct  : tamanho máximo por posição (% do portfolio)
    transaction_cost_pct : custo por trade (ambas as pernas)
    """
    df = _load_alert_db()
    if df.empty:
        raise RuntimeError("alert_db.csv vazio ou não encontrado")

    # Parse de datas
    df["_date"] = pd.to_datetime(df.get("date_iso", ""), errors="coerce").dt.date
    df = df.dropna(subset=["_date"])

    today = date.today()
    p_start = date.fromisoformat(period_start) if period_start else df["_date"].min()
    p_end   = date.fromisoformat(period_end) if period_end else today

    df = df[(df["_date"] >= p_start) & (df["_date"] <= p_end)].copy()
    if df.empty:
        raise RuntimeError(f"Sem alertas no período {p_start} → {p_end}")

    df = df.sort_values("_date").reset_index(drop=True)

    # Usar return_3m como saída standard (≈ 91 dias)
    # fallback: return_60d se return_3m não disponível
    df["_return_pct"] = df.apply(
        lambda r: _safe_float(r.get("return_3m"))
                  if not math.isnan(_safe_float(r.get("return_3m")))
                  else _safe_float(r.get("return_60d")),
        axis=1,
    )
    df["_spy_pct"]    = df.apply(
        lambda r: _safe_float(r.get("spy_return_60d")),   # melhor proxy disponível
        axis=1,
    )
    df["_score"]   = df["score"].apply(_safe_float)
    df["_verdict"] = df.get("verdict", "").fillna("").astype(str)
    df["_category"]= df.get("category", "").fillna("").astype(str)
    df["_symbol"]  = df.get("symbol", "").fillna("").astype(str)

    trades: list[Trade] = []
    n_skip_outcome = 0
    n_skip_score   = 0
    portfolio      = initial_capital

    for _, row in df.iterrows():
        score   = row["_score"]
        verdict = row["_verdict"]
        ret_pct = row["_return_pct"]
        spy_pct = row["_spy_pct"]
        alert_d = row["_date"]
        symbol  = row["_symbol"]
        category= row["_category"]

        # Filtrar por verdict e score
        if verdict not in verdict_filter:
            n_skip_score += 1
            continue
        if score < score_threshold or math.isnan(score):
            n_skip_score += 1
            continue

        # Filtrar por outcome resolvido
        if math.isnan(ret_pct):
            n_skip_outcome += 1
            continue

        # Posição
        amount = _position_size(score, portfolio, score_threshold, max_position_pct)
        if amount < min_trade_eur:
            continue

        # Transaction cost (entrada + saída)
        tc = amount * transaction_cost_pct * 2
        pnl_gross = amount * (ret_pct / 100.0)
        pnl_net   = pnl_gross - tc
        portfolio += pnl_net

        alpha = ret_pct - spy_pct if not math.isnan(spy_pct) else float("nan")
        exit_d = alert_d + timedelta(days=exit_horizon_days)

        trades.append(Trade(
            symbol=symbol,
            alert_date=alert_d,
            exit_date=min(exit_d, today),
            score=score,
            verdict=verdict,
            category=category,
            entry_amount=amount,
            return_pct=ret_pct,
            spy_return_pct=spy_pct if not math.isnan(spy_pct) else 0.0,
            pnl_eur=round(pnl_net, 2),
            alpha_pct=alpha if not math.isnan(alpha) else 0.0,
        ))

    if not trades:
        raise RuntimeError(
            f"Nenhum trade executável no período. "
            f"Skipped: {n_skip_outcome} sem outcome, {n_skip_score} abaixo do threshold."
        )

    # ── Métricas ─────────────────────────────────────────────────────────────

    total_ret   = (portfolio - initial_capital) / initial_capital
    period_days = (p_end - p_start).days
    period_yrs  = max(period_days / 365.25, 1e-6)
    annual_ret  = (1 + total_ret) ** (1 / period_yrs) - 1

    # SPY buy-and-hold no mesmo período (tentamos via yfinance, fallback à média)
    spy_bah = float("nan")
    try:
        import yfinance as yf
        spy_hist = yf.Ticker("SPY").history(
            start=p_start.isoformat(),
            end=(p_end + timedelta(days=1)).isoformat(),
            auto_adjust=True,
        )
        if not spy_hist.empty:
            spy_entry = float(spy_hist["Close"].iloc[0])
            spy_exit  = float(spy_hist["Close"].iloc[-1])
            spy_bah   = (spy_exit / spy_entry - 1) * 100  # %
    except Exception as e:
        log.warning(f"[portfolio] SPY fetch falhou: {e} — usando média dos spy_return_60d")
        spy_vals = [t.spy_return_pct for t in trades if t.spy_return_pct != 0]
        if spy_vals:
            # Rough annualise da média de retornos de 60d
            spy_60d_mean = float(np.mean(spy_vals))
            spy_bah      = spy_60d_mean * 365 / 60  # anualizado approximado

    spy_bah_annual = ((1 + spy_bah / 100) ** (1 / period_yrs) - 1) * 100 if not math.isnan(spy_bah) else float("nan")
    alpha = annual_ret * 100 - spy_bah_annual if not math.isnan(spy_bah_annual) else float("nan")

    # Volatilidade e Sharpe (usando trade returns como série)
    rets_arr = np.array([t.return_pct / 100 for t in trades])
    annual_vol = float(np.std(rets_arr) * np.sqrt(len(trades) / max(period_yrs, 0.1)))
    if annual_vol > 1e-6:
        sharpe = (annual_ret - _RISK_FREE_RATE) / annual_vol
    else:
        sharpe = 0.0

    # Max drawdown (sobre equity curve reconstruída por trade)
    cap = initial_capital
    peak = cap
    max_dd = 0.0
    for t in sorted(trades, key=lambda x: x.exit_date):
        cap += t.pnl_eur
        if cap > peak:
            peak = cap
        dd = (cap - peak) / peak
        if dd < max_dd:
            max_dd = dd

    wins   = [t for t in trades if t.pnl_eur > 0]
    losses = [t for t in trades if t.pnl_eur <= 0]

    best  = max(trades, key=lambda t: t.return_pct) if trades else None
    worst = min(trades, key=lambda t: t.return_pct) if trades else None

    # Retornos mensais (agrupados pelo mês de exit)
    monthly: dict[str, float] = {}
    for t in trades:
        key = t.exit_date.strftime("%Y-%m")
        monthly[key] = monthly.get(key, 0.0) + t.pnl_eur

    # Warnings
    warnings: list[str] = []
    if n_skip_outcome > len(trades):
        warnings.append(
            f"{n_skip_outcome} alertas sem outcome resolvido — preenche outcomes via /mldata update"
        )
    if period_yrs < 0.5:
        warnings.append("Período < 6 meses — resultados podem ser pouco representativos")

    return PortfolioResult(
        period_start=p_start,
        period_end=p_end,
        period_years=round(period_yrs, 2),
        initial_capital=initial_capital,
        final_capital=round(portfolio, 2),
        score_threshold=score_threshold,
        total_return_pct=round(total_ret * 100, 2),
        annual_return_pct=round(annual_ret * 100, 2),
        benchmark_spy_pct=round(spy_bah_annual, 2) if not math.isnan(spy_bah_annual) else 0.0,
        alpha_pct=round(alpha, 2) if not math.isnan(alpha) else 0.0,
        sharpe_ratio=round(sharpe, 2),
        max_drawdown_pct=round(max_dd * 100, 2),
        annual_vol_pct=round(annual_vol * 100, 2),
        n_trades=len(trades),
        n_wins=len(wins),
        n_losses=len(losses),
        win_rate_pct=round(len(wins) / len(trades) * 100, 1),
        best_trade=best,
        worst_trade=worst,
        trades=trades,
        monthly_returns=dict(sorted(monthly.items())),
        n_skipped_no_outcome=n_skip_outcome,
        n_skipped_low_score=n_skip_score,
        warnings=warnings,
    )


# ── Formatação Telegram ────────────────────────────────────────────────────────

def format_portfolio_result(r: PortfolioResult) -> str:
    """Formata o resultado do backtest para Telegram (Markdown)."""
    sign = "+" if r.total_return_pct >= 0 else ""
    ann_sign = "+" if r.annual_return_pct >= 0 else ""
    alpha_sign = "+" if r.alpha_pct >= 0 else ""
    spy_sign = "+" if r.benchmark_spy_pct >= 0 else ""

    # Emoji de performance
    if r.annual_return_pct >= 15:
        perf_emoji = "🟢"
    elif r.annual_return_pct >= 8:
        perf_emoji = "🟡"
    else:
        perf_emoji = "🔴"

    alpha_emoji = "📈" if r.alpha_pct > 0 else "📉"

    # Sharpe label
    if r.sharpe_ratio >= 1.5:
        sharpe_label = "_(excelente)_"
    elif r.sharpe_ratio >= 0.8:
        sharpe_label = "_(bom)_"
    elif r.sharpe_ratio >= 0.5:
        sharpe_label = "_(aceitável)_"
    else:
        sharpe_label = "_(fraco)_"

    lines = [
        f"📊 *Simulação de Portfolio DipRadar*",
        f"_{r.period_start} → {r.period_end} ({r.period_years:.1f} anos) | score ≥ {r.score_threshold:.0f}_",
        "",
        f"💶 Capital: €{r.initial_capital:,.0f} → *€{r.final_capital:,.0f}*",
        "",
        f"*📈 Retornos:*",
        f"  {perf_emoji} Total:           *{sign}{r.total_return_pct:.1f}%*",
        f"  {perf_emoji} Anualizado:      *{ann_sign}{r.annual_return_pct:.1f}%*",
        f"  {alpha_emoji} SPY buy-and-hold: {spy_sign}{r.benchmark_spy_pct:.1f}%",
        f"  {alpha_emoji} Alpha gerado:    *{alpha_sign}{r.alpha_pct:.1f}pp*",
        "",
        f"*⚡ Risco:*",
        f"  Sharpe Ratio:    *{r.sharpe_ratio:.2f}* {sharpe_label}",
        f"  Max Drawdown:    *{r.max_drawdown_pct:.1f}%*",
        f"  Volatilidade:    *{r.annual_vol_pct:.1f}%/ano*",
        "",
        f"*🎯 Trades ({r.n_trades} executados):*",
        f"  WIN: *{r.n_wins}* ({r.win_rate_pct:.0f}%)  |  LOSS: *{r.n_losses}*",
    ]

    if r.best_trade:
        b = r.best_trade
        lines.append(
            f"  Melhor: *{b.symbol}* +{b.return_pct:.1f}% "
            f"({b.alert_date.strftime('%b %y')})"
        )
    if r.worst_trade:
        w = r.worst_trade
        lines.append(
            f"  Pior:   *{w.symbol}* {w.return_pct:+.1f}% "
            f"({w.alert_date.strftime('%b %y')})"
        )

    # Monthly breakdown (últimos 6 meses)
    if r.monthly_returns:
        lines.append("")
        lines.append("*📅 P&L mensal (últimos 6 meses):*")
        months = sorted(r.monthly_returns.keys())[-6:]
        for m in months:
            pnl = r.monthly_returns[m]
            em = "🟢" if pnl > 0 else "🔴"
            lines.append(f"  {em} {m}: *€{pnl:+.0f}*")

    if r.warnings:
        lines.append("")
        for w in r.warnings:
            lines.append(f"_⚠️ {w}_")

    if r.n_skipped_no_outcome > 0:
        lines.append(
            f"\n_{r.n_skipped_no_outcome} alertas sem outcome (ainda dentro dos 90d ou sem dados)._"
        )

    lines.append("\n_⚠️ Simulação sem custo de tempo real ou slippage. "
                 "Não é garantia de performance futura._")

    return "\n".join(lines)
