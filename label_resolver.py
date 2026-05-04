"""
label_resolver.py — Motor de Resolução de Labels para o Flywheel de ML.

Responsabilidade:
  Varre o alert_db.csv em busca de alertas com label_win == None
  (i.e. ainda não resolvidos) cuja data de alerta seja >= 60 dias atrás.
  Para cada um, busca a série de preços dos 60 dias seguintes ao alerta
  e calcula as duas labels que o train_model.py precisa:

    label_win (int, Model A target):
      1  se o preço máximo nos 60 dias seguintes >= preço_alerta * 1.15
      0  caso contrário

    label_further_drop (float, Model B target):
      Drawdown máximo observado ANTES do primeiro cruzamento do target +15%.
      Se o target nunca for atingido, é o drawdown máximo de todo o período.
      Mede o "calor" que tiveste de aguentar antes de ganhar dinheiro.
      Ex: -0.042 significa que a acção caiu mais 4.2% antes de recuperar.

Point-in-Time Guarantee:
  O preço de referência (price_alert) e todas as features foram congelados
  no momento do alerta. Esta função nunca retoca as features — só preenche
  as labels com dados que seriam conhecidos 60 dias depois do alerta.
  Não há look-ahead bias.

Eficiência de Rede:
  Agrupa os tickers com labels pendentes e faz UM único request por ticker
  (janela alargada que cobre todos os alertas pendentes desse ticker).
  Evita rate limits em scans com dezenas de linhas pendentes.

Robustez de Calendário:
  A data_alerta + 60 dias no calendário civil pode cair num sábado ou
  num feriado onde não há preço. O resolver normaliza para o dia útil
  mais próximo disponível na série (nunca ultrapassa a data de hoje).

Idempotência:
  Linhas já resolvidas (label_win não é None/NaN/'') são ignoradas.
  Seguro correr diariamente.

Fontes de preço (por ordem de prioridade):
  1. Tiingo EOD  (TIINGO_API_KEY no env)
  2. yfinance    (fallback automático)

Outputs:
  - alert_db.csv actualizado em disco
  - dict com estatísticas do run {resolved, skipped, errors, total_pending}

Uso:
  python label_resolver.py              # corre uma vez e imprime stats
  python label_resolver.py --dry-run   # mostra pendentes sem escrever

Job no scheduler (main.py):
  Cron: todos os dias às 03:00 Lisboa — após fecho US + 5h de buffer.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Caminhos ──────────────────────────────────────────────────────────────────

_DATA_DIR = Path("/data") if Path("/data").exists() else Path("/tmp")
_ALERT_DB = _DATA_DIR / "alert_db.csv"

# ── Configuração ──────────────────────────────────────────────────────────────

_RESOLUTION_DAYS   = 60    # janela de observação após o alerta
_WIN_THRESHOLD     = 0.15  # +15% para label_win = 1
_MIN_AGE_DAYS      = 60    # só resolve alertas com pelo menos esta antiguidade
_TIINGO_API_KEY    = os.environ.get("TIINGO_API_KEY", "")
_TIINGO_RATE_SLEEP = 0.5   # segundos entre requests Tiingo
_YFINANCE_SLEEP    = 1.0   # segundos entre requests yfinance


# ── 1. Carga e filtragem do CSV ───────────────────────────────────────────────

def _load_pending(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega o alert_db.csv e separa em:
      pending: linhas sem label_win resolvido com >= _MIN_AGE_DAYS
      rest:    todas as outras linhas (já resolvidas ou demasiado recentes)

    Devolve (pending_df, full_df) — full_df inclui todas as linhas originais.

    Usa on_bad_lines='warn' para não crashar com linhas mal formadas
    (ex: campos de texto com vírgulas em CSVs antigos sem quoting).
    """
    if not db_path.exists():
        logger.warning(f"[resolver] {db_path} não existe — nada a resolver.")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(db_path, dtype=str, on_bad_lines="warn", engine="python")
    if df.empty:
        return pd.DataFrame(), df

    # Normaliza a coluna de data (aceita 'date_iso' ou 'date')
    date_col = "date_iso" if "date_iso" in df.columns else "date" if "date" in df.columns else None
    if date_col is None:
        logger.error("[resolver] alert_db.csv não tem coluna 'date_iso' nem 'date'.")
        return pd.DataFrame(), df

    df["_date_parsed"] = pd.to_datetime(df[date_col], errors="coerce").dt.date

    cutoff = date.today() - timedelta(days=_MIN_AGE_DAYS)

    def _is_unresolved(row) -> bool:
        val = row.get("label_win", None)
        if val is None:
            return True
        val_str = str(val).strip()
        return val_str == "" or val_str.lower() in ("none", "nan", "null", "")

    pending_mask = (
        df.apply(_is_unresolved, axis=1)
        & (df["_date_parsed"].notna())
        & (df["_date_parsed"] <= cutoff)
    )

    pending = df[pending_mask].copy()
    logger.info(
        f"[resolver] Total linhas: {len(df)} | "
        f"Pendentes (>={_MIN_AGE_DAYS}d): {len(pending)}"
    )
    return pending, df


# ── 2. Fetch de preços por ticker ─────────────────────────────────────────────

def _fetch_prices_tiingo(
    ticker: str,
    start: date,
    end: date,
) -> Optional[pd.Series]:
    """
    Busca série diária de closes via Tiingo EOD API.
    Devolve pd.Series com index=date, values=float (adjusted close).
    Devolve None se o request falhar ou a chave não existir.
    """
    if not _TIINGO_API_KEY:
        return None
    try:
        import requests
        url = (
            f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
            f"?startDate={start.isoformat()}&endDate={end.isoformat()}"
            f"&resampleFreq=daily&token={_TIINGO_API_KEY}"
        )
        resp = requests.get(url, timeout=15)
        if resp.status_code == 404:
            logger.debug(f"[resolver][tiingo] {ticker}: 404 — ticker desconhecido")
            return None
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        series = pd.Series(
            {pd.to_datetime(row["date"]).date(): float(row["adjClose"])
             for row in data if row.get("adjClose") is not None}
        ).sort_index()
        return series if len(series) >= 2 else None
    except Exception as e:
        logger.warning(f"[resolver][tiingo] {ticker}: {e}")
        return None


def _fetch_prices_yfinance(
    ticker: str,
    start: date,
    end: date,
) -> Optional[pd.Series]:
    """
    Fallback: busca serie diária de closes via yfinance.
    Devolve pd.Series com index=date, values=float.
    """
    try:
        import yfinance as yf
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            progress=False,
            auto_adjust=True,
        )
        if df.empty or "Close" not in df.columns:
            return None
        series = df["Close"].dropna()
        series.index = series.index.date
        return series.sort_index() if len(series) >= 2 else None
    except Exception as e:
        logger.warning(f"[resolver][yfinance] {ticker}: {e}")
        return None


def _fetch_prices(
    ticker: str,
    start: date,
    end: date,
    use_sleep: bool = True,
) -> Optional[pd.Series]:
    """
    Orchestrador: tenta Tiingo primeiro, cai para yfinance.
    Limita end a date.today() para não pedir dados futuros.
    """
    safe_end = min(end, date.today())
    if start >= safe_end:
        return None

    series = _fetch_prices_tiingo(ticker, start, safe_end)
    if use_sleep:
        time.sleep(_TIINGO_RATE_SLEEP)

    if series is None:
        series = _fetch_prices_yfinance(ticker, start, safe_end)
        if use_sleep:
            time.sleep(_YFINANCE_SLEEP)

    return series


# ── 3. Cálculo das labels ─────────────────────────────────────────────────────

def _nearest_business_price(
    series: pd.Series,
    target_date: date,
) -> Optional[float]:
    """
    Devolve o preço de fecho mais próximo de target_date na série.
    Procura primeiro o próprio dia, depois retrocede até 5 dias úteis.
    Devolve None se não encontrar nada num raio de 5 dias.
    """
    for delta in range(0, 6):
        check_date = target_date - timedelta(days=delta)
        if check_date in series.index:
            return float(series[check_date])
    for delta in range(1, 4):
        check_date = target_date + timedelta(days=delta)
        if check_date in series.index:
            return float(series[check_date])
    return None


def _compute_labels(
    series: pd.Series,
    alert_date: date,
    price_alert: float,
) -> tuple[int, float]:
    """
    Calcula label_win e label_further_drop para um alerta.

    label_win = 1 se max(precos_janela) >= price_alert * (1 + WIN_THRESHOLD)

    label_further_drop:
      Se label_win == 1: drawdown máximo até ao primeiro dia >= target.
      Se label_win == 0: drawdown máximo de toda a janela.
    """
    window_start = alert_date + timedelta(days=1)
    window_end   = alert_date + timedelta(days=_RESOLUTION_DAYS)

    window_prices = series[
        (series.index >= window_start) & (series.index <= window_end)
    ]

    if window_prices.empty or price_alert <= 0:
        return -1, 0.0  # sentinela de erro

    target_price   = price_alert * (1.0 + _WIN_THRESHOLD)
    prices         = window_prices.values.astype(float)
    max_price      = float(np.max(prices))

    label_win: int = 1 if max_price >= target_price else 0

    if label_win == 1:
        first_win_idx = next(
            (i for i, p in enumerate(prices) if p >= target_price),
            len(prices) - 1,
        )
        sub_prices = prices[:first_win_idx + 1]
    else:
        sub_prices = prices

    if len(sub_prices) == 0:
        label_further_drop = 0.0
    else:
        min_price          = float(np.min(sub_prices))
        label_further_drop = round((min_price / price_alert) - 1.0, 6)
        label_further_drop = min(label_further_drop, 0.0)

    return label_win, label_further_drop


# ── 4. Orquestração principal ─────────────────────────────────────────────────

def resolve_pending_labels(
    db_path: Path = _ALERT_DB,
    dry_run: bool = False,
) -> dict:
    """
    Entry point principal.

    1. Carrega pending do alert_db.csv
    2. Agrupa por ticker (1 request por stock, janela alargada)
    3. Para cada alerta pendente desse ticker, calcula as labels
    4. Actualiza o CSV
    5. Devolve stats
    """
    stats = {"resolved": 0, "skipped": 0, "errors": 0, "total_pending": 0}

    pending, full_df = _load_pending(db_path)
    if pending.empty:
        logger.info("[resolver] Nenhum alerta pendente. Nada a fazer.")
        return stats

    stats["total_pending"] = len(pending)

    ticker_col = "symbol" if "symbol" in pending.columns else "ticker"
    if ticker_col not in pending.columns:
        logger.error("[resolver] alert_db.csv não tem coluna 'symbol' nem 'ticker'.")
        return stats

    ticker_groups: dict[str, list[dict]] = {}
    for idx, row in pending.iterrows():
        ticker = str(row.get(ticker_col, "")).strip().upper()
        if not ticker:
            stats["errors"] += 1
            continue

        alert_date = row.get("_date_parsed")
        if not isinstance(alert_date, date):
            stats["errors"] += 1
            continue

        price_raw = row.get("price") or row.get("price_alert") or row.get("close")
        try:
            price_alert = float(str(price_raw).replace(",", "."))
        except (TypeError, ValueError):
            logger.warning(f"[resolver] {ticker} idx={idx}: preço inválido ({price_raw!r}) — a ignorar")
            stats["skipped"] += 1
            continue

        if price_alert <= 0:
            stats["skipped"] += 1
            continue

        ticker_groups.setdefault(ticker, []).append({
            "idx":         idx,
            "alert_date":  alert_date,
            "price_alert": price_alert,
        })

    logger.info(f"[resolver] Tickers únicos com pendentes: {len(ticker_groups)}")

    for ticker, alerts in ticker_groups.items():
        all_dates    = [a["alert_date"] for a in alerts]
        fetch_start  = min(all_dates)
        fetch_end    = max(all_dates) + timedelta(days=_RESOLUTION_DAYS + 5)

        logger.info(
            f"[resolver] {ticker}: {len(alerts)} alerta(s) | "
            f"fetch {fetch_start} → {fetch_end}"
        )

        series = _fetch_prices(ticker, fetch_start, fetch_end, use_sleep=True)

        if series is None or series.empty:
            logger.warning(f"[resolver] {ticker}: sem dados de preço — a saltar {len(alerts)} alertas")
            stats["skipped"] += len(alerts)
            continue

        for alert in alerts:
            idx         = alert["idx"]
            alert_date  = alert["alert_date"]
            price_alert = alert["price_alert"]

            window_end   = alert_date + timedelta(days=_RESOLUTION_DAYS)
            window_prices = series[
                (series.index > alert_date) & (series.index <= window_end)
            ]

            if len(window_prices) < 5:
                logger.debug(
                    f"[resolver] {ticker} @ {alert_date}: "
                    f"apenas {len(window_prices)} dias na janela — a saltar"
                )
                stats["skipped"] += 1
                continue

            label_win, label_further_drop = _compute_labels(
                series, alert_date, price_alert
            )

            if label_win == -1:
                stats["errors"] += 1
                continue

            if not dry_run:
                full_df.at[idx, "label_win"]          = label_win
                full_df.at[idx, "label_further_drop"] = label_further_drop

            stats["resolved"] += 1
            logger.info(
                f"[resolver] {ticker} @ {alert_date} (${price_alert:.2f}): "
                f"label_win={label_win} | "
                f"further_drop={label_further_drop:.3f} | "
                f"n_days={len(window_prices)}"
            )

    # ── Persistir CSV ─────────────────────────────────────────────────────────
    if not dry_run and stats["resolved"] > 0:
        if "_date_parsed" in full_df.columns:
            full_df = full_df.drop(columns=["_date_parsed"])

        try:
            full_df.to_csv(db_path, index=False)
            logger.info(
                f"[resolver] CSV actualizado: {stats['resolved']} labels preenchidas "
                f"| {db_path}"
            )
        except Exception as e:
            logger.error(f"[resolver] Erro ao guardar CSV: {e}")
            stats["errors"] += 1
    else:
        if "_date_parsed" in full_df.columns:
            full_df = full_df.drop(columns=["_date_parsed"])

    if stats["resolved"] > 0 and not dry_run:
        try:
            fresh = pd.read_csv(db_path, dtype=str, on_bad_lines="warn", engine="python")
            wins  = (fresh["label_win"] == "1").sum()
            loses = (fresh["label_win"] == "0").sum()
            total = wins + loses
            win_rate = wins / total * 100 if total > 0 else 0
            logger.info(
                f"[resolver] Snapshot pós-run: "
                f"WIN={wins} | LOSS={loses} | WR={win_rate:.1f}%"
            )
        except Exception:
            pass

    return stats


# ── 5. Wrapper para o scheduler ───────────────────────────────────────────────

def run_label_resolver_job(
    send_telegram_fn=None,
    telegram_chat_id: str = "",
) -> None:
    """
    Wrapper síncrono chamado pelo APScheduler às 03:00.
    Executa resolve_pending_labels() e envia sumário pelo Telegram.
    """
    from datetime import datetime
    import pytz
    lisbon_tz = pytz.timezone("Europe/Lisbon")
    now_str   = datetime.now(lisbon_tz).strftime("%d/%m/%Y %H:%M")

    logger.info("[label_resolver] Job das 03:00 a iniciar...")
    try:
        stats = resolve_pending_labels()
    except Exception as e:
        logger.error(f"[label_resolver] Erro inesperado: {e}", exc_info=True)
        if send_telegram_fn:
            send_telegram_fn(
                f"❌ *Label Resolver — Erro*\n"
                f"`{e}`\n_⏰ {now_str}_"
            )
        return

    resolved      = stats["resolved"]
    skipped       = stats["skipped"]
    errors        = stats["errors"]
    total_pending = stats["total_pending"]

    if total_pending == 0:
        logger.info("[label_resolver] Nenhum pendente — sem notificação.")
        return

    win_rate_str = ""
    try:
        if _ALERT_DB.exists():
            df      = pd.read_csv(_ALERT_DB, dtype=str, on_bad_lines="warn", engine="python")
            labeled = df[df["label_win"].notna() & (df["label_win"] != "") & (~df["label_win"].isin(["None", "nan"]))]
            wins    = (labeled["label_win"] == "1").sum()
            total_l = len(labeled)
            if total_l > 0:
                wr = wins / total_l * 100
                win_rate_str = (
                    f"\n\n*📊 Win Rate acumulado:*\n"
                    f"  WIN: *{wins}* | TOTAL: *{total_l}* | *{wr:.1f}%*"
                )
    except Exception:
        pass

    lines = [
        f"🏷️ *Label Resolver — Run Diário*",
        f"_{now_str}_",
        "",
        f"*📋 Pendentes encontrados:* {total_pending}",
        f"*✅ Labels preenchidas:*   {resolved}",
        f"*⏩ Ignorados:*            {skipped}  _(dados insuficientes)_",
    ]
    if errors:
        lines.append(f"*❌ Erros:*                 {errors}")
    if win_rate_str:
        lines.append(win_rate_str)

    lines.append("")
    try:
        if _ALERT_DB.exists():
            df      = pd.read_csv(_ALERT_DB, dtype=str, on_bad_lines="warn", engine="python")
            labeled = df[df["label_win"].notna() & (df["label_win"] != "") & (~df["label_win"].isin(["None", "nan"]))]
            total_l = len(labeled)
            if total_l < 30:
                lines.append(f"_🕐 {30 - total_l} alertas até ao mínimo de treino (30)_")
            elif total_l < 200:
                lines.append(f"_🤖 {total_l}/200 — SMOTE activo até 200 amostras_")
            else:
                lines.append(f"_🟢 {total_l} amostras — modelo já treina sem SMOTE_")
    except Exception:
        pass

    if send_telegram_fn:
        send_telegram_fn("\n".join(lines))
    else:
        print("\n".join(lines))

    logger.info(
        f"[label_resolver] Concluído: "
        f"resolved={resolved} skipped={skipped} errors={errors}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="DipRadar — Label Resolver")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Mostra alertas pendentes e labels calculadas sem escrever no CSV."
    )
    parser.add_argument(
        "--db", default=str(_ALERT_DB),
        help=f"Caminho para o alert_db.csv (default: {_ALERT_DB})"
    )
    args = parser.parse_args()

    if args.dry_run:
        print(f"[dry-run] A ler: {args.db}")
        stats = resolve_pending_labels(db_path=Path(args.db), dry_run=True)
        print(f"\nResultado (dry-run):\n{json.dumps(stats, indent=2)}")
        print("\n[dry-run] Nenhum ficheiro foi modificado.")
    else:
        run_label_resolver_job()
