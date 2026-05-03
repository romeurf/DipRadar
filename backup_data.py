"""
backup_data.py — Backup diário dos dados críticos do DipRadar

Arquiva em /data/backups/YYYY-MM-DD_HH-MM.zip:
  - alert_db.sqlite (ou .db) — base de dados de alertas ML
  - state/*.json     — alertas, recovery watch, weekly log, backtest, etc.
  - data/*.parquet   — snapshots do universo
  - ml_training_*.parquet — datasets de treino
  - dip_models_v3.pkl — bundle ML em produção (schema v3)
  - ml_report_v3.json — métricas do último treino

Retenção: últimos 30 backups (≈1 mês).
Agendado em main.py: seg-sex às 23:30 Lisboa.

Chamada manual:
  python backup_data.py
"""

import os
import glob
import logging
import zipfile
from datetime import datetime
from pathlib import Path

import pytz

log = logging.getLogger(__name__)

LISBON_TZ   = pytz.timezone("Europe/Lisbon")
BACKUP_DIR  = Path("/data/backups")
KEEP_LAST_N = 30

# Padrões de ficheiros a incluir no backup (relativos à raiz do projecto)
_BACKUP_PATTERNS: list[str] = [
    # Base de dados de alertas (SQLite)
    "/data/alert_db.sqlite",
    "/data/alert_db.db",
    # State JSON files
    "/data/alerts.json",
    "/data/recovery_watch.json",
    "/data/weekly_log.json",
    "/data/backtest_log.json",
    "/data/rejected_log.json",
    "/data/persistent_dips.json",
    "/data/positions.json",
    "/data/prediction_log.json",
    # Universe snapshots (parquet)
    "/data/universe_snapshot*.parquet",
    # Datasets de treino ML (raiz do repo)
    "ml_training_merged.parquet",
    "ml_training_fund.parquet",
    "ml_training_price.parquet",
    # Bundle ML em produção (v3 single-file)
    "dip_models_v3.pkl",
    "/data/dip_models_v3.pkl",
    # Relatório de métricas
    "ml_report_v3.json",
    "/data/ml_report_v3.json",
]


def _resolve_files() -> list[Path]:
    """Devolve lista de ficheiros existentes que correspondem aos padrões."""
    found: list[Path] = []
    for pattern in _BACKUP_PATTERNS:
        # glob suporta wildcards; Path.glob não funciona com paths absolutos
        matches = glob.glob(pattern)
        for m in matches:
            p = Path(m)
            if p.exists() and p.is_file():
                found.append(p)
    # Deduplica mantendo ordem
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def _prune_old_backups() -> int:
    """Remove backups mais antigos, mantendo apenas os últimos KEEP_LAST_N."""
    zips = sorted(BACKUP_DIR.glob("*.zip"))
    to_delete = zips[: max(0, len(zips) - KEEP_LAST_N)]
    for z in to_delete:
        try:
            z.unlink()
            log.info(f"[backup] Removido backup antigo: {z.name}")
        except Exception as e:
            log.warning(f"[backup] Não foi possível remover {z.name}: {e}")
    return len(to_delete)


def run_backup() -> dict:
    """
    Cria um ficheiro ZIP com todos os dados críticos.

    Devolve um dict com:
      - zip_path   : caminho absoluto do ZIP criado
      - file_count : número de ficheiros incluídos
      - size_mb    : tamanho do ZIP em MB
      - pruned     : número de backups antigos removidos
      - ts         : timestamp da execução (ISO)
      - error      : mensagem de erro (ou None)
    """
    now     = datetime.now(LISBON_TZ)
    ts      = now.strftime("%Y-%m-%d_%H-%M")
    ts_iso  = now.isoformat()

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = BACKUP_DIR / f"{ts}.zip"

    files   = _resolve_files()
    result  = {
        "zip_path":   str(zip_path),
        "file_count": 0,
        "size_mb":    0.0,
        "pruned":     0,
        "ts":         ts_iso,
        "error":      None,
    }

    if not files:
        log.warning("[backup] Nenhum ficheiro encontrado para fazer backup.")
        result["error"] = "Nenhum ficheiro encontrado"
        return result

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for f in files:
                arcname = str(f).lstrip("/")
                zf.write(f, arcname=arcname)
                log.debug(f"[backup] + {arcname}")
        size_mb = zip_path.stat().st_size / 1_048_576
        result["file_count"] = len(files)
        result["size_mb"]    = round(size_mb, 2)
        log.info(f"[backup] ZIP criado: {zip_path.name} ({size_mb:.2f} MB, {len(files)} ficheiros)")
    except Exception as e:
        log.error(f"[backup] Erro ao criar ZIP: {e}", exc_info=True)
        result["error"] = str(e)
        return result

    result["pruned"] = _prune_old_backups()
    return result


def build_telegram_summary(result: dict) -> str:
    """Constrói a mensagem Telegram para o resultado do backup."""
    now_str = datetime.now(LISBON_TZ).strftime("%d/%m/%Y %H:%M")
    if result.get("error"):
        return (
            f"❌ *Backup Diário — Erro*\n"
            f"_{result['error']}_\n"
            f"_⏰ {now_str}_"
        )
    pruned_str = f" | 🗑️ {result['pruned']} antigo(s) removido(s)" if result["pruned"] else ""
    zip_name   = Path(result["zip_path"]).name
    return (
        f"💾 *Backup Diário — OK*\n"
        f"_⏰ {now_str}_\n\n"
        f"  📦 `{zip_name}`\n"
        f"  📁 {result['file_count']} ficheiros incluídos\n"
        f"  📏 {result['size_mb']} MB{pruned_str}\n"
        f"  📂 `/data/backups/`\n\n"
        f"_Retenção: últimos {KEEP_LAST_N} backups_"
    )


# ── Entrada directa ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    res = run_backup()
    print(build_telegram_summary(res))
