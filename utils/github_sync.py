"""
github_sync.py — Push de artefactos ML para o repositório GitHub.

Elimina a dependência do Colab: o bot trata de tudo autonomamente.
Após retrain bem-sucedido, os artefactos são guardados no repo GitHub
para persistirem entre deploys do Railway e servirem de auditoria.

Artefactos DipRadar:
  ml_training_base.parquet        — dataset de treino
  ml_training/dip_models.pkl      — bundle do modelo em produção
  ml_training/ml_report.json      — métricas e IC do modelo

Artefactos MomentumRadar:
  momentum_radar/momentum_training.parquet
  momentum_radar/momentum_model.pkl
  momentum_radar/momentum_report.json

Env vars necessárias no Railway:
  GITHUB_TOKEN   — Personal Access Token com permissão repo:write
  GITHUB_REPO    — ex: "romeurf/DipRadar"
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_TOKEN = os.getenv("GITHUB_TOKEN", "")
_REPO  = os.getenv("GITHUB_REPO", "")


def _is_configured() -> bool:
    return bool(_TOKEN and _REPO)


def push_file(
    local_path: Path,
    repo_path: str,
    commit_message: str,
    branch: str = "main",
) -> bool:
    """Faz push de um ficheiro local para o GitHub via API.

    Cria ou actualiza o ficheiro no repo. Retorna True em caso de sucesso.
    Falha silenciosamente (log.warning) se credenciais não estiverem configuradas.
    """
    if not _is_configured():
        log.warning(
            "[github_sync] GITHUB_TOKEN ou GITHUB_REPO não configurados. "
            "Adiciona estas env vars no Railway para activar o sync automático."
        )
        return False

    if not local_path.exists():
        log.warning(f"[github_sync] Ficheiro não encontrado: {local_path}")
        return False

    import requests

    api_url = f"https://api.github.com/repos/{_REPO}/contents/{repo_path}"
    headers = {
        "Authorization": f"token {_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Obter SHA actual (necessário para actualizar ficheiro existente)
    sha: Optional[str] = None
    try:
        r = requests.get(api_url, headers=headers, params={"ref": branch}, timeout=15)
        if r.status_code == 200:
            sha = r.json().get("sha")
    except Exception as e:
        log.debug(f"[github_sync] GET {repo_path}: {e}")

    # Ler e codificar ficheiro
    try:
        content_b64 = base64.b64encode(local_path.read_bytes()).decode()
    except Exception as e:
        log.error(f"[github_sync] Erro ao ler {local_path}: {e}")
        return False

    payload: dict = {
        "message": commit_message,
        "content": content_b64,
        "branch":  branch,
    }
    if sha:
        payload["sha"] = sha

    try:
        r = requests.put(api_url, headers=headers, json=payload, timeout=60)
        if r.status_code in (200, 201):
            size_mb = local_path.stat().st_size / 1e6
            log.info(f"[github_sync] {repo_path} ({size_mb:.1f}MB) → GitHub ✓")
            return True
        else:
            log.warning(f"[github_sync] PUT {repo_path}: {r.status_code} {r.text[:200]}")
            return False
    except Exception as e:
        log.error(f"[github_sync] Push {repo_path} falhou: {e}")
        return False


def sync_training_artifacts(data_dir: Path) -> dict:
    """Sincroniza parquet + bundle + report para o repo GitHub.

    Chamado automaticamente após retrain bem-sucedido.
    """
    if not _is_configured():
        return {"skipped": True, "reason": "GITHUB_TOKEN/GITHUB_REPO não configurados"}

    from datetime import datetime
    ts   = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    results = {}

    _ARTIFACTS: list[tuple[Path, str, str]] = [
        # (local_source, repo_path, label)
        # ── DipRadar ───────────────────────────────────────────────────────
        (data_dir / "ml_training_base.parquet",  "ml_training_base.parquet",         "dip_parquet"),
        (data_dir / "dip_models.pkl",            "ml_training/dip_models.pkl",       "dip_bundle"),
        (data_dir / "ml_report.json",            "ml_training/ml_report.json",       "dip_report"),
        # ── MomentumRadar ──────────────────────────────────────────────────
        (data_dir / "momentum_training.parquet", "momentum_radar/momentum_training.parquet", "mom_parquet"),
        (data_dir / "momentum_model.pkl",        "momentum_radar/momentum_model.pkl",        "mom_bundle"),
        (data_dir / "momentum_report.json",      "momentum_radar/momentum_report.json",      "mom_report"),
    ]

    for local, repo_path, label in _ARTIFACTS:
        if local.exists():
            ok = push_file(local, repo_path, f"artifacts: update {repo_path.split('/')[-1]} [{ts}]")
            results[label] = "ok" if ok else "failed"

    log.info(f"[github_sync] sync_training_artifacts: {results}")
    return results


def sync_momentum_artifacts(data_dir: Path) -> dict:
    """Sincroniza só os artefactos MomentumRadar. Chamado após momentum retrain."""
    if not _is_configured():
        return {"skipped": True, "reason": "GITHUB_TOKEN/GITHUB_REPO não configurados"}

    from datetime import datetime
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    results = {}

    for local, repo_path, label in [
        (data_dir / "momentum_training.parquet", "momentum_radar/momentum_training.parquet", "mom_parquet"),
        (data_dir / "momentum_model.pkl",        "momentum_radar/momentum_model.pkl",        "mom_bundle"),
        (data_dir / "momentum_report.json",      "momentum_radar/momentum_report.json",      "mom_report"),
    ]:
        if local.exists():
            ok = push_file(local, repo_path, f"artifacts: update {repo_path.split('/')[-1]} [{ts}]")
            results[label] = "ok" if ok else "failed"

    log.info(f"[github_sync] sync_momentum_artifacts: {results}")
    return results
