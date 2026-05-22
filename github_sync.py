"""
github_sync.py — Push de artefactos ML para o repositório GitHub.

Elimina a dependência do Colab: o bot trata de tudo autonomamente.
Após retrain bem-sucedido, os artefactos são guardados no repo GitHub
para persistirem entre deploys do Railway e servirem de auditoria.

Artefactos sincronizados:
  ml_training_base.parquet  — dataset de treino completo
  dip_models.pkl             — bundle do modelo em produção
  ml_report.json             — métricas e IC do modelo

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

    # 1. Parquet de treino
    pq = data_dir / "ml_training_base.parquet"
    if pq.exists():
        ok = push_file(pq, "ml_training_base.parquet",
                       f"data: update ml_training_base.parquet [{ts}]")
        results["parquet"] = "ok" if ok else "failed"

    # 2. Bundle do modelo
    pkl = data_dir / "dip_models.pkl"
    if pkl.exists():
        ok = push_file(pkl, "dip_models.pkl",
                       f"model: update dip_models.pkl [{ts}]")
        results["bundle"] = "ok" if ok else "failed"

    # 3. Report de métricas
    report = data_dir / "ml_report.json"
    if report.exists():
        ok = push_file(report, "ml_report.json",
                       f"model: update ml_report.json [{ts}]")
        results["report"] = "ok" if ok else "failed"

    log.info(f"[github_sync] sync_training_artifacts: {results}")
    return results
