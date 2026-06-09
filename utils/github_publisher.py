"""
github_publisher.py — Pushes chart PNGs to a GitHub repository via the Contents API.

Required env vars:
  GITHUB_TOKEN        — personal access token with repo write scope
  GITHUB_CHARTS_REPO  — "owner/repo" (e.g. "romeurf/dipr-public-charts")

Optional env vars:
  GITHUB_CHARTS_BRANCH — branch to push to (default: "main")
  GITHUB_CHARTS_PATH   — remote directory path (default: "public/charts")
"""
from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Optional
from urllib import request, error as url_error

logger = logging.getLogger(__name__)

_API_BASE = "https://api.github.com"


def _github_request(
    method: str,
    url: str,
    token: str,
    payload: Optional[dict] = None,
) -> dict:
    """Send a GitHub API request and return the parsed JSON response."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }
    data = json.dumps(payload).encode() if payload is not None else None
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except url_error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(
            f"GitHub API {method} {url} → HTTP {exc.code}: {body}"
        ) from exc


def _get_file_sha(url: str, token: str) -> Optional[str]:
    """Return SHA of existing file, or None if it doesn't exist."""
    try:
        resp = _github_request("GET", url, token)
        return resp.get("sha")
    except RuntimeError as exc:
        if "HTTP 404" in str(exc):
            return None
        raise


def push_file_to_github(
    local_path: Path,
    remote_path: str,
    token: str,
    repo: str,
    branch: str = "main",
    commit_message: Optional[str] = None,
) -> str:
    """
    Push a single file to GitHub via the Contents API.

    Returns the raw GitHub URL of the pushed file.
    """
    content_b64 = base64.b64encode(local_path.read_bytes()).decode()
    api_url = f"{_API_BASE}/repos/{repo}/contents/{remote_path}"

    sha = _get_file_sha(api_url, token)

    message = commit_message or f"Update {Path(remote_path).name} [DipRadar auto]"
    payload: dict = {
        "message": message,
        "content": content_b64,
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    method = "PUT"
    resp = _github_request(method, api_url, token, payload)

    # For Markdown files, return the github.com viewer URL (renders the report)
    # For PNGs and other binaries, return the raw URL (for direct embedding)
    if local_path.suffix.lower() == ".md":
        public_url = f"https://github.com/{repo}/blob/{branch}/{remote_path}"
    else:
        public_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{remote_path}"

    logger.info("Pushed %s → %s", local_path.name, public_url)
    return public_url


def push_charts_to_github(
    chart_paths: list[Path],
    *,
    token: Optional[str] = None,
    repo: Optional[str] = None,
    branch: Optional[str] = None,
    remote_dir: Optional[str] = None,
    commit_message: Optional[str] = None,
) -> dict[str, str | None]:
    """
    Push a list of PNG charts to GitHub. Returns a dict mapping filename → URL (or error string).

    Reads configuration from env vars if not passed explicitly:
      GITHUB_TOKEN, GITHUB_CHARTS_REPO, GITHUB_CHARTS_BRANCH, GITHUB_CHARTS_PATH
    """
    _token      = token      or os.environ.get("GITHUB_TOKEN", "")
    _repo       = repo       or os.environ.get("GITHUB_CHARTS_REPO", "")
    _branch     = branch     or os.environ.get("GITHUB_CHARTS_BRANCH", "main")
    _remote_dir = remote_dir or os.environ.get("GITHUB_CHARTS_PATH", "public/charts")

    if not _token:
        raise ValueError("GITHUB_TOKEN is not set — cannot push charts.")
    if not _repo:
        raise ValueError("GITHUB_CHARTS_REPO is not set — cannot push charts.")

    results: dict[str, str | None] = {}
    for path in chart_paths:
        if not path.exists():
            logger.warning("Chart not found, skipping: %s", path)
            results[path.name] = None
            continue
        remote_path = f"{_remote_dir.rstrip('/')}/{path.name}"
        try:
            url = push_file_to_github(
                path,
                remote_path,
                _token,
                _repo,
                branch=_branch,
                commit_message=commit_message,
            )
            results[path.name] = url
        except Exception as exc:
            logger.error("Failed to push %s: %s", path.name, exc)
            results[path.name] = f"ERROR: {exc}"

    return results


def push_all_charts(chart_dir: Optional[Path] = None) -> dict[str, str | None]:
    """Push all PNG charts and Markdown reports from chart_dir to GitHub.

    Picks up *.png and *.md files from the charts directory.
    The DRIP ledger (drip_ledger.md) is treated with the same priority as charts —
    both are public transparency artifacts.

    Returns mapping of filename → GitHub raw URL or error string.
    """
    _dir = chart_dir or Path(os.environ.get("CHART_DIR", "/data/charts"))
    if not _dir.exists():
        logger.warning("Chart directory does not exist: %s", _dir)
        return {}

    files = sorted(_dir.glob("*.png")) + sorted(_dir.glob("*.md"))
    if not files:
        logger.warning("No publishable files (*.png, *.md) found in %s", _dir)
        return {}

    logger.info("Publishing %d files to GitHub (%d charts, %d reports)…",
                len(files),
                sum(1 for f in files if f.suffix == ".png"),
                sum(1 for f in files if f.suffix == ".md"))
    return push_charts_to_github(files)
