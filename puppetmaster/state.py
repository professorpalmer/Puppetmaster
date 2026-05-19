from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union


STATE_DIR_ENV = "PUPPETMASTER_STATE_DIR"


def resolve_state_dir(value: Optional[Union[Path, str]] = None, cwd: Optional[Path] = None) -> Path:
    """Resolve Puppetmaster state without dirtying the target repository by default."""
    root = cwd or Path.cwd()
    if value:
        return _resolve_user_path(value, root)
    env_value = os.environ.get(STATE_DIR_ENV)
    if env_value:
        return _resolve_user_path(env_value, root)
    return default_state_dir(root)


def default_state_dir(cwd: Optional[Path] = None) -> Path:
    workspace = _git_root(cwd or Path.cwd()) or (cwd or Path.cwd())
    resolved = workspace.resolve()
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", resolved.name).strip("-") or "workspace"
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    return app_state_root() / "projects" / f"{slug}-{digest}"


def app_state_root() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "puppetmaster"
    if os.name == "nt":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "puppetmaster"
    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home) / "puppetmaster"
    return Path.home() / ".local" / "state" / "puppetmaster"


def list_project_state_dirs() -> list[Path]:
    """Return every project-scoped state directory currently on disk.

    The MCP server and CLI both compute a per-workspace state dir hashed
    from the resolved workspace path, so `puppetmaster show <job_id>`
    only finds jobs created from the same workspace by default. This
    helper lets callers iterate every known project to support cross-
    workspace job lookup without forcing users to memorize the hash.
    """
    projects_root = app_state_root() / "projects"
    if not projects_root.is_dir():
        return []
    return sorted(p for p in projects_root.iterdir() if p.is_dir())


def find_state_dir_for_job(job_id: str) -> Optional[Path]:
    """Locate the project state dir that owns ``job_id``, if any.

    Scans every project state dir and returns the first one whose
    ``jobs/<job_id>`` directory exists. The SQLite store stores jobs
    on disk under ``<state_dir>/jobs/<job_id>/`` (we don't need to
    actually open the DB to detect ownership — a directory check is
    enough and avoids spinning up the WAL writer just for a search).

    Returns None when no project knows about the job — callers should
    surface the user-facing error in that case rather than swallowing.
    """
    if not job_id:
        return None
    for project in list_project_state_dirs():
        job_dir = project / "jobs" / job_id
        if job_dir.is_dir():
            return project
    return None


def _resolve_user_path(value: Union[Path, str], cwd: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else cwd / path


def _git_root(cwd: Path) -> Optional[Path]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    return Path(output) if output else None
