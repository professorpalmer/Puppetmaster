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
