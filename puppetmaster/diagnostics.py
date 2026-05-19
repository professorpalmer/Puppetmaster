from __future__ import annotations

import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from puppetmaster.adapters import ADAPTER_INFO, AdapterInfo
from puppetmaster.codegraph import (
    codegraph_available,
    codegraph_initialized,
    codegraph_native_sqlite_broken,
    codegraph_status_command,
)
from puppetmaster.state import resolve_state_dir


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def run_doctor(root: Path, state_dir: Optional[Path] = None) -> list[Check]:
    state_path = state_dir or resolve_state_dir(cwd=root)
    checks = [
        Check("python", "ok", sys.version.split()[0]),
        Check("sqlite", "ok", sqlite3.sqlite_version),
        _command_check("git", ["git", "--version"]),
        _command_check("node", ["node", "--version"]),
        _command_check("npm", ["npm", "--version"]),
        _cursor_sdk_check(root),
        _claude_code_check(),
        _codegraph_check(root),
        _env_check("CURSOR_API_KEY"),
        _sqlite_state_check(state_path / "state.sqlite3"),
        _git_clean_check(root),
    ]
    return checks


def _codegraph_check(root: Path) -> Check:
    if not codegraph_available():
        return Check(
            "codegraph",
            "optional",
            "install codegraph for shared repo intelligence (npx @colbymchenry/codegraph)",
        )
    if not codegraph_initialized(root):
        return Check(
            "codegraph",
            "optional",
            "codegraph installed; run `codegraph init` in target repos to enable shared context",
        )
    status = codegraph_status_command(root)
    combined = (status.get("stdout") or "") + "\n" + (status.get("stderr") or "")
    if codegraph_native_sqlite_broken(combined):
        return Check(
            "codegraph",
            "warn",
            "native better-sqlite3 broken; codegraph is using the slow WASM fallback. "
            "Rebuild with `cd \"$(npm root -g)/@colbymchenry/codegraph\" && "
            "npm rebuild better-sqlite3`, then re-run `codegraph status`.",
        )
    return Check("codegraph", "ok", "codegraph installed and target workspace initialized")


def adapter_status(root: Path) -> list[dict[str, object]]:
    cursor_installed = _cursor_sdk_installed(root)
    cursor_key = bool(os.environ.get("CURSOR_API_KEY"))
    claude_installed = _claude_code_installed()
    rows = []
    for info in ADAPTER_INFO:
        configured = info.status == "built-in"
        if info.name == "cursor":
            configured = cursor_installed and cursor_key
        if info.name == "claude-code":
            configured = claude_installed
        if info.status == "stub":
            configured = False
        rows.append(
            {
                "name": info.name,
                "status": info.status,
                "configured": configured,
                "description": info.description,
                "requires": info.requires,
            }
        )
    return rows


def starter_config() -> str:
    return """{
  "lease_seconds": 5,
  "workers": [
    {
      "role": "explore",
      "instruction": "Map the goal and emit evidenced findings."
    },
    {
      "role": "verify-runtime",
      "instruction": "Verify Python is available before deeper work.",
      "adapter": "shell",
      "depends_on": ["explore"],
      "payload": {
        "command": ["python", "--version"],
        "timeout_seconds": 10
      }
    },
    {
      "role": "architect",
      "instruction": "Choose the smallest useful architecture and record decisions.",
      "depends_on": ["verify-runtime"]
    }
  ]
}
"""


def _command_check(name: str, command: list[str]) -> Check:
    if shutil.which(command[0]) is None:
        return Check(name, "missing", f"{command[0]} not found on PATH")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    output = (completed.stdout or completed.stderr).strip()
    status = "ok" if completed.returncode == 0 else "warn"
    return Check(name, status, output or f"exit code {completed.returncode}")


def _cursor_sdk_check(root: Path) -> Check:
    if _cursor_sdk_installed(root):
        return Check("cursor-sdk", "ok", "@cursor/sdk installed")
    return Check("cursor-sdk", "optional", "run npm install to enable the cursor adapter")


def _cursor_sdk_installed(root: Path) -> bool:
    return (root / "node_modules" / "@cursor" / "sdk").exists()


def _claude_code_check() -> Check:
    if _claude_code_installed():
        return Check("claude-code", "ok", _claude_code_command())
    return Check(
        "claude-code",
        "missing",
        "install Claude Code or set CLAUDE_CODE_COMMAND to the CLI executable",
    )


def _claude_code_installed() -> bool:
    command = shlex.split(_claude_code_command())
    if not command:
        return False
    first = command[0]
    if Path(first).expanduser().exists():
        return True
    return shutil.which(first) is not None


def _claude_code_command() -> str:
    return os.environ.get("CLAUDE_CODE_COMMAND", "claude")


def _env_check(name: str) -> Check:
    if os.environ.get(name):
        return Check(name, "ok", "set")
    return Check(name, "optional", "not set")


def _sqlite_state_check(path: Path) -> Check:
    if not path.exists():
        return Check("sqlite-state", "optional", "no local sqlite state yet")
    try:
        with sqlite3.connect(path) as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            journal = connection.execute("PRAGMA journal_mode").fetchone()
    except sqlite3.Error as exc:
        return Check("sqlite-state", "warn", str(exc))
    version = row[0] if row else "unknown"
    integrity_status = integrity[0] if integrity else "unknown"
    journal_mode = journal[0] if journal else "unknown"
    status = "ok" if integrity_status == "ok" and journal_mode == "wal" else "warn"
    return Check(
        "sqlite-state",
        status,
        f"schema={version}; journal={journal_mode}; integrity={integrity_status}",
    )


def _git_clean_check(root: Path) -> Check:
    if shutil.which("git") is None:
        return Check("git-status", "optional", "git not available")
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return Check("git-status", "optional", "not a git repository")
    detail = completed.stdout.strip()
    return Check("git-status", "ok" if not detail else "warn", detail or "clean")

