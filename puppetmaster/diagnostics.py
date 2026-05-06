from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from puppetmaster.adapters import ADAPTER_INFO, AdapterInfo


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def run_doctor(root: Path) -> list[Check]:
    checks = [
        Check("python", "ok", sys.version.split()[0]),
        Check("sqlite", "ok", sqlite3.sqlite_version),
        _command_check("git", ["git", "--version"]),
        _command_check("node", ["node", "--version"]),
        _command_check("npm", ["npm", "--version"]),
        _cursor_sdk_check(root),
        _env_check("CURSOR_API_KEY"),
        _git_clean_check(root),
    ]
    return checks


def adapter_status(root: Path) -> list[dict[str, object]]:
    cursor_installed = _cursor_sdk_installed(root)
    cursor_key = bool(os.environ.get("CURSOR_API_KEY"))
    rows = []
    for info in ADAPTER_INFO:
        configured = info.status == "built-in"
        if info.name == "cursor":
            configured = cursor_installed and cursor_key
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


def _env_check(name: str) -> Check:
    if os.environ.get(name):
        return Check(name, "ok", "set")
    return Check(name, "optional", "not set")


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

