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
    resolve_codegraph_invocation,
)
from puppetmaster.mcp_registry import list_entries as registry_list_entries
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
        _mcp_servers_check(),
        _env_check("CURSOR_API_KEY"),
        _sqlite_state_check(state_path / "state.sqlite3"),
        _git_clean_check(root),
    ]
    return checks


def _mcp_servers_check() -> Check:
    """Flag dead-but-tracked or stale-but-alive Puppetmaster MCP servers.

    Either condition is the smoking gun behind a stale `Tool execution
    error. Not connected` symptom: dead entries mean a prior server
    crashed and left state behind; stale-but-alive entries mean a
    Cursor parent went away but the MCP child is still consuming
    resources.
    """
    try:
        entries = registry_list_entries()
    except Exception as exc:
        return Check("mcp-servers", "warn", f"registry unreadable: {exc}")
    alive = [entry for entry in entries if entry.is_alive()]
    dead = [entry for entry in entries if not entry.is_alive()]
    stale_alive = [entry for entry in alive if entry.is_stale()]
    if dead and stale_alive:
        return Check(
            "mcp-servers",
            "warn",
            (
                f"{len(dead)} dead tracking file(s) and {len(stale_alive)} stale-but-alive "
                "server(s) detected. Run `python -m puppetmaster mcp cleanup --kill-stale` "
                "to clean up; this is the common root cause of `Tool execution error. "
                "Not connected` after a Cursor MCP restart."
            ),
        )
    if dead:
        return Check(
            "mcp-servers",
            "warn",
            (
                f"{len(dead)} dead tracking file(s) from prior MCP server crashes. "
                "Run `python -m puppetmaster mcp cleanup` to reclaim them."
            ),
        )
    if stale_alive:
        return Check(
            "mcp-servers",
            "warn",
            (
                f"{len(stale_alive)} Puppetmaster MCP server(s) alive but stale "
                "(no heartbeat in >5min). Cursor parent likely gone. Run "
                "`python -m puppetmaster mcp cleanup --kill-stale` to terminate."
            ),
        )
    if alive:
        return Check("mcp-servers", "ok", f"{len(alive)} healthy server(s) tracked")
    return Check("mcp-servers", "ok", "no MCP servers currently tracked")


def _codegraph_check(root: Path) -> Check:
    """Verify codegraph is healthy from the runtime Puppetmaster MCP uses.

    Pre-v0.5.4 this called ``codegraph status`` via the shim on PATH,
    which on macOS-with-Homebrew machines is invoked under Homebrew's
    Node — a *different* runtime from the one Puppetmaster's MCP server
    actually runs ``codegraph`` under after v0.5.4. The shell-side
    backend could report WASM (because better-sqlite3 was built for
    Cursor's Node) while MCP's backend was happily native, producing a
    misleading ``warn``.

    We now verify against the same invocation Puppetmaster uses at
    runtime: :func:`resolve_codegraph_invocation` returns Cursor's Node
    + ``codegraph.js`` when available. That is the *real* signal that
    matters for MCP operation.
    """
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
            "native better-sqlite3 broken under the runtime Puppetmaster MCP uses; "
            "codegraph is falling back to slow WASM SQLite. "
            "Fix with `python -m puppetmaster repair-codegraph` (rebuilds against "
            "Cursor's bundled Node so MCP picks it up). Common cause: shell Node "
            "ABI differs from Cursor Node ABI.",
        )
    invocation = resolve_codegraph_invocation()
    detail = "codegraph installed and target workspace initialized"
    if len(invocation) >= 2 and "Cursor.app" in invocation[0]:
        detail += " (verified under Cursor's bundled Node)"
    return Check("codegraph", "ok", detail)


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
    # stdin=DEVNULL is critical when this runs inside the MCP server: by
    # default subprocess inherits fd 0 from the parent, and certain
    # children (or just the kernel under fd pressure from many parallel
    # spawns) can cause the parent's stdin reader to receive a phantom
    # EOF — silently exiting the server with code 0. See
    # bench/mcp_stress.py for the repro.
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout or completed.stderr).strip()
    status = "ok" if completed.returncode == 0 else "warn"
    return Check(name, status, output or f"exit code {completed.returncode}")


def _cursor_sdk_check(root: Path) -> Check:
    location = _find_cursor_sdk_install(root)
    if location is not None:
        return Check("cursor-sdk", "ok", f"@cursor/sdk installed ({location})")
    return Check(
        "cursor-sdk",
        "optional",
        "run `npm install` in the Puppetmaster package dir to enable the cursor adapter",
    )


def _cursor_sdk_installed(root: Path) -> bool:
    """Whether the @cursor/sdk package is resolvable for Puppetmaster's runtime.

    This intentionally checks BOTH the user's workspace ``root/node_modules``
    AND the Puppetmaster package install dir, because the SDK is bundled
    with the Puppetmaster package itself (`cursor_sdk_runner.mjs` resolves
    `@cursor/sdk` from there at runtime) — not from whatever repo the
    user happens to be cd'd in. Before this fix, ``puppetmaster doctor``
    and ``puppetmaster adapters`` would falsely report
    ``cursor: configured=false`` from any non-Puppetmaster workspace.
    """
    return _find_cursor_sdk_install(root) is not None


def _find_cursor_sdk_install(root: Path) -> Optional[Path]:
    """Return the on-disk location of @cursor/sdk, or None if not found."""
    candidates: list[Path] = []
    if root is not None:
        candidates.append(Path(root) / "node_modules" / "@cursor" / "sdk")
    # The Puppetmaster package directory ships its own package.json and
    # node_modules; cursor_sdk_runner.mjs uses Node's resolution which
    # walks upward from its own file path. We mirror that here so
    # diagnostics agree with runtime behavior.
    package_root = Path(__file__).resolve().parent.parent
    candidates.append(package_root / "node_modules" / "@cursor" / "sdk")
    # An editable install ($PUPPETMASTER_HOME / install dir) may live
    # somewhere else entirely; honor an explicit override so users on
    # weird layouts can self-correct without code changes.
    env_root = os.environ.get("PUPPETMASTER_HOME")
    if env_root:
        candidates.append(Path(env_root).expanduser() / "node_modules" / "@cursor" / "sdk")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


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
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return Check("git-status", "optional", "not a git repository")
    detail = completed.stdout.strip()
    return Check("git-status", "ok" if not detail else "warn", detail or "clean")

