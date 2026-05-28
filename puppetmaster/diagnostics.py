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
        _codex_check(),
        _codegraph_check(root),
        _mcp_servers_check(),
        _env_check("CURSOR_API_KEY"),
        _env_check("OPENAI_API_KEY"),
        _sqlite_state_check(state_path / "state.sqlite3"),
        _git_clean_check(root),
        _agent_rules_check(root),
    ]
    checks.extend(_billing_checks())
    return checks


def _billing_checks() -> list[Check]:
    """Report each adapter's billing posture: plan (in-subscription) vs api
    (out-of-pocket) vs unknown/unauthenticated. This is the at-a-glance answer
    to "will this cost me extra, and can it even run?"."""
    from puppetmaster.platform_billing import detect_adapter_billing

    checks: list[Check] = []
    for adapter in ("cursor", "claude-code", "codex"):
        try:
            status = detect_adapter_billing(adapter)
        except Exception as exc:  # pragma: no cover - defensive
            checks.append(Check(f"billing:{adapter}", "warn", f"probe failed: {exc}"))
            continue
        state = "ok" if status.healthy else "warn"
        checks.append(
            Check(f"billing:{adapter}", state, f"{status.billing} — {status.detail}")
        )
    return checks


def _agent_rules_check(root: Path) -> Check:
    """Warn when MCP is wired but no agent rule file is present.

    The MCP installers give Cursor / Codex / Claude Code the *capability*
    to call Puppetmaster, but a host agent won't reflexively reach for
    those tools without a workspace rule nudging it. This check catches
    the common half-installed state where `install-cursor-mcp` or
    `install-codex-mcp` was run but `install-rules` was not.

    Returns ``optional`` rather than ``warn`` if no MCP integration is
    detected either (no MCP = no point in rules), and ``ok`` once any
    rule file is present at one of the canonical locations.
    """
    candidate_paths = [
        root / ".cursor" / "rules" / "puppetmaster.mdc",
        root / "AGENTS.md",
        root / "CLAUDE.md",
        Path.home() / ".codex" / "instructions.md",
        Path.home() / ".claude" / "CLAUDE.md",
    ]
    rule_present_paths: list[Path] = []
    for path in candidate_paths:
        try:
            if not path.is_file():
                continue
            if path.name == "puppetmaster.mdc":
                rule_present_paths.append(path)
                continue
            text = path.read_text(encoding="utf-8")
            if "puppetmaster:rules:begin" in text or "puppetmaster_route_task" in text:
                rule_present_paths.append(path)
        except OSError:
            continue
    if rule_present_paths:
        rel = ", ".join(
            str(p.relative_to(Path.home())) if str(p).startswith(str(Path.home())) else str(p)
            for p in rule_present_paths[:2]
        )
        return Check("agent-rules", "ok", f"agent rule present ({rel})")
    cursor_mcp = (root / ".cursor" / "mcp.json").is_file() or (Path.home() / ".cursor" / "mcp.json").is_file()
    if not cursor_mcp:
        return Check(
            "agent-rules",
            "optional",
            "no agent rule files detected (run `puppetmaster install-rules` after registering an MCP host)",
        )
    return Check(
        "agent-rules",
        "warn",
        (
            "Puppetmaster MCP is registered but no agent rule file was found — "
            "the host agent will not reflexively reach for Puppetmaster on multi-file tasks. "
            "Fix: `puppetmaster install-rules` (workspace) or `puppetmaster install-rules --global` (user-level)."
        ),
    )


def _codex_check() -> Check:
    if _codex_cli_installed():
        return Check("codex", "ok", _codex_command())
    return Check(
        "codex",
        "optional",
        (
            "install the OpenAI Codex CLI with `npm install -g @openai/codex` "
            "then `printenv OPENAI_API_KEY | codex login --with-api-key`, or "
            "set CODEX_COMMAND to its path. Required only if you want to "
            "route to codex/* tiers."
        ),
    )


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
    codex_installed = _codex_cli_installed()
    openai_key = bool(os.environ.get("OPENAI_API_KEY"))
    rows = []
    for info in ADAPTER_INFO:
        configured = info.status == "built-in"
        if info.name == "cursor":
            configured = cursor_installed and cursor_key
        elif info.name == "claude-code":
            configured = claude_installed
        elif info.name == "openai":
            configured = openai_key
        elif info.name == "codex":
            # Codex needs BOTH: the CLI installed AND OpenAI auth. We don't
            # introspect `codex login` state from here (the auth file is
            # outside our purview), so OPENAI_API_KEY is a decent proxy
            # that we don't false-positive on a never-authed Codex install.
            configured = codex_installed and openai_key
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
        "optional",
        (
            "install Claude Code (`npm install -g @anthropic-ai/claude-code` or "
            "`npx -y @anthropic-ai/claude-code`) or set CLAUDE_CODE_COMMAND. "
            "Required only if you want to route to claude-code/* tiers."
        ),
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


def _codex_cli_installed() -> bool:
    command = shlex.split(_codex_command())
    if not command:
        return False
    first = command[0]
    if Path(first).expanduser().exists():
        return True
    return shutil.which(first) is not None


def _codex_command() -> str:
    return os.environ.get("CODEX_COMMAND", "codex")


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

