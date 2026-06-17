from __future__ import annotations

import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from puppetmaster.adapters import ADAPTER_INFO
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
    evidence: list[str] = field(default_factory=list)


def _guard(name: str, fn: "Callable[[], Check]") -> Check:
    """Run a check, converting any exception into an ``error`` Check.

    A probe that shells out to a CLI can raise on some platforms (e.g. a
    Windows node-shim that isn't a valid executable raises ``OSError`` /
    ``FileNotFoundError``). Doctor must always return a full report rather
    than crashing because one optional probe blew up.
    """
    try:
        return fn()
    except Exception as exc:  # never let one probe abort the whole report
        return Check(name, "error", f"check raised: {type(exc).__name__}: {exc}")


def _guard_many(fn: "Callable[[], list[Check]]") -> list[Check]:
    try:
        return list(fn())
    except Exception as exc:
        return [Check("billing", "error", f"billing checks raised: {type(exc).__name__}: {exc}")]


def run_doctor(root: Path, state_dir: Optional[Path] = None) -> list[Check]:
    state_path = state_dir or resolve_state_dir(cwd=root)
    checks = [
        Check("python", "ok", sys.version.split()[0]),
        Check("sqlite", "ok", sqlite3.sqlite_version),
        _guard("git", lambda: _command_check("git", ["git", "--version"])),
        _guard("node", lambda: _command_check("node", ["node", "--version"])),
        _guard("npm", lambda: _command_check("npm", ["npm", "--version"])),
        _guard("cursor-sdk", lambda: _cursor_sdk_check(root)),
        _guard("claude-code", _claude_code_check),
        _guard("codex", _codex_check),
        _guard("codegraph", lambda: _codegraph_check(root)),
        _guard("mcp-servers", _mcp_servers_check),
        _guard("CURSOR_API_KEY", lambda: _env_check("CURSOR_API_KEY")),
        _guard("OPENAI_API_KEY", lambda: _env_check("OPENAI_API_KEY")),
        _guard("sqlite-state", lambda: _sqlite_state_check(state_path / "state.sqlite3")),
        _guard("git-status", lambda: _git_clean_check(root)),
        _guard("agent-rules", lambda: _agent_rules_check(root)),
    ]
    checks.extend(_guard_many(_credential_env_checks))
    checks.extend(_guard_many(_billing_checks))
    checks.append(_guard("catalog-freshness", _catalog_freshness_check))
    checks.append(_guard("platform-lock", _platform_lock_check))
    return checks


def _platform_lock_check() -> Check:
    """Report whether a platform lock is narrowing the adapter set.

    When active, only the listed platforms can be routed to, auto-discovered,
    or used for fallback — a disabled platform can never run, even if its CLI
    is installed and funded. Off by default (every platform enabled)."""
    from puppetmaster import platform_lock as pl

    enabled = pl.enabled_adapters()
    if not pl.is_restricted():
        return Check("platform-lock", "ok", "off — all platforms enabled")
    disabled = sorted(set(pl.KNOWN_ADAPTERS) - enabled)
    detail = (
        f"active — only {', '.join(sorted(enabled)) or '(none)'} "
        f"(disabled: {', '.join(disabled) or 'none'})"
    )
    if (os.environ.get(pl.ONLY_ENV) or "").strip():
        detail += f"; via ${pl.ONLY_ENV}"
    return Check("platform-lock", "ok", detail)


def _catalog_freshness_check() -> Check:
    """Nudge when a discovered model catalog is stale (or never refreshed).

    Model catalogs drift — platforms add/retire models. ``models discover``
    records when each source was last enumerated; this surfaces a gentle
    reminder so routing doesn't quietly run against an out-of-date view."""
    from puppetmaster.model_registry import (
        catalog_staleness_days,
        read_discovery_meta,
    )

    meta = read_discovery_meta()
    if not meta:
        return Check(
            "catalog-freshness",
            "optional",
            "no catalog discovery recorded yet — run `puppetmaster models discover --write` "
            "to enumerate plan-billed models and keep routing current.",
        )
    stale_threshold = 30.0
    stale: list[str] = []
    fresh: list[str] = []
    for source in meta:
        age = catalog_staleness_days(meta, source)
        if age is None:
            continue
        label = f"{source} {age:.0f}d"
        (stale if age > stale_threshold else fresh).append(label)
    if stale:
        return Check(
            "catalog-freshness",
            "warn",
            f"catalog stale (>{stale_threshold:.0f}d): {', '.join(stale)}. "
            "Re-run `puppetmaster models discover --write` to refresh.",
        )
    return Check(
        "catalog-freshness",
        "ok",
        f"catalog fresh ({', '.join(fresh) or 'recently discovered'})",
    )


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
            Check(
                f"billing:{adapter}",
                state,
                f"{status.billing} — {status.detail}",
                evidence=list(status.evidence),
            )
        )
    return checks


_PROVIDER_CREDENTIAL_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "cursor": ("CURSOR_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "codex": ("CODEX_HOME", "OPENAI_API_KEY"),
    "claude-code": (
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_USE_BEDROCK",
        "AWS_PROFILE",
        "AWS_BEARER_TOKEN_BEDROCK",
    ),
}


def _credential_env_checks() -> list[Check]:
    checks: list[Check] = []
    seen: set[tuple[str, str]] = set()
    for provider, keys in _PROVIDER_CREDENTIAL_ENV_KEYS.items():
        for key in keys:
            marker = (provider, key)
            if marker in seen:
                continue
            seen.add(marker)
            status = "ok" if os.environ.get(key) else "optional"
            detail = (
                f"{key} visible to this process (value hidden)"
                if status == "ok"
                else f"{key} not visible to this process"
            )
            checks.append(
                Check(
                    f"credential-env:{provider}:{key}",
                    status,
                    detail,
                    evidence=[f"provider:{provider}", f"env:{key}", f"visible:{status == 'ok'}"],
                )
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
            "codegraph not found — it powers the shared repo intelligence injected "
            "into every worker (CodeGraph search/context). It is a Node/npm package: "
            "`npm install -g @colbymchenry/codegraph` (or run via `npx "
            "@colbymchenry/codegraph`). Note: do NOT `pip install codegraph` — the "
            "PyPI package of that name is unrelated and will not work.",
        )
    if not codegraph_initialized(root):
        return Check(
            "codegraph",
            "optional",
            "codegraph installed but this repo isn't indexed yet — run "
            "`python -m puppetmaster codegraph init --index` here to enable shared "
            "context (always invoke via `python -m puppetmaster codegraph …`, never a "
            "bare `codegraph`, so it runs under Cursor's bundled Node).",
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
    try:
        from puppetmaster.platform_billing import detect_codex_billing

        codex_auth = detect_codex_billing()
    except Exception:
        codex_auth = None
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
            # Availability is intentionally separate from billing context.
            # The CLI plus *any* credential signal — an OPENAI_API_KEY in the
            # environment, or a healthy Codex auth context ($CODEX_HOME/auth.json
            # or `codex login`) — marks Codex usable. Which account work bills to
            # is reported on its own by the `billing:codex` doctor check, so an
            # OPENAI_API_KEY-only setup is never silently demoted to "unconfigured".
            configured = codex_installed and (
                openai_key or bool(codex_auth and codex_auth.healthy)
            )
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


def _resolve_probe_command(command: list[str]) -> Optional[list[str]]:
    """Resolve ``command`` to a form ``subprocess`` can actually launch.

    ``shutil.which`` honors ``PATHEXT``, so on Windows it locates the
    ``npm.cmd`` / ``npx.cmd`` shims that a bare ``npm`` would miss. But
    ``subprocess`` with ``shell=False`` cannot launch a batch shim directly
    (``CreateProcess`` only runs PE binaries), so probing bare ``npm`` there
    raised ``FileNotFoundError`` (WinError 2) even though npm was installed —
    surfacing a misleading ``error`` row in ``doctor``. Route a resolved
    ``.cmd`` / ``.bat`` through the command processor instead.

    Returns ``None`` when the executable isn't on PATH.
    """
    resolved = shutil.which(command[0])
    if resolved is None:
        return None
    if os.name == "nt" and resolved.lower().endswith((".cmd", ".bat")):
        comspec = os.environ.get("COMSPEC", "cmd.exe")
        return [comspec, "/c", resolved, *command[1:]]
    return [resolved, *command[1:]]


def _command_check(name: str, command: list[str]) -> Check:
    resolved = _resolve_probe_command(command)
    if resolved is None:
        return Check(name, "missing", f"{command[0]} not found on PATH")
    # stdin=DEVNULL is critical when this runs inside the MCP server: by
    # default subprocess inherits fd 0 from the parent, and certain
    # children (or just the kernel under fd pressure from many parallel
    # spawns) can cause the parent's stdin reader to receive a phantom
    # EOF — silently exiting the server with code 0. See
    # bench/mcp_stress.py for the repro.
    completed = subprocess.run(
        resolved,
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
        "run `puppetmaster install-cursor-mcp` to bootstrap @cursor/sdk (needs Node/npm)",
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
    # cursor_sdk_runner.mjs resolves @cursor/sdk with Node's resolution,
    # which walks node_modules upward from the runner's own directory.
    # Mirror the full walk so diagnostics agree with runtime: a probe
    # pinned to one fixed level (the old `parent.parent`) missed valid
    # installs at e.g. site-packages/puppetmaster/node_modules — Node's
    # *first* hop — and reported "SDK not found" on working machines.
    package_dir = Path(__file__).resolve().parent
    for ancestor in [package_dir, *package_dir.parents]:
        candidates.append(ancestor / "node_modules" / "@cursor" / "sdk")
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
        return Check(name, "ok", "set", evidence=[f"env:{name}", "visible:true"])
    return Check(name, "optional", "not set", evidence=[f"env:{name}", "visible:false"])


def _mcp_env_check(name: str) -> Check:
    if os.environ.get(name):
        return Check(f"mcp-env:{name}", "ok", "visible to this process (value hidden)")
    return Check(f"mcp-env:{name}", "optional", "not visible to this process")


def _sqlite_state_check(path: Path) -> Check:
    if not path.exists():
        return Check("sqlite-state", "optional", "no local sqlite state yet")
    try:
        # ``with sqlite3.connect(...)`` only commits — it does not close the
        # handle. On Windows a lingering handle locks the file and breaks a
        # later unlink (WinError 32), so close it explicitly.
        connection = sqlite3.connect(path)
        try:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            journal = connection.execute("PRAGMA journal_mode").fetchone()
        finally:
            connection.close()
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
