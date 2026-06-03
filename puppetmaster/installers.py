"""One-shot installers that wire Puppetmaster's MCP server into Cursor and
the OpenAI Codex CLI.

Both installers solve the same friction point: when a user types
"register Puppetmaster as an MCP server", the registration command needs
the exact Python that has Puppetmaster importable — not whatever
``python`` happens to be on PATH when the host application (Cursor or
Codex) launches the subprocess. We always use :data:`sys.executable` so
the registered command is unambiguous.

A short handshake test (spawn the MCP server, send ``tools/list``, count
the tools returned) verifies the registration actually works before the
installer reports success. That catches:

- Pyenv shim / venv mismatches (registered Python can't import puppetmaster)
- Missing required Python deps
- A broken MCP server module

Both installers are idempotent and preserve existing config: the Cursor
installer merges into ``mcpServers`` without touching other servers and
keeps any env vars already set on the puppetmaster entry; the Codex
installer no-ops cleanly when the entry already matches what we would
write.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class HandshakeResult:
    """Result of spawning the MCP server and asking it to list its tools.

    A successful handshake (``ok=True``) means the registered command
    can actually launch the MCP server end-to-end. ``tool_count`` is
    the number of tools the server reports; ``error`` is populated
    when the handshake failed for any reason (timeout, non-JSON output,
    JSON-RPC error response).
    """

    ok: bool
    tool_count: int = 0
    error: str = ""


@dataclass
class InstallResult:
    """Structured outcome of one install run.

    ``status`` is one of ``"installed"`` (registration created or
    updated), ``"unchanged"`` (entry already matches what we would
    write — idempotent no-op), ``"would_install"`` (dry-run preview),
    or ``"error"`` (something blocked the install).

    ``target`` is the absolute path being touched (``~/.codex/config.toml``
    or whichever ``.cursor/mcp.json``). ``messages`` accumulates the
    human-readable lines we'd print to the user — populating this dict
    lets tests assert on the install's narrative without parsing stdout.
    """

    status: str
    target: str
    python_executable: str
    handshake: Optional[HandshakeResult] = None
    messages: list[str] = field(default_factory=list)


def handshake_mcp_server(
    python_executable: Optional[str] = None,
    timeout_seconds: float = 10.0,
) -> HandshakeResult:
    """Smoke-test the registered MCP launch command.

    Spawns ``<python> -m puppetmaster.mcp_server`` as a subprocess,
    sends a single ``tools/list`` JSON-RPC request on stdin, reads one
    line from stdout, and parses the response. The puppetmaster MCP
    server is designed to answer ``tools/list`` without requiring a
    prior ``initialize`` handshake, so this is a true end-to-end
    smoke test of the same launch command Cursor or Codex will run.

    A successful handshake proves three things at once: (1) the Python
    interpreter exists, (2) it can ``import puppetmaster``, and (3) the
    MCP server module starts cleanly. Any of those failing surfaces as
    ``ok=False`` with a descriptive ``error`` string instead of an
    exception, so callers can decide whether to print a warning and
    continue or treat it as a hard failure.
    """
    python = python_executable or sys.executable
    if not Path(python).expanduser().exists() and shutil.which(python) is None:
        return HandshakeResult(ok=False, error=f"python executable not resolvable: {python}")
    request = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    ) + "\n"
    try:
        completed = subprocess.run(
            [python, "-m", "puppetmaster.mcp_server"],
            input=request,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return HandshakeResult(
            ok=False,
            error=f"MCP server did not respond within {timeout_seconds}s (stderr={(exc.stderr or '')[-300:]!r})",
        )
    except OSError as exc:
        return HandshakeResult(ok=False, error=f"failed to spawn MCP server: {exc!r}")

    if completed.returncode != 0 and not completed.stdout:
        return HandshakeResult(
            ok=False,
            error=(
                f"MCP server exited rc={completed.returncode} with no stdout; "
                f"stderr={(completed.stderr or '')[-300:]!r}"
            ),
        )
    first_line = ""
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("{"):
            first_line = stripped
            break
    if not first_line:
        return HandshakeResult(
            ok=False,
            error=f"MCP server returned no JSON line; stdout={(completed.stdout or '')[-300:]!r}",
        )
    try:
        response = json.loads(first_line)
    except json.JSONDecodeError as exc:
        return HandshakeResult(ok=False, error=f"MCP response not valid JSON: {exc!r}")
    if "error" in response:
        return HandshakeResult(
            ok=False,
            error=f"MCP server returned JSON-RPC error: {response['error']}",
        )
    tools = (response.get("result") or {}).get("tools") or []
    if not isinstance(tools, list):
        return HandshakeResult(ok=False, error=f"MCP response.result.tools not a list: {tools!r}")
    return HandshakeResult(ok=True, tool_count=len(tools))


def install_cursor_mcp(
    *,
    target_path: Path,
    python_executable: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
    skip_handshake: bool = False,
) -> InstallResult:
    """Register or update Puppetmaster's MCP entry in a Cursor ``mcp.json``.

    ``target_path`` is the absolute path to a Cursor MCP config file —
    typically ``~/.cursor/mcp.json`` for a global install or
    ``<cwd>/.cursor/mcp.json`` for a workspace-local install.

    Idempotency: if the existing entry already matches the registration
    we would write, the function reports ``status="unchanged"`` and
    leaves the file untouched. ``force=True`` rewrites the entry even
    when it already matches (useful when re-running after Python
    interpreter relocation).

    Env preservation: ANY ``env`` block already on the puppetmaster
    entry is preserved verbatim. Users typically populate ``env`` with
    API keys (``CURSOR_API_KEY``, ``OPENAI_API_KEY``) and command
    overrides (``CLAUDE_CODE_COMMAND``); silently overwriting that
    would be a UX and security regression.

    Other entries in ``mcpServers`` are never touched.
    """
    python = python_executable or sys.executable
    messages: list[str] = []
    desired_entry = {
        "command": python,
        "args": ["-m", "puppetmaster.mcp_server"],
    }
    if target_path.exists():
        try:
            existing = json.loads(target_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except json.JSONDecodeError as exc:
            messages.append(f"existing config at {target_path} is not valid JSON: {exc!r}")
            return InstallResult(
                status="error",
                target=str(target_path),
                python_executable=python,
                messages=messages,
            )
    else:
        existing = {}
    mcp_servers = existing.setdefault("mcpServers", {})
    if not isinstance(mcp_servers, dict):
        messages.append(f"'mcpServers' in {target_path} is not an object; cannot merge")
        return InstallResult(
            status="error",
            target=str(target_path),
            python_executable=python,
            messages=messages,
        )
    prior = mcp_servers.get("puppetmaster") if isinstance(mcp_servers.get("puppetmaster"), dict) else None
    if prior and isinstance(prior.get("env"), dict):
        desired_entry["env"] = dict(prior["env"])
    needs_write = True
    if prior is not None:
        existing_command = prior.get("command")
        existing_args = prior.get("args")
        existing_env = prior.get("env") if isinstance(prior.get("env"), dict) else None
        same_command = existing_command == desired_entry["command"]
        same_args = list(existing_args or []) == desired_entry["args"]
        same_env = (existing_env or {}) == desired_entry.get("env", {})
        if same_command and same_args and same_env and not force:
            messages.append(
                f"puppetmaster entry in {target_path} already matches sys.executable; nothing to do"
            )
            return InstallResult(
                status="unchanged",
                target=str(target_path),
                python_executable=python,
                handshake=None,
                messages=messages,
            )
        if not same_command:
            messages.append(
                f"updating command: {existing_command!r} -> {desired_entry['command']!r}"
            )
        if not same_args:
            messages.append(
                f"updating args: {existing_args!r} -> {desired_entry['args']!r}"
            )

    handshake: Optional[HandshakeResult] = None
    if not skip_handshake:
        handshake = handshake_mcp_server(python)
        if not handshake.ok:
            messages.append(
                f"handshake FAILED — refusing to write a broken registration. "
                f"Reason: {handshake.error}"
            )
            return InstallResult(
                status="error",
                target=str(target_path),
                python_executable=python,
                handshake=handshake,
                messages=messages,
            )
        messages.append(
            f"handshake OK ({handshake.tool_count} tools advertised by {python})"
        )

    mcp_servers["puppetmaster"] = desired_entry

    if dry_run:
        messages.append(f"DRY RUN — would write to {target_path}")
        return InstallResult(
            status="would_install",
            target=str(target_path),
            python_executable=python,
            handshake=handshake,
            messages=messages,
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(existing, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, target_path)
    messages.append(f"wrote puppetmaster MCP entry to {target_path}")
    if needs_write and prior is not None and isinstance(prior.get("env"), dict) and prior["env"]:
        messages.append(
            f"preserved existing env block ({len(prior['env'])} key(s))"
        )
    return InstallResult(
        status="installed",
        target=str(target_path),
        python_executable=python,
        handshake=handshake,
        messages=messages,
    )


def install_codex_mcp(
    *,
    python_executable: Optional[str] = None,
    codex_executable: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
    skip_handshake: bool = False,
) -> InstallResult:
    """Register Puppetmaster as an MCP server in the OpenAI Codex CLI.

    Shells out to ``codex mcp add`` / ``codex mcp remove`` rather than
    hand-editing ``~/.codex/config.toml``: Codex owns the TOML schema
    and may change it across versions, so going through its CLI is the
    forward-compatible path.

    Idempotency: if ``codex mcp get puppetmaster`` already exists and
    its command + args match what we would register, the function
    reports ``status="unchanged"``. With ``force=True``, the existing
    entry is removed and re-added so a stale Python path can be fixed
    without manual intervention.

    Note: Codex stores no per-server env block on stdio MCP entries by
    default in the same way Cursor does, so this installer does not
    forward env vars. Users who need to pass env to the MCP subprocess
    should re-run with ``--env KEY=VAL`` after install via
    ``codex mcp add puppetmaster --env KEY=VAL -- <python> -m ...``
    or hand-edit the TOML.
    """
    python = python_executable or sys.executable
    codex = codex_executable or "codex"
    resolved_codex = shutil.which(codex) or (codex if Path(codex).expanduser().exists() else None)
    messages: list[str] = []
    if resolved_codex is None:
        messages.append(
            f"`codex` CLI not found on PATH (looked for {codex!r}). "
            f"Install with `npm install -g @openai/codex`, then re-run."
        )
        return InstallResult(
            status="error",
            target="~/.codex/config.toml",
            python_executable=python,
            messages=messages,
        )

    desired_command = python
    desired_args = ["-m", "puppetmaster.mcp_server"]
    existing_command = None
    existing_args: list[str] = []
    try:
        get_result = subprocess.run(
            [resolved_codex, "mcp", "get", "puppetmaster"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        messages.append(f"`codex mcp get puppetmaster` failed: {exc!r}")
        get_result = None

    if get_result is not None and get_result.returncode == 0:
        for raw in (get_result.stdout or "").splitlines():
            stripped = raw.strip()
            if stripped.startswith("command:"):
                existing_command = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("args:"):
                arg_blob = stripped.split(":", 1)[1].strip()
                existing_args = arg_blob.split()
        if (
            existing_command == desired_command
            and existing_args == desired_args
            and not force
        ):
            messages.append(
                "codex `puppetmaster` MCP entry already matches sys.executable; nothing to do"
            )
            return InstallResult(
                status="unchanged",
                target="~/.codex/config.toml",
                python_executable=python,
                handshake=None,
                messages=messages,
            )
        if existing_command:
            messages.append(
                f"existing codex entry will be replaced ({existing_command!r} -> {desired_command!r})"
            )

    handshake: Optional[HandshakeResult] = None
    if not skip_handshake:
        handshake = handshake_mcp_server(python)
        if not handshake.ok:
            messages.append(
                f"handshake FAILED — refusing to register a broken MCP entry in Codex. "
                f"Reason: {handshake.error}"
            )
            return InstallResult(
                status="error",
                target="~/.codex/config.toml",
                python_executable=python,
                handshake=handshake,
                messages=messages,
            )
        messages.append(
            f"handshake OK ({handshake.tool_count} tools advertised by {python})"
        )

    if dry_run:
        messages.append(
            f"DRY RUN — would run: "
            f"`{resolved_codex} mcp add puppetmaster -- {desired_command} {' '.join(desired_args)}`"
        )
        return InstallResult(
            status="would_install",
            target="~/.codex/config.toml",
            python_executable=python,
            handshake=handshake,
            messages=messages,
        )

    if get_result is not None and get_result.returncode == 0:
        try:
            subprocess.run(
                [resolved_codex, "mcp", "remove", "puppetmaster"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            messages.append(f"failed to remove existing entry: {exc!r}")

    add_cmd = [
        resolved_codex,
        "mcp",
        "add",
        "puppetmaster",
        "--",
        desired_command,
        *desired_args,
    ]
    try:
        add_result = subprocess.run(
            add_cmd,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        messages.append(f"`codex mcp add` failed: {exc!r}")
        return InstallResult(
            status="error",
            target="~/.codex/config.toml",
            python_executable=python,
            handshake=handshake,
            messages=messages,
        )
    if add_result.returncode != 0:
        messages.append(
            f"`codex mcp add` exited rc={add_result.returncode}: "
            f"stderr={(add_result.stderr or '')[-300:]!r}"
        )
        return InstallResult(
            status="error",
            target="~/.codex/config.toml",
            python_executable=python,
            handshake=handshake,
            messages=messages,
        )
    messages.append(
        f"registered puppetmaster MCP entry with Codex via `{resolved_codex} mcp add`"
    )
    return InstallResult(
        status="installed",
        target="~/.codex/config.toml",
        python_executable=python,
        handshake=handshake,
        messages=messages,
    )


CODEX_SANDBOX_GUIDANCE = (
    "Codex sandboxes MCP-server subprocesses inside the agent's sandbox. "
    "Puppetmaster's MCP server reads/writes ~/.puppetmaster/ (its durable "
    "state dir), which sits OUTSIDE any workspace. Two clean ways to use "
    "the MCP tools from Codex:\n"
    "  1. Interactive TUI: run `codex` (no flags). You'll get a one-time "
    "approval prompt the first time the MCP tool needs to read ~/.puppetmaster/; "
    "approve once and subsequent calls in the session pass cleanly. Best for "
    "daily-driver use.\n"
    "  2. Non-interactive `codex exec`: pass "
    "`--dangerously-bypass-approvals-and-sandbox`. Functionally equivalent "
    "to running Puppetmaster's CLI directly."
)


CURSOR_NEXT_STEPS_GUIDANCE = (
    "Restart Cursor or open a fresh chat to pick up the new MCP server.\n"
    "Cursor Agent should now see 32+ puppetmaster_* tools alongside its native tools.\n"
    "Verify by asking the agent to call `puppetmaster_doctor` and report the results."
)
