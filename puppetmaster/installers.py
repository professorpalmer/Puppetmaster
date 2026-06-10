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
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Codex's MCP client enforces a hard per-tool timeout (`tool_timeout_sec`,
# default 60s) and a startup timeout (`startup_timeout_sec`, default 10s).
# The 60s tool cap guillotines any long call — long-poll follows, the first
# cold codegraph index, sync verbs — which is what made Codex read healthy
# swarms as a dead tool. The server now caps blocking handlers under 60s on
# its own (PUPPETMASTER_MCP_MAX_BLOCK_SECONDS), but we also raise Codex's
# ceilings at install time so a genuinely-long single call survives and a
# cold Python import doesn't trip the 10s startup window. These are
# documented, stable Codex config keys; we only set them when absent so a
# user override is never clobbered.
CODEX_STARTUP_TIMEOUT_SEC = 30
CODEX_TOOL_TIMEOUT_SEC = 300


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
class UninstallResult:
    """Structured outcome of one uninstall run.

    ``status`` is one of ``"removed"`` (artifact deleted or entry stripped),
    ``"unchanged"`` (nothing Puppetmaster-owned was present),
    ``"would_remove"`` (dry-run preview), or ``"error"``.
    """

    status: str
    target: str
    messages: list[str] = field(default_factory=list)


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


def _ensure_codex_timeouts(config_path: Path) -> list[str]:
    """Set Codex tool/startup timeouts on the puppetmaster MCP entry if absent.

    ``codex mcp add`` writes a ``[mcp_servers.puppetmaster]`` table but never
    sets timeout keys, leaving the server exposed to Codex's 60s tool cap and
    10s startup cap. We insert ``tool_timeout_sec`` / ``startup_timeout_sec``
    directly after the table header when they aren't already present, so an
    explicit user value is preserved. Best-effort: any failure returns a
    warning message and never raises — the MCP entry is still functional.
    """
    messages: list[str] = []
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"could not read {config_path} to set Codex timeouts: {exc!r}"]

    lines = text.splitlines()
    header = re.compile(r"^\s*\[\s*mcp_servers\.puppetmaster\s*\]\s*$")
    next_table = re.compile(r"^\s*\[")
    header_idx = next((i for i, line in enumerate(lines) if header.match(line)), None)
    if header_idx is None:
        # Don't risk creating a duplicate table; leave config untouched.
        return [
            "skipped Codex timeout tuning: no [mcp_servers.puppetmaster] table "
            f"found in {config_path}. The server still works on Codex via the "
            "server-side block cap; set tool_timeout_sec manually for long sync calls."
        ]

    # Scan only this table's body (until the next table header or EOF).
    body_end = len(lines)
    for i in range(header_idx + 1, len(lines)):
        if next_table.match(lines[i]):
            body_end = i
            break
    body = lines[header_idx + 1 : body_end]
    has_tool = any(re.match(r"^\s*tool_timeout_sec\s*=", line) for line in body)
    has_startup = any(re.match(r"^\s*startup_timeout_sec\s*=", line) for line in body)

    inserts: list[str] = []
    if not has_startup:
        inserts.append(f"startup_timeout_sec = {CODEX_STARTUP_TIMEOUT_SEC}")
    if not has_tool:
        inserts.append(f"tool_timeout_sec = {CODEX_TOOL_TIMEOUT_SEC}")
    if not inserts:
        messages.append("Codex timeouts already set on puppetmaster entry; left as-is")
        return messages

    lines[header_idx + 1 : header_idx + 1] = inserts
    try:
        config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        return [f"could not write Codex timeouts to {config_path}: {exc!r}"]
    messages.append(
        f"set Codex timeouts on puppetmaster entry ({', '.join(inserts)})"
    )
    return messages


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
    forward-compatible path. The one exception is a minimal, absence-only
    insert of ``startup_timeout_sec`` / ``tool_timeout_sec`` after the
    server table (:func:`_ensure_codex_timeouts`), since ``codex mcp add``
    exposes no flag for them and Codex's 60s tool cap is what strangled
    long Puppetmaster calls.

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
            f", then set startup_timeout_sec={CODEX_STARTUP_TIMEOUT_SEC}/"
            f"tool_timeout_sec={CODEX_TOOL_TIMEOUT_SEC} on the entry if unset"
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
    messages.extend(_ensure_codex_timeouts(Path("~/.codex/config.toml").expanduser()))
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
    "to running Puppetmaster's CLI directly.\n"
    "\n"
    "Long jobs: Codex caps each tool call at tool_timeout_sec (default 60s); "
    "this installer raises it on the puppetmaster entry and the server now "
    "caps its own blocking calls under that ceiling. Still prefer the async "
    "pattern — fire a `start_*` verb, then re-poll with bounded "
    "`live_artifacts_follow` / `await_job` — instead of asking for one "
    "multi-minute block."
)


CURSOR_NEXT_STEPS_GUIDANCE = (
    "Restart Cursor or open a fresh chat to pick up the new MCP server.\n"
    "Cursor Agent should now see 32+ puppetmaster_* tools alongside its native tools.\n"
    "Verify by asking the agent to call `puppetmaster_doctor` and report the results."
)


def _remove_codex_puppetmaster_table(config_path: Path) -> tuple[bool, list[str]]:
    """Remove the ``[mcp_servers.puppetmaster]`` table from Codex config TOML.

    Uses the same line-oriented, regex-scoped editing as
    :func:`_ensure_codex_timeouts` so we never parse or rewrite unrelated
    tables. Returns ``(removed, messages)``.
    """
    messages: list[str] = []
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, [f"could not read {config_path}: {exc!r}"]

    lines = text.splitlines()
    header = re.compile(r"^\s*\[\s*mcp_servers\.puppetmaster\s*\]\s*$")
    next_table = re.compile(r"^\s*\[")
    header_idx = next((i for i, line in enumerate(lines) if header.match(line)), None)
    if header_idx is None:
        return False, []

    body_end = len(lines)
    for i in range(header_idx + 1, len(lines)):
        if next_table.match(lines[i]):
            body_end = i
            break
    new_lines = lines[:header_idx] + lines[body_end:]
    new_text = "\n".join(new_lines)
    if new_text and not new_text.endswith("\n"):
        new_text += "\n"
    if new_text == text:
        return False, []
    try:
        config_path.write_text(new_text, encoding="utf-8")
    except OSError as exc:
        return False, [f"could not write {config_path}: {exc!r}"]
    messages.append(f"removed [mcp_servers.puppetmaster] table from {config_path}")
    return True, messages


def uninstall_cursor_mcp(
    *,
    target_path: Path,
    dry_run: bool = False,
) -> UninstallResult:
    """Remove Puppetmaster's MCP entry from a Cursor ``mcp.json``.

    Other ``mcpServers`` entries are never touched. Idempotent: reports
    ``unchanged`` when no puppetmaster entry exists.
    """
    messages: list[str] = []
    if not target_path.is_file():
        messages.append(f"no config at {target_path}; nothing to remove")
        return UninstallResult(status="unchanged", target=str(target_path), messages=messages)

    try:
        existing = json.loads(target_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        messages.append(f"existing config at {target_path} is not valid JSON: {exc!r}")
        return UninstallResult(status="error", target=str(target_path), messages=messages)

    if not isinstance(existing, dict):
        messages.append(f"config at {target_path} is not a JSON object")
        return UninstallResult(status="error", target=str(target_path), messages=messages)

    mcp_servers = existing.get("mcpServers")
    if not isinstance(mcp_servers, dict) or "puppetmaster" not in mcp_servers:
        messages.append(f"no puppetmaster entry in {target_path}")
        return UninstallResult(status="unchanged", target=str(target_path), messages=messages)

    if dry_run:
        messages.append(f"DRY RUN — would remove puppetmaster entry from {target_path}")
        return UninstallResult(status="would_remove", target=str(target_path), messages=messages)

    del mcp_servers["puppetmaster"]
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(existing, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, target_path)
    messages.append(f"removed puppetmaster MCP entry from {target_path}")
    return UninstallResult(status="removed", target=str(target_path), messages=messages)


def uninstall_codex_mcp(
    *,
    codex_executable: Optional[str] = None,
    dry_run: bool = False,
) -> UninstallResult:
    """Remove Puppetmaster's MCP entry from the OpenAI Codex CLI config.

    Prefers ``codex mcp remove puppetmaster`` when the CLI is available,
    then verifies with :func:`_remove_codex_puppetmaster_table` so a stale
    table cannot survive a missing CLI.
    """
    codex = codex_executable or "codex"
    config_path = Path("~/.codex/config.toml").expanduser()
    target_label = str(config_path)
    messages: list[str] = []
    resolved_codex = shutil.which(codex) or (codex if Path(codex).expanduser().exists() else None)

    has_entry = False
    if resolved_codex is not None:
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
            has_entry = True

    table_present = False
    if config_path.is_file():
        header = re.compile(r"^\s*\[\s*mcp_servers\.puppetmaster\s*\]\s*$")
        table_present = any(header.match(line) for line in config_path.read_text(encoding="utf-8").splitlines())

    if not has_entry and not table_present:
        messages.append("no puppetmaster MCP entry in Codex config")
        return UninstallResult(status="unchanged", target=target_label, messages=messages)

    if dry_run:
        if has_entry and resolved_codex is not None:
            messages.append(
                f"DRY RUN — would run: `{resolved_codex} mcp remove puppetmaster`"
            )
        if table_present:
            messages.append(
                f"DRY RUN — would remove [mcp_servers.puppetmaster] from {config_path}"
            )
        return UninstallResult(status="would_remove", target=target_label, messages=messages)

    removed_any = False
    if has_entry and resolved_codex is not None:
        try:
            remove_result = subprocess.run(
                [resolved_codex, "mcp", "remove", "puppetmaster"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            messages.append(f"`codex mcp remove puppetmaster` failed: {exc!r}")
            remove_result = None
        if remove_result is not None and remove_result.returncode == 0:
            removed_any = True
            messages.append(
                f"removed puppetmaster MCP entry from Codex via `{resolved_codex} mcp remove`"
            )
        elif remove_result is not None and remove_result.returncode != 0:
            messages.append(
                f"`codex mcp remove` exited rc={remove_result.returncode}: "
                f"stderr={(remove_result.stderr or '')[-300:]!r}"
            )

    removed_table, table_messages = _remove_codex_puppetmaster_table(config_path)
    messages.extend(table_messages)
    removed_any = removed_any or removed_table

    if not removed_any:
        if messages:
            return UninstallResult(status="error", target=target_label, messages=messages)
        return UninstallResult(status="unchanged", target=target_label, messages=messages)

    return UninstallResult(status="removed", target=target_label, messages=messages)
