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
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from puppetmaster.redaction import redact_secrets, register_secret_value


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
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_KEY_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
_CODEX_WRAPPER_PATH = Path("~/.config/puppetmaster/codex-mcp-wrapper.py").expanduser()
_CODEX_MANAGED_ENV_PATH = Path("~/.config/puppetmaster/codex-mcp.env.json").expanduser()


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


@dataclass
class SdkBootstrapResult:
    """Outcome of an :func:`ensure_cursor_sdk` run.

    ``status`` is ``"installed"`` (npm fetched the SDK), ``"unchanged"``
    (already resolvable), ``"skipped"`` (no npm to bootstrap with), or
    ``"error"``. ``location`` is the resolved ``@cursor/sdk`` directory
    when one exists.
    """

    status: str
    detail: str
    location: Optional[str] = None


def ensure_cursor_sdk(
    root: Optional[Path] = None,
    *,
    package_root: Optional[Path] = None,
    npm_executable: Optional[str] = None,
    timeout_seconds: int = 180,
) -> SdkBootstrapResult:
    """Make ``@cursor/sdk`` resolvable for the installed Puppetmaster package.

    PyPI wheels cannot ship ``node_modules``, so a pip/pipx install lands
    without the SDK that ``cursor_sdk_runner.mjs`` resolves at runtime —
    leaving the cursor adapter dead and platform detection reporting
    "not detected" for users who are literally running Cursor. This
    bootstraps the SDK with ``npm install @cursor/sdk --prefix <package
    dir>``: the same directory Node's resolution walks up to from the
    runner script, so diagnostics and runtime agree afterward.
    """
    from puppetmaster.diagnostics import _find_cursor_sdk_install

    probe_root = root if root is not None else Path.cwd()
    existing = _find_cursor_sdk_install(probe_root)
    if existing is not None:
        return SdkBootstrapResult(
            "unchanged", f"@cursor/sdk already installed ({existing})", str(existing)
        )

    npm = npm_executable or shutil.which("npm")
    if npm is None:
        return SdkBootstrapResult(
            "skipped",
            "npm not on PATH — install Node.js, then re-run "
            "`puppetmaster install-cursor-mcp` to bootstrap @cursor/sdk",
        )

    install_root = package_root or Path(__file__).resolve().parent.parent
    try:
        completed = subprocess.run(
            # as_posix(): npm accepts forward slashes on every platform, and
            # native backslashes get mangled by Git-Bash-style shells on
            # Windows (same class of bug as the v0.9.34 hook-path fix).
            [npm, "install", "@cursor/sdk", "--prefix", install_root.as_posix()],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return SdkBootstrapResult(
            "error", f"npm install @cursor/sdk timed out after {timeout_seconds}s"
        )
    except OSError as exc:
        return SdkBootstrapResult("error", f"failed to spawn npm: {exc!r}")
    if completed.returncode != 0:
        output_lines = (completed.stderr or completed.stdout or "").strip().splitlines()
        tail = output_lines[-1] if output_lines else "no output"
        return SdkBootstrapResult(
            "error",
            f"npm install @cursor/sdk failed (exit {completed.returncode}): {tail}",
        )
    location = _find_cursor_sdk_install(probe_root)
    if location is None:
        return SdkBootstrapResult(
            "error",
            f"npm install succeeded but @cursor/sdk is still not resolvable "
            f"under {install_root}/node_modules",
        )
    return SdkBootstrapResult(
        "installed", f"@cursor/sdk bootstrapped into {location}", str(location)
    )


@dataclass(frozen=True)
class McpEnvRequest:
    """User-requested MCP environment sources.

    ``direct`` comes from ``--env KEY=VALUE``. ``inherit`` comes from
    ``--inherit-env KEY[,KEY...]`` and is copied from the installer process.
    ``env_files`` are parsed as simple shell-style env files. ``map_env`` maps
    a provider's canonical key from a local user key (``TARGET=SOURCE``).
    ``force`` lets requested values override existing MCP env keys; otherwise
    existing keys win to avoid silently clobbering user-owned credentials on
    reinstall.
    """

    direct: tuple[str, ...] = ()
    inherit: tuple[str, ...] = ()
    env_files: tuple[Path, ...] = ()
    map_env: tuple[str, ...] = ()
    force: bool = False


@dataclass
class McpEnvResolution:
    env: dict[str, str] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    secret_keys: set[str] = field(default_factory=set)
    requested_keys: set[str] = field(default_factory=set)
    uses_env_file: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors


def _is_valid_env_key(key: str) -> bool:
    return bool(_ENV_KEY_RE.match(key))


def _is_secret_like_key(key: str) -> bool:
    upper = key.upper()
    return any(hint in upper for hint in _SECRET_KEY_HINTS)


def _register_env_secrets(env: Mapping[str, str]) -> set[str]:
    secret_keys: set[str] = set()
    for key, value in env.items():
        if _is_secret_like_key(key):
            secret_keys.add(key)
            register_secret_value(value)
            continue
        if re.search(r"sk-[A-Za-z0-9_\-]{8,}", value) or re.search(
            r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}", value
        ):
            secret_keys.add(key)
            register_secret_value(value)
    return secret_keys


def _parse_direct_env(assignments: tuple[str, ...]) -> tuple[dict[str, str], list[str]]:
    env: dict[str, str] = {}
    errors: list[str] = []
    for raw in assignments:
        if "=" not in raw:
            errors.append("--env entries must be KEY=VALUE")
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        if not _is_valid_env_key(key):
            errors.append(f"invalid env key {key!r}")
            continue
        env[key] = value
    return env, errors


def _parse_env_mappings(mappings: tuple[str, ...]) -> tuple[list[tuple[str, str]], list[str]]:
    parsed: list[tuple[str, str]] = []
    errors: list[str] = []
    for raw in mappings:
        if "=" not in raw:
            errors.append("--map-env entries must be TARGET=SOURCE")
            continue
        target, source = (part.strip() for part in raw.split("=", 1))
        if not _is_valid_env_key(target):
            errors.append(f"invalid target env key {target!r}")
            continue
        if not _is_valid_env_key(source):
            errors.append(f"invalid source env key {source!r}")
            continue
        parsed.append((target, source))
    return parsed, errors


def _split_inherit_keys(raw_keys: tuple[str, ...]) -> tuple[list[str], list[str]]:
    keys: list[str] = []
    errors: list[str] = []
    seen: set[str] = set()
    for raw in raw_keys:
        for part in raw.split(","):
            key = part.strip()
            if not key:
                continue
            if not _is_valid_env_key(key):
                errors.append(f"invalid env key {key!r}")
                continue
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return keys, errors


def _parse_env_file(path: Path) -> tuple[dict[str, str], list[str], list[str]]:
    """Parse a conservative shell-style env file without executing it."""

    env: dict[str, str] = {}
    warnings: list[str] = []
    errors: list[str] = []
    expanded = path.expanduser()
    if not expanded.is_file():
        return env, warnings, [f"env file not found: {expanded}"]
    try:
        mode = expanded.stat().st_mode
    except OSError as exc:
        return env, warnings, [f"could not stat env file {expanded}: {exc!r}"]
    if mode & 0o077:
        warnings.append(
            f"env file {expanded} is group/world readable; recommended permissions are 0600 "
            f"(`chmod 600 {expanded}`)"
        )
    try:
        lines = expanded.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return env, warnings, [f"could not read env file {expanded}: {exc!r}"]

    for lineno, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            parts = shlex.split(stripped, comments=True, posix=True)
        except ValueError as exc:
            errors.append(f"{expanded}:{lineno}: could not parse shell env line: {exc}")
            continue
        if not parts:
            continue
        if parts[0] == "export":
            parts = parts[1:]
        elif parts[0] in {"set", "setenv", "source", "."}:
            warnings.append(f"{expanded}:{lineno}: ignored unsupported shell command")
            continue
        elif "=" not in parts[0]:
            warnings.append(f"{expanded}:{lineno}: ignored unsupported shell command")
            continue
        parsed_any = False
        for part in parts:
            if "=" not in part:
                warnings.append(f"{expanded}:{lineno}: ignored token without KEY=VALUE")
                continue
            key, value = part.split("=", 1)
            if not _is_valid_env_key(key):
                errors.append(f"{expanded}:{lineno}: invalid env key {key!r}")
                continue
            env[key] = value
            parsed_any = True
        if not parsed_any and parts:
            warnings.append(f"{expanded}:{lineno}: no env assignments found")
    return env, warnings, errors


def resolve_mcp_env(
    request: Optional[McpEnvRequest],
    *,
    existing_env: Optional[Mapping[str, str]] = None,
    source_env: Optional[Mapping[str, str]] = None,
) -> McpEnvResolution:
    """Resolve requested MCP env with deterministic precedence."""

    resolution = McpEnvResolution()
    if request is None:
        request = McpEnvRequest()
    source_env = source_env if source_env is not None else os.environ
    requested: dict[str, str] = {}

    for env_file in request.env_files:
        file_env, warnings, errors = _parse_env_file(env_file)
        resolution.uses_env_file = True
        resolution.messages.extend(warnings)
        resolution.errors.extend(errors)
        requested.update(file_env)
    inherit_keys, inherit_errors = _split_inherit_keys(request.inherit)
    resolution.errors.extend(inherit_errors)
    for key in inherit_keys:
        if key in source_env:
            requested[key] = str(source_env[key])
        else:
            resolution.messages.append(f"requested inherited env {key} is not set in installer environment")
    mappings, mapping_errors = _parse_env_mappings(request.map_env)
    resolution.errors.extend(mapping_errors)
    for target, source in mappings:
        if source in source_env:
            requested[target] = str(source_env[source])
        else:
            resolution.messages.append(
                f"requested env mapping {target}={source} skipped because {source} is not set"
            )
    direct, direct_errors = _parse_direct_env(request.direct)
    resolution.errors.extend(direct_errors)
    requested.update(direct)
    resolution.requested_keys = set(requested)

    existing = dict(existing_env or {})
    if request.force:
        merged = {**existing, **requested}
        overridden = sorted(set(existing) & set(requested))
        if overridden:
            resolution.messages.append(
                f"--force-env overriding existing env key(s): {', '.join(overridden)}"
            )
    else:
        merged = {**requested, **existing}
        preserved = sorted(set(existing) & set(requested))
        if preserved:
            resolution.messages.append(
                "preserved existing env key(s) "
                f"{', '.join(preserved)}; pass --force-env to override"
            )
    resolution.env = merged
    resolution.secret_keys = _register_env_secrets(merged)
    return resolution


def _managed_env_content(env: Mapping[str, str]) -> str:
    """Private, machine-readable managed env content for wrapper launchers."""
    return json.dumps({key: str(env[key]) for key in sorted(env)}, indent=2, sort_keys=True) + "\n"


def _write_private_file(path: Path, content: str, mode: int) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    current = None
    if path.exists():
        try:
            current = path.read_text(encoding="utf-8")
        except OSError:
            current = None
    if current == content:
        try:
            os.chmod(path, mode)
        except OSError:
            pass
        return False
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.chmod(tmp_path, mode)
    os.replace(tmp_path, path)
    return True


def _codex_wrapper_content(managed_env_path: Path) -> str:
    # Python wrapper instead of shell/zsh keeps the env-file/secret fallback
    # portable across Unix shells and Windows process launchers.
    return (
        "#!/usr/bin/env python3\n"
        "# Managed by Puppetmaster; re-run `puppetmaster install-codex-mcp` to update.\n"
        "import json\n"
        "import os\n"
        "import runpy\n"
        "from pathlib import Path\n"
        f"env_path = Path({str(managed_env_path)!r})\n"
        "if env_path.is_file():\n"
        "    with env_path.open('r', encoding='utf-8') as handle:\n"
        "        for key, value in json.load(handle).items():\n"
        "            os.environ[str(key)] = str(value)\n"
        "runpy.run_module('puppetmaster.mcp_server', run_name='__main__')\n"
    )


def _codex_config_path(env: Optional[Mapping[str, str]] = None) -> Path:
    env = env if env is not None else os.environ
    codex_home = env.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser() / "config.toml"
    return Path("~/.codex/config.toml").expanduser()


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


def _parse_toml_string(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return stripped[1:-1]
    if stripped.startswith("'") and stripped.endswith("'"):
        return stripped[1:-1]
    return stripped


def _read_codex_puppetmaster_env(config_path: Path) -> dict[str, str]:
    """Read the existing ``[mcp_servers.puppetmaster.env]`` table best-effort."""

    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    header = re.compile(r"^\s*\[\s*mcp_servers\.puppetmaster\.env\s*\]\s*$")
    next_table = re.compile(r"^\s*\[")
    header_idx = next((i for i, line in enumerate(lines) if header.match(line)), None)
    if header_idx is None:
        return {}
    env: dict[str, str] = {}
    for line in lines[header_idx + 1 :]:
        if next_table.match(line):
            break
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip().strip('"')
        if _is_valid_env_key(key):
            env[key] = _parse_toml_string(value)
    return env


def _codex_supports_stdio_env(codex: str) -> bool:
    try:
        result = subprocess.run(
            [codex, "mcp", "add", "--help"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    text = f"{result.stdout}\n{result.stderr}"
    return "--env <KEY=VALUE>" in text or "--env" in text


def _codex_needs_wrapper(
    resolution: McpEnvResolution,
    *,
    codex_supports_env: bool,
) -> bool:
    if not resolution.env:
        return False
    if not codex_supports_env:
        return True
    return resolution.uses_env_file or bool(resolution.secret_keys)


def install_codex_mcp(
    *,
    python_executable: Optional[str] = None,
    codex_executable: Optional[str] = None,
    force: bool = False,
    force_env: bool = False,
    env: tuple[str, ...] = (),
    inherit_env: tuple[str, ...] = (),
    env_files: tuple[Path, ...] = (),
    map_env: tuple[str, ...] = (),
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

    Env handling: ``env`` maps to explicit ``--env KEY=VALUE`` values,
    ``inherit_env`` copies selected keys from this installer process,
    ``map_env`` maps a canonical provider key from a local key, and
    ``env_files`` parses shell-style env files. Existing Codex env table keys
    are preserved unless ``force_env=True``. When the local Codex CLI supports
    stdio ``--env`` and the requested values are non-secret, we register native
    Codex env entries. For env-files, secret-like values, or Codex CLIs without
    stdio env support, we register a deterministic Puppetmaster-owned Python
    wrapper that loads a private 0600 JSON env file instead, keeping raw
    credentials out of Codex TOML and wrapper code.
    """
    python = python_executable or sys.executable
    codex = codex_executable or "codex"
    resolved_codex = shutil.which(codex) or (codex if Path(codex).expanduser().exists() else None)
    messages: list[str] = []
    target_path = _codex_config_path()
    if resolved_codex is None:
        messages.append(
            f"`codex` CLI not found on PATH (looked for {codex!r}). "
            f"Install with `npm install -g @openai/codex`, then re-run."
        )
        return InstallResult(
            status="error",
            target=str(target_path),
            python_executable=python,
            messages=messages,
        )

    existing_env = _read_codex_puppetmaster_env(target_path)
    env_request = McpEnvRequest(
        direct=tuple(env),
        inherit=tuple(inherit_env),
        env_files=tuple(Path(p).expanduser() for p in env_files),
        map_env=tuple(map_env),
        force=force_env,
    )
    env_resolution = resolve_mcp_env(env_request, existing_env=existing_env)
    messages.extend(env_resolution.messages)
    if env_resolution.errors:
        messages.extend(env_resolution.errors)
        return InstallResult(
            status="error",
            target=str(target_path),
            python_executable=python,
            messages=[redact_secrets(m) or "" for m in messages],
        )

    codex_supports_env = _codex_supports_stdio_env(resolved_codex) if env_resolution.env else True
    use_wrapper = bool(env_resolution.env) and _codex_needs_wrapper(
        env_resolution, codex_supports_env=codex_supports_env
    )
    wrapper_path = _CODEX_WRAPPER_PATH
    managed_env_path = _CODEX_MANAGED_ENV_PATH
    desired_command = python
    desired_args = [str(wrapper_path)] if use_wrapper else ["-m", "puppetmaster.mcp_server"]
    if use_wrapper:
        # Wrapper mode exists specifically to keep secret-like values and
        # env-file contents out of Codex TOML. Put the full effective env in
        # the private managed env file and clear native Codex env entries.
        desired_env = {}
    else:
        desired_env = env_resolution.env
    desired_env = dict(desired_env)
    existing_command = None
    existing_args: list[str] = []
    managed_files_current = True
    if use_wrapper:
        managed_env_content = _managed_env_content(env_resolution.env)
        wrapper_content = _codex_wrapper_content(managed_env_path)
        try:
            managed_files_current = (
                managed_env_path.read_text(encoding="utf-8") == managed_env_content
                and wrapper_path.read_text(encoding="utf-8") == wrapper_content
            )
        except OSError:
            managed_files_current = False
    try:
        get_result = subprocess.run(
            [resolved_codex, "mcp", "get", "puppetmaster"],
            stdin=subprocess.DEVNULL,
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
            and existing_env == desired_env
            and managed_files_current
            and not force
        ):
            timeout_messages = _ensure_codex_timeouts(target_path)
            messages.extend(timeout_messages)
            if any(message.startswith("set Codex timeouts") for message in timeout_messages):
                return InstallResult(
                    status="installed",
                    target=str(target_path),
                    python_executable=python,
                    handshake=None,
                    messages=[redact_secrets(m) or "" for m in messages],
                )
            messages.append(
                "codex `puppetmaster` MCP entry already matches sys.executable; nothing to do"
            )
            return InstallResult(
                status="unchanged",
                target=str(target_path),
                python_executable=python,
                handshake=None,
                messages=[redact_secrets(m) or "" for m in messages],
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
                target=str(target_path),
                python_executable=python,
                handshake=handshake,
                messages=[redact_secrets(m) or "" for m in messages],
            )
        messages.append(
            f"handshake OK ({handshake.tool_count} tools advertised by {python})"
        )

    if dry_run:
        env_note = ""
        if env_resolution.env:
            keys = ", ".join(sorted(env_resolution.env))
            env_note = f"; env key(s): {keys}"
        wrapper_note = ""
        if use_wrapper:
            wrapper_note = (
                f"; would register managed Python wrapper {wrapper_path} backed by private env file "
                f"{managed_env_path}"
            )
        elif env_resolution.env:
            wrapper_note = "; would use Codex native --env entries"
        messages.append(
            f"DRY RUN — would run: "
            f"`{resolved_codex} mcp add puppetmaster"
            f"{' [--env KEY=<redacted> ...]' if desired_env else ''}"
            f" -- {desired_command} {' '.join(desired_args)}`"
            f", then set startup_timeout_sec={CODEX_STARTUP_TIMEOUT_SEC}/"
            f"tool_timeout_sec={CODEX_TOOL_TIMEOUT_SEC} on the entry if unset"
            f"{env_note}{wrapper_note}"
        )
        return InstallResult(
            status="would_install",
            target=str(target_path),
            python_executable=python,
            handshake=handshake,
            messages=[redact_secrets(m) or "" for m in messages],
        )

    if use_wrapper:
        env_changed = _write_private_file(managed_env_path, managed_env_content, 0o600)
        wrapper_changed = _write_private_file(wrapper_path, wrapper_content, 0o700)
        messages.append(
            "using managed Codex MCP Python wrapper with private env file "
            f"({len(env_resolution.env)} key(s); values not printed)"
        )
        if env_changed:
            messages.append(f"wrote private managed env file {managed_env_path} (0600)")
        if wrapper_changed:
            messages.append(f"wrote managed wrapper {wrapper_path} (0700)")
    elif env_resolution.env:
        messages.append(
            f"using Codex native MCP env entries for key(s): {', '.join(sorted(env_resolution.env))}"
        )

    if get_result is not None and get_result.returncode == 0:
        try:
            subprocess.run(
                [resolved_codex, "mcp", "remove", "puppetmaster"],
                stdin=subprocess.DEVNULL,
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
    ]
    for key in sorted(desired_env):
        add_cmd.extend(["--env", f"{key}={desired_env[key]}"])
    add_cmd.extend(["--", desired_command, *desired_args])
    try:
        add_result = subprocess.run(
            add_cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        messages.append(f"`codex mcp add` failed: {exc!r}")
        return InstallResult(
            status="error",
            target=str(target_path),
            python_executable=python,
            handshake=handshake,
            messages=[redact_secrets(m) or "" for m in messages],
        )
    if add_result.returncode != 0:
        messages.append(
            f"`codex mcp add` exited rc={add_result.returncode}: "
            f"stderr={(add_result.stderr or '')[-300:]!r}"
        )
        return InstallResult(
            status="error",
            target=str(target_path),
            python_executable=python,
            handshake=handshake,
            messages=[redact_secrets(m) or "" for m in messages],
        )
    messages.append(
        f"registered puppetmaster MCP entry with Codex via `{resolved_codex} mcp add`"
    )
    messages.extend(_ensure_codex_timeouts(target_path))
    return InstallResult(
        status="installed",
        target=str(target_path),
        python_executable=python,
        handshake=handshake,
        messages=[redact_secrets(m) or "" for m in messages],
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


def _remove_codex_managed_files(*, dry_run: bool = False) -> tuple[bool, list[str]]:
    """Remove Puppetmaster-owned Codex wrapper/env artifacts if present."""

    removed_any = False
    messages: list[str] = []
    for path in (_CODEX_WRAPPER_PATH, _CODEX_MANAGED_ENV_PATH):
        if not path.exists():
            continue
        if dry_run:
            removed_any = True
            messages.append(f"DRY RUN — would remove managed Codex MCP file {path}")
            continue
        try:
            path.unlink()
        except OSError as exc:
            messages.append(f"could not remove managed Codex MCP file {path}: {exc!r}")
            continue
        removed_any = True
        messages.append(f"removed managed Codex MCP file {path}")
    return removed_any, messages


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
    config_path = _codex_config_path()
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

    managed_files_present = any(path.exists() for path in (_CODEX_WRAPPER_PATH, _CODEX_MANAGED_ENV_PATH))

    if not has_entry and not table_present and not managed_files_present:
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
        _, managed_messages = _remove_codex_managed_files(dry_run=True)
        messages.extend(managed_messages)
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

    removed_managed, managed_messages = _remove_codex_managed_files()
    messages.extend(managed_messages)
    removed_any = removed_any or removed_managed

    if not removed_any:
        if messages:
            return UninstallResult(status="error", target=target_label, messages=messages)
        return UninstallResult(status="unchanged", target=target_label, messages=messages)

    return UninstallResult(status="removed", target=target_label, messages=messages)


# ---------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------

_CLAUDE_USER_CONFIG_LABEL = "~/.claude.json"

CLAUDE_NEXT_STEPS_GUIDANCE = (
    "Restart Claude Code (or start a fresh session) to pick up the new MCP server.\n"
    "Verify with `claude mcp list` — puppetmaster should show as connected — or ask\n"
    "Claude to call `puppetmaster_doctor` and report the results.\n"
    "Long jobs: prefer the async pattern — fire a `start_*` verb, then re-poll with\n"
    "bounded `live_artifacts_follow` / `await_job` — instead of one multi-minute block."
)


def resolve_claude_command(claude_executable: Optional[str] = None) -> Optional[list[str]]:
    """Resolve the Claude Code CLI into an argv prefix, or ``None`` if absent.

    Resolution order: explicit ``claude_executable`` (may be multi-word, e.g.
    ``npx -y @anthropic-ai/claude-code``), then the ``CLAUDE_CODE_COMMAND`` env
    var the adapters already honor, then a bare ``claude`` on PATH. The first
    token must resolve to a real file or PATH entry; the rest ride along.
    """
    candidate = claude_executable or os.environ.get("CLAUDE_CODE_COMMAND") or "claude"
    try:
        parts = shlex.split(candidate, posix=(os.name != "nt"))
    except ValueError:
        return None
    if not parts:
        return None
    head = parts[0]
    resolved = shutil.which(head) or (head if Path(head).expanduser().exists() else None)
    if resolved is None:
        return None
    return [resolved, *parts[1:]]


def _parse_claude_mcp_get(stdout: str) -> tuple[Optional[str], list[str]]:
    """Extract (command, args) from ``claude mcp get`` human-readable output."""
    command: Optional[str] = None
    args: list[str] = []
    for raw in (stdout or "").splitlines():
        stripped = raw.strip()
        lower = stripped.lower()
        if lower.startswith("command:"):
            command = stripped.split(":", 1)[1].strip()
        elif lower.startswith("args:"):
            args = stripped.split(":", 1)[1].strip().split()
    return command, args


def install_claude_mcp(
    *,
    python_executable: Optional[str] = None,
    claude_executable: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
    skip_handshake: bool = False,
) -> InstallResult:
    """Register Puppetmaster as a user-scope MCP server in Claude Code.

    Shells out to ``claude mcp add --scope user`` rather than hand-editing
    ``~/.claude.json`` — Claude Code owns that file's schema (it also stores
    OAuth/onboarding state there), so going through its CLI is the only safe
    path. User scope makes the server available in every project, matching
    how people actually drive Puppetmaster from Claude Code.

    Idempotency: when ``claude mcp get puppetmaster`` reports a command + args
    matching what we would register, reports ``status="unchanged"``.
    ``force=True`` removes and re-adds the entry.

    Known limit: an existing *local/project*-scope puppetmaster entry shadows
    the user-scope one in Claude's precedence order; we manage only the user
    scope and surface a message instead of touching other scopes.
    """
    python = python_executable or sys.executable
    messages: list[str] = []
    claude_cmd = resolve_claude_command(claude_executable)
    if claude_cmd is None:
        messages.append(
            "Claude Code CLI not found (looked for "
            f"{claude_executable or os.environ.get('CLAUDE_CODE_COMMAND') or 'claude'!r}). "
            "Install with `npm install -g @anthropic-ai/claude-code` or set "
            "CLAUDE_CODE_COMMAND, then re-run."
        )
        return InstallResult(
            status="error",
            target=_CLAUDE_USER_CONFIG_LABEL,
            python_executable=python,
            messages=messages,
        )

    desired_command = python
    desired_args = ["-m", "puppetmaster.mcp_server"]

    get_result: Optional[subprocess.CompletedProcess] = None
    try:
        get_result = subprocess.run(
            [*claude_cmd, "mcp", "get", "puppetmaster"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        messages.append(f"`claude mcp get puppetmaster` failed: {exc!r}")

    entry_exists = get_result is not None and get_result.returncode == 0
    if entry_exists:
        existing_command, existing_args = _parse_claude_mcp_get(get_result.stdout)
        if (
            existing_command == desired_command
            and existing_args == desired_args
            and not force
        ):
            messages.append(
                "claude `puppetmaster` MCP entry already matches sys.executable; nothing to do"
            )
            return InstallResult(
                status="unchanged",
                target=_CLAUDE_USER_CONFIG_LABEL,
                python_executable=python,
                handshake=None,
                messages=messages,
            )
        if existing_command:
            messages.append(
                f"existing claude entry will be replaced ({existing_command!r} -> {desired_command!r})"
            )

    handshake: Optional[HandshakeResult] = None
    if not skip_handshake:
        handshake = handshake_mcp_server(python)
        if not handshake.ok:
            messages.append(
                f"handshake FAILED — refusing to register a broken MCP entry in Claude Code. "
                f"Reason: {handshake.error}"
            )
            return InstallResult(
                status="error",
                target=_CLAUDE_USER_CONFIG_LABEL,
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
            f"`{' '.join(claude_cmd)} mcp add --scope user puppetmaster -- "
            f"{desired_command} {' '.join(desired_args)}`"
        )
        return InstallResult(
            status="would_install",
            target=_CLAUDE_USER_CONFIG_LABEL,
            python_executable=python,
            handshake=handshake,
            messages=messages,
        )

    if entry_exists:
        try:
            subprocess.run(
                [*claude_cmd, "mcp", "remove", "--scope", "user", "puppetmaster"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            messages.append(f"failed to remove existing entry: {exc!r}")

    add_cmd = [
        *claude_cmd,
        "mcp",
        "add",
        "--scope",
        "user",
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
            timeout=60,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        messages.append(f"`claude mcp add` failed: {exc!r}")
        return InstallResult(
            status="error",
            target=_CLAUDE_USER_CONFIG_LABEL,
            python_executable=python,
            handshake=handshake,
            messages=messages,
        )
    if add_result.returncode != 0:
        messages.append(
            f"`claude mcp add` exited rc={add_result.returncode}: "
            f"stderr={(add_result.stderr or '')[-300:]!r}"
        )
        return InstallResult(
            status="error",
            target=_CLAUDE_USER_CONFIG_LABEL,
            python_executable=python,
            handshake=handshake,
            messages=messages,
        )
    messages.append(
        "registered puppetmaster MCP entry with Claude Code (user scope) via `claude mcp add`"
    )
    return InstallResult(
        status="installed",
        target=_CLAUDE_USER_CONFIG_LABEL,
        python_executable=python,
        handshake=handshake,
        messages=messages,
    )


def uninstall_claude_mcp(
    *,
    claude_executable: Optional[str] = None,
    dry_run: bool = False,
) -> UninstallResult:
    """Remove Puppetmaster's user-scope MCP entry from Claude Code.

    Idempotent: reports ``unchanged`` when no entry exists or the Claude CLI
    is absent (nothing we could have installed through it).
    """
    messages: list[str] = []
    claude_cmd = resolve_claude_command(claude_executable)
    if claude_cmd is None:
        messages.append("Claude Code CLI not found; no claude MCP entry to remove")
        return UninstallResult(
            status="unchanged", target=_CLAUDE_USER_CONFIG_LABEL, messages=messages
        )

    try:
        get_result = subprocess.run(
            [*claude_cmd, "mcp", "get", "puppetmaster"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        messages.append(f"`claude mcp get puppetmaster` failed: {exc!r}")
        return UninstallResult(
            status="error", target=_CLAUDE_USER_CONFIG_LABEL, messages=messages
        )

    if get_result.returncode != 0:
        messages.append("no puppetmaster MCP entry in Claude Code config")
        return UninstallResult(
            status="unchanged", target=_CLAUDE_USER_CONFIG_LABEL, messages=messages
        )

    if dry_run:
        messages.append(
            f"DRY RUN — would run: `{' '.join(claude_cmd)} mcp remove --scope user puppetmaster`"
        )
        return UninstallResult(
            status="would_remove", target=_CLAUDE_USER_CONFIG_LABEL, messages=messages
        )

    try:
        remove_result = subprocess.run(
            [*claude_cmd, "mcp", "remove", "--scope", "user", "puppetmaster"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        messages.append(f"`claude mcp remove puppetmaster` failed: {exc!r}")
        return UninstallResult(
            status="error", target=_CLAUDE_USER_CONFIG_LABEL, messages=messages
        )
    if remove_result.returncode != 0:
        messages.append(
            f"`claude mcp remove` exited rc={remove_result.returncode}: "
            f"stderr={(remove_result.stderr or '')[-300:]!r}"
        )
        return UninstallResult(
            status="error", target=_CLAUDE_USER_CONFIG_LABEL, messages=messages
        )
    messages.append(
        "removed puppetmaster MCP entry from Claude Code via `claude mcp remove`"
    )
    return UninstallResult(
        status="removed", target=_CLAUDE_USER_CONFIG_LABEL, messages=messages
    )


# --- Hermes (NousResearch hermes-agent) -----------------------------------
#
# Hermes stores MCP servers under ``mcp_servers.<name>`` in ``~/.hermes/
# config.yaml`` (overridable via ``$HERMES_HOME``). Unlike Codex and Claude
# Code — whose CLIs expose a non-interactive ``mcp add`` we can shell out to —
# ``hermes mcp add`` is discovery-first: it probes the server, then drops into
# an interactive tool-selection checklist that aborts without a TTY. So we
# register Puppetmaster the same way :func:`install_cursor_mcp` does for
# ``mcp.json``: a direct, idempotent edit of the config file Hermes owns,
# preserving every other key. ``yaml`` is imported lazily so Puppetmaster's
# zero-runtime-dependency contract holds for users who never touch Hermes;
# it is declared in the optional ``[hermes]`` extra.

HERMES_NEXT_STEPS_GUIDANCE = (
    "Restart Hermes (or start a fresh session) to pick up the new MCP server.\n"
    "Verify with `hermes mcp list` — puppetmaster should appear — or ask Hermes to\n"
    "call `puppetmaster_doctor` and report the results.\n"
    "Auto-invocation: the install also wired Hermes' native pre_llm_call / pre_tool_call\n"
    "hooks, so focused single edits get steered to `puppetmaster_edit` and broad work to\n"
    "a swarm automatically. Approve them once at the TTY prompt (or pass --accept-hooks),\n"
    "then check `hermes hooks list` / `hermes hooks doctor`. Disable anytime with\n"
    "PUPPETMASTER_AUTO_INVOKE_DISABLED=1.\n"
    "Single edits: `puppetmaster_edit \"<instruction>\"` (cheap model + CodeGraph, in place,\n"
    "returns a diff). Long jobs: prefer the async pattern — fire a `start_*` verb, then\n"
    "re-poll with bounded `live_artifacts_follow` / `await_job` instead of one long block."
)

_HERMES_PIP_HINT = (
    "PyYAML is required to edit Hermes' config.yaml but is not importable in "
    "this Python environment. Install it with `pip install puppetmaster-ai[hermes]` "
    "(or `pip install pyyaml`) into the same interpreter that runs Puppetmaster, "
    "then re-run `puppetmaster install-hermes-mcp`."
)


def hermes_config_path(env: Optional[Mapping[str, str]] = None) -> Path:
    """Return the path to Hermes' ``config.yaml``.

    Honors ``$HERMES_HOME`` (the same override Hermes itself reads) and falls
    back to ``~/.hermes/config.yaml``.
    """
    env = env if env is not None else os.environ
    hermes_home = env.get("HERMES_HOME")
    base = Path(hermes_home).expanduser() if hermes_home else Path("~/.hermes").expanduser()
    return base / "config.yaml"


def build_hermes_mcp_entry(python_executable: str, *, prior: Optional[Mapping] = None) -> dict:
    """Build the ``mcp_servers.puppetmaster`` entry for Hermes' config.

    Starts from any ``prior`` entry so user-set keys (``env``, ``tools``
    filters, ``timeout``) survive a re-install, and overrides only the launch
    ``command``/``args`` — mirroring how :func:`install_cursor_mcp` preserves
    an existing ``env`` block.
    """
    entry: dict = dict(prior) if isinstance(prior, Mapping) else {}
    # A stale HTTP transport from a hand-edited entry must not coexist with a
    # stdio command, or Hermes can't tell which transport to use.
    entry.pop("url", None)
    entry.pop("headers", None)
    entry["command"] = python_executable
    entry["args"] = ["-m", "puppetmaster.mcp_server"]
    return entry


def install_hermes_mcp(
    *,
    python_executable: Optional[str] = None,
    target_path: Optional[Path] = None,
    force: bool = False,
    dry_run: bool = False,
    skip_handshake: bool = False,
) -> InstallResult:
    """Register Puppetmaster as an MCP server in Hermes' ``config.yaml``.

    Idempotent: when the existing ``mcp_servers.puppetmaster`` entry already
    launches this interpreter's MCP server, reports ``status="unchanged"`` and
    leaves the file untouched. ``force=True`` rewrites even on a match. Any
    other keys on the entry (``env``, ``tools``, ``timeout``) and every other
    server/section in the config are preserved verbatim.
    """
    python = python_executable or sys.executable
    target = target_path or hermes_config_path()
    messages: list[str] = []

    try:
        import yaml  # type: ignore
    except Exception:  # pragma: no cover - exercised on hosts without PyYAML
        messages.append(_HERMES_PIP_HINT)
        return InstallResult(
            status="error",
            target=str(target),
            python_executable=python,
            messages=messages,
        )

    if target.exists():
        try:
            loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            messages.append(f"existing config at {target} is not valid YAML: {exc!r}")
            return InstallResult(
                status="error",
                target=str(target),
                python_executable=python,
                messages=messages,
            )
        config = loaded if isinstance(loaded, dict) else {}
        if loaded is not None and not isinstance(loaded, dict):
            messages.append(f"top-level of {target} is not a mapping; cannot merge")
            return InstallResult(
                status="error",
                target=str(target),
                python_executable=python,
                messages=messages,
            )
    else:
        config = {}

    mcp_servers = config.setdefault("mcp_servers", {})
    if not isinstance(mcp_servers, dict):
        messages.append(f"'mcp_servers' in {target} is not a mapping; cannot merge")
        return InstallResult(
            status="error",
            target=str(target),
            python_executable=python,
            messages=messages,
        )

    prior = mcp_servers.get("puppetmaster")
    prior = prior if isinstance(prior, dict) else None
    desired_entry = build_hermes_mcp_entry(python, prior=prior)

    if prior is not None and not force:
        same_command = prior.get("command") == desired_entry["command"]
        same_args = list(prior.get("args") or []) == desired_entry["args"]
        if same_command and same_args:
            messages.append(
                f"puppetmaster entry in {target} already launches {python}; nothing to do"
            )
            return InstallResult(
                status="unchanged",
                target=str(target),
                python_executable=python,
                messages=messages,
            )
        if not same_command:
            messages.append(
                f"updating command: {prior.get('command')!r} -> {desired_entry['command']!r}"
            )
        if not same_args:
            messages.append(
                f"updating args: {prior.get('args')!r} -> {desired_entry['args']!r}"
            )

    handshake: Optional[HandshakeResult] = None
    if not skip_handshake:
        handshake = handshake_mcp_server(python)
        if not handshake.ok:
            messages.append(
                "handshake FAILED — refusing to write a broken registration. "
                f"Reason: {handshake.error}"
            )
            return InstallResult(
                status="error",
                target=str(target),
                python_executable=python,
                handshake=handshake,
                messages=messages,
            )
        messages.append(
            f"handshake OK ({handshake.tool_count} tools advertised by {python})"
        )

    mcp_servers["puppetmaster"] = desired_entry

    if dry_run:
        messages.append(f"DRY RUN — would write puppetmaster MCP entry to {target}")
        return InstallResult(
            status="would_install",
            target=str(target),
            python_executable=python,
            handshake=handshake,
            messages=messages,
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(
        config,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=4096,
    )
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    tmp_path.write_text(rendered, encoding="utf-8")
    os.replace(tmp_path, target)
    messages.append(f"wrote puppetmaster MCP entry to {target}")
    if prior and isinstance(prior.get("env"), dict) and prior["env"]:
        messages.append(f"preserved existing env block ({len(prior['env'])} key(s))")
    return InstallResult(
        status="installed",
        target=str(target),
        python_executable=python,
        handshake=handshake,
        messages=messages,
    )


def uninstall_hermes_mcp(
    *,
    target_path: Optional[Path] = None,
    dry_run: bool = False,
) -> UninstallResult:
    """Remove Puppetmaster's MCP entry from Hermes' ``config.yaml``.

    Idempotent: reports ``unchanged`` when the config, the ``mcp_servers``
    block, or the ``puppetmaster`` entry is absent. Every other server and
    section is preserved.
    """
    target = target_path or hermes_config_path()
    messages: list[str] = []

    if not target.exists():
        messages.append(f"no Hermes config at {target}; nothing to remove")
        return UninstallResult(status="unchanged", target=str(target), messages=messages)

    try:
        import yaml  # type: ignore
    except Exception:  # pragma: no cover - exercised on hosts without PyYAML
        messages.append(_HERMES_PIP_HINT)
        return UninstallResult(status="error", target=str(target), messages=messages)

    try:
        loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        messages.append(f"existing config at {target} is not valid YAML: {exc!r}")
        return UninstallResult(status="error", target=str(target), messages=messages)

    config = loaded if isinstance(loaded, dict) else {}
    mcp_servers = config.get("mcp_servers")
    if not isinstance(mcp_servers, dict) or "puppetmaster" not in mcp_servers:
        messages.append(f"no puppetmaster MCP entry in {target}")
        return UninstallResult(status="unchanged", target=str(target), messages=messages)

    if dry_run:
        messages.append(f"DRY RUN — would remove puppetmaster MCP entry from {target}")
        return UninstallResult(status="would_remove", target=str(target), messages=messages)

    del mcp_servers["puppetmaster"]
    if not mcp_servers:
        config.pop("mcp_servers", None)
    rendered = yaml.safe_dump(
        config,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=4096,
    )
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    tmp_path.write_text(rendered, encoding="utf-8")
    os.replace(tmp_path, target)
    messages.append(f"removed puppetmaster MCP entry from {target}")
    return UninstallResult(status="removed", target=str(target), messages=messages)
