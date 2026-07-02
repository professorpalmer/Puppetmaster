"""`puppetmaster keys` — a guided setup for agentic provider API keys.

The agentic adapter calls provider HTTP APIs directly with the user's own key
(see :mod:`puppetmaster.providers`), resolving credentials purely from the
process environment. When Puppetmaster is driven through Cursor's MCP client,
that "process" is the MCP server Cursor launches — so a key only reaches the
worker if it lives in the ``env`` block of the ``puppetmaster`` entry in a
Cursor ``mcp.json``.

This module makes that setup a walk-through instead of hand-editing JSON:

* ``keys`` (no subcommand) — interactive wizard over the known providers.
* ``keys status`` — show, per provider, whether a key is visible to this
  process and whether one is stored in the target MCP config (values never
  printed).
* ``keys set <provider>`` — non-interactive single-provider write, reading the
  value from a hidden prompt or ``--stdin``.

Secrets are registered with :func:`register_secret_value` the moment we handle
them, so any later log/artifact output is scrubbed, and the config file is
tightened to ``0600`` because it now holds credentials. The value itself is
never echoed back.
"""
from __future__ import annotations

import getpass
import json
import os
import sys
from pathlib import Path
from typing import Callable, Mapping, Optional, TextIO

from puppetmaster.providers import (
    PROVIDER_REGISTRY,
    ProviderDescriptor,
    is_available,
)
from puppetmaster.redaction import register_secret_value

_PUPPETMASTER_ENTRY = "puppetmaster"


def _keyed_providers() -> list[ProviderDescriptor]:
    """Providers the wizard can set a key for, in a stable, friendly order.

    Keyless local endpoints (Ollama / LM Studio) are excluded: they are opted
    into with a base-URL/presence env var, not an API key, so a key prompt for
    them would be meaningless.
    """
    keyed = [desc for desc in PROVIDER_REGISTRY.values() if not desc.keyless]
    # Dedupe on the primary env var so ``openai`` and ``openai-api`` (which
    # share OPENAI_API_KEY) don't both appear as separate rows to fill in.
    seen: set[str] = set()
    unique: list[ProviderDescriptor] = []
    for desc in sorted(keyed, key=lambda d: d.label or d.slug):
        primary = desc.api_key_env_vars[0]
        if primary in seen:
            continue
        seen.add(primary)
        unique.append(desc)
    return unique


# --- MCP config I/O ---------------------------------------------------------


def default_cursor_config_path(*, workspace: bool = False) -> Path:
    """The Cursor ``mcp.json`` the wizard writes to by default.

    Global (``~/.cursor/mcp.json``) is the default because that is the config
    Cursor loads for every workspace; ``--workspace`` targets the repo-local
    ``.cursor/mcp.json`` instead.
    """
    root = Path.cwd() if workspace else Path.home()
    return (root / ".cursor" / "mcp.json").resolve()


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def _puppetmaster_env(config: Mapping) -> dict:
    """The stored env block on the puppetmaster MCP entry (read-only view)."""
    servers = config.get("mcpServers")
    if not isinstance(servers, dict):
        return {}
    entry = servers.get(_PUPPETMASTER_ENTRY)
    if not isinstance(entry, dict):
        return {}
    env = entry.get("env")
    return dict(env) if isinstance(env, dict) else {}


def _write_env_key(path: Path, env_var: str, value: str) -> None:
    """Persist ``env_var=value`` onto the puppetmaster entry's env block.

    Creates a minimal puppetmaster entry (matching what ``install-cursor-mcp``
    writes) if the config or entry is absent, so the wizard is self-sufficient
    on a fresh machine. Tightens the file to ``0600`` since it now holds a
    credential.
    """
    config = _load_config(path)
    servers = config.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise ValueError(f"'mcpServers' in {path} is not an object; cannot merge")
    entry = servers.get(_PUPPETMASTER_ENTRY)
    if not isinstance(entry, dict):
        entry = {"command": sys.executable, "args": ["-m", "puppetmaster.mcp_server"]}
        servers[_PUPPETMASTER_ENTRY] = entry
    env = entry.get("env")
    if not isinstance(env, dict):
        env = {}
        entry["env"] = env
    env[env_var] = value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


# --- shared write path ------------------------------------------------------


def _apply_key(
    desc: ProviderDescriptor, value: str, path: Path
) -> tuple[bool, str]:
    """Register + persist one provider key. Returns ``(ok, message)``.

    The value is registered as a secret immediately and never returned in the
    message, so callers can print the result without leaking the credential.
    """
    value = (value or "").strip()
    if not value:
        return False, f"{desc.label}: empty value — nothing written."
    register_secret_value(value)
    env_var = desc.api_key_env_vars[0]
    try:
        _write_env_key(path, env_var, value)
    except (OSError, ValueError) as exc:
        return False, f"{desc.label}: could not write {path}: {exc}"
    return True, f"{desc.label}: set {env_var} in {path}"


# --- status -----------------------------------------------------------------


def _run_status(path: Path, env: Mapping[str, str]) -> int:
    stored = _puppetmaster_env(_load_config(path)) if path.exists() else {}
    print(f"Provider API keys ({path}{'' if path.exists() else ' — not created yet'})")
    print(f"{'PROVIDER':<16}  {'ENV VAR':<20}  {'PROCESS':<9}  STORED")
    for desc in _keyed_providers():
        env_var = desc.api_key_env_vars[0]
        in_process = "visible" if is_available(desc, env) else "-"
        in_config = "yes" if any(k in stored for k in desc.api_key_env_vars) else "-"
        print(f"{desc.label:<16}  {env_var:<20}  {in_process:<9}  {in_config}")
    print()
    print("PROCESS = key visible to this shell now.  STORED = key saved in the MCP config.")
    print("Values are never displayed. Restart the Cursor MCP server to pick up new keys.")
    return 0


# --- interactive wizard -----------------------------------------------------


class KeyWizard:
    """Walk the user through setting agentic provider keys into an MCP config.

    ``read_secret`` is injectable so tests can drive the wizard without a TTY;
    it defaults to :func:`getpass.getpass`, which reads without echoing.
    """

    def __init__(
        self,
        path: Path,
        stdin: TextIO,
        stdout: TextIO,
        *,
        env: Optional[Mapping[str, str]] = None,
        read_secret: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.path = path
        self.stdin = stdin
        self.stdout = stdout
        self.env = env if env is not None else os.environ
        self.read_secret = read_secret or getpass.getpass
        self.wrote_any = False

    def run(self) -> int:
        try:
            return self._run()
        except EOFError:
            self._write("\nInput closed — exiting.")
            return 0 if self.wrote_any else 1

    def _run(self) -> int:
        providers = _keyed_providers()
        self._write("Puppetmaster — set agentic provider API keys")
        self._write(f"Writing to MCP config: {self.path}")
        self._write("Keys are stored in the puppetmaster MCP entry's env block; "
                     "values are never displayed.")
        self._write("")
        for index, desc in enumerate(providers, 1):
            marker = "already visible" if is_available(desc, self.env) else "not set"
            self._write(f"[{index}] {desc.label:<16} {desc.api_key_env_vars[0]:<20} ({marker})")
        self._write("")
        self._write("Enter a number to set that provider's key, or 'q' to finish.")

        while True:
            choice = self._prompt("> ").strip().lower()
            if choice in ("q", "quit", "exit", ""):
                break
            if not choice.isdigit() or not (1 <= int(choice) <= len(providers)):
                self._write(f"Please enter 1–{len(providers)} or 'q'.")
                continue
            desc = providers[int(choice) - 1]
            value = self.read_secret(f"{desc.label} key (input hidden): ")
            ok, message = _apply_key(desc, value, self.path)
            self._write(message)
            if ok:
                self.wrote_any = True

        if self.wrote_any:
            self._write("")
            self._write("Done. Restart the Cursor MCP server (or reload the window) so the "
                        "new keys reach agentic workers, then run "
                        "`puppetmaster models discover --source agentic --write` to seed models.")
        else:
            self._write("No keys set.")
        return 0

    def _write(self, line: str) -> None:
        self.stdout.write(line + "\n")

    def _prompt(self, text: str) -> str:
        self.stdout.write(text)
        self.stdout.flush()
        line = self.stdin.readline()
        if line == "":
            raise EOFError("input closed")
        return line.rstrip("\n")


# --- dispatch ---------------------------------------------------------------


def _resolve_target(args) -> Path:
    explicit = getattr(args, "target", None)
    if explicit:
        return Path(explicit).expanduser().resolve()
    return default_cursor_config_path(workspace=getattr(args, "workspace", False))


def _run_set(args, path: Path) -> int:
    slug = str(args.provider).strip().lower()
    desc = PROVIDER_REGISTRY.get(slug)
    if desc is None or desc.keyless:
        known = ", ".join(d.slug for d in _keyed_providers())
        print(f"error: unknown or keyless provider {args.provider!r}. Known: {known}",
              file=sys.stderr)
        return 1
    if getattr(args, "stdin", False):
        value = sys.stdin.readline().rstrip("\n")
    else:
        value = getpass.getpass(f"{desc.label} key (input hidden): ")
    ok, message = _apply_key(desc, value, path)
    print(message, file=sys.stdout if ok else sys.stderr)
    if ok:
        print("Restart the Cursor MCP server to pick up the new key.")
    return 0 if ok else 1


def run_keys_subcommand(args) -> int:
    """Dispatch ``python -m puppetmaster keys ...``."""
    path = _resolve_target(args)
    sub = getattr(args, "keys_command", None)
    if sub == "status":
        return _run_status(path, os.environ)
    if sub == "set":
        return _run_set(args, path)
    return KeyWizard(path, sys.stdin, sys.stdout).run()
