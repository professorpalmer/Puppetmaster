"""Platform lock: restrict which adapters (platforms) Puppetmaster may use.

Some users only pay for one platform — e.g. a Cursor plan — and never want
Puppetmaster to touch Claude Code, Codex, or the OpenAI API, even if those
CLIs happen to be installed and logged in. Others want the opposite: flip
several platforms on and bounce across their subscription / free tiers.

This module is the single source of truth for *which adapters are enabled*.
It is consulted at every decision point that could pick a platform:

* routing (`router.route_task` via `TaskSignals.allowed_adapters`),
* first-run plan-catalog auto-discovery (`orchestrator._ensure_plan_catalog`),
* auto-fallback rerouting (`orchestrator._reroute_recoverable_failures`).

State is a tiny denylist persisted next to the model registry
(`~/.puppetmaster/platform.json`), so "default = everything on" needs no
file at all. An env override (`PUPPETMASTER_ONLY_ADAPTERS=cursor,openai`)
wins over the file for ephemeral / CI use.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from puppetmaster.model_registry import default_registry_path

# The user-billable adapters the lock governs. ``shell`` and any future
# internal adapter are intentionally excluded — they are never platform-billed
# and must not be blocked by a platform lock.
KNOWN_ADAPTERS: tuple[str, ...] = ("cursor", "claude-code", "codex", "openai")

ONLY_ENV = "PUPPETMASTER_ONLY_ADAPTERS"


def platform_config_path(registry_path: Optional[Path] = None) -> Path:
    """Where the lock lives: ``platform.json`` beside the model registry."""
    base = (registry_path or default_registry_path()).parent
    return base / "platform.json"


def _parse_adapters(raw: str) -> set[str]:
    return {part.strip() for part in raw.split(",") if part.strip()}


def _read_disabled(registry_path: Optional[Path] = None) -> set[str]:
    path = platform_config_path(registry_path)
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()
    disabled = data.get("disabled", []) if isinstance(data, dict) else []
    return {str(a).strip() for a in disabled if str(a).strip()}


def _write_disabled(disabled: set[str], registry_path: Optional[Path] = None) -> Path:
    path = platform_config_path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Persist a stable, known-only, sorted list so the file stays readable
    # and forward entries we don't recognise don't accumulate.
    payload = {"disabled": sorted(a for a in disabled if a in KNOWN_ADAPTERS)}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def enabled_adapters(registry_path: Optional[Path] = None) -> set[str]:
    """The set of adapters currently allowed to run.

    Resolution order:

    1. ``$PUPPETMASTER_ONLY_ADAPTERS`` (comma list) — an explicit allowlist
       that wins over the file. Empty/unset falls through.
    2. The persisted denylist: ``KNOWN_ADAPTERS`` minus disabled.

    Defaults to every known adapter (no restriction).
    """
    env = os.environ.get(ONLY_ENV)
    if env and env.strip():
        return _parse_adapters(env)
    return set(KNOWN_ADAPTERS) - _read_disabled(registry_path)


def is_restricted(registry_path: Optional[Path] = None) -> bool:
    """True when a lock is actually narrowing the platform set."""
    return enabled_adapters(registry_path) != set(KNOWN_ADAPTERS)


def is_adapter_enabled(adapter: str, registry_path: Optional[Path] = None) -> bool:
    """Whether ``adapter`` may be used.

    Non-billable / internal adapters (anything outside ``KNOWN_ADAPTERS``,
    e.g. ``shell``) are never blocked by a platform lock.
    """
    if adapter not in KNOWN_ADAPTERS:
        return True
    return adapter in enabled_adapters(registry_path)


def active_allowlist(registry_path: Optional[Path] = None) -> Optional[frozenset[str]]:
    """Return the enabled set as a frozenset when restricted, else ``None``.

    ``None`` means "no restriction" — callers building ``TaskSignals`` should
    leave ``allowed_adapters`` unset in that case so routing behaves exactly
    as before for unlocked users.
    """
    if not is_restricted(registry_path):
        return None
    return frozenset(enabled_adapters(registry_path))


# --- mutators (used by the CLI) --------------------------------------------

def set_enabled(adapters: set[str], registry_path: Optional[Path] = None) -> Path:
    """Enable exactly ``adapters`` (the ``only`` command). Others get disabled."""
    disabled = set(KNOWN_ADAPTERS) - {a for a in adapters if a in KNOWN_ADAPTERS}
    return _write_disabled(disabled, registry_path)


def enable(adapters: set[str], registry_path: Optional[Path] = None) -> Path:
    """Turn ``adapters`` back on (remove from the denylist)."""
    disabled = _read_disabled(registry_path) - adapters
    return _write_disabled(disabled, registry_path)


def disable(adapters: set[str], registry_path: Optional[Path] = None) -> Path:
    """Turn ``adapters`` off (add to the denylist)."""
    disabled = _read_disabled(registry_path) | {a for a in adapters if a in KNOWN_ADAPTERS}
    return _write_disabled(disabled, registry_path)


def reset(registry_path: Optional[Path] = None) -> Path:
    """Clear the lock — every known adapter enabled again."""
    return _write_disabled(set(), registry_path)
