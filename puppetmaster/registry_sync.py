"""Starter-registry staleness check and append-only merge.

``puppetmaster models init`` writes the starter registry once. Later
releases add starter entries, and a registry written by an old release
does not get them. ``models init --force`` replaces the whole file, which
destroys local capability scores, prices and disabled flags.

This module finds the starter ids that a registry lacks and appends them.
It never changes an entry that is already in the file.

Rule for an appended entry (one rule, used by the CLI and the tests):

* Default: the entry is written with ``enabled: false``. Its
  ``disabled_reason`` names ``puppetmaster models sync`` and its
  ``disabled_authority`` is ``models_sync``. A new entry cannot win
  auto-routing and cannot resolve a pin until the owner enables it.
* ``enable=True``: the entry keeps the starter's own state, but only when
  the platform lock allows its adapter. An entry on an adapter that the
  platform lock disables is always written with ``enabled: false``.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Optional

from puppetmaster import platform_lock
from puppetmaster.fs_permissions import write_private_text
from puppetmaster.interprocess_lock import InterProcessFileLock
from puppetmaster.model_registry import (
    ModelSpec,
    _execution_identity,
    _spec_to_jsonable,
    default_registry_path,
    load_registry,
    resolve_model_pin,
    starter_registry,
)

SYNC_AUTHORITY = "models_sync"
SYNC_COMMAND = "puppetmaster models sync"


@dataclass(frozen=True)
class SyncAddition:
    """One starter entry that the registry lacks, as ``--apply`` writes it."""

    spec: ModelSpec
    adapter_locked: bool


@dataclass(frozen=True)
class SyncPlan:
    """The difference between the starter registry and one registry file."""

    path: Path
    registry_exists: bool
    additions: list[SyncAddition] = field(default_factory=list)
    # (starter id, existing id): the registry has the same adapter, wire model
    # and payload defaults below a different id. The starter id is not added.
    covered: list[tuple[str, str]] = field(default_factory=list)
    locked_adapters: list[str] = field(default_factory=list)
    applied: bool = False

    def by_adapter(self) -> dict[str, list[SyncAddition]]:
        grouped: dict[str, list[SyncAddition]] = {}
        for addition in self.additions:
            grouped.setdefault(addition.spec.adapter, []).append(addition)
        return dict(sorted(grouped.items()))


def enable_command(model_id: str) -> str:
    return (
        f"puppetmaster models set {model_id} enabled=true "
        "disabled_reason= disabled_authority="
    )


def _registry_arg(registry_path: Optional[Path]) -> str:
    """The `--registry-path` text for advice about a registry that is not the default one."""
    if registry_path is None:
        return ""
    path = Path(registry_path).expanduser().resolve()
    if path == default_registry_path().expanduser().resolve():
        return ""
    # One argument for the shell of this platform. POSIX quoting of a Windows
    # path gives single quotes, which cmd.exe does not read as quotes.
    text = subprocess.list2cmdline([str(path)]) if os.name == "nt" else shlex.quote(str(path))
    return f" --registry-path {text}"


def _synced_spec(spec: ModelSpec, *, adapter_locked: bool, enable: bool) -> ModelSpec:
    if not spec.is_routable or (enable and not adapter_locked):
        return spec
    if adapter_locked:
        reason = (
            f"added by `{SYNC_COMMAND}`; the platform lock disables adapter "
            f"{spec.adapter}"
        )
    else:
        reason = f"added by `{SYNC_COMMAND}`; enable with `{enable_command(spec.id)}`"
    return replace(
        spec,
        enabled=False,
        disabled_reason=reason,
        disabled_authority=SYNC_AUTHORITY,
    )


def plan_starter_sync(
    existing: Iterable[ModelSpec],
    *,
    enabled_adapters: Iterable[str],
    enable: bool = False,
) -> tuple[list[SyncAddition], list[tuple[str, str]]]:
    """Return ``(additions, covered)`` for a loaded registry."""
    materialized = list(existing)
    allowed = set(enabled_adapters)
    ids = {spec.id for spec in materialized}
    identities = {_execution_identity(spec): spec.id for spec in materialized}
    additions: list[SyncAddition] = []
    covered: list[tuple[str, str]] = []
    for spec in starter_registry():
        if spec.id in ids:
            continue
        same_identity = identities.get(_execution_identity(spec))
        if same_identity is not None:
            covered.append((spec.id, same_identity))
            continue
        locked = (
            spec.adapter in platform_lock.KNOWN_ADAPTERS
            and spec.adapter not in allowed
        )
        additions.append(
            SyncAddition(
                spec=_synced_spec(spec, adapter_locked=locked, enable=enable),
                adapter_locked=locked,
            )
        )
    return additions, covered


def _plan_for_path(path: Path, *, enable: bool) -> SyncPlan:
    enabled = platform_lock.enabled_adapters(path)
    locked = sorted(set(platform_lock.KNOWN_ADAPTERS) - enabled)
    if not path.is_file():
        return SyncPlan(path=path, registry_exists=False, locked_adapters=locked)
    additions, covered = plan_starter_sync(
        load_registry(path), enabled_adapters=enabled, enable=enable
    )
    return SyncPlan(
        path=path,
        registry_exists=True,
        additions=additions,
        covered=covered,
        locked_adapters=locked,
    )


def sync_starter_registry(
    path: Optional[Path] = None,
    *,
    apply: bool = False,
    enable: bool = False,
) -> SyncPlan:
    """Plan the starter merge for ``path``. With ``apply``, write it.

    The write appends rows to the parsed JSON document. It does not rebuild
    the rows that are already there, so their fields, their key order and
    the fields this release does not know stay as they are. A registry file
    that does not exist is not created: that is the job of ``models init``.

    Raises ``RuntimeError`` when the registry file is not valid.
    """
    resolved = path or default_registry_path()
    if not apply:
        return _plan_for_path(resolved, enable=enable)
    with InterProcessFileLock.for_target(resolved):
        plan = _plan_for_path(resolved, enable=enable)
        if not plan.registry_exists or not plan.additions:
            return plan
        document = json.loads(resolved.read_text(encoding="utf-8"))
        rows = document if isinstance(document, list) else document["models"]
        rows.extend(_spec_to_jsonable(addition.spec) for addition in plan.additions)
        write_private_text(
            resolved,
            json.dumps(document, indent=2, sort_keys=False) + "\n",
            lock=False,
        )
    return replace(plan, applied=True)


def unroutable_pin_hint(
    source: str,
    registry: Iterable[ModelSpec],
    *,
    adapter: str,
    registry_path: Optional[Path] = None,
) -> str:
    """Say why a pin did not resolve when the starter registry knows the id.

    Returns an empty string when the starter registry does not have the pin.
    ``registry_path`` is the file that ``registry`` came from. With it, the
    advice names a registry file that does not exist, and each command acts
    on that registry and not on the default one.
    """
    try:
        starter_pin = resolve_model_pin(source, starter_registry(), adapter=adapter)
    except ValueError:
        return ""
    if starter_pin is None:
        return ""
    where = _registry_arg(registry_path)
    if registry_path is not None and not Path(registry_path).expanduser().is_file():
        return (
            f"no model registry file is at {registry_path}. Run "
            f"`puppetmaster models init{where}` to make it."
        )
    specs = list(registry)
    present = next(
        (spec for spec in specs if spec.id == starter_pin.registry_id), None
    )
    if present is None:
        # `models sync` skips a starter id that another entry covers, so the
        # covering entry is the one the owner must fix.
        starter_spec = next(
            (s for s in starter_registry() if s.id == starter_pin.registry_id),
            None,
        )
        if starter_spec is not None:
            identity = _execution_identity(starter_spec)
            present = next(
                (spec for spec in specs if _execution_identity(spec) == identity),
                None,
            )
    if present is None:
        return (
            f"{starter_pin.registry_id!r} is a starter registry entry that this "
            f"registry does not have. Run `{SYNC_COMMAND}{where}` to list the absent "
            f"entries and `{SYNC_COMMAND} --apply{where}` to add them."
        )
    if present.retired:
        return f"{present.id!r} is retired: {present.retirement_reason}"
    if not present.enabled:
        detail = f": {present.disabled_reason}" if present.disabled_reason else ""
        enable = f"{enable_command(present.id)}{where}"
        locked = (
            present.adapter in platform_lock.KNOWN_ADAPTERS
            and present.adapter not in platform_lock.enabled_adapters(registry_path)
        )
        if locked:
            # An enabled entry on a locked adapter still fails at dispatch.
            lock = (
                ""
                if "platform lock" in (present.disabled_reason or "")
                else f" The platform lock disables adapter {present.adapter}."
            )
            return (
                f"{present.id!r} is disabled{detail}.{lock} To use this model, "
                f"first run `puppetmaster platform enable {present.adapter}`, "
                f"then `{enable}`."
            )
        return f"{present.id!r} is disabled{detail}. To enable it, run `{enable}`."
    return ""
