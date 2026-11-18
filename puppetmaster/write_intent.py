"""Shared write-intent matrix for every registered adapter.

``spec_edits_files`` (dirty-tree / swarm mode) and ``edit_admission`` (file
claims) must agree on whether a task may mutate the working tree. The
classifier lives here so a new adapter cannot fall out of the admission
layer just because a local frozenset was stale.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional


def canonicalize_adapter_name(adapter: str) -> str:
    try:
        from puppetmaster.platform_lock import canonicalize_adapter

        return canonicalize_adapter(adapter)
    except Exception:
        return str(adapter or "").strip().lower()


def adapter_may_write(
    adapter: str,
    payload: Optional[Mapping[str, Any]] = None,
    *,
    swarm_mode: Optional[str] = None,
) -> bool:
    """Return True when this adapter/payload pair may mutate the workspace.

    Conservative for unknown adapters: treat them as writers so claims and
    dirty-tree guards still fire.
    """
    name = canonicalize_adapter_name(adapter)
    payload = dict(payload or {})
    implement = (
        str(payload.get("mode") or "").strip().lower() == "implement"
        or bool(payload.get("implement"))
    )
    # ``swarm_mode`` is the swarm-level verdict from ``workers.swarm_mode``
    # ("analysis" when no spec in the wave can edit at all). It was declared
    # and then discarded, so callers believed they were constraining the
    # decision when nothing read it. Honour it as a conservative one-way
    # override: it can only ever remove a write claim, never grant one, and an
    # explicit implement payload still wins (matching ``spec_edits_files``).
    if str(swarm_mode or "").strip().lower() == "analysis" and not implement:
        return False
    if name in {"local", "openai"}:
        return implement
    if name == "shell":
        return True
    if name in {"agentic", "cursor", "hermes"}:
        # An explicit analysis / read-only payload is a hard no-edit fence even
        # for a full-edit adapter. This is the same predicate as
        # ``puppetmaster.workers.payload_forbids_writes`` (and therefore
        # ``spec_edits_files`` / ``swarm_mode``); inlined to avoid an import
        # cycle with ``workers``. Without it, every read-only agentic swarm
        # worker took an exclusive file claim on its whole write scope
        # (default "."), so a 5-role swarm became 1 winner + 4
        # EditAdmissionTimeout failures ("swarm exited with incomplete tasks").
        if (
            payload.get("read_only")
            or payload.get("no_edit")
            or payload.get("dry_run")
            or str(payload.get("sandbox") or "").strip().lower() == "read-only"
        ):
            return False
        return True
    if name == "claude-code":
        default = (
            "plan"
            if (payload.get("read_only") or payload.get("sandbox") == "read-only")
            else "acceptEdits"
        )
        return str(payload.get("permission_mode", default)) != "plan"
    if name == "codex":
        sandbox = str(payload.get("sandbox", "workspace-write"))
        bypass = bool(payload.get("dangerously_bypass_approvals_and_sandbox"))
        return sandbox != "read-only" or bypass
    if name in {"antigravity", "agy"}:
        # Use the adapter's own mode map, so "analyze" (run as agy --mode
        # plan) is read-only here too. Imported late to avoid an import cycle.
        from puppetmaster.adapters.antigravity import resolve_antigravity_mode

        return resolve_antigravity_mode(payload) != "plan"
    return True
