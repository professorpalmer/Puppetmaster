"""Install deterministic auto-invocation hooks into Cursor and Claude Code.

Where :mod:`puppetmaster.rules` writes *soft* prompt text (which the model may
ignore), this writes *hard* hooks: lifecycle commands the host runs
deterministically. Each hook shells back into
``<python> -m puppetmaster invocation-gate`` (see :mod:`puppetmaster.hook_runner`),
which reads the host payload on stdin and prints a host-specific verdict.

Targets:

* ``.cursor/hooks.json`` — Cursor workspace hooks. We register
  ``beforeSubmitPrompt`` (inject a delegate directive) plus
  ``beforeShellExecution`` / ``beforeReadFile`` (deny-redirect broad native
  exploration).
* ``.claude/settings.json`` — Claude Code hooks: ``UserPromptSubmit`` (inject)
  and ``PreToolUse`` matched on ``Grep|Glob|Task`` (deny-redirect).

Both writers are **idempotent and non-destructive**: they merge our entries in,
identified by the ``puppetmaster invocation-gate`` command string, and leave any
user-authored hooks untouched. Re-running replaces only our entries. The user
disables them by deleting our entries (or setting
``PUPPETMASTER_AUTO_INVOKE_DISABLED=1`` to neuter them at runtime).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

_GATE_MARKER = "puppetmaster invocation-gate"

VALID_HOOK_TARGETS = {"cursor", "claude"}


@dataclass
class HookOutcome:
    target: str
    path: str
    status: str  # installed | unchanged | would_install | skipped | error
    reason: str = ""


@dataclass
class HooksInstallResult:
    outcomes: list[HookOutcome] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    @property
    def overall_status(self) -> str:
        statuses = [o.status for o in self.outcomes]
        if any(s == "error" for s in statuses):
            return "error"
        if statuses and all(s == "unchanged" for s in statuses):
            return "unchanged"
        if any(s == "would_install" for s in statuses):
            return "would_install"
        if statuses and all(s == "skipped" for s in statuses):
            return "skipped"
        return "installed"


def _gate_command(host: str, event: str, python: Optional[str] = None) -> str:
    exe = python or sys.executable or "python3"
    return f"{exe} -m puppetmaster invocation-gate --host {host} --event {event}"


def _is_ours(entry: object) -> bool:
    """True if a hook entry is one we wrote (matched by the gate command)."""
    return _GATE_MARKER in json.dumps(entry)


def render_cursor_hooks(python: Optional[str] = None) -> dict:
    """The Cursor hook entries Puppetmaster owns."""
    return {
        "beforeSubmitPrompt": [{"command": _gate_command("cursor", "user-prompt", python)}],
        "beforeShellExecution": [{"command": _gate_command("cursor", "pre-tool", python)}],
        "beforeReadFile": [{"command": _gate_command("cursor", "pre-tool", python)}],
    }


def render_claude_hooks(python: Optional[str] = None) -> dict:
    """The Claude Code hook entries Puppetmaster owns."""
    return {
        "UserPromptSubmit": [
            {"hooks": [{"type": "command", "command": _gate_command("claude", "user-prompt", python)}]}
        ],
        "PreToolUse": [
            {
                "matcher": "Grep|Glob|Task",
                "hooks": [{"type": "command", "command": _gate_command("claude", "pre-tool", python)}],
            }
        ],
    }


def merge_hook_maps(existing: dict, ours: dict) -> tuple[dict, bool]:
    """Merge our per-event hook lists into ``existing['hooks']`` idempotently.

    Returns ``(merged, changed)``. For each event we drop any prior
    Puppetmaster entries, keep the user's, and append ours. ``changed`` is False
    when the result is byte-identical to the input (so callers can report
    ``unchanged``).
    """
    merged = dict(existing) if isinstance(existing, dict) else {}
    hooks = dict(merged.get("hooks") or {})
    for event, our_entries in ours.items():
        prior = [e for e in (hooks.get(event) or []) if not _is_ours(e)]
        hooks[event] = prior + list(our_entries)
    merged["hooks"] = hooks
    changed = json.dumps(existing or {}, sort_keys=True) != json.dumps(merged, sort_keys=True)
    return merged, changed


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError, ValueError):
        return {}


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _install_cursor(cwd: Path, *, dry_run: bool, force: bool, python: Optional[str]) -> HookOutcome:
    path = cwd / ".cursor" / "hooks.json"
    existing = _read_json(path)
    if "version" not in existing:
        existing = {"version": 1, **existing}
    merged, changed = merge_hook_maps(existing, render_cursor_hooks(python))
    if not changed and not force:
        return HookOutcome("cursor", str(path), "unchanged", ".cursor/hooks.json already current")
    if dry_run:
        return HookOutcome("cursor", str(path), "would_install", "would register Cursor beforeSubmitPrompt + deny-redirect hooks")
    _write_atomic(path, json.dumps(merged, indent=2) + "\n")
    return HookOutcome("cursor", str(path), "installed", "wrote Cursor prompt-inject + native-tool deny-redirect hooks")


def _install_claude(cwd: Path, *, dry_run: bool, force: bool, python: Optional[str]) -> HookOutcome:
    path = cwd / ".claude" / "settings.json"
    existing = _read_json(path)
    merged, changed = merge_hook_maps(existing, render_claude_hooks(python))
    if not changed and not force:
        return HookOutcome("claude", str(path), "unchanged", ".claude/settings.json already current")
    if dry_run:
        return HookOutcome("claude", str(path), "would_install", "would register Claude UserPromptSubmit + PreToolUse deny-redirect hooks")
    _write_atomic(path, json.dumps(merged, indent=2) + "\n")
    return HookOutcome("claude", str(path), "installed", "wrote Claude UserPromptSubmit + PreToolUse(Grep|Glob|Task) hooks")


def install_hooks(
    *,
    cwd: Optional[Path] = None,
    targets: Optional[Iterable[str]] = None,
    dry_run: bool = False,
    force: bool = False,
    python: Optional[str] = None,
) -> HooksInstallResult:
    """Install Puppetmaster auto-invocation hooks for ``targets``.

    Defaults to both Cursor and Claude Code. Pass ``targets`` to restrict.
    """
    cwd = cwd or Path.cwd()
    result = HooksInstallResult()
    selected = list(targets) if targets else ["cursor", "claude"]
    for target in selected:
        if target == "cursor":
            result.outcomes.append(_install_cursor(cwd, dry_run=dry_run, force=force, python=python))
        elif target == "claude":
            result.outcomes.append(_install_claude(cwd, dry_run=dry_run, force=force, python=python))
        else:
            result.outcomes.append(HookOutcome(target, "", "error", f"unknown hook target: {target!r}"))
    return result
