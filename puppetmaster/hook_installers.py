"""Install deterministic auto-invocation hooks into Cursor and Claude Code.

Where :mod:`puppetmaster.rules` writes *soft* prompt text (which the model may
ignore), this writes *hard* hooks: lifecycle commands the host runs
deterministically. Each hook shells back into
``<python> -m puppetmaster invocation-gate`` (see :mod:`puppetmaster.hook_runner`),
which reads the host payload on stdin and prints a host-specific verdict.

Targets:

* ``.cursor/hooks.json`` — Cursor hooks. We register ``beforeSubmitPrompt``
  (inject a delegate directive) plus ``beforeShellExecution`` /
  ``beforeReadFile`` (deny-redirect genuinely recursive shell searches + Task
  fan-out; read-only inspection passes through).
* ``.claude/settings.json`` — Claude Code hooks: ``UserPromptSubmit`` (inject)
  and ``PreToolUse`` matched on ``Grep|Glob|Task`` (deny-redirect).

Each target has two **scopes**, differing only in the base directory the same
subpath hangs off:

* ``project`` (default) — ``<cwd>/.cursor/hooks.json`` and
  ``<cwd>/.claude/settings.json``. Covers this repo only; can be checked in.
* ``global`` — ``~/.cursor/hooks.json`` and ``~/.claude/settings.json``. Covers
  every repo the user opens, so they don't re-run setup per project. Our hook
  command is an absolute ``python -m puppetmaster invocation-gate`` (no relative
  script path), so it resolves identically regardless of the host's cwd — which
  is what makes a user-level Cursor hook (run from ``~/.cursor/``) work.

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
from typing import Iterable, Mapping, Optional

_GATE_MARKER = "puppetmaster invocation-gate"

VALID_HOOK_TARGETS = {"cursor", "claude"}
VALID_HOOK_SCOPES = {"project", "global"}


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
        if any(s == "would_remove" for s in statuses):
            return "would_remove"
        if any(s == "removed" for s in statuses):
            return "removed"
        if statuses and all(s == "skipped" for s in statuses):
            return "skipped"
        return "installed"


def _shell_safe_executable(exe: str) -> str:
    """Make a Python path safe to embed in a hook command string.

    Hosts execute hook commands through a POSIX-style shell — on Windows
    that's Git Bash, where unquoted backslashes are escape characters, so
    ``C:\\Users\\me\\python.exe`` arrives as ``C:Usersmepython.exe`` (field
    report: Claude Code on Windows). Forward slashes are valid Windows path
    separators everywhere this command runs, and quoting protects spaces
    (``Program Files``-style paths) in any shell.
    """
    safe = exe.replace("\\", "/")
    return f'"{safe}"' if " " in safe else safe


def _gate_command(host: str, event: str, python: Optional[str] = None) -> str:
    exe = _shell_safe_executable(python or sys.executable or "python3")
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


def strip_hook_maps(existing: dict) -> tuple[dict, bool]:
    """Remove Puppetmaster hook entries from ``existing['hooks']``.

    Returns ``(stripped, changed)``. User-authored hooks are preserved.
    """
    merged = dict(existing) if isinstance(existing, dict) else {}
    hooks = dict(merged.get("hooks") or {})
    changed = False
    for event, entries in list(hooks.items()):
        prior = [e for e in (entries or []) if not _is_ours(e)]
        if len(prior) != len(entries or []):
            changed = True
        if prior:
            hooks[event] = prior
        elif event in hooks:
            del hooks[event]
    if hooks:
        merged["hooks"] = hooks
    elif "hooks" in merged:
        del merged["hooks"]
        changed = True
    if json.dumps(existing or {}, sort_keys=True) != json.dumps(merged, sort_keys=True):
        changed = True
    return merged, changed


def _hooks_file_is_empty(data: dict) -> bool:
    """True when a hooks/settings file has no meaningful content left."""
    if not data:
        return True
    hooks = data.get("hooks") or {}
    if hooks:
        return False
    other_keys = set(data.keys()) - {"version", "hooks"}
    return len(other_keys) == 0


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


def _scoped_label(scope: str, relpath: str) -> str:
    """Human label for a hook file, e.g. ``~/.cursor/hooks.json`` (global)."""
    return f"~/{relpath}" if scope == "global" else relpath


def _install_cursor(base_dir: Path, *, scope: str, dry_run: bool, force: bool, python: Optional[str]) -> HookOutcome:
    path = base_dir / ".cursor" / "hooks.json"
    label = _scoped_label(scope, ".cursor/hooks.json")
    existing = _read_json(path)
    if "version" not in existing:
        existing = {"version": 1, **existing}
    merged, changed = merge_hook_maps(existing, render_cursor_hooks(python))
    if not changed and not force:
        return HookOutcome("cursor", str(path), "unchanged", f"{label} already current")
    if dry_run:
        return HookOutcome("cursor", str(path), "would_install", f"would register Cursor {scope} beforeSubmitPrompt + deny-redirect hooks")
    _write_atomic(path, json.dumps(merged, indent=2) + "\n")
    return HookOutcome("cursor", str(path), "installed", f"wrote Cursor {scope} prompt-inject + native-tool deny-redirect hooks")


def _install_claude(base_dir: Path, *, scope: str, dry_run: bool, force: bool, python: Optional[str]) -> HookOutcome:
    path = base_dir / ".claude" / "settings.json"
    label = _scoped_label(scope, ".claude/settings.json")
    existing = _read_json(path)
    merged, changed = merge_hook_maps(existing, render_claude_hooks(python))
    if not changed and not force:
        return HookOutcome("claude", str(path), "unchanged", f"{label} already current")
    if dry_run:
        return HookOutcome("claude", str(path), "would_install", f"would register Claude {scope} UserPromptSubmit + PreToolUse deny-redirect hooks")
    _write_atomic(path, json.dumps(merged, indent=2) + "\n")
    return HookOutcome("claude", str(path), "installed", f"wrote Claude {scope} UserPromptSubmit + PreToolUse(Grep|Glob|Task) hooks")


_HOOK_TARGET_ADAPTERS = {"cursor": "cursor", "claude": "claude-code"}


# ---------------------------------------------------------------------------
# Hermes hooks (YAML config.yaml, not a JSON sidecar)
# ---------------------------------------------------------------------------
#
# Hermes drives auto-invocation through its native shell-hook system
# (agent/shell_hooks.py): a ``hooks:`` block in ``~/.hermes/config.yaml`` maps
# each event to a list of ``{command: ...}`` entries. Hermes runs the command,
# pipes the hook payload as JSON on stdin, and reads the command's stdout:
#
#   * ``pre_llm_call``  → honors ``{"context": "<str>"}`` (injected into the
#                         user message; the per-turn delegate directive).
#   * ``pre_tool_call`` → honors ``{"action": "block", "message": ...}`` (the
#                         broad-native-search deny-redirect).
#
# Both events shell into the same ``python -m puppetmaster invocation-gate
# --host hermes`` brain the Cursor/Claude hooks use, so there is exactly one
# policy across every host. We register both entries, identified (for
# idempotent re-install / clean uninstall) by the gate-command marker.

#: Hermes events we own and the canonical gate event each maps to.
_HERMES_HOOK_EVENTS = {
    "pre_llm_call": "user-prompt",
    "pre_tool_call": "pre-tool",
}


def render_hermes_hooks(python: Optional[str] = None) -> dict:
    """The Hermes ``hooks:`` entries Puppetmaster owns, keyed by event."""
    return {
        event: [{"command": _gate_command("hermes", gate_event, python)}]
        for event, gate_event in _HERMES_HOOK_EVENTS.items()
    }


def merge_hermes_hooks(existing_hooks: object, ours: dict) -> tuple[dict, bool]:
    """Merge our Hermes hook entries into an existing ``hooks:`` mapping.

    Returns ``(merged, changed)``. For each event we drop any prior Puppetmaster
    entries (matched by the gate-command marker), keep the user's, and append
    ours. ``changed`` is False when the result is byte-identical to the input so
    callers can report ``unchanged``. ``existing_hooks`` may be ``{}`` (Hermes'
    default), ``None``, or a populated mapping.
    """
    merged = dict(existing_hooks) if isinstance(existing_hooks, dict) else {}
    for event, our_entries in ours.items():
        prior_list = merged.get(event)
        prior = [e for e in (prior_list or []) if not _is_ours(e)] if isinstance(prior_list, list) else []
        merged[event] = prior + list(our_entries)
    changed = json.dumps(existing_hooks if isinstance(existing_hooks, dict) else {}, sort_keys=True) != json.dumps(merged, sort_keys=True)
    return merged, changed


def strip_hermes_hooks(existing_hooks: object) -> tuple[dict, bool]:
    """Remove Puppetmaster entries from a Hermes ``hooks:`` mapping.

    Returns ``(stripped, changed)``. User-authored hook entries are preserved;
    an event whose only entries were ours is dropped entirely.
    """
    merged = dict(existing_hooks) if isinstance(existing_hooks, dict) else {}
    changed = False
    for event, entries in list(merged.items()):
        if not isinstance(entries, list):
            continue
        prior = [e for e in entries if not _is_ours(e)]
        if len(prior) != len(entries):
            changed = True
        if prior:
            merged[event] = prior
        else:
            del merged[event]
    return merged, changed


def install_hermes_hooks(
    *,
    target_path: Optional[Path] = None,
    dry_run: bool = False,
    force: bool = False,
    python: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> HookOutcome:
    """Register Puppetmaster auto-invocation hooks in Hermes' ``config.yaml``.

    Edits only the ``hooks:`` block; every other server/section is preserved
    verbatim. Idempotent and non-destructive. Requires PyYAML (ships with the
    ``puppetmaster-ai[hermes]`` extra and with Hermes itself).
    """
    from puppetmaster.installers import hermes_config_path

    target = target_path or hermes_config_path(env)
    label = str(target)
    try:
        import yaml  # type: ignore
    except Exception:
        return HookOutcome(
            "hermes", label, "error",
            "PyYAML is required to edit Hermes' config.yaml but is not "
            "importable. Install it (pip install pyyaml) and re-run.",
        )

    if target.exists():
        try:
            loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            return HookOutcome("hermes", label, "error", f"{target} is not valid YAML: {exc!r}")
        config = loaded if isinstance(loaded, dict) else {}
        if loaded is not None and not isinstance(loaded, dict):
            return HookOutcome("hermes", label, "error", f"top-level of {target} is not a mapping; cannot merge")
    else:
        config = {}

    existing_hooks = config.get("hooks")
    if existing_hooks is not None and not isinstance(existing_hooks, dict):
        return HookOutcome("hermes", label, "error", f"'hooks' in {target} is not a mapping; cannot merge")

    merged_hooks, changed = merge_hermes_hooks(existing_hooks, render_hermes_hooks(python))
    if not changed and not force:
        return HookOutcome("hermes", label, "unchanged", f"{label} already current")
    if dry_run:
        return HookOutcome("hermes", label, "would_install", f"would register Hermes pre_llm_call + pre_tool_call hooks in {label}")

    config["hooks"] = merged_hooks
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(config, sort_keys=False, allow_unicode=True, default_flow_style=False, width=4096)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(rendered, encoding="utf-8")
    tmp.replace(target)
    return HookOutcome(
        "hermes", label, "installed",
        f"wrote Hermes pre_llm_call (prompt-inject) + pre_tool_call (deny-redirect) hooks to {label}",
    )


def uninstall_hermes_hooks(
    *,
    target_path: Optional[Path] = None,
    dry_run: bool = False,
    env: Optional[Mapping[str, str]] = None,
) -> HookOutcome:
    """Remove Puppetmaster auto-invocation hooks from Hermes' ``config.yaml``."""
    from puppetmaster.installers import hermes_config_path

    target = target_path or hermes_config_path(env)
    label = str(target)
    if not target.is_file():
        return HookOutcome("hermes", label, "unchanged", f"no {label}")
    try:
        import yaml  # type: ignore
    except Exception:
        return HookOutcome("hermes", label, "error", "PyYAML required to edit Hermes config.yaml; not importable")
    try:
        loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return HookOutcome("hermes", label, "error", f"{target} is not valid YAML: {exc!r}")
    config = loaded if isinstance(loaded, dict) else {}
    existing_hooks = config.get("hooks")
    if not isinstance(existing_hooks, dict):
        return HookOutcome("hermes", label, "unchanged", f"{label} has no Puppetmaster hooks")

    stripped, changed = strip_hermes_hooks(existing_hooks)
    if not changed:
        return HookOutcome("hermes", label, "unchanged", f"{label} has no Puppetmaster hooks")
    if dry_run:
        return HookOutcome("hermes", label, "would_remove", f"would remove Puppetmaster hooks from {label}")

    # Preserve Hermes' default empty ``hooks: {}`` so the key shape is unchanged.
    config["hooks"] = stripped
    rendered = yaml.safe_dump(config, sort_keys=False, allow_unicode=True, default_flow_style=False, width=4096)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(rendered, encoding="utf-8")
    tmp.replace(target)
    return HookOutcome("hermes", label, "removed", f"removed Puppetmaster hooks from {label}")


def install_hooks(
    *,
    cwd: Optional[Path] = None,
    targets: Optional[Iterable[str]] = None,
    dry_run: bool = False,
    force: bool = False,
    python: Optional[str] = None,
    scope: str = "project",
    home: Optional[Path] = None,
    enabled_adapters: Optional[set[str]] = None,
) -> HooksInstallResult:
    """Install Puppetmaster auto-invocation hooks for ``targets``.

    ``scope`` selects where the hooks land:

    * ``"project"`` (default) — workspace-local under ``cwd``; covers this repo
      only and can be checked in.
    * ``"global"`` — user-level under ``home`` (``~`` by default); covers every
      repo the user opens without re-running setup per project.

    Only the base directory differs between scopes — the ``.cursor`` / ``.claude``
    subpaths are identical. Defaults to both Cursor and Claude Code; pass
    ``targets`` to restrict. ``home`` is an injectable override of ``~`` for tests.

    ``enabled_adapters`` (when provided) filters the *default* target set to the
    platforms the user actually routes to — e.g. a Cursor-only user shouldn't get
    a ``.claude/settings.json`` hook written. ``None`` (the default) means "don't
    filter", preserving standalone ``install-hooks`` behavior. Ignored when
    ``targets`` is given (explicit intent always wins).
    """
    result = HooksInstallResult()
    if scope not in VALID_HOOK_SCOPES:
        result.outcomes.append(HookOutcome("", "", "error", f"unknown hook scope: {scope!r}"))
        return result
    base = (home or Path.home()) if scope == "global" else (cwd or Path.cwd())
    if targets:
        selected = list(targets)
    else:
        selected = [
            target
            for target in ("cursor", "claude")
            if enabled_adapters is None
            or _HOOK_TARGET_ADAPTERS[target] in enabled_adapters
        ]
        if not selected:
            result.outcomes.append(
                HookOutcome(
                    "",
                    "",
                    "unchanged",
                    "no enabled platform needs host hooks (cursor + claude-code both "
                    "disabled by the platform lock)",
                )
            )
            return result
    for target in selected:
        if target == "cursor":
            result.outcomes.append(_install_cursor(base, scope=scope, dry_run=dry_run, force=force, python=python))
        elif target == "claude":
            result.outcomes.append(_install_claude(base, scope=scope, dry_run=dry_run, force=force, python=python))
        else:
            result.outcomes.append(HookOutcome(target, "", "error", f"unknown hook target: {target!r}"))
    return result


def _uninstall_cursor(
    base_dir: Path,
    *,
    scope: str,
    dry_run: bool,
    python: Optional[str],
) -> HookOutcome:
    path = base_dir / ".cursor" / "hooks.json"
    label = _scoped_label(scope, ".cursor/hooks.json")
    if not path.is_file():
        return HookOutcome("cursor", str(path), "unchanged", f"no {label}")
    existing = _read_json(path)
    stripped, changed = strip_hook_maps(existing)
    if not changed:
        return HookOutcome("cursor", str(path), "unchanged", f"{label} has no Puppetmaster hooks")
    if dry_run:
        if _hooks_file_is_empty(stripped):
            return HookOutcome(
                "cursor",
                str(path),
                "would_remove",
                f"would remove Puppetmaster hooks and delete {label}",
            )
        return HookOutcome(
            "cursor",
            str(path),
            "would_remove",
            f"would remove Puppetmaster hooks from {label}",
        )
    if _hooks_file_is_empty(stripped):
        path.unlink(missing_ok=True)
        return HookOutcome("cursor", str(path), "removed", f"removed Puppetmaster hooks and deleted {label}")
    _write_atomic(path, json.dumps(stripped, indent=2) + "\n")
    return HookOutcome("cursor", str(path), "removed", f"removed Puppetmaster hooks from {label}")


def _uninstall_claude(
    base_dir: Path,
    *,
    scope: str,
    dry_run: bool,
    python: Optional[str],
) -> HookOutcome:
    path = base_dir / ".claude" / "settings.json"
    label = _scoped_label(scope, ".claude/settings.json")
    if not path.is_file():
        return HookOutcome("claude", str(path), "unchanged", f"no {label}")
    existing = _read_json(path)
    stripped, changed = strip_hook_maps(existing)
    if not changed:
        return HookOutcome("claude", str(path), "unchanged", f"{label} has no Puppetmaster hooks")
    if dry_run:
        if _hooks_file_is_empty(stripped):
            return HookOutcome(
                "claude",
                str(path),
                "would_remove",
                f"would remove Puppetmaster hooks and delete {label}",
            )
        return HookOutcome(
            "claude",
            str(path),
            "would_remove",
            f"would remove Puppetmaster hooks from {label}",
        )
    if _hooks_file_is_empty(stripped):
        path.unlink(missing_ok=True)
        return HookOutcome("claude", str(path), "removed", f"removed Puppetmaster hooks and deleted {label}")
    _write_atomic(path, json.dumps(stripped, indent=2) + "\n")
    return HookOutcome("claude", str(path), "removed", f"removed Puppetmaster hooks from {label}")


def uninstall_hooks(
    *,
    cwd: Optional[Path] = None,
    targets: Optional[Iterable[str]] = None,
    dry_run: bool = False,
    python: Optional[str] = None,
    scopes: Optional[Iterable[str]] = None,
    home: Optional[Path] = None,
) -> HooksInstallResult:
    """Remove Puppetmaster auto-invocation hooks for ``targets``.

    By default removes hooks at both ``project`` and ``global`` scopes so
    a prior ``setup`` or ``install-hooks --global`` is fully reversed.
    """
    result = HooksInstallResult()
    selected_scopes = list(scopes) if scopes is not None else ["project", "global"]
    selected = list(targets) if targets else ["cursor", "claude"]
    for scope in selected_scopes:
        if scope not in VALID_HOOK_SCOPES:
            result.outcomes.append(HookOutcome("", "", "error", f"unknown hook scope: {scope!r}"))
            continue
        base = (home or Path.home()) if scope == "global" else (cwd or Path.cwd())
        for target in selected:
            if target == "cursor":
                result.outcomes.append(
                    _uninstall_cursor(base, scope=scope, dry_run=dry_run, python=python)
                )
            elif target == "claude":
                result.outcomes.append(
                    _uninstall_claude(base, scope=scope, dry_run=dry_run, python=python)
                )
            else:
                result.outcomes.append(
                    HookOutcome(target, "", "error", f"unknown hook target: {target!r}")
                )
    return result
