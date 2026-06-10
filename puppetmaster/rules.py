"""Cross-tool agent rule installer.

The MCP installers in :mod:`puppetmaster.installers` give Cursor / Codex
/ Claude Code the *capability* to call Puppetmaster's tools, but they
do not tell the host agent *when* to reach for them. Without that
nudge, a Cursor or Codex session will tend to do multi-file audits
itself — slowly, expensively, single-threaded — instead of fanning out
to a Puppetmaster swarm.

This module fixes that by writing short "use Puppetmaster on these
patterns" rule files into the conventions each host respects:

- ``.cursor/rules/puppetmaster.mdc`` (Cursor workspace rules with
  ``alwaysApply: true``)
- ``AGENTS.md`` (the cross-tool convention at https://agents.md/ now
  respected by Codex, Claude Code, and several other agents — workspace
  scope only)
- ``~/.codex/instructions.md`` (Codex user-level instructions, global
  scope)
- ``~/.claude/CLAUDE.md`` (Claude Code user-level instructions, global
  scope)

For the multi-line markdown targets (``AGENTS.md``, ``CLAUDE.md``,
``instructions.md``), the writer uses an HTML-comment-delimited block
so re-running ``install-rules`` replaces only the Puppetmaster block
and leaves any other content in the file untouched. The user can
disable the rule by deleting the marked block; we never overwrite
content outside it.

For Cursor ``.mdc`` files, the file is owned wholesale by Puppetmaster
(rule files in ``.cursor/rules/`` are atomic — one rule per file by
convention) so we simply write the file.
"""

from __future__ import annotations

import shutil
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


BEGIN_MARKER = "<!-- puppetmaster:rules:begin -->"
END_MARKER = "<!-- puppetmaster:rules:end -->"


RULE_BODY = textwrap.dedent(
    """\
    # Puppetmaster orchestration

    Puppetmaster is an MCP-based agent orchestrator with structured worker
    swarms, durable SQLite state, tiered model routing, and zero-token
    follow-ups via stored artifacts. When Puppetmaster's MCP server is
    registered (`puppetmaster install-cursor-mcp` or
    `puppetmaster install-codex-mcp`), the `puppetmaster_*` MCP tools are
    available in this environment.

    ## Trigger convention (must obey)

    When the user says **"Use Puppetmaster to …"**, **"PM this …"**, or
    otherwise names Puppetmaster for a task, route that work through the
    `puppetmaster_*` MCP tools — do not answer inline.

    ## Delegate-first gate (default path)

    Before attempting multi-step work inline, start a Puppetmaster verb
    (`puppetmaster_start_cursor_swarm`, `puppetmaster_start_swarm`,
    `puppetmaster_start_implement`, or the matching sync verbs) when the
    task is any of:

    - Multi-file (3+ files) or cross-cutting refactor/migration
    - An audit, review, or "find all X" search
    - Work whose result will be reused later in this or a future session

    Swarms and reviews run read-only analysis; building goes through
    implement. Recall prior results with `puppetmaster_artifacts <job_id>`
    at zero token cost.

    Reach for a Puppetmaster verb **before** native broad search/exploration:
    prefer `puppetmaster_codegraph_search` / `_context` over a repo-wide
    `Grep`/`Glob`/`find`, and a swarm over the built-in `Task` tool, for any
    multi-file investigation. When unsure whether a task qualifies, run the
    classifier-backed gate — `puppetmaster_route_task` (or
    `puppetmaster should-delegate "<prompt>"`) — which returns a delegate /
    inline verdict and a suggested verb with zero LLM cost.

    For deterministic enforcement, the user can install host hooks
    (`puppetmaster install-hooks`) that inject this directive on prompt submit
    and deny-redirect broad native exploration automatically. The kill switch
    is `PUPPETMASTER_AUTO_INVOKE_DISABLED=1`.

    ## When NOT to use Puppetmaster (stay inline)

    - Trivial single-file edits, typos, one-line fixes
    - Quick factual questions
    - Fast interactive iteration where the user is steering turn-by-turn

    Routing those through Puppetmaster wastes tokens and latency.

    ## Fallback

    If `puppetmaster_*` tools are not connected, fall back to native
    tooling — do not pretend the tools exist.

    ## Usage

    1. `puppetmaster_route_task <prompt> --role <role>` — dry-run that
       returns the chosen model, estimated cost, and reasoning. Use
       whenever spend matters or the task is ambiguous.
    2. `puppetmaster_start_cursor_swarm` / `puppetmaster_start_swarm` for
       read-only analysis; `puppetmaster_start_implement` /
       `puppetmaster_start_claude_implement` / `puppetmaster_start_codex`
       for full-edit builds.
    3. `puppetmaster_artifacts <job_id>` — read structured outputs at zero
       token cost (results persist in SQLite).
    4. `puppetmaster_dashboard [job_id]` — when the user asks to see/open
       the job dashboard, call this (it starts the local server if needed)
       and open the returned URL in a browser tab for them. CLI fallback:
       `python -m puppetmaster dashboard [job_id]`.
    5. `puppetmaster_doctor` — sanity-check Puppetmaster's runtime
       dependencies once per session.

    If `puppetmaster_doctor` reports critical failures, surface them to
    the user before continuing.
    """
)


@dataclass
class TargetOutcome:
    """One install-rules action's result.

    ``target`` is the symbolic name of the rule destination
    (``"cursor"``, ``"agents"``, ``"claude_global"``,
    ``"codex_global"``, etc.). ``path`` is the absolute file written
    or that would have been written. ``status`` matches the installer
    contract: ``installed``, ``unchanged``, ``would_install``, ``skipped``,
    or ``error``. ``reason`` is a one-line explanation surfaced to the
    user.
    """

    target: str
    path: str
    status: str
    reason: str = ""


@dataclass
class RulesInstallResult:
    """Aggregate result for an :func:`install_rules` run."""

    outcomes: list[TargetOutcome] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    @property
    def overall_status(self) -> str:
        """Return ``"error"`` if any target errored, else best-effort summary."""
        statuses = [o.status for o in self.outcomes]
        if any(s == "error" for s in statuses):
            return "error"
        if all(s == "skipped" for s in statuses):
            return "skipped"
        if all(s == "unchanged" for s in statuses):
            return "unchanged"
        if any(s == "would_install" for s in statuses):
            return "would_install"
        if any(s == "would_remove" for s in statuses):
            return "would_remove"
        if any(s == "removed" for s in statuses):
            return "removed"
        return "installed"


def render_cursor_mdc() -> str:
    """Return the Cursor ``.mdc`` rule content (YAML frontmatter + body).

    Cursor's rule format requires a frontmatter block with at least
    ``description`` and ``alwaysApply``. ``alwaysApply: true`` means the
    rule fires on every agent turn in the workspace, which is what we
    want — the rule should bias the agent's tool-selection on every
    user message, not just when a glob matches.
    """
    description = (
        "Delegate multi-file refactors, audits, and reusable work to "
        "Puppetmaster MCP swarms; obey 'Use Puppetmaster to …' triggers."
    )
    frontmatter = (
        "---\n"
        f"description: {description}\n"
        "alwaysApply: true\n"
        "---\n\n"
    )
    return frontmatter + RULE_BODY


def render_agents_block() -> str:
    """Wrap :data:`RULE_BODY` in the begin/end markers for merge targets."""
    return (
        f"{BEGIN_MARKER}\n"
        "<!-- managed by `puppetmaster install-rules`; delete this whole "
        "block to disable -->\n\n"
        f"{RULE_BODY.rstrip()}\n\n"
        f"{END_MARKER}\n"
    )


def merge_block_into_text(existing: str, new_block: str) -> tuple[str, str]:
    """Insert or replace the Puppetmaster block in ``existing``.

    Returns ``(merged_text, action)`` where ``action`` is one of
    ``"created"`` (no markers found, block appended), ``"replaced"``
    (markers found, content between them swapped), or ``"unchanged"``
    (existing markers wrap content byte-identical to ``new_block``).

    The merge protocol is deliberately literal: we look for the exact
    ``BEGIN_MARKER`` and ``END_MARKER`` strings on their own lines. If
    the user hand-edited inside the block, those edits get overwritten
    on the next ``install-rules`` run — which is correct behavior; the
    block is owned by Puppetmaster. To customize, the user deletes the
    block (we'll re-create it next run) or deletes one of the markers
    (we'll see no marker pair and append a fresh block, leaving the
    hand-edited version alone). The latter is an honest escape hatch.
    """
    begin_idx = existing.find(BEGIN_MARKER)
    end_idx = existing.find(END_MARKER)
    if begin_idx == -1 or end_idx == -1 or end_idx < begin_idx:
        separator = "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
        if not existing:
            return new_block, "created"
        return existing + separator + new_block, "created"
    end_line_end = existing.find("\n", end_idx)
    if end_line_end == -1:
        end_line_end = len(existing)
    else:
        end_line_end += 1
    before = existing[:begin_idx]
    after = existing[end_line_end:]
    if not before.endswith("\n") and before:
        before = before + "\n"
    candidate = before + new_block + (after.lstrip("\n") if after else "")
    if candidate == existing:
        return existing, "unchanged"
    return candidate, "replaced"


def strip_block_from_text(existing: str) -> tuple[str, str]:
    """Remove the Puppetmaster block from ``existing`` if present.

    Returns ``(stripped_text, action)`` where ``action`` is ``"removed"``
    when the block was stripped, or ``"unchanged"`` when no markers were
    found. Surrounding content outside the markers is preserved byte-for-byte.
    """
    begin_idx = existing.find(BEGIN_MARKER)
    end_idx = existing.find(END_MARKER)
    if begin_idx == -1 or end_idx == -1 or end_idx < begin_idx:
        return existing, "unchanged"
    end_line_end = existing.find("\n", end_idx)
    if end_line_end == -1:
        end_line_end = len(existing)
    else:
        end_line_end += 1
    stripped = existing[:begin_idx] + existing[end_line_end:]
    if stripped == existing:
        return existing, "unchanged"
    return stripped, "removed"


def _text_is_empty(content: str) -> bool:
    return not content or not content.strip()


def _write_or_delete_markdown(path: Path, content: str, *, dry_run: bool) -> str:
    """Write ``content`` to ``path``, deleting the file when whitespace-only."""
    if _text_is_empty(content):
        if dry_run:
            return "would_delete"
        if path.is_file():
            path.unlink()
        return "deleted"
    if dry_run:
        return "would_write"
    _write_atomic(path, content)
    return "written"


def _detect_cursor(cwd: Path) -> bool:
    """Return True if a Cursor workspace rule directory makes sense here.

    We consider Cursor "present in this workspace" if either (a) the
    ``.cursor/`` directory exists already, or (b) the parent is a git
    repository (most Puppetmaster users running ``install-rules`` will
    be inside a project repo and intend the rule to be checked in).
    """
    if (cwd / ".cursor").exists():
        return True
    if (cwd / ".git").exists():
        return True
    return False


def _detect_codex_cli() -> bool:
    return shutil.which("codex") is not None or (Path.home() / ".codex").exists()


def _detect_claude_cli() -> bool:
    return shutil.which("claude") is not None or (Path.home() / ".claude").exists()


def _write_atomic(path: Path, content: str) -> None:
    """Write to a temp sibling and rename, so a partial write never leaves a corrupt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _install_cursor_workspace(
    cwd: Path, *, dry_run: bool, force: bool
) -> TargetOutcome:
    target_path = cwd / ".cursor" / "rules" / "puppetmaster.mdc"
    desired = render_cursor_mdc()
    if target_path.exists():
        existing = target_path.read_text(encoding="utf-8")
        if existing == desired and not force:
            return TargetOutcome(
                target="cursor",
                path=str(target_path),
                status="unchanged",
                reason=".cursor/rules/puppetmaster.mdc already up to date",
            )
    if dry_run:
        return TargetOutcome(
            target="cursor",
            path=str(target_path),
            status="would_install",
            reason="would write .cursor/rules/puppetmaster.mdc with alwaysApply: true",
        )
    _write_atomic(target_path, desired)
    return TargetOutcome(
        target="cursor",
        path=str(target_path),
        status="installed",
        reason="wrote .cursor/rules/puppetmaster.mdc (alwaysApply: true)",
    )


def _install_agents_md_workspace(
    cwd: Path, *, dry_run: bool, force: bool
) -> TargetOutcome:
    target_path = cwd / "AGENTS.md"
    new_block = render_agents_block()
    existing = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
    merged, action = merge_block_into_text(existing, new_block)
    if action == "unchanged" and not force:
        return TargetOutcome(
            target="agents",
            path=str(target_path),
            status="unchanged",
            reason="AGENTS.md already has an up-to-date Puppetmaster block",
        )
    if dry_run:
        verb = "create" if action == "created" else "replace Puppetmaster block in"
        return TargetOutcome(
            target="agents",
            path=str(target_path),
            status="would_install",
            reason=f"would {verb} {target_path.name} (cross-tool: Codex + Claude Code + others honor AGENTS.md)",
        )
    if force and action == "unchanged":
        merged = (existing.replace(new_block, "") + "\n" + new_block).strip() + "\n"
    _write_atomic(target_path, merged)
    verb = "created" if action == "created" else "updated"
    return TargetOutcome(
        target="agents",
        path=str(target_path),
        status="installed",
        reason=f"{verb} AGENTS.md (cross-tool nudge; Codex + Claude Code both read this)",
    )


def _install_codex_global(*, dry_run: bool, force: bool) -> TargetOutcome:
    target_path = Path.home() / ".codex" / "instructions.md"
    new_block = render_agents_block()
    existing = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
    merged, action = merge_block_into_text(existing, new_block)
    if action == "unchanged" and not force:
        return TargetOutcome(
            target="codex_global",
            path=str(target_path),
            status="unchanged",
            reason="~/.codex/instructions.md already has an up-to-date block",
        )
    if dry_run:
        return TargetOutcome(
            target="codex_global",
            path=str(target_path),
            status="would_install",
            reason=f"would update {target_path} (applies to every codex session)",
        )
    _write_atomic(target_path, merged)
    return TargetOutcome(
        target="codex_global",
        path=str(target_path),
        status="installed",
        reason="wrote ~/.codex/instructions.md (applies to every codex session)",
    )


def _install_claude_global(*, dry_run: bool, force: bool) -> TargetOutcome:
    target_path = Path.home() / ".claude" / "CLAUDE.md"
    new_block = render_agents_block()
    existing = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
    merged, action = merge_block_into_text(existing, new_block)
    if action == "unchanged" and not force:
        return TargetOutcome(
            target="claude_global",
            path=str(target_path),
            status="unchanged",
            reason="~/.claude/CLAUDE.md already has an up-to-date block",
        )
    if dry_run:
        return TargetOutcome(
            target="claude_global",
            path=str(target_path),
            status="would_install",
            reason=f"would update {target_path} (applies to every claude session)",
        )
    _write_atomic(target_path, merged)
    return TargetOutcome(
        target="claude_global",
        path=str(target_path),
        status="installed",
        reason="wrote ~/.claude/CLAUDE.md (applies to every claude session)",
    )


def install_rules(
    *,
    cwd: Optional[Path] = None,
    targets: Optional[Iterable[str]] = None,
    install_global: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> RulesInstallResult:
    """Detect host tools and install Puppetmaster rule files.

    Auto-detection rules (when ``targets`` is None):

    - Workspace ``cursor`` is included if ``.cursor/`` exists or the cwd
      is inside a git repo (heuristic: a Puppetmaster user running this
      command inside a project repo intends the rule to be checked in).
    - Workspace ``agents`` is always included — ``AGENTS.md`` is the
      portable convention and adding the block is harmless even if the
      user isn't currently using Codex or Claude Code; if they install
      one later, the rule is already there.
    - ``codex_global`` is included only with ``--global`` AND when codex
      is detected (``codex`` on PATH or ``~/.codex/`` present).
    - ``claude_global`` is included only with ``--global`` AND when
      claude is detected (``claude`` on PATH or ``~/.claude/`` present).

    Pass an explicit ``targets`` iterable to override detection.
    """
    cwd = cwd or Path.cwd()
    result = RulesInstallResult()

    detected: list[str] = []
    if targets is None:
        if _detect_cursor(cwd):
            detected.append("cursor")
        detected.append("agents")
        if install_global:
            if _detect_codex_cli():
                detected.append("codex_global")
            else:
                result.messages.append(
                    "codex CLI not detected — skipping ~/.codex/instructions.md "
                    "(install `codex` and re-run with --global to enable)"
                )
            if _detect_claude_cli():
                detected.append("claude_global")
            else:
                result.messages.append(
                    "claude CLI not detected — skipping ~/.claude/CLAUDE.md "
                    "(install `claude` and re-run with --global to enable)"
                )
    else:
        detected = list(targets)

    for target in detected:
        if target == "cursor":
            result.outcomes.append(
                _install_cursor_workspace(cwd, dry_run=dry_run, force=force)
            )
        elif target == "agents":
            result.outcomes.append(
                _install_agents_md_workspace(cwd, dry_run=dry_run, force=force)
            )
        elif target == "codex_global":
            result.outcomes.append(_install_codex_global(dry_run=dry_run, force=force))
        elif target == "claude_global":
            result.outcomes.append(_install_claude_global(dry_run=dry_run, force=force))
        else:
            result.outcomes.append(
                TargetOutcome(
                    target=target,
                    path="",
                    status="error",
                    reason=f"unknown rule target: {target!r}",
                )
            )

    if install_global:
        result.messages.append(
            "Cursor User Rules (global, in-app) cannot be written from outside Cursor; "
            "to install at the Cursor User Rule level, ask the Cursor agent: "
            '"add a Cursor User Rule from puppetmaster install-rules output".'
        )

    return result


VALID_TARGETS = {"cursor", "agents", "codex_global", "claude_global"}
VALID_UNINSTALL_TARGETS = VALID_TARGETS | {"claude_workspace"}


def _uninstall_cursor_workspace(
    cwd: Path, *, dry_run: bool
) -> TargetOutcome:
    target_path = cwd / ".cursor" / "rules" / "puppetmaster.mdc"
    if not target_path.is_file():
        return TargetOutcome(
            target="cursor",
            path=str(target_path),
            status="unchanged",
            reason="no .cursor/rules/puppetmaster.mdc",
        )
    if dry_run:
        return TargetOutcome(
            target="cursor",
            path=str(target_path),
            status="would_remove",
            reason="would delete .cursor/rules/puppetmaster.mdc",
        )
    target_path.unlink()
    return TargetOutcome(
        target="cursor",
        path=str(target_path),
        status="removed",
        reason="deleted .cursor/rules/puppetmaster.mdc",
    )


def _uninstall_markdown_block_file(
    target_path: Path,
    *,
    target: str,
    dry_run: bool,
    label: str,
) -> TargetOutcome:
    if not target_path.is_file():
        existing = ""
    else:
        existing = target_path.read_text(encoding="utf-8")
    stripped, action = strip_block_from_text(existing)
    if action == "unchanged":
        if not target_path.is_file():
            return TargetOutcome(
                target=target,
                path=str(target_path),
                status="unchanged",
                reason=f"no {label}",
            )
        return TargetOutcome(
            target=target,
            path=str(target_path),
            status="unchanged",
            reason=f"{label} has no Puppetmaster block",
        )
    if dry_run:
        if _text_is_empty(stripped):
            return TargetOutcome(
                target=target,
                path=str(target_path),
                status="would_remove",
                reason=f"would strip Puppetmaster block and delete {label}",
            )
        return TargetOutcome(
            target=target,
            path=str(target_path),
            status="would_remove",
            reason=f"would strip Puppetmaster block from {label}",
        )
    write_action = _write_or_delete_markdown(target_path, stripped, dry_run=False)
    if write_action == "deleted":
        return TargetOutcome(
            target=target,
            path=str(target_path),
            status="removed",
            reason=f"stripped Puppetmaster block and deleted {label}",
        )
    return TargetOutcome(
        target=target,
        path=str(target_path),
        status="removed",
        reason=f"stripped Puppetmaster block from {label}",
    )


def uninstall_rules(
    *,
    cwd: Optional[Path] = None,
    targets: Optional[Iterable[str]] = None,
    dry_run: bool = False,
) -> RulesInstallResult:
    """Remove Puppetmaster rule files and marked blocks installed by :func:`install_rules`."""
    cwd = cwd or Path.cwd()
    result = RulesInstallResult()
    selected = list(targets) if targets is not None else [
        "cursor",
        "agents",
        "claude_workspace",
        "claude_global",
        "codex_global",
    ]

    for target in selected:
        if target == "cursor":
            result.outcomes.append(_uninstall_cursor_workspace(cwd, dry_run=dry_run))
        elif target == "agents":
            result.outcomes.append(
                _uninstall_markdown_block_file(
                    cwd / "AGENTS.md",
                    target="agents",
                    dry_run=dry_run,
                    label="AGENTS.md",
                )
            )
        elif target == "claude_workspace":
            result.outcomes.append(
                _uninstall_markdown_block_file(
                    cwd / "CLAUDE.md",
                    target="claude_workspace",
                    dry_run=dry_run,
                    label="CLAUDE.md",
                )
            )
        elif target == "claude_global":
            result.outcomes.append(
                _uninstall_markdown_block_file(
                    Path.home() / ".claude" / "CLAUDE.md",
                    target="claude_global",
                    dry_run=dry_run,
                    label="~/.claude/CLAUDE.md",
                )
            )
        elif target == "codex_global":
            result.outcomes.append(
                _uninstall_markdown_block_file(
                    Path.home() / ".codex" / "instructions.md",
                    target="codex_global",
                    dry_run=dry_run,
                    label="~/.codex/instructions.md",
                )
            )
        else:
            result.outcomes.append(
                TargetOutcome(
                    target=target,
                    path="",
                    status="error",
                    reason=f"unknown rule target: {target!r}",
                )
            )

    return result
