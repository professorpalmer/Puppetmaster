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
- ``$CODEX_HOME/AGENTS.md`` (Codex user-level guidance, global scope;
  default ``~/.codex``. Codex ignores ``instructions.md``)
- ``~/.claude/CLAUDE.md`` (Claude Code user-level instructions, global
  scope)
- ``~/.hermes/SOUL.md`` (NousResearch Hermes global system-prompt file,
  injected into every Hermes session — global scope; honors ``$HERMES_HOME``)

For the multi-line markdown targets (``AGENTS.md``, ``CLAUDE.md``,
``SOUL.md``), the writer uses an HTML-comment-delimited block
so re-running ``install-rules`` replaces only the Puppetmaster block
and leaves any other content in the file untouched. The user can
disable the rule by deleting the marked block; we never overwrite
content outside it.

For Cursor ``.mdc`` files, the file is owned wholesale by Puppetmaster
(rule files in ``.cursor/rules/`` are atomic — one rule per file by
convention) so we simply write the file.
"""

from __future__ import annotations

import os
import shutil
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional


BEGIN_MARKER = "<!-- puppetmaster:rules:begin -->"
END_MARKER = "<!-- puppetmaster:rules:end -->"


RULE_BODY = textwrap.dedent(
    """\
    # Puppetmaster orchestration

    Puppetmaster runs durable worker jobs and flow graphs for you through the
    `puppetmaster_*` MCP tools: workers that survive restarts, per-item check
    and repair, cheap worker models under an expensive pilot, and follow-ups
    that resume a worker's own session.

    ## Are you a Puppetmaster worker? (check this first)

    **If `PUPPETMASTER_WORKER` is `1`, or Puppetmaster issued your prompt,
    every delegation rule below is void for you.** You *are* the worker it
    delegated to. Do the analysis or the edit yourself and return the
    artifacts your prompt asks for.

    You are a Puppetmaster worker if `PUPPETMASTER_WORKER=1` in the
    environment, or your prompt contains a `Puppetmaster artifact contract:`
    block, a `Role: <role>` + `Goal: <goal>` header, or an instruction to
    finish by calling `submit_findings` / `submit_report`. Nested job starts
    are refused while that env is set (override:
    `PUPPETMASTER_ALLOW_NESTED=1`). Workers run as plain agent CLIs with **no
    `puppetmaster_*` MCP tools**, so delegating is impossible. Use your own
    native tools.

    ## Trigger convention (must obey)

    When the user says **"Use Puppetmaster to ..."**, **"PM this ..."**, or
    otherwise names Puppetmaster for a task, route that work through the
    `puppetmaster_*` MCP tools rather than answering inline.

    ## Solo first; fan out when it pays

    Do the work yourself unless parallel workers clearly finish sooner or
    better. One session that holds the whole problem beats any fan-out on
    small work: every worker pays a fixed start (its own context, reading,
    checks) and you pay to integrate. Fan out when the work splits into
    independent units that each take a worker minutes, or when there is more
    of it than you can finish well in one session (you are running long, or
    your context is filling with work that does not depend on itself). Many
    small units you could write in a few minutes are still solo work.

    - Exact edits, typos, small follow-ups and revisions: make them yourself.
      If your instruction to a worker would spell out the change, it is
      cheaper to make it.
    - A revision that needs a prior worker's context: continue its flow run
      with `continue_from`, or resume the worker with `resume_from`, instead
      of starting fresh sessions.
    - Marionette decides this for you from your plan and measured pace; on
      other hosts, `puppetmaster sizing` gives the same decision.

    ## Fan out with one flow

    Write ONE flow graph and start it with `puppetmaster_flow` (action
    `run`), then call action `wait` (or end your turn). Puppetmaster walks it
    durably and wakes you only when it is done, failed, stuck, interrupted or
    waiting at a gate; do not launch, poll and hand off each step yourself.
    The tool description has the node shapes and a fan-out example.

    - **Size workers to the work.** Group many small units into a few `map`
      items (each an object with its unit names and files; 16 small modules:
      3-5 workers) and set concurrency to the number of items so they all
      run at once. Give a unit its own worker only when it is minutes of work.
    - **Check each unit.** Put the unit's own check (`shell`) after its build
      with a `fail` edge back to the build (`max` 2), inside the map item, so
      each unit is repaired alone. A unit's `files` include what its own check
      generates (its render or report dir as a glob); shared assembly outputs
      belong to one integrate agent after the map.
    - **Armor for craft.** When the result is judged by how it looks, reads
      or feels (visuals, geometry, UI, prose) and not only by a test, add a
      `shell` step that produces the observable result (render, run,
      screenshot) and a `judge` whose task is a numbered rubric from the
      user's request, with its FAIL edge back to the build (`max` 2).
      Passing tests is not the bar a user grades.
    - Leave the model unpinned unless the user named one: workers then run
      the model you are configured with.

    ## Label every job you start (do it by default)

    When you start any job verb (`puppetmaster_start_*`, `puppetmaster_edit`,
    or the matching sync verbs), pass a short human-readable `label` (3-6
    words, e.g. `"auth refactor audit"`). It becomes the job's headline on the
    dashboard and in `puppetmaster_jobs`.

    ## CodeGraph for unfamiliar code

    When you must find where something is, what calls it or what a change
    affects in code you have not read, ask the graph instead of crawling the
    tree: `puppetmaster_codegraph_status`, then `puppetmaster_codegraph_init`
    (`index: true`) if there is no index, then `puppetmaster_codegraph_search`
    / `_context` / `_affected`, and read only the files it points to. Skip it
    for a small repository you can list at a glance, for files you already
    know, and for plain-text matches (log strings, config values). If a
    codegraph MCP call fails, use `python -m puppetmaster codegraph ...`,
    never a bare `codegraph` from the shell.

    ## Fallback

    If a `puppetmaster_*` observation tool is not connected, continue the same
    durable job through `python -m puppetmaster status|await|show <job_id>`.
    Check recent jobs before retrying a start; a dropped MCP reply is not proof
    that no job was created. Use native tooling only when no Puppetmaster job
    exists and the task itself permits inline work.

    ## Other verbs

    - `puppetmaster_start_swarm` for read-only analysis across several
      lenses; `puppetmaster_edit` for one focused in-place edit that builds on
      uncommitted work; `puppetmaster_start_implement` for one coupled change
      in an isolated worktree. With a provider API key but no vendor CLI
      (keys-only), use `puppetmaster_agentic` / `puppetmaster_start_agentic`.
    - Every asynchronous `start_*` response is a resumable contract: follow
      its returned `monitor_with` tool with the exact `job_ref`. Treat only
      `delivery.verdict == "delivered"` as success.
    - `puppetmaster_artifacts <job_id>` reads stored results at zero token
      cost; `puppetmaster_route_task` previews the routed model and price when
      spend matters; `puppetmaster_dashboard` opens the job dashboard when the
      user asks.
    - `puppetmaster_doctor` when a Puppetmaster call fails or behaves
      unexpectedly; surface critical failures to the user. Not a ritual.
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
        "Work solo until parallel workers clearly pay, then fan out with one "
        "Puppetmaster flow; obey 'Use Puppetmaster to …' triggers."
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


def _detect_hermes_cli() -> bool:
    """Return True if NousResearch Hermes looks present on this machine.

    Mirrors :func:`puppetmaster.diagnostics._hermes_cli_installed`: a `hermes`
    executable on PATH (honoring the ``HERMES_COMMAND`` override) or a
    ``~/.hermes`` (or ``$HERMES_HOME``) directory left by a prior run.
    """
    command = os.environ.get("HERMES_COMMAND", "hermes")
    parts = command.split()
    first = parts[0] if parts else ""
    if first and (Path(first).expanduser().exists() or shutil.which(first) is not None):
        return True
    return hermes_soul_path().parent.exists()


def hermes_soul_path(env: Optional[Mapping[str, str]] = None) -> Path:
    """Return the path to Hermes' global ``SOUL.md``.

    ``SOUL.md`` is the file Hermes injects into *every* session's system
    prompt, so it is the correct global-bias surface for Hermes — the
    counterpart to ``~/.claude/CLAUDE.md`` / ``$CODEX_HOME/AGENTS.md`` for
    the other hosts. Honors ``$HERMES_HOME`` (the same override Hermes and the
    MCP installer read) and falls back to ``~/.hermes/SOUL.md``.
    """
    env = env if env is not None else os.environ
    hermes_home = env.get("HERMES_HOME")
    base = Path(hermes_home).expanduser() if hermes_home else Path("~/.hermes").expanduser()
    return base / "SOUL.md"


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


def codex_global_rules_path() -> Path:
    """Codex's user-level guidance file: ``$CODEX_HOME/AGENTS.md``.

    Codex 0.160 loads AGENTS.md from CODEX_HOME and ignores instructions.md;
    rules written there never reached a Codex pilot (verified with a marker
    word: AGENTS.md answered, instructions.md did not).
    """
    home = os.environ.get("CODEX_HOME")
    return (Path(home).expanduser() if home else Path.home() / ".codex") / "AGENTS.md"


def _legacy_codex_rules_path() -> Path:
    home = os.environ.get("CODEX_HOME")
    return (Path(home).expanduser() if home else Path.home() / ".codex") / "instructions.md"


def _install_codex_global(*, dry_run: bool, force: bool) -> TargetOutcome:
    target_path = codex_global_rules_path()
    new_block = render_agents_block()
    existing = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
    merged, action = merge_block_into_text(existing, new_block)
    legacy = _legacy_codex_rules_path()
    legacy_text = legacy.read_text(encoding="utf-8") if legacy.exists() else ""
    stripped_legacy, legacy_action = strip_block_from_text(legacy_text)
    if action == "unchanged" and legacy_action == "unchanged" and not force:
        return TargetOutcome(
            target="codex_global",
            path=str(target_path),
            status="unchanged",
            reason=f"{target_path} already has an up-to-date block",
        )
    if dry_run:
        return TargetOutcome(
            target="codex_global",
            path=str(target_path),
            status="would_install",
            reason=f"would update {target_path} (applies to every codex session)",
        )
    _write_atomic(target_path, merged)
    if legacy_action != "unchanged":
        # Move an old block out of the file Codex never read.
        _write_or_delete_markdown(legacy, stripped_legacy, dry_run=False)
    return TargetOutcome(
        target="codex_global",
        path=str(target_path),
        status="installed",
        reason=f"wrote {target_path} (applies to every codex session)",
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


def _install_hermes_global(*, dry_run: bool, force: bool) -> TargetOutcome:
    target_path = hermes_soul_path()
    new_block = render_agents_block()
    existing = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
    merged, action = merge_block_into_text(existing, new_block)
    if action == "unchanged" and not force:
        return TargetOutcome(
            target="hermes_global",
            path=str(target_path),
            status="unchanged",
            reason="Hermes SOUL.md already has an up-to-date block",
        )
    if dry_run:
        return TargetOutcome(
            target="hermes_global",
            path=str(target_path),
            status="would_install",
            reason=f"would update {target_path} (applies to every Hermes session)",
        )
    if force and action == "unchanged":
        merged = (existing.replace(new_block, "") + "\n" + new_block).strip() + "\n"
    _write_atomic(target_path, merged)
    return TargetOutcome(
        target="hermes_global",
        path=str(target_path),
        status="installed",
        reason="wrote Hermes SOUL.md block (applies to every Hermes session)",
    )


def install_rules(
    *,
    cwd: Optional[Path] = None,
    targets: Optional[Iterable[str]] = None,
    install_global: bool = False,
    dry_run: bool = False,
    force: bool = False,
    enabled_adapters: Optional[set[str]] = None,
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
    - ``hermes_global`` is included only with ``--global`` AND when Hermes
      is detected (``hermes`` on PATH, ``$HERMES_COMMAND``, or ``~/.hermes/``
      present). Writes the managed block into Hermes' global ``SOUL.md`` —
      the only host-global surface Hermes injects into every session.

    ``enabled_adapters`` (when provided) further filters the auto-detected
    set to the platforms the user actually routes to — e.g. a Claude-Code-only
    user shouldn't get a ``.cursor/rules/`` file written just because the cwd
    is a git repo. ``None`` (the default) means "don't filter", preserving
    standalone ``install-rules`` behavior. Ignored when ``targets`` is given
    (explicit intent always wins). The portable ``agents`` target is never
    filtered: ``AGENTS.md`` is cross-tool and harmless.

    Pass an explicit ``targets`` iterable to override detection.
    """
    cwd = cwd or Path.cwd()
    result = RulesInstallResult()

    def _adapter_enabled(adapter: str) -> bool:
        return enabled_adapters is None or adapter in enabled_adapters

    detected: list[str] = []
    if targets is None:
        if _detect_cursor(cwd) and _adapter_enabled("cursor"):
            detected.append("cursor")
        detected.append("agents")
        if install_global:
            if _detect_codex_cli() and _adapter_enabled("codex"):
                detected.append("codex_global")
            elif _detect_codex_cli():
                result.messages.append(
                    "codex detected but disabled by the platform lock — "
                    "skipping $CODEX_HOME/AGENTS.md"
                )
            else:
                result.messages.append(
                    "codex CLI not detected — skipping $CODEX_HOME/AGENTS.md "
                    "(install `codex` and re-run with --global to enable)"
                )
            if _detect_claude_cli() and _adapter_enabled("claude-code"):
                detected.append("claude_global")
            elif _detect_claude_cli():
                result.messages.append(
                    "claude detected but disabled by the platform lock — "
                    "skipping ~/.claude/CLAUDE.md"
                )
            else:
                result.messages.append(
                    "claude CLI not detected — skipping ~/.claude/CLAUDE.md "
                    "(install `claude` and re-run with --global to enable)"
                )
            if _detect_hermes_cli() and _adapter_enabled("hermes"):
                detected.append("hermes_global")
            elif _detect_hermes_cli():
                result.messages.append(
                    "Hermes detected but disabled by the platform lock — "
                    "skipping Hermes SOUL.md"
                )
            else:
                result.messages.append(
                    "Hermes not detected — skipping SOUL.md "
                    "(install Hermes and re-run with --global to enable)"
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
        elif target == "hermes_global":
            result.outcomes.append(_install_hermes_global(dry_run=dry_run, force=force))
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


VALID_TARGETS = {"cursor", "agents", "codex_global", "claude_global", "hermes_global"}
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
        "hermes_global",
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
                    codex_global_rules_path(),
                    target="codex_global",
                    dry_run=dry_run,
                    label=str(codex_global_rules_path()),
                )
            )
        elif target == "hermes_global":
            soul = hermes_soul_path()
            result.outcomes.append(
                _uninstall_markdown_block_file(
                    soul,
                    target="hermes_global",
                    dry_run=dry_run,
                    label=str(soul),
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
