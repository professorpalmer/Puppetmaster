"""Standalone, provider-agnostic agentic worker adapter.

This is Puppetmaster's answer to "run workers on the user's keys, with no
external agent CLI." Where :mod:`hermes`, :mod:`cursor`, :mod:`claude_code`, and
:mod:`codex` shell out to a third-party agent binary, ``AgenticAdapter`` runs
its OWN tool-use loop against a provider HTTP API directly (via
:func:`puppetmaster.providers.provider_chat`), so a fresh install needs nothing
but provider keys. It is the provider-native worker that powers a standalone
harness ("providers, not platforms") -- so it is built to be a legitimate
engine, not a keys-only toy.

Two modes, one loop:

* ``analyze`` (read-only): the worker may read files, list directories, and
  search the tree -- including CodeGraph symbol search when the repo is graphed
  -- to ground its answer, then emits the same structured finding/risk/decision
  artifacts the rest of Puppetmaster reasons over. A single JSON-only reprompt
  recovers a clean run that returned unstructured prose before it is accepted as
  degraded.
* ``implement`` (full-edit): the worker additionally gets ``write_file`` /
  ``edit_file`` / ``delete_file`` and, to self-verify, a guarded
  ``run_terminal`` (on by default in implement mode). Edits are attributed via
  git diff through :class:`FullEditWorkerAdapter` -- the exact same
  snapshot/guard/PATCH machinery the CLI adapters use -- so we never hand-roll
  diff attribution. A run that stops without touching a file is nudged once to
  actually implement, and is reported ``degraded`` (never ``passed``) when it
  still produces no diff.

The worker-grade guardrails are lifted (as patterns, not vendored code) from the
Hermes agent core: filesystem tools are confined to the worker ``cwd`` with
symlink-escape and binary-write protection; the headless ``run_terminal`` is
gated by a destructive-command denylist because there is no human to confirm;
and a budget governor bounds turns, wall time, and (optionally) total tokens so
a runaway model can never spin forever. All tool output, diffs, and provider
error bodies are scrubbed with :func:`puppetmaster.redaction.redact_secrets`
before they enter an artifact.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

from puppetmaster.codegraph import (
    codegraph_context,
    codegraph_query,
    codegraph_ready,
    enrich_prompt_with_codegraph,
)
from puppetmaster.models import Artifact, ArtifactType, Task
from puppetmaster.providers import AssistantTurn, ProviderError, get_provider, provider_chat
from puppetmaster.redaction import redact_secrets

from ._base import (
    FullEditWorkerAdapter,
    make_patch_artifact,
    verification_artifact,
    _should_emit_patch_artifact,
)
from ._facade import facade
from ._prompts import (
    _ANALYZE_JSON_ONLY_RETRY,
    _IMPLEMENT_NOOP_NUDGE,
    build_implement_prompt,
    build_structured_prompt,
    prompt_with_memory,
    prompt_with_skills,
    with_repo_census,
    with_report_contract,
)
from .cursor import cursor_result_artifacts, implement_report_artifacts

# Budget governor defaults, mode-aware: implementation is a long-horizon task
# and 12 turns / 5 minutes forces a no-op on anything non-trivial, so it gets a
# far larger envelope than a read-only analysis pass.
DEFAULT_ANALYZE_MAX_TURNS = 14
DEFAULT_IMPLEMENT_MAX_TURNS = 50
DEFAULT_ANALYZE_TIMEOUT_SECONDS = 300
DEFAULT_IMPLEMENT_TIMEOUT_SECONDS = 900

# Back-compat aliases (older callers/tests referenced the analyze-tier defaults).
DEFAULT_MAX_TURNS = DEFAULT_ANALYZE_MAX_TURNS
DEFAULT_TIMEOUT_SECONDS = DEFAULT_ANALYZE_TIMEOUT_SECONDS

_TOOL_OUTPUT_LIMIT = 12000  # per tool result, chars, before truncation note
_READ_FILE_LIMIT = 16000
_SEARCH_FILE_CAP = 400  # files scanned per search_code call
_SEARCH_HIT_CAP = 60
_TERMINAL_TIMEOUT_SECONDS = 120
_MUTATING_TOOLS = frozenset({"write_file", "edit_file", "delete_file"})

# Headless destructive-command denylist (guardrail SHAPE lifted from Hermes'
# tool_guardrails): there is no human to confirm a worker's shell command, so a
# small conservative denylist of unambiguously catastrophic patterns is prudent.
# It is intentionally narrow -- it blocks the obviously-irreversible, not merely
# "risky" -- so it never gets in the way of a legitimate `pytest` / `npm test`.
_DESTRUCTIVE_COMMAND_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\brm\s+(-[a-z]*\s+)*-?[a-z]*[rf][a-z]*\s+(-[a-z]*\s+)*(/|~|\$HOME|\.\s*$|\*)",
        r"\brm\s+-rf\b.*\s(/|~|\*)",
        r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",  # fork bomb
        r"\bmkfs\b",
        r"\bdd\b[^\n]*\bof=/dev/",
        r">\s*/dev/(sd|nvme|disk|hd)",
        r"\bgit\s+push\b[^\n]*(--force\b|\s-f\b)",
        r"\bgit\s+reset\s+--hard\b[^\n]*\borigin/",
        r"\b(shutdown|reboot|halt|poweroff)\b",
        r"\bchmod\s+-R\s+0*777\s+/",
        r"\bchown\s+-R\b[^\n]*\s/(?:\s|$)",
        r"\b(curl|wget)\b[^\n]*\|\s*(sudo\s+)?(sh|bash|zsh)\b",  # pipe-to-shell
        r"\bsudo\b",
    )
)

_BINARY_SNIFF_BYTES = 8000


class AgenticAdapter(FullEditWorkerAdapter):
    """Provider-agnostic direct-API worker with an in-process tool loop."""

    name = "agentic"

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        implement = bool(
            task.payload.get("mode") == "implement" or task.payload.get("implement")
        )
        provider = self._resolve_provider(task)
        model = task.payload.get("model") or task.payload.get("adapter_model_name")
        evidence_base = [f"adapter:agentic", f"provider:{provider}", f"model:{model}"]

        if get_provider(provider) is None:
            return [self._fail(task, worker_id, evidence_base, "unknown_provider",
                               f"No provider descriptor for {provider!r}.")]
        if not model:
            return [self._fail(task, worker_id, evidence_base, "no_model",
                               "No model name on the task payload.")]

        if implement:
            return self._run_implement(task, goal, worker_id, provider, model, evidence_base)
        return self._run_analyze(task, goal, worker_id, provider, model, evidence_base)

    # --- provider / prompt plumbing ----------------------------------------

    def _resolve_provider(self, task: Task) -> str:
        """The provider slug for this task, from the routed payload.

        The catalog stamps ``payload_defaults.provider`` (merged into the task
        payload by the router), mirroring the Hermes provider-stamp pattern.
        Falls back to ``openai`` so a bare model still has a sane wire.
        """
        return str(task.payload.get("provider") or "openai").strip().lower()

    def _extra_params(self, task: Task) -> dict:
        """Optional generation knobs, passed through when present."""
        extra: dict[str, Any] = {}
        for src, dst in (
            ("max_output_tokens", "max_tokens"),
            ("max_completion_tokens", "max_tokens"),
            ("max_tokens", "max_tokens"),
        ):
            if task.payload.get(src) is not None:
                extra[dst] = int(task.payload[src])
                break
        if task.payload.get("temperature") is not None:
            extra["temperature"] = float(task.payload["temperature"])
        if task.payload.get("reasoning_effort"):
            extra["reasoning_effort"] = str(task.payload["reasoning_effort"])
        return extra

    def _graph_enabled(self, task: Task, cwd: Path) -> bool:
        """True when CodeGraph is indexed for ``cwd`` and not disabled.

        Gates registration of the ``graph_search`` / ``graph_context`` tools so
        the worker only offers repo-graph tools when they will actually resolve.
        """
        if task.payload.get("disable_codegraph", False):
            return False
        try:
            return bool(codegraph_ready(cwd))
        except Exception:
            return False

    # --- analyze mode ------------------------------------------------------

    def _run_analyze(
        self, task: Task, goal: str, worker_id: str, provider: str, model: str,
        evidence_base: list[str],
    ) -> list[Artifact]:
        cwd = Path(task.payload.get("cwd") or ".").resolve()
        base_prompt = task.payload.get("prompt") or task.instruction
        prompt, codegraph_used = facade("enrich_prompt_with_codegraph")(
            prompt_with_skills(
                prompt_with_memory(
                    build_structured_prompt(base_prompt, final_message_note=True), task
                ),
                task,
            ),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )
        prompt = facade("with_repo_census")(prompt, str(cwd))
        if codegraph_used:
            evidence_base = evidence_base + ["context:codegraph"]

        graph_on = self._graph_enabled(task, cwd)
        tools = self._tool_schema(implement=False, task=task, graph_on=graph_on)

        # One stricter JSON-only reprompt before accepting a degrade: a clean run
        # that returned prose the parser couldn't structure usually recovers on a
        # single retry. Gated by analyze_retry (default on).
        retry_enabled = bool(task.payload.get("analyze_retry", True))
        state = {"attempted": False, "recovered": False}

        def _on_stop(final_text: str, mutated: bool) -> Optional[str]:
            if not retry_enabled or state["attempted"]:
                return None
            if cursor_result_artifacts(task, worker_id, final_text, adapter="agentic"):
                return None  # already structured
            if not (final_text or "").strip():
                return None  # nothing to reshape
            state["attempted"] = True
            return _ANALYZE_JSON_ONLY_RETRY

        try:
            loop = self._agent_loop(
                task, cwd, provider, model, prompt, tools, implement=False,
                on_stop=_on_stop,
            )
        except ProviderError as exc:
            return [self._fail(task, worker_id, evidence_base, exc.reason,
                               redact_secrets(exc.body or str(exc)) or str(exc),
                               status=exc.status)]

        final_text, usage, turns, _mutated, stop_reason = loop
        parsed = cursor_result_artifacts(task, worker_id, final_text, adapter="agentic")
        state["recovered"] = state["attempted"] and bool(parsed)
        degraded = not parsed and bool(final_text)
        result = "degraded" if degraded else "passed"
        evidence = evidence_base + [f"turns:{turns}", f"stop:{stop_reason}"]
        if state["recovered"]:
            evidence.append("retry:recovered")
        elif state["attempted"]:
            evidence.append("retry:exhausted")
        artifacts: list[Artifact] = [
            verification_artifact(
                task=task, worker_id=worker_id, adapter="agentic",
                check=task.instruction, result=result,
                confidence=0.65 if degraded else 0.9,
                evidence=evidence,
                payload={
                    "model": model, "provider": provider, "cwd": str(cwd),
                    "turns": turns, "stop_reason": stop_reason, **usage,
                    "failure": "empty_or_unstructured_agentic_result" if degraded else None,
                    "stdout": (final_text or "")[-_TOOL_OUTPUT_LIMIT:],
                },
            )
        ]
        if degraded:
            artifacts.append(Artifact(
                job_id=task.job_id, task_id=task.id, type=ArtifactType.RISK,
                created_by=worker_id, confidence=0.85,
                evidence=["adapter:agentic", "result:empty-or-unstructured"],
                payload={
                    "risk": "Agentic worker completed without structured findings.",
                    "mitigation": "Rerun with a stricter prompt or a higher-capability model.",
                    "stdout_excerpt": redact_secrets(final_text or "")[:2000],
                },
            ))
        artifacts.extend(parsed)
        return artifacts

    # --- implement mode ----------------------------------------------------

    def _run_implement(
        self, task: Task, goal: str, worker_id: str, provider: str, model: str,
        evidence_base: list[str],
    ) -> list[Artifact]:
        cwd = Path(task.payload.get("cwd") or ".").resolve()
        blocked, before = self.guard_full_edit_run(task, worker_id, "agentic", cwd)
        if blocked is not None:
            return blocked

        base_prompt = with_report_contract(task.payload.get("prompt") or task.instruction)
        prompt, codegraph_used = facade("enrich_prompt_with_codegraph")(
            prompt_with_skills(
                prompt_with_memory(build_implement_prompt(base_prompt), task),
                task,
            ),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )
        if codegraph_used:
            evidence_base = evidence_base + ["context:codegraph"]

        graph_on = self._graph_enabled(task, cwd)
        tools = self._tool_schema(implement=True, task=task, graph_on=graph_on)

        # Nudge once if the model stops having changed nothing: the single most
        # common failure mode is a worker that "answers" implement work as prose
        # and returns a no-op diff. Gated by noop_nudge (default on).
        nudge_enabled = bool(task.payload.get("noop_nudge", True))
        nudged = {"done": False}

        def _on_stop(final_text: str, mutated: bool) -> Optional[str]:
            if not nudge_enabled or nudged["done"] or mutated:
                return None
            nudged["done"] = True
            return _IMPLEMENT_NOOP_NUDGE

        try:
            loop = self._agent_loop(
                task, cwd, provider, model, prompt, tools, implement=True,
                on_stop=_on_stop,
            )
        except ProviderError as exc:
            after = facade("git_snapshot")(cwd, base_tree=str(before.get("tree") or "") or None)
            arts = [self._fail(task, worker_id, evidence_base, exc.reason,
                               redact_secrets(exc.body or str(exc)) or str(exc),
                               status=exc.status)]
            if _should_emit_patch_artifact(before, after):
                arts.append(make_patch_artifact(
                    task, worker_id, before, after, adapter="agentic",
                    status="failed", change="Agentic worker modified files before failing.",
                    sidecar_name="agentic_implement",
                ))
            return arts

        final_text, usage, turns, mutated, stop_reason = loop
        after = facade("git_snapshot")(cwd, base_tree=str(before.get("tree") or "") or None)
        has_work = _should_emit_patch_artifact(before, after)
        # A run that produced no attributable diff is NOT a pass -- surface it as
        # degraded so a no-op can never masquerade as a successful implementation.
        degraded = not has_work
        parsed = implement_report_artifacts(task, worker_id, final_text, adapter="agentic")
        evidence = evidence_base + [f"turns:{turns}", f"stop:{stop_reason}"]
        if nudged["done"]:
            evidence.append("nudge:applied")
        artifacts: list[Artifact] = [
            verification_artifact(
                task=task, worker_id=worker_id, adapter="agentic",
                check=task.instruction,
                result="degraded" if degraded else "passed",
                confidence=0.6 if degraded else 0.9,
                evidence=evidence,
                payload={
                    "model": model, "provider": provider, "cwd": str(cwd),
                    "turns": turns, "stop_reason": stop_reason, "has_work": has_work,
                    "changed_files": after.get("changed_files", []),
                    "untracked_files": after.get("untracked_files", []),
                    **usage,
                    "failure": "no_diff_produced" if degraded else None,
                    "stdout": (final_text or "")[-_TOOL_OUTPUT_LIMIT:],
                },
            )
        ]
        if degraded:
            artifacts.append(Artifact(
                job_id=task.job_id, task_id=task.id, type=ArtifactType.RISK,
                created_by=worker_id, confidence=0.85,
                evidence=["adapter:agentic", "result:no-diff"],
                payload={
                    "risk": "Agentic implement worker finished without changing any files.",
                    "mitigation": (
                        "Treat as a no-op: rerun with a higher-capability model, a "
                        "sharper task, or verify the change was not already present."
                    ),
                    "stdout_excerpt": redact_secrets(final_text or "")[:2000],
                },
            ))
        artifacts.extend(parsed)
        if has_work:
            artifacts.append(make_patch_artifact(
                task, worker_id, before, after, adapter="agentic",
                status="applied", change="Agentic worker modified repository files.",
                sidecar_name="agentic_implement",
            ))
        return artifacts

    # --- the tool loop -----------------------------------------------------

    def _agent_loop(
        self, task: Task, cwd: Path, provider: str, model: str, system_prompt: str,
        tools: list[dict], *, implement: bool,
        on_stop: Optional[Callable[[str, bool], Optional[str]]] = None,
    ) -> tuple[str, dict, int, bool, str]:
        """Run the provider tool-use loop until the model stops calling tools.

        Returns ``(final_text, usage_totals, turns, mutated, stop_reason)``. Each
        turn: send the conversation, execute any tool calls, append their
        results, repeat. A budget governor bounds the run three ways -- max
        turns, a per-call wall-clock timeout, and an optional cumulative
        ``token_budget`` -- so a runaway model can never spin forever.

        When the model stops calling tools, ``on_stop`` (if given) may return a
        follow-up user message to inject and continue (a JSON-only reprompt or a
        no-op nudge); returning ``None`` ends the loop. ``mutated`` records
        whether any write/edit/delete tool actually changed the tree.
        """
        max_turns = int(task.payload.get(
            "max_turns",
            DEFAULT_IMPLEMENT_MAX_TURNS if implement else DEFAULT_ANALYZE_MAX_TURNS,
        ))
        timeout = int(task.payload.get(
            "timeout_seconds",
            DEFAULT_IMPLEMENT_TIMEOUT_SECONDS if implement else DEFAULT_ANALYZE_TIMEOUT_SECONDS,
        ))
        token_budget = task.payload.get("token_budget")
        token_budget = int(token_budget) if token_budget else None
        extra = self._extra_params(task)
        messages: list[dict] = [{"role": "user", "content": system_prompt}]
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        final_text = ""
        mutated = False
        turns = 0
        stop_reason = "max_turns"

        for turns in range(1, max_turns + 1):
            turn: AssistantTurn = provider_chat(
                provider=provider, model=model, messages=messages,
                tools=tools or None, extra=extra, timeout=timeout,
            )
            for key in usage_total:
                usage_total[key] += int(turn.usage.get(key, 0))
            final_text = turn.text or final_text
            if not turn.tool_calls:
                follow_up = on_stop(final_text, mutated) if on_stop else None
                if follow_up:
                    messages.append({"role": "user", "content": follow_up})
                    if token_budget and usage_total["total_tokens"] >= token_budget:
                        stop_reason = "token_budget"
                        break
                    continue
                stop_reason = "model_stopped"
                break
            # Record the assistant's tool-call turn, then each tool's result.
            messages.append({
                "role": "assistant",
                "content": turn.text or "",
                "tool_calls": [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                    for c in turn.tool_calls
                ],
            })
            for call in turn.tool_calls:
                output = self._execute_tool(call["name"], call["arguments"], cwd, implement, task)
                if call["name"] in _MUTATING_TOOLS and not output.startswith("error"):
                    mutated = True
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": _truncate(redact_secrets(output) or "", _TOOL_OUTPUT_LIMIT),
                })
            if token_budget and usage_total["total_tokens"] >= token_budget:
                stop_reason = "token_budget"
                break
        return final_text, {
            "tokens_in": usage_total["prompt_tokens"],
            "tokens_out": usage_total["completion_tokens"],
            "tokens_total": usage_total["total_tokens"],
        }, turns, mutated, stop_reason

    def _tool_schema(self, *, implement: bool, task: Task, graph_on: bool = False) -> list[dict]:
        """OpenAI-format tool specs; provider_chat translates for Anthropic."""
        def fn(name, desc, props, required):
            return {"type": "function", "function": {
                "name": name, "description": desc,
                "parameters": {"type": "object", "properties": props, "required": required},
            }}

        tools = [
            fn("read_file", "Read a UTF-8 text file within the workspace.",
               {"path": {"type": "string"},
                "start_line": {"type": "integer", "description": "1-indexed start line (optional)"},
                "limit": {"type": "integer", "description": "max lines to read (optional)"}},
               ["path"]),
            fn("list_dir", "List entries of a directory within the workspace.",
               {"path": {"type": "string"}}, ["path"]),
            fn("search_code", "Plain-text/regex search over the workspace; returns matching path:line snippets. Best for log strings, config values, and comments.",
               {"query": {"type": "string"}, "glob": {"type": "string", "description": "optional filename filter, e.g. *.py"}},
               ["query"]),
        ]
        if graph_on:
            tools.append(fn(
                "graph_search",
                "Search the CodeGraph symbol index for definitions/references by name. "
                "Prefer this over search_code for 'where is X defined / what is Y' symbol questions.",
                {"query": {"type": "string"},
                 "kind": {"type": "string", "description": "optional node kind filter, e.g. function, class, method"},
                 "limit": {"type": "integer", "description": "max results (optional)"}},
                ["query"]))
            tools.append(fn(
                "graph_context",
                "Pull a task-scoped CodeGraph subgraph (the most relevant symbols and their edges) for a natural-language task description.",
                {"task": {"type": "string"},
                 "max_nodes": {"type": "integer", "description": "max nodes (optional, default 15)"}},
                ["task"]))
        if implement:
            tools.append(fn("write_file", "Create or overwrite a text file within the workspace.",
                            {"path": {"type": "string"}, "content": {"type": "string"}},
                            ["path", "content"]))
            tools.append(fn("edit_file", "Replace an exact occurrence of old_string with new_string in a file. Set replace_all=true to replace every occurrence.",
                            {"path": {"type": "string"}, "old_string": {"type": "string"},
                             "new_string": {"type": "string"},
                             "replace_all": {"type": "boolean", "description": "replace all occurrences (default false)"}},
                            ["path", "old_string", "new_string"]))
            tools.append(fn("delete_file", "Delete a file within the workspace.",
                            {"path": {"type": "string"}}, ["path"]))
            if self._terminal_enabled(task):
                tools.append(fn("run_terminal", "Run a bounded shell command in the workspace (e.g. to run focused tests) and return its output. Destructive commands are refused.",
                                {"command": {"type": "string"}}, ["command"]))
        if bool(task.payload.get("allow_web", False)):
            tools.append(fn("web_fetch", "Fetch a URL and return its text content.",
                            {"url": {"type": "string"}}, ["url"]))
        return tools

    def _terminal_enabled(self, task: Task) -> bool:
        """Whether implement-mode workers may self-verify with ``run_terminal``.

        On by default in implement mode -- a harness that cannot run its own
        tests is not a legitimate implement engine -- but the destructive-command
        denylist still guards every invocation, and ``allow_terminal=false``
        turns it off entirely for locked-down runs.
        """
        return bool(task.payload.get("allow_terminal", True))

    def _execute_tool(
        self, name: str, args: dict, cwd: Path, implement: bool, task: Task
    ) -> str:
        """Dispatch one tool call. Returns a text result (never raises)."""
        try:
            if name == "read_file":
                return self._tool_read_file(args, cwd)
            if name == "list_dir":
                return self._tool_list_dir(args, cwd)
            if name == "search_code":
                return self._tool_search_code(args, cwd)
            if name == "graph_search":
                return self._tool_graph_search(args, cwd, task)
            if name == "graph_context":
                return self._tool_graph_context(args, cwd, task)
            if name == "write_file" and implement:
                return self._tool_write_file(args, cwd)
            if name == "edit_file" and implement:
                return self._tool_edit_file(args, cwd)
            if name == "delete_file" and implement:
                return self._tool_delete_file(args, cwd)
            if name == "run_terminal" and implement and self._terminal_enabled(task):
                return self._tool_run_terminal(args, cwd)
            if name == "web_fetch" and bool(task.payload.get("allow_web", False)):
                return self._tool_web_fetch(args)
            return f"error: tool {name!r} is not available in this mode"
        except Exception as exc:  # a tool failure must not kill the worker
            return f"error: {type(exc).__name__}: {exc}"

    # --- confined filesystem tools -----------------------------------------

    def _confine(self, cwd: Path, rel: str) -> Path:
        """Resolve ``rel`` under ``cwd``, rejecting traversal outside the tree.

        Both sides are fully resolved before comparison so symlinked temp roots
        (macOS ``/var`` -> ``/private/var``) and ``..`` segments can't smuggle a
        path outside the workspace.
        """
        root = cwd.resolve()
        target = (root / rel).resolve() if not os.path.isabs(rel) else Path(rel).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"path {rel!r} escapes the workspace")
        return target

    def _tool_read_file(self, args: dict, cwd: Path) -> str:
        path = self._confine(cwd, str(args.get("path", "")))
        if _looks_binary(path):
            return "error: refusing to read an apparent binary file as text"
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        start = max(1, int(args["start_line"])) if args.get("start_line") else 1
        limit = int(args["limit"]) if args.get("limit") else len(lines)
        chunk = lines[start - 1:start - 1 + limit]
        body = "\n".join(chunk)
        return _truncate(body, _READ_FILE_LIMIT)

    def _tool_list_dir(self, args: dict, cwd: Path) -> str:
        path = self._confine(cwd, str(args.get("path", ".")))
        entries = sorted(
            (f"{e.name}/" if e.is_dir() else e.name) for e in path.iterdir()
        )
        return "\n".join(entries) if entries else "(empty)"

    def _tool_search_code(self, args: dict, cwd: Path) -> str:
        query = str(args.get("query", ""))
        glob = str(args.get("glob") or "")
        if not query:
            return "error: empty query"
        import fnmatch
        import re as _re
        try:
            pattern = _re.compile(query)
        except _re.error:
            pattern = _re.compile(_re.escape(query))
        hits: list[str] = []
        scanned = 0
        skip_dirs = {".git", "node_modules", ".venv", "__pycache__", "dist", "build", ".codegraph"}
        for root, dirs, files in os.walk(cwd):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for fname in files:
                if glob and not fnmatch.fnmatch(fname, glob):
                    continue
                scanned += 1
                if scanned > _SEARCH_FILE_CAP:
                    hits.append("... (search file cap reached)")
                    return "\n".join(hits)
                fpath = Path(root) / fname
                try:
                    for i, line in enumerate(fpath.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                        if pattern.search(line):
                            rel = fpath.relative_to(cwd)
                            hits.append(f"{rel}:{i}: {line.strip()[:200]}")
                            if len(hits) >= _SEARCH_HIT_CAP:
                                hits.append("... (hit cap reached)")
                                return "\n".join(hits)
                except (OSError, ValueError):
                    continue
        return "\n".join(hits) if hits else "(no matches)"

    def _tool_graph_search(self, args: dict, cwd: Path, task: Task) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return "error: empty query"
        result = codegraph_query(
            query, cwd,
            kind=str(args["kind"]) if args.get("kind") else None,
            limit=int(args["limit"]) if args.get("limit") else 25,
        )
        if not result.get("ok"):
            return f"error: codegraph unavailable ({result.get('error') or result.get('stderr') or 'unknown'})"
        return _truncate(str(result.get("stdout") or "(no matches)"), _TOOL_OUTPUT_LIMIT)

    def _tool_graph_context(self, args: dict, cwd: Path, task: Task) -> str:
        query = str(args.get("task", "")).strip()
        if not query:
            return "error: empty task"
        context = codegraph_context(
            query, cwd,
            max_nodes=int(args["max_nodes"]) if args.get("max_nodes") else 15,
        )
        if not context:
            return "(no codegraph context)"
        return _truncate(context, _TOOL_OUTPUT_LIMIT)

    def _tool_write_file(self, args: dict, cwd: Path) -> str:
        path = self._confine(cwd, str(args.get("path", "")))
        content = str(args.get("content", ""))
        if "\x00" in content:
            return "error: refusing to write NUL bytes (binary content) via write_file"
        if path.exists() and _looks_binary(path):
            return "error: refusing to overwrite an apparent binary file"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"wrote {_rel(path, cwd)} ({len(content)} chars)"

    def _tool_edit_file(self, args: dict, cwd: Path) -> str:
        path = self._confine(cwd, str(args.get("path", "")))
        old = str(args.get("old_string", ""))
        new = str(args.get("new_string", ""))
        replace_all = bool(args.get("replace_all", False))
        text = path.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            return "error: old_string not found (must match exactly)" + _near_miss_hint(text, old)
        if count > 1 and not replace_all:
            return (
                "error: old_string is not unique; add more surrounding context "
                "or set replace_all=true"
            )
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        path.write_text(updated, encoding="utf-8")
        n = count if replace_all else 1
        return f"edited {_rel(path, cwd)} ({n} replacement{'s' if n != 1 else ''})"

    def _tool_delete_file(self, args: dict, cwd: Path) -> str:
        path = self._confine(cwd, str(args.get("path", "")))
        if not path.exists():
            return "error: file does not exist"
        if path.is_dir():
            return "error: path is a directory; delete_file only removes files"
        rel = _rel(path, cwd)
        path.unlink()
        return f"deleted {rel}"

    def _tool_run_terminal(self, args: dict, cwd: Path) -> str:
        command = str(args.get("command", ""))
        if not command.strip():
            return "error: empty command"
        blocked = _destructive_command_match(command)
        if blocked is not None:
            return (
                "error: refusing to run a potentially destructive command "
                f"(matched guardrail: {blocked}). Narrow the command to a "
                "specific, reversible action."
            )
        try:
            proc = subprocess.run(
                command, shell=True, cwd=str(cwd), capture_output=True,
                text=True, timeout=_TERMINAL_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return f"error: command timed out ({_TERMINAL_TIMEOUT_SECONDS}s)"
        out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        return f"exit={proc.returncode}\n{_truncate(out, _TOOL_OUTPUT_LIMIT)}"

    def _tool_web_fetch(self, args: dict) -> str:
        import urllib.request
        url = str(args.get("url", ""))
        if not (url.startswith("http://") or url.startswith("https://")):
            return "error: url must be http(s)"
        req = urllib.request.Request(url, headers={"User-Agent": "puppetmaster-agentic"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return _truncate(body, _TOOL_OUTPUT_LIMIT)

    # --- helpers -----------------------------------------------------------

    def _fail(
        self, task: Task, worker_id: str, evidence: list[str], reason: str,
        detail: str, *, status: Optional[int] = None,
    ) -> Artifact:
        return verification_artifact(
            task=task, worker_id=worker_id, adapter="agentic",
            check=task.instruction, result="failed", confidence=0.55,
            evidence=evidence + [reason],
            payload={"failure": reason, "returncode": status, "stderr": detail[:8000]},
        )


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated, {len(text) - limit} more chars)"


def _rel(path: Path, cwd: Path) -> str:
    """Workspace-relative display path, resilient to symlinked roots.

    ``_confine`` resolves symlinks (``/var`` -> ``/private/var`` on macOS), so
    the confined path can't be made relative to an unresolved ``cwd``. Compare
    against the resolved root; fall back to the bare name if anything is off.
    """
    try:
        return str(path.relative_to(cwd.resolve()))
    except ValueError:
        return path.name


def _looks_binary(path: Path) -> bool:
    """Heuristic: a file is binary if its head contains a NUL byte.

    Lifted from Hermes' file_safety patterns -- the headless worker must not try
    to read or clobber a binary as UTF-8 text (which would corrupt it or produce
    garbage tool output). Never raises: an unreadable file is treated as
    non-binary so the normal read/write path can surface the real OS error.
    """
    try:
        with path.open("rb") as handle:
            return b"\x00" in handle.read(_BINARY_SNIFF_BYTES)
    except OSError:
        return False


def _destructive_command_match(command: str) -> Optional[str]:
    """Return the matched guardrail pattern for a destructive command, else None."""
    for pattern in _DESTRUCTIVE_COMMAND_PATTERNS:
        if pattern.search(command):
            return pattern.pattern
    return None


def _near_miss_hint(text: str, old: str, *, max_len: int = 200) -> str:
    """A short hint when an edit's old_string isn't found, to help the model
    self-correct instead of giving up. Surfaces the first line of the intended
    match if that line exists in the file with different surrounding whitespace.
    """
    first_line = (old.strip().splitlines() or [""])[0].strip()
    if not first_line:
        return ""
    for line in text.splitlines():
        if first_line and first_line in line and line.strip() != old.strip():
            return (
                ". Closest line in file: "
                + repr(line.strip()[:max_len])
                + " (whitespace/context differs — match it exactly)"
            )
    return ""
