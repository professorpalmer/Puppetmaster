"""Standalone, provider-agnostic agentic worker adapter.

This is Puppetmaster's answer to "run workers on the user's keys, with no
external agent CLI." Where :mod:`hermes`, :mod:`cursor`, :mod:`claude_code`, and
:mod:`codex` shell out to a third-party agent binary, ``AgenticAdapter`` runs
its OWN tool-use loop against a provider HTTP API directly (via
:func:`puppetmaster.providers.provider_chat`), so a fresh install needs nothing
but provider keys.

Two modes, one loop:

* ``analyze`` (read-only): the worker may read files, list directories, and
  search the tree to ground its answer, then emits the same structured
  finding/risk/decision artifacts the rest of Puppetmaster reasons over. This
  is the standalone replacement for read-only swarms/audits.
* ``implement`` (full-edit): the worker additionally gets ``write_file`` /
  ``edit_file`` and, opt-in, ``run_terminal`` / ``web_fetch``. Edits are
  attributed via git diff through :class:`FullEditWorkerAdapter` -- the exact
  same snapshot/guard/PATCH machinery the CLI adapters use -- so we never
  hand-roll diff attribution.

Security posture: every filesystem tool is confined to the worker ``cwd``;
terminal and web tools are OFF unless the task opts in; all tool output,
diffs, and provider error bodies are scrubbed with
:func:`puppetmaster.redaction.redact_secrets` before they enter an artifact.
The from-scratch loop is deliberately conservative (bounded turns, bounded
output) -- it is newer and less battle-tested than the mature CLIs, so it errs
toward stopping rather than thrashing.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

from puppetmaster.codegraph import enrich_prompt_with_codegraph
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
    build_implement_prompt,
    build_structured_prompt,
    prompt_with_memory,
    with_repo_census,
    with_report_contract,
)
from .cursor import cursor_result_artifacts

DEFAULT_MAX_TURNS = 12
DEFAULT_TIMEOUT_SECONDS = 300
_TOOL_OUTPUT_LIMIT = 12000  # per tool result, chars, before truncation note
_READ_FILE_LIMIT = 16000
_SEARCH_FILE_CAP = 400  # files scanned per search_code call
_SEARCH_HIT_CAP = 60


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

    # --- analyze mode ------------------------------------------------------

    def _run_analyze(
        self, task: Task, goal: str, worker_id: str, provider: str, model: str,
        evidence_base: list[str],
    ) -> list[Artifact]:
        cwd = Path(task.payload.get("cwd") or ".").resolve()
        base_prompt = task.payload.get("prompt") or task.instruction
        prompt, codegraph_used = facade("enrich_prompt_with_codegraph")(
            prompt_with_memory(build_structured_prompt(base_prompt, final_message_note=True), task),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )
        prompt = facade("with_repo_census")(prompt, str(cwd))
        if codegraph_used:
            evidence_base = evidence_base + ["context:codegraph"]

        tools = self._tool_schema(implement=False, task=task)
        try:
            final_text, usage, turns = self._agent_loop(
                task, cwd, provider, model, prompt, tools, implement=False,
            )
        except ProviderError as exc:
            return [self._fail(task, worker_id, evidence_base, exc.reason,
                               redact_secrets(exc.body or str(exc)) or str(exc),
                               status=exc.status)]

        parsed = cursor_result_artifacts(task, worker_id, final_text, adapter="agentic")
        degraded = not parsed and bool(final_text)
        result = "degraded" if degraded else "passed"
        artifacts: list[Artifact] = [
            verification_artifact(
                task=task, worker_id=worker_id, adapter="agentic",
                check=task.instruction, result=result,
                confidence=0.65 if degraded else 0.9,
                evidence=evidence_base + [f"turns:{turns}"],
                payload={
                    "model": model, "provider": provider, "cwd": str(cwd),
                    "turns": turns, **usage,
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
            prompt_with_memory(build_implement_prompt(base_prompt), task),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )
        if codegraph_used:
            evidence_base = evidence_base + ["context:codegraph"]

        tools = self._tool_schema(implement=True, task=task)
        try:
            final_text, usage, turns = self._agent_loop(
                task, cwd, provider, model, prompt, tools, implement=True,
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

        after = facade("git_snapshot")(cwd, base_tree=str(before.get("tree") or "") or None)
        has_work = _should_emit_patch_artifact(before, after)
        parsed = cursor_result_artifacts(task, worker_id, final_text, adapter="agentic")
        artifacts: list[Artifact] = [
            verification_artifact(
                task=task, worker_id=worker_id, adapter="agentic",
                check=task.instruction, result="passed",
                confidence=0.9, evidence=evidence_base + [f"turns:{turns}"],
                payload={
                    "model": model, "provider": provider, "cwd": str(cwd),
                    "turns": turns, "has_work": has_work,
                    "changed_files": after.get("changed_files", []),
                    "untracked_files": after.get("untracked_files", []),
                    **usage,
                    "stdout": (final_text or "")[-_TOOL_OUTPUT_LIMIT:],
                },
            )
        ]
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
    ) -> tuple[str, dict, int]:
        """Run the provider tool-use loop until the model stops calling tools.

        Returns ``(final_text, usage_totals, turns)``. Each turn: send the
        conversation, execute any tool calls, append their results, repeat.
        Bounded by ``max_turns`` and a per-call timeout so a runaway model can
        never spin forever.
        """
        max_turns = int(task.payload.get("max_turns", DEFAULT_MAX_TURNS))
        timeout = int(task.payload.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
        extra = self._extra_params(task)
        messages: list[dict] = [{"role": "user", "content": system_prompt}]
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        final_text = ""
        turns = 0

        for turns in range(1, max_turns + 1):
            turn: AssistantTurn = provider_chat(
                provider=provider, model=model, messages=messages,
                tools=tools or None, extra=extra, timeout=timeout,
            )
            for key in usage_total:
                usage_total[key] += int(turn.usage.get(key, 0))
            final_text = turn.text or final_text
            if not turn.tool_calls:
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
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": _truncate(redact_secrets(output) or "", _TOOL_OUTPUT_LIMIT),
                })
        return final_text, {
            "tokens_in": usage_total["prompt_tokens"],
            "tokens_out": usage_total["completion_tokens"],
            "tokens_total": usage_total["total_tokens"],
        }, turns

    def _tool_schema(self, *, implement: bool, task: Task) -> list[dict]:
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
            fn("search_code", "Search the workspace for a substring/regex; returns matching path:line snippets.",
               {"query": {"type": "string"}, "glob": {"type": "string", "description": "optional filename filter, e.g. *.py"}},
               ["query"]),
        ]
        if implement:
            tools.append(fn("write_file", "Create or overwrite a text file within the workspace.",
                            {"path": {"type": "string"}, "content": {"type": "string"}},
                            ["path", "content"]))
            tools.append(fn("edit_file", "Replace the first exact occurrence of old_string with new_string in a file.",
                            {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
                            ["path", "old_string", "new_string"]))
            if bool(task.payload.get("allow_terminal", False)):
                tools.append(fn("run_terminal", "Run a bounded shell command in the workspace and return its output.",
                                {"command": {"type": "string"}}, ["command"]))
        if bool(task.payload.get("allow_web", False)):
            tools.append(fn("web_fetch", "Fetch a URL and return its text content.",
                            {"url": {"type": "string"}}, ["url"]))
        return tools

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
            if name == "write_file" and implement:
                return self._tool_write_file(args, cwd)
            if name == "edit_file" and implement:
                return self._tool_edit_file(args, cwd)
            if name == "run_terminal" and implement and bool(task.payload.get("allow_terminal", False)):
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

    def _tool_write_file(self, args: dict, cwd: Path) -> str:
        path = self._confine(cwd, str(args.get("path", "")))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(args.get("content", "")), encoding="utf-8")
        return f"wrote {path.relative_to(cwd)} ({len(str(args.get('content', '')))} chars)"

    def _tool_edit_file(self, args: dict, cwd: Path) -> str:
        path = self._confine(cwd, str(args.get("path", "")))
        old = str(args.get("old_string", ""))
        new = str(args.get("new_string", ""))
        text = path.read_text(encoding="utf-8", errors="replace")
        if old not in text:
            return "error: old_string not found (must match exactly, once)"
        if text.count(old) > 1:
            return "error: old_string is not unique; add more surrounding context"
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"edited {path.relative_to(cwd)}"

    def _tool_run_terminal(self, args: dict, cwd: Path) -> str:
        command = str(args.get("command", ""))
        if not command.strip():
            return "error: empty command"
        try:
            proc = subprocess.run(
                command, shell=True, cwd=str(cwd), capture_output=True,
                text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            return "error: command timed out (120s)"
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
