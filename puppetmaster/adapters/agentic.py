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

import concurrent.futures
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Optional

from puppetmaster.codegraph import (
    codegraph_context,
    codegraph_query,
    codegraph_ready,
    enrich_prompt_with_codegraph,
)
from puppetmaster.cancellation import JobCancelled, is_cancelled
from puppetmaster.models import Artifact, ArtifactType, Task
from puppetmaster.provider_circuit import (
    get_provider_circuit_breaker,
    resolve_circuit_key,
)
from puppetmaster.providers import (
    AssistantTurn,
    ProviderError,
    get_provider,
    is_retryable_provider_error,
    provider_chat,
    provider_chat_streaming,
    provider_key_pool,
    provider_retry_backoff_seconds,
)
from puppetmaster.redaction import redact_secrets
from puppetmaster.state import resolve_state_dir
from puppetmaster.hashline import (
    SnapshotStore,
    apply_patch,
    content_tag,
    format_apply_success,
    format_numbered_read,
    fs_cache_enabled,
    hashline_enabled,
    normalize_text,
)
from puppetmaster.tool_batch import (
    parallel_executor_max_workers,
    plan_tool_batch_segments,
)
from puppetmaster.tool_offload import offload_tool_output, read_offload_blob

from ._base import (
    FullEditWorkerAdapter,
    make_patch_artifact,
    verification_artifact,
    _should_emit_patch_artifact,
)
from ._context_budget import (
    DEFAULT_COMPACT_AFTER_TURNS,
    DEFAULT_KEEP_RECENT,
    compress_history,
)
from ._delta_bus import delta_sink_for
from ._delta_stream import DurableDeltaWriter
from ._facade import facade
from ._streaming import _resolve_sidecar_state_dir
from ._prompts import (
    _ANALYZE_JSON_ONLY_RETRY,
    _EMPTY_RESPONSE_NUDGE,
    _IMPLEMENT_NOOP_NUDGE,
    _LENGTH_CONTINUATION_NUDGE,
    build_implement_prompt,
    build_structured_prompt,
    prompt_with_memory,
    prompt_with_skills,
    split_prompt_messages,
    with_job_brief,
    with_report_contract,
)
from .cursor import (
    cursor_artifact_from_item,
    cursor_result_artifacts,
    implement_report_artifacts,
    parse_cursor_artifact_payload,
)

# Budget governor defaults, mode-aware: implementation is a long-horizon task
# and 12 turns / 5 minutes forces a no-op on anything non-trivial, so it gets a
# far larger envelope than a read-only analysis pass. The envelope also has to
# absorb plan-tool turns and verify-before-submit fix iterations without
# starving the actual work, so implement runs a generous turn budget.
DEFAULT_ANALYZE_MAX_TURNS = 16
DEFAULT_IMPLEMENT_MAX_TURNS = 64
DEFAULT_ANALYZE_TIMEOUT_SECONDS = 300
DEFAULT_IMPLEMENT_TIMEOUT_SECONDS = 900

# Transient-failure retry envelope (Hermes parity): a single 429/5xx/timeout on
# the wire must not sink an otherwise-healthy worker. We retry the provider call
# with jittered exponential backoff on classifiably-transient failures only;
# auth/quota/4xx (except 429) surface immediately, unretried.
DEFAULT_PROVIDER_MAX_RETRIES = 2

# Bounded recovery counts so a pathological model can't loop forever on the
# empty-response nudge or the length-continuation retry.
_MAX_EMPTY_RECOVERIES = 1
_MAX_LENGTH_CONTINUATIONS = 3

# Consecutive assistant turns that emit text/reasoning but zero tool_calls.
# Pure reasoners (or endpoints that drop the tools schema) otherwise grind the
# token budget on chain-of-thought without ever calling submit_findings /
# submit_report. Fail fast with stop_reason=no_tool_calls instead.
DEFAULT_NO_TOOL_CALLS_STREAK = 3

# Reserve the final fraction of token_budget for a forced submit turn. When
# cumulative usage crosses (1 - fraction) of the budget, the next turn is
# compelled via tool_choice to call submit_findings / submit_report with
# whatever partial findings exist -- so a CoT-heavy reasoner cannot burn the
# entire budget without ever submitting. Override with PUPPETMASTER_SUBMIT_RESERVE.
SUBMIT_RESERVE_FRACTION = 0.2

# Default reasoning_effort for OpenRouter / OpenAI-compatible swarm workers when
# the caller did not set one. Caps chain-of-thought so CoT cannot consume the
# whole token budget; payload.reasoning_effort still wins when present.
DEFAULT_SWARM_REASONING_EFFORT = "low"

# Prompt-only first-turn nudge: deltas show turn 1 is often pure reasoning on
# deep reasoners. Require a real tool call before analysis prose (no artificial
# no-op force -- the model still chooses which tool).
_FIRST_TURN_TOOL_NUDGE = (
    "FIRST TURN REQUIREMENT: Your first response MUST include a tool call "
    "(start with search_code, list_dir, read_file, or graph_search before any "
    "analysis prose). Do not spend turn 1 on reasoning-only text."
)

_BUDGET_FORCE_SUBMIT_ANALYZE = (
    "Token budget reserve reached. Call `submit_findings` NOW with whatever "
    "partial findings you have (an empty array is fine if you found nothing). "
    "Do not continue exploring or reasoning -- submit immediately."
)

_BUDGET_FORCE_SUBMIT_IMPLEMENT = (
    "Token budget reserve reached. Call `submit_report` NOW with a short "
    "summary of whatever work you completed (partial progress is acceptable). "
    "Do not continue exploring -- submit immediately."
)

# Verify-before-submit loop (Codex/Claude-Code parity): before accepting an
# implement worker's submit_report, run the project's verification command
# (tests / typecheck / lint) and, on failure, feed the output back so the model
# fixes its own regression before finishing -- an edit that doesn't pass the
# checks is not a done edit. Bounded so a stubborn failure can't loop forever.
# A baseline run on the clean tree separates a regression the change introduced
# (gating: block on it) from a suite that was already red (advisory: report it,
# never fail an otherwise-legitimate diff over pre-existing breakage).
DEFAULT_VERIFY_MAX_RETRIES = 2
_VERIFY_TIMEOUT_SECONDS = 300

# Prompt-token budget before older tool outputs are elided (Hermes-style live
# compression). Conservative default that fits common 128k-window models; the
# caller can raise/lower it per model via payload['context_token_budget'].
# Turn-count compaction (DEFAULT_COMPACT_AFTER_TURNS) also stubs older tool
# results even under budget; kill with PUPPETMASTER_HISTORY_COMPACT=0.
DEFAULT_CONTEXT_TOKEN_BUDGET = 120_000

# Back-compat aliases (older callers/tests referenced the analyze-tier defaults).
DEFAULT_MAX_TURNS = DEFAULT_ANALYZE_MAX_TURNS
DEFAULT_TIMEOUT_SECONDS = DEFAULT_ANALYZE_TIMEOUT_SECONDS

_TOOL_OUTPUT_LIMIT = 12000  # artifact stdout tail / excerpt cap (not model-facing)
_SEARCH_FILE_CAP = 400  # files scanned per search_code call
_SEARCH_HIT_CAP = 60
_TERMINAL_TIMEOUT_SECONDS = 120
_MUTATING_TOOLS = frozenset({"write_file", "edit_file", "delete_file", "apply_hashline"})

# Terminal "submit" tools. The parity fix (v2 overhaul): structured output rides
# the provider-native tool-calling channel -- the model calls ``submit_findings``
# (analyze) or ``submit_report`` (implement) with schema-constrained arguments --
# instead of hoping the model emits a parseable JSON blob as free-text. This is
# how Codex/Claude Code/Hermes get reliable structure across weak and strong
# models: the provider constrains the tool arguments, so even a cheap model can't
# "return prose the parser can't structure." Free-text JSON stays a fallback for
# providers/models without tool calling.
_SUBMIT_FINDINGS_TOOL = "submit_findings"
_SUBMIT_REPORT_TOOL = "submit_report"
_SUBMIT_TOOLS = frozenset({_SUBMIT_FINDINGS_TOOL, _SUBMIT_REPORT_TOOL})

# Plan/TODO tool (Codex/Claude-Code parity): a non-terminal tool the model calls
# to record and update its step-by-step plan. It keeps a long, multi-step run
# organized and gives the host a visible task list, without touching the repo.
_PLAN_TOOL = "update_plan"
_PLAN_STATUSES = ("pending", "in_progress", "done")

# Provider -> the env var whose key to fix, named in the auth-failure RISK so a
# dead/revoked key is diagnosable at a glance instead of laundered into a
# generic "no structured findings" degrade.
_PROVIDER_ENV_HINTS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY (or GOOGLE_API_KEY)",
    "google": "GOOGLE_API_KEY (or GEMINI_API_KEY)",
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "bedrock": (
        "AWS_PROFILE / default ~/.aws profile, AWS_ACCESS_KEY_ID + "
        "AWS_SECRET_ACCESS_KEY, or AWS_BEARER_TOKEN_BEDROCK"
    ),
}

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

_BROWSER_TOOL_NAMES = frozenset((
    "browser_navigate", "browser_snapshot", "browser_click", "browser_type",
    "browser_scroll", "browser_back", "browser_get_text", "browser_screenshot",
))


def _submit_reserve_fraction() -> float:
    """Resolve the submit-reserve fraction (module default or env override)."""
    raw = os.environ.get("PUPPETMASTER_SUBMIT_RESERVE")
    if raw is None or str(raw).strip() == "":
        return SUBMIT_RESERVE_FRACTION
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return SUBMIT_RESERVE_FRACTION
    if value < 0.0 or value >= 1.0:
        return SUBMIT_RESERVE_FRACTION
    return value


def _budget_force_threshold(token_budget: int, reserve: float) -> float:
    """Usage at/above which the next turn must be a forced submit."""
    return float(token_budget) * (1.0 - reserve)


def _provider_is_openai_compatible(provider: str) -> bool:
    """True for OpenRouter and other OpenAI-wire providers (reasoning_effort)."""
    slug = (provider or "").strip().lower()
    if slug == "openrouter":
        return True
    desc = get_provider(slug)
    return bool(desc is not None and getattr(desc, "wire", None) == "openai")


def _with_first_turn_tool_nudge(prompt: str) -> str:
    """Append the first-turn tool-call requirement to a worker prompt."""
    text = prompt or ""
    if _FIRST_TURN_TOOL_NUDGE in text:
        return text
    return text + "\n\n" + _FIRST_TURN_TOOL_NUDGE


class AgenticAdapter(FullEditWorkerAdapter):
    """Provider-agnostic direct-API worker with an in-process tool loop."""

    name = "agentic"

    def __init__(self) -> None:
        super().__init__()
        # Per-run session state for Hashline tags + mtime read cache. Reset at
        # the start of each agent loop so tags never leak across jobs.
        self._hashline_store = SnapshotStore()
        self._fs_cache: dict[str, tuple[int, int, str]] = {}

    def _reset_session_caches(self) -> None:
        self._hashline_store = SnapshotStore()
        self._fs_cache = {}

    def _invalidate_fs_cache(self, path: Path) -> None:
        self._fs_cache.pop(str(path.resolve()), None)

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
        elif _provider_is_openai_compatible(self._resolve_provider(task)):
            # Swarm workers on OpenRouter / OpenAI-compatible endpoints often
            # default to unbounded CoT; pin a low effort unless the caller set one.
            extra["reasoning_effort"] = DEFAULT_SWARM_REASONING_EFFORT
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
                    facade("with_repo_census")(
                        with_job_brief(
                            build_structured_prompt(base_prompt, final_message_note=True),
                            task,
                        ),
                        str(cwd),
                    ),
                    task,
                ),
                task,
            ),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )
        if codegraph_used:
            evidence_base = evidence_base + ["context:codegraph"]

        prompt = _with_first_turn_tool_nudge(prompt)
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

        plan_holder: dict = {"steps": []}

        def _on_plan(steps: list) -> None:
            plan_holder["steps"] = steps

        primary_provider, primary_model = provider, model
        try:
            loop, provider, model = self._run_loop_with_failover(
                task=task, cwd=cwd, prompt=prompt, tools=tools, implement=False,
                on_stop=_on_stop, on_plan=_on_plan,
                worker_id=worker_id, provider=provider, model=model,
            )
        except ProviderError as exc:
            detail = redact_secrets(exc.body or str(exc)) or str(exc)
            failure = exc.failure
            if (provider or "").lower() == "bedrock":
                from puppetmaster.provider_health import bedrock_failure_for_recovery

                failure = bedrock_failure_for_recovery(exc)
            arts = [self._fail(task, worker_id, evidence_base, failure,
                               detail, status=exc.status, provider_reason=exc.reason)]
            auth_risk = self._auth_failure_risk(
                task, worker_id, provider, exc.status or 0, detail, reason=failure)
            if auth_risk is not None:
                arts.append(auth_risk)
            return arts
        failed_over = (provider, model) != (primary_provider, primary_model)

        final_text, usage, turns, _mutated, stop_reason, submitted = loop

        # Structured output has two channels. Preferred: the model called
        # submit_findings (native, schema-constrained) -- ``submitted`` is the
        # item list, possibly empty for an honest "found nothing". Fallback: the
        # model emitted a JSON blob as free-text (older/tool-less models). A run
        # is degraded ONLY when neither channel produced structure AND the model
        # actually said something -- an explicit empty submission is a clean pass.
        # no_tool_calls is a hard fail: the model never used the tool channel.
        no_tools = stop_reason == "no_tool_calls"
        if no_tools:
            parsed = []
            structured_ok = False
            channel = "none"
        elif submitted is not None:
            parsed = _items_to_artifacts(task, worker_id, submitted)
            structured_ok = True
            channel = "tool"
        else:
            parsed = cursor_result_artifacts(task, worker_id, final_text, adapter="agentic")
            # JSON fallback counts only when it produced typed artifacts. An
            # empty ``{"artifacts":[]}`` shell must not mark the run passed —
            # that path previously hid max_turns / no-submit degrades.
            structured_ok = bool(parsed)
            channel = "json"
        state["recovered"] = state["attempted"] and structured_ok
        # Tool-channel empty submit (submitted is not None) is an honest pass.
        # JSON/no-submit with zero typed artifacts is always degraded — including
        # silent max_turns runs where final_text is empty (previously those
        # incorrectly passed because degraded required bool(final_text)).
        degraded = (not no_tools) and (not structured_ok)
        if no_tools:
            result = "failed"
            failure: Optional[str] = "no_tool_calls"
        else:
            result = "degraded" if degraded else "passed"
            failure = "empty_or_unstructured_agentic_result" if degraded else None
        evidence = evidence_base + [f"turns:{turns}", f"stop:{stop_reason}"]
        if failed_over:
            evidence.append("failover:used")
        submit_forced_budget = bool(usage.get("submit_forced_budget"))
        submit_forced_max_turns = bool(usage.get("submit_forced_max_turns"))
        if not degraded and not no_tools:
            if submit_forced_budget:
                evidence.append("submit:forced_budget")
            elif submit_forced_max_turns:
                evidence.append("submit:forced_max_turns")
            else:
                evidence.append(f"submit:{channel}")
        if state["recovered"]:
            evidence.append("retry:recovered")
        elif state["attempted"]:
            evidence.append("retry:exhausted")
        if plan_holder["steps"]:
            evidence.append(f"plan:{len(plan_holder['steps'])}")
        diagnosis = (
            _no_tool_calls_diagnosis(provider, model, turns) if no_tools else None
        )
        artifacts: list[Artifact] = [
            verification_artifact(
                task=task, worker_id=worker_id, adapter="agentic",
                check=task.instruction, result=result,
                confidence=0.55 if no_tools else (0.65 if degraded else 0.9),
                evidence=evidence,
                payload={
                    "model": model, "provider": provider, "cwd": str(cwd),
                    "turns": turns, "stop_reason": stop_reason, **usage,
                    "failure": failure,
                    "stderr": diagnosis,
                    "stdout": (final_text or "")[-_TOOL_OUTPUT_LIMIT:],
                },
            )
        ]
        if no_tools:
            artifacts.append(Artifact(
                job_id=task.job_id, task_id=task.id, type=ArtifactType.RISK,
                created_by=worker_id, confidence=0.9,
                evidence=["adapter:agentic", "result:no-tool-calls"],
                payload={
                    "risk": diagnosis,
                    "mitigation": (
                        "Route this role to a model whose registry entry carries "
                        "the 'tools' tag (known tool-caller), or pin a provider/"
                        "model that honors the tools schema on this endpoint."
                    ),
                    "failure": "no_tool_calls",
                    "stdout_excerpt": redact_secrets(final_text or "")[:2000],
                },
            ))
        elif degraded:
            artifacts.append(Artifact(
                job_id=task.job_id, task_id=task.id, type=ArtifactType.RISK,
                created_by=worker_id, confidence=0.85,
                evidence=["adapter:agentic", "result:empty-or-unstructured"],
                payload={
                    "risk": "Agentic worker completed without structured findings.",
                    "mitigation": (
                        "Treat this swarm as degraded; rerun with a higher-capability model, "
                        "a higher max_turns budget, or a stricter prompt."
                    ),
                    "stdout_excerpt": redact_secrets(final_text or "")[:2000],
                },
            ))
        artifacts.extend(parsed)
        plan_artifact = self._plan_artifact(task, worker_id, plan_holder["steps"])
        if plan_artifact is not None:
            artifacts.append(plan_artifact)
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
                prompt_with_memory(
                    with_job_brief(build_implement_prompt(base_prompt), task),
                    task,
                ),
                task,
            ),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )
        if codegraph_used:
            evidence_base = evidence_base + ["context:codegraph"]

        prompt = _with_first_turn_tool_nudge(prompt)
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

        # Verify-before-submit: resolve the repo's verification command and, when
        # present, capture a clean-tree baseline so a suite that was already red
        # can't be blamed on this change (gating vs advisory). The _on_submit
        # hook runs the command each time the model calls submit_report and, in
        # gating mode, bounces failures back for a bounded number of fixes.
        verify_command = self._resolve_verify_command(task, cwd)
        verify_retries = int(task.payload.get("verify_retries", DEFAULT_VERIFY_MAX_RETRIES))
        verify_state = {
            "command": verify_command, "attempts": 0,
            "passed": None, "output": "", "mode": "skipped",
        }
        if verify_command:
            if bool(task.payload.get("verify_baseline", True)):
                baseline_ok, _baseline_out = self._run_verification(cwd, verify_command)
                verify_state["mode"] = "gating" if baseline_ok else "advisory"
            else:
                verify_state["mode"] = "gating"

        def _on_submit(report_text: str) -> Optional[str]:
            """Gate submit_report on verification. Return rejection feedback to
            bounce the model back for another fix, or None to accept and finish.
            """
            if not verify_command:
                return None
            ok, output = self._run_verification(cwd, verify_command)
            verify_state["attempts"] += 1
            verify_state["passed"] = ok
            verify_state["output"] = output
            if ok or verify_state["mode"] == "advisory":
                return None
            if verify_state["attempts"] > verify_retries:
                return None  # budget exhausted -- accept, but stamp verify:failed
            return (
                "Your change does not pass verification. The command "
                f"`{verify_command}` failed:\n\n{output}\n\n"
                "Fix the cause of the failure, then call submit_report again. "
                "Do not call submit_report until the command passes."
            )

        plan_holder: dict = {"steps": []}

        def _on_plan(steps: list) -> None:
            plan_holder["steps"] = steps

        primary_provider, primary_model = provider, model
        try:
            loop, provider, model = self._run_loop_with_failover(
                task=task, cwd=cwd, prompt=prompt, tools=tools, implement=True,
                on_stop=_on_stop, on_submit=_on_submit, on_plan=_on_plan,
                worker_id=worker_id, provider=provider, model=model,
            )
        except ProviderError as exc:
            after = facade("git_snapshot")(cwd, base_tree=str(before.get("tree") or "") or None)
            detail = redact_secrets(exc.body or str(exc)) or str(exc)
            failure = exc.failure
            if (provider or "").lower() == "bedrock":
                from puppetmaster.provider_health import bedrock_failure_for_recovery

                failure = bedrock_failure_for_recovery(exc)
            arts = [self._fail(task, worker_id, evidence_base, failure,
                               detail, status=exc.status, provider_reason=exc.reason)]
            auth_risk = self._auth_failure_risk(
                task, worker_id, provider, exc.status or 0, detail, reason=failure)
            if auth_risk is not None:
                arts.append(auth_risk)
            if _should_emit_patch_artifact(before, after):
                arts.append(make_patch_artifact(
                    task, worker_id, before, after, adapter="agentic",
                    status="failed", change="Agentic worker modified files before failing.",
                    sidecar_name="agentic_implement",
                ))
            return arts

        final_text, usage, turns, mutated, stop_reason, _submitted = loop
        after = facade("git_snapshot")(cwd, base_tree=str(before.get("tree") or "") or None)
        has_work = _should_emit_patch_artifact(before, after)
        # A change that fails the repo's own verification (in gating mode) is not
        # a clean pass -- an edit whose tests are red is a regression, not a done
        # task. Surface it as degraded alongside the no-op case so neither a
        # no-diff run nor a red-tests run can masquerade as a success.
        verify_failed = verify_state["mode"] == "gating" and verify_state["passed"] is False
        no_tools = stop_reason == "no_tool_calls"
        degraded = (not has_work) or verify_failed
        parsed = implement_report_artifacts(task, worker_id, final_text, adapter="agentic")
        evidence = evidence_base + [f"turns:{turns}", f"stop:{stop_reason}"]
        if (provider, model) != (primary_provider, primary_model):
            evidence.append("failover:used")
        if nudged["done"]:
            evidence.append("nudge:applied")
        if usage.get("submit_forced_budget"):
            evidence.append("submit:forced_budget")
        if plan_holder["steps"]:
            evidence.append(f"plan:{len(plan_holder['steps'])}")
        evidence.append(f"verify:{_verify_evidence_tag(verify_state)}")
        diagnosis = (
            _no_tool_calls_diagnosis(provider, model, turns) if no_tools else None
        )
        if no_tools:
            failure = "no_tool_calls"
            result = "failed"
            confidence = 0.55
        else:
            failure = (
                "no_diff_produced" if not has_work
                else "verification_failed" if verify_failed
                else None
            )
            result = "degraded" if degraded else "passed"
            confidence = 0.6 if degraded else 0.9
        artifacts: list[Artifact] = [
            verification_artifact(
                task=task, worker_id=worker_id, adapter="agentic",
                check=task.instruction,
                result=result,
                confidence=confidence,
                evidence=evidence,
                payload={
                    "model": model, "provider": provider, "cwd": str(cwd),
                    "turns": turns, "stop_reason": stop_reason, "has_work": has_work,
                    "changed_files": after.get("changed_files", []),
                    "untracked_files": after.get("untracked_files", []),
                    **usage,
                    "verification_command": verify_state["command"],
                    "verification_mode": verify_state["mode"],
                    "verification_passed": verify_state["passed"],
                    "verification_attempts": verify_state["attempts"],
                    "failure": failure,
                    "stderr": diagnosis,
                    "stdout": (final_text or "")[-_TOOL_OUTPUT_LIMIT:],
                },
            )
        ]
        if no_tools:
            artifacts.append(Artifact(
                job_id=task.job_id, task_id=task.id, type=ArtifactType.RISK,
                created_by=worker_id, confidence=0.9,
                evidence=["adapter:agentic", "result:no-tool-calls"],
                payload={
                    "risk": diagnosis,
                    "mitigation": (
                        "Route this role to a model whose registry entry carries "
                        "the 'tools' tag (known tool-caller), or pin a provider/"
                        "model that honors the tools schema on this endpoint."
                    ),
                    "failure": "no_tool_calls",
                    "stdout_excerpt": redact_secrets(final_text or "")[:2000],
                },
            ))
        elif not has_work:
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
        elif verify_failed:
            artifacts.append(Artifact(
                job_id=task.job_id, task_id=task.id, type=ArtifactType.RISK,
                created_by=worker_id, confidence=0.85,
                evidence=["adapter:agentic", "result:verify-failed"],
                payload={
                    "risk": (
                        "Agentic implement worker produced a diff that does not pass "
                        f"verification (`{verify_state['command']}`)."
                    ),
                    "mitigation": (
                        "Review the diff and the failing output before applying; rerun "
                        "with a higher-capability model or a sharper task if needed."
                    ),
                    "verification_output_excerpt": redact_secrets(
                        verify_state["output"] or ""
                    )[-2000:],
                },
            ))
        artifacts.extend(parsed)
        plan_artifact = self._plan_artifact(task, worker_id, plan_holder["steps"])
        if plan_artifact is not None:
            artifacts.append(plan_artifact)
        if has_work and not no_tools:
            artifacts.append(make_patch_artifact(
                task, worker_id, before, after, adapter="agentic",
                status="applied", change="Agentic worker modified repository files.",
                sidecar_name="agentic_implement",
            ))
        return artifacts

    @staticmethod
    def _plan_artifact(
        task: Task, worker_id: str, steps: list
    ) -> "Optional[Artifact]":
        """A DECISION artifact capturing the worker's final plan, so the host can
        show the task list the model worked through. ``None`` when no plan was set.
        """
        if not steps:
            return None
        done = sum(1 for s in steps if isinstance(s, dict) and s.get("status") == "done")
        return Artifact(
            job_id=task.job_id, task_id=task.id, type=ArtifactType.DECISION,
            created_by=worker_id, confidence=0.8,
            evidence=["adapter:agentic", f"plan:{done}/{len(steps)}-done"],
            payload={
                "decision": f"Worker plan ({done}/{len(steps)} steps complete)",
                "why": "Records the step-by-step plan the worker committed to via update_plan, so the host can audit the intended approach.",
                "plan": steps,
                "plan_rendered": _render_plan(steps),
            },
        )

    # --- the tool loop -----------------------------------------------------

    def _provider_call(
        self, *, provider: str, model: str, messages: list[dict],
        tools: Optional[list[dict]], extra: dict, timeout: int, max_retries: int,
        key_pool: "Optional[list[str]]" = None,
        on_delta: Optional[Callable[[str, str], None]] = None,
    ) -> AssistantTurn:
        """One model turn, wrapped in the transient-failure retry envelope.

        Two recovery layers, tried in order on failure:

        * **Credential rotation** -- on an auth or rate-limit failure (401/403/
          429) with another key in ``key_pool``, retry immediately with the next
          key. A single throttled/revoked key never sinks a worker that has a
          good one on hand.
        * **Backoff retry** -- other transient failures (5xx/timeout/network) get
          jittered exponential backoff. Terminal failures propagate immediately.

        When ``on_delta`` is set, tokens stream to it as they arrive.

        Admission is gated by the process-local provider circuit breaker: after a
        streak of consecutive retryable failures the same provider/model/base-URL
        key is refused with a recoverable :class:`ProviderError` so failover can
        still run. Auth / malformed / other non-retryable errors never trip it.
        """
        keys: list[Optional[str]] = list(key_pool) if key_pool else [None]
        key_index = 0
        attempt = 0
        last: Optional[ProviderError] = None
        breaker = get_provider_circuit_breaker()
        admission_key = resolve_circuit_key(provider, model)
        while True:
            # Refuse before dialing when the breaker is open (recoverable error).
            breaker.before_call(admission_key)
            recorded = False
            api_key = keys[key_index]
            kwargs: dict = dict(
                provider=provider, model=model, messages=messages,
                tools=tools or None, extra=extra, timeout=timeout,
            )
            # The first attempt lets provider_chat resolve the key itself, so the
            # default single-credential path (and hermetic mocks that don't
            # accept api_key) is untouched. Only an explicit rotation to a later
            # key passes api_key. Bedrock's provider_key_pool only returns bearer
            # tokens (never access-key ids); IAM/SigV4 stays inside bedrock_chat.
            if api_key is not None and key_index > 0:
                kwargs["api_key"] = api_key
            try:
                try:
                    if on_delta is not None:
                        turn = provider_chat_streaming(on_delta=on_delta, **kwargs)
                    else:
                        turn = provider_chat(**kwargs)
                    breaker.record_success(admission_key)
                    recorded = True
                    return turn
                except ProviderError as exc:
                    last = exc
                    breaker.record_failure(admission_key, exc)
                    recorded = True
                    if exc.status in (401, 403, 429) and key_index + 1 < len(keys):
                        key_index += 1
                        continue
                    if attempt >= max_retries or not is_retryable_provider_error(exc):
                        raise
                    time.sleep(provider_retry_backoff_seconds(attempt))
                    attempt += 1
            finally:
                if not recorded:
                    breaker.release_admission(admission_key)
        assert last is not None
        raise last

    def _agent_loop(
        self, task: Task, cwd: Path, provider: str, model: str, system_prompt: str,
        tools: list[dict], *, implement: bool,
        on_stop: Optional[Callable[[str, bool], Optional[str]]] = None,
        on_delta: Optional[Callable[[str, str], None]] = None,
        on_submit: Optional[Callable[[str], Optional[str]]] = None,
        on_plan: Optional[Callable[[list], None]] = None,
    ) -> tuple[str, dict, int, bool, str, Optional[list[dict]]]:
        """Run the provider tool-use loop until the model finishes.

        Returns ``(final_text, usage_totals, turns, mutated, stop_reason,
        submitted)``. ``submitted`` is ``None`` when the model never called
        ``submit_findings``; otherwise it is the (possibly empty) list of
        submitted artifact items -- so an explicit empty submission ("I found
        nothing") is distinguishable from a model that just went silent.

        Each turn: send the conversation, execute any tool calls, append their
        results, repeat. Structured output rides the native tool channel: a
        ``submit_findings`` / ``submit_report`` call is the terminal signal. A
        budget governor bounds the run three ways -- max turns, a per-call
        wall-clock timeout, and an optional cumulative ``token_budget``.

        Robustness envelope (Hermes parity): transient wire failures are retried
        with backoff (:meth:`_provider_call`); a model that goes silent right
        after a tool result is nudged once to continue; a length-truncated turn
        is continued a bounded number of times; when the analyze retry fires
        the next turn *forces* the submit tool via ``tool_choice`` so a compliant
        model can't wander back into prose; and when cumulative usage enters the
        submit-reserve fraction of ``token_budget`` the next turn is similarly
        forced to ``submit_findings`` / ``submit_report`` (tagged
        ``submit:forced_budget`` on success).
        """
        self._reset_session_caches()
        max_retries = int(task.payload.get("provider_max_retries", DEFAULT_PROVIDER_MAX_RETRIES))
        key_pool = provider_key_pool(provider)
        context_budget = int(
            task.payload.get("context_token_budget", DEFAULT_CONTEXT_TOKEN_BUDGET)
        )
        context_compressions = 0
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
        submit_reserve = _submit_reserve_fraction()
        budget_force_threshold = (
            _budget_force_threshold(token_budget, submit_reserve)
            if token_budget else None
        )
        no_tool_streak_limit = int(
            task.payload.get("no_tool_calls_streak", DEFAULT_NO_TOOL_CALLS_STREAK)
        )
        if no_tool_streak_limit < 1:
            no_tool_streak_limit = DEFAULT_NO_TOOL_CALLS_STREAK
        base_extra = self._extra_params(task)
        # Static + job-stable prefix as a true system message; per-task CodeGraph
        # + instruction as the first user message — enables provider prompt caches
        # (OpenAI implicit prefix / Anthropic body['system'] breakpoints).
        system_prefix, user_suffix = split_prompt_messages(system_prompt)
        if system_prefix:
            messages: list[dict] = [
                {"role": "system", "content": system_prefix},
                {"role": "user", "content": user_suffix or system_prompt},
            ]
        else:
            messages = [{"role": "user", "content": system_prompt}]
        usage_total = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "cached_tokens": 0, "cost_usd": 0.0,
        }
        final_text = ""
        mutated = False
        turns = 0
        stop_reason = "max_turns"
        submitted: Optional[list[dict]] = None
        empty_recoveries = 0
        length_continuations = 0
        force_submit_next = False
        budget_force_pending = False
        budget_force_attempted = False
        submit_forced_budget = False
        consecutive_no_tool_turns = 0

        # Cancellation points: a host that requested cancel (see
        # puppetmaster.cancellation) stops this worker (a) mid-stream, by the
        # wrapped delta sink raising out of the provider stream within one
        # chunk, and (b) before each provider call. Threads can't be killed;
        # these two checks make cancel land in seconds, not turns.
        job_id = getattr(task, "job_id", "") or ""
        if on_delta is not None and job_id:
            _inner_delta = on_delta

            def on_delta(kind: str, text: str) -> None:  # noqa: F811 - deliberate shadow
                if is_cancelled(job_id):
                    raise JobCancelled(job_id)
                _inner_delta(kind, text)

        for turns in range(1, max_turns + 1):
            if job_id and is_cancelled(job_id):
                stop_reason = "cancelled"
                break
            # Shed older tool outputs before the call when the running
            # conversation is nearing the context budget *or* the turn count
            # crosses the compaction threshold, so a long run degrades
            # gracefully instead of 413-ing mid-flight. Static system prefix
            # bytes stay untouched (prompt-cache invariant).
            keep_recent = int(task.payload.get("history_keep_recent", DEFAULT_KEEP_RECENT))
            compact_after = int(
                task.payload.get("history_compact_after_turns", DEFAULT_COMPACT_AFTER_TURNS)
            )
            messages, compressed = compress_history(
                messages,
                budget_tokens=context_budget,
                keep_recent=keep_recent,
                turn_count=turns,
                compact_after_turns=compact_after,
            )
            if compressed:
                context_compressions += 1
            call_extra = dict(base_extra)
            # Force the submit tool after a structure retry *or* when the token
            # budget has entered the submit-reserve zone, so a CoT-heavy model
            # is compelled to call submit_findings / submit_report instead of
            # burning the rest of the budget on prose.
            this_turn_budget_force = budget_force_pending
            if force_submit_next or budget_force_pending:
                call_extra["force_tool"] = (
                    _SUBMIT_REPORT_TOOL if implement else _SUBMIT_FINDINGS_TOOL
                )
            if budget_force_pending:
                budget_force_attempted = True
            force_submit_next = False
            budget_force_pending = False

            try:
                turn: AssistantTurn = self._provider_call(
                    provider=provider, model=model, messages=messages,
                    tools=tools or None, extra=call_extra, timeout=timeout,
                    max_retries=max_retries, key_pool=key_pool, on_delta=on_delta,
                )
            except JobCancelled:
                stop_reason = "cancelled"
                break
            for key in usage_total:
                if key == "cost_usd":
                    usage_total[key] += float(turn.usage.get(key, 0.0) or 0.0)
                else:
                    usage_total[key] += int(turn.usage.get(key, 0) or 0)
            final_text = turn.text or final_text

            if not turn.tool_calls:
                text_present = bool((turn.text or "").strip())
                # A model that returns nothing right after a tool result usually
                # just needs a poke to keep going or to submit -- recover once.
                if (
                    not text_present
                    and empty_recoveries < _MAX_EMPTY_RECOVERIES
                    and _last_message_role(messages) == "tool"
                ):
                    empty_recoveries += 1
                    messages.append({"role": "user", "content": _EMPTY_RESPONSE_NUDGE})
                    continue
                # A truncated final turn (hit the output cap) is continued a
                # bounded number of times so a long report isn't lost mid-word.
                if (
                    turn.finish_reason == "length"
                    and length_continuations < _MAX_LENGTH_CONTINUATIONS
                ):
                    length_continuations += 1
                    messages.append({"role": "user", "content": _LENGTH_CONTINUATION_NUDGE})
                    continue
                # Prose/reasoning with zero tool_calls. Count consecutive misses
                # so a pure reasoner fails fast instead of burning token_budget.
                # A budget-forced turn that still returns prose falls through to
                # the same no_tool_calls / token_budget handling below.
                consecutive_no_tool_turns += 1
                if consecutive_no_tool_turns >= no_tool_streak_limit:
                    stop_reason = "no_tool_calls"
                    break
                follow_up = on_stop(final_text, mutated) if on_stop else None
                if follow_up:
                    messages.append({"role": "user", "content": follow_up})
                    force_submit_next = not implement
                    # Force-submit check runs BEFORE the hard budget break so a
                    # single huge turn that leaps past the whole budget still
                    # gets one final forced submit instead of dying silently.
                    if (
                        budget_force_threshold is not None
                        and not budget_force_attempted
                        and usage_total["total_tokens"] >= budget_force_threshold
                    ):
                        budget_force_pending = True
                        messages.append({
                            "role": "user",
                            "content": (
                                _BUDGET_FORCE_SUBMIT_IMPLEMENT if implement
                                else _BUDGET_FORCE_SUBMIT_ANALYZE
                            ),
                        })
                        continue
                    if token_budget and usage_total["total_tokens"] >= token_budget:
                        stop_reason = "token_budget"
                        break
                    continue
                if (
                    budget_force_threshold is not None
                    and not budget_force_attempted
                    and usage_total["total_tokens"] >= budget_force_threshold
                ):
                    budget_force_pending = True
                    messages.append({
                        "role": "user",
                        "content": (
                            _BUDGET_FORCE_SUBMIT_IMPLEMENT if implement
                            else _BUDGET_FORCE_SUBMIT_ANALYZE
                        ),
                    })
                    continue
                if token_budget and usage_total["total_tokens"] >= token_budget:
                    stop_reason = "token_budget"
                    break
                stop_reason = "model_stopped"
                break

            # A tool-calling turn is progress on the wire; reset the prose streak.
            # Implement-mode mutations also count as progress (mutated is set
            # below when a mutating tool succeeds).
            consecutive_no_tool_turns = 0

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
            submitted_this_turn = False
            # Walk calls in emission order. Contiguous regular tools use
            # Hermes-style segments (parallel-safe runs concurrent); submit/
            # plan stay barriers so tool-result order matches the model turn.
            idx = 0
            n_calls = len(turn.tool_calls)
            while idx < n_calls:
                name = turn.tool_calls[idx]["name"]
                if name in _SUBMIT_TOOLS or name == _PLAN_TOOL:
                    call = turn.tool_calls[idx]
                    idx += 1
                    if name in _SUBMIT_TOOLS:
                        if name == _SUBMIT_FINDINGS_TOOL:
                            items = _coerce_submit_findings(call.get("arguments"))
                            if submitted is None:
                                submitted = []
                            submitted.extend(items)
                            ack = f"Recorded {len(items)} artifact(s). Analysis complete."
                        else:  # submit_report
                            report = _coerce_submit_report(call.get("arguments"))
                            rejection = on_submit(report) if on_submit else None
                            if rejection is not None:
                                messages.append({
                                    "role": "tool", "tool_call_id": call["id"],
                                    "content": self._model_facing_tool_result(
                                        rejection, task=task, cwd=cwd,
                                        tool_name=name, tool_call_id=call["id"],
                                    ),
                                })
                                continue
                            if report:
                                final_text = report
                            ack = "Report recorded. Task complete."
                        submitted_this_turn = True
                        messages.append({
                            "role": "tool", "tool_call_id": call["id"], "content": ack,
                        })
                        continue
                    steps = _coerce_plan_steps(call.get("arguments"))
                    if on_plan is not None:
                        on_plan(steps)
                    messages.append({
                        "role": "tool", "tool_call_id": call["id"],
                        "content": "Plan updated.\n" + _render_plan(steps),
                    })
                    continue

                regular_calls = []
                while idx < n_calls:
                    nxt = turn.tool_calls[idx]["name"]
                    if nxt in _SUBMIT_TOOLS or nxt == _PLAN_TOOL:
                        break
                    regular_calls.append(turn.tool_calls[idx])
                    idx += 1
                segments = plan_tool_batch_segments(regular_calls)
                for segment_kind, segment_calls in segments:
                    if segment_kind == "parallel":
                        results = self._execute_tool_segment_parallel(
                            segment_calls, cwd, implement, task
                        )
                    else:
                        results = self._execute_tool_segment_sequential(
                            segment_calls, cwd, implement, task
                        )
                    for call, output in results:
                        tname = call["name"]
                        if tname in _MUTATING_TOOLS and not output.startswith("error"):
                            mutated = True
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": self._model_facing_tool_result(
                                redact_secrets(output) or "",
                                task=task, cwd=cwd,
                                tool_name=tname, tool_call_id=call["id"],
                            ),
                        })
            if submitted_this_turn:
                stop_reason = "submitted"
                if this_turn_budget_force:
                    submit_forced_budget = True
                break
            # Force-submit check runs BEFORE the hard budget break (same
            # rationale as above): exhaustion without a prior forced attempt
            # still earns exactly one forced submit turn.
            if (
                budget_force_threshold is not None
                and not budget_force_attempted
                and usage_total["total_tokens"] >= budget_force_threshold
            ):
                budget_force_pending = True
                messages.append({
                    "role": "user",
                    "content": (
                        _BUDGET_FORCE_SUBMIT_IMPLEMENT if implement
                        else _BUDGET_FORCE_SUBMIT_ANALYZE
                    ),
                })
                continue
            if token_budget and usage_total["total_tokens"] >= token_budget:
                stop_reason = "token_budget"
                break

        # Last-chance forced submit: models that keep calling tools until
        # max_turns never hit the prose/budget force paths and leave
        # stop:max_turns with zero findings. Grant analyze mode exactly one
        # extra submit_findings turn. Prefer tool_choice force; if the
        # provider rejects force_tool (some OpenRouter models 400), retry
        # once with the nudge alone.
        submit_forced_max_turns = False
        if (
            not implement
            and submitted is None
            and stop_reason == "max_turns"
            and tools
        ):
            messages.append({
                "role": "user",
                "content": _BUDGET_FORCE_SUBMIT_ANALYZE,
            })
            force_attempts: list[dict] = [
                {**dict(base_extra), "force_tool": _SUBMIT_FINDINGS_TOOL},
                dict(base_extra),
            ]
            for call_extra in force_attempts:
                try:
                    turn = self._provider_call(
                        provider=provider, model=model, messages=messages,
                        tools=tools or None, extra=call_extra, timeout=timeout,
                        max_retries=max_retries, key_pool=key_pool, on_delta=on_delta,
                    )
                except JobCancelled:
                    stop_reason = "cancelled"
                    break
                except ProviderError:
                    continue
                turns += 1
                for key in usage_total:
                    if key == "cost_usd":
                        usage_total[key] += float(turn.usage.get(key, 0.0) or 0.0)
                    else:
                        usage_total[key] += int(turn.usage.get(key, 0) or 0)
                final_text = turn.text or final_text
                for call in turn.tool_calls or []:
                    if call.get("name") == _SUBMIT_FINDINGS_TOOL:
                        items = _coerce_submit_findings(call.get("arguments"))
                        if submitted is None:
                            submitted = []
                        submitted.extend(items)
                        stop_reason = "submitted"
                        submit_forced_max_turns = True
                        break
                if submit_forced_max_turns or stop_reason == "cancelled":
                    break
                # Model returned prose/tools without submit — stop retrying.
                break

        usage_out = {
            "tokens_in": usage_total["prompt_tokens"],
            "tokens_out": usage_total["completion_tokens"],
            "tokens_total": usage_total["total_tokens"],
            "context_compressions": context_compressions,
            "tokens_cached": usage_total["cached_tokens"],
            "submit_forced_budget": submit_forced_budget,
            "submit_forced_max_turns": submit_forced_max_turns,
        }
        if usage_total["cost_usd"] > 0:
            usage_out["real_cost_usd"] = round(usage_total["cost_usd"], 6)
        return final_text, usage_out, turns, mutated, stop_reason, submitted

    def _loop_targets(
        self, task: Task, provider: str, model: str
    ) -> "list[tuple[str, str]]":
        """The primary (provider, model) plus any configured failover targets.

        ``payload['failover_models']`` is an opt-in list of ``{"provider"?,
        "model"}`` dicts. When a target omits ``provider`` it inherits the
        primary's. Failover is off unless the caller supplies this list, so
        default runs are unchanged.
        """
        targets: list[tuple[str, str]] = [(provider, model)]
        for entry in task.payload.get("failover_models") or []:
            if isinstance(entry, dict) and entry.get("model"):
                targets.append((str(entry.get("provider") or provider), str(entry["model"])))
        return targets

    def _run_loop_with_failover(
        self, *, task: Task, cwd: Path, prompt: str, tools: list[dict],
        implement: bool, on_stop: Optional[Callable[[str, bool], Optional[str]]],
        worker_id: str, provider: str, model: str,
        on_submit: Optional[Callable[[str], Optional[str]]] = None,
        on_plan: Optional[Callable[[list], None]] = None,
    ) -> "tuple[tuple, str, str]":
        """Run the agent loop, failing over to each configured alternate on a
        hard :class:`ProviderError`. Returns ``(loop_result, used_provider,
        used_model)``; raises the last error only when every target fails.

        Failover restarts the loop from a clean conversation on the new provider
        (rather than switching mid-conversation, which would leave provider-
        shaped tool-call history that the next provider can't parse).
        """
        last: Optional[ProviderError] = None
        # Durable token stream: persist deltas to an NDJSON file under the job
        # state dir so a subprocess/CLI/MCP follower can tail them live -- the
        # streaming parity the in-process delta bus can only give inline hosts.
        durable = (
            DurableDeltaWriter.for_task(task, worker_id)
            if bool(task.payload.get("stream_deltas", True)) else None
        )
        try:
            for target_provider, target_model in self._loop_targets(task, provider, model):
                try:
                    loop = self._agent_loop(
                        task, cwd, target_provider, target_model, prompt, tools,
                        implement=implement, on_stop=on_stop, on_submit=on_submit,
                        on_plan=on_plan,
                        on_delta=self._compose_delta_sink(worker_id, durable),
                    )
                    return loop, target_provider, target_model
                except ProviderError as exc:
                    last = exc
                    continue
            assert last is not None
            raise last
        finally:
            if durable is not None:
                durable.close()

    @staticmethod
    def _compose_delta_sink(
        worker_id: str, durable: "Optional[DurableDeltaWriter]"
    ) -> "Optional[Callable[[str, str], None]]":
        """Fan a run's token deltas to both the in-process bus (inline hosts) and
        the durable NDJSON file (subprocess/CLI/MCP followers). Returns ``None``
        when neither sink is active, so the loop keeps its non-streaming path.
        """
        inproc = delta_sink_for(worker_id)
        if inproc is None and durable is None:
            return None

        def sink(kind: str, text: str) -> None:
            if inproc is not None:
                try:
                    inproc(kind, text)
                except Exception:  # noqa: BLE001 - a UI sink error must not sink the run
                    pass
            if durable is not None:
                durable.emit(kind, text)

        return sink

    def _tool_schema(self, *, implement: bool, task: Task, graph_on: bool = False) -> list[dict]:
        """OpenAI-format tool specs; provider_chat translates for Anthropic."""
        def fn(name, desc, props, required):
            return {"type": "function", "function": {
                "name": name, "description": desc,
                "parameters": {"type": "object", "properties": props, "required": required},
            }}

        read_file_desc = "Read a UTF-8 text file within the workspace."
        if hashline_enabled():
            read_file_desc = (
                "Read a UTF-8 text file within the workspace. Returns `[path#TAG]` "
                "plus `N:line` rows — use that TAG with `apply_hashline` or "
                "`edit_file.expected_tag` for safe mutations."
            )
        tools = [
            fn("read_file", read_file_desc,
               {"path": {"type": "string"},
                "start_line": {"type": "integer", "description": "1-indexed start line (optional)"},
                "limit": {"type": "integer", "description": "max lines to read (optional)"}},
               ["path"]),
            fn("read_offload",
               "Read a previously offloaded tool-output blob from this worker's "
               "state_dir/tool_offload directory. Use the path from an "
               "[tool output offloaded] stub. Refuses workspace escapes and "
               "foreign paths — only self-offload blobs are readable.",
               {"path": {"type": "string",
                         "description": "Absolute or blob-relative path under state_dir/tool_offload"},
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
            edit_props = {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {
                    "type": "boolean",
                    "description": "replace all occurrences (default false)",
                },
            }
            edit_desc = (
                "Replace an exact occurrence of old_string with new_string in a file. "
                "Set replace_all=true to replace every occurrence."
            )
            if hashline_enabled():
                edit_props["expected_tag"] = {
                    "type": "string",
                    "description": (
                        "Optional 4-hex content tag from a prior tagged read_file "
                        "([path#TAG]). When set, the edit is refused if the live "
                        "normalized file no longer matches — re-read and retry. "
                        "Prefer apply_hashline for surgical line edits."
                    ),
                }
                edit_desc = (
                    "Replace an exact occurrence of old_string with new_string in a "
                    "file. Set replace_all=true to replace every occurrence. After a "
                    "tagged read_file, pass expected_tag from that read's #TAG for "
                    "optimistic concurrency (or prefer apply_hashline for surgical "
                    "line edits)."
                )
            tools.append(fn(
                "edit_file", edit_desc, edit_props,
                ["path", "old_string", "new_string"],
            ))
            if hashline_enabled():
                tools.append(fn(
                    "apply_hashline",
                    "Apply a Hashline patch (content-hash-anchored line edits). Prefer this "
                    "for surgical edits after a tagged read_file. Format: [path#TAG] then "
                    "SWAP N.=M: / DEL N.=M / INS.PRE|POST|HEAD|TAIL / REM / MV. Body rows "
                    "are +TEXT. Numbers refer to ORIGINAL lines; stale tags are rejected. "
                    "Block ops (*.BLK) are unsupported — use line ranges.",
                    {"patch": {"type": "string", "description": "Full hashline patch text"}},
                    ["patch"],
                ))
            tools.append(fn("delete_file", "Delete a file within the workspace.",
                            {"path": {"type": "string"}}, ["path"]))
            if self._terminal_enabled(task):
                tools.append(fn("run_terminal", "Run a bounded shell command in the workspace (e.g. to run focused tests) and return its output. Destructive commands are refused.",
                                {"command": {"type": "string"}}, ["command"]))
        if bool(task.payload.get("allow_web", False)):
            tools.append(fn("web_fetch", "Fetch a URL and return its text content.",
                            {"url": {"type": "string"}}, ["url"]))
        if self._browser_enabled(task):
            tools.append(fn("browser_navigate", "Open a URL in a real headless browser. Call browser_snapshot next to see clickable elements.",
                            {"url": {"type": "string"}}, ["url"]))
            tools.append(fn("browser_snapshot", "Return the current page's interactable elements with @e1-style refs. Snapshot before clicking/typing so you have fresh refs.",
                            {}, []))
            tools.append(fn("browser_click", "Click the element with the given ref (from browser_snapshot, e.g. @e3).",
                            {"ref": {"type": "string"}}, ["ref"]))
            tools.append(fn("browser_type", "Type text into the input/textarea element with the given ref.",
                            {"ref": {"type": "string"}, "text": {"type": "string"}}, ["ref", "text"]))
            tools.append(fn("browser_scroll", "Scroll the page 'up' or 'down'.",
                            {"direction": {"type": "string", "description": "up or down"}}, []))
            tools.append(fn("browser_back", "Navigate the browser back one page.", {}, []))
            tools.append(fn("browser_get_text", "Return the page's main readable text (document body innerText).", {}, []))
            tools.append(fn("browser_screenshot", "Capture a PNG screenshot of the current page; returns a file path you can view_image.", {}, []))
        if bool(task.payload.get("plan_tool", True)):
            tools.append(fn(
                _PLAN_TOOL,
                "Record or update your step-by-step plan for this task. Call it "
                "once as you start on any multi-step work, then again to mark "
                "steps in_progress/done as you go. Keeps a long run organized and "
                "gives the user a live task list. It changes nothing in the repo.",
                {"steps": {"type": "array", "description": "the ordered plan steps",
                           "items": {"type": "object", "properties": {
                               "step": {"type": "string"},
                               "status": {"type": "string",
                                          "description": "one of pending, in_progress, done"}}}}},
                ["steps"]))
        tools.append(self._submit_tool(implement=implement, fn=fn))
        return tools

    @staticmethod
    def _submit_tool(*, implement: bool, fn: Callable) -> dict:
        """The terminal tool that carries structured output on the native
        tool-calling channel. Analyze workers finish by calling
        ``submit_findings`` with a schema-constrained ``artifacts`` array;
        implement workers finish by calling ``submit_report``. Provider-native
        argument schemas make this reliable where a free-text JSON contract is
        not -- the model literally cannot return unparseable prose here.
        """
        if implement:
            return fn(
                _SUBMIT_REPORT_TOOL,
                "Submit your final report and finish the task. Call this ONCE, "
                "after you have made all your edits, to record what you changed, "
                "which files you touched, and how you verified the change.",
                {
                    "summary": {"type": "string", "description": "What you changed and why."},
                    "files_changed": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Workspace-relative paths you created, edited, or deleted.",
                    },
                    "verification": {
                        "type": "string",
                        "description": "Exactly what you ran to verify the change (e.g. the test command and its result).",
                    },
                },
                ["summary"],
            )
        return fn(
            _SUBMIT_FINDINGS_TOOL,
            "Submit your final structured findings and finish the task. Call this "
            "EXACTLY ONCE when your analysis is complete. Pass an 'artifacts' "
            "array. If you genuinely found nothing for your role, submit an empty "
            "array -- do not invent a finding.",
            {
                "artifacts": {
                    "type": "array",
                    "description": "Zero or more finding/risk/decision artifacts grounded in concrete files or symbols.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["finding", "risk", "decision"],
                                "description": "Artifact kind.",
                            },
                            "claim": {"type": "string", "description": "For type=finding: the claim."},
                            "risk": {"type": "string", "description": "For type=risk: the risk."},
                            "mitigation": {"type": "string", "description": "For type=risk: how to mitigate it."},
                            "decision": {"type": "string", "description": "For type=decision: the decision."},
                            "why": {"type": "string", "description": "For type=decision: the rationale."},
                            "evidence": {
                                "type": "array", "items": {"type": "string"},
                                "description": "Concrete file paths or symbols that ground this artifact.",
                            },
                            "confidence": {
                                "type": "number",
                                "description": "0.0-1.0 confidence in this artifact.",
                            },
                        },
                        "required": ["type", "evidence"],
                    },
                }
            },
            ["artifacts"],
        )

    def _terminal_enabled(self, task: Task) -> bool:
        """Whether implement-mode workers may self-verify with ``run_terminal``.

        On by default in implement mode -- a harness that cannot run its own
        tests is not a legitimate implement engine -- but the destructive-command
        denylist still guards every invocation, and ``allow_terminal=false``
        turns it off entirely for locked-down runs.
        """
        return bool(task.payload.get("allow_terminal", True))

    def _browser_enabled(self, task: Task) -> bool:
        """Whether this worker gets the CDP browser toolset (navigate/snapshot/
        click/type/scroll/back/get_text/screenshot).

        Enabled when the spec opts in via ``payload.allow_browser`` OR lists
        ``browser`` in its comma-separated ``payload.toolsets`` (the convention
        PM's browser-swarm specs already use). This lets browser-capable swarms
        run on the standalone ``agentic`` adapter + the user's own keys, without
        the Hermes adapter / agent-browser CLI.
        """
        payload = task.payload or {}
        if payload.get("allow_browser"):
            return True
        toolsets = payload.get("toolsets")
        if isinstance(toolsets, str):
            return "browser" in {p.strip() for p in toolsets.split(",")}
        return False

    def _resolve_verify_command(self, task: Task, cwd: Path) -> Optional[str]:
        """The command that verifies an implement change (tests/typecheck/lint).

        Resolution order: an explicit ``payload['verify_command']`` wins; a
        falsey ``payload['verify']`` (``False`` / ``"off"`` / ``"none"``) disables
        verification; otherwise, when ``verify`` is unset or ``"auto"``, detect a
        standard command for the repo. Verification runs a shell command, so it
        requires the terminal enabled -- a locked-down run has no verification.
        """
        if not self._terminal_enabled(task):
            return None
        explicit = task.payload.get("verify_command")
        if explicit:
            return str(explicit)
        mode = task.payload.get("verify", "auto")
        if mode in (False, "off", "none", "false", "0", 0):
            return None
        return _detect_verify_command(cwd)

    def _run_verification(self, cwd: Path, command: str) -> "tuple[bool, str]":
        """Run ``command`` in ``cwd`` (bounded + destructive-guarded). Returns
        ``(passed, output)`` where ``passed`` is exit-code 0. Never raises: an
        unrunnable command is reported as a failure with its cause as output."""
        blocked = _destructive_command_match(command)
        if blocked is not None:
            return False, f"refused destructive verification command (matched {blocked})"
        try:
            proc = subprocess.run(
                command, shell=True, cwd=str(cwd), capture_output=True,
                text=True, timeout=_VERIFY_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return False, f"verification timed out ({_VERIFY_TIMEOUT_SECONDS}s): {command}"
        except Exception as exc:  # noqa: BLE001 - report, never crash the worker
            return False, f"verification could not run: {type(exc).__name__}: {exc}"
        out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        return proc.returncode == 0, out

    def _execute_tool(
        self, name: str, args: dict, cwd: Path, implement: bool, task: Task
    ) -> str:
        """Dispatch one tool call. Returns a text result (never raises)."""
        try:
            if name == "read_file":
                return self._tool_read_file(args, cwd)
            if name == "read_offload":
                return self._tool_read_offload(args, cwd)
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
            if name == "apply_hashline" and implement and hashline_enabled():
                return self._tool_apply_hashline(args, cwd)
            if name == "delete_file" and implement:
                return self._tool_delete_file(args, cwd)
            if name == "run_terminal" and implement and self._terminal_enabled(task):
                return self._tool_run_terminal(args, cwd)
            if name == "web_fetch" and bool(task.payload.get("allow_web", False)):
                return self._tool_web_fetch(args)
            if name in _BROWSER_TOOL_NAMES and self._browser_enabled(task):
                from puppetmaster import browser_cdp as _bcdp
                out_dir = None
                try:
                    _cwd = task.payload.get("cwd")
                    if _cwd:
                        out_dir = str(_cwd)
                except Exception:
                    out_dir = None
                result = _bcdp.dispatch(name, args, out_dir=out_dir)
                return result if result is not None else f"error: unknown browser tool {name!r}"
            return f"error: tool {name!r} is not available in this mode"
        except Exception as exc:  # a tool failure must not kill the worker
            return f"error: {type(exc).__name__}: {exc}"

    def _execute_tool_segment_parallel(
        self, calls: list[dict], cwd: Path, implement: bool, task: Task
    ) -> list[tuple[dict, str]]:
        """Execute a segment of parallel-safe tool calls concurrently.
        
        Returns a list of (call, output) tuples in the original call order.
        """
        def _execute_one(call: dict) -> tuple[dict, str]:
            try:
                name = call["name"]
                args = call.get("arguments") or {}
                output = self._execute_tool(name, args, cwd, implement, task)
                return (call, output)
            except Exception as exc:
                return (call, f"error: {type(exc).__name__}: {exc}")
        
        try:
            # Cap threads so large safe-read batches cannot unbounded-thread.
            # Segmentation / barrier rules are unchanged; only pool size is bounded.
            worker_count = parallel_executor_max_workers(len(calls))
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {executor.submit(_execute_one, call): idx for idx, call in enumerate(calls)}
                results = [None] * len(calls)
                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as exc:
                        call = calls[idx]
                        results[idx] = (call, f"error: {type(exc).__name__}: {exc}")
            return results
        except Exception as exc:
            # Fall back to sequential execution on any threading error
            return self._execute_tool_segment_sequential(calls, cwd, implement, task)

    def _execute_tool_segment_sequential(
        self, calls: list[dict], cwd: Path, implement: bool, task: Task
    ) -> list[tuple[dict, str]]:
        """Execute a segment of barrier tool calls sequentially.
        
        Returns a list of (call, output) tuples in the original call order.
        """
        results = []
        for call in calls:
            try:
                name = call["name"]
                args = call.get("arguments") or {}
                output = self._execute_tool(name, args, cwd, implement, task)
                results.append((call, output))
            except Exception as exc:
                results.append((call, f"error: {type(exc).__name__}: {exc}"))
        return results

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
        text = self._read_text_cached(path)
        lines = text.splitlines()
        start = max(1, int(args["start_line"])) if args.get("start_line") else 1
        limit = int(args["limit"]) if args.get("limit") else len(lines)
        chunk = lines[start - 1:start - 1 + limit]
        # Full body returns to the loop; model-facing offload/hard-cap happens
        # in ``_model_facing_tool_result`` (savings-gated spill, not blunt truncate).
        if hashline_enabled():
            rel = _rel(path, cwd).replace("\\", "/")
            tag = self._hashline_store.record(rel, normalize_text(text))
            return format_numbered_read(rel, tag, chunk, start_line=start)
        return "\n".join(chunk)

    def _tool_read_offload(self, args: dict, cwd: Path) -> str:
        """Read a durable self-offload blob; confined to state_dir/tool_offload."""
        state_dir = _resolve_sidecar_state_dir()
        if state_dir is None:
            try:
                state_dir = resolve_state_dir(cwd=cwd)
            except Exception:
                state_dir = None
        if state_dir is None:
            return "error: no state_dir available to read offload blobs"
        start = args.get("start_line", 1)
        limit = args.get("limit")
        return read_offload_blob(
            str(args.get("path", "")),
            state_dir=state_dir,
            start_line=start if start is not None else 1,
            limit=limit,
        )

    def _read_text_cached(self, path: Path) -> str:
        """Read UTF-8 text, optionally caching by (mtime_ns, size)."""
        if not fs_cache_enabled():
            return path.read_text(encoding="utf-8", errors="replace")
        key = str(path.resolve())
        try:
            st = path.stat()
            mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
            size = int(st.st_size)
        except OSError:
            return path.read_text(encoding="utf-8", errors="replace")
        cached = self._fs_cache.get(key)
        if cached is not None and cached[0] == mtime_ns and cached[1] == size:
            return cached[2]
        text = path.read_text(encoding="utf-8", errors="replace")
        self._fs_cache[key] = (mtime_ns, size, text)
        return text

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
        return str(result.get("stdout") or "(no matches)")

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
        return context

    def _tool_write_file(self, args: dict, cwd: Path) -> str:
        path = self._confine(cwd, str(args.get("path", "")))
        content = str(args.get("content", ""))
        if "\x00" in content:
            return "error: refusing to write NUL bytes (binary content) via write_file"
        if path.exists() and _looks_binary(path):
            return "error: refusing to overwrite an apparent binary file"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self._invalidate_fs_cache(path)
        rel = _rel(path, cwd).replace("\\", "/")
        if hashline_enabled():
            tag = self._hashline_store.record(rel, normalize_text(content))
            return f"wrote {rel} ({len(content)} chars) [{rel}#{tag}]"
        return f"wrote {rel} ({len(content)} chars)"

    def _tool_edit_file(self, args: dict, cwd: Path) -> str:
        path = self._confine(cwd, str(args.get("path", "")))
        old = str(args.get("old_string", ""))
        new = str(args.get("new_string", ""))
        replace_all = bool(args.get("replace_all", False))
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = _rel(path, cwd).replace("\\", "/")
        expected_raw = args.get("expected_tag")
        if (
            hashline_enabled()
            and expected_raw is not None
            and str(expected_raw).strip()
        ):
            expected = str(expected_raw).strip().upper()
            live_tag = content_tag(text)
            if live_tag != expected:
                return (
                    f"error: StaleTagError: {rel}#{expected}: stale expected_tag — "
                    f"live file is #{live_tag}; re-read before editing"
                )
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
        self._invalidate_fs_cache(path)
        n = count if replace_all else 1
        if hashline_enabled():
            tag = self._hashline_store.record(rel, normalize_text(updated))
            return (
                f"edited {rel} ({n} replacement{'s' if n != 1 else ''}) "
                f"[{rel}#{tag}]"
            )
        return f"edited {rel} ({n} replacement{'s' if n != 1 else ''})"

    def _tool_apply_hashline(self, args: dict, cwd: Path) -> str:
        patch = str(args.get("patch", ""))
        if not patch.strip():
            return "error: empty hashline patch"
        try:
            result = apply_patch(cwd, patch, self._hashline_store)
        except Exception as exc:  # surface parse/stale/bounds cleanly to the model
            return f"error: {type(exc).__name__}: {exc}"
        for path in result.touched:
            self._invalidate_fs_cache(path)
        return format_apply_success(result)

    def _tool_delete_file(self, args: dict, cwd: Path) -> str:
        path = self._confine(cwd, str(args.get("path", "")))
        if not path.exists():
            return "error: file does not exist"
        if path.is_dir():
            return "error: path is a directory; delete_file only removes files"
        rel = _rel(path, cwd).replace("\\", "/")
        path.unlink()
        self._invalidate_fs_cache(path)
        self._hashline_store.invalidate(rel)
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
        return f"exit={proc.returncode}\n{out}"

    def _tool_web_fetch(self, args: dict) -> str:
        import urllib.request
        url = str(args.get("url", ""))
        if not (url.startswith("http://") or url.startswith("https://")):
            return "error: url must be http(s)"
        req = urllib.request.Request(url, headers={"User-Agent": "puppetmaster-agentic"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return body

    def _model_facing_tool_result(
        self,
        text: str,
        *,
        task: Task,
        cwd: Path,
        tool_name: str = "",
        tool_call_id: str = "",
    ) -> str:
        """Savings-gated offload (or hard-cap) for a model-facing tool result."""
        state_dir = _resolve_sidecar_state_dir()
        if state_dir is None:
            try:
                state_dir = resolve_state_dir(cwd=cwd)
            except Exception:
                state_dir = None
        model_text, _meta = offload_tool_output(
            text or "",
            state_dir=state_dir,
            job_id=getattr(task, "job_id", "") or "",
            task_id=getattr(task, "id", "") or "",
            tool_name=tool_name,
            tool_call_id=tool_call_id,
        )
        return model_text

    # --- helpers -----------------------------------------------------------

    def _fail(
        self, task: Task, worker_id: str, evidence: list[str], reason: str,
        detail: str, *, status: Optional[int] = None,
        provider_reason: Optional[str] = None,
    ) -> Artifact:
        payload = {
            "failure": reason,
            "returncode": status,
            "stderr": detail[:8000],
        }
        if provider_reason is not None:
            payload["provider_reason"] = provider_reason
        return verification_artifact(
            task=task, worker_id=worker_id, adapter="agentic",
            check=task.instruction, result="failed", confidence=0.55,
            evidence=evidence + [reason],
            payload=payload,
        )

    def _auth_failure_risk(
        self, task: Task, worker_id: str, provider: str, status: int, detail: str,
        reason: str = "",
    ) -> "Optional[Artifact]":
        """Loud, unmistakable RISK artifact for a provider auth rejection.

        A 401/403 after key-pool rotation is exhausted is a DEAD/REVOKED/WRONG
        key -- not a weak model and not a bad prompt. Without this, the failure
        was laundered into a generic verification-failed / "completed without
        structured findings" artifact, sending everyone hunting for a model or
        prompt problem instead of the real cause. We surface the provider and
        the exact env var to fix so the diagnosis is immediate.
        """
        # An auth rejection reaches us several ways: an HTTP 401/403 (status
        # set, or reason "http_status:401/403"), a pre-flight
        # "not_authenticated" (key missing/blank before any call, status None),
        # or the canonical classifier category "forbidden" when status was lost
        # after a 403. Catch all of them -- every one is a credential problem,
        # not a model or prompt problem.
        r = (reason or "").lower()
        is_auth = (
            status in (401, 403)
            or r == "not_authenticated"
            or r == "forbidden"
            or r in ("http_status:401", "http_status:403")
        )
        if not is_auth:
            return None
        code = status if status in (401, 403) else (
            401 if r in ("not_authenticated", "http_status:401") else 403)
        status = code
        env_var = _PROVIDER_ENV_HINTS.get((provider or "").lower(), f"the {provider} API key")
        return Artifact(
            job_id=task.job_id, task_id=task.id, type=ArtifactType.RISK,
            created_by=worker_id, confidence=0.95,
            evidence=["adapter:agentic", f"provider:{provider}",
                      f"auth_failed:{status}", "keys:exhausted"],
            payload={
                "risk": (
                    f"AUTH FAILURE: provider '{provider}' rejected the API key "
                    f"(HTTP {status}) after trying every configured key. This is a "
                    f"dead, revoked, or wrong key -- NOT a weak model or a bad "
                    f"prompt. The worker never reached the model."
                ),
                "mitigation": (
                    f"Fix or remove {env_var} (or disable the '{provider}' provider), "
                    f"then retry. Verify the key with a direct provider API call."
                ),
                "failure": f"auth_failed:{status}",
                "provider": provider,
                "stderr_excerpt": redact_secrets(detail or "")[:2000],
            },
        )


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


def _detect_verify_command(cwd: Path) -> Optional[str]:
    """Best-effort detection of a repo's verification command. Conservative --
    returns a command only for clear, common signals, else ``None`` (so an
    undetectable repo simply runs without verification rather than guessing).
    """
    try:
        pkg = cwd / "package.json"
        if pkg.is_file():
            data = json.loads(pkg.read_text(encoding="utf-8"))
            scripts = data.get("scripts") if isinstance(data, dict) else None
            if isinstance(scripts, dict) and scripts.get("test"):
                return "npm test --silent"
        if (cwd / "pytest.ini").is_file() or (cwd / "tox.ini").is_file():
            return "python -m pytest -q"
        pyproject = cwd / "pyproject.toml"
        if pyproject.is_file() and "pytest" in pyproject.read_text(
            encoding="utf-8", errors="ignore"
        ):
            return "python -m pytest -q"
        setup_cfg = cwd / "setup.cfg"
        if setup_cfg.is_file() and "pytest" in setup_cfg.read_text(
            encoding="utf-8", errors="ignore"
        ):
            return "python -m pytest -q"
        tests_dir = cwd / "tests"
        if tests_dir.is_dir() and any(tests_dir.glob("test_*.py")):
            return "python -m pytest -q"
    except Exception:  # noqa: BLE001 - detection is best-effort, never fatal
        return None
    return None


def _verify_evidence_tag(verify_state: dict) -> str:
    """The ``verify:<tag>`` evidence suffix summarizing a run's verification.

    ``skipped`` (no command), ``passed`` / ``failed`` (gating), or
    ``advisory-passed`` / ``advisory-failed`` / ``advisory`` when the clean-tree
    baseline was already red so the result is reported but not gated on.
    """
    mode = verify_state.get("mode")
    passed = verify_state.get("passed")
    if mode == "skipped" or not verify_state.get("command"):
        return "skipped"
    if mode == "advisory":
        if passed is True:
            return "advisory-passed"
        if passed is False:
            return "advisory-failed"
        return "advisory"
    return "passed" if passed else "failed"


def _last_message_role(messages: list[dict]) -> str:
    """The role of the most recent message, or '' when the log is empty."""
    return str(messages[-1].get("role") or "") if messages else ""


def _no_tool_calls_diagnosis(provider: str, model: str, turns: int) -> str:
    """Loud human-readable diagnosis when a model never emits tool_calls."""
    return (
        f"model {provider}/{model} produced {turns} turns of prose but never "
        "called any tool -- it is not tool-calling on this endpoint; route "
        "this role to a tool-capable model"
    )


def _coerce_submit_findings(args: object) -> list[dict]:
    """Normalize a ``submit_findings`` tool payload into artifact-item dicts.

    Tolerant on purpose: the canonical shape is ``{"artifacts": [ ... ]}``, but a
    model may pass a single finding at the top level or a lone dict. Anything
    that clearly isn't an artifact item is dropped rather than fabricated.
    """
    if not isinstance(args, dict):
        return []
    items = args.get("artifacts")
    if items is None:
        if any(key in args for key in ("claim", "risk", "decision", "finding", "summary")):
            items = [args]
        else:
            items = []
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return []
    normalized: list[dict] = []
    for item in items:
        if isinstance(item, dict):
            normalized.append({**item, "type": item.get("type") or "finding"})
    return normalized


def _items_to_artifacts(task: Task, worker_id: str, items: list[dict]) -> list[Artifact]:
    """Convert submitted artifact items into durable Artifacts, dropping any that
    don't satisfy the finding/risk/decision contract."""
    artifacts: list[Artifact] = []
    for item in items:
        artifact = cursor_artifact_from_item(task, worker_id, item, adapter="agentic")
        if artifact is not None:
            artifacts.append(artifact)
    return artifacts


def _coerce_submit_report(args: object) -> str:
    """Fold a ``submit_report`` tool payload into a single report string that the
    existing ``implement_report_artifacts`` path can turn into a durable finding.
    """
    if not isinstance(args, dict):
        return ""
    parts: list[str] = []
    summary = str(args.get("summary") or "").strip()
    if summary:
        parts.append(summary)
    files = args.get("files_changed")
    if isinstance(files, list) and files:
        parts.append("Files changed: " + ", ".join(str(f) for f in files))
    verification = str(args.get("verification") or "").strip()
    if verification:
        parts.append("Verification: " + verification)
    return "\n\n".join(parts)


def _coerce_plan_steps(args: object) -> list[dict]:
    """Normalize an ``update_plan`` payload into a list of ``{"step", "status"}``
    dicts. Tolerant of a bare list of strings, or a JSON string, so a model that
    slightly misshapes the argument still gets a usable plan."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return []
    raw = args.get("steps") if isinstance(args, dict) else args
    if not isinstance(raw, list):
        return []
    steps: list[dict] = []
    for item in raw:
        if isinstance(item, str):
            steps.append({"step": item.strip(), "status": "pending"})
        elif isinstance(item, dict):
            step = str(item.get("step") or item.get("title") or "").strip()
            if not step:
                continue
            status = str(item.get("status") or "pending").strip().lower()
            if status not in _PLAN_STATUSES:
                status = "pending"
            steps.append({"step": step, "status": status})
    return steps


def _render_plan(steps: list[dict]) -> str:
    """A compact checklist rendering of a plan for the tool ack (and artifact)."""
    marks = {"done": "[x]", "in_progress": "[~]", "pending": "[ ]"}
    return "\n".join(
        f"{marks.get(s.get('status', 'pending'), '[ ]')} {s.get('step', '')}"
        for s in steps
    ) or "(empty plan)"


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
