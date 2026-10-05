from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from puppetmaster.codegraph import repo_file_census
from puppetmaster.models import Task

_ARTIFACT_GROUNDING = (
    "Your analysis target is THIS repository's code and configuration — not "
    "these instructions, not this artifact contract, and not the run itself. "
    "Ground every artifact in concrete files, functions, or symbols."
)


_ARTIFACT_EMPTY_GUIDANCE = (
    "If the repository genuinely yields nothing for your role (e.g. it is tiny "
    'or sound), return an empty list {"artifacts":[]} — never invent a finding '
    "or a risk about the prompt, the contract, or the run being degraded."
)


# What a worker's PASS/PARTIAL/FAIL means. Without it, read-only analysis
# workers graded themselves PARTIAL for not running tests they cannot run, and
# every clean audit was reported degraded.
WORKER_VERDICT_SEMANTICS = (
    "Grade the verdict against your assigned scope: PASS when you covered it, "
    "PARTIAL when part of it went uncovered (say which part), FAIL when you "
    "could not do the task. Read-only analysis cannot execute code; not running "
    "tests or reproducing at runtime is not a reason for PARTIAL."
)


_IMPLEMENT_REPORT_CONTRACT = (
    "Reporting contract: when you are done, end your final message with a short "
    "report — what you changed and why, the files you touched, and exactly what "
    "you ran to verify it. Puppetmaster persists that report as a durable "
    "artifact; without it the run looks like it did nothing."
)


# Opening line of every analysis prompt. gpt-5.6 models read the first noun
# phrase they see as the subject of the request, so a prompt that opened on
# "Puppetmaster artifact contract:" got answered *about the contract*. Naming
# where the assignment actually lives, before anything else, is what fixes it.
# Static, so it stays inside the shared cacheable prefix.
# NB: do not write the literal "Your task:" (with the colon) here --
# split_prompt_messages() splits the system/user seam on the first bare
# occurrence of TASK_INSTRUCTION_HEADER, so an inline mention would move the
# seam to this line and collapse the shared system prefix.
_PROMPT_ORIENTATION = (
    "Read this entire prompt before acting. Your assignment is the final "
    'section, the one headed "Your task". Everything above it -- output '
    "format, contract, repository context -- tells you HOW to answer and is "
    "never itself the thing to analyze, define, or ask about."
)


_PUPPETMASTER_ARTIFACT_CONTRACT_LINES = (
    "OUTPUT FORMAT (the Puppetmaster artifact contract for your reply):",
    "Return only JSON, with no markdown wrapper, in this shape:",
    '{"artifacts":[{"type":"finding","claim":"...","evidence":["path or symbol"],"confidence":0.8}]}',
    "Allowed artifact types:",
    '- finding: requires "claim", "evidence", "confidence". Also "done" (short what you finished), "deviations" (list of strings), "concerns" (list of strings). Empty lists are allowed.',
    '- risk: requires "risk", "mitigation", "evidence", "confidence".',
    '- decision: requires "decision", "why", "evidence", "confidence".',
    "Constraints beat reminders: No TODOs, no partial implementations. Do not instruct for generic engineering the model already knows — only this repo's tests, deploy, and forbidden deps.",
)


# Assembly seam for static-first / instruction-last prompts. Builders emit this
# header immediately before the per-task instruction; job-stable helpers and
# CodeGraph enrichment insert their sections before it so sibling workers share
# a cacheable prefix. Keep the header text stable — adapters and tests key on it.
TASK_INSTRUCTION_HEADER = "Your task:"

# CodeGraph's prompt section title — also a per-task boundary for agentic
# system/user splits (see split_prompt_messages).
CODEGRAPH_SECTION_HEADER = "Shared CodeGraph context for this task:"


def _task_instruction_index(prompt: str) -> int:
    """Return the index of the task-instruction header, or -1 if absent."""
    if not prompt:
        return -1
    needle = TASK_INSTRUCTION_HEADER + "\n"
    if prompt.startswith(needle) or prompt == TASK_INSTRUCTION_HEADER:
        return 0
    embedded = "\n" + needle
    idx = prompt.find(embedded)
    if idx >= 0:
        return idx + 1
    if prompt.endswith("\n" + TASK_INSTRUCTION_HEADER):
        return len(prompt) - len(TASK_INSTRUCTION_HEADER)
    return -1


def insert_before_task(prompt: str, section: str) -> str:
    """Insert ``section`` before the task instruction (static-first seam).

    When the prompt has no ``Your task:`` marker (legacy / unmarked strings),
    append so isolated helper call sites keep their prior behavior. Never raises.
    """
    try:
        if not section:
            return prompt
        section = section.strip("\n")
        if not section:
            return prompt
        anchor = _task_instruction_index(prompt)
        if anchor < 0:
            if not prompt:
                return section
            if prompt.endswith("\n\n"):
                return prompt + section
            if prompt.endswith("\n"):
                return prompt + "\n" + section
            return prompt + "\n\n" + section
        before = prompt[:anchor].rstrip("\n")
        after = prompt[anchor:]
        if before:
            return before + "\n\n" + section + "\n\n" + after
        return section + "\n\n" + after
    except Exception:
        return prompt


def split_prompt_messages(prompt: str) -> tuple[str, str]:
    """Split an assembled prompt into ``(system_prefix, user_suffix)``.

    System carries static boilerplate + job-stable sections (census / memory /
    skills). User carries per-task CodeGraph context (when present) plus the
    ``Your task:`` instruction block. Falls back to ``("", prompt)`` when no
    seam is found so callers can keep a single user message. Never raises.
    """
    try:
        if not prompt:
            return "", ""
        split_at = -1
        for header in (CODEGRAPH_SECTION_HEADER, TASK_INSTRUCTION_HEADER):
            idx = prompt.find(header)
            if idx >= 0 and (split_at < 0 or idx < split_at):
                split_at = idx
        if split_at < 0:
            return "", prompt
        system = prompt[:split_at].rstrip("\n")
        user = prompt[split_at:].lstrip("\n")
        return system, user
    except Exception:
        return "", prompt


def build_structured_prompt(
    prompt: str,
    *,
    final_message_note: bool = False,
    acceptance_criteria: object = None,
    terminal_verdict: bool = False,
) -> str:
    from puppetmaster.acceptance_criteria import (
        ensure_acceptance_criteria_in_text,
        normalize_acceptance_criteria,
        parse_acceptance_criteria_block,
    )

    criteria = normalize_acceptance_criteria(acceptance_criteria)
    if not criteria:
        criteria = parse_acceptance_criteria_block(prompt or "")
    task_body = (
        ensure_acceptance_criteria_in_text(prompt or "", criteria)
        if criteria
        else (prompt or "")
    )

    lines: list[str] = [_PROMPT_ORIENTATION, ""]
    if final_message_note:
        # Primary contract: finish by CALLING the submit_findings tool. The
        # provider constrains the tool's arguments, so structure is reliable even
        # on cheap models -- this is the parity mechanism that ends the "returned
        # prose the parser couldn't structure" degrade. The JSON-object shape is
        # kept as an explicit fallback for any model/provider without tool calls.
        submit_lines = [
            _PUPPETMASTER_ARTIFACT_CONTRACT_LINES[0],
            "When your analysis is complete, finish by CALLING the "
            "`submit_findings` tool exactly once. Pass an `artifacts` array of "
            "finding/risk/decision objects grounded in concrete files or "
            "symbols. If you genuinely found nothing for your role, call "
            "`submit_findings` with an empty array -- never invent a finding.",
        ]
        if criteria:
            submit_lines.extend(
                [
                    "When the task lists Acceptance criteria, also pass an "
                    "`acceptance_criteria` array on `submit_findings`: one record "
                    "per criterion you observed, each with `criterion` (exact task "
                    "text), `status` (`passed`, `failed`, or `unknown`), and "
                    "`evidence` (current-dispatch proof for passed/failed).",
                    "Acceptance criteria define what you must prove — report them "
                    "structurally, not as prose on every finding. Do not copy the "
                    "whole checklist onto each artifact and do not infer criterion "
                    "status from narrative text. Criteria you did not observe stay "
                    "unknown (omit them or set status unknown).",
                ]
            )
        submit_lines.extend(
            [
                "Each artifact object takes:",
                _PUPPETMASTER_ARTIFACT_CONTRACT_LINES[4],
                _PUPPETMASTER_ARTIFACT_CONTRACT_LINES[5],
                _PUPPETMASTER_ARTIFACT_CONTRACT_LINES[6],
                "Fallback only if you cannot call tools: emit ONLY a single JSON "
                'object {"artifacts":[...]} as your final message (no prose, no '
                "markdown fences).",
            ]
        )
        if terminal_verdict:
            submit_lines.append(
                "This is a review task. Submit exactly one advisory worker verdict "
                "using the native `worker_verdict` object: `verdict` is PASS, FAIL, "
                "or PARTIAL and `reason` is a short non-empty string. Do not also "
                "emit a VERDICT line when using that structured channel. "
                + WORKER_VERDICT_SEMANTICS
            )
        lines.extend(submit_lines)
    else:
        lines.extend(_PUPPETMASTER_ARTIFACT_CONTRACT_LINES)
        if criteria:
            lines.extend(
                [
                    "When the task lists Acceptance criteria, finish with "
                    "`submit_findings` and include an `acceptance_criteria` array: "
                    "one record per observed criterion (`criterion`, `status`, "
                    "`evidence`). passed/failed require current-dispatch evidence; "
                    "unobserved criteria stay unknown.",
                ]
            )
        if terminal_verdict:
            lines.append(
                "This is a review task. Include exactly one top-level "
                '`"worker_verdict":{"verdict":"PASS|FAIL|PARTIAL","reason":"..."}` '
                "object in JSON-native output. If only free text is available, "
                "instead make the final non-blank line `VERDICT: PASS - reason` "
                "(or FAIL/PARTIAL). Never use both verdict channels. "
                + WORKER_VERDICT_SEMANTICS
            )
    lines.extend([_ARTIFACT_GROUNDING, _ARTIFACT_EMPTY_GUIDANCE])
    if final_message_note:
        lines.append(
            "You may use your read/search tools to inspect the code along the way; "
            "just make sure you FINISH by calling `submit_findings`."
        )
    lines.extend(["", TASK_INSTRUCTION_HEADER, task_body])
    return "\n".join(lines)


_PLANNER_TASK_CONTRACT = (
    "You are the planner for this job. Do not write code. Do not edit files. "
    "You own the scope: emit an intent_spec decision (architecture, out_of_scope, "
    "dependency_philosophy, resource_timeouts, fanout_min, fanout_max) before any "
    "worker fan-out. Then emit 8-20 disjoint same-job tasks via enqueue_subtasks "
    "on a finding (numeric range, not 'many'). Workers are unaware of you; they "
    "return one handoff (done, deviations, concerns). After they finish you will "
    "run again. Before kind=scope_complete, enqueue a snapshot-lane task to take "
    "a green fixup pass. Constraints: no TODOs, no partial implementations, no "
    "nested job starts, no worker-to-worker coordination."
)


def structured_prompt_for_task(
    task: Task,
    *,
    final_message_note: bool = False,
    prompt: object = None,
) -> str:
    """Build the analyze artifact-contract prompt with canonical criteria.

    Threads ``task.payload["acceptance_criteria"]`` (or the instruction block)
    into :func:`build_structured_prompt` so no adapter path depends only on a
    later re-anchor pass.
    """
    from puppetmaster.acceptance_criteria import acceptance_criteria_for_task
    from puppetmaster.continuous_plan import is_planner_task

    if prompt is None:
        base = task.payload.get("prompt") or task.instruction
    else:
        base = prompt
    body = str(base or "")
    if is_planner_task(task):
        body = _PLANNER_TASK_CONTRACT + "\n\n" + body
    return build_structured_prompt(
        body,
        final_message_note=final_message_note,
        acceptance_criteria=acceptance_criteria_for_task(task),
        terminal_verdict="review" in str(task.role or "").lower(),
    )


_HASHLINE_EDIT_RULES = (
    "Prefer `apply_hashline` for surgical edits after a tagged `read_file` "
    "(Hashline format inspired by Oh My Pi / @oh-my-pi/hashline). "
    "`read_file` returns `[path#TAG]` plus `N:line` rows — use that TAG and those "
    "ORIGINAL line numbers. Ops: `SWAP N.=M:` (body `+` rows), `DEL N.=M` / `DEL N`, "
    "`INS.PRE`/`INS.POST`/`INS.HEAD`/`INS.TAIL`, `REM`, `MV`. "
    "Body rows are only `+TEXT` (literal `+` alone = blank). Ranges are tight and "
    "never shift as hunks apply. After every apply, re-ground on the new `#TAG` "
    "(or a fresh read). On stale-tag rejection: STOP and re-read. "
    "Block ops (`*.BLK`) are unsupported — use line ranges. "
    "When using `edit_file` after a tagged read, pass `expected_tag` from that "
    "read's `#TAG` so a concurrent change is refused cleanly; omit it only when "
    "you intentionally skip optimistic concurrency. "
    "Keep untagged `edit_file` / `write_file` for whole-file rewrites or when "
    "hashline is awkward."
)


def build_implement_prompt(prompt: str) -> str:
    return "\n".join(
        [
            "Implement mode: you are running as a full-edit Puppetmaster worker "
            "inside the user's repository. Actually make the code changes — create, "
            "edit, and delete files as needed to complete the task end to end. Do not "
            "just describe a plan or return findings.",
            "For anything beyond a trivial one-line change, call the `update_plan` "
            "tool first with your ordered steps, then update it (in_progress/done) as "
            "you go — it keeps the work organized and shows the user your progress.",
            _HASHLINE_EDIT_RULES,
            "Constraints: No TODOs, no partial implementations. Stay inside this "
            "task; do not wander off to fix unrelated failures. Other workers own "
            "those. End the report with deviations (list) and concerns (list); empty "
            "lists are allowed.",
            "Keep the change focused on the task; run any obvious local checks you can. "
            "Puppetmaster captures the resulting git diff as a PATCH artifact, so leave "
            "the working tree containing your final intended changes.",
            "Before you finish, VERIFY your work: run the project's tests (or the most "
            "relevant focused subset) with the `run_terminal` tool and make them pass. "
            "Your submission may be checked against the repo's verification command — if "
            "it fails you will be asked to fix it and submit again, so verify first.",
            "When all edits are done AND your checks pass, finish by CALLING the "
            "`submit_report` tool with a short summary, the files you changed, and how "
            "you verified. Include exactly one terminal verdict: PASS, FAIL, or PARTIAL "
            "with a short reason. If you cannot call tools, end the final message with "
            "`VERDICT: PASS - reason` (or FAIL/PARTIAL); do not put a verdict inside JSON.",
            _IMPLEMENT_REPORT_CONTRACT,
            "",
            TASK_INSTRUCTION_HEADER,
            prompt,
        ]
    )


_CLI_BUILD_CONTRACT = (
    "Build mode: you are a full-edit worker in this repository. Deliver working "
    "files, not a description of them: create and edit the files the task "
    "assigns you, and leave every other file alone; other workers own them.",
    "Work in a build-check loop: make the change, run the checks the task names "
    "(or the most relevant tests), read the results and fix what fails. Passing "
    "checks are the floor, not the finish: when the task asks for quality the "
    "checks do not measure and time remains, improve it and run the checks again.",
    _IMPLEMENT_REPORT_CONTRACT,
    "Keep that report to a few lines and end it with exactly one line "
    "`VERDICT: PASS - reason` (or FAIL / PARTIAL). Do not return findings JSON.",
)


def build_cli_implement_prompt(task: Task, *, prompt: object = None) -> str:
    """Build contract for a write-capable CLI worker (Codex, Claude Code).

    The analysis contract tells a worker its deliverable is a findings report.
    A worker that is building files read that literally: it stopped once its
    checks passed and spent its output on the report, so the files it was
    asked to craft got less of the turn.
    """
    from puppetmaster.acceptance_criteria import (
        acceptance_criteria_for_task,
        ensure_acceptance_criteria_in_text,
    )

    body = str((task.payload.get("prompt") or task.instruction) if prompt is None else prompt or "")
    criteria = acceptance_criteria_for_task(task)
    if criteria:
        body = ensure_acceptance_criteria_in_text(body, criteria)
    return "\n".join([_PROMPT_ORIENTATION, "", *_CLI_BUILD_CONTRACT, "", TASK_INSTRUCTION_HEADER, body])


_ANALYZE_JSON_ONLY_RETRY = (
    "\n\nIMPORTANT: your previous response did not submit the required structured "
    "output. Finish now by CALLING the `submit_findings` tool with an `artifacts` "
    "array (each item a finding/risk/decision grounded in concrete files or "
    "symbols). If you genuinely found nothing for your role, call `submit_findings` "
    'with an empty array. If you cannot call tools, respond with ONLY a single JSON '
    'object {"artifacts": [...]} — no prose, no explanation, no markdown fences.'
)


# Injected once when a model returns an empty turn right after a tool result --
# usually it just needs a nudge to keep going or to submit, not a degrade.
_EMPTY_RESPONSE_NUDGE = (
    "You returned an empty response. If your analysis is complete, call "
    "`submit_findings` now with your artifacts (or an empty array if you found "
    "nothing). Otherwise, continue using your tools to finish the task."
)


# Injected when a turn was truncated at the output-token cap, so a long final
# report/tool batch is continued instead of lost mid-word.
_LENGTH_CONTINUATION_NUDGE = (
    "Your previous response was cut off at the output limit. Continue exactly "
    "where you left off; when finished, call the appropriate submit tool."
)


_IMPLEMENT_NOOP_NUDGE = (
    "You ended the turn without changing any files. Your job is to IMPLEMENT the "
    "task, not describe it — actually create, edit, or delete files now with your "
    "apply_hashline / write_file / edit_file / delete_file tools, then run any "
    "focused checks you can to verify the change. If the task is genuinely already "
    "satisfied by the current code, do not invent an edit: say so explicitly and "
    "cite the exact file and lines that already satisfy it."
)


def _repo_census_section(cwd: Union[Path, str, None]) -> str:
    """Build the repo-census block (no prompt wrapping)."""
    sample, total = repo_file_census(cwd)
    if total <= 0:
        return (
            "Repository file census: none enumerated. Do not assert the "
            "repository is empty unless your own tools also show no files — if "
            "they error, report a tooling failure, not an empty repository."
        )
    shown = ", ".join(sample)
    overflow = total - len(sample)
    more = f" (+{overflow} more)" if overflow > 0 else ""
    return (
        f"Repository file census (ground truth — {total} file(s) under the "
        f"working directory): {shown}{more}.\nThis census is authoritative: the "
        "repository is NOT empty. Read the relevant files before reporting. Never "
        "claim the repo is empty or 'starting from scratch' when files are listed "
        "here; if your own tools cannot read them, report a tooling failure, not "
        "an empty repository."
    )


def with_repo_census(prompt: str, cwd: Union[Path, str, None]) -> str:
    """Inject an authoritative repo file census before the task instruction.

    When files exist, the census states plainly that the repo is NOT empty and
    tells the worker to read them (and to report a tooling failure rather than
    assert emptiness if its own tools can't). When nothing can be enumerated we
    add only a soft boundary — we never assert emptiness ourselves, since an
    enumeration miss is not proof of an empty tree.

    No-op when a job-level brief is already present — that brief already carries
    the census so sibling workers keep a single shared prefix segment.
    """
    try:
        from puppetmaster.job_brief import JOB_BRIEF_SECTION_HEADER

        if JOB_BRIEF_SECTION_HEADER in (prompt or ""):
            return prompt
        return insert_before_task(prompt, _repo_census_section(cwd))
    except Exception:
        return prompt


def _reanchor_acceptance_criteria(prompt: str, task: Task) -> str:
    """Keep explicit acceptance criteria after ``Your task:`` through enrichment."""
    try:
        from puppetmaster.acceptance_criteria import (
            acceptance_criteria_for_task,
            ensure_acceptance_criteria_in_text,
            format_acceptance_criteria_block,
        )

        criteria = acceptance_criteria_for_task(task)
        if not criteria:
            return prompt
        anchor = _task_instruction_index(prompt)
        if anchor < 0:
            return ensure_acceptance_criteria_in_text(prompt, criteria)
        before = prompt[:anchor]
        after = prompt[anchor:]
        header = TASK_INSTRUCTION_HEADER
        body = after[len(header) :]
        if body.startswith("\n"):
            body = body[1:]
        body = ensure_acceptance_criteria_in_text(body, criteria)
        # If structured criteria exist but the free-text block used different
        # wording, still guarantee the canonical block is present.
        block = format_acceptance_criteria_block(criteria)
        if block and block not in body:
            body = ensure_acceptance_criteria_in_text(body, criteria)
        return before + header + "\n" + body
    except Exception:
        return prompt


def with_job_brief(prompt: str, task: Task, *, shared_brief: bool = True) -> str:
    """Inject the job-stable shared CodeGraph / repo brief before the task.

    Reads bytes persisted at job start (see ``puppetmaster.job_brief``) so every
    sibling worker gets an identical prefix segment. Lands in the system prefix
    via ``split_prompt_messages`` (distinct header from per-task CodeGraph).
    Best-effort; never raises. Kill switch: ``PUPPETMASTER_JOB_BRIEF=0``.

    ``shared_brief=False`` skips that section (prewalk plan and criteria still
    apply) for a worker that receives its own task-scoped CodeGraph context.

    Also applies :func:`with_prewalk_plan` so every implement-mode adapter that
    already funnels through this helper gets upstream plan injection for free.
    """
    if shared_brief:
        try:
            from puppetmaster.job_brief import resolve_job_brief_for_task

            section = resolve_job_brief_for_task(task)
            if section:
                prompt = insert_before_task(prompt, section.strip("\n"))
        except Exception:
            pass
    prompt = with_prewalk_plan(prompt, task)
    return _reanchor_acceptance_criteria(prompt, task)


def _open_store_for_task(task: Task, *, discover: bool = True):
    """Best-effort open of the active store for ``task.job_id``.

    Mirrors :func:`puppetmaster.job_brief.resolve_job_brief_for_task` state-dir
    resolution (sidecar env → find_state_dir_for_job → ``PUPPETMASTER_STATE_DIR``)
    so worker subprocesses see the same store the orchestrator wrote into.
    Returns ``None`` on any miss or error.
    """
    import os

    job_id = getattr(task, "job_id", None) or ""
    if not job_id:
        return None
    from puppetmaster.adapters._streaming import _resolve_sidecar_state_dir
    from puppetmaster.state import STATE_DIR_ENV, find_state_dir_for_job, resolve_state_dir
    from puppetmaster.store_factory import create_store

    state_dir = _resolve_sidecar_state_dir()
    if state_dir is None and discover:
        state_dir = find_state_dir_for_job(job_id)
    if state_dir is None and os.environ.get(STATE_DIR_ENV):
        try:
            state_dir = resolve_state_dir()
        except Exception:
            state_dir = None
    if state_dir is None:
        return None
    backend = "sqlite" if (Path(state_dir) / "state.sqlite3").is_file() else "file"
    return create_store(backend, state_dir)


def _load_job_artifacts_for_task(task: Task) -> list:
    """Best-effort load of job-scoped artifacts from the active store."""
    store = _open_store_for_task(task)
    if store is None:
        return []
    job_id = getattr(task, "job_id", None) or ""
    return list(store.list_artifacts(job_id))


def _load_upstream_artifacts_via_edges(task: Task, *, record_consumes: bool = False) -> list:
    """Resolve only artifacts produced by upstream tasks through graph edges.

    Returns an empty list when no produces edges exist (compatibility fallback
    to whole-job artifact load). Consumes edges are recorded only when the
    caller passes ``record_consumes=True`` for artifacts that will actually be
    injected.
    """
    store = _open_store_for_task(task)
    if store is None:
        return []
    return list(store.resolve_artifacts_via_edges(task, record_consumes=record_consumes))


def _prewalk_injection_body(
    prompt: str,
    artifacts: list,
    *,
    verify_role: bool,
    cwd: Optional[Any] = None,
    store: Optional[Any] = None,
) -> str:
    """Return the formattable injection body, or "" when nothing usable exists."""
    from puppetmaster.prewalk import (
        PREWALK_PLAN_SECTION_HEADER,
        format_plan_artifacts_for_injection,
        format_upstream_artifacts_for_injection,
    )

    if verify_role:
        return format_upstream_artifacts_for_injection(
            artifacts, cwd=cwd, store=store
        )
    if PREWALK_PLAN_SECTION_HEADER in (prompt or ""):
        plan_text = format_plan_artifacts_for_injection(
            artifacts, cwd=cwd, store=store
        )
        if plan_text:
            return plan_text
    return (
        format_upstream_artifacts_for_injection(artifacts, cwd=cwd, store=store)
        or format_plan_artifacts_for_injection(artifacts, cwd=cwd, store=store)
    )


def _artifacts_used_for_injection(
    artifacts: list,
    *,
    verify_role: bool,
    prompt: str,
    cwd: Optional[Any] = None,
    store: Optional[Any] = None,
) -> list:
    """Filter edge-resolved artifacts to those that contribute injection text."""
    used: list = []
    for artifact in artifacts:
        if _prewalk_injection_body(
            prompt, [artifact], verify_role=verify_role, cwd=cwd, store=store
        ):
            used.append(artifact)
    return used


def with_prewalk_plan(prompt: str, task: Task) -> str:
    """Inject upstream artifacts for prewalk implement/verify workers.

    No-op unless ``task.payload["prewalk"]`` is truthy. Prefers provenance-edge
    resolution (artifacts produced by ``depends_on`` tasks) and records
    ``consumes`` edges only for formattable artifacts that are actually
    injected. ROUTING-only / empty edge results fall back to the legacy
    whole-job artifact load instead of leaving placeholders unchanged.
    ``payload["prewalk_artifacts"]`` may supply an explicit list for tests.
    Best-effort; never raises.
    """
    try:
        payload = getattr(task, "payload", None) or {}
        if not payload.get("prewalk"):
            return prompt
        from puppetmaster.prewalk import (
            VERIFY_ROLE,
            inject_plan_into_prompt,
            inject_upstream_into_prompt,
        )

        cwd = payload.get("cwd") or payload.get("workspace")
        if cwd == "":
            cwd = None
        role = str(payload.get("prewalk_role") or getattr(task, "role", "") or "")
        verify_role = role == VERIFY_ROLE
        inline = payload.get("prewalk_artifacts")
        if inline is not None:
            artifacts = list(inline)
            if not artifacts:
                return prompt
            store = _open_store_for_task(task, discover=False)
            if verify_role:
                return inject_upstream_into_prompt(
                    prompt, artifacts, cwd=cwd, store=store
                )
            return inject_plan_into_prompt(
                prompt, artifacts, cwd=cwd, store=store
            )

        store = _open_store_for_task(task)
        edge_artifacts: list = []
        if store is not None:
            edge_artifacts = list(
                store.resolve_artifacts_via_edges(task, record_consumes=False)
            )
        if edge_artifacts:
            used = _artifacts_used_for_injection(
                edge_artifacts,
                verify_role=verify_role,
                prompt=prompt,
                cwd=cwd,
                store=store,
            )
            if used and _prewalk_injection_body(
                prompt, used, verify_role=verify_role, cwd=cwd, store=store
            ):
                store.record_consumes(
                    task.job_id,
                    task.id,
                    [
                        getattr(artifact, "id", None)
                        or (artifact.get("id") if isinstance(artifact, dict) else None)
                        for artifact in used
                    ],
                )
                return inject_upstream_into_prompt(
                    prompt, used, cwd=cwd, store=store
                )

        artifacts = _load_job_artifacts_for_task(task)
        if not artifacts:
            return prompt
        if verify_role:
            return inject_upstream_into_prompt(
                prompt, artifacts, cwd=cwd, store=store
            )
        return inject_plan_into_prompt(
            prompt, artifacts, cwd=cwd, store=store
        )
    except Exception:
        return prompt


_MEMORY_MAX_ITEMS = 5


_MEMORY_STATEMENT_MAX_CHARS = 280


def _truncate_statement(statement: str) -> str:
    collapsed = " ".join(statement.split())
    if len(collapsed) <= _MEMORY_STATEMENT_MAX_CHARS:
        return collapsed
    return collapsed[: _MEMORY_STATEMENT_MAX_CHARS - 1].rstrip() + "…"


def _distill_memory_lines(retrieved: list) -> list[str]:
    """Dedupe promoted memory and cap each statement so a handful of verbose
    prior decisions can't balloon every worker prompt with thousands of tokens
    of duplicated instructions. Full statements remain in the memory store; only
    the injected copy is trimmed."""
    lines: list[str] = []
    seen: set[str] = set()
    for memory in retrieved:
        statement = str(memory.get("statement", "")).strip()
        if not statement:
            continue
        key = " ".join(statement.lower().split())
        if key in seen:
            continue
        seen.add(key)
        scope = memory.get("scope", "memory")
        lines.append(f"- [{scope}] {_truncate_statement(statement)}")
        if len(lines) >= _MEMORY_MAX_ITEMS:
            break
    return lines


def _memory_section(task: Task) -> str:
    retrieved = task.payload.get("retrieved_memory") or []
    if not retrieved:
        return ""
    distilled = _distill_memory_lines(retrieved)
    if not distilled:
        return ""
    lines = [
        "Relevant promoted Puppetmaster memory (distilled facts/decisions):",
        *distilled,
        "",
        "Use this as retrieved context, but verify claims before relying on them.",
    ]
    return "\n".join(lines)


def prompt_with_memory(prompt: str, task: Task) -> str:
    try:
        section = _memory_section(task)
        if section:
            prompt = insert_before_task(prompt, section)
        return _reanchor_acceptance_criteria(prompt, task)
    except Exception:
        return prompt


def prompt_with_skills(prompt: str, task: Task) -> str:
    """Inject the orchestrator-selected live-skill packet before the task instruction.

    The mirror image of :func:`prompt_with_memory`: the trusted planner fills
    ``task.payload["injected_skills"]`` (a list of ``{"name", "body"}``) and the
    worker merely renders it. This is the return leg of the puppetmaster-learn
    flywheel (skill -> worker). It injects skill BODIES only — never the
    persona/rules layer, which ``--ignore-rules`` keeps suppressed — so the
    worker's access surface is unchanged. No-op when nothing was injected.
    """
    try:
        injected = task.payload.get("injected_skills") or []
        if not injected:
            return prompt
        from puppetmaster.skill_injection import render_skill_packet

        packet = render_skill_packet(injected)
        if not packet:
            return prompt
        return insert_before_task(prompt, packet)
    except Exception:
        return prompt


def with_report_contract(prompt: str) -> str:
    """Inject the implement reporting contract before the task instruction.

    No-op when the prompt already carries a structured artifact contract
    (swarm review/plan prompts do) or the implement reporting contract.
    Falls back to append when no ``Your task:`` marker is present.
    """
    if "Puppetmaster artifact contract" in prompt or _IMPLEMENT_REPORT_CONTRACT in prompt:
        return prompt
    return insert_before_task(prompt, _IMPLEMENT_REPORT_CONTRACT)
