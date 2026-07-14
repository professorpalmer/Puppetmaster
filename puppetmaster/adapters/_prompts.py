from __future__ import annotations

from pathlib import Path
from typing import Union

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


_IMPLEMENT_REPORT_CONTRACT = (
    "Reporting contract: when you are done, end your final message with a short "
    "report — what you changed and why, the files you touched, and exactly what "
    "you ran to verify it. Puppetmaster persists that report as a durable "
    "artifact; without it the run looks like it did nothing."
)


_PUPPETMASTER_ARTIFACT_CONTRACT_LINES = (
    "Puppetmaster artifact contract:",
    "Return only JSON, with no markdown wrapper, in this shape:",
    '{"artifacts":[{"type":"finding","claim":"...","evidence":["path or symbol"],"confidence":0.8}]}',
    "Allowed artifact types:",
    '- finding: requires "claim", "evidence", "confidence".',
    '- risk: requires "risk", "mitigation", "evidence", "confidence".',
    '- decision: requires "decision", "why", "evidence", "confidence".',
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


def build_structured_prompt(prompt: str, *, final_message_note: bool = False) -> str:
    lines: list[str] = []
    if final_message_note:
        # Primary contract: finish by CALLING the submit_findings tool. The
        # provider constrains the tool's arguments, so structure is reliable even
        # on cheap models -- this is the parity mechanism that ends the "returned
        # prose the parser couldn't structure" degrade. The JSON-object shape is
        # kept as an explicit fallback for any model/provider without tool calls.
        lines.extend(
            [
                _PUPPETMASTER_ARTIFACT_CONTRACT_LINES[0],
                "When your analysis is complete, finish by CALLING the "
                "`submit_findings` tool exactly once. Pass an `artifacts` array of "
                "finding/risk/decision objects grounded in concrete files or "
                "symbols. If you genuinely found nothing for your role, call "
                "`submit_findings` with an empty array -- never invent a finding.",
                "Each artifact object takes:",
                _PUPPETMASTER_ARTIFACT_CONTRACT_LINES[4],
                _PUPPETMASTER_ARTIFACT_CONTRACT_LINES[5],
                _PUPPETMASTER_ARTIFACT_CONTRACT_LINES[6],
                "Fallback only if you cannot call tools: emit ONLY a single JSON "
                'object {"artifacts":[...]} as your final message (no prose, no '
                "markdown fences).",
            ]
        )
    else:
        lines.extend(_PUPPETMASTER_ARTIFACT_CONTRACT_LINES)
    lines.extend([_ARTIFACT_GROUNDING, _ARTIFACT_EMPTY_GUIDANCE])
    if final_message_note:
        lines.append(
            "You may use your read/search tools to inspect the code along the way; "
            "just make sure you FINISH by calling `submit_findings`."
        )
    lines.extend(["", TASK_INSTRUCTION_HEADER, prompt])
    return "\n".join(lines)


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
            "Keep the change focused on the task; run any obvious local checks you can. "
            "Puppetmaster captures the resulting git diff as a PATCH artifact, so leave "
            "the working tree containing your final intended changes.",
            "Before you finish, VERIFY your work: run the project's tests (or the most "
            "relevant focused subset) with the `run_terminal` tool and make them pass. "
            "Your submission may be checked against the repo's verification command — if "
            "it fails you will be asked to fix it and submit again, so verify first.",
            "When all edits are done AND your checks pass, finish by CALLING the "
            "`submit_report` tool with a short summary, the files you changed, and how "
            "you verified. If you cannot call tools, end with the same report as your "
            "final message instead.",
            _IMPLEMENT_REPORT_CONTRACT,
            "",
            TASK_INSTRUCTION_HEADER,
            prompt,
        ]
    )


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
    "write_file / edit_file / delete_file tools, then run any focused checks you "
    "can to verify the change. If the task is genuinely already satisfied by the "
    "current code, do not invent an edit: say so explicitly and cite the exact "
    "file and lines that already satisfy it."
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


def with_job_brief(prompt: str, task: Task) -> str:
    """Inject the job-stable shared CodeGraph / repo brief before the task.

    Reads bytes persisted at job start (see ``puppetmaster.job_brief``) so every
    sibling worker gets an identical prefix segment. Lands in the system prefix
    via ``split_prompt_messages`` (distinct header from per-task CodeGraph).
    Best-effort; never raises. Kill switch: ``PUPPETMASTER_JOB_BRIEF=0``.

    Also applies :func:`with_prewalk_plan` so every implement-mode adapter that
    already funnels through this helper gets upstream plan injection for free.
    """
    try:
        from puppetmaster.job_brief import resolve_job_brief_for_task

        section = resolve_job_brief_for_task(task)
        if section:
            prompt = insert_before_task(prompt, section.strip("\n"))
    except Exception:
        pass
    return with_prewalk_plan(prompt, task)


def _load_job_artifacts_for_task(task: Task) -> list:
    """Best-effort load of job-scoped artifacts from the active store.

    Mirrors :func:`puppetmaster.job_brief.resolve_job_brief_for_task` state-dir
    resolution (sidecar env → find_state_dir_for_job → ``PUPPETMASTER_STATE_DIR``)
    so worker subprocesses see the same store the orchestrator wrote into.
    Returns an empty list on any miss or error.
    """
    import os

    job_id = getattr(task, "job_id", None) or ""
    if not job_id:
        return []
    from puppetmaster.adapters._streaming import _resolve_sidecar_state_dir
    from puppetmaster.state import STATE_DIR_ENV, find_state_dir_for_job, resolve_state_dir
    from puppetmaster.store_factory import create_store

    state_dir = _resolve_sidecar_state_dir()
    if state_dir is None:
        state_dir = find_state_dir_for_job(job_id)
    if state_dir is None and os.environ.get(STATE_DIR_ENV):
        try:
            state_dir = resolve_state_dir()
        except Exception:
            state_dir = None
    if state_dir is None:
        return []
    backend = "sqlite" if (Path(state_dir) / "state.sqlite3").is_file() else "file"
    store = create_store(backend, state_dir)
    return list(store.list_artifacts(job_id))


def with_prewalk_plan(prompt: str, task: Task) -> str:
    """Inject upstream plan/decision artifacts for prewalk implement workers.

    No-op unless ``task.payload["prewalk"]`` is truthy. Loads job artifacts from
    the store (same state-dir resolution as the job brief) and replaces the
    placeholder plan section via :func:`puppetmaster.prewalk.inject_plan_into_prompt`.
    ``payload["prewalk_artifacts"]`` may supply an explicit list for tests.
    Best-effort; never raises.
    """
    try:
        payload = getattr(task, "payload", None) or {}
        if not payload.get("prewalk"):
            return prompt
        inline = payload.get("prewalk_artifacts")
        if inline is not None:
            artifacts = inline
        else:
            artifacts = _load_job_artifacts_for_task(task)
        if not artifacts:
            return prompt
        from puppetmaster.prewalk import inject_plan_into_prompt

        return inject_plan_into_prompt(prompt, artifacts)
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
        if not section:
            return prompt
        return insert_before_task(prompt, section)
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
