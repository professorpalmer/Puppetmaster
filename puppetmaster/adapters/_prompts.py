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


def build_structured_prompt(prompt: str, *, final_message_note: bool = False) -> str:
    lines = [prompt, ""]
    if final_message_note:
        lines.extend(
            [
                _PUPPETMASTER_ARTIFACT_CONTRACT_LINES[0],
                "When you are finished, emit ONLY a single JSON object as your final agent message "
                "(no prose around it, no markdown fences), in this shape:",
                _PUPPETMASTER_ARTIFACT_CONTRACT_LINES[2],
                _PUPPETMASTER_ARTIFACT_CONTRACT_LINES[3],
                _PUPPETMASTER_ARTIFACT_CONTRACT_LINES[4],
                _PUPPETMASTER_ARTIFACT_CONTRACT_LINES[5],
                _PUPPETMASTER_ARTIFACT_CONTRACT_LINES[6],
            ]
        )
    else:
        lines.extend(_PUPPETMASTER_ARTIFACT_CONTRACT_LINES)
    lines.extend([_ARTIFACT_GROUNDING, _ARTIFACT_EMPTY_GUIDANCE])
    if final_message_note:
        lines.append(
            "You may still use your tools to read files and inspect code along the way; just "
            "make sure the FINAL agent message is the JSON object described above."
        )
    return "\n".join(lines)


def build_implement_prompt(prompt: str) -> str:
    return "\n".join(
        [
            prompt,
            "",
            "Implement mode: you are running as a full-edit Puppetmaster worker "
            "inside the user's repository. Actually make the code changes — create, "
            "edit, and delete files as needed to complete the task end to end. Do not "
            "just describe a plan or return findings.",
            "Keep the change focused on the task; run any obvious local checks you can. "
            "Puppetmaster captures the resulting git diff as a PATCH artifact, so leave "
            "the working tree containing your final intended changes.",
            _IMPLEMENT_REPORT_CONTRACT,
        ]
    )


_ANALYZE_JSON_ONLY_RETRY = (
    "\n\nIMPORTANT: your previous response did not contain the required "
    "structured output. Respond with ONLY a single JSON object of the form "
    '{"artifacts": [...]} exactly as specified above — no prose, no explanation, '
    "no markdown fences, nothing before or after the JSON. If you genuinely found "
    'nothing for your role, return {"artifacts": []}.'
)


_IMPLEMENT_NOOP_NUDGE = (
    "You ended the turn without changing any files. Your job is to IMPLEMENT the "
    "task, not describe it — actually create, edit, or delete files now with your "
    "write_file / edit_file / delete_file tools, then run any focused checks you "
    "can to verify the change. If the task is genuinely already satisfied by the "
    "current code, do not invent an edit: say so explicitly and cite the exact "
    "file and lines that already satisfy it."
)


def with_repo_census(prompt: str, cwd: Union[Path, str, None]) -> str:
    """Append an authoritative repo file census so a worker can't hallucinate
    an empty repository.

    When files exist, the census states plainly that the repo is NOT empty and
    tells the worker to read them (and to report a tooling failure rather than
    assert emptiness if its own tools can't). When nothing can be enumerated we
    add only a soft boundary — we never assert emptiness ourselves, since an
    enumeration miss is not proof of an empty tree.
    """
    sample, total = repo_file_census(cwd)
    if total <= 0:
        return (
            prompt
            + "\n\nRepository file census: none enumerated. Do not assert the "
            "repository is empty unless your own tools also show no files — if "
            "they error, report a tooling failure, not an empty repository."
        )
    shown = ", ".join(sample)
    overflow = total - len(sample)
    more = f" (+{overflow} more)" if overflow > 0 else ""
    return (
        prompt
        + f"\n\nRepository file census (ground truth — {total} file(s) under the "
        f"working directory): {shown}{more}.\nThis census is authoritative: the "
        "repository is NOT empty. Read the relevant files before reporting. Never "
        "claim the repo is empty or 'starting from scratch' when files are listed "
        "here; if your own tools cannot read them, report a tooling failure, not "
        "an empty repository."
    )


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


def prompt_with_memory(prompt: str, task: Task) -> str:
    retrieved = task.payload.get("retrieved_memory") or []
    if not retrieved:
        return prompt
    distilled = _distill_memory_lines(retrieved)
    if not distilled:
        return prompt
    lines = [
        prompt,
        "",
        "Relevant promoted Puppetmaster memory (distilled facts/decisions):",
    ]
    lines.extend(distilled)
    lines.append("")
    lines.append("Use this as retrieved context, but verify claims before relying on them.")
    return "\n".join(lines)


def prompt_with_skills(prompt: str, task: Task) -> str:
    """Append the orchestrator-selected live-skill packet to a worker prompt.

    The mirror image of :func:`prompt_with_memory`: the trusted planner fills
    ``task.payload["injected_skills"]`` (a list of ``{"name", "body"}``) and the
    worker merely renders it. This is the return leg of the puppetmaster-learn
    flywheel (skill -> worker). It injects skill BODIES only — never the
    persona/rules layer, which ``--ignore-rules`` keeps suppressed — so the
    worker's access surface is unchanged. No-op when nothing was injected.
    """
    injected = task.payload.get("injected_skills") or []
    if not injected:
        return prompt
    from puppetmaster.skill_injection import render_skill_packet

    packet = render_skill_packet(injected)
    if not packet:
        return prompt
    return "\n".join([prompt, "", packet])


def with_report_contract(prompt: str) -> str:
    """Append the implement reporting contract unless the prompt already
    carries a structured artifact contract (swarm review/plan prompts do)."""
    if "Puppetmaster artifact contract" in prompt or _IMPLEMENT_REPORT_CONTRACT in prompt:
        return prompt
    return f"{prompt}\n\n{_IMPLEMENT_REPORT_CONTRACT}"

