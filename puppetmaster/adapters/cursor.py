from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Union

from puppetmaster.codegraph import enrich_prompt_with_codegraph, inject_worker_cli_env
from puppetmaster.failure import classify_cursor_failure
from puppetmaster.models import Artifact, ArtifactType, Task
from puppetmaster.ports import apply_worktree_ports
from puppetmaster.redaction import redact_secrets
from puppetmaster.usage import token_usage

from ._base import (
    CliInvocation,
    CliWorkerAdapter,
    diff_source_payload,
    make_patch_artifact,
    missing_cli_artifact,
    resolve_command,
    verification_artifact,
)
from ._facade import facade
from ._prompts import (
    build_implement_prompt,
    build_structured_prompt,
    prompt_with_memory,
    with_job_brief,
)
from ._streaming import (
    StreamedProcess,
    _STDOUT_HEAD_CHARS,
    _STDOUT_TAIL_CHARS,
    _redacted_tail,
    capture_subprocess_stdout,
    run_streamed_subprocess,
)
from ._base import _should_emit_patch_artifact

_CURSOR_RUNNER = Path(__file__).resolve().parent.parent / "cursor_sdk_runner.mjs"

class CursorAdapter(CliWorkerAdapter):
    name = "cursor"

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        # Two execution modes share the same Cursor SDK runner. ``analyze`` (the
        # default) steers the agent to return findings/risks and never touches
        # the tree; ``implement`` lets it edit files and captures the resulting
        # diff as a PATCH artifact — the same full-edit contract Claude Code and
        # Codex already honour, so a cursor-only platform lock can still ship code.
        if task.payload.get("mode") == "implement" or task.payload.get("implement"):
            return self._run_implement(task, goal, worker_id)
        return self._run_analyze(task, goal, worker_id)

    def _run_implement(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        return self._run_cli_lifecycle(task, goal, worker_id, pre_guard=True)

    def _early_run_guard(
        self, task: Task, worker_id: str, cwd: Path
    ) -> tuple[Optional[list[Artifact]], dict]:
        return self.guard_full_edit_run(
            task,
            worker_id,
            "cursor",
            cwd,
            adapter_label="cursor-sdk",
            extra_dirty_message=(
                " For focused edits on a dirty tree (docs, tests), use puppetmaster_edit — it edits "
                "in place and needs no clean tree."
            ),
        )

    def _resolve_cli_executable(self, task: Task) -> tuple[str, Optional[str]]:
        resolved = facade("resolve_command")("node")
        if resolved is None:
            return "node", None
        return "node", resolved

    def _missing_cli(
        self, task: Task, worker_id: str, executable_label: str
    ) -> list[Artifact]:
        return missing_cli_artifact(
            task,
            worker_id,
            "cursor",
            executable_label,
            "Node.js was not found. Install Node.js to run the Cursor SDK adapter.",
        )

    def _prepare_cli_invocation(
        self,
        task: Task,
        goal: str,
        worker_id: str,
        cwd: Path,
        resolved: str,
    ) -> Union[list[Artifact], CliInvocation]:
        base_prompt = task.payload.get("prompt") or task.instruction
        model = task.payload.get("model", "default")
        prompt, codegraph_used = facade("enrich_prompt_with_codegraph")(
            prompt_with_memory(
                with_job_brief(build_implement_prompt(base_prompt), task),
                task,
            ),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )
        runner = _CURSOR_RUNNER
        return CliInvocation(
            command=[resolved, str(runner)],
            sidecar_name="cursor_implement",
            extras={
                "prompt": prompt,
                "codegraph_used": codegraph_used,
                "model": model,
                "cwd": str(cwd),
            },
        )

    def _invoke_cli(
        self,
        task: Task,
        prepared: CliInvocation,
        cwd: Path,
        timeout_seconds: int,
    ) -> StreamedProcess:
        environment = inject_worker_cli_env(os.environ.copy())
        apply_worktree_ports(environment, cwd)
        environment["PUPPETMASTER_CURSOR_INPUT"] = json.dumps(
            {
                "prompt": prepared.extras["prompt"],
                "cwd": prepared.extras["cwd"],
                "model": prepared.extras["model"],
            },
            sort_keys=True,
        )
        return facade("run_streamed_subprocess")(
            command=prepared.command,
            env=environment,
            task=task,
            sidecar_name=prepared.sidecar_name,
            timeout_seconds=timeout_seconds,
        )

    def _finalize_cli_run(
        self,
        task: Task,
        worker_id: str,
        goal: str,
        prepared: CliInvocation,
        before: dict,
        after: dict,
        completed: StreamedProcess,
    ) -> list[Artifact]:
        prompt = str(prepared.extras.get("prompt") or "")
        codegraph_used = bool(prepared.extras.get("codegraph_used"))
        model = prepared.extras.get("model", "default")
        cwd = Path(prepared.extras.get("cwd") or ".")
        timeout_seconds = int(task.payload.get("timeout_seconds", 900))
        if completed.timed_out:
            artifacts: list[Artifact] = [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="cursor",
                    check=task.instruction,
                    result="failed",
                    confidence=0.55,
                    evidence=["adapter:cursor-sdk", "mode:implement", "timeout"],
                    payload={
                        "failure": "timeout",
                        "returncode": None,
                        "stdout": _redacted_tail(completed.stdout, _STDOUT_TAIL_CHARS),
                        "stderr": _redacted_tail(completed.stderr, _STDOUT_TAIL_CHARS),
                        "timeout_seconds": timeout_seconds,
                        "live_log": completed.live_log_path,
                        "base_sha": before["sha"],
                        "head_sha": after["sha"],
                        "changed_files": after["changed_files"],
                        "untracked_files": after["untracked_files"],
                        **diff_source_payload(before, after),
                    },
                )
            ]
            if _should_emit_patch_artifact(before, after):
                artifacts.append(
                    make_patch_artifact(
                        task,
                        worker_id,
                        before,
                        after,
                        adapter="cursor",
                        status="failed",
                        change="Cursor agent modified repository files.",
                        sidecar_name="cursor_implement",
                        evidence_adapter="cursor-sdk",
                    )
                )
            return artifacts

        failure = classify_cursor_failure(completed.stderr + completed.stdout)
        cursor_status, result_text = cursor_result_text(completed.stdout)
        usage = token_usage(
            sdk_usage=sdk_usage_from_stdout(completed.stdout),
            prompt_text=prompt,
            output_text=completed.stdout,
        )
        stdout_capture = capture_subprocess_stdout(
            text=completed.stdout, task=task, sidecar_name="cursor_implement_stdout", tail_chars=12000
        )
        stderr_capture = capture_subprocess_stdout(
            text=completed.stderr, task=task, sidecar_name="cursor_implement_stderr"
        )
        artifacts = [
            verification_artifact(
                task=task,
                worker_id=worker_id,
                adapter="cursor",
                check=task.instruction,
                result="passed" if completed.returncode == 0 else "failed",
                confidence=0.9 if completed.returncode == 0 else 0.55,
                evidence=(
                    ["adapter:cursor-sdk", "mode:implement", f"node:{sys.platform}"]
                    + (["context:codegraph"] if codegraph_used else [])
                ),
                payload={
                    "failure": None if completed.returncode == 0 else failure,
                    "returncode": completed.returncode,
                    "stdout": _redacted_tail(completed.stdout, 12000),
                    "stderr": _redacted_tail(completed.stderr, _STDOUT_TAIL_CHARS),
                    "stdout_capture": stdout_capture,
                    "stderr_capture": stderr_capture,
                    "live_log": completed.live_log_path,
                    "model": model,
                    "cwd": str(cwd),
                    "cursor_status": cursor_status,
                    "base_sha": before["sha"],
                    "head_sha": after["sha"],
                    "changed_files": after["changed_files"],
                    "untracked_files": after["untracked_files"],
                    **diff_source_payload(before, after),
                    **usage,
                },
            )
        ]
        if completed.returncode == 0:
            artifacts.extend(
                implement_report_artifacts(
                    task, worker_id, result_text, adapter="cursor-sdk"
                )
            )
        if _should_emit_patch_artifact(before, after):
            artifacts.append(
                make_patch_artifact(
                    task,
                    worker_id,
                    before,
                    after,
                    adapter="cursor",
                    status="applied" if completed.returncode == 0 else "failed",
                    change="Cursor agent modified repository files.",
                    sidecar_name="cursor_implement",
                    evidence_adapter="cursor-sdk",
                )
            )
        return artifacts

    def _run_analyze(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        base_prompt = task.payload.get("prompt") or task.instruction
        cwd = task.payload.get("cwd")
        model = task.payload.get("model", "default")
        prompt, codegraph_used = facade("enrich_prompt_with_codegraph")(
            prompt_with_memory(
                facade("with_repo_census")(
                    with_job_brief(build_structured_prompt(base_prompt), task),
                    cwd,
                ),
                task,
            ),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )
        runner = _CURSOR_RUNNER
        environment = inject_worker_cli_env(os.environ.copy())
        apply_worktree_ports(environment, cwd)
        environment["PUPPETMASTER_CURSOR_INPUT"] = json.dumps(
            {"prompt": prompt, "cwd": cwd, "model": model},
            sort_keys=True,
        )
        timeout_seconds = int(task.payload.get("timeout_seconds", 300))
        completed = facade("run_streamed_subprocess")(
            command=["node", str(runner)],
            env=environment,
            task=task,
            sidecar_name="cursor_analyze",
            timeout_seconds=timeout_seconds,
        )
        if completed.timed_out:
            stdout = completed.stdout
            stderr = completed.stderr
            stdout_capture = capture_subprocess_stdout(
                text=stdout,
                task=task,
                sidecar_name="cursor_stdout_timeout",
            )
            stderr_capture = capture_subprocess_stdout(
                text=stderr,
                task=task,
                sidecar_name="cursor_stderr_timeout",
            )
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="cursor",
                    check=task.instruction,
                    result="failed",
                    confidence=0.55,
                    evidence=["adapter:cursor-sdk", "timeout"],
                    payload={
                        "returncode": None,
                        "stdout": _redacted_tail(stdout, _STDOUT_TAIL_CHARS),
                        "stderr": _redacted_tail(stderr, _STDOUT_TAIL_CHARS),
                        "stdout_capture": stdout_capture,
                        "stderr_capture": stderr_capture,
                        "model": model,
                        "failure": "timeout",
                        "live_log": completed.live_log_path,
                    },
                )
            ]
        failure = classify_cursor_failure(completed.stderr + completed.stdout)
        status, result_text = cursor_result_text(completed.stdout)
        usage = token_usage(
            sdk_usage=sdk_usage_from_stdout(completed.stdout),
            prompt_text=prompt,
            output_text=result_text or completed.stdout,
        )
        parsed_artifacts = (
            cursor_result_artifacts(task, worker_id, result_text)
            if completed.returncode == 0
            else []
        )
        # Salvage (#3): structured content can sit in raw stdout even when the
        # SDK's `result` field was empty (a dry-run that printed but never
        # "finished") or the run exited non-zero. Parse raw stdout before
        # declaring the run degraded, so a real review isn't lost to a manual
        # log read.
        if not parsed_artifacts:
            salvaged = cursor_result_artifacts(task, worker_id, completed.stdout)
            if salvaged:
                parsed_artifacts = salvaged
        degraded = completed.returncode == 0 and not parsed_artifacts
        stdout_capture = capture_subprocess_stdout(
            text=completed.stdout,
            task=task,
            sidecar_name="cursor_stdout",
        )
        stderr_capture = capture_subprocess_stdout(
            text=completed.stderr,
            task=task,
            sidecar_name="cursor_stderr",
        )
        artifacts = [
            verification_artifact(
                task=task,
                worker_id=worker_id,
                adapter="cursor",
                check=task.instruction,
                result=(
                    "degraded"
                    if degraded
                    else "passed"
                    if completed.returncode == 0
                    else "failed"
                ),
                confidence=0.65 if degraded else 0.9 if completed.returncode == 0 else 0.55,
                evidence=(
                    ["adapter:cursor-sdk", f"node:{sys.platform}"]
                    + (["context:codegraph"] if codegraph_used else [])
                ),
                payload={
                    "returncode": completed.returncode,
                    "stdout": _redacted_tail(completed.stdout, _STDOUT_TAIL_CHARS),
                    "stderr": _redacted_tail(completed.stderr, _STDOUT_TAIL_CHARS),
                    "stdout_capture": stdout_capture,
                    "stderr_capture": stderr_capture,
                    "model": model,
                    "cursor_status": status,
                    "failure": (
                        "empty_or_unstructured_cursor_result"
                        if degraded
                        else failure
                        if completed.returncode != 0
                        else None
                    ),
                    **usage,
                },
            )
        ]
        if degraded:
            artifacts.append(
                cursor_degraded_artifact(
                    task,
                    worker_id,
                    result_text,
                    stdout_capture=stdout_capture,
                )
            )
        artifacts.extend(parsed_artifacts)
        return artifacts


def sdk_usage_from_stdout(stdout: str) -> Any:
    """Pull a top-level SDK ``usage`` object out of JSON stdout, if any.

    Shared by the Cursor and Claude Code adapters, whose runners both emit a
    single JSON result object carrying ``usage``. Returns ``None`` when stdout
    isn't JSON or has no usage, so the caller falls back to an approximation."""
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(payload, dict):
        return payload.get("usage")
    return None


def cursor_result_text(stdout: str) -> tuple[Optional[str], str]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return None, stdout.strip()
    if not isinstance(payload, dict):
        return None, stdout.strip()
    if "result" not in payload and (
        "artifacts" in payload or "findings" in payload or "risks" in payload or "decisions" in payload
    ):
        return None, stdout.strip()
    result = payload.get("result")
    return (
        str(payload.get("status")) if payload.get("status") is not None else None,
        str(result).strip() if result is not None else "",
    )


_REPORT_TAIL_CHARS = 20000


def implement_report_artifacts(
    task: Task,
    worker_id: str,
    result_text: str,
    *,
    adapter: str,
) -> list[Artifact]:
    """Turn a full-edit worker's final message into durable artifacts.

    Field report (the v0.9.40 CI fix): a cursor implement worker correctly
    diagnosed both failures and shipped the right patch, but its final report
    lived only in a stdout tail — the stitched summary showed "Findings: None"
    and a perfectly good run read as degraded. Structured JSON parses into
    typed artifacts; anything else is preserved verbatim as a FINDING so the
    report survives synthesis instead of dying in a log nobody reads.
    """
    parsed = cursor_result_artifacts(task, worker_id, result_text, adapter=adapter)
    if parsed:
        return parsed
    report = redact_secrets(result_text or "").strip()
    if not report:
        return []
    headline = next(
        (line.strip().lstrip("#").strip() for line in report.splitlines() if line.strip()),
        "Worker report",
    )
    return [
        Artifact(
            job_id=task.job_id,
            task_id=task.id,
            type=ArtifactType.FINDING,
            created_by=worker_id,
            confidence=0.75,
            evidence=[f"adapter:{adapter}", "report:final-message"],
            payload={
                "claim": headline[:300],
                "report": report[-_REPORT_TAIL_CHARS:],
            },
        )
    ]


def cursor_result_artifacts(
    task: Task,
    worker_id: str,
    result_text: object,
    *,
    adapter: str = "cursor-sdk",
) -> list[Artifact]:
    payload = parse_cursor_artifact_payload(result_text)
    if payload is None:
        return []

    raw_artifacts = []
    if isinstance(payload, list):
        raw_artifacts = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("artifacts"), list):
            raw_artifacts = payload["artifacts"]
        else:
            raw_artifacts.extend(_typed_items(payload, "findings", "finding"))
            raw_artifacts.extend(_typed_items(payload, "risks", "risk"))
            raw_artifacts.extend(_typed_items(payload, "decisions", "decision"))

    artifacts = []
    for item in raw_artifacts:
        artifact = cursor_artifact_from_item(task, worker_id, item, adapter=adapter)
        if artifact is not None:
            artifacts.append(artifact)
    return artifacts


def parse_cursor_artifact_payload(result_text: object) -> Optional[Any]:
    text = "" if result_text is None else str(result_text)
    text = text.strip()
    if not text:
        return None
    for candidate in [text, _strip_json_fence(text), _json_object_slice(text)]:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    # Final fallback: a valid JSON object/array followed by a brace-bearing
    # trailer (a Hermes -Q "[session abc | 1,240 tokens | $0.003]" footer, a
    # courtesy "Done {ok}", or a second JSON blob) makes every greedy
    # candidate above fail — the slice over-reaches into the trailer and
    # json.loads raises. raw_decode parses one complete value from the first
    # opener and ignores trailing junk, recovering the real payload without
    # salvaging genuine prose (which has no parseable opener → still None).
    return _json_prefix_decode(text)


def cursor_artifact_from_item(
    task: Task,
    worker_id: str,
    item: object,
    *,
    adapter: str = "cursor-sdk",
) -> Optional[Artifact]:
    if not isinstance(item, dict):
        return None
    artifact_type = str(item.get("type") or "").lower().strip()
    if artifact_type in {"findings", "swarm.finding"}:
        artifact_type = "finding"
    if artifact_type not in {"finding", "risk", "decision"}:
        return None

    evidence = _string_list(item.get("evidence")) or [f"adapter:{adapter}"]
    confidence = _confidence(item.get("confidence"))
    payload = {key: value for key, value in item.items() if key not in {"type", "evidence", "confidence"}}

    if artifact_type == "finding":
        claim = payload.get("claim") or payload.get("finding") or payload.get("summary")
        if not claim:
            return None
        payload["claim"] = str(claim)
        kind = ArtifactType.FINDING
    elif artifact_type == "risk":
        risk = payload.get("risk") or payload.get("claim") or payload.get("summary")
        if not risk:
            return None
        payload["risk"] = str(risk)
        payload["mitigation"] = str(payload.get("mitigation") or "Review and verify before implementation.")
        kind = ArtifactType.RISK
    else:
        decision = payload.get("decision") or payload.get("claim") or payload.get("summary")
        if not decision:
            return None
        payload["decision"] = str(decision)
        payload["why"] = str(payload.get("why") or "Recommended by automated analysis.")
        kind = ArtifactType.DECISION

    return Artifact(
        job_id=task.job_id,
        task_id=task.id,
        type=kind,
        created_by=worker_id,
        confidence=confidence,
        evidence=evidence,
        payload=payload,
    )


def cursor_degraded_artifact(
    task: Task,
    worker_id: str,
    result_text: str,
    *,
    adapter: str = "cursor-sdk",
    stdout_capture: Optional[dict[str, Any]] = None,
) -> Artifact:
    payload: dict[str, Any] = {
        "risk": f"{adapter} completed without structured Puppetmaster findings.",
        "mitigation": "Treat this swarm as degraded; rerun with a stricter prompt or inspect the repo directly before implementation.",
        "stdout_excerpt": result_text[:_STDOUT_HEAD_CHARS],
    }
    if stdout_capture is None:
        stdout_capture = capture_subprocess_stdout(
            text=result_text,
            task=task,
            sidecar_name="cursor_stdout",
        )
    payload["stdout_capture"] = stdout_capture
    return Artifact(
        job_id=task.job_id,
        task_id=task.id,
        type=ArtifactType.RISK,
        created_by=worker_id,
        confidence=0.85,
        evidence=[f"adapter:{adapter}", "cursor-result:empty-or-unstructured"],
        payload=payload,
    )


def _typed_items(payload: dict[str, Any], key: str, artifact_type: str) -> list[dict[str, Any]]:
    items = payload.get(key)
    if not isinstance(items, list):
        return []
    return [{**item, "type": item.get("type", artifact_type)} for item in items if isinstance(item, dict)]


def _strip_json_fence(text: str) -> Optional[str]:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return None
    lines = stripped.splitlines()
    if len(lines) < 3:
        return None
    return "\n".join(lines[1:-1]).strip()


def _json_object_slice(text: str) -> Optional[str]:
    starts = [index for index in [text.find("{"), text.find("[")] if index >= 0]
    if not starts:
        return None
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    return text[start : end + 1] if end > start else None


def _json_prefix_decode(text: str) -> Optional[Any]:
    """Decode the first complete JSON object/array, ignoring any trailing junk.

    Scans for each ``{``/``[`` opener and lets ``raw_decode`` consume exactly
    one JSON value from there, so a valid payload followed by a non-JSON trailer
    still parses. Only structured containers count as artifact payloads — a bare
    scalar that happens to sit after a brace (or genuine prose with no parseable
    opener) yields ``None`` so real degrades aren't masked.
    """
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            decoded, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, (dict, list)):
            return decoded
    return None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _confidence(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.75
    return min(1.0, max(0.0, parsed))

