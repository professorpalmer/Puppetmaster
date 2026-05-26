from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Union

from puppetmaster.codegraph import enrich_prompt_with_codegraph
from puppetmaster.models import Artifact, ArtifactType, Task


@dataclass(frozen=True)
class AdapterInfo:
    name: str
    status: str
    description: str
    requires: list[str]


class WorkerAdapter(Protocol):
    name: str

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        """Execute a task and return structured artifacts."""


class LocalAdapter:
    name = "local"

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        role_map = {
            "explore": self._explore,
            "architect": self._architect,
            "implement": self._implement,
            "redteam": self._redteam,
            "test": self._test,
        }
        return role_map.get(task.role, self._explore)(task, goal, worker_id)

    def _explore(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        return [
            Artifact(
                job_id=task.job_id,
                task_id=task.id,
                type=ArtifactType.FINDING,
                created_by=worker_id,
                confidence=0.9,
                evidence=["concept:independent-workers", "concept:structured-artifacts"],
                payload={
                    "claim": "The swarm should share durable state, not inherited transcript.",
                    "goal": goal,
                    "implication": "Workers can stay small, replayable, and independently replaceable.",
                },
            ),
            Artifact(
                job_id=task.job_id,
                task_id=task.id,
                type=ArtifactType.FINDING,
                created_by=worker_id,
                confidence=0.84,
                evidence=["concept:redis-gunicorn-analogy"],
                payload={
                    "claim": "Redis/Gunicorn is a useful operating model for agent swarms.",
                    "implication": "Use streams, locks, worker runs, and artifacts as first-class objects.",
                },
            ),
        ]

    def _architect(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        return [
            Artifact(
                job_id=task.job_id,
                task_id=task.id,
                type=ArtifactType.DECISION,
                created_by=worker_id,
                confidence=0.88,
                evidence=["mvp:local-filesystem", "mvp:no-runtime-dependencies"],
                payload={
                    "decision": "Start with a file-backed store that mirrors Redis key spaces.",
                    "why": "It makes jobs, streams, artifacts, and memory inspectable before adding Redis.",
                    "next": "Swap the store adapter for Redis Streams when live coordination matters.",
                },
            ),
            Artifact(
                job_id=task.job_id,
                task_id=task.id,
                type=ArtifactType.DECISION,
                created_by=worker_id,
                confidence=0.86,
                evidence=["rule:state-not-context"],
                payload={
                    "decision": "Synthesis reads JSON artifacts only.",
                    "why": "The final agent should not need raw worker transcripts to reconstruct state.",
                    "goal": goal,
                },
            ),
        ]

    def _implement(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        return [
            Artifact(
                job_id=task.job_id,
                task_id=task.id,
                type=ArtifactType.PATCH,
                created_by=worker_id,
                confidence=0.82,
                evidence=["module:orchestrator", "module:store", "module:stitcher"],
                payload={
                    "change": "Create a CLI that spawns role-specific workers and writes artifacts.",
                    "files": [
                        "puppetmaster/orchestrator.py",
                        "puppetmaster/store.py",
                        "puppetmaster/stitcher.py",
                    ],
                    "goal": goal,
                },
            )
        ]

    def _redteam(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        return [
            Artifact(
                job_id=task.job_id,
                task_id=task.id,
                type=ArtifactType.RISK,
                created_by=worker_id,
                confidence=0.8,
                evidence=["risk:artifact-quality"],
                payload={
                    "risk": "If workers emit vague prose, stitching collapses into summary theater.",
                    "mitigation": "Validate artifact type, confidence, evidence, and payload shape.",
                    "goal": goal,
                },
            ),
            Artifact(
                job_id=task.job_id,
                task_id=task.id,
                type=ArtifactType.RISK,
                created_by=worker_id,
                confidence=0.76,
                evidence=["risk:write-conflicts"],
                payload={
                    "risk": "Independent workers can collide when touching the same scope.",
                    "mitigation": "Use locks for file scopes and promote memory only after synthesis.",
                },
            ),
        ]

    def _test(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        return [
            Artifact(
                job_id=task.job_id,
                task_id=task.id,
                type=ArtifactType.VERIFICATION,
                created_by=worker_id,
                confidence=0.81,
                evidence=["test:job-produces-artifacts", "test:stitcher-promotes-memory"],
                payload={
                    "check": "A run should produce tasks, artifacts, a stitched summary, and promoted memory.",
                    "result": "designed",
                    "goal": goal,
                },
            )
        ]


class ShellAdapter:
    name = "shell"

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        command = task.payload.get("command")
        if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
            raise ValueError("shell adapter requires payload.command as a list of strings")

        timeout_seconds = int(task.payload.get("timeout_seconds", 30))
        cwd = task.payload.get("cwd")
        try:
            completed = subprocess.run(
                command,
                cwd=str(Path(cwd)) if cwd else None,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="shell",
                    check=task.instruction,
                    result="failed",
                    confidence=0.7,
                    evidence=[f"command:{' '.join(command)}", "timeout"],
                    payload={
                        "returncode": None,
                        "stdout": (exc.stdout or "")[-4000:],
                        "stderr": (exc.stderr or "")[-4000:],
                        "timeout_seconds": timeout_seconds,
                    },
                )
            ]
        return [
            verification_artifact(
                task=task,
                worker_id=worker_id,
                adapter="shell",
                check=task.instruction,
                result="passed" if completed.returncode == 0 else "failed",
                confidence=0.95 if completed.returncode == 0 else 0.65,
                evidence=[f"command:{' '.join(command)}"],
                payload={
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-4000:],
                    "stderr": completed.stderr[-4000:],
                },
            )
        ]


class CursorAdapter:
    name = "cursor"

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        base_prompt = task.payload.get("prompt") or task.instruction
        cwd = task.payload.get("cwd")
        model = task.payload.get("model", "default")
        prompt, codegraph_used = enrich_prompt_with_codegraph(
            self._prompt_with_memory(
                self._structured_prompt(base_prompt),
                task,
            ),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )
        runner = Path(__file__).with_name("cursor_sdk_runner.mjs")
        environment = os.environ.copy()
        environment["PUPPETMASTER_CURSOR_INPUT"] = json.dumps(
            {"prompt": prompt, "cwd": cwd, "model": model},
            sort_keys=True,
        )
        try:
            completed = subprocess.run(
                ["node", str(runner)],
                capture_output=True,
                text=True,
                timeout=int(task.payload.get("timeout_seconds", 300)),
                check=False,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            stderr = exc.stderr or ""
            stdout = exc.stdout or ""
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
                        "stdout": stdout[-8000:],
                        "stderr": stderr[-8000:],
                        "model": model,
                        "failure": classify_cursor_failure(stderr + stdout),
                    },
                )
            ]
        failure = classify_cursor_failure(completed.stderr + completed.stdout)
        status, result_text = cursor_result_text(completed.stdout)
        parsed_artifacts = (
            cursor_result_artifacts(task, worker_id, result_text)
            if completed.returncode == 0
            else []
        )
        degraded = completed.returncode == 0 and not parsed_artifacts
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
                    "stdout": completed.stdout[-8000:],
                    "stderr": completed.stderr[-8000:],
                    "model": model,
                    "cursor_status": status,
                    "failure": (
                        "empty_or_unstructured_cursor_result"
                        if degraded
                        else failure
                        if completed.returncode != 0
                        else None
                    ),
                },
            )
        ]
        if degraded:
            artifacts.append(cursor_degraded_artifact(task, worker_id, result_text))
        artifacts.extend(parsed_artifacts)
        return artifacts

    @staticmethod
    def _structured_prompt(prompt: str) -> str:
        return "\n".join(
            [
                prompt,
                "",
                "Puppetmaster artifact contract:",
                "Return only JSON, with no markdown wrapper, in this shape:",
                '{"artifacts":[{"type":"finding","claim":"...","evidence":["path or symbol"],"confidence":0.8}]}',
                "Allowed artifact types:",
                '- finding: requires "claim", "evidence", "confidence".',
                '- risk: requires "risk", "mitigation", "evidence", "confidence".',
                '- decision: requires "decision", "why", "evidence", "confidence".',
                "Use concrete file/function evidence. If there are no concrete findings, return a risk artifact explaining why the run is degraded.",
            ]
        )

    @staticmethod
    def _prompt_with_memory(prompt: str, task: Task) -> str:
        retrieved = task.payload.get("retrieved_memory") or []
        if not retrieved:
            return prompt
        lines = [
            prompt,
            "",
            "Relevant promoted Puppetmaster memory:",
        ]
        for memory in retrieved[:5]:
            lines.append(
                f"- [{memory.get('scope', 'memory')}] {memory.get('statement', '')}"
            )
        lines.append("")
        lines.append("Use this as retrieved context, but verify claims before relying on them.")
        return "\n".join(lines)


class ClaudeCodeAdapter:
    name = "claude-code"

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        base_prompt = task.payload.get("prompt") or task.instruction
        cwd = Path(task.payload.get("cwd") or ".").resolve()
        prompt, codegraph_used = enrich_prompt_with_codegraph(
            prompt_with_memory(base_prompt, task),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )
        timeout_seconds = int(task.payload.get("timeout_seconds", 600))
        executable = task.payload.get("executable") or os.environ.get("CLAUDE_CODE_COMMAND") or "claude"
        command_base = command_parts(executable)
        resolved = resolve_command(command_base[0])
        if resolved is None:
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="claude-code",
                    check=task.instruction,
                    result="blocked",
                    confidence=0.45,
                    evidence=["adapter:claude-code", "status:missing-cli"],
                    payload={
                        "failure": "missing_cli",
                        "message": (
                            "Claude Code CLI was not found. Install it or set "
                            "CLAUDE_CODE_COMMAND / payload.executable."
                        ),
                        "executable": executable,
                    },
                )
            ]

        command = build_claude_code_command(
            prompt=prompt,
            executable=[resolved, *command_base[1:]],
            model=task.payload.get("model"),
            output_format=task.payload.get("output_format", "json"),
            permission_mode=task.payload.get("permission_mode", "acceptEdits"),
            allowed_tools=task.payload.get("allowed_tools"),
            disallowed_tools=task.payload.get("disallowed_tools"),
            extra_args=task.payload.get("extra_args", []),
        )
        before = git_snapshot(cwd)
        if not task.payload.get("allow_dirty", False) and (
            before["changed_files"] or before["untracked_files"]
        ):
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="claude-code",
                    check=task.instruction,
                    result="blocked",
                    confidence=0.8,
                    evidence=["adapter:claude-code", "status:dirty-repo"],
                    payload={
                        "failure": "dirty_worktree",
                        "message": (
                            "Claude Code full-edit runs require a clean working tree by default "
                            "so Puppetmaster can attribute resulting diffs correctly. Commit, stash, "
                            "use a worktree, or set payload.allow_dirty=true."
                        ),
                        "changed_files": before["changed_files"],
                        "untracked_files": before["untracked_files"],
                    },
                )
            ]
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                input="",
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="claude-code",
                    check=task.instruction,
                    result="failed",
                    confidence=0.6,
                    evidence=["adapter:claude-code", "timeout"],
                    payload={
                        "failure": "timeout",
                        "returncode": None,
                        "stdout": stdout[-8000:],
                        "stderr": stderr[-8000:],
                        "timeout_seconds": timeout_seconds,
                    },
                )
            ]

        after = git_snapshot(cwd)
        verification = verification_artifact(
            task=task,
            worker_id=worker_id,
            adapter="claude-code",
            check=task.instruction,
            result="passed" if completed.returncode == 0 else "failed",
            confidence=0.9 if completed.returncode == 0 else 0.55,
            evidence=(
                [
                    "adapter:claude-code",
                    f"permission_mode:{task.payload.get('permission_mode', 'acceptEdits')}",
                ]
                + (["context:codegraph"] if codegraph_used else [])
            ),
            payload={
                "failure": None if completed.returncode == 0 else classify_claude_code_failure(completed.stderr + completed.stdout),
                "returncode": completed.returncode,
                "stdout": completed.stdout[-12000:],
                "stderr": completed.stderr[-8000:],
                "cwd": str(cwd),
                "permission_mode": task.payload.get("permission_mode", "acceptEdits"),
                "base_sha": before["sha"],
                "head_sha": after["sha"],
                "changed_files": after["changed_files"],
                "untracked_files": after["untracked_files"],
            },
        )
        artifacts = [verification]
        if after["diff"].strip():
            artifacts.append(
                Artifact(
                    job_id=task.job_id,
                    task_id=task.id,
                    type=ArtifactType.PATCH,
                    created_by=worker_id,
                    confidence=0.8 if completed.returncode == 0 else 0.5,
                    evidence=["adapter:claude-code", f"base:{before['sha']}"],
                    payload={
                        "change": "Claude Code modified tracked repository files.",
                        "files": after["changed_files"],
                        "unified_diff": after["diff"][-20000:],
                        "base_sha": before["sha"],
                        "head_sha": after["sha"],
                        "status": "applied" if completed.returncode == 0 else "failed",
                        "revert": "Review the diff, then use git restore / git checkout or your VCS workflow to revert unwanted changes.",
                    },
                )
            )
        return artifacts


class UnconfiguredProviderAdapter:
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        return [
            verification_artifact(
                task=task,
                worker_id=worker_id,
                adapter=self.name,
                check=task.instruction,
                result="blocked",
                confidence=0.4,
                evidence=[f"adapter:{self.name}", "status:not-configured"],
                payload={
                    "message": (
                        f"{self.description} is a provider stub. Add a concrete adapter "
                        "implementation before using it for live work."
                    )
                },
            )
        ]


def verification_artifact(
    *,
    task: Task,
    worker_id: str,
    adapter: str,
    check: str,
    result: str,
    confidence: float,
    evidence: list[str],
    payload: dict,
) -> Artifact:
    return Artifact(
        job_id=task.job_id,
        task_id=task.id,
        type=ArtifactType.VERIFICATION,
        created_by=worker_id,
        confidence=confidence,
        evidence=evidence,
        payload={
            "adapter": adapter,
            "check": check,
            "result": result,
            **payload,
        },
    )


def prompt_with_memory(prompt: str, task: Task) -> str:
    retrieved = task.payload.get("retrieved_memory") or []
    if not retrieved:
        return prompt
    lines = [
        prompt,
        "",
        "Relevant promoted Puppetmaster memory:",
    ]
    for memory in retrieved[:5]:
        lines.append(f"- [{memory.get('scope', 'memory')}] {memory.get('statement', '')}")
    lines.append("")
    lines.append("Use this as retrieved context, but verify claims before relying on them.")
    return "\n".join(lines)


def command_parts(command: object) -> list[str]:
    if isinstance(command, list) and all(isinstance(part, str) for part in command):
        return command
    if isinstance(command, str):
        return shlex.split(command)
    raise ValueError("provider executable must be a string or list of strings")


def resolve_command(executable: str) -> Optional[str]:
    path = Path(executable).expanduser()
    if path.exists():
        return str(path)
    return shutil.which(executable)


def build_claude_code_command(
    *,
    prompt: str,
    executable: Union[str, list[str]] = "claude",
    model: object = None,
    output_format: str = "json",
    permission_mode: str = "acceptEdits",
    allowed_tools: object = None,
    disallowed_tools: object = None,
    extra_args: object = None,
) -> list[str]:
    command = command_parts(executable)
    command.extend(["--print", prompt, "--output-format", output_format])
    if model:
        command.extend(["--model", str(model)])
    if permission_mode:
        command.extend(["--permission-mode", permission_mode])
    if allowed_tools:
        command.extend(["--allowedTools", tool_list(allowed_tools)])
    if disallowed_tools:
        command.extend(["--disallowedTools", tool_list(disallowed_tools)])
    if extra_args:
        command.extend(command_parts(extra_args))
    return command


def tool_list(value: object) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def git_snapshot(cwd: Path) -> dict[str, object]:
    return {
        "sha": git_output(cwd, ["rev-parse", "HEAD"]) or "uncommitted",
        "changed_files": git_lines(cwd, ["diff", "--name-only"]),
        "untracked_files": git_untracked_files(cwd),
        "diff": git_output(cwd, ["diff", "--binary"]) or "",
    }


def git_output(cwd: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def git_lines(cwd: Path, args: list[str]) -> list[str]:
    output = git_output(cwd, args)
    return [line for line in output.splitlines() if line.strip()]


def git_untracked_files(cwd: Path) -> list[str]:
    output = git_output(cwd, ["status", "--short"])
    files = []
    for line in output.splitlines():
        if line.startswith("?? "):
            files.append(line[3:])
    return files


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


def cursor_result_artifacts(task: Task, worker_id: str, result_text: str) -> list[Artifact]:
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
        artifact = cursor_artifact_from_item(task, worker_id, item)
        if artifact is not None:
            artifacts.append(artifact)
    return artifacts


def parse_cursor_artifact_payload(result_text: str) -> Optional[Any]:
    text = result_text.strip()
    if not text:
        return None
    for candidate in [text, _strip_json_fence(text), _json_object_slice(text)]:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def cursor_artifact_from_item(task: Task, worker_id: str, item: object) -> Optional[Artifact]:
    if not isinstance(item, dict):
        return None
    artifact_type = str(item.get("type") or "").lower().strip()
    if artifact_type in {"findings", "swarm.finding"}:
        artifact_type = "finding"
    if artifact_type not in {"finding", "risk", "decision"}:
        return None

    evidence = _string_list(item.get("evidence")) or ["adapter:cursor-sdk"]
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
        payload["why"] = str(payload.get("why") or "Recommended by Cursor SDK analysis.")
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


def cursor_degraded_artifact(task: Task, worker_id: str, result_text: str) -> Artifact:
    return Artifact(
        job_id=task.job_id,
        task_id=task.id,
        type=ArtifactType.RISK,
        created_by=worker_id,
        confidence=0.85,
        evidence=["adapter:cursor-sdk", "cursor-result:empty-or-unstructured"],
        payload={
            "risk": "Cursor SDK completed without structured Puppetmaster findings.",
            "mitigation": "Treat this swarm as degraded; rerun with a stricter prompt or inspect the repo directly before implementation.",
            "stdout_excerpt": result_text[:1000],
        },
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


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"


class OpenAIAdapter:
    """Calls the OpenAI Chat Completions API directly via OPENAI_API_KEY.

    Mirrors CursorAdapter semantics: enrich the prompt with CodeGraph + promoted
    memory, demand a strict JSON artifact contract, then parse and emit the same
    finding/risk/decision artifacts that the rest of Puppetmaster reasons over.
    """

    name = "openai"

    def run(self, task: Task, goal: str, worker_id: str) -> list[Artifact]:
        base_prompt = task.payload.get("prompt") or task.instruction
        cwd = task.payload.get("cwd")
        model = task.payload.get("model") or DEFAULT_OPENAI_MODEL
        prompt, codegraph_used = enrich_prompt_with_codegraph(
            prompt_with_memory(self._structured_prompt(base_prompt), task),
            task_description=task.payload.get("codegraph_task") or task.instruction or goal,
            cwd=cwd,
            disabled=bool(task.payload.get("disable_codegraph", False)),
        )

        api_key = task.payload.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
        base_url = (
            task.payload.get("openai_base_url")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_OPENAI_BASE_URL
        ).rstrip("/")
        organization = task.payload.get("openai_organization") or os.environ.get(
            "OPENAI_ORG_ID"
        )
        timeout_seconds = int(task.payload.get("timeout_seconds", 300))
        evidence_base = [
            "adapter:openai",
            f"model:{model}",
        ] + (["context:codegraph"] if codegraph_used else [])

        if not api_key:
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="openai",
                    check=task.instruction,
                    result="failed",
                    confidence=0.5,
                    evidence=evidence_base + ["missing_api_key"],
                    payload={
                        "returncode": None,
                        "model": model,
                        "failure": "missing_api_key",
                        "stderr": (
                            "OPENAI_API_KEY is not set. Export it or pass openai_api_key "
                            "in the task payload."
                        ),
                    },
                )
            ]

        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        # Optional knobs — only included when the caller explicitly opts in.
        # OpenAI's GPT-5+ family deprecated `max_tokens` in favor of
        # `max_completion_tokens`. We send the new name by default and let
        # the caller force the legacy name for OpenAI-compatible providers
        # that still require it.
        max_completion = task.payload.get(
            "max_output_tokens",
            task.payload.get("max_completion_tokens", task.payload.get("max_tokens")),
        )
        if max_completion is not None:
            if task.payload.get("legacy_max_tokens"):
                body["max_tokens"] = int(max_completion)
            else:
                body["max_completion_tokens"] = int(max_completion)
        if "temperature" in task.payload:
            body["temperature"] = float(task.payload["temperature"])
        if task.payload.get("reasoning_effort"):
            body["reasoning_effort"] = str(task.payload["reasoning_effort"])

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "puppetmaster-openai-adapter",
        }
        if organization:
            headers["OpenAI-Organization"] = str(organization)

        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                status_code = response.getcode()
                raw_body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:  # 4xx/5xx
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="openai",
                    check=task.instruction,
                    result="failed",
                    confidence=0.55,
                    evidence=evidence_base + [f"http_status:{exc.code}"],
                    payload={
                        "returncode": exc.code,
                        "model": model,
                        "failure": classify_openai_failure(err_body, exc.code),
                        "stderr": err_body[-8000:],
                    },
                )
            ]
        except (socket.timeout, TimeoutError):
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="openai",
                    check=task.instruction,
                    result="failed",
                    confidence=0.55,
                    evidence=evidence_base + ["timeout"],
                    payload={
                        "returncode": None,
                        "model": model,
                        "failure": "timeout",
                        "stderr": f"OpenAI request exceeded {timeout_seconds}s",
                    },
                )
            ]
        except urllib.error.URLError as exc:
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="openai",
                    check=task.instruction,
                    result="failed",
                    confidence=0.55,
                    evidence=evidence_base + ["network_error"],
                    payload={
                        "returncode": None,
                        "model": model,
                        "failure": "network_error",
                        "stderr": str(exc),
                    },
                )
            ]

        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError:
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="openai",
                    check=task.instruction,
                    result="failed",
                    confidence=0.55,
                    evidence=evidence_base + ["malformed_response"],
                    payload={
                        "returncode": status_code,
                        "model": model,
                        "failure": "malformed_response",
                        "stderr": raw_body[-8000:],
                    },
                )
            ]

        choices = data.get("choices") or []
        message = (
            choices[0].get("message", {}) if isinstance(choices, list) and choices else {}
        )
        result_text = str(message.get("content") or "").strip()
        finish_reason = (
            choices[0].get("finish_reason") if isinstance(choices, list) and choices else None
        )
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))

        parsed_artifacts = cursor_result_artifacts(task, worker_id, result_text)
        degraded = not parsed_artifacts and bool(result_text)
        failed = finish_reason not in (None, "stop", "length") and not parsed_artifacts

        verification = verification_artifact(
            task=task,
            worker_id=worker_id,
            adapter="openai",
            check=task.instruction,
            result=(
                "failed"
                if failed
                else "degraded"
                if degraded
                else "passed"
            ),
            confidence=0.55 if failed else 0.65 if degraded else 0.9,
            evidence=evidence_base + [f"finish:{finish_reason}"] if finish_reason else evidence_base,
            payload={
                "returncode": status_code,
                "model": model,
                "finish_reason": finish_reason,
                "stdout": result_text[-8000:],
                "stderr": "",
                "tokens_in": prompt_tokens,
                "tokens_out": completion_tokens,
                "tokens_total": total_tokens,
                "failure": (
                    "empty_or_unstructured_openai_result"
                    if degraded
                    else None
                    if not failed
                    else "unexpected_finish_reason"
                ),
            },
        )
        artifacts: list[Artifact] = [verification]
        if degraded:
            artifacts.append(
                Artifact(
                    job_id=task.job_id,
                    task_id=task.id,
                    type=ArtifactType.RISK,
                    created_by=worker_id,
                    confidence=0.85,
                    evidence=["adapter:openai", "result:empty-or-unstructured"],
                    payload={
                        "risk": "OpenAI call completed without structured Puppetmaster findings.",
                        "mitigation": (
                            "Treat this swarm as degraded; rerun with a stricter prompt or "
                            "inspect the repo directly before implementation."
                        ),
                        "stdout_excerpt": result_text[:1000],
                    },
                )
            )
        artifacts.extend(parsed_artifacts)
        return artifacts

    @staticmethod
    def _structured_prompt(prompt: str) -> str:
        return "\n".join(
            [
                prompt,
                "",
                "Puppetmaster artifact contract:",
                "Return only JSON, with no markdown wrapper, in this shape:",
                '{"artifacts":[{"type":"finding","claim":"...","evidence":["path or symbol"],"confidence":0.8}]}',
                "Allowed artifact types:",
                '- finding: requires "claim", "evidence", "confidence".',
                '- risk: requires "risk", "mitigation", "evidence", "confidence".',
                '- decision: requires "decision", "why", "evidence", "confidence".',
                "Use concrete file/function evidence. If there are no concrete findings, return a risk artifact explaining why the run is degraded.",
            ]
        )


def classify_openai_failure(body: str, http_status: Optional[int] = None) -> str:
    if http_status == 401:
        return "missing_api_key"
    if http_status == 403:
        return "forbidden"
    if http_status == 404:
        return "model_unavailable"
    if http_status == 429:
        return "rate_limit"
    if http_status is not None and 500 <= http_status < 600:
        return "openai_server_error"
    lowered = (body or "").lower()
    if "api key" in lowered or "authorization" in lowered:
        return "missing_api_key"
    if "rate limit" in lowered:
        return "rate_limit"
    if "model" in lowered and ("not found" in lowered or "unavailable" in lowered):
        return "model_unavailable"
    if "context length" in lowered or "maximum context" in lowered:
        return "context_length_exceeded"
    return "unknown"


ADAPTERS: dict[str, WorkerAdapter] = {
    "local": LocalAdapter(),
    "shell": ShellAdapter(),
    "cursor": CursorAdapter(),
    "claude-code": ClaudeCodeAdapter(),
    "openai": OpenAIAdapter(),
    "codex": UnconfiguredProviderAdapter("codex", "Codex"),
}


ADAPTER_INFO = [
    AdapterInfo(
        name="local",
        status="built-in",
        description="Deterministic structured artifacts for demo/runtime roles.",
        requires=[],
    ),
    AdapterInfo(
        name="shell",
        status="built-in",
        description="Runs bounded shell commands and emits verification artifacts.",
        requires=[],
    ),
    AdapterInfo(
        name="cursor",
        status="optional",
        description="Runs local Cursor SDK one-shot agents.",
        requires=["node", "npm install", "CURSOR_API_KEY"],
    ),
    AdapterInfo(
        name="claude-code",
        status="optional",
        description="Runs the Claude Code CLI in non-interactive full-edit mode.",
        requires=["claude CLI", "Claude Code auth"],
    ),
    AdapterInfo(
        name="openai",
        status="optional",
        description="Calls the OpenAI Chat Completions API directly with OPENAI_API_KEY.",
        requires=["OPENAI_API_KEY"],
    ),
    AdapterInfo(
        name="codex",
        status="stub",
        description="Provider-neutral placeholder for a future Codex adapter.",
        requires=["adapter implementation"],
    ),
]


def get_adapter(name: str) -> WorkerAdapter:
    if name not in ADAPTERS:
        raise ValueError(f"unsupported adapter: {name}")
    return ADAPTERS[name]


def classify_cursor_failure(output: str) -> str:
    lowered = output.lower()
    if "cursor_api_key" in lowered or "api key" in lowered:
        return "missing_api_key"
    if "cannot find package" in lowered or "@cursor/sdk" in lowered and "not found" in lowered:
        return "sdk_not_installed"
    if "model" in lowered and ("unavailable" in lowered or "not found" in lowered or "invalid" in lowered):
        return "model_unavailable"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "status" in lowered and "error" in lowered:
        return "run_status_error"
    return "unknown"


def classify_claude_code_failure(output: str) -> str:
    lowered = output.lower()
    if "credit balance" in lowered or "billing" in lowered or "quota" in lowered:
        return "billing_or_quota"
    if "command not found" in lowered or "not found" in lowered:
        return "missing_cli"
    if "auth" in lowered or "login" in lowered or "api key" in lowered:
        return "not_authenticated"
    if "permission" in lowered or "not allowed" in lowered or "denied" in lowered:
        return "permission_denied"
    if "model" in lowered and ("unavailable" in lowered or "invalid" in lowered or "not found" in lowered):
        return "model_unavailable"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    return "unknown"

