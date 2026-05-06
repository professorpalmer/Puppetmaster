from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Union

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
        prompt = self._prompt_with_memory(task.payload.get("prompt") or task.instruction, task)
        cwd = task.payload.get("cwd")
        model = task.payload.get("model", "default")
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
        return [
            verification_artifact(
                task=task,
                worker_id=worker_id,
                adapter="cursor",
                check=task.instruction,
                result="passed" if completed.returncode == 0 else "failed",
                confidence=0.9 if completed.returncode == 0 else 0.55,
                evidence=["adapter:cursor-sdk", f"node:{sys.platform}"],
                payload={
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-8000:],
                    "stderr": completed.stderr[-8000:],
                    "model": model,
                    "failure": failure if completed.returncode != 0 else None,
                },
            )
        ]

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
        prompt = prompt_with_memory(task.payload.get("prompt") or task.instruction, task)
        cwd = Path(task.payload.get("cwd") or ".").resolve()
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
            evidence=["adapter:claude-code", f"permission_mode:{task.payload.get('permission_mode', 'acceptEdits')}"],
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


ADAPTERS: dict[str, WorkerAdapter] = {
    "local": LocalAdapter(),
    "shell": ShellAdapter(),
    "cursor": CursorAdapter(),
    "claude-code": ClaudeCodeAdapter(),
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

