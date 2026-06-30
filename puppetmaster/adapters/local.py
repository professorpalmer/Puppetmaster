from __future__ import annotations

import subprocess
from pathlib import Path

from puppetmaster.models import Artifact, ArtifactType, Task
from puppetmaster.redaction import redact_secrets

from ._base import verification_artifact
from ._facade import facade
from ._streaming import _redacted_tail

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
            completed = facade("subprocess").run(
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
                        "stdout": _redacted_tail(exc.stdout, 4000),
                        "stderr": _redacted_tail(exc.stderr, 4000),
                        "timeout_seconds": timeout_seconds,
                    },
                )
            ]
        except OSError as exc:
            # A missing executable or invalid cwd raises OSError
            # (FileNotFoundError/NotADirectoryError) before any output exists.
            # Surface a failed verification artifact instead of letting the
            # exception escape and crash the worker run.
            message = redact_secrets(f"{type(exc).__name__}: {exc}") or "spawn_error"
            return [
                verification_artifact(
                    task=task,
                    worker_id=worker_id,
                    adapter="shell",
                    check=task.instruction,
                    result="failed",
                    confidence=0.7,
                    evidence=[f"command:{' '.join(command)}", "spawn_error"],
                    payload={
                        "returncode": None,
                        "failure": "spawn_error",
                        "executable": command[0] if command else None,
                        "cwd": str(cwd) if cwd else None,
                        "message": message,
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
                    "stdout": _redacted_tail(completed.stdout, 4000),
                    "stderr": _redacted_tail(completed.stderr, 4000),
                },
            )
        ]


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

