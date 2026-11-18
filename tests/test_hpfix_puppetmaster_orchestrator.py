from __future__ import annotations

import unittest
from dataclasses import replace
from tempfile import TemporaryDirectory

from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
from puppetmaster.orchestrator import Orchestrator
from puppetmaster.store import SwarmStore


def _auth_failure(job_id: str, task_id: str, artifact_id: str) -> Artifact:
    return Artifact(
        job_id=job_id, task_id=task_id, type=ArtifactType.VERIFICATION,
        created_by="w",
        payload={"failure": "auth", "passed": False, "check": "auth", "result": "fail"},
        confidence=0.5, evidence=["auth rejected"], id=artifact_id,
    )


def _fallback_marker(job_id: str, task_id: str, attempt: int, artifact_id: str) -> Artifact:
    return Artifact(
        job_id=job_id, task_id=task_id, type=ArtifactType.ROUTING,
        created_by="router-fallback",
        payload={"model_id": "claude-code/opus-5-5", "adapter": "claude-code",
                 "policy": "balanced", "fallback_attempt": attempt,
                 "fallback_reason": "auth"},
        confidence=0.9, evidence=["reason:auth"], id=artifact_id,
    )


class StaleFallbackFailureTests(unittest.TestCase):
    def _setup(self, tmp: str):
        store = SwarmStore(tmp)
        job = store.create_job("goal")
        task = Task(job_id=job.id, role="explore", instruction="x", id="task_a",
                    adapter="cursor", status=TaskStatus.FAILED)
        store.save_task(task)
        store.save_artifact(_auth_failure(job.id, task.id, "artifact_000000000001"))
        return store, job, task

    def test_hard_failure_after_requeue_without_marker(self) -> None:
        with TemporaryDirectory() as tmp:
            store, job, task = self._setup(tmp)
            store.save_task(replace(task, adapter="claude-code",
                                    payload={"fallback_attempts": 1},
                                    status=TaskStatus.FAILED))
            orch = Orchestrator(store)
            self.assertEqual(orch._recoverable_failure_by_task(job), {})
            self.assertTrue(orch._has_hard_failure(job, {task.id}))
            self.assertTrue(orch._should_fail_closed(job, {task.id}))

    def test_hard_failure_after_requeue_with_marker(self) -> None:
        with TemporaryDirectory() as tmp:
            store, job, task = self._setup(tmp)
            store.save_artifact(_fallback_marker(job.id, task.id, 1, "artifact_000000000002"))
            store.save_task(replace(task, adapter="claude-code",
                                    payload={"fallback_attempts": 1},
                                    status=TaskStatus.FAILED))
            orch = Orchestrator(store)
            self.assertEqual(orch._recoverable_failure_by_task(job), {})
            self.assertTrue(orch._has_hard_failure(job, {task.id}))

    def test_new_recoverable_failure_after_requeue_still_counts(self) -> None:
        with TemporaryDirectory() as tmp:
            store, job, task = self._setup(tmp)
            store.save_artifact(_fallback_marker(job.id, task.id, 1, "artifact_000000000002"))
            store.save_task(replace(task, adapter="claude-code",
                                    payload={"fallback_attempts": 1},
                                    status=TaskStatus.FAILED))
            store.save_artifact(_auth_failure(job.id, task.id, "artifact_000000000003"))
            orch = Orchestrator(store)
            self.assertEqual(orch._recoverable_failure_by_task(job), {task.id: "auth"})
            self.assertFalse(orch._has_hard_failure(job, {task.id}))

    def test_first_attempt_failure_unchanged(self) -> None:
        with TemporaryDirectory() as tmp:
            store, job, task = self._setup(tmp)
            orch = Orchestrator(store)
            self.assertEqual(orch._recoverable_failure_by_task(job), {task.id: "auth"})
            self.assertFalse(orch._has_hard_failure(job, {task.id}))
            self.assertFalse(orch._should_fail_closed(job, {task.id}))


if __name__ == "__main__":
    unittest.main()
