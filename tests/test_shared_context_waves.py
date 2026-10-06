"""Waves 2–4: selective unfold, adaptive enqueue, frontier observability."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from puppetmaster.attempts import ExecutionAttempt
from puppetmaster.budget import BudgetLiability, BudgetPolicy
from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.store import SwarmStore


class AdaptiveEnqueueTests(unittest.TestCase):
    def test_enqueue_subtask_links_parent_and_dedupes(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("enqueue")
            parent = Task(
                job_id=job.id,
                role="explore",
                instruction="explore root",
                status=TaskStatus.COMPLETE,
            )
            store.save_task(parent)

            child = store.enqueue_subtask(
                job.id,
                parent_task_id=parent.id,
                role="audit",
                instruction="dig into module X",
                created_by="worker-1",
            )
            self.assertIsNotNone(child)
            assert child is not None
            self.assertEqual(child.depends_on, [parent.id])
            self.assertTrue(child.payload.get("enqueued_from_parent"))
            self.assertEqual(child.payload.get("enqueue_depth"), 1)

            dup = store.enqueue_subtask(
                job.id,
                parent_task_id=parent.id,
                role="audit",
                instruction="dig into module X",
            )
            self.assertIsNotNone(dup)
            assert dup is not None
            self.assertEqual(dup.id, child.id)

            events = store.read_events(job.id)
            self.assertTrue(any(e.get("event") == "task.enqueued" for e in events))
            self.assertTrue(
                any(e.get("event") == "task.enqueue_deduped" for e in events)
            )

    def test_enqueue_respects_depth_limit(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("depth")
            root = Task(
                job_id=job.id,
                role="explore",
                instruction="root",
                status=TaskStatus.COMPLETE,
            )
            store.save_task(root)
            current = root
            for index in range(3):
                child = store.enqueue_subtask(
                    job.id,
                    parent_task_id=current.id,
                    role="explore",
                    instruction=f"layer {index}",
                    max_depth=3,
                )
                self.assertIsNotNone(child)
                assert child is not None
                current = child
            refused = store.enqueue_subtask(
                job.id,
                parent_task_id=current.id,
                role="explore",
                instruction="too deep",
                max_depth=3,
            )
            self.assertIsNone(refused)
            events = store.read_events(job.id)
            self.assertTrue(
                any(
                    e.get("event") == "task.enqueue_refused"
                    and (e.get("payload") or {}).get("reason") == "max_depth"
                    for e in events
                )
            )

    def test_enqueue_refuses_when_job_budget_is_exhausted(self) -> None:
        for store_type in (SwarmStore, SQLiteSwarmStore):
            with self.subTest(store=store_type.backend_name):
                with TemporaryDirectory() as tmp:
                    store = store_type(Path(tmp) / ".puppetmaster")
                    store.init()
                    job = store.create_job(
                        "budget-enqueue",
                        budget_policy=BudgetPolicy(max_usd=1),
                    )
                    parent = Task(
                        job_id=job.id,
                        role="explore",
                        instruction="root",
                        status=TaskStatus.COMPLETE,
                    )
                    store.save_task(parent)
                    first = store.enqueue_subtask(
                        job.id,
                        parent_task_id=parent.id,
                        role="audit",
                        instruction="first child",
                    )
                    self.assertIsNotNone(first)
                    attempt = ExecutionAttempt(
                        job.id,
                        parent.id,
                        "run-1",
                        "inv-1",
                        "2026-09-08T00:00:00+00:00",
                        "codex",
                    )
                    store.reserve_dispatch(
                        attempt,
                        BudgetLiability(
                            billing="api",
                            cost_state="known",
                            api_usd=1,
                        ),
                    )
                    store.adopt_dispatch(job.id, attempt.attempt_id, adoption_id="owner")
                    store.reconcile_reservation(
                        job.id,
                        attempt.attempt_id,
                        reconciliation_id="final",
                        liability=BudgetLiability(
                            billing="api",
                            cost_state="known",
                            api_usd=1.5,
                        ),
                        final=True,
                        evidence="authoritative cumulative SDK snapshot",
                    )
                    replay = store.enqueue_subtask(
                        job.id,
                        parent_task_id=parent.id,
                        role="audit",
                        instruction="first child",
                    )
                    self.assertIsNotNone(replay)
                    self.assertEqual(replay.id, first.id)
                    refused = store.enqueue_subtask(
                        job.id,
                        parent_task_id=parent.id,
                        role="audit",
                        instruction="second child after overspend",
                    )
                    self.assertIsNone(refused)
                    events = store.read_events(job.id)
                    self.assertTrue(
                        any(
                            e.get("event") == "task.enqueue_refused"
                            and (e.get("payload") or {}).get("reason")
                            == "budget_exhausted"
                            for e in events
                        )
                    )

    def test_follow_ups_from_artifact_payload(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("followups")
            parent = Task(
                job_id=job.id,
                role="explore",
                instruction="root",
                status=TaskStatus.COMPLETE,
            )
            store.save_task(parent)
            finding = Artifact(
                job_id=job.id,
                task_id=parent.id,
                type=ArtifactType.FINDING,
                created_by="worker",
                confidence=0.9,
                evidence=["x"],
                payload={
                    "claim": "needs more work",
                    "enqueue_subtasks": [
                        {"role": "review", "instruction": "review module"},
                        {"role": "audit", "instruction": "audit risks"},
                    ],
                },
            )
            store.save_artifact(finding)
            created = store.maybe_enqueue_follow_ups_from_artifact(
                finding, parent_task_id=parent.id
            )
            self.assertEqual(len(created), 2)
            roles = {task.role for task in created}
            self.assertEqual(roles, {"review", "audit"})


class FrontierObservabilityTests(unittest.TestCase):
    def test_status_snapshot_includes_frontier(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("frontier")
            parent = Task(
                job_id=job.id,
                role="explore",
                instruction="root",
                status=TaskStatus.COMPLETE,
            )
            store.save_task(parent)
            store.enqueue_subtask(
                job.id,
                parent_task_id=parent.id,
                role="audit",
                instruction="follow",
            )
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=parent.id,
                    type=ArtifactType.GIST,
                    created_by="worker",
                    confidence=0.9,
                    evidence=["e"],
                    payload={
                        "claim": "admitted",
                        "source_artifact_ids": ["a1"],
                        "admission": "admitted",
                    },
                )
            )
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=parent.id,
                    type=ArtifactType.GIST,
                    created_by="worker",
                    confidence=0.5,
                    evidence=["e"],
                    payload={
                        "claim": "pending",
                        "source_artifact_ids": ["a2"],
                        "admission": "pending",
                    },
                )
            )
            snap = store.status_snapshot(job.id)
            frontier = snap.get("frontier") or {}
            self.assertEqual(frontier.get("queued"), 1)
            self.assertEqual(frontier.get("enqueued_from_parent"), 1)
            gists = frontier.get("gists") or {}
            self.assertEqual(gists.get("total"), 2)
            self.assertEqual(gists.get("admitted"), 1)
            self.assertEqual(gists.get("pending"), 1)


if __name__ == "__main__":
    unittest.main()
