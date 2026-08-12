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

from puppetmaster.gist_admission import (
    CONTEXT_LEVEL_GIST,
    CONTEXT_LEVEL_RAW,
    CONTEXT_LEVEL_SUMMARY,
    format_unfolded_for_injection,
    unfold_shared_context,
)
from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
from puppetmaster.store import SwarmStore


class SelectiveUnfoldTests(unittest.TestCase):
    def test_unfold_gist_summary_and_raw(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("unfold")
            source = Artifact(
                job_id=job.id,
                task_id="task-src",
                type=ArtifactType.FINDING,
                created_by="worker",
                confidence=0.9,
                evidence=["file.py:10"],
                payload={"claim": "full finding body with detail"},
            )
            store.save_artifact(source)
            gist = Artifact(
                job_id=job.id,
                task_id="task-src",
                type=ArtifactType.GIST,
                created_by="worker",
                confidence=0.9,
                evidence=[f"source:{source.id}"],
                payload={
                    "claim": "compact claim",
                    "source_artifact_ids": [source.id],
                    "admission": "admitted",
                    "level": "gist",
                    "summary_ref": source.id,
                },
            )
            store.save_artifact(gist)

            gist_view = unfold_shared_context(
                store, gist, level=CONTEXT_LEVEL_GIST
            )
            self.assertEqual(gist_view["level"], "gist")
            self.assertEqual(gist_view["body"], "compact claim")
            self.assertIn("Gist: compact claim", format_unfolded_for_injection(gist_view))

            summary_view = unfold_shared_context(
                store, gist, level=CONTEXT_LEVEL_SUMMARY
            )
            self.assertEqual(summary_view["level"], "summary")
            self.assertIn("full finding body", summary_view["body"])
            self.assertIn(
                "Summary:", format_unfolded_for_injection(summary_view)
            )

            raw_view = unfold_shared_context(store, gist, level=CONTEXT_LEVEL_RAW)
            self.assertEqual(raw_view["level"], "raw")
            self.assertEqual(len(raw_view["sources"]), 1)
            self.assertEqual(raw_view["sources"][0]["id"], source.id)
            self.assertIn("Source (", format_unfolded_for_injection(raw_view))


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
