"""Dashboard frontier sidecar + gist activity timeline."""
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

from puppetmaster.dashboard import (
    _PAGE_APP_JS,
    _build_task_activity,
    build_job_snapshot,
)
from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
from puppetmaster.store import SwarmStore


class DashboardFrontierTests(unittest.TestCase):
    def test_build_job_snapshot_includes_frontier(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("frontier dashboard")
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
                instruction="follow up",
            )
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=parent.id,
                    type=ArtifactType.GIST,
                    created_by="worker",
                    confidence=0.9,
                    evidence=["e1"],
                    payload={
                        "claim": "admitted discovery",
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
                    evidence=["e2"],
                    payload={
                        "claim": "pending discovery",
                        "source_artifact_ids": ["a2"],
                        "admission": "pending",
                    },
                )
            )

            snap = build_job_snapshot(store, job.id)
            frontier = snap.get("frontier") or {}
            self.assertEqual(frontier.get("queued"), 1)
            self.assertEqual(frontier.get("enqueued_from_parent"), 1)
            gists = frontier.get("gists") or {}
            self.assertEqual(gists.get("total"), 2)
            self.assertEqual(gists.get("admitted"), 1)
            self.assertEqual(gists.get("pending"), 1)

    def test_gist_activity_includes_claim_and_admission(self) -> None:
        gist = Artifact(
            job_id="job-1",
            task_id="task-1",
            type=ArtifactType.GIST,
            created_by="worker",
            confidence=0.9,
            evidence=["file.py:10"],
            payload={
                "claim": "shared compact fact",
                "source_artifact_ids": ["finding-1"],
                "admission": "admitted",
            },
        )
        activity = _build_task_activity("task-1", [gist])
        self.assertEqual(len(activity), 1)
        self.assertEqual(activity[0]["type"], "gist")
        self.assertEqual(activity[0]["text"], "Gist: shared compact fact (admitted)")
        self.assertEqual(activity[0]["message"], "Gist: shared compact fact (admitted)")

    def test_gist_activity_omits_empty_admission(self) -> None:
        gist = Artifact(
            job_id="job-1",
            task_id="task-1",
            type=ArtifactType.GIST,
            created_by="worker",
            confidence=0.9,
            evidence=["file.py:10"],
            payload={
                "claim": "no admission field",
                "source_artifact_ids": ["finding-1"],
            },
        )
        activity = _build_task_activity("task-1", [gist])
        self.assertEqual(activity[0]["text"], "Gist: no admission field")

    def test_renderer_includes_frontier_sidecar(self) -> None:
        self.assertIn("renderFrontierSidecar", _PAGE_APP_JS)
        self.assertIn("frontier-sidecar", _PAGE_APP_JS)
        self.assertIn("gists admitted", _PAGE_APP_JS)
        self.assertIn("renderGistSection", _PAGE_APP_JS)
        self.assertIn("frontier-gist-list", _PAGE_APP_JS)

    def test_snapshot_exposes_gist_claims(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("gist panel")
            task = Task(
                job_id=job.id,
                role="explore",
                instruction="seed",
                status=TaskStatus.COMPLETE,
            )
            store.save_task(task)
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.GIST,
                    created_by="worker",
                    confidence=0.91,
                    evidence=["e"],
                    payload={
                        "claim": "visible shared claim",
                        "source_artifact_ids": ["src-1"],
                        "admission": "admitted",
                    },
                )
            )
            snap = build_job_snapshot(store, job.id)
            gists = (snap.get("artifacts") or {}).get("gist") or []
            self.assertEqual(len(gists), 1)
            self.assertEqual(gists[0]["statement"], "visible shared claim")
            self.assertEqual(gists[0]["admission"], "admitted")
            self.assertEqual(gists[0]["source_artifact_ids"], ["src-1"])


if __name__ == "__main__":
    unittest.main()
