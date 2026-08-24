"""End-to-end validation for DeLM shared-context waves + dashboard API."""
from __future__ import annotations

import json
import os
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

import unittest

from puppetmaster.dashboard import build_job_snapshot, make_handler
from puppetmaster.gist_admission import (
    CONTEXT_LEVEL_GIST,
    CONTEXT_LEVEL_RAW,
    CONTEXT_LEVEL_SUMMARY,
    format_unfolded_for_injection,
    maybe_admit_finding_as_gist,
    unfold_shared_context,
)
from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
from puppetmaster.prewalk import format_upstream_artifacts_for_injection
from puppetmaster.sqlite_store import SQLiteSwarmStore


class DelmSharedContextE2ETests(unittest.TestCase):
    def test_admission_unfold_enqueue_dashboard_http(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("DeLM E2E validation", label="delm-e2e")
            explore = Task(
                job_id=job.id,
                role="explore",
                instruction="find shared-context issues",
                status=TaskStatus.COMPLETE,
                adapter="local",
            )
            implement = Task(
                job_id=job.id,
                role="implement",
                instruction="apply admitted discoveries",
                status=TaskStatus.QUEUED,
                depends_on=[explore.id],
                adapter="local",
            )
            store.save_task(explore)
            store.save_task(implement)

            finding = Artifact(
                job_id=job.id,
                task_id=explore.id,
                type=ArtifactType.FINDING,
                created_by="worker-explore",
                confidence=0.92,
                evidence=["puppetmaster/gist_admission.py:120"],
                payload={
                    "claim": "Peers must only see admitted gists",
                    "enqueue_subtasks": [
                        {
                            "role": "review",
                            "instruction": "review admission filter edges",
                        }
                    ],
                },
            )
            store.save_artifact(finding)
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=explore.id,
                    type=ArtifactType.VERIFICATION,
                    created_by="worker-explore",
                    confidence=0.4,
                    evidence=[finding.id],
                    payload={
                        "check": "Peers must only see admitted gists",
                        "result": "passed",
                    },
                )
            )

            gist = maybe_admit_finding_as_gist(store, finding)
            self.assertIsNotNone(gist)
            assert gist is not None
            self.assertEqual(gist.payload.get("admission"), "admitted")

            pending = Artifact(
                job_id=job.id,
                task_id=explore.id,
                type=ArtifactType.GIST,
                created_by="worker-explore",
                confidence=0.4,
                evidence=["manual"],
                payload={
                    "claim": "unverified rumor",
                    "source_artifact_ids": [finding.id],
                    "admission": "pending",
                    "level": "gist",
                },
            )
            store.save_artifact(pending)

            resolved_ids = {
                artifact.id for artifact in store.resolve_artifacts_via_edges(implement)
            }
            self.assertIn(finding.id, resolved_ids)
            self.assertIn(gist.id, resolved_ids)
            self.assertNotIn(pending.id, resolved_ids)

            injected = format_upstream_artifacts_for_injection(
                [finding, gist, pending]
            )
            self.assertIn("Finding: Peers must only see admitted gists", injected)
            self.assertIn("Gist: Peers must only see admitted gists", injected)
            self.assertNotIn("unverified rumor", injected)

            gist_view = unfold_shared_context(
                store, gist, level=CONTEXT_LEVEL_GIST
            )
            summary_view = unfold_shared_context(
                store, gist, level=CONTEXT_LEVEL_SUMMARY
            )
            raw_view = unfold_shared_context(store, gist, level=CONTEXT_LEVEL_RAW)
            self.assertEqual(
                gist_view.get("body"), "Peers must only see admitted gists"
            )
            self.assertTrue(summary_view.get("body"))
            self.assertTrue(
                any(
                    source.get("id") == finding.id
                    for source in raw_view.get("sources") or []
                )
            )
            self.assertIn("Gist:", format_unfolded_for_injection(gist_view))

            created = store.maybe_enqueue_follow_ups_from_artifact(
                finding, parent_task_id=explore.id, created_by="worker-explore"
            )
            self.assertEqual(len(created), 1)
            self.assertEqual(created[0].role, "review")
            deduped = store.maybe_enqueue_follow_ups_from_artifact(
                finding, parent_task_id=explore.id, created_by="worker-explore"
            )
            self.assertEqual(deduped[0].id, created[0].id)

            snap = build_job_snapshot(store, job.id)
            frontier = snap.get("frontier") or {}
            self.assertGreaterEqual(frontier.get("queued", 0), 1)
            self.assertGreaterEqual(frontier.get("enqueued_from_parent", 0), 1)
            gists = frontier.get("gists") or {}
            self.assertGreaterEqual(gists.get("admitted", 0), 1)
            self.assertGreaterEqual(gists.get("pending", 0), 1)

            handler = make_handler(lambda: store)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/job?id={job.id}", timeout=5
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
                self.assertIsInstance(body.get("frontier"), dict)
                self.assertGreaterEqual(
                    (body.get("frontier") or {})
                    .get("gists", {})
                    .get("admitted", 0),
                    1,
                )
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/", timeout=5
                ) as response:
                    html = response.read().decode("utf-8")
                self.assertIn("renderFrontierSidecar", html)
                self.assertIn("frontier-sidecar", html)
            finally:
                server.shutdown()
                server.server_close()

            event_names = {event.get("event") for event in store.read_events(job.id)}
            self.assertIn("gist.admitted", event_names)
            self.assertIn("task.enqueued", event_names)


if __name__ == "__main__":
    unittest.main()
