"""Focused tests for durable verified-gist admission (Wave 1)."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

from puppetmaster.gist_admission import (
    admit_gist,
    build_pending_gist,
    filter_shared_context_artifacts,
    is_admitted_for_shared_context,
    maybe_admit_finding_as_gist,
    reject_gist,
)
from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
from puppetmaster.prewalk import format_upstream_artifacts_for_injection
from puppetmaster.store import SwarmStore


def _finding(
    *,
    job_id: str = "job-gist",
    task_id: str = "task-gist",
    claim: str = "Durable discoveries need admission",
    confidence: float = 0.9,
) -> Artifact:
    return Artifact(
        job_id=job_id,
        task_id=task_id,
        type=ArtifactType.FINDING,
        created_by="worker-gist",
        confidence=confidence,
        evidence=["puppetmaster/gist_admission.py"],
        payload={"claim": claim},
    )


def _gist(
    *,
    admission: str,
    claim: str = "compact discovery",
    job_id: str = "job-gist",
    task_id: str = "task-gist",
    source_ids: Optional[List[str]] = None,
) -> Artifact:
    return Artifact(
        job_id=job_id,
        task_id=task_id,
        type=ArtifactType.GIST,
        created_by="worker-gist",
        confidence=0.9,
        evidence=["source:finding-1"],
        payload={
            "claim": claim,
            "source_artifact_ids": source_ids or ["finding-1"],
            "admission": admission,
            "level": "gist",
        },
    )


class GistAdmissionFilterTests(unittest.TestCase):
    def test_pending_gist_filtered_from_shared_context(self) -> None:
        pending = _gist(admission="pending")
        self.assertFalse(is_admitted_for_shared_context(pending))
        self.assertEqual(filter_shared_context_artifacts([pending]), [])

    def test_admitted_gist_included(self) -> None:
        admitted = _gist(admission="admitted", claim="peers may see this")
        self.assertTrue(is_admitted_for_shared_context(admitted))
        filtered = filter_shared_context_artifacts([admitted])
        self.assertEqual(filtered, [admitted])
        text = format_upstream_artifacts_for_injection([admitted])
        self.assertIn("Gist: peers may see this", text)

    def test_rejected_gist_filtered(self) -> None:
        rejected = _gist(admission="rejected")
        self.assertFalse(is_admitted_for_shared_context(rejected))
        self.assertEqual(filter_shared_context_artifacts([rejected]), [])
        text = format_upstream_artifacts_for_injection([rejected])
        self.assertEqual(text, "")

    def test_legacy_finding_still_injects(self) -> None:
        finding = _finding(claim="legacy finding claim")
        self.assertTrue(is_admitted_for_shared_context(finding))
        filtered = filter_shared_context_artifacts([finding, _gist(admission="pending")])
        self.assertEqual([item.id for item in filtered], [finding.id])
        text = format_upstream_artifacts_for_injection(
            [finding, _gist(admission="rejected")]
        )
        self.assertIn("Finding: legacy finding claim", text)
        self.assertNotIn("Gist:", text)

    def test_stale_validation_status_refused_without_cwd(self) -> None:
        from dataclasses import replace

        finding = _finding(claim="cited bytes have moved")
        payload = dict(finding.payload)
        payload["validation"] = {
            "status": "stale",
            "source_digests": {"src/a.py": "abc123"},
        }
        stale = replace(finding, payload=payload, sha256=None)
        self.assertFalse(is_admitted_for_shared_context(stale))
        self.assertEqual(filter_shared_context_artifacts([stale]), [])


class GistAdmissionEventTests(unittest.TestCase):
    def test_event_emission_on_admit_and_reject(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("gist events")
            pending = build_pending_gist(
                job_id=job.id,
                task_id="task-1",
                created_by="worker-gist",
                claim="needs review",
                source_artifact_ids=["art-src"],
            )
            store.save_artifact(pending)

            admitted = admit_gist(store, pending, verifier_result={"result": "accepted"})
            self.assertEqual(admitted.payload["admission"], "admitted")
            events = store.read_events(job.id)
            admitted_events = [e for e in events if e.get("event") == "gist.admitted"]
            self.assertEqual(len(admitted_events), 1)
            self.assertEqual(
                admitted_events[0]["payload"]["artifact_id"], admitted.id
            )

            rejected = reject_gist(
                store, admitted, verifier_result={"result": "rejected"}
            )
            self.assertEqual(rejected.payload["admission"], "rejected")
            events = store.read_events(job.id)
            rejected_events = [e for e in events if e.get("event") == "gist.rejected"]
            self.assertEqual(len(rejected_events), 1)
            self.assertEqual(
                rejected_events[0]["payload"]["artifact_id"], rejected.id
            )


class MaybeAdmitFindingTests(unittest.TestCase):
    def test_maybe_admit_finding_as_gist_creates_admitted_gist(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("auto gist")
            finding = _finding(
                job_id=job.id,
                task_id="task-auto",
                claim="High confidence discovery",
                confidence=0.85,
            )
            store.save_artifact(finding)
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id="task-auto",
                    type=ArtifactType.VERIFICATION,
                    created_by="worker-gist",
                    confidence=0.4,
                    evidence=[finding.id],
                    payload={
                        "check": finding.payload["claim"],
                        "result": "passed",
                    },
                )
            )

            gist = maybe_admit_finding_as_gist(store, finding)
            self.assertIsNotNone(gist)
            assert gist is not None
            self.assertEqual(gist.type, ArtifactType.GIST)
            self.assertEqual(gist.payload["admission"], "admitted")
            self.assertEqual(gist.payload["claim"], "High confidence discovery")
            self.assertEqual(gist.payload["source_artifact_ids"], [finding.id])
            self.assertEqual(gist.payload.get("level"), "gist")

            saved = store.list_artifacts(job.id)
            gist_ids = [a.id for a in saved if a.type == ArtifactType.GIST]
            self.assertIn(gist.id, gist_ids)

            events = store.read_events(job.id)
            self.assertTrue(
                any(e.get("event") == "gist.admitted" for e in events)
            )

    def test_self_rating_alone_does_not_admit_gist(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("skip gist")
            finding = _finding(
                job_id=job.id, claim="self-rated theater", confidence=0.95
            )
            store.save_artifact(finding)
            self.assertIsNone(maybe_admit_finding_as_gist(store, finding))

    def test_independent_verification_admits_even_low_self_rating(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("verify gist")
            finding = _finding(
                job_id=job.id,
                task_id="task-v",
                claim="independently checked",
                confidence=0.2,
            )
            store.save_artifact(finding)
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id="task-v",
                    type=ArtifactType.VERIFICATION,
                    created_by="worker-gist",
                    confidence=0.99,
                    evidence=[finding.id],
                    payload={"check": "independently checked", "result": "passed"},
                )
            )
            gist = maybe_admit_finding_as_gist(store, finding)
            self.assertIsNotNone(gist)
            assert gist is not None
            self.assertEqual(gist.payload["admission"], "admitted")


class ResolveEdgesAdmissionTests(unittest.TestCase):
    def test_resolve_artifacts_via_edges_filters_pending_gists(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("edge filter")
            upstream = Task(
                job_id=job.id,
                role="explore",
                instruction="explore",
                status=TaskStatus.COMPLETE,
            )
            downstream = Task(
                job_id=job.id,
                role="implement",
                instruction="implement",
                status=TaskStatus.QUEUED,
                depends_on=[upstream.id],
            )
            store.save_task(upstream)
            store.save_task(downstream)

            finding = _finding(job_id=job.id, task_id=upstream.id)
            admitted = _gist(
                admission="admitted",
                job_id=job.id,
                task_id=upstream.id,
                claim="admitted edge gist",
                source_ids=[finding.id],
            )
            pending = _gist(
                admission="pending",
                job_id=job.id,
                task_id=upstream.id,
                claim="pending edge gist",
                source_ids=[finding.id],
            )
            store.save_artifact(finding)
            store.save_artifact(admitted)
            store.save_artifact(pending)

            resolved = store.resolve_artifacts_via_edges(downstream)
            resolved_ids = {item.id for item in resolved}
            self.assertIn(finding.id, resolved_ids)
            self.assertIn(admitted.id, resolved_ids)
            self.assertNotIn(pending.id, resolved_ids)

            # Raw store listing remains unfiltered for tooling/MCP.
            listed_ids = {item.id for item in store.list_artifacts(job.id)}
            self.assertIn(pending.id, listed_ids)


if __name__ == "__main__":
    unittest.main()
