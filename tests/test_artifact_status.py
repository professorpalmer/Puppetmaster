"""Issue #88: status split, compat mapping, self-rating cannot admit."""
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

from puppetmaster.artifact_status import (
    CLAIM_SUPPORT_INDEPENDENT,
    CLAIM_SUPPORT_UNKNOWN,
    CLAIM_SUPPORT_WORKER_ASSERTED,
    CRITERION_MET,
    EXECUTION_COMPLETED,
    GROUNDING_CITED,
    durable_admission_allowed,
    format_status_label,
    status_fields,
    worker_self_rating_of,
)
from puppetmaster.gist_admission import maybe_admit_finding_as_gist
from puppetmaster.models import Artifact, ArtifactType, artifact_from_dict
from puppetmaster.store import SwarmStore
from puppetmaster.validation import compact_artifact_ref


def _finding(**kwargs) -> Artifact:
    payload = kwargs.pop("payload", {"claim": "self-rated claim"})
    return Artifact(
        job_id=kwargs.pop("job_id", "job-88"),
        task_id=kwargs.pop("task_id", "task-88"),
        type=ArtifactType.FINDING,
        created_by=kwargs.pop("created_by", "worker"),
        payload=payload,
        confidence=kwargs.pop("confidence", 0.95),
        evidence=kwargs.pop("evidence", ["file.py:1"]),
        **kwargs,
    )


class ArtifactStatusCompatTests(unittest.TestCase):
    def test_old_confidence_maps_to_worker_self_rating_not_independent_support(self) -> None:
        loaded = artifact_from_dict(
            {
                "id": "artifact_old",
                "job_id": "job-old",
                "task_id": "task-old",
                "type": "finding",
                "created_by": "worker",
                "payload": {"claim": "legacy record"},
                "confidence": 0.95,
                "evidence": ["legacy.py:3"],
                "created_at": "2026-08-01T00:00:00+00:00",
            }
        )
        self.assertEqual(loaded.confidence, 0.95)
        self.assertEqual(loaded.worker_self_rating, 0.95)
        self.assertEqual(loaded.claim_support_status, CLAIM_SUPPORT_UNKNOWN)
        self.assertEqual(loaded.grounding_status, GROUNDING_CITED)
        self.assertNotEqual(loaded.claim_support_status, CLAIM_SUPPORT_INDEPENDENT)
        self.assertEqual(worker_self_rating_of(loaded), 0.95)

    def test_worker_payload_cannot_self_certify_independent_support(self) -> None:
        artifact = _finding(
            payload={
                "claim": "I am independently supported",
                "claim_support_status": "independently_supported",
            }
        )
        self.assertEqual(artifact.claim_support_status, CLAIM_SUPPORT_WORKER_ASSERTED)
        self.assertFalse(durable_admission_allowed(artifact, peers=[]))

    def test_verification_result_is_criterion_status_not_parse_probability(self) -> None:
        artifact = Artifact(
            job_id="j",
            task_id="t",
            type=ArtifactType.VERIFICATION,
            created_by="pm",
            payload={"check": "parsed ok", "result": "passed"},
            confidence=0.9,
            evidence=["adapter:local"],
        )
        self.assertEqual(artifact.execution_status, EXECUTION_COMPLETED)
        self.assertEqual(artifact.criterion_status, CRITERION_MET)
        self.assertIn("criterion_status=met", format_status_label(artifact))
        self.assertNotIn("%", format_status_label(artifact))


class SelfRatingCannotAdmitTests(unittest.TestCase):
    def test_self_rating_cannot_admit_memory_or_gist(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("no theater")
            finding = _finding(
                job_id=job.id,
                task_id="task-hot",
                confidence=0.95,
                payload={"claim": "maximum self-rating"},
            )
            store.save_artifact(finding)
            self.assertFalse(durable_admission_allowed(finding, store=store))
            self.assertIsNone(maybe_admit_finding_as_gist(store, finding))

    def test_same_task_verification_admits_gist(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("verified")
            finding = _finding(
                job_id=job.id,
                task_id="task-v",
                confidence=0.11,
                payload={"claim": "checked claim"},
            )
            store.save_artifact(finding)
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id="task-v",
                    type=ArtifactType.VERIFICATION,
                    created_by="worker",
                    confidence=0.9,
                    evidence=[finding.id],
                    payload={"check": "checked claim", "result": "passed"},
                )
            )
            self.assertTrue(durable_admission_allowed(finding, store=store))
            gist = maybe_admit_finding_as_gist(store, finding)
            self.assertIsNotNone(gist)
            assert gist is not None
            self.assertEqual(gist.payload["admission"], "admitted")


class CompactRefDisplayTests(unittest.TestCase):
    def test_compact_ref_keeps_confidence_and_adds_statuses(self) -> None:
        artifact = _finding(confidence=0.88)
        ref = compact_artifact_ref(artifact)
        self.assertEqual(ref["confidence"], 0.88)
        self.assertEqual(ref["worker_self_rating"], 0.88)
        self.assertIn("execution_status", ref)
        self.assertIn("claim_support_status", ref)
        fields = status_fields(artifact)
        self.assertEqual(ref["claim_support_status"], fields["claim_support_status"])
        self.assertNotIn("%", format_status_label(artifact))


if __name__ == "__main__":
    unittest.main()
