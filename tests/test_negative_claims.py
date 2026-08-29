"""Wave 3: negative-claim fingerprints skip fresh GATE / gist / ci_failed repeats."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

from puppetmaster.gist_admission import build_pending_gist, reject_gist
from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
from puppetmaster.metr_seams import load_host_document
from puppetmaster.negative_claims import (
    REASON_NEGATIVE_CLAIM,
    ci_failed_negative_claim,
    gate_negative_claim,
    negative_claim_fingerprint,
    should_skip_negative,
    stamp_negative_claim,
)
from puppetmaster.scm_observe import facts_from_snapshot, observe_scm, snapshot_from_gh_payload
from puppetmaster.store import SwarmStore


def _git_init_with_file(root: Path, rel: str, content: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", rel], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _events(store: SwarmStore, job_id: str, name: str) -> list:
    return [event for event in store.read_events(job_id) if event.get("event") == name]


def _refused_reasons(store: SwarmStore, job_id: str) -> list:
    return [
        str((event.get("payload") or {}).get("reason") or "")
        for event in _events(store, job_id, "task.enqueue_refused")
    ]


def _parent(store: SwarmStore, job_id: str, **payload) -> Task:
    parent = Task(
        job_id=job_id,
        role="explore",
        instruction="root",
        status=TaskStatus.COMPLETE,
        payload=dict(payload),
    )
    store.save_task(parent)
    return parent


def _finding(job_id: str, task_id: str, extra=None) -> Artifact:
    payload = {"claim": "follow up work"}
    if extra:
        payload.update(extra)
    return Artifact(
        job_id=job_id,
        task_id=task_id,
        type=ArtifactType.FINDING,
        created_by="worker",
        confidence=0.9,
        evidence=["src/a.py"],
        payload=payload,
    )


def _gh_payload(**overrides):
    body = {
        "url": "https://github.com/example/repo/pull/7",
        "number": 7,
        "title": "fix ci",
        "state": "OPEN",
        "mergeable": "MERGEABLE",
        "reviewDecision": "",
        "statusCheckRollup": [{"name": "tests", "conclusion": "FAILURE"}],
    }
    body.update(overrides)
    return body


class FingerprintTests(unittest.TestCase):
    def test_fingerprint_stable_under_whitespace_and_case(self) -> None:
        left = negative_claim_fingerprint("  Tokens   LEAK ", "  Src/A.py ")
        right = negative_claim_fingerprint("tokens leak", "src/a.py")
        self.assertEqual(left, right)
        self.assertEqual(len(left), 64)

    def test_fingerprint_changes_when_claim_or_scope_changes(self) -> None:
        base = negative_claim_fingerprint("tokens leak", "src/a.py")
        self.assertNotEqual(base, negative_claim_fingerprint("other claim", "src/a.py"))
        self.assertNotEqual(base, negative_claim_fingerprint("tokens leak", "src/b.py"))


class FailedGateEnqueueTests(unittest.TestCase):
    def test_failed_gate_refuses_identical_enqueue_and_allows_after_bytes_change(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init_with_file(root, "src/a.py", "alpha\n")
            store = SwarmStore(root / ".puppetmaster")
            store.init()
            job = store.create_job("negative gate")
            parent = _parent(
                store,
                job.id,
                cwd=str(root),
                source_scope=["src/a.py"],
            )
            gate = Artifact(
                job_id=job.id,
                task_id=parent.id,
                type=ArtifactType.GATE,
                created_by="worker",
                confidence=0.9,
                evidence=["gate:require_diff", "failed"],
                payload={
                    "gate": "require_diff",
                    "kind": "require_diff",
                    "passed": False,
                    "reason": "no_diff",
                },
            )
            store.save_artifact(gate)
            saved = [item for item in store.list_artifacts(job.id) if item.id == gate.id][0]
            self.assertEqual(saved.payload.get("passed"), False)
            negative = saved.payload.get("negative_claim") or {}
            self.assertEqual(negative.get("kind"), "gate")
            claim = gate_negative_claim("require_diff", "no_diff")
            self.assertEqual(negative.get("claim"), claim)
            self.assertEqual(negative.get("scope"), "src/a.py")
            self.assertTrue(negative.get("source_digests"))

            finding = _finding(
                job.id,
                parent.id,
                extra={
                    "enqueue_subtasks": [
                        {
                            "role": "implement",
                            "instruction": claim,
                            "scope": "src/a.py",
                        }
                    ]
                },
            )
            created = store.maybe_enqueue_follow_ups_from_artifact(
                finding, parent_task_id=parent.id, created_by="worker-1", cwd=root
            )
            self.assertEqual(created, [])
            self.assertIn(REASON_NEGATIVE_CLAIM, _refused_reasons(store, job.id))
            self.assertEqual(len(store.list_tasks(job.id)), 1)

            (root / "src/a.py").write_text("alpha-dirty\n", encoding="utf-8")
            retry = store.maybe_enqueue_follow_ups_from_artifact(
                finding, parent_task_id=parent.id, created_by="worker-1", cwd=root
            )
            self.assertEqual(len(retry), 1)
            self.assertEqual(retry[0].instruction, claim)
            self.assertEqual(len(store.list_tasks(job.id)), 2)


class RejectedGistEnqueueTests(unittest.TestCase):
    def test_reject_gist_stamps_fingerprint_and_second_enqueue_skips(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("negative gist")
            parent = _parent(store, job.id)
            pending = build_pending_gist(
                job_id=job.id,
                task_id=parent.id,
                created_by="worker-gist",
                claim="compact discovery must not retry",
                source_artifact_ids=["finding-src", "verify-src"],
            )
            store.save_artifact(pending)
            rejected = reject_gist(store, pending, verifier_result={"result": "rejected"})
            self.assertEqual(rejected.payload["admission"], "rejected")
            negative = rejected.payload.get("negative_claim") or {}
            self.assertEqual(negative.get("kind"), "gist")
            self.assertEqual(negative.get("claim"), "compact discovery must not retry")
            self.assertEqual(negative.get("scope"), "finding-src,verify-src")
            self.assertEqual(
                negative.get("fingerprint"),
                negative_claim_fingerprint(
                    "compact discovery must not retry", "finding-src,verify-src"
                ),
            )

            finding = _finding(
                job.id,
                parent.id,
                extra={
                    "source_artifact_ids": ["finding-src", "verify-src"],
                    "enqueue_subtasks": [
                        {
                            "role": "explore",
                            "instruction": "compact discovery must not retry",
                            "scope": "finding-src,verify-src",
                        }
                    ],
                },
            )
            created = store.maybe_enqueue_follow_ups_from_artifact(
                finding, parent_task_id=parent.id, created_by="worker-1"
            )
            self.assertEqual(created, [])
            self.assertIn(REASON_NEGATIVE_CLAIM, _refused_reasons(store, job.id))
            self.assertEqual(len(store.list_tasks(job.id)), 1)


class CiFailedSkipTests(unittest.TestCase):
    def test_should_skip_negative_true_for_same_ci_failed_false_when_scope_changes(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("negative ci")
            parent = _parent(store, job.id)
            snap = snapshot_from_gh_payload(_gh_payload())
            result = observe_scm(store, job.id, snap)
            self.assertEqual(result["reactions"][0]["reason"], "enqueued")
            self.assertEqual(len(store.list_tasks(job.id)), 2)
            fact = [item for item in facts_from_snapshot(snap) if item.kind == "ci_failed"][0]
            claim = ci_failed_negative_claim(fact.kind, fact.instruction)
            document = load_host_document(store, job.id)
            stamped = None
            for row in document.get("observations") or []:
                negative = row.get("negative_claim")
                if isinstance(negative, dict) and negative.get("kind") == "ci_failed":
                    stamped = negative
                    break
            self.assertIsNotNone(stamped)
            assert stamped is not None
            self.assertEqual(stamped.get("claim"), claim)
            self.assertEqual(stamped.get("scope"), fact.key)
            self.assertTrue(should_skip_negative(store, job.id, claim, fact.key))
            self.assertFalse(
                should_skip_negative(
                    store, job.id, claim, "ci:https://example.invalid/pull/9"
                )
            )
            artifact = stamp_negative_claim(
                Artifact(
                    job_id=job.id,
                    task_id=parent.id,
                    type=ArtifactType.FINDING,
                    created_by="host",
                    confidence=1.0,
                    evidence=["ci"],
                    payload={"claim": "ci failed"},
                ),
                kind="ci_failed",
                claim=ci_failed_negative_claim("ci_failed", "fix checks"),
                scope="ci:other-key",
            )
            store.save_artifact(artifact)
            self.assertTrue(
                should_skip_negative(
                    store,
                    job.id,
                    ci_failed_negative_claim("ci_failed", "fix checks"),
                    "ci:other-key",
                )
            )
            self.assertFalse(
                should_skip_negative(
                    store,
                    job.id,
                    ci_failed_negative_claim("ci_failed", "fix checks"),
                    "ci:changed-scope",
                )
            )


if __name__ == "__main__":
    unittest.main()
