"""A later evaluation of the same gate on the same task supersedes an earlier one."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hermetic_env  # noqa: F401,E402

from puppetmaster.models import Artifact, ArtifactType  # noqa: E402
from puppetmaster.quality import assess_run_quality  # noqa: E402


def _gate(task_id: str, passed: bool, at: str, gate: str = "review") -> Artifact:
    return Artifact(
        job_id="j", task_id=task_id, type=ArtifactType.GATE, created_by="worker",
        confidence=0.9, evidence=["test:gate"],
        payload={"gate": gate, "kind": gate, "passed": passed, "reason": "x",
                 "reviewed_artifact_fingerprint": "sha256:abc"},
        created_at=at,
    )


def _work(task_id: str) -> Artifact:
    return Artifact(
        job_id="j", task_id=task_id, type=ArtifactType.PATCH, created_by="worker",
        confidence=0.9, evidence=["test:patch"],
        payload={"change": "edit", "files": ["a.py"], "diff": "+x"},
    )


class SupersededGateTests(unittest.TestCase):
    def test_repaired_review_is_not_blocked_by_the_first_rejection(self) -> None:
        quality = assess_run_quality([
            _work("t1"),
            _gate("t1", False, "2026-10-04T03:49:33+00:00"),
            _gate("t1", True, "2026-10-04T03:50:35+00:00"),
        ])
        self.assertNotEqual(quality["quality"], "blocked", quality["reasons"])
        self.assertEqual(quality["semantic_quality"], "passed")

    def test_a_later_failure_still_blocks(self) -> None:
        quality = assess_run_quality([
            _work("t1"),
            _gate("t1", True, "2026-10-04T03:49:33+00:00"),
            _gate("t1", False, "2026-10-04T03:50:35+00:00"),
        ])
        self.assertEqual(quality["quality"], "blocked")

    def test_same_second_failure_wins_and_other_tasks_and_gates_stay_independent(self) -> None:
        tie = assess_run_quality([
            _work("t1"),
            _gate("t1", True, "2026-10-04T03:50:35+00:00"),
            _gate("t1", False, "2026-10-04T03:50:35+00:00"),
        ])
        self.assertEqual(tie["quality"], "blocked")
        other = assess_run_quality([
            _work("t1"),
            _gate("t1", True, "2026-10-04T03:50:35+00:00"),
            _gate("t2", False, "2026-10-04T03:49:00+00:00"),
        ])
        self.assertEqual(other["quality"], "blocked")
        gates = assess_run_quality([
            _work("t1"),
            _gate("t1", False, "2026-10-04T03:49:00+00:00", gate="require_diff"),
            _gate("t1", True, "2026-10-04T03:50:35+00:00", gate="review"),
        ])
        self.assertEqual(gates["quality"], "blocked")


if __name__ == "__main__":
    unittest.main()
