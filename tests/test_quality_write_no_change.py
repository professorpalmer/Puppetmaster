"""A write-capable run that changed nothing is not a delivered job."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hermetic_env  # noqa: F401,E402

from puppetmaster.models import Artifact, ArtifactType  # noqa: E402
from puppetmaster.quality import assess_run_quality  # noqa: E402


def art(type_, **payload):
    return Artifact(job_id="j", task_id="t", type=type_, created_by="w", payload=payload,
                    confidence=1.0, evidence=[])


class WriteRunNoChangeTests(unittest.TestCase):
    def test_refusal_from_an_edit_run_is_degraded(self):
        # A Claude Code implement worker could not read its input and said so.
        arts = [art(ArtifactType.VERIFICATION, result="passed", permission_mode="acceptEdits",
                    worker_diff_present=False, changed_files=[]),
                art(ArtifactType.FINDING, claim="I can't proceed: the audit list is unreachable")]
        verdict = assess_run_quality(arts)
        self.assertEqual(verdict["quality"], "degraded")
        self.assertFalse(verdict["trustworthy"])

    def test_edit_run_with_a_change_or_read_only_run_stays_ok(self):
        changed = [art(ArtifactType.VERIFICATION, sandbox="workspace-write", worker_diff_present=True),
                   art(ArtifactType.VERIFICATION, kind="worker_verdict", verdict="PASS", reason="built it"),
                   art(ArtifactType.FINDING, claim="built it")]
        self.assertEqual(assess_run_quality(changed)["quality"], "ok")
        read_only = [art(ArtifactType.VERIFICATION, permission_mode="plan", worker_diff_present=False),
                     art(ArtifactType.FINDING, claim="audit finding")]
        self.assertEqual(assess_run_quality(read_only)["quality"], "ok")


class UnfinishedEditRunTests(unittest.TestCase):
    def _run(self, *extra):
        return [art(ArtifactType.VERIFICATION, permission_mode="acceptEdits", worker_diff_present=True),
                art(ArtifactType.FINDING, claim="Committed. Not pushed."), *extra]

    def test_edit_run_without_a_verdict_is_unverified(self):
        # A cleanup worker did 2 of 26 items, said "Committed." and the job read as delivered.
        verdict = assess_run_quality(self._run())
        self.assertEqual(verdict["quality"], "degraded")
        self.assertIn("did not report whether it finished", verdict["reasons"][0])

    def test_partial_or_failed_verdict_is_degraded(self):
        for kind in ("PARTIAL", "FAIL"):
            run = self._run(art(ArtifactType.VERIFICATION, kind="worker_verdict", verdict=kind,
                                reason="skipped 24 of 26 items"))
            verdict = assess_run_quality(run)
            self.assertEqual(verdict["quality"], "degraded")
            self.assertIn(f"worker reported {kind}: skipped 24 of 26 items", verdict["reasons"][0])


if __name__ == "__main__":
    unittest.main()
