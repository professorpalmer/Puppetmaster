"""A write-capable run that changed nothing is not a delivered job."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hermetic_env  # noqa: F401,E402

from puppetmaster.models import Artifact, ArtifactType  # noqa: E402
from puppetmaster.quality import assess_run_quality  # noqa: E402


def art(kind, **payload):
    return Artifact(job_id="j", task_id="t", type=kind, created_by="w", payload=payload,
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
                   art(ArtifactType.FINDING, claim="built it")]
        self.assertEqual(assess_run_quality(changed)["quality"], "ok")
        read_only = [art(ArtifactType.VERIFICATION, permission_mode="plan", worker_diff_present=False),
                     art(ArtifactType.FINDING, claim="audit finding")]
        self.assertEqual(assess_run_quality(read_only)["quality"], "ok")


if __name__ == "__main__":
    unittest.main()
