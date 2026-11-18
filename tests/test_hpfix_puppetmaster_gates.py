"""Regression tests for write_scope attribution path matching."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from puppetmaster.gates import _gate_write_scope
from puppetmaster.models import Artifact, ArtifactType
from puppetmaster.worker_attribution import (
    WorkerReferences,
    path_is_referenced,
    path_was_dirty_before,
)


def _patch(files, referenced, baseline):
    return Artifact(
        job_id="j", task_id="t", type=ArtifactType.PATCH, created_by="w",
        confidence=1.0, evidence=[],
        payload={
            "worker_delta_attributed": True,
            "files": files,
            "worker_referenced_paths": referenced,
            "baseline_dirty_paths": baseline,
        },
    )


def _gate(files, referenced, baseline):
    with TemporaryDirectory() as tmp:
        result = _gate_write_scope(
            "ws", {"scope": ["src/*"]}, [_patch(files, referenced, baseline)], Path(tmp)
        )
    detail = result.detail or {}
    return result.passed, detail.get("concurrent_changes"), detail.get("out_of_scope")


class ReferencedPathMatchTests(unittest.TestCase):
    """Bug 56e0b53a-1: a root path does not name a file in a subdirectory."""

    def test_root_reference_does_not_claim_pre_dirty_subdir_file(self) -> None:
        passed, concurrent, out = _gate(["docs/README.md"], ["README.md"], ["docs/README.md"])
        self.assertTrue(passed)
        self.assertEqual(concurrent, ["docs/README.md"])
        self.assertIsNone(out)

    def test_exact_reference_is_still_charged(self) -> None:
        passed, concurrent, out = _gate(
            ["docs/README.md"], ["docs/README.md"], ["docs/README.md"]
        )
        self.assertFalse(passed)
        self.assertEqual(out, ["docs/README.md"])

    def test_absolute_reference_outside_cwd_still_matches(self) -> None:
        refs = WorkerReferences(["/repo/docs/README.md"])
        self.assertTrue(path_is_referenced("docs/README.md", refs))
        self.assertFalse(path_is_referenced("README.md", WorkerReferences(["docs/README.md"])))


class BaselineDirtyMatchTests(unittest.TestCase):
    """Bug 56e0b53a-4: a dirty root file does not excuse a subdirectory file."""

    def test_dirty_root_file_does_not_excuse_new_subdir_file(self) -> None:
        passed, concurrent, out = _gate(["docs/README.md"], [], ["README.md"])
        self.assertFalse(passed)
        self.assertEqual(out, ["docs/README.md"])
        self.assertIsNone(concurrent)
        self.assertFalse(path_was_dirty_before("pkg/x.py", ["x.py"]))

    def test_exact_and_directory_baseline_still_excuse(self) -> None:
        self.assertTrue(path_was_dirty_before("docs/README.md", ["docs/README.md"]))
        self.assertTrue(path_was_dirty_before("docs/README.md", ["docs/"]))
        passed, concurrent, _ = _gate(["docs/README.md"], [], ["docs/README.md"])
        self.assertTrue(passed)
        self.assertEqual(concurrent, ["docs/README.md"])


if __name__ == "__main__":
    unittest.main()
