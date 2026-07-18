"""Focused tests for Wave 2 bounded artifact persistence."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional
from unittest.mock import patch

from puppetmaster.artifact_bounds import (
    BOUNDS_KEY,
    DEFAULT_MAX_ARTIFACT_BYTES,
    artifact_bounds_summary,
    bounds_enabled,
    prepare_artifact_for_persist,
    serialized_artifact_bytes,
)
from puppetmaster.models import Artifact, ArtifactType
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.store import SwarmStore

def _finding(*, claim: str, evidence: Optional[List[str]] = None) -> Artifact:
    return Artifact(
        job_id="job-bounds",
        task_id="task-bounds",
        type=ArtifactType.FINDING,
        created_by="worker-bounds",
        confidence=0.9,
        evidence=evidence or ["puppetmaster/artifact_bounds.py"],
        payload={"claim": claim},
    )

def _patch(*, diff: str, files: Optional[List[str]] = None) -> Artifact:
    return Artifact(
        job_id="job-bounds",
        task_id="task-bounds",
        type=ArtifactType.PATCH,
        created_by="worker-bounds",
        confidence=0.8,
        evidence=["adapter:test", "base:abc"],
        payload={
            "change": "oversized patch",
            "files": files if files is not None else ["a.py"],
            "unified_diff": diff,
        },
    )

class ArtifactBoundsUnitTests(unittest.TestCase):
    def test_small_artifact_passthrough(self) -> None:
        artifact = _finding(claim="tiny claim")
        with TemporaryDirectory() as tmp:
            prepared = prepare_artifact_for_persist(artifact, state_dir=tmp)
        self.assertEqual(prepared.payload["claim"], "tiny claim")
        self.assertNotIn(BOUNDS_KEY, prepared.payload)

    def test_oversized_finding_offloads_with_preview(self) -> None:
        # Force a tiny global byte cap so a long claim must be bounded.
        claim = "HEAD-" + ("body" * 20_000) + "-TAIL"
        artifact = _finding(claim=claim)
        self.assertGreater(serialized_artifact_bytes(artifact), 8_000)
        with TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {
                    "PUPPETMASTER_ARTIFACT_MAX_BYTES": "4096",
                    "PUPPETMASTER_ARTIFACT_TEXT_FIELD_MAX_CHARS": "2000",
                    "PUPPETMASTER_ARTIFACT_PREVIEW_HEAD_CHARS": "80",
                    "PUPPETMASTER_ARTIFACT_PREVIEW_TAIL_CHARS": "80",
                },
            ):
                prepared = prepare_artifact_for_persist(artifact, state_dir=tmp)
            self.assertLessEqual(
                serialized_artifact_bytes(prepared),
                4096,
            )
            self.assertNotEqual(prepared.payload["claim"], claim)
            self.assertTrue(prepared.payload["claim"].startswith("HEAD-"))
            self.assertTrue(prepared.payload["claim"].endswith("-TAIL"))
            self.assertIn("omitted", prepared.payload["claim"])
            bounds = artifact_bounds_summary(prepared)
            self.assertTrue(bounds.get("truncated"))
            field = bounds["fields"]["claim"]
            self.assertTrue(field["truncated"])
            self.assertTrue(field["offloaded"])
            offload = Path(field["offload_path"])
            self.assertTrue(offload.is_file())
            self.assertEqual(offload.read_text(encoding="utf-8"), claim)
            # Evidence is preserved for reviewability.
            self.assertTrue(prepared.evidence)

    def test_oversized_patch_diff_sets_sidecar_and_flags(self) -> None:
        diff = "diff --git a/f b/f\n" + ("+line\n" * 30_000)
        artifact = _patch(diff=diff)
        with TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {
                    "PUPPETMASTER_ARTIFACT_PATCH_DIFF_MAX_CHARS": "1500",
                    "PUPPETMASTER_ARTIFACT_PREVIEW_HEAD_CHARS": "120",
                    "PUPPETMASTER_ARTIFACT_PREVIEW_TAIL_CHARS": "120",
                    "PUPPETMASTER_ARTIFACT_MAX_BYTES": str(DEFAULT_MAX_ARTIFACT_BYTES),
                },
            ):
                prepared = prepare_artifact_for_persist(artifact, state_dir=tmp)
            inline = prepared.payload["unified_diff"]
            self.assertLessEqual(len(inline), 1500 + 80)  # allow omission marker slack
            self.assertTrue(prepared.payload["diff_truncated"])
            self.assertGreaterEqual(prepared.payload["diff_total_chars"], len(diff))
            sidecar = prepared.payload.get("unified_diff_sidecar_path")
            self.assertIsNotNone(sidecar)
            self.assertTrue(Path(sidecar).is_file())
            self.assertEqual(Path(sidecar).read_text(encoding="utf-8"), diff)
            bounds = artifact_bounds_summary(prepared)
            self.assertTrue(bounds["fields"]["unified_diff"]["offloaded"])

    def test_patch_files_list_cap(self) -> None:
        files = [f"file_{i}.py" for i in range(1_000)]
        artifact = _patch(diff="diff --git a/a b/a\n", files=files)
        with patch.dict(os.environ, {"PUPPETMASTER_ARTIFACT_PATCH_FILES_MAX": "10"}):
            prepared = prepare_artifact_for_persist(artifact, state_dir=None)
        self.assertEqual(len(prepared.payload["files"]), 10)
        self.assertTrue(prepared.payload["files_truncated"])
        self.assertEqual(prepared.payload["files_total"], 1_000)

    def test_kill_switch_skips_bounding(self) -> None:
        claim = "x" * 100_000
        artifact = _finding(claim=claim)
        with TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {
                    "PUPPETMASTER_ARTIFACT_BOUNDS": "0",
                    "PUPPETMASTER_ARTIFACT_MAX_BYTES": "1024",
                },
            ):
                self.assertFalse(bounds_enabled())
                prepared = prepare_artifact_for_persist(artifact, state_dir=tmp)
            self.assertEqual(prepared.payload["claim"], claim)
            self.assertEqual(
                list(Path(tmp).joinpath("jobs").glob("**/artifact_offload/**/*.txt")),
                [],
            )

    def test_missing_state_dir_soft_truncates_without_raise(self) -> None:
        claim = "Y" * 50_000
        artifact = _finding(claim=claim)
        with patch.dict(
            os.environ,
            {
                "PUPPETMASTER_ARTIFACT_MAX_BYTES": "2048",
                "PUPPETMASTER_ARTIFACT_TEXT_FIELD_MAX_CHARS": "1000",
            },
        ):
            prepared = prepare_artifact_for_persist(artifact, state_dir=None)
        self.assertLess(len(prepared.payload["claim"]), len(claim))
        bounds = artifact_bounds_summary(prepared)
        self.assertTrue(bounds.get("truncated"))
        self.assertFalse(bounds["fields"]["claim"]["offloaded"])

    def test_untrusted_offload_path_is_never_opened(self) -> None:
        """Re-persist must not disclose arbitrary local files via bounds metadata."""
        secret = "SECRET-SHOULD-NOT-LEAK-" + ("x" * 200)
        with TemporaryDirectory() as tmp:
            bait = Path(tmp) / "bait-secret.txt"
            bait.write_text(secret, encoding="utf-8")
            # Oversized claim so shrink runs; plant an attacker-controlled path
            # that would be opened by the old offload_path re-read path.
            claim = "SAFE-HEAD-" + ("body" * 15_000) + "-SAFE-TAIL"
            artifact = _finding(claim=claim)
            artifact.payload[BOUNDS_KEY] = {
                "truncated": True,
                "original_bytes": 99999,
                "stored_bytes": 99999,
                "fields": {
                    "claim": {
                        "truncated": True,
                        "original_chars": len(claim),
                        "preview_chars": 100,
                        "offloaded": True,
                        "offload_path": str(bait),
                        "reason": "attacker",
                    }
                },
            }
            with patch.dict(
                os.environ,
                {
                    "PUPPETMASTER_ARTIFACT_MAX_BYTES": "2048",
                    "PUPPETMASTER_ARTIFACT_TEXT_FIELD_MAX_CHARS": "800",
                    "PUPPETMASTER_ARTIFACT_PREVIEW_HEAD_CHARS": "60",
                    "PUPPETMASTER_ARTIFACT_PREVIEW_TAIL_CHARS": "60",
                },
            ), patch.object(Path, "read_text", autospec=True) as read_text:
                # Any Path.read_text during prepare would be a disclosure bug;
                # force a failure so the call cannot silently succeed.
                read_text.side_effect = AssertionError(
                    "artifact persistence must not open filesystem paths"
                )
                prepared = prepare_artifact_for_persist(artifact, state_dir=tmp)
            self.assertNotIn(secret, prepared.payload.get("claim", ""))
            self.assertNotIn(secret, json.dumps(prepared.payload))
            # Still produces reviewable bounded output from the claim itself.
            self.assertTrue(prepared.payload["claim"].startswith("SAFE-HEAD-"))
            self.assertTrue(prepared.payload["claim"].endswith("-SAFE-TAIL"))
            self.assertIn("omitted", prepared.payload["claim"])
            self.assertLessEqual(serialized_artifact_bytes(prepared), 2048)

class ArtifactBoundsStoreTests(unittest.TestCase):
    def test_sqlite_save_bounds_oversized_finding(self) -> None:
        claim = "Z" * 80_000
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(tmp)
            store.init()
            job = store.create_job("bound me")
            artifact = replace(_finding(claim=claim), job_id=job.id)
            with patch.dict(
                os.environ,
                {
                    "PUPPETMASTER_ARTIFACT_MAX_BYTES": "8192",
                    "PUPPETMASTER_ARTIFACT_TEXT_FIELD_MAX_CHARS": "2500",
                    "PUPPETMASTER_ARTIFACT_PREVIEW_HEAD_CHARS": "100",
                    "PUPPETMASTER_ARTIFACT_PREVIEW_TAIL_CHARS": "100",
                },
            ):
                store.save_artifact(artifact)
            loaded = store.list_artifacts(job.id)
            self.assertEqual(len(loaded), 1)
            saved = loaded[0]
            self.assertLessEqual(serialized_artifact_bytes(saved), 8192)
            self.assertNotEqual(saved.payload["claim"], claim)
            bounds = artifact_bounds_summary(saved)
            self.assertTrue(bounds["truncated"])
            offload = Path(bounds["fields"]["claim"]["offload_path"])
            self.assertTrue(offload.is_file())
            self.assertEqual(offload.read_text(encoding="utf-8"), claim)
            self.assertIsNotNone(saved.sha256)

    def test_file_store_save_bounds_patch_diff(self) -> None:
        diff = "diff --git a/x b/x\n" + ("+n\n" * 40_000)
        with TemporaryDirectory() as tmp:
            store = SwarmStore(tmp)
            store.init()
            job = store.create_job("bound patch")
            artifact = replace(_patch(diff=diff), job_id=job.id)
            with patch.dict(
                os.environ,
                {
                    "PUPPETMASTER_ARTIFACT_PATCH_DIFF_MAX_CHARS": "2000",
                    "PUPPETMASTER_ARTIFACT_PREVIEW_HEAD_CHARS": "150",
                    "PUPPETMASTER_ARTIFACT_PREVIEW_TAIL_CHARS": "150",
                },
            ):
                store.save_artifact(artifact)
            saved = store.list_artifacts(job.id)[0]
            self.assertTrue(saved.payload["diff_truncated"])
            sidecar = Path(saved.payload["unified_diff_sidecar_path"])
            self.assertTrue(sidecar.is_file())
            self.assertEqual(sidecar.read_text(encoding="utf-8"), diff)
            # Recoverable preview remains inline for review UIs.
            self.assertIn("omitted", saved.payload["unified_diff"])

if __name__ == "__main__":
    unittest.main()
