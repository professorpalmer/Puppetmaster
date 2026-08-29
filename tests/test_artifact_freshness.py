"""Cited-artifact freshness at inject and effort-index (Sketch B)."""
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

from puppetmaster.gist_admission import (
    filter_shared_context_artifacts,
    is_admitted_for_shared_context,
)
from puppetmaster.lifecycle import index_effort_artifacts, tag_job_effort
from puppetmaster.models import Artifact, ArtifactType
from puppetmaster.store import SwarmStore
from puppetmaster.validation import (
    VALIDATION_STATUSES,
    compute_validation_fingerprint,
    refresh_cited_freshness,
    validation_status_of,
)


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


def _stamped_finding(
    root: Path,
    *,
    job_id: str = "job-fresh",
    task_id: str = "task-fresh",
    claim: str = "cited path is current",
    status: str = "fresh",
) -> Artifact:
    result = compute_validation_fingerprint(root, ["src/a.py"], strict=False)
    payload = {
        "claim": claim,
        "validation": result.to_payload(status=status),
    }
    return Artifact(
        job_id=job_id,
        task_id=task_id,
        type=ArtifactType.FINDING,
        created_by="tester",
        payload=payload,
        confidence=0.9,
        evidence=["src/a.py"],
    )


def _unlabeled_finding(
    *,
    job_id: str = "job-fresh",
    task_id: str = "task-fresh",
    claim: str = "legacy unlabeled finding",
) -> Artifact:
    return Artifact(
        job_id=job_id,
        task_id=task_id,
        type=ArtifactType.FINDING,
        created_by="tester",
        payload={"claim": claim},
        confidence=0.9,
        evidence=["src/a.py"],
    )


class ArtifactFreshnessTests(unittest.TestCase):
    def test_unprovable_is_reason_not_status(self) -> None:
        self.assertNotIn("unprovable", VALIDATION_STATUSES)
        self.assertEqual(
            VALIDATION_STATUSES, frozenset({"fresh", "reused", "stale", "superseded"})
        )

    def test_matching_tree_stays_admitted(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init_with_file(root, "src/a.py", "alpha\n")
            finding = _stamped_finding(root)
            refreshed = refresh_cited_freshness(finding, root)
            self.assertEqual(validation_status_of(refreshed), "fresh")
            self.assertNotIn("freshness_reason", refreshed.payload["validation"])
            admitted = filter_shared_context_artifacts([refreshed], cwd=root)
            self.assertEqual([item.id for item in admitted], [finding.id])

    def test_mutate_cited_bytes_marks_stale_and_drops_from_inject(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init_with_file(root, "src/a.py", "alpha\n")
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("mutate freshness")
            finding = _stamped_finding(root, job_id=job.id)
            store.save_artifact(finding)
            (root / "src/a.py").write_text("alpha-dirty\n", encoding="utf-8")

            admitted = filter_shared_context_artifacts(
                [finding], cwd=root, store=store
            )
            self.assertEqual(admitted, [])

            loaded = [
                item for item in store.list_artifacts(job.id) if item.id == finding.id
            ]
            self.assertEqual(len(loaded), 1)
            stale = loaded[0]
            self.assertEqual(validation_status_of(stale), "stale")
            self.assertEqual(
                stale.payload["validation"]["freshness_reason"], "digest_mismatch"
            )
            self.assertFalse(is_admitted_for_shared_context(stale))

    def test_missing_cited_path_is_unprovable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init_with_file(root, "src/a.py", "alpha\n")
            finding = _stamped_finding(root)
            (root / "src/a.py").unlink()
            refreshed = refresh_cited_freshness(finding, root)
            self.assertEqual(validation_status_of(refreshed), "stale")
            validation = refreshed.payload["validation"]
            self.assertEqual(validation["freshness_reason"], "unprovable")
            self.assertIn("src/a.py", validation.get("missing_paths") or [])
            self.assertEqual(
                filter_shared_context_artifacts([refreshed], cwd=root), []
            )

    def test_unlabeled_finding_still_admitted(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init_with_file(root, "src/a.py", "alpha\n")
            finding = _unlabeled_finding()
            (root / "src/a.py").write_text("changed\n", encoding="utf-8")
            refreshed = refresh_cited_freshness(finding, root)
            self.assertIsNone(validation_status_of(refreshed))
            self.assertTrue(is_admitted_for_shared_context(refreshed))
            admitted = filter_shared_context_artifacts([refreshed], cwd=root)
            self.assertEqual([item.id for item in admitted], [finding.id])

    def test_superseded_stays_superseded(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init_with_file(root, "src/a.py", "alpha\n")
            finding = _stamped_finding(root, status="superseded")
            (root / "src/a.py").write_text("alpha-dirty\n", encoding="utf-8")
            refreshed = refresh_cited_freshness(finding, root)
            self.assertEqual(validation_status_of(refreshed), "superseded")
            self.assertNotIn("freshness_reason", refreshed.payload["validation"])
            self.assertFalse(is_admitted_for_shared_context(refreshed))
            self.assertEqual(
                filter_shared_context_artifacts([refreshed], cwd=root), []
            )

    def test_matching_tree_restores_previously_stale(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init_with_file(root, "src/a.py", "alpha\n")
            finding = _stamped_finding(root)
            (root / "src/a.py").write_text("alpha-dirty\n", encoding="utf-8")
            stale = refresh_cited_freshness(finding, root)
            self.assertEqual(validation_status_of(stale), "stale")
            (root / "src/a.py").write_text("alpha\n", encoding="utf-8")
            restored = refresh_cited_freshness(stale, root)
            self.assertEqual(validation_status_of(restored), "fresh")
            self.assertNotIn("freshness_reason", restored.payload["validation"])
            self.assertTrue(is_admitted_for_shared_context(restored))

    def test_reused_status_kept_when_digests_match(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init_with_file(root, "src/a.py", "alpha\n")
            finding = _stamped_finding(root, status="reused")
            refreshed = refresh_cited_freshness(finding, root)
            self.assertEqual(validation_status_of(refreshed), "reused")
            self.assertEqual(
                [item.id for item in filter_shared_context_artifacts([refreshed], cwd=root)],
                [finding.id],
            )

    def test_effort_index_omits_stale(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init_with_file(root, "src/a.py", "alpha\n")
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("effort freshness")
            tag_job_effort(store, job.id, "freshness-effort")
            fresh = _stamped_finding(
                root, job_id=job.id, claim="fresh cited claim"
            )
            stale = _stamped_finding(
                root, job_id=job.id, claim="stale cited claim"
            )
            store.save_artifact(fresh)
            store.save_artifact(stale)
            (root / "src/a.py").write_text("alpha-dirty\n", encoding="utf-8")
            refresh_cited_freshness(stale, root, store=store)

            payload = index_effort_artifacts(
                [store], effort_id="freshness-effort"
            )
            ids = {row["id"] for row in payload["refs"]}
            self.assertIn(fresh.id, ids)
            self.assertNotIn(stale.id, ids)
            raw_ids = {item.id for item in store.list_artifacts(job.id)}
            self.assertIn(stale.id, raw_ids)


if __name__ == "__main__":
    unittest.main()
