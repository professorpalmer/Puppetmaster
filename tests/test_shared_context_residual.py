"""Wave 2 catalog residual: last-wins FINDING/GIST/RISK handles."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import unittest
from typing import Any, Optional

from puppetmaster.models import Artifact, ArtifactType
from puppetmaster.prewalk import (
    format_plan_artifacts_for_injection,
    format_upstream_artifacts_for_injection,
)
from puppetmaster.working_set import (
    format_shared_context_residual,
    last_wins_residual,
    residual_story_key,
    write_artifact_index,
)


OLDER_BODY = "OLDER_FINDING_BODY_MUST_NOT_APPEAR"
OLDER_EVIDENCE = "OLDER_EVIDENCE_MUST_NOT_APPEAR"
NEWER_HEADLINE = "tokens leak in the request log"


def _finding(
    *,
    artifact_id: str,
    claim: str,
    created_at: str = "2026-08-01T00:00:00+00:00",
    validation: Optional[dict] = None,
    details: Optional[str] = None,
    evidence: Optional[list] = None,
    extra_payload: Optional[dict] = None,
) -> Artifact:
    payload: dict[str, Any] = {"claim": claim}
    if details is not None:
        payload["details"] = details
    if extra_payload:
        payload.update(extra_payload)
    if validation is not None:
        payload["validation"] = validation
    return Artifact(
        job_id="job-residual",
        task_id="task-residual",
        type=ArtifactType.FINDING,
        created_by="tester",
        payload=payload,
        confidence=0.9,
        evidence=evidence or ["src/a.py"],
        id=artifact_id,
        created_at=created_at,
    )


def _decision() -> dict:
    return {
        "type": "decision",
        "payload": {
            "decision": "Add retry helper in client.py",
            "why": "Centralize backoff",
            "plan": ["Create retry_with_backoff", "Wire callers"],
        },
    }


class ResidualStoryKeyTests(unittest.TestCase):
    def test_normalizes_claim_whitespace_and_case(self) -> None:
        artifact = _finding(artifact_id="art-key", claim="  Tokens   LEAK  ")
        self.assertEqual(residual_story_key(artifact), "tokens leak")

    def test_empty_claim_is_not_grouped(self) -> None:
        artifact = _finding(artifact_id="art-empty", claim="")
        self.assertEqual(residual_story_key(artifact), "")


class LastWinsResidualTests(unittest.TestCase):
    def test_newer_created_at_wins_same_claim(self) -> None:
        older = _finding(
            artifact_id="art-old",
            claim=NEWER_HEADLINE,
            created_at="2026-01-01T00:00:00+00:00",
            details=OLDER_BODY,
        )
        newer = _finding(
            artifact_id="art-new",
            claim=NEWER_HEADLINE,
            created_at="2026-08-29T00:00:00+00:00",
        )
        result = last_wins_residual([older, newer])
        self.assertEqual([item.id for item in result["selected"]], [newer.id])
        self.assertEqual(len(result["omitted"]), 1)
        self.assertEqual(result["omitted"][0]["id"], older.id)
        self.assertEqual(result["omitted"][0]["superseded_by"], newer.id)

    def test_generation_beats_created_at(self) -> None:
        older_higher_gen = _finding(
            artifact_id="art-gen",
            claim=NEWER_HEADLINE,
            created_at="2026-01-01T00:00:00+00:00",
            validation={"status": "fresh", "generation": 2},
        )
        newer_lower_gen = _finding(
            artifact_id="art-late",
            claim=NEWER_HEADLINE,
            created_at="2026-08-29T00:00:00+00:00",
            validation={"status": "fresh", "generation": 1},
        )
        result = last_wins_residual([older_higher_gen, newer_lower_gen])
        self.assertEqual(
            [item.id for item in result["selected"]], [older_higher_gen.id]
        )
        self.assertEqual(result["omitted"][0]["id"], newer_lower_gen.id)

    def test_plan_and_patch_are_not_grouped(self) -> None:
        finding = _finding(artifact_id="art-find", claim="retry helper")
        decision = {
            "id": "art-dec",
            "type": "decision",
            "payload": {"decision": "retry helper", "why": "same words"},
        }
        patch = {
            "id": "art-patch",
            "type": "patch",
            "payload": {"change": "retry helper", "files": ["client.py"]},
        }
        result = last_wins_residual([finding, decision, patch])
        self.assertEqual(len(result["selected"]), 3)
        self.assertEqual(result["omitted"], [])


class SharedContextResidualInjectTests(unittest.TestCase):
    def test_last_wins_injects_newer_handle_and_overlap_query(self) -> None:
        older = _finding(
            artifact_id="art-old",
            claim=NEWER_HEADLINE,
            created_at="2026-01-01T00:00:00+00:00",
            details=OLDER_BODY,
            evidence=[OLDER_EVIDENCE],
            extra_payload={
                "source_digests": {"src/a.py": "digest-must-not-appear"},
            },
        )
        newer = _finding(
            artifact_id="art-new",
            claim=NEWER_HEADLINE,
            created_at="2026-08-29T00:00:00+00:00",
        )
        text = format_upstream_artifacts_for_injection([older, newer])
        self.assertIn(newer.id, text)
        self.assertIn(NEWER_HEADLINE, text)
        self.assertNotIn(OLDER_BODY, text)
        self.assertNotIn(OLDER_EVIDENCE, text)
        self.assertNotIn("digest-must-not-appear", text)
        self.assertIn("Overlap:", text)
        self.assertIn("last-wins", text)
        self.assertIn("puppetmaster show", text)
        self.assertIn(older.id, text)
        self.assertIn("effort-index", text)

    def test_unlabeled_finding_still_gets_a_handle(self) -> None:
        finding = _finding(artifact_id="art-unlabeled", claim="unlabeled story")
        text = format_shared_context_residual([finding])
        self.assertIn(finding.id, text)
        self.assertIn("unlabeled story", text)
        self.assertNotIn("Overlap:", text)

    def test_stale_finding_does_not_appear(self) -> None:
        stale = _finding(
            artifact_id="art-stale",
            claim="cited bytes have moved",
            validation={"status": "stale"},
        )
        fresh = _finding(
            artifact_id="art-fresh",
            claim="a different live claim",
        )
        text = format_upstream_artifacts_for_injection([stale, fresh])
        self.assertNotIn(stale.id, text)
        self.assertNotIn("cited bytes have moved", text)
        self.assertIn(fresh.id, text)
        self.assertNotIn("Overlap:", text)

    def test_distinct_claims_both_appear_without_overlap(self) -> None:
        first = _finding(artifact_id="art-a", claim="first distinct claim")
        second = _finding(artifact_id="art-b", claim="second distinct claim")
        text = format_upstream_artifacts_for_injection([first, second])
        self.assertIn(first.id, text)
        self.assertIn("first distinct claim", text)
        self.assertIn(second.id, text)
        self.assertIn("second distinct claim", text)
        self.assertNotIn("Overlap:", text)

    def test_plan_decision_still_inject_via_format_plan(self) -> None:
        finding = _finding(artifact_id="art-side", claim="unrelated residual")
        artifacts = [_decision(), finding]
        plan_text = format_plan_artifacts_for_injection(artifacts)
        self.assertIn("Decision: Add retry helper in client.py", plan_text)
        self.assertIn("Why: Centralize backoff", plan_text)
        self.assertIn("Create retry_with_backoff", plan_text)
        self.assertNotIn(finding.id, plan_text)

        upstream = format_upstream_artifacts_for_injection(artifacts)
        self.assertIn("Decision: Add retry helper in client.py", upstream)
        self.assertIn("Why: Centralize backoff", upstream)
        self.assertIn(finding.id, upstream)
        self.assertIn("unrelated residual", upstream)

    def test_empty_after_filter_is_empty_string(self) -> None:
        stale = _finding(
            artifact_id="art-only-stale",
            claim="gone",
            validation={"status": "stale"},
        )
        self.assertEqual(format_shared_context_residual([stale]), "")
        self.assertEqual(format_upstream_artifacts_for_injection([stale]), "")

    def test_index_keeps_all_refs_and_records_residual_ids(self) -> None:
        from tempfile import TemporaryDirectory
        from pathlib import Path
        import json

        older = _finding(
            artifact_id="art-old",
            claim=NEWER_HEADLINE,
            created_at="2026-01-01T00:00:00+00:00",
        )
        newer = _finding(
            artifact_id="art-new",
            claim=NEWER_HEADLINE,
            created_at="2026-08-29T00:00:00+00:00",
        )
        with TemporaryDirectory() as tmp:
            path = write_artifact_index(Path(tmp) / "job", [older, newer])
            self.assertIsNotNone(path)
            assert path is not None
            raw = json.loads(path.read_text(encoding="utf-8"))
            ids = {item["id"] for item in raw["artifacts"]}
            self.assertEqual(ids, {older.id, newer.id})
            self.assertEqual(raw["residual"]["selected_ids"], [newer.id])
            self.assertEqual(raw["residual"]["omitted_ids"], [older.id])


if __name__ == "__main__":
    unittest.main()
