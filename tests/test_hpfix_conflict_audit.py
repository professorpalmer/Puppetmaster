"""Regression tests for the latent faults that the Hard Proof conflict audit confirmed."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from puppetmaster import codex_home
from puppetmaster.adapters import codex_rollout
from puppetmaster.models import Artifact, ArtifactType
from puppetmaster.usage import select_usage_records, usage_record_score


def _usage(task_id: str, payload: dict) -> Artifact:
    return Artifact(job_id="job", task_id=task_id, type=ArtifactType.VERIFICATION, created_by="worker",
                    payload=payload, confidence=1.0, evidence=[])


class UsageTextCountTests(unittest.TestCase):
    def test_text_that_is_not_a_number_does_not_stop_the_selection(self) -> None:
        records = select_usage_records([_usage("t", {"tokens_in": "a", "tokens_out": 7}),
                                        _usage("u", {"tokens_in": 3, "tokens_out": 4})])
        self.assertEqual(records["t"]["tokens_in"], 0)
        self.assertEqual(records["t"]["tokens_out"], 7)
        self.assertEqual(records["u"]["tokens_in"], 3)

    def test_score_of_a_payload_with_text_counts(self) -> None:
        self.assertEqual(usage_record_score({"tokens_in": "a", "tokens_out": "b"})[-1], 0)

    def test_a_numeric_text_keeps_its_value(self) -> None:
        self.assertEqual(select_usage_records([_usage("t", {"tokens_in": "12", "tokens_out": 1})])["t"]["tokens_in"], 12)


class RolloutExactIdTests(unittest.TestCase):
    def _sessions(self, tmp: str, *ids: str) -> Path:
        day = Path(tmp) / "sessions" / "2026" / "10" / "04"
        day.mkdir(parents=True)
        for session_id in ids:
            (day / f"rollout-2026-10-04T09-00-00-{session_id}.jsonl").write_text("{}\n", encoding="utf-8")
        return Path(tmp) / "sessions"

    def test_the_end_of_a_different_id_is_not_a_match(self) -> None:
        with TemporaryDirectory() as tmp:
            root = self._sessions(tmp, "1-5", "0199aaaa-bbbb-7ccc-8ddd-eeeeeeeeeeee")
            self.assertIsNone(codex_home.rollout_file(root, "5"))
            self.assertIsNone(codex_home.rollout_file(root, "eeeeeeeeeeee"))
            self.assertIsNone(codex_rollout.rollout_path(Path(tmp), "5"))

    def test_the_exact_id_is_a_match(self) -> None:
        with TemporaryDirectory() as tmp:
            root = self._sessions(tmp, "1-5", "0199aaaa-bbbb-7ccc-8ddd-eeeeeeeeeeee")
            self.assertEqual(codex_home.rollout_file(root, "1-5").name, "rollout-2026-10-04T09-00-00-1-5.jsonl")
            found = codex_rollout.rollout_path(Path(tmp), "0199aaaa-bbbb-7ccc-8ddd-eeeeeeeeeeee")
            self.assertIsNotNone(found)


if __name__ == "__main__":
    unittest.main()
