from __future__ import annotations

import json
import unittest

from puppetmaster.adapters import cursor as c
from puppetmaster.models import Task


def _task() -> Task:
    return Task(job_id="j", role="r", instruction="i")


class EvidenceNullEntryTests(unittest.TestCase):
    def test_null_evidence_entry_is_skipped(self) -> None:
        art = c.cursor_artifact_from_item(
            _task(), "w", {"type": "finding", "claim": "x", "evidence": ["a.py:1", None]}
        )
        self.assertEqual(art.evidence, ["a.py:1"])

    def test_only_null_evidence_uses_adapter_fallback(self) -> None:
        art = c.cursor_artifact_from_item(
            _task(), "w", {"type": "finding", "claim": "x", "evidence": [None]}, adapter="codex"
        )
        self.assertEqual(art.evidence, ["adapter:codex"])

    def test_non_string_evidence_keeps_old_result(self) -> None:
        art = c.cursor_artifact_from_item(
            _task(), "w", {"type": "finding", "claim": "x", "evidence": ["a.py:1", 0, "  "]}
        )
        self.assertEqual(art.evidence, ["a.py:1", "0"])


class ConfidenceNonFiniteTests(unittest.TestCase):
    def test_json_nan_confidence_gets_default(self) -> None:
        item = json.loads('{"type": "finding", "claim": "x", "confidence": NaN}')
        art = c.cursor_artifact_from_item(_task(), "w", item)
        self.assertEqual(art.confidence, 0.75)

    def test_string_nan_and_inf_get_default(self) -> None:
        for value in ("nan", "inf", float("-inf")):
            art = c.cursor_artifact_from_item(
                _task(), "w", {"type": "finding", "claim": "x", "confidence": value}
            )
            self.assertEqual(art.confidence, 0.75, value)

    def test_finite_confidence_keeps_clamp(self) -> None:
        self.assertEqual(c._confidence(-3), 0.0)
        self.assertEqual(c._confidence(7), 1.0)
        self.assertEqual(c._confidence("0.4"), 0.4)
        self.assertEqual(c._confidence("bad"), 0.75)


class WrappedEmptyTypeTests(unittest.TestCase):
    def test_wrapped_empty_inner_type_uses_wrapper_key(self) -> None:
        art = c.cursor_artifact_from_item(
            _task(), "w", {"finding": {"type": "", "claim": "x"}}, adapter="codex"
        )
        self.assertIsNotNone(art)
        self.assertEqual(art.payload["claim"], "x")

    def test_wrapped_explicit_inner_type_wins(self) -> None:
        art = c.cursor_artifact_from_item(
            _task(), "w", {"finding": {"type": "risk", "risk": "r"}}, adapter="codex"
        )
        self.assertEqual(art.payload["risk"], "r")

    def test_wrapped_unknown_inner_type_still_dropped(self) -> None:
        art = c.cursor_artifact_from_item(
            _task(), "w", {"finding": {"type": "note", "claim": "x"}}, adapter="codex"
        )
        self.assertIsNone(art)


class PrefixDecodeScalarListTests(unittest.TestCase):
    def test_bracket_citation_before_payload_keeps_findings(self) -> None:
        text = 'Review of file [1] done. {"findings":[{"claim":"bug"}]} Done {ok}'
        arts = c.cursor_result_artifacts(_task(), "w", text)
        self.assertEqual([a.payload.get("claim") for a in arts], ["bug"])

    def test_prefix_decode_skips_scalar_list(self) -> None:
        text = 'see [1] and [2, "x"] then {"findings": []} trailer {x}'
        self.assertEqual(c._json_prefix_decode(text), {"findings": []})

    def test_list_of_objects_still_first_match(self) -> None:
        text = 'prose [{"type": "finding", "claim": "a"}] then {"findings": []} {x}'
        self.assertEqual(
            c._json_prefix_decode(text), [{"type": "finding", "claim": "a"}]
        )

    def test_prose_without_container_still_none(self) -> None:
        self.assertIsNone(c._json_prefix_decode("see [1] and {oops"))
        self.assertIsNone(c.parse_cursor_artifact_payload("plain prose, no payload"))


if __name__ == "__main__":
    unittest.main()
