from __future__ import annotations

import unittest

from puppetmaster.acceptance_criteria import (
    MAX_CRITERION_CHARS,
    criterion_status_records,
)


def _rows(criteria, reported):
    rows = criterion_status_records(criteria, reported)
    return [(r["criterion"], r["status"], r["evidence"]) for r in rows]


class ReportedMarkerEchoTests(unittest.TestCase):
    def test_dash_marker_echo_matches(self) -> None:
        self.assertEqual(
            _rows(
                ["- Tests pass"],
                [{"criterion": "- Tests pass", "status": "passed", "evidence": "ok"}],
            ),
            [("Tests pass", "passed", "ok")],
        )

    def test_numbered_marker_echo_matches(self) -> None:
        self.assertEqual(
            _rows(
                ["Tests pass"],
                [{"criterion": "1. Tests pass", "status": "failed", "evidence": "pytest red"}],
            ),
            [("Tests pass", "failed", "pytest red")],
        )

    def test_long_echo_matches_cut_criterion(self) -> None:
        text = "x" * (MAX_CRITERION_CHARS + 50)
        self.assertEqual(
            _rows(
                [text],
                [{"criterion": "- " + text, "status": "passed", "evidence": "ok"}],
            ),
            [("x" * MAX_CRITERION_CHARS, "passed", "ok")],
        )

    def test_plain_key_keeps_old_result(self) -> None:
        self.assertEqual(
            _rows(
                ["Tests pass"],
                [{"criterion": "Tests pass", "status": "passed", "evidence": "ok"}],
            ),
            [("Tests pass", "passed", "ok")],
        )

    def test_other_text_stays_unknown(self) -> None:
        self.assertEqual(
            _rows(
                ["Tests pass"],
                [{"criterion": "- Lint passes", "status": "passed", "evidence": "ok"}],
            ),
            [("Tests pass", "unknown", "not_reported")],
        )

    def test_marker_only_key_is_ignored(self) -> None:
        self.assertEqual(
            _rows(
                ["Tests pass"],
                [{"criterion": "-", "status": "passed", "evidence": "ok"}],
            ),
            [("Tests pass", "unknown", "not_reported")],
        )


if __name__ == "__main__":
    unittest.main()
