from __future__ import annotations

import unittest

from puppetmaster.rate_limit_headers import parse_rate_limit_headers, reset_at

NOW = 1767225600.0  # 2026-01-01T00:00:00Z


class Rfc3339ResetTests(unittest.TestCase):
    def test_anthropic_rfc3339_reset_blocks_exhausted_window(self) -> None:
        snap = parse_rate_limit_headers(
            {
                "anthropic-ratelimit-requests-remaining": "0",
                "anthropic-ratelimit-requests-reset": "2026-01-01T00:01:00Z",
            },
            now=NOW,
        )
        self.assertIsNotNone(snap)
        self.assertEqual(snap.requests.reset_at, NOW + 60.0)
        self.assertEqual(snap.blocking_reset_at(now=NOW), NOW + 60.0)
        self.assertTrue(snap.is_exhausted(now=NOW))

    def test_rfc3339_with_offset(self) -> None:
        self.assertEqual(reset_at("2026-01-01T02:01:00+02:00", now=NOW), NOW + 60.0)

    def test_rfc2822_reset_keeps_old_result(self) -> None:
        self.assertEqual(
            reset_at("Thu, 01 Jan 2026 00:01:00 GMT", now=NOW), NOW + 60.0
        )

    def test_duration_and_garbage_keep_old_result(self) -> None:
        self.assertEqual(reset_at("2m59s", now=NOW), NOW + 179.0)
        self.assertEqual(reset_at("30", now=NOW), NOW + 30.0)
        self.assertIsNone(reset_at("not-a-time", now=NOW))

    def test_healthy_window_with_rfc3339_reset_does_not_block(self) -> None:
        snap = parse_rate_limit_headers(
            {
                "anthropic-ratelimit-requests-remaining": "5",
                "anthropic-ratelimit-requests-reset": "2026-01-01T00:01:00Z",
            },
            now=NOW,
        )
        self.assertIsNone(snap.blocking_reset_at(now=NOW))


if __name__ == "__main__":
    unittest.main()
