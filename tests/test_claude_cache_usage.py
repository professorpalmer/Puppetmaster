"""Claude Code reports cache usage under Anthropic's field names."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hermetic_env  # noqa: F401,E402

from puppetmaster.usage import selected_token_usage, usage_from_sdk  # noqa: E402

# Shape of a real `claude --print --output-format json` usage block (2026-10-04).
CLAUDE_USAGE = {
    "input_tokens": 26,
    "cache_creation_input_tokens": 5493,
    "cache_read_input_tokens": 260510,
    "output_tokens": 3729,
}


class ClaudeCacheUsageTests(unittest.TestCase):
    def test_sdk_usage_keeps_anthropic_cache_counts(self) -> None:
        usage = usage_from_sdk(CLAUDE_USAGE)
        self.assertEqual(usage["tokens_in"], 26)
        self.assertEqual(usage["tokens_out"], 3729)
        self.assertEqual(usage["cache_read_tokens"], 260510)
        self.assertEqual(usage["cache_write_tokens"], 5493)

    def test_selected_usage_keeps_anthropic_cache_counts(self) -> None:
        selected = selected_token_usage(CLAUDE_USAGE)["selected_facts"]
        self.assertEqual(selected["cache_read_tokens"], 260510)
        self.assertEqual(selected["cache_write_tokens"], 5493)

    def test_cursor_names_still_win_and_missing_stays_absent(self) -> None:
        cursor = usage_from_sdk({"inputTokens": 5, "outputTokens": 7, "cacheReadTokens": 11})
        self.assertEqual(cursor["cache_read_tokens"], 11)
        self.assertNotIn("cache_write_tokens", cursor)


if __name__ == "__main__":
    unittest.main()
