from __future__ import annotations

import unittest

from puppetmaster.write_intent import adapter_may_write


class AntigravityAnalyzeModeTests(unittest.TestCase):
    def test_analyze_mode_is_read_only(self) -> None:
        # The adapter runs "analyze" as agy --mode plan, which cannot write.
        self.assertFalse(adapter_may_write("antigravity", {"mode": "analyze"}))
        self.assertFalse(adapter_may_write("agy", {"mode": "Analyze"}))

    def test_near_inputs_keep_their_result(self) -> None:
        self.assertFalse(adapter_may_write("antigravity", {"mode": "plan"}))
        self.assertFalse(adapter_may_write("antigravity", {"read_only": True}))
        self.assertTrue(adapter_may_write("antigravity", {"mode": "accept-edits"}))
        self.assertTrue(adapter_may_write("antigravity", {"mode": "implement"}))
        self.assertTrue(adapter_may_write("antigravity", {}))


if __name__ == "__main__":
    unittest.main()
