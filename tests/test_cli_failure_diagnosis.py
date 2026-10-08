"""Failure classes come from a CLI's own diagnostics, never from the worker's answer."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import hermetic_env  # noqa: F401
from puppetmaster.failure import classify_fx_failure, classify_hermes_failure, hermes_diagnostic

ANSWER = "The login flow returned 401 Unauthorized and then hit a 429 rate limit."


class FxFailureTests(unittest.TestCase):
    def test_typed_fields_and_stderr_classify(self) -> None:
        self.assertEqual(classify_fx_failure("", {"auth_failure": {"reason": "http_unauthorized"}}),
                         "not_authenticated")
        self.assertEqual(classify_fx_failure("", {"error": "MissingCredentials"}), "not_authenticated")
        self.assertEqual(classify_fx_failure("", {"error": "RateLimitExceeded"}), "rate_limit")
        self.assertEqual(classify_fx_failure("error: 429 Too Many Requests", {}), "rate_limit")

    def test_the_answer_is_never_read(self) -> None:
        for result in ({"final_output": ANSWER}, {"assistant_output": ANSWER}, None):
            self.assertIsNone(classify_fx_failure("", result))


class HermesFailureTests(unittest.TestCase):
    def test_after_a_turn_only_stderr_is_read(self) -> None:
        ran = "Error: provider returned 503\n\nsession_id: 20261007_120000_abc"
        self.assertEqual(hermes_diagnostic(ANSWER, ran), ran)
        self.assertNotEqual(classify_hermes_failure(hermes_diagnostic(ANSWER, "\nsession_id: s1")),
                            "not_authenticated")
        self.assertEqual(classify_hermes_failure(hermes_diagnostic(
            ANSWER, "Error: HTTP 429 Too Many Requests\nsession_id: s1")), "rate_limit")

    def test_setup_failures_on_stdout_still_classify(self) -> None:
        for stdout in ("\n⚠️  No inference provider is configured.\n   Run 'hermes model'",
                       "\n⚠️  No API key found for provider 'openrouter'."):
            with self.subTest(stdout=stdout):
                self.assertEqual(classify_hermes_failure(hermes_diagnostic(stdout, "")), "not_authenticated")


if __name__ == "__main__":
    unittest.main()
