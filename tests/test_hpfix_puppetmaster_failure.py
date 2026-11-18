from __future__ import annotations

import unittest

from puppetmaster.failure import NOT_AUTHENTICATED, UNKNOWN, classify_adapter_failure


class SentenceEnd401Tests(unittest.TestCase):
    """Bug 99395a2a-1: a 401 at the end of a sentence is an auth failure."""

    def test_status_401_at_sentence_end_is_not_authenticated(self) -> None:
        for text in (
            "Request failed with status code 401.",
            "Error: request failed with status 401.",
        ):
            with self.subTest(text=text):
                self.assertEqual(classify_adapter_failure("openai", text), NOT_AUTHENTICATED)

    def test_decimal_number_with_401_stays_unknown(self) -> None:
        for text in ("elapsed 401.5 seconds", "build 2.401 failed", "value 401x"):
            with self.subTest(text=text):
                self.assertEqual(classify_adapter_failure("openai", text), UNKNOWN)


if __name__ == "__main__":
    unittest.main()
