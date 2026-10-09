"""Adapter parity: shared failure classification and live-probe verdicts."""
from __future__ import annotations

import unittest

from puppetmaster.failure import (
    BILLING_OR_QUOTA,
    MODEL_UNAVAILABLE,
    NOT_AUTHENTICATED,
    RATE_LIMIT,
    SERVER_ERROR,
    UNKNOWN,
    classify_claude_code_failure,
    classify_codex_failure,
    classify_cursor_failure,
    classify_provider_failure,
)


class ProviderFourXxTests(unittest.TestCase):
    """F54: direct-provider 4xx gets the same classes as the openai adapter."""

    def test_404_is_model_unavailable(self) -> None:
        body = '{"error":{"message":"No endpoints found for model x/y"}}'
        self.assertEqual(classify_provider_failure("http_status:404", 404, body=body), MODEL_UNAVAILABLE)

    def test_402_is_billing_or_quota(self) -> None:
        body = '{"error":{"message":"Insufficient credits"}}'
        self.assertEqual(classify_provider_failure("http_status:402", 402, body=body), BILLING_OR_QUOTA)

    def test_other_4xx_body_uses_shared_rules(self) -> None:
        body = ('{"detail":"The \'gpt-5.6-luna-pro\' model is not supported when '
                'using Codex with a ChatGPT account."}')
        self.assertEqual(classify_provider_failure("http_status:400", 400, body=body), MODEL_UNAVAILABLE)

    def test_unclassified_4xx_keeps_the_raw_reason(self) -> None:
        self.assertEqual(
            classify_provider_failure("http_status:400", 400, body='{"error":"bad field"}'),
            "http_status:400",
        )

    def test_provider_error_reroutes(self) -> None:
        from puppetmaster.providers import ProviderError
        from puppetmaster.workers import RECOVERABLE_FAILURES

        error = ProviderError("x", reason="http_status:402", status=402, body="insufficient credits")
        self.assertIn(error.failure, RECOVERABLE_FAILURES)

    def test_existing_statuses_unchanged(self) -> None:
        self.assertEqual(classify_provider_failure("http_status:401", 401), NOT_AUTHENTICATED)
        self.assertEqual(classify_provider_failure("http_status:429", 429), RATE_LIMIT)
        self.assertEqual(classify_provider_failure("http_status:503", 503), SERVER_ERROR)


class LiveProbeBlockingTests(unittest.TestCase):
    """F55: an explicit auth or rate-limit diagnosis blocks every adapter."""

    def test_logged_out_codex_blocks(self) -> None:
        from puppetmaster.preflight import live_probe

        result = live_probe("codex", "gpt-5.5", prober=lambda a, m: (1, "", "You are not logged in"))
        self.assertFalse(result.ok)
        self.assertIn("live_probe:auth", result.evidence)

    def test_rate_limited_claude_blocks_like_cursor(self) -> None:
        from puppetmaster.preflight import live_probe

        result = live_probe(
            "claude-code", "claude-opus-5-5",
            prober=lambda a, m: (1, "", "rate limit exceeded"),
        )
        self.assertFalse(result.ok)
        self.assertIn("live_probe:billing_or_quota", result.evidence)

    def test_classifier_output_is_normalized(self) -> None:
        from puppetmaster.preflight import classify_live_probe

        self.assertEqual(classify_live_probe("codex", 0, "You are not logged in"), "auth")
        self.assertEqual(classify_live_probe("openai", 1, "rate limit exceeded"), "billing_or_quota")

    def test_hermes_uses_its_own_classifier(self) -> None:
        from puppetmaster.preflight import classify_live_probe

        self.assertEqual(
            classify_live_probe("hermes", 1, "No inference provider is configured"), "auth"
        )

    def test_missing_cli_still_does_not_block(self) -> None:
        from puppetmaster.preflight import live_probe

        result = live_probe("claude-code", "m", prober=lambda a, m: (127, "", "command not found"))
        self.assertTrue(result.ok)
        self.assertIn("live_probe:skipped_unverified", result.evidence)


class NodeStackAndNumberTests(unittest.TestCase):
    """F56: Node stack frames and numbers that contain 429 are not diagnoses."""

    def test_node_stack_line_401_is_not_a_logout(self) -> None:
        text = ("TypeError: Cannot read properties of undefined (reading 'id')\n"
                "    at foo (/opt/runner/node_modules/@cursor/sdk/dist/index.js:401:17)\n"
                "    at async main (file:///opt/runner/cursor_sdk_runner.mjs:88:5)")
        self.assertEqual(classify_cursor_failure(text), UNKNOWN)
        self.assertEqual(classify_claude_code_failure(text), UNKNOWN)

    def test_number_with_429_inside_is_not_a_rate_limit(self) -> None:
        self.assertEqual(classify_codex_failure("internal error, request_id req_8a4291f"), UNKNOWN)
        self.assertEqual(classify_codex_failure("SyntaxError at line 1429"), UNKNOWN)

    def test_real_429_stays_rate_limit(self) -> None:
        for text in ("HTTP 429 Too Many Requests", "status=429", "error (429)"):
            with self.subTest(text=text):
                self.assertEqual(classify_codex_failure(text), RATE_LIMIT)

    def test_auth_text_after_node_stack_still_counts(self) -> None:
        text = ("Error: 401 Unauthorized\n"
                "    at request (/opt/runner/dist/index.js:12:3)")
        self.assertEqual(classify_cursor_failure(text), NOT_AUTHENTICATED)


class BareAuthWordTests(unittest.TestCase):
    """F61: a path or checksum word is not a credential diagnosis."""

    def test_authentication_path_is_not_a_logout(self) -> None:
        self.assertEqual(classify_codex_failure("error in src/authentication/session.ts"), UNKNOWN)

    def test_checksum_verification_is_not_a_logout(self) -> None:
        self.assertEqual(classify_codex_failure("checksum verification failed"), UNKNOWN)

    def test_explicit_diagnoses_stay_not_authenticated(self) -> None:
        for text in ("Authentication failed", "authentication error", "Invalid API key provided",
                     "Incorrect API key provided", "Missing API key", "No API key found",
                     "login verification failed", "account verification required",
                     "Invalid authentication credentials", "api key missing",
                     "Provider verification failed for anthropic"):
            with self.subTest(text=text):
                self.assertEqual(classify_codex_failure(text), NOT_AUTHENTICATED)


if __name__ == "__main__":
    unittest.main()
