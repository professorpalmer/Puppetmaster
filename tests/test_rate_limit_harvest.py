"""Hermetic tests for passive rate-limit harvest + preemptive admission."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

from puppetmaster.provider_circuit import (
    ProviderCircuitBreaker,
    is_admission_blocked_error,
    reset_provider_circuit_breaker,
)
from puppetmaster.providers import ProviderError
from puppetmaster.rate_limit_headers import (
    parse_rate_limit_headers,
    reset_at,
)
from puppetmaster.rate_limit_state import (
    QUOTA_EXHAUSTED_BODY,
    admit_or_raise,
    doctor_rate_limit_summary,
    get_rate_limit_store,
    harvest_enabled,
    harvesting_rate_limits,
    is_quota_admission_error,
    record_from_headers,
    reset_rate_limit_store_cache,
)
from puppetmaster.secret_entry import (
    looks_doubled_secret,
    secret_entry_feedback,
    secret_entry_problem,
)


class RateLimitHeaderParseTests(unittest.TestCase):
    def test_openai_style_exhausted_tokens(self) -> None:
        now = 1_700_000_000.0
        snap = parse_rate_limit_headers(
            {
                "x-ratelimit-limit-tokens": "100000",
                "x-ratelimit-remaining-tokens": "0",
                "x-ratelimit-reset-tokens": "60",
                "x-ratelimit-limit-requests": "500",
                "x-ratelimit-remaining-requests": "499",
                "x-ratelimit-reset-requests": "30",
            },
            now=now,
        )
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertEqual(snap.tokens.remaining, 0)
        self.assertEqual(snap.tokens.reset_at, now + 60)
        self.assertTrue(snap.is_exhausted(now=now))
        self.assertEqual(snap.blocking_reset_at(now=now), now + 60)

    def test_anthropic_style_headers(self) -> None:
        now = 1_700_000_000.0
        snap = parse_rate_limit_headers(
            {
                "anthropic-ratelimit-requests-limit": "50",
                "anthropic-ratelimit-requests-remaining": "10",
                "anthropic-ratelimit-requests-reset": "2m30s",
            },
            now=now,
        )
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertEqual(snap.requests.remaining, 10)
        self.assertAlmostEqual(snap.requests.reset_at or 0, now + 150, places=3)
        self.assertFalse(snap.is_exhausted(now=now))

    def test_retry_after_on_429(self) -> None:
        now = 1_700_000_000.0
        snap = parse_rate_limit_headers(
            {"retry-after": "45"},
            now=now,
            http_status=429,
        )
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertEqual(snap.retry_after_at, now + 45)
        self.assertTrue(snap.is_exhausted(now=now))

    def test_reset_at_duration_and_epoch(self) -> None:
        now = 1_700_000_000.0
        self.assertEqual(reset_at("1h2m3s", now=now), now + 3723)
        self.assertEqual(reset_at("1700000060", now=now), 1_700_000_060.0)
        self.assertIsNone(reset_at(""))

    def test_empty_headers(self) -> None:
        self.assertIsNone(parse_rate_limit_headers({}))
        self.assertIsNone(parse_rate_limit_headers(None))


class RateLimitStoreAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self._tmpdir.name) / "rate_limits.sqlite3"
        os.environ["PUPPETMASTER_RATE_LIMIT_PATH"] = str(self.path)
        os.environ["PUPPETMASTER_RATE_LIMIT_HARVEST"] = "1"
        os.environ["PUPPETMASTER_RATE_LIMIT_ADMISSION"] = "1"
        reset_rate_limit_store_cache()

    def tearDown(self) -> None:
        reset_rate_limit_store_cache()
        for name in (
            "PUPPETMASTER_RATE_LIMIT_PATH",
            "PUPPETMASTER_RATE_LIMIT_HARVEST",
            "PUPPETMASTER_RATE_LIMIT_ADMISSION",
        ):
            os.environ.pop(name, None)
        self._tmpdir.cleanup()

    def test_record_and_admit(self) -> None:
        now = 1_700_000_000.0
        key = "openrouter\x1fmymodel\x1fhttps://openrouter.ai/api/v1"
        with harvesting_rate_limits(key, provider="openrouter", model="mymodel"):
            record = record_from_headers(
                {
                    "x-ratelimit-remaining-tokens": "0",
                    "x-ratelimit-reset-tokens": "120",
                    "x-ratelimit-limit-tokens": "1000",
                },
                http_status=200,
                now=now,
            )
        self.assertIsNotNone(record)
        with self.assertRaises(ProviderError) as ctx:
            admit_or_raise(key, now=now + 1)
        self.assertTrue(is_quota_admission_error(ctx.exception))
        self.assertTrue(is_admission_blocked_error(ctx.exception))
        self.assertEqual(ctx.exception.body, QUOTA_EXHAUSTED_BODY)
        # After reset, admission opens again.
        admit_or_raise(key, now=now + 200)

    def test_harvest_kill_switch(self) -> None:
        os.environ["PUPPETMASTER_RATE_LIMIT_HARVEST"] = "0"
        self.assertFalse(harvest_enabled())
        self.assertIsNone(
            record_from_headers(
                {"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "10"},
                admission_key="k",
            )
        )

    def test_fail_open_without_data(self) -> None:
        admit_or_raise("never-seen-key")

    def test_circuit_ignores_quota_admission(self) -> None:
        breaker = ProviderCircuitBreaker(enabled=True, threshold=2, cooldown=30.0)
        reset_provider_circuit_breaker(breaker)
        key = "p\x1fm\x1fhttps://example"
        err = ProviderError(
            "quota",
            reason="rate_limit",
            status=429,
            body=QUOTA_EXHAUSTED_BODY,
        )
        breaker.record_failure(key, err)
        breaker.record_failure(key, err)
        self.assertEqual(breaker.state_for(key), "closed")

    def test_doctor_summary(self) -> None:
        now = 1_700_000_000.0
        store = get_rate_limit_store()
        from puppetmaster.rate_limit_headers import RateLimitSnapshot, RateLimitWindow

        store.upsert(
            admission_key="openrouter\x1fx\x1fhttps://x",
            provider="openrouter",
            model="x",
            snapshot=RateLimitSnapshot(
                tokens=RateLimitWindow(
                    kind="tokens", limit=1, remaining=0, reset_at=now + 90
                )
            ),
            now=now,
        )
        with mock.patch("puppetmaster.rate_limit_state.time.time", return_value=now + 1):
            detail = doctor_rate_limit_summary()
        self.assertTrue(detail.startswith("exhausted"), detail)


class SecretEntryTests(unittest.TestCase):
    def test_feedback_and_doubled(self) -> None:
        self.assertEqual(secret_entry_feedback(""), "No characters were received.")
        self.assertEqual(secret_entry_feedback("abcd"), "Received 4 characters.")
        self.assertTrue(looks_doubled_secret("sk-abcdefgsk-abcdefg"))
        self.assertTrue(looks_doubled_secret("sk-abcdef sk-abcdef"))
        self.assertFalse(looks_doubled_secret("sk-abcdefghijklmnop"))
        self.assertEqual(secret_entry_problem(""), "empty")
        self.assertEqual(secret_entry_problem("abcdefghabcdefgh"), "doubled")
        self.assertIsNone(secret_entry_problem("sk-unique-key-value"))


class KeysDoubledPasteTests(unittest.TestCase):
    def test_apply_key_rejects_doubled(self) -> None:
        from puppetmaster.cli.commands_keys import _apply_key
        from puppetmaster.providers import get_provider

        desc = get_provider("openrouter")
        self.assertIsNotNone(desc)
        assert desc is not None
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mcp.json"
            ok, message = _apply_key(desc, "sk-abcdefghsk-abcdefgh", path)
            self.assertFalse(ok)
            self.assertIn("pasted twice", message)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
