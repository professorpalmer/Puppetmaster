"""Regression coverage for credential-scoped passive rate-limit admission."""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

from puppetmaster.provider_circuit import resolve_circuit_key
from puppetmaster.providers import AssistantTurn, ProviderError
from puppetmaster.rate_limit_headers import RateLimitSnapshot, RateLimitWindow
from puppetmaster.rate_limit_state import (
    RateLimitStore,
    admit_or_raise,
    get_rate_limit_store,
    reset_rate_limit_store_cache,
)


class CredentialScopedRateLimitTests(unittest.TestCase):
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

    def test_exhausted_key_a_does_not_block_key_b_and_never_persists_raw_keys(self) -> None:
        # Arrange: two distinct credentials call the same provider/model/endpoint.
        key_a_secret = "sk-test-rate-limit-a-not-for-storage"
        key_b_secret = "sk-test-rate-limit-b-not-for-storage"
        admission_a = resolve_circuit_key(
            "openai", "gpt-test", api_key=key_a_secret,
        )
        admission_b = resolve_circuit_key(
            "openai", "gpt-test", api_key=key_b_secret,
        )
        now = 1_700_000_000.0
        store = get_rate_limit_store()
        store.upsert(
            admission_key=admission_a,
            provider="openai",
            model="gpt-test",
            snapshot=RateLimitSnapshot(
                requests=RateLimitWindow(
                    kind="requests", limit=1, remaining=0, reset_at=now + 60,
                )
            ),
            now=now,
        )

        # Act / Assert: only the exhausted credential is refused.
        self.assertNotEqual(admission_a, admission_b)
        with self.assertRaises(ProviderError):
            admit_or_raise(admission_a, now=now + 1)
        admit_or_raise(admission_b, now=now + 1)

        # Raw credential material must not occur in SQLite keys or values.
        connection = sqlite3.connect(str(self.path))
        try:
            persisted = "\n".join(
                str(value)
                for row in connection.execute("SELECT * FROM rate_limits")
                for value in row
                if value is not None
            )
        finally:
            connection.close()
        self.assertNotIn(key_a_secret, persisted)
        self.assertNotIn(key_b_secret, persisted)

    def test_existing_legacy_row_is_retired_on_reopen(self) -> None:
        # Arrange: emulate a pre-credential-scoping database row.
        legacy = "openai\x1fgpt-test\x1fhttps://api.openai.com/v1"
        now = 1_700_000_000.0
        first_store = RateLimitStore(self.path)
        first_store.upsert(
            admission_key=legacy,
            provider="openai",
            model="gpt-test",
            snapshot=RateLimitSnapshot(
                requests=RateLimitWindow(
                    kind="requests", limit=1, remaining=0, reset_at=now + 60,
                )
            ),
            now=now,
        )

        # Act: a fresh process opens the same database after the schema upgrade.
        reopened = RateLimitStore(self.path)

        # Assert: legacy shared rows cannot block a credential-scoped caller and
        # are removed rather than retained forever.
        scoped = resolve_circuit_key(
            "openai", "gpt-test", api_key="sk-test-replacement",
        )
        self.assertIsNone(reopened.get(legacy))
        admit_or_raise(scoped, now=now + 1)

    def test_preemptively_blocked_key_rotates_to_next_credential(self) -> None:
        # Arrange: key A is harvested as exhausted while key B is untouched.
        from puppetmaster.adapters.agentic import AgenticAdapter
        from puppetmaster.adapters import agentic

        key_a = "sk-test-rotating-a"
        key_b = "sk-test-rotating-b"
        admission_a = resolve_circuit_key("openai", "gpt-test", api_key=key_a)
        now = 1_700_000_000.0
        get_rate_limit_store().upsert(
            admission_key=admission_a,
            provider="openai",
            model="gpt-test",
            snapshot=RateLimitSnapshot(
                requests=RateLimitWindow(
                    kind="requests", limit=1, remaining=0, reset_at=now + 60,
                )
            ),
            now=now,
        )

        # Act: admission must advance the existing rotation pool, rather than
        # fail the whole turn before provider_chat sees key B.
        with mock.patch.object(
            agentic,
            "provider_chat",
            return_value=AssistantTurn(text="key-b-worked"),
        ) as provider_chat:
            with mock.patch("puppetmaster.rate_limit_state.time.time", return_value=now + 1):
                turn = AgenticAdapter()._provider_call(
                    provider="openai",
                    model="gpt-test",
                    messages=[],
                    tools=None,
                    extra={},
                    timeout=30,
                    max_retries=0,
                    key_pool=[key_a, key_b],
                )

        # Assert: no call leaked through with exhausted key A; the deliberate
        # rotation reaches B, still without persisting either raw key.
        self.assertEqual(turn.text, "key-b-worked")
        self.assertEqual(provider_chat.call_count, 1)
        self.assertEqual(provider_chat.call_args.kwargs["api_key"], key_b)

    def test_ambient_bedrock_credentials_are_scoped_without_persisting_identity(self) -> None:
        # Arrange / Act: SigV4 callers have no api_key argument, but their
        # resolved AWS identity is still a credential boundary.
        first_access_key = "AKIAEXAMPLEONE"
        second_access_key = "AKIAEXAMPLETWO"
        first = resolve_circuit_key(
            "bedrock",
            "anthropic.claude-test",
            env={
                "AWS_ACCESS_KEY_ID": first_access_key,
                "AWS_SECRET_ACCESS_KEY": "secret-one-not-for-storage",
            },
        )
        second = resolve_circuit_key(
            "bedrock",
            "anthropic.claude-test",
            env={
                "AWS_ACCESS_KEY_ID": second_access_key,
                "AWS_SECRET_ACCESS_KEY": "secret-two-not-for-storage",
            },
        )

        # Assert: distinct ambient identities do not share admission state, and
        # the scope holds only a fingerprint (not an AWS access-key id).
        self.assertNotEqual(first, second)
        self.assertNotIn(first_access_key, first)
        self.assertNotIn(second_access_key, second)


if __name__ == "__main__":
    unittest.main()
