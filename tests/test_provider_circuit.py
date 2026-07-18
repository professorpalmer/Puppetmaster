"""Deterministic tests for process-local provider circuit-breaker admission."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import os
import unittest
from unittest import mock

from puppetmaster.provider_circuit import (
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_FAILURE_THRESHOLD,
    ProviderCircuitBreaker,
    admission_blocked_error,
    circuit_enabled,
    circuit_key,
    cooldown_seconds,
    failure_threshold,
    get_provider_circuit_breaker,
    is_admission_blocked_error,
    reset_provider_circuit_breaker,
    resolve_circuit_key,
)
from puppetmaster.providers import AssistantTurn, ProviderError

class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

def _retryable(message: str = "boom") -> ProviderError:
    return ProviderError(message, reason="timeout")

def _server_error() -> ProviderError:
    return ProviderError("unavailable", reason="http_status:503", status=503)

def _rate_limit() -> ProviderError:
    return ProviderError("slow down", reason="http_status:429", status=429)

def _auth_error() -> ProviderError:
    return ProviderError("401", reason="http_status:401", status=401, body="bad key")

def _malformed() -> ProviderError:
    return ProviderError("malformed response", reason="malformed_response", body="<html>")

class CircuitConfigTests(unittest.TestCase):
    def tearDown(self) -> None:
        for name in (
            "PUPPETMASTER_PROVIDER_CIRCUIT",
            "PUPPETMASTER_PROVIDER_CIRCUIT_THRESHOLD",
            "PUPPETMASTER_PROVIDER_CIRCUIT_COOLDOWN_SECONDS",
            "PUPPETMASTER_PROVIDER_CIRCUIT_MAX_BUCKETS",
            "PUPPETMASTER_PROVIDER_CIRCUIT_CLOSED_TTL_SECONDS",
        ):
            os.environ.pop(name, None)

    def test_defaults_and_kill_switch(self) -> None:
        from puppetmaster.provider_circuit import (
            DEFAULT_CLOSED_BUCKET_TTL_SECONDS,
            DEFAULT_MAX_BUCKETS,
            closed_bucket_ttl_seconds,
            max_buckets,
        )

        self.assertTrue(circuit_enabled())
        self.assertEqual(failure_threshold(), DEFAULT_FAILURE_THRESHOLD)
        self.assertEqual(cooldown_seconds(), DEFAULT_COOLDOWN_SECONDS)
        self.assertEqual(max_buckets(), DEFAULT_MAX_BUCKETS)
        self.assertEqual(closed_bucket_ttl_seconds(), DEFAULT_CLOSED_BUCKET_TTL_SECONDS)
        os.environ["PUPPETMASTER_PROVIDER_CIRCUIT"] = "0"
        self.assertFalse(circuit_enabled())
        os.environ["PUPPETMASTER_PROVIDER_CIRCUIT"] = "off"
        self.assertFalse(circuit_enabled())

    def test_threshold_and_cooldown_clamped(self) -> None:
        from puppetmaster.provider_circuit import (
            closed_bucket_ttl_seconds,
            max_buckets,
        )

        os.environ["PUPPETMASTER_PROVIDER_CIRCUIT_THRESHOLD"] = "0"
        self.assertEqual(failure_threshold(), 1)
        os.environ["PUPPETMASTER_PROVIDER_CIRCUIT_THRESHOLD"] = "999"
        self.assertEqual(failure_threshold(), 100)
        os.environ["PUPPETMASTER_PROVIDER_CIRCUIT_THRESHOLD"] = "nope"
        self.assertEqual(failure_threshold(), DEFAULT_FAILURE_THRESHOLD)
        os.environ["PUPPETMASTER_PROVIDER_CIRCUIT_COOLDOWN_SECONDS"] = "0.1"
        self.assertEqual(cooldown_seconds(), 1.0)
        os.environ["PUPPETMASTER_PROVIDER_CIRCUIT_COOLDOWN_SECONDS"] = "9999"
        self.assertEqual(cooldown_seconds(), 600.0)
        os.environ["PUPPETMASTER_PROVIDER_CIRCUIT_MAX_BUCKETS"] = "0"
        self.assertEqual(max_buckets(), 1)
        os.environ["PUPPETMASTER_PROVIDER_CIRCUIT_MAX_BUCKETS"] = "999999"
        self.assertEqual(max_buckets(), 10_000)
        os.environ["PUPPETMASTER_PROVIDER_CIRCUIT_CLOSED_TTL_SECONDS"] = "0.1"
        self.assertEqual(closed_bucket_ttl_seconds(), 1.0)
        os.environ["PUPPETMASTER_PROVIDER_CIRCUIT_CLOSED_TTL_SECONDS"] = "999999"
        self.assertEqual(closed_bucket_ttl_seconds(), 86_400.0)

class ProviderCircuitBreakerUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = _FakeClock()
        self.breaker = ProviderCircuitBreaker(
            clock=self.clock, threshold=3, cooldown=10.0, enabled=True,
        )
        self.key = circuit_key("anthropic", "claude", base_url="https://api.anthropic.com")

    def test_closed_by_default_and_opens_after_threshold(self) -> None:
        self.assertEqual(self.breaker.state_for(self.key), "closed")
        self.breaker.before_call(self.key)
        self.breaker.record_failure(self.key, _retryable())
        self.breaker.before_call(self.key)
        self.breaker.record_failure(self.key, _server_error())
        self.assertEqual(self.breaker.state_for(self.key), "closed")
        self.breaker.before_call(self.key)
        self.breaker.record_failure(self.key, _rate_limit())
        self.assertEqual(self.breaker.state_for(self.key), "open")
        with self.assertRaises(ProviderError) as raised:
            self.breaker.before_call(self.key)
        self.assertTrue(is_admission_blocked_error(raised.exception))
        self.assertEqual(raised.exception.failure, "rate_limit")

    def test_half_open_probe_then_recovery(self) -> None:
        for _ in range(3):
            self.breaker.before_call(self.key)
            self.breaker.record_failure(self.key, _retryable())
        self.assertEqual(self.breaker.state_for(self.key), "open")
        self.clock.advance(10.0)
        # First caller becomes the half-open probe.
        self.breaker.before_call(self.key)
        self.assertEqual(self.breaker.state_for(self.key), "half_open")
        # Concurrent callers blocked while probe is in flight.
        with self.assertRaises(ProviderError):
            self.breaker.before_call(self.key)
        self.breaker.record_success(self.key)
        self.assertEqual(self.breaker.state_for(self.key), "closed")
        self.breaker.before_call(self.key)  # admitted again

    def test_half_open_qualifying_failure_reopens(self) -> None:
        for _ in range(3):
            self.breaker.before_call(self.key)
            self.breaker.record_failure(self.key, _retryable())
        self.clock.advance(10.0)
        self.breaker.before_call(self.key)
        self.breaker.record_failure(self.key, _server_error())
        self.assertEqual(self.breaker.state_for(self.key), "open")
        with self.assertRaises(ProviderError):
            self.breaker.before_call(self.key)
        # Still inside cooldown — stays open.
        self.clock.advance(5.0)
        with self.assertRaises(ProviderError):
            self.breaker.before_call(self.key)

    def test_non_qualifying_errors_do_not_trip(self) -> None:
        for err in (_auth_error(), _malformed()):
            self.breaker.before_call(self.key)
            self.breaker.record_failure(self.key, err)
        self.assertEqual(self.breaker.state_for(self.key), "closed")
        # Mix with qualifying — only qualifying counts.
        self.breaker.record_failure(self.key, _retryable())
        self.breaker.record_failure(self.key, _auth_error())
        self.breaker.record_failure(self.key, _retryable())
        self.assertEqual(self.breaker.state_for(self.key), "closed")
        self.breaker.record_failure(self.key, _retryable())
        self.assertEqual(self.breaker.state_for(self.key), "open")

    def test_half_open_non_qualifying_closes(self) -> None:
        for _ in range(3):
            self.breaker.before_call(self.key)
            self.breaker.record_failure(self.key, _retryable())
        self.clock.advance(10.0)
        self.breaker.before_call(self.key)
        self.breaker.record_failure(self.key, _auth_error())
        self.assertEqual(self.breaker.state_for(self.key), "closed")

    def test_disabled_never_blocks_or_trips(self) -> None:
        breaker = ProviderCircuitBreaker(
            clock=self.clock, threshold=1, cooldown=10.0, enabled=False,
        )
        for _ in range(5):
            breaker.before_call(self.key)
            breaker.record_failure(self.key, _retryable())
        self.assertEqual(breaker.state_for(self.key), "closed")
        breaker.before_call(self.key)

    def test_key_isolation(self) -> None:
        other = circuit_key("openai", "gpt", base_url="https://api.openai.com")
        for _ in range(3):
            self.breaker.before_call(self.key)
            self.breaker.record_failure(self.key, _retryable())
        self.assertEqual(self.breaker.state_for(self.key), "open")
        self.assertEqual(self.breaker.state_for(other), "closed")
        self.breaker.before_call(other)
        self.breaker.record_success(other)

    def test_success_resets_consecutive_failures(self) -> None:
        self.breaker.record_failure(self.key, _retryable())
        self.breaker.record_failure(self.key, _retryable())
        self.breaker.record_success(self.key)
        self.breaker.record_failure(self.key, _retryable())
        self.breaker.record_failure(self.key, _retryable())
        self.assertEqual(self.breaker.state_for(self.key), "closed")
        self.breaker.record_failure(self.key, _retryable())
        self.assertEqual(self.breaker.state_for(self.key), "open")

    def test_admission_error_is_recoverable_and_not_self_tripping(self) -> None:
        err = admission_blocked_error(self.key)
        self.assertTrue(is_admission_blocked_error(err))
        from puppetmaster.providers import is_retryable_provider_error

        self.assertTrue(is_retryable_provider_error(err))
        self.breaker.record_failure(self.key, err)
        self.assertEqual(self.breaker.state_for(self.key), "closed")

    def test_release_admission_unblocks_abandoned_half_open_probe(self) -> None:
        for _ in range(3):
            self.breaker.before_call(self.key)
            self.breaker.record_failure(self.key, _retryable())
        self.clock.advance(10.0)
        self.breaker.before_call(self.key)
        self.assertEqual(self.breaker.state_for(self.key), "half_open")
        with self.assertRaises(ProviderError):
            self.breaker.before_call(self.key)
        self.breaker.release_admission(self.key)
        self.assertEqual(self.breaker.state_for(self.key), "open")
        # Cooldown already elapsed — next before_call becomes a new probe.
        self.breaker.before_call(self.key)
        self.assertEqual(self.breaker.state_for(self.key), "half_open")

    def test_closed_stale_buckets_evicted_by_ttl(self) -> None:
        breaker = ProviderCircuitBreaker(
            clock=self.clock,
            threshold=3,
            cooldown=10.0,
            enabled=True,
            max_buckets=64,
            closed_ttl=30.0,
        )
        keys = [
            circuit_key("p", f"m{i}", base_url=f"https://h{i}.example")
            for i in range(5)
        ]
        for key in keys:
            breaker.before_call(key)
            breaker.record_success(key)
        self.assertEqual(breaker.bucket_count(), 5)
        self.clock.advance(30.0)
        # Touching a fresh key triggers stale closed eviction.
        fresh = circuit_key("p", "fresh", base_url="https://fresh.example")
        breaker.before_call(fresh)
        self.assertEqual(breaker.bucket_count(), 1)
        self.assertEqual(breaker.state_for(fresh), "closed")
        # Previously retained closed keys are gone (still report closed).
        self.assertEqual(breaker.state_for(keys[0]), "closed")
        self.assertEqual(breaker.bucket_count(), 1)

    def test_hard_cap_evicts_oldest_closed_preserves_open(self) -> None:
        breaker = ProviderCircuitBreaker(
            clock=self.clock,
            threshold=1,
            cooldown=10.0,
            enabled=True,
            max_buckets=3,
            closed_ttl=10_000.0,
        )
        closed_keys = [
            circuit_key("p", f"c{i}", base_url=f"https://c{i}.example")
            for i in range(2)
        ]
        for key in closed_keys:
            breaker.before_call(key)
            breaker.record_success(key)
            self.clock.advance(1.0)
        open_key = circuit_key("p", "open", base_url="https://open.example")
        breaker.before_call(open_key)
        breaker.record_failure(open_key, _retryable())
        self.assertEqual(breaker.state_for(open_key), "open")
        self.clock.advance(1.0)
        # Inserting a fourth distinct key must evict an oldest closed bucket,
        # never the open breaker that still gates admission.
        extra = circuit_key("p", "extra", base_url="https://extra.example")
        breaker.before_call(extra)
        self.assertLessEqual(breaker.bucket_count(), 3)
        self.assertEqual(breaker.state_for(open_key), "open")
        with self.assertRaises(ProviderError):
            breaker.before_call(open_key)
        # Key isolation: a different closed key is unaffected in behavior.
        other = circuit_key("openai", "gpt", base_url="https://api.openai.com")
        breaker.before_call(other)
        self.assertEqual(breaker.state_for(other), "closed")

    def test_half_open_survives_cap_pressure(self) -> None:
        breaker = ProviderCircuitBreaker(
            clock=self.clock,
            threshold=1,
            cooldown=5.0,
            enabled=True,
            max_buckets=2,
            closed_ttl=10_000.0,
        )
        hot = circuit_key("p", "hot", base_url="https://hot.example")
        breaker.before_call(hot)
        breaker.record_failure(hot, _retryable())
        self.clock.advance(5.0)
        breaker.before_call(hot)
        self.assertEqual(breaker.state_for(hot), "half_open")
        # Flood with new keys while probe is in flight.
        for i in range(5):
            key = circuit_key("p", f"flood{i}", base_url=f"https://f{i}.example")
            breaker.before_call(key)
            breaker.record_success(key)
            self.clock.advance(0.1)
        self.assertEqual(breaker.state_for(hot), "half_open")
        with self.assertRaises(ProviderError):
            breaker.before_call(hot)
        breaker.record_success(hot)
        self.assertEqual(breaker.state_for(hot), "closed")

class CircuitKeyTests(unittest.TestCase):
    def test_resolve_includes_provider_base_url(self) -> None:
        key = resolve_circuit_key("anthropic", "claude-sonnet")
        self.assertIn("anthropic", key)
        self.assertIn("claude-sonnet", key)
        self.assertIn("api.anthropic.com", key)

    def test_singleton_reset(self) -> None:
        a = get_provider_circuit_breaker()
        b = reset_provider_circuit_breaker()
        self.assertIsNot(a, b)
        self.assertIs(get_provider_circuit_breaker(), b)

class ProviderCallCircuitIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = _FakeClock()
        self.breaker = ProviderCircuitBreaker(
            clock=self.clock, threshold=2, cooldown=5.0, enabled=True,
        )
        reset_provider_circuit_breaker(self.breaker)

    def tearDown(self) -> None:
        reset_provider_circuit_breaker()
        os.environ.pop("PUPPETMASTER_PROVIDER_CIRCUIT", None)

    def _adapter(self):
        from puppetmaster.adapters.agentic import AgenticAdapter

        return AgenticAdapter()

    def test_provider_call_opens_and_blocks_then_recovers(self) -> None:
        from puppetmaster.adapters import agentic

        calls = {"n": 0}

        def flaky(*, provider, model, messages, tools, extra, timeout):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise ProviderError("timed out", reason="timeout")
            return AssistantTurn(text="ok")

        with mock.patch.object(agentic, "provider_chat", side_effect=flaky), \
                mock.patch("time.sleep", lambda *_: None):
            with self.assertRaises(ProviderError) as first:
                self._adapter()._provider_call(
                    provider="anthropic",
                    model="m",
                    messages=[],
                    tools=None,
                    extra={},
                    timeout=30,
                    max_retries=0,
                )
            self.assertEqual(first.exception.reason, "timeout")

            with self.assertRaises(ProviderError) as second:
                self._adapter()._provider_call(
                    provider="anthropic",
                    model="m",
                    messages=[],
                    tools=None,
                    extra={},
                    timeout=30,
                    max_retries=0,
                )
            self.assertEqual(second.exception.reason, "timeout")

            # Circuit open — admission blocks before dialing.
            with self.assertRaises(ProviderError) as blocked:
                self._adapter()._provider_call(
                    provider="anthropic",
                    model="m",
                    messages=[],
                    tools=None,
                    extra={},
                    timeout=30,
                    max_retries=0,
                )
            self.assertTrue(is_admission_blocked_error(blocked.exception))
            self.assertEqual(calls["n"], 2)

            self.clock.advance(5.0)
            turn = self._adapter()._provider_call(
                provider="anthropic",
                model="m",
                messages=[],
                tools=None,
                extra={},
                timeout=30,
                max_retries=0,
            )
        self.assertEqual(turn.text, "ok")
        self.assertEqual(calls["n"], 3)
        self.assertEqual(self.breaker.state_for(resolve_circuit_key("anthropic", "m")), "closed")

    def test_provider_call_auth_does_not_open_circuit(self) -> None:
        from puppetmaster.adapters import agentic

        def auth_fail(*, provider, model, messages, tools, extra, timeout):
            raise ProviderError("401", reason="http_status:401", status=401)

        with mock.patch.object(agentic, "provider_chat", side_effect=auth_fail), \
                mock.patch("time.sleep", lambda *_: None):
            for _ in range(5):
                with self.assertRaises(ProviderError):
                    self._adapter()._provider_call(
                        provider="anthropic",
                        model="m-auth",
                        messages=[],
                        tools=None,
                        extra={},
                        timeout=30,
                        max_retries=2,
                    )
        self.assertEqual(
            self.breaker.state_for(resolve_circuit_key("anthropic", "m-auth")),
            "closed",
        )

    def test_provider_call_respects_kill_switch(self) -> None:
        from puppetmaster.adapters import agentic

        os.environ["PUPPETMASTER_PROVIDER_CIRCUIT"] = "0"
        # Rebuild singleton so enabled reads env (override not set).
        reset_provider_circuit_breaker(ProviderCircuitBreaker(clock=self.clock, threshold=1))
        calls = {"n": 0}

        def always_fail(*, provider, model, messages, tools, extra, timeout):
            calls["n"] += 1
            raise ProviderError("timed out", reason="timeout")

        with mock.patch.object(agentic, "provider_chat", side_effect=always_fail), \
                mock.patch("time.sleep", lambda *_: None):
            for _ in range(3):
                with self.assertRaises(ProviderError) as raised:
                    self._adapter()._provider_call(
                        provider="anthropic",
                        model="m-off",
                        messages=[],
                        tools=None,
                        extra={},
                        timeout=30,
                        max_retries=0,
                    )
                self.assertEqual(raised.exception.reason, "timeout")
        self.assertEqual(calls["n"], 3)

if __name__ == "__main__":
    unittest.main()
