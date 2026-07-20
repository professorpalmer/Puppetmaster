"""Hermetic tests for Bedrock invoke-health gating (stale routing fix)."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401


class BedrockInvokeHealthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="pm-bedrock-health-")
        self.db = Path(self.tmp) / "provider_health.sqlite3"
        self.env_patch = mock.patch.dict(
            os.environ,
            {"PUPPETMASTER_PROVIDER_HEALTH_PATH": str(self.db)},
            clear=False,
        )
        self.env_patch.start()
        from puppetmaster.provider_health import reset_provider_health_store_cache

        reset_provider_health_store_cache()
        self.aws_env = {
            "AWS_ACCESS_KEY_ID": "AKIATESTEXAMPLE0001",
            "AWS_SECRET_ACCESS_KEY": "secret-material-must-not-persist",
            "AWS_REGION": "us-east-1",
            "HOME": self.tmp,
            "USERPROFILE": self.tmp,
        }

    def tearDown(self) -> None:
        self.env_patch.stop()
        from puppetmaster.provider_health import reset_provider_health_store_cache

        reset_provider_health_store_cache()
        try:
            import shutil

            shutil.rmtree(self.tmp, ignore_errors=True)
        except Exception:
            pass

    def _store(self):
        from puppetmaster.provider_health import ProviderHealthStore

        return ProviderHealthStore(self.db)

    def test_stale_default_profile_presence_alone_not_auto_routable(self) -> None:
        from puppetmaster import providers
        from puppetmaster.bedrock import BedrockCredentials

        aws = Path(self.tmp) / ".aws"
        aws.mkdir(parents=True)
        (aws / "credentials").write_text(
            "[default]\naws_access_key_id = AKIASTALE0000000001\n"
            "aws_secret_access_key = stale-secret\n",
            encoding="utf-8",
        )
        env = {"HOME": self.tmp, "USERPROFILE": self.tmp, "AWS_REGION": "us-east-1"}
        desc = providers.get_provider("bedrock")
        self.assertTrue(providers.credentials_present(desc, env))
        self.assertFalse(providers.is_available(desc, env))
        self.assertNotIn("bedrock", providers.available_providers(env))
        # Fingerprint helper never needs the secret.
        from puppetmaster.provider_health import fingerprint_bedrock_credentials

        fp = fingerprint_bedrock_credentials(
            BedrockCredentials(
                kind="access_key",
                access_key_id="AKIASTALE0000000001",
                secret_access_key="stale-secret",
                profile="default",
            )
        )
        self.assertNotIn("stale-secret", fp)
        self.assertNotIn("AKIASTALE", fp)

    def test_successful_invoke_marks_verified_and_becomes_routable(self) -> None:
        from puppetmaster import bedrock, providers

        canned = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "ok"}],
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }
        with mock.patch.object(bedrock, "_post_bedrock", return_value=canned):
            turn = providers.provider_chat(
                provider="bedrock",
                model="amazon.nova-micro-v1:0",
                messages=[{"role": "user", "content": "hi"}],
                env=self.aws_env,
            )
        self.assertEqual(turn.text, "ok")
        desc = providers.get_provider("bedrock")
        self.assertTrue(providers.is_available(desc, self.aws_env))
        self.assertIn("bedrock", providers.available_providers(self.aws_env))

    def test_auth_denial_persists_across_fresh_store_and_drops_router(self) -> None:
        from puppetmaster import bedrock, providers
        from puppetmaster.model_registry import ModelSpec
        from puppetmaster.provider_health import (
            ProviderHealthStore,
            assert_no_secrets_in_health_state,
            reset_provider_health_store_cache,
        )
        from puppetmaster.providers import ProviderError
        from puppetmaster.router import NoEligibleModelError, TaskSignals, route_task

        def _raise(*_a, **_k):
            raise ProviderError(
                "HTTP 401",
                reason="http_status:401",
                status=401,
                body='{"message":"UnrecognizedClientException"}',
            )

        with mock.patch.object(bedrock, "_post_bedrock", side_effect=_raise):
            with self.assertRaises(ProviderError):
                providers.provider_chat(
                    provider="bedrock",
                    model="amazon.nova-micro-v1:0",
                    messages=[{"role": "user", "content": "hi"}],
                    env=self.aws_env,
                )

        reset_provider_health_store_cache()
        fresh = ProviderHealthStore(self.db)
        self.assertFalse(
            providers.is_available(providers.get_provider("bedrock"), self.aws_env)
        )
        from puppetmaster.provider_health import read_bedrock_invoke_health

        self.assertEqual(
            read_bedrock_invoke_health(self.aws_env, store=fresh), "denied"
        )
        assert_no_secrets_in_health_state(fresh)

        specs = [
            ModelSpec(
                id="agentic/nova",
                adapter="agentic",
                adapter_model_name="amazon.nova-micro-v1:0",
                payload_defaults={"provider": "bedrock"},
                billing="api",
                capability_score=70,
            )
        ]
        with mock.patch(
            "puppetmaster.providers.available_providers",
            return_value=set(),
        ):
            with self.assertRaises(NoEligibleModelError):
                route_task(
                    TaskSignals(instruction="explore", role="explore"),
                    specs,
                    policy="balanced",
                )

    def test_changed_fingerprint_bypasses_old_denial_but_stays_unverified(self) -> None:
        from puppetmaster import bedrock, providers
        from puppetmaster.provider_health import read_bedrock_invoke_health
        from puppetmaster.providers import ProviderError

        def _raise(*_a, **_k):
            raise ProviderError(
                "HTTP 401",
                reason="http_status:401",
                status=401,
                body="unauthorized",
            )

        with mock.patch.object(bedrock, "_post_bedrock", side_effect=_raise):
            with self.assertRaises(ProviderError):
                providers.provider_chat(
                    provider="bedrock",
                    model="amazon.nova-micro-v1:0",
                    messages=[{"role": "user", "content": "hi"}],
                    env=self.aws_env,
                )
        self.assertEqual(read_bedrock_invoke_health(self.aws_env), "denied")

        other = dict(self.aws_env)
        other["AWS_ACCESS_KEY_ID"] = "AKIATESTEXAMPLE0002"
        self.assertEqual(read_bedrock_invoke_health(other), "unknown")
        self.assertFalse(
            providers.is_available(providers.get_provider("bedrock"), other)
        )

    def test_transient_failure_does_not_poison_health(self) -> None:
        from puppetmaster import bedrock, providers
        from puppetmaster.provider_health import (
            STATUS_VERIFIED,
            read_bedrock_invoke_health,
            record_bedrock_invoke_success,
        )
        from puppetmaster.providers import ProviderError

        creds = bedrock.resolve_bedrock_credentials(self.aws_env)
        record_bedrock_invoke_success(self.aws_env, creds=creds)
        self.assertEqual(read_bedrock_invoke_health(self.aws_env), STATUS_VERIFIED)

        def _raise(*_a, **_k):
            raise ProviderError(
                "HTTP 503",
                reason="http_status:503",
                status=503,
                body="slow down",
            )

        with mock.patch.object(bedrock, "_post_bedrock", side_effect=_raise):
            with self.assertRaises(ProviderError):
                providers.provider_chat(
                    provider="bedrock",
                    model="amazon.nova-micro-v1:0",
                    messages=[{"role": "user", "content": "hi"}],
                    env=self.aws_env,
                )
        self.assertEqual(read_bedrock_invoke_health(self.aws_env), STATUS_VERIFIED)

    def test_generic_proxy_403_forbidden_does_not_poison_health(self) -> None:
        from puppetmaster import bedrock
        from puppetmaster.provider_health import (
            STATUS_UNKNOWN,
            is_terminal_bedrock_auth_failure,
            read_bedrock_invoke_health,
            record_bedrock_invoke_failure,
        )
        from puppetmaster.providers import ProviderError

        error = ProviderError(
            "HTTP 403",
            reason="http_status:403",
            status=403,
            body="<html><title>403 Forbidden</title></html>",
        )
        self.assertFalse(is_terminal_bedrock_auth_failure(error))
        creds = bedrock.resolve_bedrock_credentials(self.aws_env)
        self.assertIsNone(
            record_bedrock_invoke_failure(
                self.aws_env,
                creds=creds,
                error=error,
            )
        )
        self.assertEqual(
            read_bedrock_invoke_health(self.aws_env),
            STATUS_UNKNOWN,
        )

    def test_stream_access_denied_maps_and_persists_denied_health(self) -> None:
        from puppetmaster import bedrock
        from puppetmaster.provider_health import (
            STATUS_DENIED,
            read_bedrock_invoke_health,
        )
        from puppetmaster.providers import ProviderError

        headers = {
            ":message-type": "exception",
            ":exception-type": "accessDeniedException",
        }
        payload = b'{"message":"AccessDeniedException: not authorized"}'
        with mock.patch.object(
            bedrock,
            "_iter_eventstream_messages",
            return_value=iter([(headers, payload)]),
        ):
            with self.assertRaises(ProviderError) as raised:
                list(bedrock._iter_converse_stream_events(object()))
        error = raised.exception
        self.assertEqual(error.status, 403)
        self.assertEqual(error.reason, "http_status:403")

        response = mock.MagicMock()
        with mock.patch.object(
            bedrock,
            "_open_bedrock_event_stream",
            return_value=response,
        ), mock.patch.object(
            bedrock,
            "_iter_converse_stream_events",
            side_effect=error,
        ):
            with self.assertRaises(ProviderError):
                bedrock.bedrock_chat_stream(
                    model="amazon.nova-micro-v1:0",
                    messages=[{"role": "user", "content": "hi"}],
                    env=self.aws_env,
                )
        response.close.assert_called_once()
        self.assertEqual(
            read_bedrock_invoke_health(self.aws_env),
            STATUS_DENIED,
        )

    def test_discovery_list_without_verification_cannot_reenable(self) -> None:
        from puppetmaster.bedrock import merge_bedrock_discovered_into_registry
        from puppetmaster.model_registry import ModelSpec

        existing = [
            ModelSpec(
                id="agentic/old",
                adapter="agentic",
                adapter_model_name="amazon.nova-micro-v1:0",
                payload_defaults={"provider": "bedrock"},
                billing="api",
                enabled=False,
            )
        ]
        with mock.patch(
            "puppetmaster.bedrock.list_chat_model_ids",
            return_value=["amazon.nova-micro-v1:0", "deepseek.v3.2"],
        ):
            merged, report = merge_bedrock_discovered_into_registry(
                existing, env=self.aws_env
            )
        self.assertFalse(report["available"])
        self.assertTrue(report["credentials_present"])
        self.assertEqual(report["invoke_health"], "unknown")
        self.assertEqual(report["added"], 0)
        self.assertEqual(len(merged), 1)
        self.assertFalse(merged[0].enabled)

    def test_probe_success_and_failure_report(self) -> None:
        from puppetmaster import bedrock
        from puppetmaster.provider_health import (
            probe_bedrock_runtime,
            reset_provider_health_store_cache,
        )
        from puppetmaster.providers import ProviderError

        canned = {
            "output": {
                "message": {"role": "assistant", "content": [{"text": "p"}]}
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }
        with mock.patch.object(
            bedrock, "list_chat_model_ids", return_value=["amazon.nova-micro-v1:0"]
        ), mock.patch.object(bedrock, "_post_bedrock", return_value=canned):
            ok = probe_bedrock_runtime(env=self.aws_env)
        self.assertTrue(ok["ok"])
        self.assertEqual(ok["invoke_health"], "verified")

        # Separate DB for the denial case (Windows holds WAL locks on open DBs).
        deny_db = Path(self.tmp) / "provider_health_deny.sqlite3"
        with mock.patch.dict(
            os.environ,
            {"PUPPETMASTER_PROVIDER_HEALTH_PATH": str(deny_db)},
            clear=False,
        ):
            reset_provider_health_store_cache()

            def _raise(*_a, **_k):
                raise ProviderError(
                    "HTTP 403",
                    reason="http_status:403",
                    status=403,
                    body='{"message":"AccessDeniedException: not authorized"}',
                )

            with mock.patch.object(
                bedrock, "list_chat_model_ids", return_value=["amazon.nova-micro-v1:0"]
            ), mock.patch.object(bedrock, "_post_bedrock", side_effect=_raise):
                bad = probe_bedrock_runtime(env=self.aws_env)
            self.assertFalse(bad["ok"])
            self.assertEqual(bad["invoke_health"], "denied")
            reset_provider_health_store_cache()

    def test_models_discover_json_emits_post_probe_health(self) -> None:
        from puppetmaster import cli

        args = SimpleNamespace(
            source="agentic",
            json=True,
            write=False,
            probe=True,
            prune=False,
            registry_path=None,
        )
        report = {
            "source": "agentic",
            "adapter": "agentic",
            "discovered_count": 1,
            "added": [],
            "bedrock": {
                "credentials_present": True,
                "invoke_health": "unknown",
                "available": False,
            },
        }
        probe = {
            "probed": True,
            "ok": True,
            "model_id": "amazon.nova-micro-v1:0",
            "invoke_health": "verified",
        }
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            cli,
            "_discover_one_source",
            return_value=([], report, [{"id": "gpt-5"}]),
        ), mock.patch(
            "puppetmaster.bedrock.bedrock_credentials_present",
            return_value=True,
        ), mock.patch(
            "puppetmaster.provider_health.probe_bedrock_runtime",
            return_value=probe,
        ), contextlib.redirect_stdout(out):
            rc = cli._run_models_discover(args, Path(tmp) / "models.json")

        self.assertEqual(rc, 0)
        payload = json.loads(out.getvalue())
        bedrock = payload["reports"][0]["bedrock"]
        self.assertTrue(bedrock["available"])
        self.assertEqual(bedrock["invoke_health"], "verified")
        self.assertEqual(bedrock["probe"]["model_id"], "amazon.nova-micro-v1:0")

    def test_ttl_expiry_returns_unknown(self) -> None:
        from puppetmaster.bedrock import resolve_bedrock_credentials
        from puppetmaster.provider_health import (
            STATUS_UNKNOWN,
            STATUS_VERIFIED,
            fingerprint_bedrock_credentials,
            read_bedrock_invoke_health,
        )

        store = self._store()
        creds = resolve_bedrock_credentials(self.aws_env)
        store.mark_verified(
            provider="bedrock",
            fingerprint=fingerprint_bedrock_credentials(creds),
            region="us-east-1",
            now=1.0,
        )
        # Force expiry by rewriting expires_at in the past via upsert with tiny TTL.
        store.upsert(
            provider="bedrock",
            fingerprint=fingerprint_bedrock_credentials(creds),
            region="us-east-1",
            status=STATUS_VERIFIED,
            ttl_seconds=1,
            now=1.0,
        )
        # expires_at = 2.0; current time is far in the future
        status = read_bedrock_invoke_health(self.aws_env, store=store)
        self.assertEqual(status, STATUS_UNKNOWN)

    def test_no_secrets_in_state(self) -> None:
        from puppetmaster import bedrock, providers
        from puppetmaster.provider_health import assert_no_secrets_in_health_state

        canned = {
            "output": {
                "message": {"role": "assistant", "content": [{"text": "ok"}]}
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }
        with mock.patch.object(bedrock, "_post_bedrock", return_value=canned):
            providers.provider_chat(
                provider="bedrock",
                model="amazon.nova-micro-v1:0",
                messages=[{"role": "user", "content": "hi"}],
                env=self.aws_env,
            )
        assert_no_secrets_in_health_state(self._store())
        raw = self.db.read_bytes()
        self.assertNotIn(b"secret-material-must-not-persist", raw)
        self.assertNotIn(b"AKIATESTEXAMPLE0001", raw)

    def test_direct_pinned_bedrock_works_without_prior_verification(self) -> None:
        from puppetmaster import bedrock, providers

        canned = {
            "output": {
                "message": {"role": "assistant", "content": [{"text": "direct"}]}
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }
        desc = providers.get_provider("bedrock")
        self.assertFalse(providers.is_available(desc, self.aws_env))
        with mock.patch.object(bedrock, "_post_bedrock", return_value=canned):
            turn = providers.provider_chat(
                provider="bedrock",
                model="amazon.nova-micro-v1:0",
                messages=[{"role": "user", "content": "hi"}],
                env=self.aws_env,
            )
        self.assertEqual(turn.text, "direct")
        # Success of the direct call itself verifies health for subsequent routing.
        self.assertTrue(providers.is_available(desc, self.aws_env))

    def test_auth_403_maps_to_recoverable_not_authenticated(self) -> None:
        from puppetmaster.provider_health import bedrock_failure_for_recovery
        from puppetmaster.providers import ProviderError
        from puppetmaster.workers import RECOVERABLE_FAILURES

        err = ProviderError(
            "HTTP 403",
            reason="http_status:403",
            status=403,
            body="AccessDeniedException: User is not authorized",
        )
        failure = bedrock_failure_for_recovery(err)
        self.assertEqual(failure, "not_authenticated")
        self.assertIn(failure, RECOVERABLE_FAILURES)

    def test_cross_provider_reroute_after_bedrock_auth_failure(self) -> None:
        from puppetmaster.model_registry import ModelSpec
        from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
        from puppetmaster.orchestrator import Orchestrator
        from puppetmaster.platform_billing import BillingStatus
        from puppetmaster.store import SwarmStore

        openai_spec = ModelSpec(
            id="agentic/gpt",
            adapter="agentic",
            adapter_model_name="gpt-4o-mini",
            payload_defaults={"provider": "openai"},
            billing="api",
            capability_score=80,
            enabled=True,
        )
        bedrock_spec = ModelSpec(
            id="agentic/nova",
            adapter="agentic",
            adapter_model_name="amazon.nova-micro-v1:0",
            payload_defaults={"provider": "bedrock"},
            billing="api",
            capability_score=70,
            enabled=True,
        )

        def _billing(adapter, **_kwargs):
            return BillingStatus(
                adapter=adapter,
                billing="api",
                healthy=True,
                detail="ok",
                evidence=[],
            )

        store = SwarmStore(Path(self.tmp) / ".puppetmaster")
        job = store.create_job("bedrock auth recovery")
        task = Task(
            job_id=job.id,
            role="explore",
            instruction="go",
            adapter="agentic",
            status=TaskStatus.FAILED,
            payload={
                "auto_route": True,
                "router_model_id": "agentic/nova",
                "provider": "bedrock",
                "fallback_attempts": 0,
                "allow_api_billing": True,
            },
        )
        store.save_task(task)
        store.save_artifact(
            Artifact(
                job_id=job.id,
                task_id=task.id,
                type=ArtifactType.VERIFICATION,
                created_by="w",
                payload={
                    "check": "bedrock invoke",
                    "failure": "not_authenticated",
                    "result": "failed",
                    "adapter": "agentic",
                },
                confidence=0.5,
                evidence=["provider:bedrock", "not_authenticated"],
            )
        )
        orch = Orchestrator(store)
        with mock.patch(
            "puppetmaster.model_registry.load_registry",
            return_value=[bedrock_spec, openai_spec],
        ), mock.patch(
            "puppetmaster.platform_billing.detect_adapter_billing_cached",
            side_effect=_billing,
        ), mock.patch(
            "puppetmaster.preflight.adapter_cli_present",
            return_value=True,
        ), mock.patch(
            "puppetmaster.platform_lock.is_adapter_enabled",
            return_value=True,
        ), mock.patch(
            "puppetmaster.providers.available_providers",
            return_value={"openai"},
        ):
            n = orch._reroute_recoverable_failures(job)
        self.assertEqual(n, 1)
        updated = store.get_task_by_id(task.id)
        self.assertEqual(updated.status, TaskStatus.QUEUED)
        self.assertEqual(updated.payload.get("router_model_id"), "agentic/gpt")
        self.assertEqual(updated.payload.get("model"), "gpt-4o-mini")

    def test_credential_hint_names_profiles_and_bearer(self) -> None:
        from puppetmaster.bedrock import missing_bedrock_credentials_message

        msg = missing_bedrock_credentials_message()
        self.assertIn("AWS_PROFILE", msg)
        self.assertIn("default", msg.lower())
        self.assertIn("AWS_ACCESS_KEY_ID", msg)
        self.assertIn("AWS_SECRET_ACCESS_KEY", msg)
        self.assertIn("AWS_BEARER_TOKEN_BEDROCK", msg)

    def test_doctor_billing_reports_present_but_unverified(self) -> None:
        from puppetmaster.platform_billing import detect_agentic_billing

        status = detect_agentic_billing(self.aws_env, home=Path(self.tmp))
        self.assertFalse(status.healthy)
        self.assertIn("unverified", status.detail.lower())
        self.assertTrue(
            any("bedrock_invoke_health:unknown" in e for e in status.evidence)
        )


if __name__ == "__main__":
    unittest.main()
