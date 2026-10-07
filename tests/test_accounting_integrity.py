"""Usage quality survives the ledger, pricing keeps unknown counts unknown,
and failure classification needs an explicit credential diagnosis."""

from __future__ import annotations

import json
import unittest

import hermetic_env  # noqa: F401
import test_store_contracts
from puppetmaster.adapters.codex import codex_diagnostic_text, parse_codex_events
from puppetmaster.attempts import ExecutionAttempt, UsageObservation, canonical_record
from puppetmaster.cost import price_job
from puppetmaster.failure import classify_codex_failure
from puppetmaster.model_registry import ModelSpec
from puppetmaster.models import Artifact, ArtifactType


class LedgerQualityTests(unittest.TestCase):
    stores = test_store_contracts.StoreContractTests.stores

    def test_quality_flags_persist_and_replay_in_both_stores(self) -> None:
        for store, job, task, run, ref in self.stores():
            with self.subTest(store=type(store).__name__):
                attempt = ExecutionAttempt.from_run(run, adapter="codex")
                store.record_attempt(attempt)
                flagged = UsageObservation(job.id, attempt.attempt_id, "codex:rollout",
                                           "rollout_token_usage_records", run.started_at,
                                           usage_state="measured", tokens_in=100, tokens_out=5,
                                           quality=("disputed:input_tokens", "conflict:cached_exceeds_input"))
                plain = UsageObservation(job.id, attempt.attempt_id, "process:exit", "codex",
                                         run.started_at, returncode=0)
                self.assertTrue(store.record_usage_observation(flagged))
                self.assertTrue(store.record_usage_observation(plain))
                self.assertFalse(store.record_usage_observation(flagged))
                reopened = type(store)(store.root)
                stored = {o.observation_id: o for o in reopened.list_usage_observations(job.id)}
                self.assertEqual(stored["codex:rollout"].quality,
                                 ("conflict:cached_exceeds_input", "disputed:input_tokens"))
                self.assertEqual(stored["process:exit"].quality, ())

    def test_unflagged_record_omits_quality_for_older_readers(self) -> None:
        plain = UsageObservation("j", "a", "o", "s", "2026-10-06T00:00:00Z")
        self.assertNotIn("quality", json.loads(canonical_record(plain)))
        flagged = UsageObservation("j", "a", "o", "s", "2026-10-06T00:00:00Z", quality=["partial:x"])
        self.assertEqual(json.loads(canonical_record(flagged))["quality"], ["partial:x"])
        with self.assertRaises(ValueError):
            UsageObservation("j", "a", "o", "s", "2026-10-06T00:00:00Z", quality=[""])


def _usage_artifact(task_id: str, **payload) -> Artifact:
    return Artifact(job_id="j", task_id=task_id, type=ArtifactType.VERIFICATION, created_by="test",
                    confidence=1.0, evidence=[], payload=dict(payload, model="m"))


class PresencePricingTests(unittest.TestCase):
    def _price(self, billing: str, **payload):
        spec = ModelSpec(id="codex/m", adapter="codex", adapter_model_name="m", billing=billing,
                         input_per_mtok_usd=1, output_per_mtok_usd=2)
        return price_job([_usage_artifact("t", **payload)], [spec]).tasks[0]

    def test_known_counts_still_price(self) -> None:
        task = self._price("api", tokens_in=1_000_000, tokens_out=0, cached_input_tokens=0)
        self.assertTrue(task.priced)
        self.assertEqual(task.marginal_cost_usd, 1.0)
        self.assertEqual((task.usage_unknown, task.usage_invalid), ([], []))

    def test_unknown_billable_count_is_unpriced_not_zero(self) -> None:
        for payload, unknown in (({"tokens_in": None, "tokens_out": 10}, ["tokens_in"]),
                                 ({"tokens_in": 10, "tokens_out": None}, ["tokens_out"]),
                                 ({"tokens_in": 10, "tokens_out": 1, "cached_input_tokens": None},
                                  ["cached_input_tokens"])):
            with self.subTest(payload=payload):
                task = self._price("api", **payload)
                self.assertFalse(task.priced)
                self.assertIsNone(task.api_equivalent_cost_usd)
                self.assertEqual(task.usage_unknown, unknown)

    def test_cache_over_input_is_invalid_and_unpriced(self) -> None:
        task = self._price("api", tokens_in=100, tokens_out=1, cached_input_tokens=300)
        self.assertFalse(task.priced)
        self.assertEqual(task.usage_invalid, ["cached_exceeds_input"])

    def test_plan_marginal_settles_while_valuation_stays_unknown(self) -> None:
        task = self._price("plan", tokens_in=None, tokens_out=10)
        self.assertTrue(task.priced)
        self.assertEqual(task.marginal_cost_usd, 0.0)
        self.assertIsNone(task.api_equivalent_cost_usd)

    def test_reported_provider_cost_does_not_need_counts(self) -> None:
        task = self._price("api", tokens_in=None, tokens_out=None, real_cost_usd=0.25)
        self.assertTrue(task.priced)
        self.assertEqual(task.marginal_cost_usd, 0.25)


class AuthClassificationTests(unittest.TestCase):
    def test_traceback_symbol_is_not_a_logout(self) -> None:
        text = ('Traceback (most recent call last):\n'
                '  File "launcher.py", line 1, in production_authority\n'
                'pm131_home.ProfileError: unknown_home_surface')
        self.assertEqual(classify_codex_failure(text), "unknown")
        self.assertEqual(classify_codex_failure("pm131_home.ProfileError: unknown_home_surface"), "unknown")

    def test_quoted_source_and_line_numbers_are_not_auth(self) -> None:
        text = ('Traceback (most recent call last):\n'
                '  File "/srv/login/app.py", line 401, in handler\n'
                '    session = login(user)\n'
                '              ^^^^^^^^^^^\n'
                'KeyError: user')
        self.assertEqual(classify_codex_failure(text), "unknown")
        self.assertEqual(classify_codex_failure("digest a4019f0e mismatch"), "unknown")

    def test_explicit_credential_diagnoses_stay_authentication(self) -> None:
        for text in ("Error: Not logged in. Run `codex login`.",
                     "unexpected status 401 Unauthorized: Missing bearer",
                     "HTTP 401", "auth error: token expired", "login required",
                     "Please re-authenticate", "invalid api key provided", "Authentication failed",
                     "Traceback (most recent call last):\n  File \"x.py\", line 3, in f\n"
                     "RuntimeError: auth failed: token revoked"):
            with self.subTest(text=text):
                self.assertEqual(classify_codex_failure(text), "not_authenticated")

    def test_codex_transcript_is_not_diagnostic(self) -> None:
        stdout = "\n".join([
            "Reading additional input from stdin...",
            json.dumps({"type": "item.completed", "item": {"type": "agent_message",
                        "text": "Fixed the login flow and the 401 handler in auth.py."}}),
            json.dumps({"type": "error", "message": "stream disconnected before completion"}),
        ])
        text = codex_diagnostic_text(stdout, parse_codex_events(stdout))
        self.assertNotIn("login flow", text)
        self.assertIn("stream disconnected", text)
        self.assertEqual(classify_codex_failure(text), "network_error")
        failed = json.dumps({"type": "error", "message": "unexpected status 401 Unauthorized"})
        self.assertEqual(classify_codex_failure(codex_diagnostic_text(failed, parse_codex_events(failed))),
                         "not_authenticated")


if __name__ == "__main__":
    unittest.main()
