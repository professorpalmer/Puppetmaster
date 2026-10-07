"""Real worker dispatch/acceptance seams for the additive consumption ledger."""
from __future__ import annotations

import json
import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hermetic_env  # noqa: F401

from puppetmaster.adapters import CliInvocation, CliWorkerAdapter, StreamedProcess
from puppetmaster.adapters.agentic import AgenticAdapter
from puppetmaster.invocation import execution_scope, invocation
from puppetmaster.models import Artifact, ArtifactType, Task, TaskStatus
from puppetmaster.providers import AssistantTurn, ProviderError
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.store import SwarmStore
from puppetmaster.usage import aggregate_token_usage
from puppetmaster.worker_runtime import WorkerRuntime


def verification(task, **usage):
    return Artifact(job_id=task.job_id, task_id=task.id,
                    type=ArtifactType.VERIFICATION, created_by="test",
                    confidence=1.0, evidence=["test:adapter"],
                    payload={"adapter": task.adapter, "check": "invocation",
                             "result": "passed", **usage})


class Adapter:
    def __init__(self, call):
        self.call = call

    def run(self, task, goal, worker_id):
        return self.call(task)


class CliAdapter(CliWorkerAdapter):
    name = "codex"

    def __init__(self, call):
        self.call = call

    def run(self, task, goal, worker_id):
        return self._run_cli_lifecycle(task, goal, worker_id)

    def _resolve_cli_executable(self, task):
        return "test", "test"

    def _prepare_cli_invocation(self, *args):
        return CliInvocation(command=["test"], sidecar_name="test")

    def _apply_pre_run_guards(self, *args):
        return None, {}

    def _invoke_cli(self, task, prepared, cwd, timeout_seconds):
        return self.call(task)

    def _finalize_cli_run(self, task, *args):
        # Deliberately overlap a raw usage source with legacy selected output.
        return [verification(task, tokens_in=0, tokens_out=0, tokens_estimated=False)]


class ProviderAdapter(AgenticAdapter):
    def run(self, task, goal, worker_id):
        turn = self._provider_call(provider="openai", model="test-model",
                                   messages=[], tools=None, extra={}, timeout=1,
                                   max_retries=1)
        return [verification(task, tokens_in=turn.usage.get("prompt_tokens", 0))]


class RuntimeAccountingContract:
    store_type = SwarmStore

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "state"
        self.store = self.store_type(self.root)
        self.store.init()
        self.job = self.store.create_job("invocation ledger runtime")
        self.task = Task(job_id=self.job.id, role="explore", instruction="inspect",
                         adapter="local", payload={"reuse_artifacts": False})
        self.store.save_task(self.task)
        self.runtime = WorkerRuntime(self.store, self.job.id, self.task.role, "worker",
                                     lease_seconds=30)

    def run_adapter(self, adapter):
        with mock.patch("puppetmaster.workers.get_adapter", return_value=adapter), \
                mock.patch("puppetmaster.adapters.git_snapshot", return_value={}):
            self.assertTrue(self.runtime.run_once())

    def records(self):
        reopened = self.store_type(self.root)
        return (reopened.list_attempts(self.job.id),
                reopened.list_usage_observations(self.job.id))

    def test_usage_persisted_before_acceptance_and_reopen(self):
        def invoke(task):
            attempts, observations = self.records()
            self.assertEqual(len(attempts), 1)
            self.assertEqual(observations, [])
            self.assertNotEqual(attempts[0].attempt_id, attempts[0].run_id)
            return [verification(task, tokens_in=7, tokens_out=3,
                                 real_cost_usd=0.125, cost_basis="api")]
        self.run_adapter(Adapter(invoke))
        attempts, observations = self.records()
        self.assertEqual(len(attempts), 1)
        self.assertEqual(len(observations), 1)
        self.assertEqual((observations[0].tokens_in, observations[0].tokens_out), (7, 3))
        self.assertEqual(observations[0].cost_usd, 0.125)
        self.assertEqual(observations[0].cost_basis, "api")
        self.assertEqual(self.store.get_task_by_id(self.task.id).status, TaskStatus.COMPLETE)

    def test_missing_and_measured_zero(self):
        self.run_adapter(Adapter(lambda task: [verification(task)]))
        _, observations = self.records()
        self.assertEqual(observations[0].usage_state, "unknown")
        self.assertIsNone(observations[0].tokens_in)
        self.store.reset_subgraph(self.job.id, [self.task.id])
        self.run_adapter(Adapter(lambda task: [verification(
            task, tokens_in=0, tokens_out=0, cost_usd=0, cost_basis="plan_marginal")]))
        attempts, observations = self.records()
        self.assertEqual(len(attempts), 2)
        zero = next(o for o in observations if o.usage_state == "measured")
        self.assertEqual(zero.tokens_in, 0)
        self.assertEqual(zero.cost_usd, 0)

    def test_exception_records_unknown(self):
        self.run_adapter(Adapter(mock.Mock(side_effect=RuntimeError("adapter failed"))))
        attempts, observations = self.records()
        self.assertEqual(len(attempts), 1)
        self.assertEqual(observations[0].usage_state, "unknown")
        self.assertEqual(self.store.get_task_by_id(self.task.id).status, TaskStatus.FAILED)

    def test_preflight_refusal_creates_no_attempt(self):
        blocked = verification(self.task, result="blocked", failure="preflight_blocked")
        with mock.patch("puppetmaster.workers.LocalWorker._preflight", return_value=blocked):
            self.run_adapter(Adapter(mock.Mock(side_effect=AssertionError("blocked"))))
        self.assertEqual(self.records(), ([], []))
        self.assertEqual(self.store.get_task_by_id(self.task.id).status, TaskStatus.FAILED)

    def test_failed_gate_keeps_usage(self):
        failed = SimpleNamespace(passed=False, artifacts=[], failed_reason="test")
        with mock.patch.object(self.runtime, "_evaluate_gates", return_value=failed):
            self.run_adapter(Adapter(lambda task: [verification(task, tokens_in=11)]))
        self.assertEqual(self.records()[1][0].tokens_in, 11)
        self.assertEqual(self.store.get_task_by_id(self.task.id).status, TaskStatus.FAILED)

    def test_lease_loss_before_artifact_save_keeps_usage(self):
        def invoke(task):
            self.runtime._lease_lost.set()
            return [verification(task, tokens_in=13)]
        self.run_adapter(Adapter(invoke))
        self.assertEqual(self.records()[1][0].tokens_in, 13)
        self.assertEqual(self.store.list_artifacts(self.job.id), [])

    def test_reset_fallback_and_selected_economics(self):
        self.run_adapter(Adapter(lambda task: [verification(task, tokens_in=31)]))
        first = self.records()
        self.store.reset_subgraph(self.job.id, [self.task.id])
        reset = self.store.get_task_by_id(self.task.id)
        self.assertEqual(reset.attempts, 0)
        self.store.save_task(replace(reset, adapter="shell"))
        self.run_adapter(Adapter(lambda task: [verification(task, tokens_in=9)]))
        attempts, observations = self.records()
        self.assertEqual(len({a.attempt_id for a in attempts}), 2)
        self.assertEqual({a.adapter for a in attempts}, {"local", "shell"})
        self.assertIn(first[1][0], observations)
        artifacts = self.store.list_artifacts(self.job.id)
        # Only the currently selected artifacts enter the existing economics.
        expected = aggregate_token_usage([verification(self.task, tokens_in=9)])
        actual = aggregate_token_usage(artifacts)
        self.assertEqual(actual, expected)

    def test_real_fallback_resets_counter_without_reusing_attempt(self):
        from puppetmaster.model_registry import ModelSpec, save_registry
        from puppetmaster.orchestrator import Orchestrator
        from puppetmaster.routing_authority import bind_registry_authority

        models = [ModelSpec(id=f"codex/{name}", adapter="codex",
                            adapter_model_name=name, capability_score=99, billing="plan")
                  for name in ("old", "new")]
        path = Path(self.tmp.name) / "models.json"
        save_registry(models, path)
        payload = bind_registry_authority({"auto_route": True, "model": "old",
                                           "router_model_id": "codex/old",
                                           "skip_preflight": True}, path, models)
        self.store.save_task(replace(self.task, adapter="codex", payload=payload))
        self.run_adapter(Adapter(lambda task: [verification(
            task, result="failed", failure="model_unavailable", tokens_in=21)]))
        first = self.records()[0][0]
        with mock.patch("puppetmaster.platform_billing.detect_adapter_billing_cached",
                        return_value=SimpleNamespace(healthy=True, billing="plan")), \
                mock.patch("puppetmaster.preflight.adapter_cli_present", return_value=True), \
                mock.patch("puppetmaster.platform_lock.is_adapter_enabled", return_value=True):
            self.assertEqual(Orchestrator(self.store)._reroute_recoverable_failures(self.job), 1)
        fallback = self.store.get_task_by_id(self.task.id)
        self.assertEqual(fallback.attempts, 0)
        self.assertEqual(fallback.payload["model"], "new")
        self.run_adapter(Adapter(lambda task: [verification(task, tokens_in=7)]))
        attempts, observations = self.records()
        self.assertEqual(len(attempts), 2)
        self.assertIn(first, attempts)
        self.assertEqual({a.model for a in attempts}, {"old", "new"})
        self.assertEqual({o.tokens_in for o in observations if o.tokens_in is not None}, {21, 7})

    def test_cli_raw_missing_zero_and_partial_not_selected_return(self):
        for raw_usage in (None, {"input_tokens": 0, "output_tokens": 0}, {"input_tokens": 17}):
            if self.records()[0]:
                self.store.reset_subgraph(self.job.id, [self.task.id])
            event = {"type": "turn.completed"}
            if raw_usage is not None:
                event["usage"] = raw_usage
            self.run_adapter(CliAdapter(lambda task: StreamedProcess(0, json.dumps(event), "")))
        attempts, observations = self.records()
        self.assertEqual(len(attempts), 3)
        self.assertEqual(len(observations), 6)  # usage and exit per invocation
        observations = [o for o in observations if o.observation_id != "process:exit"]
        self.assertEqual(len(observations), 3)  # no shared-return duplicate
        self.assertEqual(sorted(o.tokens_in for o in observations if o.tokens_in is not None), [0, 17])
        partial = next(o for o in observations if o.tokens_in == 17)
        self.assertIsNone(partial.tokens_out)
        self.assertEqual(sum(o.usage_state == "unknown" for o in observations), 1)

    def test_cli_process_outcome_survives_successful_delivery(self):
        from puppetmaster.models import JobStatus
        from puppetmaster.orchestrator import Orchestrator
        from puppetmaster.quality import assess_run_quality
        from puppetmaster.receipt import build_job_receipt

        for rc, timed_out in ((-9, True), (7, False), (0, False)):
            with self.subTest(returncode=rc, timed_out=timed_out):
                if self.records()[0]:
                    self.store.reset_subgraph(self.job.id, [self.task.id])
                previous = {a.attempt_id for a in self.records()[0]}
                event = {"type": "turn.completed", "usage": {
                    "input_tokens": 17, "output_tokens": 5}}
                adapter = CliAdapter(lambda task: StreamedProcess(
                    rc, json.dumps(event), "", timed_out=timed_out,
                    elapsed_seconds=1804))
                patch = Artifact(job_id=self.job.id, task_id=self.task.id,
                                 type=ArtifactType.PATCH, created_by="test",
                                 confidence=1.0, evidence=["test:retained-patch"],
                                 payload={"change": "Retained consumer patch", "files": ["example"], "diff": "--- a/example\n+++ b/example\n"})
                with mock.patch.object(adapter, "_finalize_cli_run", return_value=[
                        patch, verification(self.task, tokens_in=17, tokens_out=5)]):
                    self.run_adapter(adapter)
                reopened = self.store_type(self.root)
                self.assertEqual(reopened.get_task_by_id(self.task.id).status,
                                 TaskStatus.COMPLETE)
                final_status = Orchestrator(reopened)._final_job_status(self.job)
                self.assertEqual(final_status, JobStatus.COMPLETE)
                reopened.update_job_status(self.job.id, final_status)
                artifacts = reopened.list_artifacts(self.job.id)
                self.assertIn(patch.id, [a.id for a in artifacts])
                quality = assess_run_quality(artifacts)
                self.assertEqual(quality["quality"], "ok")
                self.assertTrue(quality["trustworthy"])
                receipt = json.loads(json.dumps(build_job_receipt(reopened, self.job.id)))
                self.assertEqual(receipt["status"], "complete")
                self.assertTrue(receipt["delivery"]["successful"])
                rows = [r for r in receipt["attempt_consumption"]["attempts"]
                        if r["attempt"]["attempt_id"] not in previous]
                self.assertEqual(len(rows), 1)
                row = rows[0]
                outcomes = row["process_outcomes"]
                self.assertEqual(len(outcomes), 1)
                self.assertEqual(outcomes[0]["returncode"], rc)
                self.assertEqual(outcomes[0]["timed_out"], timed_out)
                observations = reopened.list_usage_observations(
                    self.job.id, attempt_id=row["attempt"]["attempt_id"])
                outcome = next(o for o in observations if o.observation_id == "process:exit")
                self.assertEqual((outcome.returncode, outcome.timed_out), (rc, timed_out))
                self.assertFalse(reopened.record_usage_observation(outcome))
                self.assertEqual(row["totals"]["tokens_in"]["total"], 17)
                self.assertEqual(row["totals"]["tokens_out"]["total"], 5)

    def test_cli_finalizer_exception_keeps_raw_usage(self):
        event = {"usage": {"input_tokens": 12}}
        adapter = CliAdapter(lambda task: StreamedProcess(0, json.dumps(event), ""))
        with mock.patch.object(adapter, "_finalize_cli_run", side_effect=RuntimeError("finalize")):
            self.run_adapter(adapter)
        raw = next(o for o in self.records()[1] if o.observation_id == "stdout:0")
        self.assertEqual(raw.tokens_in, 12)
        outcome = next(o for o in self.records()[1] if o.observation_id == "process:exit")
        self.assertEqual((outcome.returncode, outcome.timed_out), (0, False))
        self.assertEqual(self.store.get_task_by_id(self.task.id).status, TaskStatus.FAILED)

    def test_provider_retry_keeps_unknown_and_raw_usage(self):
        returned = AssistantTurn(usage={"prompt_tokens": 0, "completion_tokens": 0},
                                 accounting_usage={"prompt_tokens": 19, "cost": 0})
        with mock.patch("puppetmaster.adapters.agentic.provider_chat", side_effect=[
                ProviderError("busy", reason="rate_limit", status=429), returned]), \
                mock.patch("puppetmaster.adapters.agentic.get_provider_circuit_breaker"), \
                mock.patch("puppetmaster.rate_limit_state.admit_or_raise"), \
                mock.patch("puppetmaster.adapters.agentic.time.sleep"):
            self.run_adapter(ProviderAdapter())
        attempts, observations = self.records()
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len({a.run_id for a in attempts}), 1)
        self.assertEqual(len(observations), 2)
        measured = next(o for o in observations if o.tokens_in == 19)
        self.assertIsNone(measured.tokens_out)
        self.assertEqual(measured.cost_usd, 0)
        self.assertEqual(measured.cost_basis, "api")

    def test_plan_cli_cost_is_separate_from_api_equivalent(self):
        self.store.save_task(replace(self.task, payload={"billing": "plan"}))
        event = {"usage": {"input_tokens": 5}, "total_cost_usd": 0.4}
        self.run_adapter(CliAdapter(lambda task: StreamedProcess(0, json.dumps(event), "")))
        attempts, observations = self.records()
        self.assertEqual(len(attempts), 1)
        by_basis = {o.cost_basis: o for o in observations}
        self.assertEqual(by_basis["api_equivalent"].cost_usd, 0.4)
        self.assertEqual(by_basis["api_equivalent"].cost_state, "estimated")
        self.assertEqual(by_basis["plan_marginal"].cost_usd, 0)
        self.assertEqual(by_basis["plan_marginal"].cost_state, "measured")

    def test_session_cumulative_cost_is_not_this_attempts_cost(self):
        # A resumed Claude Code run reports the session's total_cost_usd to
        # date; its token usage is this invocation's own.
        self.store.save_task(replace(self.task, payload={"billing": "api"}))
        event = {"type": "result", "usage": {"input_tokens": 5, "output_tokens": 2}, "total_cost_usd": 0.4}
        self.run_adapter(CliAdapter(lambda task: StreamedProcess(
            0, json.dumps(event), "", session_cumulative_cost=True)))
        raw = next(o for o in self.records()[1] if o.observation_id == "stdout:0")
        self.assertEqual((raw.tokens_in, raw.tokens_out), (5, 2))
        self.assertIsNone(raw.cost_usd)
        self.assertEqual(raw.cost_state, "unknown")

    def test_cli_billing_vocabulary(self):
        for billing, basis, cost in (("api", "api", 0.4),
                                     ("unknown", "unknown", None)):
            with self.subTest(billing=billing):
                if self.records()[0]:
                    self.store.reset_subgraph(self.job.id, [self.task.id])
                self.store.save_task(replace(self.store.get_task_by_id(self.task.id),
                                             payload={"billing": billing}))
                before = {a.attempt_id for a in self.records()[0]}
                event = {"usage": {"input_tokens": 5}, "total_cost_usd": 0.4}
                self.run_adapter(CliAdapter(lambda task: StreamedProcess(
                    0, json.dumps(event), "")))
                attempts, observations = self.records()
                fresh = [a for a in attempts if a.attempt_id not in before]
                self.assertEqual(len(fresh), 1)
                current = [o for o in observations if o.attempt_id == fresh[0].attempt_id]
                self.assertEqual(len(current), 2)
                raw = next(o for o in current if o.observation_id == "stdout:0")
                self.assertEqual((raw.cost_basis, raw.cost_usd), (basis, cost))
                self.assertEqual(raw.tokens_in, 5)

    def test_provider_failover_uses_actual_billing(self):
        for provider, basis, cost in (("openai", "api", 0.25),
                                      ("opencode-go", "api_equivalent", 0.25),
                                      (" OpenCode-Go ", "api_equivalent", 0.25),
                                      ("openai-codex", "api", 0.25),
                                      (" OpenAI-Codex ", "api", 0.25),
                                      ("unknown-provider", "unknown", None)):
            for streaming in (False, True):
                self._check_provider_failover(provider, basis, cost, streaming)

    def _check_provider_failover(self, provider, basis, cost, streaming):
        canonical = provider.strip().lower()
        plan = basis == "api_equivalent"
        with self.subTest(provider=provider, streaming=streaming):
            task = replace(self.task, payload={
                "billing": "api" if plan else "plan", "stream_deltas": streaming,
                "provider_max_retries": 0,
                "failover_models": [{"provider": provider, "model": "backup"}],
            })
            returned = AssistantTurn(
                tool_calls=[{"id": "s1", "name": "submit_findings",
                             "arguments": {"artifacts": []}}],
                usage={"prompt_tokens": 2, "completion_tokens": 1},
                accounting_usage={"prompt_tokens": 2, "completion_tokens": 1,
                                  "cost": 0.25},
            )
            before = {a.attempt_id for a in self.records()[0]}
            with execution_scope(self.store, SimpleNamespace(id="failover-run"), task), \
                    mock.patch.object(AgenticAdapter, "_compose_delta_sink",
                                      return_value=(lambda *_: None) if streaming else None), \
                    mock.patch("puppetmaster.adapters.agentic." + (
                        "provider_chat_streaming" if streaming else "provider_chat"), side_effect=[
                        ProviderError("failed", reason="http_status:500", status=500), returned]) as call, \
                    mock.patch("puppetmaster.adapters.agentic.get_provider_circuit_breaker"), \
                    mock.patch("puppetmaster.rate_limit_state.admit_or_raise"):
                _, selected_provider, _ = AgenticAdapter()._run_loop_with_failover(
                    task=task, cwd=Path(self.tmp.name), prompt="inspect", tools=[],
                    implement=False, on_stop=None, worker_id="worker",
                    provider=" OpenCode-Go ", model="primary")
            self.assertEqual(selected_provider, provider)
            self.assertEqual(call.call_count, 2)
            attempts, observations = self.records()
            fresh = [a for a in attempts if a.attempt_id not in before]
            self.assertEqual(len(fresh), 2)
            backup = next(a for a in fresh if a.model == "backup")
            current = [o for o in observations if o.attempt_id == backup.attempt_id]
            priced = next(o for o in current if o.source == f"provider:{canonical}")
            self.assertEqual((priced.cost_basis, priced.cost_usd), (basis, cost))
            self.assertEqual(len(current), 2 if plan else 1)
            self.assertEqual(sum(o.cost_basis == "plan_marginal" for o in current),
                             int(plan))
            primary = next(a for a in fresh if a.model == "primary")
            failed = [o for o in observations if o.attempt_id == primary.attempt_id]
            self.assertEqual(len(failed), 2)
            unavailable = next(o for o in failed if o.cost_basis == "unknown")
            self.assertEqual(unavailable.source, "provider:opencode-go:usage_unavailable")
            self.assertEqual(unavailable.usage_state, "unknown")
            self.assertIsNone(unavailable.cost_usd)
            self.assertTrue(any(o.attempt_id == primary.attempt_id and
                                o.cost_basis == "plan_marginal" for o in observations))

    def test_failed_failover_retains_each_provider(self):
        task = replace(self.task, payload={
            "billing": "plan", "stream_deltas": False, "provider_max_retries": 0,
            "failover_models": [{"provider": " OpenAI-Codex ", "model": "backup"}],
        })
        error = ProviderError("failed", reason="http_status:500", status=500)
        with execution_scope(self.store, SimpleNamespace(id="failed-run"), task), \
                mock.patch("puppetmaster.adapters.agentic.provider_chat", side_effect=error), \
                mock.patch("puppetmaster.adapters.agentic.get_provider_circuit_breaker"), \
                mock.patch("puppetmaster.rate_limit_state.admit_or_raise"):
            with self.assertRaises(ProviderError) as raised:
                AgenticAdapter()._run_loop_with_failover(
                    task=task, cwd=Path(self.tmp.name), prompt="inspect", tools=[],
                    implement=False, on_stop=None, worker_id="worker",
                    provider=" OpenCode-Go ", model="primary")
        self.assertIs(raised.exception, error)
        attempts, observations = self.records()
        self.assertEqual(len({a.attempt_id for a in attempts}), 2)
        self.assertEqual(len(observations), 3)
        for attempt in attempts:
            current = [o for o in observations if o.attempt_id == attempt.attempt_id]
            failed = next(o for o in current if o.cost_basis == "unknown")
            provider = "opencode-go" if attempt.model == "primary" else "openai-codex"
            self.assertEqual(failed.source, f"provider:{provider}:usage_unavailable")
            self.assertEqual(failed.usage_state, "unknown")
            self.assertIsNone(failed.cost_usd)
            self.assertEqual(len(current), 2 if attempt.model == "primary" else 1)

    def test_streamed_api_retry_overrides_plan_task(self):
        task = replace(self.task, payload={"billing": "plan"})
        returned = AssistantTurn(accounting_usage={"cost_usd": 0.3, "input_tokens": 9})
        with execution_scope(self.store, SimpleNamespace(id="stream-run"), task), \
                mock.patch("puppetmaster.adapters.agentic.provider_chat_streaming", side_effect=[
                    ProviderError("busy", reason="rate_limit", status=429), returned]), \
                mock.patch("puppetmaster.adapters.agentic.get_provider_circuit_breaker"), \
                mock.patch("puppetmaster.rate_limit_state.admit_or_raise"), \
                mock.patch("puppetmaster.adapters.agentic.time.sleep"):
            result = AgenticAdapter()._provider_call(
                provider="openai", model="test", messages=[], tools=None,
                extra={}, timeout=1, max_retries=1, on_delta=lambda *_: None)
        self.assertIs(result, returned)
        attempts, observations = self.records()
        self.assertEqual(len({a.attempt_id for a in attempts}), 2)
        self.assertEqual(len(observations), 2)
        self.assertFalse(any(o.cost_basis == "plan_marginal" for o in observations))
        self.assertEqual(sum(o.cost_usd or 0 for o in observations), 0.3)
        failed = next(o for o in observations if o.usage_state == "unknown")
        self.assertEqual(failed.source, "provider:openai:usage_unavailable")
        self.assertIsNone(failed.cost_usd)

    def test_direct_openai_overrides_plan_task(self):
        from puppetmaster.adapters.openai import OpenAIAdapter

        task = replace(self.task, payload={"billing": "plan", "openai_api_key": "test",
                                           "disable_codegraph": True})
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.getcode.return_value = 200
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": '{"artifacts": []}'}}],
            "usage": {"prompt_tokens": 3, "cost": 0.2},
        }).encode()
        with execution_scope(self.store, SimpleNamespace(id="openai-run"), task), \
                mock.patch("puppetmaster.adapters.openai.urllib.request.urlopen", return_value=response):
            OpenAIAdapter().run(task, "inspect", "worker")
        attempts, observations = self.records()
        self.assertEqual(len(attempts), 1)
        self.assertEqual(len(observations), 1)
        self.assertEqual((observations[0].cost_basis, observations[0].cost_usd), ("api", 0.2))

    def test_duplicate_source_and_shared_return_do_not_duplicate_observations(self):
        class ReportingAdapter:
            accounts_invocations = True

            def run(self, task, goal, worker_id):
                with invocation() as capture:
                    capture.observe({"tokens_in": 8})
                    capture.observe({"tokens_in": 8})
                return [verification(task, tokens_in=8)]
        self.run_adapter(ReportingAdapter())
        attempts, observations = self.records()
        self.assertEqual(len(attempts), 1)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].tokens_in, 8)

    def test_successful_persisted_reuse_creates_no_attempt(self):
        from test_working_set import _analysis_payload, _finding, _git_init_with_file
        from puppetmaster.working_set import stamp_fresh_validation

        repo = Path(self.tmp.name) / "repo"
        _git_init_with_file(repo, "src/a.py", "alpha\n")
        queued = replace(self.task, payload=_analysis_payload(repo))
        self.store.save_task(queued)
        source = replace(queued, id="source", status=TaskStatus.COMPLETE)
        self.store.save_task(source)
        finding = _finding(self.job.id, source.id, "cached finding")
        self.store.save_artifact(stamp_fresh_validation(source, [finding])[0])
        adapter = Adapter(mock.Mock(side_effect=AssertionError("reuse must skip invocation")))
        self.run_adapter(adapter)
        adapter.call.assert_not_called()
        self.assertEqual(self.store.get_task_by_id(self.task.id).status, TaskStatus.COMPLETE)
        self.assertEqual(self.records(), ([], []))

        from puppetmaster.consumption import build_attempt_consumption_report
        consumption = build_attempt_consumption_report(self.store, self.job.id)
        self.assertEqual(consumption.attempt_count, 0)
        self.assertIsNone(consumption.totals.tokens_in.total)

    def test_write_failures_do_not_erase_accepted_result(self):
        for operation in ("record_attempt", "record_usage_observation"):
            if self.store.get_task_by_id(self.task.id).status == TaskStatus.COMPLETE:
                self.store.reset_subgraph(self.job.id, [self.task.id])
            with mock.patch.object(self.store, operation, side_effect=OSError("disk")), \
                    self.assertLogs("puppetmaster.invocation", level="WARNING"):
                self.run_adapter(Adapter(lambda task: [verification(task, tokens_in=4)]))
            self.assertEqual(self.store.get_task_by_id(self.task.id).status, TaskStatus.COMPLETE)
            self.assertTrue(self.store.list_artifacts(self.job.id))
        events = self.store.read_events_since(self.job.id, 0)
        self.assertTrue(any(e["event"] == "consumption.persistence_failed" for e in events))

    def test_transient_start_failure_retries_same_identity(self):
        original = self.store.record_attempt
        calls = []
        def record(attempt):
            calls.append(attempt)
            if len(calls) == 1:
                raise OSError("busy")
            return original(attempt)
        with mock.patch.object(self.store, "record_attempt", side_effect=record):
            self.run_adapter(Adapter(lambda task: [verification(task, tokens_in=2)]))
        self.assertEqual(len(set(calls)), 1)
        self.assertEqual(len(self.records()[0]), 1)


class FileRuntimeAccountingTests(RuntimeAccountingContract, unittest.TestCase):
    pass


class SQLiteRuntimeAccountingTests(RuntimeAccountingContract, unittest.TestCase):
    store_type = SQLiteSwarmStore


if __name__ == "__main__":
    unittest.main()
