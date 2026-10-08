"""Canonical launch billing reaches the durable invocation budget."""
import json
import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hermetic_env  # noqa: F401
from test_invocation_accounting import CliAdapter

from puppetmaster.adapters import StreamedProcess
from puppetmaster.cli._dispatch import _main
from puppetmaster.model_registry import ModelSpec, save_registry, stamp_model_billing
from puppetmaster.orchestrator import Orchestrator, merge_routing_payload
from puppetmaster.platform_billing import BillingStatus
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.store import SwarmStore
from puppetmaster.worker_runtime import WorkerRuntime
from puppetmaster.workers import WorkerSpec
from puppetmaster.budget import BudgetPolicy


class BillingProvenanceTests(unittest.TestCase):
    def setUp(self):
        from puppetmaster.platform_billing import BillingStatus
        detector = patch('puppetmaster.platform_billing.detect_adapter_billing',
                         return_value=BillingStatus('codex', 'unknown', True, 'fixture'))
        detector.start()
        self.addCleanup(detector.stop)

    def test_adapter_invocations_honor_persisted_billing_through_receipt(self):
        from contextlib import ExitStack
        from unittest.mock import MagicMock
        from puppetmaster.adapters.openai import OpenAIAdapter
        from puppetmaster.cost import price_job
        from puppetmaster.invocation import execution_scope
        from puppetmaster.models import JobStatus
        from puppetmaster.providers import AssistantTurn
        from test_invocation_accounting import ProviderAdapter, verification

        cases = [('agentic', provider, streaming)
                 for provider in ('openai', 'opencode-go') for streaming in (False, True)]
        cases += [('openai', None, False), ('codex', None, False)]
        for adapter_name, provider, streaming in cases:
            for billing in ('plan', 'api', 'unknown'):
                with self.subTest(adapter=adapter_name, provider=provider, streaming=streaming,
                                  billing=billing), TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    # Opposite registry default proves the explicit pin is authoritative.
                    spec = ModelSpec(id=f'{adapter_name}/shared', adapter=adapter_name,
                        adapter_model_name='shared', billing=('unknown' if billing == 'unknown'
                            else 'api' if billing == 'plan' else 'plan'),
                        input_per_mtok_usd=25000, output_per_mtok_usd=0)
                    registry = root / 'models.json'
                    save_registry([spec], registry)
                    store = SwarmStore(root / 'state')
                    store.init()
                    job = store.create_job('billing boundary', budget_policy=BudgetPolicy(max_attempts=1))
                    with patch('puppetmaster.platform_lock.is_adapter_enabled', return_value=True):
                        task = Orchestrator(store)._create_tasks(job, [WorkerSpec(
                            role='explore', instruction='inspect', adapter=adapter_name, payload={
                                'model': 'shared', 'billing': billing,
                                'registry_path': str(registry), 'disable_codegraph': True,
                                'openai_api_key': 'test-key', 'cwd': tmp})])[0]
                    self.assertEqual(task.payload['billing'], billing)
                    usage = {'prompt_tokens': 10, 'completion_tokens': 2, 'cost_usd': 0.25}
                    with ExitStack() as stack:
                        stack.enter_context(execution_scope(store, SimpleNamespace(id='run'), task))
                        if adapter_name == 'agentic':
                            stack.enter_context(patch('puppetmaster.adapters.agentic.' + (
                                'provider_chat_streaming' if streaming else 'provider_chat'),
                                return_value=AssistantTurn(text='ok', usage=usage)))
                            stack.enter_context(patch('puppetmaster.adapters.agentic.get_provider_circuit_breaker'))
                            stack.enter_context(patch('puppetmaster.rate_limit_state.admit_or_raise'))
                            ProviderAdapter()._provider_call(provider=provider, model='shared',
                                messages=[], tools=None, extra={}, timeout=1, max_retries=0,
                                on_delta=(lambda *_: None) if streaming else None)
                            artifacts = [verification(task, tokens_in=10, tokens_out=2)]
                        elif adapter_name == 'openai':
                            response = MagicMock()
                            response.__enter__.return_value = response
                            response.getcode.return_value = 200
                            response.read.return_value = json.dumps({'usage': usage, 'choices': [
                                {'message': {'content': '{"artifacts":[]}'}, 'finish_reason': 'stop'}]}).encode()
                            stack.enter_context(patch('urllib.request.urlopen', return_value=response))
                            artifacts = OpenAIAdapter().run(task, job.goal, 'worker')
                        else:
                            stack.enter_context(patch('puppetmaster.adapters.git_snapshot', return_value={}))
                            artifacts = CliAdapter(lambda task: StreamedProcess(returncode=0,
                                stdout=json.dumps({'type': 'turn.completed', 'usage': usage}),
                                stderr='')).run(task, job.goal, 'worker')
                    snapshot = store.budget_snapshot(job.id)
                    reservation = snapshot['reservations'][0]
                    self.assertEqual(reservation['state'], 'pending_reconciliation'
                                     if billing == 'unknown' else 'settled')
                    self.assertEqual(snapshot['totals']['api_usd']['total'],
                                     0.25 if billing == 'api' else None if billing == 'unknown' else 0)
                    self.assertEqual(snapshot['totals']['plan_marginal_usd']['total'],
                                     None if billing == 'unknown' else 0)
                    observations = store.list_usage_observations(job.id)
                    measured = [o for o in observations if o.usage_state == 'measured']
                    self.assertEqual(len(measured), 1)
                    self.assertEqual(measured[0].cost_basis, 'api_equivalent'
                                     if billing == 'plan' else billing)
                    if billing == 'unknown':
                        continue
                    for artifact in artifacts:
                        store.save_artifact(artifact)
                    report = price_job(store.list_artifacts(job.id), [spec])
                    self.assertEqual(report.tasks[0].billing, billing)
                    with patch('puppetmaster.cost.load_registry', return_value=[spec]):
                        store.update_job_status(job.id, JobStatus.COMPLETE)
                    receipt = store.get_job(job.id).cost_receipt
                    self.assertEqual(receipt['actual_cost']['tasks'][0]['billing'], billing)

    def test_stale_canonical_suffix_cannot_preserve_billing(self):
        selected = ModelSpec(id='agentic/new-provider/shared', adapter='agentic',
                             adapter_model_name='shared', billing='api')
        payload = {'router_model_id': 'agentic/old-provider/shared',
                   'model': 'shared', 'billing': 'plan'}
        self.assertEqual(stamp_model_billing(payload, selected,
                                            registry=[selected])['billing'], 'api')

    def test_billing_requires_registered_unambiguous_identity(self):
        selected = ModelSpec(id='agentic/new-provider/shared', adapter='agentic',
                             adapter_model_name='shared', billing='api')
        other = replace(selected, id='agentic/other-provider/shared')
        for identity, registry, expected in (
            (selected.id, [selected, other], 'plan'),
            ('shared', [selected], 'plan'),
            ('shared', [selected, other], 'api'),
            ('agentic/old-provider/shared', [selected], 'api'),
            ('unregistered', [selected], 'api'),
            ('', [selected], 'api'),
        ):
            with self.subTest(identity=identity, registry=registry):
                self.assertEqual(stamp_model_billing(
                    {'model': identity, 'billing': 'plan'}, selected,
                    registry=registry)['billing'], expected)
        from puppetmaster.model_registry import resolve_model_pin
        self.assertEqual(resolve_model_pin('agentic/old-provider/shared',
                                          [selected]).registry_id, selected.id)

    def test_cost_uses_effective_route_billing(self):
        from puppetmaster.cost import price_job
        from puppetmaster.models import Artifact, ArtifactType
        from puppetmaster.router import RoutingDecision
        spec = ModelSpec(id='codex/gpt-6-astra', adapter='codex',
                         adapter_model_name='gpt-6-astra', billing='plan',
                         input_per_mtok_usd=1, output_per_mtok_usd=2)
        decision = RoutingDecision(model=spec, policy='balanced', capability_needed=1,
            estimated_tokens_in=10, estimated_tokens_out=2, estimated_cost_usd=0,
            reason='test')
        usage = Artifact(job_id='j', task_id='t', type=ArtifactType.VERIFICATION,
                         created_by='test', confidence=1.0, evidence=[], payload={'tokens_in': 1000000,
                         'tokens_out': 0, 'model': spec.adapter_model_name})
        for effective, expected in (('api', 1.0), ('plan', 0.0)):
            with self.subTest(billing=effective):
                payload = decision.to_artifact_payload(effective_billing=effective)
                self.assertEqual(payload['registry_billing'], 'plan')
                route = Artifact(job_id='j', task_id='t', type=ArtifactType.ROUTING,
                                 created_by='router', confidence=1.0, evidence=[], payload=payload)
                result = price_job([route, usage], [spec])
                self.assertEqual(result.tasks[0].billing, effective)
                self.assertEqual(result.total_marginal_cost_usd, expected)

    def test_direct_codex_parser_persistence_and_runtime(self):
        for store_type in (SwarmStore, SQLiteSwarmStore):
            for billing in ('plan', 'api', 'unknown'):
                with self.subTest(store=store_type.__name__, billing=billing), TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    registry = root / 'models.json'
                    save_registry([ModelSpec(id='codex/gpt-6-astra', adapter='codex',
                        adapter_model_name='gpt-6-astra', billing=billing,
                        enabled=True, retired=False)], registry)
                    store = store_type(root / 'state')
                    store.init()
                    captured = []

                    def run(instance, goal, specs, **kwargs):
                        job = store.create_job(goal, budget_policy=kwargs['budget_policy'])
                        task = instance._create_tasks(job, specs)[0]
                        captured.append((job, task))
                        runtime = WorkerRuntime(store, job.id, task.role, 'worker', lease_seconds=30)
                        event = {'type': 'turn.completed', 'usage': {
                            'input_tokens': 188965, 'output_tokens': 604,
                            'cached_input_tokens': 169600}}
                        adapter = CliAdapter(lambda task: StreamedProcess(
                            returncode=0, stdout=json.dumps(event), stderr=''))
                        # The fake CLI is ready regardless of host Codex login state.
                        with patch('puppetmaster.workers.get_adapter', return_value=adapter), \
                                patch('puppetmaster.preflight.detect_adapter_billing',
                                      return_value=BillingStatus('codex', billing, True,
                                                                 'fixture CLI ready')), \
                                patch('puppetmaster.adapters.git_snapshot', return_value={}):
                            self.assertTrue(runtime.run_once())
                        raise InterruptedError('captured real dispatch')

                    with patch.dict(os.environ, {'PUPPETMASTER_MODELS_PATH': str(registry),
                                                  'PUPPETMASTER_WORKER': '0'}), \
                            patch('puppetmaster.routing_authority.default_registry_path',
                                  return_value=registry), \
                            patch('puppetmaster.cli._dispatch.create_store', return_value=store), \
                            patch.object(Orchestrator, 'run', autospec=True, side_effect=run):
                        with self.assertRaisesRegex(InterruptedError, 'captured real dispatch'):
                            _main(['codex', 'inspect', '--model', 'gpt-6-astra',
                                   '--budget-max-attempts', '1', '--max-cost-usd', '9'])
                    job, task = captured[0]
                    reopened = store_type(root / 'state')
                    persisted = reopened.get_task_by_id(task.id)
                    self.assertEqual(persisted.payload['billing'], billing)
                    self.assertEqual(persisted.payload['router_model_id'], 'codex/gpt-6-astra')
                    self.assertEqual(persisted.payload['pinned_model'], 'codex/gpt-6-astra')
                    self.assertEqual(persisted.payload['registry_path'], str(registry.resolve()))
                    self.assertEqual(persisted.status.value, 'complete')
                    snapshot = reopened.budget_snapshot(job.id)
                    self.assertEqual(snapshot['reservations'][0]['state'],
                                     'settled' if billing == 'plan' else 'pending_reconciliation')
                    self.assertEqual(snapshot['totals']['attempts'], 1)
                    self.assertEqual(snapshot['totals']['tokens_in']['known_subtotal'], 188965)
                    self.assertEqual(snapshot['totals']['tokens_out']['known_subtotal'], 604)
                    self.assertEqual(snapshot['totals']['plan_marginal_usd']['total'],
                                     0 if billing == 'plan' else None)
                    observations = reopened.list_usage_observations(job.id)
                    measured = [o for o in observations if o.usage_state == 'measured']
                    self.assertEqual(len(measured), 1)
                    self.assertEqual(measured[0].cache_read_tokens, 169600)

    def test_routed_billing_and_changed_identity(self):
        plan = ModelSpec(id='codex/one', adapter='codex', adapter_model_name='one', billing='plan')
        api = ModelSpec(id='agentic/two', adapter='agentic', adapter_model_name='two',
                        billing='api', payload_defaults={'provider': 'openrouter'})
        def route(payload, spec):
            return merge_routing_payload(payload, SimpleNamespace(model=spec, policy='balanced',
                capability_needed=50, estimated_cost_usd=0, allowed_model_ids=None))
        first = route({'max_cost_usd': 9}, plan)
        self.assertEqual(first['billing'], 'plan')
        fallback = route(first, api)
        self.assertEqual(fallback['billing'], 'api')
        self.assertEqual(fallback['provider'], 'openrouter')
        self.assertEqual(fallback['max_cost_usd'], 9)
        self.assertEqual(route(fallback, replace(plan, billing='unknown'))['billing'], 'unknown')
        self.assertEqual(route({'billing': 'api'}, plan)['billing'], 'plan')
        self.assertEqual(stamp_model_billing({'model': 'gpt-6-astra'})['billing'], 'unknown')
        self.assertEqual(stamp_model_billing({'billing': 'invalid'}, plan)['billing'], 'plan')

    def test_a_stamped_allowance_follows_the_routed_billing(self):
        # A launch stamps the allowance before routing, as billing "unknown".
        # Routing then set billing "plan", and every routed worker under a job
        # cap failed: "budget allowance billing conflicts with invocation".
        from puppetmaster.budget import BudgetLiability, stamp_payload_budget_allowance
        plan = ModelSpec(id='codex/one', adapter='codex', adapter_model_name='one', billing='plan')
        api = ModelSpec(id='agentic/two', adapter='agentic', adapter_model_name='two', billing='api')
        def route(payload, spec):
            return merge_routing_payload(payload, SimpleNamespace(model=spec, policy='balanced',
                capability_needed=50, estimated_cost_usd=0, allowed_model_ids=None))
        stamped = stamp_payload_budget_allowance(BudgetPolicy(max_elapsed_seconds=300),
                                                 {'auto_route': True, 'timeout_seconds': 120},
                                                 adapter='codex')
        self.assertEqual(stamped['budget_allowance']['billing'], 'unknown')
        first = route(stamped, plan)
        self.assertEqual(first['budget_allowance'], {'billing': 'plan', 'plan_marginal_usd': 0,
                                                     'cost_state': 'known', 'elapsed_seconds': 120.0})
        fallback = route(first, api)
        self.assertEqual(fallback['budget_allowance'], {'billing': 'api', 'elapsed_seconds': 120.0})
        for payload in (first, fallback):
            BudgetLiability(**payload['budget_allowance'])
            self.assertEqual(payload['budget_allowance']['billing'], payload['billing'])
        self.assertNotIn('budget_allowance', route({'auto_route': True}, plan))

    def test_reroute_takes_the_new_models_provider_binding(self):
        # A rate-limited opencode-go route fell back to an OpenRouter model, but
        # the first route's injected provider stayed in the payload, so the
        # fallback went back to the exhausted opencode-go account.
        go = ModelSpec(id='agentic/deepseek-v4-pro', adapter='agentic', adapter_model_name='deepseek-v4-pro',
                       billing='plan', payload_defaults={'provider': 'opencode-go'})
        router = ModelSpec(id='agentic/z-ai/glm-5.3', adapter='agentic', adapter_model_name='z-ai/glm-5.3',
                           billing='api', payload_defaults={'provider': 'openrouter'})
        def route(payload, spec):
            return merge_routing_payload(payload, SimpleNamespace(model=spec, policy='balanced',
                capability_needed=85, estimated_cost_usd=0, allowed_model_ids=None),
                registry=[go, router], previous_adapter='agentic')
        first = route({'auto_route': True}, go)
        self.assertEqual(first['provider'], 'opencode-go')
        fallback = route(first, router)
        self.assertEqual((fallback['provider'], fallback['model']), ('openrouter', 'z-ai/glm-5.3'))
        # A provider the caller chose (not injected by the previous route) still wins.
        self.assertEqual(route(dict(first, provider='custom'), router)['provider'], 'custom')

    def test_billing_identity_aliases_and_ambiguous_models(self):
        plan = ModelSpec(id='claude-code/sonnet-4-5', adapter='claude-code',
                         adapter_model_name='sonnet-4.5', billing='plan')
        for identity in (plan.id, plan.adapter_model_name):
            for key in ('model', 'router_model_id'):
                with self.subTest(identity=identity, key=key):
                    self.assertEqual(stamp_model_billing(
                        {key: identity, 'billing': 'api'}, plan,
                        registry=[plan], previous_adapter='claude')['billing'], 'api')
        other = replace(plan, id='agentic/sonnet-4-5', adapter='agentic', billing='api')
        self.assertEqual(stamp_model_billing(
            {'model': 'sonnet-4.5', 'billing': 'api'}, plan,
            registry=[plan, other], previous_adapter='agentic')['billing'], 'plan')
        unknown = replace(plan, billing='unknown')
        self.assertEqual(stamp_model_billing(
            {'model': 'unregistered', 'billing': 'api'}, unknown)['billing'], 'unknown')
        self.assertEqual(stamp_model_billing(
            {'model': 'sonnet-4.5', 'billing': 'api'}, unknown,
            registry=[unknown, other])['billing'], 'unknown')
        defaults = replace(plan, payload_defaults={'billing': 'api'})
        decision = SimpleNamespace(model=defaults, policy='balanced', capability_needed=50,
                                   estimated_cost_usd=0, allowed_model_ids=None)
        self.assertEqual(merge_routing_payload({}, decision)['billing'], 'plan')

    def test_auto_route_preserves_same_identity_explicit_api(self):
        from puppetmaster.platform_billing import RegistryReconciliation
        from puppetmaster.invocation import execution_scope, invocation
        for identity_key in ('router_model_id', 'model'):
            for identity in ('codex/gpt-6-astra', 'gpt-6-astra'):
                with self.subTest(key=identity_key, identity=identity), TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    registry = root / 'models.json'
                    model = ModelSpec(id='codex/gpt-6-astra', adapter='codex',
                                      adapter_model_name='gpt-6-astra', billing='plan',
                                      capability_score=100)
                    save_registry([model], registry)
                    store = SQLiteSwarmStore(root / 'state')
                    store.init()
                    job = store.create_job('inspect', budget_policy=BudgetPolicy(max_attempts=1))
                    with patch('puppetmaster.platform_billing.reconcile_registry',
                               return_value=RegistryReconciliation([model], [], [])), \
                            patch('puppetmaster.preflight.adapter_cli_present', return_value=True):
                        task = Orchestrator(store)._create_tasks(job, [WorkerSpec(
                            role='explore', instruction='inspect', adapter='codex', payload={
                                'auto_route': True, 'registry_path': str(registry),
                                identity_key: identity, 'billing': 'api',
                                'min_capability': 1, 'max_cost_usd': 9})])[0]
                    self.assertEqual(task.payload['billing'], 'api')
                    routes = [a for a in store.list_artifacts(job.id)
                              if a.type.value == 'routing']
                    self.assertEqual(routes[-1].payload['billing'], task.payload['billing'])
                    self.assertEqual(task.payload['router_model_id'], model.id)
                    self.assertEqual(task.payload['max_cost_usd'], 9)
                    self.assertNotIn('pinned_model', task.payload)
                    with execution_scope(store, SimpleNamespace(id='run'), task):
                        with invocation() as capture:
                            capture.observe({'tokens_in': 12, 'tokens_out': 3, 'cost_usd': 0.25},
                                            cost_basis='api', final=True)
                    snapshot = store.budget_snapshot(job.id)
                    self.assertEqual(snapshot['totals']['api_usd']['total'], 0.25)
                    self.assertEqual(snapshot['reservations'][0]['state'], 'settled')

    def test_pinned_provider_billing_defaults_and_overrides(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = root / 'models.json'
            save_registry([ModelSpec(id='agentic/provider-model', adapter='agentic',
                adapter_model_name='provider-model', billing='api',
                payload_defaults={'provider': 'openrouter'})], registry)
            store = SwarmStore(root / 'state')
            store.init()
            for explicit in (None, 'plan'):
                payload = {'model': 'provider-model', 'registry_path': str(registry)}
                if explicit:
                    payload['billing'] = explicit
                job = store.create_job('provider', budget_policy=BudgetPolicy(max_attempts=1))
                task = Orchestrator(store)._create_tasks(job, [WorkerSpec(
                    role='inspect', instruction='inspect', adapter='agentic', payload=payload)])[0]
                self.assertEqual(task.payload['billing'], explicit or 'api')
                self.assertEqual(task.payload['provider'], 'openrouter')

    def test_actual_router_replaces_stale_plan_billing_before_persistence(self):
        from puppetmaster.platform_billing import RegistryReconciliation
        from puppetmaster.invocation import execution_scope, invocation
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = root / 'models.json'
            spec = ModelSpec(id='codex/metered', adapter='codex',
                             adapter_model_name='metered', capability_score=100, billing='api')
            save_registry([spec], registry)
            store = SQLiteSwarmStore(root / 'state')
            store.init()
            job = store.create_job('inspect', budget_policy=BudgetPolicy(max_attempts=1))
            with patch('puppetmaster.platform_billing.reconcile_registry',
                       return_value=RegistryReconciliation([spec], [], [])), \
                    patch('puppetmaster.preflight.adapter_cli_present', return_value=True):
                task = Orchestrator(store)._create_tasks(job, [WorkerSpec(
                    role='explore', instruction='inspect', adapter='codex', payload={
                        'auto_route': True, 'registry_path': str(registry),
                        'router_model_id': 'codex/old-plan', 'billing': 'plan',
                        'min_capability': 1, 'max_cost_usd': 9})])[0]
            self.assertEqual(task.payload['billing'], 'api')
            self.assertEqual(task.payload['router_model_id'], spec.id)
            self.assertEqual(task.payload['max_cost_usd'], 9)
            with execution_scope(store, SimpleNamespace(id='run'), task):
                with invocation() as capture:
                    capture.observe({'tokens_in': 12, 'tokens_out': 3, 'cost_usd': 0.25},
                                    cost_basis='api', final=True)
            snapshot = store.budget_snapshot(job.id)
            self.assertEqual(snapshot['reservations'][0]['state'], 'settled')
            self.assertEqual(snapshot['totals']['api_usd']['total'], 0.25)
            self.assertEqual(snapshot['totals']['tokens_in']['total'], 12)

    def test_explicit_pin_qualified_alias_preserves_billing(self):
        from puppetmaster.routing_authority import (
            RegistryAuthorityError, resolve_and_bind_explicit_pin, validate_pinned_dispatch,
        )
        plan = ModelSpec(id='codex/shared', adapter='codex',
                         adapter_model_name='shared', billing='plan')
        api = replace(plan, id='openai/shared', adapter='openai', billing='api')
        with TemporaryDirectory() as tmp:
            registry = Path(tmp) / 'models.json'
            save_registry([plan, api], registry)
            for spec, override in ((plan, 'api'), (api, 'plan')):
                for identity in ('shared', spec.id):
                    with self.subTest(adapter=spec.adapter, identity=identity):
                        result = resolve_and_bind_explicit_pin(
                            {'model': identity, 'billing': override},
                            adapter=spec.adapter, registry_path=registry)
                        self.assertEqual(result['billing'], override)
                        self.assertEqual(result['router_model_id'], spec.id)
                        self.assertEqual(validate_pinned_dispatch(
                            result, adapter=spec.adapter)['billing'], override)
            result = resolve_and_bind_explicit_pin(
                {'model': 'shared', 'router_model_id': api.id, 'billing': 'api'},
                adapter='codex', registry_path=registry)
            self.assertEqual(result['billing'], 'plan')
            save_registry([plan], registry)
            unique = resolve_and_bind_explicit_pin(
                {'model': 'shared', 'billing': 'api'},
                adapter='codex', registry_path=registry)
            self.assertEqual(unique['billing'], 'api')
            with self.assertRaises(RegistryAuthorityError):
                resolve_and_bind_explicit_pin(
                    {'model': 'unresolved', 'billing': 'api'},
                    adapter='codex', registry_path=registry)
