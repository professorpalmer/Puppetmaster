"""Billing admission and durable final execution provenance regressions."""
import itertools
import os
import sys
from types import SimpleNamespace
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hermetic_env  # noqa: F401

from puppetmaster.cost import final_routing_artifacts, price_job, build_current_registry_cost_report
from puppetmaster.model_registry import ModelSpec, save_registry
from puppetmaster.models import Artifact, ArtifactType, JobStatus
from puppetmaster.budget import BudgetPolicy
from puppetmaster.invocation import execution_scope, invocation
from puppetmaster.orchestrator import Orchestrator, merge_routing_payload
from puppetmaster.router import NoEligibleModelError, route_task, signals_from_worker_spec
from puppetmaster.store import SwarmStore
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.workers import WorkerSpec


class EconomicsProvenanceTests(unittest.TestCase):
    def setUp(self):
        from puppetmaster.platform_billing import BillingStatus
        detector = patch('puppetmaster.platform_billing.detect_adapter_billing',
                         return_value=BillingStatus('codex', 'unknown', True, 'fixture'))
        detector.start()
        self.addCleanup(detector.stop)

    def test_removed_registry_model_retains_execution_billing(self):
        from puppetmaster.cost import build_cost_report

        for store_type, billing, reported in itertools.product(
                (SwarmStore, SQLiteSwarmStore), ('plan', 'api', 'unknown'), (0.25, None)):
            with self.subTest(store=store_type.__name__, billing=billing,
                              reported=reported), TemporaryDirectory() as tmp:
                root = Path(tmp)
                registry = root / 'models.json'
                model = ModelSpec(id='codex/shared', adapter='codex',
                                  adapter_model_name='shared', billing=billing)
                save_registry([model], registry)
                store = store_type(root / 'state')
                store.init()
                job = store.create_job('historical billing')
                task = Orchestrator(store)._create_tasks(job, [WorkerSpec(
                    role='explore', instruction='inspect', adapter='codex', payload={
                        'model': model.id, 'billing': billing,
                        'registry_path': str(registry)})])[0]
                store.save_artifact(Artifact(job_id=job.id, task_id=task.id,
                    type=ArtifactType.VERIFICATION, created_by='worker', confidence=1,
                    evidence=['measured usage'], payload={'check': 'usage', 'result': 'passed',
                        'model': 'shared', 'tokens_in': 10, 'tokens_out': 2,
                        'real_cost_usd': reported}))
                save_registry([], registry)
                expected = 0.0 if billing == 'plan' else reported if billing == 'api' else None
                reopened = store_type(root / 'state')
                artifacts = reopened.list_artifacts(job.id)
                priced = price_job(artifacts, [])
                self.assertEqual(priced.tasks[0].billing, billing)
                self.assertEqual(priced.tasks[0].model_id, model.id)
                self.assertEqual(priced.tasks[0].priced, expected is not None)
                self.assertEqual(priced.total_marginal_cost_usd, expected or 0)
                report = build_current_registry_cost_report(job.id, artifacts, [])
                self.assertEqual(report['actual_cost']['total_marginal_cost_usd'], expected)
                self.assertEqual(report['actual_cost']['tasks'][0]['billing'], billing)
                with patch('puppetmaster.cost.load_registry', return_value=[]):
                    reopened.update_job_status(job.id, JobStatus.COMPLETE)
                terminal = store_type(root / 'state')
                receipt = terminal.get_job(job.id).cost_receipt
                self.assertEqual(receipt['actual_cost']['total_marginal_cost_usd'], expected)
                self.assertEqual(receipt['actual_cost']['tasks'][0]['billing'], billing)
                report = build_cost_report(terminal, job.id, [model])
                # Only the comparison is priced again on the current registry.
                report.pop('counterfactual')
                frozen = dict(receipt)
                frozen.pop('counterfactual')
                self.assertEqual(report, frozen)

    def test_removed_model_route_revision_overrides_pin_and_legacy_stays_unknown(self):
        usage = Artifact(job_id='j', task_id='t', type=ArtifactType.VERIFICATION,
            created_by='worker', confidence=1, evidence=['usage'], payload={
                'model': 'shared', 'tokens_in': 10, 'tokens_out': 2, 'real_cost_usd': 0.25})
        pin = Artifact(job_id='j', task_id='t', type=ArtifactType.VERIFICATION,
            created_by='orchestrator', confidence=1, evidence=['validated pin'], payload={
                'check': 'execution_billing', 'model_id': 'codex/shared', 'billing': 'plan'})
        routes = [Artifact(job_id='j', task_id='t', type=ArtifactType.ROUTING,
            created_by='router-fallback', confidence=1, evidence=['route'], payload={
                'model_id': 'codex/shared', 'billing': billing, 'route_revision': revision})
            for revision, billing in ((1, 'plan'), (2, 'api'))]
        for replay in itertools.permutations([pin, *routes]):
            priced = price_job([usage, *replay], [])
            self.assertEqual(priced.tasks[0].billing, 'api')
            self.assertEqual(priced.total_marginal_cost_usd, 0.25)
        legacy = price_job([usage], [])
        self.assertEqual(legacy.tasks[0].billing, 'unknown')
        self.assertFalse(legacy.tasks[0].priced)
        self.assertEqual(legacy.total_marginal_cost_usd, 0)

    def model(self):
        return ModelSpec(id='codex/gpt-6-astra', adapter='codex',
                         adapter_model_name='gpt-6-astra', billing='plan',
                         capability_score=100, input_per_mtok_usd=1,
                         output_per_mtok_usd=2)

    def test_effective_api_billing_fails_zero_cap_without_mutating_registry(self):
        model = self.model()
        for billing in ('api', 'plan'):
            worker = WorkerSpec(role='explore', instruction='inspect', adapter='codex',
                payload={'model': model.id, 'billing': billing, 'min_capability': 1,
                         'estimated_tokens_in': 1000000, 'estimated_tokens_out': 0,
                         'max_cost_usd': 0})
            signals = signals_from_worker_spec(worker)
            if billing == 'api':
                with self.assertRaises(NoEligibleModelError):
                    route_task(signals, [model])
            else:
                self.assertEqual(route_task(signals, [model]).estimated_cost_usd, 0)
        self.assertEqual(model.billing, 'plan')

    def test_auto_route_admission_and_persisted_estimate(self):
        from puppetmaster.platform_billing import RegistryReconciliation
        model = self.model()
        for billing, cap in (("api", 0), ("api", 2), ("plan", 0)):
            with self.subTest(billing=billing, cap=cap), TemporaryDirectory() as tmp:
                root = Path(tmp)
                registry = root / 'models.json'
                save_registry([model], registry)
                store = SwarmStore(root / 'state')
                store.init()
                job = store.create_job('inspect')
                worker = WorkerSpec(role='explore', instruction='inspect', adapter='codex',
                    payload={'auto_route': True, 'registry_path': str(registry),
                             'model': model.id, 'billing': billing, 'min_capability': 1,
                             'estimated_tokens_in': 1000000, 'estimated_tokens_out': 0,
                             'max_cost_usd': cap})
                with patch('puppetmaster.platform_billing.reconcile_registry',
                           return_value=RegistryReconciliation([model], [], [])), \
                        patch('puppetmaster.preflight.adapter_cli_present', return_value=True):
                    if billing == 'api' and cap == 0:
                        with self.assertRaises(NoEligibleModelError):
                            Orchestrator(store)._create_tasks(job, [worker])
                        continue
                    task = Orchestrator(store)._create_tasks(job, [worker])[0]
                route = final_routing_artifacts(store.list_artifacts(job.id))[task.id]
                self.assertEqual(task.payload['billing'], billing)
                self.assertEqual(route.payload['billing'], billing)
                self.assertEqual(route.payload['registry_billing'], 'plan')
                self.assertEqual(route.payload['route_revision'], task.payload['route_revision'])
                self.assertEqual(route.payload['estimated_cost_usd'], 1 if billing == 'api' else 0)
                self.assertEqual(task.payload['router_estimated_cost_usd'], route.payload['estimated_cost_usd'])

    def test_pinned_billing_reaches_report(self):
        model = self.model()
        for billing, store_type in itertools.product(('api', 'plan'), (SwarmStore, SQLiteSwarmStore)):
            with self.subTest(billing=billing, store=store_type.__name__), TemporaryDirectory() as tmp:
                root = Path(tmp)
                registry = root / 'models.json'
                save_registry([model], registry)
                store = store_type(root / 'state')
                store.init()
                job = store.create_job('inspect', budget_policy=BudgetPolicy(max_attempts=1))
                task = Orchestrator(store)._create_tasks(job, [WorkerSpec(
                    role='explore', instruction='inspect', adapter='codex',
                    payload={'model': model.id, 'billing': billing,
                             'registry_path': str(registry)})])[0]
                store.save_artifact(Artifact(job_id=job.id, task_id=task.id,
                    type=ArtifactType.VERIFICATION, created_by='worker', confidence=1,
                    evidence=['measured usage'], payload={'model': model.adapter_model_name,
                    'check': 'usage', 'result': 'passed', 'tokens_in': 1000000, 'tokens_out': 0, 'real_cost_usd': 1}))
                artifacts = store.list_artifacts(job.id)
                self.assertFalse(any(a.type == ArtifactType.ROUTING for a in artifacts))
                report = build_current_registry_cost_report(job.id, artifacts, [model])
                result = price_job(artifacts, [model])
                self.assertEqual(result.tasks[0].billing, task.payload['billing'])
                self.assertEqual(result.total_marginal_cost_usd, 1 if billing == 'api' else 0)
                self.assertEqual(report['actual_cost']['tasks'][0]['billing'], billing)
                with execution_scope(store, SimpleNamespace(id='run'), task):
                    with invocation() as capture:
                        capture.observe({'tokens_in': 1000000, 'tokens_out': 0,
                                         'cost_usd': 1 if billing == 'api' else 0},
                                        cost_basis=billing, final=True)
                totals = store.budget_snapshot(job.id)['totals']
                self.assertEqual(totals['api_usd' if billing == 'api' else 'plan_marginal_usd']['total'],
                                 result.total_marginal_cost_usd)
                with patch('puppetmaster.cost.load_registry', return_value=[model]):
                    store.update_job_status(job.id, JobStatus.COMPLETE)
                receipt = store.get_job(job.id).cost_receipt
                self.assertEqual(receipt['actual_cost']['tasks'][0]['billing'], billing)


    def test_final_revision_survives_replay_and_timestamp_ties(self):
        for creators in (('router', 'router-fallback', 'router-fallback'),
                         ('router', 'router-escalation', 'router-fallback'),
                         ('router', 'router-review-escalation', 'router-fallback')):
            routes = [Artifact(job_id='j', task_id='t', type=ArtifactType.ROUTING,
                created_by=creator, confidence=1, evidence=[], created_at='same',
                payload={'route_revision': i, 'model_id': self.model().id,
                         'billing': 'api' if i == 3 else 'plan'})
                for i, creator in enumerate(creators, 1)]
            for replay in itertools.permutations(routes):
                self.assertEqual(final_routing_artifacts([*replay, replay[0]])['t'], routes[-1])
                usage = Artifact(job_id='j', task_id='t', type=ArtifactType.VERIFICATION,
                    created_by='worker', confidence=1, evidence=['usage'], payload={
                        'model': self.model().adapter_model_name,
                        'tokens_in': 1000000, 'tokens_out': 0})
                priced = price_job([*replay, usage], [self.model()])
                self.assertEqual(priced.tasks[0].billing, 'api')
                self.assertEqual(priced.total_marginal_cost_usd, 1)


    def test_legacy_order_is_deterministic(self):
        routes = [Artifact(job_id='j', task_id='t', type=ArtifactType.ROUTING,
            created_by=creator, confidence=1, evidence=[], created_at=stamp, payload={})
            for creator, stamp in [('router-escalation', '1'), ('router-fallback', '2')]]
        self.assertEqual(final_routing_artifacts(routes)['t'], routes[-1])
        self.assertEqual(final_routing_artifacts(reversed(routes))['t'], routes[-1])
        tied = [Artifact(job_id='j', task_id='t', type=ArtifactType.ROUTING,
            created_by='router-fallback', confidence=1, evidence=[], created_at='same', payload={})
            for _ in range(2)]
        self.assertEqual(final_routing_artifacts(tied), final_routing_artifacts(reversed(tied)))

    def test_merge_increments_revision(self):
        decision = route_task(signals_from_worker_spec(WorkerSpec(
            role='explore', instruction='inspect', adapter='codex')), [self.model()])
        payload = {}
        for expected in range(1, 5):
            payload = merge_routing_payload(payload, decision)
            self.assertEqual(payload['route_revision'], expected)

    def test_pin_artifacts_are_known_types_and_transient_rows_reopen(self):
        import json
        import sqlite3
        from enum import Enum
        from puppetmaster.models import artifact_from_dict
        from dataclasses import asdict as to_dict
        from puppetmaster.stitcher import Stitcher

        class ReleasedArtifactType(str, Enum):
            FINDING = 'finding'
            DECISION = 'decision'
            PATCH = 'patch'
            VERIFICATION = 'verification'
            RISK = 'risk'
            MEMORY_SUMMARY = 'memory_summary'
            ROUTING = 'routing'
            GATE = 'gate'
            GIST = 'gist'

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = root / 'models.json'
            model = self.model()
            save_registry([model], registry)
            store = SQLiteSwarmStore(root / 'state')
            store.init()
            job = store.create_job('pin compatibility')
            task = Orchestrator(store)._create_tasks(job, [WorkerSpec(
                role='explore', instruction='inspect', adapter='codex',
                payload={'model': model.id, 'billing': 'api',
                         'registry_path': str(registry)})])[0]
            artifacts = store.list_artifacts(job.id)
            for artifact in artifacts:
                ReleasedArtifactType(artifact.type.value)
                artifact_from_dict(to_dict(artifact)).validate()
            pin = next(a for a in artifacts if a.payload.get('model_id') == model.id)
            legacy = to_dict(pin)
            legacy['id'] = 'artifact_transient'
            legacy['type'] = 'execution_billing'
            legacy['payload'] = {'model_id': model.id, 'billing': 'api'}
            with closing(sqlite3.connect(store.db_path)) as connection, connection:
                connection.execute(
                    'INSERT INTO artifacts(id, job_id, task_id, type, data) VALUES (?, ?, ?, ?, ?)',
                    (legacy['id'], job.id, task.id, legacy['type'], json.dumps(legacy)))
            reopened = SQLiteSwarmStore(root / 'state')
            mixed = reopened.list_artifacts(job.id)
            self.assertEqual(len(mixed), len(artifacts) + 1)
            converted = next(a for a in mixed if a.id == legacy['id'])
            self.assertEqual(converted.type, ArtifactType.VERIFICATION)
            converted.validate()
            self.assertEqual(converted.payload['billing'], 'api')
            usage = Artifact(job_id=job.id, task_id=task.id,
                type=ArtifactType.VERIFICATION, created_by='worker', confidence=1,
                evidence=['usage'], payload={'check': 'usage', 'result': 'passed',
                    'model': model.adapter_model_name, 'tokens_in': 1000000,
                    'tokens_out': 0})
            for history in ([converted, usage], [*mixed, usage]):
                priced = price_job(history, [model])
                self.assertEqual(priced.total_marginal_cost_usd, 1)
                self.assertEqual(priced.tasks[0].billing, 'api')
            route = Artifact(job_id=job.id, task_id=task.id,
                type=ArtifactType.ROUTING, created_by='router', confidence=1,
                evidence=['route'], payload={'model_id': model.id, 'adapter': 'codex',
                    'policy': 'balanced', 'billing': 'plan', 'route_revision': 1})
            self.assertEqual(price_job([*mixed, usage, route], [model]).tasks[0].billing,
                             'plan')
            reopened.save_artifact(converted)
            with closing(sqlite3.connect(store.db_path)) as connection, connection:
                for kind, raw in connection.execute('SELECT type, data FROM artifacts'):
                    ReleasedArtifactType(kind)
                    ReleasedArtifactType(json.loads(raw)['type'])
            Stitcher(reopened).stitch(job.id)
            self.assertFalse(final_routing_artifacts(mixed))
