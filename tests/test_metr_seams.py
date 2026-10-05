"""v1.22.37 METR seam closures: dispatcher, listings, coordinator, receipts, owner."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from puppetmaster.artifact_status import (
    CLAIM_SUPPORT_INDEPENDENT,
    CLAIM_SUPPORT_WORKER_ASSERTED,
    infer_claim_support_status,
)
from puppetmaster.gist_admission import (
    admit_gist,
    build_pending_gist,
    is_admitted_for_shared_context,
    maybe_admit_finding_as_gist,
)
from unittest import mock

from puppetmaster.metr_seams import (
    HOLD_STATE,
    REASON_CROSS_JOB,
    REASON_GATE_FAILED,
    REASON_NEW_JOB,
    REASON_PARENT_MISMATCH,
    REASON_SUBGRAPH_HOLD,
    REASON_SUBGRAPH_VETO,
    REASON_SUBGRAPH_WRITER,
    REASON_WORKER_JOB_COMPLETE,
    REASON_WORKER_PROTOCOL,
    VETO_STATE,
    WAIT_EXTERNAL,
    WAIT_USER,
    delivery_claim_support_status,
    host_observed_kind,
    record_host_observation,
)
from puppetmaster.models import (
    AgentRun,
    Artifact,
    ArtifactType,
    JobStatus,
    Task,
    TaskStatus,
    job_from_dict,
    to_jsonable,
)
from puppetmaster.receipt import build_job_receipt, record_host_delivery_observation
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.store import SwarmStore


def _events(store: SwarmStore, job_id: str, name: str) -> list[dict]:
    return [event for event in store.read_events(job_id) if event.get("event") == name]


def _refused_reasons(store: SwarmStore, job_id: str) -> list[str]:
    return [
        str((event.get("payload") or {}).get("reason") or "")
        for event in _events(store, job_id, "task.enqueue_refused")
    ]


def _parent(store: SwarmStore, job_id: str, instruction: str = "root") -> Task:
    parent = Task(
        job_id=job_id,
        role="explore",
        instruction=instruction,
        status=TaskStatus.COMPLETE,
    )
    store.save_task(parent)
    return parent


def _finding(
    job_id: str,
    task_id: str,
    claim: str,
    *,
    extra: dict | None = None,
) -> Artifact:
    payload = {"claim": claim}
    if extra:
        payload.update(extra)
    return Artifact(
        job_id=job_id,
        task_id=task_id,
        type=ArtifactType.FINDING,
        created_by="worker",
        confidence=0.9,
        evidence=["test.py:1"],
        payload=payload,
    )


class GraphDispatcherTests(unittest.TestCase):
    def test_legitimate_same_job_follow_ups_still_work(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("dispatcher")
            parent = _parent(store, job.id)
            finding = _finding(
                job.id,
                parent.id,
                "needs more work",
                extra={
                    "enqueue_subtasks": [
                        {"role": "review", "instruction": "review module"},
                        {"role": "audit", "instruction": "audit risks"},
                    ]
                },
            )
            store.save_artifact(finding)
            created = store.maybe_enqueue_follow_ups_from_artifact(
                finding, parent_task_id=parent.id, created_by="worker-1"
            )
            self.assertEqual(len(created), 2)
            self.assertEqual({task.role for task in created}, {"review", "audit"})
            for child in created:
                self.assertEqual(child.job_id, job.id)
                self.assertEqual(child.depends_on, [parent.id])

    def test_foreign_job_id_and_new_job_and_wrong_parent_are_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("same job only")
            other = store.create_job("other job")
            parent = _parent(store, job.id)
            stranger = _parent(store, job.id, "stranger")
            finding = _finding(
                job.id,
                parent.id,
                "spawn elsewhere",
                extra={
                    "enqueue_subtasks": [
                        {
                            "role": "review",
                            "instruction": "review elsewhere",
                            "job_id": other.id,
                        },
                        {
                            "role": "audit",
                            "instruction": "open a new job",
                            "create_job": True,
                        },
                        {
                            "role": "review",
                            "instruction": "wrong parent",
                            "parent_task_id": stranger.id,
                        },
                    ]
                },
            )
            store.save_artifact(finding)
            created = store.maybe_enqueue_follow_ups_from_artifact(
                finding, parent_task_id=parent.id, created_by="worker-1"
            )
            self.assertEqual(created, [])
            reasons = set(_refused_reasons(store, job.id))
            self.assertIn(REASON_CROSS_JOB, reasons)
            self.assertIn(REASON_NEW_JOB, reasons)
            self.assertIn(REASON_PARENT_MISMATCH, reasons)
            self.assertEqual(store.list_tasks(other.id), [])

    def test_worker_recruit_hold_veto_mailbox_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("no worker protocol")
            parent = _parent(store, job.id)
            finding = _finding(
                job.id,
                parent.id,
                "coordinate peers",
                extra={
                    "enqueue_subtasks": [
                        {"role": "recruit", "instruction": "recruit peers"},
                        {"role": "HOLD", "instruction": "take HOLD of the subgraph"},
                        {"role": "VETO", "instruction": "VETO the merge"},
                        {
                            "role": "review",
                            "instruction": "open a mailbox protocol with job B",
                        },
                    ]
                },
            )
            store.save_artifact(finding)
            created = store.maybe_enqueue_follow_ups_from_artifact(
                finding, parent_task_id=parent.id, created_by="worker-1"
            )
            self.assertEqual(created, [])
            self.assertTrue(
                all(reason == REASON_WORKER_PROTOCOL for reason in _refused_reasons(store, job.id))
            )
            self.assertIsNone(
                store.hold_subgraph(job.id, actor="worker-1")
            )
            self.assertIsNone(store.veto_subgraph(job.id, actor="worker-1"))
            job_after = store.get_job(job.id)
            self.assertIsNone(job_after.subgraph_hold)


class CrossJobListingTests(unittest.TestCase):
    def test_protocol_gist_is_not_admitted_or_injected(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("no board")
            gist = build_pending_gist(
                job_id=job.id,
                task_id="task-1",
                created_by="worker",
                claim="HOLD the other job and recruit peers",
                source_artifact_ids=["src-1"],
            )
            store.save_artifact(gist)
            admitted = admit_gist(store, gist, verifier_result=True)
            self.assertEqual(admitted.payload["admission"], "rejected")
            self.assertFalse(is_admitted_for_shared_context(admitted))

    def test_worker_asserted_finding_does_not_inject_across_jobs(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job_a = store.create_job("job a")
            job_b = store.create_job("job b")
            finding = _finding(job_a.id, "task-a", "mailbox ping from A")
            store.save_artifact(finding)
            self.assertTrue(is_admitted_for_shared_context(finding, for_job_id=job_a.id))
            self.assertFalse(is_admitted_for_shared_context(finding, for_job_id=job_b.id))
            supported = Artifact(
                job_id=job_a.id,
                task_id="task-a",
                type=ArtifactType.FINDING,
                created_by="host",
                confidence=0.9,
                evidence=["host-gate"],
                payload={"claim": "host-admitted finding"},
                claim_support_status=CLAIM_SUPPORT_INDEPENDENT,
            )
            self.assertTrue(
                is_admitted_for_shared_context(supported, for_job_id=job_b.id)
            )

    def test_worker_cannot_self_upgrade_then_leak(self) -> None:
        finding = _finding(
            "job-a",
            "task-a",
            "I am independently supported",
            extra={"claim_support_status": "independently_supported"},
        )
        self.assertEqual(finding.claim_support_status, CLAIM_SUPPORT_WORKER_ASSERTED)
        self.assertFalse(is_admitted_for_shared_context(finding, for_job_id="job-b"))
        self.assertIsNone(maybe_admit_finding_as_gist(None, finding))


class CoordinatorOutlivesWorkersTests(unittest.TestCase):
    def test_job_persists_goal_criteria_and_authority(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("ship the patch")
            updated = store.bind_job_contract(
                job.id,
                acceptance_criteria=["tests pass", "diff is scoped"],
                granted_authority={"may_edit": ["src/"], "may_ship": False},
            )
            self.assertEqual(updated.goal, "ship the patch")
            self.assertEqual(
                updated.acceptance_criteria, ["tests pass", "diff is scoped"]
            )
            self.assertEqual(updated.granted_authority["may_edit"], ["src/"])
            reloaded = store.get_job(job.id)
            self.assertEqual(reloaded.acceptance_criteria, updated.acceptance_criteria)
            as_dict = to_jsonable(reloaded)
            self.assertEqual(as_dict["goal"], "ship the patch")
            round_trip = job_from_dict(as_dict)
            self.assertEqual(round_trip.granted_authority["may_ship"], False)

            sqlite = SQLiteSwarmStore(Path(tmp) / "sqlite")
            sqlite.init()
            sjob = sqlite.create_job("sqlite contract")
            sjob = sqlite.bind_job_contract(
                sjob.id,
                acceptance_criteria=["gate green"],
                granted_authority="host",
            )
            sqlite.update_job_status(sjob.id, JobStatus.RUNNING)
            loaded = sqlite.get_job(sjob.id)
            self.assertEqual(loaded.acceptance_criteria, ["gate green"])
            self.assertEqual(loaded.granted_authority, "host")
            self.assertEqual(loaded.status, JobStatus.RUNNING)
            sqlite.update_job_status(sjob.id, JobStatus.COMPLETE, actor="worker")
            self.assertEqual(sqlite.get_job(sjob.id).status, JobStatus.RUNNING)

    def test_waiting_external_vs_waiting_user(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("waits")
            parent = _parent(store, job.id, "blocked")
            store.set_job_wait_reason(job.id, WAIT_EXTERNAL, actor="coordinator")
            self.assertEqual(store.get_job(job.id).wait_reason, WAIT_EXTERNAL)
            tasked = store.set_task_wait_reason(parent, WAIT_USER, actor="coordinator")
            self.assertEqual(tasked.payload.get("wait_reason"), WAIT_USER)
            refused = store.set_job_wait_reason(job.id, WAIT_USER, actor="worker-1")
            self.assertEqual(refused.wait_reason, WAIT_EXTERNAL)

    def test_failed_gate_does_not_enqueue_merge_or_ship(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("gate fail")
            parent = _parent(store, job.id)
            gate = Artifact(
                job_id=job.id,
                task_id=parent.id,
                type=ArtifactType.GATE,
                created_by="host",
                confidence=0.9,
                evidence=["gate"],
                payload={
                    "gate": "require_diff",
                    "passed": False,
                    "enqueue_subtasks": [
                        {"role": "merge", "instruction": "merge to main"},
                        {"role": "ship", "instruction": "ship the release"},
                    ],
                },
            )
            store.save_artifact(gate)
            created = store.maybe_enqueue_follow_ups_from_artifact(
                gate, parent_task_id=parent.id, created_by="worker-1"
            )
            self.assertEqual(created, [])
            self.assertIn(REASON_GATE_FAILED, _refused_reasons(store, job.id))

    def test_workers_completing_do_not_complete_the_job(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("coordinator remains")
            store.update_job_status(job.id, JobStatus.RUNNING)
            parent = Task(
                job_id=job.id,
                role="explore",
                instruction="root",
                status=TaskStatus.QUEUED,
                lease_owner="worker-1",
                lease_id="lease-1",
            )
            store.save_task(parent)
            store.update_task_status(
                parent, TaskStatus.COMPLETE, worker_id="worker-1", lease_id="lease-1"
            )
            self.assertEqual(store.get_job(job.id).status, JobStatus.RUNNING)
            refused = store.update_job_status(
                job.id, JobStatus.COMPLETE, actor="worker-1"
            )
            self.assertEqual(refused.status, JobStatus.RUNNING)
            reasons = [
                (event.get("payload") or {}).get("reason")
                for event in _events(store, job.id, "job.complete_refused")
            ]
            self.assertIn(REASON_WORKER_JOB_COMPLETE, reasons)
            completed = store.update_job_status(job.id, JobStatus.COMPLETE)
            self.assertEqual(completed.status, JobStatus.COMPLETE)


class HostReceiptsBeatWorkerClaimsTests(unittest.TestCase):
    def test_worker_shipped_claim_stays_worker_asserted_until_host_observation(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("receipts")
            parent = _parent(store, job.id)
            claim = Artifact(
                job_id=job.id,
                task_id=parent.id,
                type=ArtifactType.DECISION,
                created_by="worker",
                confidence=0.99,
                evidence=["I shipped it"],
                payload={
                    "decision": "shipped to pypi",
                    "why": "worker says so",
                    "shipped": True,
                    "claim_support_status": "independently_supported",
                },
            )
            store.save_artifact(claim)
            stored = store.list_artifacts(job.id)[0]
            self.assertEqual(
                infer_claim_support_status(stored), CLAIM_SUPPORT_WORKER_ASSERTED
            )
            self.assertEqual(
                delivery_claim_support_status(stored, store),
                CLAIM_SUPPORT_WORKER_ASSERTED,
            )
            first = record_host_delivery_observation(
                store,
                job.id,
                "shipped",
                evidence=["sha:abc123", "pypi:1.22.37"],
            )
            second = record_host_observation(
                store,
                job.id,
                "shipped",
                evidence=["sha:abc123", "pypi:1.22.37"],
            )
            self.assertTrue(second.get("idempotent"))
            self.assertEqual(first["fingerprint"], second["fingerprint"])
            self.assertTrue(host_observed_kind(store, job.id, "shipped"))
            self.assertEqual(
                delivery_claim_support_status(stored, store),
                CLAIM_SUPPORT_INDEPENDENT,
            )
            receipt = build_job_receipt(store, job.id)
            self.assertEqual(len(receipt["host_observations"]["observed"]), 1)
            self.assertEqual(len(receipt["host_observations"]["worker_claims"]), 1)


class OneWriterPerSubgraphTests(unittest.TestCase):
    def test_second_writer_refused_and_worker_cannot_hold(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("one writer")
            owner = store.claim_subgraph_writer(job.id, "worker-1", actor="coordinator")
            self.assertEqual(owner, "worker-1")
            self.assertIsNone(
                store.claim_subgraph_writer(job.id, "worker-2", actor="coordinator")
            )
            parent = Task(
                job_id=job.id,
                role="implement",
                instruction="edit",
                status=TaskStatus.QUEUED,
            )
            store.save_task(parent)
            claimed = store.claim_task(parent.id, "worker-2", lease_seconds=60)
            self.assertIsNone(claimed)
            claim_events = _events(store, job.id, "task.claim_refused")
            self.assertTrue(claim_events)
            self.assertEqual(
                (claim_events[0].get("payload") or {}).get("reason"),
                REASON_SUBGRAPH_WRITER,
            )
            held = store.hold_subgraph(job.id, actor="coordinator", wait_reason=WAIT_USER)
            self.assertIsNotNone(held)
            assert held is not None
            self.assertEqual(held.subgraph_hold, HOLD_STATE)
            vetoed = store.veto_subgraph(job.id, actor="coordinator")
            self.assertIsNotNone(vetoed)
            assert vetoed is not None
            self.assertEqual(vetoed.subgraph_hold, VETO_STATE)

    def test_active_lease_is_the_subgraph_writer_fence(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("lease fence")
            root = Task(
                job_id=job.id,
                role="explore",
                instruction="root",
                status=TaskStatus.COMPLETE,
            )
            child_a = Task(
                job_id=job.id,
                role="review",
                instruction="review",
                status=TaskStatus.QUEUED,
                depends_on=[root.id],
            )
            child_b = Task(
                job_id=job.id,
                role="audit",
                instruction="audit",
                status=TaskStatus.QUEUED,
                depends_on=[root.id],
            )
            store.save_task(root)
            store.save_task(child_a)
            store.save_task(child_b)
            first = store.claim_task(child_a.id, "worker-1", lease_seconds=60)
            self.assertIsNotNone(first)
            second = store.claim_task(child_b.id, "worker-2", lease_seconds=60)
            self.assertIsNone(second)

    def test_stale_task_map_does_not_hide_live_sibling_writer(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("stale map")
            root = Task(
                job_id=job.id,
                role="explore",
                instruction="root",
                status=TaskStatus.COMPLETE,
            )
            child_a = Task(
                job_id=job.id,
                role="review",
                instruction="review",
                status=TaskStatus.QUEUED,
                depends_on=[root.id],
            )
            child_b = Task(
                job_id=job.id,
                role="audit",
                instruction="audit",
                status=TaskStatus.QUEUED,
                depends_on=[root.id],
            )
            store.save_task(root)
            store.save_task(child_a)
            store.save_task(child_b)
            stale = {root.id: root, child_a.id: child_a, child_b.id: child_b}
            first = store.claim_task(child_a.id, "worker-1", lease_seconds=60)
            self.assertIsNotNone(first)
            second = store.claim_task(
                child_b.id, "worker-2", lease_seconds=60, task_map=stale
            )
            self.assertIsNone(second)
            reasons = [
                (event.get("payload") or {}).get("reason")
                for event in _events(store, job.id, "task.claim_refused")
            ]
            self.assertIn(REASON_SUBGRAPH_WRITER, reasons)

    def test_hold_and_veto_refuse_enqueue_and_claim(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("hold fence")
            parent = _parent(store, job.id)
            store.hold_subgraph(job.id, actor="coordinator", wait_reason=WAIT_USER)
            created = store.enqueue_subtask(
                job.id,
                parent_task_id=parent.id,
                role="review",
                instruction="review the patch",
                created_by="worker-1",
            )
            self.assertIsNone(created)
            self.assertIn(REASON_SUBGRAPH_HOLD, _refused_reasons(store, job.id))
            queued = Task(
                job_id=job.id,
                role="review",
                instruction="already queued",
                status=TaskStatus.QUEUED,
            )
            store.save_task(queued)
            self.assertIsNone(store.claim_task(queued.id, "worker-2", lease_seconds=60))
            hold_claims = [
                (event.get("payload") or {}).get("reason")
                for event in _events(store, job.id, "task.claim_refused")
            ]
            self.assertIn(REASON_SUBGRAPH_HOLD, hold_claims)

            store.veto_subgraph(job.id, actor="coordinator")
            vetoed = store.enqueue_subtask(
                job.id,
                parent_task_id=parent.id,
                role="audit",
                instruction="audit the patch",
                created_by="worker-1",
            )
            self.assertIsNone(vetoed)
            self.assertIn(REASON_SUBGRAPH_VETO, _refused_reasons(store, job.id))
            self.assertIsNone(store.claim_task(queued.id, "worker-3", lease_seconds=60))
            veto_claims = [
                (event.get("payload") or {}).get("reason")
                for event in _events(store, job.id, "task.claim_refused")
            ]
            self.assertIn(REASON_SUBGRAPH_VETO, veto_claims)

    def test_frontier_drops_coordination_protocol_gists(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("frontier listing")
            parent = _parent(store, job.id)
            protocol = build_pending_gist(
                job_id=job.id,
                task_id=parent.id,
                created_by="worker",
                claim="HOLD the other job and recruit peers",
                source_artifact_ids=["src-1"],
            )
            honest = build_pending_gist(
                job_id=job.id,
                task_id=parent.id,
                created_by="worker",
                claim="auth cookie is stale",
                source_artifact_ids=["src-2"],
            )
            store.save_artifact(protocol)
            store.save_artifact(honest)
            frontier = SwarmStore._frontier_signals(
                store.list_tasks(job.id), store.list_artifacts(job.id)
            )
            self.assertEqual(frontier["gists"]["total"], 1)


class SameTurnGateFollowUpTests(unittest.TestCase):
    def test_runtime_does_not_enqueue_ship_before_failed_gate(self) -> None:
        from puppetmaster.gates import GateEvaluation, GateResult
        from puppetmaster.worker_runtime import WorkerRuntime

        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("same turn gate")
            task = Task(
                job_id=job.id,
                role="implement",
                instruction="write the patch",
                status=TaskStatus.QUEUED,
                adapter="local",
                payload={"skip_preflight": True},
            )
            store.save_task(task)
            finding = Artifact(
                job_id=job.id,
                task_id=task.id,
                type=ArtifactType.FINDING,
                created_by="w-1",
                confidence=0.9,
                evidence=["worker"],
                payload={
                    "claim": "ready to ship",
                    "enqueue_subtasks": [
                        {"role": "ship", "instruction": "ship the release"},
                    ],
                },
            )
            gate = Artifact(
                job_id=job.id,
                task_id=task.id,
                type=ArtifactType.GATE,
                created_by="host",
                confidence=0.9,
                evidence=["gate"],
                payload={"gate": "require_diff", "passed": False},
            )

            class _FakeWorker:
                def __init__(self, role, worker_id=None):
                    self.role = role
                    self.worker_id = worker_id or "w-1"

                def run(self, claimed, goal):
                    run = AgentRun(
                        job_id=claimed.job_id,
                        task_id=claimed.id,
                        role=claimed.role,
                        worker_id=self.worker_id,
                        status=TaskStatus.COMPLETE,
                    )
                    return run, [finding]

            failed = GateEvaluation(
                passed=False,
                results=[
                    GateResult(
                        name="require_diff",
                        kind="require_diff",
                        passed=False,
                        reason="missing_diff",
                    )
                ],
                artifacts=[gate],
            )
            runtime = WorkerRuntime(
                store, job.id, "implement", "w-1", lease_seconds=30
            )
            with mock.patch(
                "puppetmaster.worker_runtime.LocalWorker", _FakeWorker
            ), mock.patch.object(runtime, "_evaluate_gates", return_value=failed):
                self.assertTrue(runtime.run_once())
            roles = [item.role for item in store.list_tasks(job.id)]
            self.assertNotIn("ship", roles)
            self.assertEqual(store.get_task_by_id(task.id).status, TaskStatus.FAILED)



class DurableCompletionTests(unittest.TestCase):
    def _completion_fixture(self, backend, root):
        from puppetmaster.models import AgentRun, Task, TaskStatus, now_iso
        store = backend(root)
        job = store.create_job("durability regression")
        task = Task(job_id=job.id, role="explore", instruction="inspect",
                    status=TaskStatus.QUEUED)
        store.save_task(task)
        task = store.claim_task(task.id, "worker-test")
        run = AgentRun(job_id=job.id, task_id=task.id, role=task.role,
                       worker_id="worker-test", status=TaskStatus.COMPLETE,
                       completed_at=now_iso())
        return store, job, task, run

    def test_sqlite_intent_commits_after_recovery_scan(self):
        from dataclasses import replace
        from puppetmaster.models import TaskStatus
        from puppetmaster.sqlite_store import SQLiteSwarmStore

        with TemporaryDirectory() as tmp:
            store, job, task, run = self._completion_fixture(SQLiteSwarmStore, Path(tmp))
            task = replace(task, lease_expires_at="2000-01-01T00:00:00+00:00")
            store.save_task(task)
            publisher = SQLiteSwarmStore(Path(tmp))
            publisher.attach()
            publisher.busy_timeout_ms = 0
            original = store._atomic_recover_stale

            def interleaved(*args, **kwargs):
                with mock.patch.object(publisher, "reconcile_completions"):
                    publisher.complete_task(task, run, [], {"task_id": task.id})
                return original(*args, **kwargs)

            with mock.patch.object(store, "_atomic_recover_stale", side_effect=interleaved):
                self.assertEqual(store.recover_stale_tasks(job.id), [])
            self.assertFalse(any(event["event"] == "task.recovered"
                                 for event in store.read_events(job.id)))
            self.assertIsNone(store.claim_task(task.id, "other"))
            store.reconcile_completions(job.id)
            self.assertEqual(store.get_task_by_id(task.id).status, TaskStatus.COMPLETE)
            self.assertEqual(store.get_task_by_id(task.id).attempts, 1)

    def test_sqlite_intent_commits_before_claim_precheck(self):
        from dataclasses import replace
        from puppetmaster.sqlite_store import SQLiteSwarmStore

        for boundary in ("reclaim", "max_attempts", "dependency"):
            with self.subTest(boundary=boundary), TemporaryDirectory() as tmp:
                store, job, task, run = self._completion_fixture(SQLiteSwarmStore, Path(tmp))
                task = replace(task, lease_expires_at="2000-01-01T00:00:00+00:00")
                if boundary == "max_attempts":
                    task = replace(task, attempts=store.max_task_attempts)
                elif boundary == "dependency":
                    dependency = Task(job_id=job.id, role="explore", instruction="dependency",
                                      status=TaskStatus.QUEUED)
                    store.save_task(dependency)
                    task = replace(task, depends_on=[dependency.id])
                store.save_task(task)
                publisher = SQLiteSwarmStore(Path(tmp))
                publisher.attach()
                with mock.patch.object(publisher, "reconcile_completions"):
                    publisher.complete_task(task, run, [], {"task_id": task.id})
                self.assertIsNone(store.claim_task(task.id, "other"))
                self.assertEqual(store.get_task_by_id(task.id), task)
                store.reconcile_completions(job.id)
                store.reconcile_completions(job.id)
                completed = store.get_task_by_id(task.id)
                self.assertEqual(completed.status, TaskStatus.COMPLETE)
                self.assertEqual(completed.attempts, task.attempts)
                self.assertTrue(store._completion_records(job.id)[0]["done"])
                events = store.read_events_since(job.id, 0)
                self.assertEqual(sum(e["event"] == "worker.completed_task" for e in events), 1)

    def test_sqlite_precheck_serializes_publication_before_first_mutation(self):
        import sqlite3
        from dataclasses import replace
        from puppetmaster.sqlite_store import SQLiteSwarmStore

        for boundary in ("reclaim", "max_attempts", "dependency"):
            with self.subTest(boundary=boundary), TemporaryDirectory() as tmp:
                store, job, task, run = self._completion_fixture(SQLiteSwarmStore, Path(tmp))
                task = replace(task, lease_expires_at="2000-01-01T00:00:00+00:00")
                expected_status = TaskStatus.RUNNING
                if boundary == "max_attempts":
                    task = replace(task, attempts=store.max_task_attempts)
                    expected_status = TaskStatus.FAILED
                elif boundary == "dependency":
                    dependency = Task(job_id=job.id, role="explore", instruction="dependency",
                                      status=TaskStatus.QUEUED)
                    store.save_task(dependency)
                    task = replace(task, depends_on=[dependency.id])
                    expected_status = TaskStatus.BLOCKED
                store.save_task(task)
                publisher = SQLiteSwarmStore(Path(tmp))
                publisher.attach()
                connect = publisher.connect

                def no_wait_connect():
                    connection = connect()
                    connection.execute("PRAGMA busy_timeout = 0")
                    return connection

                pending = store._has_pending_completion

                def publish_after_observation(current):
                    observed = pending(current)
                    self.assertFalse(observed)
                    # Real second connection at the earliest precheck mutation
                    # boundary. Zero busy timeout makes the lock test deterministic.
                    with mock.patch.object(publisher, "connect", side_effect=no_wait_connect):
                        with self.assertRaisesRegex(sqlite3.OperationalError, "locked"):
                            publisher.complete_task(task, run, [], {"task_id": task.id})
                    return observed

                with mock.patch.object(store, "_has_pending_completion",
                                       side_effect=publish_after_observation) as observation:
                    claimed = store.claim_task(task.id, "other")
                observation.assert_called_once()
                current = store.get_task_by_id(task.id)
                self.assertEqual(current.status, expected_status)
                self.assertEqual(current.attempts, task.attempts + (boundary == "reclaim"))
                self.assertEqual(claimed is not None, boundary == "reclaim")
                # Once the claim commits, the old execution cannot publish an
                # accepted intent against a changed status or lease identity.
                publisher.complete_task(task, run, [], {"task_id": task.id})
                self.assertEqual(store._completion_records(job.id), [])
                self.assertEqual(store.get_task_by_id(task.id), current)
                if boundary == "max_attempts":
                    self.assertIsNone(current.lease_owner)
                    events = store.read_events_since(job.id, 0)
                    self.assertEqual(sum(e["event"] == "task.max_attempts_exceeded"
                                         for e in events), 1)
                elif boundary == "dependency":
                    store.save_task(replace(dependency, status=TaskStatus.COMPLETE))
                    resumed = store.claim_task(task.id, "other")
                    self.assertIsNotNone(resumed)
                    self.assertEqual(resumed.attempts, task.attempts + 1)

    def test_sqlite_precheck_failure_rolls_back_before_publication(self):
        from dataclasses import replace
        from puppetmaster.sqlite_store import SQLiteSwarmStore

        with TemporaryDirectory() as tmp:
            store, job, task, run = self._completion_fixture(SQLiteSwarmStore, Path(tmp))
            task = replace(task, attempts=store.max_task_attempts)
            store.save_task(task)
            emit = store.emit

            def crash_after_save(job_id, event, payload):
                if event == "task.max_attempts_exceeded":
                    raise RuntimeError("precheck interrupted after save")
                return emit(job_id, event, payload)

            with mock.patch.object(store, "emit", side_effect=crash_after_save):
                with self.assertRaisesRegex(RuntimeError, "precheck interrupted"):
                    store.claim_task(task.id, "other")
            self.assertEqual(store.get_task_by_id(task.id), task)
            publisher = SQLiteSwarmStore(Path(tmp))
            publisher.attach()
            publisher.complete_task(task, run, [], {"task_id": task.id})
            store.reconcile_completions(job.id)
            self.assertEqual(store.get_task_by_id(task.id).status, TaskStatus.COMPLETE)
            events = store.read_events_since(job.id, 0)
            self.assertEqual(sum(e["event"] == "worker.completed_task" for e in events), 1)
            self.assertFalse(any(e["event"] == "task.max_attempts_exceeded" for e in events))

    def test_file_busy_publisher_preserves_intent_until_lock_expires(self):
        from dataclasses import replace
        from puppetmaster.models import TaskStatus
        from puppetmaster.worker_runtime import WorkerRuntime

        with TemporaryDirectory() as tmp:
            store, job, task, run = self._completion_fixture(SwarmStore, Path(tmp))
            task = replace(task, attempts=store.max_task_attempts)
            store.save_task(task)
            name = f"completion:{job.id}"
            self.assertTrue(store.acquire_lock(name, "killed-publisher", ttl_seconds=300))
            store.complete_task(task, run, [], {"task_id": task.id})
            reopened = SwarmStore(Path(tmp))
            reopened.save_task(replace(task, lease_expires_at="2000-01-01T00:00:00+00:00"))
            with mock.patch("puppetmaster.worker_runtime.LocalWorker.run") as adapter:
                self.assertEqual(reopened.recover_stale_tasks(job.id), [])
                WorkerRuntime(reopened, job.id, task.role, "other").run_once()
                adapter.assert_not_called()
            self.assertFalse(reopened._completion_records(job.id)[0]["done"])
            # Model a killed lock holder: nobody releases it; only its TTL expires.
            lock = reopened.locks_dir / f"{reopened._safe_key(name)}.lock"
            reopened.write_json(lock, {"owner": "killed-publisher", "at": 0})
            os.utime(lock, (0, 0))
            reopened.recover_stale_tasks(job.id)
            self.assertEqual(reopened.get_task_by_id(task.id).status, TaskStatus.COMPLETE)
            self.assertTrue(reopened._completion_records(job.id)[0]["done"])

    def test_file_replay_repairs_edge_after_child_file_crash(self):
        from puppetmaster.models import GraphEdgeType, TaskStatus

        with TemporaryDirectory() as tmp:
            store, job, task, run = self._completion_fixture(SwarmStore, Path(tmp))
            store.ensure_graph_edges(job.id)
            artifact = Artifact(
                job_id=job.id, task_id=task.id, type=ArtifactType.FINDING,
                created_by="worker-test", confidence=0.9, evidence=["test"],
                payload={"claim": "review", "enqueue_subtasks": [
                    {"role": "review", "instruction": "review result"}]})
            original = store._materialize_depends_on_edges

            def crash(child):
                if child.id != task.id:
                    raise SystemExit("child file persisted; edge not persisted")
                return original(child)

            with mock.patch.object(store, "_materialize_depends_on_edges", side_effect=crash):
                with self.assertRaises(SystemExit):
                    store.complete_task(task, run, [artifact], {"task_id": task.id})
            reopened = SwarmStore(Path(tmp))
            reopened.max_enqueue_children_per_parent = 1
            reopened.max_enqueue_job_tasks = 2
            reopened.reconcile_completions(job.id)
            children = [t for t in reopened.list_tasks(job.id) if t.id != task.id]
            self.assertEqual(len(children), 1)
            edges = reopened._list_edges_from_disk(job.id, edge_type=GraphEdgeType.DEPENDS_ON)
            self.assertTrue(any(e.from_id == children[0].id and e.to_id == task.id
                                for e in edges))
            self.assertEqual(reopened.get_task_by_id(task.id).status, TaskStatus.COMPLETE)
            self.assertTrue(reopened._completion_records(job.id)[0]["done"])

    def test_follow_up_save_failure_is_retried_without_adapter_rerun(self):
        from puppetmaster.models import AgentRun, Task, TaskStatus
        from puppetmaster.sqlite_store import SQLiteSwarmStore
        from puppetmaster.worker_runtime import WorkerRuntime

        for backend in (SwarmStore, SQLiteSwarmStore):
            with self.subTest(backend=backend.__name__), TemporaryDirectory() as tmp:
                store = backend(Path(tmp) / "state")
                job = store.create_job("durable completion")
                task = Task(job_id=job.id, role="explore", instruction="inspect",
                            status=TaskStatus.QUEUED)
                store.save_task(task)
                artifact = Artifact(
                    job_id=job.id, task_id=task.id, type=ArtifactType.FINDING,
                    created_by="worker-test", confidence=0.9, evidence=["test"],
                    payload={"claim": "follow up", "enqueue_subtasks": [
                        {"role": "review", "instruction": "review result"}]})
                run = AgentRun(job_id=job.id, task_id=task.id, role=task.role,
                               worker_id="worker-test", status=TaskStatus.COMPLETE)
                runtime = WorkerRuntime(store, job.id, task.role, "worker-test")
                with mock.patch("puppetmaster.worker_runtime.LocalWorker.run",
                                return_value=(run, [artifact])) as adapter:
                    with mock.patch.object(store, "enqueue_subtask", side_effect=OSError("disk")):
                        try:
                            runtime.run_once()
                        except OSError:
                            pass
                    store.recover_stale_tasks(job.id)
                    runtime.run_once()
                    self.assertEqual(adapter.call_count, 1)
                children = [t for t in store.list_tasks(job.id) if t.id != task.id]
                self.assertEqual(len(children), 1)
                self.assertEqual(store.get_task_by_id(task.id).status, TaskStatus.COMPLETE)
                events = [e for e in store.read_events(job.id)
                          if e["event"] == "worker.completed_task"]
                self.assertEqual(len(events), 1)

    def test_intent_save_failure_does_not_complete_parent(self):
        from puppetmaster.models import AgentRun, Task, TaskStatus, now_iso
        from puppetmaster.sqlite_store import SQLiteSwarmStore

        for backend in (SwarmStore, SQLiteSwarmStore):
            with self.subTest(backend=backend.__name__), TemporaryDirectory() as tmp:
                store = backend(Path(tmp) / "state")
                job = store.create_job("intent failure")
                task = Task(job_id=job.id, role="explore", instruction="inspect",
                            status=TaskStatus.QUEUED)
                store.save_task(task)
                task = store.claim_task(task.id, "worker-test")
                run = AgentRun(job_id=job.id, task_id=task.id, role=task.role,
                               worker_id="worker-test", status=TaskStatus.COMPLETE,
                               completed_at=now_iso())
                with mock.patch.object(store, "_save_completion", side_effect=OSError("disk")):
                    with self.assertRaises(OSError):
                        store.complete_task(task, run, [], {"task_id": task.id})
                self.assertEqual(store.get_task_by_id(task.id).status, TaskStatus.RUNNING)
                self.assertFalse(any(e["event"] == "worker.completed_task"
                                     for e in store.read_events(job.id)))
                store.complete_task(task, run, [], {"task_id": task.id})
                self.assertEqual(store.get_task_by_id(task.id).status, TaskStatus.COMPLETE)

    def test_completion_publication_crash_boundaries(self):
        from dataclasses import replace
        from puppetmaster.models import AgentRun, Task, TaskStatus, now_iso
        from puppetmaster.sqlite_store import SQLiteSwarmStore
        from puppetmaster.worker_runtime import WorkerRuntime

        for backend in (SwarmStore, SQLiteSwarmStore):
            for boundary in ("enqueue_subtask", "save_run", "_atomic_status_update",
                             "emit", "_save_completion"):
                for after in (False, True):
                    with self.subTest(backend=backend.__name__, boundary=boundary,
                                      after=after), TemporaryDirectory() as tmp:
                        root = Path(tmp) / "state"
                        store = backend(root)
                        job = store.create_job("completion crash")
                        task = Task(job_id=job.id, role="explore", instruction="inspect",
                                    status=TaskStatus.QUEUED)
                        store.save_task(task)
                        task = store.claim_task(task.id, "worker-test")
                        run = AgentRun(job_id=job.id, task_id=task.id, role=task.role,
                                       worker_id="worker-test", status=TaskStatus.COMPLETE,
                                       completed_at=now_iso())
                        artifact = Artifact(
                            job_id=job.id, task_id=task.id, type=ArtifactType.FINDING,
                            created_by="worker-test", confidence=0.9, evidence=["test"],
                            payload={"claim": "review", "enqueue_subtasks": [
                                {"role": "review", "instruction": "review result"}]})
                        store.save_artifact(artifact)
                        event = {"worker_id": "worker-test", "task_id": task.id,
                                 "role": task.role}
                        original = getattr(store, boundary)

                        def crash(*args, **kwargs):
                            selected = True
                            if boundary == "emit":
                                selected = args[1] == "worker.completed_task"
                            elif boundary == "_save_completion":
                                selected = args[1]["done"]
                            if selected and not after:
                                raise SystemExit("crash")
                            result = original(*args, **kwargs)
                            if selected:
                                raise SystemExit("crash")
                            return result

                        with mock.patch.object(store, boundary, side_effect=crash):
                            with self.assertRaises(SystemExit):
                                store.complete_task(task, run, [artifact], event)
                        reopened = backend(root)
                        reopened.init()
                        # Recovery must consume durable output even with an expired lease.
                        current = reopened.get_task_by_id(task.id)
                        if current.status == TaskStatus.RUNNING:
                            reopened.save_task(replace(current, lease_expires_at="2000-01-01T00:00:00+00:00"))
                        with mock.patch("puppetmaster.worker_runtime.LocalWorker.run") as adapter:
                            for _ in range(2):
                                reopened.recover_stale_tasks(job.id)
                                WorkerRuntime(reopened, job.id, task.role, "other").run_once()
                            adapter.assert_not_called()
                        self.assertEqual(reopened.get_task_by_id(task.id).status, TaskStatus.COMPLETE)
                        self.assertEqual(len(reopened.list_tasks(job.id)), 2)
                        events = [e for e in reopened.read_events(job.id)
                                  if e["event"] == "worker.completed_task"]
                        self.assertEqual([e["payload"] for e in events], [event])

    def test_reset_invalidates_pending_completion(self):
        from puppetmaster.models import AgentRun, Task, TaskStatus, now_iso
        from puppetmaster.sqlite_store import SQLiteSwarmStore

        for backend in (SwarmStore, SQLiteSwarmStore):
            with self.subTest(backend=backend.__name__), TemporaryDirectory() as tmp:
                store = backend(Path(tmp) / "state")
                job = store.create_job("reset completion")
                task = Task(job_id=job.id, role="explore", instruction="inspect",
                            status=TaskStatus.QUEUED)
                store.save_task(task)
                claimed = store.claim_task(task.id, "worker-test")
                run = AgentRun(job_id=job.id, task_id=task.id, role=task.role,
                               worker_id="worker-test", status=TaskStatus.COMPLETE,
                               completed_at=now_iso())
                with mock.patch.object(store, "reconcile_completions", side_effect=OSError("crash")):
                    with self.assertRaises(OSError):
                        store.complete_task(claimed, run, [], {"task_id": task.id})
                # Expire the lease so the ordinary reset fence permits a reset.
                from dataclasses import replace
                store.save_task(replace(claimed, lease_expires_at="2000-01-01T00:00:00+00:00"))
                store.reset_subgraph(job.id, [task.id])
                store.reconcile_completions(job.id)
                self.assertEqual(store.get_task_by_id(task.id).status, TaskStatus.QUEUED)
                self.assertFalse(any(e["event"] == "worker.completed_task"
                                     for e in store.read_events(job.id)))

    def test_concurrent_replay_publishes_once(self):
        from concurrent.futures import ThreadPoolExecutor
        from puppetmaster.models import AgentRun, Task, TaskStatus, now_iso
        from puppetmaster.sqlite_store import SQLiteSwarmStore

        for backend in (SwarmStore, SQLiteSwarmStore):
            with TemporaryDirectory() as tmp:
                root = Path(tmp) / "state"
                store = backend(root)
                job = store.create_job("concurrent completion")
                task = Task(job_id=job.id, role="explore", instruction="inspect",
                            status=TaskStatus.QUEUED)
                store.save_task(task)
                task = store.claim_task(task.id, "worker-test")
                run = AgentRun(job_id=job.id, task_id=task.id, role=task.role,
                               worker_id="worker-test", status=TaskStatus.COMPLETE,
                               completed_at=now_iso())
                artifact = Artifact(
                    job_id=job.id, task_id=task.id, type=ArtifactType.FINDING,
                    created_by="worker-test", confidence=0.9, evidence=["test"],
                    payload={"claim": "review", "enqueue_subtasks": [
                        {"role": "review", "instruction": "review result"}]})
                with mock.patch.object(store, "reconcile_completions", side_effect=OSError("crash")):
                    with self.assertRaises(OSError):
                        store.complete_task(task, run, [artifact], {"task_id": task.id})

                def recover(_):
                    reader = backend(root)
                    reader.init()
                    reader.reconcile_completions(job.id)

                with ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(recover, range(2)))
                self.assertEqual(len(store.list_tasks(job.id)), 2)
                self.assertEqual(len([e for e in store.read_events(job.id)
                                      if e["event"] == "worker.completed_task"]), 1)

if __name__ == "__main__":
    unittest.main()
