"""Focused tests for the durable execution/provenance graph vertical slice."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import io
import json
import os
import sqlite3
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from puppetmaster.adapters._prompts import with_job_brief, with_prewalk_plan
from puppetmaster.adapters.local import LocalAdapter
from puppetmaster.cli._dispatch import _main
from puppetmaster.diagnostics import run_doctor
from puppetmaster.models import (
    Artifact,
    ArtifactType,
    GraphEdgeType,
    GraphNodeKind,
    Task,
    TaskStatus,
    graph_edge_identity,
    make_graph_edge,
    seconds_from_now,
    to_jsonable,
)
from puppetmaster.prewalk import (
    IMPLEMENT_ROLE,
    PLAN_ROLE,
    PREWALK_PLAN_SECTION_HEADER,
    PREWALK_UPSTREAM_SECTION_HEADER,
    VERIFY_ROLE,
    build_prewalk_specs,
)
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.store import ActiveTaskLeaseError, SwarmStore, _CONSUMES_JOURNAL_PREFIX
from puppetmaster.store_factory import create_store

_NON_LOCAL_ADAPTERS = (
    "hermes",
    "claude-code",
    "codex",
    "agentic",
    "openai",
    "cursor",
)

def _decision_artifact(job_id: str, task_id: str, decision: str) -> Artifact:
    return Artifact(
        job_id=job_id,
        task_id=task_id,
        type=ArtifactType.DECISION,
        created_by="plan",
        payload={"decision": decision, "why": "test"},
        confidence=0.9,
        evidence=["test"],
    )

def _failure_artifact(
    job_id: str, task_id: str, failure: str
) -> Artifact:
    return Artifact(
        job_id=job_id,
        task_id=task_id,
        type=ArtifactType.RISK,
        created_by="worker",
        payload={
            "risk": failure,
            "mitigation": "retry",
            "failure": failure,
        },
        confidence=0.9,
        evidence=["test"],
    )

class GraphEdgeIdentityTests(unittest.TestCase):
    def test_identity_is_stable_and_idempotent(self) -> None:
        first = graph_edge_identity(
            "job1", "depends_on", "task", "a", "task", "b"
        )
        second = graph_edge_identity(
            "job1", "depends_on", "task", "a", "task", "b"
        )
        third = graph_edge_identity(
            "job1", "depends_on", "task", "a", "task", "c"
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertTrue(first.startswith("edge_"))

class GraphStoreParityTests(unittest.TestCase):
    def _backends(self, root: Path):
        return (
            ("file", SwarmStore(root / "file")),
            ("sqlite", SQLiteSwarmStore(root / "sqlite")),
        )

    def test_edge_upsert_list_idempotent_file_and_sqlite(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for _name, store in self._backends(root):
                store.init()
                job = store.create_job("graph edge parity")
                plan = Task(
                    job_id=job.id,
                    role=PLAN_ROLE,
                    instruction="plan",
                    status=TaskStatus.COMPLETE,
                )
                implement = Task(
                    job_id=job.id,
                    role=IMPLEMENT_ROLE,
                    instruction="implement",
                    status=TaskStatus.BLOCKED,
                    depends_on=[plan.id],
                )
                store.save_task(plan)
                store.save_task(implement)
                artifact = _decision_artifact(job.id, plan.id, "Touch x.py")
                store.save_artifact(artifact)

                depends = store.list_edges(
                    job.id, edge_type=GraphEdgeType.DEPENDS_ON
                )
                produces = store.list_edges(
                    job.id, edge_type=GraphEdgeType.PRODUCES
                )
                self.assertEqual(len(depends), 1)
                self.assertEqual(depends[0].from_id, implement.id)
                self.assertEqual(depends[0].to_id, plan.id)
                self.assertEqual(len(produces), 1)
                self.assertEqual(produces[0].to_id, artifact.id)

                # Idempotent re-save keeps a single edge identity/created_at.
                created_at = depends[0].created_at
                store.save_task(implement)
                store.save_artifact(artifact)
                depends_again = store.list_edges(
                    job.id, edge_type=GraphEdgeType.DEPENDS_ON
                )
                produces_again = store.list_edges(
                    job.id, edge_type=GraphEdgeType.PRODUCES
                )
                self.assertEqual(len(depends_again), 1)
                self.assertEqual(depends_again[0].id, depends[0].id)
                self.assertEqual(depends_again[0].created_at, created_at)
                self.assertEqual(len(produces_again), 1)
                self.assertEqual(produces_again[0].id, produces[0].id)

                got = store.get_edge(job.id, depends[0].id)
                self.assertIsNotNone(got)
                assert got is not None
                self.assertEqual(got.type, GraphEdgeType.DEPENDS_ON)

    def test_sqlite_migration_backfills_v1_edges(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "legacy"
            root.mkdir()
            db_path = root / "state.sqlite3"
            # Hand-roll a v1 database (no graph_edges, schema_version=1).
            connection = sqlite3.connect(db_path)
            try:
                connection.executescript(
                    """
                    CREATE TABLE jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                    CREATE TABLE tasks (
                      id TEXT PRIMARY KEY,
                      job_id TEXT NOT NULL,
                      role TEXT NOT NULL,
                      status TEXT NOT NULL,
                      data TEXT NOT NULL
                    );
                    CREATE TABLE artifacts (
                      id TEXT PRIMARY KEY,
                      job_id TEXT NOT NULL,
                      task_id TEXT NOT NULL,
                      type TEXT NOT NULL,
                      data TEXT NOT NULL
                    );
                    CREATE TABLE runs (
                      id TEXT PRIMARY KEY,
                      job_id TEXT NOT NULL,
                      task_id TEXT NOT NULL,
                      data TEXT NOT NULL
                    );
                    CREATE TABLE memory (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                    CREATE TABLE events (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      job_id TEXT NOT NULL,
                      at TEXT NOT NULL,
                      event TEXT NOT NULL,
                      payload TEXT NOT NULL
                    );
                    CREATE TABLE metadata (
                      key TEXT PRIMARY KEY,
                      value TEXT NOT NULL
                    );
                    INSERT INTO metadata(key, value) VALUES('schema_version', '1');
                    """
                )
                plan = Task(
                    id="task_plan",
                    job_id="job_legacy",
                    role=PLAN_ROLE,
                    instruction="plan",
                    status=TaskStatus.COMPLETE,
                )
                implement = Task(
                    id="task_impl",
                    job_id="job_legacy",
                    role=IMPLEMENT_ROLE,
                    instruction="implement",
                    status=TaskStatus.BLOCKED,
                    depends_on=[plan.id],
                )
                artifact = _decision_artifact(
                    "job_legacy", plan.id, "Migrate me"
                )
                job_payload = {
                    "id": "job_legacy",
                    "goal": "legacy",
                    "label": None,
                    "status": "running",
                    "created_at": plan.created_at,
                    "completed_at": None,
                }
                from puppetmaster.models import to_jsonable

                connection.execute(
                    "INSERT INTO jobs(id, data) VALUES(?, ?)",
                    ("job_legacy", json.dumps(to_jsonable(job_payload))),
                )
                for task in (plan, implement):
                    connection.execute(
                        "INSERT INTO tasks(id, job_id, role, status, data) "
                        "VALUES(?, ?, ?, ?, ?)",
                        (
                            task.id,
                            task.job_id,
                            task.role,
                            str(task.status),
                            json.dumps(to_jsonable(task)),
                        ),
                    )
                connection.execute(
                    "INSERT INTO artifacts(id, job_id, task_id, type, data) "
                    "VALUES(?, ?, ?, ?, ?)",
                    (
                        artifact.id,
                        artifact.job_id,
                        artifact.task_id,
                        str(artifact.type),
                        json.dumps(to_jsonable(artifact)),
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            store = SQLiteSwarmStore(root)
            store.init()
            status = store.schema_status()
            self.assertEqual(status["schema_version"], "2")
            self.assertEqual(status["expected_schema_version"], "2")
            depends = store.list_edges(
                "job_legacy", edge_type=GraphEdgeType.DEPENDS_ON
            )
            produces = store.list_edges(
                "job_legacy", edge_type=GraphEdgeType.PRODUCES
            )
            self.assertEqual(len(depends), 1)
            self.assertEqual(depends[0].from_id, "task_impl")
            self.assertEqual(depends[0].to_id, "task_plan")
            self.assertEqual(len(produces), 1)
            self.assertEqual(produces[0].to_id, artifact.id)

    def test_depends_on_reconcile_drops_stale_edges_file_and_sqlite(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for _name, store in self._backends(root):
                store.init()
                job = store.create_job("stale depends")
                plan_a = Task(
                    job_id=job.id,
                    role=PLAN_ROLE,
                    instruction="a",
                    status=TaskStatus.COMPLETE,
                )
                plan_b = Task(
                    job_id=job.id,
                    role="alt-plan",
                    instruction="b",
                    status=TaskStatus.COMPLETE,
                )
                implement = Task(
                    job_id=job.id,
                    role=IMPLEMENT_ROLE,
                    instruction="implement",
                    status=TaskStatus.BLOCKED,
                    depends_on=[plan_a.id, plan_b.id],
                )
                store.save_task(plan_a)
                store.save_task(plan_b)
                store.save_task(implement)
                depends = store.list_edges(
                    job.id, edge_type=GraphEdgeType.DEPENDS_ON, from_id=implement.id
                )
                self.assertEqual(
                    sorted(edge.to_id for edge in depends),
                    sorted([plan_a.id, plan_b.id]),
                )
                narrowed = Task(
                    id=implement.id,
                    job_id=job.id,
                    role=IMPLEMENT_ROLE,
                    instruction="implement",
                    status=TaskStatus.BLOCKED,
                    depends_on=[plan_a.id],
                    created_at=implement.created_at,
                )
                store.save_task(narrowed)
                depends_after = store.list_edges(
                    job.id, edge_type=GraphEdgeType.DEPENDS_ON, from_id=implement.id
                )
                self.assertEqual(len(depends_after), 1)
                self.assertEqual(depends_after[0].to_id, plan_a.id)

    def test_file_store_lazy_backfill_materializes_legacy_jobs(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("legacy file graph")
            plan = Task(
                job_id=job.id,
                role=PLAN_ROLE,
                instruction="plan",
                status=TaskStatus.COMPLETE,
            )
            implement = Task(
                job_id=job.id,
                role=IMPLEMENT_ROLE,
                instruction="implement",
                status=TaskStatus.BLOCKED,
                depends_on=[plan.id],
            )
            # Write tasks/artifacts without going through save_* edge materialization.
            store.write_json(
                store.job_dir(job.id) / "tasks" / f"{plan.id}.json", plan
            )
            store.write_json(
                store.job_dir(job.id) / "tasks" / f"{implement.id}.json", implement
            )
            artifact = _decision_artifact(job.id, plan.id, "Backfill me")
            store.write_json(
                store.job_dir(job.id) / "artifacts" / f"{artifact.id}.json",
                artifact,
            )
            edges_dir = store.job_dir(job.id) / "edges"
            if edges_dir.exists():
                for path in edges_dir.glob("*.json"):
                    path.unlink()
            marker = store.job_dir(job.id) / ".graph_edges_materialized"
            if marker.exists():
                marker.unlink()

            depends = store.list_edges(job.id, edge_type=GraphEdgeType.DEPENDS_ON)
            produces = store.list_edges(job.id, edge_type=GraphEdgeType.PRODUCES)
            self.assertEqual(len(depends), 1)
            self.assertEqual(depends[0].from_id, implement.id)
            self.assertEqual(depends[0].to_id, plan.id)
            self.assertEqual(len(produces), 1)
            self.assertEqual(produces[0].to_id, artifact.id)
            self.assertTrue(marker.exists())

class FailurePropagationTests(unittest.TestCase):
    def test_hard_failed_dependency_cascades_blocked_child(self) -> None:
        for backend in ("file", "sqlite"):
            with self.subTest(backend=backend):
                with TemporaryDirectory() as tmp:
                    store = create_store(backend, Path(tmp) / ".puppetmaster")
                    store.init()
                    job = store.create_job("hard fail cascade")
                    upstream = Task(
                        job_id=job.id,
                        role="upstream",
                        instruction="fail hard",
                        status=TaskStatus.FAILED,
                    )
                    child = Task(
                        job_id=job.id,
                        role="child",
                        instruction="wait",
                        status=TaskStatus.BLOCKED,
                        depends_on=[upstream.id],
                    )
                    store.save_task(upstream)
                    store.save_task(child)
                    store.save_artifact(
                        _failure_artifact(job.id, upstream.id, "exception")
                    )
                    ready = store.refresh_blocked_tasks(job.id)
                    self.assertEqual(ready, [])
                    updated = store.get_task_by_id(child.id)
                    self.assertEqual(updated.status, TaskStatus.FAILED)
                    events = store.read_events(job.id)
                    self.assertTrue(
                        any(
                            event["event"] == "task.dependency_failed"
                            for event in events
                        )
                    )

    def test_recoverable_failed_dependency_keeps_child_blocked(self) -> None:
        for backend in ("file", "sqlite"):
            with self.subTest(backend=backend):
                with TemporaryDirectory() as tmp:
                    store = create_store(backend, Path(tmp) / ".puppetmaster")
                    store.init()
                    job = store.create_job("recoverable fail")
                    upstream = Task(
                        job_id=job.id,
                        role="upstream",
                        instruction="billing",
                        status=TaskStatus.FAILED,
                    )
                    child = Task(
                        job_id=job.id,
                        role="child",
                        instruction="wait",
                        status=TaskStatus.BLOCKED,
                        depends_on=[upstream.id],
                    )
                    store.save_task(upstream)
                    store.save_task(child)
                    store.save_artifact(
                        _failure_artifact(job.id, upstream.id, "billing_or_quota")
                    )
                    ready = store.refresh_blocked_tasks(job.id)
                    self.assertEqual(ready, [])
                    updated = store.get_task_by_id(child.id)
                    self.assertEqual(updated.status, TaskStatus.BLOCKED)

    def test_complete_dependency_still_unblocks(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("complete unblocks")
            upstream = Task(
                job_id=job.id,
                role="upstream",
                instruction="done",
                status=TaskStatus.COMPLETE,
            )
            child = Task(
                job_id=job.id,
                role="child",
                instruction="go",
                status=TaskStatus.BLOCKED,
                depends_on=[upstream.id],
            )
            store.save_task(upstream)
            store.save_task(child)
            ready = store.refresh_blocked_tasks(job.id)
            self.assertEqual(len(ready), 1)
            self.assertEqual(ready[0].id, child.id)
            self.assertEqual(ready[0].status, TaskStatus.QUEUED)

class PrewalkEdgeResolutionTests(unittest.TestCase):
    def test_build_prewalk_includes_verify_on_implement(self) -> None:
        specs = build_prewalk_specs("Add verify", cwd="/repo")
        self.assertEqual([spec.role for spec in specs], [
            PLAN_ROLE,
            IMPLEMENT_ROLE,
            VERIFY_ROLE,
        ])
        self.assertEqual(specs[2].depends_on_roles, [IMPLEMENT_ROLE])

    def test_edge_only_resolution_records_consumes(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("edge resolve")
            plan = Task(
                job_id=job.id,
                role=PLAN_ROLE,
                instruction="plan",
                status=TaskStatus.COMPLETE,
            )
            implement = Task(
                job_id=job.id,
                role=IMPLEMENT_ROLE,
                instruction="implement",
                status=TaskStatus.QUEUED,
                depends_on=[plan.id],
                payload={"prewalk": True, "mode": "implement"},
            )
            noise = Task(
                job_id=job.id,
                role="noise",
                instruction="noise",
                status=TaskStatus.COMPLETE,
            )
            store.save_task(plan)
            store.save_task(implement)
            store.save_task(noise)
            plan_artifact = _decision_artifact(job.id, plan.id, "Edit only api.py")
            noise_artifact = _decision_artifact(
                job.id, noise.id, "Should not appear"
            )
            store.save_artifact(plan_artifact)
            store.save_artifact(noise_artifact)

            resolved = store.resolve_artifacts_via_edges(implement)
            self.assertEqual([item.id for item in resolved], [plan_artifact.id])
            consumes = store.list_edges(
                job.id, edge_type=GraphEdgeType.CONSUMES, from_id=implement.id
            )
            self.assertEqual(len(consumes), 1)
            self.assertEqual(consumes[0].to_id, plan_artifact.id)

            prompt = (
                f"Apply the plan\n\n{PREWALK_PLAN_SECTION_HEADER}\n"
                "(placeholder stub)"
            )
            with mock.patch(
                "puppetmaster.adapters._prompts._open_store_for_task",
                return_value=store,
            ):
                injected = with_prewalk_plan(prompt, implement)
            self.assertIn("Decision: Edit only api.py", injected)
            self.assertNotIn("Should not appear", injected)
            self.assertNotIn("(placeholder stub)", injected)

    def test_fallback_to_broad_load_without_edges(self) -> None:
        task = Task(
            job_id="job-fallback",
            role=IMPLEMENT_ROLE,
            instruction="implement",
            depends_on=["missing-plan"],
            payload={"prewalk": True, "mode": "implement"},
        )
        artifacts = [
            {
                "type": "decision",
                "payload": {"decision": "Broad fallback", "why": "no edges"},
            }
        ]
        prompt = f"{PREWALK_PLAN_SECTION_HEADER}\n(stub)"
        with mock.patch(
            "puppetmaster.adapters._prompts._open_store_for_task",
            return_value=None,
        ), mock.patch(
            "puppetmaster.adapters._prompts._load_job_artifacts_for_task",
            return_value=artifacts,
        ):
            result = with_prewalk_plan(prompt, task)
        self.assertIn("Decision: Broad fallback", result)

    def test_routing_only_edges_fall_back_without_consumes(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("routing only")
            plan = Task(
                job_id=job.id,
                role=PLAN_ROLE,
                instruction="plan",
                status=TaskStatus.COMPLETE,
            )
            implement = Task(
                job_id=job.id,
                role=IMPLEMENT_ROLE,
                instruction="implement",
                status=TaskStatus.QUEUED,
                depends_on=[plan.id],
                payload={"prewalk": True, "mode": "implement"},
            )
            store.save_task(plan)
            store.save_task(implement)
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=plan.id,
                    type=ArtifactType.ROUTING,
                    created_by="router",
                    payload={
                        "model_id": "frontier/95",
                        "adapter": "cursor",
                        "policy": "quality",
                        "estimated_cost_usd": 0.1,
                        "baseline_cost_usd": 0.1,
                    },
                    confidence=0.9,
                    evidence=["routing"],
                )
            )
            decision = _decision_artifact(job.id, plan.id, "From broad load")
            # Persist decision without a produces edge so edge resolve stays ROUTING-only.
            store.write_json(
                store.job_dir(job.id) / "artifacts" / f"{decision.id}.json",
                decision,
            )
            prompt = (
                f"Apply the plan\n\n{PREWALK_PLAN_SECTION_HEADER}\n"
                "(placeholder stub)"
            )
            with mock.patch(
                "puppetmaster.adapters._prompts._open_store_for_task",
                return_value=store,
            ):
                injected = with_prewalk_plan(prompt, implement)
            self.assertIn("Decision: From broad load", injected)
            self.assertNotIn("(placeholder stub)", injected)
            consumes = store.list_edges(
                job.id, edge_type=GraphEdgeType.CONSUMES, from_id=implement.id
            )
            self.assertEqual(consumes, [])

    def test_plan_implement_verify_handoff_records_consumes(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("prewalk handoff")
            plan = Task(
                job_id=job.id,
                role=PLAN_ROLE,
                instruction="plan",
                status=TaskStatus.COMPLETE,
            )
            implement = Task(
                job_id=job.id,
                role=IMPLEMENT_ROLE,
                instruction="implement",
                status=TaskStatus.COMPLETE,
                depends_on=[plan.id],
                payload={"prewalk": True, "mode": "implement"},
            )
            verify = Task(
                job_id=job.id,
                role=VERIFY_ROLE,
                instruction="verify",
                status=TaskStatus.QUEUED,
                depends_on=[implement.id],
                payload={
                    "prewalk": True,
                    "prewalk_role": VERIFY_ROLE,
                    "mode": "analyze",
                },
            )
            store.save_task(plan)
            store.save_task(implement)
            store.save_task(verify)
            plan_artifact = _decision_artifact(job.id, plan.id, "Edit only api.py")
            patch = Artifact(
                job_id=job.id,
                task_id=implement.id,
                type=ArtifactType.PATCH,
                created_by="implement",
                payload={"change": "Touched api.py", "files": ["api.py"]},
                confidence=0.9,
                evidence=["test"],
            )
            store.save_artifact(plan_artifact)
            store.save_artifact(patch)

            implement_prompt = (
                f"Apply the plan\n\n{PREWALK_PLAN_SECTION_HEADER}\n(placeholder)"
            )
            verify_prompt = (
                f"Verify\n\n{PREWALK_UPSTREAM_SECTION_HEADER}\n(placeholder)"
            )
            with mock.patch(
                "puppetmaster.adapters._prompts._open_store_for_task",
                return_value=store,
            ):
                implement_injected = with_prewalk_plan(implement_prompt, implement)
                verify_injected = with_prewalk_plan(verify_prompt, verify)
            self.assertIn("Decision: Edit only api.py", implement_injected)
            self.assertIn("Change: Touched api.py", verify_injected)
            implement_consumes = store.list_edges(
                job.id, edge_type=GraphEdgeType.CONSUMES, from_id=implement.id
            )
            verify_consumes = store.list_edges(
                job.id, edge_type=GraphEdgeType.CONSUMES, from_id=verify.id
            )
            self.assertEqual(
                [edge.to_id for edge in implement_consumes], [plan_artifact.id]
            )
            self.assertEqual([edge.to_id for edge in verify_consumes], [patch.id])

    def test_local_adapter_plan_and_verify_roles(self) -> None:
        adapter = LocalAdapter()
        plan_task = Task(
            job_id="job-local",
            role=PLAN_ROLE,
            instruction="plan",
        )
        verify_task = Task(
            job_id="job-local",
            role=VERIFY_ROLE,
            instruction="verify",
        )
        plan_arts = adapter.run(plan_task, "Ship retries", "w1")
        verify_arts = adapter.run(verify_task, "Ship retries", "w1")
        self.assertTrue(
            any(art.type == ArtifactType.DECISION for art in plan_arts)
        )
        self.assertIn("decision", plan_arts[0].payload)
        self.assertTrue(
            any(art.type == ArtifactType.VERIFICATION for art in verify_arts)
        )

class SubgraphResetTests(unittest.TestCase):
    def _seed_chain(self, store: SwarmStore):
        job = store.create_job("subgraph reset")
        plan = Task(
            job_id=job.id,
            role=PLAN_ROLE,
            instruction="plan",
            status=TaskStatus.COMPLETE,
            completed_at="2026-01-01T00:00:00+00:00",
        )
        implement = Task(
            job_id=job.id,
            role=IMPLEMENT_ROLE,
            instruction="implement",
            status=TaskStatus.COMPLETE,
            depends_on=[plan.id],
            attempts=2,
            completed_at="2026-01-01T00:01:00+00:00",
            lease_owner="w1",
            lease_expires_at="2026-01-01T00:02:00+00:00",
            lease_id="lease_old",
        )
        verify = Task(
            job_id=job.id,
            role=VERIFY_ROLE,
            instruction="verify",
            status=TaskStatus.FAILED,
            depends_on=[implement.id],
            attempts=3,
            completed_at="2026-01-01T00:03:00+00:00",
        )
        store.save_task(plan)
        store.save_task(implement)
        store.save_task(verify)
        plan_artifact = _decision_artifact(job.id, plan.id, "Keep me")
        store.save_artifact(plan_artifact)
        return job, plan, implement, verify

    def test_reset_is_idempotent_and_contained(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job, plan, implement, verify = self._seed_chain(store)

            first = store.reset_subgraph(job.id, [implement.id])
            by_id = {task.id: task for task in first}
            self.assertIn(implement.id, by_id)
            self.assertIn(verify.id, by_id)
            self.assertEqual(by_id[implement.id].status, TaskStatus.QUEUED)
            self.assertEqual(by_id[implement.id].attempts, 0)
            self.assertEqual(by_id[verify.id].attempts, 0)
            self.assertIsNone(by_id[implement.id].lease_owner)
            self.assertIsNone(by_id[implement.id].completed_at)
            self.assertEqual(by_id[verify.id].status, TaskStatus.BLOCKED)

            # Upstream plan remains complete with artifacts/edges intact.
            self.assertEqual(
                store.get_task_by_id(plan.id).status, TaskStatus.COMPLETE
            )
            self.assertEqual(len(store.list_artifacts(job.id)), 1)
            self.assertTrue(
                store.list_edges(job.id, edge_type=GraphEdgeType.PRODUCES)
            )

            second = store.reset_subgraph(job.id, [implement.id])
            self.assertEqual(
                {task.id: task.status for task in first},
                {task.id: task.status for task in second},
            )

    def test_reset_include_descendants_false_skips_consumers(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job, _plan, implement, verify = self._seed_chain(store)
            reset = store.reset_subgraph(
                job.id, [implement.id], include_descendants=False
            )
            self.assertEqual([task.id for task in reset], [implement.id])
            self.assertEqual(
                store.get_task_by_id(implement.id).status, TaskStatus.QUEUED
            )
            self.assertEqual(
                store.get_task_by_id(verify.id).status, TaskStatus.FAILED
            )
            self.assertEqual(store.get_task_by_id(verify.id).attempts, 3)

    def test_reset_rejects_active_lease(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job, _plan, implement, verify = self._seed_chain(store)
            live = Task(
                id=implement.id,
                job_id=job.id,
                role=IMPLEMENT_ROLE,
                instruction="implement",
                status=TaskStatus.RUNNING,
                depends_on=implement.depends_on,
                attempts=1,
                lease_owner="worker-live",
                lease_expires_at=seconds_from_now(120),
                lease_id="lease_live",
            )
            store.save_task(live)
            with self.assertRaises(ActiveTaskLeaseError):
                store.reset_subgraph(job.id, [implement.id])
            self.assertEqual(
                store.get_task_by_id(implement.id).status, TaskStatus.RUNNING
            )
            self.assertEqual(
                store.get_task_by_id(verify.id).status, TaskStatus.FAILED
            )

    def test_sqlite_reset_attempts_and_active_lease(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job, _plan, implement, verify = self._seed_chain(store)
            reset = store.reset_subgraph(job.id, [implement.id])
            by_id = {task.id: task for task in reset}
            self.assertEqual(by_id[implement.id].attempts, 0)
            self.assertEqual(by_id[verify.id].attempts, 0)
            live = Task(
                id=implement.id,
                job_id=job.id,
                role=IMPLEMENT_ROLE,
                instruction="implement",
                status=TaskStatus.RUNNING,
                depends_on=implement.depends_on,
                attempts=1,
                lease_owner="worker-live",
                lease_expires_at=seconds_from_now(120),
                lease_id="lease_live",
            )
            store.save_task(live)
            with self.assertRaises(ActiveTaskLeaseError):
                store.reset_subgraph(job.id, [implement.id])

class PrewalkAdapterPinTests(unittest.TestCase):
    def test_non_local_plan_and_verify_pin_allowed_adapters(self) -> None:
        for plan_adapter, implement_adapter, verify_adapter in (
            ("hermes", "claude-code", "codex"),
            ("claude-code", "hermes", "agentic"),
            ("codex", "openai", "hermes"),
            ("agentic", "cursor", "claude-code"),
            ("openai", "hermes", "openai"),
            ("cursor", "codex", None),  # verify defaults to plan adapter
        ):
            with self.subTest(
                plan=plan_adapter,
                implement=implement_adapter,
                verify=verify_adapter,
            ):
                specs = build_prewalk_specs(
                    "Pin adapters under auto-route",
                    cwd="/repo",
                    plan_adapter=plan_adapter,
                    implement_adapter=implement_adapter,
                    verify_adapter=verify_adapter,
                )
                plan, implement, verify = specs
                expected_verify = verify_adapter or plan_adapter
                self.assertEqual(plan.payload.get("allowed_adapters"), [plan_adapter])
                self.assertEqual(
                    implement.payload.get("allowed_adapters"), [implement_adapter]
                )
                self.assertEqual(
                    verify.payload.get("allowed_adapters"), [expected_verify]
                )
                self.assertTrue(plan.payload.get("auto_route"))
                self.assertTrue(implement.payload.get("auto_route"))
                self.assertTrue(verify.payload.get("auto_route"))

    def test_legacy_local_defaults_omit_allowed_adapters(self) -> None:
        specs = build_prewalk_specs("Legacy local defaults", cwd="/repo")
        for spec in specs:
            self.assertEqual(spec.adapter, "local")
            self.assertNotIn("allowed_adapters", spec.payload)
            self.assertTrue(spec.payload.get("auto_route"))

class SharedPrewalkPromptPathTests(unittest.TestCase):
    """Every non-local adapter funnels through with_job_brief → with_prewalk_plan."""

    def test_adapters_import_shared_prompt_helpers(self) -> None:
        import puppetmaster.adapters.agentic as agentic
        import puppetmaster.adapters.claude_code as claude_code
        import puppetmaster.adapters.codex as codex
        import puppetmaster.adapters.cursor as cursor
        import puppetmaster.adapters.hermes as hermes
        import puppetmaster.adapters.openai as openai

        for module in (hermes, claude_code, codex, agentic, openai, cursor):
            with self.subTest(adapter=module.__name__):
                self.assertIs(module.with_job_brief, with_job_brief)

    def test_edge_injection_and_routing_fallback_identical_per_adapter(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("shared prompt path")
            plan = Task(
                job_id=job.id,
                role=PLAN_ROLE,
                instruction="plan",
                status=TaskStatus.COMPLETE,
            )
            store.save_task(plan)
            decision = _decision_artifact(job.id, plan.id, "Touch shared.py")
            store.save_artifact(decision)

            edge_prompt = (
                f"Apply\n\n{PREWALK_PLAN_SECTION_HEADER}\n(placeholder stub)"
            )
            fallback_artifacts = [
                {
                    "type": "decision",
                    "payload": {"decision": "Broad fallback", "why": "no edges"},
                }
            ]

            # Path A: edge-resolved decision injection (via with_job_brief).
            for adapter_name in _NON_LOCAL_ADAPTERS:
                implement = Task(
                    job_id=job.id,
                    role=IMPLEMENT_ROLE,
                    instruction="implement",
                    adapter=adapter_name,
                    status=TaskStatus.QUEUED,
                    depends_on=[plan.id],
                    payload={
                        "prewalk": True,
                        "mode": "implement",
                        "allowed_adapters": [adapter_name],
                    },
                )
                store.save_task(implement)
                with self.subTest(adapter=adapter_name, path="edge"):
                    with mock.patch(
                        "puppetmaster.adapters._prompts._open_store_for_task",
                        return_value=store,
                    ):
                        via_brief = with_job_brief(edge_prompt, implement)
                        via_direct = with_prewalk_plan(edge_prompt, implement)
                    self.assertEqual(via_brief, via_direct)
                    self.assertIn("Decision: Touch shared.py", via_brief)
                    self.assertNotIn("(placeholder stub)", via_brief)

            # Path B: ROUTING-only produces → broad-load fallback, no consumes.
            for edge in list(
                store.list_edges(
                    job.id, edge_type=GraphEdgeType.PRODUCES, from_id=plan.id
                )
            ):
                if edge.to_id == decision.id:
                    store.delete_edge(job.id, edge.id)
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=plan.id,
                    type=ArtifactType.ROUTING,
                    created_by="router",
                    payload={
                        "model_id": "frontier/95",
                        "adapter": "cursor",
                        "policy": "quality",
                        "estimated_cost_usd": 0.1,
                        "baseline_cost_usd": 0.1,
                    },
                    confidence=0.9,
                    evidence=["routing"],
                )
            )
            # Keep decision bytes on disk without a produces edge.
            store.write_json(
                store.job_dir(job.id) / "artifacts" / f"{decision.id}.json",
                decision,
            )

            for adapter_name in _NON_LOCAL_ADAPTERS:
                implement = Task(
                    job_id=job.id,
                    role=IMPLEMENT_ROLE,
                    instruction="implement",
                    adapter=adapter_name,
                    status=TaskStatus.QUEUED,
                    depends_on=[plan.id],
                    payload={"prewalk": True, "mode": "implement"},
                )
                store.save_task(implement)
                with self.subTest(adapter=adapter_name, path="routing_fallback"):
                    with mock.patch(
                        "puppetmaster.adapters._prompts._open_store_for_task",
                        return_value=store,
                    ):
                        injected = with_job_brief(edge_prompt, implement)
                    self.assertIn("Decision: Touch shared.py", injected)
                    self.assertNotIn("(placeholder stub)", injected)
                    consumes = store.list_edges(
                        job.id,
                        edge_type=GraphEdgeType.CONSUMES,
                        from_id=implement.id,
                    )
                    self.assertEqual(consumes, [])

            # Path C: no store → credential-free broad-load fallback.
            for adapter_name in _NON_LOCAL_ADAPTERS:
                task = Task(
                    job_id=job.id,
                    role=IMPLEMENT_ROLE,
                    instruction="implement",
                    adapter=adapter_name,
                    depends_on=["missing-plan"],
                    payload={"prewalk": True, "mode": "implement"},
                )
                with self.subTest(adapter=adapter_name, path="no_store_fallback"):
                    with mock.patch(
                        "puppetmaster.adapters._prompts._open_store_for_task",
                        return_value=None,
                    ), mock.patch(
                        "puppetmaster.adapters._prompts._load_job_artifacts_for_task",
                        return_value=fallback_artifacts,
                    ):
                        result = with_job_brief(
                            f"{PREWALK_PLAN_SECTION_HEADER}\n(stub)", task
                        )
                    self.assertIn("Decision: Broad fallback", result)

class AmbiguousModelPinBoundaryTests(unittest.TestCase):
    def _ambiguous_registry(self, path: Path):
        from puppetmaster.model_registry import ModelSpec, save_registry

        save_registry(
            [
                ModelSpec(
                    id="cursor/grok-a",
                    adapter="cursor",
                    adapter_model_name="grok-a",
                    enabled=True,
                ),
                ModelSpec(
                    id="cursor/grok-a-alt",
                    adapter="cursor",
                    adapter_model_name="grok-a",
                    enabled=True,
                ),
            ],
            path,
        )

    def test_cli_cursor_dispatch_returns_structured_preflight_blocked(self) -> None:
        with TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "models.json"
            self._ambiguous_registry(registry_path)
            state_dir = Path(tmp) / ".puppetmaster"
            buf = io.StringIO()
            with mock.patch.dict(
                os.environ,
                {"PUPPETMASTER_MODELS_PATH": str(registry_path)},
                clear=False,
            ), redirect_stdout(buf):
                code = _main(
                    [
                        "--state-dir",
                        str(state_dir),
                        "cursor",
                        "check pin",
                        "--cwd",
                        tmp,
                        "--model",
                        "grok-a",
                        "--dry-run",
                    ]
                )
            self.assertEqual(code, 1)
            payload = json.loads(buf.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["failure"], "preflight_blocked")
            self.assertEqual(payload["result"], "blocked")
            self.assertIn("preflight:ambiguous_model_pin", payload["evidence"])
            self.assertIn("ambiguous model pin", payload["reason"])

    def test_mcp_generated_config_returns_structured_preflight_blocked(self) -> None:
        from puppetmaster.mcp_server import start_cursor_swarm, start_swarm

        with TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "models.json"
            self._ambiguous_registry(registry_path)
            args = {
                "goal": "audit ambiguous pin",
                "cwd": tmp,
                "state_dir": str(Path(tmp) / "state"),
                "model": "grok-a",
                "adapter": "cursor",
            }
            with mock.patch.dict(
                os.environ,
                {"PUPPETMASTER_MODELS_PATH": str(registry_path)},
                clear=False,
            ):
                for starter in (start_swarm, start_cursor_swarm):
                    with self.subTest(starter=starter.__name__):
                        result = starter(args)
                        self.assertTrue(result.get("isError"))
                        blob = json.dumps(result)
                        self.assertIn("preflight_blocked", blob)
                        self.assertIn("ambiguous model pin", blob)
                        self.assertIn("preflight:ambiguous_model_pin", blob)
                        self.assertNotIn("Traceback", blob)

class FileConsumesJournalTests(unittest.TestCase):
    def test_record_consumes_journal_replays_after_crash(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("consumes journal")
            task = Task(
                job_id=job.id,
                role=IMPLEMENT_ROLE,
                instruction="implement",
                status=TaskStatus.QUEUED,
            )
            store.save_task(task)
            a1 = _decision_artifact(job.id, task.id, "one")
            a2 = _decision_artifact(job.id, task.id, "two")
            store.save_artifact(a1)
            store.save_artifact(a2)

            real_upsert = store.upsert_edges
            calls = {"n": 0}

            def flaky_upsert(edges):
                calls["n"] += 1
                edge_list = list(edges)
                if calls["n"] == 1 and len(edge_list) >= 2:
                    # Crash after journaling + first edge write.
                    real_upsert(edge_list[:1])
                    raise RuntimeError("simulated crash mid record_consumes")
                return real_upsert(edge_list)

            with mock.patch.object(store, "upsert_edges", side_effect=flaky_upsert):
                with self.assertRaises(RuntimeError):
                    store.record_consumes(job.id, task.id, [a1.id, a2.id])

            journal = store.job_dir(job.id) / (
                f"{_CONSUMES_JOURNAL_PREFIX}{store._safe_key(task.id)}.json"
            )
            self.assertTrue(journal.exists())

            # Recovery path: replay finishes the batch idempotently.
            recovered = store.record_consumes(job.id, task.id, [a1.id, a2.id])
            self.assertEqual(len(recovered), 2)
            self.assertFalse(journal.exists())
            consumes = store.list_edges(
                job.id, edge_type=GraphEdgeType.CONSUMES, from_id=task.id
            )
            self.assertEqual(
                sorted(edge.to_id for edge in consumes), sorted([a1.id, a2.id])
            )

    def test_delete_edge_uses_windows_lock_retry(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("delete retry")
            edge = make_graph_edge(
                job_id=job.id,
                type=GraphEdgeType.DEPENDS_ON,
                from_kind=GraphNodeKind.TASK,
                from_id="a",
                to_kind=GraphNodeKind.TASK,
                to_id="b",
            )
            store.upsert_edge(edge)
            path = store.job_dir(job.id) / "edges" / f"{edge.id}.json"
            self.assertTrue(path.exists())
            with mock.patch(
                "puppetmaster.store._retry_on_windows_lock",
                side_effect=lambda op: op(),
            ) as retry:
                self.assertTrue(store.delete_edge(job.id, edge.id))
            retry.assert_called()
            self.assertFalse(path.exists())

class SqliteAtomicResetAndDoctorTests(unittest.TestCase):
    def test_sqlite_reset_subgraph_is_single_transaction(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("atomic reset")
            plan = Task(
                job_id=job.id,
                role=PLAN_ROLE,
                instruction="plan",
                status=TaskStatus.COMPLETE,
            )
            implement = Task(
                job_id=job.id,
                role=IMPLEMENT_ROLE,
                instruction="implement",
                status=TaskStatus.COMPLETE,
                depends_on=[plan.id],
                attempts=2,
            )
            verify = Task(
                job_id=job.id,
                role=VERIFY_ROLE,
                instruction="verify",
                status=TaskStatus.FAILED,
                depends_on=[implement.id],
                attempts=3,
            )
            store.save_task(plan)
            store.save_task(implement)
            store.save_task(verify)

            # Must batch via one write session — never per-task save_task.
            with mock.patch.object(
                store,
                "save_task",
                side_effect=AssertionError("reset_subgraph must not call save_task"),
            ):
                reset = store.reset_subgraph(job.id, [implement.id])
            by_id = {task.id: task for task in reset}
            self.assertEqual(by_id[implement.id].status, TaskStatus.QUEUED)
            self.assertEqual(by_id[verify.id].status, TaskStatus.BLOCKED)
            self.assertEqual(by_id[implement.id].attempts, 0)

            # A busy/crash before commit must not leave a partial reset.
            store.save_task(
                Task(
                    id=implement.id,
                    job_id=job.id,
                    role=IMPLEMENT_ROLE,
                    instruction="implement",
                    status=TaskStatus.COMPLETE,
                    depends_on=[plan.id],
                    attempts=4,
                    completed_at="2026-01-01T00:01:00+00:00",
                )
            )
            store.save_task(
                Task(
                    id=verify.id,
                    job_id=job.id,
                    role=VERIFY_ROLE,
                    instruction="verify",
                    status=TaskStatus.FAILED,
                    depends_on=[implement.id],
                    attempts=5,
                    completed_at="2026-01-01T00:03:00+00:00",
                )
            )
            real_session = store._session

            from contextlib import contextmanager

            @contextmanager
            def boom_after_writes():
                with real_session() as connection:
                    yield connection
                    raise sqlite3.OperationalError("simulated busy/crash")

            with mock.patch.object(store, "_session", boom_after_writes):
                with self.assertRaises(sqlite3.OperationalError):
                    store.reset_subgraph(job.id, [implement.id])
            self.assertEqual(
                store.get_task_by_id(implement.id).status, TaskStatus.COMPLETE
            )
            self.assertEqual(store.get_task_by_id(implement.id).attempts, 4)
            self.assertEqual(
                store.get_task_by_id(verify.id).status, TaskStatus.FAILED
            )
            self.assertEqual(store.get_task_by_id(verify.id).attempts, 5)

    def test_doctor_warns_on_schema_version_mismatch(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / ".puppetmaster"
            store = SQLiteSwarmStore(state_dir)
            store.init()
            connection = sqlite3.connect(state_dir / "state.sqlite3")
            try:
                connection.execute(
                    "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
                    ("999",),
                )
                connection.commit()
            finally:
                connection.close()
            checks = {check.name: check for check in run_doctor(root, state_dir)}
            sqlite_check = checks["sqlite-state"]
            self.assertEqual(sqlite_check.status, "warn")
            self.assertIn("expected=2", sqlite_check.detail)
            self.assertIn("999", sqlite_check.detail)
            self.assertIn("differs", sqlite_check.detail)

class GraphCliMcpTests(unittest.TestCase):
    def test_cli_graph_outputs_nodes_and_edges(self) -> None:
        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".puppetmaster"
            store = SwarmStore(state_dir)
            store.init()
            job = store.create_job("cli graph")
            plan = Task(
                job_id=job.id,
                role=PLAN_ROLE,
                instruction="plan",
                status=TaskStatus.COMPLETE,
            )
            implement = Task(
                job_id=job.id,
                role=IMPLEMENT_ROLE,
                instruction="implement",
                status=TaskStatus.BLOCKED,
                depends_on=[plan.id],
            )
            store.save_task(plan)
            store.save_task(implement)
            store.save_artifact(
                _decision_artifact(job.id, plan.id, "Ship it")
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = _main(
                    [
                        "--state-dir",
                        str(state_dir),
                        "--backend",
                        "file",
                        "graph",
                        job.id,
                    ]
                )
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["job_id"], job.id)
            self.assertTrue(payload["nodes"])
            self.assertTrue(payload["edges"])

    def test_job_graph_shape_and_mcp_tool_registered(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("mcp graph")
            plan = Task(
                job_id=job.id,
                role=PLAN_ROLE,
                instruction="plan",
                status=TaskStatus.COMPLETE,
            )
            implement = Task(
                job_id=job.id,
                role=IMPLEMENT_ROLE,
                instruction="implement",
                depends_on=[plan.id],
                status=TaskStatus.QUEUED,
            )
            store.save_task(plan)
            store.save_task(implement)
            artifact = _decision_artifact(job.id, plan.id, "Graph me")
            store.save_artifact(artifact)
            graph = store.job_graph(job.id)
            self.assertEqual(graph["job_id"], job.id)
            kinds = {node["kind"] for node in graph["nodes"]}
            self.assertEqual(kinds, {"task", "artifact"})
            edge_types = {edge["type"] for edge in graph["edges"]}
            self.assertIn("depends_on", edge_types)
            self.assertIn("produces", edge_types)

        from puppetmaster.mcp_server import tools

        names = {tool.name for tool in tools()}
        self.assertIn("puppetmaster_job_graph", names)

    def test_make_graph_edge_optional_derived_from(self) -> None:
        edge = make_graph_edge(
            job_id="j",
            type=GraphEdgeType.DERIVED_FROM,
            from_kind=GraphNodeKind.ARTIFACT,
            from_id="a1",
            to_kind=GraphNodeKind.ARTIFACT,
            to_id="a0",
        )
        self.assertEqual(edge.type, GraphEdgeType.DERIVED_FROM)
        self.assertEqual(
            edge.id,
            graph_edge_identity(
                "j", "derived_from", "artifact", "a1", "artifact", "a0"
            ),
        )

    def test_cli_graph_covers_migrated_sqlite_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "legacy"
            root.mkdir()
            db_path = root / "state.sqlite3"
            connection = sqlite3.connect(db_path)
            try:
                connection.executescript(
                    """
                    CREATE TABLE jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                    CREATE TABLE tasks (
                      id TEXT PRIMARY KEY,
                      job_id TEXT NOT NULL,
                      role TEXT NOT NULL,
                      status TEXT NOT NULL,
                      data TEXT NOT NULL
                    );
                    CREATE TABLE artifacts (
                      id TEXT PRIMARY KEY,
                      job_id TEXT NOT NULL,
                      task_id TEXT NOT NULL,
                      type TEXT NOT NULL,
                      data TEXT NOT NULL
                    );
                    CREATE TABLE runs (
                      id TEXT PRIMARY KEY,
                      job_id TEXT NOT NULL,
                      task_id TEXT NOT NULL,
                      data TEXT NOT NULL
                    );
                    CREATE TABLE memory (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                    CREATE TABLE events (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      job_id TEXT NOT NULL,
                      at TEXT NOT NULL,
                      event TEXT NOT NULL,
                      payload TEXT NOT NULL
                    );
                    CREATE TABLE metadata (
                      key TEXT PRIMARY KEY,
                      value TEXT NOT NULL
                    );
                    INSERT INTO metadata(key, value) VALUES('schema_version', '1');
                    """
                )
                plan = Task(
                    id="task_plan",
                    job_id="job_legacy",
                    role=PLAN_ROLE,
                    instruction="plan",
                    status=TaskStatus.COMPLETE,
                )
                implement = Task(
                    id="task_impl",
                    job_id="job_legacy",
                    role=IMPLEMENT_ROLE,
                    instruction="implement",
                    status=TaskStatus.BLOCKED,
                    depends_on=[plan.id],
                )
                artifact = _decision_artifact("job_legacy", plan.id, "CLI graph me")
                job_payload = {
                    "id": "job_legacy",
                    "goal": "legacy",
                    "label": None,
                    "status": "running",
                    "created_at": plan.created_at,
                    "completed_at": None,
                }
                connection.execute(
                    "INSERT INTO jobs(id, data) VALUES(?, ?)",
                    ("job_legacy", json.dumps(to_jsonable(job_payload))),
                )
                for task in (plan, implement):
                    connection.execute(
                        "INSERT INTO tasks(id, job_id, role, status, data) "
                        "VALUES(?, ?, ?, ?, ?)",
                        (
                            task.id,
                            task.job_id,
                            task.role,
                            str(task.status),
                            json.dumps(to_jsonable(task)),
                        ),
                    )
                connection.execute(
                    "INSERT INTO artifacts(id, job_id, task_id, type, data) "
                    "VALUES(?, ?, ?, ?, ?)",
                    (
                        artifact.id,
                        artifact.job_id,
                        artifact.task_id,
                        str(artifact.type),
                        json.dumps(to_jsonable(artifact)),
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            buf = io.StringIO()
            with redirect_stdout(buf):
                code = _main(
                    [
                        "--state-dir",
                        str(root),
                        "--backend",
                        "sqlite",
                        "graph",
                        "job_legacy",
                    ]
                )
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["job_id"], "job_legacy")
            self.assertTrue(payload["nodes"])
            self.assertTrue(payload["edges"])
            edge_types = {edge["type"] for edge in payload["edges"]}
            self.assertIn("depends_on", edge_types)
            self.assertIn("produces", edge_types)
            status = SQLiteSwarmStore(root).schema_status()
            self.assertEqual(status["schema_version"], "2")

if __name__ == "__main__":
    unittest.main()
