"""Focused tests for dependency-aware reusable validation protocol."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

from puppetmaster.cli._dispatch import _main
from puppetmaster.mcp_server import call_tool, tools
from puppetmaster.models import (
    Artifact,
    ArtifactType,
    GraphEdgeType,
    GraphNodeKind,
    Task,
    TaskStatus,
    seconds_from_now,
)
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.store import ActiveTaskLeaseError, SwarmStore
from puppetmaster.validation import (
    ValidationFingerprintError,
    compact_artifact_ref,
    compute_validation_fingerprint,
    validation_status_of,
    with_validation_status,
)


def _git_init_with_file(root: Path, rel: str, content: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", rel], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _finding(
    job_id: str,
    task_id: str,
    claim: str,
    *,
    validation: Optional[dict[str, Any]] = None,
    artifact_type: ArtifactType = ArtifactType.FINDING,
) -> Artifact:
    payload: dict[str, Any] = {"claim": claim}
    if artifact_type == ArtifactType.VERIFICATION:
        payload = {"check": claim, "result": "pass"}
    elif artifact_type == ArtifactType.DECISION:
        payload = {"decision": claim, "why": "test"}
    if validation is not None:
        payload["validation"] = validation
    return Artifact(
        job_id=job_id,
        task_id=task_id,
        type=artifact_type,
        created_by="test",
        payload=payload,
        confidence=0.9,
        evidence=["src/a.py"],
    )


class ValidationFingerprintTests(unittest.TestCase):
    def test_fingerprint_stable_and_changes_on_dirty_scoped_bytes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init_with_file(root, "src/a.py", "alpha\n")
            first = compute_validation_fingerprint(
                root, ["src/a.py"], rules_version="r1", evaluator_version="e1"
            )
            second = compute_validation_fingerprint(
                root, ["src/a.py"], rules_version="r1", evaluator_version="e1"
            )
            self.assertEqual(first.fingerprint, second.fingerprint)
            self.assertTrue(first.complete)
            self.assertFalse(first.dirty_scoped)

            (root / "src/a.py").write_text("alpha-dirty\n", encoding="utf-8")
            dirty = compute_validation_fingerprint(
                root, ["src/a.py"], rules_version="r1", evaluator_version="e1"
            )
            self.assertNotEqual(first.fingerprint, dirty.fingerprint)
            self.assertTrue(dirty.dirty_scoped)
            self.assertEqual(
                dirty.source_digests["src/a.py"],
                __import__("hashlib")
                .sha256(b"alpha-dirty\n")
                .hexdigest(),
            )

    def test_fingerprint_changes_on_rules_and_evaluator(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init_with_file(root, "src/a.py", "alpha\n")
            (root / "rules.txt").write_text("rule-a\n", encoding="utf-8")
            base = compute_validation_fingerprint(
                root,
                ["src/a.py"],
                rules_paths=["rules.txt"],
                rules_version="1",
                evaluator_version="eval-1",
            )
            (root / "rules.txt").write_text("rule-b\n", encoding="utf-8")
            rules_changed = compute_validation_fingerprint(
                root,
                ["src/a.py"],
                rules_paths=["rules.txt"],
                rules_version="1",
                evaluator_version="eval-1",
            )
            self.assertNotEqual(base.fingerprint, rules_changed.fingerprint)
            eval_changed = compute_validation_fingerprint(
                root,
                ["src/a.py"],
                rules_paths=["rules.txt"],
                rules_version="1",
                evaluator_version="eval-2",
            )
            self.assertNotEqual(rules_changed.fingerprint, eval_changed.fingerprint)

    def test_missing_scoped_path_fails_closed(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init_with_file(root, "src/a.py", "alpha\n")
            with self.assertRaises(ValidationFingerprintError) as ctx:
                compute_validation_fingerprint(root, ["src/missing.py"])
            self.assertIn("src/missing.py", ctx.exception.missing_paths)
            incomplete = compute_validation_fingerprint(
                root, ["src/missing.py"], strict=False
            )
            self.assertFalse(incomplete.complete)
            with self.assertRaises(ValidationFingerprintError):
                incomplete.to_payload(status="fresh")

    def test_path_containment_rejects_absolute_and_traversal(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init_with_file(root, "src/a.py", "alpha\n")
            outside = Path(tmp).resolve().parent / f"pm-outside-{os.getpid()}.txt"
            try:
                outside.write_text("secret\n", encoding="utf-8")
                for bad in (
                    str(outside),
                    "../" + outside.name,
                    "..\\outside.txt",
                    "/etc/passwd",
                    "C:/Windows/System32/drivers/etc/hosts",
                ):
                    with self.subTest(path=bad):
                        with self.assertRaises(ValidationFingerprintError) as ctx:
                            compute_validation_fingerprint(root, [bad])
                        message = str(ctx.exception).lower()
                        self.assertTrue(
                            ctx.exception.rejected_paths
                            or "escape" in message
                            or "rejected" in message
                            or "inside" in message,
                            msg=message,
                        )
                        # Never surface outside file bytes via digests/errors.
                        self.assertNotIn("secret\n", str(ctx.exception))
                        self.assertNotIn("secret", getattr(ctx.exception, "missing_paths", []))
            finally:
                if outside.exists():
                    outside.unlink()

    def test_path_containment_rejects_symlink_escape(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init_with_file(root, "src/a.py", "alpha\n")
            outside = Path(tmp).resolve().parent / f"pm-symlink-target-{os.getpid()}.txt"
            link = root / "leak.txt"
            try:
                outside.write_text("symlink-secret\n", encoding="utf-8")
                try:
                    link.symlink_to(outside)
                except OSError as exc:
                    self.skipTest(f"symlinks unavailable on this host: {exc}")
                with self.assertRaises(ValidationFingerprintError) as ctx:
                    compute_validation_fingerprint(root, ["leak.txt"])
                self.assertTrue(
                    ctx.exception.rejected_paths
                    or "escape" in str(ctx.exception).lower()
                )
                # Confirm we never hashed the outside target.
                with self.assertRaises(ValidationFingerprintError):
                    compute_validation_fingerprint(
                        root, ["leak.txt"], rules_paths=["leak.txt"]
                    )
            finally:
                if link.exists() or link.is_symlink():
                    link.unlink()
                if outside.exists():
                    outside.unlink()


class ValidationStoreParityTests(unittest.TestCase):
    def _seed_job(self, store):
        job = store.create_job("validate")
        task = Task(job_id=job.id, role="review", instruction="review")
        store.save_task(task)
        return job, task

    def test_old_artifact_compatibility_and_lookup_filters(self) -> None:
        for store_cls in (SwarmStore, SQLiteSwarmStore):
            with self.subTest(store=store_cls.__name__), TemporaryDirectory() as tmp:
                store = store_cls(Path(tmp) / ".puppetmaster")
                store.init()
                job, task = self._seed_job(store)
                legacy = _finding(job.id, task.id, "legacy unlabeled")
                store.save_artifact(legacy)
                loaded = store.list_artifacts(job.id)[0]
                self.assertNotIn("validation", loaded.payload)

                fp = "abc123fingerprint"
                fresh = _finding(
                    job.id,
                    task.id,
                    "fresh claim",
                    validation={
                        "fingerprint": fp,
                        "status": "fresh",
                        "head_sha": "deadbeef",
                        "scope": ["src/a.py"],
                    },
                )
                stale = _finding(
                    job.id,
                    task.id,
                    "stale claim",
                    validation={
                        "fingerprint": fp,
                        "status": "stale",
                        "head_sha": "deadbeef",
                        "scope": ["src/a.py"],
                    },
                )
                superseded = _finding(
                    job.id,
                    task.id,
                    "superseded claim",
                    validation={
                        "fingerprint": fp,
                        "status": "superseded",
                        "head_sha": "deadbeef",
                        "scope": ["src/a.py"],
                    },
                )
                routing = Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.ROUTING,
                    created_by="router",
                    payload={
                        "model_id": "m",
                        "adapter": "cursor",
                        "policy": "balanced",
                        "validation": {"fingerprint": fp, "status": "fresh"},
                    },
                    confidence=1.0,
                    evidence=["router"],
                )
                store.save_artifacts([fresh, stale, superseded, routing])
                found = store.lookup_artifacts_by_validation_fingerprint(fp)
                self.assertEqual([item.payload["claim"] for item in found], ["fresh claim"])
                self.assertTrue(
                    all(validation_status_of(item) == "fresh" for item in found)
                )

    def test_record_derived_from_idempotent_and_ownership(self) -> None:
        for store_cls in (SwarmStore, SQLiteSwarmStore):
            with self.subTest(store=store_cls.__name__), TemporaryDirectory() as tmp:
                store = store_cls(Path(tmp) / ".puppetmaster")
                store.init()
                job, task = self._seed_job(store)
                source = _finding(
                    job.id,
                    task.id,
                    "source",
                    validation={"fingerprint": "f1", "status": "fresh"},
                )
                derived = _finding(
                    job.id,
                    task.id,
                    "derived",
                    validation={
                        "fingerprint": "f1",
                        "status": "reused",
                        "source_artifact_ids": [],
                    },
                )
                store.save_artifacts([source, derived])
                edge1 = store.record_derived_from(job.id, source.id, derived.id)
                edge2 = store.record_derived_from(job.id, source.id, derived.id)
                self.assertEqual(edge1.id, edge2.id)
                edges = store.list_edges(job.id, edge_type=GraphEdgeType.DERIVED_FROM)
                self.assertEqual(len(edges), 1)
                self.assertEqual(edges[0].from_kind, GraphNodeKind.ARTIFACT)
                self.assertEqual(edges[0].from_id, derived.id)
                self.assertEqual(edges[0].to_id, source.id)

                other = store.create_job("other")
                with self.assertRaises(ValueError):
                    store.record_derived_from(other.id, source.id, derived.id)
                with self.assertRaises(ValueError):
                    store.record_derived_from(job.id, source.id, source.id)

    def test_reset_refuses_active_lease_and_labels_superseded(self) -> None:
        for store_cls in (SwarmStore, SQLiteSwarmStore):
            with self.subTest(store=store_cls.__name__), TemporaryDirectory() as tmp:
                store = store_cls(Path(tmp) / ".puppetmaster")
                store.init()
                job = store.create_job("reset")
                plan = Task(
                    job_id=job.id,
                    role="plan",
                    instruction="plan",
                    status=TaskStatus.COMPLETE,
                )
                implement = Task(
                    job_id=job.id,
                    role="implement",
                    instruction="implement",
                    status=TaskStatus.COMPLETE,
                    depends_on=[plan.id],
                )
                store.save_task(plan)
                store.save_task(implement)
                plan_art = _finding(
                    job.id,
                    plan.id,
                    "keep upstream",
                    validation={"fingerprint": "up", "status": "fresh"},
                )
                impl_art = _finding(
                    job.id,
                    implement.id,
                    "will supersede",
                    validation={"fingerprint": "im", "status": "fresh"},
                )
                store.save_artifacts([plan_art, impl_art])

                live = Task(
                    id=implement.id,
                    job_id=job.id,
                    role="implement",
                    instruction="implement",
                    status=TaskStatus.RUNNING,
                    depends_on=[plan.id],
                    lease_owner="worker-live",
                    lease_expires_at=seconds_from_now(120),
                    lease_id="lease_live",
                )
                store.save_task(live)
                with self.assertRaises(ActiveTaskLeaseError):
                    store.reset_subgraph(job.id, [implement.id])
                self.assertEqual(
                    validation_status_of(store.get_artifacts_by_ids(job.id, [impl_art.id])[impl_art.id]),
                    "fresh",
                )

                cleared = Task(
                    id=implement.id,
                    job_id=job.id,
                    role="implement",
                    instruction="implement",
                    status=TaskStatus.COMPLETE,
                    depends_on=[plan.id],
                )
                store.save_task(cleared)
                result = store.reset_subgraph(job.id, [implement.id])
                after = store.get_artifacts_by_ids(job.id, [impl_art.id, plan_art.id])
                self.assertEqual(validation_status_of(after[impl_art.id]), "superseded")
                self.assertEqual(validation_status_of(after[plan_art.id]), "fresh")
                self.assertEqual(len(store.list_artifacts(job.id)), 2)
                self.assertEqual(result.superseded_artifact_ids, [impl_art.id])
                # Superseded must not be reusable via fingerprint lookup.
                self.assertEqual(
                    store.lookup_artifacts_by_validation_fingerprint("im"),
                    [],
                )
                reset_events = [
                    event
                    for event in store.read_events(job.id)
                    if event.get("event") == "subgraph.reset"
                ]
                self.assertTrue(reset_events)
                payload = reset_events[-1].get("payload") or {}
                self.assertEqual(payload.get("superseded_artifact_ids"), [impl_art.id])
                self.assertNotIn(
                    "subgraph.reset.superseded",
                    {event.get("event") for event in store.read_events(job.id)},
                )


class ValidationCliMcpTests(unittest.TestCase):
    def test_compact_refs_omit_repo_root_and_truncate_evidence(self) -> None:
        long_evidence = "e" * 500
        absolute_root = str(Path("/tmp/secret-repo").resolve())
        artifact = Artifact(
            job_id="job_x",
            task_id="task_x",
            type=ArtifactType.FINDING,
            created_by="test",
            payload={
                "claim": "c" * 300,
                "details": "d" * 5000,
                "validation": {
                    "fingerprint": "fp",
                    "status": "fresh",
                    "head_sha": "abc",
                    "repo_root": absolute_root,
                    "scope": ["src/a.py"],
                    "source_digests": {"src/a.py": "00" * 32},
                },
            },
            confidence=0.8,
            evidence=[long_evidence, "src/b.py", "src/c.py"],
        )
        ref = compact_artifact_ref(artifact, evidence_limit=2)
        encoded = json.dumps(ref)
        self.assertNotIn(absolute_root, encoded)
        self.assertNotIn("/tmp/secret-repo", encoded)
        self.assertNotIn("repo_root", ref.get("validation") or {})
        self.assertNotIn("source_digests", ref.get("validation") or {})
        self.assertNotIn("details", ref)
        self.assertTrue(str(ref["claim"]).endswith("..."))
        self.assertLessEqual(len(ref["claim"]), 240)
        self.assertEqual(ref["evidence_summary"]["count"], 3)
        self.assertEqual(len(ref["evidence_summary"]["items"]), 2)
        self.assertLessEqual(len(ref["evidence_summary"]["items"][0]), 240)
        self.assertTrue(str(ref["evidence_summary"]["items"][0]).endswith("..."))
        self.assertNotIn(long_evidence, encoded)

    def test_cli_and_mcp_reset_and_refs_round_trip(self) -> None:
        with TemporaryDirectory() as tmp:
            state = Path(tmp) / ".puppetmaster"
            store = SQLiteSwarmStore(state)
            store.init()
            job = store.create_job("cli")
            task = Task(
                job_id=job.id,
                role="implement",
                instruction="implement",
                status=TaskStatus.COMPLETE,
            )
            store.save_task(task)
            big_body = "x" * 5000
            absolute_root = str(Path(tmp).resolve())
            artifact = Artifact(
                job_id=job.id,
                task_id=task.id,
                type=ArtifactType.FINDING,
                created_by="test",
                payload={
                    "claim": "compact me",
                    "details": big_body,
                    "validation": {
                        "fingerprint": "fp-cli",
                        "status": "fresh",
                        "head_sha": "abc",
                        "repo_root": absolute_root,
                        "scope": ["src/a.py"],
                        "source_digests": {"src/a.py": "00" * 32},
                    },
                },
                confidence=0.8,
                evidence=["src/a.py", "e" * 400],
            )
            store.save_artifact(artifact)

            # Compact refs omit large bodies and absolute repo roots.
            ref = compact_artifact_ref(artifact)
            encoded = json.dumps(ref)
            self.assertNotIn(big_body, encoded)
            self.assertNotIn(absolute_root, encoded)
            self.assertNotIn("details", ref)
            self.assertEqual(ref["claim"], "compact me")
            self.assertIn("validation", ref)
            self.assertNotIn("source_digests", ref["validation"])
            self.assertNotIn("repo_root", ref["validation"])
            self.assertEqual(ref["evidence_summary"]["count"], 2)
            self.assertLessEqual(len(ref["evidence_summary"]["items"][1]), 240)

            # CLI --refs
            from contextlib import redirect_stdout
            from io import StringIO

            buf = StringIO()
            with redirect_stdout(buf):
                code = _main(
                    [
                        "--state-dir",
                        str(state),
                        "--backend",
                        "sqlite",
                        "artifacts",
                        job.id,
                        "--refs",
                    ]
                )
            self.assertEqual(code, 0)
            refs_payload = json.loads(buf.getvalue())
            self.assertEqual(len(refs_payload), 1)
            self.assertNotIn("details", refs_payload[0])
            self.assertEqual(refs_payload[0]["claim"], "compact me")
            self.assertNotIn(absolute_root, json.dumps(refs_payload))

            # CLI reset-subgraph
            buf2 = StringIO()
            with redirect_stdout(buf2):
                code = _main(
                    [
                        "--state-dir",
                        str(state),
                        "--backend",
                        "sqlite",
                        "reset-subgraph",
                        job.id,
                        "--task",
                        task.id,
                    ]
                )
            self.assertEqual(code, 0)
            reset_body = json.loads(buf2.getvalue())
            self.assertEqual(reset_body["reset_count"], 1)
            self.assertEqual(reset_body["superseded_artifact_ids"], [artifact.id])
            self.assertEqual(
                validation_status_of(
                    store.get_artifacts_by_ids(job.id, [artifact.id])[artifact.id]
                ),
                "superseded",
            )

            # MCP tool registration + refs round trip
            names = {tool.name for tool in tools()}
            self.assertIn("puppetmaster_reset_subgraph", names)
            self.assertIn("puppetmaster_artifacts", names)

            # Re-seed a fresh artifact for MCP refs/reset (prior one is superseded).
            fresh = with_validation_status(
                _finding(
                    job.id,
                    task.id,
                    "mcp claim",
                    validation={
                        "fingerprint": "fp-mcp",
                        "status": "fresh",
                        "head_sha": "abc",
                        "scope": ["src/a.py"],
                        "details_should_not_appear": big_body,
                    },
                ),
                "fresh",
            )
            # Put large body on payload root for omission check.
            from dataclasses import replace

            fresh = replace(
                fresh,
                payload={
                    **fresh.payload,
                    "details": big_body,
                    "validation": {
                        "fingerprint": "fp-mcp",
                        "status": "fresh",
                        "head_sha": "abc",
                        "repo_root": absolute_root,
                        "scope": ["src/a.py"],
                    },
                },
                sha256=None,
            )
            store.save_artifact(fresh)

            result = call_tool(
                "puppetmaster_artifacts",
                {
                    "cwd": tmp,
                    "state_dir": str(state),
                    "job_id": job.id,
                    "refs": True,
                },
            )
            self.assertFalse(result.get("isError"), result)
            stdout = json.loads(result["content"][0]["text"])["stdout"]
            mcp_refs = json.loads(stdout)
            self.assertTrue(any(item.get("claim") == "mcp claim" for item in mcp_refs))
            for item in mcp_refs:
                self.assertNotIn("details", item)
                self.assertNotIn(big_body, json.dumps(item))
                self.assertNotIn(absolute_root, json.dumps(item))
                if "validation" in item:
                    self.assertNotIn("repo_root", item["validation"])

            # MCP reset refuses active lease
            live = Task(
                id=task.id,
                job_id=job.id,
                role="implement",
                instruction="implement",
                status=TaskStatus.RUNNING,
                lease_owner="worker-live",
                lease_expires_at=seconds_from_now(120),
                lease_id="lease_live",
            )
            store.save_task(live)
            refused = call_tool(
                "puppetmaster_reset_subgraph",
                {
                    "cwd": tmp,
                    "state_dir": str(state),
                    "job_id": job.id,
                    "task_ids": [task.id],
                },
            )
            self.assertTrue(refused.get("isError"), refused)
            err_text = refused["content"][0]["text"]
            self.assertIn("ActiveTaskLeaseError", err_text)

            # Clear lease and confirm MCP reset surface matches CLI ids field.
            store.save_task(
                Task(
                    id=task.id,
                    job_id=job.id,
                    role="implement",
                    instruction="implement",
                    status=TaskStatus.COMPLETE,
                )
            )
            ok = call_tool(
                "puppetmaster_reset_subgraph",
                {
                    "cwd": tmp,
                    "state_dir": str(state),
                    "job_id": job.id,
                    "task_ids": [task.id],
                },
            )
            self.assertFalse(ok.get("isError"), ok)
            mcp_reset = json.loads(ok["content"][0]["text"])
            self.assertIn("superseded_artifact_ids", mcp_reset)
            self.assertNotIn("superseded_artifacts", mcp_reset)
            self.assertEqual(mcp_reset["superseded_artifact_ids"], [fresh.id])


if __name__ == "__main__":
    unittest.main()
