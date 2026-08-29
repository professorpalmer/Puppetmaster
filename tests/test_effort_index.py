from __future__ import annotations

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from tempfile import TemporaryDirectory

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

from puppetmaster.lifecycle import (
    index_effort_artifacts,
    latest_tagged_effort_id,
    rollup_stores,
    tag_job_effort,
)
from puppetmaster.models import Artifact, ArtifactType, Job, Task
from puppetmaster.store import SwarmStore


def _task(store: SwarmStore, job_id: str, role: str = "explore") -> Task:
    task = Task(job_id=job_id, role=role, instruction=f"{role} work")
    store.save_task(task)
    return task


def _finding(store: SwarmStore, job: Job, task: Task, claim: str, **payload) -> Artifact:
    body = {"claim": claim}
    body.update(payload)
    artifact = Artifact(
        job_id=job.id,
        task_id=task.id,
        type=ArtifactType.FINDING,
        created_by="tester",
        payload=body,
        confidence=0.9,
        evidence=["tests"],
    )
    store.save_artifact(artifact)
    return artifact


def _verification(store: SwarmStore, job: Job, task: Task, check: str) -> Artifact:
    artifact = Artifact(
        job_id=job.id,
        task_id=task.id,
        type=ArtifactType.VERIFICATION,
        created_by="tester",
        payload={"check": check, "result": "pass"},
        confidence=0.8,
        evidence=["gate"],
    )
    store.save_artifact(artifact)
    return artifact


def _decision(store: SwarmStore, job: Job, task: Task, decision: str) -> Artifact:
    artifact = Artifact(
        job_id=job.id,
        task_id=task.id,
        type=ArtifactType.DECISION,
        created_by="tester",
        payload={"decision": decision, "why": "because"},
        confidence=0.7,
        evidence=["plan"],
    )
    store.save_artifact(artifact)
    return artifact


class EffortIndexTests(unittest.TestCase):
    def test_index_is_compact_and_cross_job(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(tmp)
            store.init()
            job_a = store.create_job("first")
            job_b = store.create_job("second")
            untagged = store.create_job("noise")
            task_a = _task(store, job_a.id)
            task_b = _task(store, job_b.id, "review")
            task_u = _task(store, untagged.id)
            finding = _finding(
                store,
                job_a,
                task_a,
                "auth cookie is stale",
                transcript="NEVER INCLUDE THIS TRANSCRIPT",
            )
            _decision(store, job_b, task_b, "rewrite the session store")
            _finding(store, untagged, task_u, "untagged should not appear")
            tag_job_effort(store, job_a.id, "mig-auth")
            tag_job_effort(store, job_b.id, "mig-auth")

            payload = index_effort_artifacts([store], effort_id="mig-auth")
            self.assertEqual(payload["effort_id"], "mig-auth")
            self.assertEqual(payload["jobs"], 2)
            self.assertEqual(payload["count"], 2)
            ids = {row["id"] for row in payload["refs"]}
            self.assertIn(finding.id, ids)
            for row in payload["refs"]:
                self.assertIn("job_id", row)
                self.assertIn("task_id", row)
                self.assertIn("sha256", row)
                self.assertIn("created_at", row)
                self.assertNotIn("payload", row)
                self.assertNotIn("transcript", row)
                dumped = json.dumps(row)
                self.assertNotIn("NEVER INCLUDE THIS TRANSCRIPT", dumped)
            ledger = rollup_stores([store], effort_id="mig-auth")
            self.assertEqual(ledger["jobs"], 2)
            self.assertEqual(ledger["artifacts"], 2)

    def test_latest_tagged_effort_uses_sidecar_timestamp(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(tmp)
            store.init()
            older = store.create_job("older")
            newer = store.create_job("newer")
            tag_job_effort(store, older.id, "effort-old")
            tag_job_effort(store, newer.id, "effort-new")
            older_path = store.job_dir(older.id) / "effort.json"
            newer_path = store.job_dir(newer.id) / "effort.json"
            store.write_json(
                older_path, {"effort_id": "effort-old", "tagged_at": "2020-01-01T00:00:00+00:00"}
            )
            store.write_json(
                newer_path, {"effort_id": "effort-new", "tagged_at": "2026-08-23T12:00:00+00:00"}
            )
            _finding(store, older, _task(store, older.id), "old claim")
            _finding(store, newer, _task(store, newer.id), "new claim")

            self.assertEqual(latest_tagged_effort_id([store]), "effort-new")
            payload = index_effort_artifacts([store])
            self.assertEqual(payload["effort_id"], "effort-new")
            self.assertEqual(payload["requested_effort_id"], None)
            self.assertEqual([row["claim"] for row in payload["refs"]], ["new claim"])

    def test_type_and_query_filters(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(tmp)
            store.init()
            job = store.create_job("filters")
            task = _task(store, job.id)
            tag_job_effort(store, job.id, "filt")
            _finding(store, job, task, "stale cookie on login")
            _verification(store, job, task, "cookie refresh path")
            _decision(store, job, task, "keep the cookie jar")

            by_type = index_effort_artifacts(
                [store], effort_id="filt", artifact_type="verification"
            )
            self.assertEqual(by_type["count"], 1)
            self.assertEqual(by_type["refs"][0]["type"], "verification")
            self.assertEqual(by_type["refs"][0]["check"], "cookie refresh path")

            by_query = index_effort_artifacts(
                [store], effort_id="filt", query="cookie"
            )
            self.assertEqual(by_query["count"], 3)
            miss = index_effort_artifacts(
                [store], effort_id="filt", query="oauth"
            )
            self.assertEqual(miss["count"], 0)

    def test_expand_omits_transcript_keys(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(tmp)
            store.init()
            job = store.create_job("expand")
            task = _task(store, job.id)
            tag_job_effort(store, job.id, "exp")
            _finding(
                store,
                job,
                task,
                "compact by default",
                transcript="secret-transcript",
                extra="keep-me",
            )
            compact = index_effort_artifacts([store], effort_id="exp")
            self.assertNotIn("payload", compact["refs"][0])
            expanded = index_effort_artifacts([store], effort_id="exp", expand=True)
            payload = expanded["refs"][0]["payload"]
            self.assertEqual(payload["claim"], "compact by default")
            self.assertEqual(payload["extra"], "keep-me")
            self.assertNotIn("transcript", payload)

    def test_cli_and_mcp_surface(self) -> None:
        from puppetmaster.cli import build_parser, main
        from puppetmaster.mcp_remote import SUPERVISE_TOOL_NAMES
        from puppetmaster.mcp_server import effort_index_schema, tools

        names = {tool.name for tool in tools()}
        self.assertIn("puppetmaster_effort_index", names)
        self.assertIn("puppetmaster_effort_index", SUPERVISE_TOOL_NAMES)
        schema = effort_index_schema()
        for key in ("effort_id", "type", "query", "expand", "all_projects"):
            self.assertIn(key, schema["properties"])

        parser = build_parser()
        args = parser.parse_args(["effort-index", "--effort", "x", "--type", "finding", "--query", "q", "--json"])
        self.assertEqual(args.command, "effort-index")
        self.assertEqual(args.effort, "x")
        self.assertEqual(args.artifact_type, "finding")
        self.assertTrue(args.json)

        with TemporaryDirectory() as tmp:
            store = SwarmStore(tmp)
            store.init()
            job = store.create_job("cli")
            task = _task(store, job.id)
            tag_job_effort(store, job.id, "cli-effort")
            _finding(store, job, task, "cli visible claim")
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(
                    [
                        "--backend",
                        "file",
                        "--state-dir",
                        tmp,
                        "effort-index",
                        "--effort",
                        "cli-effort",
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0)
            body = json.loads(buf.getvalue())
            self.assertEqual(body["effort_id"], "cli-effort")
            self.assertEqual(body["count"], 1)
            self.assertEqual(body["refs"][0]["claim"], "cli visible claim")
            self.assertNotIn("payload", body["refs"][0])

    def test_stale_finding_omitted_fresh_finding_remains(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(tmp)
            store.init()
            job = store.create_job("freshness index")
            task = _task(store, job.id)
            tag_job_effort(store, job.id, "freshness-effort")
            stale = _finding(
                store,
                job,
                task,
                "stale cited claim",
                validation={
                    "status": "stale",
                    "source_digests": {"src/a.py": "deadbeef"},
                },
            )
            fresh = _finding(
                store,
                job,
                task,
                "fresh cited claim",
                validation={
                    "status": "fresh",
                    "source_digests": {"src/a.py": "abcd"},
                },
            )
            unlabeled = _finding(store, job, task, "unlabeled cited claim")
            superseded = _finding(
                store,
                job,
                task,
                "superseded cited claim",
                validation={"status": "superseded"},
            )
            payload = index_effort_artifacts(
                [store], effort_id="freshness-effort"
            )
            ids = {row["id"] for row in payload["refs"]}
            self.assertIn(fresh.id, ids)
            self.assertIn(unlabeled.id, ids)
            self.assertNotIn(stale.id, ids)
            self.assertNotIn(superseded.id, ids)


if __name__ == "__main__":
    unittest.main()
