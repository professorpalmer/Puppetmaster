from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from puppetmaster.artifact_status import CLAIM_SUPPORT_INDEPENDENT
from puppetmaster.mcp_server import call_tool
from puppetmaster.models import Artifact, ArtifactType, JobStatus, Task, TaskStatus
from puppetmaster.sqlite_store import SQLiteSwarmStore
from puppetmaster.stitcher import Stitcher

GOAL = (
    "Audit the payment ingestion pipeline end to end and report concrete defects. "
    + "Cover retries, idempotency keys, ledger reconciliation, and webhook replay. " * 12
).strip()

ROLES = ("explore", "audit", "review", "redteam")

DUPLICATE_CLAIM = "Webhook replay skips idempotency key check in ingest.py handler"

DISTINCT_CLAIMS = {
    "explore": (
        "Ledger rounding drift accumulates across currency conversions in batch close",
        "Exponential backoff ignores Retry-After headers sent by the processor",
        "Nullable refund_reason column breaks downstream export joins",
        "Service token cached past rotation window keeps stale credentials alive",
    ),
    "audit": (
        "Settlement cron overlaps itself when a prior run exceeds fifteen minutes",
        "Chargeback notifications are dropped whenever the queue consumer restarts",
        "Partial captures never release the remaining authorization hold",
        "Audit trail omits operator identity for manual ledger adjustments",
    ),
    "review": (
        "Decimal parsing accepts locale commas and silently scales amounts",
        "Payout batching sorts by merchant name instead of stable merchant id",
        "Fraud score threshold is hardcoded and bypasses configuration overrides",
        "Dead letter table grows unbounded without any retention sweep",
    ),
    "redteam": (
        "Signature verification compares digests with a timing-unsafe equality",
        "Callback URL allowlist permits arbitrary subdomains of tenant domains",
        "Debug endpoint exposes raw card fingerprints without authentication",
        "Replay window tolerance of one day lets attackers resend captured events",
    ),
}


class AwaitCompactTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        self.store = SQLiteSwarmStore(Path(self.tmp) / ".puppetmaster")
        self.store.init()
        env = patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("PUPPETMASTER_AWAIT_SUMMARY", None)

    def _fixture_job(self) -> tuple[str, list[Task]]:
        store = self.store
        job = store.create_job(GOAL)
        tasks = [
            Task(
                job_id=job.id,
                role=role,
                instruction=f"Role: {role}\nGoal: {GOAL}",
                status=TaskStatus.COMPLETE,
            )
            for role in ROLES
        ]
        store.save_tasks(tasks)
        artifacts = []
        for index, task in enumerate(tasks):
            artifacts.append(
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.FINDING,
                    created_by=task.role,
                    confidence=0.8 + index * 0.01,
                    evidence=["ingest.py:42", f"adapter:{task.role}"],
                    payload={"claim": DUPLICATE_CLAIM},
                    claim_support_status=CLAIM_SUPPORT_INDEPENDENT,
                )
            )
            artifacts.extend(
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.FINDING,
                    created_by=task.role,
                    confidence=0.7,
                    evidence=[f"{task.role}_{position}.py:{10 + index}"],
                    payload={"claim": claim},
                    claim_support_status=CLAIM_SUPPORT_INDEPENDENT,
                )
                for position, claim in enumerate(DISTINCT_CLAIMS[task.role])
            )
            failed = task.role == "redteam"
            artifacts.append(
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.VERIFICATION,
                    created_by=task.role,
                    confidence=0.2 if failed else 0.9,
                    evidence=[f"adapter:{task.role}"],
                    payload={
                        "check": f"Role: {task.role}\nGoal: {GOAL}",
                        "result": "failed" if failed else "passed",
                        **({"failure": "rate_limit"} if failed else {}),
                    },
                )
            )
        artifacts.append(
            Artifact(
                job_id=job.id,
                task_id=tasks[0].id,
                type=ArtifactType.DECISION,
                created_by="explore",
                confidence=0.85,
                evidence=["ingest.py:42"],
                payload={
                    "decision": "Gate replay on a persisted idempotency key",
                    "why": "Replays currently double-post ledger entries",
                },
                claim_support_status=CLAIM_SUPPORT_INDEPENDENT,
            )
        )
        store.save_artifacts(artifacts)
        store.update_job_status(job.id, JobStatus.COMPLETE)
        Stitcher(store).stitch(job.id)
        return job.id, tasks

    def _await(self, job_id: str, **extra) -> tuple[dict, dict]:
        args = {
            "cwd": self.tmp,
            "state_dir": str(self.store.root),
            "backend": "sqlite",
            "job_id": job_id,
            "timeout_seconds": 1,
        }
        args.update(extra)
        result = call_tool("puppetmaster_await_job", args)
        return result, json.loads(result["content"][0]["text"])

    def test_default_compact_digest(self) -> None:
        job_id, tasks = self._fixture_job()
        result, body = self._await(job_id)
        _, full = self._await(job_id, summary="full")

        self.assertEqual(body["summary_mode"], "compact")
        self.assertNotIn("summary", body)
        digest = body["digest"]
        self.assertEqual(digest["counts"]["finding"], 20)
        self.assertEqual(digest["counts"]["verification"], 4)
        self.assertEqual(digest["counts"]["decision"], 1)

        self.assertEqual(digest["findings"][0]["text"], DUPLICATE_CLAIM)
        self.assertEqual(digest["findings"][0]["reported_by"], 4)
        self.assertEqual(
            sum(f["text"] == DUPLICATE_CLAIM for f in digest["findings"]), 1
        )
        self.assertEqual(len(digest["findings"]), 8)
        self.assertEqual(digest["decisions"][0]["reported_by"], 1)
        self.assertEqual(digest["truncated"], {"findings": 9})

        redteam = next(t for t in tasks if t.role == "redteam")
        self.assertEqual(
            digest["exceptions"],
            [
                {
                    "task_id": redteam.id,
                    "role": "redteam",
                    "result": "failed",
                    "reason": "rate_limit",
                }
            ],
        )

        text = result["content"][0]["text"]
        self.assertNotIn(GOAL[:200], text)
        stitched = self.store.job_dir(job_id) / "summaries" / "stitched.md"
        self.assertEqual(body["summary_ref"]["path"], str(stitched))
        self.assertEqual(body["summary_ref"]["chars"], len(full["summary"]))
        self.assertEqual(len(body["summary_ref"]["sha256"]), 64)

        compact_bytes = len(json.dumps(body))
        full_bytes = len(json.dumps(full))
        self.assertGreaterEqual(full_bytes, 5 * compact_bytes, (full_bytes, compact_bytes))
        self.assertEqual(result["isError"], self._await(job_id, summary="full")[0]["isError"])

    def test_full_matches_stitched_text(self) -> None:
        job_id, _ = self._fixture_job()
        _, body = self._await(job_id, summary="full")
        stitched = self.store.job_dir(job_id) / "summaries" / "stitched.md"
        self.assertEqual(body["summary_mode"], "full")
        self.assertEqual(body["summary"], stitched.read_text(encoding="utf-8"))
        self.assertNotIn("digest", body)

    def test_none_returns_state_only(self) -> None:
        job_id, _ = self._fixture_job()
        _, body = self._await(job_id, summary="none")
        self.assertEqual(body["summary_mode"], "none")
        for key in ("summary", "digest", "summary_ref"):
            self.assertNotIn(key, body)
        self.assertEqual(body["status"], "complete")

    def test_env_override_and_explicit_arg_wins(self) -> None:
        job_id, _ = self._fixture_job()
        os.environ["PUPPETMASTER_AWAIT_SUMMARY"] = "full"
        _, body = self._await(job_id)
        self.assertEqual(body["summary_mode"], "full")
        self.assertIn("summary", body)
        _, body = self._await(job_id, summary="compact")
        self.assertEqual(body["summary_mode"], "compact")
        self.assertNotIn("summary", body)

    def test_timed_out_await_unchanged_apart_from_mode(self) -> None:
        job = self.store.create_job("never finishes")
        result, body = self._await(job.id, timeout_seconds=0.3, poll_interval_seconds=0.05)
        self.assertTrue(body["timed_out"])
        self.assertFalse(body["terminal"])
        self.assertEqual(body["summary"], "")
        self.assertEqual(body["summary_mode"], "compact")
        self.assertNotIn("digest", body)
        self.assertNotIn("summary_ref", body)
        self.assertFalse(result["isError"])

    def test_invalid_summary_mode_fails_before_blocking(self) -> None:
        import time

        job = self.store.create_job("never finishes")
        args = {
            "cwd": self.tmp,
            "state_dir": str(self.store.root),
            "backend": "sqlite",
            "job_id": job.id,
            "timeout_seconds": 20,
            "summary": "bogus",
        }
        started = time.monotonic()
        result = call_tool("puppetmaster_await_job", args)
        self.assertTrue(result["isError"])
        self.assertIn("summary must be one of", result["content"][0]["text"])
        self.assertLess(time.monotonic() - started, 5)

    def test_digest_failure_falls_back(self) -> None:
        job_id, _ = self._fixture_job()
        _, baseline = self._await(job_id)
        with patch.object(Stitcher, "digest", side_effect=RuntimeError("boom")):
            result, body = self._await(job_id)
        self.assertEqual(result["isError"], self._await(job_id)[0]["isError"])
        self.assertNotIn("digest", body)
        self.assertNotIn("summary", body)
        self.assertIn("boom", body["digest_error"])
        self.assertEqual(body["summary_ref"], baseline["summary_ref"])
        self.assertEqual(body["status"], "complete")


class StitcherDigestTests(unittest.TestCase):
    def test_truncation_and_clipping(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SQLiteSwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("goal")
            task = Task(job_id=job.id, role="audit", instruction="x", status=TaskStatus.COMPLETE)
            store.save_tasks([task])
            store.save_artifacts(
                Artifact(
                    job_id=job.id,
                    task_id=task.id,
                    type=ArtifactType.FINDING,
                    created_by="audit",
                    confidence=0.5,
                    evidence=[f"file{i}.py:1"],
                    payload={"claim": f"{'zq' * i} unrelated{i} " + "x" * 400},
                )
                for i in range(1, 6)
            )
            digest = Stitcher(store).digest(job.id, limit=3, max_chars=50)
            self.assertEqual(len(digest["findings"]), 3)
            self.assertEqual(digest["truncated"], {"findings": 2})
            for item in digest["findings"]:
                self.assertLessEqual(len(item["text"]), 50)
                self.assertTrue(item["text"].endswith("..."))
            self.assertEqual(digest["exceptions"][0]["reason"], "no verification artifact")


if __name__ == "__main__":
    unittest.main()
