"""Hermetic tests for Durable Autoresearch (puppetmaster.research)."""
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

from puppetmaster.models import ArtifactType, GraphEdgeType, TaskStatus, seconds_from_now
from puppetmaster.research import (
    RESEARCH_KIND_BEST,
    RESEARCH_KIND_HYPOTHESIS,
    RESEARCH_KIND_INSIGHT,
    RESEARCH_KIND_RESULT,
    RESEARCH_KIND_VERIFICATION,
    SCORE_KEY,
    TOY_HARNESS_ID,
    ResearchLab,
    ToyCompressionHarness,
    analyze_swarm_from_artifacts,
    claim_lock_name,
    experiment_fingerprint,
)
from puppetmaster.store import SwarmStore
from puppetmaster.store_factory import create_store


class ToyCompressionHarnessTests(unittest.TestCase):
    def test_deterministic_and_lower_is_better_ordering(self) -> None:
        harness = ToyCompressionHarness()
        a = harness.run({"level": 1, "seed": 3, "size": 2048})
        b = harness.run({"level": 1, "seed": 3, "size": 2048})
        self.assertEqual(a[SCORE_KEY], b[SCORE_KEY])
        high = harness.run({"level": 9, "seed": 3, "size": 2048})
        # Higher zlib level should not worsen compression on this corpus.
        self.assertLessEqual(high[SCORE_KEY], a[SCORE_KEY])


class ResearchClaimTests(unittest.TestCase):
    def test_claim_exclusivity_and_dedup(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            lab_a = ResearchLab(store=store, worker_id="worker-a")
            lab_b = ResearchLab(store=store, worker_id="worker-b")
            job = lab_a.init_lab("claim exclusivity", label="autoresearch-lab")
            config = {"level": 6, "seed": 1, "size": 1024}
            first = lab_a.claim_experiment(
                job.id, "h1", TOY_HARNESS_ID, config, lease_seconds=60
            )
            self.assertIsNotNone(first)
            assert first is not None
            self.assertEqual(first.status, TaskStatus.RUNNING)
            second = lab_b.claim_experiment(
                job.id, "h1", TOY_HARNESS_ID, config, lease_seconds=60
            )
            self.assertIsNone(second)
            # Same fingerprint must not create a second open task.
            open_same = [
                t
                for t in store.list_tasks(job.id)
                if t.payload.get("fingerprint")
                == experiment_fingerprint("h1", TOY_HARNESS_ID, config)
                and t.status in {TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.BLOCKED}
            ]
            self.assertEqual(len(open_same), 1)

    def test_lease_expiry_allows_reclaim(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            lab_a = ResearchLab(store=store, worker_id="worker-a", lease_seconds=1)
            lab_b = ResearchLab(store=store, worker_id="worker-b", lease_seconds=30)
            job = lab_a.init_lab("reclaim after stale lease")
            config = {"level": 2, "seed": 9, "size": 1024}
            from dataclasses import replace

            first = lab_a.claim_experiment(
                job.id, "stale", TOY_HARNESS_ID, config, lease_seconds=1
            )
            self.assertIsNotNone(first)
            assert first is not None
            # Force lease expiry without waiting on wall clock jitter.
            expired = replace(first, lease_expires_at=seconds_from_now(-2))
            store.save_task(expired)
            # Drop the claim lock so B can acquire (TTL would also expire).
            store.release_lock(
                claim_lock_name(
                    job.id, experiment_fingerprint("stale", TOY_HARNESS_ID, config)
                ),
                owner="worker-a",
            )
            reclaimed = lab_b.claim_experiment(
                job.id, "stale", TOY_HARNESS_ID, config, lease_seconds=30
            )
            self.assertIsNotNone(reclaimed)
            assert reclaimed is not None
            self.assertEqual(reclaimed.id, first.id)
            self.assertEqual(reclaimed.lease_owner, "worker-b")


class ResearchPublishVerifyTests(unittest.TestCase):
    def test_publish_result_schema_and_verify(self) -> None:
        with TemporaryDirectory() as tmp:
            store = create_store("sqlite", Path(tmp) / ".puppetmaster")
            store.init()
            lab = ResearchLab(store=store, worker_id="pub")
            job = lab.init_lab("publish/verify")
            config = {"level": 6, "seed": 4, "size": 2048}
            claimed = lab.claim_experiment(job.id, "ratio", TOY_HARNESS_ID, config)
            self.assertIsNotNone(claimed)
            assert claimed is not None
            metrics = lab.run_claimed(job.id, task_id=claimed.id)
            art = lab.publish_result(job.id, metrics, task_id=claimed.id, keep=True)
            self.assertEqual(art.type, ArtifactType.FINDING)
            self.assertEqual(art.payload.get("research_kind"), RESEARCH_KIND_RESULT)
            self.assertIn("claim", art.payload)
            self.assertIn("metrics", art.payload)
            self.assertIn(SCORE_KEY, art.payload)
            self.assertTrue(art.payload.get("keep"))
            completed = store.get_task_by_id(claimed.id)
            self.assertEqual(completed.status, TaskStatus.COMPLETE)

            verification = lab.verify_claim(job.id, art.id)
            self.assertEqual(verification.type, ArtifactType.VERIFICATION)
            self.assertEqual(
                verification.payload.get("research_kind"), RESEARCH_KIND_VERIFICATION
            )
            self.assertTrue(verification.payload.get("passed"))
            self.assertEqual(verification.payload.get("result"), "pass")
            edges = store.list_edges(job.id, edge_type=GraphEdgeType.DERIVED_FROM)
            self.assertTrue(
                any(
                    e.from_id == verification.id and e.to_id == art.id for e in edges
                )
            )

    def test_verify_claim_fail_on_tampered_score(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            lab = ResearchLab(store=store, worker_id="tamper")
            job = lab.init_lab("verify fail")
            config = {"level": 3, "seed": 2, "size": 1024}
            claimed = lab.claim_experiment(job.id, "bad", TOY_HARNESS_ID, config)
            assert claimed is not None
            metrics = dict(lab.run_claimed(job.id, task_id=claimed.id))
            art = lab.publish_result(job.id, metrics, task_id=claimed.id)
            # Tamper the stored score so re-run disagrees.
            from dataclasses import replace

            tampered = replace(
                art,
                payload={
                    **art.payload,
                    SCORE_KEY: float(art.payload[SCORE_KEY]) + 1.0,
                    "metrics": {
                        **art.payload["metrics"],
                        SCORE_KEY: float(art.payload[SCORE_KEY]) + 1.0,
                    },
                },
            )
            # Overwrite artifact file directly (file store).
            path = store.job_dir(job.id) / "artifacts" / f"{art.id}.json"
            path.write_text(
                json.dumps(
                    {
                        "job_id": tampered.job_id,
                        "task_id": tampered.task_id,
                        "type": str(tampered.type),
                        "created_by": tampered.created_by,
                        "payload": tampered.payload,
                        "confidence": tampered.confidence,
                        "evidence": tampered.evidence,
                        "id": tampered.id,
                        "created_at": tampered.created_at,
                        "sha256": tampered.sha256,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            verification = lab.verify_claim(job.id, art.id)
            self.assertFalse(verification.payload.get("passed"))
            self.assertEqual(verification.payload.get("result"), "fail")


class ResearchLeaderboardDemoTests(unittest.TestCase):
    def test_leaderboard_ordering(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            lab = ResearchLab(store=store, worker_id="board")
            job = lab.init_lab("leaderboard")
            for level in (1, 9, 5):
                config = {"level": level, "seed": 11, "size": 2048}
                claimed = lab.claim_experiment(
                    job.id, f"L{level}", TOY_HARNESS_ID, config
                )
                assert claimed is not None
                metrics = lab.run_claimed(job.id, task_id=claimed.id)
                lab.publish_result(job.id, metrics, task_id=claimed.id, keep=True)
            board = lab.leaderboard(job.id)
            self.assertEqual(len(board), 3)
            scores = [float(a.payload[SCORE_KEY]) for a in board]
            self.assertEqual(scores, sorted(scores))
            best = lab.pull_best(job.id)
            self.assertIsNotNone(best)
            assert best is not None
            self.assertEqual(best.payload.get("research_kind"), RESEARCH_KIND_BEST)

    def test_demo_smoke_temp_state_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = create_store("sqlite", root / ".puppetmaster")
            store.init()
            lab = ResearchLab(store=store, worker_id="demo")
            brief = root / "brief.md"
            result = lab.run_demo(brief_path=brief)
            self.assertTrue(brief.is_file())
            text = brief.read_text(encoding="utf-8")
            self.assertIn("durable", text.lower())
            self.assertIn("zero", text.lower())
            self.assertIn("ExperimentHarness", text)
            self.assertGreaterEqual(len(result["results"]), 2)
            self.assertEqual(len(result["results"]), len(result["verifications"]))
            self.assertTrue(result["leaderboard"])
            status = result["status"]
            self.assertGreaterEqual(status["by_research_kind"].get(RESEARCH_KIND_RESULT, 0), 2)
            self.assertGreaterEqual(
                status["by_research_kind"].get(RESEARCH_KIND_VERIFICATION, 0), 2
            )
            self.assertGreaterEqual(
                status["by_research_kind"].get(RESEARCH_KIND_INSIGHT, 0), 1
            )
            self.assertGreaterEqual(
                status["by_research_kind"].get(RESEARCH_KIND_HYPOTHESIS, 0), 1
            )
            think = analyze_swarm_from_artifacts(store, result["job_id"])
            self.assertTrue(think["leaderboard"])


class ResearchCliSmokeTests(unittest.TestCase):
    def test_research_demo_help(self) -> None:
        from puppetmaster.cli._parser import build_parser
        from io import StringIO
        from contextlib import redirect_stdout

        parser = build_parser()
        buf = StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stdout(buf):
                parser.parse_args(["research", "demo", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("ToyCompressionHarness", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
