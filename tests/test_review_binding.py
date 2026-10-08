"""A judge's PASS binds to the files it reviewed.

Codex benchmark: four judges passed, then a repair worker that a stop had
missed edited the shared helper. The delivered source was not the reviewed
source. A judge now waits for live writers on its files, records a digest of
them, turns a PASS on files that changed during review into PARTIAL, and a run
whose reviewed files changed after the PASS ends failed, not done.
"""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import threading
import time
from unittest.mock import patch

from puppetmaster import flow
from puppetmaster.file_claims import FileClaimRegistry
from puppetmaster.flow import JobNodeExecutor, NodeOutcome

from tests.test_flow_runtime import Base, agent, graph, judge


class ReviewBindingTests(Base):
    def setUp(self):
        super().setUp()
        self.claims_db = self.root / "claims.sqlite3"
        patcher = patch("puppetmaster.file_claims.default_file_claim_db_path", return_value=self.claims_db)
        patcher.start()
        self.addCleanup(patcher.stop)
        (self.work / "src").mkdir()
        (self.work / "src" / "a.py").write_text("v1\n", encoding="utf-8")

    def judge_run(self, on_launch=None, verdict="PASS"):
        g = graph([agent("build", files=["src/*.py"]), judge("review")],
                  [{"from": "build", "to": "review"}])
        run = flow.new_run(self.state, g, cwd=str(self.work))
        run.current = "review"
        run.inflight = {"node": "review", "visit": 1, "attempt": 0, "job_ids": []}
        executor = JobNodeExecutor(self.state)
        executor.poll_seconds = 0.05

        def launch(*_args, **_kwargs):
            if on_launch:
                on_launch()
            return "job_r", None, None

        with patch.object(JobNodeExecutor, "_launch", side_effect=launch), \
                patch.object(flow, "task_outcome",
                             return_value=NodeOutcome(ok=True, verdict=verdict, task_id="t")):
            return run, executor(run.graph["nodes"][1], run, "")

    def test_a_pass_records_the_reviewed_files(self):
        _, outcome = self.judge_run()
        self.assertEqual(outcome.verdict, "PASS")
        self.assertEqual(outcome.reviewed["count"], 1)
        self.assertIn("src/a.py", outcome.reviewed["hashes"])
        self.assertEqual(outcome.reviewed["scope"], ["src/*.py"])

    def test_a_change_during_review_turns_pass_into_partial(self):
        def concurrent_write():
            (self.work / "src" / "a.py").write_text("v2\n", encoding="utf-8")

        _, outcome = self.judge_run(on_launch=concurrent_write)
        self.assertEqual(outcome.verdict, "PARTIAL")
        self.assertIn("changed during the review", outcome.reason)
        self.assertIn("src/a.py", outcome.reason)

    def test_the_judge_waits_for_a_live_writer(self):
        registry = FileClaimRegistry(self.claims_db)
        claims = registry.acquire_many(self.work, ["."], "worker-repair", 30)
        released_at = {}

        def finish_writing():
            time.sleep(0.4)
            (self.work / "src" / "a.py").write_text("final\n", encoding="utf-8")
            registry.release_many(self.work, [(c.path, c.claim_id) for c in claims])
            released_at["t"] = time.monotonic()

        writer = threading.Thread(target=finish_writing)
        writer.start()
        launched_at = {}
        run, outcome = self.judge_run(on_launch=lambda: launched_at.setdefault("t", time.monotonic()))
        writer.join()
        self.assertGreaterEqual(launched_at["t"], released_at["t"])
        self.assertEqual(outcome.verdict, "PASS")
        self.assertEqual(run.inflight["review_waits_for"][0]["owner"], "worker-repair")
        # The digest is of the final source, written before the review began.
        before = flow._source_state(str(self.work), ["src/*.py"])
        self.assertEqual(outcome.reviewed["digest"], before["digest"])

    def test_a_change_after_pass_fails_the_run(self):
        g = graph([agent("build", files=["src/*.py"]), judge("review"), {"id": "end", "kind": "end"}],
                  [{"from": "build", "to": "review"}, {"from": "review", "to": "end", "when": "PASS"}])

        def execute(node, run, prev):
            if node["kind"] == "agent":
                return NodeOutcome(ok=True, verdict="PASS")
            state = flow._source_state(run.graph["cwd"], ["src/*.py"])
            state["scope"] = ["src/*.py"]
            # A late writer edits the reviewed file after the PASS.
            (self.work / "src" / "a.py").write_text("late\n", encoding="utf-8")
            return NodeOutcome(ok=True, verdict="PASS", reviewed=state)

        run = self.start(g, execute)
        self.assertEqual(run.status, "failed")
        self.assertIn("stale review", run.reason)
        self.assertIn("src/a.py", run.reason)

    def test_an_unchanged_pass_ends_done(self):
        g = graph([agent("build", files=["src/*.py"]), judge("review"), {"id": "end", "kind": "end"}],
                  [{"from": "build", "to": "review"}, {"from": "review", "to": "end", "when": "PASS"}])

        def execute(node, run, prev):
            if node["kind"] == "agent":
                return NodeOutcome(ok=True, verdict="PASS")
            state = flow._source_state(run.graph["cwd"], ["src/*.py"])
            state["scope"] = ["src/*.py"]
            return NodeOutcome(ok=True, verdict="PASS", reviewed=state)

        self.assertEqual(self.start(g, execute).status, "done")

    def test_no_declared_files_means_no_binding(self):
        g = graph([agent("build"), judge("review")], [{"from": "build", "to": "review"}])
        run = flow.new_run(self.state, g, cwd=str(self.work))
        self.assertIsNone(JobNodeExecutor(self.state)._prepare_review(run, ""))


if __name__ == "__main__":
    import unittest
    unittest.main()
