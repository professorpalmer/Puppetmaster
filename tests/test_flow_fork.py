"""Map builders fork one shared session; judges stay fresh.

Codex R169/R179 (PM 1.45.1, N16 grouped, Sol 6.1 high): every cold group
builder started a fresh session and read the same contracts, rubric, world
API and checks again (178-261 s for 4 regions; native built all 16 in about
350 s). Native multiworker builders start from the pilot's context. A map
``fork: {"from": <agent>}`` gives PM builders the same start: one session
reads the shared inputs once, and each item's code builder forks it. Judges
never fork, so a review never shares a builder's context.
"""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import tempfile
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from puppetmaster import flow
from puppetmaster.adapters import CodexAdapter
from puppetmaster.adapters import codex as codex_module
from puppetmaster.adapters.codex import build_codex_resume_command
from puppetmaster.flow import JobNodeExecutor, NodeOutcome
from puppetmaster.models import Task
from puppetmaster.session_lease import acquire_codex_thread
from puppetmaster.worker_resume import claim_resumed_session, resolve_worker_resume

from test_worker_resume import FakeStore, _codex_events, _run_adapter, _task as _store_task

_HOMES = tempfile.TemporaryDirectory()
_ENV = patch.dict("os.environ", {
    "CODEX_HOME": str(Path(_HOMES.name) / "codex"),
    "CLAUDE_CONFIG_DIR": str(Path(_HOMES.name) / "claude"),
    "PUPPETMASTER_HOME": str(Path(_HOMES.name) / "pm"),
})


def setUpModule() -> None:
    _ENV.start()


def tearDownModule() -> None:
    _ENV.stop()
    _HOMES.cleanup()


SOURCE = "0199a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b"
FORKED = "0199a1b2-ffff-7e5f-8a9b-0c1d2e3f4a5b"


class ForkResolutionTests(unittest.TestCase):
    def setUp(self):
        self.store = FakeStore()
        self.store.add("job-prime", _store_task("t-prime", "prime", "codex", job_id="job-prime"),
                       {"adapter": "codex", "ephemeral": False, "thread_id": SOURCE})

    def test_fork_from_resolves_like_resume_from_and_says_fork(self):
        record = resolve_worker_resume(self.store, {"fork_from": {"job_id": "job-prime", "task_id": "t-prime"}},
                                       "codex")
        self.assertEqual(record, {"status": "resolved", "adapter": "codex", "session_id": SOURCE,
                                  "from_job_id": "job-prime", "from_task_id": "t-prime", "mode": "fork"})
        missing = resolve_worker_resume(self.store, {"fork_from": {"job_id": "nope"}}, "codex")
        self.assertEqual((missing["status"], missing["mode"]), ("unavailable", "fork"))

    def test_any_number_of_tasks_can_fork_one_codex_thread(self):
        record = {"status": "resolved", "adapter": "codex", "session_id": SOURCE, "mode": "fork"}
        claimed: dict = {}
        self.assertEqual(claim_resumed_session(record, claimed, "group_00"), record)
        self.assertEqual(claim_resumed_session(record, claimed, "group_01"), record)
        self.assertEqual(claimed, {})

    def test_resume_from_wins_over_fork_from(self):
        record = resolve_worker_resume(self.store, {"resume_from": {"job_id": "job-prime", "task_id": "t-prime"},
                                                    "fork_from": {"job_id": "job-prime"}}, "codex")
        self.assertNotIn("mode", record)


class CodexForkTests(unittest.TestCase):
    def task(self, **payload) -> Task:
        return Task(job_id="job-b", role="build", instruction="Build regions/a.py.", adapter="codex",
                    payload={"cwd": str(Path.cwd()), "model": "gpt-6.1-sol", "write_scope": ["regions/a.py"],
                             **payload})

    def record(self):
        return {"status": "resolved", "adapter": "codex", "session_id": SOURCE,
                "from_job_id": "job-prime", "from_task_id": "t-prime", "mode": "fork"}

    def test_fork_argv(self):
        command = build_codex_resume_command(executable="codex", session_id=SOURCE, fork=True,
                                             sandbox="workspace-write", model="gpt-6.1-sol")
        self.assertEqual(command[:5], ["codex", "exec", "fork", SOURCE, "--json"])
        self.assertIn('sandbox_mode="workspace-write"', command)
        self.assertEqual(command[-2:], ["--", "-"])

    def test_a_fork_needs_no_thread_lease_sends_the_build_contract_and_records_its_own_thread(self):
        held = acquire_codex_thread(SOURCE)  # another job continues the source in place
        self.assertIsNotNone(held)
        stdout = _codex_events(
            {"type": "thread.started", "thread_id": FORKED},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "VERDICT: PASS - built"}},
            {"type": "turn.completed", "usage": {"input_tokens": 126420, "cached_input_tokens": 84096,
                                                 "output_tokens": 83}},
        )
        try:
            artifacts, kwargs, enrich, _ = _run_adapter(CodexAdapter(), self.task(resume=self.record()), stdout)
        finally:
            held.release()
        command = kwargs["command"]
        self.assertEqual(command[:5], ["/usr/bin/codex", "exec", "fork", SOURCE, "--json"])
        prompt = kwargs.get("stdin_data") or ""
        self.assertIn("Build mode", prompt)
        self.assertIn("Build regions/a.py.", prompt)
        enrich.assert_not_called()  # the source session already holds the shared context
        receipt = artifacts[0].payload
        self.assertEqual(receipt["resume"]["mode"], "fork")
        self.assertEqual(receipt["thread_id"], FORKED)
        self.assertFalse(receipt["ephemeral"])  # a later repair resumes the fork itself
        # The SDK total counts the copied source turns: never the attempt's own.
        self.assertEqual(receipt["sdk_usage_scope"], "session_cumulative")

    def test_fork_usage_comes_from_the_new_threads_rollout(self):
        seen = []

        def rollout_path(home, thread):
            seen.append(thread)
            return Path(f"/rollouts/{thread}.jsonl")

        stdout = _codex_events({"type": "thread.started", "thread_id": FORKED})
        with patch.object(codex_module.codex_rollout, "rollout_path", side_effect=rollout_path), \
                patch.object(codex_module.codex_rollout, "attempt_usage",
                             side_effect=lambda path, base: {"path": str(path), "base": sorted(base)}):
            found = codex_module._rollout_attempt_usage(Path("/home"), self.record(),
                                                        frozenset({"source-turn"}), stdout)
        self.assertEqual(seen, [FORKED])
        self.assertEqual(found, {"path": f"/rollouts/{FORKED}.jsonl", "base": ["source-turn"]})


ITEM = {"entry": "build",
        "nodes": [{"id": "build", "kind": "agent", "task": "Build {{item.id}}.", "files": "{{item.files}}"},
                  {"id": "review", "kind": "judge", "task": "Review {{item.id}}.", "files": "{{item.files}}"},
                  {"id": "done", "kind": "end", "status": "pass"}],
        "edges": [{"from": "build", "to": "review"}, {"from": "review", "to": "done", "when": "PASS"}]}


def graph(fork=True):
    groups = {"id": "groups", "kind": "map", "concurrency": 2, "graph": ITEM,
              "items": [{"id": key, "files": [f"regions/{key}.py"]} for key in ("group_00", "group_01")]}
    if fork:
        groups["fork"] = {"from": "prime"}
    return {"id": "fork-flow", "entry": "prime", "defaults": {"adapter": "codex"},
            "nodes": [{"id": "prime", "kind": "agent", "role": "explore", "task": "Read the shared inputs."},
                      groups, {"id": "done", "kind": "end", "status": "pass"}],
            "edges": [{"from": "prime", "to": "groups"}, {"from": "groups", "to": "done", "when": "ok"}]}


class FlowForkTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state, self.work = Path(self._tmp.name) / "state", Path(self._tmp.name) / "work"
        self.state.mkdir()
        (self.work / "regions").mkdir(parents=True)

    def test_validation(self):
        bad = graph()
        bad["nodes"][1]["fork"] = {"from": "done"}
        self.assertIn("map 'groups' fork from 'done' is not an agent node", flow.validate_graph(bad))
        bad["nodes"][1]["fork"] = {"source": "prime"}
        self.assertIn("map 'groups' fork must be an object with from: the agent node to fork",
                      flow.validate_graph(bad))
        self.assertEqual(flow.validate_graph(graph()), [])

    def child(self, fork_from, sessions=None):
        parent = {"run_id": "flow_parent00000", "node": "groups", "key": "group_00", "depth": 1,
                  "peer_scopes": ["regions/group_01.py"], "fork_from": fork_from}
        run = flow.new_run(self.state, {**ITEM, "defaults": {"adapter": "codex"}}, "",
                           state={"item": {"id": "group_00", "files": ["regions/group_00.py"]}},
                           parent=parent, cwd=str(self.work))
        run.sessions.update(sessions or {})
        return run

    def spec(self, run, node_id, visit=1):
        node = next(node for node in run.graph["nodes"] if node["id"] == node_id)
        return JobNodeExecutor(self.state)._spec(node_id, node, run, run.input, visit)

    def test_builders_fork_and_judges_stay_fresh(self):
        run = self.child({"job_id": "job-prime", "task_id": "t-prime"})
        build = self.spec(run, "build")
        self.assertEqual(build.payload["fork_from"], {"job_id": "job-prime", "task_id": "t-prime"})
        self.assertNotIn("resume_from", build.payload)
        self.assertIn("Build group_00.", build.instruction)  # the full task, not a delta
        review = self.spec(run, "review")
        self.assertNotIn("fork_from", review.payload)

    def test_a_repair_visit_resumes_the_builders_own_session(self):
        own = {"build": {"job_id": "job-b", "task_id": "t-b", "run_id": "flow_x", "visit": 1}}
        build = self.spec(self.child({"job_id": "job-prime", "task_id": "t-prime"}, own), "build", visit=2)
        self.assertEqual(build.payload["resume_from"], {"job_id": "job-b", "task_id": "t-b"})
        self.assertNotIn("fork_from", build.payload)

    def test_no_source_session_means_no_fork(self):
        self.assertNotIn("fork_from", self.spec(self.child(None), "build").payload)

    def test_the_map_hands_its_children_the_source_session(self):
        def execute(node, run, prev):
            if node["kind"] == "agent" and node.get("role") != "explore":
                return NodeOutcome(ok=True, verdict="PASS", output="built", files=run.state["item"]["files"])
            return NodeOutcome(ok=True, verdict="PASS", output="ok")

        def spawn(state_dir, run_id):
            thread = threading.Thread(target=flow.walk, args=(state_dir, run_id), kwargs={"execute": execute})
            thread.start()
            self.addCleanup(thread.join, 10)

        executor = JobNodeExecutor(self.state, spawn=spawn, poll_seconds=0.02)
        route = lambda node, run, prev: executor(node, run, prev) if node["kind"] == "map" else execute(node, run, prev)
        run = flow.new_run(self.state, graph(), "", cwd=str(self.work))
        run.sessions["prime"] = {"job_id": "job-prime", "task_id": "t-prime", "adapter": "codex"}
        flow.save_run(self.state, run)
        run = flow.walk(self.state, run.run_id, execute=route)
        self.assertEqual(run.status, "done", run.reason)
        for item in run.results["groups"]["items"].values():
            child = flow.load_run(self.state, item["run_id"])
            self.assertEqual(child.parent["fork_from"], {"job_id": "job-prime", "task_id": "t-prime"})


if __name__ == "__main__":
    unittest.main()
