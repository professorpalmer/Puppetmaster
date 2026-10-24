from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from puppetmaster import flow


def nested_graph(inprocess=True):
    check = f'"{sys.executable}" -c "print(\'VERDICT: PASS - {{{{item}}}} ok\')"'
    region = {"entry": "check", "nodes": [{"id": "check", "kind": "shell", "command": check},
                                         {"id": "done", "kind": "end", "status": "pass"}],
              "edges": [{"from": "check", "to": "done", "when": "ok"}]}
    group = {"entry": "regions", "nodes": [{"id": "regions", "kind": "map", "items": "{{item.regions}}",
                                            "concurrency": 2, "graph": region},
                                           {"id": "done", "kind": "end", "status": "pass"}],
             "edges": [{"from": "regions", "to": "done", "when": "ok"}]}
    return {"id": "inproc", "entry": "groups", "defaults": {"inprocess": inprocess},
            "nodes": [{"id": "groups", "kind": "map", "concurrency": 2, "graph": group,
                       "items": [{"id": "g0", "regions": ["r0", "r1"]}, {"id": "g1", "regions": ["r2", "r3"]}]},
                      {"id": "done", "kind": "end", "status": "pass"}],
            "edges": [{"from": "groups", "to": "done", "when": "ok"}]}


class InprocessWalkTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.state = self.folder / "state"

    def run_graph(self, graph):
        run = flow.new_run(self.state, graph, cwd=str(self.folder))
        return flow.walk(self.state, run.run_id, execute=flow.JobNodeExecutor(self.state, poll_seconds=0.05))

    def test_nested_map_walks_without_a_new_process(self):
        def no_process(*_a, **_k):
            raise AssertionError("an in-process walk started a walker process")
        threads = []
        real = flow.walk_in_thread
        with patch.object(flow, "spawn_background_walk", no_process), \
                patch.object(flow, "walk_in_thread", lambda s, r: threads.append(real(s, r)) or threads[-1]):
            run = self.run_graph(nested_graph())
        self.assertEqual((run.status, run.reason), ("done", "pass"), run.steps)
        self.assertEqual(len(threads), 6)  # 2 groups and 4 regions
        for thread in threads:
            thread.join(5)
            self.assertFalse(thread.is_alive())

    def test_graph_false_wins_over_the_operator(self):
        self.assertFalse(flow.inprocess_requested({"defaults": {"inprocess": False}},
                                                  env={flow.INPROCESS_ENV: "1"}))
        self.assertTrue(flow.inprocess_requested({}, env={flow.INPROCESS_ENV: "1"}))
        self.assertFalse(flow.inprocess_requested({}, env={}))
        graph = nested_graph()
        graph["defaults"]["inprocess"] = "yes"
        self.assertIn("defaults.inprocess must be true or false", flow.validate_graph(graph))

    def test_a_walk_that_raises_is_started_again(self):
        calls = {"n": 0}
        real_walk = flow.walk
        lock = threading.Lock()

        def flaky(state_dir, run_id, **kwargs):
            run = flow.load_run(state_dir, run_id)
            with lock:
                first = run.parent is not None and calls["n"] == 0
                if first:
                    calls["n"] += 1
            if first:
                raise RuntimeError("walker crashed")
            return real_walk(state_dir, run_id, **kwargs)
        with patch.object(flow, "walk", flaky):
            run = flow.new_run(self.state, nested_graph(), cwd=str(self.folder))
            run = real_walk(self.state, run.run_id, execute=flow.JobNodeExecutor(self.state, poll_seconds=0.05))
        self.assertEqual((run.status, run.reason), ("done", "pass"), run.steps)
        self.assertEqual(calls["n"], 1)


if __name__ == "__main__":
    unittest.main()
