"""Release audit of v1.53.0 (hpf, v1.52.1..de4ed0ac): one test for each confirmed bug."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import argparse
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from puppetmaster import flow, flow_events
from puppetmaster.cli import commands_flow
from puppetmaster.flow import NodeOutcome
from test_flow import ScriptedExecutor, agent, graph  # noqa: E402


class StateTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state = Path(self._tmp.name)
        flow_events._CACHE.clear()


class EdgeNumberTests(unittest.TestCase):
    def test_nan_text_compares_as_text(self):
        outcome = NodeOutcome(ok=True)
        self.assertTrue(flow.edge_matches("state.v = nan", outcome, {"v": "nan"}))
        self.assertFalse(flow.edge_matches("state.v != nan", outcome, {"v": "nan"}))

    def test_digit_separator_is_not_a_number(self):
        outcome = NodeOutcome(ok=True)
        self.assertFalse(flow.edge_matches("state.build = 10", outcome, {"build": "1_0"}))
        self.assertTrue(flow.edge_matches("state.build = 10", outcome, {"build": "10.0"}))
        self.assertTrue(flow.edge_matches("state.build > 9", outcome, {"build": " 1e1 "}))


class WalkTests(StateTestCase):
    def test_no_edge_finish_after_failed_work_is_a_failure(self):
        g = graph([agent("a"), agent("b")], [{"from": "a", "to": "b"}])
        run = flow.new_run(self.state, g, "goal")
        run = flow.walk(self.state, run.run_id, execute=ScriptedExecutor(
            {"a": [NodeOutcome(ok=False, error="boom")]}))
        self.assertEqual(run.status, "failed")

    def test_restart_of_an_interrupted_run_applies_its_limits(self):
        g = graph([agent("a"), agent("b")], [{"from": "a", "to": "b"}])
        run = flow.new_run(self.state, g, "goal")
        run.status, run.edge_counts = "interrupted", {"0": 3}
        run.steps = [{"node": "a"}, {"node": "a"}]
        flow.save_run(self.state, run)
        run, should_walk = flow.prepare_resume(self.state, run.run_id, restart=True,
                                               reset_loops=True, extra_steps=5)
        self.assertTrue(should_walk)
        self.assertEqual(run.graph["limits"]["maxSteps"], 7)
        self.assertEqual(run.edge_counts, {})

    def test_explicit_worker_mode_wins_over_the_continued_run(self):
        g = graph([agent("a")])
        prior = flow.new_run(self.state, g, "goal")
        flow.walk(self.state, prior.run_id, execute=ScriptedExecutor({}))
        follow = flow.new_run(self.state, g, "more", continue_from=prior.run_id,
                              worker_mode="subprocess")
        self.assertEqual(follow.worker_mode, "subprocess")
        same = flow.new_run(self.state, g, "more", continue_from=prior.run_id)
        self.assertEqual(same.worker_mode, prior.worker_mode)


class SchemaTests(unittest.TestCase):
    def test_boolean_is_not_the_integer_one(self):
        event = {"v": True, "id": "r.1", "seq": 1, "run_id": "r", "type": "run.started", "key": "run",
                 "at": "t", "parent": None, "graph_id": None, "entry": None, "parent_run": None,
                 "continued_from": None}
        self.assertTrue(flow_events.check(event))
        self.assertEqual(flow_events.check({**event, "v": 1}), [])
        rows = flow_events.check_lines([json.dumps(event)])
        self.assertEqual(len(rows), 1)


class CliTests(StateTestCase):
    def test_children_follow_nested_maps(self):
        g = graph([agent("a")])
        top = flow.new_run(self.state, g)
        mid = flow.new_run(self.state, g, parent={"run_id": top.run_id, "node": "m", "key": "x"})
        leaf = flow.new_run(self.state, g, parent={"run_id": mid.run_id, "node": "m2", "key": "y"})
        for run, child in ((top, mid), (mid, leaf)):
            run.steps = [{"node": "m", "kind": "map", "ok": True, "children": {"k": child.run_id}}]
            flow.save_run(self.state, run)
            flow_events.sync(self.state, run)
        args = argparse.Namespace(flow_action="events", run_id=top.run_id, since=0, follow=False,
                                  timeout=0.0, children=True)
        out = io.StringIO()
        with redirect_stdout(out):
            commands_flow.run_flow_command(args, self.state)
        runs = [json.loads(line)["run_id"] for line in out.getvalue().splitlines()]
        self.assertIn(leaf.run_id, runs)
        self.assertLess(runs.index(mid.run_id), runs.index(leaf.run_id))

    def test_bad_stdin_graph_is_a_usage_error(self):
        args = argparse.Namespace(flow_action="validate", graph="-", backend="sqlite")
        with mock.patch.object(sys, "stdin", io.StringIO("{not json")), redirect_stderr(io.StringIO()):
            self.assertEqual(commands_flow.run_flow_command(args, self.state), commands_flow.EXIT_USAGE)

    def test_missing_graph_is_a_value_error(self):
        for action in ("validate", "save", "run"):
            with self.assertRaises(ValueError):
                commands_flow.flow_action(self.state, action, {})


if __name__ == "__main__":
    unittest.main()
