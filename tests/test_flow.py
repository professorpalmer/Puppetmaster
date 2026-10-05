"""Flow graph runtime: validation, routing language, walker semantics, persistence."""
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

from puppetmaster import flow
from puppetmaster.flow import FlowError, NodeOutcome


def graph(nodes, edges=(), **extra):
    return {"id": "test-flow", "entry": nodes[0]["id"], "defaults": {"adapter": "codex"},
            "nodes": list(nodes), "edges": list(edges), **extra}


def agent(nid, **extra):
    return {"id": nid, "kind": "agent", "task": f"do {nid}", **extra}


def judge(nid, **extra):
    return {"id": nid, "kind": "judge", "task": f"judge {nid}", **extra}


class ScriptedExecutor:
    """Returns queued outcomes per node id; records every call."""

    def __init__(self, script):
        self.script = {key: list(value) for key, value in script.items()}
        self.calls = []

    def __call__(self, node, run, prev):
        self.calls.append((node["id"], prev))
        queue = self.script.get(node["id"])
        if not queue:
            return NodeOutcome(ok=True, output=f"{node['id']} done")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item


class FlowTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state = Path(self._tmp.name)

    def start(self, g, execute, input_text="goal", answer=None):
        run = flow.new_run(self.state, g, input_text)
        return flow.walk(self.state, run.run_id, execute=execute, answer=answer)


class ValidationTests(FlowTestCase):
    def test_valid_graph_has_no_problems(self):
        g = graph([agent("a"), judge("b"), {"id": "z", "kind": "end"}],
                  [{"from": "a", "to": "b"}, {"from": "b", "to": "z", "when": "PASS"}])
        self.assertEqual(flow.validate_graph(g), [])

    def test_problems_are_all_reported(self):
        g = {"id": "Bad_ID", "entry": "missing",
             "nodes": [{"id": "a", "kind": "agent"}, {"id": "a", "kind": "warp"},
                       {"id": "p", "kind": "parallel", "branches": ["nope"]}],
             "edges": [{"from": "a", "to": "ghost", "when": "sometimes"}, {"from": "a", "to": "a", "max": 0}]}
        problems = "\n".join(flow.validate_graph(g))
        for fragment in ("kebab-case", "duplicate node id", "kind 'warp'", "needs a task",
                         "needs an adapter", "branch 'nope'", "entry 'missing'", "ghost",
                         "'sometimes'", "max must be"):
            self.assertIn(fragment, problems)

    def test_parallel_branch_must_be_agent_or_judge(self):
        g = graph([{"id": "p", "kind": "parallel", "branches": ["s"]},
                   {"id": "s", "kind": "shell", "command": "true"}])
        self.assertIn("must be an agent or judge", "\n".join(flow.validate_graph(g)))

    def test_new_run_refuses_an_invalid_graph(self):
        with self.assertRaises(FlowError):
            flow.new_run(self.state, {"id": "x", "nodes": []})


class EdgeConditionTests(unittest.TestCase):
    def outcome(self, **kw):
        return NodeOutcome(**{"ok": True, **kw})

    def test_outcome_and_verdict_conditions(self):
        self.assertTrue(flow.edge_matches("always", self.outcome(ok=False), {}))
        self.assertTrue(flow.edge_matches("ok", self.outcome(), {}))
        self.assertTrue(flow.edge_matches("fail", self.outcome(ok=False), {}))
        self.assertTrue(flow.edge_matches("FAIL", self.outcome(verdict="FAIL"), {}))
        self.assertFalse(flow.edge_matches("PASS", self.outcome(verdict="PARTIAL"), {}))

    def test_answer_is_normalized(self):
        self.assertTrue(flow.edge_matches("answer=Ship It", self.outcome(answer="  ship it "), {}))
        self.assertTrue(flow.edge_matches("answer=Café", self.outcome(answer="café"), {}))
        self.assertFalse(flow.edge_matches("answer=yes", self.outcome(), {}))

    def test_output_substring(self):
        self.assertTrue(flow.edge_matches("out~=tests pass", self.outcome(output="all tests pass"), {}))

    def test_state_predicates_compare_numbers_numerically(self):
        state = {"coverage": "0.85", "mode": "fast", "count": 9}
        self.assertTrue(flow.edge_matches("state.coverage >= 0.8", self.outcome(), state))
        self.assertTrue(flow.edge_matches("state.count < 10", self.outcome(), state))
        self.assertFalse(flow.edge_matches("state.count > 10", self.outcome(), state))
        self.assertTrue(flow.edge_matches("state.mode = fast", self.outcome(), state))
        self.assertTrue(flow.edge_matches("state.mode != slow", self.outcome(), state))
        self.assertTrue(flow.edge_matches("state.mode ~= as", self.outcome(), state))
        self.assertFalse(flow.edge_matches("state.missing > 1", self.outcome(), state))

    def test_aggregate_verdict_precedence(self):
        self.assertEqual(flow.aggregate_verdict(["PASS", "FAIL", "PARTIAL"], [True] * 3), "FAIL")
        self.assertEqual(flow.aggregate_verdict(["PASS", "PARTIAL"], [True, True]), "PARTIAL")
        self.assertEqual(flow.aggregate_verdict(["PASS", "PASS"], [True, False]), "PARTIAL")
        self.assertEqual(flow.aggregate_verdict(["PASS", None], [True, True]), "PASS")
        self.assertIsNone(flow.aggregate_verdict([None, None], [True, True]))


class WalkTests(FlowTestCase):
    def test_linear_flow_finishes_done_with_templates(self):
        g = graph([agent("a"), agent("b", task="use {{prev}} and {{out.a}} for {{input}}"),
                   {"id": "z", "kind": "end", "summary": "built {{out.b}}"}],
                  [{"from": "a", "to": "b"}, {"from": "b", "to": "z"}])
        seen = {}

        def execute(node, run, prev):
            if node["id"] == "b":
                seen["task"] = flow.render(node["task"], run, prev=prev)
            return NodeOutcome(ok=True, output=f"{node['id']}-out")

        run = self.start(g, execute)
        self.assertEqual(run.status, "done")
        self.assertEqual(seen["task"], "use a-out and a-out for goal")
        self.assertEqual(run.reason, "built b-out")
        self.assertEqual([step["node"] for step in run.steps], ["a", "b", "z"])

    def test_first_matching_edge_in_declaration_order_wins(self):
        g = graph([agent("a"), agent("x"), agent("y")],
                  [{"from": "a", "to": "x", "when": "ok"}, {"from": "a", "to": "y", "when": "always"}])
        run = self.start(g, ScriptedExecutor({}))
        self.assertEqual([step["node"] for step in run.steps], ["a", "x"])

    def test_judge_fail_loops_back_until_pass(self):
        g = graph([agent("build"), judge("review"), {"id": "z", "kind": "end"}],
                  [{"from": "build", "to": "review"},
                   {"from": "review", "to": "z", "when": "PASS"},
                   {"from": "review", "to": "build", "when": "FAIL"}])
        execute = ScriptedExecutor({"review": [NodeOutcome(ok=True, verdict="FAIL", output="fix x"),
                                               NodeOutcome(ok=True, verdict="PASS")]})
        run = self.start(g, execute)
        self.assertEqual(run.status, "done")
        self.assertEqual([c[0] for c in execute.calls], ["build", "review", "build", "review"])
        self.assertEqual(execute.calls[2][1], "fix x")  # the rebuild sees the judge's feedback as {{prev}}

    def test_back_edge_budget_ends_in_stuck_not_a_spin(self):
        g = graph([agent("build"), judge("review")],
                  [{"from": "build", "to": "review"},
                   {"from": "review", "to": "build", "when": "FAIL", "max": 2}])
        execute = ScriptedExecutor({"review": [NodeOutcome(ok=True, verdict="FAIL")]})
        run = self.start(g, execute)
        self.assertEqual(run.status, "stuck")
        self.assertIn("review->build taken 2 times", run.reason)
        self.assertEqual(len([c for c in execute.calls if c[0] == "build"]), 3)

    def test_default_loop_budget_comes_from_limits(self):
        g = graph([agent("a")], [{"from": "a", "to": "a"}], limits={"maxLoops": 4})
        run = self.start(g, ScriptedExecutor({}))
        self.assertEqual(run.status, "stuck")
        self.assertEqual(len(run.steps), 5)

    def test_step_limit(self):
        g = graph([agent("a"), agent("b")],
                  [{"from": "a", "to": "b", "max": 99}, {"from": "b", "to": "a", "max": 99}],
                  limits={"maxSteps": 7})
        run = self.start(g, ScriptedExecutor({}))
        self.assertEqual(run.status, "stuck")
        self.assertEqual(run.reason, "step-limit 7 reached")
        self.assertEqual(len(run.steps), 7)

    def test_judge_without_verdict_is_stuck_not_pass(self):
        g = graph([judge("review"), agent("next")], [{"from": "review", "to": "next", "when": "PASS"}])
        run = self.start(g, ScriptedExecutor({"review": [NodeOutcome(ok=True, verdict=None)]}))
        self.assertEqual(run.status, "stuck")
        self.assertIn("gave no verdict", run.reason)

    def test_failure_without_a_fail_edge_fails_the_run(self):
        g = graph([agent("a"), agent("b")], [{"from": "a", "to": "b", "when": "ok"}])
        run = self.start(g, ScriptedExecutor({"a": [NodeOutcome(ok=False, error="boom")]}))
        self.assertEqual(run.status, "failed")
        self.assertIn("boom", run.reason)

    def test_fail_edge_routes_the_failure(self):
        g = graph([agent("a"), agent("fallback")], [{"from": "a", "to": "fallback", "when": "fail"}])
        run = self.start(g, ScriptedExecutor({"a": [NodeOutcome(ok=False, error="boom")]}))
        self.assertEqual(run.status, "done")
        self.assertEqual(run.steps[-1]["node"], "fallback")

    def test_node_without_edges_ends_the_run(self):
        run = self.start(graph([agent("only")]), ScriptedExecutor({}))
        self.assertEqual(run.status, "done")
        failed = self.start(graph([agent("only")]), ScriptedExecutor({"only": [NodeOutcome(ok=False)]}))
        self.assertEqual(failed.status, "failed")

    def test_unmatched_edges_are_stuck_with_context(self):
        g = graph([judge("r"), agent("x")], [{"from": "r", "to": "x", "when": "PASS"}])
        run = self.start(g, ScriptedExecutor({"r": [NodeOutcome(ok=True, verdict="PARTIAL")]}))
        self.assertEqual(run.status, "stuck")
        self.assertIn("verdict PARTIAL", run.reason)

    def test_end_node_status(self):
        g = graph([agent("a"), {"id": "bad", "kind": "end", "status": "fail", "summary": "nope"}],
                  [{"from": "a", "to": "bad"}])
        run = self.start(g, ScriptedExecutor({}))
        self.assertEqual((run.status, run.reason), ("failed", "nope"))

    def test_retries_stop_at_a_verdict_and_retry_transport_failures(self):
        g = graph([judge("r", retries=3), agent("x")], [{"from": "r", "to": "x", "when": "FAIL"}])
        execute = ScriptedExecutor({"r": [NodeOutcome(ok=False, verdict="FAIL")]})
        self.start(g, execute)
        self.assertEqual([c[0] for c in execute.calls].count("r"), 1)

        g2 = graph([agent("a", retries=2)])
        flaky = ScriptedExecutor({"a": [NodeOutcome(ok=False, error="503"), RuntimeError("spawn"),
                                        NodeOutcome(ok=True)]})
        run = self.start(g2, flaky)
        self.assertEqual(run.status, "done")
        self.assertEqual(len(flaky.calls), 3)

    def test_set_node_and_save_as_feed_state_predicates(self):
        g = graph([agent("measure", saveAs="coverage"),
                   {"id": "flag", "kind": "set", "values": {"checked": "yes"}},
                   agent("ship"), agent("more_tests")],
                  [{"from": "measure", "to": "flag"},
                   {"from": "flag", "to": "ship", "when": "state.coverage >= 0.8"},
                   {"from": "flag", "to": "more_tests", "when": "always"}])

        def execute(node, run, prev):
            if node["kind"] == "set":
                return flow.JobNodeExecutor(self.state)(node, run, prev)
            return NodeOutcome(ok=True, output="0.91" if node["id"] == "measure" else "ok")

        run = self.start(g, execute)
        self.assertEqual(run.state, {"coverage": "0.91", "checked": "yes"})
        self.assertEqual(run.steps[-1]["node"], "ship")

    def test_usage_rolls_up_per_node_and_run(self):
        g = graph([agent("a"), agent("b")], [{"from": "a", "to": "b"}])
        execute = ScriptedExecutor({"a": [NodeOutcome(ok=True, usage={"tokens_in": 10, "tokens_out": 2})],
                                    "b": [NodeOutcome(ok=True, usage={"tokens_in": 5})]})
        run = self.start(g, execute)
        self.assertEqual(run.usage, {"tokens_in": 15, "tokens_out": 2})
        self.assertEqual(run.steps[0]["usage"], {"tokens_in": 10, "tokens_out": 2})


class GateTests(FlowTestCase):
    def test_gate_waits_then_routes_on_the_answer(self):
        g = graph([agent("plan"), {"id": "approve", "kind": "gate", "question": "Ship {{out.plan}}?",
                                   "options": ["yes", "no"]},
                   agent("ship"), {"id": "abort", "kind": "end", "status": "fail"}],
                  [{"from": "plan", "to": "approve"},
                   {"from": "approve", "to": "ship", "when": "answer=yes"},
                   {"from": "approve", "to": "abort", "when": "answer=no"}])
        execute = ScriptedExecutor({"plan": [NodeOutcome(ok=True, output="v2")]})
        run = self.start(g, execute)
        self.assertEqual(run.status, "waiting")
        self.assertEqual(run.gate, {"node": "approve", "question": "Ship v2?", "options": ["yes", "no"]})
        self.assertEqual(flow.walk(self.state, run.run_id, execute=execute).status, "waiting")

        done = flow.walk(self.state, run.run_id, execute=execute, answer="Yes")
        self.assertEqual(done.status, "done")
        self.assertEqual([step["node"] for step in done.steps], ["plan", "approve", "ship"])
        self.assertIsNone(done.gate)


class PersistenceTests(FlowTestCase):
    def test_run_record_round_trips_and_lists(self):
        g = graph([agent("a")])
        run = self.start(g, ScriptedExecutor({}))
        loaded = flow.load_run(self.state, run.run_id)
        self.assertEqual(loaded.to_dict(), run.to_dict())
        self.assertEqual(flow.list_runs(self.state)[0]["run_id"], run.run_id)

    def test_terminal_runs_are_not_walked_again(self):
        run = self.start(graph([agent("a")]), ScriptedExecutor({}))
        execute = ScriptedExecutor({})
        self.assertEqual(flow.walk(self.state, run.run_id, execute=execute).status, "done")
        self.assertEqual(execute.calls, [])

    def test_stop_request_halts_before_the_next_node(self):
        g = graph([agent("a"), agent("b")], [{"from": "a", "to": "b"}])
        run = flow.new_run(self.state, g)

        def execute(node, current, prev):
            flow.request_stop(self.state, run.run_id)
            return NodeOutcome(ok=True)

        stopped = flow.walk(self.state, run.run_id, execute=execute)
        self.assertEqual(stopped.status, "stopped")
        # A stopped node's forced result is not recorded or routed; a restart
        # continues the same visit as a new attempt.
        self.assertEqual(stopped.steps, [])
        self.assertEqual((stopped.current, stopped.inflight["node"]), ("a", "a"))
        resumed = flow.walk(self.state, run.run_id, execute=ScriptedExecutor({}), restart=True)
        self.assertEqual(resumed.status, "done")
        self.assertEqual([step["node"] for step in resumed.steps], ["a", "b"])
        self.assertEqual(resumed.visits, {"a": 1, "b": 1})

    def test_crashed_walker_resumes_at_the_current_node(self):
        g = graph([agent("a"), agent("b"), agent("c")], [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}])
        run = flow.new_run(self.state, g)

        class Crash(BaseException):
            pass

        def crashing(node, current, prev):
            if node["id"] == "b":
                raise Crash()
            return NodeOutcome(ok=True)

        with self.assertRaises(Crash):
            flow.walk(self.state, run.run_id, execute=crashing)
        mid = flow.load_run(self.state, run.run_id)
        self.assertEqual((mid.status, mid.current), ("running", "b"))
        finished = flow.walk(self.state, run.run_id, execute=ScriptedExecutor({}))
        self.assertEqual(finished.status, "done")
        self.assertEqual([step["node"] for step in finished.steps], ["a", "b", "c"])

    def test_saved_graphs_load_by_id_or_path(self):
        g = graph([agent("a")])
        path = flow.save_graph(self.state, g)
        self.assertEqual(flow.load_graph(self.state, "test-flow"), g)
        self.assertEqual(flow.load_graph(self.state, str(path)), g)
        with self.assertRaises(FlowError):
            flow.load_graph(self.state, "no-such-graph")

    def test_run_ids_are_validated(self):
        with self.assertRaises(FlowError):
            flow.load_run(self.state, "../../etc")

    def test_wait_returns_on_wake_states(self):
        run = self.start(graph([agent("a")]), ScriptedExecutor({}))
        self.assertEqual(flow.wait_for_event(self.state, run.run_id, timeout_seconds=1).status, "done")


if __name__ == "__main__":
    unittest.main()
