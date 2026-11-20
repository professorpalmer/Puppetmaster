"""Flow event log: lineage, approval records, idempotent sync, schema, tap, run compare."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from puppetmaster import flow, flow_compare, flow_events
from puppetmaster.flow import NodeOutcome

sys.path.insert(0, _HERMETIC_DIR)
from test_flow import ScriptedExecutor, agent, graph, judge  # noqa: E402


def check_loop():
    return graph(
        [agent("build"), judge("review"), {"id": "z", "kind": "end"}],
        [{"from": "build", "to": "review"},
         {"from": "review", "to": "build", "when": "FAIL", "max": 2},
         {"from": "review", "to": "z", "when": "PASS"}],
    )


class EventLogTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state = Path(self._tmp.name)
        flow_events._CACHE.clear()

    def walk(self, g, script, answer=None):
        run = flow.new_run(self.state, g, "goal")
        return flow.walk(self.state, run.run_id, execute=ScriptedExecutor(script), answer=answer)

    def test_repair_step_points_at_the_failed_review(self):
        run = self.walk(check_loop(), {"review": [NodeOutcome(ok=True, verdict="FAIL", reason="bad"),
                                                  NodeOutcome(ok=True, verdict="PASS")]})
        self.assertEqual(run.status, "done")
        events = flow_events.read(self.state, run.run_id)
        kinds = [event["type"] for event in events]
        self.assertEqual(kinds, ["run.started", "step", "step", "step", "step", "step", "run.finished"])
        by_id = {event["id"]: event for event in events}
        repair = [event for event in events if event["type"] == "step" and event["node"] == "build"][1]
        self.assertEqual(repair["visit"], 2)
        self.assertEqual(repair["via"], {"from": "review", "when": "FAIL", "edge": 1})
        cause = by_id[repair["parent"]]
        self.assertEqual((cause["node"], cause["verdict"], cause["reason"]), ("review", "FAIL", "bad"))
        self.assertIsNone(events[1]["via"])  # the entry step has no edge
        self.assertEqual([event["seq"] for event in events], list(range(1, 8)))

    def test_every_event_matches_the_published_schema(self):
        run = self.walk(check_loop(), {"review": [NodeOutcome(ok=True, verdict="PASS")]})
        lines = flow_events.events_path(self.state, run.run_id).read_text().splitlines()
        self.assertEqual(flow_events.check_lines(lines), [])

    def test_schema_check_names_the_problem(self):
        bad = {"v": 1, "id": "x.1", "seq": 1, "run_id": "x", "type": "step", "key": "step:0",
               "at": "t", "parent": None}
        problems = flow_events.check(bad)
        self.assertTrue(any("missing 'node'" in problem for problem in problems), problems)
        self.assertTrue(flow_events.check({**bad, "type": "warp"}))

    def test_sync_is_idempotent_and_fills_a_gap(self):
        run = self.walk(check_loop(), {"review": [NodeOutcome(ok=True, verdict="PASS")]})
        path = flow_events.events_path(self.state, run.run_id)
        before = path.read_text()
        flow_events.sync(self.state, run)
        self.assertEqual(path.read_text(), before)
        # A walker that died after save_run but before the append: drop the
        # last two lines and leave a torn tail; the next sync repairs both.
        lines = before.splitlines(keepends=True)
        path.write_text("".join(lines[:-2]) + lines[-2][:10])
        flow_events._CACHE.clear()
        flow_events.sync(self.state, run)
        events = flow_events.read(self.state, run.run_id)
        self.assertEqual([event["key"] for event in events],
                         [json.loads(line)["key"] for line in lines])
        self.assertEqual(len({event["id"] for event in events}), len(events))

    def test_gate_answer_is_recorded_with_who_and_what_they_saw(self):
        g = graph([{"id": "ask", "kind": "gate", "question": "Ship {{input}}?", "options": ["yes", "no"]},
                   {"id": "z", "kind": "end"}],
                  [{"from": "ask", "to": "z", "when": "answer=yes"}])
        g["defaults"] = {}
        run = self.walk(g, {})
        self.assertEqual(run.status, "waiting")
        waiting = flow_events.read(self.state, run.run_id)[-1]
        self.assertEqual(waiting["type"], "gate.waiting")
        self.assertEqual(waiting["question_digest"], flow_events.question_digest("Ship goal?"))
        summary = flow.run_summary(flow.load_run(self.state, run.run_id))
        run = flow.walk(self.state, run.run_id, execute=ScriptedExecutor({}), answer="yes",
                        answered_by="cli:tester")
        self.assertEqual(run.status, "done")
        step = next(event for event in flow_events.read(self.state, run.run_id)
                    if event["type"] == "step" and event["node"] == "ask")
        approval = step["approval"]
        self.assertEqual(approval["answer"], "yes")
        self.assertEqual(approval["answered_by"], "cli:tester")
        self.assertEqual(approval["question_digest"], waiting["question_digest"])
        self.assertEqual(approval["options"], ["yes", "no"])
        self.assertTrue(approval["answered_at"])
        self.assertNotIn("answered_by", json.dumps(summary))

    def test_pilot_summary_does_not_grow(self):
        run = self.walk(check_loop(), {"review": [NodeOutcome(ok=True, verdict="FAIL"),
                                                  NodeOutcome(ok=True, verdict="PASS")]})
        text = json.dumps(flow.run_summary(run))
        for key in ("via", "children", "approval", "arrived"):
            self.assertNotIn(f'"{key}"', text)

    def test_failed_append_never_fails_the_walk(self):
        original = flow_events._append

        def broken(*args, **kwargs):
            raise OSError("disk full")

        flow_events._append = broken
        try:
            with self.assertLogs("puppetmaster.flow_events", level="WARNING"):
                run = self.walk(check_loop(), {"review": [NodeOutcome(ok=True, verdict="PASS")]})
        finally:
            flow_events._append = original
        self.assertEqual(run.status, "done")

    def test_follow_ends_after_the_finish_event(self):
        run = self.walk(check_loop(), {"review": [NodeOutcome(ok=True, verdict="PASS")]})
        events = list(flow_events.follow(self.state, run.run_id, timeout=5, interval=0.01))
        self.assertEqual(events[-1]["type"], "run.finished")
        tail = list(flow_events.follow(self.state, run.run_id, since=events[-2]["seq"], timeout=5))
        self.assertEqual([event["type"] for event in tail], ["run.finished"])

    def test_follow_on_a_restarted_run_waits_for_the_new_finish(self):
        g = check_loop()
        g["edges"][1]["max"] = 1
        run = self.walk(g, {"review": [NodeOutcome(ok=True, verdict="FAIL")]})
        self.assertEqual(run.status, "stuck")
        run, should_walk = flow.prepare_resume(self.state, run.run_id, restart=True, reset_loops=True)
        self.assertTrue(should_walk)
        events = list(flow.flow_events.follow(self.state, run.run_id, timeout=0.3, interval=0.01))
        # The old finish event is out, but the restarted run is not finished:
        # the tap stays open until its timeout.
        self.assertEqual([event["type"] for event in events].count("run.finished"), 1)
        run = flow.walk(self.state, run.run_id,
                        execute=ScriptedExecutor({"review": [NodeOutcome(ok=True, verdict="PASS")]}))
        self.assertEqual(run.status, "done")
        tail = list(flow_events.follow(self.state, run.run_id, since=events[-1]["seq"], timeout=5))
        self.assertEqual(tail[-1]["type"], "run.finished")
        self.assertEqual(tail[-1]["status"], "done")

    def test_child_run_names_its_parent(self):
        parent = flow.new_run(self.state, check_loop(), "goal")
        child = flow.new_run(self.state, check_loop(), "goal",
                             parent={"run_id": parent.run_id, "node": "fan", "key": "a", "depth": 1})
        started = flow_events.read(self.state, child.run_id)[0]
        self.assertEqual(started["parent_run"], {"run_id": parent.run_id, "node": "fan", "key": "a"})
        self.assertEqual(flow_events.check(started), [])


class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state = Path(self._tmp.name)

    def test_events_schema_check_and_compare_commands(self):
        import argparse

        from puppetmaster.cli.commands_flow import flow_action, run_flow_command

        first = flow.new_run(self.state, check_loop(), "goal")
        flow.walk(self.state, first.run_id,
                  execute=ScriptedExecutor({"review": [NodeOutcome(ok=True, verdict="PASS")]}))
        second = flow.new_run(self.state, check_loop(), "goal")
        flow.walk(self.state, second.run_id, execute=ScriptedExecutor(
            {"review": [NodeOutcome(ok=True, verdict="FAIL"), NodeOutcome(ok=True, verdict="PASS")]}))

        out = io.StringIO()
        args = argparse.Namespace(flow_action="events", run_id=first.run_id, since=0, follow=False,
                                  timeout=0.0, children=False)
        with redirect_stdout(out):
            self.assertEqual(run_flow_command(args, self.state), 0)
        self.assertEqual(len(out.getvalue().splitlines()), 5)

        body, code = flow_action(self.state, "check-events", {"source": second.run_id})
        self.assertEqual((code, body["valid"], body["events"]), (0, True, 7))
        body, code = flow_action(self.state, "schema", {})
        self.assertEqual(body["$id"].rsplit("/", 1)[-1], "flow-event.v1.json")

        body, _ = flow_action(self.state, "compare", {"run_ids": [first.run_id, second.run_id]})
        delta = body["runs"][0]["vs_baseline"]
        self.assertTrue(delta["same_graph"])
        self.assertEqual(delta["revisits"], 2)
        self.assertEqual(delta["visits"], {"build": 1, "review": 1})
        self.assertEqual(delta["shape_score"], 0.6)
        with self.assertRaises(ValueError):
            flow_action(self.state, "compare", {"run_ids": [first.run_id]})


class CompareTests(unittest.TestCase):
    def test_edit_distance_and_score(self):
        self.assertEqual(flow_compare.edit_distance(list("abc"), list("abc")), 0)
        self.assertEqual(flow_compare.edit_distance(list("abc"), list("abxbc")), 2)
        self.assertEqual(flow_compare.shape_score([], []), 1.0)
        self.assertEqual(flow_compare.shape_score(["a"], []), 0.0)

    def test_graph_digest_ignores_the_checkout(self):
        g = check_loop()
        self.assertEqual(flow_compare.graph_digest({**g, "cwd": "/a"}),
                         flow_compare.graph_digest({**g, "cwd": "/b"}))
        self.assertNotEqual(flow_compare.graph_digest(g),
                            flow_compare.graph_digest({**g, "entry": "review"}))


if __name__ == "__main__":
    unittest.main()
