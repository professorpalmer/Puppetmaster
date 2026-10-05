"""Flow runtime: durability, sessions, executors on real jobs, map fan-out, CLI/MCP."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import contextlib
import io
import json
import subprocess
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from puppetmaster import flow
from puppetmaster.flow import FlowError, JobNodeExecutor, NodeOutcome
from puppetmaster.models import Artifact, ArtifactType, JobStatus, Task, TaskStatus
from puppetmaster.sqlite_store import SQLiteSwarmStore


def graph(nodes, edges=(), **extra):
    return {"id": "rt-flow", "entry": nodes[0]["id"], "defaults": {"adapter": "codex"},
            "nodes": list(nodes), "edges": list(edges), **extra}


def agent(nid, **extra):
    return {"id": nid, "kind": "agent", "task": f"do {nid}", **extra}


def judge(nid, **extra):
    return {"id": nid, "kind": "judge", "task": f"judge {nid}", **extra}


def py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


class Scripted:
    def __init__(self, script=None):
        self.script = {key: list(value) for key, value in (script or {}).items()}
        self.calls = []

    def __call__(self, node, run, prev):
        self.calls.append(node["id"])
        queue = self.script.get(node["id"])
        if not queue:
            return NodeOutcome(ok=True, output=f"{node['id']} done")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, BaseException):
            raise item
        return item


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.state = self.root / "state"
        self.state.mkdir()
        self.work = self.root / "work"
        self.work.mkdir()

    def start(self, g, execute, input_text="goal", **kw):
        run = flow.new_run(self.state, g, input_text, cwd=str(self.work), **kw)
        return flow.walk(self.state, run.run_id, execute=execute)


class WalkerSemanticsTests(Base):
    def test_forward_edge_in_a_loop_is_not_budgeted(self):
        g = graph([agent("build"), judge("review"), {"id": "z", "kind": "end"}],
                  [{"from": "build", "to": "review"},
                   {"from": "review", "to": "z", "when": "PASS"},
                   {"from": "review", "to": "build", "when": "FAIL", "max": 5}])
        fails = [NodeOutcome(ok=True, verdict="FAIL")] * 4 + [NodeOutcome(ok=True, verdict="PASS")]
        run = self.start(g, Scripted({"review": fails}))
        self.assertEqual(run.status, "done", run.reason)
        self.assertEqual(flow.back_edges(g), {2})

    def test_judge_without_verdict_is_stuck_even_with_an_ok_edge(self):
        g = graph([judge("review"), agent("next")], [{"from": "review", "to": "next", "when": "ok"}])
        run = self.start(g, Scripted({"review": [NodeOutcome(ok=True)]}))
        self.assertEqual(run.status, "stuck")
        self.assertIn("gave no verdict", run.reason)

    def test_gate_answer_must_be_an_option_and_persists_for_templates(self):
        seen = {}
        g = graph([{"id": "ask", "kind": "gate", "question": "go?", "options": ["yes", "no"]},
                   agent("act", task="answer was {{answer}}")],
                  [{"from": "ask", "to": "act", "when": "answer=yes"}])

        def execute(node, run, prev):
            seen["task"] = flow.render(node["task"], run, prev)
            return NodeOutcome(ok=True)

        run = self.start(g, execute)
        self.assertEqual(run.status, "waiting")
        typo = flow.walk(self.state, run.run_id, execute=execute, answer="yep")
        self.assertEqual(typo.status, "waiting")
        self.assertIn('"yep" is not one of', typo.reason)
        done = flow.walk(self.state, run.run_id, execute=execute, answer="YES")
        self.assertEqual(done.status, "done")
        self.assertEqual(seen["task"], "answer was YES")

    def test_restart_reopens_a_stuck_run_and_done_runs_stay_done(self):
        g = graph([agent("a"), judge("r"), {"id": "z", "kind": "end"}],
                  [{"from": "a", "to": "r"}, {"from": "r", "to": "z", "when": "PASS"},
                   {"from": "r", "to": "a", "when": "FAIL", "max": 1}])
        execute = Scripted({"r": [NodeOutcome(ok=True, verdict="FAIL"), NodeOutcome(ok=True, verdict="FAIL"),
                                  NodeOutcome(ok=True, verdict="PASS")]})
        run = self.start(g, execute)
        self.assertEqual(run.status, "stuck")
        self.assertEqual(flow.walk(self.state, run.run_id, execute=execute).status, "stuck")
        again = flow.walk(self.state, run.run_id, execute=execute, restart=True, reset_loops=True)
        self.assertEqual(again.status, "done")
        self.assertEqual(flow.walk(self.state, run.run_id, execute=execute, restart=True).status, "done")

    def test_stop_on_an_idle_run_is_immediate_and_restart_clears_it(self):
        g = graph([{"id": "ask", "kind": "gate", "question": "go?"}, agent("act")], [{"from": "ask", "to": "act"}])
        run = self.start(g, Scripted())
        self.assertEqual(flow.request_stop(self.state, run.run_id).status, "stopped")
        restarted = flow.walk(self.state, run.run_id, execute=Scripted(), restart=True)
        self.assertEqual(restarted.status, "waiting")

    def test_cut_fails_the_node_in_flight_and_its_fail_edge_takes_over(self):
        g = graph([agent("slow"), agent("fallback")], [{"from": "slow", "to": "fallback", "when": "fail"}])
        run = flow.new_run(self.state, g, cwd=str(self.work))

        def execute(node, current, prev):
            if node["id"] == "slow":
                flow.cut_node(self.state, run.run_id, "too slow")
            return NodeOutcome(ok=True, verdict="PASS")

        finished = flow.walk(self.state, run.run_id, execute=execute)
        self.assertEqual(finished.status, "done")
        self.assertEqual(finished.steps[0]["error"], "cut off by the user: too slow")
        self.assertEqual(finished.steps[-1]["node"], "fallback")
        self.assertFalse((flow.run_path(self.state, run.run_id).with_name("cut")).exists())

    def test_dead_walker_is_reported_as_interrupted_and_resumes(self):
        run = flow.new_run(self.state, graph([agent("a")]), cwd=str(self.work))
        path = flow.run_path(self.state, run.run_id)
        record = json.loads(path.read_text(encoding="utf-8"))
        record["pid"] = 2 ** 22 + 12345  # no such process
        record["updated_at"] = "2026-01-01T00:00:00+00:00"  # past the start-up grace window
        path.write_text(json.dumps(record), encoding="utf-8")
        self.assertEqual(flow.wait_for_event(self.state, run.run_id, timeout_seconds=1).status, "interrupted")
        self.assertEqual(flow.walk(self.state, run.run_id, execute=Scripted()).status, "done")

    def test_crashed_node_visit_is_adopted_not_repeated(self):
        g = graph([agent("a"), agent("b")], [{"from": "a", "to": "b"}])
        run = flow.new_run(self.state, g, cwd=str(self.work))
        seen = []

        class Crash(BaseException):
            pass

        def crashing(node, current, prev):
            seen.append((node["id"], current.inflight["visit"], list(current.inflight["job_ids"])))
            if node["id"] == "a" and not current.inflight["job_ids"]:
                current.inflight["job_ids"].append("job_started_before_crash")
                flow.save_run(self.state, current)
                raise Crash()
            return NodeOutcome(ok=True, job_ids=list(current.inflight["job_ids"]))

        with self.assertRaises(Crash):
            flow.walk(self.state, run.run_id, execute=crashing)
        flow.walk(self.state, run.run_id, execute=crashing)
        self.assertEqual(seen[1], ("a", 1, ["job_started_before_crash"]))
        finished = flow.load_run(self.state, run.run_id)
        self.assertEqual(finished.visits, {"a": 1, "b": 1})
        self.assertEqual(finished.steps[0]["job_ids"], ["job_started_before_crash"])

    def test_retry_attempts_are_counted_in_the_inflight_record(self):
        g = graph([agent("a", retries=2)])
        attempts = []

        def flaky(node, current, prev):
            attempts.append(current.inflight["attempt"])
            return NodeOutcome(ok=len(attempts) == 3, error="503")

        run = self.start(g, flaky)
        self.assertEqual(run.status, "done")
        self.assertEqual(attempts, [0, 1, 2])

    def test_continue_from_carries_sessions_results_and_state(self):
        first = self.start(graph([agent("a", saveAs="note")]), Scripted({"a": [NodeOutcome(ok=True, output="v1")]}))
        record = flow.load_run(self.state, first.run_id)
        record.sessions["a"] = {"job_id": "job_1", "task_id": "task_1", "adapter": "codex"}
        flow.save_run(self.state, record)
        follow = flow.new_run(self.state, graph([agent("a")]), "now add x", continue_from=first.run_id)
        self.assertEqual(follow.sessions["a"]["task_id"], "task_1")
        self.assertEqual(follow.state["note"], "v1")
        self.assertEqual(follow.continued_from, first.run_id)

    def test_summary_is_a_delta_with_the_last_problem(self):
        g = graph([agent("a"), judge("r")], [{"from": "a", "to": "r"}])
        run = self.start(g, Scripted({"r": [NodeOutcome(ok=True, verdict="FAIL", reason="x.py:3 broken")]}))
        summary = flow.run_summary(run, since=1)
        self.assertEqual([step["node"] for step in summary["steps"]], ["r"])
        self.assertEqual(summary["last_problem"]["reason"], "x.py:3 broken")
        self.assertEqual(summary["next_since"], 2)

    def test_relative_graph_cwd_resolves_against_the_callers_workspace(self):
        (self.work / "sub").mkdir()
        dot = flow.new_run(self.state, graph([agent("a")], cwd="."), cwd=str(self.work))
        self.assertEqual(dot.graph["cwd"], str(self.work.resolve()))
        sub = flow.new_run(self.state, graph([agent("a")], cwd="sub"), cwd=str(self.work))
        self.assertEqual(sub.graph["cwd"], str((self.work / "sub").resolve()))
        absolute = flow.new_run(self.state, graph([agent("a")], cwd=str(self.root)), cwd=str(self.work))
        self.assertEqual(absolute.graph["cwd"], str(self.root.resolve()))

    def test_continue_from_a_live_run_is_refused(self):
        live = flow.new_run(self.state, graph([agent("a")]), cwd=str(self.work))
        with self.assertRaises(FlowError):
            flow.new_run(self.state, graph([agent("a")]), continue_from=live.run_id)

    def test_a_cut_aimed_at_an_earlier_visit_is_ignored(self):
        run = flow.new_run(self.state, graph([agent("a"), agent("b")], [{"from": "a", "to": "b"}]),
                           cwd=str(self.work))
        path = flow.run_path(self.state, run.run_id).with_name("cut")
        path.write_text(json.dumps({"node": "a", "visit": 1, "reason": "late"}), encoding="utf-8")

        def execute(node, current, prev):
            return NodeOutcome(ok=True)

        finished = flow.walk(self.state, run.run_id, execute=execute)
        self.assertEqual(finished.steps[0]["error"], "cut off by the user: late")
        # a stale marker for a visit that is no longer in flight never fails "b"
        path.write_text(json.dumps({"node": "a", "visit": 1, "reason": "stale"}), encoding="utf-8")
        g = graph([agent("a"), agent("b")], [{"from": "a", "to": "b"}])
        second = flow.new_run(self.state, g, cwd=str(self.work))
        flow.run_path(self.state, second.run_id).with_name("cut").write_text(
            json.dumps({"node": "b", "visit": 7, "reason": "stale"}), encoding="utf-8")
        clean = flow.walk(self.state, second.run_id, execute=execute)
        self.assertEqual(clean.status, "done")
        self.assertTrue(all(step.get("error") is None for step in clean.steps))

    def test_stop_halts_retries(self):
        g = graph([agent("a", retries=5)])
        run = flow.new_run(self.state, g, cwd=str(self.work))
        calls = []

        def failing(node, current, prev):
            calls.append(1)
            flow.request_stop(self.state, run.run_id)
            return NodeOutcome(ok=False, error="503")

        self.assertEqual(flow.walk(self.state, run.run_id, execute=failing).status, "stopped")
        self.assertEqual(len(calls), 1)


class HonestEndTests(Base):
    def test_failed_node_carried_to_an_implicit_end_fails_the_run(self):
        g = graph([agent("build"), {"id": "done", "kind": "end"}], [{"from": "build", "to": "done"}])
        run = self.start(g, Scripted({"build": [NodeOutcome(ok=False, error="task failed")]}))
        self.assertEqual(run.status, "failed")
        self.assertIn("build: task failed", run.reason)

    def test_explicit_end_status_still_wins(self):
        g = graph([agent("build"), {"id": "done", "kind": "end", "status": "pass"}],
                  [{"from": "build", "to": "done"}])
        run = self.start(g, Scripted({"build": [NodeOutcome(ok=False, error="task failed")]}))
        self.assertEqual(run.status, "done")

    def test_repaired_failure_does_not_fail_the_end(self):
        g = graph([agent("build"), {"id": "check", "kind": "shell", "command": "x"},
                   {"id": "done", "kind": "end"}],
                  [{"from": "build", "to": "check"}, {"from": "check", "to": "build", "when": "fail", "max": 2},
                   {"from": "check", "to": "done", "when": "ok"}])
        run = self.start(g, Scripted({"check": [NodeOutcome(ok=False, error="red"), NodeOutcome(ok=True)]}))
        self.assertEqual(run.status, "done", run.reason)

    def test_mcp_flow_example_graph_validates(self):
        from puppetmaster import mcp_server
        text = mcp_server.flow_schema()["properties"]["graph"]["description"]
        example = json.loads(text[text.index('{"id": "mods"'):text.index(". Full reference")])
        self.assertEqual(flow.validate_graph(example), [])


class TemplateAndSafetyTests(Base):
    def run_with(self, **state):
        return flow.FlowRun(run_id="flow_000000000000", graph={}, state=state, input="in")

    def test_render_value_keeps_structured_types(self):
        run = self.run_with(item={"files": ["a.py", "b.py"], "n": 3})
        self.assertEqual(flow.render_value("{{item.files}}", run), ["a.py", "b.py"])
        self.assertEqual(flow.render_value("{{item.n}}", run), 3)
        self.assertEqual(flow.render_value("n={{item.n}}", run), "n=3")
        self.assertEqual(flow.render("{{nope}} {{input}}", run), "{{nope}} in")

    def test_shell_rendering_refuses_unsafe_values(self):
        run = self.run_with(name="region_003", evil="x; rm -rf ~")
        self.assertEqual(flow.render("check {{state.name}}", run, shell=True), "check region_003")
        with self.assertRaises(FlowError):
            flow.render("check {{state.evil}}", run, shell=True)
        with self.assertRaises(FlowError):
            flow.render("echo {{prev}}", run, "$(whoami)", shell=True)

    def test_item_feedback_keeps_only_the_items_lines(self):
        feedback = "shapes/circle.py:1 - missing __all__\nshapes/square.py:4 - wrong message\nsquare: rounding"
        self.assertEqual(flow.item_feedback(feedback, "square", ["shapes/square.py"]),
                         "shapes/square.py:4 - wrong message\nsquare: rounding")
        self.assertEqual(flow.item_feedback("general note", "square", ["shapes/square.py"]), "general note")

    def test_unowned_paths_are_reported(self):
        feedback = ("shapes/circle.py:1 - missing __all__\nshapes/__init__.py:1 - defines nothing\n"
                    "see README.md and shapes/square.py:4")
        self.assertEqual(flow.unowned_paths(feedback, {"shapes/circle.py", "shapes/square.py"}),
                         ["shapes/__init__.py"])

    def test_item_keys_and_target_selection(self):
        self.assertEqual(flow.item_key({}, {"id": "r 1/x"}, 0), "r-1-x")
        self.assertEqual(flow.item_key({}, "alpha", 3), "alpha")
        self.assertEqual(flow.item_key({"key": "{{item.n}}-{{index}}"}, {"n": 7}, 2), "7-2")
        prior = {"a": {"ok": True, "verdict": "PASS", "files": ["src/a.py"]},
                 "b": {"ok": False, "verdict": "FAIL"}, "c": {"ok": True, "verdict": "PASS"}}
        # A repair reruns what the feedback names plus what failed.
        self.assertEqual(flow.select_targets(["a", "b", "c"], prior, "src/a.py:3 is wrong"), {"a", "b"})
        self.assertEqual(flow.select_targets(["a", "b", "c"], prior, "item c needs work"), {"b"})
        # A follow-up reruns what the request names, else everything.
        self.assertEqual(flow.select_targets(["a", "b", "c"], prior, "src/a.py too", repair=False), {"a"})
        self.assertEqual(flow.select_targets(["a", "b", "c"], prior, "add logging", repair=False),
                         {"a", "b", "c"})
        # Digit keys are never matched by name: they collide with line numbers.
        numbered = {"12": {"ok": True, "verdict": "PASS"}, "13": {"ok": False}}
        self.assertEqual(flow.select_targets(["12", "13"], numbered, "x.py:12 is wrong"), {"13"})
        self.assertEqual(flow.select_targets(["r1", "r2"], {"r1": {"ok": True}, "r2": {"ok": True}},
                                             "r2 roads are broken"), {"r2"})
        self.assertEqual(flow.select_targets(["a", "b", "c"], prior, "general feedback"), {"b"})
        healthy = {key: {"ok": True, "verdict": "PASS"} for key in "abc"}
        self.assertEqual(flow.select_targets(["a", "b", "c"], healthy, "general"), {"a", "b", "c"})


class ValidationV2Tests(unittest.TestCase):
    def test_map_rules(self):
        base = {"id": "m-flow", "entry": "m", "defaults": {"adapter": "codex"}}
        ok = {**base, "nodes": [{"id": "m", "kind": "map", "items": "{{state.items}}", "concurrency": 4,
                                 "node": {"kind": "agent", "task": "build {{item}}", "files": "{{item.files}}"}}]}
        self.assertEqual(flow.validate_graph(ok), [])
        no_files = {**base, "nodes": [{"id": "m", "kind": "map", "items": [], "node": {"kind": "agent", "task": "t"}}]}
        self.assertIn("needs files", "\n".join(flow.validate_graph(no_files)))
        gate = {**base, "nodes": [{"id": "m", "kind": "map", "items": [],
                                   "graph": {"entry": "g", "nodes": [{"id": "g", "kind": "gate", "question": "?"}]}}]}
        self.assertIn("cannot run inside a map", "\n".join(flow.validate_graph(gate)))
        both = {**base, "nodes": [{"id": "m", "kind": "map", "items": [], "node": {}, "graph": {}}]}
        self.assertIn("exactly one of node or graph", "\n".join(flow.validate_graph(both)))
        bad_pass = {**base, "nodes": [{"id": "m", "kind": "map", "items": [], "pass": 2,
                                       "node": {"kind": "judge", "task": "t"}}]}
        self.assertIn("pass must be", "\n".join(flow.validate_graph(bad_pass)))

    def test_map_depth_is_bounded(self):
        inner = {"kind": "judge", "task": "t"}
        for _ in range(flow.MAX_MAP_DEPTH + 1):
            inner = {"kind": "map", "items": [], "node": inner}
        g = {"id": "deep", "entry": "item", "defaults": {"adapter": "codex"}, "nodes": [{**inner, "id": "item"}]}
        self.assertIn("nests deeper", "\n".join(flow.validate_graph(g)))

    def test_parallel_code_branches_need_disjoint_files(self):
        g = {"id": "p-flow", "entry": "p", "defaults": {"adapter": "codex"},
             "nodes": [{"id": "p", "kind": "parallel", "branches": ["a", "b", "c"]},
                       {"id": "a", "kind": "agent", "task": "t", "files": ["x.py"]},
                       {"id": "b", "kind": "agent", "task": "t", "files": ["x.py"]},
                       {"id": "c", "kind": "agent", "task": "t"}]}
        problems = "\n".join(flow.validate_graph(g))
        self.assertIn("both own 'x.py'", problems)
        self.assertIn("'c' needs files", problems)

    def test_retries_and_types_are_checked_not_crashed_on(self):
        g = {"id": "t-flow", "entry": "a", "defaults": [], "limits": 5,
             "nodes": [{"id": "a", "kind": "agent", "task": "t", "adapter": "codex", "retries": "two"}]}
        problems = "\n".join(flow.validate_graph(g))
        self.assertIn("defaults must be an object", problems)
        self.assertIn("limits must be an object", problems)
        self.assertIn("retries must be", problems)


class SpecTests(Base):
    def make_run(self, g, **kw):
        return flow.new_run(self.state, g, "the goal", cwd=str(self.work), **kw)

    def spec(self, run, key, visit=1, prev=""):
        node = next(item for item in run.graph["nodes"] if item["id"] == key)
        return JobNodeExecutor(self.state)._spec(key, node, run, prev, visit)

    def test_judge_is_read_only_with_a_verdict_contract(self):
        run = self.make_run(graph([judge("review", task="Review {{input}}")]))
        spec = self.spec(run, "review")
        self.assertTrue(spec.payload["read_only"])
        self.assertEqual(spec.payload["sandbox"], "read-only")
        self.assertEqual(spec.payload["permission_mode"], "plan")
        self.assertTrue(spec.payload["terminal_verdict"])
        self.assertIn("Review the goal", spec.instruction)
        self.assertIn("VERDICT: FAIL - <what is wrong, file:line>", spec.instruction)
        self.assertEqual(spec.role, "review")
        self.assertFalse(spec.payload["ephemeral"])

    def test_builder_gets_write_scope_from_templated_files(self):
        g = graph([agent("build", files="{{state.files}}", model="gpt-5.6-luna")], state={"files": ["src/a.py"]})
        spec = self.spec(self.make_run(g), "build")
        self.assertEqual(spec.payload["sandbox"], "workspace-write")
        self.assertEqual(spec.payload["write_scope"], ["src/a.py"])
        self.assertEqual(spec.payload["model"], "gpt-5.6-luna")
        self.assertEqual(spec.payload["cwd"], str(self.work.resolve()))
        self.assertIn("VERDICT: PASS - <what you built and checked>", spec.instruction)

    def test_revisit_resumes_the_session_with_only_the_feedback(self):
        run = self.make_run(graph([agent("build"), judge("review")], [{"from": "build", "to": "review"}]))
        run.sessions = {"build": {"job_id": "job_1", "task_id": "task_1"},
                        "review": {"job_id": "job_2", "task_id": "task_2"}}
        spec = self.spec(run, "build", visit=2, prev="x.py:3 off by one")
        self.assertEqual(spec.payload["resume_from"], {"job_id": "job_1", "task_id": "task_1"})
        delta = spec.payload["resume_prompt"]
        self.assertIn("Revision 2", delta)
        self.assertIn("x.py:3 off by one", delta)
        self.assertNotIn("do build", delta)
        review = self.spec(run, "review", visit=2)
        self.assertIn("confirm the problems you flagged are fixed", review.payload["resume_prompt"])
        self.assertIn("judge review", review.payload["resume_prompt"])

    def test_continued_run_sends_the_follow_up_and_revise_overrides(self):
        g = graph([agent("build"), agent("polish", revise="Polish: {{input}}")])
        run = self.make_run(g)
        run.sessions = {"build": {"job_id": "j", "task_id": "t"}, "polish": {"job_id": "j", "task_id": "u"}}
        run.input = "rename foo to bar"
        self.assertIn("Follow-up request:\nrename foo to bar", self.spec(run, "build").payload["resume_prompt"])
        self.assertTrue(self.spec(run, "polish").payload["resume_prompt"].startswith("Polish: rename foo to bar"))

    def test_resume_false_opts_out(self):
        run = self.make_run(graph([agent("build", resume=False)]))
        run.sessions = {"build": {"job_id": "j", "task_id": "t"}}
        self.assertNotIn("resume_from", self.spec(run, "build", visit=2).payload)


class TaskOutcomeTests(Base):
    def test_latest_attempt_verdict_report_files_and_usage(self):
        store = SQLiteSwarmStore(self.state)
        store.init()
        job = store.create_job("g")
        task = Task(job_id=job.id, role="build", instruction="i", status=TaskStatus.COMPLETE, adapter="codex")
        store.save_tasks([task])

        def art(kind, payload, at, evidence=("e",)):
            return Artifact(job_id=job.id, task_id=task.id, type=kind, created_by="w", confidence=0.9,
                            evidence=list(evidence), payload=payload, created_at=at)

        store.save_artifacts([
            art(ArtifactType.VERIFICATION, {"check": "c", "result": "failed", "adapter": "cursor",
                                            "tokens_in": 5, "tokens_out": 1}, "2026-10-05T10:00:00+00:00"),
            art(ArtifactType.FINDING, {"claim": "old attempt report"}, "2026-10-05T10:00:00+00:00"),
            art(ArtifactType.VERIFICATION, {"check": "c", "kind": "worker_verdict", "verdict": "FAIL",
                                            "reason": "old", "result": "failed"}, "2026-10-05T10:00:00+00:00"),
            art(ArtifactType.VERIFICATION, {"check": "c", "result": "passed", "adapter": "codex",
                                            "tokens_in": 900, "tokens_out": 40, "real_cost_usd": 0.01,
                                            "last_message": "built\nVERDICT: PASS - all checks pass"},
                "2026-10-05T10:05:00+00:00"),
            art(ArtifactType.FINDING, {"claim": "Built a.py", "report": "Built a.py; checks pass"},
                "2026-10-05T10:05:00+00:00"),
            art(ArtifactType.PATCH, {"change": "c", "files": ["a.py"]}, "2026-10-05T10:05:00+00:00"),
        ])
        outcome = flow.task_outcome(store, job.id, "build")
        self.assertTrue(outcome.ok)
        self.assertEqual((outcome.verdict, outcome.reason), ("PASS", "all checks pass"))
        self.assertEqual(outcome.output, "Built a.py; checks pass")
        self.assertEqual(outcome.files, ["a.py"])
        self.assertEqual(outcome.usage["tokens_in"], 900)
        self.assertEqual(outcome.usage["cost_usd"], 0.01)
        self.assertEqual(outcome.task_id, task.id)


class JobIntegrationTests(Base):
    """Real Orchestrator jobs with the shell adapter: no model spend."""

    def shell_node(self, nid, code, kind="agent", **extra):
        return {"id": nid, "kind": kind, "adapter": "shell", "task": f"{nid} task",
                "payload": {"command": py(code), "timeout_seconds": 30}, **extra}

    def executor(self):
        return JobNodeExecutor(self.state, worker_mode="inline", poll_seconds=0.05)

    def test_build_review_repair_loop_on_real_jobs(self):
        counter = self.work / "reviews"
        review = ("import pathlib;p=pathlib.Path(%r);n=int(p.read_text()) if p.exists() else 0;"
                  "p.write_text(str(n+1));print('VERDICT: FAIL - x.py:1 wrong' if n==0 else 'VERDICT: PASS - fixed')"
                  % str(counter))
        g = graph([self.shell_node("build", "print('built\\nVERDICT: PASS - ok')", files=["x.py"]),
                   self.shell_node("review", review, kind="judge"),
                   {"id": "done", "kind": "end"}],
                  [{"from": "build", "to": "review"},
                   {"from": "review", "to": "done", "when": "PASS"},
                   {"from": "review", "to": "build", "when": "FAIL"}])
        run = self.start(g, self.executor())
        self.assertEqual(run.status, "done", run.reason)
        self.assertEqual([step["node"] for step in run.steps], ["build", "review", "build", "review", "done"])
        self.assertEqual([step.get("verdict") for step in run.steps[:4]], ["PASS", "FAIL", "PASS", "PASS"])
        jobs = [step["job_ids"][0] for step in run.steps[:4]]
        self.assertEqual(len(set(jobs)), 4)  # one durable job per node visit
        store = SQLiteSwarmStore(self.state)
        keys = {store.get_job(job_id).launch_key for job_id in jobs}
        self.assertEqual(len(keys), 4)
        self.assertIn("build", run.sessions)
        # The second build visit asked to resume the first (shell has no sessions,
        # so the worker ran fresh and recorded why).
        second = [task for task in store.list_tasks(jobs[2])][0]
        self.assertEqual(second.payload["resume_from"]["task_id"], run.steps[0] and
                         store.list_tasks(jobs[0])[0].id)

    def test_relaunching_a_visit_adopts_the_existing_job(self):
        from puppetmaster.orchestrator import Orchestrator

        real_run = Orchestrator.run

        def create_only(self_, goal, specs=None, on_job_created=None, launch_key=None, **kw):
            job, created = self_.store.create_or_get_job(goal, launch_key=launch_key,
                                                         launch_fingerprint="fp")
            if created:
                self_.store.save_tasks([Task(job_id=job.id, role=spec.role, instruction=spec.instruction,
                                             adapter=spec.adapter, payload=spec.payload) for spec in specs])
            on_job_created(job)

        g = graph([self.shell_node("build", "print('VERDICT: PASS - ok')")])
        with patch.object(Orchestrator, "run", create_only):
            run = self.start(g, self.executor())
        self.assertEqual(run.status, "done", run.reason)
        store = SQLiteSwarmStore(self.state)
        job = store.get_job(run.steps[0]["job_ids"][0])
        self.assertEqual(job.status, JobStatus.COMPLETE)
        events = [event["event"] for event in store.read_events(job.id)]
        self.assertIn("job.adopted", events)
        self.assertIsNotNone(real_run)

    def test_parallel_reruns_only_the_branch_the_feedback_names(self):
        flaky = self.work / "flaky"
        b_code = ("import pathlib;p=pathlib.Path(%r);n=int(p.read_text()) if p.exists() else 0;"
                  "p.write_text(str(n+1));print('VERDICT: PASS - b ok' if n else 'VERDICT: FAIL - b broken')"
                  % str(flaky))
        g = graph([{"id": "fan", "kind": "parallel", "branches": ["a", "b"]},
                   self.shell_node("a", "print('VERDICT: PASS - a ok')", files=["a.py"]),
                   self.shell_node("b", b_code, files=["b.py"]),
                   {"id": "done", "kind": "end"}],
                  [{"from": "fan", "to": "done", "when": "PASS"},
                   {"from": "fan", "to": "fan", "when": "FAIL"}])
        run = self.start(g, self.executor())
        self.assertEqual(run.status, "done", run.reason)
        first, second = run.steps[0], run.steps[1]
        self.assertEqual(first["verdict"], "FAIL")
        self.assertIn("b: b broken", first["reason"])
        self.assertEqual(second["verdict"], "PASS")
        store = SQLiteSwarmStore(self.state)
        rerun_roles = [task.role for job_id in second["job_ids"] for task in store.list_tasks(job_id)]
        self.assertEqual(rerun_roles, ["b"])

    def test_shell_node_context_file_timeout_and_verdict(self):
        g = graph([{"id": "check", "kind": "shell",
                    "command": f'"{sys.executable}" -c "import json,os;'
                               f'c=json.load(open(os.environ[\'PM_FLOW_CONTEXT\']));'
                               f'print(c[\'input\']);print(\'VERDICT: PASS - ctx ok\')"'},
                   {"id": "slow", "kind": "shell", "timeoutMs": 300,
                    "command": f'"{sys.executable}" -c "import time;time.sleep(30)"'}],
                  [{"from": "check", "to": "slow", "when": "PASS"}])
        started = time.monotonic()
        run = self.start(g, self.executor(), input_text="hello flow")
        self.assertLess(time.monotonic() - started, 20)
        self.assertIn("hello flow", run.outputs["check"])
        self.assertEqual(run.steps[0]["verdict"], "PASS")
        self.assertEqual(run.status, "failed")
        self.assertIn("timed out", run.steps[1]["error"])

    def test_unsafe_shell_substitution_fails_the_node(self):
        g = graph([{"id": "a", "kind": "set", "values": {"x": "a; rm -rf /"}},
                   {"id": "check", "kind": "shell", "command": "echo {{state.x}}"}],
                  [{"from": "a", "to": "check"}])
        run = self.start(g, self.executor())
        self.assertEqual(run.status, "failed")
        self.assertIn("not safe in a shell command", run.steps[-1]["error"])


class MapTests(Base):
    def setUp(self):
        super().setUp()
        self.live = 0
        self.peak = 0
        self.lock = threading.Lock()
        self.child_calls = []

    def item_executor(self, fail_once=()):
        failed = set()

        def execute(node, run, prev):
            key = run.state["key"]
            with self.lock:
                self.live += 1
                self.peak = max(self.peak, self.live)
                self.child_calls.append((key, run.input, run.continued_from))
            time.sleep(0.05)
            with self.lock:
                self.live -= 1
            if key in fail_once and key not in failed:
                failed.add(key)
                return NodeOutcome(ok=True, verdict="FAIL", reason=f"{key} broken",
                                   files=[f"regions/{key}.py"])
            return NodeOutcome(ok=True, verdict="PASS", output=f"{key} built", files=[f"regions/{key}.py"])

        return execute

    def map_executor(self, item_execute):
        def spawn(state_dir, run_id):
            thread = threading.Thread(target=flow.walk, args=(state_dir, run_id), kwargs={"execute": item_execute})
            thread.start()
            self.addCleanup(thread.join, 10)

        return JobNodeExecutor(self.state, spawn=spawn, poll_seconds=0.02)

    def map_graph(self, **map_extra):
        return graph([{"id": "regions", "kind": "map", "items": "{{state.regions}}", "concurrency": 2,
                       "node": {"kind": "judge", "task": "check {{item.id}}"}, **map_extra},
                      {"id": "done", "kind": "end"}],
                     [{"from": "regions", "to": "done", "when": "PASS"},
                      {"from": "regions", "to": "regions", "when": "FAIL"}],
                     state={"regions": [{"id": "r1"}, {"id": "r2"}, {"id": "r3"}, {"id": "r4"}]})

    def test_map_fans_out_bounded_and_repairs_only_failing_items(self):
        run = self.start(self.map_graph(), self.map_executor(self.item_executor(fail_once={"r3"})))
        self.assertEqual(run.status, "done", run.reason)
        self.assertLessEqual(self.peak, 2)
        first, second = run.steps[0], run.steps[1]
        self.assertEqual(first["verdict"], "FAIL")
        self.assertEqual(first["items"], {"done": 4})
        self.assertIn("r3", run.results["regions"]["items"])
        reruns = [call for call in self.child_calls[4:]]
        self.assertEqual([call[0] for call in reruns], ["r3"])
        first_r3 = flow.load_run(self.state, flow.load_run(self.state, run.run_id).results["regions"]["items"]["r3"]["run_id"])
        self.assertIsNotNone(first_r3.continued_from)  # the repair continued r3's own child flow
        self.assertEqual(second["verdict"], "PASS")
        self.assertEqual(run.outputs["regions"].splitlines()[0], "map regions: 4/4 items passed")

    def test_items_learn_their_peers_scopes_and_the_reason_names_failures(self):
        g = graph([{"id": "m", "kind": "map", "items": ["a1", "b2"],
                    "node": {"kind": "agent", "task": "build {{item}}", "files": ["src/{{item}}.py"]}}])
        captured = {}

        def item_execute(node, run, prev):
            spec = JobNodeExecutor(self.state)._spec(node["id"], node, run, prev, 1)
            captured[run.state["key"]] = (spec.payload["write_scope"], spec.payload["peer_write_scopes"])
            if run.state["key"] == "b2":
                return NodeOutcome(ok=False, error="broke")
            return NodeOutcome(ok=True, verdict="PASS")

        run = self.start(g, self.map_executor(item_execute))
        self.assertEqual(captured["a1"], (["src/a1.py"], ["src/b2.py"]))
        self.assertEqual(captured["b2"], (["src/b2.py"], ["src/a1.py"]))
        self.assertIn("b2 (", run.reason)

    def test_an_item_whose_node_failed_does_not_pass_even_if_its_flow_ended(self):
        g = graph([{"id": "m", "kind": "map", "items": ["a1", "b2"],
                    "graph": {"entry": "build",
                              "nodes": [{"id": "build", "kind": "agent", "task": "t", "files": ["src/{{item}}.py"]},
                                        {"id": "ok", "kind": "end"}],
                              "edges": [{"from": "build", "to": "ok"}]}}])

        def item_execute(node, run, prev):
            if run.state["key"] == "b2":
                return NodeOutcome(ok=False, error="task failed: gate write_scope")
            return NodeOutcome(ok=True, verdict="PASS")

        run = self.start(g, self.map_executor(item_execute))
        self.assertEqual(run.status, "failed")
        items = run.results["m"]["items"]
        self.assertTrue(items["a1"]["ok"])
        self.assertFalse(items["b2"]["ok"])
        self.assertEqual(items["b2"]["status"], "failed")
        self.assertIn("build: task failed: gate write_scope", run.reason)

    def test_pass_fraction_policy(self):
        g = self.map_graph(**{"pass": 0.75})
        run = self.start(g, self.map_executor(self.item_executor(fail_once={"r2"})))
        self.assertEqual(run.status, "done")
        self.assertEqual(len(run.steps), 2)  # 3/4 passed meets 0.75 on the first visit
        self.assertEqual(run.steps[0]["verdict"], "PASS")

    def test_map_item_flow_children_are_hidden_from_list(self):
        self.start(self.map_graph(), self.map_executor(self.item_executor()))
        self.assertEqual(len(flow.list_runs(self.state)), 1)
        self.assertEqual(len(flow.list_runs(self.state, include_children=True)), 5)

    def test_map_feedback_template_targets_from_the_named_node(self):
        g = graph([{"id": "regions", "kind": "map", "items": "{{state.regions}}", "concurrency": 2,
                    "feedback": "{{out.consistency}}", "node": {"kind": "judge", "task": "check {{item.id}}"}},
                   judge("consistency"), agent("integrate"), {"id": "done", "kind": "end"}],
                  [{"from": "regions", "to": "consistency", "when": "PASS"},
                   {"from": "consistency", "to": "done", "when": "PASS"},
                   {"from": "consistency", "to": "integrate", "when": "FAIL"},
                   {"from": "integrate", "to": "regions"}],
                  state={"regions": [{"id": "r1"}, {"id": "r2"}, {"id": "r3"}]})
        verdicts = iter([NodeOutcome(ok=True, verdict="FAIL", output="r2 roads broken\nr9 nothing"),
                         NodeOutcome(ok=True, verdict="PASS")])
        item_exec = self.item_executor()

        class Parent:
            def __init__(inner, base):
                inner.base = base

            def __call__(inner, node, run, prev):
                if node["id"] == "consistency":
                    return next(verdicts)
                if node["id"] == "integrate":
                    return NodeOutcome(ok=True, output="integrated shared files")
                return inner.base(node, run, prev)

        run = self.start(g, Parent(self.map_executor(item_exec)))
        self.assertEqual(run.status, "done", run.reason)
        repaired = [call for call in self.child_calls[3:]]
        self.assertEqual([call[0] for call in repaired], ["r2"])
        self.assertEqual(repaired[0][1], "r2 roads broken")

    def test_follow_up_run_continues_only_named_items(self):
        first = self.start(self.map_graph(), self.map_executor(self.item_executor()))
        self.child_calls.clear()
        follow = flow.new_run(self.state, self.map_graph(), "make r2 taller", continue_from=first.run_id)
        finished = flow.walk(self.state, follow.run_id, execute=self.map_executor(self.item_executor()))
        self.assertEqual(finished.status, "done", finished.reason)
        self.assertEqual([call[0] for call in self.child_calls], ["r2"])
        self.assertEqual(self.child_calls[0][1], "make r2 taller")
        self.assertIsNotNone(self.child_calls[0][2])



    def test_a_cut_before_items_start_creates_none(self):
        run = flow.new_run(self.state, self.map_graph(), cwd=str(self.work))
        run.inflight = {"node": "regions", "visit": 1, "attempt": 0, "job_ids": []}
        flow.save_run(self.state, run)
        flow.cut_node(self.state, run.run_id, "enough")
        outcome = self.map_executor(self.item_executor())._run_map(run.graph["nodes"][0], run, "")
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.error, "cut off by the user: enough")
        self.assertEqual(run.inflight.get("children") or {}, {})
        self.assertEqual(self.child_calls, [])

    def test_a_cut_stops_running_items_and_starts_no_more(self):
        release, started = threading.Event(), []

        def execute(node, run, prev):
            with self.lock:
                started.append(run.state["key"])
            release.wait(5)
            return NodeOutcome(ok=True, verdict="PASS")

        run = flow.new_run(self.state, self.map_graph(), cwd=str(self.work))
        run.inflight = {"node": "regions", "visit": 1, "attempt": 0, "job_ids": []}
        flow.save_run(self.state, run)

        def cut_once_two_run():
            deadline = time.monotonic() + 5
            while len(started) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            flow.cut_node(self.state, run.run_id, "enough")
            release.set()

        cutter = threading.Thread(target=cut_once_two_run)
        cutter.start()
        self.addCleanup(cutter.join, 10)
        outcome = self.map_executor(execute)._run_map(run.graph["nodes"][0], run, "")
        self.assertFalse(outcome.ok)
        self.assertEqual(sorted(started), ["r1", "r2"])
        statuses = {flow.load_run(self.state, child).status for child in run.inflight["children"].values()}
        self.assertEqual(statuses, {"stopped"})


class ReviewFixTests(Base):
    """Regressions for the pre-release adversarial review."""

    def item_map(self, items, files_template, **extra):
        return graph([{"id": "m", "kind": "map", "items": items,
                       "node": {"kind": "agent", "task": "t", "files": [files_template]}, **extra}])

    def test_a_typo_in_map_items_fails_instead_of_passing_empty(self):
        run = self.start(self.item_map("{{state.regionz}}", "src/{{item}}.py"),
                         JobNodeExecutor(self.state, spawn=lambda *a: None, poll_seconds=0.01))
        self.assertEqual(run.status, "failed")
        self.assertIn("resolved to nothing", run.steps[0]["error"])
        empty = self.start(self.item_map([], "src/{{item}}.py"),
                           JobNodeExecutor(self.state, spawn=lambda *a: None, poll_seconds=0.01))
        self.assertEqual(empty.status, "done")

    def test_overlapping_item_scopes_are_refused(self):
        g = self.item_map([{"id": "a", "f": "x.py"}, {"id": "b", "f": "x.py"}], "{{item.f}}")
        run = self.start(g, JobNodeExecutor(self.state, spawn=lambda *a: None, poll_seconds=0.01))
        self.assertEqual(run.status, "failed")
        self.assertIn("both own 'x.py'", run.steps[0]["error"])

    def test_stop_kills_a_long_shell_command_promptly(self):
        g = graph([{"id": "slow", "kind": "shell", "command": f'"{sys.executable}" -c "import time;time.sleep(60)"'}])
        run = flow.new_run(self.state, g, cwd=str(self.work))
        timer = threading.Timer(1.0, flow.request_stop, args=(self.state, run.run_id))
        timer.start()
        started = time.monotonic()
        finished = flow.walk(self.state, run.run_id, execute=JobNodeExecutor(self.state))
        timer.join()
        self.assertLess(time.monotonic() - started, 20)
        self.assertEqual(finished.status, "stopped")

    def test_cut_kills_a_shell_command_and_takes_the_fail_edge(self):
        g = graph([{"id": "slow", "kind": "shell", "command": f'"{sys.executable}" -c "import time;time.sleep(60)"'},
                   {"id": "after", "kind": "set", "values": {"routed": "yes"}}],
                  [{"from": "slow", "to": "after", "when": "fail"}])
        run = flow.new_run(self.state, g, cwd=str(self.work))

        def cut_when_running():
            for _ in range(100):
                if (flow.load_run(self.state, run.run_id).inflight or {}).get("shell"):
                    flow.cut_node(self.state, run.run_id, "too slow")
                    return
                time.sleep(0.05)

        thread = threading.Thread(target=cut_when_running)
        thread.start()
        finished = flow.walk(self.state, run.run_id, execute=JobNodeExecutor(self.state))
        thread.join()
        self.assertEqual(finished.status, "done")
        self.assertIn("cut off by the user: too slow", finished.steps[0]["error"])
        self.assertEqual(finished.state["routed"], "yes")

    def test_feedback_reaches_a_fresh_start_and_retries_get_a_continue_prompt(self):
        run = flow.new_run(self.state, graph([agent("build", resume=False)]), "goal", cwd=str(self.work))
        executor = JobNodeExecutor(self.state)
        node = run.graph["nodes"][0]
        fresh = executor._spec("build", node, run, "x.py:3 off by one", 2)
        self.assertNotIn("resume_from", fresh.payload)
        self.assertIn("This is revision 2", fresh.instruction)
        self.assertIn("x.py:3 off by one", fresh.instruction)
        run.graph["nodes"][0].pop("resume")
        run.sessions["build"] = {"job_id": "j", "task_id": "t", "run_id": run.run_id, "visit": 2, "attempt": 0}
        retry = executor._spec("build", run.graph["nodes"][0], run, "x.py:3 off by one", 2)
        self.assertIn("previous attempt at this task stopped", retry.payload["resume_prompt"])

    def test_a_continued_run_sends_follow_ups_not_a_retry_prompt(self):
        first = flow.new_run(self.state, graph([agent("build"), judge("review")],
                                               [{"from": "build", "to": "review"}]), "goal",
                             cwd=str(self.work))
        for node, role in (("build", "implement"), ("review", "review")):
            first.sessions[node] = {"job_id": f"j_{node}", "task_id": f"t_{node}", "adapter": "codex",
                                    "run_id": first.run_id, "visit": 1, "attempt": 0}
        flow._finish(self.state, first, "done", "ok")
        follow = flow.new_run(self.state, first.graph, "also handle empty input",
                              continue_from=first.run_id, cwd=str(self.work))
        executor = JobNodeExecutor(self.state)
        build = executor._spec("build", follow.graph["nodes"][0], follow, "", 1)
        review = executor._spec("review", follow.graph["nodes"][1], follow, "", 1)
        self.assertEqual(build.payload["resume_from"]["job_id"], "j_build")
        self.assertNotIn("previous attempt", build.payload["resume_prompt"])
        self.assertIn("Follow-up request", build.payload["resume_prompt"])
        self.assertIn("also handle empty input", build.payload["resume_prompt"])
        self.assertNotIn("previous attempt", review.payload["resume_prompt"])

    def test_item_files_are_limited_to_its_scope(self):
        run = flow.new_run(self.state, graph([agent("build", files=["src/a.py"])]), cwd=str(self.work))
        run.inflight = {"node": "build", "visit": 1, "attempt": 0, "job_ids": []}
        executor = JobNodeExecutor(self.state)
        with patch.object(JobNodeExecutor, "_launch", return_value=("job_x", None, None)), \
                patch.object(flow, "task_outcome",
                             return_value=NodeOutcome(ok=True, files=["src/a.py", "src/b.py"], task_id="t")):
            outcome = executor(run.graph["nodes"][0], run, "")
        self.assertEqual(outcome.files, ["src/a.py"])

    def test_nested_flow_start_is_refused_inside_a_worker(self):
        from puppetmaster.cli.commands_flow import flow_action

        with patch.dict(os.environ, {"PUPPETMASTER_WORKER": "1"}), self.assertRaises(ValueError):
            flow_action(self.state, "run", {"graph": graph([agent("a")])})
        with patch.dict(os.environ, {"PUPPETMASTER_WORKER": "1"}):
            body, _ = flow_action(self.state, "list", {})
        self.assertIn("runs", body)

    def test_relative_graph_path_resolves_against_the_callers_cwd(self):
        (self.work / "g.json").write_text(json.dumps(graph([agent("a")])), encoding="utf-8")
        self.assertEqual(flow.load_graph(self.state, "g.json", base=str(self.work))["id"], "rt-flow")
        with self.assertRaises(FlowError):
            flow.load_graph(self.state, "g.json", base=str(self.root))


class CliAndMcpTests(Base):
    def cli(self, *argv):
        from puppetmaster.cli import main

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--state-dir", str(self.state), *argv])
        return code, out.getvalue()

    def write_graph(self, g):
        path = self.root / "graph.json"
        path.write_text(json.dumps(g), encoding="utf-8")
        return str(path)

    def shell_graph(self, verdict="PASS"):
        return {"id": "cli-flow", "entry": "check", "cwd": str(self.work),
                "nodes": [{"id": "check", "kind": "shell",
                           "command": f'"{sys.executable}" -c "print(\'VERDICT: {verdict} - cli\')"'},
                          {"id": "ask", "kind": "gate", "question": "ship?", "options": ["yes"]},
                          {"id": "done", "kind": "end"}],
                "edges": [{"from": "check", "to": "ask", "when": "PASS"},
                          {"from": "ask", "to": "done", "when": "answer=yes"}]}

    def test_cli_validate_run_answer_status_list(self):
        path = self.write_graph(self.shell_graph())
        self.assertEqual(self.cli("flow", "validate", path)[0], 0)
        code, text = self.cli("flow", "run", path, "--foreground", "--input", "x")
        body = json.loads(text)
        self.assertEqual((code, body["status"]), (4, "waiting"))
        self.assertEqual(body["gate"]["question"], "ship?")
        code, text = self.cli("flow", "resume", body["run_id"], "--answer=yes")
        self.assertEqual((code, json.loads(text)["status"]), (0, "done"))
        code, text = self.cli("flow", "status", body["run_id"], "--since", "1")
        self.assertEqual([step["node"] for step in json.loads(text)["steps"]], ["ask", "done"])
        self.assertEqual(json.loads(self.cli("flow", "list")[1])["runs"][0]["run_id"], body["run_id"])

    def test_cli_invalid_graph_exits_2(self):
        path = self.write_graph({"id": "Bad", "nodes": []})
        code, text = self.cli("flow", "validate", path)
        self.assertEqual(code, 2)
        self.assertFalse(json.loads(text)["valid"])

    def test_background_walker_runs_detached_to_completion(self):
        path = self.write_graph(self.shell_graph())
        code, text = self.cli("flow", "run", path, "--wait", "--timeout", "60")
        body = json.loads(text)
        self.assertEqual(body["status"], "waiting", body)
        code, text = self.cli("flow", "resume", body["run_id"], "--answer=yes", "--background")
        final = flow.wait_for_event(self.state, body["run_id"], timeout_seconds=60)
        self.assertEqual(final.status, "done", final.reason)

    def test_mcp_flow_tool(self):
        from puppetmaster.mcp_server import call_tool

        base = {"state_dir": str(self.state), "cwd": str(self.work)}
        result = call_tool("puppetmaster_flow", {**base, "action": "validate", "graph": self.shell_graph()})
        self.assertTrue(json.loads(result["content"][0]["text"])["valid"])
        started = json.loads(call_tool("puppetmaster_flow", {**base, "action": "run", "graph": self.shell_graph(),
                                                             "wait": True, "timeout_seconds": 60})["content"][0]["text"])
        self.assertEqual(started["status"], "waiting", started)
        call_tool("puppetmaster_flow", {**base, "action": "resume", "run_id": started["run_id"], "answer": "yes"})
        final = json.loads(call_tool("puppetmaster_flow", {**base, "action": "wait", "run_id": started["run_id"],
                                                           "timeout_seconds": 60})["content"][0]["text"])
        self.assertEqual(final["status"], "done", final)
        bad = call_tool("puppetmaster_flow", {**base, "action": "status", "run_id": "../x"})
        self.assertTrue(bad.get("isError"))


class AdoptTests(Base):
    def test_adopt_creates_the_tasks_of_a_job_that_died_before_creating_them(self):
        from puppetmaster.orchestrator import Orchestrator
        from puppetmaster.workers import WorkerSpec

        store = SQLiteSwarmStore(self.state)
        store.init()
        job = store.create_job("born empty")
        spec = WorkerSpec(role="check", instruction="i", adapter="shell",
                          payload={"command": py("print('ok')"), "cwd": str(self.work)})
        result = Orchestrator(store).adopt(job.id, worker_mode="inline", specs=[spec])
        self.assertEqual(result.job.status, JobStatus.COMPLETE)
        self.assertEqual([task.role for task in store.list_tasks(job.id)], ["check"])

    def test_adopt_drives_a_dead_coordinators_queued_tasks(self):
        from puppetmaster.orchestrator import Orchestrator

        store = SQLiteSwarmStore(self.state)
        store.init()
        job = store.create_job("orphaned job")
        store.save_tasks([Task(job_id=job.id, role="check", instruction="i", adapter="shell",
                               payload={"command": py("print('ok')"), "cwd": str(self.work)})])
        result = Orchestrator(store).adopt(job.id, worker_mode="inline")
        self.assertEqual(result.job.status, JobStatus.COMPLETE)
        self.assertEqual(store.list_tasks(job.id)[0].status, TaskStatus.COMPLETE)
        again = Orchestrator(store).adopt(job.id, worker_mode="inline")
        self.assertEqual(again.job.status, JobStatus.COMPLETE)


class SecondPassTests(Base):
    """Process identity for walkers and shell orphans; launches, adopt and validation."""

    def sleeper(self) -> subprocess.Popen:
        kwargs = {"creationflags": 0x00000200} if os.name == "nt" else {"start_new_session": True}
        process = subprocess.Popen(py("import time; time.sleep(60)"), **kwargs)
        reaper = threading.Thread(target=process.wait, daemon=True)
        reaper.start()

        def cleanup():
            if process.returncode is None:
                try:
                    process.kill()
                except OSError:
                    pass
            reaper.join(5)

        self.addCleanup(cleanup)
        return process

    def shell_run(self, command, **extra):
        node = {"id": "sh", "kind": "shell", "command": command, **extra}
        run = flow.new_run(self.state, graph([node]), cwd=str(self.work))
        run.inflight = {"node": "sh", "visit": 1, "attempt": 0, "job_ids": []}
        flow.save_run(self.state, run)
        return run, run.graph["nodes"][0]

    def test_a_free_run_lock_outweighs_a_live_pid_on_the_run(self):
        other = self.sleeper()
        run = flow.new_run(self.state, graph([agent("a")]), cwd=str(self.work))
        run.pid = other.pid
        flow.save_run(self.state, run)
        self.assertFalse(flow.walker_alive(self.state, run.run_id))

    def test_the_spawn_marker_counts_only_for_the_process_it_names(self):
        from puppetmaster.proc_identity import process_identity

        child = self.sleeper()
        identity = process_identity(child.pid)
        if identity is None:
            self.skipTest("no process identity on this platform")
        run = flow.new_run(self.state, graph([agent("a")]), cwd=str(self.work))
        marker = flow._marker(self.state, run.run_id, "walker.pid")
        for at, proc, alive in ((time.time(), identity, True), (time.time(), "a-later-process", False),
                                (time.time() - 3600, identity, False)):
            marker.write_text(json.dumps({"pid": child.pid, "at": at, "proc": proc}), encoding="utf-8")
            self.assertEqual(flow.walker_alive(self.state, run.run_id), alive, proc)

    def test_a_resumed_shell_node_kills_its_own_orphan_and_runs_again(self):
        from puppetmaster.proc_identity import process_identity

        orphan = self.sleeper()
        identity = process_identity(orphan.pid)
        if identity is None:
            self.skipTest("no process identity on this platform")
        run, node = self.shell_run("echo again")
        run.inflight["shell"] = {"pid": orphan.pid, "proc": identity}
        outcome = JobNodeExecutor(self.state)._run_shell(node, run, "")
        self.assertTrue(outcome.ok, outcome.error)
        self.assertIn("again", outcome.output)
        orphan.wait(5)
        self.assertIsNotNone(orphan.returncode)

    def test_a_pid_that_now_names_another_process_is_left_alone(self):
        from puppetmaster.liveness import _pid_alive

        other = self.sleeper()
        run, node = self.shell_run("echo fine")
        run.inflight["shell"] = {"pid": other.pid, "proc": "a-command-that-died"}
        started = time.monotonic()
        outcome = JobNodeExecutor(self.state)._run_shell(node, run, "")
        self.assertTrue(outcome.ok, outcome.error)
        self.assertLess(time.monotonic() - started, 5)
        self.assertTrue(_pid_alive(other.pid))

    def test_waiting_out_an_unidentified_orphan_still_honours_a_stop(self):
        from puppetmaster.liveness import _pid_alive

        orphan = self.sleeper()
        run, node = self.shell_run("echo never", timeoutMs=60_000)
        run.inflight["shell"] = {"pid": orphan.pid}  # recorded before identities existed
        flow._marker(self.state, run.run_id, "stop").write_text("now", encoding="utf-8")
        started = time.monotonic()
        outcome = JobNodeExecutor(self.state)._run_shell(node, run, "")
        self.assertEqual(outcome.error, "stopped by request")
        self.assertLess(time.monotonic() - started, 5)
        self.assertTrue(_pid_alive(orphan.pid))

    def test_a_launch_that_creates_no_job_reports_its_own_error(self):
        from puppetmaster.orchestrator import Orchestrator

        store = SQLiteSwarmStore(self.state)
        store.init()
        earlier = store.create_job("attempt 0", launch_key="flow:x:a:1:0")
        run = flow.new_run(self.state, graph([agent("a")]), cwd=str(self.work))
        run.inflight = {"node": "a", "visit": 1, "attempt": 1, "job_ids": [earlier.id]}
        with patch.object(Orchestrator, "run", side_effect=RuntimeError("boom")):
            job_id, error, _ = JobNodeExecutor(self.state)._launch(run, [], "flow:x:a:1:1", "goal")
        self.assertIsNone(job_id)
        self.assertEqual(error, "RuntimeError: boom")

    def test_adopt_prepares_the_tasks_it_creates_like_a_launch(self):
        from puppetmaster.orchestrator import Orchestrator
        from puppetmaster.workers import WorkerSpec

        store = SQLiteSwarmStore(self.state)
        store.init()
        job = store.create_job("born empty")
        spec = WorkerSpec(role="check", instruction="i", adapter="shell",
                          payload={"command": py("print('ok')"), "cwd": str(self.work)})
        with patch.object(Orchestrator, "_ensure_job_brief", autospec=True) as brief, \
                patch.object(Orchestrator, "_with_retrieved_memory", autospec=True,
                             side_effect=lambda self, specs, goal, job_id=None: specs) as memory:
            result = Orchestrator(store).adopt(job.id, worker_mode="inline", specs=[spec])
        self.assertEqual(result.job.status, JobStatus.COMPLETE)
        brief.assert_called_once()
        self.assertEqual(memory.call_args.args[2], "born empty")

    def test_payloads_must_be_objects(self):
        problems = flow.validate_graph(graph([agent("a", payload="x")], defaults={"payload": "y"}))
        self.assertIn("defaults.payload must be an object", problems)
        self.assertIn("node 'a' payload must be an object", problems)


class AdapterHookTests(unittest.TestCase):
    def test_terminal_verdict_payload_flag_asks_for_a_verdict(self):
        from puppetmaster.adapters._prompts import structured_prompt_for_task

        plain = structured_prompt_for_task(Task(job_id="j", role="audit", instruction="look"))
        flagged = structured_prompt_for_task(Task(job_id="j", role="audit", instruction="look",
                                                  payload={"terminal_verdict": True}))
        self.assertNotIn("VERDICT: PASS", plain)
        self.assertIn("VERDICT: PASS", flagged)


if __name__ == "__main__":
    unittest.main()


class EffortLaneTests(Base):
    def test_node_lane_and_graph_effort_precedence(self):
        defaults = {"effort": "medium", "lanes": {"explore": "low", "judge": "high"}}
        self.assertEqual(flow.node_effort({"kind": "agent", "role": "explore"}, defaults, 1), "low")
        self.assertEqual(flow.node_effort({"kind": "judge"}, defaults, 1), "high")
        self.assertEqual(flow.node_effort({"kind": "agent"}, defaults, 1), "medium")
        self.assertEqual(flow.node_effort({"kind": "agent", "effort": "xhigh"}, defaults, 1), "xhigh")
        self.assertIsNone(flow.node_effort({"kind": "agent"}, {}, 1))

    def test_escalation_thinks_harder_on_each_repair_visit(self):
        node = {"kind": "agent", "effort": "low", "escalate": True}
        self.assertEqual([flow.node_effort(node, {}, visit) for visit in (1, 2, 3, 4, 5)],
                         ["low", "medium", "high", "xhigh", "xhigh"])
        self.assertEqual(flow.node_effort({"kind": "agent"}, {"escalate": True}, 2), "high")

    def test_spec_pins_reasoning_effort_for_the_visit(self):
        g = graph([agent("build", effort="low", escalate=True)], defaults={"adapter": "codex"})
        run = flow.new_run(self.state, g, "goal", cwd=str(self.work))
        executor = JobNodeExecutor(self.state)
        self.assertEqual(executor._spec("build", run.graph["nodes"][0], run, "", 1).payload["reasoning_effort"], "low")
        self.assertEqual(executor._spec("build", run.graph["nodes"][0], run, "fix x", 2).payload["reasoning_effort"], "medium")

    def test_effort_validation(self):
        bad = graph([agent("a", effort="max", escalate="yes")],
                    defaults={"adapter": "codex", "lanes": {"review": "low"}, "effort": "huge"})
        problems = flow.validate_graph(bad)
        self.assertTrue(any("node 'a' effort" in p for p in problems))
        self.assertTrue(any("escalate must be true or false" in p for p in problems))
        self.assertTrue(any("defaults.lanes" in p for p in problems))
        self.assertTrue(any("defaults effort" in p for p in problems))

    def test_item_graph_lanes_merge_over_the_parent(self):
        merged = flow._merged_defaults({"lanes": {"explore": "low"}, "adapter": "codex"}, {"lanes": {"judge": "high"}})
        self.assertEqual(merged["lanes"], {"explore": "low", "judge": "high"})

