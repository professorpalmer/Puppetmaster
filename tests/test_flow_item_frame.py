"""A map item's builder knows its peers, and an inherited result shows its origin.

Codex R163 (PM 1.45.0, reuse on, follow-up): the group_03 builder ran the
world check while the group_01 builder was still at work. The check failed
only in group_01's regions, the builder ended FAIL, and that FAIL failed the
item although its own fresh review passed. The run stopped at the map and
still held the cold world review's PASS with no mark of where it came from.
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
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from puppetmaster import flow
from puppetmaster.flow import JobNodeExecutor, NodeOutcome

ITEM_GRAPH = {"entry": "build", "defaults": {"adapter": "codex"},
              "nodes": [{"id": "build", "kind": "agent", "task": "Build {{item.id}}.", "files": "{{item.files}}",
                         "revise": "Apply this: {{prev}}"},
                        {"id": "review", "kind": "judge", "task": "Review {{item.id}}.", "files": "{{item.files}}"},
                        {"id": "done", "kind": "end", "status": "pass"}],
              "edges": [{"from": "build", "to": "review"}, {"from": "review", "to": "done", "when": "PASS"}]}


def graph():
    items = [{"id": key, "files": [f"regions/{key}.py"]} for key in ("group_00", "group_01")]
    item_graph = {name: value for name, value in ITEM_GRAPH.items() if name != "defaults"}
    return {"id": "frame-flow", "entry": "groups", "defaults": {"adapter": "codex"},
            "nodes": [{"id": "groups", "kind": "map", "items": items, "concurrency": 2, "graph": item_graph},
                      {"id": "world", "kind": "judge", "task": "Review the world.", "files": ["regions/*.py"]},
                      {"id": "done", "kind": "end", "status": "pass"}],
            "edges": [{"from": "groups", "to": "world", "when": "ok"},
                      {"from": "world", "to": "done", "when": "PASS"}]}


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.state, self.work = root / "state", root / "work"
        self.state.mkdir()
        (self.work / "regions").mkdir(parents=True)
        for key in ("group_00", "group_01"):
            (self.work / "regions" / f"{key}.py").write_text(f"# {key}\n")


class ItemFrameTests(Base):
    def item_run(self, continued=False):
        parent = {"run_id": "flow_parent00000", "node": "groups", "key": "group_00", "depth": 1,
                  "peer_scopes": ["regions/group_01.py"]}
        prior = None
        if continued:
            prior = flow.new_run(self.state, {**ITEM_GRAPH}, "", state={"item": {"id": "group_00"}},
                                 parent=parent, cwd=str(self.work))
            prior.status = "done"
            prior.sessions["build"] = {"job_id": "job_1", "task_id": "task_1", "run_id": prior.run_id, "visit": 1}
            flow.save_run(self.state, prior)
        return flow.new_run(self.state, {**ITEM_GRAPH}, "Revise group_00 and group_01.",
                            state={"item": {"id": "group_00", "files": ["regions/group_00.py"]}},
                            parent=parent, cwd=str(self.work),
                            continue_from=prior.run_id if prior else None)

    def spec(self, run, node_id):
        node = next(node for node in run.graph["nodes"] if node["id"] == node_id)
        return JobNodeExecutor(self.state)._spec(node_id, node, run, run.input, 1)

    def test_a_map_item_builder_knows_its_peers_and_owns_its_verdict(self):
        task = self.spec(self.item_run(), "build").instruction
        self.assertIn("You build item group_00 of map groups.", task)
        self.assertIn("Other items build at the same time in this workspace and own: regions/group_01.py.", task)
        self.assertIn("fails only in files that another item owns is not your failure", task)
        self.assertNotIn("follow-up request is for the whole map", task)
        self.assertTrue(task.rstrip().endswith("`VERDICT: PARTIAL - <what you could not finish or verify>`."))

    def test_a_resumed_follow_up_builder_gets_the_frame_before_its_revise_text(self):
        spec = self.spec(self.item_run(continued=True), "build")
        delta = spec.payload["resume_prompt"]
        self.assertTrue(delta.startswith("You build item group_00 of map groups."))
        self.assertIn("The follow-up request is for the whole map.", delta)
        self.assertIn("Apply this: Revise group_00 and group_01.", delta)
        self.assertEqual(delta.count("You build item group_00"), 1)

    def test_a_judge_and_a_top_level_builder_get_no_frame(self):
        self.assertNotIn("You build item", self.spec(self.item_run(), "review").instruction)
        top = flow.new_run(self.state, {"id": "t", "entry": "build", "defaults": {"adapter": "codex"},
                                        "nodes": [{"id": "build", "kind": "agent", "task": "Build."}],
                                        "edges": []}, "", cwd=str(self.work))
        self.assertNotIn("You build item", self.spec(top, "build").instruction)


class NestedPeerTests(Base):
    """R179 replica, one builder per region inside each group: a region builder's write-scope
    gate charged it with 79 render files that builders in the other groups wrote meanwhile."""

    def test_a_nested_map_item_knows_the_outer_peers(self):
        inner = {"entry": "build",
                 "nodes": [{"id": "build", "kind": "agent", "task": "Build {{item}}.", "files": ["regions/{{item}}.py"]},
                           {"id": "built", "kind": "end", "status": "pass"}],
                 "edges": [{"from": "build", "to": "built", "when": "ok"}]}
        groups = {"id": "groups", "kind": "map", "concurrency": 2,
                  "items": [{"id": "group_00", "regions": ["r0", "r1"]}, {"id": "group_01", "regions": ["r2", "r3"]}],
                  "graph": {"entry": "regions",
                            "nodes": [{"id": "regions", "kind": "map", "items": "{{item.regions}}", "graph": inner},
                                      {"id": "done", "kind": "end", "status": "pass"}],
                            "edges": [{"from": "regions", "to": "done", "when": "ok"}]}}
        graph = {"id": "nested-flow", "entry": "groups", "defaults": {"adapter": "codex"},
                 "nodes": [groups, {"id": "done", "kind": "end", "status": "pass"}],
                 "edges": [{"from": "groups", "to": "done", "when": "ok"}]}

        def execute(node, run, prev):
            return NodeOutcome(ok=True, verdict="PASS", output="built", files=[f"regions/{run.state['item']}.py"])

        def spawn(state_dir, run_id):
            thread = threading.Thread(target=flow.walk, args=(state_dir, run_id), kwargs={"execute": route})
            thread.start()
            self.addCleanup(thread.join, 10)

        executor = JobNodeExecutor(self.state, spawn=spawn, poll_seconds=0.02)

        def route(node, run, prev):
            return executor(node, run, prev) if node["kind"] == "map" else execute(node, run, prev)

        run = flow.new_run(self.state, graph, "", cwd=str(self.work))
        run = flow.walk(self.state, run.run_id, execute=route)
        self.assertEqual(run.status, "done", run.reason)
        peers = {run_.parent["key"]: set(run_.parent["peer_scopes"]) for run_ in map(
            lambda path: flow.load_run(self.state, path.parent.name), self.state.glob("flows/runs/*/run.json"))
                 if (run_.parent or {}).get("key") in ("r0", "r1", "r2", "r3")}
        self.assertEqual(peers["r0"], {"regions/r1.py", "regions/r2.py", "regions/r3.py"})
        self.assertEqual(peers["r3"], {"regions/r0.py", "regions/r1.py", "regions/r2.py"})


class BarrierTests(Base):
    """A fast item's local PASS never stands in for the world check after the join."""

    def walk(self, execute, text="", continue_from=None):
        def spawn(state_dir, run_id):
            thread = threading.Thread(target=flow.walk, args=(state_dir, run_id), kwargs={"execute": execute})
            thread.start()
            self.addCleanup(thread.join, 10)

        executor = JobNodeExecutor(self.state, spawn=spawn, poll_seconds=0.02)

        def route(node, run, prev):
            return executor(node, run, prev) if node["kind"] == "map" else execute(node, run, prev)

        run = flow.new_run(self.state, graph(), text, cwd=str(self.work), continue_from=continue_from)
        return flow.walk(self.state, run.run_id, execute=route)

    def worker(self, world="PASS", build_fail=None):
        order = []

        def execute(node, run, prev):
            key = run.state.get("key") or "root"
            if node["kind"] == "agent":
                if key == "group_01":
                    time.sleep(0.2)  # the slow peer
                order.append((key, "build", time.monotonic()))
                verdict = "FAIL" if key == build_fail else "PASS"
                return NodeOutcome(ok=True, verdict=verdict, output=f"VERDICT: {verdict}",
                                   files=run.state["item"]["files"])
            order.append((key, node["id"], time.monotonic()))
            scope = flow.render_files(node.get("files"), run, prev)
            outcome = NodeOutcome(ok=True, verdict=world if node["id"] == "world" else "PASS", output="ok")
            outcome.reviewed = {**flow._source_state(run.graph["cwd"], scope), "scope": scope}
            return outcome

        return execute, order

    def test_the_world_review_runs_after_the_slow_peer(self):
        execute, order = self.worker()
        run = self.walk(execute)
        self.assertEqual(run.status, "done", run.reason)
        world = next(at for key, node, at in order if node == "world")
        self.assertTrue(all(at < world for key, node, at in order if node == "build"))

    def test_a_world_fail_still_fails_the_run(self):
        execute, _ = self.worker(world="FAIL")
        run = self.walk(execute)
        self.assertNotEqual(run.status, "done")

    def test_a_real_local_fail_still_fails_the_item_and_stops_before_the_world(self):
        execute, order = self.worker(build_fail="group_00")
        run = self.walk(execute)
        self.assertEqual(run.status, "failed")
        self.assertFalse(run.results["groups"]["items"]["group_00"]["ok"])
        self.assertNotIn("world", [node for _, node, _ in order])

    def test_an_inherited_result_shows_its_origin_and_whether_it_is_current(self):
        execute, _ = self.worker()
        cold = self.walk(execute)
        (self.work / "regions" / "group_01.py").write_text("# revised\n")
        follow = flow.new_run(self.state, graph(), "group_01", cwd=str(self.work), continue_from=cold.run_id)
        inherited = flow.run_summary(follow)["inherited"]
        self.assertEqual(inherited["world"], {"from": cold.run_id, "verdict": "PASS", "current": False})
        self.assertEqual(inherited["groups"]["from"], cold.run_id)
        execute, _ = self.worker(build_fail="group_01")
        follow = flow.walk(self.state, follow.run_id, execute=self._route(execute))
        self.assertEqual(follow.status, "failed")
        summary = flow.run_summary(follow)["inherited"]
        self.assertNotIn("groups", summary)  # the map ran again in this run
        self.assertEqual(summary["world"]["from"], cold.run_id)  # the world review did not

    def test_a_chained_continuation_keeps_the_first_origin(self):
        execute, _ = self.worker()
        cold = self.walk(execute)
        second = flow.new_run(self.state, graph(), "", cwd=str(self.work), continue_from=cold.run_id)
        second.status = "failed"
        flow.save_run(self.state, second)
        third = flow.new_run(self.state, graph(), "", cwd=str(self.work), continue_from=second.run_id)
        self.assertEqual(third.results["world"]["inherited_from"], cold.run_id)

    def _route(self, execute):
        def spawn(state_dir, run_id):
            thread = threading.Thread(target=flow.walk, args=(state_dir, run_id), kwargs={"execute": execute})
            thread.start()
            self.addCleanup(thread.join, 10)

        executor = JobNodeExecutor(self.state, spawn=spawn, poll_seconds=0.02)
        return lambda node, run, prev: executor(node, run, prev) if node["kind"] == "map" else execute(node, run, prev)


if __name__ == "__main__":
    unittest.main()
