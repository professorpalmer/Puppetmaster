"""Continuation reuse: an unchanged map item keeps its result on a follow-up.

Codex R157 (PM 1.44.0, N16 grouped): every follow-up rebuilt and reviewed all
four groups. The request named only revised regions, so unchanged groups
spent build and review time on revalidation (one unchanged group held the map
finish at 202 s). Opt-in ``reuse`` keeps an item whose task, declared inputs,
own files and review digests did not change.
"""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from puppetmaster import flow
from puppetmaster.flow import JobNodeExecutor, NodeOutcome

GROUPS = {
    "group_00": ["region_000", "region_004"],
    "group_01": ["region_001", "region_005"],
    "group_02": ["region_002", "region_006"],
    "group_03": ["region_003", "region_007"],
}
FOLLOW_UP = ("Read current contracts: region_001 and region_007 are revision 2. "
             "Apply them within the original task and preserve unaffected source.")


def items(order=None):
    keys = order or list(GROUPS)
    return [{"id": key, "regions": GROUPS[key],
             "files": [f"regions/{region}.py" for region in GROUPS[key]],
             "inputs": [f"contracts/{region}.json" for region in GROUPS[key]] + ["rubric.json"]}
            for key in keys]


def graph(order=None, reuse=True, task="Build {{item.id}}. Read each current contract."):
    groups = {"id": "groups", "kind": "map", "items": items(order), "concurrency": 4,
              "graph": {"entry": "build",
                        "nodes": [{"id": "build", "kind": "agent", "task": task, "files": "{{item.files}}"},
                                  {"id": "review", "kind": "judge", "task": "Review {{item.id}}.",
                                   "files": "{{item.files}}"},
                                  {"id": "done", "kind": "end", "status": "pass"}],
                        "edges": [{"from": "build", "to": "review"},
                                  {"from": "review", "to": "done", "when": "PASS"},
                                  {"from": "review", "to": "build", "when": "FAIL", "max": 2}]}}
    if reuse:
        groups["reuse"] = {"inputs": "{{item.inputs}}", "names": "{{item.regions}}"}
    return {"id": "reuse-flow", "entry": "groups", "defaults": {"adapter": "codex"},
            "nodes": [groups, {"id": "world", "kind": "judge", "task": "Review the world.",
                               "files": ["regions/*.py"]},
                      {"id": "done", "kind": "end", "status": "pass"}],
            "edges": [{"from": "groups", "to": "world", "when": "ok"},
                      {"from": "world", "to": "done", "when": "PASS"}]}


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.state, self.work = root / "state", root / "work"
        self.state.mkdir()
        for key, regions in GROUPS.items():
            for region in regions:
                self.write(f"regions/{region}.py", f"# {region} v1\n")
                self.write(f"contracts/{region}.json", '{"revision": 1}')
        self.write("rubric.json", "{}")
        self.calls = []
        self.lock = threading.Lock()

    def write(self, rel, text):
        path = self.work / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def execute(self, node, run, prev):
        """A builder that edits nothing and a judge whose PASS binds to the item files."""
        key = run.state.get("key")
        with self.lock:
            self.calls.append((key or node["id"], node["id"]))
        if node["kind"] == "agent":
            return NodeOutcome(ok=True, verdict="PASS", output="built", files=run.state["item"]["files"],
                               usage={"input_tokens": 100})
        scope = flow.render_files(node.get("files"), run, prev)
        outcome = NodeOutcome(ok=True, verdict="PASS", output="looks right")
        outcome.reviewed = {**flow._source_state(run.graph["cwd"], scope), "scope": scope}
        return outcome

    def executor(self):
        def spawn(state_dir, run_id):
            thread = threading.Thread(target=flow.walk, args=(state_dir, run_id), kwargs={"execute": self.execute})
            thread.start()
            self.addCleanup(thread.join, 10)

        executor = JobNodeExecutor(self.state, spawn=spawn, poll_seconds=0.02)

        def route(node, run, prev):
            return executor(node, run, prev) if node["kind"] == "map" else self.execute(node, run, prev)

        return route

    def cold(self, **kw):
        run = flow.new_run(self.state, graph(**kw), "", cwd=str(self.work))
        run = flow.walk(self.state, run.run_id, execute=self.executor())
        self.assertEqual(run.status, "done", run.reason)
        return run

    def follow(self, prior, text=FOLLOW_UP, **kw):
        self.calls.clear()
        run = flow.new_run(self.state, graph(**kw), text, cwd=str(self.work), continue_from=prior.run_id)
        return flow.walk(self.state, run.run_id, execute=self.executor())

    def built(self):
        return sorted({key for key, node in self.calls if node == "build"})

    def revise(self, *regions):
        for region in regions:
            self.write(f"contracts/{region}.json", '{"revision": 2}')

    def test_unchanged_unnamed_groups_skip_build_and_review(self):
        cold = self.cold()
        self.revise("region_001", "region_007")
        run = self.follow(cold)
        self.assertEqual(run.status, "done", run.reason)
        self.assertEqual(self.built(), ["group_01", "group_03"])
        reviewed = sorted({key for key, node in self.calls if node == "review"})
        self.assertEqual(reviewed, ["group_01", "group_03"])
        self.assertIn("world", [key for key, _ in self.calls])  # world review still runs
        found = run.results["groups"]["items"]
        self.assertTrue(found["group_00"]["reuse"]["reused"])
        self.assertEqual(found["group_00"]["reuse"]["source_run"],
                         flow.load_run(self.state, cold.run_id).results["groups"]["items"]["group_00"]["run_id"])
        self.assertEqual(found["group_00"]["reuse"]["source_parent"], cold.run_id)
        self.assertEqual(found["group_01"]["reuse"], {"reused": False, "reason": "the request names this item"})
        self.assertEqual(run.steps[0]["items"], {"done": 4, "reused": 2})
        self.assertTrue(run.outputs["groups"].startswith("map groups: 4/4 items passed (2 reused, 2 ran)"))

    def test_a_changed_input_runs_an_unnamed_item(self):
        cold = self.cold()
        self.revise("region_001", "region_007", "region_002")
        run = self.follow(cold)
        self.assertEqual(self.built(), ["group_01", "group_02", "group_03"])
        reason = run.results["groups"]["items"]["group_02"]["reuse"]["reason"]
        self.assertEqual(reason, "input files changed: contracts/region_002.json")

    def test_a_shared_input_change_runs_every_item(self):
        cold = self.cold()
        self.revise("region_001")
        self.write("rubric.json", '{"strict": true}')
        self.follow(cold)
        self.assertEqual(self.built(), sorted(GROUPS))

    def test_reordering_the_items_keeps_reuse(self):
        cold = self.cold()
        self.revise("region_001", "region_007")
        self.follow(cold, order=list(reversed(list(GROUPS))))
        self.assertEqual(self.built(), ["group_01", "group_03"])

    def test_a_request_that_names_no_item_runs_every_item(self):
        cold = self.cold()
        run = self.follow(cold, text="Make every palette warmer.")
        self.assertEqual(self.built(), sorted(GROUPS))
        self.assertIn("names no item", run.results["groups"]["items"]["group_00"]["reuse"]["reason"])

    def test_a_changed_task_or_output_runs_the_item(self):
        cold = self.cold()
        self.write("regions/region_000.py", "# hand edit\n")
        run = self.follow(cold, text="region_001 only")
        self.assertEqual(self.built(), ["group_00", "group_01"])
        self.assertIn("output files changed: regions/region_000.py",
                      run.results["groups"]["items"]["group_00"]["reuse"]["reason"])
        cold2 = flow.load_run(self.state, run.run_id)
        self.follow(cold2, text="region_001 only", task="Build {{item.id}} with care.")
        self.assertEqual(self.built(), sorted(GROUPS))

    def test_a_missing_input_or_no_fingerprint_prevents_reuse(self):
        cold = self.cold(reuse=False)
        run = self.follow(cold, text="region_001 only")
        self.assertEqual(self.built(), sorted(GROUPS))
        self.assertEqual(run.results["groups"]["items"]["group_00"]["reuse"]["reason"], "no recorded fingerprint")
        (self.work / "contracts/region_002.json").unlink()
        run = self.follow(flow.load_run(self.state, run.run_id), text="region_001 only")
        self.assertIn("declared input missing: contracts/region_002.json",
                      run.results["groups"]["items"]["group_02"]["reuse"]["reason"])

    def test_a_reused_item_that_changes_before_the_end_fails_the_run(self):
        cold = self.cold()
        self.revise("region_001", "region_007")
        world = self.execute

        def execute(node, run, prev):
            if node["id"] == "world":
                self.write("regions/region_002.py", "# changed after reuse\n")
            return world(node, run, prev)

        self.execute = execute
        run = self.follow(cold)
        self.assertEqual(run.status, "failed")
        self.assertIn("groups/group_02: reused result is stale: output files changed: regions/region_002.py",
                      run.reason)

    def test_reuse_chains_and_keeps_the_original_source(self):
        cold = self.cold()
        self.revise("region_001", "region_007")
        second = self.follow(cold)
        third = self.follow(flow.load_run(self.state, second.run_id), text="region_005 again")
        self.assertEqual(self.built(), ["group_01"])
        source = third.results["groups"]["items"]["group_00"]["reuse"]
        self.assertEqual(source["source_parent"], cold.run_id)

    def test_a_reused_item_adds_no_usage(self):
        cold = self.cold()
        self.revise("region_001", "region_007")
        self.assertEqual(cold.steps[0]["usage"], {"input_tokens": 400})
        run = self.follow(cold)
        self.assertEqual(run.steps[0]["usage"], {"input_tokens": 200})  # two built, two reused

    def test_reuse_off_keeps_the_old_follow_up(self):
        cold = self.cold(reuse=False)
        run = self.follow(cold, text="region_001 only", reuse=False)
        self.assertEqual(self.built(), sorted(GROUPS))  # aliases need reuse.names
        self.assertNotIn("reuse", run.results["groups"]["items"]["group_00"])
        self.assertNotIn("fingerprint", run.results["groups"]["items"]["group_00"])

    def test_a_run_without_an_end_node_still_checks_reuse(self):
        def endless(**kw):
            shape = graph(**kw)
            shape["nodes"] = [node for node in shape["nodes"] if node["id"] != "done"]
            shape["edges"] = [edge for edge in shape["edges"] if edge["to"] != "done"]
            return shape
        run = flow.new_run(self.state, endless(), "", cwd=str(self.work))
        cold = flow.walk(self.state, run.run_id, execute=self.executor())
        self.assertEqual(cold.status, "done", cold.reason)
        self.revise("region_001", "region_007")
        world = self.execute

        def execute(node, run, prev):
            if node["id"] == "world":
                self.write("regions/region_002.py", "# changed after reuse\n")
            return world(node, run, prev)

        self.execute = execute
        follow = flow.new_run(self.state, endless(), FOLLOW_UP, cwd=str(self.work), continue_from=cold.run_id)
        follow = flow.walk(self.state, follow.run_id, execute=self.executor())
        self.assertEqual(follow.status, "failed")
        self.assertIn("groups/group_02: reused result is stale", follow.reason)

    def test_a_feedback_template_does_not_hide_the_request(self):
        def with_feedback(**kw):
            shape = graph(**kw)
            shape["nodes"][0]["feedback"] = "Notes: group_02 palette"
            return shape
        run = flow.new_run(self.state, with_feedback(), "", cwd=str(self.work))
        cold = flow.walk(self.state, run.run_id, execute=self.executor())
        self.calls.clear()
        follow = flow.new_run(self.state, with_feedback(), "region_001 only", cwd=str(self.work),
                              continue_from=cold.run_id)
        flow.walk(self.state, follow.run_id, execute=self.executor())
        self.assertEqual(self.built(), ["group_01", "group_02"])

    def test_state_index_and_the_shell_context_make_the_position_count(self):
        for task in ("Build {{ state.index }}.", "Build it. PM_FLOW_CONTEXT has the item."):
            shape = {"entry": "build", "nodes": [{"id": "build", "kind": "agent", "task": task}], "edges": []}
            first = flow._task_fingerprint(shape, {"id": "a"}, 0)
            self.assertNotEqual(first, flow._task_fingerprint(shape, {"id": "a"}, 1), task)
        plain = {"entry": "build", "nodes": [{"id": "build", "kind": "agent", "task": "Build {{item.id}}."}],
                 "edges": []}
        self.assertEqual(flow._task_fingerprint(plain, {"id": "a"}, 0), flow._task_fingerprint(plain, {"id": "a"}, 1))

    def test_a_declared_file_names_its_item(self):
        cold = self.cold()
        self.follow(cold, text="Tidy regions/region_002.py and region_001.")
        self.assertEqual(self.built(), ["group_01", "group_02"])

    def test_an_item_graph_that_reads_the_request_never_reuses(self):
        cold = self.cold(task="Build {{item.id}}: {{input}}")
        run = self.follow(cold, text="region_001 only", task="Build {{item.id}}: {{input}}")
        self.assertEqual(self.built(), sorted(GROUPS))
        self.assertEqual(run.results["groups"]["items"]["group_00"]["reuse"]["reason"],
                         "its item graph reads the request")

    def test_prev_in_a_repair_template_is_feedback_not_the_request(self):
        def shape(**fields):
            return {"entry": "build", "nodes": [{"id": "build", "kind": "agent", "task": "t", **fields}],
                    "edges": []}
        self.assertFalse(flow._reads_request(shape(revise="Fix: {{prev}}")))
        self.assertTrue(flow._reads_request(shape(task="Do {{prev}}")))
        self.assertTrue(flow._reads_request(shape(revise="Fix {{input}}")))

    def test_an_item_a_request_cannot_name_runs(self):
        decisions = flow._reuse_decisions(
            {"id": "m", "kind": "map", "reuse": {"inputs": ["rubric.json"]},
             "node": {"id": "build", "kind": "agent", "task": "t"}},
            flow.FlowRun(run_id="flow_000000000001", graph={"id": "g", "cwd": str(self.work)},
                         input="fix 1 and region_x/a.py"),
            ["1", "ab"], [{"id": "1"}, {"id": "ab", "f": 1}],
            {"1": {}, "ab": {"files": ["region_x/a.py"]}}, "")
        self.assertIn("cannot name this item", decisions["1"]["reason"])
        self.assertEqual(decisions["ab"]["reason"], "the request names this item")

    def test_turning_reuse_off_drops_earlier_reuse_records(self):
        cold = self.cold()
        self.revise("region_001", "region_007")
        second = self.follow(cold)
        self.write("regions/region_002.py", "# hand edit\n")
        third = self.follow(flow.load_run(self.state, second.run_id), text="region_001 only", reuse=False)
        self.assertEqual(third.status, "done", third.reason)
        self.assertNotIn("reuse", third.results["groups"]["items"]["group_00"])

    def test_a_trailing_double_star_input_names_the_files_below_it(self):
        state = flow._source_state(str(self.work), ["contracts/**"])
        self.assertEqual(state["count"], 8)
        (self.work / "empty").mkdir()
        self.assertEqual(flow._missing_files(str(self.work), ["empty/**", "contracts/**"]), ["empty/**"])

    def test_validation(self):
        bad = graph()
        bad["nodes"][0]["reuse"] = {"names": "{{item.regions}}"}
        self.assertIn("map 'groups' reuse needs inputs: the files each item's result depends on",
                      flow.validate_graph(bad))
        bad["nodes"][0]["reuse"] = {"inputs": "x", "extra": 1}
        self.assertIn("map 'groups' reuse must be an object with inputs and optional names",
                      flow.validate_graph(bad))


if __name__ == "__main__":
    unittest.main()
