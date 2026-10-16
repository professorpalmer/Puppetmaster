"""Builders see their gates; judges see what changed and which unit reviews hold.

Codex R138: every cold repair visit followed a review FAIL on a criterion the
builder could have checked itself, each re-review read every view again, and
the world review read all 144 regional views after 16 current unit reviews.
"""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

from puppetmaster import flow
from puppetmaster.flow import FlowRun, JobNodeExecutor

from tests.test_flow_runtime import Base, agent, graph, judge


def item_graph(**build):
    return graph([agent("build", files=["src/*.py"], **build),
                  {"id": "check", "kind": "shell", "command": "python3 check.py {{item.id}}"},
                  {"id": "render", "kind": "shell", "command": "python3 render.py"},
                  judge("review", task="Review the renders:\n{{out.render}}\nName failed criteria."),
                  {"id": "done", "kind": "end"}],
                 [{"from": "build", "to": "check"}, {"from": "check", "to": "render", "when": "ok"},
                  {"from": "check", "to": "build", "when": "fail"},
                  {"from": "render", "to": "review", "when": "ok"},
                  {"from": "review", "to": "done", "when": "PASS"},
                  {"from": "review", "to": "build", "when": "FAIL"}])


class GatePreviewTests(Base):
    def spec(self, g, node_id="build", visit=1):
        run = flow.new_run(self.state, g, cwd=str(self.work), state={"item": {"id": "group_00"}})
        node = next(n for n in run.graph["nodes"] if n["id"] == node_id)
        return JobNodeExecutor(self.state)._spec(node_id, node, run, "", visit)

    def test_the_builder_sees_its_checks_and_review_in_order(self):
        task = self.spec(item_graph()).instruction
        preview = task[task.index("After you finish"):]
        self.assertLess(preview.index("1. Command: python3 check.py group_00"),
                        preview.index("2. Command: python3 render.py"))
        self.assertLess(preview.index("2. Command"), preview.index("3. Review:"))
        self.assertIn("Name failed criteria.", preview)
        self.assertNotIn("{{out.render}}", preview)
        self.assertTrue(task.rstrip().endswith("what you could not finish or verify>`."))

    def test_a_build_with_no_gate_has_no_preview(self):
        task = self.spec(graph([agent("build"), {"id": "done", "kind": "end"}],
                               [{"from": "build", "to": "done"}])).instruction
        self.assertNotIn("After you finish", task)

    def test_preview_false_turns_it_off(self):
        self.assertNotIn("After you finish", self.spec(item_graph(preview=False)).instruction)

    def test_a_judge_gets_no_build_preview(self):
        self.assertNotIn("After you finish", self.spec(item_graph(), node_id="review").instruction)

    def test_preview_and_evidence_must_be_booleans(self):
        g = item_graph(preview="no")
        g["nodes"][3]["evidence"] = 1
        problems = flow.validate_graph(g)
        self.assertIn("node 'build' preview must be true or false", problems)
        self.assertIn("node 'review' evidence must be true or false", problems)


class ReviewDeltaTests(Base):
    def test_a_re_review_names_changed_and_unchanged_files(self):
        (self.work / "src").mkdir()
        for name in ("a.py", "b.py", "c.py"):
            (self.work / "src" / name).write_text("v1\n", encoding="utf-8")
        g = graph([agent("build", files=["src/*.py"]), judge("review")], [{"from": "build", "to": "review"}])
        run = flow.new_run(self.state, g, cwd=str(self.work))
        state = flow._source_state(str(self.work), ["src/*.py"])
        state["scope"] = ["src/*.py"]
        run.results["review"] = {"verdict": "FAIL", "reviewed": state}
        (self.work / "src" / "b.py").write_text("v2\n", encoding="utf-8")
        (self.work / "src" / "d.py").write_text("new\n", encoding="utf-8")
        node = run.graph["nodes"][1]
        text = JobNodeExecutor._delta(node, run, "", 2, "judge it", session={"run_id": "other"})
        self.assertIn("Changed since your last review: src/b.py, src/d.py.", text)
        self.assertIn("Unchanged since your last review: src/a.py, src/c.py.", text)
        self.assertTrue(text.endswith("judge it"))

    def test_no_recorded_review_keeps_the_plain_re_review(self):
        g = graph([agent("build"), judge("review")], [{"from": "build", "to": "review"}])
        run = flow.new_run(self.state, g, cwd=str(self.work))
        text = JobNodeExecutor._delta(run.graph["nodes"][1], run, "", 2, "judge it", session={"run_id": "x"})
        self.assertNotIn("Changed since", text)


class UnitEvidenceTests(Base):
    def world_run(self):
        (self.work / "regions").mkdir()
        (self.work / "regions" / "r0.py").write_text("v1\n", encoding="utf-8")
        (self.work / "regions" / "r1.py").write_text("v1\n", encoding="utf-8")
        g = graph([{"id": "groups", "kind": "map", "items": [{"id": "g0"}, {"id": "g1"}],
                    "graph": graph([agent("build", files=["regions/r{{index}}.py"])])},
                   judge("world", files=["regions/*.py"])], [{"from": "groups", "to": "world", "when": "ok"}])
        run = flow.new_run(self.state, g, cwd=str(self.work))
        items = {}
        for index in (0, 1):
            scope = [f"regions/r{index}.py"]
            state = flow._source_state(str(self.work), scope)
            items[f"g{index}"] = {"ok": True, "verdict": "PASS", "reviews": [
                {"node": "review", "verdict": "PASS",
                 "reviewed": {"digest": state["digest"], "count": state["count"], "scope": scope}}]}
        run.results["groups"] = {"items": items}
        return run

    def test_the_world_judge_sees_current_and_stale_unit_reviews(self):
        run = self.world_run()
        (self.work / "regions" / "r1.py").write_text("v2\n", encoding="utf-8")
        world = run.graph["nodes"][1]
        task = JobNodeExecutor(self.state)._spec("world", world, run, "", 1).instruction
        self.assertIn("- g0: review PASS on 1 files (regions/r0.py); current: no file changed since", task)
        self.assertIn("- g1: review PASS on 1 files (regions/r1.py); NOT current: files changed since", task)
        self.assertIn("how the units fit together", task)

    def test_evidence_false_turns_it_off(self):
        run = self.world_run()
        world = {**run.graph["nodes"][1], "evidence": False}
        task = JobNodeExecutor(self.state)._spec("world", world, run, "", 1).instruction
        self.assertNotIn("Unit reviews in this run", task)

    def test_an_item_result_carries_its_review_records(self):
        child = FlowRun(run_id="flow_child", graph={"nodes": []}, status="done")
        child.results = {"review": {"verdict": "PASS", "ok": True,
                                    "reviewed": {"digest": "d", "count": 3, "scope": ["x"], "hashes": {}}}}
        child.steps = [{"node": "review"}]
        result = flow._item_result(child)
        self.assertEqual(result["reviews"], [{"node": "review", "verdict": "PASS",
                                              "reviewed": {"digest": "d", "count": 3, "scope": ["x"]}}])


if __name__ == "__main__":
    import unittest
    unittest.main()
