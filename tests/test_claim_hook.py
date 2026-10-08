"""Pilot claim hook: a pilot edit during a flow walk becomes a pilot claim."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import io
import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from puppetmaster import claim_hook, flow
from puppetmaster.flow import NodeOutcome
from puppetmaster.hook_installers import (
    merge_hook_maps,
    render_claude_hooks,
    render_codex_hooks,
    render_cursor_hooks,
    strip_hook_maps,
)
from puppetmaster.state import walking_runs_dir

GRAPH = {"id": "regions", "entry": "build", "defaults": {"adapter": "codex"},
         "nodes": [{"id": "build", "kind": "agent", "task": "build"}, {"id": "z", "kind": "end"}],
         "edges": [{"from": "build", "to": "z"}]}


def claude_write(path: Path, cwd: Path) -> dict:
    return {"hook_event_name": "PostToolUse", "tool_name": "Write", "cwd": str(cwd),
            "tool_input": {"file_path": str(path), "content": "x"}}


def codex_patch(cwd: Path, *names: str) -> dict:
    body = "\n".join(f"*** Add File: {name}\n+x" for name in names)
    return {"hook_event_name": "PostToolUse", "tool_name": "apply_patch", "cwd": str(cwd),
            "tool_input": {"command": f"*** Begin Patch\n{body}\n*** End Patch"}}


def hook(payload: dict, env=None) -> None:
    claim_hook.run(stdin=io.StringIO(json.dumps(payload)), env=env if env is not None else {})


class EditedPathTests(unittest.TestCase):
    def test_each_host_payload_names_its_paths(self):
        cwd = Path(os.path.abspath("/work/repo"))
        self.assertEqual(claim_hook.edited_paths(claude_write(cwd / "a.py", cwd)), [cwd / "a.py"])
        notebook = {"cwd": str(cwd), "tool_input": {"notebook_path": "nb.ipynb"}}
        self.assertEqual(claim_hook.edited_paths(notebook), [cwd / "nb.ipynb"])
        patch_text = ("*** Begin Patch\n*** Add File: new.py\n+x\n*** Update File: old.py\n"
                      "*** Move to: moved.py\n@@\n-a\n+b\n*** Delete File: gone.py\n*** End Patch")
        codex = {"cwd": str(cwd), "tool_input": {"command": patch_text}}
        self.assertEqual(claim_hook.edited_paths(codex),
                         [cwd / "new.py", cwd / "old.py", cwd / "moved.py", cwd / "gone.py"])
        cursor = {"hook_event_name": "afterFileEdit", "file_path": str(cwd / "c.py"), "edits": []}
        self.assertEqual(claim_hook.edited_paths(cursor), [cwd / "c.py"])

    def test_a_shell_command_is_not_an_edit(self):
        bash = {"cwd": "/w", "tool_input": {"command": "python render.py > out.json"}}
        self.assertEqual(claim_hook.edited_paths(bash), [])


class ClaimHookTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.state = Path(tmp.name) / "state"
        self.repo = (Path(tmp.name) / "repo").resolve()
        (self.repo / ".git").mkdir(parents=True)
        self.outside = Path(tmp.name) / "elsewhere.py"

    def walk_with(self, during):
        """Walk one agent node; ``during`` runs while the node works (a pilot edit)."""
        run = flow.new_run(self.state, GRAPH, "goal", cwd=str(self.repo))

        def execute(node, current, prev):
            during(current)
            return NodeOutcome(ok=True, output="built")

        done = flow.walk(self.state, run.run_id, execute=execute)
        self.assertEqual(done.status, "done")
        return run.run_id

    def test_a_pilot_write_during_a_walk_is_claimed_and_the_index_is_cleared(self):
        def during(run):
            self.assertTrue((walking_runs_dir() / f"{run.run_id}.json").is_file())
            hook(claude_write(self.repo / "preview_world.py", self.repo))
            hook(codex_patch(self.repo, "preliminary-validation.json", "renders/integrated/top.png"))
            hook(claude_write(self.outside, self.repo))

        run_id = self.walk_with(during)
        self.assertEqual(flow.pilot_claims(self.state, run_id),
                         ["preliminary-validation.json", "preview_world.py", "renders/integrated/top.png"])
        self.assertFalse((walking_runs_dir() / f"{run_id}.json").exists())

    def test_a_worker_edit_is_never_claimed(self):
        def during(run):
            hook(claude_write(self.repo / "preview_world.py", self.repo), env={"PUPPETMASTER_WORKER": "1"})
            with patch.dict(os.environ, {"PUPPETMASTER_WORKER": "1"}):
                claim_hook.run(stdin=io.StringIO(json.dumps(claude_write(self.repo / "b.py", self.repo))))

        self.assertEqual(flow.pilot_claims(self.state, self.walk_with(during)), [])

    def test_no_walk_means_no_claim_and_a_stale_index_entry_is_removed(self):
        run = flow.new_run(self.state, GRAPH, "goal", cwd=str(self.repo))
        hook(claude_write(self.repo / "a.py", self.repo))
        self.assertEqual(flow.pilot_claims(self.state, run.run_id), [])

        # A walker that crashed leaves its entry; the run is no longer running.
        flow._mark_walking(self.state, run)
        stored = flow.load_run(self.state, run.run_id)
        stored.status = "interrupted"
        flow.save_run(self.state, stored)
        hook(claude_write(self.repo / "a.py", self.repo))
        self.assertEqual(flow.pilot_claims(self.state, run.run_id), [])
        self.assertFalse((walking_runs_dir() / f"{run.run_id}.json").exists())

    def test_bad_input_never_fails_the_host(self):
        self.assertEqual(claim_hook.run(stdin=io.StringIO("not json"), env={}), 0)
        self.assertEqual(claim_hook.run(stdin=io.StringIO("[]"), env={}), 0)

    def test_with_no_walk_the_hook_stays_light(self):
        code = ("import io, sys; from puppetmaster import claim_hook; "
                "claim_hook.run(stdin=io.StringIO('{\"tool_input\": {\"file_path\": \"/x/a.py\"}}'), env={}); "
                "print(sorted(m for m in sys.modules if m.startswith('puppetmaster')))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                             env={**os.environ, "PUPPETMASTER_APP_STATE_ROOT": str(self.state / "app")})
        self.assertIn("puppetmaster.claim_hook", out.stdout)
        self.assertNotIn("puppetmaster.flow", out.stdout)


class InstallTests(unittest.TestCase):
    def test_each_host_gets_the_claim_hook_and_strip_removes_it(self):
        def commands(entries):
            return json.dumps(entries)

        claude = render_claude_hooks("/py")
        edit = [e for e in claude["PostToolUse"] if e["matcher"] == "Write|Edit|MultiEdit|NotebookEdit"]
        self.assertEqual(edit[0]["hooks"][0]["command"], "/py -m puppetmaster.claim_hook")
        codex = render_codex_hooks("/py")
        self.assertIn("apply_patch", [e["matcher"] for e in codex["PostToolUse"]])
        self.assertIn("puppetmaster.claim_hook", commands(render_cursor_hooks("/py")["afterFileEdit"]))

        user = {"matcher": "Write", "hooks": [{"type": "command", "command": "my-formatter"}]}
        merged, changed = merge_hook_maps({"hooks": {"PostToolUse": [user]}}, claude)
        self.assertTrue(changed)
        again, changed = merge_hook_maps(merged, claude)
        self.assertFalse(changed)
        stripped, _ = strip_hook_maps(again)
        self.assertEqual(stripped["hooks"]["PostToolUse"], [user])


if __name__ == "__main__":
    unittest.main()
