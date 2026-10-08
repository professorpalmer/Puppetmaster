"""Operator default for Codex worker retention (PUPPETMASTER_CODEX_EPHEMERAL)."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from puppetmaster.adapters import CodexAdapter
from puppetmaster.adapters._streaming import StreamedProcess
from puppetmaster.adapters.codex import CODEX_EPHEMERAL_ENV, worker_ephemeral
from puppetmaster.models import Task

CLEAN = {"sha": "a" * 40, "tree": "b" * 40, "changed_files": [], "untracked_files": [], "diff": ""}


class PrecedenceTests(unittest.TestCase):
    def test_each_rule_in_order(self):
        off = {CODEX_EPHEMERAL_ENV: "0"}
        on = {CODEX_EPHEMERAL_ENV: "1"}
        cases = [
            ({}, False, {}, (True, "default")),
            ({}, False, off, (False, "operator_default")),
            ({}, False, on, (True, "operator_default")),
            ({"ephemeral": True}, False, off, (True, "payload")),
            ({"ephemeral": False}, False, on, (False, "payload")),
            ({"review_loop": True}, False, on, (False, "review_loop")),
            ({"ephemeral": True}, True, on, (False, "resume")),
        ]
        for payload, resumed, env, expected in cases:
            with self.subTest(payload=payload, resumed=resumed, env=env):
                self.assertEqual(worker_ephemeral(payload, resumed=resumed, env=env), expected)

    def test_a_bad_value_is_refused_not_ignored(self):
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            worker_ephemeral({}, resumed=False, env={CODEX_EPHEMERAL_ENV: "keep"})


class AdapterRunTests(unittest.TestCase):
    """An unpinned start (no model, no registry defaults) honors the operator default."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name) / "codex"
        self.home.mkdir()

    def run_task(self, env, **payload):
        task = Task(job_id="job-e", role="audit", instruction="Look.", adapter="codex",
                    payload={"cwd": str(Path.cwd()), "sandbox": "read-only",
                             "disable_codegraph": True, **payload})
        argv = []

        def launch(command, *_args, **_kwargs):
            argv.extend(command)
            out = "\n".join(json.dumps(e) for e in (
                {"type": "thread.started", "thread_id": "019a0000-0000-7000-8000-000000000001"},
                {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}}))
            return StreamedProcess(returncode=0, stdout=out, stderr="", timed_out=False)

        base = {"CODEX_HOME": str(self.home), "PUPPETMASTER_CODEX_LEAN_HOME": "0",
                "PUPPETMASTER_HOME": str(self.home / "pm")}
        with patch.dict("os.environ", {**base, **env}), \
                patch("puppetmaster.adapters.resolve_command", side_effect=lambda n: f"/usr/bin/{n}"), \
                patch("puppetmaster.adapters.worktree_guard", return_value=None), \
                patch("puppetmaster.adapters.git_snapshot", return_value=CLEAN), \
                patch("puppetmaster.adapters.run_streamed_subprocess", side_effect=launch):
            artifacts = CodexAdapter().run(task, "goal", "worker")
        receipt = next(a.payload for a in artifacts if "ephemeral_source" in a.payload)
        return argv, receipt

    def test_operator_zero_keeps_an_unpinned_session(self):
        argv, receipt = self.run_task({CODEX_EPHEMERAL_ENV: "0"})
        self.assertNotIn("--ephemeral", argv)
        self.assertEqual((receipt["ephemeral"], receipt["ephemeral_source"]), (False, "operator_default"))

    def test_without_the_setting_the_default_is_unchanged(self):
        argv, receipt = self.run_task({})
        self.assertIn("--ephemeral", argv)
        self.assertEqual((receipt["ephemeral"], receipt["ephemeral_source"]), (True, "default"))

    def test_an_explicit_caller_value_beats_the_operator(self):
        argv, receipt = self.run_task({CODEX_EPHEMERAL_ENV: "0"}, ephemeral=True)
        self.assertIn("--ephemeral", argv)
        self.assertEqual(receipt["ephemeral_source"], "payload")

    def test_the_app_server_path_starts_the_thread_it_reports(self):
        task = Task(job_id="job-e", role="audit", instruction="Look.", adapter="codex",
                    payload={"cwd": str(Path.cwd()), "native_steer": True, "disable_codegraph": True})
        with patch.dict("os.environ", {CODEX_EPHEMERAL_ENV: "0"}), \
                patch("puppetmaster.adapters.enrich_prompt_with_codegraph", side_effect=lambda p, **_: (p, False)), \
                patch("puppetmaster.adapters.with_repo_census", side_effect=lambda p, cwd: p):
            prepared = CodexAdapter()._prepare_cli_invocation(task, "goal", "w", Path("."), "/usr/bin/codex")
        seen = {}

        def session(**kwargs):
            seen.update(kwargs)
            return SimpleNamespace(messages=["done"], status="completed", error="")

        with patch("puppetmaster.adapters.codex_session.run_codex_session", side_effect=session):
            CodexAdapter()._invoke_cli(task, prepared, Path("."), 30)
        self.assertIs(seen["ephemeral"], False)
        self.assertEqual(prepared.extras["ephemeral_source"], "operator_default")


if __name__ == "__main__":
    unittest.main()
