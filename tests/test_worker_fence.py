"""Nested job start is refused when PUPPETMASTER_WORKER=1."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from puppetmaster.worker_fence import (
    ALLOW_NESTED_ENV,
    JOB_ID_ENV,
    TASK_ID_ENV,
    WORKER_ENV,
    WORKER_VALUE,
    is_worker_process,
    nested_start_blocked,
    stamp_worker_env,
)


class WorkerFenceTests(unittest.TestCase):
    def test_stamp_sets_worker_flag_and_ids(self) -> None:
        env = stamp_worker_env({}, job_id="job_a", task_id="task_b", role="explore")
        self.assertEqual(env[WORKER_ENV], WORKER_VALUE)
        self.assertEqual(env[JOB_ID_ENV], "job_a")
        self.assertEqual(env[TASK_ID_ENV], "task_b")
        self.assertEqual(env["PUPPETMASTER_ROLE"], "explore")

    def test_stamp_skips_blank_ids(self) -> None:
        env = stamp_worker_env({"PATH": "/bin"})
        self.assertEqual(env[WORKER_ENV], WORKER_VALUE)
        self.assertNotIn(JOB_ID_ENV, env)
        self.assertEqual(env["PATH"], "/bin")

    def test_swarm_blocked_only_inside_worker(self) -> None:
        self.assertIsNone(nested_start_blocked("swarm", {}))
        self.assertIsNone(nested_start_blocked("codegraph", {WORKER_ENV: WORKER_VALUE}))
        self.assertIsNone(nested_start_blocked("status", {WORKER_ENV: WORKER_VALUE}))
        self.assertIsNone(nested_start_blocked("artifacts", {WORKER_ENV: WORKER_VALUE}))
        msg = nested_start_blocked("swarm", {WORKER_ENV: WORKER_VALUE})
        self.assertIsNotNone(msg)
        self.assertIn("PUPPETMASTER_WORKER=1", msg)
        self.assertIn("PUPPETMASTER_ALLOW_NESTED=1", msg)

    def test_job_start_verbs_are_blocked(self) -> None:
        env = {WORKER_ENV: WORKER_VALUE}
        for command in ("run", "edit", "prewalk", "browser", "agentic", "rerun"):
            self.assertIsNotNone(nested_start_blocked(command, env), command)

    def test_allow_nested_override(self) -> None:
        env = {WORKER_ENV: WORKER_VALUE, ALLOW_NESTED_ENV: "1"}
        self.assertIsNone(nested_start_blocked("swarm", env))
        env[ALLOW_NESTED_ENV] = "true"
        self.assertIsNone(nested_start_blocked("run", env))

    def test_cli_swarm_exits_2_inside_worker(self) -> None:
        from puppetmaster.cli import main

        with patch.dict(os.environ, {WORKER_ENV: WORKER_VALUE}, clear=False):
            self.assertTrue(is_worker_process())
            buf = io.StringIO()
            with redirect_stderr(buf):
                code = main(["swarm", "should not launch"])
        self.assertEqual(code, 2)
        self.assertIn("nested job start refused", buf.getvalue())

    def test_inject_worker_cli_env_stamps_worker_flag(self) -> None:
        from puppetmaster.codegraph import inject_worker_cli_env

        env = inject_worker_cli_env({})
        self.assertEqual(env[WORKER_ENV], WORKER_VALUE)

    def test_mcp_start_cli_refuses_nested_swarm(self) -> None:
        from puppetmaster.mcp_server import start_cli

        with patch.dict(os.environ, {WORKER_ENV: WORKER_VALUE}, clear=False):
            result = start_cli(["swarm", "nope"], {"cwd": os.getcwd()})
        self.assertTrue(result.get("isError"))
        text = result["content"][0]["text"]
        self.assertIn("nested job start refused", text)

    def test_rules_name_the_env_flag(self) -> None:
        from puppetmaster.rules import RULE_BODY, render_agents_block, render_cursor_mdc

        for content in (RULE_BODY, render_cursor_mdc(), render_agents_block()):
            flattened = " ".join(content.split())
            self.assertIn("PUPPETMASTER_WORKER", flattened)
            self.assertLess(
                flattened.index("PUPPETMASTER_WORKER"),
                flattened.index("Delegate-first gate"),
            )

    def test_hand_maintained_rules_name_the_env_flag(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for path in (
            root / "AGENTS.md",
            root / ".cursor" / "rules" / "puppetmaster-workflow.mdc",
        ):
            self.assertIn("PUPPETMASTER_WORKER", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
