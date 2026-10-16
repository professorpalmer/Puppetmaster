"""A flow revisit sends the resumed session its delta alone.

Codex R138 replica: a repair visit resumed the builder's thread, but the
adapter wrapped the flow's short revise text in the full build contract,
write scope and acceptance criteria again. The builder treated the message
as a new task: it read every contract, reran every check and inspected
every view again (226 s for one review FAIL; native fixed the same kind of
FAIL in about 35 s).
"""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import unittest
from pathlib import Path
from unittest.mock import patch

from puppetmaster.adapters import ClaudeCodeAdapter, CodexAdapter
from puppetmaster.models import Task

DELTA = "Repair only criterion 4.\n\nWhen you finish, end your reply with exactly one line: `VERDICT: PASS - x`."
CONTRACT = "Build mode: you are a full-edit worker"


def task(adapter: str, resumed: bool, **payload) -> Task:
    base = {"cwd": str(Path.cwd()), "disable_codegraph": True, "write_scope": ["src/a.py"],
            "resume_prompt": DELTA}
    if resumed:
        base["resume"] = {"status": "resolved", "adapter": adapter,
                          "session_id": "019a0000-0000-7000-8000-000000000001"}
    return Task(job_id="job-r", role="build", instruction="Build src/a.py.", adapter=adapter,
                payload={**base, **payload})


class CodexResumeDeltaTests(unittest.TestCase):
    def prompt(self, t: Task) -> str:
        with patch("puppetmaster.adapters.codex._lease_codex_thread", side_effect=lambda r: (r, None)), \
                patch("puppetmaster.adapters.enrich_prompt_with_codegraph", side_effect=lambda p, **_: (p, False)):
            prepared = CodexAdapter()._prepare_cli_invocation(t, "goal", "w", Path("."), "/usr/bin/codex")
        return prepared.extras["prompt"]

    def test_a_resumed_build_gets_the_delta_alone(self):
        self.assertEqual(self.prompt(task("codex", True)), DELTA)

    def test_a_resumed_review_gets_the_delta_alone(self):
        self.assertEqual(self.prompt(task("codex", True, sandbox="read-only", terminal_verdict=True)), DELTA)

    def test_a_fresh_start_keeps_the_contract(self):
        prompt = self.prompt(task("codex", False))
        self.assertIn(CONTRACT, prompt)
        self.assertIn("Build src/a.py.", prompt)
        self.assertNotIn("Repair only criterion 4.", prompt)


class ClaudeCodeResumeDeltaTests(unittest.TestCase):
    def prompt(self, t: Task) -> str:
        with patch("puppetmaster.adapters.enrich_prompt_with_codegraph", side_effect=lambda p, **_: (p, False)):
            prepared = ClaudeCodeAdapter()._prepare_cli_invocation(t, "goal", "w", Path("."), "/usr/bin/claude")
        return prepared.extras["prompt"]

    def test_a_resumed_build_gets_the_delta_alone(self):
        self.assertEqual(self.prompt(task("claude-code", True, permission_mode="acceptEdits")), DELTA)

    def test_a_fresh_start_keeps_the_contract(self):
        prompt = self.prompt(task("claude-code", False, permission_mode="acceptEdits"))
        self.assertIn(CONTRACT, prompt)


if __name__ == "__main__":
    unittest.main()
