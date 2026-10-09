"""Cursor SDK and Hermes adapters match the shared CLI worker contract."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401,E402

from puppetmaster.adapters._streaming import StreamedProcess  # noqa: E402
from puppetmaster.adapters.cursor import CursorAdapter  # noqa: E402
from puppetmaster.adapters.hermes import HermesAdapter  # noqa: E402
from puppetmaster.models import ArtifactType, Task  # noqa: E402

CLEAN = {"sha": "s", "changed_files": [], "untracked_files": [], "diff": ""}
REPORT = json.dumps({"artifacts": [{"type": "finding", "claim": "Inspected", "evidence": ["a.py:1"]}]})
CURSOR_STDOUT = json.dumps({"status": "finished", "result": REPORT, "usage": {"inputTokens": 5, "outputTokens": 3}})
RESUME = {"status": "unavailable", "adapter": "cursor", "reason": "cursor workers never resume"}


def _run(adapter, task: Task, stdout: str, **streamed):
    completed = StreamedProcess(
        returncode=streamed.pop("returncode", 0), stdout=stdout, stderr="", **streamed
    )
    with patch("puppetmaster.adapters.resolve_command", side_effect=lambda name: f"/usr/bin/{name}"), patch(
        "puppetmaster.adapters.worktree_guard", return_value=None
    ), patch("puppetmaster.adapters.git_snapshot", side_effect=[CLEAN, CLEAN]), patch(
        "puppetmaster.adapters.run_streamed_subprocess", return_value=completed
    ) as run, patch(
        "puppetmaster.adapters.enrich_prompt_with_codegraph", side_effect=lambda prompt, **_: (prompt, False)
    ), patch(
        "puppetmaster.adapters.with_repo_census", side_effect=lambda prompt, cwd: prompt
    ), patch(
        "puppetmaster.adapters.prune_hermes_tool_sessions", return_value=None
    ), patch("puppetmaster.adapters.cursor.cursor_runner", return_value=Path("/runner.mjs")):
        artifacts = adapter.run(task, "goal", "worker")
    return artifacts, run


def _task(adapter: str, implement: bool, **payload) -> Task:
    base = {"cwd": str(Path.cwd()), "allow_dirty": True, "analyze_retry": False}
    if implement:
        base["mode"] = "implement"
    return Task(job_id="job", id="task", role="builder" if implement else "explore",
                instruction="Build a.py.", adapter=adapter, payload={**base, **payload})


CASES = (
    (CursorAdapter, "cursor", CURSOR_STDOUT),
    (HermesAdapter, "hermes", REPORT),
)


def _receipt(artifacts):
    return next(a for a in artifacts if a.type == ArtifactType.VERIFICATION).payload


class OutputLimitTests(unittest.TestCase):
    """F9, F19, F20, F21, F22: every Cursor and Hermes path enforces max_output_bytes."""

    def test_cap_reaches_the_subprocess(self) -> None:
        for adapter_type, name, stdout in CASES:
            for implement in (True, False):
                with self.subTest(adapter=name, implement=implement):
                    _, run = _run(adapter_type(), _task(name, implement, max_output_bytes=200000), stdout)
                    self.assertEqual(run.call_args.kwargs.get("max_output_bytes"), 200000)

    def test_cap_hit_is_a_blocked_receipt_not_a_timeout(self) -> None:
        for adapter_type, name, stdout in CASES:
            for implement in (True, False):
                with self.subTest(adapter=name, implement=implement):
                    artifacts, run = _run(
                        adapter_type(), _task(name, implement, max_output_bytes=100), stdout,
                        returncode=None, timed_out=True, output_limit_hit=True,
                    )
                    receipt = _receipt(artifacts)
                    self.assertEqual(receipt["result"], "blocked")
                    self.assertEqual(receipt["failure"], "runtime_budget_exceeded")
                    self.assertEqual(receipt["max_output_bytes"], 100)
                    self.assertIn("tool output", receipt["counted_stream"])
                    self.assertEqual(run.call_count, 1)

    def test_cursor_analyze_receipt_names_its_capture(self) -> None:
        artifacts, _ = _run(CursorAdapter(), _task("cursor", False), CURSOR_STDOUT)
        receipt = _receipt(artifacts)
        for field in ("live_log", "attempt_id", "dispatch_receipt"):
            self.assertIn(field, receipt)


class WriteCapableTests(unittest.TestCase):
    """F38, F39: implement receipts say they could write; analysis receipts do not."""

    def test_implement_receipts_stamp_write_capable(self) -> None:
        for adapter_type, name, stdout in CASES:
            for timed_out in (False, True):
                with self.subTest(adapter=name, timed_out=timed_out):
                    artifacts, _ = _run(adapter_type(), _task(name, True), stdout,
                                        returncode=None if timed_out else 0, timed_out=timed_out)
                    self.assertIs(_receipt(artifacts).get("write_capable"), True)

    def test_analysis_receipts_do_not(self) -> None:
        for adapter_type, name, stdout in CASES:
            with self.subTest(adapter=name):
                artifacts, _ = _run(adapter_type(), _task(name, False), stdout)
                self.assertNotIn("write_capable", _receipt(artifacts))


class BuildContractTests(unittest.TestCase):
    """F46: the build contract names no tool that Cursor SDK or Hermes lacks."""

    def test_implement_prompt_is_the_cli_build_contract(self) -> None:
        for adapter_type, name, stdout in CASES:
            with self.subTest(adapter=name):
                _, run = _run(adapter_type(), _task(name, True), stdout)
                kwargs = run.call_args.kwargs
                if name == "cursor":
                    prompt = json.loads(kwargs["env"]["PUPPETMASTER_CURSOR_INPUT"])["prompt"]
                else:
                    command = kwargs["command"]
                    prompt = command[command.index("-q") + 1]
                self.assertIn("Build mode", prompt)
                self.assertIn("VERDICT: PASS", prompt)
                for tool in ("update_plan", "apply_hashline", "run_terminal", "submit_report"):
                    self.assertNotIn(tool, prompt)
                self.assertIn("Your task:\nBuild a.py.", prompt)


class ResumeRecordTests(unittest.TestCase):
    """F25: a fresh run records why a requested resume did not happen."""

    def test_receipts_carry_the_resume_record(self) -> None:
        for adapter_type, name, stdout in CASES:
            for implement in (True, False):
                for timed_out in (False, True):
                    with self.subTest(adapter=name, implement=implement, timed_out=timed_out):
                        artifacts, _ = _run(adapter_type(), _task(name, implement, resume=dict(RESUME)), stdout,
                                            returncode=None if timed_out else 0, timed_out=timed_out)
                        self.assertEqual(_receipt(artifacts).get("resume"), RESUME)

    def test_no_resume_record_without_a_request(self) -> None:
        for adapter_type, name, stdout in CASES:
            with self.subTest(adapter=name):
                artifacts, _ = _run(adapter_type(), _task(name, True), stdout)
                self.assertNotIn("resume", _receipt(artifacts))


class SdkUsagePresenceTests(unittest.TestCase):
    """F35: a missing SDK count stays unknown, never a measured zero."""

    def test_missing_side_is_none(self) -> None:
        from puppetmaster.usage import token_usage, usage_from_sdk

        self.assertEqual(usage_from_sdk({"outputTokens": 800}), {"tokens_in": None, "tokens_out": 800})
        record = token_usage(sdk_usage={"outputTokens": 800}, prompt_text="x" * 400, output_text="y")
        self.assertIsNone(record["tokens_in"])
        self.assertEqual(record["tokens_out"], 800)
        self.assertIsNone(record["selected_facts"]["tokens_in"])

    def test_missing_side_is_unpriced(self) -> None:
        from puppetmaster.models import Artifact
        from puppetmaster.cost import price_job
        from puppetmaster.model_registry import ModelSpec
        from puppetmaster.usage import token_usage

        payload = dict(token_usage(sdk_usage={"outputTokens": 800}), model="m")
        artifact = Artifact(job_id="j", task_id="t", type=ArtifactType.VERIFICATION, created_by="test",
                            confidence=1.0, evidence=[], payload=payload)
        spec = ModelSpec(id="cursor/m", adapter="cursor", adapter_model_name="m", billing="api",
                         input_per_mtok_usd=1, output_per_mtok_usd=2)
        task = price_job([artifact], [spec]).tasks[0]
        self.assertFalse(task.priced)
        self.assertEqual(task.usage_unknown, ["tokens_in"])


if __name__ == "__main__":
    unittest.main()
