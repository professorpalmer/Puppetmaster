"""Parity fixes for the fx adapter: build contract, usage presence, resume record.

Hermetic: the subprocess, git snapshots and capture are patched through the
``puppetmaster.adapters`` facade, like tests/test_fx_adapter.py.
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

from puppetmaster.adapters._streaming import StreamedProcess
from puppetmaster.adapters.fx import FxAdapter
from puppetmaster.models import ArtifactType, Task

CLEAN = {"sha": "base000", "tree": "tree-base", "changed_files": [], "untracked_files": [], "diff": ""}


def _task(**payload) -> Task:
    return Task(job_id="job_fx", role="implement", instruction="Build regions/a.py.",
                adapter="fx", payload={"cwd": "/tmp", **payload})


def _run(task: Task, stdout: str, *, timed_out: bool = False):
    completed = StreamedProcess(returncode=None if timed_out else 0, stdout=stdout, stderr="",
                                timed_out=timed_out, live_log_path=None)
    with mock.patch("puppetmaster.adapters.resolve_command", return_value="/usr/bin/fx"), \
         mock.patch("puppetmaster.adapters.worktree_guard", return_value=None), \
         mock.patch("puppetmaster.adapters.git_snapshot", side_effect=[CLEAN, CLEAN]), \
         mock.patch("puppetmaster.adapters.run_streamed_subprocess", return_value=completed) as run, \
         mock.patch("puppetmaster.adapters.enrich_prompt_with_codegraph",
                    side_effect=lambda prompt, **_: (prompt + "\n[codegraph]", True)), \
         mock.patch("puppetmaster.adapters.with_repo_census",
                    side_effect=lambda prompt, cwd: prompt + "\n[census]") as census, \
         mock.patch("puppetmaster.adapters.fx.capture_subprocess_stdout", side_effect=lambda **kw: None):
        artifacts = FxAdapter().run(task, "goal", "worker")
    return artifacts, run.call_args.kwargs, census


def _receipt(artifacts):
    return next(a for a in artifacts if a.type == ArtifactType.VERIFICATION).payload


class FxBuildContractTests(unittest.TestCase):
    """F43: a write-capable fx worker gets the build contract, not the findings contract."""

    def test_write_capable_worker_gets_the_build_contract(self) -> None:
        _, kwargs, census = _run(_task(), '{"final_output":"done"}')
        prompt = kwargs["stdin_data"]
        self.assertIn("Build mode", prompt)
        self.assertIn("VERDICT: PASS", prompt)
        self.assertNotIn("submit_findings", prompt)
        census.assert_not_called()

    def test_read_only_worker_keeps_the_analysis_contract(self) -> None:
        _, kwargs, census = _run(_task(read_only=True), '{"final_output":"done"}')
        self.assertIn("submit_findings", kwargs["stdin_data"])
        self.assertNotIn("Build mode", kwargs["stdin_data"])
        census.assert_called_once()

    def test_builder_skips_the_job_brief_only_when_it_gets_task_codegraph(self) -> None:
        with mock.patch("puppetmaster.job_brief.resolve_job_brief_for_task", return_value="JOB-BRIEF-SECTION"):
            with_graph = _run(_task(), "{}")[1]["stdin_data"]
            without_graph = _run(_task(disable_codegraph=True), "{}")[1]["stdin_data"]
            analysis = _run(_task(read_only=True), "{}")[1]["stdin_data"]
        self.assertNotIn("JOB-BRIEF-SECTION", with_graph)
        self.assertIn("JOB-BRIEF-SECTION", without_graph)
        self.assertIn("JOB-BRIEF-SECTION", analysis)

    def test_build_report_survives_as_a_finding(self) -> None:
        artifacts, _, _ = _run(
            _task(), '{"final_output":"Built regions/a.py.\\nVERDICT: PASS - checks pass"}'
        )
        self.assertEqual(_receipt(artifacts)["result"], "passed")
        self.assertIn(ArtifactType.FINDING, [a.type for a in artifacts])


class FxUsagePresenceTests(unittest.TestCase):
    """F32: an unknown count is NULL, never a measured zero."""

    def test_a_result_without_usage_records_null_counts(self) -> None:
        receipt = _receipt(_run(_task(), '{"final_output":"ok","session_id":"s1"}')[0])
        self.assertIsNone(receipt["tokens_in"])
        self.assertIsNone(receipt["tokens_out"])
        self.assertIsNone(receipt["tokens_total"])
        self.assertFalse(receipt["usage_reported"])

    def test_a_missing_result_records_null_counts(self) -> None:
        receipt = _receipt(_run(_task(), "")[0])
        self.assertTrue(receipt["result_missing"])
        self.assertIsNone(receipt["tokens_in"])
        self.assertFalse(receipt["usage_reported"])

    def test_one_missing_counter_stays_null(self) -> None:
        receipt = _receipt(_run(_task(), '{"final_output":"ok","usage":{"input_tokens":7}}')[0])
        self.assertEqual((receipt["tokens_in"], receipt["tokens_out"]), (7, None))
        self.assertIsNone(receipt["tokens_total"])
        self.assertTrue(receipt["usage_reported"])

    def test_reported_usage_is_kept(self) -> None:
        receipt = _receipt(_run(_task(), '{"usage":{"input_tokens":7,"output_tokens":2}}')[0])
        self.assertEqual((receipt["tokens_in"], receipt["tokens_out"], receipt["tokens_total"]), (7, 2, 9))


class FxWriteCapableStampTests(unittest.TestCase):
    """F41: both fx receipts say whether the run could write."""

    def test_success_and_timeout_receipts_stamp_write_capable(self) -> None:
        self.assertIs(_receipt(_run(_task(), "{}")[0])["write_capable"], True)
        self.assertIs(_receipt(_run(_task(), "", timed_out=True)[0])["write_capable"], True)
        self.assertIs(_receipt(_run(_task(read_only=True), "", timed_out=True)[0])["write_capable"], False)


class FxResumeRecordTests(unittest.TestCase):
    """F24: fx obeys the stamped resume record and keeps a flow session."""

    UNAVAILABLE = {"status": "unavailable", "reason": "adapter 'fx' does not support session resume"}

    def test_a_stamped_unavailable_record_runs_fresh_and_says_why(self) -> None:
        artifacts, kwargs, _ = _run(_task(resume_session_id="S", resume=self.UNAVAILABLE), "{}")
        self.assertNotIn("--resume-id", kwargs["command"])
        receipt = _receipt(artifacts)
        self.assertIsNone(receipt["resume_session_id"])
        self.assertEqual(receipt["resume"], self.UNAVAILABLE)

    def test_a_stamped_resolved_record_resumes_its_session(self) -> None:
        record = {"status": "resolved", "adapter": "fx", "session_id": "S2"}
        artifacts, kwargs, _ = _run(_task(resume=record), "", timed_out=True)
        command = kwargs["command"]
        self.assertEqual(command[command.index("--resume-id") + 1], "S2")
        receipt = _receipt(artifacts)
        self.assertEqual((receipt["session_id"], receipt["resume"]), ("S2", record))

    def test_a_direct_caller_id_still_resumes(self) -> None:
        _, kwargs, _ = _run(_task(resume_session_id="S3"), "{}")
        self.assertIn("S3", kwargs["command"])

    def test_a_non_ephemeral_node_saves_its_session(self) -> None:
        self.assertNotIn("--no-save", _run(_task(ephemeral=False), "{}")[1]["command"])
        self.assertIn("--no-save", _run(_task(), "{}")[1]["command"])


if __name__ == "__main__":
    unittest.main()
