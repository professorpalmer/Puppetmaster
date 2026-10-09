"""Antigravity adapter parity with the Codex / Claude Code receipt contract."""
from __future__ import annotations

import json
import os
import sys
import unittest
from unittest import mock

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

from puppetmaster.adapters.antigravity import AntigravityAdapter
from puppetmaster.failure import NOT_AUTHENTICATED
from puppetmaster.models import ArtifactType, Task

CLEAN = {"sha": "abc1234", "changed_files": [], "untracked_files": [], "tree": "t1", "diff": ""}


def _completed(stdout: str, *, returncode=0, timed_out=False) -> mock.MagicMock:
    return mock.MagicMock(returncode=returncode, stdout=stdout, stderr="",
                          timed_out=timed_out, live_log_path="/tmp/live.log")


def _run(task: Task, stdout: str, **completed):
    with mock.patch("puppetmaster.adapters.resolve_command", return_value="/usr/local/bin/agy"), \
            mock.patch("puppetmaster.adapters.git_snapshot", return_value=CLEAN), \
            mock.patch("puppetmaster.adapters.worktree_guard", return_value=None), \
            mock.patch("puppetmaster.adapters.enrich_prompt_with_codegraph",
                       side_effect=lambda prompt, **_: (prompt, False)), \
            mock.patch("puppetmaster.job_brief.resolve_job_brief_for_task",
                       return_value="JOB-BRIEF-SECTION"), \
            mock.patch("puppetmaster.adapters.run_streamed_subprocess",
                       return_value=_completed(stdout, **completed)) as run:
        artifacts = AntigravityAdapter().run(task, goal="goal", worker_id="w1")
    return artifacts, run.call_args.kwargs


def _task(**payload) -> Task:
    return Task(job_id="job-agy", id="task-agy", role="implement", instruction="Build regions/a.py.",
                payload={"allow_dirty": True, "allow_non_worktree": True, **payload})


def _verification(artifacts):
    return next(a for a in artifacts if a.type == ArtifactType.VERIFICATION)


class WriteCapableReceiptTests(unittest.TestCase):
    """F40: the receipt names its write capability for quality.py."""

    def test_accept_edits_receipt_stamps_write_capable(self) -> None:
        artifacts, _ = _run(_task(mode="accept-edits"), json.dumps({"status": "SUCCESS", "response": "ok"}))
        self.assertIs(_verification(artifacts).payload["write_capable"], True)

    def test_plan_receipt_is_not_write_capable(self) -> None:
        artifacts, _ = _run(_task(mode="plan"), json.dumps({"status": "SUCCESS", "response": "ok"}))
        self.assertIs(_verification(artifacts).payload["write_capable"], False)

    def test_timeout_receipt_stamps_write_capable(self) -> None:
        artifacts, _ = _run(_task(mode="accept-edits"), "partial", returncode=None, timed_out=True)
        self.assertIs(_verification(artifacts).payload["write_capable"], True)


class BuildContractTests(unittest.TestCase):
    """F45: an accept-edits worker gets the build contract and no job-wide brief."""

    def _prompt(self, **payload) -> str:
        _, kwargs = _run(_task(**payload), json.dumps({"status": "SUCCESS", "response": "ok"}))
        return json.loads(kwargs["stdin_data"].strip())["message"]["content"]

    def test_builder_gets_the_build_contract_without_the_job_brief(self) -> None:
        prompt = self._prompt(mode="accept-edits")
        self.assertIn("Build mode", prompt)
        self.assertIn("VERDICT: PASS", prompt)
        self.assertNotIn("JOB-BRIEF-SECTION", prompt)

    def test_builder_without_codegraph_keeps_the_job_brief(self) -> None:
        self.assertIn("JOB-BRIEF-SECTION", self._prompt(mode="accept-edits", disable_codegraph=True))

    def test_plan_worker_keeps_the_report_contract_and_the_brief(self) -> None:
        prompt = self._prompt(mode="plan")
        self.assertNotIn("Build mode", prompt)
        self.assertIn("JOB-BRIEF-SECTION", prompt)


class UsagePresenceTests(unittest.TestCase):
    """F31: a missing or bad counter is NULL, never a measured 0."""

    def test_missing_usage_is_null_with_a_reason(self) -> None:
        artifacts, _ = _run(_task(mode="plan"), json.dumps({"status": "SUCCESS", "response": "ok"}))
        payload = _verification(artifacts).payload
        for field in ("tokens_in", "tokens_out", "tokens_total", "cached_input_tokens",
                      "reasoning_output_tokens"):
            self.assertIsNone(payload[field], field)
        self.assertEqual(payload["usage_unlinked_reason"], "usage_missing")

    def test_bad_counter_is_null_and_does_not_raise(self) -> None:
        stdout = json.dumps({"status": "SUCCESS", "response": "ok",
                             "usage": {"input_tokens": {"n": 1}, "output_tokens": 7}})
        payload = _verification(_run(_task(mode="plan"), stdout)[0]).payload
        self.assertIsNone(payload["tokens_in"])
        self.assertEqual(payload["tokens_out"], 7)
        self.assertIsNone(payload["tokens_total"])
        self.assertIsNone(payload["cached_input_tokens"])
        self.assertIsNone(payload["usage_unlinked_reason"])

    def test_known_counts_still_record(self) -> None:
        stdout = json.dumps({"status": "SUCCESS", "response": "ok", "usage": {
            "input_tokens": 10, "output_tokens": 2, "thinking_tokens": 1, "cache_read_tokens": 0}})
        payload = _verification(_run(_task(mode="plan"), stdout)[0]).payload
        self.assertEqual((payload["tokens_in"], payload["tokens_out"], payload["tokens_total"]), (10, 2, 12))
        self.assertEqual(payload["cached_input_tokens"], 0)
        self.assertEqual(payload["reasoning_output_tokens"], 1)


class TimeoutConversationTests(unittest.TestCase):
    """F26: a timeout receipt records the one conversation the partial stream shows."""

    @staticmethod
    def _stream(*events) -> str:
        return "\n".join(json.dumps(e) for e in events) + "\nnot json"

    def _timeout(self, stdout: str):
        artifacts, _ = _run(_task(mode="plan"), stdout, returncode=None, timed_out=True)
        return _verification(artifacts).payload

    def test_one_observed_conversation_is_recorded(self) -> None:
        payload = self._timeout(self._stream(
            {"event": "init", "init": {"conversation_id": "conv-1"}},
            {"event": "step_update", "conversation_id": "conv-1"}))
        self.assertEqual(payload["conversation_id"], "conv-1")
        self.assertEqual(payload["failure"], "timeout")

    def test_missing_or_ambiguous_conversations_record_none(self) -> None:
        for stdout in ("partial progress...",
                       self._stream({"event": "init", "init": {"conversation_id": "a"}},
                                    {"event": "step_update", "conversation_id": "b"})):
            with self.subTest(stdout=stdout[:20]):
                self.assertIsNone(self._timeout(stdout)["conversation_id"])


class EnvelopeDiagnosisTests(unittest.TestCase):
    """F59: a pretty-printed envelope diagnoses from its error fields only."""

    CHATTER = "Fixed the 401 handler and added an api key check."

    def test_indented_envelope_response_is_not_a_diagnosis(self) -> None:
        stdout = json.dumps({"status": "ERROR", "response": self.CHATTER}, indent=2)
        payload = _verification(_run(_task(mode="plan"), stdout, returncode=1)[0]).payload
        self.assertNotEqual(payload["failure"], NOT_AUTHENTICATED)

    def test_indented_envelope_error_still_diagnoses(self) -> None:
        stdout = json.dumps({"status": "ERROR", "error": "not authenticated"}, indent=2)
        payload = _verification(_run(_task(mode="plan"), stdout, returncode=1)[0]).payload
        self.assertEqual(payload["failure"], NOT_AUTHENTICATED)


if __name__ == "__main__":
    unittest.main()
