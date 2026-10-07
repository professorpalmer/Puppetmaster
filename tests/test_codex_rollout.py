"""Attempt-local Codex usage from the session rollout (cold, resume, compaction)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from puppetmaster.adapters import CodexAdapter
from puppetmaster.adapters import codex_rollout
from puppetmaster.adapters._streaming import StreamedProcess
from puppetmaster.adapters.codex import _attempt_accounting
from puppetmaster.invocation import Invocation
from puppetmaster.models import Task

SID = "0199a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b"
CLEAN = {"sha": "s", "changed_files": [], "untracked_files": [], "diff": ""}


def _usage(inp, cached, out, reasoning=0, write=0):
    return {"input_tokens": inp, "cached_input_tokens": cached, "cache_write_input_tokens": write,
            "output_tokens": out, "reasoning_output_tokens": reasoning}


def _started(turn):
    return {"type": "event_msg", "payload": {"type": "task_started", "turn_id": turn}}


def _record(turn, response, usage, turn_snapshot=None, thread_snapshot=None):
    payload = {"turn_id": turn, "response_id": response, "usage": usage}
    if turn_snapshot is not None:
        payload["turn_token_usage"] = turn_snapshot
    if thread_snapshot is not None:
        payload["thread_token_usage"] = thread_snapshot
    return {"type": "token_usage_record", "payload": payload}


def _compacted():
    return {"type": "compacted", "payload": {"message": "summary"}}


class RolloutFixture(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        day = self.home / "sessions" / "2026" / "10" / "06"
        day.mkdir(parents=True)
        self.path = day / f"rollout-2026-10-06T15-25-21-{SID}.jsonl"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write(self, *records) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")


# Turn 1 (cold): two requests. Turn 2 (resume): a request, a compaction, then
# two more. The SDK's resumed turn.completed reports the session total minus
# the pre-compaction request, so neither it nor its difference is exact.
COLD = [_started("t1"), _record("t1", "r1", _usage(1000, 800, 50, 10)),
        _record("t1", "r2", _usage(1200, 1000, 60, 20), turn_snapshot=_usage(2200, 1800, 110, 30))]
RESUME = [_started("t2"), _record("t2", "r3", _usage(1500, 1200, 70, 5)), _compacted(),
          _record("t2", "r4", _usage(400, 0, 30, 3)),
          _record("t2", "r5", _usage(600, 300, 40, 2), turn_snapshot=_usage(2500, 1500, 140, 10))]
RESUME_EXACT = _usage(2500, 1500, 140, 10)


class AttemptUsageTests(RolloutFixture):
    def test_cold_turn_is_exact_unique_request_sum(self) -> None:
        self.write(*COLD)
        found = codex_rollout.attempt_usage(self.path, frozenset())
        self.assertEqual(found["usage_scope"], "attempt")
        self.assertEqual(found["rollout_turn_id"], "t1")
        self.assertEqual(found["rollout_request_count"], 2)
        self.assertEqual(found["usage"], _usage(2200, 1800, 110, 30))
        self.assertEqual(found["usage_conflicts"], [])
        self.assertEqual(found["usage_partial_fields"], [])

    def test_resume_with_compaction_counts_only_new_turn_including_pre_compaction_request(self) -> None:
        self.write(*COLD)
        baseline = codex_rollout.turn_ids(self.path)
        self.write(*RESUME)
        found = codex_rollout.attempt_usage(self.path, baseline)
        self.assertEqual(found["rollout_turn_id"], "t2")
        self.assertEqual(found["rollout_request_count"], 3)
        self.assertEqual(found["usage"], RESUME_EXACT)
        self.assertEqual(found["usage_conflicts"], [])

    def test_replayed_response_id_is_one_request(self) -> None:
        self.write(*COLD, _record("t1", "r2", _usage(1200, 1000, 60, 20)))
        found = codex_rollout.attempt_usage(self.path, frozenset())
        self.assertEqual(found["rollout_request_count"], 2)
        self.assertEqual(found["usage"]["input_tokens"], 2200)

    def test_conflicting_replay_is_disputed_in_either_order(self) -> None:
        first, changed = _usage(1200, 1000, 60, 20), _usage(1300, 1000, 60, 25)
        found = []
        for order in ((first, changed), (changed, first)):
            self.path.unlink(missing_ok=True)
            self.write(_started("t1"), _record("t1", "r1", _usage(1000, 800, 50, 10)),
                       *(_record("t1", "r2", usage) for usage in order))
            found.append(codex_rollout.attempt_usage(self.path, frozenset()))
        self.assertEqual(found[0], found[1])
        usage = found[0]["usage"]
        self.assertIsNone(usage["input_tokens"])
        self.assertIsNone(usage["reasoning_output_tokens"])
        self.assertEqual((usage["cached_input_tokens"], usage["output_tokens"]), (1800, 110))
        self.assertEqual(found[0]["usage_disputed_fields"], ["input_tokens", "reasoning_output_tokens"])
        self.assertIn("response_id_replay_conflict", found[0]["usage_conflicts"])
        self.assertEqual(found[0]["rollout_request_count"], 2)

    def test_same_response_id_in_another_turn_is_not_attempt_local(self) -> None:
        self.write(*COLD, _started("t2"), _record("t2", "r2", _usage(1200, 1000, 60, 20)),
                   _record("t2", "r9", _usage(10, 0, 1)))
        found = codex_rollout.attempt_usage(self.path, frozenset({"t1"}))
        self.assertEqual(set(found["usage"].values()), {None})
        self.assertIn("response_id_in_other_turn", found["usage_conflicts"])
        self.assertEqual(found["usage_disputed_fields"], sorted(codex_rollout.FIELDS))

    def test_unknown_counter_stays_null_and_partial(self) -> None:
        missing = _usage(500, 100, 20)
        del missing["reasoning_output_tokens"]
        self.write(_started("t1"), _record("t1", "r1", _usage(1000, 800, 50, 10)), _record("t1", "r2", missing))
        found = codex_rollout.attempt_usage(self.path, frozenset())
        self.assertIsNone(found["usage"]["reasoning_output_tokens"])
        self.assertEqual(found["usage"]["input_tokens"], 1500)
        self.assertEqual(found["usage_partial_fields"], ["reasoning_output_tokens"])

    def test_snapshot_mismatch_and_cache_over_input_are_conflicts(self) -> None:
        self.write(_started("t1"), _record("t1", "r1", _usage(100, 300, 5), turn_snapshot=_usage(999, 300, 5)))
        found = codex_rollout.attempt_usage(self.path, frozenset())
        self.assertEqual(found["usage"]["input_tokens"], 100)
        self.assertEqual(found["usage_conflicts"], ["input_tokens", "cached_exceeds_input"])

    def test_unlinked_reasons_are_null_never_session_totals(self) -> None:
        self.assertEqual(codex_rollout.attempt_usage(None, frozenset())["usage_unlinked_reason"], "rollout_missing")
        self.write(*COLD)
        same = codex_rollout.attempt_usage(self.path, codex_rollout.turn_ids(self.path))
        self.assertEqual(same["usage_unlinked_reason"], "rollout_no_new_turn")
        self.write(*RESUME)
        both = codex_rollout.attempt_usage(self.path, frozenset())
        self.assertEqual(both["usage_unlinked_reason"], "rollout_ambiguous_turns")
        self.write(_started("t3"))
        empty = codex_rollout.attempt_usage(self.path, frozenset({"t1", "t2"}))
        self.assertEqual(empty["usage_unlinked_reason"], "rollout_turn_without_requests")
        for unlinked in (same, both, empty):
            self.assertEqual(unlinked["usage_scope"], "unknown")
            self.assertEqual(set(unlinked["usage"].values()), {None})


class LedgerTests(unittest.TestCase):
    def _observed(self, attempt, stdout=""):
        inv = Invocation.__new__(Invocation)
        inv.billing = "plan"
        inv.outcome_complete = True
        seen = []
        inv.observe = lambda data=None, **kw: seen.append(dict(kw, data=data))
        inv.stdout(stdout, attempt_usage=attempt)
        return seen

    def test_linked_is_final_and_unlinked_is_not(self) -> None:
        linked = {"usage_scope": "attempt", "usage_provenance": "rollout_token_usage_records",
                  "usage": _usage(2500, 1500, 140, 10)}
        (row,) = self._observed(linked)
        self.assertEqual((row["key"], row["final"], row["cost_basis"]), ("codex:rollout", True, "api_equivalent"))
        self.assertEqual(row["data"]["input_tokens"], 2500)
        self.assertEqual(row["data"]["cached_input_tokens"], 1500)
        (row,) = self._observed(codex_rollout.unlinked("rollout_missing"))
        self.assertFalse(row["final"])
        self.assertEqual(set(row["data"].values()), {None})

    def test_quality_flags_reach_the_ledger_row(self) -> None:
        attempt = {"usage_scope": "attempt", "usage_provenance": "rollout_token_usage_records",
                   "usage": _usage(100, 300, 5), "usage_partial_fields": ["reasoning_output_tokens"],
                   "usage_disputed_fields": ["input_tokens"],
                   "usage_conflicts": ["response_id_replay_conflict", "cached_exceeds_input"]}
        (row,) = self._observed(attempt)
        self.assertEqual(set(row["quality"]), {
            "partial:reasoning_output_tokens", "disputed:input_tokens",
            "conflict:response_id_replay_conflict", "conflict:cached_exceeds_input"})
        (row,) = self._observed(codex_rollout.unlinked("rollout_missing"))
        self.assertIn("unlinked:rollout_missing", row["quality"])

    def test_attempt_usage_suppresses_cumulative_stdout_usage(self) -> None:
        stdout = json.dumps({"type": "turn.completed", "usage": _usage(9000, 8000, 300)})
        rows = self._observed(codex_rollout.unlinked("rollout_missing"), stdout)
        self.assertEqual([r["key"] for r in rows], ["codex:rollout"])


class AdapterAccountingTests(RolloutFixture):
    def _task(self, **payload) -> Task:
        return Task(job_id="job-rev", role="audit", instruction="Revise.", adapter="codex",
                    payload={"cwd": str(Path.cwd()), "sandbox": "read-only", "model": "gpt-5.4-mini",
                             "disable_codegraph": True, **payload})

    def _run(self, task, stdout, append):
        def launch(*_args, **_kwargs):
            self.write(*append)
            return StreamedProcess(returncode=0, stdout=stdout, stderr="", timed_out=False)

        env = {"CODEX_HOME": str(self.home), "PUPPETMASTER_CODEX_LEAN_HOME": "0",
               "PUPPETMASTER_HOME": str(self.home / "pm")}
        with patch.dict("os.environ", env), \
                patch("puppetmaster.adapters.resolve_command", side_effect=lambda n: f"/usr/bin/{n}"), \
                patch("puppetmaster.adapters.worktree_guard", return_value=None), \
                patch("puppetmaster.adapters.git_snapshot", return_value=CLEAN), \
                patch("puppetmaster.adapters.run_streamed_subprocess", side_effect=launch):
            artifacts = CodexAdapter().run(task, "goal", "worker")
        return next(a for a in artifacts if "usage_scope" in a.payload).payload

    @staticmethod
    def _stdout(usage):
        return "\n".join(json.dumps(e) for e in (
            {"type": "thread.started", "thread_id": SID},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
            {"type": "turn.completed", "usage": usage}))

    def test_cold_then_resume_reconcile_exact_unique_sums(self) -> None:
        cold = self._run(self._task(ephemeral=False), self._stdout(_usage(2200, 1800, 110, 30)), COLD)
        self.assertEqual(cold["usage_provenance"], "rollout_token_usage_records")
        self.assertEqual(cold["sdk_usage_scope"], "attempt")
        self.assertEqual((cold["tokens_in"], cold["cached_input_tokens"], cold["tokens_out"]), (2200, 1800, 110))

        record = {"status": "resolved", "adapter": "codex", "session_id": SID,
                  "from_job_id": "job-prior", "from_task_id": "t-codex"}
        # Session-cumulative SDK usage that also dropped the pre-compaction request r3.
        sdk = _usage(2200 + 1000, 1800 + 300, 110 + 70, 30 + 5)
        resumed = self._run(self._task(resume=record), self._stdout(sdk), RESUME)
        self.assertEqual(resumed["usage_scope"], "attempt")
        self.assertEqual(resumed["rollout_turn_id"], "t2")
        self.assertEqual(resumed["rollout_request_count"], 3)
        self.assertEqual(resumed["tokens_in"], RESUME_EXACT["input_tokens"])
        self.assertEqual(resumed["cached_input_tokens"], RESUME_EXACT["cached_input_tokens"])
        self.assertEqual(resumed["tokens_out"], RESUME_EXACT["output_tokens"])
        self.assertEqual(resumed["reasoning_output_tokens"], RESUME_EXACT["reasoning_output_tokens"])
        self.assertEqual(resumed["sdk_usage_scope"], "session_cumulative")
        self.assertEqual(resumed["sdk_input_tokens"], sdk["input_tokens"])
        # Neither the snapshot nor snapshot-minus-previous is the answer.
        self.assertNotEqual(resumed["tokens_in"], sdk["input_tokens"] - 2200)
        self.assertEqual(resumed["selected_facts"]["tokens_in"], 2500)

    def test_resume_without_rollout_is_null_with_reason(self) -> None:
        record = {"status": "resolved", "adapter": "codex", "session_id": SID,
                  "from_job_id": "job-prior", "from_task_id": "t-codex"}
        self.path.unlink(missing_ok=True)
        payload = self._run(self._task(resume=record), self._stdout(_usage(9000, 8000, 300)), [])
        self.assertEqual(payload["usage_scope"], "unknown")
        self.assertIsNone(payload["tokens_in"])
        self.assertIsNone(payload["tokens_total"])
        self.assertIsNone(payload["selected_facts"]["tokens_in"])
        self.assertEqual(payload["sdk_input_tokens"], 9000)


class AccountingShapeTests(unittest.TestCase):
    def test_cold_without_rollout_keeps_attempt_local_sdk(self) -> None:
        out = _attempt_accounting(None, _usage(10, 4, 2), resumed=False)
        self.assertEqual(out["usage_provenance"], "sdk_turn_completed")
        self.assertEqual(out["usage"]["input_tokens"], 10)

    def test_cold_without_any_usage_is_null_not_zero(self) -> None:
        out = _attempt_accounting(None, {}, resumed=False)
        self.assertEqual(out["usage_unlinked_reason"], "sdk_usage_missing")
        self.assertIsNone(out["usage"]["input_tokens"])


if __name__ == "__main__":
    unittest.main()
