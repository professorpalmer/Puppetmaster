"""Durable session command ledger — evaluate + mark-before-execute."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from puppetmaster.session_commands import (
    CommandBasedOn,
    CommandDisposition,
    EvaluationContext,
    JobCommandLedger,
    SessionCommandKind,
    SessionCommandStatus,
    evaluate_command,
)


class EvaluateCommandTests(unittest.TestCase):
    def _entry(self, **kwargs):
        from puppetmaster.session_commands import SessionCommandEntry

        defaults = dict(
            id="cmd_1",
            kind=SessionCommandKind.STEER,
            issued_by="device-a",
            issued_at_ms=1000,
            status=SessionCommandStatus.PENDING,
            payload={"prompt": "nudge"},
        )
        defaults.update(kwargs)
        return SessionCommandEntry(**defaults)

    def test_processed_is_skip(self) -> None:
        entry = self._entry()
        cx = EvaluationContext(
            is_processed=lambda cid: cid == entry.id,
            now_ms=1500,
            entries=(entry,),
        )
        self.assertEqual(evaluate_command(entry, cx), CommandDisposition.SKIP)

    def test_ttl_expired(self) -> None:
        entry = self._entry(issued_at_ms=1000, expires_at_ms=1100)
        cx = EvaluationContext(
            is_processed=lambda _cid: False,
            now_ms=1100,
            entries=(entry,),
        )
        self.assertEqual(evaluate_command(entry, cx), CommandDisposition.EXPIRED)

    def test_newer_steer_supersedes(self) -> None:
        older = self._entry(id="cmd_old", issued_at_ms=1000)
        newer = self._entry(id="cmd_new", issued_at_ms=2000)
        cx = EvaluationContext(
            is_processed=lambda _cid: False,
            now_ms=2500,
            entries=(older, newer),
        )
        self.assertEqual(evaluate_command(older, cx), CommandDisposition.SUPERSEDED)
        self.assertEqual(evaluate_command(newer, cx), CommandDisposition.EXECUTE)

    def test_interrupt_past_turn_superseded(self) -> None:
        entry = self._entry(
            id="cmd_int",
            kind=SessionCommandKind.INTERRUPT,
            based_on=CommandBasedOn(turn_id="turn_1"),
            payload={},
        )
        cx = EvaluationContext(
            is_processed=lambda _cid: False,
            now_ms=2500,
            entries=(entry,),
            current_turn_id="turn_2",
            turn_is_past=lambda turn: turn == "turn_1",
        )
        self.assertEqual(evaluate_command(entry, cx), CommandDisposition.SUPERSEDED)

    def test_interrupt_current_turn_executes(self) -> None:
        entry = self._entry(
            id="cmd_int",
            kind=SessionCommandKind.INTERRUPT,
            based_on=CommandBasedOn(turn_id="turn_1"),
            payload={},
        )
        cx = EvaluationContext(
            is_processed=lambda _cid: False,
            now_ms=2500,
            entries=(entry,),
            current_turn_id="turn_1",
            turn_is_past=lambda _turn: False,
        )
        self.assertEqual(evaluate_command(entry, cx), CommandDisposition.EXECUTE)


class JobCommandLedgerTests(unittest.TestCase):
    def test_mark_before_execute_then_replay_skips(self) -> None:
        with TemporaryDirectory() as tmp:
            ledger = JobCommandLedger(tmp, "job_demo")
            run = ledger.append(
                SessionCommandKind.RUN,
                issued_by="pilot",
                payload={"goal": "ship it", "message_id": "m1"},
            )
            interrupt = ledger.append(
                SessionCommandKind.INTERRUPT,
                issued_by="pilot",
                based_on=CommandBasedOn(turn_id="turn_9"),
            )
            pending = ledger.evaluate_pending(now_ms=run.issued_at_ms + 10)
            self.assertEqual(len(pending), 2)
            for entry, disposition in pending:
                self.assertEqual(disposition, CommandDisposition.EXECUTE)
                ledger.apply_disposition(entry, disposition)
            self.assertIn(run.id, ledger.processed_ids())
            self.assertIn(interrupt.id, ledger.processed_ids())
            # Replay is Skip after mark-processed.
            again = ledger.evaluate_pending(now_ms=run.issued_at_ms + 20)
            kinds = {disposition for _entry, disposition in again}
            self.assertEqual(kinds, {CommandDisposition.SKIP})

    def test_rewrite_status_is_visible_to_listing(self) -> None:
        with TemporaryDirectory() as tmp:
            ledger = JobCommandLedger(Path(tmp), "job_x")
            entry = ledger.append(SessionCommandKind.STEER, issued_by="device-a", payload={})
            self.assertEqual(entry.status, SessionCommandStatus.PENDING)
            ledger.rewrite_status(entry.id, SessionCommandStatus.APPLIED)
            updated = ledger.list_entries()[0]
            self.assertEqual(updated.status, SessionCommandStatus.APPLIED)


if __name__ == "__main__":
    unittest.main()
