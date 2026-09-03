"""Host lifecycle classification, boot record, and lifecycle-default event query."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from puppetmaster.host_lifecycle import (
    HOST_BOOT_FILENAME,
    HOST_EVENT_RECOVERED,
    HOST_EVENT_STARTED,
    HOST_LIFECYCLE_EVENTS,
    NOISY_TURN_EVENTS,
    classify_host_start,
    filter_events,
    mark_clean_shutdown,
    read_lifecycle_events,
    record_host_start,
    reset_host_start_guard,
)
from puppetmaster.models import JobStatus
from puppetmaster.store import SwarmStore
from puppetmaster.worker_fence import WORKER_ENV, WORKER_VALUE


def _forget_host_start(_store: SwarmStore) -> None:
    reset_host_start_guard()


def _boot_path(store: SwarmStore) -> Path:
    return Path(store.root) / HOST_BOOT_FILENAME


def _event_names(store: SwarmStore, job_id: str) -> list[str]:
    return [str(item.get("event")) for item in store.read_events(job_id)]


class ClassifyHostStartTests(unittest.TestCase):
    def test_none_is_first_start(self) -> None:
        self.assertEqual(
            classify_host_start(None),
            (HOST_EVENT_STARTED, "first"),
        )

    def test_clean_shutdown_is_reboot(self) -> None:
        self.assertEqual(
            classify_host_start({"clean_shutdown": True}),
            (HOST_EVENT_STARTED, "reboot"),
        )

    def test_unclean_previous_is_crash(self) -> None:
        self.assertEqual(
            classify_host_start({"clean_shutdown": False}),
            (HOST_EVENT_RECOVERED, "crash"),
        )
        self.assertEqual(
            classify_host_start({}),
            (HOST_EVENT_RECOVERED, "crash"),
        )


class FilterEventsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events = [
            {"event": HOST_EVENT_STARTED},
            {"event": HOST_EVENT_RECOVERED},
            {"event": "job.stalled"},
            {"event": "task.lease_renewed"},
            {"event": "run.heartbeat"},
            {"event": "task.saved"},
            {"event": "job.status"},
        ]

    def test_lifecycle_keeps_host_and_stalled(self) -> None:
        names = [item["event"] for item in filter_events(self.events)]
        self.assertEqual(
            names,
            [HOST_EVENT_STARTED, HOST_EVENT_RECOVERED, "job.stalled"],
        )

    def test_quiet_drops_noisy_turn_events(self) -> None:
        names = [item["event"] for item in filter_events(self.events, include="quiet")]
        for noisy in NOISY_TURN_EVENTS:
            self.assertNotIn(noisy, names)
        self.assertIn("job.status", names)
        self.assertIn(HOST_EVENT_STARTED, names)

    def test_all_is_identity(self) -> None:
        filtered = filter_events(self.events, include="all")
        self.assertIs(filtered, self.events)

    def test_unknown_include_matches_lifecycle(self) -> None:
        self.assertEqual(
            filter_events(self.events, include="nope"),
            filter_events(self.events, include="lifecycle"),
        )


class HostStartRecordTests(unittest.TestCase):
    def test_empty_store_first_start(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            path = _boot_path(store)
            self.assertTrue(path.is_file())
            boot = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(boot["kind"], HOST_EVENT_STARTED)
            self.assertEqual(boot["reason"], "first")
            self.assertFalse(boot["clean_shutdown"])
            self.assertEqual(boot["pid"], os.getpid())
            self.assertIn("boot_id", boot)
            self.assertIn("host", boot)
            self.assertIn("started_at", boot)
            self.assertEqual(store.list_jobs(), [])
            self.assertEqual(
                {HOST_EVENT_STARTED, HOST_EVENT_RECOVERED},
                set(HOST_LIFECYCLE_EVENTS),
            )

    def test_crash_fanout_to_live_jobs_not_complete(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            running = store.create_job("keep running")
            stitching = store.create_job("keep stitching")
            complete = store.create_job("already done")
            store.update_job_status(running.id, JobStatus.RUNNING)
            store.update_job_status(stitching.id, JobStatus.STITCHING)
            store.update_job_status(complete.id, JobStatus.COMPLETE)

            _forget_host_start(store)
            record = record_host_start(store)
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record.kind, HOST_EVENT_RECOVERED)
            self.assertEqual(record.reason, "crash")
            self.assertEqual(record.fanned_out, 2)
            self.assertEqual(record.skipped, 0)
            self.assertFalse(record.idempotent)

            self.assertIn(HOST_EVENT_RECOVERED, _event_names(store, running.id))
            self.assertIn(HOST_EVENT_RECOVERED, _event_names(store, stitching.id))
            self.assertNotIn(HOST_EVENT_RECOVERED, _event_names(store, complete.id))

    def test_second_call_is_process_idempotent(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            running = store.create_job("live")
            other = store.create_job("also live")
            store.update_job_status(running.id, JobStatus.RUNNING)
            store.update_job_status(other.id, JobStatus.STITCHING)

            _forget_host_start(store)
            first = record_host_start(store)
            self.assertIsNotNone(first)
            before = {
                running.id: store.read_events(running.id),
                other.id: store.read_events(other.id),
            }
            second = record_host_start(store)
            self.assertIsNotNone(second)
            assert first is not None and second is not None
            self.assertTrue(second.idempotent)
            self.assertEqual(second.boot_id, first.boot_id)
            self.assertEqual(second.kind, first.kind)
            self.assertEqual(store.read_events(running.id), before[running.id])
            self.assertEqual(store.read_events(other.id), before[other.id])

    def test_emit_failure_on_one_job_still_fans_the_other(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            job_a = store.create_job("emit fails")
            job_b = store.create_job("emit works")
            store.update_job_status(job_a.id, JobStatus.RUNNING)
            store.update_job_status(job_b.id, JobStatus.RUNNING)

            real_emit = store.emit

            def _flaky_emit(job_id: str, event: str, payload: dict) -> None:
                if job_id == job_a.id:
                    raise RuntimeError("emit failed")
                real_emit(job_id, event, payload)

            _forget_host_start(store)
            with patch.object(store, "emit", side_effect=_flaky_emit):
                record = record_host_start(store)
            self.assertIsNotNone(record)
            assert record is not None
            self.assertGreaterEqual(record.skipped, 1)
            self.assertEqual(record.fanned_out, 1)
            self.assertNotIn(HOST_EVENT_RECOVERED, _event_names(store, job_a.id))
            self.assertIn(HOST_EVENT_RECOVERED, _event_names(store, job_b.id))

    def test_worker_env_is_noop(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            with patch.dict(os.environ, {WORKER_ENV: WORKER_VALUE}, clear=False):
                self.assertIsNone(record_host_start(store))
            self.assertFalse(_boot_path(store).exists())

    def test_mark_clean_shutdown_classifies_reboot(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            first = record_host_start(store)
            self.assertIsNotNone(first)
            marked = mark_clean_shutdown(store)
            self.assertIsNotNone(marked)
            assert marked is not None
            self.assertTrue(marked["clean_shutdown"])
            boot = json.loads(_boot_path(store).read_text(encoding="utf-8"))
            self.assertTrue(boot["clean_shutdown"])
            self.assertEqual(
                classify_host_start(boot),
                (HOST_EVENT_STARTED, "reboot"),
            )


class ReadLifecycleEventsTests(unittest.TestCase):
    def test_default_include_is_lifecycle(self) -> None:
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            job = store.create_job("query events")
            store.emit(job.id, HOST_EVENT_STARTED, {"reason": "first"})
            store.emit(job.id, "run.heartbeat", {"n": 1})
            store.emit(job.id, "task.lease_renewed", {"task_id": "t1"})
            store.emit(job.id, "task.saved", {"task_id": "t1"})
            store.emit(job.id, "job.stalled", {"reason": "dead"})
            store.emit(job.id, "job.status", {"status": "running"})

            names = [
                str(item.get("event"))
                for item in read_lifecycle_events(store, job.id)
            ]
            self.assertEqual(names, [HOST_EVENT_STARTED, "job.stalled"])
            self.assertEqual(
                [item["event"] for item in read_lifecycle_events(store, job.id)],
                [
                    item["event"]
                    for item in read_lifecycle_events(
                        store, job.id, include="lifecycle"
                    )
                ],
            )
            raw = [str(item.get("event")) for item in store.read_events(job.id)]
            self.assertIn("run.heartbeat", raw)


if __name__ == "__main__":
    unittest.main()
