"""Host SCM observe: independent reactions, suppress ≠ delivered, derived attention."""
from __future__ import annotations

import json
import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from puppetmaster.metr_seams import (
    WAIT_USER,
    load_host_document,
    record_host_observation,
)
from puppetmaster.models import Artifact, ArtifactType, JobStatus, Task, TaskStatus
from puppetmaster.receipt import build_job_receipt
from puppetmaster.scm_observe import (
    ATTENTION_DONE,
    ATTENTION_NEEDS_YOU,
    ATTENTION_QUEUED,
    ATTENTION_READY,
    ATTENTION_WORKING,
    OUTCOME_ACCOUNTED,
    OUTCOME_SKIPPED,
    OUTCOME_SUPPRESSED,
    SCMSnapshot,
    derive_attention,
    facts_from_snapshot,
    fetch_github_pr,
    observe_scm,
    snapshot_from_gh_payload,
)
from puppetmaster.store import SwarmStore


def _payload(**overrides):
    body = {
        "url": "https://github.com/example/repo/pull/7",
        "number": 7,
        "title": "fix ci",
        "state": "OPEN",
        "mergeable": "MERGEABLE",
        "reviewDecision": "",
        "statusCheckRollup": [
            {"name": "tests", "conclusion": "SUCCESS"},
        ],
    }
    body.update(overrides)
    return body


def _parent(store, job_id, instruction="root"):
    parent = Task(
        job_id=job_id,
        role="implement",
        instruction=instruction,
        status=TaskStatus.COMPLETE,
    )
    store.save_task(parent)
    return parent


class GhPayloadTests(unittest.TestCase):
    def test_snapshot_collects_failing_checks_and_errors_independently(self):
        snap = snapshot_from_gh_payload(
            _payload(
                statusCheckRollup=[
                    {"name": "tests", "conclusion": "FAILURE"},
                    {"name": "lint", "conclusion": "ERROR"},
                    {"name": "ok", "conclusion": "SUCCESS"},
                    {"name": "", "conclusion": "FAILURE"},
                ],
                reviewDecision="CHANGES_REQUESTED",
                mergeable="CONFLICTING",
            )
        )
        self.assertEqual(snap.failing_checks, ("tests", "lint"))
        self.assertEqual(snap.review_decision, "CHANGES_REQUESTED")
        self.assertEqual(snap.mergeable, "CONFLICTING")
        self.assertEqual(snap.fetch_errors, ())

        missing = snapshot_from_gh_payload(_payload(statusCheckRollup=None))
        self.assertIn("checks: missing statusCheckRollup", missing.fetch_errors)
        facts = facts_from_snapshot(missing)
        kinds = {fact.kind for fact in facts}
        self.assertNotIn("ci_failed", kinds)
        self.assertNotIn("ci_passing", kinds)

    def test_facts_are_independent(self):
        snap = snapshot_from_gh_payload(
            _payload(
                statusCheckRollup=[{"name": "tests", "conclusion": "FAILURE"}],
                reviewDecision="CHANGES_REQUESTED",
                mergeable="CONFLICTING",
            )
        )
        kinds = [fact.kind for fact in facts_from_snapshot(snap)]
        self.assertEqual(
            kinds, ["ci_failed", "changes_requested", "conflicting"]
        )

    def test_merged_is_observation_only(self):
        snap = snapshot_from_gh_payload(_payload(state="MERGED"))
        kinds = [fact.kind for fact in facts_from_snapshot(snap)]
        self.assertIn("merged", kinds)
        self.assertIn("ci_passing", kinds)

    def test_fetch_github_pr_no_pr_is_none(self):
        def runner(*_args, **_kwargs):
            return SimpleNamespace(returncode=1, stdout="", stderr="no pull requests found")

        self.assertIsNone(fetch_github_pr("/tmp", runner=runner))

    def test_fetch_github_pr_parses_json(self):
        def runner(*_args, **_kwargs):
            return SimpleNamespace(
                returncode=0, stdout=json.dumps(_payload()), stderr=""
            )

        snap = fetch_github_pr("/tmp", runner=runner)
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertEqual(snap.number, 7)
        self.assertEqual(snap.failing_checks, ())


class ObserveScmTests(unittest.TestCase):
    def test_ci_failed_enqueues_follow_up(self):
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("scm")
            parent = _parent(store, job.id)
            snap = snapshot_from_gh_payload(
                _payload(statusCheckRollup=[{"name": "tests", "conclusion": "FAILURE"}])
            )
            result = observe_scm(store, job.id, snap)
            self.assertEqual(len(result["reactions"]), 1)
            self.assertEqual(result["reactions"][0]["outcome"], OUTCOME_ACCOUNTED)
            self.assertEqual(result["reactions"][0]["reason"], "enqueued")
            tasks = store.list_tasks(job.id)
            self.assertEqual(len(tasks), 2)
            child = [task for task in tasks if task.id != parent.id][0]
            self.assertEqual(child.role, "implement")
            self.assertIn("CI is failing", child.instruction)
            self.assertEqual(child.depends_on, [parent.id])
            self.assertEqual(result["attention"], ATTENTION_NEEDS_YOU)

    def test_independent_reactions_do_not_hide_each_other(self):
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("scm")
            _parent(store, job.id)
            snap = snapshot_from_gh_payload(
                _payload(
                    statusCheckRollup=[{"name": "tests", "conclusion": "FAILURE"}],
                    reviewDecision="CHANGES_REQUESTED",
                    mergeable="CONFLICTING",
                )
            )
            result = observe_scm(store, job.id, snap)
            kinds = [row["kind"] for row in result["reactions"]]
            self.assertEqual(kinds, ["ci_failed", "changes_requested", "conflicting"])
            self.assertEqual(
                {row["outcome"] for row in result["reactions"]}, {OUTCOME_ACCOUNTED}
            )
            self.assertEqual(len(store.list_tasks(job.id)), 4)

    def test_waiting_user_suppresses_and_retries_later(self):
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("scm")
            _parent(store, job.id)
            store.set_job_wait_reason(job.id, WAIT_USER, actor="coordinator")
            snap = snapshot_from_gh_payload(
                _payload(statusCheckRollup=[{"name": "tests", "conclusion": "FAILURE"}])
            )
            first = observe_scm(store, job.id, snap)
            self.assertEqual(first["reactions"][0]["outcome"], OUTCOME_SUPPRESSED)
            self.assertEqual(first["reactions"][0]["reason"], WAIT_USER)
            self.assertEqual(len(store.list_tasks(job.id)), 1)
            document = load_host_document(store, job.id)
            key = first["reactions"][0]["key"]
            self.assertEqual(document["reactions"][key]["outcome"], OUTCOME_SUPPRESSED)

            store.set_job_wait_reason(job.id, None, actor="coordinator")
            second = observe_scm(store, job.id, snap)
            self.assertEqual(second["reactions"][0]["outcome"], OUTCOME_ACCOUNTED)
            self.assertEqual(len(store.list_tasks(job.id)), 2)

    def test_signature_change_re_enqueues(self):
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("scm")
            _parent(store, job.id)
            first_snap = snapshot_from_gh_payload(
                _payload(statusCheckRollup=[{"name": "tests", "conclusion": "FAILURE"}])
            )
            observe_scm(store, job.id, first_snap)
            dup = observe_scm(store, job.id, first_snap)
            self.assertEqual(dup["reactions"][0]["reason"], "deduped")
            self.assertEqual(len(store.list_tasks(job.id)), 2)
            second_snap = snapshot_from_gh_payload(
                _payload(
                    statusCheckRollup=[
                        {"name": "tests", "conclusion": "FAILURE"},
                        {"name": "lint", "conclusion": "FAILURE"},
                    ]
                )
            )
            changed = observe_scm(store, job.id, second_snap)
            self.assertEqual(changed["reactions"][0]["reason"], "enqueued")
            self.assertEqual(len(store.list_tasks(job.id)), 3)

    def test_hold_and_veto_suppress(self):
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("scm")
            _parent(store, job.id)
            store.hold_subgraph(job.id, actor="coordinator")
            snap = snapshot_from_gh_payload(
                _payload(statusCheckRollup=[{"name": "tests", "conclusion": "FAILURE"}])
            )
            held = observe_scm(store, job.id, snap)
            self.assertEqual(held["reactions"][0]["outcome"], OUTCOME_SUPPRESSED)
            self.assertEqual(held["reactions"][0]["reason"], "hold")
            self.assertEqual(len(store.list_tasks(job.id)), 1)
            store.resume_subgraph(job.id, actor="coordinator")
            store.veto_subgraph(job.id, actor="coordinator")
            vetoed = observe_scm(store, job.id, snap)
            self.assertEqual(vetoed["reactions"][0]["reason"], "veto")
            self.assertEqual(len(store.list_tasks(job.id)), 1)

    def test_terminal_job_records_but_does_not_enqueue(self):
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("scm")
            _parent(store, job.id)
            store.update_job_status(job.id, JobStatus.COMPLETE, actor="coordinator")
            snap = snapshot_from_gh_payload(
                _payload(statusCheckRollup=[{"name": "tests", "conclusion": "FAILURE"}])
            )
            result = observe_scm(store, job.id, snap)
            self.assertTrue(result["observations"])
            self.assertEqual(result["reactions"][0]["outcome"], OUTCOME_SKIPPED)
            self.assertEqual(result["reactions"][0]["reason"], "job_terminal")
            self.assertEqual(len(store.list_tasks(job.id)), 1)
            self.assertEqual(result["attention"], ATTENTION_DONE)

    def test_reactions_survive_new_observation(self):
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("scm")
            _parent(store, job.id)
            snap = snapshot_from_gh_payload(
                _payload(statusCheckRollup=[{"name": "tests", "conclusion": "FAILURE"}])
            )
            observe_scm(store, job.id, snap)
            before = load_host_document(store, job.id)
            self.assertTrue(before.get("reactions"))
            record_host_observation(
                store, job.id, "shipped", evidence=["sha:abc"], source="host"
            )
            after = load_host_document(store, job.id)
            self.assertEqual(before["reactions"], after["reactions"])
            kinds = [row["kind"] for row in after["observations"]]
            self.assertIn("ci_failed", kinds)
            self.assertIn("shipped", kinds)

    def test_ci_passing_does_not_enqueue(self):
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("scm")
            _parent(store, job.id)
            snap = snapshot_from_gh_payload(_payload())
            result = observe_scm(store, job.id, snap)
            self.assertEqual(result["reactions"], [])
            self.assertEqual(result["observations"][0]["kind"], "ci_passing")
            self.assertEqual(len(store.list_tasks(job.id)), 1)

    def test_attention_lanes(self):
        with TemporaryDirectory() as tmp:
            store = SwarmStore(Path(tmp) / ".puppetmaster")
            store.init()
            job = store.create_job("scm")
            self.assertEqual(derive_attention(store, job.id), ATTENTION_QUEUED)
            parent = _parent(store, job.id)
            store.update_job_status(job.id, JobStatus.RUNNING, actor="coordinator")
            self.assertEqual(derive_attention(store, job.id), ATTENTION_WORKING)
            store.set_job_wait_reason(job.id, WAIT_USER, actor="coordinator")
            self.assertEqual(derive_attention(store, job.id), ATTENTION_NEEDS_YOU)
            store.set_job_wait_reason(job.id, None, actor="coordinator")
            store.save_artifact(
                Artifact(
                    job_id=job.id,
                    task_id=parent.id,
                    type=ArtifactType.PATCH,
                    created_by="worker",
                    confidence=0.9,
                    evidence=["diff"],
                    payload={"change": "fix", "files": ["x.py"]},
                )
            )
            observe_scm(store, job.id, snapshot_from_gh_payload(_payload()), enqueue=False)
            self.assertEqual(derive_attention(store, job.id), ATTENTION_READY)
            receipt = build_job_receipt(store, job.id)
            self.assertEqual(receipt["attention"], ATTENTION_READY)

    def test_cli_parses_observe_scm(self):
        from puppetmaster.cli._parser import build_parser

        args = build_parser().parse_args(
            ["observe-scm", "job_123", "--json", "--no-enqueue", "--cwd", "/tmp/repo"]
        )
        self.assertEqual(args.command, "observe-scm")
        self.assertEqual(args.job_id, "job_123")
        self.assertTrue(args.json)
        self.assertTrue(args.no_enqueue)
        self.assertEqual(args.cwd, "/tmp/repo")
