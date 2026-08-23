"""Hermetic coverage for the effort-level artifact index."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from puppetmaster.cli import main as cli_main
from puppetmaster.lifecycle import (
    index_effort_artifacts,
    latest_tagged_effort_id,
    tag_job_effort,
)
from puppetmaster.models import Artifact, ArtifactType
from puppetmaster.store_factory import create_store
from puppetmaster.validation import compact_artifact_ref


def _store(tmp: str):
    store = create_store("file", Path(tmp) / ".puppetmaster")
    store.init()
    return store


def _retag(store, job_id: str, effort_id: str, tagged_at: str) -> None:
    store.write_json(
        store.job_dir(job_id) / "effort.json",
        {"effort_id": effort_id, "tagged_at": tagged_at},
    )


def _finding(job_id: str, claim: str, **payload) -> Artifact:
    body = {"claim": claim, "details": "SECRET-BODY-" + ("x" * 4000)}
    body.update(payload)
    return Artifact(
        job_id=job_id,
        task_id="task-1",
        type=ArtifactType.FINDING,
        created_by="worker",
        confidence=0.8,
        evidence=["src/a.py"],
        payload=body,
    )


def _decision(job_id: str, decision: str) -> Artifact:
    return Artifact(
        job_id=job_id,
        task_id="task-2",
        type=ArtifactType.DECISION,
        created_by="worker",
        confidence=0.9,
        evidence=["src/b.py"],
        payload={"decision": decision, "why": "because"},
    )


def _check(job_id: str, check: str) -> Artifact:
    return Artifact(
        job_id=job_id,
        task_id="task-3",
        type=ArtifactType.VERIFICATION,
        created_by="worker",
        confidence=0.7,
        evidence=["tests"],
        payload={"check": check, "result": "passed"},
    )


class EffortArtifactIndexTests(unittest.TestCase):
    def test_index_filters_effort_type_and_query(self) -> None:
        with TemporaryDirectory() as tmp:
            store = _store(tmp)
            old = store.create_job("old effort")
            new = store.create_job("new effort")
            other = store.create_job("untagged")
            _retag(store, old.id, "eff-old", "2026-01-01T00:00:00+00:00")
            _retag(store, new.id, "eff-new", "2026-08-01T00:00:00+00:00")
            store.save_artifact(_finding(old.id, "legacy auth claim"))
            store.save_artifact(_finding(new.id, "fresh auth claim"))
            store.save_artifact(_decision(new.id, "ship the auth fix"))
            store.save_artifact(_check(new.id, "pytest -q"))
            store.save_artifact(_finding(other.id, "should not appear"))

            latest = latest_tagged_effort_id([store])
            self.assertEqual(latest, "eff-new")

            defaulted = index_effort_artifacts([store])
            self.assertEqual(defaulted["effort_id"], "eff-new")
            self.assertEqual(defaulted["requested_effort_id"], None)
            self.assertEqual(defaulted["latest_effort_id"], "eff-new")
            self.assertEqual(defaulted["jobs"], 1)
            claims = {ref.get("claim") for ref in defaulted["refs"]}
            self.assertIn("fresh auth claim", claims)
            self.assertNotIn("legacy auth claim", claims)
            encoded = json.dumps(defaulted)
            self.assertNotIn("SECRET-BODY-", encoded)
            self.assertNotIn("transcript", encoded.lower())
            for ref in defaulted["refs"]:
                self.assertNotIn("payload", ref)
                self.assertIn("id", ref)
                self.assertIn("type", ref)
                self.assertIn("sha256", ref)
                self.assertIn("confidence", ref)
                self.assertIn("job_id", ref)
                self.assertEqual(ref["job_id"], new.id)
                self.assertIn("task_id", ref)
                self.assertIn("created_at", ref)
                compact = compact_artifact_ref(
                    next(
                        a
                        for a in store.list_artifacts(new.id)
                        if a.id == ref["id"]
                    )
                )
                for key in ("id", "type", "sha256", "confidence", "task_id"):
                    self.assertEqual(ref[key], compact[key])

            typed = index_effort_artifacts([store], effort_id="eff-new", artifact_type="decision")
            self.assertEqual(typed["count"], 1)
            self.assertEqual(typed["refs"][0]["type"], "decision")
            self.assertEqual(typed["refs"][0]["decision"], "ship the auth fix")

            queried = index_effort_artifacts([store], effort_id="eff-new", query="pytest")
            self.assertEqual(queried["count"], 1)
            self.assertEqual(queried["refs"][0]["check"], "pytest -q")

            old_index = index_effort_artifacts([store], effort_id="eff-old")
            self.assertEqual(old_index["count"], 1)
            self.assertEqual(old_index["refs"][0]["claim"], "legacy auth claim")

            expanded = index_effort_artifacts([store], effort_id="eff-new", query="fresh", expand=True)
            self.assertEqual(expanded["count"], 1)
            self.assertIn("payload", expanded["refs"][0])
            self.assertIn("SECRET-BODY-", expanded["refs"][0]["payload"]["details"])

    def test_skips_transcript_types(self) -> None:
        class FakeArtifact:
            def __init__(self) -> None:
                self.id = "artifact_transcript"
                self.type = "transcript"
                self.task_id = "t"
                self.sha256 = "00" * 32
                self.confidence = 1.0
                self.created_at = "2026-08-01T00:00:00+00:00"
                self.job_id = "job_x"
                self.payload = {"transcript": "WORKER STDOUT"}
                self.evidence = []

        class FakeStore:
            def list_jobs(self):
                return [SimpleNamespace(id="job_x", created_at="2026-08-01T00:00:00+00:00")]

            def list_artifacts(self, job_id):
                return [FakeArtifact()]

            def job_dir(self, job_id):
                return Path("/nonexistent")

        with patch(
            "puppetmaster.lifecycle.job_effort_id", return_value="eff-x"
        ), patch(
            "puppetmaster.lifecycle.job_effort_sidecar",
            return_value={"effort_id": "eff-x", "tagged_at": "2026-08-01T00:00:00+00:00"},
        ):
            payload = index_effort_artifacts([FakeStore()])
        self.assertEqual(payload["effort_id"], "eff-x")
        self.assertEqual(payload["refs"], [])
        self.assertNotIn("WORKER STDOUT", json.dumps(payload))

    def test_cli_json_matches_index_and_omits_bodies(self) -> None:
        with TemporaryDirectory() as tmp:
            store = _store(tmp)
            job = store.create_job("cli")
            tag_job_effort(store, job.id, "eff-cli")
            store.save_artifact(_finding(job.id, "cli claim"))
            expected = index_effort_artifacts([store], effort_id="eff-cli")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = cli_main(
                    [
                        "--state-dir",
                        str(Path(tmp) / ".puppetmaster"),
                        "--backend",
                        "file",
                        "effort-index",
                        "--effort",
                        "eff-cli",
                        "--json",
                    ]
                )
            self.assertEqual(code, 0)
            body = json.loads(out.getvalue())
            self.assertEqual(body, expected)
            self.assertNotIn("SECRET-BODY-", out.getvalue())

    def test_mcp_verb_and_cli_argv(self) -> None:
        import puppetmaster.mcp_server as mcp
        from puppetmaster.mcp_remote import SUPERVISE_TOOL_NAMES, tool_allowed

        names = {tool.name for tool in mcp.tools()}
        self.assertIn("puppetmaster_effort_index", names)
        self.assertIn("puppetmaster_effort_index", SUPERVISE_TOOL_NAMES)
        self.assertTrue(tool_allowed("puppetmaster_effort_index", "supervise"))

        captured = []

        def fake_run_cli(command, args):
            captured.append(command)
            return {"content": [], "isError": False}

        with patch.object(mcp, "run_cli", side_effect=fake_run_cli):
            mcp.run_effort_index(
                {
                    "effort_id": "eff-1",
                    "type": "finding",
                    "query": "auth",
                    "expand": True,
                    "all_projects": True,
                }
            )
        self.assertEqual(
            captured[0],
            [
                "effort-index",
                "--json",
                "--effort",
                "eff-1",
                "--type",
                "finding",
                "--query",
                "auth",
                "--expand",
                "--all-projects",
            ],
        )


if __name__ == "__main__":
    unittest.main()
