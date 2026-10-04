"""Worker final reports parse into one artifact per reported item.

Field report (2026-09-30): a 4-role claude-code audit swarm returned 34
findings as fenced JSON with prose after the fence, each keyed ``title`` /
``headline`` + ``symptom`` rather than ``claim``. Every item was dropped as
nameless, so each role's report collapsed into one FINDING whose claim was
literally "```json".
"""

from __future__ import annotations

import json
import unittest

from puppetmaster.adapters.cursor import cursor_artifact_from_item, implement_report_artifacts
from puppetmaster.models import ArtifactType, Task


def _report(items: list) -> str:
    body = json.dumps({"artifacts": items}, indent=2)
    return f"```json\n{body}\n```\n\nNotes: verified by reading the code; tests to add are listed above."


class WorkerReportArtifactsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.task = Task(job_id="job_x", role="audit", instruction="audit", adapter="claude-code")

    def test_fenced_report_with_titled_findings_yields_one_artifact_each(self) -> None:
        items = [
            {"type": "FINDING", "severity": "felt-every-turn", "file": "a.tsx", "line": 3,
             "title": "Caret gets its own line box", "symptom": "Feed jumps 22px", "fix": "Inline it"},
            {"type": "finding", "file": "b.tsx", "headline": "Spinner never resolves", "detail": "No error state"},
            {"type": "FINDING", "file": "c.css", "symptom": "Periwinkle badge beside amber accent"},
        ]
        artifacts = implement_report_artifacts(self.task, "w1", _report(items), adapter="claude-code")
        findings = [a for a in artifacts if a.type == ArtifactType.FINDING]
        self.assertEqual(
            [f.payload["claim"] for f in findings],
            ["Caret gets its own line box", "Spinner never resolves", "Periwinkle badge beside amber accent"],
        )
        self.assertEqual(findings[0].payload["file"], "a.tsx")
        self.assertFalse(any(f.payload.get("claim", "").startswith("```") for f in findings))

    def test_a_typed_item_without_any_headline_is_kept_not_dropped(self) -> None:
        artifact = cursor_artifact_from_item(
            self.task, "w1", {"type": "finding", "file": "d.py", "line": 9, "fix": "Guard at the boundary"},
            adapter="claude-code",
        )
        self.assertIsNotNone(artifact)
        self.assertIn("d.py:9", artifact.payload["claim"])

    def test_artifact_shaped_items_lift_their_nested_payload(self) -> None:
        items = [
            {"type": "finding", "confidence": 0.9,
             "payload": {"claim": "Digest picks verifications by random id", "evidence": ["stitcher.py:205"], "severity": "medium"}},
            {"type": "risk", "payload": {"risk": "Rotation race", "mitigation": "Lock and re-check"}},
            {"type": "decision", "payload": {"decision": "Conditional ship", "why": "Default path regression"}},
        ]
        artifacts = [cursor_artifact_from_item(self.task, "w1", item, adapter="claude-code") for item in items]
        finding, risk, decision = artifacts
        self.assertEqual(finding.payload["claim"], "Digest picks verifications by random id")
        self.assertEqual(finding.evidence, ["stitcher.py:205"])
        self.assertEqual(finding.payload["severity"], "medium")
        self.assertNotIn("payload", finding.payload)
        self.assertEqual(risk.payload["risk"], "Rotation race")
        self.assertEqual(decision.payload["decision"], "Conditional ship")

    def test_top_level_headline_wins_over_a_nested_payload(self) -> None:
        artifact = cursor_artifact_from_item(
            self.task, "w1",
            {"type": "finding", "claim": "Top-level claim", "payload": {"claim": "Nested claim"}},
            adapter="claude-code",
        )
        self.assertEqual(artifact.payload["claim"], "Top-level claim")

    def test_untyped_prose_still_falls_back_to_one_report(self) -> None:
        artifacts = implement_report_artifacts(self.task, "w1", "Fixed the parser.\n\nAll tests pass.", adapter="claude-code")
        findings = [a for a in artifacts if a.type == ArtifactType.FINDING]
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].payload["claim"], "Fixed the parser.")


if __name__ == "__main__":
    unittest.main()


class ImplementAllowedToolsTest(unittest.TestCase):
    """A headless acceptEdits run refuses every unlisted shell command, so a
    full-edit worker could edit but never test or commit (field report
    2026-09-30: three workers returned untested, uncommitted diffs)."""

    def test_write_capable_run_can_verify_and_commit_but_not_push(self) -> None:
        from puppetmaster.adapters.claude_code import build_claude_code_command, implement_allowed_tools

        tools = implement_allowed_tools({}, write_capable=True)
        command = build_claude_code_command(permission_mode="acceptEdits", allowed_tools=tools)
        allowed = command[command.index("--allowedTools") + 1]
        for rule in ("Bash(git commit:*)", "Bash(npx vitest:*)", "Bash(python -m unittest:*)", "Bash(pytest:*)"):
            self.assertIn(rule, allowed)
        self.assertNotIn("git push", allowed)

    def test_explicit_tools_win_and_read_only_runs_get_read_tools(self) -> None:
        from puppetmaster.adapters.claude_code import implement_allowed_tools

        self.assertEqual(implement_allowed_tools({"allowed_tools": ["Read"]}, write_capable=True), ["Read"])
        # dontAsk denies anything not allowlisted, so read-only runs need the read tools.
        self.assertEqual(implement_allowed_tools({}, write_capable=False), ["Read", "Grep", "Glob"])
