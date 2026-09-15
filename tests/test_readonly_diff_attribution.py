"""Concurrent repository edits are not output from read-only CLI workers."""
import json
import os
import sys
import unittest
from unittest import mock

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

from puppetmaster.adapters.codex import CodexAdapter
from puppetmaster.adapters.claude_code import ClaudeCodeAdapter
from puppetmaster.adapters.antigravity import AntigravityAdapter
from puppetmaster.adapters.fx import FxAdapter
from puppetmaster.adapters._git import GitSnapshot
from puppetmaster.adapters._streaming import StreamedProcess
from puppetmaster.models import ArtifactType, Task


class ReadonlyDiffAttributionTests(unittest.TestCase):
    def test_concurrent_changes_across_cli_outcomes(self):
        cases = [
            (CodexAdapter, {"sandbox": "read-only"}, {"sandbox": "workspace-write"}),
            (ClaudeCodeAdapter, {"permission_mode": "plan"}, {"permission_mode": "acceptEdits"}),
            (AntigravityAdapter, {"mode": "plan"}, {"mode": "implement"}),
            (FxAdapter, {"read_only": True, "sandbox": "read-only"}, {"permission_mode": "auto"}),
        ]
        for adapter_type, readonly, writable in cases:
            for outcome in ("success", "failure", "timeout", "budget"):
                for write_capable, settings in ((False, readonly), (True, writable)):
                    with self.subTest(adapter=adapter_type.__name__, outcome=outcome, write_capable=write_capable):
                        before = GitSnapshot("base", True, [], [], "", tree="base-tree")
                        after = GitSnapshot("other-commit", True, ["other.py"], ["new.py"], "+ambient",
                                            tree="other-tree", worker_changed_files=["other.py"],
                                            worker_untracked_files=["new.py"], worker_diff="+ambient")
                        adapter = adapter_type()
                        task = Task(job_id="job", id="task", role="explore", instruction="Inspect code",
                                    payload={**settings, "allow_dirty": True, "disable_codegraph": True})
                        report = json.dumps({"artifacts": [{
                            "type": "finding", "claim": "Inspected code",
                            "evidence": ["module.py:1"], "confidence": 0.8,
                        }]})
                        stdout = report
                        if adapter_type is CodexAdapter:
                            stdout = json.dumps({"type": "item.completed", "item": {
                                "type": "agent_message", "text": report,
                            }})
                        elif adapter_type is FxAdapter:
                            stdout = json.dumps({
                                "output": report,
                                "final_output": report,
                                "exit_code": 0 if outcome == "success" else 1,
                                "model": "test-model",
                                "session_id": "",
                                "steps": 0,
                                "tool_calls": [],
                                "usage": {"input_tokens": 1, "output_tokens": 1},
                            })
                        completed = StreamedProcess(
                            returncode=0 if outcome == "success" else 1,
                            stdout=stdout, stderr="",
                            timed_out=outcome == "timeout", output_limit_hit=outcome == "budget")
                        with mock.patch.object(adapter, "_resolve_cli_executable", return_value=("cli", "/fake/cli")), \
                             mock.patch.object(adapter, "_invoke_cli", return_value=completed), \
                             mock.patch("puppetmaster.adapters.git_snapshot", side_effect=[before, after]), \
                             mock.patch("puppetmaster.adapters.worktree_guard", return_value=None):
                            artifacts = adapter.run(task, "Inspect code", "worker")
                        patches = [a for a in artifacts if a.type == ArtifactType.PATCH]
                        self.assertEqual(bool(patches), write_capable)
                        verifications = [a for a in artifacts if a.type == ArtifactType.VERIFICATION]
                        self.assertTrue(verifications)
                        if outcome == "success":
                            self.assertEqual(verifications[0].payload["result"], "passed")
                        if not write_capable:
                            for artifact in verifications:
                                for field in ("worker_diff_present", "base_sha", "head_sha", "changed_files", "untracked_files"):
                                    self.assertNotIn(field, artifact.payload)
                                self.assertEqual(artifact.payload["repository_diff_attribution"], "none")
                        else:
                            self.assertEqual(patches[0].payload["unified_diff"], "+ambient")
