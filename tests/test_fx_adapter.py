"""Tests for the fx CLI worker adapter.

Hermetic: the subprocess, git snapshots, and sidecar capture are patched through
the ``puppetmaster.adapters`` facade, so nothing here spawns fx or touches a real
repository. The one place we deliberately do not mock is the JSON contract, which
was captured from a real ``fx ask --json`` run.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

from puppetmaster.adapters import ADAPTERS, get_adapter
from puppetmaster.adapters._streaming import StreamedProcess
from puppetmaster.adapters.fx import (
    DISABLE_MCP_ENV,
    FxAdapter,
    WORKER_DEPTH_ENV,
    _FX_READ_ONLY_PREAMBLE,
    build_fx_command,
    fx_report_text,
    fx_usage_from_result,
    parse_fx_result,
    resolve_fx_permission_mode,
    resolve_fx_read_only_intent,
    resolve_fx_worker_depth,
    resolve_mcp_disabled,
)
from puppetmaster.adapters.registry import ADAPTER_INFO
from puppetmaster.models import ArtifactType, Task
from puppetmaster.platform_lock import KNOWN_ADAPTERS


# The exact shape a real `fx ask --json` run emits, captured from the built binary.
REAL_FX_RESULT = (
    '{"output":"OK","final_output":"OK","exit_code":0,'
    '"model":"deepseek/deepseek-v4.1-flash","session_id":"",'
    '"steps":0,"tool_calls":[],"usage":{"input_tokens":19859,"output_tokens":2}}'
)


def _task(**payload) -> Task:
    return Task(
        job_id="job_fx",
        role="implement",
        instruction="Fix the flaky resize test",
        adapter="fx",
        payload=dict(payload),
    )


def _snapshot_has_diff(snapshot: dict) -> bool:
    return bool(
        str(snapshot.get("diff") or "").strip()
        or snapshot.get("changed_files")
        or snapshot.get("untracked_files")
    )


class FxCommandBuilderTests(unittest.TestCase):
    def test_defaults_are_unattended_and_keep_prompt_out_of_argv(self) -> None:
        cmd = build_fx_command(executable=["fx"])
        self.assertEqual(cmd[:3], ["fx", "ask", "--json"])
        self.assertIn("--auto", cmd)
        self.assertIn("--no-save", cmd)
        self.assertNotIn("--full-access", cmd)
        # The prompt is never argv: it travels on stdin.
        self.assertNotIn("Fix the flaky resize test", cmd)

    def test_full_access_is_opt_in_only(self) -> None:
        cmd = build_fx_command(executable=["fx"], permission_mode="full-access")
        self.assertIn("--full-access", cmd)
        self.assertNotIn("--auto", cmd)

    def test_ask_mode_emits_no_permission_flag(self) -> None:
        cmd = build_fx_command(executable=["fx"], permission_mode="ask")
        self.assertNotIn("--auto", cmd)
        self.assertNotIn("--full-access", cmd)

    def test_rejects_unknown_permission_mode(self) -> None:
        with self.assertRaises(ValueError):
            build_fx_command(executable=["fx"], permission_mode="yolo-nope")

    def test_resume_id_suppresses_no_save(self) -> None:
        # --no-save and --resume-id are mutually exclusive in fx.
        cmd = build_fx_command(executable=["fx"], resume_session_id="sess-1")
        self.assertIn("--resume-id", cmd)
        self.assertIn("sess-1", cmd)
        self.assertNotIn("--no-save", cmd)

    def test_session_is_saved_when_requested(self) -> None:
        cmd = build_fx_command(executable=["fx"], no_save=False)
        self.assertNotIn("--no-save", cmd)

    def test_system_prompt_and_extra_args_forward(self) -> None:
        cmd = build_fx_command(
            executable=["fx"],
            system_prompt="be terse",
            extra_args=["--no-color"],
        )
        self.assertEqual(cmd[cmd.index("--system") + 1], "be terse")
        self.assertIn("--no-color", cmd)

    def test_absolute_executable_with_spaces_survives(self) -> None:
        cmd = build_fx_command(executable=["/opt/my fx/bin/fx"], permission_mode="auto")
        self.assertEqual(cmd[0], "/opt/my fx/bin/fx")


class FxPermissionModeTests(unittest.TestCase):
    def test_default_is_auto_not_full_access(self) -> None:
        self.assertEqual(resolve_fx_permission_mode(_task()), "auto")

    def test_explicit_payload_wins(self) -> None:
        self.assertEqual(
            resolve_fx_permission_mode(_task(permission_mode="full-access")),
            "full-access",
        )

    def test_full_access_boolean_is_honored(self) -> None:
        self.assertEqual(resolve_fx_permission_mode(_task(full_access=True)), "full-access")

    def test_invalid_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_fx_permission_mode(_task(permission_mode="banana"))


class FxReadOnlyIntentTests(unittest.TestCase):
    def test_analysis_no_edit_payload_is_read_only(self) -> None:
        self.assertTrue(
            resolve_fx_read_only_intent(
                _task(read_only=True, sandbox="read-only")
            )
        )

    def test_analyze_mode_is_read_only(self) -> None:
        self.assertTrue(resolve_fx_read_only_intent(_task(mode="analyze")))
        self.assertTrue(resolve_fx_read_only_intent(_task(mode="plan")))

    def test_implement_default_is_write_capable_intent(self) -> None:
        self.assertFalse(resolve_fx_read_only_intent(_task()))

    def test_no_edit_and_dry_run_flags(self) -> None:
        self.assertTrue(resolve_fx_read_only_intent(_task(no_edit=True)))
        self.assertTrue(resolve_fx_read_only_intent(_task(dry_run=True)))


class FxWriteCapablePrepareTests(unittest.TestCase):
    def _prepare(self, **payload):
        adapter = FxAdapter()
        task = _task(cwd="/tmp", disable_codegraph=True, **payload)
        with mock.patch("puppetmaster.adapters.resolve_command", return_value="/usr/bin/fx"), \
             mock.patch("puppetmaster.adapters.with_repo_census", side_effect=lambda p, cwd: p), \
             mock.patch(
                 "puppetmaster.adapters.enrich_prompt_with_codegraph",
                 side_effect=lambda prompt, **kw: (prompt, False),
             ):
            prepared = adapter._prepare_cli_invocation(
                task, "goal", "worker_1", Path("/tmp"), "/usr/bin/fx"
            )
        self.assertIsInstance(prepared, object)
        return prepared

    def test_analysis_payload_sets_write_capable_false(self) -> None:
        prepared = self._prepare(read_only=True, sandbox="read-only")
        self.assertFalse(prepared.extras["write_capable"])
        self.assertTrue(prepared.extras["read_only_intent"])
        self.assertEqual(prepared.extras["enforcement"], "prompt-only")
        self.assertIn("--auto", prepared.command)
        stdin = prepared.subprocess_kwargs["stdin_data"]
        self.assertTrue(str(stdin).startswith(_FX_READ_ONLY_PREAMBLE))

    def test_implement_default_is_write_capable(self) -> None:
        prepared = self._prepare()
        self.assertTrue(prepared.extras["write_capable"])
        self.assertFalse(prepared.extras["read_only_intent"])
        self.assertEqual(prepared.extras["enforcement"], "cli")
        stdin = prepared.subprocess_kwargs["stdin_data"]
        self.assertFalse(str(stdin).startswith(_FX_READ_ONLY_PREAMBLE))

    def test_ask_permission_mode_is_not_write_capable(self) -> None:
        prepared = self._prepare(permission_mode="ask")
        self.assertFalse(prepared.extras["write_capable"])
        self.assertEqual(prepared.extras["enforcement"], "prompt-only")


class FxWorkerDepthTests(unittest.TestCase):
    def test_absent_is_zero(self) -> None:
        self.assertEqual(resolve_fx_worker_depth({}), 0)

    def test_parses_int(self) -> None:
        self.assertEqual(resolve_fx_worker_depth({WORKER_DEPTH_ENV: "3"}), 3)

    def test_garbage_and_negatives_clamp_to_zero(self) -> None:
        self.assertEqual(resolve_fx_worker_depth({WORKER_DEPTH_ENV: "abc"}), 0)
        self.assertEqual(resolve_fx_worker_depth({WORKER_DEPTH_ENV: "-4"}), 0)


class FxResultParsingTests(unittest.TestCase):
    def test_parses_a_real_fx_result(self) -> None:
        parsed = parse_fx_result(REAL_FX_RESULT)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["final_output"], "OK")
        self.assertEqual(parsed["usage"]["input_tokens"], 19859)

    def test_parses_result_wrapped_in_noise(self) -> None:
        noisy = f"starting fx\n{REAL_FX_RESULT}\ndone\n"
        parsed = parse_fx_result(noisy)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["model"], "deepseek/deepseek-v4.1-flash")

    def test_empty_and_non_object_return_none(self) -> None:
        self.assertIsNone(parse_fx_result(""))
        self.assertIsNone(parse_fx_result("   "))
        self.assertIsNone(parse_fx_result("[1, 2, 3]"))
        self.assertIsNone(parse_fx_result("not json at all"))

    def test_usage_maps_onto_pm_token_fields(self) -> None:
        parsed = parse_fx_result(REAL_FX_RESULT)
        self.assertEqual(fx_usage_from_result(parsed), (19859, 2))

    def test_usage_is_zero_without_a_result(self) -> None:
        self.assertEqual(fx_usage_from_result(None), (0, 0))
        self.assertEqual(fx_usage_from_result({"usage": {}}), (0, 0))

    def test_usage_rejects_bool_and_string_tokens(self) -> None:
        # Bools are ints in Python; a truthy flag must not read as a token count.
        self.assertEqual(
            fx_usage_from_result({"usage": {"input_tokens": True, "output_tokens": "7"}}),
            (0, 0),
        )

    def test_report_prefers_final_output(self) -> None:
        self.assertEqual(
            fx_report_text({"output": "streamed", "final_output": "final"}, "raw"),
            "final",
        )

    def test_report_falls_back_to_output_then_empty(self) -> None:
        self.assertEqual(fx_report_text({"output": "streamed"}, "raw"), "streamed")
        self.assertEqual(fx_report_text({}, "raw"), "")
        self.assertEqual(fx_report_text(None, "raw"), "")


class FxRegistrationTests(unittest.TestCase):
    def test_adapter_is_registered(self) -> None:
        self.assertIn("fx", ADAPTERS)
        self.assertIsInstance(get_adapter("fx"), FxAdapter)

    def test_adapter_info_declares_the_cli_requirement(self) -> None:
        info = next(i for i in ADAPTER_INFO if i.name == "fx")
        self.assertEqual(info.status, "optional")
        self.assertTrue(any("fx CLI" in r for r in info.requires))

    def test_platform_lock_knows_fx(self) -> None:
        # fx is billed through whatever provider fx is configured with, so a
        # platform lock must be able to disable it.
        self.assertIn("fx", KNOWN_ADAPTERS)

    def test_fx_is_implement_capable(self) -> None:
        # fx produced a real PATCH artifact in the live drive, so it belongs in
        # the full-edit priority list that start_implement / edit import.
        from puppetmaster.workers import IMPLEMENT_ADAPTER_PRIORITY

        self.assertIn("fx", IMPLEMENT_ADAPTER_PRIORITY)

    def test_fx_is_review_capable(self) -> None:
        # The review-capable capability set and the platform-lock set are kept in
        # lockstep by tests/test_default_reviewer_platform.py, so fx must be in
        # both or neither.
        from puppetmaster.workers import REVIEW_ADAPTERS

        self.assertIn("fx", REVIEW_ADAPTERS)


class FxNestingGuardTests(unittest.TestCase):
    def test_inner_fx_worker_is_blocked(self) -> None:
        adapter = FxAdapter()
        with mock.patch.dict(os.environ, {WORKER_DEPTH_ENV: "1"}):
            artifacts = adapter._prepare_cli_invocation(
                _task(), "goal", "worker_1", Path("/tmp"), "/usr/bin/fx"
            )
        self.assertIsInstance(artifacts, list)
        assert isinstance(artifacts, list)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].type, ArtifactType.VERIFICATION)
        self.assertEqual(artifacts[0].payload["failure"], "nested_fx_worker")
        self.assertEqual(artifacts[0].payload["result"], "blocked")
        self.assertEqual(artifacts[0].payload["depth"], 1)

    def test_outermost_run_is_not_blocked(self) -> None:
        adapter = FxAdapter()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(WORKER_DEPTH_ENV, None)
            with mock.patch("puppetmaster.adapters.resolve_command", return_value="/usr/bin/fx"), \
                 mock.patch("puppetmaster.adapters.with_repo_census", side_effect=lambda p, cwd: p), \
                 mock.patch(
                     "puppetmaster.adapters.enrich_prompt_with_codegraph",
                     return_value=("prompt", False),
                 ):
                prepared = adapter._prepare_cli_invocation(
                    _task(), "goal", "worker_1", Path("/tmp"), "/usr/bin/fx"
                )
        self.assertNotIsInstance(prepared, list)

    def test_explicit_depth_opt_in_allows_one_nested_level(self) -> None:
        adapter = FxAdapter()
        with mock.patch.dict(os.environ, {WORKER_DEPTH_ENV: "1"}), \
             mock.patch("puppetmaster.adapters.with_repo_census", side_effect=lambda p, cwd: p), \
             mock.patch(
                 "puppetmaster.adapters.enrich_prompt_with_codegraph",
                 return_value=("prompt", False),
             ):
            prepared = adapter._prepare_cli_invocation(
                _task(max_worker_depth=1), "goal", "worker_1", Path("/tmp"), "/usr/bin/fx"
            )
        self.assertNotIsInstance(prepared, list)


class FxMcpSuppressionTests(unittest.TestCase):
    """A spawned worker must not inherit the orchestrator's MCP surface."""

    def test_workers_disable_mcp_by_default(self) -> None:
        self.assertTrue(resolve_mcp_disabled(_task()))

    def test_payload_can_opt_back_in(self) -> None:
        self.assertFalse(resolve_mcp_disabled(_task(allow_mcp=True)))

    def test_non_boolean_allow_mcp_falls_back_to_the_safe_default(self) -> None:
        # A string or a number is a malformed payload, not an opt-in.
        for raw in ("true", "1", 1, [], {}):
            with self.subTest(raw=raw):
                self.assertTrue(resolve_mcp_disabled(_task(allow_mcp=raw)))

    def test_invocation_sets_the_env_and_records_the_flag(self) -> None:
        adapter = FxAdapter()
        with mock.patch.dict(os.environ, {}, clear=False), \
             mock.patch("puppetmaster.adapters.with_repo_census", side_effect=lambda p, cwd: p), \
             mock.patch(
                 "puppetmaster.adapters.enrich_prompt_with_codegraph",
                 return_value=("prompt", False),
             ):
            os.environ.pop(WORKER_DEPTH_ENV, None)
            prepared = adapter._prepare_cli_invocation(
                _task(), "goal", "worker_1", Path("/tmp"), "/usr/bin/fx"
            )
        self.assertNotIsInstance(prepared, list)
        self.assertEqual(prepared.env[DISABLE_MCP_ENV], "1")
        self.assertTrue(prepared.extras["mcp_disabled"])
        # Suppression travels in the environment, never as an argv flag an older
        # fx build would reject.
        self.assertNotIn("--no-mcp", prepared.command)

    def test_opt_in_leaves_the_environment_untouched(self) -> None:
        adapter = FxAdapter()
        with mock.patch.dict(os.environ, {}, clear=False), \
             mock.patch("puppetmaster.adapters.with_repo_census", side_effect=lambda p, cwd: p), \
             mock.patch(
                 "puppetmaster.adapters.enrich_prompt_with_codegraph",
                 return_value=("prompt", False),
             ):
            os.environ.pop(WORKER_DEPTH_ENV, None)
            os.environ.pop(DISABLE_MCP_ENV, None)
            prepared = adapter._prepare_cli_invocation(
                _task(allow_mcp=True), "goal", "worker_1", Path("/tmp"), "/usr/bin/fx"
            )
        self.assertNotIsInstance(prepared, list)
        self.assertNotIn(DISABLE_MCP_ENV, prepared.env)
        self.assertFalse(prepared.extras["mcp_disabled"])

    def test_a_stale_ambient_disable_is_not_mistaken_for_a_policy(self) -> None:
        # The worker environment starts from the parent's, so an inherited
        # FX_DISABLE_MCP must not be read back as this adapter's own decision.
        adapter = FxAdapter()
        with mock.patch.dict(os.environ, {DISABLE_MCP_ENV: "1"}):
            prepared = adapter._prepare_cli_invocation(
                _task(allow_mcp=True), "goal", "worker_1", Path("/tmp"), "/usr/bin/fx"
            )
        self.assertFalse(prepared.extras["mcp_disabled"])


class FxLifecycleTests(unittest.TestCase):
    """Full snapshot -> guard -> CLI -> artifact path with the facade patched."""

    def _patch_facade(self, stdout: str, returncode: int = 0, timed_out: bool = False):
        before = {
            "sha": "base000",
            "tree": "tree-base",
            "changed_files": [],
            "untracked_files": [],
            "diff": "",
        }
        after = {
            "sha": "head111",
            "tree": "tree-head",
            "changed_files": ["puppetmaster/foo.py"],
            "untracked_files": [],
            "diff": "diff --git a/puppetmaster/foo.py b/puppetmaster/foo.py\n+x = 1\n",
            "worker_diff": None,
        }
        completed = StreamedProcess(
            returncode=returncode,
            stdout=stdout,
            stderr="",
            timed_out=timed_out,
            live_log_path=None,
        )
        return before, after, completed

    def test_success_run_records_tokens_and_patch(self) -> None:
        before, after, completed = self._patch_facade(REAL_FX_RESULT)
        adapter = FxAdapter()
        with mock.patch("puppetmaster.adapters.resolve_command", return_value="/usr/bin/fx"), \
             mock.patch("puppetmaster.adapters.with_repo_census", side_effect=lambda p, cwd: p), \
             mock.patch(
                 "puppetmaster.adapters.enrich_prompt_with_codegraph",
                 return_value=("prompt", False),
             ), \
             mock.patch("puppetmaster.adapters.worktree_guard", return_value=None), \
             mock.patch("puppetmaster.adapters.snapshot_has_diff", side_effect=_snapshot_has_diff), \
             mock.patch(
                 "puppetmaster.adapters.git_snapshot",
                 side_effect=lambda cwd, base_tree=None: after if base_tree else before,
             ), \
             mock.patch(
                 "puppetmaster.adapters.run_streamed_subprocess", return_value=completed
             ), \
             mock.patch(
                 "puppetmaster.adapters.fx.capture_subprocess_stdout",
                 side_effect=lambda **kw: None,
             ):
            artifacts = adapter.run(_task(cwd="/tmp"), "goal", "worker_1")

        kinds = [a.type for a in artifacts]
        self.assertIn(ArtifactType.VERIFICATION, kinds)
        self.assertIn(ArtifactType.PATCH, kinds)

        verification = next(a for a in artifacts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verification.payload["tokens_in"], 19859)
        self.assertEqual(verification.payload["tokens_out"], 2)
        self.assertEqual(verification.payload["tokens_total"], 19861)
        self.assertEqual(verification.payload["model"], "deepseek/deepseek-v4.1-flash")
        self.assertEqual(verification.payload["permission_mode"], "auto")
        self.assertTrue(verification.payload["usage_reported"])
        self.assertFalse(verification.payload["result_missing"])
        self.assertIsNone(verification.payload["failure"])
        self.assertEqual(verification.payload["base_sha"], "base000")
        self.assertEqual(verification.payload["head_sha"], "head111")

    def test_unparseable_output_is_flagged_not_silently_passed(self) -> None:
        before, after, completed = self._patch_facade("not json", returncode=0)
        adapter = FxAdapter()
        with mock.patch("puppetmaster.adapters.resolve_command", return_value="/usr/bin/fx"), \
             mock.patch("puppetmaster.adapters.with_repo_census", side_effect=lambda p, cwd: p), \
             mock.patch(
                 "puppetmaster.adapters.enrich_prompt_with_codegraph",
                 return_value=("prompt", False),
             ), \
             mock.patch("puppetmaster.adapters.worktree_guard", return_value=None), \
             mock.patch("puppetmaster.adapters.snapshot_has_diff", side_effect=_snapshot_has_diff), \
             mock.patch(
                 "puppetmaster.adapters.git_snapshot",
                 side_effect=lambda cwd, base_tree=None: after if base_tree else before,
             ), \
             mock.patch(
                 "puppetmaster.adapters.run_streamed_subprocess", return_value=completed
             ), \
             mock.patch(
                 "puppetmaster.adapters.fx.capture_subprocess_stdout",
                 side_effect=lambda **kw: None,
             ):
            artifacts = adapter.run(_task(cwd="/tmp"), "goal", "worker_1")

        verification = next(a for a in artifacts if a.type == ArtifactType.VERIFICATION)
        self.assertTrue(verification.payload["result_missing"])
        self.assertEqual(verification.payload["failure"], "fx_unparseable_result")
        self.assertNotEqual(verification.payload["result"], "passed")

    def test_prose_only_write_capable_run_passes_not_degraded(self) -> None:
        """The live-run correction: a coding worker that reports in prose is fine.

        The PATCH artifact is its evidence. Only non-writing ('ask') postures
        treat an unstructured report as degraded.
        """
        prose = json.dumps(
            {
                "output": "I created the file and verified it.",
                "final_output": "I created the file and verified it.",
                "exit_code": 0,
                "model": "deepseek/deepseek-v4.1-flash",
                "session_id": "",
                "steps": 3,
                "tool_calls": [],
                "usage": {"input_tokens": 94998, "output_tokens": 2429},
            }
        )
        before, after, completed = self._patch_facade(prose)
        adapter = FxAdapter()
        with mock.patch("puppetmaster.adapters.resolve_command", return_value="/usr/bin/fx"), \
             mock.patch("puppetmaster.adapters.with_repo_census", side_effect=lambda p, cwd: p), \
             mock.patch(
                 "puppetmaster.adapters.enrich_prompt_with_codegraph",
                 return_value=("prompt", False),
             ), \
             mock.patch("puppetmaster.adapters.worktree_guard", return_value=None), \
             mock.patch("puppetmaster.adapters.snapshot_has_diff", side_effect=_snapshot_has_diff), \
             mock.patch(
                 "puppetmaster.adapters.git_snapshot",
                 side_effect=lambda cwd, base_tree=None: after if base_tree else before,
             ), \
             mock.patch(
                 "puppetmaster.adapters.run_streamed_subprocess", return_value=completed
             ), \
             mock.patch(
                 "puppetmaster.adapters.fx.capture_subprocess_stdout",
                 side_effect=lambda **kw: None,
             ):
            artifacts = adapter.run(_task(cwd="/tmp"), "goal", "worker_1")

        verification = next(a for a in artifacts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verification.payload["result"], "passed")
        self.assertEqual(verification.payload["tokens_in"], 94998)

    def test_prose_only_read_only_run_is_degraded(self) -> None:
        prose = json.dumps({"final_output": "some prose", "exit_code": 0})
        before, after, completed = self._patch_facade(prose)
        adapter = FxAdapter()
        with mock.patch("puppetmaster.adapters.resolve_command", return_value="/usr/bin/fx"), \
             mock.patch("puppetmaster.adapters.with_repo_census", side_effect=lambda p, cwd: p), \
             mock.patch(
                 "puppetmaster.adapters.enrich_prompt_with_codegraph",
                 return_value=("prompt", False),
             ), \
             mock.patch("puppetmaster.adapters.worktree_guard", return_value=None), \
             mock.patch("puppetmaster.adapters.snapshot_has_diff", side_effect=_snapshot_has_diff), \
             mock.patch(
                 "puppetmaster.adapters.git_snapshot",
                 side_effect=lambda cwd, base_tree=None: after if base_tree else before,
             ), \
             mock.patch(
                 "puppetmaster.adapters.run_streamed_subprocess", return_value=completed
             ), \
             mock.patch(
                 "puppetmaster.adapters.fx.capture_subprocess_stdout",
                 side_effect=lambda **kw: None,
             ):
            artifacts = adapter.run(
                _task(cwd="/tmp", permission_mode="ask"), "goal", "worker_1"
            )

        verification = next(a for a in artifacts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verification.payload["result"], "degraded")

    def test_nonzero_exit_is_a_failure_even_with_a_valid_result(self) -> None:
        failing = REAL_FX_RESULT.replace('"exit_code":0', '"exit_code":1')
        before, after, completed = self._patch_facade(failing, returncode=1)
        adapter = FxAdapter()
        with mock.patch("puppetmaster.adapters.resolve_command", return_value="/usr/bin/fx"), \
             mock.patch("puppetmaster.adapters.with_repo_census", side_effect=lambda p, cwd: p), \
             mock.patch(
                 "puppetmaster.adapters.enrich_prompt_with_codegraph",
                 return_value=("prompt", False),
             ), \
             mock.patch("puppetmaster.adapters.worktree_guard", return_value=None), \
             mock.patch("puppetmaster.adapters.snapshot_has_diff", side_effect=_snapshot_has_diff), \
             mock.patch(
                 "puppetmaster.adapters.git_snapshot",
                 side_effect=lambda cwd, base_tree=None: after if base_tree else before,
             ), \
             mock.patch(
                 "puppetmaster.adapters.run_streamed_subprocess", return_value=completed
             ), \
             mock.patch(
                 "puppetmaster.adapters.fx.capture_subprocess_stdout",
                 side_effect=lambda **kw: None,
             ):
            artifacts = adapter.run(_task(cwd="/tmp"), "goal", "worker_1")

        verification = next(a for a in artifacts if a.type == ArtifactType.VERIFICATION)
        self.assertEqual(verification.payload["result"], "failed")
        self.assertEqual(verification.payload["failure"], "fx_exit_code")

    def test_requested_model_is_forwarded_via_environment_not_argv(self) -> None:
        before, after, completed = self._patch_facade(REAL_FX_RESULT)
        adapter = FxAdapter()
        captured: dict = {}

        def _spy(**kwargs):
            captured.update(kwargs)
            return completed

        with mock.patch("puppetmaster.adapters.resolve_command", return_value="/usr/bin/fx"), \
             mock.patch("puppetmaster.adapters.with_repo_census", side_effect=lambda p, cwd: p), \
             mock.patch(
                 "puppetmaster.adapters.enrich_prompt_with_codegraph",
                 return_value=("prompt", False),
             ), \
             mock.patch("puppetmaster.adapters.worktree_guard", return_value=None), \
             mock.patch("puppetmaster.adapters.snapshot_has_diff", side_effect=_snapshot_has_diff), \
             mock.patch(
                 "puppetmaster.adapters.git_snapshot",
                 side_effect=lambda cwd, base_tree=None: after if base_tree else before,
             ), \
             mock.patch("puppetmaster.adapters.run_streamed_subprocess", side_effect=_spy), \
             mock.patch(
                 "puppetmaster.adapters.fx.capture_subprocess_stdout",
                 side_effect=lambda **kw: None,
             ):
            adapter.run(_task(cwd="/tmp", model="some/model"), "goal", "worker_1")

        command = captured["command"]
        self.assertNotIn("some/model", command)
        self.assertEqual(captured["env"]["FX_MODEL"], "some/model")
        # The nesting counter is bumped for every spawned worker.
        self.assertEqual(captured["env"][WORKER_DEPTH_ENV], "1")
        # MCP is off by default, so the worker cannot call back into this PM.
        self.assertEqual(captured["env"][DISABLE_MCP_ENV], "1")


if __name__ == "__main__":
    unittest.main()
