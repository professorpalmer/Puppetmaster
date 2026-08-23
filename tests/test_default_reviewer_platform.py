"""Focused tests for user-configurable default reviewer platform.

Tests cover:
- Resolution precedence: explicit > configured default > fail-closed
- Configured default use when no explicit adapter
- Unset fail-closed behavior with actionable error
- Disabled/unavailable configured defaults fail clearly
- Platform-specific review commands remain platform-specific
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from puppetmaster import mcp_server, platform_lock
from puppetmaster.cli._parser import build_parser
from puppetmaster.cli._dispatch import _main as run_cli_main
from puppetmaster.cli.commands_platform import _run_platform_subcommand
from puppetmaster.workers import (
    NoReviewAdapterError,
    REVIEW_ADAPTERS,
    pick_review_adapter,
)


class DefaultReviewerPlatformTests(TestCase):
    def test_explicit_adapter_wins_over_configured_default(self) -> None:
        """Explicit adapter selection takes precedence over configured default."""
        enabled = {"cursor", "claude-code", "codex"}
        # Configured default is cursor, but explicit request is claude-code
        adapter = pick_review_adapter(
            enabled, requested="claude-code", configured_default="cursor"
        )
        self.assertEqual(adapter, "claude-code")

    def test_configured_default_used_when_no_explicit_request(self) -> None:
        """Configured default reviewer is used when no explicit adapter requested."""
        enabled = {"cursor", "claude-code", "codex"}
        adapter = pick_review_adapter(
            enabled,
            requested=None,
            configured_default="cursor",
            is_available=lambda _adapter: True,
        )
        self.assertEqual(adapter, "cursor")

    def test_fail_closed_when_no_configured_default_and_no_request(self) -> None:
        """Fail closed with actionable error when no default configured and no explicit request."""
        enabled = {"cursor", "claude-code"}
        with self.assertRaises(NoReviewAdapterError) as ctx:
            pick_review_adapter(enabled, requested=None, configured_default=None)
        exc = ctx.exception
        self.assertEqual(exc.enabled, enabled)
        self.assertIsNone(exc.requested)
        self.assertIsNone(exc.configured_default)
        self.assertIn("No default reviewer configured", str(exc))
        self.assertIn("platform reviewer", str(exc))

    def test_fail_closed_lists_enabled_platforms(self) -> None:
        """Fail-closed error lists enabled valid reviewer platforms."""
        enabled = {"cursor", "codex", "hermes"}
        with self.assertRaises(NoReviewAdapterError) as ctx:
            pick_review_adapter(enabled, requested=None, configured_default=None)
        exc = ctx.exception
        error_msg = str(exc)
        self.assertIn("cursor", error_msg)
        self.assertIn("codex", error_msg)
        self.assertIn("hermes", error_msg)

    def test_explicit_disabled_adapter_fails_clearly(self) -> None:
        """Explicit request for disabled adapter fails with clear error."""
        enabled = {"cursor"}  # codex is disabled
        with self.assertRaises(NoReviewAdapterError) as ctx:
            pick_review_adapter(enabled, requested="codex", configured_default=None)
        exc = ctx.exception
        self.assertEqual(exc.requested, "codex")
        self.assertIn("disabled by the platform lock", str(exc))

    def test_configured_default_disabled_fails_clearly(self) -> None:
        """Configured default that is disabled fails with clear error and remediation."""
        enabled = {"cursor"}  # codex is disabled
        with self.assertRaises(NoReviewAdapterError) as ctx:
            pick_review_adapter(enabled, requested=None, configured_default="codex")
        exc = ctx.exception
        self.assertEqual(exc.configured_default, "codex")
        self.assertIn("disabled by the platform lock", str(exc))
        self.assertIn("platform enable codex", str(exc))
        self.assertIn("platform reviewer", str(exc))

    def test_configured_default_unavailable_fails_clearly(self) -> None:
        """Configured default that is enabled but unavailable fails with clear error."""
        enabled = {"cursor", "codex"}

        def always_unavailable(adapter: str) -> bool:
            return False

        with self.assertRaises(NoReviewAdapterError) as ctx:
            pick_review_adapter(
                enabled,
                requested=None,
                configured_default="cursor",
                is_available=always_unavailable,
            )
        exc = ctx.exception
        self.assertEqual(exc.configured_default, "cursor")
        self.assertIn("not available", str(exc))
        self.assertIn("CLI/credentials missing", str(exc))

    def test_explicit_request_not_review_capable_fails(self) -> None:
        """Explicit request for non-review-capable adapter fails."""
        enabled = {"cursor", "local"}  # local is not a billable review platform
        with self.assertRaises(NoReviewAdapterError) as ctx:
            pick_review_adapter(enabled, requested="local", configured_default=None)
        exc = ctx.exception
        self.assertEqual(exc.requested, "local")
        self.assertIn("cannot review", str(exc))

    def test_configured_default_not_review_capable_fails(self) -> None:
        """Configured default that is not review-capable fails."""
        enabled = {"cursor", "shell"}
        with self.assertRaises(NoReviewAdapterError) as ctx:
            pick_review_adapter(enabled, requested=None, configured_default="shell")
        exc = ctx.exception
        self.assertEqual(exc.configured_default, "shell")
        self.assertIn("not review-capable", str(exc))
        self.assertIn("platform reviewer", str(exc))

    def test_no_enabled_review_platforms_fails(self) -> None:
        """No enabled review-capable platforms fails with clear guidance."""
        enabled = set()  # no platforms enabled
        with self.assertRaises(NoReviewAdapterError) as ctx:
            pick_review_adapter(enabled, requested=None, configured_default=None)
        exc = ctx.exception
        self.assertIn("No review-capable platform is enabled", str(exc))
        self.assertIn("platform enable", str(exc))

    def test_all_billable_analysis_platforms_can_be_reviewers(self) -> None:
        self.assertEqual(set(REVIEW_ADAPTERS), set(platform_lock.KNOWN_ADAPTERS))


class GenericReviewDispatchTests(TestCase):
    def _start_result(self) -> dict:
        return {"content": [{"type": "text", "text": "{}"}], "isError": False}

    @patch.object(mcp_server, "start_cli")
    @patch.object(mcp_server, "write_generated_swarm_config")
    @patch("puppetmaster.platform_lock.get_default_reviewer", return_value="cursor")
    @patch("puppetmaster.platform_lock.enabled_adapters", return_value={"cursor", "codex"})
    def test_explicit_platform_overrides_configured_default(
        self, _enabled, _default, write_config, start_cli
    ) -> None:
        write_config.return_value = Path("review.json")
        start_cli.return_value = self._start_result()

        result = mcp_server.start_review(
            {"goal": "Review the parser", "cwd": ".", "adapter": "codex"}
        )

        self.assertFalse(result["isError"])
        roles = [{"role": "review", "instruction": "Review the parser"}]
        write_config.assert_called_once()
        self.assertEqual(write_config.call_args.args[1:], (roles, "codex"))
        command = start_cli.call_args.args[0]
        self.assertEqual(command[:2], ["run", "Review the parser"])
        self.assertNotIn("--review", command)
        self.assertNotIn("--dry-run", command)

    @patch.object(mcp_server, "start_cli")
    @patch.object(mcp_server, "write_generated_swarm_config")
    @patch("puppetmaster.workers.adapter_is_available", return_value=True)
    @patch("puppetmaster.platform_lock.get_default_reviewer", return_value="openai")
    @patch("puppetmaster.platform_lock.enabled_adapters", return_value={"openai", "cursor"})
    def test_configured_default_uses_platform_neutral_analysis_runtime(
        self, _enabled, _default, _available, write_config, start_cli
    ) -> None:
        write_config.return_value = Path("review.json")
        start_cli.return_value = self._start_result()

        result = mcp_server.start_review({"goal": "Review routing", "cwd": "."})

        self.assertFalse(result["isError"])
        self.assertEqual(write_config.call_args.args[2], "openai")
        self.assertEqual(start_cli.call_args.args[0][0], "run")

    @patch.object(mcp_server, "start_cli")
    @patch("puppetmaster.platform_lock.get_default_reviewer", return_value=None)
    @patch("puppetmaster.platform_lock.enabled_adapters", return_value={"cursor", "codex"})
    def test_unset_default_fails_without_starting_job(
        self, _enabled, _default, start_cli
    ) -> None:
        result = mcp_server.start_review({"goal": "Review routing", "cwd": "."})

        self.assertTrue(result["isError"])
        self.assertIn("No default reviewer configured", result["content"][0]["text"])
        self.assertIn("platform reviewer <platform>", result["content"][0]["text"])
        start_cli.assert_not_called()

    @patch.object(mcp_server, "start_cli")
    @patch("puppetmaster.platform_lock.get_default_reviewer", return_value="codex")
    @patch("puppetmaster.platform_lock.enabled_adapters", return_value={"cursor"})
    def test_disabled_configured_default_fails_without_fallback(
        self, _enabled, _default, start_cli
    ) -> None:
        result = mcp_server.start_review({"goal": "Review routing", "cwd": "."})

        self.assertTrue(result["isError"])
        self.assertIn("disabled by the platform lock", result["content"][0]["text"])
        self.assertIn('"configured_default": "codex"', result["content"][0]["text"])
        start_cli.assert_not_called()

    @patch.object(mcp_server, "start_cli")
    @patch("puppetmaster.workers.adapter_is_available", return_value=False)
    @patch("puppetmaster.platform_lock.get_default_reviewer", return_value="codex")
    @patch("puppetmaster.platform_lock.enabled_adapters", return_value={"cursor", "codex"})
    def test_unavailable_configured_default_fails_without_fallback(
        self, _enabled, _default, _available, start_cli
    ) -> None:
        result = mcp_server.start_review({"goal": "Review routing", "cwd": "."})

        self.assertTrue(result["isError"])
        self.assertIn("enabled but not available", result["content"][0]["text"])
        start_cli.assert_not_called()

    @patch.object(mcp_server, "start_cli")
    @patch("puppetmaster.platform_lock.get_default_reviewer", return_value="cursor")
    @patch("puppetmaster.platform_lock.enabled_adapters", return_value={"cursor", "codex"})
    def test_conflicting_adapter_and_platform_fails_closed(
        self, _enabled, _default, start_cli
    ) -> None:
        result = mcp_server.start_review(
            {
                "goal": "Review routing",
                "cwd": ".",
                "adapter": "cursor",
                "platform": "codex",
            }
        )

        self.assertTrue(result["isError"])
        self.assertIn("pass only one", result["content"][0]["text"])
        start_cli.assert_not_called()


class PlatformLockDefaultReviewerTests(TestCase):
    def test_get_default_reviewer_when_unset(self) -> None:
        """get_default_reviewer returns None when no default is configured."""
        with TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "models.json"
            registry_path.touch()
            result = platform_lock.get_default_reviewer(registry_path)
            self.assertIsNone(result)

    def test_set_and_get_default_reviewer(self) -> None:
        """set_default_reviewer persists and get_default_reviewer retrieves it."""
        with TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "models.json"
            registry_path.touch()
            platform_lock.set_default_reviewer("cursor", registry_path)
            result = platform_lock.get_default_reviewer(registry_path)
            self.assertEqual(result, "cursor")

    def test_set_default_reviewer_canonicalizes_alias(self) -> None:
        """set_default_reviewer canonicalizes adapter aliases."""
        with TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "models.json"
            registry_path.touch()
            # agy is an alias for antigravity
            platform_lock.set_default_reviewer("agy", registry_path)
            result = platform_lock.get_default_reviewer(registry_path)
            self.assertEqual(result, "antigravity")

    def test_set_default_reviewer_none_clears_default(self) -> None:
        """set_default_reviewer with None clears the configured default."""
        with TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "models.json"
            registry_path.touch()
            platform_lock.set_default_reviewer("cursor", registry_path)
            self.assertEqual(platform_lock.get_default_reviewer(registry_path), "cursor")
            platform_lock.set_default_reviewer(None, registry_path)
            self.assertIsNone(platform_lock.get_default_reviewer(registry_path))

    def test_set_default_reviewer_preserves_other_platform_json_fields(self) -> None:
        """set_default_reviewer preserves other fields in platform.json."""
        with TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "models.json"
            registry_path.touch()
            # Disable some adapters first
            platform_lock.disable({"codex"}, registry_path)
            disabled_before = platform_lock._read_disabled(registry_path)
            # Set default reviewer
            platform_lock.set_default_reviewer("cursor", registry_path)
            # Verify disabled list is preserved
            disabled_after = platform_lock._read_disabled(registry_path)
            self.assertEqual(disabled_before, disabled_after)
            self.assertIn("codex", disabled_after)

    def test_get_default_reviewer_returns_none_on_malformed_json(self) -> None:
        """get_default_reviewer returns None if platform.json is malformed."""
        with TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "models.json"
            registry_path.touch()
            config_path = platform_lock.platform_config_path(registry_path)
            config_path.write_text("not valid json", encoding="utf-8")
            result = platform_lock.get_default_reviewer(registry_path)
            self.assertIsNone(result)

    def test_set_default_reviewer_rejects_unknown_platform(self) -> None:
        with TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "models.json"
            with self.assertRaisesRegex(ValueError, "unknown reviewer platform"):
                platform_lock.set_default_reviewer("not-a-platform", registry_path)


class DefaultReviewerCliTests(TestCase):
    def test_cli_sets_reports_and_clears_default_reviewer(self) -> None:
        parser = build_parser()
        with TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "models.json"

            set_args = parser.parse_args(
                [
                    "platform",
                    "reviewer",
                    "codex",
                    "--registry-path",
                    str(registry_path),
                ]
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(_run_platform_subcommand(set_args), 0)
            self.assertEqual(platform_lock.get_default_reviewer(registry_path), "codex")

            show_args = parser.parse_args(
                [
                    "platform",
                    "reviewer",
                    "--registry-path",
                    str(registry_path),
                ]
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(_run_platform_subcommand(show_args), 0)
            self.assertIn("Default reviewer platform: codex", output.getvalue())

            status_args = parser.parse_args(
                [
                    "platform",
                    "status",
                    "--json",
                    "--registry-path",
                    str(registry_path),
                ]
            )
            status_output = io.StringIO()
            with redirect_stdout(status_output):
                self.assertEqual(_run_platform_subcommand(status_args), 0)
            self.assertIn('"default_reviewer": "codex"', status_output.getvalue())

            clear_args = parser.parse_args(
                [
                    "platform",
                    "reviewer",
                    "--clear",
                    "--registry-path",
                    str(registry_path),
                ]
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(_run_platform_subcommand(clear_args), 0)
            self.assertIsNone(platform_lock.get_default_reviewer(registry_path))

    def test_cli_rejects_platform_and_clear_together(self) -> None:
        parser = build_parser()
        with TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "models.json"
            args = parser.parse_args(
                [
                    "platform",
                    "reviewer",
                    "cursor",
                    "--clear",
                    "--registry-path",
                    str(registry_path),
                ]
            )
            error = io.StringIO()
            with redirect_stderr(error):
                self.assertEqual(_run_platform_subcommand(args), 1)
            self.assertIn("not both", error.getvalue())
            self.assertIsNone(platform_lock.get_default_reviewer(registry_path))

    @patch("puppetmaster.swarm_launch.detach_analysis_swarm")
    @patch("puppetmaster.workers.adapter_is_available", return_value=True)
    @patch("puppetmaster.platform_lock.get_default_reviewer", return_value="codex")
    @patch("puppetmaster.platform_lock.enabled_adapters", return_value={"cursor", "codex"})
    def test_generic_review_cli_uses_configured_default(
        self, _enabled, _default, _available, detach
    ) -> None:
        detach.return_value = {"job_id": "job_review"}
        with TemporaryDirectory() as tmpdir:
            output = io.StringIO()
            with redirect_stdout(output):
                result = run_cli_main(
                    [
                        "--state-dir",
                        str(Path(tmpdir) / "state"),
                        "review",
                        "Review routing",
                        "--cwd",
                        tmpdir,
                        "--json",
                    ]
                )

        self.assertEqual(result, 0)
        self.assertEqual(detach.call_args.kwargs["adapter"], "codex")
        self.assertEqual(detach.call_args.kwargs["roles"], ["review"])
        self.assertIn("job_review", output.getvalue())

    @patch("puppetmaster.swarm_launch.detach_analysis_swarm")
    @patch("puppetmaster.platform_lock.get_default_reviewer", return_value=None)
    @patch("puppetmaster.platform_lock.enabled_adapters", return_value={"cursor", "codex"})
    def test_generic_review_cli_fails_closed_when_unset(
        self, _enabled, _default, detach
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            error = io.StringIO()
            with redirect_stderr(error):
                result = run_cli_main(
                    [
                        "--state-dir",
                        str(Path(tmpdir) / "state"),
                        "review",
                        "Review routing",
                        "--cwd",
                        tmpdir,
                    ]
                )

        self.assertEqual(result, 2)
        self.assertIn("No default reviewer configured", error.getvalue())
        detach.assert_not_called()


class ReviewAdapterAvailabilityTests(TestCase):
    def test_available_adapter_is_selected(self) -> None:
        """When configured default is available, it is selected."""
        enabled = {"cursor", "codex"}

        def cursor_available(adapter: str) -> bool:
            return adapter == "cursor"

        adapter = pick_review_adapter(
            enabled,
            requested=None,
            configured_default="cursor",
            is_available=cursor_available,
        )
        self.assertEqual(adapter, "cursor")

    def test_unavailable_configured_default_fails(self) -> None:
        """When configured default is unavailable, fail with clear error."""
        enabled = {"cursor", "codex"}

        def none_available(adapter: str) -> bool:
            return False

        with self.assertRaises(NoReviewAdapterError) as ctx:
            pick_review_adapter(
                enabled,
                requested=None,
                configured_default="cursor",
                is_available=none_available,
            )
        exc = ctx.exception
        self.assertIn("not available", str(exc))
        self.assertIn("CLI/credentials missing", str(exc))

    def test_explicit_request_skips_availability_check(self) -> None:
        """Explicit adapter request is honored even if unavailable (fails later with precise error)."""
        enabled = {"cursor", "codex"}

        def none_available(adapter: str) -> bool:
            return False

        # Explicit request is honored regardless of availability
        adapter = pick_review_adapter(
            enabled,
            requested="cursor",
            configured_default=None,
            is_available=none_available,
        )
        self.assertEqual(adapter, "cursor")
