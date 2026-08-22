"""Regression coverage for CLI state routing when a worker has ``--cwd``."""

from __future__ import annotations

import contextlib
import io
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import puppetmaster.cli._dispatch as dispatch
from puppetmaster.state import resolve_state_dir


class CliStateRoutingTests(unittest.TestCase):
    """The launcher shell and a worker workspace can be different projects."""

    @contextlib.contextmanager
    def _launcher_cwd(self, path: Path):
        old_cwd = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(old_cwd)

    def _start_detached_swarm(self, target: Path, *prefix: str) -> MagicMock:
        """Start the CLI path far enough to capture state-store/config routing."""
        detached = MagicMock(
            return_value={"job_id": "job_state_route", "launcher_pid": 4242}
        )
        store = MagicMock()
        argv = [*prefix, "swarm", "state routing", "--cwd", str(target)]
        with patch.object(dispatch, "create_store", return_value=store) as create_store, patch(
            "puppetmaster.platform_lock.is_adapter_enabled", return_value=True
        ), patch("puppetmaster.swarm_launch.detach_analysis_swarm", detached), contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(dispatch._main(argv), 0)
        self.create_store = create_store
        return detached

    def _assert_launcher_relative_state(
        self, actual: Path, launcher: Path, name: str
    ) -> None:
        """Compare path identity across symlink and Windows short-name aliases."""
        self.assertEqual(actual.name, name)
        self.assertTrue(actual.parent.samefile(launcher))

    def test_detached_swarm_uses_target_workspace_for_store_and_launcher_config(self) -> None:
        """A nested target repo must not persist under the launcher folder."""
        with TemporaryDirectory() as tmp:
            outer = Path(tmp) / "puppetmaster"
            target = outer / "Puppetmaster"
            target.mkdir(parents=True)
            with self._launcher_cwd(outer), patch(
                "puppetmaster.state.app_state_root", return_value=outer / "app-state"
            ):
                expected = resolve_state_dir(cwd=target)
                launcher_state = resolve_state_dir(cwd=outer)
                detached = self._start_detached_swarm(target)

            self.assertNotEqual(expected, launcher_state)
            self.assertEqual(self.create_store.call_args.args[1], expected)
            self.assertEqual(detached.call_args.kwargs["state_dir"], expected)
            self.assertEqual(detached.call_args.kwargs["cwd"], str(target))

    def test_all_worker_commands_default_state_to_their_cwd(self) -> None:
        """Every write-side CLI command with --cwd shares the routing rule."""
        commands = [
            ["cursor", "prompt"],
            ["claude", "prompt"],
            ["openai", "prompt"],
            ["codex", "prompt"],
            ["hermes", "prompt"],
            ["agentic", "prompt"],
            ["swarm", "goal"],
            ["browser", "mission"],
            ["edit", "instruction"],
            ["prewalk", "goal"],
        ]
        with TemporaryDirectory() as tmp:
            outer = Path(tmp) / "launcher"
            target = outer / "target"
            target.mkdir(parents=True)
            with self._launcher_cwd(outer), patch(
                "puppetmaster.state.app_state_root", return_value=outer / "app-state"
            ):
                expected = resolve_state_dir(cwd=target)
                for command in commands:
                    with self.subTest(command=command[0]):
                        args = dispatch.build_parser().parse_args(
                            [*command, "--cwd", str(target)]
                        )
                        self.assertEqual(dispatch._resolve_command_state_dir(args), expected)

    def test_absolute_cli_state_dir_still_overrides_worker_cwd(self) -> None:
        """An explicit absolute --state-dir retains its existing precedence."""
        with TemporaryDirectory() as tmp:
            outer = Path(tmp) / "launcher"
            target = outer / "target"
            explicit = Path(tmp) / "explicit-state"
            target.mkdir(parents=True)
            outer.mkdir(exist_ok=True)
            with self._launcher_cwd(outer):
                detached = self._start_detached_swarm(target, "--state-dir", str(explicit))

            self.assertEqual(self.create_store.call_args.args[1], explicit)
            self.assertEqual(detached.call_args.kwargs["state_dir"], explicit)

    def test_relative_cli_state_dir_stays_relative_to_launcher_shell(self) -> None:
        """Changing default routing does not reinterpret an explicit relative path."""
        with TemporaryDirectory() as tmp:
            outer = Path(tmp) / "launcher"
            target = outer / "target"
            target.mkdir(parents=True)
            with self._launcher_cwd(outer):
                detached = self._start_detached_swarm(target, "--state-dir", "relative-state")

            self._assert_launcher_relative_state(
                self.create_store.call_args.args[1], outer, "relative-state"
            )
            self._assert_launcher_relative_state(
                detached.call_args.kwargs["state_dir"], outer, "relative-state"
            )

    def test_relative_environment_state_dir_stays_relative_to_launcher_shell(self) -> None:
        """The environment override keeps its precedence and path interpretation."""
        with TemporaryDirectory() as tmp:
            outer = Path(tmp) / "launcher"
            target = outer / "target"
            target.mkdir(parents=True)
            with self._launcher_cwd(outer), patch.dict(
                os.environ, {"PUPPETMASTER_STATE_DIR": "environment-state"}
            ):
                detached = self._start_detached_swarm(target)

            self._assert_launcher_relative_state(
                self.create_store.call_args.args[1], outer, "environment-state"
            )
            self._assert_launcher_relative_state(
                detached.call_args.kwargs["state_dir"], outer, "environment-state"
            )

    def test_absolute_environment_state_dir_still_overrides_worker_cwd(self) -> None:
        """An absolute environment override retains its existing precedence."""
        with TemporaryDirectory() as tmp:
            outer = Path(tmp) / "launcher"
            target = outer / "target"
            explicit = Path(tmp) / "environment-state"
            target.mkdir(parents=True)
            with self._launcher_cwd(outer), patch.dict(
                os.environ, {"PUPPETMASTER_STATE_DIR": str(explicit)}
            ):
                detached = self._start_detached_swarm(target)

            self.assertEqual(self.create_store.call_args.args[1], explicit)
            self.assertEqual(detached.call_args.kwargs["state_dir"], explicit)


if __name__ == "__main__":
    unittest.main()
