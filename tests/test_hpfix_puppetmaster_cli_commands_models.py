from __future__ import annotations

import contextlib
import io
import json
import os
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from puppetmaster.cli import main as cli_main
from puppetmaster.model_registry import ModelSpec, load_registry, save_registry


def _set_effort(spec: ModelSpec, level: str) -> dict:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "models.json"
        save_registry([spec], path)
        with contextlib.redirect_stdout(io.StringIO()):
            code = cli_main(
                ["models", "set", "--registry-path", str(path), spec.id, f"effort={level}"]
            )
        assert code == 0
        return {s.id: s for s in load_registry(path)}[spec.id].payload_defaults


class ModelsSetEffortKeepsExtraArgsTests(unittest.TestCase):
    def test_claude_code_effort_keeps_other_extra_args(self) -> None:
        spec = ModelSpec(
            id="claude/opus",
            adapter="claude-code",
            adapter_model_name="opus",
            payload_defaults={"extra_args": ["--foo", "--effort", "low"], "temperature": 0.2},
        )
        self.assertEqual(
            _set_effort(spec, "high"),
            {"extra_args": ["--foo", "--effort", "high"], "temperature": 0.2},
        )

    def test_codex_effort_keeps_other_extra_args(self) -> None:
        spec = ModelSpec(
            id="codex/gpt",
            adapter="codex",
            adapter_model_name="gpt-5.5",
            payload_defaults={
                "extra_args": ["-c", "model_reasoning_effort=low", "-c", "n=2"]
            },
        )
        self.assertEqual(
            _set_effort(spec, "high"),
            {"extra_args": ["-c", "n=2", "-c", "model_reasoning_effort=high"]},
        )

    def test_effort_without_existing_extra_args_is_unchanged(self) -> None:
        spec = ModelSpec(
            id="claude/opus",
            adapter="claude-code",
            adapter_model_name="opus",
            payload_defaults={"temperature": 0.2},
        )
        self.assertEqual(
            _set_effort(spec, "high"),
            {"temperature": 0.2, "extra_args": ["--effort", "high"]},
        )

    def test_reasoning_effort_adapter_leaves_extra_args_alone(self) -> None:
        spec = ModelSpec(
            id="openai/gpt",
            adapter="openai",
            adapter_model_name="gpt-5.5",
            payload_defaults={"extra_args": ["--foo"]},
        )
        self.assertEqual(
            _set_effort(spec, "high"),
            {"extra_args": ["--foo"], "reasoning_effort": "high"},
        )


def _sync(path: Path, *flags: str) -> "tuple[int, str]":
    err = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
        code = cli_main(["models", "sync", "--registry-path", str(path), *flags])
    return code, err.getvalue()


class ModelsSyncFileErrorTests(unittest.TestCase):
    """A registry that cannot be written gives one error line and exit code 1, not a traceback."""

    def test_an_os_error_of_the_sync_is_an_error_line(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            path.write_text("[]", encoding="utf-8")
            refused = PermissionError(13, "Permission denied", str(path))
            with patch("puppetmaster.registry_sync.sync_starter_registry", side_effect=refused):
                code, err = _sync(path, "--apply")
        self.assertEqual(code, 1)
        self.assertTrue(err.startswith("error: "), err)
        self.assertIn("Permission denied", err)

    @unittest.skipIf(os.name == "nt", "a folder mode does not stop a write on Windows")
    def test_apply_in_a_folder_that_is_not_writable(self) -> None:
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root can write to each folder")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            path.write_text("[]", encoding="utf-8")
            os.chmod(tmp, stat.S_IRUSR | stat.S_IXUSR)
            try:
                code, err = _sync(path, "--apply")
            finally:
                os.chmod(tmp, stat.S_IRWXU)
            self.assertEqual(code, 1)
            self.assertTrue(err.startswith("error: "), err)
            self.assertEqual(path.read_text(encoding="utf-8"), "[]")


if __name__ == "__main__":
    unittest.main()
