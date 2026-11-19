from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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


if __name__ == "__main__":
    unittest.main()
