"""Grok Bot contained path: agentic default when Cursor is not runnable.

Hermetic unittest only — no pytest imports.
"""
from __future__ import annotations

import json
import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from puppetmaster.diagnostics import grok_bot_path_check
from puppetmaster.workers import (
    CURSOR_ONLY_LOCK_FIX,
    NoImplementAdapterError,
    agentic_provider_keys_visible,
    pick_implement_adapter,
    pick_swarm_adapter,
)


class AgenticKeyVisibilityTests(unittest.TestCase):
    def test_named_provider_keys(self) -> None:
        self.assertFalse(agentic_provider_keys_visible({}))
        for name in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            self.assertTrue(agentic_provider_keys_visible({name: "k"}), name)


class ImplementPickerGrokBotTests(unittest.TestCase):
    def test_falls_to_agentic_when_cursor_not_runnable(self) -> None:
        enabled = {
            "cursor",
            "claude-code",
            "codex",
            "hermes",
            "antigravity",
            "agentic",
        }
        picked = pick_implement_adapter(
            enabled, is_available=lambda a: a == "agentic"
        )
        self.assertEqual(picked, "agentic")

    def test_prefers_cursor_when_runnable(self) -> None:
        enabled = {"cursor", "agentic", "claude-code"}
        picked = pick_implement_adapter(
            enabled, is_available=lambda a: a in {"cursor", "agentic"}
        )
        self.assertEqual(picked, "cursor")

    def test_does_not_fail_for_missing_cursor_when_agentic_keys(self) -> None:
        enabled = {"cursor", "claude-code", "codex", "hermes", "agentic"}
        # Missing Cursor (and other CLIs) is fine if agentic is runnable.
        picked = pick_implement_adapter(
            enabled, is_available=lambda a: a == "agentic"
        )
        self.assertEqual(picked, "agentic")

    def test_cursor_only_lock_fails_closed_with_fix(self) -> None:
        with self.assertRaises(NoImplementAdapterError) as ctx:
            pick_implement_adapter({"cursor"}, is_available=lambda a: False)
        message = str(ctx.exception)
        self.assertIn("cursor-only", message)
        self.assertIn("enable agentic", message)
        self.assertEqual(ctx.exception.fix, CURSOR_ONLY_LOCK_FIX)
        self.assertIn("unlock", CURSOR_ONLY_LOCK_FIX)
        self.assertIn("install Cursor", CURSOR_ONLY_LOCK_FIX)


class SwarmPickerGrokBotTests(unittest.TestCase):
    def test_swarm_falls_to_agentic_when_cursor_not_runnable(self) -> None:
        picked = pick_swarm_adapter(
            {"cursor", "agentic", "local"},
            is_available=lambda a: a == "agentic",
        )
        self.assertEqual(picked, "agentic")

    def test_swarm_cursor_only_lock_fails_closed(self) -> None:
        with self.assertRaises(NoImplementAdapterError) as ctx:
            pick_swarm_adapter({"cursor", "local"}, is_available=lambda a: False)
        self.assertIn("enable agentic", str(ctx.exception))


class StartImplementMcpGrokBotTests(unittest.TestCase):
    def test_start_implement_uses_agentic_when_cursor_missing(self) -> None:
        from puppetmaster import mcp_server

        captured: dict = {}

        def fake_start_cli(command, args):
            captured["command"] = command
            return {"ok": True}

        with patch(
            "puppetmaster.platform_lock.enabled_adapters",
            return_value={"cursor", "claude-code", "codex", "hermes", "agentic"},
        ), patch(
            "puppetmaster.workers.adapter_is_available",
            side_effect=lambda name, **_: name == "agentic",
        ), patch.object(mcp_server, "_worktree_preflight", return_value=None), patch.object(
            mcp_server, "start_cli", side_effect=fake_start_cli
        ):
            result = mcp_server.start_implement({"goal": "ship it", "cwd": "."})

        self.assertFalse(result.get("isError"))
        self.assertEqual(result["implement_adapter"], "agentic")
        self.assertEqual(captured["command"][0], "agentic")

    def test_start_implement_cursor_only_lock_returns_fix(self) -> None:
        from puppetmaster import mcp_server

        with patch(
            "puppetmaster.platform_lock.enabled_adapters",
            return_value={"cursor"},
        ), patch(
            "puppetmaster.workers.adapter_is_available",
            return_value=False,
        ):
            result = mcp_server.start_implement({"goal": "ship it", "cwd": "."})

        self.assertTrue(result.get("isError"))
        body = json.loads(result["content"][0]["text"])
        self.assertIn("enable agentic", body.get("fix", ""))
        self.assertIn("cursor-only", body["error"])

    def test_start_prewalk_does_not_fail_for_missing_cursor(self) -> None:
        from puppetmaster import mcp_server

        with patch(
            "puppetmaster.platform_lock.enabled_adapters",
            return_value={"cursor", "agentic"},
        ), patch(
            "puppetmaster.workers.adapter_is_available",
            side_effect=lambda name, **_: name == "agentic",
        ), patch.object(mcp_server, "start_cli", return_value={"ok": True}):
            result = mcp_server.start_prewalk(
                {"goal": "plan then ship", "cwd": ".", "allow_non_worktree": True}
            )
        self.assertFalse(result.get("isError"))


class DoctorGrokBotPathTests(unittest.TestCase):
    def test_healthy_when_cursor_missing_and_agentic_keys_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            check = grok_bot_path_check(
                Path(tmp),
                env={"OPENAI_API_KEY": "sk-test"},
                enabled={"cursor", "agentic", "claude-code"},
            )
        self.assertEqual(check.status, "ok")
        self.assertIn("healthy Grok Bot path", check.detail)
        self.assertIn("agentic", check.detail)

    def test_cursor_only_lock_fails_closed_with_fix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            check = grok_bot_path_check(
                Path(tmp),
                env={},
                enabled={"cursor"},
            )
        self.assertEqual(check.status, "error")
        self.assertIn("enable agentic", check.detail)
        self.assertIn("unlock", check.detail)
        self.assertIn("install Cursor", check.detail)


if __name__ == "__main__":
    unittest.main()
