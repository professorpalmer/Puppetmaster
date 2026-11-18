from __future__ import annotations

import unittest

from puppetmaster.preflight import adapter_cli_present


def _only(name):
    return lambda candidate: f"/usr/bin/{candidate}" if candidate == name else None


class MultiWordCommandTests(unittest.TestCase):
    """Bug 06d11534-1: a multi-word command resolves only its first word."""

    def test_claude_code_npx_command_reads_as_present(self):
        env = {"CLAUDE_CODE_COMMAND": "npx -y @anthropic-ai/claude-code"}
        self.assertTrue(adapter_cli_present("claude-code", env=env, resolver=_only("npx")))

    def test_codex_multi_word_command_reads_as_present(self):
        env = {"CODEX_COMMAND": "npx -y @openai/codex"}
        self.assertTrue(adapter_cli_present("codex", env=env, resolver=_only("npx")))

    def test_multi_word_command_with_missing_first_word_is_absent(self):
        env = {"CLAUDE_CODE_COMMAND": "npx -y @anthropic-ai/claude-code"}
        self.assertFalse(adapter_cli_present("claude-code", env=env, resolver=_only("claude")))

    def test_single_word_command_keeps_old_result(self):
        env = {"CLAUDE_CODE_COMMAND": "claude-next"}
        self.assertTrue(adapter_cli_present("claude-code", env=env, resolver=_only("claude-next")))
        self.assertFalse(adapter_cli_present("claude-code", env=env, resolver=_only("claude")))

    def test_default_command_keeps_old_result(self):
        self.assertTrue(adapter_cli_present("claude-code", env={}, resolver=_only("claude")))
        self.assertFalse(adapter_cli_present("claude-code", env={}, resolver=lambda _n: None))

    def test_adapter_without_cli_keeps_old_result(self):
        self.assertTrue(adapter_cli_present("openai", env={}, resolver=lambda _n: None))


if __name__ == "__main__":
    unittest.main()
