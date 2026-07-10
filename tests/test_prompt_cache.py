"""Anthropic prompt-caching markers for agentic workers."""
from __future__ import annotations

import json
import unittest
from unittest import mock

from puppetmaster import providers


def _count_cache_markers(obj) -> int:
    """Count cache_control dicts anywhere in a request body."""
    if isinstance(obj, dict):
        n = 1 if "cache_control" in obj else 0
        return n + sum(_count_cache_markers(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(_count_cache_markers(v) for v in obj)
    return 0


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "d",
            "parameters": {"type": "object", "properties": {}},
        },
    }


class ApplyAnthropicCacheControlTests(unittest.TestCase):
    def test_system_string_becomes_marked_block_list(self) -> None:
        body = {
            "model": "claude",
            "system": "be terse",
            "messages": [{"role": "user", "content": "go"}],
            "max_tokens": 64,
        }
        out = providers._apply_anthropic_cache_control(body)
        self.assertIsInstance(out["system"], list)
        self.assertEqual(len(out["system"]), 1)
        block = out["system"][0]
        self.assertEqual(block["type"], "text")
        self.assertEqual(block["text"], "be terse")
        # Stable breakpoint: 1h TTL by default.
        self.assertEqual(block["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        # Original body must stay unmarked (helper deep-copies).
        self.assertEqual(body["system"], "be terse")

    def test_last_tool_marked_earlier_tools_not(self) -> None:
        body = {
            "model": "claude",
            "system": "sys",
            "messages": [{"role": "user", "content": "go"}],
            "tools": [
                {"name": "a", "description": "", "input_schema": {"type": "object"}},
                {"name": "b", "description": "", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 64,
        }
        out = providers._apply_anthropic_cache_control(body)
        self.assertNotIn("cache_control", out["tools"][0])
        self.assertEqual(
            out["tools"][1]["cache_control"],
            {"type": "ephemeral", "ttl": "1h"},
        )

    def test_last_and_second_to_last_messages_marked(self) -> None:
        body = {
            "model": "claude",
            "system": "sys",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
                {"role": "user", "content": "third"},
            ],
            "max_tokens": 64,
        }
        out = providers._apply_anthropic_cache_control(body)
        msgs = out["messages"]
        self.assertNotIn("cache_control", str(msgs[0]))
        # Moving history markers omit ttl (Anthropic default 5m write).
        self.assertEqual(msgs[1]["content"][0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("ttl", msgs[1]["content"][0]["cache_control"])
        self.assertEqual(msgs[2]["content"][0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("ttl", msgs[2]["content"][0]["cache_control"])
        # System stays on the stable 1h path.
        self.assertEqual(
            out["system"][0]["cache_control"],
            {"type": "ephemeral", "ttl": "1h"},
        )

    def test_whitespace_only_last_text_block_is_not_marked(self) -> None:
        body = {
            "model": "claude",
            "system": "sys",
            "messages": [
                {"role": "user", "content": "stable"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "ok"},
                        {"type": "text", "text": "   \n"},
                    ],
                },
            ],
            "max_tokens": 64,
        }
        out = providers._apply_anthropic_cache_control(body)
        last = out["messages"][-1]["content"][-1]
        self.assertNotIn("cache_control", last)
        # Second-to-last still gets a moving (no-ttl) marker.
        prev = out["messages"][-2]["content"][0]
        self.assertEqual(prev["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("ttl", prev["cache_control"])

    def test_env_5m_forces_stable_markers_without_ttl(self) -> None:
        body = {
            "model": "claude",
            "system": "sys",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
                {"role": "user", "content": "third"},
            ],
            "tools": [
                {"name": "a", "description": "", "input_schema": {"type": "object"}},
                {"name": "b", "description": "", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 64,
        }
        for force in ("5m", "5", "off", "disabled", "0", "false", "no"):
            with self.subTest(force=force):
                with mock.patch.dict(
                    "os.environ",
                    {"PUPPETMASTER_ANTHROPIC_CACHE_TTL": force},
                ):
                    out = providers._apply_anthropic_cache_control(body)
                self.assertEqual(
                    out["system"][0]["cache_control"],
                    {"type": "ephemeral"},
                )
                self.assertNotIn("ttl", out["system"][0]["cache_control"])
                self.assertEqual(
                    out["tools"][-1]["cache_control"],
                    {"type": "ephemeral"},
                )
                self.assertNotIn("ttl", out["tools"][-1]["cache_control"])
                self.assertEqual(
                    out["messages"][-1]["content"][0]["cache_control"],
                    {"type": "ephemeral"},
                )
                self.assertLessEqual(_count_cache_markers(out), 4)

    def test_at_most_four_markers(self) -> None:
        body = {
            "model": "claude",
            "system": "sys",
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
                {"role": "assistant", "content": "d"},
            ],
            "tools": [
                {"name": "t1", "description": "", "input_schema": {"type": "object"}},
                {"name": "t2", "description": "", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 64,
        }
        out = providers._apply_anthropic_cache_control(body)
        self.assertLessEqual(_count_cache_markers(out), 4)
        self.assertEqual(_count_cache_markers(out), 4)

    def test_kill_switch_leaves_body_unmarked(self) -> None:
        body = {
            "model": "claude",
            "system": "be terse",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"},
            ],
            "tools": [
                {"name": "a", "description": "", "input_schema": {"type": "object"}},
                {"name": "b", "description": "", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 64,
        }
        expected = json.dumps(body, sort_keys=True)
        for off in ("0", "false", "no", "off", "FALSE", "Off"):
            with self.subTest(off=off):
                with mock.patch.dict("os.environ", {"PUPPETMASTER_PROMPT_CACHE": off}):
                    out = providers._apply_anthropic_cache_control(body)
                self.assertEqual(json.dumps(out, sort_keys=True), expected)
                self.assertEqual(_count_cache_markers(out), 0)

    def test_helper_exception_returns_unmarked_body(self) -> None:
        body = {
            "model": "claude",
            "system": "sys",
            "messages": [{"role": "user", "content": "go"}],
            "max_tokens": 64,
        }
        with mock.patch.object(providers.copy, "deepcopy", side_effect=RuntimeError("boom")):
            out = providers._apply_anthropic_cache_control(body)
        self.assertIs(out, body)
        self.assertEqual(out["system"], "sys")
        self.assertEqual(_count_cache_markers(out), 0)


class AnthropicChatCacheIntegrationTests(unittest.TestCase):
    def test_sync_path_applies_markers_and_parses_cache_write(self) -> None:
        canned = {
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 5,
                "cache_read_input_tokens": 40,
                "cache_creation_input_tokens": 60,
            },
        }
        captured = {}

        def _capture(url, *, headers, body, timeout):
            captured["body"] = body
            return canned

        with mock.patch.object(providers, "_post_json", side_effect=_capture):
            turn = providers.provider_chat(
                provider="anthropic",
                model="claude",
                api_key="k",
                messages=[
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "second"},
                    {"role": "user", "content": "third"},
                ],
                tools=[_tool("edit_file"), _tool("read_file")],
            )
        body = captured["body"]
        self.assertIsInstance(body["system"], list)
        self.assertEqual(
            body["system"][0]["cache_control"],
            {"type": "ephemeral", "ttl": "1h"},
        )
        self.assertNotIn("cache_control", body["tools"][0])
        self.assertEqual(
            body["tools"][-1]["cache_control"],
            {"type": "ephemeral", "ttl": "1h"},
        )
        # History markers stay on the default 5m path (no ttl field).
        msgs = body["messages"]
        self.assertEqual(msgs[-2]["content"][0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("ttl", msgs[-2]["content"][0]["cache_control"])
        self.assertEqual(msgs[-1]["content"][0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("ttl", msgs[-1]["content"][0]["cache_control"])
        self.assertLessEqual(_count_cache_markers(body), 4)
        self.assertEqual(turn.usage["cached_tokens"], 40)
        self.assertEqual(turn.usage["cache_write_tokens"], 60)

    def test_kill_switch_sends_unmarked_body_on_wire(self) -> None:
        canned = {
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        captured = {}

        def _capture(url, *, headers, body, timeout):
            captured["body"] = body
            return canned

        with mock.patch.dict("os.environ", {"PUPPETMASTER_PROMPT_CACHE": "0"}):
            with mock.patch.object(providers, "_post_json", side_effect=_capture):
                providers.provider_chat(
                    provider="anthropic",
                    model="claude",
                    api_key="k",
                    messages=[
                        {"role": "system", "content": "be terse"},
                        {"role": "user", "content": "go"},
                    ],
                    tools=[_tool("edit_file")],
                )
        body = captured["body"]
        self.assertEqual(body["system"], "be terse")
        self.assertEqual(_count_cache_markers(body), 0)
        self.assertIsInstance(body["messages"][0]["content"], str)

    def test_helper_failure_still_sends_request(self) -> None:
        canned = {
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        captured = {}

        def _capture(url, *, headers, body, timeout):
            captured["body"] = body
            return canned

        # Force an internal failure inside the helper; try/except must fall
        # back to the unmarked body so the HTTP call still goes out.
        with mock.patch.object(providers.copy, "deepcopy", side_effect=RuntimeError("boom")):
            with mock.patch.object(providers, "_post_json", side_effect=_capture):
                turn = providers.provider_chat(
                    provider="anthropic",
                    model="claude",
                    api_key="k",
                    messages=[
                        {"role": "system", "content": "be terse"},
                        {"role": "user", "content": "go"},
                    ],
                )
        self.assertEqual(turn.text, "ok")
        self.assertEqual(captured["body"]["system"], "be terse")
        self.assertEqual(_count_cache_markers(captured["body"]), 0)

    def test_stream_path_parses_cache_write_tokens(self) -> None:
        lines = [
            (
                b'data: {"type":"message_start","message":{"usage":{'
                b'"input_tokens":100,"cache_read_input_tokens":40,'
                b'"cache_creation_input_tokens":60}}}\n'
            ),
            b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n',
            b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n',
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\n',
            b'data: {"type":"message_stop"}\n',
        ]

        class FakeResp:
            def __iter__(self):
                return iter(lines)

            def close(self):
                pass

        captured = {}

        def _capture(url, *, headers, body, timeout):
            captured["body"] = body
            return FakeResp()

        with mock.patch.object(providers, "_open_stream", side_effect=_capture):
            turn = providers.provider_chat_streaming(
                provider="anthropic",
                model="claude",
                api_key="k",
                messages=[
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "go"},
                ],
                tools=[_tool("edit_file")],
            )
        self.assertEqual(turn.text, "hi")
        self.assertEqual(turn.usage["cached_tokens"], 40)
        self.assertEqual(turn.usage["cache_write_tokens"], 60)
        self.assertIsInstance(captured["body"]["system"], list)
        self.assertEqual(
            captured["body"]["system"][0]["cache_control"],
            {"type": "ephemeral", "ttl": "1h"},
        )


if __name__ == "__main__":
    unittest.main()
