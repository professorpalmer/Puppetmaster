"""Prompt-caching markers for agentic workers (Anthropic + OpenAI-wire)."""
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
        # All-1h policy: history markers get ttl:1h alongside system/tools.
        self.assertEqual(
            msgs[1]["content"][0]["cache_control"],
            {"type": "ephemeral", "ttl": "1h"},
        )
        self.assertEqual(
            msgs[2]["content"][0]["cache_control"],
            {"type": "ephemeral", "ttl": "1h"},
        )
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
        # Second-to-last still gets a history marker (all-1h).
        prev = out["messages"][-2]["content"][0]
        self.assertEqual(prev["cache_control"], {"type": "ephemeral", "ttl": "1h"})

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
        # History markers follow all-1h (ttl:1h) alongside system/tools.
        msgs = body["messages"]
        self.assertEqual(
            msgs[-2]["content"][0]["cache_control"],
            {"type": "ephemeral", "ttl": "1h"},
        )
        self.assertEqual(
            msgs[-1]["content"][0]["cache_control"],
            {"type": "ephemeral", "ttl": "1h"},
        )
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


class OpenAIExplicitCacheKindTests(unittest.TestCase):
    def test_claude_slugs(self) -> None:
        for model in (
            "anthropic/claude-sonnet-4",
            "claude-3-5-sonnet",
            "openrouter/anthropic/claude-3.5-sonnet",
        ):
            with self.subTest(model=model):
                self.assertEqual(providers._openai_explicit_cache_kind(model), "claude")

    def test_qwen_slugs(self) -> None:
        for model in (
            "qwen/qwen3-coder",
            "alibaba/qwen-max",
            "qwen-plus",
            "qwen2.5-coder-32b",
        ):
            with self.subTest(model=model):
                self.assertEqual(providers._openai_explicit_cache_kind(model), "qwen")

    def test_automatic_providers_unmarked(self) -> None:
        for model in (
            "gpt-4o",
            "openai/gpt-4.1",
            "google/gemini-2.5-pro",
            "deepseek/deepseek-chat",
            "x-ai/grok-3",
            "moonshotai/kimi-k2",
        ):
            with self.subTest(model=model):
                self.assertEqual(providers._openai_explicit_cache_kind(model), "")


class ApplyOpenAIExplicitCacheTests(unittest.TestCase):
    def _claude_body(self) -> dict:
        return {
            "model": "anthropic/claude-sonnet-4",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
                {"role": "user", "content": "third"},
            ],
            "tools": [_tool("edit_file"), _tool("read_file")],
        }

    def test_claude_gets_all_1h_including_history(self) -> None:
        body = self._claude_body()
        out = providers._apply_openai_explicit_cache(
            body,
            model="anthropic/claude-sonnet-4",
            base_url="https://openrouter.ai/api/v1",
        )
        sys_msg = out["messages"][0]
        self.assertEqual(
            sys_msg["content"][0]["cache_control"],
            {"type": "ephemeral", "ttl": "1h"},
        )
        self.assertNotIn("cache_control", out["tools"][0])
        self.assertEqual(
            out["tools"][-1]["cache_control"],
            {"type": "ephemeral", "ttl": "1h"},
        )
        # History markers also get ttl:1h (all-1h policy).
        self.assertEqual(
            out["messages"][-2]["content"][0]["cache_control"],
            {"type": "ephemeral", "ttl": "1h"},
        )
        self.assertEqual(
            out["messages"][-1]["content"][0]["cache_control"],
            {"type": "ephemeral", "ttl": "1h"},
        )
        # Prefer per-block markers; do not set conflicting top-level cache_control.
        self.assertNotIn("cache_control", out)
        self.assertIn("session_id", out)
        self.assertEqual(len(out["session_id"]), 32)
        # Original body stays unmarked.
        self.assertEqual(body["messages"][0]["content"], "be terse")

    def test_gpt_model_is_not_marked(self) -> None:
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "go"},
            ],
            "tools": [_tool("edit_file")],
        }
        expected = json.dumps(body, sort_keys=True)
        out = providers._apply_openai_explicit_cache(
            body,
            model="gpt-4o",
            base_url="https://openrouter.ai/api/v1",
        )
        self.assertEqual(json.dumps(out, sort_keys=True), expected)
        self.assertEqual(_count_cache_markers(out), 0)
        self.assertNotIn("session_id", out)

    def test_qwen_gets_ephemeral_without_ttl(self) -> None:
        body = {
            "model": "qwen/qwen3-coder",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
                {"role": "user", "content": "third"},
            ],
            "tools": [_tool("edit_file"), _tool("read_file")],
        }
        out = providers._apply_openai_explicit_cache(
            body,
            model="qwen/qwen3-coder",
            base_url="https://openrouter.ai/api/v1",
        )
        for marker in (
            out["messages"][0]["content"][0]["cache_control"],
            out["tools"][-1]["cache_control"],
            out["messages"][-2]["content"][0]["cache_control"],
            out["messages"][-1]["content"][0]["cache_control"],
        ):
            self.assertEqual(marker, {"type": "ephemeral"})
            self.assertNotIn("ttl", marker)
        self.assertIn("session_id", out)

    def test_kill_switch_leaves_openai_body_unmarked(self) -> None:
        body = self._claude_body()
        expected = json.dumps(body, sort_keys=True)
        with mock.patch.dict("os.environ", {"PUPPETMASTER_PROMPT_CACHE": "0"}):
            out = providers._apply_openai_explicit_cache(
                body,
                model="anthropic/claude-sonnet-4",
                base_url="https://openrouter.ai/api/v1",
            )
        self.assertEqual(json.dumps(out, sort_keys=True), expected)
        self.assertEqual(_count_cache_markers(out), 0)

    def test_env_5m_forces_claude_all_markers_without_ttl(self) -> None:
        body = self._claude_body()
        with mock.patch.dict(
            "os.environ",
            {"PUPPETMASTER_ANTHROPIC_CACHE_TTL": "5m"},
        ):
            out = providers._apply_openai_explicit_cache(
                body,
                model="anthropic/claude-sonnet-4",
                base_url="https://api.openai.com/v1",
            )
        for marker in (
            out["messages"][0]["content"][0]["cache_control"],
            out["tools"][-1]["cache_control"],
            out["messages"][-2]["content"][0]["cache_control"],
            out["messages"][-1]["content"][0]["cache_control"],
        ):
            self.assertEqual(marker, {"type": "ephemeral"})
            self.assertNotIn("ttl", marker)
        # Non-OpenRouter hosts do not get session_id sticky routing.
        self.assertNotIn("session_id", out)

    def test_explicit_session_id_from_extra(self) -> None:
        body = self._claude_body()
        out = providers._apply_openai_explicit_cache(
            body,
            model="anthropic/claude-sonnet-4",
            base_url="https://openrouter.ai/api/v1",
            extra={"session_id": "sticky-loop-1"},
        )
        self.assertEqual(out["session_id"], "sticky-loop-1")


class OpenAIChatCacheIntegrationTests(unittest.TestCase):
    def test_openrouter_claude_sync_path_applies_markers(self) -> None:
        canned = {
            "choices": [{
                "message": {"content": "ok", "tool_calls": []},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "total_tokens": 105,
                "prompt_tokens_details": {"cached_tokens": 40},
            },
        }
        captured = {}

        def _capture(url, *, headers, body, timeout):
            captured["body"] = body
            return canned

        with mock.patch.object(providers, "_post_json", side_effect=_capture):
            turn = providers.provider_chat(
                provider="openrouter",
                model="anthropic/claude-sonnet-4",
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
        self.assertEqual(
            body["messages"][0]["content"][0]["cache_control"],
            {"type": "ephemeral", "ttl": "1h"},
        )
        self.assertEqual(
            body["tools"][-1]["cache_control"],
            {"type": "ephemeral", "ttl": "1h"},
        )
        self.assertEqual(
            body["messages"][-1]["content"][0]["cache_control"],
            {"type": "ephemeral", "ttl": "1h"},
        )
        self.assertNotIn("cache_control", body)
        self.assertIn("session_id", body)
        self.assertEqual(turn.usage["cached_tokens"], 40)

    def test_openrouter_gpt_sync_path_unmarked(self) -> None:
        canned = {
            "choices": [{
                "message": {"content": "ok", "tool_calls": []},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        captured = {}

        def _capture(url, *, headers, body, timeout):
            captured["body"] = body
            return canned

        with mock.patch.object(providers, "_post_json", side_effect=_capture):
            providers.provider_chat(
                provider="openrouter",
                model="openai/gpt-4o",
                api_key="k",
                messages=[
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "go"},
                ],
                tools=[_tool("edit_file")],
            )
        body = captured["body"]
        self.assertEqual(_count_cache_markers(body), 0)
        self.assertIsInstance(body["messages"][0]["content"], str)
        self.assertNotIn("session_id", body)

    def test_openrouter_qwen_sync_path_ephemeral_only(self) -> None:
        canned = {
            "choices": [{
                "message": {"content": "ok", "tool_calls": []},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        captured = {}

        def _capture(url, *, headers, body, timeout):
            captured["body"] = body
            return canned

        with mock.patch.object(providers, "_post_json", side_effect=_capture):
            providers.provider_chat(
                provider="openrouter",
                model="qwen/qwen3-coder",
                api_key="k",
                messages=[
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "go"},
                ],
                tools=[_tool("edit_file")],
            )
        body = captured["body"]
        self.assertEqual(
            body["messages"][0]["content"][0]["cache_control"],
            {"type": "ephemeral"},
        )
        self.assertNotIn("ttl", body["messages"][0]["content"][0]["cache_control"])
        self.assertEqual(
            body["tools"][-1]["cache_control"],
            {"type": "ephemeral"},
        )
        self.assertNotIn("ttl", body["tools"][-1]["cache_control"])
        self.assertIn("session_id", body)

    def test_stream_path_applies_markers_for_claude(self) -> None:
        lines = [
            (
                b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}\n'
            ),
            (
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                b'"usage":{"prompt_tokens":10,"completion_tokens":1,'
                b'"total_tokens":11,"prompt_tokens_details":{"cached_tokens":4}}}\n'
            ),
            b"data: [DONE]\n",
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
                provider="openrouter",
                model="anthropic/claude-sonnet-4",
                api_key="k",
                messages=[
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "go"},
                ],
                tools=[_tool("edit_file")],
            )
        self.assertEqual(turn.text, "hi")
        self.assertEqual(turn.usage["cached_tokens"], 4)
        self.assertEqual(
            captured["body"]["messages"][0]["content"][0]["cache_control"],
            {"type": "ephemeral", "ttl": "1h"},
        )
        self.assertIn("session_id", captured["body"])


if __name__ == "__main__":
    unittest.main()
