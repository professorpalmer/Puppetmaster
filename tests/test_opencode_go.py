"""OpenCode Go: descriptor, flat-id normalization, per-model endpoint routing,
Responses transport, upstream-block classification, and agentic selection.

Hermetic: no real network. Provider HTTP is stubbed at ``providers._post_json``
/ ``providers._open_stream``.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from unittest import mock

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

from puppetmaster import opencode_go as go
from puppetmaster import providers
from puppetmaster.failure import (
    NOT_AUTHENTICATED,
    SERVER_ERROR,
    classify_provider_failure,
    is_upstream_provider_block,
)


def _sse_response(events: list[dict]):
    """Build a fake urllib streaming response from Responses SSE event dicts."""
    lines = [
        f"data: {json.dumps(event)}\n".encode("utf-8") for event in events
    ]

    class FakeResp:
        def __iter__(self):
            return iter(lines)

        def close(self):
            pass

    return FakeResp()


class OpenCodeGoCatalogTests(unittest.TestCase):
    def test_curated_catalog_is_flat_and_current(self) -> None:
        self.assertTrue(go.CURATED_MODELS)
        for model in go.CURATED_MODELS:
            self.assertNotIn("/", model, msg=f"{model} carries a vendor namespace")
            self.assertEqual(model, model.lower())

    def test_deepseek_entry_is_v4_not_stale_alias(self) -> None:
        self.assertIn("deepseek-v4-flash", go.CURATED_MODELS)
        stale = [
            m for m in go.CURATED_MODELS
            if m.startswith(("deepseek-v3", "deepseek-v2"))
        ]
        self.assertEqual(stale, [])

    def test_curated_catalog_covers_published_endpoint_table(self) -> None:
        for model in (
            "grok-4.6", "grok-4.5", "gpt-5.6-luna", "glm-5.2", "glm-5.1", "kimi-k3",
            "kimi-k2.7-code", "kimi-k2.6", "mimo-v2.5", "mimo-v2.5-pro",
            "minimax-m3", "minimax-m2.7", "qwen3.7-max", "qwen3.7-plus",
            "qwen3.6-plus", "deepseek-v4-pro", "deepseek-v4-flash", "hy3",
        ):
            self.assertIn(model, go.CURATED_MODELS)


class OpenCodeGoNormalizeAndRouteTests(unittest.TestCase):
    def test_normalize_model_id(self) -> None:
        cases = (
            ("kimi-k3", "kimi-k3"),
            ("opencode-go/kimi-k3", "kimi-k3"),
            ("moonshotai/kimi-k2.7-code", "kimi-k2.7-code"),
            ("  deepseek-v4-flash  ", "deepseek-v4-flash"),
            ("z-ai/glm-5.2", "glm-5.2"),
            ("", ""),
            (None, ""),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(go.normalize_model_id(raw), expected)

    def test_normalize_preserves_dots(self) -> None:
        self.assertEqual(
            go.normalize_model_id("opencode-go/mimo-v2.5-pro"),
            "mimo-v2.5-pro",
        )

    def test_api_mode_for_model(self) -> None:
        cases = (
            ("glm-5.2", go.CHAT_COMPLETIONS),
            ("kimi-k3", go.CHAT_COMPLETIONS),
            ("deepseek-v4-flash", go.CHAT_COMPLETIONS),
            ("mimo-v2.5-pro", go.CHAT_COMPLETIONS),
            ("grok-4.5", go.CHAT_COMPLETIONS),
            ("grok-4.6", go.CHAT_COMPLETIONS),
            ("hy3", go.CHAT_COMPLETIONS),
            ("minimax-m3", go.ANTHROPIC_MESSAGES),
            ("minimax-m2.7", go.ANTHROPIC_MESSAGES),
            ("qwen3.7-max", go.ANTHROPIC_MESSAGES),
            ("qwen3.6-plus", go.ANTHROPIC_MESSAGES),
            ("gpt-5.6-luna", go.OPENAI_RESPONSES),
            ("opencode-go/gpt-5.6-luna", go.OPENAI_RESPONSES),
            ("", go.CHAT_COMPLETIONS),
        )
        for model, mode in cases:
            with self.subTest(model=model):
                self.assertEqual(go.api_mode_for_model(model), mode)

    def test_every_curated_model_routes_somewhere_known(self) -> None:
        modes = {go.api_mode_for_model(m) for m in go.CURATED_MODELS}
        self.assertTrue(
            modes
            <= {go.CHAT_COMPLETIONS, go.ANTHROPIC_MESSAGES, go.OPENAI_RESPONSES}
        )

    def test_driver_base_url_heals_stripped_v1(self) -> None:
        self.assertEqual(
            go.driver_base_url("https://opencode.ai/zen/go"),
            go.BASE_URL,
        )
        self.assertEqual(go.driver_base_url(go.BASE_URL), go.BASE_URL)
        self.assertEqual(go.driver_base_url(go.BASE_URL + "/"), go.BASE_URL)
        self.assertEqual(go.driver_base_url(None), go.BASE_URL)

    def test_driver_base_url_leaves_custom_relays_alone(self) -> None:
        self.assertEqual(
            go.driver_base_url("https://relay.internal/go"),
            "https://relay.internal/go",
        )

    def test_mimo_pro_output_ceiling(self) -> None:
        self.assertEqual(go.max_tokens_for_model("mimo-v2.5-pro", 262144), 131072)
        self.assertEqual(
            go.max_tokens_for_model("opencode-go/mimo-v2.5-pro", 262144),
            131072,
        )
        self.assertEqual(go.max_tokens_for_model("mimo-v2.5-pro", 8000), 8000)
        self.assertEqual(go.max_tokens_for_model("kimi-k3", 262144), 262144)

    def test_glm_5_2_reasoning_dialect(self) -> None:
        self.assertEqual(go.reasoning_body_extras("glm-5.2", "none"), {})
        self.assertEqual(
            go.reasoning_body_extras("glm-5.2", "low"),
            {"reasoning_effort": "high"},
        )
        self.assertEqual(
            go.reasoning_body_extras("glm-5.2", "xhigh"),
            {"reasoning_effort": "max"},
        )
        for alias in ("glm-5-2", "glm-5p2", "opencode-go/glm-5.2"):
            self.assertEqual(
                go.reasoning_body_extras(alias, "high"),
                {"reasoning_effort": "high"},
            )

    def test_kimi_and_deepseek_reasoning_dialects(self) -> None:
        self.assertEqual(
            go.reasoning_body_extras("kimi-k2.6", "none"),
            {"thinking": {"type": "disabled"}},
        )
        self.assertEqual(
            go.reasoning_body_extras("kimi-k2.6", "medium"),
            {"reasoning_effort": "medium"},
        )
        self.assertEqual(
            go.reasoning_body_extras("deepseek-v4-flash", "max"),
            {"reasoning_effort": "max"},
        )

    def test_reasoning_never_sends_thinking_and_effort_together(self) -> None:
        for model in ("kimi-k2.6", "deepseek-v4-pro", "glm-5.2"):
            for effort in ("none", "low", "medium", "high", "xhigh", "max"):
                extras = go.reasoning_body_extras(model, effort)
                self.assertFalse(
                    "thinking" in extras and "reasoning_effort" in extras,
                    msg=f"{model}/{effort}: {extras}",
                )

    def test_models_without_reasoning_knob_send_nothing(self) -> None:
        for model in ("mimo-v2.5-pro", "grok-4.5", "hy3", "minimax-m3"):
            self.assertEqual(go.reasoning_body_extras(model, "high"), {})


class OpenCodeGoProviderDescriptorTests(unittest.TestCase):
    def test_provider_is_registered_with_subscription_key(self) -> None:
        desc = providers.get_provider("opencode-go")
        self.assertIsNotNone(desc)
        assert desc is not None
        self.assertEqual(desc.api_key_env_vars, ("OPENCODE_GO_API_KEY",))
        self.assertEqual(desc.base_url, go.BASE_URL)
        self.assertEqual(desc.wire, "openai")
        self.assertEqual(desc.label, "OpenCode Go")

    def test_available_from_subscription_key(self) -> None:
        self.assertIn(
            "opencode-go",
            providers.available_providers({"OPENCODE_GO_API_KEY": "sk-go-test"}),
        )
        self.assertNotIn("opencode-go", providers.available_providers({}))

    def test_key_pool_includes_numbered_siblings(self) -> None:
        pool = providers.provider_key_pool(
            "opencode-go",
            env={
                "OPENCODE_GO_API_KEY": "sk-a",
                "OPENCODE_GO_API_KEY_2": "sk-b",
            },
        )
        self.assertEqual(pool, ["sk-a", "sk-b"])

    def test_resolve_api_key_and_base_url(self) -> None:
        desc = providers.get_provider("opencode-go")
        assert desc is not None
        self.assertEqual(
            providers.resolve_api_key(desc, {"OPENCODE_GO_API_KEY": " sk-go "}),
            "sk-go",
        )
        self.assertEqual(
            providers.resolve_base_url(
                desc, {"OPENCODE_GO_BASE_URL": "https://opencode.ai/zen/go"}
            ),
            go.BASE_URL,
        )


class OpenCodeGoRequestConstructionTests(unittest.TestCase):
    def test_chat_completions_url_headers_body(self) -> None:
        captured: dict = {}

        def fake_post(url, *, headers, body, timeout):
            captured.update(url=url, headers=headers, body=body, timeout=timeout)
            return {
                "choices": [{
                    "message": {"content": "ok", "role": "assistant"},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }

        with mock.patch.object(providers, "_post_json", side_effect=fake_post):
            turn = providers.provider_chat(
                provider="opencode-go",
                model="opencode-go/deepseek-v4-flash",
                messages=[{"role": "user", "content": "hi"}],
                api_key="sk-go-test",
                extra={"max_tokens": 2048, "reasoning_effort": "medium"},
            )
        self.assertEqual(
            captured["url"],
            f"{go.BASE_URL}/chat/completions",
        )
        self.assertEqual(
            captured["headers"]["Authorization"],
            "Bearer sk-go-test",
        )
        self.assertEqual(captured["body"]["model"], "deepseek-v4-flash")
        self.assertEqual(captured["body"]["reasoning_effort"], "medium")
        self.assertEqual(turn.text, "ok")

    def test_anthropic_messages_url_for_minimax(self) -> None:
        captured: dict = {}

        def fake_post(url, *, headers, body, timeout):
            captured.update(url=url, headers=headers, body=body)
            return {
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        with mock.patch.object(providers, "_post_json", side_effect=fake_post):
            providers.provider_chat(
                provider="opencode-go",
                model="minimax-m3",
                messages=[{"role": "user", "content": "hi"}],
                api_key="sk-go-test",
                extra={"max_tokens": 1024},
            )
        self.assertEqual(captured["url"], f"{go.BASE_URL}/messages")
        self.assertEqual(captured["headers"]["x-api-key"], "sk-go-test")
        self.assertEqual(captured["body"]["model"], "minimax-m3")
        self.assertEqual(captured["body"]["max_tokens"], 1024)

    def test_qwen_uses_messages_not_chat_completions(self) -> None:
        urls: list[str] = []

        def fake_post(url, *, headers, body, timeout):
            urls.append(url)
            return {
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        with mock.patch.object(providers, "_post_json", side_effect=fake_post):
            providers.provider_chat(
                provider="opencode-go",
                model="qwen3.7-plus",
                messages=[{"role": "user", "content": "hi"}],
                api_key="sk-go-test",
            )
        self.assertEqual(urls, [f"{go.BASE_URL}/messages"])

    def test_gpt_luna_uses_responses_not_chat_completions(self) -> None:
        captured: dict = {}

        def fake_post(url, *, headers, body, timeout):
            captured.update(url=url, headers=headers, body=body)
            return {
                "status": "completed",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "luna-ok"}],
                }],
                "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
            }

        with mock.patch.object(providers, "_post_json", side_effect=fake_post):
            turn = providers.provider_chat(
                provider="opencode-go",
                model="gpt-5.6-luna",
                messages=[
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "hi"},
                ],
                api_key="sk-go-test",
                extra={"max_tokens": 1024},
            )
        self.assertEqual(captured["url"], f"{go.BASE_URL}/responses")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer sk-go-test")
        self.assertNotIn("originator", captured["headers"])
        self.assertNotIn("ChatGPT-Account-ID", captured["headers"])
        body = captured["body"]
        self.assertEqual(body["model"], "gpt-5.6-luna")
        self.assertEqual(body["instructions"], "Be brief.")
        self.assertEqual(body["max_output_tokens"], 1024)
        self.assertEqual(body.get("store"), False)
        self.assertNotIn("client_metadata", body)
        self.assertNotIn("messages", body)
        self.assertEqual(
            body["input"],
            [{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            }],
        )
        self.assertEqual(turn.text, "luna-ok")
        self.assertEqual(turn.usage["prompt_tokens"], 2)
        self.assertEqual(turn.usage["completion_tokens"], 3)

    def test_gpt_luna_never_posts_to_chat_completions(self) -> None:
        urls: list[str] = []

        def fake_post(url, *, headers, body, timeout):
            urls.append(url)
            return {
                "status": "completed",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}],
                }],
                "usage": {},
            }

        with mock.patch.object(providers, "_post_json", side_effect=fake_post):
            providers.provider_chat(
                provider="opencode-go",
                model="opencode-go/gpt-5.6-luna",
                messages=[{"role": "user", "content": "hi"}],
                api_key="sk-go-test",
            )
        self.assertEqual(urls, [f"{go.BASE_URL}/responses"])

    def test_retired_deepseek_fails_closed(self) -> None:
        with mock.patch.object(providers, "_post_json") as post:
            with self.assertRaises(providers.ProviderError) as ctx:
                providers.provider_chat(
                    provider="opencode-go",
                    model="deepseek-v3",
                    messages=[],
                    api_key="sk-go-test",
                )
        post.assert_not_called()
        self.assertEqual(ctx.exception.reason, "unsupported_model")

    def test_mimo_pro_clamps_max_tokens_on_wire(self) -> None:
        captured: dict = {}

        def fake_post(url, *, headers, body, timeout):
            captured["body"] = body
            return {
                "choices": [{
                    "message": {"content": "ok"},
                    "finish_reason": "stop",
                }],
                "usage": {},
            }

        with mock.patch.object(providers, "_post_json", side_effect=fake_post):
            providers.provider_chat(
                provider="opencode-go",
                model="mimo-v2.5-pro",
                messages=[{"role": "user", "content": "hi"}],
                api_key="sk-go-test",
                extra={"max_tokens": 262144},
            )
        self.assertEqual(captured["body"]["max_tokens"], 131072)

    def test_glm_reasoning_dialect_on_wire(self) -> None:
        captured: dict = {}

        def fake_post(url, *, headers, body, timeout):
            captured["body"] = body
            return {
                "choices": [{
                    "message": {"content": "ok"},
                    "finish_reason": "stop",
                }],
                "usage": {},
            }

        with mock.patch.object(providers, "_post_json", side_effect=fake_post):
            providers.provider_chat(
                provider="opencode-go",
                model="glm-5.2",
                messages=[{"role": "user", "content": "hi"}],
                api_key="sk-go-test",
                extra={"reasoning_effort": "max"},
            )
        self.assertEqual(captured["body"]["reasoning_effort"], "max")
        self.assertNotIn("thinking", captured["body"])


class OpenCodeGoResponsesTransportTests(unittest.TestCase):
    def test_supports_wire_includes_responses(self) -> None:
        self.assertTrue(go.supports_wire_in_puppetmaster("gpt-5.6-luna"))
        self.assertTrue(go.supports_wire_in_puppetmaster("deepseek-v4-flash"))
        self.assertTrue(go.supports_wire_in_puppetmaster("minimax-m3"))
        self.assertFalse(go.supports_wire_in_puppetmaster("deepseek-v3"))
        self.assertNotIn("does not implement", go.unsupported_model_message("gpt-5.6-luna"))

    def test_message_and_tool_conversion(self) -> None:
        instructions, items = providers._messages_to_responses_input([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "ask"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"a.py"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "file body",
            },
            {"role": "assistant", "content": "done"},
        ])
        self.assertEqual(instructions, "sys")
        self.assertEqual(items[0]["type"], "message")
        self.assertEqual(items[0]["role"], "user")
        self.assertEqual(items[1], {
            "type": "function_call",
            "call_id": "call_1",
            "name": "read_file",
            "arguments": '{"path":"a.py"}',
        })
        self.assertEqual(items[2], {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "file body",
        })
        self.assertEqual(items[3]["content"][0]["type"], "output_text")

        tools = providers._tools_to_responses([{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }])
        self.assertEqual(tools, [{
            "type": "function",
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        }])

    def test_nonstream_parses_function_call_and_keeps_reasoning_out_of_text(self) -> None:
        turn = providers._assistant_turn_from_responses({
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "think hard"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "progress noise"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "visible"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_9",
                    "name": "run_terminal",
                    "arguments": '{"command":"pytest"}',
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 4},
        })
        self.assertEqual(turn.text, "visible")
        self.assertNotIn("progress noise", turn.text)
        self.assertNotIn("think hard", turn.text)
        self.assertEqual(turn.reasoning, "think hard")
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.tool_calls[0]["name"], "run_terminal")
        self.assertEqual(turn.tool_calls[0]["arguments"]["command"], "pytest")
        self.assertEqual(turn.finish_reason, "tool_calls")

    def test_streaming_sse_parses_text_reasoning_tools_and_usage(self) -> None:
        events = [
            {"type": "response.output_text.delta", "delta": "Hel"},
            {"type": "response.output_text.delta", "delta": "lo"},
            {
                "type": "response.reasoning_summary_text.delta",
                "delta": "why",
            },
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "type": "function_call",
                    "call_id": "call_s",
                    "name": "read_file",
                    "arguments": "",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 1,
                "delta": '{"path":',
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 1,
                "delta": '"x.py"}',
            },
            {
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "type": "function_call",
                    "call_id": "call_s",
                    "name": "read_file",
                    "arguments": '{"path":"x.py"}',
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 7,
                        "total_tokens": 12,
                    },
                    "output": [{
                        "type": "function_call",
                        "call_id": "call_s",
                        "name": "read_file",
                        "arguments": '{"path":"x.py"}',
                    }],
                },
            },
        ]
        deltas: list[tuple[str, str]] = []
        captured: dict = {}

        def fake_open_stream(url, *, headers, body, timeout):
            captured.update(url=url, headers=headers, body=body, timeout=timeout)
            return _sse_response(events)

        with mock.patch.object(providers, "_open_stream", side_effect=fake_open_stream):
            turn = providers.provider_chat_streaming(
                provider="opencode-go",
                model="gpt-5.6-luna",
                messages=[{"role": "user", "content": "hi"}],
                api_key="sk-go-test",
                on_delta=lambda kind, text: deltas.append((kind, text)),
            )
        self.assertEqual(captured["url"], f"{go.BASE_URL}/responses")
        self.assertTrue(captured["body"]["stream"])
        self.assertEqual(turn.tool_calls[0]["name"], "read_file")
        self.assertEqual(turn.tool_calls[0]["arguments"]["path"], "x.py")
        self.assertEqual(turn.usage["total_tokens"], 12)
        self.assertEqual(turn.reasoning, "why")
        self.assertNotIn("why", turn.text)
        self.assertIn(("text", "Hel"), deltas)
        self.assertIn(("reasoning", "why"), deltas)

    def test_streaming_terminal_error_raises(self) -> None:
        with mock.patch.object(
            providers,
            "_open_stream",
            return_value=_sse_response([{
                "type": "response.failed",
                "response": {
                    "status": "failed",
                    "error": {"message": "quota exhausted"},
                },
            }]),
        ):
            with self.assertRaises(providers.ProviderError) as ctx:
                providers.provider_chat_streaming(
                    provider="opencode-go",
                    model="gpt-5.6-luna",
                    messages=[{"role": "user", "content": "hi"}],
                    api_key="sk-go-test",
                )
        self.assertEqual(ctx.exception.reason, "provider_error")
        self.assertIn("quota exhausted", str(ctx.exception))

    def test_responses_tool_round_trip_request_body(self) -> None:
        """Prior tool call + tool result convert into Responses input items."""
        captured: dict = {}

        def fake_post(url, *, headers, body, timeout):
            captured["body"] = body
            return {
                "status": "completed",
                "output": [{
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "submit_report",
                    "arguments": '{"summary":"done","files_changed":[],"verification":"n/a"}',
                }],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }

        tools = [{
            "type": "function",
            "function": {
                "name": "submit_report",
                "description": "Finish",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        messages = [
            {"role": "system", "content": "Implement mode."},
            {"role": "user", "content": "edit the file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "arguments": '{"path":"a.py","old_string":"x","new_string":"y"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "edited",
            },
        ]
        with mock.patch.object(providers, "_post_json", side_effect=fake_post):
            turn = providers.provider_chat(
                provider="opencode-go",
                model="gpt-5.6-luna",
                messages=messages,
                tools=tools,
                api_key="sk-go-test",
            )
        body = captured["body"]
        self.assertEqual(body["instructions"], "Implement mode.")
        self.assertEqual(body["tools"][0]["name"], "submit_report")
        self.assertEqual(body["tool_choice"], "auto")
        types = [item["type"] for item in body["input"]]
        self.assertEqual(
            types,
            ["message", "function_call", "function_call_output"],
        )
        self.assertEqual(turn.tool_calls[0]["name"], "submit_report")
        self.assertEqual(turn.tool_calls[0]["arguments"]["summary"], "done")


class OpenCodeGoErrorClassificationTests(unittest.TestCase):
    def test_upstream_block_patterns(self) -> None:
        body = (
            '{"error":{"message":"Request blocked by upstream provider",'
            '"type":"AuthError"}}'
        )
        self.assertTrue(is_upstream_provider_block(body))
        self.assertTrue(
            is_upstream_provider_block(
                "HTTP 401: Request blocked by upstream provider"
            )
        )
        self.assertFalse(is_upstream_provider_block("invalid_api_key"))

    def test_upstream_block_401_is_retryable_not_auth(self) -> None:
        body = (
            '{"error":{"message":"Request blocked by upstream provider",'
            '"type":"AuthError"}}'
        )
        self.assertEqual(
            classify_provider_failure("http_status:401", 401, body=body),
            SERVER_ERROR,
        )
        error = providers.ProviderError(
            "HTTP 401: Request blocked by upstream provider",
            reason="http_status:401",
            status=401,
            body=body,
        )
        self.assertEqual(error.failure, SERVER_ERROR)
        self.assertTrue(providers.is_retryable_provider_error(error))

    def test_genuine_invalid_key_still_auth(self) -> None:
        self.assertEqual(
            classify_provider_failure(
                "http_status:401",
                401,
                body="incorrect api key provided",
            ),
            NOT_AUTHENTICATED,
        )
        error = providers.ProviderError(
            "HTTP 401",
            reason="http_status:401",
            status=401,
            body="invalid_api_key",
        )
        self.assertEqual(error.failure, NOT_AUTHENTICATED)
        self.assertFalse(providers.is_retryable_provider_error(error))

    def test_http_error_message_preserves_upstream_detail(self) -> None:
        body = (
            '{"error":{"message":"Request blocked by upstream provider",'
            '"type":"AuthError"}}'
        )
        msg = providers._http_error_message(401, body)
        self.assertIn("Request blocked by upstream provider", msg)
        self.assertTrue(msg.startswith("HTTP 401:"))


class OpenCodeGoAgenticSelectionTests(unittest.TestCase):
    def test_curated_catalog_stamps_opencode_go_provider(self) -> None:
        from puppetmaster.static_catalog import curated_to_specs

        specs = curated_to_specs(
            "agentic",
            "api",
            [],
            allowed_providers={"opencode-go"},
        )
        self.assertTrue(specs)
        names = {spec.adapter_model_name for spec in specs}
        self.assertIn("gpt-5.6-luna", names)
        for spec in specs:
            self.assertEqual(spec.payload_defaults.get("provider"), "opencode-go")
            self.assertEqual(spec.billing, "plan")

    def test_router_picks_opencode_go_when_key_present(self) -> None:
        from puppetmaster.router import TaskSignals, route_task
        from puppetmaster.static_catalog import curated_to_specs

        registry = curated_to_specs("agentic", "api", [])
        signals = TaskSignals(
            role="agentic-implement",
            instruction="Fix a small typo in the docs",
            allowed_adapters={"agentic"},
        )
        with mock.patch(
            "puppetmaster.providers.available_providers",
            return_value={"opencode-go"},
        ):
            decision = route_task(signals, registry, policy="balanced")
        self.assertEqual(decision.model.adapter, "agentic")
        self.assertEqual(
            decision.model.payload_defaults["provider"],
            "opencode-go",
        )

    def test_agentic_resolve_provider_and_env_hint(self) -> None:
        from puppetmaster.adapters.agentic import AgenticAdapter, _PROVIDER_ENV_HINTS
        from puppetmaster.models import Task

        self.assertEqual(_PROVIDER_ENV_HINTS["opencode-go"], "OPENCODE_GO_API_KEY")
        adapter = AgenticAdapter()
        task = Task(
            job_id="j",
            id="t",
            role="agentic-analyze",
            instruction="x",
            adapter="agentic",
            payload={"provider": "opencode-go", "model": "deepseek-v4-flash"},
        )
        self.assertEqual(adapter._resolve_provider(task), "opencode-go")

    def test_auth_failure_risk_skips_upstream_block(self) -> None:
        from puppetmaster.adapters.agentic import AgenticAdapter
        from puppetmaster.models import Task

        adapter = AgenticAdapter()
        task = Task(
            job_id="j", id="t", role="r", instruction="x", adapter="agentic",
        )
        risk = adapter._auth_failure_risk(
            task,
            "w",
            "opencode-go",
            401,
            'Request blocked by upstream provider',
            reason="http_status:401",
        )
        self.assertIsNone(risk)

    def test_auth_failure_risk_still_fires_for_invalid_key(self) -> None:
        from puppetmaster.adapters.agentic import AgenticAdapter
        from puppetmaster.models import ArtifactType, Task

        adapter = AgenticAdapter()
        task = Task(
            job_id="j", id="t", role="r", instruction="x", adapter="agentic",
        )
        risk = adapter._auth_failure_risk(
            task,
            "w",
            "opencode-go",
            401,
            "incorrect api key provided",
            reason="http_status:401",
        )
        self.assertIsNotNone(risk)
        assert risk is not None
        self.assertEqual(risk.type, ArtifactType.RISK)
        self.assertIn("OPENCODE_GO_API_KEY", risk.payload["mitigation"])


if __name__ == "__main__":
    unittest.main()
