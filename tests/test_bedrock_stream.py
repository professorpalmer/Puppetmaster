"""Hermetic unit tests for Bedrock ConverseStream (no live AWS)."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import io
import json
import unittest
from unittest.mock import patch

def _stream_body(*events: bytes) -> bytes:
    return b"".join(events)

class BedrockEventStreamCodecTests(unittest.TestCase):
    def test_round_trip_headers_and_json_payload(self) -> None:
        from puppetmaster import bedrock

        payload = json.dumps({"delta": {"text": "hi"}, "contentBlockIndex": 0}).encode(
            "utf-8"
        )
        framed = bedrock.encode_eventstream_message(
            {
                ":message-type": "event",
                ":event-type": "contentBlockDelta",
                ":content-type": "application/json",
            },
            payload,
        )
        messages = list(bedrock._iter_eventstream_messages(io.BytesIO(framed)))
        self.assertEqual(len(messages), 1)
        headers, body = messages[0]
        self.assertEqual(headers[":event-type"], "contentBlockDelta")
        self.assertEqual(headers[":message-type"], "event")
        self.assertEqual(json.loads(body.decode("utf-8")), {
            "delta": {"text": "hi"},
            "contentBlockIndex": 0,
        })

    def test_encode_converse_stream_event_helper(self) -> None:
        from puppetmaster import bedrock

        framed = bedrock.encode_converse_stream_event(
            "messageStop", {"stopReason": "end_turn"}
        )
        events = list(bedrock._iter_converse_stream_events(io.BytesIO(framed)))
        self.assertEqual(events, [{"messageStop": {"stopReason": "end_turn"}}])

class BedrockConverseStreamTests(unittest.TestCase):
    def test_converse_stream_url(self) -> None:
        from puppetmaster import bedrock

        url = bedrock.converse_stream_model_url(
            "https://bedrock-runtime.us-east-1.amazonaws.com",
            "amazon.nova-micro-v1:0",
        )
        self.assertTrue(url.endswith("/converse-stream"))
        self.assertIn("/model/amazon.nova-micro-v1%3A0/", url)

    def test_stream_emits_incremental_text_and_reasoning_deltas(self) -> None:
        from puppetmaster import bedrock

        framed = _stream_body(
            bedrock.encode_converse_stream_event(
                "messageStart", {"role": "assistant"}
            ),
            bedrock.encode_converse_stream_event(
                "contentBlockDelta",
                {
                    "contentBlockIndex": 0,
                    "delta": {"reasoningContent": {"text": "think-"}},
                },
            ),
            bedrock.encode_converse_stream_event(
                "contentBlockDelta",
                {
                    "contentBlockIndex": 0,
                    "delta": {"reasoningContent": {"text": "ing"}},
                },
            ),
            bedrock.encode_converse_stream_event(
                "contentBlockDelta",
                {"contentBlockIndex": 1, "delta": {"text": "Hel"}},
            ),
            bedrock.encode_converse_stream_event(
                "contentBlockDelta",
                {"contentBlockIndex": 1, "delta": {"text": "lo"}},
            ),
            bedrock.encode_converse_stream_event(
                "messageStop", {"stopReason": "end_turn"}
            ),
            bedrock.encode_converse_stream_event(
                "metadata",
                {
                    "usage": {
                        "inputTokens": 10,
                        "outputTokens": 4,
                        "totalTokens": 4425,
                        "cacheReadInputTokens": 4411,
                        "cacheWriteInputTokens": 2,
                    }
                },
            ),
        )
        deltas = []
        captured = {}

        class _FakeResponse(io.BytesIO):
            def close(self) -> None:
                return None

        def _open_stream(url, *, headers, body, timeout):
            captured["url"] = url
            captured["headers"] = dict(headers)
            captured["body"] = body
            return _FakeResponse(framed)

        with patch.object(
            bedrock, "_open_bedrock_event_stream", side_effect=_open_stream
        ):
            turn = bedrock.bedrock_chat_stream(
                model="amazon.nova-micro-v1:0",
                messages=[{"role": "user", "content": "hi"}],
                env={
                    "AWS_BEARER_TOKEN_BEDROCK": "tok",
                    "AWS_REGION": "us-east-1",
                },
                on_delta=lambda kind, text: deltas.append((kind, text)),
            )

        self.assertEqual(
            deltas,
            [
                ("reasoning", "think-"),
                ("reasoning", "ing"),
                ("text", "Hel"),
                ("text", "lo"),
            ],
        )
        self.assertEqual(turn.text, "Hello")
        self.assertEqual(turn.finish_reason, "end_turn")
        self.assertEqual(turn.usage["prompt_tokens"], 4423)
        self.assertEqual(turn.usage["cached_tokens"], 4411)
        self.assertEqual(turn.usage["cache_write_tokens"], 2)
        self.assertEqual(turn.usage["completion_tokens"], 4)
        self.assertTrue(captured["url"].endswith("/converse-stream"))
        self.assertEqual(
            captured["headers"].get("Accept"),
            "application/vnd.amazon.eventstream",
        )
        self.assertEqual(
            captured["headers"].get("Authorization"), "Bearer tok"
        )
        self.assertIn("messages", captured["body"])
        self.assertIn("inferenceConfig", captured["body"])

    def test_stream_parses_tool_use_and_partial_json(self) -> None:
        from puppetmaster import bedrock

        class _FakeResponse(io.BytesIO):
            def close(self) -> None:
                return None

        framed = _stream_body(
            bedrock.encode_converse_stream_event(
                "contentBlockStart",
                {
                    "contentBlockIndex": 0,
                    "start": {
                        "toolUse": {"toolUseId": "tu1", "name": "echo"},
                    },
                },
            ),
            bedrock.encode_converse_stream_event(
                "contentBlockDelta",
                {
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": '{"text":'}},
                },
            ),
            bedrock.encode_converse_stream_event(
                "contentBlockDelta",
                {
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": ' "hi"}'}},
                },
            ),
            bedrock.encode_converse_stream_event(
                "contentBlockStop", {"contentBlockIndex": 0}
            ),
            bedrock.encode_converse_stream_event(
                "messageStop", {"stopReason": "tool_use"}
            ),
            bedrock.encode_converse_stream_event(
                "metadata",
                {"usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5}},
            ),
        )

        with patch.object(
            bedrock,
            "_open_bedrock_event_stream",
            return_value=_FakeResponse(framed),
        ):
            turn = bedrock.bedrock_chat_stream(
                model="amazon.nova-micro-v1:0",
                messages=[{"role": "user", "content": "hi"}],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }],
                env={"AWS_BEARER_TOKEN_BEDROCK": "tok"},
            )

        self.assertEqual(turn.finish_reason, "tool_use")
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.tool_calls[0]["id"], "tu1")
        self.assertEqual(turn.tool_calls[0]["name"], "echo")
        self.assertEqual(turn.tool_calls[0]["arguments"], {"text": "hi"})

    def test_bedrock_chat_with_on_delta_uses_converse_stream(self) -> None:
        from puppetmaster import bedrock

        class _FakeResponse(io.BytesIO):
            def close(self) -> None:
                return None

        framed = _stream_body(
            bedrock.encode_converse_stream_event(
                "contentBlockDelta",
                {"contentBlockIndex": 0, "delta": {"text": "a"}},
            ),
            bedrock.encode_converse_stream_event(
                "contentBlockDelta",
                {"contentBlockIndex": 0, "delta": {"text": "b"}},
            ),
            bedrock.encode_converse_stream_event(
                "messageStop", {"stopReason": "end_turn"}
            ),
            bedrock.encode_converse_stream_event(
                "metadata",
                {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}},
            ),
        )
        deltas = []
        with patch.object(
            bedrock,
            "_open_bedrock_event_stream",
            return_value=_FakeResponse(framed),
        ) as opened:
            with patch.object(bedrock, "_post_bedrock") as post:
                turn = bedrock.bedrock_chat(
                    model="amazon.nova-micro-v1:0",
                    messages=[{"role": "user", "content": "hi"}],
                    env={"AWS_BEARER_TOKEN_BEDROCK": "tok"},
                    on_delta=lambda kind, text: deltas.append((kind, text)),
                )
        post.assert_not_called()
        opened.assert_called_once()
        self.assertEqual(deltas, [("text", "a"), ("text", "b")])
        self.assertEqual(turn.text, "ab")

    def test_bedrock_chat_without_on_delta_stays_on_converse(self) -> None:
        from puppetmaster import bedrock

        canned = {
            "output": {
                "message": {"role": "assistant", "content": [{"text": "ok"}]}
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }
        with patch.object(bedrock, "_post_bedrock", return_value=canned) as post:
            with patch.object(bedrock, "_open_bedrock_event_stream") as opened:
                turn = bedrock.bedrock_chat(
                    model="amazon.nova-micro-v1:0",
                    messages=[{"role": "user", "content": "hi"}],
                    env={"AWS_BEARER_TOKEN_BEDROCK": "tok"},
                )
        opened.assert_not_called()
        post.assert_called_once()
        self.assertEqual(turn.text, "ok")

    def test_stream_exception_event_raises_provider_error(self) -> None:
        from puppetmaster import bedrock
        from puppetmaster.providers import ProviderError

        framed = bedrock.encode_eventstream_message(
            {
                ":message-type": "exception",
                ":exception-type": "ValidationException",
                ":content-type": "application/json",
            },
            json.dumps({"message": "bad model"}).encode("utf-8"),
        )

        class _FakeResponse(io.BytesIO):
            def close(self) -> None:
                return None

        with patch.object(
            bedrock,
            "_open_bedrock_event_stream",
            return_value=_FakeResponse(framed),
        ):
            with self.assertRaises(ProviderError) as ctx:
                bedrock.bedrock_chat_stream(
                    model="amazon.nova-micro-v1:0",
                    messages=[{"role": "user", "content": "hi"}],
                    env={"AWS_BEARER_TOKEN_BEDROCK": "tok"},
                )
        self.assertIn("bad model", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
