"""Z.AI / MiniMax / NVIDIA NIM agentic provider descriptors.

Hermetic: no real network. Covers get_provider, available_providers, env hints,
curated catalog stamps, and wire resolution for provider_chat.
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

from puppetmaster import providers
from puppetmaster.adapters.agentic import _PROVIDER_ENV_HINTS


class ZaiProviderTests(unittest.TestCase):
    def test_provider_is_registered(self) -> None:
        desc = providers.get_provider("zai")
        self.assertIsNotNone(desc)
        assert desc is not None
        self.assertEqual(desc.slug, "zai")
        self.assertEqual(desc.wire, "openai")
        self.assertEqual(desc.base_url, "https://api.z.ai/api/paas/v4")
        self.assertEqual(
            desc.api_key_env_vars,
            ("ZAI_API_KEY", "GLM_API_KEY", "Z_AI_API_KEY"),
        )
        self.assertEqual(desc.label, "Z.AI")

    def test_available_from_any_key_alias(self) -> None:
        for env_var in ("ZAI_API_KEY", "GLM_API_KEY", "Z_AI_API_KEY"):
            self.assertIn(
                "zai",
                providers.available_providers({env_var: "sk-zai-test"}),
                msg=env_var,
            )
        self.assertNotIn("zai", providers.available_providers({}))

    def test_env_hint(self) -> None:
        self.assertIn("ZAI_API_KEY", _PROVIDER_ENV_HINTS["zai"])


class MinimaxProviderTests(unittest.TestCase):
    def test_provider_is_registered(self) -> None:
        desc = providers.get_provider("minimax")
        self.assertIsNotNone(desc)
        assert desc is not None
        self.assertEqual(desc.slug, "minimax")
        self.assertEqual(desc.wire, "anthropic")
        self.assertEqual(desc.base_url, "https://api.minimax.io/anthropic")
        self.assertEqual(desc.api_key_env_vars, ("MINIMAX_API_KEY",))
        self.assertEqual(desc.label, "MiniMax")

    def test_available_from_key(self) -> None:
        self.assertIn(
            "minimax",
            providers.available_providers({"MINIMAX_API_KEY": "sk-mm-test"}),
        )
        self.assertNotIn("minimax", providers.available_providers({}))

    def test_resolve_base_url_appends_v1(self) -> None:
        desc = providers.get_provider("minimax")
        assert desc is not None
        self.assertEqual(
            providers.resolve_base_url(desc, {}),
            "https://api.minimax.io/anthropic/v1",
        )
        self.assertEqual(
            providers.resolve_base_url(
                desc, {"MINIMAX_BASE_URL": "https://api.minimax.io/anthropic/"}
            ),
            "https://api.minimax.io/anthropic/v1",
        )

    def test_provider_chat_uses_anthropic_messages(self) -> None:
        captured: dict = {}

        def fake_post(url, *, headers, body, timeout):
            captured.update(url=url, headers=headers, body=body)
            return {
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        with mock.patch.object(providers, "_post_json", side_effect=fake_post):
            turn = providers.provider_chat(
                provider="minimax",
                model="MiniMax-M3",
                messages=[{"role": "user", "content": "hi"}],
                api_key="sk-mm-test",
                extra={"max_tokens": 256},
            )
        self.assertEqual(
            captured["url"], "https://api.minimax.io/anthropic/v1/messages"
        )
        self.assertEqual(captured["headers"]["x-api-key"], "sk-mm-test")
        self.assertEqual(captured["body"]["model"], "MiniMax-M3")
        self.assertEqual(turn.text, "ok")

    def test_env_hint(self) -> None:
        self.assertEqual(_PROVIDER_ENV_HINTS["minimax"], "MINIMAX_API_KEY")


class NvidiaProviderTests(unittest.TestCase):
    def test_provider_is_registered(self) -> None:
        desc = providers.get_provider("nvidia")
        self.assertIsNotNone(desc)
        assert desc is not None
        self.assertEqual(desc.slug, "nvidia")
        self.assertEqual(desc.wire, "openai")
        self.assertEqual(desc.base_url, "https://integrate.api.nvidia.com/v1")
        self.assertEqual(desc.api_key_env_vars, ("NVIDIA_API_KEY",))
        self.assertEqual(desc.label, "NVIDIA NIM")

    def test_available_from_key(self) -> None:
        self.assertIn(
            "nvidia",
            providers.available_providers({"NVIDIA_API_KEY": "nvapi-test"}),
        )
        self.assertNotIn("nvidia", providers.available_providers({}))

    def test_provider_chat_uses_openai_chat_completions(self) -> None:
        captured: dict = {}

        def fake_post(url, *, headers, body, timeout):
            captured.update(url=url, headers=headers, body=body)
            return {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }

        with mock.patch.object(providers, "_post_json", side_effect=fake_post):
            turn = providers.provider_chat(
                provider="nvidia",
                model="meta/llama-3.3-70b-instruct",
                messages=[{"role": "user", "content": "hi"}],
                api_key="nvapi-test",
            )
        self.assertEqual(
            captured["url"],
            "https://integrate.api.nvidia.com/v1/chat/completions",
        )
        self.assertIn("Bearer nvapi-test", captured["headers"].get("Authorization", ""))
        self.assertEqual(captured["body"]["model"], "meta/llama-3.3-70b-instruct")
        self.assertEqual(turn.text, "ok")

    def test_env_hint(self) -> None:
        self.assertEqual(_PROVIDER_ENV_HINTS["nvidia"], "NVIDIA_API_KEY")


class MarionetteKeyedCatalogTests(unittest.TestCase):
    def test_curated_catalog_stamps_new_providers(self) -> None:
        from puppetmaster.static_catalog import curated_to_specs

        for provider, model in (
            ("zai", "glm-4.7"),
            ("minimax", "MiniMax-M3"),
            ("nvidia", "meta/llama-3.3-70b-instruct"),
        ):
            specs = curated_to_specs(
                "agentic",
                "api",
                [],
                allowed_providers={provider},
            )
            self.assertTrue(specs, msg=provider)
            names = {spec.adapter_model_name for spec in specs}
            self.assertIn(model, names, msg=provider)
            for spec in specs:
                self.assertEqual(spec.payload_defaults.get("provider"), provider)


if __name__ == "__main__":
    unittest.main()
