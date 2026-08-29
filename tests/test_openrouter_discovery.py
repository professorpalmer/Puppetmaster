"""OpenRouter model discovery: catalog parsing and registry merge."""
import json
import unittest

from puppetmaster.api_discovery import (
    ApiDiscoveryError,
    fetch_openrouter_models,
    merge_api_catalog_into_registry,
)
from puppetmaster.model_registry import ModelSpec

_CATALOG = {
    "data": [
        {
            "id": "anthropic/claude-sonnet-4.5",
            "name": "Anthropic: Claude Sonnet 4.5",
            "context_length": 1000000,
            "supported_parameters": ["tools", "reasoning"],
            "architecture": {
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
            },
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
        },
        {
            "id": "openrouter/auto",
            "name": "Auto Router",
            "context_length": 2000000,
            "supported_parameters": ["tools"],
            "architecture": {"output_modalities": ["text"]},
            "pricing": {"prompt": "-1", "completion": "-1"},
        },
        {
            "id": "some/free-model",
            "context_length": 8192,
            "supported_parameters": ["tools"],
            "architecture": {"output_modalities": ["text"]},
            "pricing": {"prompt": "0", "completion": "0"},
        },
        {
            "id": "some/no-tools",
            "supported_parameters": ["temperature"],
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        },
        {
            "id": "some/image-only",
            "supported_parameters": ["tools"],
            "architecture": {"output_modalities": ["image"]},
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        },
    ]
}


def _getter(status=200, body=None):
    body = json.dumps(_CATALOG) if body is None else body
    captured = {}

    def get(url, headers):
        captured["url"] = url
        captured["headers"] = dict(headers)
        return (status, body)

    get.captured = captured
    return get


class FetchOpenRouterModelsTest(unittest.TestCase):
    def test_requires_api_key(self):
        with self.assertRaises(ApiDiscoveryError):
            fetch_openrouter_models(env={}, getter=_getter())

    def test_http_error_is_reported(self):
        with self.assertRaises(ApiDiscoveryError) as ctx:
            fetch_openrouter_models(
                env={"OPENROUTER_API_KEY": "k"}, getter=_getter(status=401, body="nope")
            )
        self.assertIn("401", str(ctx.exception))

    def test_non_json_body_is_reported(self):
        with self.assertRaises(ApiDiscoveryError):
            fetch_openrouter_models(
                env={"OPENROUTER_API_KEY": "k"}, getter=_getter(body="<html>")
            )

    def test_missing_data_array_is_reported(self):
        with self.assertRaises(ApiDiscoveryError):
            fetch_openrouter_models(
                env={"OPENROUTER_API_KEY": "k"}, getter=_getter(body="{}")
            )

    def test_base_url_and_auth_header(self):
        get = _getter()
        fetch_openrouter_models(
            env={"OPENROUTER_API_KEY": "k", "OPENROUTER_BASE_URL": "https://proxy/api/v1/"},
            getter=get,
        )
        self.assertEqual(get.captured["url"], "https://proxy/api/v1/models")
        self.assertEqual(get.captured["headers"]["Authorization"], "Bearer k")

    def test_keeps_only_tool_capable_text_models(self):
        catalog = fetch_openrouter_models(
            env={"OPENROUTER_API_KEY": "k"}, getter=_getter()
        )
        self.assertEqual(
            [m["id"] for m in catalog],
            ["anthropic/claude-sonnet-4.5", "openrouter/auto", "some/free-model"],
        )

    def test_tools_only_can_be_disabled(self):
        catalog = fetch_openrouter_models(
            env={"OPENROUTER_API_KEY": "k"}, getter=_getter(), tools_only=False
        )
        self.assertIn("some/no-tools", [m["id"] for m in catalog])

    def test_price_context_and_tags(self):
        catalog = fetch_openrouter_models(
            env={"OPENROUTER_API_KEY": "k"}, getter=_getter()
        )
        by_id = {m["id"]: m for m in catalog}
        sonnet = by_id["anthropic/claude-sonnet-4.5"]
        self.assertEqual(sonnet["displayName"], "Anthropic: Claude Sonnet 4.5")
        self.assertEqual(sonnet["input_per_mtok_usd"], 3.0)
        self.assertEqual(sonnet["output_per_mtok_usd"], 15.0)
        self.assertEqual(sonnet["context_window"], 1000000)
        self.assertEqual(
            sonnet["tags"],
            ["long-context", "openrouter", "reasoning", "vision"],
        )
        self.assertNotIn("free", by_id["openrouter/auto"]["tags"])
        self.assertIn("free", by_id["some/free-model"]["tags"])

    def test_dynamic_pricing_is_not_negative(self):
        catalog = fetch_openrouter_models(
            env={"OPENROUTER_API_KEY": "k"}, getter=_getter()
        )
        auto = next(m for m in catalog if m["id"] == "openrouter/auto")
        self.assertEqual(auto["input_per_mtok_usd"], 0.0)
        self.assertEqual(auto["output_per_mtok_usd"], 0.0)


class MergeOpenRouterCatalogTest(unittest.TestCase):
    def _merge(self, existing):
        catalog = fetch_openrouter_models(
            env={"OPENROUTER_API_KEY": "k"}, getter=_getter()
        )
        return merge_api_catalog_into_registry(
            "agentic",
            "api",
            existing,
            catalog,
            id_namespace="openrouter",
            payload_defaults={"provider": "openrouter"},
        )

    def test_new_rows_are_namespaced_and_carry_provider(self):
        merged, _ = self._merge([])
        sonnet = next(
            s for s in merged if s.adapter_model_name == "anthropic/claude-sonnet-4.5"
        )
        self.assertEqual(sonnet.id, "openrouter/anthropic-claude-sonnet-4-5")
        self.assertEqual(sonnet.adapter, "agentic")
        self.assertEqual(sonnet.payload_defaults, {"provider": "openrouter"})
        self.assertEqual(sonnet.input_per_mtok_usd, 3.0)
        self.assertEqual(sonnet.context_window, 1000000)
        self.assertIn("openrouter", sonnet.tags)

    def test_hand_tuned_values_win_over_catalog(self):
        tuned = ModelSpec(
            id="mine/sonnet",
            adapter="agentic",
            adapter_model_name="anthropic/claude-sonnet-4.5",
            capability_score=95,
            input_per_mtok_usd=1.0,
            output_per_mtok_usd=2.0,
            context_window=123,
            billing="api",
            tags=["mine"],
        )
        merged, _ = self._merge([tuned])
        sonnet = next(s for s in merged if s.id == "mine/sonnet")
        self.assertEqual(sonnet.capability_score, 95)
        self.assertEqual(sonnet.input_per_mtok_usd, 1.0)
        self.assertEqual(sonnet.context_window, 123)
        self.assertEqual(sonnet.payload_defaults, {"provider": "openrouter"})
        self.assertIn("mine", sonnet.tags)

    def test_unrelated_entries_are_preserved(self):
        other = ModelSpec(
            id="openai/gpt-4o",
            adapter="openai",
            adapter_model_name="gpt-4o",
            capability_score=80,
            input_per_mtok_usd=2.5,
            output_per_mtok_usd=10.0,
            context_window=128000,
            billing="api",
        )
        merged, _ = self._merge([other])
        self.assertIn("openai/gpt-4o", [s.id for s in merged])


if __name__ == "__main__":
    unittest.main()
