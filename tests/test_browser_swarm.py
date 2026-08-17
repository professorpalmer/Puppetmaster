"""Focused tests for agentic/OpenRouter browser-swarm parity.

Hermes stays the preferred adapter. Agentic uses the existing stdlib CDP
hooks, the same acting-agent / network-truth / React-input / strong-model
guardrails, and is selectable from CLI + MCP.
"""
import os
import sys
import unittest
from unittest import mock

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

from puppetmaster.workers import (
    spec_edits_files,
    spec_has_side_effects,
    swarm_is_acting,
    swarm_mode,
)


class BrowserAdapterResolveTests(unittest.TestCase):
    def test_prefers_hermes_when_both_enabled(self) -> None:
        from puppetmaster.browser import (
            PREFERRED_BROWSER_ADAPTER,
            resolve_browser_adapter,
        )

        with mock.patch(
            "puppetmaster.platform_lock.is_adapter_enabled",
            side_effect=lambda name, registry_path=None: name in {"hermes", "agentic"},
        ), mock.patch(
            "puppetmaster.platform_lock.enabled_adapters",
            return_value={"hermes", "agentic", "cursor"},
        ):
            self.assertEqual(resolve_browser_adapter(), PREFERRED_BROWSER_ADAPTER)
            self.assertEqual(resolve_browser_adapter(None), "hermes")

    def test_falls_back_to_agentic_when_hermes_locked_out(self) -> None:
        from puppetmaster.browser import resolve_browser_adapter

        with mock.patch(
            "puppetmaster.platform_lock.is_adapter_enabled",
            side_effect=lambda name, registry_path=None: name == "agentic",
        ), mock.patch(
            "puppetmaster.platform_lock.enabled_adapters",
            return_value={"agentic", "cursor"},
        ):
            self.assertEqual(resolve_browser_adapter(), "agentic")

    def test_explicit_agentic_pin(self) -> None:
        from puppetmaster.browser import resolve_browser_adapter

        with mock.patch(
            "puppetmaster.platform_lock.is_adapter_enabled",
            return_value=True,
        ), mock.patch(
            "puppetmaster.platform_lock.enabled_adapters",
            return_value={"hermes", "agentic"},
        ):
            self.assertEqual(resolve_browser_adapter("agentic"), "agentic")
            self.assertEqual(resolve_browser_adapter("HERMES"), "hermes")

    def test_rejects_unknown_and_disabled(self) -> None:
        from puppetmaster.browser import (
            BrowserAdapterUnavailable,
            resolve_browser_adapter,
        )

        with mock.patch(
            "puppetmaster.platform_lock.is_adapter_enabled",
            return_value=False,
        ), mock.patch(
            "puppetmaster.platform_lock.enabled_adapters",
            return_value={"cursor"},
        ):
            with self.assertRaises(BrowserAdapterUnavailable) as unknown:
                resolve_browser_adapter("cursor")
            self.assertIn("not browser-capable", str(unknown.exception))
            with self.assertRaises(BrowserAdapterUnavailable) as disabled:
                resolve_browser_adapter("hermes")
            self.assertEqual(disabled.exception.requested, "hermes")
            self.assertIn("platform enable hermes", disabled.exception.fix or "")
            with self.assertRaises(BrowserAdapterUnavailable) as none:
                resolve_browser_adapter()
            self.assertIn("no browser-capable adapter", str(none.exception))


class AgenticBrowserSpecTests(unittest.TestCase):
    def test_agentic_spec_keeps_guardrails_and_stays_analysis(self) -> None:
        from puppetmaster.browser import (
            BROWSER_MIN_CAPABILITY,
            BROWSER_TOOLSETS,
            build_browser_spec,
        )

        with mock.patch(
            "puppetmaster.platform_lock.is_adapter_enabled",
            return_value=True,
        ), mock.patch(
            "puppetmaster.platform_lock.enabled_adapters",
            return_value={"hermes", "agentic"},
        ):
            spec = build_browser_spec(
                "QA login on https://app.example.com",
                cwd="/repo",
                adapter="agentic",
                provider="openrouter",
            )
        self.assertEqual(spec.adapter, "agentic")
        self.assertEqual(spec.payload["mode"], "analyze")
        self.assertTrue(spec.payload["allow_browser"])
        self.assertTrue(spec.payload["no_edit"])
        self.assertTrue(spec.payload["side_effecting"])
        self.assertEqual(spec.payload["provider"], "openrouter")
        self.assertEqual(spec.payload["toolsets"], BROWSER_TOOLSETS)
        self.assertIn("browser", spec.payload["toolsets"].split(","))
        self.assertTrue(spec.payload["auto_route"])
        self.assertEqual(spec.payload["allowed_adapters"], ["agentic"])
        self.assertGreaterEqual(spec.payload["min_capability"], BROWSER_MIN_CAPABILITY)
        prompt = spec.payload["prompt"]
        self.assertIn("React-controlled", prompt)
        self.assertIn("getOwnPropertyDescriptor", prompt)
        self.assertIn("200", prompt)
        self.assertIn("QA login on https://app.example.com", prompt)
        self.assertFalse(spec_edits_files(spec))
        self.assertEqual(swarm_mode([spec]), "analysis")
        self.assertTrue(spec_has_side_effects(spec))
        self.assertTrue(swarm_is_acting([spec]))

    def test_default_spec_stays_hermes(self) -> None:
        from puppetmaster.browser import BROWSER_ADAPTER, build_browser_spec

        spec = build_browser_spec("x", cwd="/repo")
        self.assertEqual(spec.adapter, BROWSER_ADAPTER)
        self.assertEqual(spec.adapter, "hermes")
        self.assertNotIn("allow_browser", spec.payload)
        self.assertNotIn("no_edit", spec.payload)

    def test_agentic_swarm_fans_out(self) -> None:
        from puppetmaster.browser import browser_swarm_specs

        with mock.patch(
            "puppetmaster.platform_lock.is_adapter_enabled",
            return_value=True,
        ):
            specs = browser_swarm_specs(
                ["QA routes", "QA airports"],
                cwd="/repo",
                adapter="agentic",
                provider="openrouter",
            )
        self.assertEqual([s.role for s in specs], ["browser-1", "browser-2"])
        self.assertTrue(all(s.adapter == "agentic" for s in specs))
        self.assertTrue(all(s.payload.get("allow_browser") for s in specs))
        self.assertTrue(all(not s.depends_on_roles for s in specs))


class AgenticBrowserToolTests(unittest.TestCase):
    def test_toolsets_and_allow_browser_enable_cdp_tools(self) -> None:
        from puppetmaster.adapters.agentic import AgenticAdapter
        from puppetmaster.models import Task

        adapter = AgenticAdapter()
        via_toolsets = Task(
            job_id="j",
            role="browser",
            instruction="qa",
            payload={"toolsets": "file,web,vision,browser"},
        )
        via_flag = Task(
            job_id="j",
            role="browser",
            instruction="qa",
            payload={"allow_browser": True},
        )
        off = Task(job_id="j", role="browser", instruction="qa", payload={})
        self.assertTrue(adapter._browser_enabled(via_toolsets))
        self.assertTrue(adapter._browser_enabled(via_flag))
        self.assertFalse(adapter._browser_enabled(off))
        names = {
            tool["function"]["name"]
            if "function" in tool
            else tool.get("name")
            for tool in adapter._tool_schema(implement=False, task=via_toolsets)
        }
        for required in (
            "browser_navigate",
            "browser_snapshot",
            "browser_click",
            "browser_type",
            "browser_network",
        ):
            self.assertIn(required, names)
        off_names = {
            tool["function"]["name"]
            if "function" in tool
            else tool.get("name")
            for tool in adapter._tool_schema(implement=False, task=off)
        }
        self.assertNotIn("browser_navigate", off_names)

    def test_execute_dispatches_cdp_network_tool(self) -> None:
        from pathlib import Path

        from puppetmaster.adapters.agentic import AgenticAdapter
        from puppetmaster.models import Task

        adapter = AgenticAdapter()
        task = Task(
            job_id="j",
            role="browser",
            instruction="qa",
            payload={"allow_browser": True, "cwd": "/repo"},
        )
        with mock.patch(
            "puppetmaster.browser_cdp.dispatch",
            return_value="Captured fetch/XHR traffic:\n[]",
        ) as dispatch:
            out = adapter._execute_tool(
                "browser_network", {}, Path("/repo"), False, task
            )
        dispatch.assert_called_once()
        self.assertEqual(dispatch.call_args[0][0], "browser_network")
        self.assertIn("Captured fetch/XHR", out)


class BrowserCliMcpSurfaceTests(unittest.TestCase):
    def test_cli_accepts_adapter_flag(self) -> None:
        from puppetmaster.cli._parser import build_parser

        args = build_parser().parse_args(
            [
                "browser",
                "QA login",
                "--adapter",
                "agentic",
                "--provider",
                "openrouter",
            ]
        )
        self.assertEqual(args.command, "browser")
        self.assertEqual(args.adapter, "agentic")
        self.assertEqual(args.provider, "openrouter")
        self.assertEqual(args.tasks, ["QA login"])

    def test_mcp_schema_exposes_adapter_enum(self) -> None:
        from puppetmaster.mcp_server import browser_swarm_schema

        schema = browser_swarm_schema()
        adapter = schema["properties"]["adapter"]
        self.assertEqual(adapter["enum"], ["hermes", "agentic"])

    def test_mcp_start_errors_when_requested_adapter_locked(self) -> None:
        from puppetmaster.browser import BrowserAdapterUnavailable
        from puppetmaster.mcp_server import start_browser_swarm

        with mock.patch(
            "puppetmaster.browser.resolve_browser_adapter",
            side_effect=BrowserAdapterUnavailable(
                "the 'hermes' adapter is disabled by the platform lock.",
                enabled={"agentic"},
                requested="hermes",
                fix="puppetmaster platform enable hermes",
            ),
        ):
            result = start_browser_swarm(
                {"tasks": ["QA login"], "adapter": "hermes"}
            )
        self.assertTrue(result.get("isError"))
        text = result["content"][0]["text"]
        self.assertIn("hermes", text)
        self.assertIn("platform enable hermes", text)


if __name__ == "__main__":
    unittest.main()
