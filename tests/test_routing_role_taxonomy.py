"""Adversarial contract tests for the external routing-role taxonomy.

These tests intentionally exercise behavior at the public router seam.  They
do not read implementation-private caches and do not duplicate the production
loader's schema validation.
"""

from __future__ import annotations

import ast
import importlib.resources
import inspect
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

from puppetmaster import router
from puppetmaster.model_registry import ModelSpec


EXPECTED_BUILTIN_ROLES = {
    "verify-runtime": (25, False),
    "shell": (20, False),
    "demo": (25, False),
    "explore": (50, True),
    "review": (55, True),
    "plan": (60, True),
    "implement": (75, True),
    "refactor": (75, True),
    "patch": (75, True),
    "fix": (70, True),
    "build": (75, True),
    "test-coverage-reviewer": (60, True),
    "architect": (85, True),
    "audit": (85, True),
    "security-review": (90, True),
    "decision-explainer": (70, True),
    "conflict-auditor": (75, True),
    "pipeline-mapper": (65, True),
}

EXPECTED_ALIASES = {
    "runtime-check": "verify-runtime",
    "codex-review": "review",
    "hermes-implement": "implement",
    "routing_quality": "audit",
    "recovery-governance": "audit",
    "evaluation_design": "architect",
}

EXPECTED_LEGACY_OUTPUT_BUDGETS = {
    "verify-runtime": 300,
    "shell": 200,
    "demo": 500,
    "explore": 1500,
    "review": 1500,
    "plan": 2000,
    "implement": 3000,
    "refactor": 3000,
    "patch": 3000,
    "fix": 1500,
    "build": 1500,
    "test-coverage-reviewer": 1500,
    "architect": 5000,
    "audit": 5000,
    "security-review": 5000,
    "decision-explainer": 1500,
    "conflict-auditor": 1500,
    "pipeline-mapper": 1500,
}


def _required_api(name: str):
    value = getattr(router, name, None)
    assert value is not None, f"router must expose {name} for deterministic taxonomy use"
    return value


def _authority_json() -> dict:
    resource = importlib.resources.files("puppetmaster").joinpath("routing_roles.json")
    return json.loads(resource.read_text(encoding="utf-8"))


class RoutingRoleTaxonomyTests(unittest.TestCase):
    def _assert_message_names_path(self, message: str, path: Path) -> None:
        """Error text should name the resolved path and/or filename, not 8.3."""
        lowered = message.lower()
        resolved = str(path.resolve()).lower()
        filename = path.name.lower()
        self.assertTrue(
            resolved in lowered or filename in lowered,
            f"{path.name} / resolved path missing from: {message}",
        )

    def test_taxonomy_is_external_package_data_without_literal_role_tables(self) -> None:
        resource = importlib.resources.files("puppetmaster").joinpath("routing_roles.json")
        self.assertTrue(resource.is_file(), "routing_roles.json must be shipped inside puppetmaster")
        source = inspect.getsource(router)
        tree = ast.parse(source)
        hardcoded_assignments = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {target.id for target in targets if isinstance(target, ast.Name)}
            value = node.value
            if "_ROLE_BASE_SCORE" in names and isinstance(value, ast.Dict):
                hardcoded_assignments.append("_ROLE_BASE_SCORE")
            if "_TOOL_LOOP_ROLES" in names:
                literal = isinstance(value, (ast.Set, ast.List, ast.Tuple))
                literal_frozenset = (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "frozenset"
                    and bool(value.args)
                    and isinstance(value.args[0], (ast.Set, ast.List, ast.Tuple))
                )
                if literal or literal_frozenset:
                    hardcoded_assignments.append("_TOOL_LOOP_ROLES")
        self.assertEqual(hardcoded_assignments, [])
        pyproject = Path(__file__).parents[1] / "pyproject.toml"
        package_data = pyproject.read_text(encoding="utf-8")
        self.assertTrue(
            '"routing_roles.json"' in package_data or '"*.json"' in package_data
        )

    def test_external_schema_defines_every_builtin_role_and_alias(self) -> None:
        data = _authority_json()
        roles = data.get("roles", {})
        actual = {
            role: (definition.get("base_capability"), definition.get("tool_loop"))
            for role, definition in roles.items()
            if role in EXPECTED_BUILTIN_ROLES
        }
        aliases = {
            alias: role
            for role, definition in roles.items()
            for alias in definition.get("aliases", [])
        }
        self.assertEqual(data.get("schema_version"), 1)
        self.assertIsInstance(data.get("taxonomy_version"), str)
        self.assertTrue(data["taxonomy_version"].strip())
        unknown = data.get("unknown_role")
        unknown_canonical = (
            unknown.get("canonical_role") if isinstance(unknown, dict) else unknown
        )
        self.assertEqual(unknown_canonical, "unknown")
        self.assertEqual(actual, EXPECTED_BUILTIN_ROLES)
        normalized_expected_aliases = {
            alias.replace("_", "-"): canonical
            for alias, canonical in EXPECTED_ALIASES.items()
        }
        self.assertTrue(aliases.items() >= normalized_expected_aliases.items())

    def test_every_builtin_role_uses_external_capability_and_tool_classification(self) -> None:
        for role, (expected_base, expected_tool_loop) in EXPECTED_BUILTIN_ROLES.items():
            with self.subTest(role):
                task = router.TaskSignals(instruction="inspect the requested target", role=role)
                capability = router.classify_capability_needed(task)
                needs_tools = router.task_needs_tool_calling(task)
                self.assertEqual(capability, expected_base)
                self.assertIs(needs_tools, expected_tool_loop)

    def test_display_role_aliases_normalize_before_classification(self) -> None:
        for display_role, canonical_role in EXPECTED_ALIASES.items():
            with self.subTest(display_role):
                normalize = _required_api("normalize_routing_role")
                display = f"  {display_role.upper()}  "
                expected_base, expected_tools = EXPECTED_BUILTIN_ROLES[canonical_role]
                task = router.TaskSignals(instruction="inspect the requested target", role=display)
                normalized = normalize(display)
                capability = router.classify_capability_needed(task)
                needs_tools = router.task_needs_tool_calling(task)
                self.assertEqual(normalized, canonical_role)
                self.assertEqual(capability, expected_base)
                self.assertEqual(needs_tools, expected_tools)

    def test_unknown_roles_fail_safe_to_external_unknown_role(self) -> None:
        normalize = _required_api("normalize_routing_role")
        task = router.TaskSignals(instruction="inspect the requested target", role="future-specialist")
        canonical = normalize(task.role)
        capability = router.classify_capability_needed(task)
        needs_tools = router.task_needs_tool_calling(task)
        self.assertEqual(canonical, "unknown")
        self.assertEqual(capability, 60)
        self.assertIs(needs_tools, True)

    def test_externalization_preserves_existing_output_token_budgets(self) -> None:
        for role, expected_tokens in EXPECTED_LEGACY_OUTPUT_BUDGETS.items():
            with self.subTest(role):
                task = router.TaskSignals(instruction="inspect the requested target", role=role)
                tokens = router.estimate_tokens_out(task)
                self.assertEqual(tokens, expected_tokens)

    def test_loader_fails_explicitly_for_missing_or_malformed_authority(self) -> None:
        cases = (
            ("missing.json", None, "missing"),
            ("invalid-json.json", "{not-json", "not valid json"),
            (
                "invalid-schema.json",
                json.dumps({"schema_version": 1, "taxonomy_version": "test", "roles": {}}),
                "unknown_role",
            ),
        )
        for filename, contents, message_fragment in cases:
            with self.subTest(filename):
                load = _required_api("load_role_taxonomy")
                error_type = _required_api("RoleTaxonomyError")
                with TemporaryDirectory() as tmp:
                    authority_path = Path(tmp) / filename
                    if contents is not None:
                        authority_path.write_text(contents, encoding="utf-8")
                    with self.assertRaises(error_type) as caught:
                        load(authority_path)
                    message = str(caught.exception).lower()
                    self._assert_message_names_path(message, authority_path)
                    self.assertIn(message_fragment, message)

    def test_loader_rejects_alias_collision_with_reserved_unknown_role(self) -> None:
        load = _required_api("load_role_taxonomy")
        error_type = _required_api("RoleTaxonomyError")
        data = _authority_json()
        data["roles"]["audit"]["aliases"].append("unknown")
        with TemporaryDirectory() as tmp:
            authority_path = Path(tmp) / "ambiguous-unknown.json"
            authority_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(error_type) as caught:
                load(authority_path)
            message = str(caught.exception).lower()
            self._assert_message_names_path(message, authority_path)
            self.assertIn("unknown", message)
            self.assertTrue("alias" in message or "ambiguous" in message)

    def test_callers_cannot_mutate_cached_authority_or_future_routing(self) -> None:
        load = _required_api("load_role_taxonomy")
        first = load()
        first["roles"]["audit"]["base_capability"] = 1
        task = router.TaskSignals(instruction="inspect the requested target", role="audit")
        second = load()
        capability = router.classify_capability_needed(task)
        self.assertEqual(second["roles"]["audit"]["base_capability"], 85)
        self.assertEqual(capability, 85)

    def test_routing_provenance_records_original_role_canonical_role_and_version(self) -> None:
        taxonomy = _authority_json()
        model = ModelSpec(
            id="test/frontier",
            adapter="test",
            adapter_model_name="frontier",
            capability_score=100,
            tags=["tools"],
        )
        task = router.TaskSignals(instruction="inspect the requested target", role="routing_quality")
        decision = router.route_task(task, [model], policy="balanced")
        payload = decision.to_artifact_payload()
        self.assertEqual(payload["role"], "routing_quality")
        self.assertEqual(payload["canonical_role"], "audit")
        self.assertEqual(payload["taxonomy_version"], taxonomy["taxonomy_version"])



if __name__ == "__main__":
    unittest.main()
