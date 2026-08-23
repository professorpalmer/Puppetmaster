"""Adversarial registry-governance tests for TASK-004.

These tests intentionally exercise public registry and discovery seams.  They
define the authority boundary before the production implementation exists:
malformed or ambiguous registries never partially load, and a local retirement
cannot be undone by catalog reconciliation.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation


def _entry(
    model_id: str = "cursor/grok-4-5",
    *,
    adapter: str = "cursor",
    adapter_model_name: str = "grok-4.5",
) -> dict:
    return {
        "id": model_id,
        "adapter": adapter,
        "adapter_model_name": adapter_model_name,
        "capability_score": 90,
    }


def _write_registry(tmp_path: Path, payload) -> Path:
    path = tmp_path / "models.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _retired_spec(*, adapter_model_name: str = "grok-4.5"):
    from puppetmaster.model_registry import ModelSpec

    return ModelSpec(
        id="cursor/retired-grok",
        adapter="cursor",
        adapter_model_name=adapter_model_name,
        capability_score=99,
        enabled=False,
        retired=True,
        retirement_reason="Superseded after local qualification",
        retirement_authority="user:registry-owner",
    )


class RegistryRetirementTests(unittest.TestCase):
    def test_load_registry_accepts_both_documented_forms(self) -> None:
        from puppetmaster.model_registry import load_registry

        payloads = {
            "documented-list-form": [_entry()],
            "documented-object-form": {"models": [_entry()]},
        }
        for case_id, payload in payloads.items():
            with self.subTest(case_id), TemporaryDirectory() as tmp:
                path = _write_registry(Path(tmp), payload)
                specs = load_registry(path)
                self.assertEqual([spec.id for spec in specs], ["cursor/grok-4-5"])

    def test_load_registry_rejects_malformed_authority_as_runtime_error(self) -> None:
        from puppetmaster.model_registry import load_registry

        cases = {
            "unsupported-object-envelope": ({"unexpected": []}, "models"),
            "non-mapping-entry": ({"models": ["not-an-object"]}, "entry 0"),
            "invalid-adapter": ({"models": [_entry(adapter="invented-adapter")]}, "adapter"),
            "missing-required-fields": ({"models": [{"id": "missing-required-fields"}]}, "entry 0"),
        }
        for case_id, (payload, expected) in cases.items():
            with self.subTest(case_id), TemporaryDirectory() as tmp:
                path = _write_registry(Path(tmp), payload)
                with self.assertRaisesRegex(RuntimeError, expected):
                    load_registry(path)

    def test_load_registry_rejects_duplicate_registry_ids(self) -> None:
        from puppetmaster.model_registry import load_registry

        with TemporaryDirectory() as tmp:
            path = _write_registry(
                Path(tmp),
                {
                    "models": [
                        _entry(),
                        _entry(adapter_model_name="grok-4.6"),
                    ]
                },
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate.*id"):
                load_registry(path)

    def test_load_registry_rejects_slug_equivalent_identity_twins(self) -> None:
        from puppetmaster.model_registry import load_registry

        with TemporaryDirectory() as tmp:
            path = _write_registry(
                Path(tmp),
                {
                    "models": [
                        _entry("cursor/grok-dot", adapter_model_name="grok-4.5"),
                        _entry("cursor/grok-dash", adapter_model_name="grok-4-5"),
                    ]
                },
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate.*identity|ambiguous.*identity"):
                load_registry(path)

    def test_retirement_requires_reason_and_authority_metadata(self) -> None:
        from puppetmaster.model_registry import ModelSpec

        common = {
            "id": "cursor/retired-grok",
            "adapter": "cursor",
            "adapter_model_name": "grok-4.5",
            "enabled": False,
            "retired": True,
        }
        with self.assertRaisesRegex(ValueError, "reason"):
            ModelSpec(**common, retirement_authority="user:registry-owner")
        with self.assertRaisesRegex(ValueError, "authority"):
            ModelSpec(**common, retirement_reason="Superseded")

    def test_retired_model_is_ineligible_for_routing_and_direct_pin(self) -> None:
        from puppetmaster.model_registry import enabled_specs, resolve_model_pin

        retired = _retired_spec()
        routable = enabled_specs([retired])
        pin = resolve_model_pin("cursor/retired-grok", [retired], adapter="cursor")
        self.assertEqual(routable, [])
        self.assertIsNone(pin)

    def test_retirement_metadata_round_trips_without_losing_quarantine(self) -> None:
        from puppetmaster.model_registry import load_registry, save_registry

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            retired = _retired_spec()
            save_registry([retired], path)
            loaded = load_registry(path)
            self.assertEqual(len(loaded), 1)
            self.assertIs(loaded[0].retired, True)
            self.assertIs(loaded[0].enabled, False)
            self.assertEqual(loaded[0].retirement_reason, retired.retirement_reason)
            self.assertEqual(loaded[0].retirement_authority, retired.retirement_authority)

    def test_cursor_discovery_preserves_slug_equivalent_retirement(self) -> None:
        from puppetmaster.cursor_discovery import merge_catalog_into_registry
        from puppetmaster.model_registry import normalize_model_token

        retired = _retired_spec(adapter_model_name="grok-4-5")
        merged, _report = merge_catalog_into_registry(
            [retired],
            [{"id": "grok-4.5", "displayName": "Grok 4.5"}],
        )
        matches = [
            spec
            for spec in merged
            if spec.adapter == "cursor"
            and normalize_model_token(spec.adapter_model_name)
            == normalize_model_token("grok-4.5")
        ]
        self.assertEqual(len(matches), 1)
        self.assertIs(matches[0].retired, True)
        self.assertIs(matches[0].enabled, False)
        self.assertEqual(matches[0].retirement_authority, "user:registry-owner")

    def test_curated_discovery_does_not_reenable_retired_identity(self) -> None:
        from puppetmaster.model_registry import ModelSpec
        from puppetmaster.static_catalog import curated_catalog, merge_curated_into_registry

        catalog_model = str(curated_catalog("antigravity")[0]["model"])
        retired = ModelSpec(
            id="antigravity/locally-retired",
            adapter="antigravity",
            adapter_model_name=catalog_model,
            enabled=False,
            retired=True,
            retirement_reason="Retired by local policy",
            retirement_authority="user:registry-owner",
        )
        merged, _report = merge_curated_into_registry(
            "antigravity", "plan", [retired]
        )
        matches = [
            spec
            for spec in merged
            if spec.adapter == "antigravity"
            and spec.adapter_model_name == catalog_model
        ]
        self.assertEqual(len(matches), 1)
        self.assertIs(matches[0].retired, True)
        self.assertIs(matches[0].enabled, False)
        self.assertEqual(matches[0].retirement_reason, "Retired by local policy")

    def test_curated_discovery_preserves_punctuation_equivalent_retirement(self) -> None:
        from puppetmaster.model_registry import ModelSpec, normalize_model_token
        from puppetmaster.static_catalog import curated_catalog, merge_curated_into_registry

        catalog_model = next(
            str(item["model"])
            for item in curated_catalog("antigravity")
            if str(item["model"]) != normalize_model_token(str(item["model"]))
        )
        retired = ModelSpec(
            id="antigravity/punctuation-retired",
            adapter="antigravity",
            adapter_model_name=normalize_model_token(catalog_model),
            enabled=False,
            retired=True,
            retirement_reason="Retired across catalog spellings",
            retirement_authority="user:registry-owner",
        )
        merged, _report = merge_curated_into_registry(
            "antigravity", "plan", [retired]
        )
        matches = [
            spec
            for spec in merged
            if spec.adapter == "antigravity"
            and normalize_model_token(spec.adapter_model_name)
            == normalize_model_token(catalog_model)
        ]
        self.assertEqual(len(matches), 1)
        self.assertIs(matches[0].retired, True)
        self.assertIs(matches[0].enabled, False)
        self.assertEqual(matches[0].retirement_authority, "user:registry-owner")

    def test_api_discovery_does_not_reenable_retired_identity(self) -> None:
        from puppetmaster.api_discovery import merge_api_catalog_into_registry
        from puppetmaster.model_registry import ModelSpec

        retired = ModelSpec(
            id="openai/locally-retired",
            adapter="openai",
            adapter_model_name="gpt-retired",
            enabled=False,
            retired=True,
            retirement_reason="Retired by local policy",
            retirement_authority="user:registry-owner",
        )
        merged, _report = merge_api_catalog_into_registry(
            "openai",
            "api",
            [retired],
            [{"id": "gpt-retired"}],
        )
        matches = [spec for spec in merged if spec.adapter_model_name == "gpt-retired"]
        self.assertEqual(len(matches), 1)
        self.assertIs(matches[0].retired, True)
        self.assertIs(matches[0].enabled, False)

    def test_api_discovery_preserves_punctuation_equivalent_retirement(self) -> None:
        from puppetmaster.api_discovery import merge_api_catalog_into_registry
        from puppetmaster.model_registry import ModelSpec, normalize_model_token

        retired = ModelSpec(
            id="openai/punctuation-retired",
            adapter="openai",
            adapter_model_name="gpt.retired",
            enabled=False,
            retired=True,
            retirement_reason="Retired across catalog spellings",
            retirement_authority="user:registry-owner",
        )
        merged, _report = merge_api_catalog_into_registry(
            "openai",
            "api",
            [retired],
            [{"id": "gpt-retired"}],
        )
        matches = [
            spec
            for spec in merged
            if spec.adapter == "openai"
            and normalize_model_token(spec.adapter_model_name)
            == normalize_model_token("gpt-retired")
        ]
        self.assertEqual(len(matches), 1)
        self.assertIs(matches[0].retired, True)
        self.assertIs(matches[0].enabled, False)
        self.assertEqual(matches[0].retirement_authority, "user:registry-owner")

    def test_registry_digest_and_schema_version_are_stable_across_round_trip(self) -> None:
        import puppetmaster.model_registry as registry_module

        retired = _retired_spec()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            before = registry_module.registry_digest([retired])
            registry_module.save_registry([retired], path)
            after = registry_module.registry_digest(registry_module.load_registry(path))
            self.assertGreaterEqual(registry_module.REGISTRY_SCHEMA_VERSION, 1)
            self.assertEqual(before, after)
            self.assertEqual(len(before), 64)
            int(before, 16)

    def test_registry_digest_changes_for_routing_or_quarantine_authority(self) -> None:
        import puppetmaster.model_registry as registry_module

        retired = _retired_spec()
        baseline = registry_module.registry_digest([retired])
        capability_changed = registry_module.registry_digest(
            [replace(retired, capability_score=retired.capability_score - 1)]
        )
        authority_changed = registry_module.registry_digest(
            [replace(retired, retirement_authority="policy:security-team")]
        )
        self.assertNotEqual(baseline, capability_changed)
        self.assertNotEqual(baseline, authority_changed)

    def test_registry_digest_changes_when_routing_precedence_order_changes(self) -> None:
        import puppetmaster.model_registry as registry_module

        first = registry_module.ModelSpec(
            id="cursor/equal-first",
            adapter="cursor",
            adapter_model_name="equal-first",
            capability_score=90,
            input_per_mtok_usd=1.0,
            output_per_mtok_usd=1.0,
        )
        second = registry_module.ModelSpec(
            id="cursor/equal-second",
            adapter="cursor",
            adapter_model_name="equal-second",
            capability_score=90,
            input_per_mtok_usd=1.0,
            output_per_mtok_usd=1.0,
        )
        first_precedence = registry_module.registry_digest([first, second])
        second_precedence = registry_module.registry_digest([second, first])
        self.assertNotEqual(first_precedence, second_precedence)


if __name__ == "__main__":
    unittest.main()
