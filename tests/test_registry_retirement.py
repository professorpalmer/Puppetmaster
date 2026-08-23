"""Adversarial registry-governance tests for TASK-004.

These tests intentionally exercise public registry and discovery seams.  They
define the authority boundary before the production implementation exists:
malformed or ambiguous registries never partially load, and a local retirement
cannot be undone by catalog reconciliation.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest


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


def _write_registry(tmp_path, payload):
    path = tmp_path / "models.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "payload",
    [
        [_entry()],
        {"models": [_entry()]},
    ],
    ids=["documented-list-form", "documented-object-form"],
)
def test_load_registry_accepts_both_documented_forms(tmp_path, payload) -> None:
    # Arrange
    from puppetmaster.model_registry import load_registry

    path = _write_registry(tmp_path, payload)

    # Act
    specs = load_registry(path)

    # Assert
    assert [spec.id for spec in specs] == ["cursor/grok-4-5"]


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"unexpected": []}, "models"),
        ({"models": ["not-an-object"]}, "entry 0"),
        ({"models": [_entry(adapter="invented-adapter")]}, "adapter"),
        ({"models": [{"id": "missing-required-fields"}]}, "entry 0"),
    ],
    ids=[
        "unsupported-object-envelope",
        "non-mapping-entry",
        "invalid-adapter",
        "missing-required-fields",
    ],
)
def test_load_registry_rejects_malformed_authority_as_runtime_error(
    tmp_path, payload, expected
) -> None:
    # Arrange
    from puppetmaster.model_registry import load_registry

    path = _write_registry(tmp_path, payload)

    # Act / Assert
    with pytest.raises(RuntimeError, match=expected):
        load_registry(path)


def test_load_registry_rejects_duplicate_registry_ids(tmp_path) -> None:
    # Arrange
    from puppetmaster.model_registry import load_registry

    path = _write_registry(
        tmp_path,
        {
            "models": [
                _entry(),
                _entry(adapter_model_name="grok-4.6"),
            ]
        },
    )

    # Act / Assert
    with pytest.raises(RuntimeError, match="duplicate.*id"):
        load_registry(path)


def test_load_registry_rejects_slug_equivalent_identity_twins(tmp_path) -> None:
    # Arrange: both entries can satisfy the same direct pin after normalization.
    from puppetmaster.model_registry import load_registry

    path = _write_registry(
        tmp_path,
        {
            "models": [
                _entry("cursor/grok-dot", adapter_model_name="grok-4.5"),
                _entry("cursor/grok-dash", adapter_model_name="grok-4-5"),
            ]
        },
    )

    # Act / Assert
    with pytest.raises(RuntimeError, match="duplicate.*identity|ambiguous.*identity"):
        load_registry(path)


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


def test_retirement_requires_reason_and_authority_metadata() -> None:
    # Arrange
    from puppetmaster.model_registry import ModelSpec

    common = {
        "id": "cursor/retired-grok",
        "adapter": "cursor",
        "adapter_model_name": "grok-4.5",
        "enabled": False,
        "retired": True,
    }

    # Act / Assert
    with pytest.raises(ValueError, match="reason"):
        ModelSpec(**common, retirement_authority="user:registry-owner")
    with pytest.raises(ValueError, match="authority"):
        ModelSpec(**common, retirement_reason="Superseded")


def test_retired_model_is_ineligible_for_routing_and_direct_pin() -> None:
    # Arrange
    from puppetmaster.model_registry import enabled_specs, resolve_model_pin

    retired = _retired_spec()

    # Act
    routable = enabled_specs([retired])
    pin = resolve_model_pin("cursor/retired-grok", [retired], adapter="cursor")

    # Assert
    assert routable == []
    assert pin is None


def test_retirement_metadata_round_trips_without_losing_quarantine(tmp_path) -> None:
    # Arrange
    from puppetmaster.model_registry import load_registry, save_registry

    path = tmp_path / "models.json"
    retired = _retired_spec()

    # Act
    save_registry([retired], path)
    loaded = load_registry(path)

    # Assert
    assert len(loaded) == 1
    assert loaded[0].retired is True
    assert loaded[0].enabled is False
    assert loaded[0].retirement_reason == retired.retirement_reason
    assert loaded[0].retirement_authority == retired.retirement_authority


def test_cursor_discovery_preserves_slug_equivalent_retirement() -> None:
    # Arrange
    from puppetmaster.cursor_discovery import merge_catalog_into_registry
    from puppetmaster.model_registry import normalize_model_token

    retired = _retired_spec(adapter_model_name="grok-4-5")

    # Act
    merged, _report = merge_catalog_into_registry(
        [retired],
        [{"id": "grok-4.5", "displayName": "Grok 4.5"}],
    )

    # Assert: discovery may refresh spelling, but cannot create an enabled twin.
    matches = [
        spec
        for spec in merged
        if spec.adapter == "cursor"
        and normalize_model_token(spec.adapter_model_name)
        == normalize_model_token("grok-4.5")
    ]
    assert len(matches) == 1
    assert matches[0].retired is True
    assert matches[0].enabled is False
    assert matches[0].retirement_authority == "user:registry-owner"


def test_curated_discovery_does_not_reenable_retired_identity() -> None:
    # Arrange
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

    # Act
    merged, _report = merge_curated_into_registry(
        "antigravity", "plan", [retired]
    )

    # Assert
    matches = [
        spec
        for spec in merged
        if spec.adapter == "antigravity"
        and spec.adapter_model_name == catalog_model
    ]
    assert len(matches) == 1
    assert matches[0].retired is True
    assert matches[0].enabled is False
    assert matches[0].retirement_reason == "Retired by local policy"


def test_curated_discovery_preserves_punctuation_equivalent_retirement() -> None:
    # Arrange: curated catalogs may spell an identity with dots while a local
    # quarantine uses its slug form. Those are the same executable identity.
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

    # Act
    merged, _report = merge_curated_into_registry(
        "antigravity", "plan", [retired]
    )

    # Assert: refresh may adopt catalog punctuation but cannot create a twin.
    matches = [
        spec
        for spec in merged
        if spec.adapter == "antigravity"
        and normalize_model_token(spec.adapter_model_name)
        == normalize_model_token(catalog_model)
    ]
    assert len(matches) == 1
    assert matches[0].retired is True
    assert matches[0].enabled is False
    assert matches[0].retirement_authority == "user:registry-owner"


def test_api_discovery_does_not_reenable_retired_identity() -> None:
    # Arrange
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

    # Act
    merged, _report = merge_api_catalog_into_registry(
        "openai",
        "api",
        [retired],
        [{"id": "gpt-retired"}],
    )

    # Assert
    matches = [spec for spec in merged if spec.adapter_model_name == "gpt-retired"]
    assert len(matches) == 1
    assert matches[0].retired is True
    assert matches[0].enabled is False


def test_api_discovery_preserves_punctuation_equivalent_retirement() -> None:
    # Arrange
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

    # Act
    merged, _report = merge_api_catalog_into_registry(
        "openai",
        "api",
        [retired],
        [{"id": "gpt-retired"}],
    )

    # Assert
    matches = [
        spec
        for spec in merged
        if spec.adapter == "openai"
        and normalize_model_token(spec.adapter_model_name)
        == normalize_model_token("gpt-retired")
    ]
    assert len(matches) == 1
    assert matches[0].retired is True
    assert matches[0].enabled is False
    assert matches[0].retirement_authority == "user:registry-owner"


def test_registry_digest_and_schema_version_are_stable_across_round_trip(
    tmp_path,
) -> None:
    # Arrange
    import puppetmaster.model_registry as registry_module

    retired = _retired_spec()
    path = tmp_path / "models.json"

    # Act
    before = registry_module.registry_digest([retired])
    registry_module.save_registry([retired], path)
    after = registry_module.registry_digest(registry_module.load_registry(path))

    # Assert
    assert registry_module.REGISTRY_SCHEMA_VERSION >= 1
    assert before == after
    assert len(before) == 64
    int(before, 16)


def test_registry_digest_changes_for_routing_or_quarantine_authority() -> None:
    # Arrange
    import puppetmaster.model_registry as registry_module

    retired = _retired_spec()

    # Act
    baseline = registry_module.registry_digest([retired])
    capability_changed = registry_module.registry_digest(
        [replace(retired, capability_score=retired.capability_score - 1)]
    )
    authority_changed = registry_module.registry_digest(
        [replace(retired, retirement_authority="policy:security-team")]
    )

    # Assert
    assert baseline != capability_changed
    assert baseline != authority_changed


def test_registry_digest_changes_when_routing_precedence_order_changes() -> None:
    # Arrange: router min/max tie-breaking retains registry insertion order for
    # otherwise equal candidates, so row order is material reproduction state.
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

    # Act
    first_precedence = registry_module.registry_digest([first, second])
    second_precedence = registry_module.registry_digest([second, first])

    # Assert
    assert first_precedence != second_precedence
