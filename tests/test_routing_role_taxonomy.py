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
from pathlib import Path
from typing import Optional

import pytest

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


def test_taxonomy_is_external_package_data_without_literal_role_tables() -> None:
    # Arrange: resolve the authority through the installed-package resource API.
    resource = importlib.resources.files("puppetmaster").joinpath("routing_roles.json")

    # Act: inspect both the resource and assignments in router.py.
    assert resource.is_file(), "routing_roles.json must be shipped inside puppetmaster"
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

    # Assert: authority is data-backed, not duplicated as Python literals, and
    # setuptools is configured to carry root JSON resources into a wheel.
    assert hardcoded_assignments == []
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    package_data = pyproject.read_text(encoding="utf-8")
    assert (
        '"routing_roles.json"' in package_data or '"*.json"' in package_data
    )


def test_external_schema_defines_every_builtin_role_and_alias() -> None:
    # Arrange: load the independent JSON authority directly.
    data = _authority_json()

    # Act: project only the behavior required by the routing classifier.
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

    # Assert: the schema/version identity and all formerly hardcoded authority
    # are represented in the packaged file.
    assert data.get("schema_version") == 1
    assert isinstance(data.get("taxonomy_version"), str)
    assert data["taxonomy_version"].strip()
    unknown = data.get("unknown_role")
    unknown_canonical = (
        unknown.get("canonical_role") if isinstance(unknown, dict) else unknown
    )
    assert unknown_canonical == "unknown"
    assert actual == EXPECTED_BUILTIN_ROLES
    normalized_expected_aliases = {
        alias.replace("_", "-"): canonical
        for alias, canonical in EXPECTED_ALIASES.items()
    }
    assert aliases.items() >= normalized_expected_aliases.items()


@pytest.mark.parametrize(
    ("role", "expected_base", "expected_tool_loop"),
    [
        pytest.param(role, values[0], values[1], id=role)
        for role, values in EXPECTED_BUILTIN_ROLES.items()
    ],
)
def test_every_builtin_role_uses_external_capability_and_tool_classification(
    role: str,
    expected_base: int,
    expected_tool_loop: bool,
) -> None:
    # Arrange: use signal-free instructions so only role authority contributes.
    task = router.TaskSignals(instruction="inspect the requested target", role=role)

    # Act: classify through the normal public router functions.
    capability = router.classify_capability_needed(task)
    needs_tools = router.task_needs_tool_calling(task)

    # Assert: every legacy built-in retains its intended behavior after moving
    # the table out of Python.
    assert capability == expected_base
    assert needs_tools is expected_tool_loop


@pytest.mark.parametrize(
    ("display_role", "canonical_role"),
    [pytest.param(alias, canonical, id=alias) for alias, canonical in EXPECTED_ALIASES.items()],
)
def test_display_role_aliases_normalize_before_classification(
    display_role: str,
    canonical_role: str,
) -> None:
    # Arrange: vary case, surrounding whitespace, and underscore separators.
    normalize = _required_api("normalize_routing_role")
    display = f"  {display_role.upper()}  "
    expected_base, expected_tools = EXPECTED_BUILTIN_ROLES[canonical_role]
    task = router.TaskSignals(instruction="inspect the requested target", role=display)

    # Act: normalize and classify using the unmodified display role.
    normalized = normalize(display)
    capability = router.classify_capability_needed(task)
    needs_tools = router.task_needs_tool_calling(task)

    # Assert: aliases cannot silently fall through to the old generic score.
    assert normalized == canonical_role
    assert capability == expected_base
    assert needs_tools is expected_tools


def test_unknown_roles_fail_safe_to_external_unknown_role() -> None:
    # Arrange: choose a display role absent from the packaged aliases.
    normalize = _required_api("normalize_routing_role")
    task = router.TaskSignals(instruction="inspect the requested target", role="future-specialist")

    # Act: normalize and classify it.
    canonical = normalize(task.role)
    capability = router.classify_capability_needed(task)
    needs_tools = router.task_needs_tool_calling(task)

    # Assert: unknown work has an explicit, conservative authority entry rather
    # than an accidental dict.get fallback.
    assert canonical == "unknown"
    assert capability == 60
    assert needs_tools is True


@pytest.mark.parametrize(
    ("role", "expected_tokens"),
    [
        pytest.param(role, tokens, id=role)
        for role, tokens in EXPECTED_LEGACY_OUTPUT_BUDGETS.items()
    ],
)
def test_externalization_preserves_existing_output_token_budgets(
    role: str,
    expected_tokens: int,
) -> None:
    # Arrange: externalization must not silently alter the cost classifier's
    # pre-existing output reservations.
    task = router.TaskSignals(instruction="inspect the requested target", role=role)

    # Act: estimate through the public router seam.
    tokens = router.estimate_tokens_out(task)

    # Assert: this task moves authority without introducing unrelated cost
    # behavior changes.
    assert tokens == expected_tokens


@pytest.mark.parametrize(
    ("filename", "contents", "message_fragment"),
    [
        ("missing.json", None, "missing"),
        ("invalid-json.json", "{not-json", "not valid json"),
        (
            "invalid-schema.json",
            json.dumps({"schema_version": 1, "taxonomy_version": "test", "roles": {}}),
            "unknown_role",
        ),
    ],
)
def test_loader_fails_explicitly_for_missing_or_malformed_authority(
    tmp_path: Path,
    filename: str,
    contents: Optional[str],
    message_fragment: str,
) -> None:
    # Arrange: create an isolated missing, syntactically invalid, or
    # semantically incomplete authority path.
    load = _required_api("load_role_taxonomy")
    error_type = _required_api("RoleTaxonomyError")
    authority_path = tmp_path / filename
    if contents is not None:
        authority_path.write_text(contents, encoding="utf-8")

    # Act: load through the explicit path seam.
    with pytest.raises(error_type) as caught:
        load(authority_path)

    # Assert: failure is deterministic and actionable; no implicit table or
    # generic-role fallback is permitted for broken authority.
    message = str(caught.value).lower()
    assert str(authority_path).lower() in message
    assert message_fragment in message


def test_loader_rejects_alias_collision_with_reserved_unknown_role(tmp_path: Path) -> None:
    # Arrange: make the reserved unknown canonical name also resolve to audit.
    load = _required_api("load_role_taxonomy")
    error_type = _required_api("RoleTaxonomyError")
    data = _authority_json()
    data["roles"]["audit"]["aliases"].append("unknown")
    authority_path = tmp_path / "ambiguous-unknown.json"
    authority_path.write_text(json.dumps(data), encoding="utf-8")

    # Act / Assert: schema validation must reject a taxonomy whose fallback
    # identity has two incompatible meanings.
    with pytest.raises(error_type) as caught:
        load(authority_path)
    message = str(caught.value).lower()
    assert str(authority_path).lower() in message
    assert "unknown" in message
    assert "alias" in message or "ambiguous" in message


def test_callers_cannot_mutate_cached_authority_or_future_routing() -> None:
    # Arrange: receive one loaded authority value and mutate the caller-owned
    # object as an accidental or hostile consumer could.
    load = _required_api("load_role_taxonomy")
    first = load()
    first["roles"]["audit"]["base_capability"] = 1
    task = router.TaskSignals(instruction="inspect the requested target", role="audit")

    # Act: load and classify again.
    second = load()
    capability = router.classify_capability_needed(task)

    # Assert: a public read cannot corrupt the cached authority and make later
    # routing non-deterministic inside the same process.
    assert second["roles"]["audit"]["base_capability"] == 85
    assert capability == 85


def test_routing_provenance_records_original_role_canonical_role_and_version() -> None:
    # Arrange: route a custom audit alias through a single unquestionably
    # eligible tool-capable model.
    taxonomy = _authority_json()
    model = ModelSpec(
        id="test/frontier",
        adapter="test",
        adapter_model_name="frontier",
        capability_score=100,
        tags=["tools"],
    )
    task = router.TaskSignals(instruction="inspect the requested target", role="routing_quality")

    # Act: materialize the persisted ROUTING payload.
    decision = router.route_task(task, [model], policy="balanced")
    payload = decision.to_artifact_payload()

    # Assert: a later audit can reproduce which taxonomy interpretation was
    # used without erasing the user-facing display role.
    assert payload["role"] == "routing_quality"
    assert payload["canonical_role"] == "audit"
    assert payload["taxonomy_version"] == taxonomy["taxonomy_version"]
