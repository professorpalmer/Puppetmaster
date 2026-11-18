from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

from puppetmaster.model_registry import ModelSpec, save_registry, starter_registry
from puppetmaster.routing_authority import (
    RegistryAuthorityError,
    _pin_failure_detail,
    resolve_and_bind_explicit_pin,
    validate_pinned_dispatch,
)


def _starter(model_id: str) -> ModelSpec:
    return next(spec for spec in starter_registry() if spec.id == model_id)


class PinFailureDetailCoveredIdTests(unittest.TestCase):
    def test_disabled_entry_with_same_identity_gets_enable_advice(self) -> None:
        spec = dataclasses.replace(
            _starter("claude-code/haiku-5-5"),
            id="claude-code/my-haiku",
            enabled=False,
            disabled_reason="x",
            disabled_authority="owner",
        )
        detail = _pin_failure_detail("claude-haiku-5-5", [spec], "claude-code")
        self.assertIn("'claude-code/my-haiku' is disabled: x", detail)
        self.assertIn("puppetmaster models set claude-code/my-haiku enabled=true", detail)
        self.assertNotIn("--apply", detail)

    def test_retired_entry_with_same_identity_gets_reason(self) -> None:
        spec = dataclasses.replace(
            _starter("claude-code/haiku-5-5"),
            id="claude-code/my-haiku",
            enabled=False,
            disabled_reason="retired",
            disabled_authority="owner",
            retired=True,
            retirement_reason="gone",
            retirement_authority="owner",
        )
        detail = _pin_failure_detail("claude-haiku-5-5", [spec], "claude-code")
        self.assertIn("'claude-code/my-haiku' is retired: gone", detail)

    def test_absent_identity_keeps_sync_advice(self) -> None:
        other = _starter("claude-code/sonnet-5-5")
        detail = _pin_failure_detail("claude-haiku-5-5", [other], "claude-code")
        self.assertIn("starter registry entry that this registry does not have", detail)
        self.assertIn("puppetmaster models sync --apply", detail)


class ValidatePinnedDispatchProRemapTests(unittest.TestCase):
    def _bind(self, spec: ModelSpec, tmp: str) -> dict:
        reg = Path(tmp) / "models.json"
        save_registry([spec], reg)
        return resolve_and_bind_explicit_pin(
            {"model": spec.id}, adapter="agentic", registry_path=reg
        )

    def test_pro_registry_wire_passes_dispatch(self) -> None:
        spec = ModelSpec(
            id="agentic/gpt-5.6-sol-pro",
            adapter="agentic",
            adapter_model_name="gpt-5.6-sol-pro",
            billing="api",
        )
        with tempfile.TemporaryDirectory() as tmp:
            bound = self._bind(spec, tmp)
            out = validate_pinned_dispatch(bound, adapter="agentic")
        self.assertEqual(bound["model"], "gpt-5.6-sol")
        self.assertEqual(out["model"], "gpt-5.6-sol")
        self.assertEqual(out["codex_pro_remapped_from"], "gpt-5.6-sol-pro")

    def test_non_pro_registry_wire_still_passes(self) -> None:
        spec = ModelSpec(
            id="agentic/gpt-5.6-sol",
            adapter="agentic",
            adapter_model_name="gpt-5.6-sol",
            billing="api",
        )
        with tempfile.TemporaryDirectory() as tmp:
            bound = self._bind(spec, tmp)
            out = validate_pinned_dispatch(bound, adapter="agentic")
        self.assertEqual(out["model"], "gpt-5.6-sol")

    def test_diverged_wire_still_fails(self) -> None:
        spec = ModelSpec(
            id="agentic/gpt-5.6-sol-pro",
            adapter="agentic",
            adapter_model_name="gpt-5.6-sol-pro",
            billing="api",
        )
        with tempfile.TemporaryDirectory() as tmp:
            bound = self._bind(spec, tmp)
            bound["model"] = "gpt-5.6-luna"
            bound["pinned_adapter_model_name"] = "gpt-5.6-luna"
            with self.assertRaisesRegex(RegistryAuthorityError, "diverge"):
                validate_pinned_dispatch(bound, adapter="agentic")


if __name__ == "__main__":
    unittest.main()
