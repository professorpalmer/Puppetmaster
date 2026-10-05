"""A curated (shipped) catalog that lags a new release must not block a registered model."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hermetic_env  # noqa: F401,E402

from puppetmaster.model_registry import (  # noqa: E402
    DISCOVERY_ORIGIN_CURATED,
    ModelSpec,
    save_registry,
    write_discovery_meta,
)
from puppetmaster.platform_billing import BillingStatus  # noqa: E402
from puppetmaster.preflight import preflight_check  # noqa: E402
from puppetmaster.static_catalog import curated_catalog  # noqa: E402


class CuratedCatalogPreflightTests(unittest.TestCase):
    def _check(self, origin: str) -> object:
        with TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "models.json"
            save_registry([ModelSpec(id="claude-code/opus-9", adapter="claude-code",
                                     adapter_model_name="claude-opus-9", capability_score=100,
                                     billing="plan")], registry_path)
            write_discovery_meta("claude", 1, registry_path, model_ids=["claude-opus-5"], origin=origin)
            with patch.dict(os.environ, {"PUPPETMASTER_MODELS_PATH": str(registry_path)}):
                return preflight_check(
                    "claude-code", "claude-opus-9",
                    billing_status=BillingStatus(adapter="claude-code", billing="plan",
                                                 healthy=True, detail="oauth", evidence=[]),
                )

    def test_curated_catalog_miss_is_advisory(self) -> None:
        result = self._check(DISCOVERY_ORIGIN_CURATED)
        self.assertTrue(result.ok, result.reason)
        self.assertNotIn("preflight:cached_model_not_in_catalog", result.evidence)

    def test_live_catalog_miss_still_blocks(self) -> None:
        result = self._check("live")
        self.assertFalse(result.ok)
        self.assertIn("preflight:cached_model_not_in_catalog", result.evidence)

    def test_curated_claude_catalog_lists_opus_5_5(self) -> None:
        self.assertIn("claude-opus-5-5", [entry["model"] for entry in curated_catalog("claude-code")])


if __name__ == "__main__":
    unittest.main()
