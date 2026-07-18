"""Regression: unittest discover must isolate host Puppetmaster env pins."""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import unittest
from pathlib import Path

from puppetmaster.platform_lock import KNOWN_ADAPTERS, ONLY_ENV


class HermeticEnvIsolationTests(unittest.TestCase):
    def test_models_path_points_at_missing_sentinel(self) -> None:
        path = Path(os.environ["PUPPETMASTER_MODELS_PATH"])
        self.assertFalse(path.is_file())
        self.assertIn("pm-test-empty-", str(path))

    def test_platform_lock_env_enables_every_known_adapter(self) -> None:
        enabled = {part.strip() for part in os.environ[ONLY_ENV].split(",") if part.strip()}
        self.assertEqual(enabled, set(KNOWN_ADAPTERS))

    def test_codegraph_runtime_pins_are_cleared(self) -> None:
        self.assertNotIn("PUPPETMASTER_CODEGRAPH_NODE", os.environ)
        self.assertNotIn("PUPPETMASTER_CODEGRAPH_JS", os.environ)

    def test_autodiscover_disabled_for_suite(self) -> None:
        self.assertEqual(os.environ.get("PUPPETMASTER_AUTODISCOVER"), "0")


if __name__ == "__main__":
    unittest.main()
