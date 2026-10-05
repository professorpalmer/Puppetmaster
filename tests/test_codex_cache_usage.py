"""Codex reports its input-inclusive cache hits as cached_input_tokens."""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hermetic_env  # noqa: F401,E402

from puppetmaster.cost import _cost_with_cache_discount  # noqa: E402
from puppetmaster.models import Artifact, ArtifactType  # noqa: E402
from puppetmaster.usage import select_usage_records  # noqa: E402

# Payload a real `codex exec --json` build worker stamped (2026-10-05).
CODEX_PAYLOAD = {
    "model": "gpt-5.6-luna",
    "tokens_in": 97124,
    "tokens_out": 1202,
    "cached_input_tokens": 92160,
    "tokens_estimated": False,
}


class CodexCacheUsageTests(unittest.TestCase):
    def test_cache_hits_are_not_priced_as_fresh_input(self) -> None:
        artifact = Artifact(job_id="job_x", task_id="task_x", type=ArtifactType.VERIFICATION,
                            created_by="worker", payload=dict(CODEX_PAYLOAD),
                            confidence=1.0, evidence=[])
        record = select_usage_records([artifact])["task_x"]
        self.assertEqual(record["tokens_cached"], 92160)
        spec = SimpleNamespace(input_per_mtok_usd=1.0, output_per_mtok_usd=0.0,
                               output_multiplier=1.0)
        cost = _cost_with_cache_discount(spec, record["tokens_in"], record["tokens_out"],
                                         record["tokens_cached"])
        # 4964 fresh at full price plus 92160 cached at a tenth.
        self.assertAlmostEqual(cost, (4964 + 9216) / 1_000_000, places=6)


if __name__ == "__main__":
    unittest.main()
