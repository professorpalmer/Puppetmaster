from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import hermetic_env  # noqa: E402

from puppetmaster.installers import omp_agent_dir, pi_agent_dir  # noqa: E402


class HermeticHostDirTests(unittest.TestCase):
    def test_pilot_installers_resolve_inside_the_test_sandbox(self) -> None:
        sandbox = Path(hermetic_env._ISOLATION_TMP).resolve()
        for resolved in (pi_agent_dir(), omp_agent_dir()):
            with self.subTest(dir=str(resolved)):
                self.assertTrue(resolved.resolve().is_relative_to(sandbox), resolved)


if __name__ == "__main__":
    unittest.main()
