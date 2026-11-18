from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from puppetmaster import codegraph as cg

_ENV = "PUPPETMASTER_CODEGRAPH_PROVISION_TIMEOUT_SECONDS"


class ProvisionTimeoutTests(unittest.TestCase):
    def _timeout(self, raw: str) -> int:
        with patch.dict(os.environ, {_ENV: raw}, clear=False):
            return cg._provision_timeout_seconds()

    def test_infinite_values_fall_back_to_default(self) -> None:
        for raw in ("inf", "-inf", "1e999", "Infinity"):
            with self.subTest(raw=raw):
                self.assertEqual(self._timeout(raw), 300)

    def test_finite_values_keep_their_result(self) -> None:
        self.assertEqual(self._timeout("600"), 600)
        self.assertEqual(self._timeout("1e3"), 1000)
        self.assertEqual(self._timeout("5"), 30)
        self.assertEqual(self._timeout("abc"), 300)
        self.assertEqual(self._timeout("nan"), 300)

    def test_provision_does_not_raise_on_infinite_timeout(self) -> None:
        cg.reset_codegraph_provisioning_state()

        def fake_which(name, *a, **k):
            if name in ("npx", "npm"):
                return "/usr/bin/" + name
            return None

        removed = (
            "PUPPETMASTER_CODEGRAPH_NODE",
            "PUPPETMASTER_CODEGRAPH_JS",
            "PUPPETMASTER_CODEGRAPH_NO_NPX",
            "PUPPETMASTER_CODEGRAPH_NO_GLOBAL_INSTALL",
        )
        try:
            with patch.object(cg.shutil, "which", side_effect=fake_which), \
                 patch.object(cg, "_cursor_codegraph_invocation", return_value=None), \
                 patch.object(cg, "_run_cli", side_effect=OSError("stub")) as run_cli, \
                 patch.dict(os.environ, {_ENV: "inf"}, clear=False):
                for var in removed:
                    os.environ.pop(var, None)
                result = cg.ensure_codegraph_provisioned()
            self.assertIsInstance(result, bool)
            for call in run_cli.call_args_list:
                self.assertEqual(call.kwargs.get("timeout"), 300)
        finally:
            cg.reset_codegraph_provisioning_state()


if __name__ == "__main__":
    unittest.main()
