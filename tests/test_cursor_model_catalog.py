"""Cursor SDK workers share one cached model catalog instead of each calling get_models."""
from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hermetic_env  # noqa: F401,E402

from puppetmaster import cursor_discovery  # noqa: E402
from puppetmaster.cursor_discovery import (  # noqa: E402
    LOCAL_CATALOG_ENV,
    CursorDiscoveryError,
    local_model_catalog_json,
    with_local_model_catalog,
)

ENV = {"CURSOR_API_KEY": "key-a"}
CATALOG = [{"id": "composer-2.5", "displayName": "Composer", "parameters": [{"id": "fast"}]},
           {"id": "grok-4.6", "aliases": ["grok-4-6"]}]


class CatalogCacheTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        home = patch.dict(os.environ, {"PUPPETMASTER_HOME": tmp.name})
        home.start()
        self.addCleanup(home.stop)
        self.calls = 0
        self.clock = 1000.0

    def fetch(self):
        self.calls += 1
        return CATALOG

    def catalog(self, model, env=ENV):
        return local_model_catalog_json(model, env=env, fetch=self.fetch, now=lambda: self.clock)

    def test_parallel_starts_fetch_once_and_get_id_and_aliases_only(self):
        results = []
        threads = [threading.Thread(target=lambda: results.append(self.catalog("composer-2.5")))
                   for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(self.calls, 1)
        self.assertEqual({json.dumps(json.loads(r)) for r in results},
                         {json.dumps([{"id": "composer-2.5"}, {"id": "grok-4.6", "aliases": ["grok-4-6"]}])})

    def test_a_cache_write_refused_by_an_open_reader_keeps_the_fetched_catalog(self):
        # Windows refuses os.replace while a lock-free reader holds the cache open.
        with patch("puppetmaster.cursor_discovery.os.replace",
                   side_effect=PermissionError(13, "Access is denied")):
            self.assertIsNotNone(self.catalog("composer-2.5"))
        self.assertEqual(self.calls, 1)
        self.assertIsNotNone(self.catalog("composer-2.5"))
        self.assertEqual(self.calls, 2)

    def test_alias_is_accepted_and_unknown_model_falls_back_to_the_sdk(self):
        self.assertIsNotNone(self.catalog("grok-4-6"))
        self.assertIsNone(self.catalog("model-added-after-refresh"))
        self.assertEqual(self.calls, 1)

    def test_refreshes_after_ttl_and_per_key(self):
        self.catalog("composer-2.5")
        self.clock += 599
        self.catalog("composer-2.5")
        self.assertEqual(self.calls, 1)
        self.clock += 2
        self.catalog("composer-2.5")
        self.assertEqual(self.calls, 2)
        self.catalog("composer-2.5", env={"CURSOR_API_KEY": "key-b"})
        self.assertEqual(self.calls, 3)

    def test_a_failed_refresh_backs_off_and_lets_the_sdk_fetch(self):
        def failing():
            self.calls += 1
            raise CursorDiscoveryError("rate limited")

        for _ in range(5):
            self.assertIsNone(local_model_catalog_json("composer-2.5", env=ENV, fetch=failing,
                                                       now=lambda: self.clock))
        self.assertEqual(self.calls, 1)
        self.clock += 61
        self.assertIsNotNone(self.catalog("composer-2.5"))

    def test_cache_never_holds_the_key(self):
        self.catalog("composer-2.5")
        path = cursor_discovery._catalog_cache_path("key-a")
        self.assertNotIn("key-a", path.read_text() + str(path))

    def test_no_key_opt_out_or_preset_catalog_leave_the_sdk_alone(self):
        self.assertIsNone(self.catalog("composer-2.5", env={}))
        self.assertIsNone(local_model_catalog_json("composer-2.5", env=ENV))  # hermetic autodiscover=0
        preset = {**ENV, LOCAL_CATALOG_ENV: "[]"}
        with patch.object(cursor_discovery, "local_model_catalog_json") as build:
            self.assertEqual(with_local_model_catalog(preset, "composer-2.5")[LOCAL_CATALOG_ENV], "[]")
        build.assert_not_called()


class AdapterLaunchTests(unittest.TestCase):
    def test_implement_launch_passes_the_cached_catalog(self):
        from pathlib import Path

        from puppetmaster.adapters import CursorAdapter
        from puppetmaster.adapters._base import CliInvocation
        from puppetmaster.models import Task

        seen = {}
        prepared = CliInvocation(command=["node", "runner.mjs"], sidecar_name="x",
                                 extras={"prompt": "p", "cwd": ".", "model": "composer-2.5"})
        task = Task(job_id="j", role="r", instruction="i", adapter="cursor", payload={})
        with patch.dict(os.environ, ENV), \
                patch.object(cursor_discovery, "local_model_catalog_json", return_value="[{\"id\":\"x\"}]"), \
                patch("puppetmaster.adapters.cursor.facade",
                      return_value=lambda **kw: seen.update(kw["env"]) or "done"):
            CursorAdapter()._invoke_cli(task, prepared, Path.cwd(), 5)
        self.assertEqual(seen[LOCAL_CATALOG_ENV], "[{\"id\":\"x\"}]")


if __name__ == "__main__":
    unittest.main()
