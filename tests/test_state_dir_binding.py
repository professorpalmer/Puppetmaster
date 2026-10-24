"""An inline run binds its store per context, never in os.environ.

Two inline runs on different stores in one process (Marionette runs swarms
inline in its backend) must each resolve their own state dir. Before the
binding, ``_run_inline_workers`` set ``PUPPETMASTER_STATE_DIR`` process-wide,
so worker output of one run could land in the other run's store.
"""
from __future__ import annotations

import os
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from puppetmaster.adapters import _delta_stream, _streaming
from puppetmaster.adapters.agentic import AgenticAdapter
from puppetmaster.models import Task
from puppetmaster.orchestrator import Orchestrator
from puppetmaster.state import STATE_DIR_ENV, bind_state_dir, resolve_state_dir, state_dir_from_env
from puppetmaster.store import SwarmStore
from puppetmaster.worker_fence import stamp_worker_env
from puppetmaster.workers import WorkerSpec


def _seen() -> dict:
    return {
        "bound": state_dir_from_env(),
        "sidecar": str(_streaming._resolve_sidecar_state_dir()),
        "delta": str(_delta_stream._resolve_state_dir()),
        "resolved": str(resolve_state_dir()),
        "env": os.environ.get(STATE_DIR_ENV),
    }


class StateDirBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(STATE_DIR_ENV, None)

    def test_binding_is_local_to_each_thread_context(self) -> None:
        inside = threading.Barrier(2, timeout=5)
        seen: dict = {}

        def run(name: str, path: str) -> None:
            with bind_state_dir(path):
                inside.wait()
                seen[name] = _seen()
                inside.wait()
            seen[name + "_after"] = state_dir_from_env()

        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            threads = [threading.Thread(target=run, args=("a", a)), threading.Thread(target=run, args=("b", b))]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            for name, path in (("a", a), ("b", b)):
                self.assertEqual(seen[name]["bound"], path)
                self.assertEqual(seen[name]["sidecar"], path)
                self.assertEqual(seen[name]["delta"], path)
                self.assertEqual(Path(seen[name]["resolved"]).resolve(), Path(path).resolve())
                self.assertIsNone(seen[name]["env"])
                self.assertIsNone(seen[name + "_after"])

    def test_binding_wins_over_the_process_env(self) -> None:
        os.environ[STATE_DIR_ENV] = "/env/state"
        self.assertEqual(state_dir_from_env(), "/env/state")
        with bind_state_dir("/bound/state"):
            self.assertEqual(state_dir_from_env(), "/bound/state")
        self.assertEqual(state_dir_from_env(), "/env/state")

    def test_child_env_gets_the_bound_state_dir(self) -> None:
        self.assertNotIn(STATE_DIR_ENV, stamp_worker_env({}))
        self.assertEqual(stamp_worker_env({STATE_DIR_ENV: "/kept"})[STATE_DIR_ENV], "/kept")
        with bind_state_dir("/bound/state"):
            self.assertEqual(stamp_worker_env({STATE_DIR_ENV: "/old"})[STATE_DIR_ENV], "/bound/state")

    def test_concurrent_inline_runs_each_see_their_own_store(self) -> None:
        from puppetmaster.worker_runtime import WorkerRuntime

        original = WorkerRuntime.run_until_idle
        inside = threading.Barrier(2, timeout=10)
        seen: dict = {}
        lock = threading.Lock()

        def run_until_idle(runtime: WorkerRuntime) -> int:
            first = False
            with lock:
                if runtime.store.root not in seen:
                    seen[runtime.store.root] = None
                    first = True
            if first:
                inside.wait()
                observed = _seen()
                inside.wait()
                with lock:
                    seen[runtime.store.root] = observed
            return original(runtime)

        with TemporaryDirectory() as tmp:
            stores = [SwarmStore(Path(tmp) / name) for name in ("plain", "sealed")]
            errors: list = []

            def run(store: SwarmStore) -> None:
                try:
                    Orchestrator(store).run(
                        "binding probe", specs=[WorkerSpec(role="explore", instruction="look")],
                        worker_mode="inline",
                    )
                except BaseException as exc:  # surfaced below
                    errors.append(exc)

            with patch("puppetmaster.orchestrator.WorkerRuntime.run_until_idle", run_until_idle):
                threads = [threading.Thread(target=run, args=(s,)) for s in stores]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
            self.assertEqual(errors, [])
            for store in stores:
                observed = seen[store.root]
                self.assertEqual(observed["bound"], str(store.root))
                self.assertEqual(observed["sidecar"], str(store.root))
                self.assertEqual(observed["delta"], str(store.root))
                self.assertIsNone(observed["env"])
            self.assertNotIn(STATE_DIR_ENV, os.environ)

    def test_agentic_parallel_tool_calls_see_the_binding(self) -> None:
        adapter = AgenticAdapter()
        seen: list = []

        def execute_tool(name, args, cwd, implement, task):
            seen.append(state_dir_from_env())
            return "ok"

        adapter._execute_tool = execute_tool
        task = Task(job_id="job_x", role="explore", instruction="look")
        calls = [{"name": "read_file", "arguments": {"path": str(i)}} for i in range(4)]
        with bind_state_dir("/bound/state"):
            results = adapter._execute_tool_segment_parallel(calls, Path("."), False, task)
        self.assertEqual([out for _, out in results], ["ok"] * 4)
        self.assertEqual(seen, ["/bound/state"] * 4)


if __name__ == "__main__":
    unittest.main()
