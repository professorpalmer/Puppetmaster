"""Regression: concurrent MCP/swarm launches must not share run_id / log paths."""

from __future__ import annotations

import concurrent.futures
import itertools
import json
import multiprocessing as mp
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from puppetmaster.mcp_server import start_cli, _spawn_codegraph_indexer
from puppetmaster.run_id import new_run_id, reserve_run_logs, write_exclusive_run_text
from puppetmaster.swarm_launch import detach_analysis_swarm, write_analysis_swarm_config
from tests.run_id_mp_worker import FROZEN_MS, FROZEN_PID, spawn_new_run_id_batch


def _assert_prefixed_run_id(test: unittest.TestCase, run_id: str, prefix: str) -> None:
    legacy = f"{prefix}_{FROZEN_MS}_{FROZEN_PID}"
    test.assertNotEqual(run_id, legacy)
    test.assertTrue(run_id.startswith(f"{legacy}_"))
    # Trailing fields are seq (hex) + entropy (8 hex chars).
    suffix = run_id[len(legacy) + 1 :]
    seq, entropy = suffix.rsplit("_", 1)
    test.assertTrue(seq)
    test.assertRegex(seq, r"^[0-9a-f]+$")
    test.assertEqual(len(entropy), 8)
    test.assertRegex(entropy, r"^[0-9a-f]{8}$")


class NewRunIdConcurrencyTests(unittest.TestCase):
    def test_new_run_id_unique_when_time_and_pid_frozen(self) -> None:
        with patch("puppetmaster.run_id.time.time", return_value=FROZEN_MS / 1000.0), patch(
            "puppetmaster.run_id.os.getpid", return_value=FROZEN_PID
        ):
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                ids = list(pool.map(lambda _: new_run_id("mcp"), range(64)))

        self.assertEqual(len(ids), len(set(ids)))
        for run_id in ids:
            _assert_prefixed_run_id(self, run_id, "mcp")

    def test_new_run_id_prefixes_for_all_call_sites(self) -> None:
        with patch("puppetmaster.run_id.time.time", return_value=FROZEN_MS / 1000.0), patch(
            "puppetmaster.run_id.os.getpid", return_value=FROZEN_PID
        ):
            for prefix in ("mcp", "codegraph_index", "swarm", "swarm_config"):
                run_id = new_run_id(prefix)
                _assert_prefixed_run_id(self, run_id, prefix)

    def test_process_entropy_keeps_ids_unique_when_seq_resets(self) -> None:
        # Emulate two processes: each restarts the process-local counter at 0
        # while sharing the same frozen ms+pid. Entropy must keep the union unique.
        with patch("puppetmaster.run_id.time.time", return_value=FROZEN_MS / 1000.0), patch(
            "puppetmaster.run_id.os.getpid", return_value=FROZEN_PID
        ):
            with patch("puppetmaster.run_id._RUN_ID_SEQ", itertools.count(0)):
                first = [new_run_id("mcp") for _ in range(16)]
            with patch("puppetmaster.run_id._RUN_ID_SEQ", itertools.count(0)):
                second = [new_run_id("mcp") for _ in range(16)]

        seqs_first = [rid.rsplit("_", 2)[1] for rid in first]
        seqs_second = [rid.rsplit("_", 2)[1] for rid in second]
        self.assertEqual(seqs_first, seqs_second)
        combined = first + second
        self.assertEqual(len(combined), len(set(combined)))
        for run_id in combined:
            _assert_prefixed_run_id(self, run_id, "mcp")

    def test_new_run_id_multiprocess_unique_with_forced_same_ms_pid(self) -> None:
        # Lightweight real multiprocess check. Worker lives in a tiny helper
        # module so spawn children do not re-import mcp_server.
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=ctx) as pool:
            batches = list(pool.map(spawn_new_run_id_batch, [32, 32, 32, 32]))
        all_ids = [run_id for batch in batches for run_id in batch]
        self.assertEqual(len(all_ids), 128)
        self.assertEqual(len(all_ids), len(set(all_ids)))
        for run_id in all_ids:
            _assert_prefixed_run_id(self, run_id, "mcp")

    def test_reserve_run_logs_retries_on_exclusive_collision(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            first_id = "mcp_1_1_0_aaaaaaaa"
            stdout_path = run_dir / f"{first_id}.stdout.log"
            stderr_path = run_dir / f"{first_id}.stderr.log"
            stdout_path.write_text("occupied", encoding="utf-8")
            stderr_path.write_text("occupied", encoding="utf-8")

            ids = iter([first_id, "mcp_1_1_1_bbbbbbbb"])
            with patch("puppetmaster.run_id.new_run_id", side_effect=lambda _prefix: next(ids)):
                run_id, out_path, err_path, out_h, err_h = reserve_run_logs(run_dir, "mcp")
                try:
                    self.assertEqual(run_id, "mcp_1_1_1_bbbbbbbb")
                    self.assertEqual(out_path, run_dir / f"{run_id}.stdout.log")
                    self.assertEqual(err_path, run_dir / f"{run_id}.stderr.log")
                    self.assertTrue(out_path.is_file())
                    self.assertTrue(err_path.is_file())
                    # Original occupant must not be truncated.
                    self.assertEqual(stdout_path.read_text(encoding="utf-8"), "occupied")
                finally:
                    out_h.close()
                    err_h.close()

    def test_reserve_run_logs_fails_closed_after_max_attempts(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            occupied = "mcp_1_1_0_cccccccc"
            (run_dir / f"{occupied}.stdout.log").write_text("occupied", encoding="utf-8")
            with patch("puppetmaster.run_id.new_run_id", return_value=occupied):
                with self.assertRaises(FileExistsError) as ctx:
                    reserve_run_logs(run_dir, "mcp", max_attempts=2)
            self.assertIn("could not reserve exclusive", str(ctx.exception))
            self.assertEqual(
                (run_dir / f"{occupied}.stdout.log").read_text(encoding="utf-8"),
                "occupied",
            )

    def test_write_exclusive_run_text_retries_on_collision(self) -> None:
        with TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            first_id = "swarm_config_1_1_0_dddddddd"
            occupied = config_dir / f"{first_id}.json"
            occupied.write_text('{"occupied": true}', encoding="utf-8")
            ids = iter([first_id, "swarm_config_1_1_1_eeeeeeee"])
            with patch("puppetmaster.run_id.new_run_id", side_effect=lambda _prefix: next(ids)):
                run_id, path = write_exclusive_run_text(
                    config_dir,
                    "swarm_config",
                    '{"ok": true}',
                    suffix=".json",
                )
            self.assertEqual(run_id, "swarm_config_1_1_1_eeeeeeee")
            self.assertEqual(path, config_dir / f"{run_id}.json")
            self.assertEqual(path.read_text(encoding="utf-8"), '{"ok": true}')
            # Original occupant must not be overwritten.
            self.assertEqual(occupied.read_text(encoding="utf-8"), '{"occupied": true}')

    def test_write_exclusive_run_text_fails_closed_and_cleans_partial(self) -> None:
        with TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            occupied_id = "swarm_config_1_1_0_ffffffff"
            (config_dir / f"{occupied_id}.json").write_text("occupied", encoding="utf-8")
            with patch("puppetmaster.run_id.new_run_id", return_value=occupied_id):
                with self.assertRaises(FileExistsError) as ctx:
                    write_exclusive_run_text(
                        config_dir,
                        "swarm_config",
                        '{"never": true}',
                        suffix=".json",
                        max_attempts=3,
                    )
            self.assertIn("could not reserve exclusive", str(ctx.exception))
            self.assertEqual(
                (config_dir / f"{occupied_id}.json").read_text(encoding="utf-8"),
                "occupied",
            )

            # Write failure after exclusive create must unlink the partial file.
            partial_id = "swarm_config_1_1_1_11111111"
            real_open = Path.open

            def open_then_fail_write(self, mode="r", *args, **kwargs):
                handle = real_open(self, mode, *args, **kwargs)
                if "x" in mode:

                    def boom(_data: str) -> int:
                        raise OSError("disk full")

                    handle.write = boom  # type: ignore[method-assign]
                return handle

            with patch(
                "puppetmaster.run_id.new_run_id", return_value=partial_id
            ), patch.object(Path, "open", open_then_fail_write):
                with self.assertRaises(OSError):
                    write_exclusive_run_text(
                        config_dir,
                        "swarm_config",
                        '{"partial": true}',
                        suffix=".json",
                    )
            self.assertFalse((config_dir / f"{partial_id}.json").exists())

    def test_write_analysis_swarm_config_retries_on_collision(self) -> None:
        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            config_dir = state_dir / "mcp-configs"
            config_dir.mkdir(parents=True)
            first_id = "swarm_config_1_1_0_22222222"
            occupied = config_dir / f"{first_id}.json"
            occupied.write_text('{"occupied": true}', encoding="utf-8")
            ids = iter([first_id, "swarm_config_1_1_1_33333333"])
            with patch("puppetmaster.run_id.new_run_id", side_effect=lambda _prefix: next(ids)):
                path = write_analysis_swarm_config(
                    goal="audit collision",
                    roles=["explore"],
                    adapter="cursor",
                    state_dir=state_dir,
                    cwd=tmp,
                )
            self.assertEqual(path.name, "swarm_config_1_1_1_33333333.json")
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["workers"][0]["role"], "explore")
            self.assertEqual(occupied.read_text(encoding="utf-8"), '{"occupied": true}')


class StartCliLogIsolationTests(unittest.TestCase):
    def test_start_cli_concurrent_launches_do_not_share_log_paths(self) -> None:
        def fake_popen(*_args, **_kwargs):
            process = MagicMock()
            process.pid = 4242
            process.poll.return_value = None
            process.wait.return_value = 0
            return process

        job_counter = iter(range(1, 100))

        def fake_wait_for_job_id(stdout_path, stderr_path, process, timeout_seconds=None):
            digest = Path(stdout_path).name
            return f"job_{digest.split('.')[0][-12:]}_{next(job_counter)}"

        with TemporaryDirectory() as tmp:
            args = {"cwd": tmp, "state_dir": ".pm-test"}
            with patch(
                "puppetmaster.run_id.time.time", return_value=FROZEN_MS / 1000.0
            ), patch(
                "puppetmaster.run_id.os.getpid", return_value=FROZEN_PID
            ), patch(
                "puppetmaster.mcp_server.subprocess.Popen", side_effect=fake_popen
            ), patch(
                "puppetmaster.mcp_server.wait_for_job_id", side_effect=fake_wait_for_job_id
            ), patch(
                "puppetmaster.mcp_server._track_async_process"
            ):
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    futures = [
                        pool.submit(start_cli, ["cursor", "review", "goal"], args)
                        for _ in range(16)
                    ]
                    results = [future.result(timeout=5) for future in futures]

            bodies = [json.loads(result["content"][0]["text"]) for result in results]
            run_ids = [body["run_id"] for body in bodies]
            job_ids = [body["job_id"] for body in bodies]
            stdout_paths = [body["stdout_path"] for body in bodies]
            stderr_paths = [body["stderr_path"] for body in bodies]

            self.assertEqual(len(run_ids), len(set(run_ids)))
            self.assertEqual(len(job_ids), len(set(job_ids)))
            self.assertEqual(len(stdout_paths), len(set(stdout_paths)))
            self.assertEqual(len(stderr_paths), len(set(stderr_paths)))

            run_dir = Path(tmp) / ".pm-test" / "mcp-runs"
            for path in stdout_paths + stderr_paths:
                self.assertTrue(Path(path).is_file())
                self.assertEqual(Path(path).parent, run_dir)

            for run_id in run_ids:
                _assert_prefixed_run_id(self, run_id, "mcp")


class CodegraphIndexLogIsolationTests(unittest.TestCase):
    def test_spawn_codegraph_indexer_concurrent_log_paths(self) -> None:
        def fake_popen(*_args, **_kwargs):
            process = MagicMock()
            process.pid = 5151
            process.poll.return_value = None
            return process

        lock = MagicMock()
        with TemporaryDirectory() as tmp:
            args = {"cwd": tmp, "state_dir": ".pm-test"}
            with patch(
                "puppetmaster.run_id.time.time", return_value=FROZEN_MS / 1000.0
            ), patch(
                "puppetmaster.run_id.os.getpid", return_value=FROZEN_PID
            ), patch(
                "puppetmaster.mcp_server.codegraph_lock_path",
                return_value=Path(tmp) / ".codegraph.lock",
            ), patch(
                "puppetmaster.mcp_server.acquire_codegraph_lock", return_value=lock
            ), patch(
                "puppetmaster.mcp_server.subprocess.Popen", side_effect=fake_popen
            ), patch(
                "puppetmaster.mcp_server._track_async_process"
            ):
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    futures = [
                        pool.submit(_spawn_codegraph_indexer, args, tmp) for _ in range(16)
                    ]
                    payloads = [future.result(timeout=5) for future in futures]

            self.assertTrue(all(payload.get("ok") for payload in payloads))
            run_ids = [payload["run_id"] for payload in payloads]
            stdout_paths = [payload["stdout_path"] for payload in payloads]
            stderr_paths = [payload["stderr_path"] for payload in payloads]
            self.assertEqual(len(run_ids), len(set(run_ids)))
            self.assertEqual(len(stdout_paths), len(set(stdout_paths)))
            self.assertEqual(len(stderr_paths), len(set(stderr_paths)))
            for run_id in run_ids:
                _assert_prefixed_run_id(self, run_id, "codegraph_index")
            self.assertEqual(lock.release.call_count, 16)


class SwarmLaunchLogIsolationTests(unittest.TestCase):
    def test_detach_analysis_swarm_concurrent_log_paths(self) -> None:
        def fake_popen(*_args, **_kwargs):
            process = MagicMock()
            process.pid = 6161
            process.poll.return_value = None
            process.wait.return_value = 0
            return process

        job_counter = iter(range(1, 100))

        def fake_wait_for_job_id(stdout_path, stderr_path, process, timeout_seconds=None):
            digest = Path(stdout_path).name
            return f"job_{digest.split('.')[0][-12:]}_{next(job_counter)}"

        with TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".pm-test"
            with patch(
                "puppetmaster.run_id.time.time", return_value=FROZEN_MS / 1000.0
            ), patch(
                "puppetmaster.run_id.os.getpid", return_value=FROZEN_PID
            ), patch(
                "puppetmaster.swarm_launch.subprocess.Popen", side_effect=fake_popen
            ), patch(
                "puppetmaster.swarm_launch.wait_for_job_id", side_effect=fake_wait_for_job_id
            ):
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    futures = [
                        pool.submit(
                            detach_analysis_swarm,
                            goal="audit concurrent",
                            roles=["explore"],
                            adapter="cursor",
                            state_dir=state_dir,
                            cwd=tmp,
                        )
                        for _ in range(16)
                    ]
                    payloads = [future.result(timeout=5) for future in futures]

            run_ids = [payload["run_id"] for payload in payloads]
            job_ids = [payload["job_id"] for payload in payloads]
            stdout_paths = [payload["stdout_path"] for payload in payloads]
            stderr_paths = [payload["stderr_path"] for payload in payloads]
            configs = [payload["config"] for payload in payloads]

            self.assertEqual(len(run_ids), len(set(run_ids)))
            self.assertEqual(len(job_ids), len(set(job_ids)))
            self.assertEqual(len(stdout_paths), len(set(stdout_paths)))
            self.assertEqual(len(stderr_paths), len(set(stderr_paths)))
            self.assertEqual(len(configs), len(set(configs)))

            run_dir = state_dir / "mcp-runs"
            for path in stdout_paths + stderr_paths:
                self.assertTrue(Path(path).is_file())
                self.assertEqual(Path(path).parent, run_dir)
            for run_id in run_ids:
                _assert_prefixed_run_id(self, run_id, "swarm")
            for config in configs:
                name = Path(config).stem
                _assert_prefixed_run_id(self, name, "swarm_config")


if __name__ == "__main__":
    unittest.main()
