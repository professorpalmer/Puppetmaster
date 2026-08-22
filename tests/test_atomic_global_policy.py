"""Regression coverage for user-global policy-document mutations.

These tests deliberately coordinate concurrent callers.  They prove the
platform read-modify-write transaction keeps independent updates, and that
readers never observe a partially-written models registry.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from puppetmaster import platform_lock
from puppetmaster.model_registry import ModelSpec, save_registry


def test_platform_concurrent_mutations_keep_both_updates() -> None:
    """Arrange two writers; Act with one paused in its critical section; Assert union."""
    with TemporaryDirectory() as tmp:
        registry = Path(tmp) / "models.json"
        entered = threading.Event()
        release_first = threading.Event()
        failures: list[BaseException] = []
        original = platform_lock._write_disabled_locked
        calls = 0

        def paused_first_writer(disabled: set[str], path: Path) -> Path:
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                assert release_first.wait(timeout=2), "second writer was not scheduled"
            return original(disabled, path)

        def run(adapters: set[str]) -> None:
            try:
                platform_lock.disable(adapters, registry)
            except BaseException as exc:  # surface thread failures to pytest
                failures.append(exc)

        with patch.object(platform_lock, "_write_disabled_locked", paused_first_writer):
            first = threading.Thread(target=run, args=({"cursor"},))
            second = threading.Thread(target=run, args=({"codex"},))
            first.start()
            assert entered.wait(timeout=2), "first writer did not enter critical section"
            second.start()
            # Give the second caller a chance to contend while the first lock
            # remains held. The old unlocked read-modify-write loses one update.
            time.sleep(0.05)
            release_first.set()
            first.join(timeout=3)
            second.join(timeout=3)

        assert not first.is_alive()
        assert not second.is_alive()
        assert not failures
        data = json.loads(platform_lock.platform_config_path(registry).read_text("utf-8"))
        assert set(data["disabled"]) == {"cursor", "codex"}


def test_models_registry_readers_never_observe_truncated_json() -> None:
    """Arrange a live reader; Act with repeated replacements; Assert every parse succeeds."""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "models.json"
        save_registry([ModelSpec("one", "cursor", "one")], path)
        start = threading.Event()
        done = threading.Event()
        parse_failures: list[BaseException] = []

        def reader() -> None:
            assert start.wait(timeout=2)
            while not done.is_set():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    assert isinstance(data["models"], list)
                except BaseException as exc:
                    parse_failures.append(exc)
                    return

        thread = threading.Thread(target=reader)
        thread.start()
        start.set()
        for index in range(80):
            save_registry(
                [ModelSpec("one", "cursor", "one"), ModelSpec("two", "codex", str(index))],
                path,
            )
        done.set()
        thread.join(timeout=3)

        assert not thread.is_alive()
        assert not parse_failures
        data = json.loads(path.read_text(encoding="utf-8"))
        assert {model["id"] for model in data["models"]} == {"one", "two"}


def test_orphaned_lock_is_recovered_by_next_writer() -> None:
    """Arrange a dead-PID lock; Act by saving; Assert stale recovery publishes JSON."""
    from puppetmaster.interprocess_lock import InterProcessFileLock

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "models.json"
        lock_path = InterProcessFileLock.for_target(path).path
        lock_path.write_text(
            json.dumps({"pid": 999_999_999, "created_at": time.time(), "token": "dead"}),
            encoding="utf-8",
        )

        save_registry([ModelSpec("one", "cursor", "one")], path)

        assert json.loads(path.read_text(encoding="utf-8"))["models"][0]["id"] == "one"
        assert not lock_path.exists()
