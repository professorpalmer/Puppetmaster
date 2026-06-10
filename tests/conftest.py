"""Pytest-level fixtures shared across the Puppetmaster suite.

The router's auto-routing path reads ``~/.puppetmaster/models.json``
by default. After v0.6.0 the built-in ``DEFAULT_WORKERS`` opt into
auto-routing (so any swarm started from MCP or the CLI picks the
right tier per task without per-spec opt-in). That's exactly what we
want for users, but it means a developer who has run
``puppetmaster models init`` on their machine would see Orchestrator
tests start invoking the real ``claude-code`` / ``cursor`` adapters
during ``pytest`` — tests must be hermetic.

This conftest forces the registry path to a nonexistent temp file
and enables every platform for the duration of the suite. ``load_registry``
returns ``[]`` for a missing file, and ``_apply_auto_routing`` is a clean
no-op when the registry is empty (one ``router.registry_empty`` event, then
the original spec passes through unchanged). Net effect: tests run
identically on every machine regardless of the developer's home directory
state or personal platform lock.

Individual tests that exercise the router directly (e.g. the cost +
auto-route end-to-end test) still write their own registry into a
``TemporaryDirectory`` and pass ``payload.registry_path`` explicitly,
so this fixture does not interfere with them.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from puppetmaster.platform_lock import KNOWN_ADAPTERS, ONLY_ENV


_ENV_BEFORE: dict[str, str | None] = {}
_ISOLATION_TMP: str | None = None


def pytest_configure(config):
    """Force routing/platform tests away from the developer's real config."""
    global _ISOLATION_TMP
    _ISOLATION_TMP = tempfile.mkdtemp(prefix="pm-test-empty-")
    sentinel = Path(_ISOLATION_TMP) / "models-does-not-exist.json"
    _ENV_BEFORE["PUPPETMASTER_MODELS_PATH"] = os.environ.get("PUPPETMASTER_MODELS_PATH")
    _ENV_BEFORE[ONLY_ENV] = os.environ.get(ONLY_ENV)
    os.environ["PUPPETMASTER_MODELS_PATH"] = str(sentinel)
    os.environ[ONLY_ENV] = ",".join(KNOWN_ADAPTERS)


def pytest_unconfigure(config):
    for key, value in _ENV_BEFORE.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    if _ISOLATION_TMP is not None:
        shutil.rmtree(_ISOLATION_TMP, ignore_errors=True)
