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
for the duration of the suite. ``load_registry`` returns ``[]`` for a
missing file, and ``_apply_auto_routing`` is a clean no-op when the
registry is empty (one ``router.registry_empty`` event, then the
original spec passes through unchanged). Net effect: tests run
identically on every machine regardless of the developer's home
directory state.

Individual tests that exercise the router directly (e.g. the cost +
auto-route end-to-end test) still write their own registry into a
``TemporaryDirectory`` and pass ``payload.registry_path`` explicitly,
so this fixture does not interfere with them.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolate_puppetmaster_models_registry():
    """Force the router to see an empty registry during pytest runs."""
    tmp = tempfile.mkdtemp(prefix="pm-test-empty-")
    sentinel = Path(tmp) / "models-does-not-exist.json"
    previous = os.environ.get("PUPPETMASTER_MODELS_PATH")
    os.environ["PUPPETMASTER_MODELS_PATH"] = str(sentinel)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PUPPETMASTER_MODELS_PATH", None)
        else:
            os.environ["PUPPETMASTER_MODELS_PATH"] = previous
