"""Pytest-level fixtures shared across the Puppetmaster suite.

The router's auto-routing path reads ``~/.puppetmaster/models.json``
by default. After v0.6.0 the built-in ``DEFAULT_WORKERS`` opt into
auto-routing (so any swarm started from MCP or the CLI picks the
right tier per task without per-spec opt-in). That's exactly what we
want for users, but it means a developer who has run
``puppetmaster models init`` on their machine would see Orchestrator
tests start invoking the real ``claude-code`` / ``cursor`` adapters
during ``pytest`` — tests must be hermetic.

Isolation lives in :mod:`hermetic_env` so the same pins apply under
``python -m unittest discover`` (which does not load this conftest).
Individual tests that exercise the router directly (e.g. the cost +
auto-route end-to-end test) still write their own registry into a
``TemporaryDirectory`` and pass ``payload.registry_path`` explicitly,
so this fixture does not interfere with them.
"""
from __future__ import annotations

import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)

import hermetic_env


def pytest_configure(config):
    """Force routing/platform tests away from the developer's real config."""
    # register_atexit=False: pytest_unconfigure restores explicitly.
    hermetic_env.apply_hermetic_isolation(register_atexit=False)


def pytest_unconfigure(config):
    hermetic_env.restore_hermetic_isolation()


def pytest_runtest_setup(item):
    """Keep process-local caches/breakers from leaking across tests."""
    from puppetmaster.codegraph import reset_cursor_codegraph_invocation_cache
    from puppetmaster.platform_billing import clear_billing_cache
    from puppetmaster.provider_circuit import reset_provider_circuit_breaker

    reset_provider_circuit_breaker()
    clear_billing_cache()
    reset_cursor_codegraph_invocation_cache()
