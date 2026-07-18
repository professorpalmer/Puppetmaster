"""Process-wide hermetic isolation for the unittest and pytest suites.

Developer hosts often carry ``PUPPETMASTER_*`` pins from
``repair-codegraph``, ``platform only``, a real
``~/.puppetmaster/models.json``, and in-flight Cursor worker env
(``PUPPETMASTER_CURSOR_INPUT`` / ``PUPPETMASTER_STATE_DIR``). Those must
not leak into tests.

Pytest applies this via ``conftest.py``. Unittest discover does not load
conftest, so every ``test_*.py`` module imports this file for its side
effect before exercising Puppetmaster code.
"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Optional


_APPLIED = False
_ENV_BEFORE: dict[str, Optional[str]] = {}
_ISOLATION_TMP: Optional[str] = None
_ORIG_TESTCASE_RUN = None
_ATEXIT_REGISTERED = False

# Host pins that short-circuit discovery / routing when left in place.
_PIN_KEYS_TO_CLEAR = (
    "PUPPETMASTER_CODEGRAPH_NODE",
    "PUPPETMASTER_CODEGRAPH_JS",
    "PUPPETMASTER_STATE_DIR",
    "PUPPETMASTER_CURSOR_INPUT",
)


def apply_hermetic_isolation(*, register_atexit: bool = True) -> None:
    """Force routing/platform/codegraph tests away from the host config."""
    global _APPLIED, _ISOLATION_TMP, _ORIG_TESTCASE_RUN, _ATEXIT_REGISTERED
    if _APPLIED:
        return
    _APPLIED = True

    from puppetmaster.platform_lock import KNOWN_ADAPTERS, ONLY_ENV

    _ISOLATION_TMP = tempfile.mkdtemp(prefix="pm-test-empty-")
    sentinel = Path(_ISOLATION_TMP) / "models-does-not-exist.json"

    _ENV_BEFORE["PUPPETMASTER_MODELS_PATH"] = os.environ.get("PUPPETMASTER_MODELS_PATH")
    _ENV_BEFORE[ONLY_ENV] = os.environ.get(ONLY_ENV)
    for key in _PIN_KEYS_TO_CLEAR:
        _ENV_BEFORE[key] = os.environ.get(key)

    os.environ["PUPPETMASTER_MODELS_PATH"] = str(sentinel)
    os.environ[ONLY_ENV] = ",".join(KNOWN_ADAPTERS)
    for key in _PIN_KEYS_TO_CLEAR:
        os.environ.pop(key, None)

    # Orchestrator plan-catalog auto-discovery shells out to the Cursor SDK
    # when CURSOR_API_KEY is set; tests that need it inject their own fetcher.
    os.environ.setdefault("PUPPETMASTER_AUTODISCOVER", "0")

    # Keep process-local provider circuit state from leaking across tests.
    # Pytest also resets via ``pytest_runtest_setup``; double-reset is a no-op.
    _ORIG_TESTCASE_RUN = unittest.TestCase.run

    def _hermetic_run(self, result=None):
        try:
            from puppetmaster.provider_circuit import reset_provider_circuit_breaker

            reset_provider_circuit_breaker()
        except Exception:
            pass
        try:
            from puppetmaster.platform_billing import clear_billing_cache

            clear_billing_cache()
        except Exception:
            pass
        try:
            from puppetmaster.codegraph import reset_cursor_codegraph_invocation_cache

            reset_cursor_codegraph_invocation_cache()
        except Exception:
            pass
        return _ORIG_TESTCASE_RUN(self, result)

    unittest.TestCase.run = _hermetic_run  # type: ignore[method-assign]

    if register_atexit and not _ATEXIT_REGISTERED:
        atexit.register(restore_hermetic_isolation)
        _ATEXIT_REGISTERED = True


def restore_hermetic_isolation() -> None:
    """Undo :func:`apply_hermetic_isolation` (pytest unconfigure / atexit)."""
    global _APPLIED, _ISOLATION_TMP, _ORIG_TESTCASE_RUN
    if not _APPLIED:
        return

    if _ORIG_TESTCASE_RUN is not None:
        unittest.TestCase.run = _ORIG_TESTCASE_RUN  # type: ignore[method-assign]
        _ORIG_TESTCASE_RUN = None

    for key, value in _ENV_BEFORE.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    _ENV_BEFORE.clear()

    if _ISOLATION_TMP is not None:
        shutil.rmtree(_ISOLATION_TMP, ignore_errors=True)
        _ISOLATION_TMP = None

    _APPLIED = False


# Import-time side effect for ``python -m unittest discover -s tests``.
apply_hermetic_isolation(register_atexit=True)
