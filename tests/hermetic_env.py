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
import contextlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional


_APPLIED = False
_ENV_BEFORE: dict[str, Optional[str]] = {}
_ISOLATION_TMP: Optional[str] = None
_ORIG_TESTCASE_RUN = None
_ATEXIT_REGISTERED = False
# How many tests left a registry behind at the sentinel path (see
# ``reap_leaked_registry``). Exposed for the harness's own tests.
_LEAKED_REGISTRY_REAPS = 0
# Test ids that leaked, kept for diagnostics (the failure itself is reported
# against the guilty test -- see _fail_leaking_test).
_LEAKED_REGISTRY_TESTS: list[str] = []
# >0 while a test leaks on purpose (the reaper's own test).
_EXPECT_REGISTRY_LEAK = 0

# Host pins that short-circuit discovery / routing when left in place.
_PIN_KEYS_TO_CLEAR = (
    "PUPPETMASTER_CODEGRAPH_NODE",
    "PUPPETMASTER_CODEGRAPH_JS",
    "PUPPETMASTER_STATE_DIR",
    "PUPPETMASTER_CURSOR_INPUT",
    # A provider the developer disconnected in Marionette Settings must not
    # change what the suite sees either — tests that care set it themselves.
    "PUPPETMASTER_DISABLED_PROVIDERS",
)


def _provider_credential_env_names() -> "tuple[str, ...]":
    """Every env var that can make a direct-API provider auto-routable.

    Derived from ``PROVIDER_REGISTRY`` rather than hardcoded, so a provider
    added later is covered without touching this file. Includes the numbered
    rotation siblings (``OPENAI_API_KEY_2`` ...) and the presence vars that
    opt keyless local endpoints (Ollama / LM Studio) in.
    """
    from puppetmaster.providers import PROVIDER_REGISTRY

    try:
        from puppetmaster.providers import _numbered_env_names
    except ImportError:  # private helper; keep isolation working if it moves
        def _numbered_env_names(name: str) -> "list[str]":
            return [name] + [f"{name}_{i}" for i in range(2, 10)]

    names: list[str] = []
    for desc in PROVIDER_REGISTRY.values():
        for var in desc.api_key_env_vars:
            names.extend(_numbered_env_names(var))
        names.extend(desc.presence_env_vars)
    # Not in PROVIDER_REGISTRY (it drives the Cursor *CLI/SDK* path, not a
    # direct-API provider) but it still turns on plan-catalog discovery.
    names.extend(_numbered_env_names("CURSOR_API_KEY"))
    # dict.fromkeys: de-dupe (gemini lists GOOGLE_API_KEY too) but keep order
    # stable so the restore path is deterministic.
    return tuple(dict.fromkeys(names))


def _clear_provider_credentials(env) -> "list[str]":
    """Remove every provider credential from ``env``; return what was removed.

    Split out from :func:`apply_hermetic_isolation` so the behaviour is
    testable on a fixture dict. Testing it against ``os.environ`` alone is
    vacuous on a keyless machine (CI) — there the assertion holds whether or
    not the clearing works.
    """
    removed = []
    for key in _provider_credential_env_names():
        if env.pop(key, None) is not None:
            removed.append(key)
    return removed


def _sentinel_paths() -> "tuple[Path, ...]":
    """The registry the harness promises is absent, plus its meta sidecar."""
    if _ISOLATION_TMP is None:
        return ()
    from puppetmaster.model_registry import discovery_meta_path

    sentinel = Path(_ISOLATION_TMP) / "models-does-not-exist.json"
    return (sentinel, discovery_meta_path(sentinel))


@contextlib.contextmanager
def expect_registry_leak():
    """Mark a deliberate leak so it doesn't fail the run at exit.

    Only the harness's own test for the reaper should need this.
    """
    global _EXPECT_REGISTRY_LEAK
    _EXPECT_REGISTRY_LEAK += 1
    try:
        yield
    finally:
        _EXPECT_REGISTRY_LEAK -= 1


_LEAK_MESSAGE = (
    "hermetic isolation was broken: this test wrote a model registry to "
    "PUPPETMASTER_MODELS_PATH, which the suite pins at a path that must stay "
    "absent. It was reaped so later tests stay isolated, but the write itself "
    "is the bug — something under test called save_registry() against "
    "default_registry_path(). With a provider credential in the environment "
    "that is exactly how routing used to retarget the whole suite onto a live "
    "API."
)


def _fail_leaking_test(case: unittest.TestCase, result) -> None:
    """Record a leak as a failure of the test that caused it.

    Reaping keeps later tests isolated, but on its own it would *silence* the
    canary that caught this class of bug — with the reaper in place,
    ``test_models_path_points_at_missing_sentinel`` can no longer fail, because
    the leak is always cleaned up before it looks. So the leak still fails the
    run; it just fails the guilty test instead of an innocent one downstream.

    (An ``atexit`` hook cannot do this job: CPython prints "Exception ignored
    in atexit callback" and still exits 0 — verified on 3.12 — so a leak would
    report as a green run.)
    """
    try:
        raise AssertionError(f"{_LEAK_MESSAGE}\n(test: {case.id()})")
    except AssertionError:
        info = sys.exc_info()
        if result is None:
            raise
        result.addFailure(case, info)


def reap_leaked_registry(test_id: str = "") -> bool:
    """Delete any registry a test wrote at the pinned sentinel path.

    ``hermetic_env`` points ``PUPPETMASTER_MODELS_PATH`` at a file that does
    not exist, but nothing stops code under test from *creating* it: routing
    persists a reconciled catalog through ``save_registry(...,
    default_registry_path())``. When that happens the registry stays populated
    for the rest of the process and every later "empty registry" assertion
    fails hundreds of tests downstream of the one that actually did it.

    So we reap after each test and name the culprit on stderr. Returns True if
    something was reaped.
    """
    global _LEAKED_REGISTRY_REAPS
    reaped = False
    for path in _sentinel_paths():
        try:
            if not path.exists():
                continue
            path.unlink()
        except OSError:
            continue
        reaped = True
    if reaped:
        _LEAKED_REGISTRY_REAPS += 1
        if not _EXPECT_REGISTRY_LEAK:
            _LEAKED_REGISTRY_TESTS.append(test_id or "<unknown test>")
        print(
            f"hermetic_env: {test_id or 'a test'} left a model registry at the "
            f"pinned sentinel path; reaped it so later tests stay isolated. "
            f"Something under test called save_registry() against "
            f"default_registry_path().",
            file=sys.stderr,
        )
    return reaped


def apply_hermetic_isolation(*, register_atexit: bool = True) -> None:
    """Force routing/platform/codegraph tests away from the host config."""
    global _APPLIED, _ISOLATION_TMP, _ORIG_TESTCASE_RUN, _ATEXIT_REGISTERED
    if _APPLIED:
        return
    _APPLIED = True

    from puppetmaster.platform_lock import KNOWN_ADAPTERS, ONLY_ENV

    _ISOLATION_TMP = tempfile.mkdtemp(prefix="pm-test-empty-")
    sentinel = Path(_ISOLATION_TMP) / "models-does-not-exist.json"
    health_db = Path(_ISOLATION_TMP) / "provider_health.sqlite3"
    rate_limit_db = Path(_ISOLATION_TMP) / "rate_limits.sqlite3"

    _ENV_BEFORE["PUPPETMASTER_MODELS_PATH"] = os.environ.get("PUPPETMASTER_MODELS_PATH")
    _ENV_BEFORE["PUPPETMASTER_PROVIDER_HEALTH_PATH"] = os.environ.get(
        "PUPPETMASTER_PROVIDER_HEALTH_PATH"
    )
    _ENV_BEFORE["PUPPETMASTER_RATE_LIMIT_PATH"] = os.environ.get(
        "PUPPETMASTER_RATE_LIMIT_PATH"
    )
    _ENV_BEFORE[ONLY_ENV] = os.environ.get(ONLY_ENV)
    for key in _PIN_KEYS_TO_CLEAR:
        _ENV_BEFORE[key] = os.environ.get(key)
    for key in _provider_credential_env_names():
        _ENV_BEFORE[key] = os.environ.get(key)

    os.environ["PUPPETMASTER_MODELS_PATH"] = str(sentinel)
    os.environ["PUPPETMASTER_PROVIDER_HEALTH_PATH"] = str(health_db)
    os.environ["PUPPETMASTER_RATE_LIMIT_PATH"] = str(rate_limit_db)
    os.environ[ONLY_ENV] = ",".join(KNOWN_ADAPTERS)
    for key in _PIN_KEYS_TO_CLEAR:
        os.environ.pop(key, None)

    # A developer's own provider API keys must not reach the suite. With, say,
    # GOOGLE_API_KEY set, ``available_providers()`` is non-empty, so the
    # auto-route path in the orchestrator reconciles the curated agentic
    # catalog, persists it to PUPPETMASTER_MODELS_PATH above (creating the
    # "missing" sentinel), and every later routing assertion sees `agentic`
    # instead of the pinned adapter — while worker tests quietly execute
    # against the live API. CI is green precisely because it has no keys;
    # clearing them here is what makes a local run mean the same thing.
    _clear_provider_credentials(os.environ)

    # Orchestrator plan-catalog auto-discovery shells out to the Cursor SDK
    # when CURSOR_API_KEY is set; tests that need it inject their own fetcher.
    # Force-assign rather than setdefault: a host PUPPETMASTER_AUTODISCOVER=1
    # would still run _ensure_plan_catalog, and file-based Claude/Codex
    # billing can persist a plan catalog onto the sentinel even with
    # CURSOR_API_KEY cleared.
    _ENV_BEFORE["PUPPETMASTER_AUTODISCOVER"] = os.environ.get("PUPPETMASTER_AUTODISCOVER")
    os.environ["PUPPETMASTER_AUTODISCOVER"] = "0"

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
            from puppetmaster.provider_health import reset_provider_health_store_cache

            reset_provider_health_store_cache()
        except Exception:
            pass
        try:
            from puppetmaster.rate_limit_state import reset_rate_limit_store_cache

            reset_rate_limit_store_cache()
        except Exception:
            pass
        try:
            from puppetmaster.codegraph import reset_cursor_codegraph_invocation_cache

            reset_cursor_codegraph_invocation_cache()
        except Exception:
            pass
        outcome = _ORIG_TESTCASE_RUN(self, result)
        # Reap AFTER the test, not before: this attributes the leak to the test
        # that actually wrote the registry, instead of the isolation silently
        # absorbing it and the damage surfacing hundreds of tests later as an
        # unrelated routing assertion. Do not swallow addFailure errors — a
        # silent except would turn a leak into a green run.
        if reap_leaked_registry(self.id()) and not _EXPECT_REGISTRY_LEAK:
            _fail_leaking_test(self, result if result is not None else outcome)
        return outcome

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
