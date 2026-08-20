"""Regression: unittest discover must isolate host Puppetmaster env pins."""
from __future__ import annotations

import contextlib
import io
import os
import sys

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401  # process-wide host-env isolation

import unittest
from pathlib import Path

from puppetmaster.platform_lock import KNOWN_ADAPTERS, ONLY_ENV


class HermeticEnvIsolationTests(unittest.TestCase):
    def test_models_path_points_at_missing_sentinel(self) -> None:
        path = Path(os.environ["PUPPETMASTER_MODELS_PATH"])
        self.assertFalse(path.is_file())
        self.assertIn("pm-test-empty-", str(path))

    def test_platform_lock_env_enables_every_known_adapter(self) -> None:
        enabled = {part.strip() for part in os.environ[ONLY_ENV].split(",") if part.strip()}
        self.assertEqual(enabled, set(KNOWN_ADAPTERS))

    def test_codegraph_runtime_pins_are_cleared(self) -> None:
        self.assertNotIn("PUPPETMASTER_CODEGRAPH_NODE", os.environ)
        self.assertNotIn("PUPPETMASTER_CODEGRAPH_JS", os.environ)

    def test_autodiscover_disabled_for_suite(self) -> None:
        self.assertEqual(os.environ.get("PUPPETMASTER_AUTODISCOVER"), "0")

    def test_no_provider_credentials_reach_the_suite(self) -> None:
        """The developer's own API keys must not make a provider routable.

        With one set, ``available_providers()`` is non-empty, the auto-route
        path reconciles the curated agentic catalog into the registry, and
        routing/worker tests silently retarget onto a live API.
        """
        from puppetmaster.providers import available_providers

        leaked = [
            name
            for name in hermetic_env._provider_credential_env_names()
            if os.environ.get(name)
        ]
        self.assertEqual(leaked, [], f"provider credentials leaked into the suite: {leaked}")
        self.assertEqual(available_providers(), set())

    def test_clearing_removes_credentials_from_a_populated_env(self) -> None:
        """The assertion above is vacuous on a keyless machine.

        On CI there are no provider keys, so "none leaked" holds whether or not
        the clearing works. Drive the clearing over a fixture that definitely
        has keys, so this fails on CI too if the mechanism breaks.
        """
        fake = {
            "GOOGLE_API_KEY": "g",
            "OPENROUTER_API_KEY": "o",
            "ANTHROPIC_API_KEY": "a",
            "OPENAI_API_KEY_3": "numbered sibling",
            "CURSOR_API_KEY": "c",
            "PATH": "/keep/me",
        }
        removed = hermetic_env._clear_provider_credentials(fake)

        self.assertEqual(fake, {"PATH": "/keep/me"}, "non-credentials must survive")
        for expected in (
            "GOOGLE_API_KEY",
            "OPENROUTER_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY_3",
            "CURSOR_API_KEY",
        ):
            self.assertIn(expected, removed)

        from puppetmaster.providers import available_providers

        self.assertEqual(
            available_providers(env={"GOOGLE_API_KEY": "g"}),
            {"gemini"},
            "control: a key really does make a provider available",
        )

    def test_credential_names_are_derived_from_the_registry(self) -> None:
        """Derived, not hardcoded, so a provider added later is covered."""
        names = hermetic_env._provider_credential_env_names()
        self.assertIn("GOOGLE_API_KEY", names)  # gemini's second var
        self.assertIn("OPENROUTER_API_KEY", names)
        self.assertIn("ANTHROPIC_API_KEY", names)
        self.assertIn("AWS_BEARER_TOKEN_BEDROCK", names)
        # numbered rotation siblings, and keyless-provider presence vars
        self.assertIn("OPENAI_API_KEY_2", names)
        presence = {
            var
            for desc in __import__(
                "puppetmaster.providers", fromlist=["PROVIDER_REGISTRY"]
            ).PROVIDER_REGISTRY.values()
            for var in desc.presence_env_vars
        }
        self.assertTrue(presence.issubset(set(names)))
        self.assertEqual(len(names), len(set(names)), "duplicate env names")

    def test_reaper_deletes_a_registry_a_test_leaked(self) -> None:
        """The guard for the next code path that writes the pinned registry.

        Clearing credentials stops today's writer; this stops tomorrow's from
        silently poisoning every later test. Reaping happens *after* each test
        so the warning names the culprit.
        """
        from puppetmaster.model_registry import ModelSpec, save_registry

        sentinel = Path(os.environ["PUPPETMASTER_MODELS_PATH"])
        before = hermetic_env._LEAKED_REGISTRY_REAPS
        leak: dict[str, bool] = {}

        class _LeakyTest(unittest.TestCase):
            def runTest(self) -> None:  # noqa: N802 - unittest API
                # Exactly what orchestrator._apply_auto_routing does: persist to
                # default_registry_path(), which resolves to the sentinel.
                save_registry(
                    [
                        ModelSpec(
                            id="x/leaked",
                            adapter="agentic",
                            adapter_model_name="leaked",
                        )
                    ]
                )
                leak["created"] = sentinel.exists()

        stderr = io.StringIO()
        result = unittest.TestResult()
        # expect_registry_leak: this leak is deliberate, so it must not be
        # reported as a failure the way an undeclared one is.
        with contextlib.redirect_stderr(stderr), hermetic_env.expect_registry_leak():
            _LeakyTest().run(result)

        # Without these two the test passes vacuously when the leak never
        # happens (a swallowed error in the inner test would leave the sentinel
        # trivially absent).
        self.assertEqual(result.errors, [], f"the leaking test itself errored: {result.errors}")
        self.assertTrue(leak.get("created"), "the inner test never created the registry")

        self.assertFalse(
            sentinel.exists(), "the reaper left the leaked registry in place"
        )
        self.assertEqual(hermetic_env._LEAKED_REGISTRY_REAPS, before + 1)
        self.assertIn("left a model registry", stderr.getvalue())
        self.assertIn("_LeakyTest", stderr.getvalue(), "the warning must name the culprit")

    def test_reaper_is_quiet_and_reports_false_when_nothing_leaked(self) -> None:
        before = hermetic_env._LEAKED_REGISTRY_REAPS
        self.assertFalse(hermetic_env.reap_leaked_registry("noop"))
        self.assertEqual(hermetic_env._LEAKED_REGISTRY_REAPS, before)

    def test_a_real_leak_fails_the_test_that_caused_it(self) -> None:
        """Reaping must not *silence* the canary.

        Cleaning up after each test keeps later tests isolated, but it also
        means test_models_path_points_at_missing_sentinel can no longer fail --
        the leak is always gone before it looks. The write is still the bug, so
        it has to fail the run somewhere. It fails the guilty test, at the
        scene, instead of an innocent one hundreds of tests later.

        (atexit cannot do this job: CPython prints "Exception ignored in atexit
        callback" and still exits 0, so a leak would report as a green run.)
        """
        from puppetmaster.model_registry import ModelSpec, save_registry

        sentinel = Path(os.environ["PUPPETMASTER_MODELS_PATH"])
        leak = {}

        class _LeakyTest(unittest.TestCase):
            def runTest(self) -> None:  # noqa: N802 - unittest API
                save_registry([ModelSpec(id="x/leaked", adapter="agentic",
                                         adapter_model_name="leaked")])
                leak["created"] = sentinel.exists()

        # No expect_registry_leak() here: this is the undeclared case, which
        # must be reported as a failure.
        result = unittest.TestResult()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            case = _LeakyTest()
            case.run(result)

        self.assertTrue(leak.get("created"), "the inner test never created the registry")
        self.assertEqual(result.errors, [], f"inner test errored: {result.errors}")
        self.assertEqual(
            len(result.failures), 1, "an undeclared leak must fail its own test"
        )
        failed_case, message = result.failures[0]
        self.assertIs(failed_case, case, "the failure must be attributed to the culprit")
        self.assertIn("hermetic isolation was broken", message)
        self.assertIn("save_registry", message)
        self.assertFalse(result.wasSuccessful())

    def test_declared_leaks_do_not_fail_their_test(self) -> None:
        """The reaper's own test leaks on purpose; that must stay green."""
        from puppetmaster.model_registry import ModelSpec, save_registry

        class _DeliberateLeak(unittest.TestCase):
            def runTest(self) -> None:  # noqa: N802 - unittest API
                save_registry([ModelSpec(id="x/ok", adapter="agentic",
                                         adapter_model_name="ok")])

        result = unittest.TestResult()
        with contextlib.redirect_stderr(io.StringIO()), hermetic_env.expect_registry_leak():
            _DeliberateLeak().run(result)
        self.assertTrue(result.wasSuccessful(), f"{result.failures}")



if __name__ == "__main__":
    unittest.main()
