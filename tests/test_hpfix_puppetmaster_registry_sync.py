"""`unroutable_pin_hint` names an entry that covers the starter id."""
from __future__ import annotations

import os
import sys
import unittest
from dataclasses import replace

_HERMETIC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HERMETIC_DIR not in sys.path:
    sys.path.insert(0, _HERMETIC_DIR)
import hermetic_env  # noqa: F401

from puppetmaster.model_registry import starter_registry
from puppetmaster.registry_sync import (
    SYNC_COMMAND,
    enable_command,
    plan_starter_sync,
    unroutable_pin_hint,
)

STARTER_ID = "claude-code/haiku-5-5"


def _starter():
    return next(spec for spec in starter_registry() if spec.id == STARTER_ID)


class UnroutablePinHintCoveredTests(unittest.TestCase):
    def test_disabled_covering_entry_is_named(self):
        own = replace(
            _starter(),
            id="local/my-haiku",
            enabled=False,
            disabled_reason="mine",
            disabled_authority="user",
        )
        # Sync adds nothing for this starter id: the own entry covers it.
        additions, covered = plan_starter_sync(
            [own], enabled_adapters=["claude-code"]
        )
        self.assertIn((STARTER_ID, "local/my-haiku"), covered)
        self.assertNotIn(STARTER_ID, [a.spec.id for a in additions])

        hint = unroutable_pin_hint("claude-haiku-5-5", [own], adapter="claude-code")
        self.assertNotIn(SYNC_COMMAND + " --apply", hint)
        self.assertIn("'local/my-haiku' is disabled: mine", hint)
        self.assertIn(enable_command("local/my-haiku"), hint)

    def test_retired_covering_entry_is_named(self):
        own = replace(
            _starter(),
            id="local/my-haiku",
            enabled=False,
            retired=True,
            retirement_reason="old",
            retirement_authority="user",
        )
        hint = unroutable_pin_hint("claude-haiku-5-5", [own], adapter="claude-code")
        self.assertEqual(hint, "'local/my-haiku' is retired: old")

    def test_absent_entry_still_points_to_sync(self):
        other = replace(
            _starter(), id="local/other", payload_defaults={"effort": "high"}
        )
        hint = unroutable_pin_hint(
            "claude-haiku-5-5", [other], adapter="claude-code"
        )
        self.assertIn(f"{STARTER_ID!r} is a starter registry entry", hint)
        self.assertIn(f"`{SYNC_COMMAND} --apply`", hint)

    def test_disabled_entry_under_starter_id_is_unchanged(self):
        own = replace(
            _starter(), enabled=False, disabled_reason="off", disabled_authority="user"
        )
        hint = unroutable_pin_hint("claude-haiku-5-5", [own], adapter="claude-code")
        self.assertEqual(
            hint,
            f"{STARTER_ID!r} is disabled: off. To enable it, run "
            f"`{enable_command(STARTER_ID)}`.",
        )


if __name__ == "__main__":
    unittest.main()
