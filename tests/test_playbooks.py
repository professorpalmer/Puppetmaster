"""Hermetic tests for universal playbook recipes (not a Cursor plugin)."""
from __future__ import annotations

import unittest

from puppetmaster.playbooks import (
    PLAYBOOK_IDS,
    match_playbook,
    merge_gates,
    recipe_for,
    stamp_payload,
)
from puppetmaster.swarm_launch import DEFAULT_SWARM_ROLES, build_analysis_swarm_specs


class MatchPlaybookTests(unittest.TestCase):
    def test_prompt_table(self) -> None:
        cases = (
            ("are we sure how the lease renews across the repo", "investigation"),
            ("how does the orchestrator pick an adapter", "investigation"),
            ("repro first then fix the retry bug", "bug-fix"),
            ("this is a bug in the lease renewer", "bug-fix"),
            ("implement new behavior behind a flag", "feature"),
            ("interrogate this diff", "interrogate"),
            ("hillclimb until the metric improves", "hillclimb"),
            ("add a CSV export endpoint", None),
            ("fix a typo in the README", None),
            ("map every module across the repo", None),
        )
        for prompt, expected in cases:
            self.assertEqual(
                match_playbook(prompt, env={}),
                expected,
                prompt,
            )

    def test_explicit_pin_wins(self) -> None:
        self.assertEqual(
            match_playbook(
                "are we sure how this works",
                explicit="bug-fix",
                env={},
            ),
            "bug-fix",
        )
        self.assertEqual(
            match_playbook("unrelated prompt. playbook: interrogate", env={}),
            "interrogate",
        )

    def test_unknown_pin_raises(self) -> None:
        with self.assertRaises(ValueError):
            match_playbook("x", explicit="nope")

    def test_disable_auto_match_keeps_explicit(self) -> None:
        env = {"PUPPETMASTER_PLAYBOOKS": "0"}
        self.assertIsNone(
            match_playbook("are we sure how the lease renews", env=env)
        )
        self.assertEqual(
            match_playbook(
                "are we sure how the lease renews",
                explicit="investigation",
                env=env,
            ),
            "investigation",
        )
        self.assertEqual(
            match_playbook("playbook: feature implement new behavior", env=env),
            "feature",
        )

    def test_delegating_false_drops_auto_investigation(self) -> None:
        self.assertIsNone(
            match_playbook(
                "are we sure how the lease renews",
                env={},
                delegating=False,
            )
        )
        self.assertEqual(
            match_playbook("interrogate this diff", env={}, delegating=False),
            "interrogate",
        )


class StampPayloadTests(unittest.TestCase):
    def test_feature_require_diff_merges_once(self) -> None:
        first = stamp_payload({}, "feature")
        kinds = [g["kind"] for g in first.get("gates") or []]
        self.assertEqual(kinds.count("require_diff"), 1)
        second = stamp_payload(first, "feature")
        kinds = [g["kind"] for g in second.get("gates") or []]
        self.assertEqual(kinds.count("require_diff"), 1)

    def test_merge_gates_skips_duplicate_kind(self) -> None:
        merged = merge_gates(
            [{"kind": "require_diff"}],
            [{"kind": "require_diff"}, {"kind": "ratchet", "command": "x"}],
        )
        self.assertEqual([g["kind"] for g in merged], ["require_diff", "ratchet"])

    def test_hillclimb_ratchet_needs_both(self) -> None:
        bare = stamp_payload({}, "hillclimb")
        self.assertNotIn("gates", bare)
        half = stamp_payload({}, "hillclimb", {"ratchet_command": "pytest -q"})
        self.assertNotIn("gates", half)
        full = stamp_payload(
            {},
            "hillclimb",
            {"ratchet_command": "pytest -q", "metric": "pass_rate"},
        )
        gates = full.get("gates") or []
        self.assertEqual(gates[0]["kind"], "ratchet")
        self.assertEqual(gates[0]["command"], "pytest -q")
        self.assertEqual(gates[0]["metric"], "pass_rate")

    def test_investigation_does_not_set_require_diff(self) -> None:
        stamped = stamp_payload({"read_only": True}, "investigation")
        self.assertEqual(stamped["playbook"], "investigation")
        self.assertFalse(
            any(g.get("kind") == "require_diff" for g in stamped.get("gates") or [])
        )
        self.assertNotIn("read_only", recipe_for("investigation").payload)

    def test_known_ids(self) -> None:
        self.assertEqual(PLAYBOOK_IDS, tuple(recipe_for(i).playbook_id for i in PLAYBOOK_IDS))


class SwarmPlaybookLaunchTests(unittest.TestCase):
    def test_investigation_fills_explore_review_when_roles_omitted(self) -> None:
        specs = build_analysis_swarm_specs(
            "are we sure how leases renew",
            [],
            adapter="cursor",
            cwd="/tmp/x",
            playbook="investigation",
        )
        self.assertEqual([s.role for s in specs], ["explore", "review"])
        for spec in specs:
            self.assertEqual(spec.payload.get("playbook"), "investigation")
            self.assertTrue(spec.payload.get("read_only"))
            self.assertFalse(
                any(
                    g.get("kind") == "require_diff"
                    for g in spec.payload.get("gates") or []
                )
            )

    def test_omitted_playbook_keeps_default_analysis_role(self) -> None:
        specs = build_analysis_swarm_specs(
            "peel the MCP surface",
            [],
            adapter="cursor",
            cwd="/tmp/x",
        )
        self.assertEqual([s.role for s in specs], list(DEFAULT_SWARM_ROLES))
        self.assertNotIn("playbook", specs[0].payload)

    def test_interrogate_prefers_quality_policy(self) -> None:
        specs = build_analysis_swarm_specs(
            "interrogate this diff",
            [],
            adapter="cursor",
            cwd="/tmp/x",
            playbook="interrogate",
        )
        self.assertEqual([s.role for s in specs], ["review", "audit"])
        for spec in specs:
            self.assertEqual(spec.payload.get("routing_policy"), "quality")
            self.assertEqual(spec.payload.get("playbook"), "interrogate")


class InvocationGatePlaybookTests(unittest.TestCase):
    def test_investigation_hard_scope(self) -> None:
        from puppetmaster.invocation_gate import should_delegate

        d = should_delegate(
            "are we sure how the lease renews across the repo",
            env={},
        )
        self.assertTrue(d.should_delegate)
        self.assertEqual(d.playbook, "investigation")
        self.assertEqual(d.suggested_verb, "puppetmaster_start_swarm")

    def test_bug_fix_broad_scope_uses_implement(self) -> None:
        from puppetmaster.invocation_gate import should_delegate

        d = should_delegate(
            "repro first then fix the retry bug across the codebase",
            env={},
        )
        self.assertTrue(d.should_delegate)
        self.assertEqual(d.playbook, "bug-fix")
        self.assertEqual(d.suggested_verb, "puppetmaster_start_implement")

    def test_interrogate_forces_delegate(self) -> None:
        from puppetmaster.invocation_gate import should_delegate

        d = should_delegate("interrogate this diff", env={})
        self.assertTrue(d.should_delegate)
        self.assertEqual(d.playbook, "interrogate")
        self.assertEqual(d.suggested_verb, "puppetmaster_start_swarm")
        self.assertIn("playbook", d.matched_signals)

    def test_typo_stays_inline_without_playbook(self) -> None:
        from puppetmaster.invocation_gate import should_delegate

        d = should_delegate("fix a typo", env={})
        self.assertFalse(d.should_delegate)
        self.assertIsNone(d.playbook)

    def test_unmatched_explore_uses_generic_swarm(self) -> None:
        from puppetmaster.invocation_gate import should_delegate

        d = should_delegate("map every module across the repo", env={})
        self.assertTrue(d.should_delegate)
        self.assertIsNone(d.playbook)
        self.assertEqual(d.suggested_verb, "puppetmaster_start_swarm")
        self.assertNotIn("cursor", d.suggested_verb)

    def test_codegraph_beats_investigation(self) -> None:
        from puppetmaster.invocation_gate import should_delegate

        d = should_delegate(
            "where is ClientError defined — are we sure how it is shaped",
            env={},
        )
        self.assertTrue(d.should_delegate)
        self.assertEqual(d.suggested_verb, "puppetmaster_codegraph_search")
        self.assertEqual(d.playbook, "investigation")

    def test_edit_beats_bug_fix_without_hard_scope(self) -> None:
        from puppetmaster.invocation_gate import should_delegate

        d = should_delegate("repro first then fix the retry helper", env={})
        self.assertTrue(d.should_delegate)
        self.assertEqual(d.playbook, "bug-fix")
        self.assertEqual(d.suggested_verb, "puppetmaster_edit")

    def test_disable_env_skips_auto_match(self) -> None:
        from puppetmaster.invocation_gate import should_delegate

        d = should_delegate(
            "are we sure how the lease renews across the repo",
            env={"PUPPETMASTER_PLAYBOOKS": "0"},
        )
        self.assertTrue(d.should_delegate)
        self.assertIsNone(d.playbook)
        self.assertEqual(d.suggested_verb, "puppetmaster_start_swarm")

    def test_in_prompt_pin_survives_disable(self) -> None:
        from puppetmaster.invocation_gate import should_delegate

        d = should_delegate(
            "playbook: investigation are we sure how leases work across the repo",
            env={"PUPPETMASTER_PLAYBOOKS": "0"},
        )
        self.assertTrue(d.should_delegate)
        self.assertEqual(d.playbook, "investigation")


if __name__ == "__main__":
    unittest.main()
