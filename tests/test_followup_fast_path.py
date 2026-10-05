from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from puppetmaster import hook_runner, invocation_gate
from puppetmaster.invocation_gate import should_delegate
from puppetmaster.rules import render_agents_block

_ENV: dict = {}
_KILL = {"PUPPETMASTER_FOLLOWUP_FAST_PATH": "0"}

_LONG_FOLLOWUP = (
    "Follow-up on the report the swarm produced yesterday: the summary table "
    "lists the regions in alphabetical order, but the finance team reads it by "
    "revenue, so sort the rows by total revenue descending and keep the totals "
    "row pinned at the bottom. Also rename the 'delta' column header to "
    "'change vs last quarter' and round the percentages to one decimal place. "
    "Nothing else in the report should change, and keep the existing styling."
)


class FollowupFastPathGateTests(unittest.TestCase):
    def test_followups_stay_inline(self):
        prompts = [
            "follow up: change the header color in the report the swarm produced",
            "now also add a --json flag to the command",
            "now change the default timeout to 30 seconds",
            "revise the summary the last run produced to be shorter",
            "tweak the wording in the error message you just generated",
            "same as before but use port 9000",
            "the previous job output has a bug in the date parsing, fix it",
            _LONG_FOLLOWUP,
        ]
        self.assertGreater(len(_LONG_FOLLOWUP), 380)
        for prompt in prompts:
            with self.subTest(prompt=prompt[:40]):
                d = should_delegate(prompt, env=_ENV)
                self.assertFalse(d.should_delegate)
                self.assertEqual(d.matched_signals, ("followup",))
                self.assertIn("follow-up", d.reason)
                self.assertIn("existing pilot", d.reason)

    def test_followup_with_hard_scope_keeps_normal_policy(self):
        for prompt in (
            "follow-up: redo every module across the repo with the new logger",
            "now also migrate all callers across the codebase",
        ):
            with self.subTest(prompt=prompt):
                d = should_delegate(prompt, env=_ENV)
                self.assertTrue(d.should_delegate)
                self.assertNotIn("followup", d.matched_signals)
                self.assertEqual(d, should_delegate(prompt, env=_KILL))

    def test_first_time_adjust_or_tweak_is_not_a_followup(self):
        for prompt in (
            "adjust the auth flow to support OAuth refresh tokens",
            "tweak the retry policy in the http client",
        ):
            with self.subTest(prompt=prompt):
                d = should_delegate(prompt, env=_ENV)
                self.assertNotIn("followup", d.matched_signals)
                self.assertEqual(d, should_delegate(prompt, env=_KILL))

    def test_codegraph_lookup_about_prior_output_still_delegates(self):
        d = should_delegate("where is the parser you just built called from?", env=_ENV)
        self.assertTrue(d.should_delegate)
        self.assertEqual(d.matched_signals, ("codegraph-lookup",))

    def test_revision_as_a_noun_is_not_a_followup(self):
        prompt = "implement the revision history API with tests"
        d = should_delegate(prompt, env=_ENV)
        self.assertNotIn("followup", d.matched_signals)
        self.assertEqual(d, should_delegate(prompt, env=_KILL))

    def test_followup_over_length_cap_keeps_normal_policy(self):
        prompt = "now also add a --json flag to the command. " + ("x" * 600)
        d = should_delegate(prompt, env=_ENV)
        self.assertNotIn("followup", d.matched_signals)

    def test_explicit_trigger_still_delegates(self):
        d = should_delegate("use puppetmaster to tweak the handler", env=_ENV)
        self.assertTrue(d.should_delegate)
        self.assertEqual(d.matched_signals, ("explicit-trigger",))

    def test_kill_switch_restores_previous_decision(self):
        prompt = "now also add a --json flag to the command"
        self.assertFalse(should_delegate(prompt, env=_ENV).should_delegate)
        for value in ("0", "false", "no", "off", " OFF "):
            with self.subTest(value=value):
                d = should_delegate(
                    prompt, env={"PUPPETMASTER_FOLLOWUP_FAST_PATH": value}
                )
                self.assertTrue(d.should_delegate)
                self.assertEqual(d.suggested_verb, "puppetmaster_edit")
                self.assertEqual(d.matched_signals, ("score",))

    def test_non_followup_decisions_unchanged(self):
        expected = [
            ("refactor the auth module across all files", True, "puppetmaster_start_implement"),
            ("audit the whole repo for races", True, "puppetmaster_start_review"),
            ("where is the retry policy defined", True, "puppetmaster_codegraph_search"),
            ("fix a typo in README", False, "puppetmaster_edit"),
            ("finish the module I just wrote and add tests", True, "puppetmaster_edit"),
            ("review the payment flow for risk", False, "puppetmaster_start_review"),
        ]
        for prompt, delegate, verb in expected:
            with self.subTest(prompt=prompt):
                d = should_delegate(prompt, env=_ENV)
                self.assertEqual(d.should_delegate, delegate)
                self.assertEqual(d.suggested_verb, verb)
                self.assertNotIn("followup", d.matched_signals)
                self.assertEqual(d, should_delegate(prompt, env=_KILL))


class FollowupRulesDirectiveTests(unittest.TestCase):
    def test_directive_has_followup_bullet_and_old_bullets(self):
        text = render_agents_block()
        self.assertIn("## When NOT to use Puppetmaster (stay inline)", text)
        self.assertIn("- Trivial single-file edits, typos, one-line fixes", text)
        self.assertIn("- Quick factual questions", text)
        self.assertIn(
            "- Fast interactive iteration where the user is steering turn-by-turn",
            text,
        )
        self.assertIn("- Small follow-ups and revisions to work a Puppetmaster job", text)
        self.assertIn("resume that worker with `resume_from`", text)
        section = text.split("## When NOT to use Puppetmaster (stay inline)", 1)[1]
        section = section.split("## Fallback", 1)[0]
        self.assertLess(section.index("Fast interactive"), section.index("Small follow-ups"))


class HookDecisionLogTests(unittest.TestCase):
    _ENV = {"PUPPETMASTER_HOOK_ASSUME_TOOLS": "1"}

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.env = dict(self._ENV, PUPPETMASTER_HOME=str(self.root))
        self.path = self.root / "invocation_decisions.jsonl"

    def _hook(self, prompt, env=None):
        return hook_runner.handle_hook(
            {"prompt": prompt}, host="cursor", event="beforeSubmitPrompt",
            env=env if env is not None else self.env,
        )

    def test_writes_one_line_without_prompt_text(self):
        prompt = "now also add a --json flag to the secret-widget command"
        self._hook(prompt)
        self.assertEqual(hook_runner.decision_log_path(self.env), self.path)
        raw = self.path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        self.assertEqual(len(lines), 1)
        self.assertNotIn("secret-widget", raw)
        rec = json.loads(lines[0])
        self.assertEqual(
            set(rec),
            {"ts", "delegate", "mode", "policy", "reason", "signals", "score", "role",
             "suggested_verb", "prompt_sha256", "prompt_chars"},
        )
        self.assertIs(rec["delegate"], False)
        self.assertEqual(rec["mode"], "pilot")
        self.assertEqual(rec["policy"], invocation_gate.GATE_POLICY_VERSION)
        self.assertEqual(rec["signals"], ["followup"])
        self.assertEqual(rec["prompt_chars"], len(prompt))
        self.assertEqual(len(rec["prompt_sha256"]), 64)
        self.assertTrue(rec["ts"].endswith("+00:00"))

    def test_rotates_past_size_cap(self):
        with patch.object(hook_runner, "DECISION_LOG_MAX_BYTES", 100):
            self._hook("audit the whole repo for races")
            self._hook("audit the whole repo for races")
            self.assertTrue(self.path.with_name(self.path.name + ".1").exists())
            self._hook("audit the whole repo for races")
        rotated = self.path.with_name(self.path.name + ".1")
        self.assertEqual(len(self.path.read_text(encoding="utf-8").splitlines()), 1)
        self.assertGreaterEqual(len(rotated.read_text(encoding="utf-8").splitlines()), 1)

    def test_stale_size_from_a_second_hook_does_not_clobber_rotated_history(self):
        rotated = self.path.with_name(self.path.name + ".1")
        with patch.object(hook_runner, "DECISION_LOG_MAX_BYTES", 100):
            self.path.write_text("x" * 500 + "\n", encoding="utf-8")
            hook_runner._rotate_decision_log(self.path)
            self.path.write_text("fresh\n", encoding="utf-8")
            real_stat = Path.stat
            stale = {"left": 1}

            def stale_first_stat(path_self, *args, **kwargs):
                result = real_stat(path_self, *args, **kwargs)
                if path_self == self.path and stale["left"]:
                    stale["left"] -= 1
                    return os.stat_result((result.st_mode, 0, 0, 0, 0, 0, 10_000, 0, 0, 0))
                return result

            with patch.object(type(self.path), "stat", stale_first_stat):
                hook_runner._rotate_decision_log(self.path)
        self.assertEqual(rotated.read_text(encoding="utf-8"), "x" * 500 + "\n")
        self.assertEqual(self.path.read_text(encoding="utf-8"), "fresh\n")

    def test_unwritable_path_is_swallowed(self):
        blocker = self.root / "blocker"
        blocker.write_text("not a dir", encoding="utf-8")
        env = dict(self._ENV, PUPPETMASTER_HOME=str(blocker / "nested"))
        r = self._hook("audit the whole repo for races", env=env)
        self.assertEqual(r.action, "allow")
        self.assertTrue(r.decision.should_delegate)

    def test_kill_switch_skips_write(self):
        env = dict(self.env, PUPPETMASTER_AUTO_INVOKE_DISABLED="1")
        decision = should_delegate("audit the whole repo for races", env=self._ENV)
        hook_runner.record_decision(decision, "audit the whole repo for races", env=env)
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
