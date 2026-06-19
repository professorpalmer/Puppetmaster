"""Classifier-gated auto-invocation decision — the delegation *gate*.

This is the pure brain that answers one question for a single user prompt (or a
single native tool call): should the host agent **delegate** this work to a
Puppetmaster verb instead of grinding through it inline?

It reuses the model router's pure-function task classifier
(:func:`puppetmaster.router.classify_capability_needed`) so there is exactly
one capability heuristic in the codebase. The gate adds the *delegate-vs-inline
policy* on top of that score — it never introduces a second classifier.

Design constraints (load-bearing — these come straight out of the redteam):

* **Pure + import-light.** No SQLite, CodeGraph, network, model registry, or
  filesystem access on the hot path. A host hook may call this before *every*
  user turn, so it must stay microsecond-scale and depend on nothing beyond the
  standard library and :mod:`puppetmaster.router`.
* **Fail open.** Callers (hooks, proxy) must treat any error as "don't block".
  The gate must never become the reason a user can't get work done.
* **Kill switch.** ``PUPPETMASTER_AUTO_INVOKE_DISABLED=1`` (alias
  ``PUPPETMASTER_INVOCATION_GATE_DISABLED=1``) forces ``should_delegate=False``
  everywhere, so a user who dislikes it can turn it off without uninstalling.
* **Trivial carve-out first.** Typos, renames, formatting, one-line/single-file
  edits, and quick factual questions stay inline even when the raw capability
  score is moderate. A false positive that blocks trivial work is exactly how
  you train users to disable the whole system.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Mapping, Optional

from puppetmaster.router import TaskSignals, classify_capability_needed

# ----- Tunables (all overridable via env so users can calibrate) -----------

#: Delegate when the classifier score is at least this high. Conservative by
#: design: a high bar means fewer false positives on borderline tasks.
DEFAULT_THRESHOLD = 60
#: Below this score a task is treated as trivially inline, full stop.
DEFAULT_TRIVIAL_THRESHOLD = 35
#: Prompts longer than this can't be "trivial" no matter what easy words match.
_TRIVIAL_MAX_CHARS = 240

_DISABLE_ENV_VARS = (
    "PUPPETMASTER_AUTO_INVOKE_DISABLED",
    "PUPPETMASTER_INVOCATION_GATE_DISABLED",
)
_THRESHOLD_ENV_VAR = "PUPPETMASTER_AUTO_INVOKE_THRESHOLD"

# ----- Intent signals ------------------------------------------------------

# Phrases that explicitly *force* delegation regardless of score — the user
# named Puppetmaster (or a swarm) directly.
_EXPLICIT_DELEGATE_PATTERNS = [
    re.compile(r"\buse puppet ?master\b"),
    re.compile(r"\bpuppet ?master this\b"),
    re.compile(r"\bpm this\b"),
    re.compile(r"\b(swarm|delegate|fan ?out) this\b"),
    re.compile(r"\bstart a swarm\b"),
]

# Phrases where the user explicitly opts *out* — stay inline even if the task
# looks heavy. This is the honest escape hatch the redteam asked for.
_EXPLICIT_INLINE_PATTERNS = [
    re.compile(r"\bno puppet ?master\b"),
    re.compile(r"\bno swarm\b"),
    re.compile(r"\bdo(n'?t| not) (use )?(puppet ?master|swarm)\b"),
    re.compile(r"\b(just|simply) answer\b"),
    re.compile(r"\bdo it inline\b"),
    re.compile(r"\binline only\b"),
]

# Multi-file / cross-cutting scope words. When any of these match we delegate
# even if the raw score lands just under the threshold — broad scope is the
# single strongest "this should have been a swarm" signal.
_HARD_SCOPE_PATTERNS = [
    re.compile(r"\baudit\b"),
    re.compile(r"\bsecurity\b"),
    re.compile(r"\brefactor\b"),
    re.compile(r"\bmigrat(e|ion)\b"),
    re.compile(r"\bacross the (repo|codebase|project)\b"),
    re.compile(r"\b(every|all) (file|files|function|functions|module|modules|usage|usages|caller|callers)\b"),
    re.compile(r"\bfind all\b"),
    re.compile(r"\b(review|analy[sz]e) (the )?(whole|entire|repo|codebase)\b"),
    re.compile(r"\bend[-\s]?to[-\s]?end\b"),
    re.compile(r"\bcross[-\s]?cutting\b"),
    # Tracing / data-flow questions are inherently multi-file even when no
    # "every/all/across" keyword appears — the redteam's "implied scope" gap.
    re.compile(r"\btrace\b"),
    re.compile(r"\bcall ?graph\b"),
    re.compile(r"\bdata ?flow\b"),
    re.compile(r"\bflows? through\b"),
]

# Trivial signals — short, obviously-inline intents.
_TRIVIAL_PATTERNS = [
    re.compile(r"\btypo\b"),
    re.compile(r"\brename\b"),
    re.compile(r"\bformat(ting)?\b"),
    re.compile(r"\blint\b"),
    re.compile(r"\badd a comment\b"),
    re.compile(r"\bone[-\s]?line(r)?\b"),
    re.compile(r"\bsingle[-\s]?file\b"),
    re.compile(r"\bwhat (is|does|are)\b"),
    re.compile(r"\bquick question\b"),
]

# Role inference → (canonical role for the classifier, suggested Puppetmaster
# verb the host should reach for). Order matters: first match wins.
_ROLE_INFERENCE = [
    (re.compile(r"\b(security|vuln|exploit|cve)\b"), "security-review", "puppetmaster_start_cursor_review"),
    (re.compile(r"\b(audit|review|risk|find issues|what could break)\b"), "review", "puppetmaster_start_cursor_review"),
    (re.compile(r"\b(architect|design|approach|trade[-\s]?off|plan|scope)\b"), "plan", "puppetmaster_start_cursor_plan"),
    # Implementation intent → a SINGLE implement worker, never a fan-out swarm.
    # Broadened beyond the old narrow verb list because plain feature work
    # ("add a CSV export endpoint", "create the webhook handler", "wire up
    # retries") previously fell through to the read-only swarm default — the
    # task-shape mismatch that makes coupled changes collide. Trivial edits
    # ("add a comment", "fix a typo") are held inline by the trivial carve-out
    # in should_delegate, not here.
    (
        re.compile(
            r"\b(refactor|migrat(e|ion)|implement|build|create|add|writ(e|ing)|"
            r"wire[-\s]?up|set[-\s]?up|scaffold|integrat(e|ion)|port|patch|fix|"
            r"rewrite|endpoint|feature|hook up)\b"
        ),
        "implement",
        "puppetmaster_start_implement",
    ),
    (re.compile(r"\b(where is|who calls|what implements|find all|trace|call graph)\b"), "explore", "puppetmaster_codegraph_search"),
]
_DEFAULT_VERB = "puppetmaster_start_cursor_swarm"

# Verbs that run a single edit-capable worker in an isolated worktree. These
# must NOT be described as a "fan-out swarm": the whole point is that a coupled
# implementation lands as one coherent change (one branch, one PATCH artifact)
# instead of parallel editors stacking commits that are unaware of each other.
_IMPLEMENT_VERBS = frozenset(
    {
        "puppetmaster_start_implement",
        "puppetmaster_start_cursor_implement",
        "puppetmaster_start_claude_implement",
    }
)

# The lightweight single in-place edit verb. Distinct from the implement verbs:
# synchronous, edits the working tree directly (no isolated worktree), cheapest
# sufficient model, returns the diff. The gate steers a *focused* implement
# intent (no broad-scope signal) here instead of the heavier implement job.
_EDIT_VERB = "puppetmaster_edit"
_EDIT_VERBS = frozenset({_EDIT_VERB})

# Verbs that resolve a "where/what/how is X" question structurally instead of
# crawling the tree. Framed as look-up, not fan-out.
_CODEGRAPH_VERBS = frozenset(
    {
        "puppetmaster_codegraph_search",
        "puppetmaster_codegraph_context",
    }
)


@dataclass(frozen=True)
class DelegationDecision:
    """The gate's verdict for one prompt or tool call.

    ``should_delegate`` is the only field a caller must act on; the rest exist
    so the directive a hook injects can explain *why* (which is what makes the
    nudge persuasive instead of an opaque "do this") and so decisions are
    auditable/tunable without re-running anything.
    """

    should_delegate: bool
    reason: str
    suggested_verb: str
    capability_score: int
    role: str
    matched_signals: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "should_delegate": self.should_delegate,
            "reason": self.reason,
            "suggested_verb": self.suggested_verb,
            "capability_score": self.capability_score,
            "role": self.role,
            "matched_signals": list(self.matched_signals),
        }

    def directive(self) -> str:
        """Instruction a hook injects into the host's context when delegating.

        The framing is verb-aware on purpose. The old one-size message told the
        host to "fan it out to a swarm" for *every* delegated task — including
        single implementations, which is exactly the misfire that makes parallel
        workers stack uncoordinated commits. So implementation work is steered
        to one worker in a clean worktree, lookups to CodeGraph, and only
        genuinely read-only analysis to a swarm.
        """
        if not self.should_delegate:
            return ""
        verb = self.suggested_verb
        tail = "If this is actually trivial, say 'do it inline' to skip."
        if verb in _EDIT_VERBS:
            return (
                f"[Puppetmaster] This is a single, focused edit (capability "
                f"{self.capability_score}, {self.reason}). Delegate it to "
                f"`{verb}` — one in-place edit on the cheapest sufficient model, "
                f"with CodeGraph to locate the site, returned as a reviewable "
                f"diff. It's the snappy path between editing inline yourself and "
                f"a full implement job; no isolated worktree, no job to poll. "
                f"Reach for `puppetmaster_start_implement` instead only if this "
                f"grows into a coupled multi-file change. {tail}"
            )
        if verb in _IMPLEMENT_VERBS:
            return (
                f"[Puppetmaster] This is a single implementation task (capability "
                f"{self.capability_score}, {self.reason}). Delegate it to ONE "
                f"implement worker in a clean worktree via `{verb}` — not a "
                f"fan-out swarm. A single worker keeps the change coherent and "
                f"captures a PATCH artifact; parallel editors stack commits that "
                f"are unaware of each other. Reserve swarms for the "
                f"explore/review/audit passes around the feature. {tail}"
            )
        if verb in _CODEGRAPH_VERBS:
            return (
                f"[Puppetmaster] Resolve this with CodeGraph first (capability "
                f"{self.capability_score}, {self.reason}): call `{verb}` to locate "
                f"the code instead of grepping, then read only what it points to. "
                f"If the MCP tool isn't reachable, run `python -m puppetmaster "
                f"codegraph search '<query>'`. {tail}"
            )
        return (
            f"[Puppetmaster] This warrants a read-only analysis pass (capability "
            f"{self.capability_score}, {self.reason}). Call `{verb}` to fan it out "
            f"to a swarm; recall results with puppetmaster_artifacts at zero token "
            f"cost. For the implementation itself, follow up with a single "
            f"`puppetmaster_start_implement` worker rather than editing in the "
            f"swarm. {tail}"
        )


def gate_disabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """True when a kill-switch env var is set to a truthy value."""
    env = env if env is not None else os.environ
    for name in _DISABLE_ENV_VARS:
        if _truthy(env.get(name)):
            return True
    return False


def infer_role_and_verb(prompt: str) -> tuple[str, str]:
    """Infer a (role, suggested_verb) pair from a free-form prompt.

    Pure pattern match — the role feeds the classifier's role base score and
    the verb is what a hook tells the host to call. Defaults to a read-only
    multi-role swarm, the safe daily-driver entry point.
    """
    lower = prompt.lower()
    for pattern, role, verb in _ROLE_INFERENCE:
        if pattern.search(lower):
            return role, verb
    return "explore", _DEFAULT_VERB


def should_delegate(
    prompt: str,
    *,
    role: Optional[str] = None,
    payload_size_chars: int = 0,
    threshold: Optional[int] = None,
    trivial_threshold: Optional[int] = None,
    env: Optional[Mapping[str, str]] = None,
) -> DelegationDecision:
    """Decide whether ``prompt`` should be delegated to a Puppetmaster verb.

    The order of checks is the policy, and it is deliberate:

    1. Kill switch — a disabled gate never delegates.
    2. Explicit inline opt-out — the user's word wins.
    3. Explicit delegate trigger — "use Puppetmaster" forces a swarm.
    4. Trivial carve-out — short + obviously-easy prompts stay inline.
    5. Score threshold — delegate at/above the (conservative) bar.
    6. Hard scope override — broad multi-file scope delegates even just under
       the bar, because scope is the strongest swarm signal.

    Always returns a decision; never raises for normal input. Callers should
    still treat exceptions as fail-open.
    """
    env = env if env is not None else os.environ
    threshold = _resolve_threshold(threshold, env)
    trivial_threshold = (
        trivial_threshold if trivial_threshold is not None else DEFAULT_TRIVIAL_THRESHOLD
    )

    inferred_role, suggested_verb = infer_role_and_verb(prompt)
    role = role or inferred_role
    lower = prompt.lower()

    if gate_disabled(env):
        return DelegationDecision(
            False, "auto-invocation disabled via kill-switch env", suggested_verb, 0, role,
            ("kill-switch",),
        )

    if _matches_any(_EXPLICIT_INLINE_PATTERNS, lower):
        return DelegationDecision(
            False, "user explicitly asked to stay inline", suggested_verb, 0, role,
            ("explicit-inline",),
        )

    if _matches_any(_EXPLICIT_DELEGATE_PATTERNS, lower):
        return DelegationDecision(
            True, "explicit Puppetmaster trigger phrase", suggested_verb, 100, role,
            ("explicit-trigger",),
        )

    signals = TaskSignals(
        instruction=prompt, role=role, payload_size_chars=payload_size_chars
    )
    score = classify_capability_needed(signals)

    has_hard_scope = _matches_any(_HARD_SCOPE_PATTERNS, lower)
    has_trivial = _matches_any(_TRIVIAL_PATTERNS, lower)

    # Scope-aware refinement: a focused implementation intent with NO broad-scope
    # signal is a single edit, not a coupled multi-file job — steer it to the
    # lightweight in-place ``edit`` verb (cheap model, CodeGraph, inline diff)
    # rather than the heavier ``start_implement`` worktree job. Broad scope
    # ("across the repo", "every caller", refactor/migrate) keeps the implement
    # verb, where an isolated worktree + one coherent PATCH is the right shape.
    if suggested_verb in _IMPLEMENT_VERBS and not has_hard_scope:
        suggested_verb = _EDIT_VERB

    # Trivial carve-out: a short prompt with an explicit easy-intent signal and
    # no broad scope stays inline — even if role inference (e.g. "add ...") put
    # it in the high-scoring `implement` bucket. An explicit trivial signal
    # ("add a comment", "fix a typo", "rename", one-line/single-file) should win
    # over a role-inflated score; otherwise broadening implement detection would
    # start delegating routine one-liners, which is how users learn to disable
    # the gate entirely.
    if (
        has_trivial
        and not has_hard_scope
        and len(prompt) <= _TRIVIAL_MAX_CHARS
    ):
        return DelegationDecision(
            False,
            f"explicit trivial signal (score {score}); staying inline",
            suggested_verb, score, role, ("trivial",),
        )

    if score < trivial_threshold and not has_hard_scope:
        return DelegationDecision(
            False, f"score {score} below trivial threshold {trivial_threshold}",
            suggested_verb, score, role,
        )

    if score >= threshold:
        return DelegationDecision(
            True, f"capability score {score} >= threshold {threshold}",
            suggested_verb, score, role, ("score",),
        )

    if has_hard_scope:
        return DelegationDecision(
            True,
            f"broad multi-file scope signal (score {score}, just under "
            f"threshold {threshold})",
            suggested_verb, score, role, ("hard-scope",),
        )

    return DelegationDecision(
        False, f"score {score} < threshold {threshold}, no broad-scope signal",
        suggested_verb, score, role,
    )


# ----- internals -----------------------------------------------------------


def _resolve_threshold(explicit: Optional[int], env: Mapping[str, str]) -> int:
    if explicit is not None:
        return explicit
    raw = env.get(_THRESHOLD_ENV_VAR)
    if raw:
        try:
            return max(0, min(100, int(raw)))
        except ValueError:
            pass
    return DEFAULT_THRESHOLD


def _matches_any(patterns, text: str) -> bool:
    return any(p.search(text) for p in patterns)


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
