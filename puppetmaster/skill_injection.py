"""Select live Hermes skills and inject their bodies into routed workers.

This is the return leg of the puppetmaster-learn flywheel. ``puppetmaster-learn``
distills a swarm into a skill CANDIDATE (swarm -> skill); this module hands a
routed Hermes worker the bodies of the user's existing *live* skills
(skill -> worker), so the worker knows what the user knows.

It is a SECOND CONSUMER of the existing trusted-planner injection pattern — the
mirror image of ``adapters.prompt_with_memory`` / the orchestrator's
``retrieved_memory`` pass. The worker's access surface never changes:
``--ignore-rules`` stays set, no ``skills`` toolset is granted, and the
persona/rules layer stays suppressed. The orchestrator — which already sees
everything — does the selection and hands the worker a curated packet.

See ``docs/specs/hermes-skill-injection.md`` for the full design. Key invariants:

* SKILL BODIES ONLY — frontmatter is parsed for selection then stripped; the
  persona (SOUL.md) / rules layer is never read or injected.
* The cap is a TOKEN BUDGET (the packet re-rides every worker episode), with a
  count cap as secondary safety.
* Storage coupling is ISOLATED to :func:`discover_hermes_skills` — the one place
  that knows both Hermes' skills directory (via ``installers.hermes_skills_dir``)
  and the SKILL.md frontmatter format. A storage reorg's blast radius is that
  one function, and the failure mode (zero discovered) is observable upstream.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

# Conservative defaults: the packet re-rides every worker episode, so keep the
# budget small. Both are tunable per call (and via env at the orchestrator).
DEFAULT_SKILL_TOKEN_BUDGET = 1200
DEFAULT_SKILL_COUNT_CAP = 3

# ~4 chars/token, matching the router's own heuristic (router.estimate_tokens_in).
_CHARS_PER_TOKEN = 4

# Tokens too generic to carry selection signal. Deliberately tiny — skill
# descriptions are written to be discriminative, so light stopwording is enough.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
        "use", "used", "using", "when", "this", "that", "your", "you", "it",
        "is", "are", "be", "as", "at", "by", "from", "into", "skill", "skills",
        "via", "any", "all", "not", "but", "if", "then", "do", "does",
    }
)


@dataclass(frozen=True)
class SkillDoc:
    """One discovered live Hermes skill.

    ``body`` is the guidance prose with frontmatter stripped — never the
    persona/rules layer. ``description`` and ``name`` come from the frontmatter
    and drive selection.
    """

    name: str
    description: str
    body: str
    path: str


def estimate_tokens(text: str) -> int:
    """Rough token count for a string (~4 chars/token, matching the router)."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _parse_frontmatter(text: str) -> "tuple[dict, str]":
    """Split a SKILL.md into (frontmatter_fields, body).

    Minimal, YAML-free (the orchestrator must not depend on PyYAML): reads the
    leading ``---`` ... ``---`` block as ``key: value`` lines and returns the
    remainder as the body. No frontmatter → ({}, full text).
    """
    if not text.startswith("---"):
        return {}, text.strip()
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text.strip()
    fields: dict = {}
    body_start = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            body_start = index + 1
            break
        match = re.match(r"\s*([A-Za-z0-9_-]+)\s*:\s*(.*)$", lines[index])
        if match:
            key = match.group(1).strip().lower()
            value = match.group(2).strip().strip("'\"")
            fields[key] = value
    if body_start is None:
        return {}, text.strip()
    return fields, "\n".join(lines[body_start:]).strip()


def discover_hermes_skills(
    *,
    skills_dir: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
) -> "list[SkillDoc]":
    """Discover the user's live Hermes skills. The ONLY storage-coupled function.

    Reads ``<skills_dir>/*/SKILL.md`` (or, by default, ``hermes_skills_dir(env)``
    which respects ``$HERMES_HOME``). A plain filesystem read — no Hermes import.
    Skills with neither a usable name nor description fall back to the directory
    name. Returns an empty list (not an error) when the directory is absent, so
    the caller can treat "no skills" as an observable, non-fatal condition.
    """
    if skills_dir is None:
        from puppetmaster.installers import hermes_skills_dir

        skills_dir = hermes_skills_dir(env)
    base = Path(skills_dir)
    if not base.is_dir():
        return []

    discovered: "list[SkillDoc]" = []
    for skill_md in sorted(base.glob("*/SKILL.md")):
        try:
            text = skill_md.read_text(encoding="utf-8")
        except OSError:
            continue
        fields, body = _parse_frontmatter(text)
        name = fields.get("name") or skill_md.parent.name
        description = fields.get("description") or _first_line(body)
        if not body.strip():
            continue
        discovered.append(
            SkillDoc(
                name=str(name),
                description=str(description),
                body=body,
                path=str(skill_md.parent),
            )
        )
    return discovered


def _first_line(text: str, *, max_len: int = 200) -> str:
    for line in text.splitlines():
        stripped = line.lstrip("#").strip()
        if stripped:
            return stripped[:max_len]
    return ""


def _tokenize(text: str) -> "set[str]":
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _relevance(instruction_tokens: "set[str]", skill: SkillDoc) -> int:
    """Shared-token overlap of the instruction against the skill's name+description.

    Deliberately simple and dependency-free; the ranking function is the
    swappable part of the design (embeddings are a v2 drop-in). The architectural
    commitment is the cap + the seam, not this scorer.
    """
    skill_tokens = _tokenize(skill.name + " " + skill.description)
    return len(instruction_tokens & skill_tokens)


def select_skills_for_task(
    instruction: str,
    skills: "list[SkillDoc]",
    *,
    token_budget: int = DEFAULT_SKILL_TOKEN_BUDGET,
    max_count: int = DEFAULT_SKILL_COUNT_CAP,
) -> "list[SkillDoc]":
    """Pick the highest-relevance skills that fit the per-turn token budget.

    Greedy by relevance (then shorter body as a tie-break — more specific, and
    cheaper to re-ride every episode). A skill is admitted only if it has
    nonzero relevance AND adding it keeps the rendered packet within
    ``token_budget``; ``max_count`` is the secondary safety cap.
    """
    if not skills or token_budget <= 0 or max_count <= 0:
        return []
    instruction_tokens = _tokenize(instruction)
    if not instruction_tokens:
        return []

    ranked = sorted(
        ((_relevance(instruction_tokens, skill), skill) for skill in skills),
        key=lambda pair: (-pair[0], len(pair[1].body), pair[1].name),
    )

    selected: "list[SkillDoc]" = []
    for score, skill in ranked:
        if score <= 0 or len(selected) >= max_count:
            break
        if estimate_tokens(render_skill_packet_from_docs(selected + [skill])) > token_budget:
            continue
        selected.append(skill)
    return selected


def packet_from_docs(skills: "list[SkillDoc]") -> "list[dict]":
    """Serialize selected skills into the payload form (a LIST, never a str).

    Stored as ``task.payload["injected_skills"]``. Keeping it a list (not a
    concatenated string) means it is NOT counted in routing's
    ``payload_size_chars`` — only the explicit ``estimated_tokens_in`` bump
    accounts for it, so there is no double-counting.
    """
    return [{"name": skill.name, "body": skill.body} for skill in skills]


def render_skill_packet(injected: "list[dict]") -> str:
    """Render the payload skill list into the prompt block the worker receives.

    The single source of truth for the injected text, used by both
    ``adapters.prompt_with_skills`` (to build the prompt) and the token-budget
    estimate (so the estimate matches what the worker actually pays for).
    """
    blocks = []
    for entry in injected or []:
        name = str(entry.get("name") or "").strip()
        body = str(entry.get("body") or "").strip()
        if not body:
            continue
        header = f"### {name}" if name else "### skill"
        blocks.append(f"{header}\n{body}")
    if not blocks:
        return ""
    intro = (
        "Relevant skills from your live library (guidance only — apply where "
        "useful, and verify before relying on them):"
    )
    return intro + "\n\n" + "\n\n".join(blocks)


def render_skill_packet_from_docs(skills: "list[SkillDoc]") -> str:
    """Convenience: render straight from :class:`SkillDoc` objects (for budgeting)."""
    return render_skill_packet(packet_from_docs(skills))
