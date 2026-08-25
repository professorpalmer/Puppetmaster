"""Optional universal playbook recipes.

pstack-style recipes as orchestrator data, not a Cursor plugin. Five frozen
ids stamp a generic verb, optional swarm roles, and complete gate specs onto
jobs so every adapter sees the same contract.

Import-light: stdlib only. Callers (invocation gate, swarm launch, CLI) own
host-specific verbs.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

PLAYBOOK_IDS = (
    "investigation",
    "bug-fix",
    "feature",
    "interrogate",
    "hillclimb",
)

PLAYBOOKS_DISABLE_ENV = "PUPPETMASTER_PLAYBOOKS"

# Recipes that opt the host in even when the capability score is below the
# delegate bar. Investigation / bug-fix / feature stay score-and-scope gated
# so they cannot steal trivial turns.
_FORCE_DELEGATE_IDS = frozenset({"interrogate", "hillclimb"})


@dataclass(frozen=True)
class PlaybookRecipe:
    playbook_id: str
    suggested_verb: str
    roles: tuple
    payload: dict
    directive: str


RECIPES = {
    "investigation": PlaybookRecipe(
        playbook_id="investigation",
        suggested_verb="puppetmaster_start_swarm",
        roles=("explore", "review"),
        payload={"playbook": "investigation"},
        directive=(
            "Playbook investigation: disjoint explore/review swarm; recall "
            "artifacts at zero token cost; do not expect a PATCH."
        ),
    ),
    "bug-fix": PlaybookRecipe(
        playbook_id="bug-fix",
        suggested_verb="puppetmaster_start_implement",
        roles=(),
        payload={"playbook": "bug-fix"},
        directive=(
            "Playbook bug-fix: reproduce first, then one implement worker. "
            "Attach a command oracle when a test exists; emit a FINDING if "
            "there is no repro."
        ),
    ),
    "feature": PlaybookRecipe(
        playbook_id="feature",
        suggested_verb="puppetmaster_start_implement",
        roles=(),
        payload={
            "playbook": "feature",
            "gates": [{"kind": "require_diff"}],
        },
        directive=(
            "Playbook feature: one implement worker, not a swarm; a real "
            "PATCH is required."
        ),
    ),
    "interrogate": PlaybookRecipe(
        playbook_id="interrogate",
        suggested_verb="puppetmaster_start_swarm",
        roles=("review", "audit"),
        payload={
            "playbook": "interrogate",
            "routing_policy": "quality",
        },
        directive=(
            "Playbook interrogate: adversarial plus quality review swarm; "
            "no PATCH expected."
        ),
    ),
    "hillclimb": PlaybookRecipe(
        playbook_id="hillclimb",
        suggested_verb="puppetmaster_start_implement",
        roles=(),
        payload={"playbook": "hillclimb"},
        directive=(
            "Playbook hillclimb: one PATCH per accepted win; prove the "
            "metric on the real artifact."
        ),
    ),
}

_PIN_IN_PROMPT = re.compile(
    r"\bplaybook\s*[:\s]\s*(" + "|".join(re.escape(i) for i in PLAYBOOK_IDS) + r")\b",
    re.IGNORECASE,
)

_PATTERN_ORDER = (
    (
        "hillclimb",
        (
            re.compile(r"\bhillclimb\b", re.I),
            re.compile(r"\bimprove metric\b", re.I),
            re.compile(r"\buntil the metric\b", re.I),
        ),
    ),
    (
        "interrogate",
        (
            re.compile(r"\binterrogate\b", re.I),
            re.compile(r"\bbreak this diff\b", re.I),
            re.compile(r"\badversarial review\b", re.I),
        ),
    ),
    (
        "investigation",
        (
            re.compile(r"\bare we sure\b", re.I),
            re.compile(r"\bhow does\b", re.I),
            re.compile(r"\bhow do we\b", re.I),
            re.compile(r"\bwhy was\b", re.I),
            re.compile(r"\bwhy is this shaped\b", re.I),
        ),
    ),
    (
        "bug-fix",
        (
            re.compile(r"\brepro first\b", re.I),
            re.compile(r"\breproduce first\b", re.I),
            re.compile(r"\broot-cause\b", re.I),
            re.compile(r"\bthis is a bug\b", re.I),
            re.compile(r"\bdefect\b", re.I),
        ),
    ),
    (
        "feature",
        (
            re.compile(
                r"\b(new behavior|behind a (feature )?flag)\b",
                re.I,
            ),
        ),
    ),
)

_FEATURE_IMPLEMENT_SIGNAL = re.compile(
    r"\b(implement|build|add|create|feature)\b",
    re.I,
)


def recipe_for(playbook_id: str) -> PlaybookRecipe:
    """Return the recipe for ``playbook_id``. Raises on unknown ids."""
    key = str(playbook_id or "").strip().lower()
    recipe = RECIPES.get(key)
    if recipe is None:
        raise ValueError(
            "unknown playbook %r; valid: %s"
            % (playbook_id, ", ".join(PLAYBOOK_IDS))
        )
    return recipe


def playbooks_auto_match_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """False when ``PUPPETMASTER_PLAYBOOKS=0`` (explicit pins still work)."""
    env = env if env is not None else os.environ
    raw = str(env.get(PLAYBOOKS_DISABLE_ENV) or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def match_playbook(
    prompt: str,
    *,
    explicit: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    delegating: Optional[bool] = None,
) -> Optional[str]:
    """Return a recipe id or None. Pure; no I/O besides reading ``env``.

    ``explicit`` always wins (and raises on unknown). Auto patterns are
    skipped when ``PUPPETMASTER_PLAYBOOKS=0``. When ``delegating`` is False,
    only force recipes (interrogate, hillclimb) and in-prompt pins apply.
    """
    if explicit is not None and str(explicit).strip():
        return recipe_for(explicit).playbook_id

    text = prompt or ""
    pinned = _pin_in_prompt(text)
    if pinned is not None:
        return pinned

    if not playbooks_auto_match_enabled(env):
        return None

    matched = _pattern_match(text)
    if matched is None:
        return None
    if delegating is False and matched not in _FORCE_DELEGATE_IDS:
        return None
    return matched


def forces_delegate(playbook_id: Optional[str]) -> bool:
    return bool(playbook_id) and playbook_id in _FORCE_DELEGATE_IDS


def merge_gates(existing: Any, incoming: Any) -> list:
    """Append incoming gate dicts whose kind is not already present."""
    out: list = []
    kinds = set()
    for raw in list(existing or []) + list(incoming or []):
        if not isinstance(raw, dict) or not raw.get("kind"):
            continue
        kind = raw["kind"]
        if kind in kinds:
            continue
        kinds.add(kind)
        out.append(dict(raw))
    return out


def stamp_payload(
    payload: Optional[Mapping[str, Any]],
    playbook_id: str,
    extras: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Merge recipe payload into a worker payload. Never duplicate gate kinds."""
    recipe = recipe_for(playbook_id)
    out = dict(payload or {})
    for key, value in recipe.payload.items():
        if key == "gates":
            out["gates"] = merge_gates(out.get("gates"), value)
        elif key == "playbook":
            out["playbook"] = value
        elif key not in out:
            out[key] = value
        elif key == "routing_policy" and not out.get("routing_policy"):
            out[key] = value
    out["playbook"] = recipe.playbook_id
    extras = extras or {}
    if recipe.playbook_id == "hillclimb":
        command = extras.get("ratchet_command")
        metric = extras.get("metric")
        if command and metric:
            out["gates"] = merge_gates(
                out.get("gates"),
                [{"kind": "ratchet", "command": command, "metric": metric}],
            )
    return out


def resolve_launch_playbook(
    goal: str,
    *,
    explicit: Optional[str] = None,
    roles: Optional[list] = None,
    roles_omitted: bool = False,
    env: Optional[Mapping[str, str]] = None,
) -> tuple:
    """Return ``(playbook_id_or_None, roles_list)`` for a swarm/implement launch."""
    playbook_id = match_playbook(goal or "", explicit=explicit, env=env)
    out_roles = list(roles) if roles else []
    if playbook_id and roles_omitted:
        recipe = recipe_for(playbook_id)
        if recipe.roles:
            out_roles = list(recipe.roles)
    return playbook_id, out_roles


def apply_playbook_to_mapping(args: dict, *, goal_keys: tuple = ("goal", "prompt", "instruction")) -> Optional[str]:
    """Mutate MCP/CLI-like ``args`` with the resolved playbook id.

    Does not fill swarm roles. Injecting recipe roles into MCP ``start_swarm``
    would trip the custom-role adapter guard. ``build_analysis_swarm_specs``
    fills omitted roles from the recipe instead.
    """
    goal = ""
    for key in goal_keys:
        raw = args.get(key)
        if isinstance(raw, str) and raw.strip():
            goal = raw
            break
    explicit = args.get("playbook")
    if explicit is not None and not str(explicit).strip():
        explicit = None
    playbook_id, _out_roles = resolve_launch_playbook(
        goal, explicit=explicit, env=None,
    )
    if playbook_id:
        args["playbook"] = playbook_id
    return playbook_id


def maybe_stamp_payload(payload: Mapping[str, Any], prompt: str, args: Any) -> dict:
    """Stamp ``payload`` when args/prompt resolve a playbook."""
    explicit = None
    extras: dict[str, Any] = {}
    if isinstance(args, Mapping):
        explicit = args.get("playbook")
        extras["ratchet_command"] = args.get("ratchet_command")
        extras["metric"] = args.get("metric")
    else:
        explicit = getattr(args, "playbook", None)
        extras["ratchet_command"] = getattr(args, "ratchet_command", None)
        extras["metric"] = getattr(args, "metric", None)
    playbook_id = match_playbook(prompt or "", explicit=explicit)
    if not playbook_id:
        return dict(payload)
    return stamp_payload(payload, playbook_id, extras)


def _pin_in_prompt(text: str) -> Optional[str]:
    match = _PIN_IN_PROMPT.search(text or "")
    if not match:
        return None
    return recipe_for(match.group(1)).playbook_id


def _pattern_match(text: str) -> Optional[str]:
    lower = text or ""
    for playbook_id, patterns in _PATTERN_ORDER:
        if playbook_id == "feature":
            if not _FEATURE_IMPLEMENT_SIGNAL.search(lower):
                continue
        if any(pattern.search(lower) for pattern in patterns):
            return playbook_id
    return None
