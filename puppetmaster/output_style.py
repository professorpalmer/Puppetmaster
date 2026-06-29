"""Optional output-style directives for worker prompts (the "Signal-maximizer").

This shapes how a worker *writes* — never what it reasons about. Output tokens
are a minority of an agentic bill (turns, tool output, and cache dominate), so
the win here is primarily **readability and latency**, with a modest cost bonus
on output-heavy roles. Treat it as a style feature, not a cost lever.

Two tiers, both off by default:

- ``terse``  — drop ceremony, filler, hedging, and restatement; one claim per
  line; state uncertainty as fact. Safe; readable. Recommended default when on.
- ``lithic`` — ``terse`` plus telegraphic glue-dropping (articles/copulas).
  Aggressive; marginal extra savings; can read poorly and occasionally drop a
  disambiguating word. Opt-in, best for machine-consumed worker artifacts.

Enable globally with ``PUPPETMASTER_OUTPUT_STYLE=terse|lithic`` or per task with
``payload.output_style``; an explicit payload value always wins over the env.
Only *form* is compressed — every fact, number, name, path, condition, and
caveat is preserved (lossless on content, lossy only on form).
"""

from __future__ import annotations

from typing import Optional

OUTPUT_STYLE_ENV = "PUPPETMASTER_OUTPUT_STYLE"

TERSE = "terse"
LITHIC = "lithic"
_VALID_STYLES = (TERSE, LITHIC)
_DISABLED_VALUES = {"", "off", "none", "0", "false", "no"}

# The safe tier. Cuts form, keeps signal. The uncertainty rule deliberately
# folds "no empty hedging" and "state real uncertainty as fact" into one line so
# the two never fight — banning hedges must not coerce false confidence.
_TERSE_RULES = (
    "Start with the answer. No greetings, sign-offs, preambles, or postambles.",
    'No self-reference ("I\'d be happy to", "let me", "as an AI").',
    'No filler ("basically", "essentially", "in order to", "it\'s worth noting").',
    'No transition padding ("furthermore", "additionally", "that said"). '
    "Sequence facts directly.",
    "Do not restate the question. No closing summary of what you just said.",
    "One claim per line. Never repeat a claim.",
    "Cut any word removable without changing meaning.",
    "Keep every fact, number, name, path, condition, and caveat that changes "
    "meaning. Never drop signal to save space.",
    'State genuine uncertainty as fact ("unconfirmed: X"). Never express false '
    'confidence and never pad with empty hedges ("I think", "perhaps", "it seems").',
)

# The aggressive add-on. Telegraphic; protects machine-exact spans explicitly.
_LITHIC_EXTRA = (
    "Drop articles, copulas, and grammatical glue wherever meaning survives "
    "(telegraphic style). Keep code, identifiers, paths, and quoted strings "
    "byte-exact.",
)

_HEADER = (
    "OUTPUT STYLE ({style}): emit essential tokens only. This constrains form, "
    "not reasoning — think as fully as needed, then write tight. Compression is "
    "lossless on content, lossy only on form.\nRules:"
)


def normalize_style(value: Optional[str]) -> Optional[str]:
    """Map a raw style value to ``terse`` / ``lithic`` / ``None`` (disabled)."""
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in _DISABLED_VALUES:
        return None
    return token if token in _VALID_STYLES else None


def resolve_output_style(
    payload_value: Optional[str], env_value: Optional[str]
) -> Optional[str]:
    """Resolve the active style for a spec. Explicit payload wins over env.

    A payload that names a *disabled* value (e.g. ``"off"``) suppresses the env
    default for that one spec, mirroring the tri-state skill/memory opt-in.
    """
    if payload_value is not None:
        return normalize_style(payload_value)
    return normalize_style(env_value)


def directive_for(style: Optional[str]) -> str:
    """Render the directive block for a resolved style (empty string if off)."""
    resolved = normalize_style(style)
    if resolved is None:
        return ""
    rules = _TERSE_RULES + (_LITHIC_EXTRA if resolved == LITHIC else ())
    lines = [_HEADER.format(style=resolved)]
    lines.extend(f"- {rule}" for rule in rules)
    return "\n".join(lines)


def apply_output_style(instruction: str, style: Optional[str]) -> str:
    """Prepend the directive to a worker instruction. No-op when disabled."""
    directive = directive_for(style)
    if not directive:
        return instruction
    return f"{directive}\n\n{instruction}"
