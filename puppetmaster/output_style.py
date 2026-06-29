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

Custom directive (bring your own rules)
---------------------------------------
The two tiers are presets. To run your *own* wording instead, supply a verbatim
directive that replaces the built-in block:

- per task: ``payload.output_style_text`` (the directive string itself), or
- globally: ``PUPPETMASTER_OUTPUT_STYLE_TEXT`` (inline) or
  ``PUPPETMASTER_OUTPUT_STYLE_FILE`` (path to a directive file).

Precedence, highest first: payload custom text, payload tier, env custom text,
env tier. A disabled value (``off``/``none``/…) at the payload layer opts that
one spec out even when the env default is on. A custom directive is used exactly
as written — you own its content (and its failure modes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

OUTPUT_STYLE_ENV = "PUPPETMASTER_OUTPUT_STYLE"
OUTPUT_STYLE_TEXT_ENV = "PUPPETMASTER_OUTPUT_STYLE_TEXT"
OUTPUT_STYLE_FILE_ENV = "PUPPETMASTER_OUTPUT_STYLE_FILE"

TERSE = "terse"
LITHIC = "lithic"
CUSTOM = "custom"
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


def normalize_custom_text(value: Optional[str]) -> Optional[str]:
    """A verbatim custom directive: trimmed text, or ``None`` when empty/disabled.

    A custom directive is used exactly as written, so only literal disable words
    (``off``/``none``/…) and blank input map to ``None``; any other string is a
    real directive the caller owns.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in _DISABLED_VALUES:
        return None
    return text


def _is_disabled(value: Optional[str]) -> bool:
    """True only when ``value`` is an explicit disable token (not merely unset)."""
    return value is not None and str(value).strip().lower() in _DISABLED_VALUES


def read_style_file(path: Optional[str]) -> Optional[str]:
    """Read a directive file, returning its stripped contents or ``None``.

    Missing path, unreadable file, or empty contents all resolve to ``None`` so
    a stale ``PUPPETMASTER_OUTPUT_STYLE_FILE`` never silently injects nothing or
    crashes a worker dispatch.
    """
    if not path:
        return None
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    text = text.strip()
    return text or None


def resolve_output(
    *,
    payload_style: Optional[str],
    payload_text: Optional[str],
    env_style: Optional[str],
    env_text: Optional[str],
) -> Optional[tuple[str, str]]:
    """Resolve the active ``(label, directive)`` for a spec, or ``None`` if off.

    ``label`` is ``terse`` / ``lithic`` / ``custom`` (useful for stamping and
    telemetry); ``directive`` is the block to prepend. Precedence, highest
    first: payload custom text, payload tier, env custom text, env tier. An
    explicit disabled value at the payload layer suppresses the env defaults for
    that one spec — mirroring the skill/memory tri-state opt-in.
    """
    custom = normalize_custom_text(payload_text)
    if custom is not None:
        return (CUSTOM, custom)
    if _is_disabled(payload_text):
        return None
    tier = normalize_style(payload_style)
    if tier is not None:
        return (tier, directive_for(tier))
    if _is_disabled(payload_style):
        return None
    custom = normalize_custom_text(env_text)
    if custom is not None:
        return (CUSTOM, custom)
    tier = normalize_style(env_style)
    if tier is not None:
        return (tier, directive_for(tier))
    return None
