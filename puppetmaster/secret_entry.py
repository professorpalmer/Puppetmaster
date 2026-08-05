"""Helpers for interactive API-key prompts that never echo the value.

When terminal echo is disabled, a paste gives no visual feedback. Reporting the
captured character count and rejecting an input that looks like the same key
pasted twice (common paste-doubling) prevents silent bad writes — lifted from
codex-router's ``secret-entry`` pattern.
"""
from __future__ import annotations

from typing import Optional

MIN_DOUBLED_SECRET_LENGTH = 8


def _normalized(value: Optional[str]) -> str:
    return str(value or "").strip()


def secret_entry_feedback(value: Optional[str]) -> str:
    """Human-readable confirmation of how many characters were captured."""
    key = _normalized(value)
    if not key:
        return "No characters were received."
    characters = len(key)
    suffix = "" if characters == 1 else "s"
    return f"Received {characters} character{suffix}."


def looks_doubled_secret(value: Optional[str]) -> bool:
    """True when ``value`` looks like the same secret pasted twice."""
    key = _normalized(value)
    if len(key) < MIN_DOUBLED_SECRET_LENGTH:
        return False
    if len(key) % 2 == 0:
        half = len(key) // 2
        return key[:half] == key[half:]
    middle = (len(key) - 1) // 2
    return key[middle].isspace() and key[:middle] == key[middle + 1 :]


def secret_entry_problem(value: Optional[str]) -> Optional[str]:
    """Return ``empty``, ``doubled``, or ``None`` when the value looks usable."""
    if not _normalized(value):
        return "empty"
    if looks_doubled_secret(value):
        return "doubled"
    return None
