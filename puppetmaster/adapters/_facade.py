"""Resolve names through the public ``puppetmaster.adapters`` facade.

Tests patch ``puppetmaster.adapters.<name>``; submodule code must look up
those names at call time so patches keep working after the package split.
"""
from __future__ import annotations

from typing import Any


def facade(name: str) -> Any:
    import puppetmaster.adapters as adapters

    return getattr(adapters, name)
