"""Map changed files -> the spec/test paths they affect (B2).

Puppetmaster (via CodeGraph) can tell you the blast radius of a change, but
*which tests cover which files* is a per-repo convention Puppetmaster must not
guess. This is the configurable seam: the caller supplies the changed files and
a small mapping config, and gets back the affected spec paths to run — so
"validate only the affected specs" works generically without Puppetmaster
hardcoding anyone's layout.

Two mapping strategies (use either or both):

- ``rules``  — a list of ``{"match": <glob>, "specs": [<glob/template>, ...]}``.
  Declarative. A spec entry may interpolate the changed path with ``{path}``,
  ``{dir}``, ``{name}``, ``{stem}`` and may itself be a glob resolved against
  ``cwd``. Example: ``{"match": "src/**/*.py", "specs": ["tests/{stem}_test.py"]}``.
- ``command`` — a shell command (string) or argv (list) that receives the
  changed files (one per line on stdin) and prints affected spec paths (one per
  line). For repos whose mapping is logic, not globs.
"""

from __future__ import annotations

import fnmatch
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Optional, Union


def affected_specs(
    changed_files: Iterable[str],
    mapping: dict[str, Any],
    *,
    cwd: Optional[Union[str, Path]] = None,
) -> list[str]:
    """Resolve the spec paths affected by ``changed_files`` under ``mapping``.

    Returns a de-duplicated, stably-ordered list. An empty changed set yields no
    specs. A mapping with neither ``rules`` nor ``command`` is a usage error.
    """
    base = Path(cwd or ".")
    changed = [str(path).strip() for path in changed_files if str(path).strip()]
    if not changed:
        return []

    rules = mapping.get("rules")
    command = mapping.get("command")
    if not rules and not command:
        raise ValueError("affected mapping must define 'rules' and/or 'command'")

    collected: list[str] = []
    if rules:
        collected.extend(_affected_via_rules(changed, rules, base))
    if command:
        collected.extend(_affected_via_command(changed, command, base))

    seen: set[str] = set()
    ordered: list[str] = []
    for spec in collected:
        if spec not in seen:
            seen.add(spec)
            ordered.append(spec)
    return ordered


def _affected_via_rules(changed: list[str], rules: list[dict], cwd: Path) -> list[str]:
    out: list[str] = []
    for path in changed:
        for rule in rules:
            match = rule.get("match")
            if match and fnmatch.fnmatch(path, str(match)):
                for spec in rule.get("specs", []) or []:
                    out.extend(_expand_spec(str(spec), path, cwd))
    return out


def _expand_spec(spec: str, changed_path: str, cwd: Path) -> list[str]:
    """Interpolate path tokens, then glob-expand against ``cwd`` if the spec is a
    pattern. A literal spec is returned as-is (even if the file doesn't exist —
    the caller decides whether a missing spec is an error)."""
    candidate = Path(changed_path)
    try:
        spec = spec.format(
            path=changed_path,
            dir=str(candidate.parent),
            name=candidate.name,
            stem=candidate.stem,
        )
    except (KeyError, IndexError):
        # An unknown placeholder shouldn't crash resolution — use the raw spec.
        pass
    if any(ch in spec for ch in "*?["):
        matches = sorted(cwd.glob(spec))
        return [str(match.relative_to(cwd)) if match.is_absolute() else str(match) for match in matches]
    return [spec]


def _affected_via_command(
    changed: list[str], command: Union[str, list], cwd: Path
) -> list[str]:
    proc = subprocess.run(
        command,
        shell=isinstance(command, str),
        cwd=str(cwd),
        input="\n".join(changed) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"affected-spec command failed (exit {proc.returncode}): {(proc.stderr or '')[-500:]}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def changed_files_from_git(cwd: Union[str, Path], ref_range: str) -> list[str]:
    """Changed file paths for a git ref range (e.g. ``HEAD~1..HEAD``). Empty on
    any git error so the caller can fall back to an explicit list."""
    proc = subprocess.run(
        ["git", "diff", "--name-only", ref_range],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def load_mapping(path: Union[str, Path]) -> dict[str, Any]:
    """Load a JSON mapping config from disk."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("affected mapping config must be a JSON object")
    return data
