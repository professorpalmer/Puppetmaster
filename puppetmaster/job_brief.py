"""Job-level shared CodeGraph / repo brief for sibling workers.

Computed once at job start (census + CodeGraph for the *job goal*), persisted
under the job state dir, and injected into every worker prompt via
``insert_before_task`` so siblings share an identical prefix segment.

Process-local ``lru_cache`` in ``codegraph.py`` remains a micro-optimization;
this module is the cross-worker / cross-process piece.

Kill switch: ``PUPPETMASTER_JOB_BRIEF=0``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

JOB_BRIEF_ENV = "PUPPETMASTER_JOB_BRIEF"
JOB_BRIEF_FILENAME = "repo_brief.md"
JOB_BRIEF_SECTION_HEADER = "Shared job CodeGraph / repo brief:"


def job_brief_enabled() -> bool:
    """Return False when the optional kill switch disables job briefs."""
    raw = (os.environ.get(JOB_BRIEF_ENV) or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def job_brief_path(job_dir: Union[Path, str]) -> Path:
    return Path(job_dir) / JOB_BRIEF_FILENAME


def build_job_brief(
    goal: str,
    cwd: Union[Path, str, None],
    *,
    max_nodes: int = 15,
) -> str:
    """Build the job-stable brief body (never raises).

    Always includes a repo census when possible. Adds a CodeGraph section for
    ``goal`` when CodeGraph is available; otherwise returns census-only.
    """
    try:
        from puppetmaster.adapters._prompts import _repo_census_section

        census = _repo_census_section(cwd).strip()
    except Exception:
        census = ""

    codegraph_block = ""
    try:
        from puppetmaster.codegraph import (
            codegraph_context,
            codegraph_prompt_section,
        )

        context = codegraph_context(goal, cwd, max_nodes=max_nodes)
        if context:
            # Re-label so this is distinct from per-task CodeGraph sections.
            section = codegraph_prompt_section(context).strip("\n")
            section = section.replace(
                "Shared CodeGraph context for this task:",
                "CodeGraph brief for the job goal:",
                1,
            )
            codegraph_block = section.strip()
    except Exception:
        codegraph_block = ""

    parts = [JOB_BRIEF_SECTION_HEADER]
    if census:
        parts.append(census)
    if codegraph_block:
        parts.append(codegraph_block)
    if len(parts) == 1:
        parts.append(
            "No repository census or CodeGraph context was available for this job."
        )
    return "\n\n".join(parts).strip() + "\n"


def write_job_brief(
    job_dir: Union[Path, str],
    goal: str,
    cwd: Union[Path, str, None],
    *,
    max_nodes: int = 15,
) -> Optional[Path]:
    """Compute and persist the job brief under ``job_dir``. Best-effort.

    Returns the path written, or ``None`` when disabled / write failed.
    """
    if not job_brief_enabled():
        return None
    try:
        root = Path(job_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = job_brief_path(root)
        text = build_job_brief(goal, cwd, max_nodes=max_nodes)
        temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, path)
        return path
    except Exception:
        try:
            temp_path = Path(job_dir) / f".{JOB_BRIEF_FILENAME}.{os.getpid()}.tmp"
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
        return None


def read_job_brief(job_dir: Union[Path, str, None]) -> str:
    """Read persisted brief bytes from ``job_dir``. Empty string on miss."""
    if job_dir is None:
        return ""
    try:
        path = job_brief_path(job_dir)
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def resolve_job_brief_for_task(task) -> str:
    """Load the job brief for ``task`` from the active state dir (never raises)."""
    if not job_brief_enabled():
        return ""
    try:
        job_id = getattr(task, "job_id", None) or ""
        if not job_id:
            return ""
        # Explicit payload override (tests / pre-baked briefs).
        payload = getattr(task, "payload", None) or {}
        inline = payload.get("job_brief")
        if isinstance(inline, str) and inline.strip():
            return inline.strip() + ("\n" if not inline.endswith("\n") else "")
        from puppetmaster.adapters._streaming import _resolve_sidecar_state_dir

        state_dir = _resolve_sidecar_state_dir()
        if state_dir is None:
            from puppetmaster.state import resolve_state_dir

            cwd = payload.get("cwd")
            try:
                state_dir = resolve_state_dir(cwd=Path(cwd) if cwd else None)
            except Exception:
                state_dir = None
        if state_dir is None:
            return ""
        return read_job_brief(Path(state_dir) / "jobs" / job_id)
    except Exception:
        return ""
