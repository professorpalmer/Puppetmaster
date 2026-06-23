"""puppetmaster-learn — the auto-/learn flywheel for Hermes.

On ``on_session_end`` this plugin distills the most recent *durable*
Puppetmaster swarm — one that COMPLETEd and produced a finding or patch
artifact — into a reusable Hermes skill CANDIDATE. Candidates are written to
``~/.hermes/skills-candidates/`` for a human to review and promote; this plugin
NEVER writes a live skill and NEVER starts an agent loop.

Conservative v1 design constraints (all deliberate):

* OPT-IN. Does nothing unless ``PUPPETMASTER_LEARN`` is truthy.
* BACKGROUNDED. All work runs in a daemon thread so session teardown is never
  blocked, even though Hermes already swallows hook exceptions.
* BEST-EFFORT. Every code path is wrapped so the hook can never raise.
* NO ``import puppetmaster``. This plugin runs in Hermes' interpreter (3.11),
  which is a different interpreter from the one that runs Puppetmaster (3.9).
  We talk to Puppetmaster by shelling out to *its* interpreter's CLI.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_OPT_IN_VALUES = {"1", "true", "yes", "on"}
# Only a swarm that finished in the recent past belongs to "this" session; an
# older COMPLETE job is unrelated work we must not re-distill.
_RECENT_WINDOW_SECONDS = 30 * 60
_SOURCE = "puppetmaster-auto-learn"


def register(ctx) -> None:
    ctx.register_hook("on_session_end", _on_session_end)


def _on_session_end(
    session_id=None,
    completed=None,
    interrupted=None,
    model=None,
    platform=None,
    **kwargs,
) -> None:
    """Opt-in, backgrounded entrypoint. Never raises."""
    try:
        if os.environ.get("PUPPETMASTER_LEARN", "").strip().lower() not in _OPT_IN_VALUES:
            return
        cwd = os.environ.get("TERMINAL_CWD") or os.getcwd()
        thread = threading.Thread(
            target=_distill_recent_swarm,
            args=(cwd,),
            name="puppetmaster-learn",
            daemon=True,
        )
        thread.start()
    except Exception:  # pragma: no cover - hook must never raise into Hermes
        logger.debug("puppetmaster-learn: on_session_end gate failed", exc_info=True)


# ---------------------------------------------------------------------------
# Distillation (background thread)
# ---------------------------------------------------------------------------

def _distill_recent_swarm(cwd: Optional[str] = None) -> Optional[str]:
    """Find the most recent durable PM swarm for ``cwd`` and write a candidate.

    Returns the candidate path it wrote (or ``None`` when nothing qualified).
    Wrapped end-to-end so a background thread can never crash the host.
    """
    try:
        cwd = cwd or os.environ.get("TERMINAL_CWD") or os.getcwd()
        pyexe = _find_pm_interpreter()
        job_id = _query_latest_job_id(pyexe, cwd)
        if not job_id:
            return None
        detail = _query_job_detail(pyexe, cwd, job_id)
        if detail is None or not _is_durable(detail):
            return None

        summary = _query_job_summary(pyexe, cwd, job_id)
        goal = detail.get("goal") or _goal_from_summary(summary)
        job = {
            "id": job_id,
            "goal": goal,
            "summary": summary,
            "cwd": cwd,
        }
        candidate = build_skill_candidate(job)
        home = _hermes_home()
        path = _write_candidate(
            candidate["slug"],
            candidate["skill_md"],
            candidate["meta"],
            home=home,
        )
        if path is not None:
            logger.info("puppetmaster-learn: wrote skill candidate %s from job %s", path, job_id)
        return path
    except Exception:  # pragma: no cover - best-effort background work
        logger.debug("puppetmaster-learn: distillation failed", exc_info=True)
        return None


def _hermes_home() -> Path:
    home = os.environ.get("HERMES_HOME")
    return Path(home).expanduser() if home else Path("~/.hermes").expanduser()


def _find_pm_interpreter() -> str:
    """Resolve the interpreter that can run Puppetmaster's CLI.

    Reads ``mcp_servers.puppetmaster.command`` from ``~/.hermes/config.yaml``
    (the entry Puppetmaster's own installer writes), preferring PyYAML when it
    happens to be importable and otherwise doing a tiny line-scoped parse so we
    never add a hard PyYAML dependency to Hermes' interpreter. Falls back to
    ``python3`` when the config or key is absent.
    """
    config_path = _hermes_home() / "config.yaml"
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        return "python3"

    command = _read_puppetmaster_command_with_yaml(text)
    if command is None:
        command = _read_puppetmaster_command_minimal(text)
    return command or "python3"


def _read_puppetmaster_command_with_yaml(text: str) -> Optional[str]:
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    try:
        loaded = yaml.safe_load(text)
    except Exception:
        return None
    if not isinstance(loaded, dict):
        return None
    servers = loaded.get("mcp_servers")
    if not isinstance(servers, dict):
        return None
    entry = servers.get("puppetmaster")
    if not isinstance(entry, dict):
        return None
    command = entry.get("command")
    return command if isinstance(command, str) and command else None


def _read_puppetmaster_command_minimal(text: str) -> Optional[str]:
    """YAML-free fallback: find ``command:`` inside the puppetmaster MCP block.

    Tracks indentation so we only read the command belonging to
    ``mcp_servers.puppetmaster`` and not some other server's command line.
    """
    lines = text.splitlines()
    in_mcp_servers = False
    in_puppetmaster = False
    puppetmaster_indent = -1
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        if not in_mcp_servers:
            if stripped.rstrip().rstrip(":") == "mcp_servers":
                in_mcp_servers = True
            continue
        if not in_puppetmaster:
            if indent == 0 and stripped.endswith(":"):
                # Left the mcp_servers block entirely.
                in_mcp_servers = False
                continue
            if stripped.rstrip().rstrip(":") == "puppetmaster":
                in_puppetmaster = True
                puppetmaster_indent = indent
            continue
        # Inside the puppetmaster entry: a sibling/parent key ends the block.
        if indent <= puppetmaster_indent:
            in_puppetmaster = False
            if stripped.rstrip().rstrip(":") == "puppetmaster":
                in_puppetmaster = True
                puppetmaster_indent = indent
            continue
        match = re.match(r"command\s*:\s*(.+)$", stripped)
        if match:
            return match.group(1).strip().strip("'\"") or None
    return None


def _run_pm(pyexe: str, cwd: str, args: list) -> Optional[subprocess.CompletedProcess]:
    """Run ``<pyexe> -m puppetmaster <args>`` in ``cwd``. Returns None on failure."""
    try:
        return subprocess.run(
            [pyexe, "-m", "puppetmaster", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None


def _extract_json(stdout: str) -> Optional[dict]:
    """Pull the first JSON object out of CLI stdout, tolerating prose lines."""
    if not stdout:
        return None
    text = stdout.strip()
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        loaded = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _query_latest_job_id(pyexe: str, cwd: str) -> Optional[str]:
    """Resolve the latest job id for ``cwd``'s project.

    Prefers ``last --json`` if that flag ever lands; otherwise ``last`` prints
    a bare job id, which is exactly what we need.
    """
    result = _run_pm(pyexe, cwd, ["last", "--json"])
    if result is not None and result.returncode == 0:
        payload = _extract_json(result.stdout)
        if payload:
            job_id = payload.get("id") or payload.get("job_id")
            if isinstance(job_id, str) and job_id:
                return job_id
    result = _run_pm(pyexe, cwd, ["last"])
    if result is None or result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        candidate = line.strip()
        if candidate.startswith("job"):
            return candidate
    first = result.stdout.strip().splitlines()
    return first[0].strip() if first else None


def _query_job_detail(pyexe: str, cwd: str, job_id: str) -> Optional[dict]:
    """Return the status snapshot dict for ``job_id`` (or None).

    Uses ``status <job>`` (full JSON) — NOT ``--compact``, which strips the
    goal/prompt bodies we need. The full snapshot carries ``job.goal``,
    ``job.status``, ``job.completed_at`` and the ``tasks`` list.
    """
    result = _run_pm(pyexe, cwd, ["status", job_id])
    if result is None or result.returncode != 0:
        return None
    payload = _extract_json(result.stdout)
    return _normalize_detail(payload) if payload else None


def _normalize_detail(payload: dict) -> dict:
    """Flatten a status snapshot into the keys the durability heuristic reads."""
    job = payload.get("job") if isinstance(payload.get("job"), dict) else payload
    outcome = payload.get("outcome") if isinstance(payload.get("outcome"), dict) else {}
    tasks = payload.get("tasks") if isinstance(payload.get("tasks"), list) else []
    return {
        "goal": job.get("goal") or "",
        "status": job.get("status") or "",
        "completed_at": job.get("completed_at"),
        "task_count": len(tasks),
        "artifact_count": payload.get("artifact_count") or outcome.get("artifact_count") or 0,
        "patch_artifact_emitted": bool(outcome.get("patch_artifact_emitted")),
    }


def _query_job_summary(pyexe: str, cwd: str, job_id: str) -> str:
    """Best-effort stitched summary markdown for ``job_id`` (empty on failure)."""
    result = _run_pm(pyexe, cwd, ["show", job_id])
    if result is None or result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def _goal_from_summary(summary: str) -> str:
    """Pull the goal out of a stitched summary's ``Goal:`` line (empty if none).

    The stitched summary begins with a ``Goal: <text>`` line; this is the
    fallback when the status snapshot omits the goal.
    """
    for line in (summary or "").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("goal:"):
            return stripped[len("goal:"):].strip()
    return ""


def _is_durable(detail: dict) -> bool:
    """Durable = COMPLETE, finished recently, and produced something concrete."""
    if str(detail.get("status")).lower() != "complete":
        return False
    if not _finished_recently(detail.get("completed_at")):
        return False
    produced_artifact = (
        detail.get("patch_artifact_emitted")
        or int(detail.get("artifact_count") or 0) > 0
        or int(detail.get("task_count") or 0) > 0
    )
    return bool(produced_artifact)


def _finished_recently(completed_at: Optional[str]) -> bool:
    """True when ``completed_at`` is missing or within the recent window.

    A missing timestamp is treated as recent: the job is the project's latest
    COMPLETE swarm, and we'd rather distill a borderline-stale candidate (which
    a human reviews anyway) than silently drop a fresh one over a clock quirk.
    """
    if not completed_at:
        return True
    parsed = _parse_iso(completed_at)
    if parsed is None:
        return True
    now = datetime.now(timezone.utc)
    delta = (now - parsed).total_seconds()
    return -60 <= delta <= _RECENT_WINDOW_SECONDS


def _parse_iso(value: str) -> Optional[datetime]:
    try:
        text = value.strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except (ValueError, AttributeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


# ---------------------------------------------------------------------------
# Pure, testable candidate builder
# ---------------------------------------------------------------------------

def build_skill_candidate(job: dict) -> dict:
    """Build a skill candidate from a job dict. Pure and deterministic.

    ``job`` carries ``id``/``job_id``, ``goal``, an optional stitched
    ``summary``, and an optional ``cwd``. Returns ``{"slug", "skill_md",
    "meta"}`` where ``skill_md`` is a valid SKILL.md (YAML frontmatter +
    body) and ``meta`` records provenance.
    """
    job_id = str(job.get("id") or job.get("job_id") or "unknown")
    goal = (job.get("goal") or "").strip() or "Untitled Puppetmaster swarm"
    summary = (job.get("summary") or "").strip()
    cwd = job.get("cwd") or ""

    slug = _slugify(goal)
    name = slug
    description = _describe(goal)
    created_iso = datetime.now(timezone.utc).isoformat()

    skill_md = _render_skill_md(
        name=name,
        description=description,
        goal=goal,
        summary=summary,
        job_id=job_id,
    )
    meta = {
        "job_id": job_id,
        "goal": goal,
        "cwd": cwd,
        "created_iso": created_iso,
        "source": _SOURCE,
    }
    return {"slug": slug, "skill_md": skill_md, "meta": meta}


def _slugify(text: str, *, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug or "puppetmaster-swarm"


def _describe(goal: str, *, max_len: int = 200) -> str:
    """First sentence/line of the goal, collapsed to a single descriptive line."""
    collapsed = re.sub(r"\s+", " ", goal).strip()
    for terminator in (". ", "\n"):
        idx = collapsed.find(terminator)
        if 0 < idx < max_len:
            collapsed = collapsed[:idx]
            break
    if len(collapsed) > max_len:
        collapsed = collapsed[: max_len - 1].rstrip() + "\u2026"
    return collapsed


def _render_skill_md(*, name: str, description: str, goal: str, summary: str, job_id: str) -> str:
    body_summary = summary if summary else "_No stitched summary was captured for this swarm._"
    provenance = f"Generated by Puppetmaster auto-/learn from job {job_id}"
    safe_description = description.replace('"', "'")
    return (
        "---\n"
        f"name: {name}\n"
        f'description: "{safe_description}"\n'
        "---\n"
        "\n"
        f"# {description}\n"
        "\n"
        "> Skill CANDIDATE auto-distilled by Puppetmaster. Review and edit before promoting\n"
        "> it into a live Hermes skill.\n"
        "\n"
        "## Goal\n"
        "\n"
        f"{goal}\n"
        "\n"
        "## What the swarm found\n"
        "\n"
        f"{body_summary}\n"
        "\n"
        "## Provenance\n"
        "\n"
        f"{provenance}\n"
    )


def _write_candidate(slug: str, skill_md: str, meta: dict, *, home: Path) -> Optional[str]:
    """Write SKILL.md + candidate.json under ``<home>/skills-candidates/``.

    Idempotent: if any existing candidate's ``candidate.json`` already records
    this ``job_id``, we skip and return ``None``. Returns the candidate dir path
    on a fresh write.
    """
    job_id = str(meta.get("job_id") or "unknown")
    base = Path(home).expanduser() / "skills-candidates"

    if _candidate_exists_for_job(base, job_id):
        return None

    date_prefix = datetime.now(timezone.utc).strftime("%Y%m%d")
    target = base / f"{date_prefix}-{slug}"
    try:
        target.mkdir(parents=True, exist_ok=True)
        (target / "SKILL.md").write_text(skill_md, encoding="utf-8")
        (target / "candidate.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError:
        return None
    return str(target)


def _candidate_exists_for_job(base: Path, job_id: str) -> bool:
    if not base.is_dir():
        return False
    for candidate_json in base.glob("*/candidate.json"):
        try:
            existing = json.loads(candidate_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(existing, dict) and str(existing.get("job_id")) == job_id:
            return True
    return False
