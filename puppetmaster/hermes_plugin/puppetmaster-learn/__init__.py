"""puppetmaster-learn: turn a finished Puppetmaster swarm into a Hermes skill candidate.

When a Hermes session ends (``on_session_finalize``), the plugin reads the
session's working directory and start time from the Hermes session store and
asks the Puppetmaster CLI for the latest job of that directory. If that job
completed during the session and produced findings or a patch, the plugin
writes a skill CANDIDATE under ``<hermes home>/skills-candidates/`` and tells
the user where it is. It never writes a live skill and never starts an agent.

The plugin runs in the Hermes interpreter and does not import ``puppetmaster``.
It runs the Puppetmaster CLI as a child process instead.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SOURCE = "puppetmaster-auto-learn"
_CLI_TIMEOUT_SECONDS = 15

# Fallback for a session with no stored row: jobs that completed before this
# time belong to an earlier session. It starts at plugin load and moves to
# "now" after each finalize.
_session_started = time.time()


def register(ctx) -> None:
    ctx.register_hook("on_session_finalize", _on_session_finalize)


def _on_session_finalize(session_id=None, platform=None, **kwargs) -> Optional[str]:
    """Write a candidate for this session's durable swarm. Never raises."""
    global _session_started
    fallback_since, _session_started = _session_started, time.time()
    try:
        cwd, since = _session_scope(session_id, fallback_since)
        path = distill_latest_job(cwd, since=since)
    except Exception:
        logger.debug("puppetmaster-learn: distillation failed", exc_info=True)
        return None
    if path is None:
        return None
    slug = Path(path).name
    return (
        f"Puppetmaster saved a skill candidate from this session's swarm: {path}\n"
        f"Review it, then promote it with `puppetmaster skills promote-candidate {slug}`."
    )


def distill_latest_job(cwd: str, *, since: float) -> Optional[str]:
    """Write a candidate for the latest durable job in ``cwd``. Returns its dir or None."""
    command = _puppetmaster_command()
    if command is None:
        return None
    latest = _run_json(command, cwd, ["last", "--json"])
    if not latest or str(latest.get("status")).lower() != "complete":
        return None
    job_id = latest.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        return None
    # Newer Puppetmaster puts completed_at in `last --json`, which saves the
    # `status` call for a job from an earlier session.
    if "completed_at" in latest and not _completed_since(latest["completed_at"], since):
        return None
    home = _hermes_home()
    if _candidate_exists_for_job(home / "skills-candidates", job_id):
        return None

    snapshot = _run_json(command, cwd, ["status", job_id])
    if not snapshot:
        return None
    job = snapshot.get("job") if isinstance(snapshot.get("job"), dict) else {}
    outcome = snapshot.get("outcome") if isinstance(snapshot.get("outcome"), dict) else {}
    produced = int(latest.get("finding_count") or 0) > 0 or bool(outcome.get("patch_artifact_emitted"))
    if not produced or not _completed_since(job.get("completed_at"), since):
        return None

    findings = _findings(_run_json_list(command, cwd, ["artifacts", job_id]))
    candidate = build_skill_candidate(
        {"id": job_id, "goal": job.get("goal") or "", "findings": findings, "cwd": cwd}
    )
    path = _write_candidate(candidate["slug"], candidate["skill_md"], candidate["meta"], home=home)
    if path is not None:
        logger.info("puppetmaster-learn: wrote skill candidate %s from job %s", path, job_id)
    return path


def _session_scope(session_id, fallback_since: float) -> tuple:
    """Return ``(cwd, since)`` for the session that ends.

    A Hermes gateway runs the hook for many sessions, so its own cwd is not the
    session's. The session row in the Hermes session store has the session's
    cwd and start time. The process cwd and ``fallback_since`` are the fallback.
    """
    row = _session_row(session_id)
    cwd = row.get("cwd")
    if not isinstance(cwd, str) or not os.path.isdir(cwd):
        cwd = os.environ.get("TERMINAL_CWD") or os.getcwd()
    started = row.get("started_at")
    since = float(started) if isinstance(started, (int, float)) else fallback_since
    return cwd, since


def _session_row(session_id) -> dict:
    if not session_id:
        return {}
    try:
        from hermes_state import SessionDB

        db = SessionDB(read_only=True)
    except Exception:
        return {}
    try:
        row = db.get_session(str(session_id))
    except Exception:
        row = None
    finally:
        db.close()
    return row if isinstance(row, dict) else {}


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def _puppetmaster_command() -> Optional[list]:
    """Return the argv prefix that runs the Puppetmaster CLI, or None.

    The ``puppetmaster`` executable on PATH comes first. When Hermes runs
    with a short PATH (a desktop launch), fall back to the interpreter in the
    ``mcp_servers.puppetmaster`` entry that ``puppetmaster setup`` writes, if
    that file still exists.
    """
    found = shutil.which("puppetmaster")
    if found:
        return [found]
    try:
        from hermes_cli.config import load_config_readonly

        entry = (load_config_readonly().get("mcp_servers") or {}).get("puppetmaster") or {}
    except Exception:
        return None
    command = entry.get("command") if isinstance(entry, dict) else None
    args = entry.get("args") if isinstance(entry, dict) else None
    if not isinstance(command, str) or not Path(command).is_file():
        return None
    if isinstance(args, list) and args[:2] == ["-m", "puppetmaster"]:
        return [command, "-m", "puppetmaster"]
    if Path(command).name.startswith("puppetmaster"):
        return [command]
    return None


def _run_text(command: list, cwd: str, args: list) -> str:
    try:
        result = subprocess.run(
            [*command, *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_CLI_TIMEOUT_SECONDS,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.TimeoutExpired, OSError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _run_json(command: list, cwd: str, args: list) -> Optional[dict]:
    text = _run_text(command, cwd, args)
    if not text:
        return None
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _run_json_list(command: list, cwd: str, args: list) -> list:
    text = _run_text(command, cwd, args)
    try:
        loaded = json.loads(text) if text else []
    except json.JSONDecodeError:
        return []
    return loaded if isinstance(loaded, list) else []


_FENCED_JSON = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)
_MAX_FINDINGS = 12
_MAX_DETAIL = 1200
_FINDING_KINDS = {"FINDING", "RISK"}


def _findings(artifacts: list) -> list:
    """Findings from the job's ``finding`` artifacts, as dicts.

    A worker report can hold a JSON block with the findings themselves, in the
    Puppetmaster shape (``{"artifacts": [{"title", "body"}]}``) or as a list
    under ``content``. Use those items. Otherwise use the artifact's claim.
    """
    found = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("type") != "finding":
            continue
        payload = artifact.get("payload") if isinstance(artifact.get("payload"), dict) else {}
        items = _report_items(str(payload.get("report") or ""))
        if not items and payload.get("claim"):
            items = [{"title": str(payload["claim"])}]
        found.extend(items)
    return found[:_MAX_FINDINGS]


def _report_items(report: str) -> list:
    items = []
    for block in _FENCED_JSON.findall(report):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        groups = data.get("artifacts") if isinstance(data, dict) else None
        for group in groups if isinstance(groups, list) else []:
            if not isinstance(group, dict):
                continue
            content = group.get("content")
            for item in content if isinstance(content, list) else [group]:
                if not isinstance(item, dict) or not (item.get("title") or item.get("summary")):
                    continue
                if str(item.get("kind") or "FINDING").upper() in _FINDING_KINDS:
                    items.append(item)
    return items


def _completed_since(completed_at: object, since: float) -> bool:
    if not isinstance(completed_at, str):
        return False
    try:
        parsed = datetime.fromisoformat(completed_at.strip().replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp() >= since


def build_skill_candidate(job: dict) -> dict:
    """Build ``{"slug", "skill_md", "meta"}`` from a job dict. Pure."""
    job_id = str(job.get("id") or job.get("job_id") or "unknown")
    goal = (job.get("goal") or "").strip() or "Untitled Puppetmaster swarm"
    findings = [f for f in job.get("findings") or [] if isinstance(f, dict)]
    slug = _slugify(goal)
    description = _describe(goal)
    skill_md = _render_skill_md(
        name=slug, description=description, goal=goal, findings=findings, job_id=job_id
    )
    meta = {
        "job_id": job_id,
        "goal": goal,
        "cwd": job.get("cwd") or "",
        "created_iso": datetime.now(timezone.utc).isoformat(),
        "source": _SOURCE,
    }
    return {"slug": slug, "skill_md": skill_md, "meta": meta}


def _slugify(text: str, *, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug or "puppetmaster-swarm"


def _describe(goal: str, *, max_len: int = 200) -> str:
    """The goal's first sentence on one line, at most ``max_len`` characters."""
    collapsed = re.sub(r"\s+", " ", goal).strip()
    end = collapsed.find(". ")
    if 0 < end < max_len:
        collapsed = collapsed[:end]
    if len(collapsed) > max_len:
        collapsed = collapsed[: max_len - 1].rstrip() + "…"
    return collapsed


def _render_finding(index: int, finding: dict) -> str:
    title = re.sub(r"\s+", " ", str(finding.get("title") or finding.get("summary"))).strip()
    where = finding.get("file")
    if where and finding.get("line"):
        where = f"{where}:{finding['line']}"
    tags = ", ".join(str(v) for v in (f"`{where}`" if where else None, finding.get("severity")) if v)
    lines = [f"{index}. **{title}**" + (f" ({tags})" if tags else "")]
    details = [("", finding.get("body")), ("Failure: ", finding.get("failure_scenario")),
               ("Evidence: ", finding.get("evidence"))]
    for label, text in details:
        text = str(text or "").strip()
        if text:
            if len(text) > _MAX_DETAIL:
                text = text[: _MAX_DETAIL - 1].rstrip() + "…"
            lines.append("   " + (label + text).replace("\n", "\n   "))
    return "\n".join(line.rstrip() for line in "\n".join(lines).split("\n"))


def _render_skill_md(*, name: str, description: str, goal: str, findings: list, job_id: str) -> str:
    body_summary = "\n\n".join(
        _render_finding(i, f) for i, f in enumerate(findings, 1)
    ) or "_The swarm stored no finding text._"
    # A JSON string is a valid YAML double-quoted scalar, so any goal text is safe here.
    return (
        "---\n"
        f"name: {name}\n"
        f"description: {json.dumps(description, ensure_ascii=False)}\n"
        "---\n"
        "\n"
        f"# {description}\n"
        "\n"
        "> Skill CANDIDATE distilled by Puppetmaster. Review and edit it before you\n"
        "> promote it to a live Hermes skill.\n"
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
        f"Puppetmaster job {job_id}\n"
    )


def _write_candidate(slug: str, skill_md: str, meta: dict, *, home: Path) -> Optional[str]:
    """Write SKILL.md + candidate.json under ``<home>/skills-candidates/``.

    Returns None when a candidate for this job already exists or the write fails.
    """
    base = Path(home).expanduser() / "skills-candidates"
    if _candidate_exists_for_job(base, str(meta.get("job_id") or "unknown")):
        return None
    target = base / f"{datetime.now(timezone.utc):%Y%m%d}-{slug}"
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
