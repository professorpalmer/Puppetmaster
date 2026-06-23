from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def emit_spawn_tree(store, job, artifacts, specs, *, env=None) -> Optional[Path]:
    """Write a completed Puppetmaster job as a Hermes spawn-tree snapshot.

    Hermes reads these files directly from disk for `/agents` history/replay;
    this must stay best-effort so Hermes integration can never affect a swarm.
    """
    try:
        env = env if env is not None else os.environ
        if env.get("PUPPETMASTER_HERMES_SPAWN_TREE") == "0":
            return None

        hermes_home_value = env.get("HERMES_HOME")
        default_home = Path("~/.hermes").expanduser()
        if hermes_home_value:
            hermes_home = Path(hermes_home_value).expanduser()
        elif default_home.exists():
            hermes_home = default_home
        else:
            return None

        session_id = env.get("HERMES_SESSION_ID") or getattr(job, "id", None) or "default"
        safe_session_id = _safe_session_id(str(session_id))
        session_dir = hermes_home / "spawn-trees" / safe_session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        started_at = _timestamp(getattr(job, "created_at", None))
        finished_at = _timestamp(getattr(job, "completed_at", None))
        if finished_at is None:
            finished_at = datetime.now(timezone.utc).timestamp()
        label = (getattr(job, "goal", "") or "")[:80]
        subagents = _subagent_entries(store, job, artifacts, specs)

        ts = datetime.utcfromtimestamp(finished_at).strftime("%Y%m%dT%H%M%S")
        snapshot_path = session_dir / f"{ts}.json"
        snapshot = {
            "session_id": str(session_id),
            "started_at": started_at,
            "finished_at": finished_at,
            "label": label,
            "subagents": subagents,
        }
        snapshot_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")

        index_entry = {
            "path": str(snapshot_path.resolve()),
            "session_id": str(session_id),
            "started_at": started_at,
            "finished_at": finished_at,
            "label": label,
            "count": len(subagents),
        }
        with (session_dir / "_index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(index_entry) + "\n")
        return snapshot_path
    except Exception:
        return None


def _safe_session_id(session_id: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in session_id) or "unknown"


def _subagent_entries(store, job, artifacts, specs) -> list[dict[str, Any]]:
    artifacts_by_task: dict[str, list[Any]] = {}
    for artifact in artifacts or []:
        artifacts_by_task.setdefault(str(getattr(artifact, "task_id", "")), []).append(artifact)

    spec_roles = {getattr(spec, "role", None) for spec in (specs or [])}
    spec_roles.discard(None)
    entries: list[dict[str, Any]] = []
    for task in store.list_tasks(job.id):
        payload = getattr(task, "payload", None) or {}
        if spec_roles and getattr(task, "role", None) not in spec_roles:
            continue
        if payload.get("internal") or str(getattr(task, "role", "")).startswith("_"):
            continue

        task_artifacts = artifacts_by_task.get(str(getattr(task, "id", "")), [])
        entry: dict[str, Any] = {
            "subagent_id": task.id,
            "role": task.role,
            "goal": task.instruction,
        }
        _put_if_present(entry, "status", _verification_status(task, task_artifacts))
        _put_if_present(entry, "summary", _finding_summary(task_artifacts))
        _put_if_present(entry, "model", _routing_model(task_artifacts))
        _put_if_present(entry, "cost_usd", _cost_usd(task, task_artifacts))
        _put_if_present(entry, "input_tokens", _token_count(task_artifacts, "input_tokens", "tokens_in"))
        _put_if_present(entry, "output_tokens", _token_count(task_artifacts, "output_tokens", "tokens_out"))
        _put_if_present(entry, "duration_seconds", _task_duration(task))
        _put_if_present(entry, "files_written", _files_written(task_artifacts))
        entries.append(entry)
    return entries


def _put_if_present(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


def _artifacts_of_type(artifacts: list[Any], artifact_type: str) -> list[Any]:
    return [artifact for artifact in artifacts if str(getattr(artifact, "type", "")) == artifact_type]


def _latest_artifact(artifacts: list[Any]) -> Optional[Any]:
    if not artifacts:
        return None
    return max(artifacts, key=lambda artifact: str(getattr(artifact, "created_at", "")))


def _verification_status(task, artifacts: list[Any]) -> Optional[str]:
    verification = _latest_artifact(_artifacts_of_type(artifacts, "verification"))
    if verification is not None:
        result = (getattr(verification, "payload", None) or {}).get("result")
        if result is not None:
            return str(result)
    status = getattr(task, "status", None)
    return str(status) if status is not None else None


def _finding_summary(artifacts: list[Any]) -> Optional[str]:
    finding = _latest_artifact(_artifacts_of_type(artifacts, "finding"))
    if finding is None:
        return None
    payload = getattr(finding, "payload", None) or {}
    summary = payload.get("claim")
    if summary is None:
        summary = payload.get("report")
    if summary is None:
        return None
    text = summary if isinstance(summary, str) else json.dumps(summary)
    return text if len(text) <= 280 else text[:277] + "..."


def _routing_model(artifacts: list[Any]) -> Optional[str]:
    routing = _latest_artifact(_artifacts_of_type(artifacts, "routing"))
    if routing is None:
        return None
    model = (getattr(routing, "payload", None) or {}).get("model")
    return str(model) if model is not None else None


def _cost_usd(task, artifacts: list[Any]) -> Optional[float]:
    payloads = [getattr(task, "payload", None) or {}]
    payloads.extend(getattr(artifact, "payload", None) or {} for artifact in artifacts)
    for payload in payloads:
        for key in ("cost_usd", "estimated_cost_usd", "router_estimated_cost_usd"):
            value = payload.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _token_count(artifacts: list[Any], *keys: str) -> Optional[int]:
    for artifact in artifacts:
        payload = getattr(artifact, "payload", None) or {}
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        for source in (payload, usage):
            for key in keys:
                value = source.get(key)
                if value is None:
                    continue
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
    return None


def _task_duration(task) -> Optional[float]:
    started_at = _timestamp(getattr(task, "created_at", None))
    finished_at = _timestamp(getattr(task, "completed_at", None))
    if started_at is None or finished_at is None:
        return None
    return max(0.0, finished_at - started_at)


def _files_written(artifacts: list[Any]) -> Optional[list[str]]:
    files: list[str] = []
    for artifact in _artifacts_of_type(artifacts, "patch"):
        value = (getattr(artifact, "payload", None) or {}).get("files")
        if isinstance(value, list):
            files.extend(str(item) for item in value)
    return files or None


def _timestamp(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (TypeError, ValueError):
        return None
