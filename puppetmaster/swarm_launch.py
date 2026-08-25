"""Shared analysis-swarm launch helpers for MCP + CLI.

The daily-driver MCP verb ``puppetmaster_start_swarm`` and the CLI
``python -m puppetmaster swarm`` must build the same worker specs. Agents that
hit ``Tool execution error. Not connected`` should run ONE command — never
hand-author a JSON config or explore ``run --help``. Cursor-specific start
verbs remain available; they are not the default.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from puppetmaster.run_id import reserve_run_logs, write_exclusive_run_text
from puppetmaster.state import state_identity
from puppetmaster.playbooks import recipe_for, stamp_payload
from puppetmaster.workers import (
    ANALYSIS_NO_EDIT_PAYLOAD,
    WorkerSpec,
    default_routing_policy_for_role,
    normalize_role_specs,
)

# Safe CLI / MCP-fallback default: one assignment unless the caller provides
# genuinely distinct structured roles or an explicit workflow config.
DEFAULT_SWARM_ROLES: tuple[str, ...] = ("analysis",)

SWARM_ANALYSIS_ADAPTERS: tuple[str, ...] = (
    "agentic",
    "cursor",
    "local",
    "claude-code",
    "codex",
    "hermes",
    "openai",
    "antigravity",
)


def analysis_swarm_prompt(*, role: str, goal: str) -> str:
    """Build the analysis worker instruction.

    Preserves an explicit ``Acceptance criteria:`` block from ``goal`` so
    harness validation requirements survive Role/Goal wrapping.
    """
    from puppetmaster.acceptance_criteria import (
        ensure_acceptance_criteria_in_text,
        parse_acceptance_criteria_block,
    )

    criteria = parse_acceptance_criteria_block(goal or "")
    body = (
        f"Role: {role}\n"
        f"Goal: {goal}\n\n"
        "Return structured findings with concrete file/function evidence. "
        "Do not modify files unless the user explicitly requested implementation. "
        "Return only Puppetmaster artifact JSON with an artifacts array."
    )
    return ensure_acceptance_criteria_in_text(body, criteria)


def build_analysis_swarm_specs(
    goal: str,
    roles: list[object],
    *,
    adapter: str = "cursor",
    cwd: str = "",
    timeout_seconds: int = 900,
    max_timeout_seconds: Optional[int] = None,
    model: Optional[str] = None,
    auto_route: Optional[bool] = None,
    routing_policy: Optional[str] = None,
    max_cost_usd: Optional[float] = None,
    min_capability: Optional[int] = None,
    required_tags: Optional[list[str]] = None,
    allowed_model_ids: Optional[list[str]] = None,
    disable_memory: bool = True,
    playbook: Optional[str] = None,
) -> list[WorkerSpec]:
    """Build read-only analysis WorkerSpecs for a multi-role swarm."""
    if not playbook:
        from puppetmaster.playbooks import match_playbook

        playbook = match_playbook(goal)
    else:
        playbook = recipe_for(playbook).playbook_id
    if playbook and not roles:
        recipe = recipe_for(playbook)
        if recipe.roles:
            roles = list(recipe.roles)
    if adapter not in SWARM_ANALYSIS_ADAPTERS:
        raise ValueError(
            f"adapter {adapter!r} cannot run an analysis swarm. Supported: "
            f"{', '.join(SWARM_ANALYSIS_ADAPTERS)}."
        )
    role_specs, duplicated_legacy_roles = normalize_role_specs(roles, goal)
    explicit_model = model
    model_name = str(explicit_model or "default")
    if auto_route is not None:
        auto_route_enabled = bool(auto_route)
    else:
        auto_route_enabled = not bool(explicit_model)

    specs: list[WorkerSpec] = []
    for role_spec in role_specs:
        role = role_spec.name
        assignment = role_spec.instruction
        prompt = analysis_swarm_prompt(role=str(role), goal=assignment)
        payload: dict[str, Any] = {
            "prompt": prompt,
            "cwd": cwd or str(Path.cwd()),
            "timeout_seconds": int(timeout_seconds),
            **ANALYSIS_NO_EDIT_PAYLOAD,
        }
        if role_spec.source_scope:
            payload["source_scope"] = list(role_spec.source_scope)
        if role_spec.negative_scope:
            payload["negative_scope"] = list(role_spec.negative_scope)
        if duplicated_legacy_roles:
            payload["duplication_warning"] = {
                "message": "Multiple bare role names received the same goal; provide structured role instructions for decomposition.",
                "fan_out_multiplier": len(role_specs),
            }
        if max_timeout_seconds is not None:
            # Orchestrator._worker_hard_cap honors this; without it the ceiling
            # is 3x the base timeout.
            payload["max_timeout_seconds"] = int(max_timeout_seconds)
        try:
            from puppetmaster.acceptance_criteria import parse_acceptance_criteria_block

            criteria = parse_acceptance_criteria_block(assignment or "")
            if criteria:
                payload["acceptance_criteria"] = criteria
        except Exception:
            pass
        if adapter == "cursor":
            if explicit_model:
                from puppetmaster.model_registry import apply_cursor_model_pin

                payload.update(apply_cursor_model_pin({}, str(explicit_model)))
            else:
                payload["model"] = model_name
        elif adapter == "agentic" and explicit_model:
            from puppetmaster.model_registry import apply_agentic_model_pin

            payload.update(apply_agentic_model_pin({}, str(explicit_model)))
        elif explicit_model:
            payload["model"] = str(explicit_model)
        if auto_route_enabled:
            payload["auto_route"] = True
            # Pin every launch adapter — including cursor. Without this,
            # start_cursor_swarm could hop onto agentic/minimax when vision
            # tags or cursor-cli keys fail, yielding empty unstructured
            # findings while the user asked for a Cursor SDK swarm.
            payload["allowed_adapters"] = [adapter]
            if isinstance(routing_policy, str) and routing_policy:
                payload["routing_policy"] = routing_policy
            else:
                role_policy = default_routing_policy_for_role(str(role))
                if role_policy:
                    payload["routing_policy"] = role_policy
            if max_cost_usd is not None:
                payload["max_cost_usd"] = float(max_cost_usd)
            if min_capability is not None:
                payload["min_capability"] = int(min_capability)
            if required_tags:
                payload["required_tags"] = [
                    str(tag) for tag in required_tags if str(tag).strip()
                ]
            if allowed_model_ids is not None:
                payload["allowed_model_ids"] = list(allowed_model_ids)
        payload["disable_memory"] = not (disable_memory is False)
        if playbook:
            payload = stamp_payload(payload, playbook)
            if playbook == "interrogate" and not (
                isinstance(routing_policy, str) and routing_policy.strip()
            ):
                payload["routing_policy"] = "quality"
        specs.append(
            WorkerSpec(
                role=str(role),
                instruction=prompt,
                adapter=adapter,
                payload=payload,
            )
        )
    return specs


def write_analysis_swarm_config(
    *,
    goal: str,
    roles: list[object],
    adapter: str,
    state_dir: Path,
    cwd: str = "",
    timeout_seconds: int = 900,
    max_timeout_seconds: Optional[int] = None,
    model: Optional[str] = None,
    auto_route: Optional[bool] = None,
    routing_policy: Optional[str] = None,
    max_cost_usd: Optional[float] = None,
    min_capability: Optional[int] = None,
    required_tags: Optional[list[str]] = None,
    allowed_model_ids: Optional[list[str]] = None,
    disable_memory: bool = True,
    lease_seconds: int = 10,
    playbook: Optional[str] = None,
) -> Path:
    """Persist a generated swarm JSON config under ``state_dir/mcp-configs``."""
    specs = build_analysis_swarm_specs(
        goal,
        roles,
        adapter=adapter,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        max_timeout_seconds=max_timeout_seconds,
        model=model,
        auto_route=auto_route,
        routing_policy=routing_policy,
        max_cost_usd=max_cost_usd,
        min_capability=min_capability,
        required_tags=required_tags,
        allowed_model_ids=allowed_model_ids,
        disable_memory=disable_memory,
        playbook=playbook,
    )
    config_dir = Path(state_dir) / "mcp-configs"
    role_specs, duplicated_legacy_roles = normalize_role_specs(roles, goal)
    workers = [
        {
            "role": spec.role,
            "instruction": spec.instruction,
            "adapter": spec.adapter,
            "payload": dict(spec.payload),
            "source_scope": next((r.source_scope for r in role_specs if r.name == spec.role), None),
            "negative_scope": next((r.negative_scope for r in role_specs if r.name == spec.role), None),
        }
        for spec in specs
    ]
    _, config_path = write_exclusive_run_text(
        config_dir,
        "swarm_config",
        json.dumps(
            {
                "lease_seconds": lease_seconds,
                "workers": workers,
                "warnings": ([{"type": "duplicate_legacy_roles", "fan_out_multiplier": len(specs)}]
                              if duplicated_legacy_roles else []),
            },
            indent=2,
        ),
        suffix=".json",
    )
    return config_path


# MCP / detach launchers poll for the early ``job_id:`` line. Five seconds was
# enough on unloaded Linux CI but flake-failed on Windows after a long suite
# (import + SQLite job create > 5s while the child was still healthy). Keep this
# generous — callers return as soon as the line appears.
EARLY_JOB_ID_TIMEOUT_SECONDS = 30.0


def wait_for_job_id(
    stdout_path: Path,
    stderr_path: Path,
    process: subprocess.Popen,
    timeout_seconds: float = EARLY_JOB_ID_TIMEOUT_SECONDS,
) -> str:
    """Poll a launcher stdout log for an early ``job_id:`` line (O(n) total)."""
    deadline = time.monotonic() + timeout_seconds
    pattern = re.compile(r"job_id:\s*(job_[A-Za-z0-9]+)")
    offset = 0
    buffer = ""
    while time.monotonic() < deadline:
        if process.poll() is not None and not stdout_path.exists():
            break
        if stdout_path.exists():
            with stdout_path.open("r", encoding="utf-8") as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset = handle.tell()
            if chunk:
                buffer += chunk
                match = pattern.search(buffer)
                if match:
                    return match.group(1)
            if process.poll() is not None:
                break
        time.sleep(0.05)
    stderr = stderr_path.read_text(encoding="utf-8")[-1000:] if stderr_path.exists() else ""
    stdout_tail = buffer[-500:] if buffer else (
        stdout_path.read_text(encoding="utf-8")[-500:] if stdout_path.exists() else ""
    )
    raise RuntimeError(
        f"started Puppetmaster process but did not receive early job_id; "
        f"pid={process.pid}; returncode={process.poll()}; "
        f"stderr={stderr}; stdout_tail={stdout_tail!r}"
    )


def terminate_launcher_tree(process: subprocess.Popen) -> None:
    """Terminate a detached launcher's exact process tree on early failure."""
    try:
        if os.name == "nt":
            from puppetmaster.win_process import kill_process_tree

            if process.pid and kill_process_tree(process.pid):
                return
    except Exception:
        pass
    try:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
    except (OSError, ProcessLookupError):
        pass


def detach_analysis_swarm(
    *,
    goal: str,
    roles: list[object],
    adapter: str,
    state_dir: Path,
    cwd: str,
    timeout_seconds: int = 900,
    max_timeout_seconds: Optional[int] = None,
    model: Optional[str] = None,
    auto_route: Optional[bool] = None,
    routing_policy: Optional[str] = None,
    max_cost_usd: Optional[float] = None,
    min_capability: Optional[int] = None,
    required_tags: Optional[list[str]] = None,
    allowed_model_ids: Optional[list[str]] = None,
    disable_memory: bool = True,
    label: Optional[str] = None,
    worker_mode: str = "subprocess",
    backend: str = "sqlite",
    job_id_timeout_seconds: float = EARLY_JOB_ID_TIMEOUT_SECONDS,
    launch_key: Optional[str] = None,
    playbook: Optional[str] = None,
) -> dict[str, Any]:
    """Write config, spawn ``run --config`` detached, return ``{job_id, ...}``."""
    _normalized_roles, duplicated_legacy_roles = normalize_role_specs(roles, goal)
    config_path = write_analysis_swarm_config(
        goal=goal,
        roles=roles,
        adapter=adapter,
        state_dir=state_dir,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        max_timeout_seconds=max_timeout_seconds,
        model=model,
        auto_route=auto_route,
        routing_policy=routing_policy,
        max_cost_usd=max_cost_usd,
        min_capability=min_capability,
        required_tags=required_tags,
        allowed_model_ids=allowed_model_ids,
        disable_memory=disable_memory,
        playbook=playbook,
    )
    run_dir = Path(state_dir) / "mcp-runs"
    run_id, stdout_path, stderr_path, stdout_handle, stderr_handle = reserve_run_logs(
        run_dir, "swarm"
    )
    goal_path = run_dir / f"{run_id}.goal"
    goal_path.write_text(goal, encoding="utf-8")
    full_command = [
        sys.executable,
        "-u",
        "-m",
        "puppetmaster",
        "--state-dir",
        str(state_dir),
        "--backend",
        backend,
        "--emit-job-id-early",
        "run",
        "--goal-file",
        str(goal_path),
        "--config",
        str(config_path),
        "--worker-mode",
        worker_mode,
    ]
    if disable_memory:
        full_command.append("--disable-memory")
    else:
        full_command.append("--enable-memory")
    if label:
        full_command.extend(["--label", label])
    if launch_key:
        full_command.extend(["--launch-key", str(launch_key)])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    source_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else source_root
    )

    try:
        process = subprocess.Popen(
            full_command,
            cwd=cwd or str(Path.cwd()),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            start_new_session=True,
        )
    except OSError:
        stdout_handle.close()
        stderr_handle.close()
        raise
    stdout_handle.close()
    stderr_handle.close()
    try:
        job_id = wait_for_job_id(
            stdout_path, stderr_path, process, timeout_seconds=job_id_timeout_seconds
        )
    except BaseException:
        terminate_launcher_tree(process)
        raise
    job_ref = {
        "job_id": job_id,
        "state_id": state_identity(state_dir),
    }
    body = {
        "ok": True,
        "job_id": job_id,
        "job_ref": job_ref,
        "monitor_with": {
            "tool": "puppetmaster_live_artifacts_follow",
            "job_id": job_id,
            "backend": backend,
            "state_id": state_identity(state_dir),
            "initial_cursor": 0,
            "arguments": {
                "job_ref": job_ref,
                "backend": backend,
                "since_cursor": 0,
                "timeout_seconds": 10,
            },
        },
        "run_id": run_id,
        "orchestrator_pid": process.pid,
        "launcher_pid": process.pid,
        "config": str(config_path),
        "cwd": cwd or str(Path.cwd()),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "next_steps": [
            f"python -m puppetmaster status {job_id}",
            f"python -m puppetmaster feed {job_id} --follow",
            f"python -m puppetmaster show {job_id}",
        ],
    }
    if duplicated_legacy_roles:
        body["warnings"] = [
            {
                "kind": "duplicate_legacy_roles",
                "fan_out_multiplier": len(_normalized_roles),
                "message": (
                    "Multiple bare role names received the same goal; use "
                    "structured role instructions for real decomposition."
                ),
            }
        ]
    return body
