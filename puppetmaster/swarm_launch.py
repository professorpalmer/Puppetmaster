"""Shared analysis-swarm launch helpers for MCP + CLI.

The daily-driver MCP verb ``puppetmaster_start_cursor_swarm`` and the CLI
``python -m puppetmaster swarm`` must build the same worker specs. Agents that
hit ``Tool execution error. Not connected`` should run ONE command — never
hand-author a JSON config or explore ``run --help``.
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

from puppetmaster.workers import (
    ANALYSIS_NO_EDIT_PAYLOAD,
    WorkerSpec,
    default_routing_policy_for_role,
)

# Sensible CLI / MCP-fallback defaults. MCP ``start_cursor_swarm`` keeps its
# historical role list when the caller omits ``roles``; CLI ``swarm`` uses this
# shorter audit trio so a one-liner matches how operators actually peel.
DEFAULT_SWARM_ROLES: tuple[str, ...] = ("explore", "audit", "review")

SWARM_ANALYSIS_ADAPTERS: tuple[str, ...] = (
    "agentic",
    "cursor",
    "local",
    "claude-code",
    "codex",
    "hermes",
    "openai",
)


def analysis_swarm_prompt(*, role: str, goal: str) -> str:
    return (
        f"Role: {role}\n"
        f"Goal: {goal}\n\n"
        "Return structured findings with concrete file/function evidence. "
        "Do not modify files unless the user explicitly requested implementation. "
        "Return only Puppetmaster artifact JSON with an artifacts array."
    )


def build_analysis_swarm_specs(
    goal: str,
    roles: list[str],
    *,
    adapter: str = "cursor",
    cwd: str = "",
    timeout_seconds: int = 900,
    model: Optional[str] = None,
    auto_route: Optional[bool] = None,
    routing_policy: Optional[str] = None,
    max_cost_usd: Optional[float] = None,
    min_capability: Optional[int] = None,
    required_tags: Optional[list[str]] = None,
    allowed_model_ids: Optional[list[str]] = None,
    disable_memory: bool = True,
) -> list[WorkerSpec]:
    """Build read-only analysis WorkerSpecs for a multi-role swarm."""
    if adapter not in SWARM_ANALYSIS_ADAPTERS:
        raise ValueError(
            f"adapter {adapter!r} cannot run an analysis swarm. Supported: "
            f"{', '.join(SWARM_ANALYSIS_ADAPTERS)}."
        )
    if not roles:
        roles = list(DEFAULT_SWARM_ROLES)
    explicit_model = model
    model_name = str(explicit_model or "default")
    if auto_route is not None:
        auto_route_enabled = bool(auto_route)
    else:
        auto_route_enabled = not bool(explicit_model)

    specs: list[WorkerSpec] = []
    for role in roles:
        prompt = analysis_swarm_prompt(role=str(role), goal=goal)
        payload: dict[str, Any] = {
            "prompt": prompt,
            "cwd": cwd or str(Path.cwd()),
            "timeout_seconds": int(timeout_seconds),
            **ANALYSIS_NO_EDIT_PAYLOAD,
        }
        if adapter == "cursor":
            if explicit_model:
                from puppetmaster.model_registry import apply_cursor_model_pin

                payload.update(apply_cursor_model_pin({}, str(explicit_model)))
            else:
                payload["model"] = model_name
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
    roles: list[str],
    adapter: str,
    state_dir: Path,
    cwd: str = "",
    timeout_seconds: int = 900,
    model: Optional[str] = None,
    auto_route: Optional[bool] = None,
    routing_policy: Optional[str] = None,
    max_cost_usd: Optional[float] = None,
    min_capability: Optional[int] = None,
    required_tags: Optional[list[str]] = None,
    allowed_model_ids: Optional[list[str]] = None,
    disable_memory: bool = True,
    lease_seconds: int = 10,
) -> Path:
    """Persist a generated swarm JSON config under ``state_dir/mcp-configs``."""
    specs = build_analysis_swarm_specs(
        goal,
        roles,
        adapter=adapter,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        model=model,
        auto_route=auto_route,
        routing_policy=routing_policy,
        max_cost_usd=max_cost_usd,
        min_capability=min_capability,
        required_tags=required_tags,
        allowed_model_ids=allowed_model_ids,
        disable_memory=disable_memory,
    )
    config_dir = Path(state_dir) / "mcp-configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"swarm_{int(time.time() * 1000)}_{os.getpid()}.json"
    workers = [
        {
            "role": spec.role,
            "instruction": spec.instruction,
            "adapter": spec.adapter,
            "payload": dict(spec.payload),
        }
        for spec in specs
    ]
    config_path.write_text(
        json.dumps({"lease_seconds": lease_seconds, "workers": workers}, indent=2),
        encoding="utf-8",
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


def detach_analysis_swarm(
    *,
    goal: str,
    roles: list[str],
    adapter: str,
    state_dir: Path,
    cwd: str,
    timeout_seconds: int = 900,
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
) -> dict[str, Any]:
    """Write config, spawn ``run --config`` detached, return ``{job_id, ...}``."""
    config_path = write_analysis_swarm_config(
        goal=goal,
        roles=roles,
        adapter=adapter,
        state_dir=state_dir,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        model=model,
        auto_route=auto_route,
        routing_policy=routing_policy,
        max_cost_usd=max_cost_usd,
        min_capability=min_capability,
        required_tags=required_tags,
        allowed_model_ids=allowed_model_ids,
        disable_memory=disable_memory,
    )
    run_dir = Path(state_dir) / "mcp-runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"swarm_{int(time.time() * 1000)}_{os.getpid()}"
    stdout_path = run_dir / f"{run_id}.stdout.log"
    stderr_path = run_dir / f"{run_id}.stderr.log"
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
        goal,
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

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    source_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else source_root
    )

    stdout_handle = stdout_path.open("w", encoding="utf-8")
    try:
        stderr_handle = stderr_path.open("w", encoding="utf-8")
    except OSError:
        stdout_handle.close()
        raise
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
        try:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        except (OSError, ProcessLookupError):
            pass
        raise
    return {
        "ok": True,
        "job_id": job_id,
        "run_id": run_id,
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
