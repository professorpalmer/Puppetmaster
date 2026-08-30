"""Fail-closed nested job start when a process is already a worker.

Prompt text (v1.22.10) told spawned agents they were the swarm. They still
ran ``python -m puppetmaster swarm`` via the shell because identity lived
only in the prompt. Stamp ``PUPPETMASTER_WORKER=1`` on worker subprocesses
and refuse job-creating CLI/MCP verbs unless ``PUPPETMASTER_ALLOW_NESTED=1``.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

WORKER_ENV = "PUPPETMASTER_WORKER"
WORKER_VALUE = "1"
ALLOW_NESTED_ENV = "PUPPETMASTER_ALLOW_NESTED"
JOB_ID_ENV = "PUPPETMASTER_JOB_ID"
TASK_ID_ENV = "PUPPETMASTER_TASK_ID"
ROLE_ENV = "PUPPETMASTER_ROLE"

# Commands that create a new orchestrator job. Read-only inspectors
# (status/show/artifacts/codegraph/doctor) stay available so a worker can
# still query its parent job.
JOB_START_COMMANDS = frozenset(
    (
        "run",
        "swarm",
        "review",
        "cursor",
        "claude",
        "openai",
        "codex",
        "hermes",
        "antigravity",
        "agentic",
        "edit",
        "prewalk",
        "browser",
        "research",
        "demo",
        "crash-demo",
        "daemon",
        "rerun",
    )
)

_TRUTHY = frozenset(("1", "true", "yes", "on"))


def _truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


def is_worker_process(env: Optional[Mapping[str, str]] = None) -> bool:
    source = os.environ if env is None else env
    return _truthy(source.get(WORKER_ENV))


def nested_starts_allowed(env: Optional[Mapping[str, str]] = None) -> bool:
    source = os.environ if env is None else env
    return _truthy(source.get(ALLOW_NESTED_ENV))


def stamp_worker_env(
    env: dict,
    *,
    job_id: str = "",
    task_id: str = "",
    role: str = "",
) -> dict:
    """Mark ``env`` as a Puppetmaster worker. Mutates and returns ``env``."""
    env[WORKER_ENV] = WORKER_VALUE
    if job_id:
        env[JOB_ID_ENV] = str(job_id)
    if task_id:
        env[TASK_ID_ENV] = str(task_id)
    if role:
        env[ROLE_ENV] = str(role)
    return env


def nested_start_blocked(
    command: Optional[str],
    env: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Return an error message when ``command`` must not start a nested job."""
    if not command or command not in JOB_START_COMMANDS:
        return None
    if not is_worker_process(env):
        return None
    if nested_starts_allowed(env):
        return None
    return (
        "nested job start refused (PUPPETMASTER_WORKER=1). This process is "
        "already a Puppetmaster worker. Do the work in-process and emit "
        "artifacts; do not start a swarm. Override: PUPPETMASTER_ALLOW_NESTED=1"
    )
