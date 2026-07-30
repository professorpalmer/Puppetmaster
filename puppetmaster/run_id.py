"""Collision-proof run ids for detached MCP / swarm launch log paths.

Stdlib-only and intentionally free of heavier puppetmaster imports so
``mcp_server`` and ``swarm_launch`` can share the helper without cycles.
"""

from __future__ import annotations

import itertools
import os
import time
from pathlib import Path
from typing import IO, Optional, Tuple
from uuid import uuid4

# Process-local monotonic component. ``itertools.count`` next() is atomic
# under CPython's GIL; combined with uuid entropy this stays unique even when
# ``time.time()`` and ``os.getpid()`` collide across concurrent callers in the
# same process (the historical ms+pid-only form). Cross-process uniqueness
# still holds when ms+pid are adversarially identical because of the random hex.
_RUN_ID_SEQ = itertools.count()

# Exclusive create retries when a reserved path already exists — should be
# vanishingly rare given seq+entropy, but fail closed instead of truncating.
_MAX_RESERVE_ATTEMPTS = 8


def new_run_id(prefix: str) -> str:
    """Build a collision-proof id for detached launch log / config paths.

    Historical shape was ``{prefix}_{ms}_{pid}``, which collides when two
    tool handlers in the same process fire in the same millisecond — both
    open the same stdout log, and ``wait_for_job_id`` collapses the launches
    onto one job_id. Keep the readable prefix + ms + pid, then append a
    monotonic sequence and random hex so concurrent and cross-process callers
    never share state paths.
    """
    return (
        f"{prefix}_{int(time.time() * 1000)}_{os.getpid()}"
        f"_{next(_RUN_ID_SEQ):x}_{uuid4().hex[:8]}"
    )


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def reserve_run_path(
    directory: Path,
    prefix: str,
    *,
    suffix: str,
    encoding: str = "utf-8",
    max_attempts: int = _MAX_RESERVE_ATTEMPTS,
) -> Tuple[str, Path, IO[str]]:
    """Reserve a unique path under ``directory`` with exclusive create.

    Returns ``(run_id, path, handle)``. On an unexpected path collision,
    regenerates the run id instead of opening with ``'w'`` / ``write_text``
    (which would overwrite another launch's file). Caller owns the handle.
    """
    directory.mkdir(parents=True, exist_ok=True)
    last_error: Optional[OSError] = None
    for _ in range(max(1, int(max_attempts))):
        run_id = new_run_id(prefix)
        path = directory / f"{run_id}{suffix}"
        try:
            handle = path.open("x", encoding=encoding)
        except FileExistsError as exc:
            last_error = exc
            continue
        return run_id, path, handle
    raise FileExistsError(
        f"could not reserve exclusive {prefix!r} path under {directory} "
        f"after {max_attempts} attempts"
    ) from last_error


def write_exclusive_run_text(
    directory: Path,
    prefix: str,
    text: str,
    *,
    suffix: str,
    encoding: str = "utf-8",
    max_attempts: int = _MAX_RESERVE_ATTEMPTS,
) -> Tuple[str, Path]:
    """Create a unique run file exclusively, write ``text``, clean up on failure.

    Returns ``(run_id, path)``. Partial files are unlinked if the write fails
    after exclusive create succeeded.
    """
    run_id, path, handle = reserve_run_path(
        directory,
        prefix,
        suffix=suffix,
        encoding=encoding,
        max_attempts=max_attempts,
    )
    try:
        handle.write(text)
    except BaseException:
        _unlink_quiet(path)
        raise
    finally:
        handle.close()
    return run_id, path


def reserve_run_logs(
    run_dir: Path,
    prefix: str,
    *,
    max_attempts: int = _MAX_RESERVE_ATTEMPTS,
) -> Tuple[str, Path, Path, IO[str], IO[str]]:
    """Reserve unique stdout/stderr log paths with exclusive create.

    Returns ``(run_id, stdout_path, stderr_path, stdout_handle, stderr_handle)``.
    On an unexpected path collision, regenerates the run id instead of opening
    with ``'w'`` (which would truncate another launch's logs).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    last_error: Optional[OSError] = None
    for _ in range(max(1, int(max_attempts))):
        run_id = new_run_id(prefix)
        stdout_path = run_dir / f"{run_id}.stdout.log"
        stderr_path = run_dir / f"{run_id}.stderr.log"
        try:
            stdout_handle = stdout_path.open("x", encoding="utf-8")
        except FileExistsError as exc:
            last_error = exc
            continue
        try:
            stderr_handle = stderr_path.open("x", encoding="utf-8")
        except FileExistsError as exc:
            stdout_handle.close()
            _unlink_quiet(stdout_path)
            last_error = exc
            continue
        except OSError:
            stdout_handle.close()
            _unlink_quiet(stdout_path)
            raise
        return run_id, stdout_path, stderr_path, stdout_handle, stderr_handle
    raise FileExistsError(
        f"could not reserve exclusive {prefix!r} log paths under {run_dir} "
        f"after {max_attempts} attempts"
    ) from last_error
