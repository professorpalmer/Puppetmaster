"""Small, dependency-free interprocess locks for shared local state.

``InterProcessFileLock.for_target(path)`` serializes writers for one target
file or directory.  It uses an adjacent ``.<name>.lock`` file, so independent
targets never contend.  Lock ownership is recorded with a PID and a random
token.  A later process recovers an orphan whose owner PID is no longer alive;
malformed/unknown owners are recovered after ``stale_after`` seconds.

The lock is advisory: every writer of a resource must use the same target.
It is deliberately suitable for both files (policy documents) and directories
(future workspace and dashboard leases).  Acquire it with a context manager:

    with InterProcessFileLock.for_target(target):
        ... read / modify / atomically replace target ...

Release removes the lock only when the random token still matches.  Acquisition
raises :class:`TimeoutError` rather than silently proceeding without a lock.
"""
from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path
from typing import Optional


_DEFAULT_TIMEOUT_SECONDS = 10.0
_DEFAULT_STALE_AFTER_SECONDS = 120.0
_POLL_SECONDS = 0.05


def _lock_path_for(target: Path) -> Path:
    return target.parent / ("." + target.name + ".lock")


def _pid_is_alive(pid: int) -> bool:
    """Delegate to the shared Windows-safe probe after imports settle.

    A top-level ``liveness`` import cycles through ``store`` →
    ``fs_permissions`` → this module.
    """
    from puppetmaster.liveness import _pid_alive

    return _pid_alive(pid)


class InterProcessFileLock:
    """An exclusive, stale-recovering lock adjacent to a target path.

    Use :meth:`for_target` rather than inventing a lock-file name.  The target
    itself need not exist.  ``timeout`` bounds waiting for an active peer;
    ``stale_after`` is only used when an owner cannot be identified safely.
    """

    def __init__(
        self,
        target: Path,
        *,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        stale_after: float = _DEFAULT_STALE_AFTER_SECONDS,
    ) -> None:
        self.target = Path(target)
        self.path = _lock_path_for(self.target)
        self.timeout = timeout
        self.stale_after = stale_after
        self._token: Optional[str] = None

    @classmethod
    def for_target(
        cls,
        target: Path,
        *,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        stale_after: float = _DEFAULT_STALE_AFTER_SECONDS,
    ) -> "InterProcessFileLock":
        """Create the standard per-target writer lock.

        The caller must keep the returned context open over its entire
        read-modify-write operation, not merely the final write.
        """
        return cls(target, timeout=timeout, stale_after=stale_after)

    def __enter__(self) -> "InterProcessFileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()

    def acquire(self) -> None:
        if self._token is not None:
            raise RuntimeError("interprocess lock is already held by this instance")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        token = secrets.token_hex(16)
        payload = json.dumps(
            {"pid": os.getpid(), "created_at": time.time(), "token": token},
            separators=(",", ":"),
        ).encode("utf-8")
        while True:
            try:
                fd = os.open(
                    self.path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600 if os.name != "nt" else 0o666,
                )
            except FileExistsError:
                self._recover_stale_owner()
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "timed out acquiring interprocess lock for " + str(self.target)
                    )
                time.sleep(_POLL_SECONDS)
                continue
            try:
                with os.fdopen(fd, "wb", closefd=False) as handle:
                    handle.write(payload)
                    handle.flush()
                    try:
                        os.fsync(handle.fileno())
                    except OSError:
                        # Some Windows/filesystem combinations do not expose
                        # fsync for this tiny advisory file. O_EXCL remains the
                        # correctness primitive; durability is best effort.
                        pass
            finally:
                os.close(fd)
            self._token = token
            return

    def release(self) -> None:
        token = self._token
        self._token = None
        if token is None:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(data, dict) and data.get("token") == token:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    def _recover_stale_owner(self) -> None:
        """Remove only an orphaned lock, with a guard against reclaim races."""
        reclaim_path = self.path.with_name(self.path.name + ".reclaim")
        try:
            reclaim_fd = os.open(
                reclaim_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600 if os.name != "nt" else 0o666,
            )
        except FileExistsError:
            return
        try:
            os.close(reclaim_fd)
            try:
                raw = self.path.read_text(encoding="utf-8")
                data = json.loads(raw)
            except (OSError, json.JSONDecodeError):
                data = None
            now = time.time()
            pid = data.get("pid") if isinstance(data, dict) else None
            created_at = data.get("created_at") if isinstance(data, dict) else None
            if not isinstance(created_at, (int, float)):
                try:
                    created_at = self.path.stat().st_mtime
                except OSError:
                    created_at = None
            owner_alive = isinstance(pid, int) and _pid_is_alive(pid)
            age = now - created_at if isinstance(created_at, (int, float)) else None
            recoverable = (isinstance(pid, int) and not owner_alive) or (
                not isinstance(pid, int) and age is not None and age >= self.stale_after
            )
            if recoverable:
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
        finally:
            try:
                reclaim_path.unlink()
            except FileNotFoundError:
                pass
