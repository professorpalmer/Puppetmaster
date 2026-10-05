"""Cross-process lease on an in-place provider session.

``codex exec resume`` continues a thread in place, so two workers resuming the
same thread interleave turns into one conversation. The per-job guard in
:func:`puppetmaster.worker_resume.claim_resumed_session` cannot see other jobs,
which may run in other processes with other state directories. This lease is
an OS file lock beside the provider's own session store: it is shared by every
job using that store and released by the kernel if the holder dies.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def codex_thread_lock_path(session_id: str) -> Path:
    root = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    return root / "puppetmaster-thread-locks" / f"{session_id}.lock"


class SessionLease:
    """A held, non-blocking exclusive lock; call :meth:`release` when the run ends."""

    def __init__(self, fd: int) -> None:
        self._fd: Optional[int] = fd

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            _unlock(fd)
        finally:
            os.close(fd)


def acquire_codex_thread(session_id: str) -> Optional[SessionLease]:
    """Lease the codex thread, or None when another live worker holds it.

    Raises ``OSError`` only when the lock file cannot be created at all.
    """
    if not _SAFE_ID.match(session_id):
        raise OSError(f"unsafe codex thread id {session_id!r}")
    path = codex_thread_lock_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if _try_lock(fd):
            return SessionLease(fd)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    return None


if os.name == "nt":  # pragma: no cover - exercised on Windows CI
    import msvcrt

    def _try_lock(fd: int) -> bool:
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import errno
    import fcntl

    def _try_lock(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                return False
            raise

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)
