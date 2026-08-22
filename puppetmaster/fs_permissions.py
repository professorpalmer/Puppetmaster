"""Owner-only filesystem permissions for sensitive Puppetmaster state.

On POSIX, directories are created as ``0700`` and files as ``0600``. Umask can
strip mode bits from ``mkdir``, so we always follow up with an explicit
``chmod``. On Windows these calls are best-effort no-ops so callers never crash.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from puppetmaster.interprocess_lock import InterProcessFileLock

_DIR_MODE = 0o700
_FILE_MODE = 0o600


def supports_posix_modes() -> bool:
    return os.name != "nt" and hasattr(os, "chmod")


def chmod_private_dir(path: os.PathLike[str] | str) -> None:
    if not supports_posix_modes():
        return
    try:
        os.chmod(path, _DIR_MODE)
    except OSError:
        pass


def chmod_private_file(path: os.PathLike[str] | str) -> None:
    if not supports_posix_modes():
        return
    try:
        os.chmod(path, _FILE_MODE)
    except OSError:
        pass


def mkdir_private(path: Path, *, parents: bool = True, exist_ok: bool = True) -> None:
    if supports_posix_modes():
        path.mkdir(mode=_DIR_MODE, parents=parents, exist_ok=exist_ok)
        chmod_private_dir(path)
    else:
        path.mkdir(parents=parents, exist_ok=exist_ok)


def open_private(path: Path, flags: int) -> int:
    """Open ``path``, creating it with owner-only mode when POSIX modes apply."""
    if supports_posix_modes():
        return os.open(path, flags, _FILE_MODE)
    return os.open(path, flags)


def write_private_text(
    path: Path,
    content: str,
    *,
    encoding: str = "utf-8",
    lock: bool = True,
) -> None:
    """Atomically replace private text, serializing writers for ``path``.

    The replacement file is written and synced in the destination directory
    before ``os.replace`` makes it visible.  Pass ``lock=False`` only while a
    caller already holds :meth:`InterProcessFileLock.for_target` over a larger
    read-modify-write transaction.
    """
    if lock:
        with InterProcessFileLock.for_target(path):
            write_private_text(path, content, encoding=encoding, lock=False)
        return
    mkdir_private(path.parent)
    temp = path.parent / ("." + path.name + "." + secrets.token_hex(12) + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if supports_posix_modes():
        fd = os.open(temp, flags, _FILE_MODE)
    else:
        fd = os.open(temp, flags)
    try:
        with os.fdopen(fd, "w", encoding=encoding, closefd=False) as handle:
            handle.write(content)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                # Atomic replacement remains correct when an uncommon local
                # filesystem cannot fsync a file descriptor.
                pass
        os.close(fd)
        fd = -1
        chmod_private_file(temp)
        os.replace(temp, path)
        chmod_private_file(path)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def append_private_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    mkdir_private(path.parent)
    if supports_posix_modes():
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0),
            _FILE_MODE,
        )
        try:
            os.write(fd, content.encode(encoding))
        finally:
            os.close(fd)
    else:
        with path.open("a", encoding=encoding) as handle:
            handle.write(content)
