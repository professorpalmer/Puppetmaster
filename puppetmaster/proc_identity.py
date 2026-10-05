"""Tell a live process from a later one that was given the same pid.

A pid is not an identity: once a process dies the OS can hand its pid to an
unrelated process. Recording ``process_identity(pid)`` next to a pid lets a
reader check that the process it finds is still the one that was recorded.

The token comes from the kernel's record of when the process started (boot
ticks on Linux, ``p_starttime`` on macOS, the creation FILETIME on Windows), so
it is fixed for the life of the process and does not move with the wall clock.
Stdlib only: the lock module imports this at load time.
"""

from __future__ import annotations

import os
import struct
import sys
from typing import Any, Optional, Tuple

_OWN: Optional[Tuple[int, Optional[str]]] = None
_SYSCTL: Any = None


def process_identity(pid: int) -> Optional[str]:
    """A token that differs between two processes that held ``pid``.

    None means the platform or permissions do not say; callers then fall back
    to trusting the pid alone.
    """
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        if sys.platform.startswith("linux"):
            with open(f"/proc/{pid}/stat", "rb") as handle:
                # Field 22, starttime: counted from the last ")" because the
                # command name may itself contain spaces and parentheses.
                return handle.read().rsplit(b")", 1)[1].split()[19].decode("ascii")
        if sys.platform == "darwin":
            return _darwin_identity(pid)
        if os.name == "nt":
            from puppetmaster.win_process import process_identity_windows

            return process_identity_windows(pid)
    except Exception:
        return None
    return None


def own_identity() -> Optional[str]:
    """This process's identity, read once and again after a fork."""
    global _OWN
    pid = os.getpid()
    if _OWN is None or _OWN[0] != pid:
        _OWN = (pid, process_identity(pid))
    return _OWN[1]


def pid_reused(pid: int, recorded: Optional[str]) -> bool:
    """True when ``pid`` now belongs to a different process than ``recorded``.

    Unknown on either side is never reuse.
    """
    if not recorded or not isinstance(pid, int):
        return False
    current = process_identity(pid)
    return current is not None and current != recorded


def _darwin_identity(pid: int) -> Optional[str]:
    """``kinfo_proc.kp_proc.p_starttime`` through sysctl, without spawning ps."""
    import ctypes

    global _SYSCTL
    if _SYSCTL is None:
        function = ctypes.CDLL(None, use_errno=True).sysctl
        function.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_uint, ctypes.c_void_p,
                             ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]
        function.restype = ctypes.c_int
        _SYSCTL = function
    mib = (ctypes.c_int * 4)(1, 14, 1, pid)  # CTL_KERN, KERN_PROC, KERN_PROC_PID
    buffer = ctypes.create_string_buffer(1024)  # sizeof(struct kinfo_proc) is 648
    size = ctypes.c_size_t(len(buffer))
    if _SYSCTL(mib, 4, buffer, ctypes.byref(size), None, 0) != 0 or size.value < 16:
        return None  # size 0: no such process
    # p_starttime (struct timeval) is the first member of kinfo_proc.
    seconds, micros = struct.unpack_from("=qi", buffer.raw)
    return f"{seconds}.{micros:06d}"
