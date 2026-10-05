"""Process-tree teardown and bounded subprocess ownership cleanup.

POSIX process groups cover descendants that remain in the original session;
inherited ownership markers also identify descendants that escape it.
Windows has no process-group SIGKILL equivalent, so timeout paths that only
call ``Popen.kill()`` leave agent-CLI grandchildren alive.

This module prefers ``taskkill /F /T`` (tree kill) and falls back to a
``CreateToolhelp32Snapshot`` walk + ``TerminateProcess``. Every step is
best-effort: missing ``taskkill``, denied handles, or unavailable Toolhelp
APIs never raise to the caller.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from typing import Iterable, Optional


_KERNEL32 = None


def _kernel32():
    """kernel32 with full signatures, never the process-global ``ctypes.windll``.

    Untyped calls truncate 64-bit HANDLE results and arguments to 32-bit int,
    and windll does not capture the error for ``ctypes.get_last_error()``.
    """
    global _KERNEL32
    if _KERNEL32 is None:
        import ctypes
        from ctypes import wintypes
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, args, result in (
            ("OpenProcess", [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            ("GetExitCodeProcess", [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL),
            ("TerminateProcess", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
            ("CreateToolhelp32Snapshot", [wintypes.DWORD, wintypes.DWORD], wintypes.HANDLE),
            ("Process32FirstW", [wintypes.HANDLE, ctypes.c_void_p], wintypes.BOOL),
            ("Process32NextW", [wintypes.HANDLE, ctypes.c_void_p], wintypes.BOOL),
            ("GetProcessTimes", [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4,
             wintypes.BOOL),
        ):
            function = getattr(api, name)
            function.argtypes, function.restype = args, result
        _KERNEL32 = api
    return _KERNEL32


def pid_alive_windows(pid: int) -> bool:
    """Non-destructive liveness probe: ``OpenProcess`` plus the exit code.

    Access denied means the process exists but belongs to someone else, so it
    counts as alive (POSIX ``EPERM``). ``os.kill(pid, 0)`` is not an option on
    Windows: it sends CTRL_C_EVENT or terminates the target.
    """
    import ctypes
    from ctypes import wintypes
    kernel32 = _kernel32()
    handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def process_identity_windows(pid: int) -> Optional[str]:
    """The creation FILETIME, fixed for the life of the process (see ``proc_identity``)."""
    import ctypes
    from ctypes import wintypes
    kernel32 = _kernel32()
    handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel32.GetProcessTimes(handle, *(ctypes.byref(value) for value in times)):
            return None
        return str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)
    finally:
        kernel32.CloseHandle(handle)


def kill_process_tree(pid: int) -> bool:
    """Kill ``pid`` and its descendants on Windows.

    Returns True when at least one kill method was attempted successfully
    enough to consider the tree addressed (taskkill ran, or Toolhelp
    terminated the root). Returns False when no method was available or
    the pid is invalid — callers should fall back to ``Popen.kill()``.
    """
    if os.name != "nt" or pid is None or int(pid) <= 0:
        return False
    pid = int(pid)
    if _taskkill_process_tree(pid):
        return True
    return _toolhelp_kill_process_tree(pid)


def _taskkill_creationflags() -> int:
    """Hide the taskkill console under console-less hosts when possible."""
    try:
        from puppetmaster.win_console import effective_creationflags

        return int(effective_creationflags(0))
    except Exception:
        return getattr(subprocess, "CREATE_NO_WINDOW", 0) or 0


def _taskkill_process_tree(pid: int, timeout: float = 15) -> bool:
    """Invoke ``taskkill /F /T /PID`` when the binary is on PATH."""
    taskkill = shutil.which("taskkill")
    if not taskkill:
        return False
    try:
        completed = subprocess.run(
            [taskkill, "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=timeout,
            creationflags=_taskkill_creationflags(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    # 0 = killed, 128 = process not found (already gone) — both fine.
    return completed.returncode in (0, 128)


def _toolhelp_kill_process_tree(pid: int, *, deadline: Optional[float] = None) -> bool:
    """Enumerate descendants via Toolhelp and TerminateProcess each one."""
    try:
        targets = _toolhelp_tree_pids(pid)
    except Exception:
        return False
    if not targets:
        return False
    killed_any = False
    for target in targets:
        if deadline is not None and time.monotonic() >= deadline:
            return False
        try:
            if _terminate_pid(target):
                killed_any = True
        except Exception:
            continue
    return killed_any


def _toolhelp_tree_pids(root_pid: int) -> list[int]:
    """Return descendants (deepest-first) followed by ``root_pid``."""
    children_by_parent = _snapshot_children_by_parent()
    if children_by_parent is None:
        return []
    return _descendant_pids_from_map(root_pid, children_by_parent)


def _snapshot_children_by_parent() -> Optional[dict[int, list[int]]]:
    import ctypes
    from ctypes import wintypes

    TH32CS_SNAPPROCESS = 0x00000002

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(wintypes.ULONG)),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32 = _kernel32()
    invalid = ctypes.c_void_p(-1).value
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not snapshot or snapshot == invalid:
        return None

    children_by_parent: dict[int, list[int]] = {}
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        if not kernel32.Process32FirstW(snapshot, ctypes.addressof(entry)):
            return None
        while True:
            parent = int(entry.th32ParentProcessID)
            child = int(entry.th32ProcessID)
            children_by_parent.setdefault(parent, []).append(child)
            if not kernel32.Process32NextW(snapshot, ctypes.addressof(entry)):
                break
    finally:
        kernel32.CloseHandle(snapshot)
    return children_by_parent


def _terminate_pid(pid: int) -> bool:
    PROCESS_TERMINATE = 0x0001
    kernel32 = _kernel32()
    handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, int(pid))
    if not handle:
        return False
    try:
        return bool(kernel32.TerminateProcess(handle, 1))
    finally:
        kernel32.CloseHandle(handle)


def _descendant_pids_from_map(
    root_pid: int, children_by_parent: dict[int, Iterable[int]]
) -> list[int]:
    """Descendants deepest-first, then ``root_pid``. Cycle-safe under PID reuse."""
    root = int(root_pid)
    descendants: list[int] = []
    seen = {root}
    stack = [int(child) for child in children_by_parent.get(root, ())]
    while stack:
        child = stack.pop()
        if child in seen:
            # Parent/child cycles appear under PID reuse; skip already-walked
            # nodes so Toolhelp fallback cannot hang the timeout path.
            continue
        seen.add(child)
        descendants.append(child)
        stack.extend(int(next_child) for next_child in children_by_parent.get(child, ()))
    descendants.reverse()
    descendants.append(root)
    return descendants


def _owned_posix_pids(owner: str, timeout: float) -> list[int]:
    """Find inherited ownership even after setsid and parent reparenting.

    Both macOS and Linux ps expose the initial environment with eww. Keep
    output private: it contains environments of other processes as well.
    """
    result = subprocess.run(["ps", "eww", "-ax", "-o", "pid=", "-o", "command="],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, timeout=timeout, check=True)
    marker = "PUPPETMASTER_PROCESS_OWNER=" + owner
    return [int(fields[0]) for line in result.stdout.splitlines()
            if (fields := line.split()) and fields[0].isdigit() and marker in fields[1:]]


def stop_owned_process(process: subprocess.Popen, owner: str, deadline: float) -> None:
    """Bound teardown by an absolute deadline, including after leader exit.

    POSIX ownership is inherited through the environment, independently of
    session and parent IDs. Windows uses the launch-owned Job Object.
    """
    if os.name == "nt":
        job = getattr(process, "_puppetmaster_job", None)
        if job is not None:
            try:
                job.terminate()
            except OSError:
                pass
    else:
        # Freeze the original group while enumerating escaped descendants.
        try:
            os.killpg(process.pid, signal.SIGSTOP)
        except Exception:
            pass
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                targets = _owned_posix_pids(owner, remaining)
            except Exception:
                # Discovery is strict before launch, but best-effort at teardown.
                break
            if not targets:
                break
            for pid in targets:
                if time.monotonic() >= deadline:
                    break
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    continue
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass
    try:
        process.kill()
    except Exception:
        pass
    remaining = deadline - time.monotonic()
    if remaining > 0:
        try:
            process.wait(timeout=remaining)
        except Exception:
            pass


def cleanup_owned_process(process: subprocess.Popen, owner: str, deadline: float):
    """Conservative cancellation cleanup using a live handle and an owner nonce.

    POSIX groups require an unreaped launch-owned session leader; escaped
    descendants require current exact nonce evidence. Windows uses a Job Object and the
    held Popen handle, never cached ancestry or descendant PIDs.
    This is bounded best effort, not containment or proof of remote cancellation.
    """
    descendant_outcome = "unknown"
    if os.name == "posix" and getattr(process, "_puppetmaster_session", False):
        # An unreaped child reserves its PID, hence its launch-time PGID.
        # Exclude concurrent Popen.wait/poll reaping until after signalling;
        # checking poll() first would release that reservation on leader exit.
        remaining = deadline - time.monotonic()
        if remaining > 0 and process._waitpid_lock.acquire(timeout=remaining):
            try:
                if process.returncode is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                        descendant_outcome = "partial"
                    except OSError:
                        pass
            finally:
                process._waitpid_lock.release()
    if os.name == "nt":
        job = getattr(process, "_puppetmaster_job", None)
        if job is not None:
            try:
                job.terminate()
                descendant_outcome = "partial"
            except OSError:
                pass
    if os.name == "posix" and owner and deadline > time.monotonic():
        try:
            targets = _owned_posix_pids(owner, deadline - time.monotonic())
            for pid in targets:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                # Revalidate each candidate immediately before signalling;
                # stale discovery from a previous cleanup is never authority.
                current = _owned_posix_pids(owner, remaining)
                if pid in current:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            descendant_outcome = "partial"
        except (OSError, subprocess.SubprocessError):
            descendant_outcome = "unknown"
    try:
        # Popen owns its process handle and guards against a reaped PID.
        if process.poll() is None:
            process.kill()
        remaining = deadline - time.monotonic()
        if remaining > 0:
            process.wait(timeout=remaining)
        exited = process.poll() is not None
    except (OSError, subprocess.SubprocessError):
        exited = False
    from puppetmaster.contracts import ProcessCleanupReceipt
    return ProcessCleanupReceipt("observed_exit" if exited else "unknown", descendant_outcome)


class WindowsJob:
    """Own descendants by kernel handle, including after the leader exits."""

    def __init__(self):
        import ctypes
        import threading
        from ctypes import wintypes
        self.lock = threading.Lock()
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
            "SetInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = args, result
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

        class BasicLimits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
                        ("max_working_set", ctypes.c_size_t), ("active_processes", wintypes.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                        ("scheduling", wintypes.DWORD)]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io", ctypes.c_uint64 * 6),
                        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                        ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

        limits = ExtendedLimits()
        limits.basic.flags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign_and_resume(self, process):
        import ctypes
        from ctypes import wintypes
        if not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
        # Popen closes the primary thread handle. Resume via the owned process
        # handle only after assignment, so no descendant can escape the job.
        resume = ctypes.WinDLL("ntdll").NtResumeProcess
        resume.argtypes, resume.restype = [wintypes.HANDLE], ctypes.c_long
        if resume(int(process._handle)) < 0:
            raise OSError("NtResumeProcess failed")

    def terminate(self):
        import ctypes
        with self.lock:
            if self.handle and not self.api.TerminateJobObject(self.handle, 1):
                raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        with self.lock:
            if self.handle:
                self.api.CloseHandle(self.handle)
                self.handle = None


def popen_owned(*args, **kwargs):
    """Fail closed if Windows containment cannot be established before execution."""
    if os.name != "nt":
        process = subprocess.Popen(*args, **kwargs)
        process._puppetmaster_session = bool(kwargs.get("start_new_session", False))
        return process
    job = WindowsJob()
    process = None
    try:
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | 0x00000004  # CREATE_SUSPENDED
        process = subprocess.Popen(*args, **kwargs)
        job.assign_and_resume(process)
        process._puppetmaster_job = job
        return process
    except BaseException:
        try:
            if process is not None:
                process.kill()
                process.wait(timeout=3)
        finally:
            job.close()
        raise


def close_owned_process(process):
    job = getattr(process, "_puppetmaster_job", None)
    if job is not None:
        try:
            job.terminate()
        except OSError:
            pass
        finally:
            job.close()
