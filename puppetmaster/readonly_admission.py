"""Kernel-owned, per-database reader admission; coordination files are permanent."""
import errno
import hashlib
import os
from pathlib import Path
import stat
import threading
import time
import weakref


_windows_locks = weakref.WeakValueDictionary()
_fds = set()
_registry_lock = threading.RLock()
_registry_pid = os.getpid()


def _after_fork():
    global _fds, _registry_lock, _registry_pid, _windows_locks
    pid = os.getpid()
    if pid == _registry_pid:
        return
    inherited = _fds
    _fds = set()
    _windows_locks = weakref.WeakValueDictionary()
    _registry_lock = threading.RLock()
    _registry_pid = pid
    # Never unlock an inherited open file description: it belongs to the parent.
    for fd in inherited:
        try:
            os.close(fd)
        except OSError:
            pass


def _before_fork():
    _after_fork()
    _registry_lock.acquire()


def _parent_fork():
    _registry_lock.release()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(before=_before_fork, after_in_parent=_parent_fork,
                        after_in_child=_after_fork)


def _identity(path, selected=None):
    if selected is not None:
        return selected
    info = os.stat(path)
    return info.st_dev, info.st_ino


def _key(identity):
    return hashlib.sha256(repr(tuple(identity)).encode('ascii')).hexdigest()


def _windows_directory():
    import ctypes
    import uuid
    from ctypes import wintypes
    shell = ctypes.WinDLL('shell32', use_last_error=True)
    ole = ctypes.WinDLL('ole32', use_last_error=True)
    folder = (ctypes.c_ubyte * 16).from_buffer_copy(
        uuid.UUID('F1B32785-6FBA-4FCF-9D55-7B8E7F157091').bytes_le)
    value = ctypes.c_wchar_p()
    shell.SHGetKnownFolderPath.argtypes = [ctypes.c_void_p, wintypes.DWORD,
                                         wintypes.HANDLE, ctypes.POINTER(ctypes.c_wchar_p)]
    shell.SHGetKnownFolderPath.restype = ctypes.c_long
    ole.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    result = shell.SHGetKnownFolderPath(ctypes.byref(folder), 0, None, ctypes.byref(value))
    if result:
        raise OSError('SHGetKnownFolderPath failed: HRESULT 0x%08x' % (result & 0xffffffff))
    try:
        return Path(value.value)
    finally:
        ole.CoTaskMemFree(value)


def _windows_api():
    import ctypes
    from ctypes import wintypes

    class Overlapped(ctypes.Structure):
        _fields_ = [('internal', ctypes.c_size_t), ('internal_high', ctypes.c_size_t),
                    ('offset', wintypes.DWORD), ('offset_high', wintypes.DWORD),
                    ('event', wintypes.HANDLE)]

    class Info(ctypes.Structure):
        _fields_ = [('attributes', wintypes.DWORD), ('creation', wintypes.FILETIME),
                    ('access', wintypes.FILETIME), ('write', wintypes.FILETIME),
                    ('volume', wintypes.DWORD), ('size_high', wintypes.DWORD),
                    ('size_low', wintypes.DWORD), ('links', wintypes.DWORD),
                    ('index_high', wintypes.DWORD), ('index_low', wintypes.DWORD)]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    for name, args, result in (
        ('CreateFileW', [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                         ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE], wintypes.HANDLE),
        ('GetFileInformationByHandle', [wintypes.HANDLE, ctypes.POINTER(Info)], wintypes.BOOL),
        ('SetHandleInformation', [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD], wintypes.BOOL),
        ('LockFileEx', [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                        wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(Overlapped)], wintypes.BOOL),
        ('CloseHandle', [wintypes.HANDLE], wintypes.BOOL),
    ):
        method = getattr(kernel, name)
        method.argtypes, method.restype = args, result

    class API:
        def open(self, target):
            # OPEN_ALWAYS, OPEN_REPARSE_POINT, no delete sharing: pin the final
            # inode until close. NULL security attributes are non-inheritable.
            handle = kernel.CreateFileW(str(target), 0xc0000000, 3, None, 4, 0x00200080, None)
            if handle == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            return handle

        def validate(self, handle):
            info = Info()
            if not kernel.GetFileInformationByHandle(handle, ctypes.byref(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            if info.attributes & (0x400 | 0x10) or info.links != 1:
                raise OSError('unsafe reader coordination file: reparse point, directory or hardlink')
            if not kernel.SetHandleInformation(handle, 1, 0):
                raise ctypes.WinError(ctypes.get_last_error())

        def lock(self, handle):
            if kernel.LockFileEx(handle, 3, 0, 1, 0, ctypes.byref(Overlapped())):
                return True
            error = ctypes.get_last_error()
            if error == 33:  # ERROR_LOCK_VIOLATION; stale files carry no ownership.
                return False
            raise ctypes.WinError(error)

        def close(self, handle):
            if not kernel.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())
    return API()


class _WindowsFileLock:
    def __init__(self, identity, deadline, clock):
        self.api = _windows_api()
        self.pid = os.getpid()
        self.handle = None
        # LocalAppData is OS-resolved, stable across sessions and environment
        # changes. Its user-private inherited ACL is the trust boundary; do not
        # use this directory if an administrator has granted other users write.
        root = _windows_directory() / 'PuppetmasterReaders'
        root.mkdir(mode=0o700, exist_ok=True)
        _safe(root, stat.S_ISDIR)
        self.handle = self.api.open(root / (_key(identity) + '.lock'))
        try:
            self.api.validate(self.handle)
            while True:
                if self.api.lock(self.handle):
                    return
                remaining = deadline - clock.monotonic()
                if remaining <= 0:
                    from puppetmaster.readonly import ReadTimeout
                    raise ReadTimeout('unable to open database: reader timed out')
                clock.sleep(min(.01, remaining))
        except BaseException:
            try:
                self.release()
            except BaseException as cleanup_error:
                cleanup_error.admission_owner = self
                raise
            raise

    def release(self):
        if self.pid != os.getpid() or self.handle is None:
            return
        # File lock ownership follows the handle, never the acquiring thread.
        # Close/process death releases it. Keep the handle retryable on failure.
        self.api.close(self.handle)
        self.handle = None


def _safe(path, mode):
    info = path.lstat()
    if (not mode(info.st_mode) or stat.S_ISLNK(info.st_mode) or
            getattr(info, 'st_file_attributes', 0) & 0x400):
        raise OSError('unsafe reader coordination path')
    if os.name != 'nt' and (info.st_uid != os.getuid() or info.st_mode & 0o077):
        raise PermissionError('reader coordination path must be private')
    return info


def _coordination_path(path, selected=None):
    # One root for every Puppetmaster process of the user, so helper spawns
    # serialize machine-wide. Test suites point it at their own directory.
    override = os.environ.get('PUPPETMASTER_READER_COORDINATION_DIR')
    root = Path(override) if override else Path('/tmp') / ('puppetmaster-readers-' + str(os.getuid()))
    root.mkdir(mode=0o700, exist_ok=True)
    _safe(root, stat.S_ISDIR)
    return root / (_key(_identity(path, selected)) + '.lock')


def _try_lock(fd):
    try:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
            raise
        return False


class ReaderAdmission:
    def __init__(self, path, deadline, *, clock=time, selected=None):
        self.pid = os.getpid()
        self.fd = None
        self.windows_lock = None
        self.local_lock = None
        _after_fork()
        identity = _identity(path, selected)
        if os.name == 'nt':
            with _registry_lock:
                local = _windows_locks.setdefault(identity, threading.Lock())
            if not local.acquire(timeout=max(0, deadline - clock.monotonic())):
                from puppetmaster.readonly import ReadTimeout
                raise ReadTimeout('unable to open database: reader timed out')
            self.local_lock = local
            try:
                self.windows_lock = _WindowsFileLock(identity, deadline, clock)
            except BaseException:
                self.local_lock = None
                local.release()
                raise
            return
        target = _coordination_path(path, identity)
        try:
            _safe(target, stat.S_ISREG)
        except FileNotFoundError:
            pass
        with _registry_lock:
            fd = os.open(target, os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0) |
                         getattr(os, 'O_NONBLOCK', 0), 0o600)
            try:
                os.set_inheritable(fd, False)
                info = _safe(target, stat.S_ISREG)
                actual = os.fstat(fd)
                if (info.st_dev, info.st_ino) != (actual.st_dev, actual.st_ino) or actual.st_nlink != 1:
                    raise OSError('reader coordination file replaced or linked')
            except BaseException:
                os.close(fd)
                raise
            self.fd = fd
            _fds.add(fd)
        try:
            while True:
                if _try_lock(fd):
                    return
                remaining = deadline - clock.monotonic()
                if remaining <= 0:
                    from puppetmaster.readonly import ReadTimeout
                    raise ReadTimeout('unable to open database: reader timed out')
                clock.sleep(min(.01, remaining))
        except BaseException:
            self.release()
            raise

    def release(self):
        _after_fork()
        if self.pid != os.getpid():
            return
        if self.windows_lock is not None:
            self.windows_lock.release()
            if self.local_lock is not None:
                local, self.local_lock = self.local_lock, None
                local.release()
            return
        with _registry_lock:
            if self.fd is None:
                return
            fd = self.fd
            # The fork barrier covers close AND unregister, including the
            # boundary where another thread could otherwise inherit this fd.
            os.close(fd)
            _fds.remove(fd)
            self.fd = None


# Cross-process gate for helper Popen only. Not reader admission: interpreter
# startup must not look like a source-open lock. Identity is process-global so
# 32 workers cannot birth 32 python.exe helpers in one Windows stampede.
_HELPER_SPAWN_IDENTITY = ("puppetmaster-helper-spawn", 1)


def helper_spawn_gate(deadline, clock=time):
    if deadline is None:
        deadline = clock.monotonic() + 5.0
    return ReaderAdmission(
        Path("helper-spawn"),
        deadline,
        clock=clock,
        selected=_HELPER_SPAWN_IDENTITY,
    )
