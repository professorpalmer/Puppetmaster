"""Private bounded SQL reader process; stdin/stdout are a local JSON protocol.

Two open modes, chosen from what the source looks like when the session starts:

* **Quiet source** (no live connection, fully checkpointed WAL): the session
  takes SQLite's shared byte range exclusively and opens the main file through
  an ``immutable=1`` descriptor URI. Nothing on disk is touched at all.
* **Live source** (a live connection, or uncheckpointed WAL frames): the
  session joins the committed WAL snapshot through a ``mode=ro`` pathname open
  and takes no byte-range lock, so writers keep committing while it reads.
  A WAL reader records read marks in the ``-shm`` index, so the promise in this
  mode is *never changes database content*, not *touches no file*: the main
  database bytes and the WAL are never written, and a ``mode=ro`` connection
  never checkpoints. Missing or damaged sidecars are never created, repaired,
  or recovered here -- they fail closed as unavailable.

Live mode has no descriptor-bound open to lean on, so a device/inode fence over
the main file and both sidecars runs before and after every statement; a
replaced or renamed source is refused instead of read.
"""
import ctypes
import json
import os
import sqlite3
import stat
import sys
import time
from pathlib import Path


def stamp(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def source_stamp(path=None, *, fd=None):
    """Use ChangeTime on Windows, where stat's ctime can mean CreationTime.

    Only Windows opens a temporary metadata descriptor here: closing an fd on
    POSIX could release another connection's process-wide SQLite locks.
    """
    if os.name != 'nt':
        return stamp(os.stat(path) if fd is None else os.fstat(fd))
    import msvcrt
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    class BasicInfo(ctypes.Structure):
        _fields_ = [('creation', ctypes.c_longlong), ('access', ctypes.c_longlong),
                    ('write', ctypes.c_longlong), ('change', ctypes.c_longlong),
                    ('attributes', wintypes.DWORD)]
    # ChangeTime tracks metadata updates, unlike CreationTime, but is not a
    # rename counter. Windows replacement safety uses deny-delete sharing.
    # https://learn.microsoft.com/windows/win32/api/winbase/ns-winbase-file_basic_info
    query = kernel.GetFileInformationByHandleEx
    query.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    query.restype = wintypes.BOOL
    owned = fd is None
    if owned:
        create = kernel.CreateFileW
        create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                           ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
        create.restype = wintypes.HANDLE
        # Attributes only, shared read/write/delete, including directories.
        handle = create(str(path), 0x80, 7, None, 3, 0x02000000, None)
        if handle == wintypes.HANDLE(-1).value:
            raise _retryable_windows_contention(ctypes.WinError(ctypes.get_last_error()))
        try:
            fd = msvcrt.open_osfhandle(handle, os.O_RDONLY)
        except BaseException:
            close = kernel.CloseHandle
            close.argtypes = (wintypes.HANDLE,)
            close.restype = wintypes.BOOL
            close(handle)
            raise
    try:
        # fstat's size and FILE_BASIC_INFO are separate native queries. Fence
        # the size query with BasicInfo so a writer cannot splice old size into
        # new timestamps (or new size into old timestamps).
        epoch = 116444736000000000
        for _ in range(8):
            first, last = BasicInfo(), BasicInfo()
            if not query(msvcrt.get_osfhandle(fd), 0, ctypes.byref(first), ctypes.sizeof(first)):
                raise _retryable_windows_contention(ctypes.WinError(ctypes.get_last_error()))
            st = os.fstat(fd)
            if not query(msvcrt.get_osfhandle(fd), 0, ctypes.byref(last), ctypes.sizeof(last)):
                raise _retryable_windows_contention(ctypes.WinError(ctypes.get_last_error()))
            write = (last.write - epoch) * 100
            if (first.write, first.change) == (last.write, last.change) and st.st_mtime_ns == write:
                return (st.st_dev, st.st_ino, st.st_size, write, (last.change - epoch) * 100)
        raise OSError('source metadata did not stabilize')
    finally:
        if owned:
            os.close(fd)


def same_store_write(before, after):
    # A changed size/mtime on the same file proves a pre-snapshot write race.
    # Windows can publish size before LastWriteTime on a live writer handle.
    # ctime-only drift (rename ABA/permissions) never earns a retry.
    return (before is not None and after is not None and
            before[:2] == after[:2] and before[2:4] != after[2:4])


def adopt_darwin_open_ctime(before, descriptor_after_open):
    """Rebase the pre-open main stamp after Darwin r+b LOCK_EX.

    APFS can bump ctime on a writable open of the same inode (Application
    Support ``state.sqlite3``). That is not a replaced file. Later full-tuple
    compares still fence ABA. Keep a list so it stays type-equal to stamps().
    """
    main = before[0]
    if (sys.platform != 'darwin' or main is None or descriptor_after_open is None
            or main[:4] != descriptor_after_open[:4]):
        return before
    adopted = list(before)
    adopted[0] = descriptor_after_open
    return adopted


def emit(value):
    print(json.dumps(value), flush=True)


def stamps(path):
    result = []
    for suffix in ('', '-wal', '-shm', '-journal'):
        try:
            result.append(source_stamp(str(path) + suffix))
        except FileNotFoundError:
            result.append(None)
    return result


def wal_index_number(header, start, stop):
    return int.from_bytes(header[start:stop], sys.byteorder)


def wal_index_header(path):
    """Return the validated constant-size WAL-index header, else None."""
    try:
        with open(str(path) + '-shm', 'rb') as index:
            header = index.read(100)
    except OSError:
        return None
    if len(header) != 100 or header[:48] != header[48:96] or header[12] != 1:
        return None
    if wal_index_number(header, 0, 4) != 3007000:
        return None
    first = second = 0
    for offset in range(0, 40, 8):
        first = (first + wal_index_number(header, offset, offset + 4) + second) & 0xffffffff
        second = (second + wal_index_number(header, offset + 4, offset + 8) + first) & 0xffffffff
    if (first, second) != (wal_index_number(header, 40, 44), wal_index_number(header, 44, 48)):
        return None
    return header


def checkpointed_sidecars(path, before, journal):
    """Constant-size WAL-index proof, under an exclusive main-file lock.

    Some SQLite builds retain a zero-length WAL and its checkpointed index on
    last close. Presence alone does not mean live or uncheckpointed WAL.
    """
    if before[3] is not None:
        return False
    if before[1] is None and before[2] is None:
        return True
    # Apple SQLite can leave only an old SHM after a successful switch to
    # DELETE mode. The main header and empty/absent WAL prove it is not live.
    if journal == b'\x01\x01' and (before[1] is None or before[1][2] == 0):
        return True
    if before[2] is None:
        return False
    header = wal_index_header(path)
    if header is None:
        return False
    return wal_index_number(header, 16, 20) == wal_index_number(header, 96, 100)


def joinable_wal(path, before, journal):
    """Can a read-only connection join this source's committed WAL snapshot?

    Only a complete, structurally sound cohort qualifies: the main header says
    WAL, no rollback journal is present, and both a non-empty -wal and a valid
    -shm index already exist. A reader never creates, repairs, or recovers a
    sidecar, so an incomplete cohort stays unavailable.
    """
    if journal != b'\x02\x02' or before[3] is not None:
        return False
    if before[1] is None or before[2] is None or before[1][2] < 32:
        return False
    return wal_index_header(path) is not None


def open_windows_source(path):
    """Pin the selected file across SQLite's independent pathname open/read."""
    import msvcrt
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    # GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, OPEN_EXISTING.
    # Omitting FILE_SHARE_DELETE excludes rename/replacement until close, even
    # when an A->B->A round trip would leave ChangeTime unchanged. Existing
    # DELETE handles make this open fail too; never fall back to a shared open.
    handle = create(str(path), 0x80000000, 3, None, 3, 0, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        close = kernel.CloseHandle
        close.argtypes = (wintypes.HANDLE,)
        close.restype = wintypes.BOOL
        close(handle)
        raise
    try:
        return os.fdopen(fd, 'rb')
    except BaseException:
        os.close(fd)
        raise


def _retryable_windows_contention(exc):
    if os.name == 'nt' and getattr(exc, 'winerror', None) in (5, 32, 33):
        exc.source_open_contention = True
    return exc


def _stamps_after_open(path):
    try:
        return stamps(path)
    except OSError as exc:
        raise _retryable_windows_contention(exc)


def descriptor_uri(fd):
    """Select a source descriptor URI; Linux additionally attests SQLite's fd."""
    expected = os.fstat(fd)
    for namespace in ('/proc/self/fd', '/dev/fd'):
        candidate = Path(namespace) / str(fd)
        try:
            probe = os.open(candidate, os.O_RDONLY)
        except OSError:
            continue
        try:
            actual = os.fstat(probe)
        finally:
            os.close(probe)
        if (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino):
            # Do not resolve(): that would restore the original pathname race.
            # main runs the checkpoint proof before opening this immutable URI,
            # which prevents SQLite from consulting namespace sidecars.
            return candidate.as_uri() + '?mode=ro&immutable=1'
    raise OSError('no usable SQLite descriptor namespace')


def connect_descriptor(uri, attempts=5):
    """Open a macOS /dev/fd URI, retrying SQLite's spurious CANTOPEN.

    Under load SQLite now and then fails to open /dev/fd/N with
    "unable to open database file" while the descriptor and the source are
    unchanged; the same open then succeeds at once. Any other error, or one
    that persists, still raises.
    """
    for attempt in range(attempts):
        try:
            return sqlite3.connect(uri, uri=True)
        except sqlite3.OperationalError as exc:
            if str(exc) != 'unable to open database file' or attempt == attempts - 1:
                raise
            time.sleep(.005 * (attempt + 1))


def linux_fd_snapshot():
    """Inventory live descriptors without opening/closing any database fd.

    Keep scandir alive while inspecting entries, so its own fd cannot be
    recycled into SQLite's main fd between enumeration and fstat. Only the
    procfs directory descriptors are omitted. Any disappearing entry fails
    closed; this private helper has one thread and no other SQLite connection.
    """
    directory = '/proc/self/fd'
    proc = os.stat(directory)
    result = {}
    with os.scandir(directory) as entries:
        for entry in entries:
            fd = int(entry.name)
            current = os.fstat(fd)
            if (stat.S_ISDIR(current.st_mode) and
                    (current.st_dev, current.st_ino) == (proc.st_dev, proc.st_ino)):
                continue
            result[fd] = (current.st_dev, current.st_ino, stat.S_IFMT(current.st_mode))
    return result


def attest_linux_database(before, source_fd, *, sidecars=()):
    """Prove every new descriptor belongs to the validated source cohort.

    SQLite's built-in immutable Unix VFS opens main during connect and keeps
    that fd until close. No queries/callbacks run between the snapshots. An
    extra descriptor (even another fd for A), a missing fd, or reuse of an
    existing number for a different identity makes attribution ambiguous.
    Do not select a matching A fd out of several: SQLite might be using B.

    A WAL join passes the validated -wal/-shm identities as ``sidecars``, since
    a snapshot read legitimately opens those too. Main must still appear, and
    any descriptor outside the cohort is still foreign.
    """
    after = linux_fd_snapshot()
    if any(after.get(fd) != identity for fd, identity in before.items()):
        raise OSError('SQLite fd attestation: existing descriptors changed')
    opened = [identity for fd, identity in after.items() if fd not in before]
    source = os.fstat(source_fd)
    expected = (source.st_dev, source.st_ino, stat.S_IFREG)
    if sidecars:
        cohort = {expected, *((dev, ino, stat.S_IFREG) for dev, ino in sidecars)}
        attested = (expected in opened and
                    all(identity in cohort for identity in opened))
    else:
        attested = opened == [expected]
    if not stat.S_ISREG(source.st_mode) or not attested:
        raise OSError('SQLite fd attestation: missing, ambiguous, or foreign main descriptor')


def main(path, *, wal_snapshot=False):
    path = Path(path)
    wal_snapshot = bool(wal_snapshot and os.name == 'nt')
    try:
        before = stamps(path)
    except OSError as exc:
        raise _retryable_windows_contention(exc)
    if before[0] is None:
        emit(dict(kind='unavailable', error='unable to open database: source missing'))
        return
    # A writable descriptor is used only for a quiet source's exclusive
    # advisory lock, which excludes new WAL openers during the read so our lock
    # cannot strand a writer's final checkpoint/sidecar cleanup. No source
    # bytes are written through it in either mode.
    if os.name == 'nt':
        try:
            source = open_windows_source(path)
        except OSError as exc:
            exc.source_open_contention = getattr(exc, 'winerror', None) in (5, 32, 33)
            raise
        exclusive = True
    else:
        try:
            source = path.open('r+b')
            exclusive = True
        except PermissionError:
            source = path.open('rb')
            exclusive = False
    with source:
        descriptor_before = source_stamp(fd=source.fileno())
        if wal_snapshot:
            bound = descriptor_before[:2] == before[0][:2]
        elif sys.platform == 'darwin' and descriptor_before[:4] == before[0][:4]:
            bound = True
            before = adopt_darwin_open_ctime(before, descriptor_before)
        else:
            bound = descriptor_before == before[0]
        if not bound:
            emit(dict(kind='unavailable', error='unable to open database: source changed',
                      same_store_write=same_store_write(before[0], descriptor_before)))
            return
        header = None
        # Set when the source turns out to be live. POSIX then joins the
        # committed WAL snapshot if the sidecar cohort allows it, and only
        # emits this refusal when it does not.
        live = None
        if os.name == 'nt':
            uri = path.resolve().as_uri() + ('?mode=ro' if wal_snapshot else '?mode=ro&immutable=1')
            if not wal_snapshot:
                import msvcrt
                from ctypes import wintypes
                class Overlapped(ctypes.Structure):
                    _fields_ = [('internal', ctypes.c_size_t), ('internal_high', ctypes.c_size_t),
                                ('offset', wintypes.DWORD), ('offset_high', wintypes.DWORD),
                                ('event', wintypes.HANDLE)]
                overlapped = Overlapped()
                overlapped.offset = 1073741826
                kernel = ctypes.WinDLL('kernel32', use_last_error=True)
                lock = kernel.LockFileEx
                lock.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                 wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(Overlapped))
                lock.restype = wintypes.BOOL
                # LockFileEx permits GENERIC_READ handles. Exclusively reserving
                # SQLite's shared range excludes live connections without writing.
                # https://learn.microsoft.com/windows/win32/api/fileapi/nf-fileapi-lockfileex
                if not lock(msvcrt.get_osfhandle(source.fileno()), 3, 0, 510, 0, ctypes.byref(overlapped)):
                    snapshot = (before[1] is not None and before[1][2] > 32 and
                                before[2] is not None)
                    emit(dict(kind='unavailable',
                              error='unable to open database: active reader; sidecars may be missing',
                              code=5, wal_snapshot=snapshot))
                    return
        else:
            import fcntl
            if sys.platform == 'darwin':
                fields = [('start', ctypes.c_longlong), ('length', ctypes.c_longlong),
                          ('pid', ctypes.c_int), ('type', ctypes.c_short), ('whence', ctypes.c_short)]
            elif sys.platform.startswith('linux'):
                fields = [('type', ctypes.c_short), ('whence', ctypes.c_short),
                          ('start', ctypes.c_longlong), ('length', ctypes.c_longlong), ('pid', ctypes.c_int)]
            else:
                emit(dict(kind='unavailable', error='unable to open database: unsupported lock ABI'))
                return
            class Lock(ctypes.Structure):
                _fields_ = fields
            # pread leaves the file position alone, so the shared read below
            # still sees these same bytes. The journal mode decides whether a
            # live source is joinable, which is known before any locking.
            header = os.pread(source.fileno(), 100, 0)
            query = Lock()
            query.type, query.start, query.length = fcntl.F_WRLCK, 1073741826, 510
            answer = Lock.from_buffer_copy(fcntl.fcntl(source, fcntl.F_GETLK, bytes(query)))
            if answer.type != fcntl.F_UNLCK:
                live = dict(kind='OperationalError' if answer.type == fcntl.F_WRLCK else 'unavailable',
                            error='database is locked' if answer.type == fcntl.F_WRLCK else 'unable to open database: active reader; sidecars may be missing',
                            code=5)  # SQLITE_BUSY, also on Python before 3.11.
            else:
                # Select the descriptor before locking: closing any descriptor
                # for this inode releases this process's POSIX record locks, and
                # this probe opens and closes one. /dev/fd stat on macOS
                # describes a device node, so validate an opened descriptor.
                uri = descriptor_uri(source.fileno())
                try:
                    fcntl.lockf(source, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB, 510, 1073741826)
                except BlockingIOError:
                    live = dict(kind='OperationalError', error='database is locked')
        if live is None:
            after = _stamps_after_open(path)
            if not wal_snapshot and after != before:
                emit(dict(kind='unavailable', error='unable to open database: source changed',
                          launch_topology_change=after[0] == before[0] and after[1:] != before[1:],
                          same_store_write=same_store_write(before[0], after[0])))
                return
            if header is None:
                header = source.read(100)
            if not wal_snapshot and not checkpointed_sidecars(path, before, header[18:20]):
                # A validated WAL/index with uncheckpointed frames is concrete
                # write contention, even if the writer dropped its byte-range
                # lock between the lock probe and this checkpoint proof.
                live = dict(kind='unavailable',
                            error='unable to open database: live sidecars; retry after checkpoint',
                            code=5)
        if live is not None:
            if os.name == 'nt' or not joinable_wal(path, before, header[18:20]):
                emit(live)
                return
            # Join the committed WAL snapshot rather than refuse. Holding the
            # shared range exclusively would make every writer connection BUSY,
            # so drop it; unlocking a range we never took is a no-op.
            fcntl.lockf(source, fcntl.LOCK_UN, 510, 1073741826)
            wal_snapshot = True
            uri = path.resolve().as_uri() + '?mode=ro'
        if not wal_snapshot:
            after = _stamps_after_open(path)
            if after != before:
                emit(dict(kind='unavailable', error='unable to open database: source changed',
                          launch_topology_change=after[0] == before[0] and after[1:] != before[1:]))
                return
        linux = os.name != 'nt' and sys.platform.startswith('linux')
        pathname_bound = wal_snapshot and os.name != 'nt'
        sidecars = tuple(side[:2] for side in before[1:3]) if pathname_bound else ()
        def cohort_state():
            """'ok', 'cohort' (the WAL pair moved), or 'replaced' (new main).

            A pathname open has no descriptor-bound main file to lean on, so
            every statement revalidates the identities the snapshot was chosen
            from. A sidecar that moved is a retryable cohort change, not proof
            that the selected database was replaced.
            """
            if not pathname_bound:
                return 'ok'
            try:
                current = stamps(path)
            except OSError:
                return 'cohort'
            if current[0] is None or current[0][:2] != descriptor_before[:2]:
                return 'replaced'
            if ([None if side is None else side[:2] for side in current[1:3]] !=
                    [None if side is None else side[:2] for side in before[1:3]]):
                return 'cohort'
            return 'ok'

        def unchanged():
            if pathname_bound:
                return cohort_state() == 'ok'
            current = source_stamp(fd=source.fileno())
            return (current[:2] == descriptor_before[:2] if wal_snapshot
                    else current == descriptor_before)
        opened_before = linux_fd_snapshot() if linux else None
        c = connect_descriptor(uri) if sys.platform == 'darwin' and not wal_snapshot else sqlite3.connect(uri, uri=True)
        try:
            if linux:
                attest_linux_database(opened_before, source.fileno(), sidecars=sidecars)
            elif os.name != 'nt' and not wal_snapshot:
                # Some Unix VFS builds resolve /proc/self/fd symlinks back to
                # a replaceable pathname. Such an open is not descriptor-bound.
                # Never expose rows or retry against the source name in that case.
                filename = c.execute('PRAGMA database_list').fetchone()[2]
                if Path(filename).as_uri() + '?mode=ro&immutable=1' != uri:
                    raise OSError('SQLite resolved the descriptor to a pathname')
            try:
                c.execute('PRAGMA foreign_keys=ON')
                c.execute('PRAGMA synchronous=NORMAL')
                c.execute('BEGIN')
                c.execute('SELECT rootpage FROM sqlite_master LIMIT 1').fetchone()
            except sqlite3.OperationalError:
                # The last writer can checkpoint and delete the WAL pair between
                # the cohort check and this open; a read-only open then cannot
                # create -shm (SQLITE_CANTOPEN). That is a cohort turnover.
                state = cohort_state() if pathname_bound else 'ok'
                if state == 'ok':
                    raise
                emit(dict(kind='unavailable',
                          error=('unable to open database: live sidecars; retry after checkpoint'
                                 if state == 'cohort' else 'unable to open database: source changed'),
                          code=5 if state == 'cohort' else None))
                return
            state = cohort_state() if wal_snapshot else (
                'ok' if _stamps_after_open(path) == before else 'replaced')
            if state == 'cohort':
                # The writer cohort turned over before the snapshot was read.
                # Retry and re-evaluate the source rather than fail the caller.
                emit(dict(kind='unavailable',
                          error='unable to open database: live sidecars; retry after checkpoint',
                          code=5))
                return
            if state != 'ok':
                emit(dict(kind='unavailable', error='unable to open database: source changed'))
                return
            emit(dict(journal='wal' if header[18:20] == b'\x02\x02' else 'delete'))
            def event(name, args):
                emit(dict(event=name, args=args))
                answer = sys.stdin.readline(1024)
                if not answer:
                    raise EOFError()
                return json.loads(answer)
            progress_interval = 0
            for line in sys.stdin:
                if len(line) > 1024 * 1024:
                    return
                if not unchanged():
                    emit(dict(kind='unavailable', error='unable to open database: source changed'))
                    return
                request = json.loads(line)
                if 'control' in request:
                    name, value = request['control'], request['value']
                    if name == 'release':
                        return True
                    if name == 'authorizer':
                        c.set_authorizer((lambda *args: event('authorize', args)) if value else None)
                    elif name == 'trace':
                        c.set_trace_callback((lambda sql: event('trace', [sql])) if value else None)
                    elif name == 'progress':
                        progress_interval = value
                    emit(dict(rows=[], names=[]))
                    continue
                sql, parameters = request['sql'], request['parameters']
                deadline = time.monotonic() + 5
                c.set_progress_handler(lambda: int(time.monotonic() > deadline) or
                                       (event('progress', []) if progress_interval else 0),
                                       progress_interval or 1000)
                try:
                    cursor = c.execute(sql, parameters)
                    rows = cursor.fetchmany(1002)
                    if len(rows) > 1001:
                        emit(dict(kind='unavailable', error='unable to open database: read row budget exceeded'))
                        continue
                    names = [d[0] for d in cursor.description] if cursor.description else []
                    response = dict(rows=rows, names=names)
                    if not unchanged() or len(json.dumps(response)) > 8 * 1024 * 1024:
                        emit(dict(kind='unavailable', error='unable to open database: changed source or read byte budget exceeded'))
                    else:
                        emit(response)
                except sqlite3.Error as exc:
                    emit(dict(kind=type(exc).__name__, error=str(exc), code=getattr(exc, 'sqlite_errorcode', None)))
        finally:
            c.close()


def serve(path=None):
    global emit
    output = emit
    wal_snapshot = False
    if path is None:
        output(dict(ready=True))
    while True:
        if path is None:
            request = sys.stdin.readline(1024 * 1024 + 1)
            if not request or len(request) > 1024 * 1024:
                return
            request = json.loads(request)
            path = request['open']
            wal_snapshot = request.get('wal_snapshot') is True
        pending = []
        opened = False

        def session_output(message):
            nonlocal opened
            if 'journal' in message:
                opened = True
            if not opened and 'error' in message:
                pending.append(message)
            else:
                output(message)

        emit = session_output
        try:
            released = False
            try:
                released = main(path, wal_snapshot=wal_snapshot)
            except OSError as exc:
                response = dict(
                    kind='unavailable',
                    error='unable to open database: source changed or unavailable: ' + str(exc),
                    source_open_contention=getattr(exc, 'source_open_contention', False),
                )
                if response['source_open_contention']:
                    emit(response)
                else:
                    output(response)
                    return
            except sqlite3.Error as exc:
                output(dict(kind=type(exc).__name__, error=str(exc), code=getattr(exc, 'sqlite_errorcode', None)))
                return
        finally:
            emit = output
        for message in pending:
            output(dict(message, session_closed=True))
        if released:
            output(dict(rows=[], names=[]))
        path = None
        wal_snapshot = False


if __name__ == '__main__':
    serve(None if sys.argv[1] == '--ready' else sys.argv[1])
