"""Fenced, non-mutating SQLite reads in an isolated descriptor owner.

No database copies. A quiet source is read through an immutable descriptor
that touches no file at all. A live one -- a live connection, or uncheckpointed
WAL frames -- is read by joining its committed WAL snapshot through a
``mode=ro`` open, so observers neither block writers nor fail while a job
writes. The guarantee in that mode is *never changes database content* rather
than *touches no file*: a WAL reader records read marks in the ``-shm`` index,
while the main database bytes and the WAL itself stay untouched and a
``mode=ro`` connection never checkpoints. Incomplete or damaged sidecars are
never created or repaired here; they stay unavailable.

Isolation matters: closing *any* source fd in the caller could release locks
held by another SQLite connection in that process.
"""
from __future__ import annotations

import atexit
from contextlib import contextmanager
from contextvars import ContextVar
from enum import IntEnum
import json
import os
import queue
import sqlite3
import subprocess
import sys
import threading
import time
import weakref
from pathlib import Path
from subprocess import Popen as ReaderProcess

from puppetmaster.readonly_worker import source_stamp
from puppetmaster.readonly_admission import ReaderAdmission, helper_spawn_gate
from puppetmaster.readonly_cleanup import CleanupRegistry


class ReadUnavailable(sqlite3.OperationalError):
    retry_after_ms = 100


class ReadTimeout(ReadUnavailable):
    """The helper exhausted its response budget without reporting contention."""


_read_deadline = ContextVar('readonly_deadline', default=None)


@contextmanager
def read_deadline(deadline):
    previous = _read_deadline.get()
    effective = previous
    if deadline is not None:
        effective = deadline if previous is None else min(previous, deadline)
    token = _read_deadline.set(effective)
    try:
        yield
    finally:
        _read_deadline.reset(token)


def _source_stamp(store):
    path = store.root / ('state.sqlite3' if store.backend_name == 'sqlite' else 'metadata.sqlite3')
    result = []
    for source in (store.root, *(Path(str(path) + suffix) for suffix in ('', '-wal', '-shm', '-journal'))):
        try:
            result.append(source_stamp(source))
        except FileNotFoundError:
            result.append(None)
    return tuple(result)


def selection(store):
    path = store.root / ('state.sqlite3' if store.backend_name == 'sqlite' else 'metadata.sqlite3')
    def identity(path):
        try:
            stat = path.stat()
            return stat.st_dev, stat.st_ino
        except FileNotFoundError:
            return None
    return identity(store.root), identity(path)


class ReadRow(tuple):
    def __new__(cls, values, names):
        row = super().__new__(cls, values)
        row.names = names
        return row

    def __getitem__(self, key):
        return super().__getitem__(self.names.index(key) if isinstance(key, str) else key)

    def keys(self):
        return self.names


class ReadCursor:
    def __init__(self, rows, names):
        self.rows = iter(ReadRow(row, names) for row in rows)

    def fetchone(self):
        return next(self.rows, None)

    def fetchmany(self, size=1):
        from itertools import islice
        return list(islice(self.rows, size))

    def fetchall(self):
        return list(self.rows)

    def __iter__(self):
        return self.rows


class _TransportState(IntEnum):
    STARTING = 0
    OPEN = 1
    CLOSING = 2
    REAPED = 3
    READER_EXITED = 4
    CLOSED = 5


_READER_EXIT_GRACE = 5.0


class _Transport:
    """Store-lifetime helper; idle helpers own pipes, never source descriptors."""
    def __init__(self, path, deadline=None):
        self.state = _TransportState.STARTING
        self.close_event = threading.Event()
        self.reader_exited = threading.Event()
        self.pid = os.getpid()
        self.busy = threading.Lock()
        self.responses = None
        self.reader = None
        self._reader_attempted = False
        process_factory = ReaderProcess
        process_args = [
            sys.executable, '-I', '-S',
            str(Path(__file__).with_name('readonly_worker.py')), '--ready',
        ]
        process_kwargs = dict(
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding='utf-8')
        self.process = None
        self.token = _cleanup.register(self, source_stamp(path)[:2])
        try:
            spawn = helper_spawn_gate(deadline)
            try:
                if isinstance(process_factory, type):
                    self.process = process_factory.__new__(process_factory)
                    process_factory.__init__(self.process, process_args, **process_kwargs)
                else:
                    self.process = process_factory(process_args, **process_kwargs)
            finally:
                spawn.release()
        except BaseException:
            _cleanup.retire(self.token)
            raise
        try:
            self.responses = queue.Queue(maxsize=1)
            def receive():
                try:
                    while True:
                        line = self.process.stdout.readline(8 * 1024 * 1024 + 1)
                        while not self.close_event.is_set():
                            try:
                                self.responses.put(line, timeout=.01)
                                break
                            except queue.Full:
                                continue
                        if self.close_event.is_set():
                            break
                        if not line or len(line) > 8 * 1024 * 1024:
                            break
                except (OSError, ValueError):
                    pass
                finally:
                    self.reader_exited.set()
            self.reader = threading.Thread(target=receive, daemon=True)
            self._reader_attempted = True
            self.reader.start()
            self.state = _TransportState.OPEN
        except BaseException:
            try:
                _cleanup.close(self.token, deadline=deadline)
            except BaseException:
                _cleanup.retire(self.token)
                raise
            raise

    @property
    def closed(self):
        return self.state == _TransportState.CLOSED

    @property
    def _reaped(self):
        return self.state >= _TransportState.REAPED

    def close(self, deadline=None):
        deadline = time.monotonic() + 1 if deadline is None else deadline
        def remaining():
            return max(0, deadline - time.monotonic())
        if self.closed or self.pid != os.getpid():
            return
        self.state = max(self.state, _TransportState.CLOSING)
        self.close_event.set()
        if self.process is None:
            raise ReadUnavailable('reader launch outcome unknown')
        if not self._reaped:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=remaining())
            except subprocess.TimeoutExpired:
                self.process.kill()
                try:
                    self.process.wait(timeout=remaining())
                except subprocess.TimeoutExpired as exc:
                    raise ReadTimeout('reader teardown timed out') from exc
            if self.process.poll() is None:
                raise ReadUnavailable('reader process remains alive')
        self.state = max(self.state, _TransportState.REAPED)
        if self._reader_attempted and not self.reader._started.wait(timeout=remaining()):
            raise ReadUnavailable('reader thread exit unconfirmed')
        # The reaped helper's pipe is at EOF, so the reader is already leaving.
        # Confirm on its own grace: under load terminate+wait can spend
        # the whole budget. Done means reader_exited (set in its finally,
        # after the last pipe read) or a dead thread (target never ran).
        if self.reader is not None and self.reader.ident is not None:
            exit_deadline = time.monotonic() + max(remaining(), _READER_EXIT_GRACE)
            while not self.reader_exited.is_set() and self.reader.is_alive():
                if time.monotonic() >= exit_deadline:
                    raise ReadUnavailable('reader thread remains alive')
                self.reader.join(timeout=.01)
        self.state = max(self.state, _TransportState.READER_EXITED)
        if not self.process.stdin.closed:
            self.process.stdin.close()
        if not self.process.stdout.closed:
            self.process.stdout.close()
        self.state = max(self.state, _TransportState.CLOSED)

    def ready(self, deadline):
        try:
            response = self.responses.get(timeout=max(0, deadline - time.monotonic()))
        except queue.Empty as exc:
            raise ReadTimeout('unable to open database: reader timed out') from exc
        if not response or json.loads(response) != {'ready': True}:
            raise ReadUnavailable('unable to open database: reader unavailable')


_cleanup = CleanupRegistry()


def _shutdown_cleanup():
    _cleanup.shutdown()


atexit.register(_shutdown_cleanup)


class _ReuseSlot:
    def __init__(self):
        self.busy = threading.Lock()
        self.transport = None
        self.finalizer = None
        self.token = None

    def discard(self, deadline=None):
        try:
            if self.transport is not None:
                _cleanup.close(self.token, deadline=deadline)
        except BaseException:
            _cleanup.retire(self.token)
            raise
        finally:
            if self.finalizer is not None:
                self.finalizer.detach()
                self.finalizer = None
            self.transport = None
            self.token = None

    def retire(self):
        if self.token is not None:
            _cleanup.retire(self.token)
        if self.finalizer is not None:
            self.finalizer.detach()
            self.finalizer = None
        self.transport = None
        self.token = None


_reuse_slots = weakref.WeakValueDictionary()
_reuse_lock = threading.Lock()
_inherited_cleanup = []


def _reset_reuse_after_fork():
    global _reuse_slots, _reuse_lock, _cleanup
    # A vanished thread may own either inherited lock. Only replace bookkeeping:
    # closing buffered pipes here can itself wait on a vanished reader thread.
    _inherited_cleanup.append((_reuse_slots, _cleanup))
    _reuse_slots = weakref.WeakValueDictionary()
    _reuse_lock = threading.Lock()
    _cleanup = CleanupRegistry()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_reset_reuse_after_fork)


def _reuse_slot(store, path):
    key = (os.getpid(), getattr(store, '_readonly_reuse_key',
                               os.path.normcase(os.path.realpath(path))))
    # Only registry bookkeeping is global; startup, waits and teardown are not.
    with _reuse_lock:
        slot = _reuse_slots.get(key)
        if slot is None:
            slot = _ReuseSlot()
            _reuse_slots[key] = slot
    store._readonly_slot = slot
    return slot


class ReadConnection:
    in_transaction = True
    row_factory = sqlite3.Row

    def __init__(self, store, timeout, *, reuse=False, attach_binding=False, launch_binding=False,
                 attach_deadline=None, ordinary_deadline=None, deadline=None, contention_window=None,
                 retry_deadline=None):
        self._pid = os.getpid()
        # Before any read budget starts: a bounded sweep of one abandoned helper.
        _cleanup.reap_idle(limit=1, deadline=time.monotonic() + 0.05)
        ordinary_window = .1 if reuse else 1.0
        self.store = store
        self.selected = selection(store)
        self._opened = False
        self.closed = False
        self._aborting = False
        self._leased = False
        self._authorizer = self._progress = self._trace = None
        self._attach_deadline = attach_deadline if attach_binding else None
        self._caller_deadline = _read_deadline.get()
        # A zero SQLite busy budget still permits one bounded helper response.
        # Interpreter startup is not evidence of database lock contention.
        self.timeout = 5.0 if timeout <= 0 else max(0.001, timeout)
        open_deadline = deadline if deadline is not None else time.monotonic() + max(0, timeout)
        write_deadline = open_deadline
        if attach_binding and attach_deadline is not None:
            open_deadline = min(open_deadline, attach_deadline)
            write_deadline = open_deadline
        response_deadline = (open_deadline if timeout > 0
                             else time.monotonic() + self.timeout)
        if retry_deadline is not None:
            write_deadline = min(write_deadline, retry_deadline)
        initial_response = True
        path = store.root / ('state.sqlite3' if store.backend_name == 'sqlite' else 'metadata.sqlite3')
        try:
            weakref.ref(store)
            cacheable = reuse
        except TypeError:
            cacheable = False
        self._slot = _reuse_slot(store, path) if cacheable else None
        queued_source = _source_stamp(store)[:2]
        admission_deadline = attach_deadline if attach_binding and attach_deadline is not None else open_deadline
        # Stable helper refresh may still try immediate admission after the
        # ordinary window, but cannot wait beyond it or extend the caller budget.
        admission_expiry = admission_deadline if timeout > 0 else response_deadline
        if retry_deadline is not None:
            # The outer loop selects the retry budget by failure kind. A new
            # helper must not recover the initial startup allowance.
            admission_deadline = min(admission_deadline, retry_deadline)
            admission_expiry = admission_deadline
        elif not (attach_binding or launch_binding) and contention_window is not None and contention_window[0] is not None:
            admission_deadline = min(admission_deadline, contention_window[0])
        if self.selected[1] is None:
            raise ReadUnavailable('unable to open database: source missing')
        self._cleanup_deadline = admission_expiry
        self._permit = None
        self._token = None
        if timeout > 0 and time.monotonic() >= admission_expiry:
            raise ReadTimeout('unable to open database: reader timed out')
        try:
            if self._slot is not None:
                if not self._slot.busy.acquire(timeout=max(0, admission_deadline - time.monotonic())):
                    raise ReadTimeout('unable to open database: reader timed out')
                try:
                    cached = self._slot.transport
                    reuse = cached is not None and not cached.closed
                    if not reuse:
                        self._slot.discard(deadline=self._cleanup_deadline)
                        self._slot.transport = _Transport(path, deadline=self._cleanup_deadline)
                        self._slot.token = self._slot.transport.token
                        self._slot.finalizer = weakref.finalize(self._slot, _cleanup.retire, self._slot.token)
                        self._slot.finalizer.atexit = False
                    self.transport = self._slot.transport
                    self._token = self._slot.token
                    self.transport.busy.acquire()
                    store._readonly_transport = self.transport
                except BaseException:
                    if self._aborting:
                        raise
                    try:
                        self._slot.discard(deadline=self._cleanup_deadline)
                    finally:
                        self._slot.busy.release()
                    raise
            else:
                reuse = False
                self.transport = _Transport(path, deadline=self._cleanup_deadline)
                self._token = self.transport.token
                self.transport.busy.acquire()
        except BaseException:
            if not self._aborting:
                self._release_permit()
            raise
        self._leased = True
        self._cached = self._slot is not None
        self.process = self.transport.process
        self.responses = self.transport.responses
        try:
            retry_error = None
            stamp = None
            wal_snapshot = False
            if not reuse:
                self.transport.ready(admission_expiry)
            _cleanup.recover(self.selected[1], admission_expiry)
            self._admit(path, admission_deadline)
            self._fence()
            for before, after in zip(queued_source, _source_stamp(store)[:2]):
                _metadata_fence(before, after)
            if timeout > 0 and time.monotonic() >= admission_expiry:
                raise ReadTimeout('unable to open database: reader timed out')
            if not launch_binding:
                if contention_window is not None:
                    if contention_window[0] is None:
                        contention_window[0] = min(open_deadline, time.monotonic() + ordinary_window)
                    ordinary_deadline = contention_window[0]
                elif ordinary_deadline is None:
                    ordinary_deadline = time.monotonic() + ordinary_window
                if not attach_binding:
                    write_deadline = min(write_deadline, ordinary_deadline)
            try:
                retry_source = source_stamp(path)
            except FileNotFoundError:
                retry_source = None
            stamp = _source_stamp(store) if reuse else None
            self.process.stdin.write(json.dumps(dict(
                open=str(path), wal_snapshot=wal_snapshot)) + '\n')
            self.process.stdin.flush()
            retry_error = None
            while True:
                if timeout > 0:
                    # Interpreter startup shares the whole attach budget. Once
                    # the helper responds, retries keep their shorter waits.
                    if initial_response and attach_deadline is not None:
                        self.timeout = attach_deadline - time.monotonic()
                    else:
                        self.timeout = min(timeout, response_deadline - time.monotonic())
                    if retry_deadline is not None:
                        self.timeout = min(self.timeout, retry_deadline - time.monotonic())
                else:
                    self.timeout = response_deadline - time.monotonic()
                initial_response = False
                try:
                    self.source_journal_mode = self._receive()['journal']
                    break
                except ReadTimeout:
                    if retry_error is not None and not (attach_binding or launch_binding):
                        raise retry_error
                    raise
                except sqlite3.OperationalError as exc:
                    write_race = getattr(exc, 'same_store_write', False)
                    topology_race = attach_binding and getattr(exc, 'launch_topology_change', False)
                    source_open_contention = (
                        (attach_binding or launch_binding) and
                        getattr(exc, 'source_open_contention', False)
                    )
                    locked = _locked(exc)
                    unavailable = isinstance(exc, ReadUnavailable) and any(
                        reason in str(exc) for reason in ('live sidecars', 'active reader'))
                    launch_transient = launch_binding and _launch_transient(exc)
                    if not (write_race or topology_race or source_open_contention or locked or
                            (launch_transient if launch_binding else unavailable)):
                        raise
                    candidate = (write_deadline if write_race or topology_race or source_open_contention or locked and not launch_binding
                                 else open_deadline if launch_transient or locked and launch_binding
                                 else ordinary_deadline)
                    retry_deadline = candidate if retry_deadline is None else min(retry_deadline, candidate)
                    remaining = retry_deadline - time.monotonic()
                    if remaining <= 0:
                        raise
                    # Retry in the same descriptor owner, after main() has
                    # closed its failed session. Do not fork a startup herd.
                    if not getattr(exc, 'session_closed', False):
                        raise
                    wal_snapshot = (wal_snapshot or
                        attach_binding and os.name == 'nt' and
                        getattr(exc, 'wal_snapshot', False))
                    self._release_permit()
                    time.sleep(min(.01 if write_race else .05, remaining))
                    remaining = retry_deadline - time.monotonic()
                    if remaining <= 0:
                        raise
                    self._admit(path, retry_deadline)
                    self._fence()
                    current_source = source_stamp(path)
                    _metadata_fence(retry_source, current_source)
                    retry_source = current_source
                    if write_race or locked:
                        retry_error = exc
                    response_deadline = retry_deadline
                    self.process.stdin.write(json.dumps(dict(
                        open=str(path), wal_snapshot=wal_snapshot)) + '\n')
                    self.process.stdin.flush()
            self._opened = True
            self.timeout = .1 if timeout <= 0 else timeout
            self._cleanup_deadline = time.monotonic() + self.timeout
            if self._caller_deadline is not None:
                self._cleanup_deadline = min(self._cleanup_deadline, self._caller_deadline)
        except BaseException as exc:
            # Caller/open budgets may be exhausted, but teardown still needs a
            # fresh bounded interval so the helper cannot retain source locks.
            self._cleanup_deadline = time.monotonic() + 1
            cleanup_error = None
            try:
                self._abort()
            except BaseException as error:
                cleanup_error = error
            if (reuse and isinstance(exc, ReadUnavailable) and
                    str(exc) == 'unable to open database: source changed' and
                    _source_stamp(store) == stamp):
                exc.stable_helper_stamp = stamp
                exc.readonly_retry_deadline = retry_deadline
                exc.readonly_retry_error = retry_error
            if cleanup_error is not None:
                raise exc from cleanup_error
            raise

    def _admit(self, path, deadline):
        try:
            self._permit = ReaderAdmission(path, deadline, clock=time, selected=self.selected[1])
        except BaseException as exc:
            self._permit = getattr(exc, 'admission_owner', None)
            if self._permit is not None:
                _cleanup.hold(self._token, self._permit)
            raise
        _cleanup.hold(self._token, self._permit)

    def _fence(self):
        from puppetmaster.identity import StoreIdentityError
        if selection(self.store) != self.selected or self.selected != self.store._read_selection:
            raise StoreIdentityError('store removed or replaced since selection; explicitly reopen')

    def _receive(self):
        response_deadline = time.monotonic() + self.timeout
        while True:
            timeout = response_deadline - time.monotonic()
            if self._caller_deadline is not None:
                timeout = min(timeout, self._caller_deadline - time.monotonic())
            if self._attach_deadline is not None:
                timeout = min(timeout, self._attach_deadline - time.monotonic())
            if timeout <= 0:
                raise ReadTimeout('unable to open database: reader timed out')
            try:
                line = self.responses.get(timeout=timeout)
            except queue.Empty as exc:
                raise ReadTimeout('unable to open database: reader timed out') from exc
            if not self._opened:
                self._fence()
            if not line or len(line) > 8 * 1024 * 1024:
                raise ReadUnavailable('unable to open database: reader unavailable')
            result = json.loads(line)
            if 'event' in result:
                callback = {'authorize': self._authorizer, 'progress': self._progress,
                            'trace': self._trace}[result['event']]
                answer = callback(*result['args']) if callback is not None else 0
                self.process.stdin.write(json.dumps(answer or 0) + '\n')
                self.process.stdin.flush()
                continue
            if 'error' in result:
                kind = result.get('kind')
                error = (ReadUnavailable if kind == 'unavailable' else
                         sqlite3.OperationalError if kind == 'OperationalError' else sqlite3.DatabaseError)(result['error'])
                error.session_closed = result.get('session_closed') is True
                error.launch_topology_change = result.get('launch_topology_change') is True
                error.source_open_contention = result.get('source_open_contention') is True
                error.same_store_write = (kind == 'unavailable' and
                    result['error'] == 'unable to open database: source changed' and
                    result.get('same_store_write') is True)
                error.wal_snapshot = result.get('wal_snapshot') is True
                if result.get('code') is not None:
                    error.sqlite_errorcode = result['code']
                raise error
            return result

    def execute(self, sql, parameters=()):
        self._check_process()
        try:
            self.process.stdin.write(json.dumps(dict(sql=sql, parameters=parameters)) + '\n')
            self.process.stdin.flush()
            result = self._receive()
            return ReadCursor(result['rows'], result['names'])
        except BaseException:
            self._abort()
            raise

    def _control(self, name, value):
        self._check_process()
        try:
            self.process.stdin.write(json.dumps(dict(control=name, value=value)) + '\n')
            self.process.stdin.flush()
            self._receive()
        except BaseException:
            self._abort()
            raise

    def set_authorizer(self, callback):
        self._authorizer = callback
        self._control('authorizer', callback is not None)

    def set_progress_handler(self, callback, instructions):
        self._progress = callback
        self._control('progress', instructions if callback is not None else 0)

    def set_trace_callback(self, callback):
        self._trace = callback
        self._control('trace', callback is not None)

    def _check_process(self):
        if self._pid != os.getpid():
            raise ReadUnavailable('reader belongs to another process; explicitly reopen')

    def _release_permit(self):
        if self._permit is not None:
            permit = self._permit
            try:
                permit.release()
            except BaseException as exc:
                # Constructor failures also need to leave a reachable owner.
                exc.admission_owner = permit
                raise
            self._permit = None
            _cleanup.released(self._token, permit)

    def _release_lease(self):
        if self._leased:
            self._leased = False
            self.transport.busy.release()
            if self._slot is not None:
                self._slot.busy.release()

    def _abort(self):
        if self._pid != os.getpid():
            self.closed = True
            return
        if self.closed:
            self._release_permit()
            return
        # Source-operation budgets do not double as teardown budgets. A
        # long-lived connection must still get one bounded reap attempt.
        self._cleanup_deadline = time.monotonic() + 1
        self._aborting = True
        try:
            if self._slot is not None:
                self._slot.discard(deadline=self._cleanup_deadline)
            else:
                _cleanup.close(self._token, deadline=self._cleanup_deadline)
        except BaseException:
            _cleanup.retire(self._token)
            self._release_lease()
            self._slot = None
            raise
        self.closed = True
        self._release_lease()
        self._release_permit()

    def close(self):
        # Inherited pipes remain inert until child exit/exec. Do not flush,
        # signal the parent's helper, or touch inherited Python I/O locks.
        if self._pid != os.getpid():
            self.closed = True
            return
        if self.closed:
            self._release_permit()
            return
        if self._aborting or not self._cached or any((self._authorizer, self._progress, self._trace)):
            self._abort()
            return
        self._control('release', True)
        self.closed = True
        try:
            self._release_permit()
        except BaseException:
            _cleanup.retire(self._token)
            self._release_lease()
            if self._slot is not None:
                self._slot.retire()
                self._slot = None
            raise
        finally:
            if self._leased:
                self._release_lease()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _locked(exc):
    code = getattr(exc, 'sqlite_errorcode', None)
    return ((isinstance(code, int) and (code & 0xff) in (5, 6)) if code is not None else
            str(exc) in ('database is locked', 'database table is locked', 'database schema is locked'))


def _launch_transient(exc):
    code = getattr(exc, 'sqlite_errorcode', None)
    return ((code is None or _locked(exc)) and isinstance(exc, ReadUnavailable) and (
        str(exc) in ('unable to open database: active reader; sidecars may be missing',
                     'unable to open database: live sidecars; retry after checkpoint') or
        (str(exc) == 'unable to open database: source changed' and
         getattr(exc, 'launch_topology_change', False))))


def _source_open_contention(exc):
    return bool(
        getattr(exc, 'source_open_contention', False)
        or getattr(exc, 'winerror', None) in (5, 32, 33)
    )


def _metadata_fence(before, after):
    from puppetmaster.identity import StoreMetadataDrift
    if (before is not None and after is not None and
            before[2:4] == after[2:4] and before[4] != after[4]):
        raise StoreMetadataDrift('store source metadata changed during binding')


def _transient_read(exc):
    if isinstance(exc, ReadUnavailable):
        return not isinstance(exc, ReadTimeout) and any(
            reason in str(exc) for reason in ('source changed', 'active reader', 'live sidecars'))
    if isinstance(exc, sqlite3.OperationalError):
        return str(exc) == 'database is locked'
    return not isinstance(exc, FileNotFoundError) and _source_open_contention(exc)


def retry_transient(read, deadline):
    """Run a control-plane read, retrying source contention until deadline.

    Ordinary listing reads stay fail-fast and report unavailable. Identity and
    contract reads must not fail because a concurrent writer held or moved
    the source while the helper opened it. Replacement still raises: the
    retry reopens through the identity fence.
    """
    while True:
        try:
            return read()
        except (sqlite3.OperationalError, OSError) as exc:
            if not _transient_read(exc) or time.monotonic() >= deadline:
                raise
            time.sleep(.01)


def connect(store, *, timeout=5, reuse=False, launch_binding=False, attach_binding=False,
            attach_deadline=None):
    from puppetmaster.identity import StoreIdentityError
    if selection(store) != store._read_selection:
        raise StoreIdentityError('store removed or replaced since selection; explicitly reopen')
    started = time.monotonic()
    deadline = (attach_deadline if attach_binding and attach_deadline is not None
                else started + max(0, timeout))
    caller_deadline = _read_deadline.get()
    if caller_deadline is not None:
        deadline = min(deadline, caller_deadline)
    if launch_binding:
        reuse = False
    # Ordinary reads keep their short window even for confirmed contention.
    # Only binding operations may spend the wider caller budget.
    contention_window = [None]
    retry_deadline = None
    refresh_stamp = None
    refresh_error = None
    launch_source = None
    while True:
        if launch_binding and selection(store) != store._read_selection:
            raise StoreIdentityError('store removed or replaced since selection; explicitly reopen')
        try:
            if launch_binding:
                current_source = _source_stamp(store)[1]
                if launch_source is not None:
                    # Reject observed metadata-only drift; ctime is not an operation counter.
                    _metadata_fence(launch_source, current_source)
                launch_source = current_source
            if refresh_stamp is not None and _source_stamp(store) != refresh_stamp:
                raise ReadUnavailable('unable to open database: source changed')
            remaining = min(timeout, (deadline if retry_deadline is None else retry_deadline) - time.monotonic())
            if timeout > 0 and remaining <= 0:
                raise ReadTimeout('unable to open database: reader timed out')
            connection = ReadConnection(store, (remaining if attach_binding else max(.001, remaining)) if timeout > 0 else 0,
                                        reuse=reuse, attach_binding=attach_binding, launch_binding=launch_binding,
                                        attach_deadline=attach_deadline,
                                        deadline=deadline, contention_window=contention_window,
                                        retry_deadline=retry_deadline)
            if refresh_stamp is not None and _source_stamp(store) != refresh_stamp:
                connection._abort()
                raise ReadUnavailable('unable to open database: source changed')
            return connection
        except OSError as exc:
            if (
                isinstance(exc, FileNotFoundError)
                or not (attach_binding or launch_binding)
                or not _source_open_contention(exc)
            ):
                raise
            if contention_window[0] is None:
                contention_window[0] = min(deadline, time.monotonic() + (0.1 if reuse else 1.0))
            next_deadline = (
                attach_deadline if attach_binding and attach_deadline is not None
                else deadline
            )
            retry_deadline = next_deadline if retry_deadline is None else min(retry_deadline, next_deadline)
            if time.monotonic() >= retry_deadline:
                raise
            time.sleep(min(.05, max(0, retry_deadline - time.monotonic())))
            if time.monotonic() >= retry_deadline:
                raise
        except sqlite3.OperationalError as exc:
            if isinstance(exc, ReadTimeout) and refresh_error is not None and not (attach_binding or launch_binding):
                raise refresh_error
            stamp = getattr(exc, 'stable_helper_stamp', None)
            # A reused helper can fail after its short retry window expires.
            # Refresh once only if the full source (including directory/sidecar
            # history) remained unchanged through helper teardown.
            if refresh_stamp is None and stamp is not None and time.monotonic() < deadline:
                refresh_stamp = stamp
                established = getattr(exc, 'readonly_retry_deadline', None)
                if established is not None:
                    retry_deadline = established if retry_deadline is None else min(retry_deadline, established)
                    refresh_error = getattr(exc, 'readonly_retry_error', None) or exc
                continue
            # Admission survives incomplete teardown. Codes take precedence over messages.
            locked = _locked(exc)
            unavailable = isinstance(exc, ReadUnavailable) and any(
                reason in str(exc) for reason in ('live sidecars', 'active reader'))
            launch_transient = launch_binding and _launch_transient(exc)
            if launch_binding:
                unavailable = launch_transient
            if (locked or unavailable) and contention_window[0] is None:
                contention_window[0] = min(deadline, time.monotonic() + (0.1 if reuse else 1.0))
            next_deadline = (attach_deadline if locked and attach_binding and attach_deadline is not None
                             else deadline if launch_transient or locked and (launch_binding or attach_binding)
                             else (contention_window[0] if contention_window[0] is not None else deadline))
            retry_deadline = next_deadline if retry_deadline is None else min(retry_deadline, next_deadline)
            if not (locked or unavailable) or time.monotonic() >= retry_deadline:
                raise
            time.sleep(min(.05, max(0, retry_deadline - time.monotonic())))
            if time.monotonic() >= retry_deadline:
                raise
