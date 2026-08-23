"""Named cells: inspectable SQLite + serial inbox + hibernate/alarm.

This is the paying slice of the Durable Objects programming model
(one named cell, one private SQLite, one event at a time, idle
hibernate, alarm resume, sqlite3+grep evidence) without celld's
fleet: no S3 coordinator, no LTX replication, no V8 isolates, no
Wrangler bundles, no distributed CAS ownership.

Built on top of the existing SwarmStore / task-lease stack. Cells
live as sibling files under ``<state>/cells/<id>.sqlite`` and do
not replace ``state.sqlite3``.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional, Union

from puppetmaster.fs_permissions import chmod_private_file, mkdir_private
from puppetmaster.models import now_iso

SQLITE_MAGIC = b"SQLite format 3\x00"
_CELL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,190}$")
_LEASE_SECONDS = 30.0
_BUSY_TIMEOUT_MS = 5000

EventHandler = Callable[[dict[str, Any]], Any]


class CellBusyError(RuntimeError):
    """Another handler already holds this cell's single-thread lease."""


class CellNotFoundError(KeyError):
    """No on-disk cell file for this id."""


class InvalidCellIdError(ValueError):
    """cell_id failed the portable filename / inspectability rules."""


def normalize_cell_id(cell_id: str) -> str:
    text = (cell_id or "").strip()
    if not _CELL_ID_RE.match(text):
        raise InvalidCellIdError(
            f"invalid cell id {cell_id!r}: use letters, digits, . _ : -"
        )
    return text


def cells_dir_for(root: Union[Path, str]) -> Path:
    return Path(root) / "cells"


def cell_path(root: Union[Path, str], cell_id: str) -> Path:
    return cells_dir_for(root) / f"{normalize_cell_id(cell_id)}.sqlite"


def is_sqlite_file(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        return handle.read(16) == SQLITE_MAGIC


class CellRegistry:
    """Per-state-dir registry of named cells.

    In-memory workers live in ``_live`` and are dropped on hibernate.
    The cell itself remains the sqlite file.
    """

    def __init__(self, root: Union[Path, str]) -> None:
        self.root = Path(root)
        self.cells_dir = cells_dir_for(self.root)
        self._live: dict[str, dict[str, Any]] = {}
        self._thread_locks: dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()

    def _thread_lock(self, cell_id: str) -> threading.Lock:
        with self._registry_lock:
            lock = self._thread_locks.get(cell_id)
            if lock is None:
                lock = threading.Lock()
                self._thread_locks[cell_id] = lock
            return lock

    def ensure(self, cell_id: str) -> Path:
        cell_id = normalize_cell_id(cell_id)
        mkdir_private(self.cells_dir)
        path = cell_path(self.root, cell_id)
        with self._connect(path) as connection:
            self._init_schema(connection, cell_id)
        chmod_private_file(path)
        return path

    @contextmanager
    def _connect(self, path: Path):
        """Open a cell DB, apply PRAGMAs, and ALWAYS close the handle.

        ``with sqlite3.connect(...) as conn`` is a *transaction* context
        manager — it commits/rolls back but leaves the OS file handle
        open. Windows holds a mandatory lock on that handle, so a later
        unlink / TemporaryDirectory cleanup fails with ``WinError 32``.
        Fetch ``journal_mode`` so the result row does not linger unread.
        """
        connection = sqlite3.connect(str(path), timeout=_BUSY_TIMEOUT_MS / 1000.0)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            connection.execute("PRAGMA journal_mode = WAL").fetchone()
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
        finally:
            connection.close()

    def close(self) -> None:
        """Drop live workers and checkpoint every cell file.

        Call this before deleting a temporary state dir so Windows can
        unlink ``*.sqlite`` (and ``-wal`` / ``-shm``) after inspect.
        Connections are already closed per ``_connect``; the checkpoint
        releases WAL sidecar locks that otherwise race teardown.
        """
        self._live.clear()
        if not self.cells_dir.is_dir():
            return
        for path in sorted(self.cells_dir.glob("*.sqlite")):
            connection = sqlite3.connect(str(path), timeout=_BUSY_TIMEOUT_MS / 1000.0)
            try:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            finally:
                connection.close()

    def _init_schema(self, connection: sqlite3.Connection, cell_id: str) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS cell_meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS inbox (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              enqueued_at TEXT NOT NULL,
              kind TEXT NOT NULL,
              payload TEXT NOT NULL,
              status TEXT NOT NULL,
              processed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_inbox_status_id
              ON inbox(status, id);
            """
        )
        defaults = {
            "cell_id": cell_id,
            "hibernating": "0",
            "next_alarm": "",
            "lease_owner": "",
            "lease_expires_at": "",
            "lease_id": "",
        }
        for key, value in defaults.items():
            connection.execute(
                """
                INSERT INTO cell_meta(key, value)
                VALUES(?, ?)
                ON CONFLICT(key) DO NOTHING
                """,
                (key, value),
            )
        connection.commit()

    def _meta_map(self, connection: sqlite3.Connection) -> dict[str, str]:
        return {
            row["key"]: row["value"]
            for row in connection.execute("SELECT key, value FROM cell_meta")
        }

    def _set_meta(self, connection: sqlite3.Connection, **fields: str) -> None:
        for key, value in fields.items():
            connection.execute(
                """
                INSERT INTO cell_meta(key, value)
                VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )

    def _inbox_depth(self, connection: sqlite3.Connection) -> int:
        row = connection.execute(
            "SELECT COUNT(*) AS n FROM inbox WHERE status = 'pending'"
        ).fetchone()
        return int(row["n"]) if row is not None else 0

    def _lease_held(self, meta: dict[str, str], *, now: float, owner: str) -> bool:
        holder = (meta.get("lease_owner") or "").strip()
        if not holder:
            return False
        expires_raw = (meta.get("lease_expires_at") or "").strip()
        try:
            expires = float(expires_raw) if expires_raw else 0.0
        except ValueError:
            expires = 0.0
        if expires and expires <= now:
            return False
        return holder != owner

    def enqueue(
        self,
        cell_id: str,
        kind: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> int:
        cell_id = normalize_cell_id(cell_id)
        path = self.ensure(cell_id)
        body = json.dumps(payload or {}, default=str, separators=(",", ":"))
        with self._connect(path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT INTO inbox(enqueued_at, kind, payload, status)
                VALUES(?, ?, ?, 'pending')
                """,
                (now_iso(), kind, body),
            )
            event_id = int(cursor.lastrowid)
            connection.commit()
        return event_id

    def process_one(
        self,
        cell_id: str,
        *,
        owner: str = "local",
        handler: Optional[EventHandler] = None,
        lease_seconds: float = _LEASE_SECONDS,
        now: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """Dequeue and handle exactly one pending event.

        Raises :class:`CellBusyError` when another owner already holds
        the cell lease so two handlers never run concurrently.
        """
        cell_id = normalize_cell_id(cell_id)
        path = self.ensure(cell_id)
        thread_lock = self._thread_lock(cell_id)
        if not thread_lock.acquire(blocking=False):
            raise CellBusyError(f"cell {cell_id} already has an in-process handler")
        current = time.time() if now is None else now
        event: Optional[dict[str, Any]] = None
        try:
            with self._connect(path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                meta = self._meta_map(connection)
                if self._lease_held(meta, now=current, owner=owner):
                    connection.rollback()
                    raise CellBusyError(
                        f"cell {cell_id} leased by {meta.get('lease_owner')!r}"
                    )
                row = connection.execute(
                    """
                    SELECT id, enqueued_at, kind, payload, status
                    FROM inbox
                    WHERE status = 'pending'
                    ORDER BY id ASC
                    LIMIT 1
                    """
                ).fetchone()
                if row is None:
                    self._set_meta(
                        connection,
                        lease_owner="",
                        lease_expires_at="",
                        lease_id="",
                    )
                    self._hibernate_locked(connection, cell_id)
                    connection.commit()
                    return None
                event = {
                    "id": int(row["id"]),
                    "enqueued_at": row["enqueued_at"],
                    "kind": row["kind"],
                    "payload": json.loads(row["payload"] or "{}"),
                    "status": "processing",
                }
                self._set_meta(
                    connection,
                    hibernating="0",
                    lease_owner=owner,
                    lease_expires_at=str(current + lease_seconds),
                    lease_id=f"lease-{owner}-{event['id']}",
                )
                connection.execute(
                    "UPDATE inbox SET status = 'processing' WHERE id = ?",
                    (event["id"],),
                )
                connection.commit()
            self._live[cell_id] = {
                "owner": owner,
                "awake_at": current,
                "processing_id": event["id"],
            }
            try:
                if handler is not None:
                    handler(event)
                final_status = "done"
            except Exception:
                final_status = "pending"
                raise
            finally:
                with self._connect(path) as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    if final_status == "done":
                        connection.execute(
                            """
                            UPDATE inbox
                            SET status = 'done', processed_at = ?
                            WHERE id = ?
                            """,
                            (now_iso(), event["id"]),
                        )
                    else:
                        connection.execute(
                            "UPDATE inbox SET status = 'pending' WHERE id = ?",
                            (event["id"],),
                        )
                    self._set_meta(
                        connection,
                        lease_owner="",
                        lease_expires_at="",
                        lease_id="",
                    )
                    if final_status == "done" and self._inbox_depth(connection) == 0:
                        self._hibernate_locked(connection, cell_id)
                    connection.commit()
            return event
        finally:
            thread_lock.release()

    def _hibernate_locked(self, connection: sqlite3.Connection, cell_id: str) -> None:
        self._set_meta(connection, hibernating="1")
        self._live.pop(cell_id, None)

    def hibernate(self, cell_id: str) -> dict[str, Any]:
        cell_id = normalize_cell_id(cell_id)
        path = self.ensure(cell_id)
        with self._connect(path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._hibernate_locked(connection, cell_id)
            connection.commit()
        return self.status(cell_id)

    def set_alarm(self, cell_id: str, due_at: Union[float, str]) -> dict[str, Any]:
        cell_id = normalize_cell_id(cell_id)
        path = self.ensure(cell_id)
        if isinstance(due_at, (int, float)):
            due_value = f"{float(due_at):.6f}"
        else:
            due_value = str(due_at).strip()
        with self._connect(path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._set_meta(connection, next_alarm=due_value)
            connection.commit()
        return self.status(cell_id)

    def wake(self, cell_id: str, *, reason: str = "manual", now: Optional[float] = None) -> dict[str, Any]:
        cell_id = normalize_cell_id(cell_id)
        path = self.ensure(cell_id)
        current = time.time() if now is None else now
        with self._connect(path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._set_meta(connection, hibernating="0")
            connection.commit()
        self._live[cell_id] = {"owner": "wake", "awake_at": current, "reason": reason}
        return self.status(cell_id)

    def _alarm_due(self, next_alarm: str, now: float) -> bool:
        text = (next_alarm or "").strip()
        if not text:
            return False
        try:
            return float(text) <= now
        except ValueError:
            return False

    def tick(self, *, now: Optional[float] = None) -> list[dict[str, Any]]:
        """Wake cells whose persisted alarm is due. Interned poll / cell-tick."""
        current = time.time() if now is None else now
        woken: list[dict[str, Any]] = []
        if not self.cells_dir.is_dir():
            return woken
        for path in sorted(self.cells_dir.glob("*.sqlite")):
            cell_id = path.stem
            try:
                normalize_cell_id(cell_id)
            except InvalidCellIdError:
                continue
            with self._connect(path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                meta = self._meta_map(connection)
                if not self._alarm_due(meta.get("next_alarm") or "", current):
                    connection.rollback()
                    continue
                self._set_meta(connection, hibernating="0", next_alarm="")
                connection.execute(
                    """
                    INSERT INTO inbox(enqueued_at, kind, payload, status)
                    VALUES(?, 'alarm', ?, 'pending')
                    """,
                    (
                        now_iso(),
                        json.dumps({"due_at": meta.get("next_alarm"), "woken_at": current}),
                    ),
                )
                connection.commit()
            self._live[cell_id] = {
                "owner": "alarm",
                "awake_at": current,
                "reason": "alarm",
            }
            woken.append(self.status(cell_id))
        return woken

    def status(self, cell_id: str) -> dict[str, Any]:
        cell_id = normalize_cell_id(cell_id)
        path = cell_path(self.root, cell_id)
        if not path.exists():
            raise CellNotFoundError(cell_id)
        with self._connect(path) as connection:
            meta = self._meta_map(connection)
            depth = self._inbox_depth(connection)
        next_alarm = meta.get("next_alarm") or None
        if next_alarm == "":
            next_alarm = None
        return {
            "cell_id": cell_id,
            "path": str(path.resolve()),
            "inbox_depth": depth,
            "hibernating": meta.get("hibernating") == "1",
            "next_alarm": next_alarm,
            "live": cell_id in self._live,
            "sqlite": is_sqlite_file(path),
        }

    def inspect(self, cell_id: str, *, limit: int = 20) -> dict[str, Any]:
        payload = self.status(cell_id)
        path = Path(payload["path"])
        with self._connect(path) as connection:
            rows = connection.execute(
                """
                SELECT id, enqueued_at, kind, payload, status, processed_at
                FROM inbox
                ORDER BY id ASC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        payload["inbox"] = [
            {
                "id": int(row["id"]),
                "enqueued_at": row["enqueued_at"],
                "kind": row["kind"],
                "payload": json.loads(row["payload"] or "{}"),
                "status": row["status"],
                "processed_at": row["processed_at"],
            }
            for row in rows
        ]
        return payload

    def list_cells(self) -> list[dict[str, Any]]:
        if not self.cells_dir.is_dir():
            return []
        listed: list[dict[str, Any]] = []
        for path in sorted(self.cells_dir.glob("*.sqlite")):
            try:
                listed.append(self.status(path.stem))
            except (InvalidCellIdError, CellNotFoundError):
                continue
        return listed


def registry_for_store(store: Any) -> CellRegistry:
    root = getattr(store, "root", None)
    if root is None:
        raise TypeError("store has no root")
    return CellRegistry(root)


def interned_poll(store: Any, *, now: Optional[float] = None) -> list[dict[str, Any]]:
    """Process due cell alarms from an in-process daemon poll."""
    registry = registry_for_store(store)
    try:
        return registry.tick(now=now)
    finally:
        registry.close()
