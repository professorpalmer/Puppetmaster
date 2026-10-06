"""Durable session command ledger (Comet/Zeron steal — orchestration, not UI).

Send / steer / interrupt / respond_input are durable entries the *host*
executes. Rules are a Python port of Zeron's ``evaluate_command``:

1. Append-only per issuer; entries are immutable once written.
2. Host alone writes outcomes; a composer may cancel only its own pending
   entries.
3. Evaluation: processed-id dedupe → Skip; expired TTL → Expired; a newer
   same-kind pending steer/interrupt supersedes; an interrupt whose
   ``based_on.turn_id`` is already past → Superseded; else Execute.
   **Mark processed BEFORE execute** (idempotent recovery).

Maps onto Puppetmaster:

- ``run`` → job task / artifact admission (host creates or claims work)
- ``steer`` → follow-up instruction (continuous-planner / enqueue path)
- ``interrupt`` → scoped durable cancellation (:mod:`store_contracts`)
- ``respond_input`` → answer a pending worker question / gate

This module formalizes the ledger + pure evaluation. It does not invent an
ACP adapter layer and does not move orchestration state into Marionette /
Automaton / Discord viewports.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union

from puppetmaster.models import new_id, now_iso

COMMAND_DEFAULT_TTL_MS = 24 * 60 * 60 * 1000  # 24h
LEDGER_DIRNAME = "command-ledgers"
PROCESSED_FILENAME = "processed.json"


class SessionCommandKind(str, Enum):
    RUN = "run"
    STEER = "steer"
    INTERRUPT = "interrupt"
    RESPOND_INPUT = "respond_input"


class SessionCommandStatus(str, Enum):
    PENDING = "pending"
    APPLIED = "applied"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    CANCELLED = "cancelled"


class CommandDisposition(str, Enum):
    SKIP = "skip"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    EXECUTE = "execute"


@dataclass(frozen=True)
class CommandBasedOn:
    turn_id: Optional[str] = None
    frontier: Optional[str] = None


@dataclass
class SessionCommandEntry:
    id: str
    kind: SessionCommandKind
    issued_by: str
    issued_at_ms: int
    status: SessionCommandStatus = SessionCommandStatus.PENDING
    payload: dict[str, Any] = field(default_factory=dict)
    based_on: Optional[CommandBasedOn] = None
    expires_at_ms: Optional[int] = None
    resolution: Optional[str] = None
    # Host maps: task_id / cancel request_id / artifact_id once applied.
    mapped_task_id: Optional[str] = None
    mapped_request_id: Optional[str] = None
    mapped_artifact_id: Optional[str] = None

    def effective_expiry_ms(self) -> int:
        if self.expires_at_ms is not None:
            return int(self.expires_at_ms)
        return int(self.issued_at_ms) + COMMAND_DEFAULT_TTL_MS

    def to_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "kind": self.kind.value,
            "issued_by": self.issued_by,
            "issued_at_ms": self.issued_at_ms,
            "status": self.status.value,
            "payload": dict(self.payload),
            "expires_at_ms": self.expires_at_ms,
            "resolution": self.resolution,
            "mapped_task_id": self.mapped_task_id,
            "mapped_request_id": self.mapped_request_id,
            "mapped_artifact_id": self.mapped_artifact_id,
        }
        if self.based_on is not None:
            data["based_on"] = asdict(self.based_on)
        else:
            data["based_on"] = None
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SessionCommandEntry":
        based = raw.get("based_on")
        based_on = None
        if isinstance(based, dict):
            based_on = CommandBasedOn(
                turn_id=(str(based["turn_id"]) if based.get("turn_id") else None),
                frontier=(str(based["frontier"]) if based.get("frontier") else None),
            )
        return cls(
            id=str(raw["id"]),
            kind=SessionCommandKind(str(raw["kind"])),
            issued_by=str(raw.get("issued_by") or ""),
            issued_at_ms=int(raw["issued_at_ms"]),
            status=SessionCommandStatus(str(raw.get("status") or "pending")),
            payload=dict(raw.get("payload") or {}),
            based_on=based_on,
            expires_at_ms=(
                int(raw["expires_at_ms"]) if raw.get("expires_at_ms") is not None else None
            ),
            resolution=(str(raw["resolution"]) if raw.get("resolution") else None),
            mapped_task_id=(
                str(raw["mapped_task_id"]) if raw.get("mapped_task_id") else None
            ),
            mapped_request_id=(
                str(raw["mapped_request_id"]) if raw.get("mapped_request_id") else None
            ),
            mapped_artifact_id=(
                str(raw["mapped_artifact_id"]) if raw.get("mapped_artifact_id") else None
            ),
        )


@dataclass(frozen=True)
class EvaluationContext:
    is_processed: Callable[[str], bool]
    now_ms: int
    entries: tuple[SessionCommandEntry, ...]
    current_turn_id: Optional[str] = None
    turn_is_past: Callable[[str], bool] = lambda _turn: False


def evaluate_command(
    entry: SessionCommandEntry,
    cx: EvaluationContext,
) -> CommandDisposition:
    """Pure Rule-3 evaluation (Zeron port)."""
    if cx.is_processed(entry.id):
        return CommandDisposition.SKIP
    if cx.now_ms >= entry.effective_expiry_ms():
        return CommandDisposition.EXPIRED
    kind = entry.kind
    if kind in {SessionCommandKind.STEER, SessionCommandKind.INTERRUPT}:
        has_newer = any(
            other.id != entry.id
            and other.kind == kind
            and other.status == SessionCommandStatus.PENDING
            and other.issued_at_ms > entry.issued_at_ms
            for other in cx.entries
        )
        if has_newer:
            return CommandDisposition.SUPERSEDED
    if kind == SessionCommandKind.INTERRUPT and entry.based_on and entry.based_on.turn_id:
        turn_id = entry.based_on.turn_id
        is_current = cx.current_turn_id == turn_id
        if not is_current and cx.turn_is_past(turn_id):
            return CommandDisposition.SUPERSEDED
    return CommandDisposition.EXECUTE


class JobCommandLedger:
    """Append-only JSONL command ledger + processed-id set per job.

    Layout under the store root::

        command-ledgers/<job_id>.jsonl
        command-ledgers/<job_id>.processed.json
    """

    def __init__(self, root: Union[Path, str], job_id: str) -> None:
        self.root = Path(root)
        self.job_id = str(job_id).strip()
        if not self.job_id:
            raise ValueError("job_id required")
        self._dir = self.root / LEDGER_DIRNAME
        self._path = self._dir / f"{_safe_id(self.job_id)}.jsonl"
        self._processed_path = self._dir / f"{_safe_id(self.job_id)}.processed.json"
        self._lock = threading.Lock()

    def append(
        self,
        kind: Union[SessionCommandKind, str],
        *,
        issued_by: str,
        payload: Optional[dict[str, Any]] = None,
        based_on: Optional[CommandBasedOn] = None,
        command_id: Optional[str] = None,
        issued_at_ms: Optional[int] = None,
        expires_at_ms: Optional[int] = None,
    ) -> SessionCommandEntry:
        kind_enum = SessionCommandKind(kind) if not isinstance(kind, SessionCommandKind) else kind
        entry = SessionCommandEntry(
            id=command_id or new_id("cmd"),
            kind=kind_enum,
            issued_by=str(issued_by or "host"),
            issued_at_ms=int(issued_at_ms if issued_at_ms is not None else _now_ms()),
            payload=dict(payload or {}),
            based_on=based_on,
            expires_at_ms=expires_at_ms,
        )
        with self._lock:
            self._dir.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
        return entry

    def list_entries(self) -> list[SessionCommandEntry]:
        if not self._path.is_file():
            return []
        entries: list[SessionCommandEntry] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                entries.append(SessionCommandEntry.from_dict(data))
        return entries

    def processed_ids(self) -> set[str]:
        if not self._processed_path.is_file():
            return set()
        try:
            data = json.loads(self._processed_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return set()
        if not isinstance(data, dict):
            return set()
        ids = data.get("ids")
        if not isinstance(ids, list):
            return set()
        return {str(item) for item in ids}

    def mark_processed(self, command_id: str) -> None:
        """Mark BEFORE execute — crash-safe idempotence."""
        with self._lock:
            self._dir.mkdir(parents=True, exist_ok=True)
            ids = self.processed_ids()
            ids.add(str(command_id))
            payload = {"ids": sorted(ids), "updated_at": now_iso()}
            tmp = self._processed_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            tmp.replace(self._processed_path)

    def rewrite_status(
        self,
        command_id: str,
        status: SessionCommandStatus,
        *,
        resolution: Optional[str] = None,
        mapped_task_id: Optional[str] = None,
        mapped_request_id: Optional[str] = None,
        mapped_artifact_id: Optional[str] = None,
    ) -> Optional[SessionCommandEntry]:
        """Host-only outcome write: rewrite the matching line in place.

        Entries stay append-only for *new* commands; outcomes are host-owned
        mutations of status fields (Zeron rule 2).
        """
        with self._lock:
            entries = self.list_entries()
            updated: Optional[SessionCommandEntry] = None
            out: list[str] = []
            for entry in entries:
                if entry.id == command_id:
                    entry.status = status
                    if resolution is not None:
                        entry.resolution = resolution
                    if mapped_task_id is not None:
                        entry.mapped_task_id = mapped_task_id
                    if mapped_request_id is not None:
                        entry.mapped_request_id = mapped_request_id
                    if mapped_artifact_id is not None:
                        entry.mapped_artifact_id = mapped_artifact_id
                    updated = entry
                out.append(json.dumps(entry.to_dict(), sort_keys=True))
            if updated is None:
                return None
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
            tmp.replace(self._path)
            return updated

    def evaluate_pending(
        self,
        *,
        now_ms: Optional[int] = None,
        current_turn_id: Optional[str] = None,
        turn_is_past: Optional[Callable[[str], bool]] = None,
    ) -> list[tuple[SessionCommandEntry, CommandDisposition]]:
        entries = self.list_entries()
        processed = self.processed_ids()
        cx = EvaluationContext(
            is_processed=lambda cid: cid in processed,
            now_ms=int(now_ms if now_ms is not None else _now_ms()),
            entries=tuple(entries),
            current_turn_id=current_turn_id,
            turn_is_past=turn_is_past or (lambda _t: False),
        )
        results: list[tuple[SessionCommandEntry, CommandDisposition]] = []
        for entry in entries:
            if entry.status != SessionCommandStatus.PENDING:
                continue
            results.append((entry, evaluate_command(entry, cx)))
        return results

    def apply_disposition(
        self,
        entry: SessionCommandEntry,
        disposition: CommandDisposition,
    ) -> SessionCommandEntry:
        """Apply ledger-side effects for a disposition (still mark-before-execute)."""
        if disposition == CommandDisposition.SKIP:
            return entry
        if disposition == CommandDisposition.EXPIRED:
            updated = self.rewrite_status(
                entry.id, SessionCommandStatus.EXPIRED, resolution="ttl"
            )
            return updated or entry
        if disposition == CommandDisposition.SUPERSEDED:
            updated = self.rewrite_status(
                entry.id, SessionCommandStatus.SUPERSEDED, resolution="newer_same_kind"
            )
            return updated or entry
        # EXECUTE: mark processed first, leave pending→applied to the host after work.
        self.mark_processed(entry.id)
        return entry


def ledger_for_store(store: Any, job_id: str) -> JobCommandLedger:
    return JobCommandLedger(Path(store.root), job_id)


def _safe_id(job_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in job_id)


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)
