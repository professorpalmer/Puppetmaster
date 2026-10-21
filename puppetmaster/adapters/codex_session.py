"""Small native Codex app-server transport used by the Codex adapter.

This is deliberately not an app-server SDK.  It owns one ephemeral stdio
session, preserves JSON-RPC request correlation, and provides the one feature
``codex exec`` cannot: steering an active turn.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from puppetmaster.redaction import redact_secrets


_CAPTURE_CHARS = 120000
_APPROVAL_METHODS = {"commandExecution/requestApproval", "fileChange/requestApproval", "execCommandApproval", "applyPatchApproval", "permissions/requestApproval"}


@dataclass
class CodexSessionResult:
    status: str
    thread_id: Optional[str] = None
    turn_id: Optional[str] = None
    messages: List[str] = field(default_factory=list)
    error: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    raw_protocol: str = ""
    protocol_log_path: Optional[str] = None
    queued_steering: List[str] = field(default_factory=list)
    accepted_steering: List[str] = field(default_factory=list)
    rejected_steering: List[str] = field(default_factory=list)


def _bounded(parts: List[str], line: str) -> None:
    parts.append(redact_secrets(line) or "")
    while sum(map(len, parts)) > _CAPTURE_CHARS:
        parts.pop(0)


def _text_from(value: Any) -> List[str]:
    """Extract agent text across the small set of app-server item shapes."""
    if not isinstance(value, dict):
        return []
    item = value.get("item", value)
    if not isinstance(item, dict):
        return []
    kind = str(item.get("type", "")).replace("_", "").lower()
    if kind not in ("agentmessage", "agent_message", "assistantmessage", "assistant_message"):
        return []
    text = item.get("text")
    if isinstance(text, str):
        return [text]
    content = item.get("content")
    if isinstance(content, list):
        return [str(x.get("text")) for x in content if isinstance(x, dict) and isinstance(x.get("text"), str)]
    return []


def _usage(params: Any) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    usage = params.get("tokenUsage", {}) if isinstance(params, dict) else {}
    total = usage.get("total", usage.get("last", {})) if isinstance(usage, dict) else {}
    if not isinstance(total, dict):
        return None, None, None
    def number(name: str) -> Optional[int]:
        value = total.get(name)
        return int(value) if isinstance(value, (int, float)) else None
    return number("inputTokens"), number("outputTokens"), number("cachedInputTokens")


def run_codex_session(
    command_prefix: Union[str, Sequence[str]], cwd: Union[str, Path], prompt: str,
    model: Optional[str] = None, sandbox: str = "workspace-write", approval_policy: str = "never",
    config: Optional[Dict[str, Any]] = None,
    effort: Optional[str] = None, ephemeral: bool = True, timeout: float = 600, env: Optional[Dict[str, str]] = None,
    log: Optional[Callable[[str], None]] = None, event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    cancellation_check: Optional[Callable[[], bool]] = None,
    pending_steering: Optional[Callable[[], Iterable[Any]]] = None,
    protocol_log_path: Optional[Union[str, Path]] = None,
    max_output_bytes: Optional[int] = None,
    on_spawn: Optional[Callable[[List[str], Dict[str, str], int], None]] = None,
) -> CodexSessionResult:
    """Run one app-server thread (ephemeral by default) and optionally steer its active turn.

    ``pending_steering`` yields ``(id, text, acknowledge)`` tuples or mappings
    with those names.  ``acknowledge`` receives ``(id, state, details)`` where
    state is queued, accepted, or failed.  A response to ``turn/steer`` means
    delivery acceptance only; it never claims that the model applied text.

    The app-server runs as an owned process (see ``win_process.popen_owned``),
    so teardown also stops the shell commands it started. Output past
    ``max_output_bytes`` ends the session with status ``output_limit``.
    ``on_spawn`` receives the launched argv, env and pid.
    """
    from puppetmaster.models import new_id
    from puppetmaster.win_process import cleanup_owned_process, close_owned_process, popen_owned

    command = [command_prefix] if isinstance(command_prefix, str) else list(command_prefix)
    command += ["app-server", "--listen", "stdio://"]
    child_env = dict(os.environ)
    child_env.update(env or {})
    # A nested worker must not discover/re-enter the parent MCP transport.
    child_env["PUPPETMASTER_WORKER"] = "1"
    child_env["PUPPETMASTER_AUTO_INVOKE_DISABLED"] = "1"
    owner = new_id("process")
    child_env["PUPPETMASTER_PROCESS_OWNER"] = owner
    proc = popen_owned(command, cwd=str(Path(cwd).resolve()), env=child_env, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
        bufsize=1, start_new_session=True)
    if on_spawn:
        on_spawn(list(command), child_env, proc.pid)
    inbox: "queue.Queue[Tuple[str, Optional[str]]]" = queue.Queue()
    captured: List[str] = []
    output_bytes = [0]
    output_lock = threading.Lock()
    def reader(name: str, stream: Any) -> None:
        try:
            for line in iter(stream.readline, ""):
                with output_lock:
                    output_bytes[0] += len(line.encode("utf-8", errors="replace"))
                inbox.put((name, line))
        finally:
            inbox.put((name, None))
    for name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr)):
        threading.Thread(target=reader, args=(name, stream), daemon=True).start()
    next_id = 1; waiting: Dict[int, Tuple[str, Optional[Callable[..., Any]]]] = {}
    result = CodexSessionResult(status="failed", protocol_log_path=str(protocol_log_path) if protocol_log_path else None)
    active_turn: Optional[str] = None; stdout_eof = False; stderr_eof = False; started = False
    deadline = time.monotonic() + max(0.01, timeout)
    def emit(obj: Dict[str, Any]) -> int:
        nonlocal next_id
        request_id = next_id; next_id += 1
        obj["id"] = request_id
        line = json.dumps(obj, separators=(",", ":"))
        _bounded(captured, "> " + line)
        if proc.stdin is None: raise OSError("app-server stdin unavailable")
        proc.stdin.write(line + "\n"); proc.stdin.flush()
        return request_id
    def request(method: str, params: Dict[str, Any], tag: str = "", ack: Optional[Callable[..., Any]] = None) -> int:
        ident = emit({"jsonrpc": "2.0", "method": method, "params": params}); waiting[ident] = (tag, ack); return ident
    def acknowledge(callback: Optional[Callable[..., Any]], msg_id: str, state: str, detail: str) -> None:
        if callback:
            try: callback(msg_id, state, detail)
            except TypeError: callback(state, detail)
    def deny_server_request(obj: Dict[str, Any]) -> None:
        ident = obj.get("id")
        if ident is not None:
            # Explicit refusal is safer than leaving a never-approved child hung.
            emit({"jsonrpc":"2.0", "id":ident, "result":{"decision":"decline"}})
    try:
        init = request("initialize", {"clientInfo":{"name":"puppetmaster","version":"1"}, "capabilities":{}})
        while time.monotonic() < deadline:
            if cancellation_check and cancellation_check():
                if result.thread_id and active_turn: request("turn/interrupt", {"threadId":result.thread_id,"turnId":active_turn})
                result.status = "cancelled"; result.error = "cancelled"; break
            if started and pending_steering and result.thread_id and active_turn:
                for pending in pending_steering() or ():
                    if isinstance(pending, dict): mid, text, ack = str(pending.get("id")), str(pending.get("text", "")), pending.get("acknowledge")
                    else: mid, text, ack = pending[0], pending[1], pending[2] if len(pending) > 2 else None
                    result.queued_steering.append(str(mid)); acknowledge(ack, str(mid), "queued", "queued for active turn")
                    request("turn/steer", {"threadId":result.thread_id,"expectedTurnId":active_turn,"input":[{"type":"text","text":str(text)}],"clientUserMessageId":str(mid)}, str(mid), ack)
            if max_output_bytes and output_bytes[0] > max_output_bytes:
                result.status = "output_limit"; result.error = f"output exceeded {max_output_bytes} bytes"; break
            try: source, line = inbox.get(timeout=min(.05, max(.001, deadline-time.monotonic())))
            except queue.Empty: continue
            if line is None:
                stdout_eof |= source == "stdout"; stderr_eof |= source == "stderr"
                if stdout_eof and proc.poll() is not None:
                    if result.status == "failed":
                        result.error = result.error or "app-server EOF"
                    break
                continue
            _bounded(captured, ("! " if source == "stderr" else "< ") + line.rstrip())
            if log: log(line.rstrip())
            if source != "stdout": continue
            try: obj = json.loads(line)
            except ValueError:
                result.status = "failed"
                result.error = "malformed app-server JSON"
                break
            if not isinstance(obj, dict): continue
            if event_sink: event_sink(obj)
            if "id" in obj and "method" in obj:
                deny_server_request(obj)
                result.status = "failed"
                result.error = "unexpected approval/server request"
                break
            if "id" in obj:
                tag, ack = waiting.pop(obj.get("id"), ("", None))
                if "error" in obj:
                    if tag:
                        result.rejected_steering.append(tag); acknowledge(ack, tag, "failed", str(obj["error"]))
                    else: result.error = str(obj["error"])
                    continue
                if obj.get("id") == init:
                    emit({"jsonrpc":"2.0", "method":"initialized", "params":{}})
                    thread = request("thread/start", {"model":model,"cwd":str(Path(cwd).resolve()),"sandbox":sandbox,"approvalPolicy":"never","ephemeral":bool(ephemeral),"config":config or {}})
                    waiting[thread] = ("thread", None)
                elif tag == "thread":
                    payload = obj.get("result", {}); result.thread_id = payload.get("thread", {}).get("id") if isinstance(payload, dict) else None
                    result.thread_id = result.thread_id or (payload.get("threadId") if isinstance(payload, dict) else None)
                    if not result.thread_id: result.error = "thread/start response missing thread id"; break
                    params: Dict[str, Any] = {"threadId":result.thread_id,"input":[{"type":"text","text":prompt}]}
                    if model: params["model"] = model
                    if effort: params["effort"] = effort
                    request("turn/start", params, "turn"); started = True
                elif tag == "turn":
                    payload = obj.get("result", {}); result.turn_id = payload.get("turn", {}).get("id") if isinstance(payload, dict) else None
                    result.turn_id = result.turn_id or (payload.get("turnId") if isinstance(payload, dict) else None); active_turn = result.turn_id
                elif tag:
                    # Do not follow a response turnId: steering is constrained by expectedTurnId.
                    result.accepted_steering.append(tag); acknowledge(ack, tag, "accepted", "server accepted delivery")
                continue
            method, params = obj.get("method"), obj.get("params", {})
            if method == "turn/started" and isinstance(params, dict): active_turn = str(params.get("turnId") or active_turn or "") or None; result.turn_id = active_turn or result.turn_id
            if method == "thread/tokenUsage/updated": result.input_tokens, result.output_tokens, result.cached_input_tokens = _usage(params)
            result.messages.extend(_text_from(params))
            if method == "turn/completed": result.status = "completed"; break
            if method in ("turn/failed", "error"): result.error = str(params); result.status = "failed"; break
        else: result.status = "timeout"; result.error = "app-server timeout"
    except (OSError, ValueError) as exc:
        result.status = "failed"; result.error = str(exc)
    finally:
        # The group and owner nonce also reach the shell commands the turn started.
        cleanup_owned_process(proc, owner, time.monotonic() + 3)
        close_owned_process(proc)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                try: stream.close()
                except OSError: pass
        result.raw_protocol = "\n".join(captured)
        if protocol_log_path:
            Path(protocol_log_path).write_text(result.raw_protocol, encoding="utf-8")
    return result


@dataclass
class HostTurnResult(CodexSessionResult):
    """One thread turn on a shared host, with the items the write-scope gate reads."""
    items: List[Dict[str, Any]] = field(default_factory=list)
    rollout_path: Optional[str] = None
    thread_started_at: Optional[float] = None
    turn_started_at: Optional[float] = None


def _exec_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """An app-server item in the ``codex exec --json`` item shape, or None."""
    kind = item.get("type")
    if kind == "fileChange":
        changes = [{"path": change.get("path"), "kind": (change.get("kind") or {}).get("type")}
                   for change in item.get("changes") or [] if isinstance(change, dict)]
        return {"type": "file_change", "changes": changes, "status": item.get("status")}
    if kind == "commandExecution":
        return {"type": "command_execution", "command": item.get("command"),
                "exit_code": item.get("exitCode"), "status": item.get("status")}
    return None


def run_host_turn(
    connect: Callable[[], Any], cwd: Union[str, Path], prompt: str, *,
    model: Optional[str] = None, sandbox: str = "workspace-write", approval_policy: str = "never",
    config: Optional[Dict[str, Any]] = None,
    effort: Optional[str] = None, ephemeral: bool = False, resume_thread: Optional[str] = None,
    timeout: float = 600, cancellation_check: Optional[Callable[[], bool]] = None,
    protocol_log_path: Optional[Union[str, Path]] = None, max_output_bytes: Optional[int] = None,
) -> HostTurnResult:
    """Run one turn as a thread on a shared ``codex app-server`` host.

    ``connect`` returns a connected :class:`codex_ws.HostSocket`. A fresh run
    starts a thread; a repair resumes ``resume_thread``. A cut or the deadline
    interrupts only this thread's turn: the host and its other threads go on.
    An approval request is declined and fails the turn, as in
    :func:`run_codex_session`. Notifications for another thread are ignored.
    """
    import socket

    from puppetmaster.adapters.codex_ws import HostClosed

    result = HostTurnResult(status="failed",
                            protocol_log_path=str(protocol_log_path) if protocol_log_path else None)
    captured: List[str] = []
    deadline = time.monotonic() + max(0.01, timeout)
    try:
        conn = connect()
    except OSError as exc:
        result.error = f"host connect failed: {exc}"
        return result
    next_id = [0]
    waiting: Dict[int, str] = {}
    output_bytes = 0
    interrupted = ""

    def request(method: str, params: Dict[str, Any], tag: str) -> None:
        next_id[0] += 1
        message = {"jsonrpc": "2.0", "id": next_id[0], "method": method, "params": params}
        _bounded(captured, "> " + json.dumps(message, separators=(",", ":")))
        conn.send(message)
        waiting[next_id[0]] = tag

    def interrupt(reason: str) -> None:
        nonlocal interrupted
        if not interrupted and result.thread_id and result.turn_id:
            request("turn/interrupt", {"threadId": result.thread_id, "turnId": result.turn_id}, "interrupt")
        interrupted = interrupted or reason

    try:
        request("initialize", {"clientInfo": {"name": "puppetmaster", "version": "1"}, "capabilities": {}},
                "initialize")
        while True:
            now = time.monotonic()
            if now >= deadline:
                if interrupted:
                    result.status, result.error = "timeout", "host turn timeout"
                    break
                interrupt("timeout")
                deadline = now + 10
            if not interrupted and cancellation_check and cancellation_check():
                interrupt("cancelled")
            conn.settimeout(max(0.05, min(0.5, deadline - time.monotonic())))
            try:
                obj = conn.recv()
            except socket.timeout:
                continue
            line = json.dumps(obj, separators=(",", ":"))
            output_bytes += len(line)
            _bounded(captured, "< " + line)
            if max_output_bytes and output_bytes > max_output_bytes and not interrupted:
                interrupt("output_limit")
            if not isinstance(obj, dict):
                continue
            if "id" in obj and "method" in obj:
                # The run never approves: refuse, then fail like the stdio session.
                conn.send({"jsonrpc": "2.0", "id": obj["id"], "result": {"decision": "decline"}})
                result.error = "unexpected approval/server request"
                interrupt("approval")
                continue
            if "id" in obj:
                tag = waiting.pop(obj["id"], "")
                payload = obj.get("result") if isinstance(obj.get("result"), dict) else {}
                if "error" in obj:
                    if tag != "interrupt":
                        result.error = str(obj["error"])
                        break
                    continue
                if tag == "initialize":
                    conn.send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
                    if resume_thread:
                        request("thread/resume", {"threadId": resume_thread, "cwd": str(Path(cwd).resolve()),
                                                  "sandbox": sandbox, "approvalPolicy": approval_policy,
                                                  "config": config or {}}, "thread")
                    else:
                        request("thread/start", {"model": model, "cwd": str(Path(cwd).resolve()),
                                                 "sandbox": sandbox, "approvalPolicy": approval_policy,
                                                 "ephemeral": bool(ephemeral), "config": config or {}}, "thread")
                elif tag == "thread":
                    thread = payload.get("thread") if isinstance(payload.get("thread"), dict) else {}
                    result.thread_id = thread.get("id") or payload.get("threadId")
                    result.rollout_path = thread.get("path")
                    result.thread_started_at = time.time()
                    if not result.thread_id:
                        result.error = "thread response missing thread id"
                        break
                    params: Dict[str, Any] = {"threadId": result.thread_id,
                                              "input": [{"type": "text", "text": prompt}]}
                    if model:
                        params["model"] = model
                    if effort:
                        params["effort"] = effort
                    request("turn/start", params, "turn")
                elif tag == "turn":
                    turn = payload.get("turn") if isinstance(payload.get("turn"), dict) else {}
                    result.turn_id = result.turn_id or turn.get("id") or payload.get("turnId")
                continue
            method, params = obj.get("method"), obj.get("params")
            if not isinstance(params, dict):
                continue
            if params.get("threadId") not in (None, result.thread_id):
                continue
            if method == "turn/started":
                turn = params.get("turn") if isinstance(params.get("turn"), dict) else {}
                result.turn_id = result.turn_id or turn.get("id") or params.get("turnId")
                result.turn_started_at = result.turn_started_at or time.time()
            elif method == "thread/tokenUsage/updated":
                result.input_tokens, result.output_tokens, result.cached_input_tokens = _usage(params)
            elif method == "item/completed":
                result.messages.extend(_text_from(params))
                item = params.get("item")
                mapped = _exec_item(item) if isinstance(item, dict) else None
                if mapped is not None:
                    result.items.append(mapped)
            elif method == "turn/completed":
                turn = params.get("turn") if isinstance(params.get("turn"), dict) else {}
                state = str(turn.get("status") or "completed")
                if interrupted:
                    result.status = {"cancelled": "cancelled", "timeout": "timeout",
                                     "output_limit": "output_limit"}.get(interrupted, "failed")
                    result.error = result.error or f"turn interrupted: {interrupted}"
                elif state in ("failed", "interrupted"):
                    result.status, result.error = "failed", str(turn.get("error") or state)
                else:
                    result.status = "completed"
                break
            elif method in ("turn/failed", "error") and not params.get("willRetry"):
                result.status, result.error = "failed", str(params)[:2000]
                break
    except (HostClosed, OSError, ValueError) as exc:
        result.status = "failed"
        result.error = f"host connection lost: {exc}"
    finally:
        try:
            conn.close()
        except OSError:
            pass
        result.raw_protocol = "\n".join(captured)
        if protocol_log_path:
            Path(protocol_log_path).write_text(result.raw_protocol, encoding="utf-8")
    return result
