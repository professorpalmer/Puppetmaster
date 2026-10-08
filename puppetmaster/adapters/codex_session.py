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
    model: Optional[str] = None, sandbox: str = "workspace-write", config: Optional[Dict[str, Any]] = None,
    effort: Optional[str] = None, ephemeral: bool = True, timeout: float = 600, env: Optional[Dict[str, str]] = None,
    log: Optional[Callable[[str], None]] = None, event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    cancellation_check: Optional[Callable[[], bool]] = None,
    pending_steering: Optional[Callable[[], Iterable[Any]]] = None,
    protocol_log_path: Optional[Union[str, Path]] = None,
) -> CodexSessionResult:
    """Run one app-server thread (ephemeral by default) and optionally steer its active turn.

    ``pending_steering`` yields ``(id, text, acknowledge)`` tuples or mappings
    with those names.  ``acknowledge`` receives ``(id, state, details)`` where
    state is queued, accepted, or failed.  A response to ``turn/steer`` means
    delivery acceptance only; it never claims that the model applied text.
    """
    command = [command_prefix] if isinstance(command_prefix, str) else list(command_prefix)
    command += ["app-server", "--listen", "stdio://"]
    child_env = dict(os.environ)
    child_env.update(env or {})
    # A nested worker must not discover/re-enter the parent MCP transport.
    child_env["PUPPETMASTER_WORKER"] = "1"
    child_env["PUPPETMASTER_AUTO_INVOKE_DISABLED"] = "1"
    proc = subprocess.Popen(command, cwd=str(Path(cwd).resolve()), env=child_env, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    inbox: "queue.Queue[Tuple[str, Optional[str]]]" = queue.Queue()
    captured: List[str] = []
    def reader(name: str, stream: Any) -> None:
        try:
            for line in iter(stream.readline, ""):
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
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=2)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                try: stream.close()
                except OSError: pass
        result.raw_protocol = "\n".join(captured)
        if protocol_log_path:
            Path(protocol_log_path).write_text(result.raw_protocol, encoding="utf-8")
    return result
