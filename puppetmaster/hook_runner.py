"""Translate host-agent hook payloads into gate decisions and host responses.

The deterministic half of auto-invocation. Host agents (Cursor, Claude Code)
can run a *command* at lifecycle events and feed it JSON on stdin; whatever the
command prints on stdout is interpreted by the host. This module is that
command's brain:

* **user-prompt** events (Cursor ``beforeSubmitPrompt`` / Claude
  ``UserPromptSubmit``): run :func:`invocation_gate.should_delegate` on the
  prompt and, when it fires, inject a "delegate now with <verb>" directive into
  the model's context *before it starts working*. We never block the prompt —
  injection beats refusal for a prompt the user just typed.
* **pre-tool** events (Cursor ``beforeShellExecution`` / ``beforeReadFile`` /
  ``preToolUse``; Claude ``PreToolUse``): when the model reaches for a broad
  native exploration tool (Grep/Glob/built-in Task/repo-wide shell search), we
  **deny and redirect** it to the Puppetmaster equivalent. This is the
  enforcement rules can't provide — the model can't take the cheap path.

Everything here is fail-open: any parse error or unknown shape yields an
"allow / no-op" response so a hook can never wedge the host session. Response
rendering is host-specific and centralized in :class:`HookResponse` so the
exact JSON keys can track each host's evolving schema in one place.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from puppetmaster.invocation_gate import DelegationDecision, gate_disabled, should_delegate

# Canonical event kinds. Host-specific event names normalize onto these.
EVENT_USER_PROMPT = "user-prompt"
EVENT_PRE_TOOL = "pre-tool"

_USER_PROMPT_ALIASES = {
    "user-prompt", "userpromptsubmit", "beforesubmitprompt", "user_prompt",
    "submitprompt", "prompt",
}
_PRE_TOOL_ALIASES = {
    "pre-tool", "pretooluse", "beforeshellexecution", "beforereadfile",
    "beforemcpexecution", "pre_tool", "tool",
}

# Native host tools that mean "broad exploration" — redirect these to
# Puppetmaster. Matched case-insensitively against the host's tool name.
_NATIVE_BROAD_TOOLS = {
    "grep", "glob", "task", "codebase_search", "search", "ripgrep",
}
_BROAD_TOOL_REDIRECT = {
    "grep": "puppetmaster_codegraph_search",
    "glob": "puppetmaster_codegraph_files",
    "codebase_search": "puppetmaster_codegraph_context",
    "search": "puppetmaster_codegraph_search",
    "ripgrep": "puppetmaster_codegraph_search",
    "task": "puppetmaster_start_cursor_swarm",
}
# Shell commands that are really a repo-wide search in disguise.
_BROAD_SHELL_RE = re.compile(
    r"\b(rg|grep|ag|ack|find)\b.*(-r|--recursive|-R|\*\*|/)", re.IGNORECASE
)


@dataclass(frozen=True)
class HookResponse:
    """A host-agnostic hook verdict, rendered per host on demand.

    ``action`` is one of ``allow`` (no-op / proceed) or ``deny`` (block + show
    ``reason``). ``context`` is optional text to inject into the model's context
    (used by user-prompt events). ``decision`` carries the underlying gate
    verdict for logging/debug.
    """

    action: str = "allow"
    reason: str = ""
    context: str = ""
    decision: Optional[DelegationDecision] = None

    def to_host_json(self, host: str) -> dict:
        host = (host or "").lower()
        if host == "claude":
            return self._claude_json()
        return self._cursor_json()

    def _cursor_json(self) -> dict:
        # Cursor reads a permission verdict and optional injected context.
        out: dict[str, Any] = {}
        if self.action == "deny":
            out["permission"] = "deny"
            out["userMessage"] = self.reason
            out["agentMessage"] = self.reason
        else:
            out["permission"] = "allow"
            out["continue"] = True
            if self.context:
                out["additionalContext"] = self.context
        return out

    def _claude_json(self) -> dict:
        # Claude Code reads hookSpecificOutput with an event-scoped decision.
        if self.action == "deny":
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": self.reason,
                }
            }
        out: dict[str, Any] = {"continue": True}
        if self.context:
            out["hookSpecificOutput"] = {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": self.context,
            }
        return out


def normalize_event(event: str) -> str:
    """Map a host-specific event name onto a canonical event kind."""
    key = (event or "").strip().lower().replace("_", "").replace("-", "")
    if key in {a.replace("_", "").replace("-", "") for a in _USER_PROMPT_ALIASES}:
        return EVENT_USER_PROMPT
    if key in {a.replace("_", "").replace("-", "") for a in _PRE_TOOL_ALIASES}:
        return EVENT_PRE_TOOL
    # Unknown → treat as user-prompt (inject-only, never blocks).
    return EVENT_USER_PROMPT


def extract_prompt(payload: Mapping[str, Any]) -> str:
    """Pull the user prompt text out of a host hook payload, tolerantly."""
    for key in ("prompt", "user_prompt", "userPrompt", "message", "text", "input"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    # Cursor sometimes nests under conversation/last message shapes.
    nested = payload.get("hook_input") or payload.get("data")
    if isinstance(nested, Mapping):
        return extract_prompt(nested)
    return ""


def extract_tool(payload: Mapping[str, Any]) -> tuple[str, Any]:
    """Return (tool_name, tool_input) from a pre-tool payload, tolerantly."""
    for key in ("tool_name", "toolName", "tool", "name"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            tool_input = (
                payload.get("tool_input")
                or payload.get("toolInput")
                or payload.get("arguments")
                or payload.get("command")
                or {}
            )
            return value, tool_input
    # Shell-execution payloads carry the command directly.
    command = payload.get("command")
    if isinstance(command, str):
        return "shell", command
    return "", {}


def classify_tool(tool_name: str, tool_input: Any) -> tuple[bool, str]:
    """Decide whether a native tool call should be redirected to Puppetmaster.

    Returns ``(should_redirect, suggested_verb)``. Conservative on purpose:
    only *broad* exploration is redirected. Anything namespaced ``puppetmaster``
    / ``mcp`` is always allowed (never block our own tools), and focused reads /
    single commands pass through.
    """
    name = (tool_name or "").strip().lower()
    if not name:
        return False, ""
    if "puppetmaster" in name or name.startswith("mcp"):
        return False, ""

    base = name.split("__")[-1]  # strip any "mcp__server__tool" prefixing
    if base in _NATIVE_BROAD_TOOLS:
        return True, _BROAD_TOOL_REDIRECT.get(base, "puppetmaster_start_cursor_swarm")

    if base == "shell" or base in {"bash", "run_terminal_cmd", "execute"}:
        command = tool_input if isinstance(tool_input, str) else json.dumps(tool_input)
        if _BROAD_SHELL_RE.search(command or ""):
            return True, "puppetmaster_codegraph_search"
    return False, ""


def handle_hook(
    payload: Mapping[str, Any],
    *,
    host: str,
    event: str,
    env: Optional[Mapping[str, str]] = None,
) -> HookResponse:
    """Core hook policy. Pure: same payload/event → same response."""
    if gate_disabled(env):
        return HookResponse(action="allow")

    kind = normalize_event(event)

    if kind == EVENT_PRE_TOOL:
        tool_name, tool_input = extract_tool(payload)
        redirect, verb = classify_tool(tool_name, tool_input)
        if redirect:
            return HookResponse(
                action="deny",
                reason=(
                    f"[Puppetmaster] Native '{tool_name}' is broad exploration. "
                    f"Use `{verb}` instead — CodeGraph/swarm is faster, cheaper, "
                    f"and auditable. (Set PUPPETMASTER_AUTO_INVOKE_DISABLED=1 to "
                    f"turn this off.)"
                ),
            )
        return HookResponse(action="allow")

    # user-prompt: inject a directive when the gate fires; never block.
    prompt = extract_prompt(payload)
    if not prompt:
        return HookResponse(action="allow")
    decision = should_delegate(prompt, env=env)
    if decision.should_delegate:
        return HookResponse(action="allow", context=decision.directive(), decision=decision)
    return HookResponse(action="allow", decision=decision)


def run(argv: Optional[list[str]] = None, *, stdin=None, stdout=None, env=None) -> int:
    """Entry point for ``python -m puppetmaster invocation-gate``.

    Reads a host hook payload as JSON from stdin, emits the host-specific
    response JSON on stdout, and always exits 0 (fail-open: a non-zero exit or
    crash here would stall the host session).
    """
    import argparse

    parser = argparse.ArgumentParser(prog="puppetmaster invocation-gate")
    parser.add_argument("--host", default="cursor", help="cursor | claude")
    parser.add_argument("--event", default="user-prompt", help="host event name")
    parser.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    raw = ""
    try:
        raw = stdin.read()
    except Exception:  # pragma: no cover - defensive
        raw = ""
    try:
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, Mapping):
            payload = {}
    except (json.JSONDecodeError, ValueError):
        payload = {}

    try:
        response = handle_hook(payload, host=args.host, event=args.event, env=env)
        out = response.to_host_json(args.host)
    except Exception as exc:  # fail open, but make the reason visible for debug
        out = {"permission": "allow", "continue": True, "_puppetmaster_error": str(exc)}

    stdout.write(json.dumps(out))
    stdout.flush()
    return 0
