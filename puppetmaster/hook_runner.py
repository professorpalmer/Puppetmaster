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
import os
import re
import sys
from dataclasses import dataclass, replace
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

# We deliberately do NOT hard-deny the built-in agent fan-out (``Task`` /
# ``Agent``). Two field reports (Claude with no Cursor; Cursor with the swarm
# tools not connected) showed the deny wedging the turn: native Task is blocked
# while the suggested ``puppetmaster_start_*swarm`` verb isn't callable. It also
# over-reaches — Puppetmaster explicitly does NOT replace several native
# subagents (``browser-use``, ``ci-investigator``, ``cursor-guide``,
# ``best-of-n-runner``), and the hook payload can't reliably tell those apart
# from a swarm-worthy fan-out. Steering toward a swarm is left to the
# non-blocking prompt-submit directive, which never blocks the native tool.
_ASSUME_TOOLS_ENV = "PUPPETMASTER_HOOK_ASSUME_TOOLS"

# Cursor-specific verbs translated for non-Cursor hosts. The ``*_cursor_*``
# verbs require the Cursor SDK platform; recommending them on a Claude Code or
# Codex host points the model at a tool it cannot run (field-reported: the
# hook denied Claude's Agent tool and suggested ``puppetmaster_start_cursor_swarm``
# on a machine with no Cursor installed, wedging the turn). The generic verbs
# route to whatever platform the lock enables, so they are safe everywhere.
_HOST_PORTABLE_VERBS = {
    "puppetmaster_start_cursor_swarm": "puppetmaster_start_swarm",
    "puppetmaster_start_cursor_review": "puppetmaster_start_swarm",
    "puppetmaster_start_cursor_plan": "puppetmaster_start_swarm",
    "puppetmaster_start_cursor_implement": "puppetmaster_start_implement",
}


def verb_for_host(verb: str, host: str) -> str:
    """Return a verb the ``host`` agent can actually invoke.

    Cursor hosts keep the cursor-specific daily drivers; every other host gets
    the platform-routing equivalent instead of a verb it may not have.
    """
    if (host or "").strip().lower() == "cursor":
        return verb
    return _HOST_PORTABLE_VERBS.get(verb, verb)

# NOTE on native search/glob tools (Grep/Glob/codebase_search): we deliberately
# do NOT hard-deny these. Field testing showed Cursor's hook payload for them
# doesn't reliably carry the call's *scope*, so we can't tell "grep one file" /
# "list one config dir" (read-only inspection) from a repo-wide sweep — and
# hard-denying *every* one obstructs legitimate work, violating the "never wedge
# the session" contract. Steering toward Puppetmaster for broad search is left to
# the non-blocking prompt-submit directive; only genuinely recursive *shell*
# searches (where the command string is visible) are redirected below.

# Read-only shell inspection that must NEVER be redirected — git history, listing
# a directory, viewing a file. Matched at the start of a command or after a
# pipe/sep, so `git log`, `ls ~/.cursor`, `cat foo.py | head` all pass.
_READONLY_SHELL_RE = re.compile(
    r"(?:^|[\n;&|]\s*)\s*(?:sudo\s+)?(?:"
    r"git\s+(?:log|show|diff|status|blame|reflog|describe|rev-parse|branch|tag|"
    r"remote|stash|config|cat-file|shortlog|whatchanged|ls-files)"
    r"|ls|ll|cat|bat|head|tail|less|more|pwd|echo|printf|which|type|file|stat|"
    r"wc|tree|env|date|whoami|basename|dirname|realpath|cd"
    r")\b",
    re.IGNORECASE,
)
# A *genuinely broad* shell search: a search tool used recursively / repo-wide.
# Requires an explicit broad signal (recursive flag, ** glob, or find -name/-path),
# so a narrow `grep pattern file.py` or `rg pattern src/app.ts` is NOT flagged.
_BROAD_SHELL_RE = re.compile(
    r"\b(?:rg|ripgrep|grep|egrep|fgrep|ag|ack)\b[^\n|;&]*?"
    r"(?:\s-{1,2}(?:r|R|recursive)\b|\*\*)"
    r"|\bfind\b[^\n|;&]*?\s-(?:i?name|i?path|regex)\b",
    re.IGNORECASE,
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

    Returns ``(should_redirect, suggested_verb)``. Deliberately conservative —
    a false deny obstructs legitimate work (the field-reported failure mode),
    which is worse than a false allow. The *only* native call we redirect is a
    shell command that is a genuinely recursive/repo-wide search, with an
    explicit read-only carve-out (``git log/show/diff``, ``ls``, ``cat`` …),
    because the command string makes the broad scope unambiguous and CodeGraph
    is the strictly better tool for it.

    Native search/glob tools (Grep/Glob/codebase_search) and the built-in agent
    fan-out (``Task``/``Agent``) are NOT denied — their scope isn't visible in
    the hook payload and the agent tool has irreplaceable native uses, so
    blocking them wedges benign work. Anything namespaced ``puppetmaster`` /
    ``mcp`` is always allowed, and plain file reads (``Read``) are never touched.
    """
    name = (tool_name or "").strip().lower()
    if not name:
        return False, ""
    if "puppetmaster" in name or name.startswith("mcp"):
        return False, ""

    base = name.split("__")[-1]  # strip any "mcp__server__tool" prefixing

    if base in {"shell", "bash", "sh", "zsh", "run_terminal_cmd", "execute"}:
        command = tool_input if isinstance(tool_input, str) else json.dumps(tool_input)
        command = command or ""
        if _READONLY_SHELL_RE.search(command):
            return False, ""
        if _BROAD_SHELL_RE.search(command):
            return True, "puppetmaster_codegraph_search"
        return False, ""

    return False, ""


def _puppetmaster_tools_available(env: Optional[Mapping[str, str]] = None) -> bool:
    """Best-effort: is a Puppetmaster MCP server actually alive to redirect to?

    The hook must never steer a native tool toward a Puppetmaster verb the agent
    can't call. Reads the per-user MCP server registry and reports whether any
    tracked server process is alive. Fails closed (``False`` → the hook becomes a
    no-op) on any error, because allowing the native tool is always safe and a
    wrongful deny is the bug we're fixing. ``PUPPETMASTER_HOOK_ASSUME_TOOLS``
    (truthy/falsy) overrides the probe for power users and tests.
    """
    environ = env if env is not None else os.environ
    override = environ.get(_ASSUME_TOOLS_ENV)
    if override is not None:
        return str(override).strip().lower() in {"1", "true", "yes", "on"}
    try:
        from puppetmaster import mcp_registry

        return any(entry.is_alive() for entry in mcp_registry.list_entries())
    except Exception:
        return False


def handle_hook(
    payload: Mapping[str, Any],
    *,
    host: str,
    event: str,
    env: Optional[Mapping[str, str]] = None,
) -> HookResponse:
    """Core hook policy. Same payload/event → same response (modulo whether a
    Puppetmaster MCP server is alive, which is what makes steering safe)."""
    if gate_disabled(env):
        return HookResponse(action="allow")

    # If no Puppetmaster MCP server is alive, the hook is a complete no-op: it
    # must never deny a native tool or inject a "use <verb>" directive for a
    # tool the agent can't call. Field reports: the hook denied the native Agent
    # tool and insisted on puppetmaster_start_swarm while the swarm tools weren't
    # connected, leaving the turn wedged.
    if not _puppetmaster_tools_available(env):
        return HookResponse(action="allow")

    kind = normalize_event(event)

    if kind == EVENT_PRE_TOOL:
        tool_name, tool_input = extract_tool(payload)
        redirect, verb = classify_tool(tool_name, tool_input)
        if redirect:
            verb = verb_for_host(verb, host)
            return HookResponse(
                action="deny",
                reason=(
                    f"[Puppetmaster] Native '{tool_name}' is a repo-wide search. "
                    f"Use `{verb}` instead — CodeGraph is faster, cheaper, and "
                    f"auditable. If the MCP tool isn't reachable, run "
                    f"`python -m puppetmaster codegraph search '<query>'`. "
                    f"(Set PUPPETMASTER_AUTO_INVOKE_DISABLED=1 to turn this off.)"
                ),
            )
        return HookResponse(action="allow")

    # user-prompt: inject a directive when the gate fires; never block.
    prompt = extract_prompt(payload)
    if not prompt:
        return HookResponse(action="allow")
    decision = should_delegate(prompt, env=env)
    decision = replace(
        decision, suggested_verb=verb_for_host(decision.suggested_verb, host)
    )
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
