"""Optional OpenAI-compatible proxy shim — the API-key/SDK enforcement gap.

Closed harnesses (Cursor) won't let us sit on the LLM provider wire, so hooks
are the enforcement layer there. But *open* clients — anything that speaks the
OpenAI Chat Completions API with a configurable base URL (Aider, Codex in
API mode, raw SDK apps, LangChain, …) — can be pointed at a local proxy. That
proxy is the only deterministic place to enforce delegation for those clients.

This module is that shim. It is deliberately **zero-dependency** (stdlib
``http.server`` only) and small. Two modes:

* ``advise`` (default, needs no upstream): inspect the inbound request, run the
  gate on the latest user message, and if it fires, return a **synthetic**
  assistant message telling the caller to invoke a Puppetmaster verb. Nothing
  is forwarded anywhere — fully local, fully testable, no key handling.
* ``inject`` (needs an upstream): prepend a system directive to the request and
  forward it to a vetted upstream. The upstream must pass
  :func:`is_upstream_allowed` (HTTPS + allowlisted host, or loopback) so a
  misconfigured proxy can never exfiltrate an API key to an arbitrary host.

The request-transform logic is pure and unit-tested; the HTTP plumbing is a
thin wrapper around it.

`headroom wrap` / its proxy are referenced only as transport *prior art* for
the ergonomics of standing between a client and a provider — Puppetmaster does
not adopt Headroom's stack (its compression/memory/code-graph layers overlap
and would duplicate what we already own).
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

from puppetmaster.invocation_gate import DelegationDecision, should_delegate

#: Hosts an ``inject``-mode proxy may forward to without an explicit override.
#: Loopback is always allowed; everything else must be HTTPS and on this list.
DEFAULT_UPSTREAM_ALLOWLIST = (
    "api.openai.com",
    "api.anthropic.com",
)
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def extract_user_prompt(payload: Mapping[str, Any]) -> str:
    """Return the latest user message text from an OpenAI-style request body.

    Handles both string content and the structured ``[{type, text}, ...]``
    content form. Falls back to a legacy top-level ``prompt`` field.
    """
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, Mapping) or message.get("role") != "user":
                continue
            return _content_to_text(message.get("content"))
    prompt = payload.get("prompt")
    if isinstance(prompt, str):
        return prompt
    return ""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return ""


def build_delegate_directive(decision: DelegationDecision) -> str:
    """System-message text injected ahead of a delegated request."""
    return decision.directive()


def inject_directive(payload: Mapping[str, Any], directive: str) -> dict:
    """Return a copy of ``payload`` with a system directive prepended."""
    body = dict(payload)
    messages = list(body.get("messages") or [])
    messages.insert(0, {"role": "system", "content": directive})
    body["messages"] = messages
    return body


def build_advice_response(decision: DelegationDecision, *, model: str = "puppetmaster-gate") -> dict:
    """A synthetic OpenAI-shaped chat completion that delivers the directive.

    Lets ``advise`` mode answer locally with no upstream call — the caller's
    SDK sees a normal assistant turn whose content is the delegate directive.
    """
    return {
        "id": "puppetmaster-gate",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": decision.directive()},
                "finish_reason": "stop",
            }
        ],
        "_puppetmaster": decision.to_dict(),
    }


def transform_chat_request(
    body: Mapping[str, Any],
    *,
    env: Optional[Mapping[str, str]] = None,
) -> tuple[dict, Optional[DelegationDecision]]:
    """Pure request transform for ``inject`` mode.

    Returns ``(possibly_modified_body, decision_or_None)``. When the gate fires
    the body gains a leading system directive; otherwise it's passed through
    unchanged.
    """
    prompt = extract_user_prompt(body)
    if not prompt:
        return dict(body), None
    decision = should_delegate(prompt, env=env)
    if not decision.should_delegate:
        return dict(body), decision
    return inject_directive(body, build_delegate_directive(decision)), decision


def is_upstream_allowed(
    base_url: str, allowlist: tuple[str, ...] = DEFAULT_UPSTREAM_ALLOWLIST
) -> bool:
    """Guardrail against forwarding API keys to an arbitrary upstream.

    Loopback (any scheme) is always allowed for local testing. Every other
    upstream must be HTTPS and have a host on ``allowlist``.
    """
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    if host in _LOOPBACK_HOSTS:
        return True
    if parsed.scheme != "https":
        return False
    return host in {h.lower() for h in allowlist}


def make_handler(*, mode: str = "advise", upstream_base_url: str = "", env=None):
    """Build a stdlib ``BaseHTTPRequestHandler`` subclass for the proxy.

    Kept as a factory so the (pure) policy above stays trivially unit-testable
    and the network surface is a thin shell.
    """
    from http.server import BaseHTTPRequestHandler

    class _ProxyHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet by default
            pass

        def do_POST(self):  # noqa: N802 (stdlib naming)
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw or b"{}")
            except (json.JSONDecodeError, ValueError):
                body = {}

            prompt = extract_user_prompt(body) if isinstance(body, Mapping) else ""
            decision = should_delegate(prompt, env=env) if prompt else None

            if mode == "advise" and decision is not None and decision.should_delegate:
                self._respond_json(200, build_advice_response(decision))
                return

            # advise mode with no delegation: tell the caller to proceed
            # normally (it should retry against the real provider directly).
            self._respond_json(
                200,
                {
                    "_puppetmaster": {
                        "delegated": False,
                        "decision": decision.to_dict() if decision else None,
                        "note": "no delegation; call your provider directly",
                    }
                },
            )

        def _respond_json(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return _ProxyHandler


def serve_proxy(
    *,
    host: str = "127.0.0.1",
    port: int = 8788,
    mode: str = "advise",
    upstream_base_url: str = "",
    env=None,
) -> None:  # pragma: no cover - network loop, exercised manually
    """Run the local proxy until interrupted.

    ``inject`` mode requires a vetted ``upstream_base_url``; we refuse to start
    otherwise so a misconfig can't silently forward keys somewhere unexpected.
    """
    from http.server import HTTPServer

    if mode == "inject":
        if not upstream_base_url or not is_upstream_allowed(upstream_base_url):
            raise ValueError(
                f"inject mode requires an allowlisted HTTPS upstream; "
                f"{upstream_base_url!r} is not permitted"
            )

    handler = make_handler(mode=mode, upstream_base_url=upstream_base_url, env=env)
    server = HTTPServer((host, port), handler)
    print(f"puppetmaster proxy listening on http://{host}:{port} (mode={mode})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
