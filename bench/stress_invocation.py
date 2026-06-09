"""End-to-end stress / mock harness for Puppetmaster auto-invocation.

Validates the full enforcement stack the way a host actually drives it:

1. **Gate battery** — a labeled set of realistic prompts run through
   ``should_delegate``; reports precision/recall and fails on any miss.
2. **Hook subprocess** — real ``python -m puppetmaster invocation-gate`` calls
   over stdin/stdout for Cursor + Claude, both events (the exact path a host
   hook executes).
3. **Live proxy** — boots the stdlib HTTP proxy on a real port and POSTs
   OpenAI-shaped requests, asserting delegate vs passthrough behavior.
4. **Installer** — installs hooks into a temp workspace, validates the emitted
   JSON, then feeds a payload through the installed command string.
5. **Kill switch** — confirms the global env disable neuters every layer.

Run: ``python bench/stress_invocation.py``  (exit 0 = all green).
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.request
from http.server import HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# 1. Gate battery
# ---------------------------------------------------------------------------

# (prompt, expected_should_delegate)
BATTERY = [
    ("audit the auth module for security vulnerabilities across the repo", True),
    ("refactor the database layer into a repository pattern", True),
    ("find all callers of getCwd and rename them", True),
    ("review the whole codebase for race conditions", True),
    ("migrate the project from npm to pnpm", True),
    ("design a caching strategy for the API", True),
    ("implement OAuth login end-to-end", True),
    ("trace how a websocket message flows through the backend", True),
    ("use puppetmaster to summarize the module", True),
    ("fix a typo in the README", False),
    ("rename this local variable", False),
    ("add a comment to this function", False),
    ("what does the walrus operator do", False),
    ("format this file", False),
    ("refactor everything but do it inline", False),
    ("reword this one sentence in the docstring", False),
]


def run_gate_battery() -> None:
    section("1. Gate battery (precision/recall on labeled prompts)")
    from puppetmaster.invocation_gate import should_delegate

    tp = fp = tn = fn = 0
    for prompt, expected in BATTERY:
        got = should_delegate(prompt).should_delegate
        ok = got == expected
        if expected and got:
            tp += 1
        elif expected and not got:
            fn += 1
        elif not expected and got:
            fp += 1
        else:
            tn += 1
        check(f"{'DELEGATE' if expected else 'inline':>8} <- {prompt[:52]}", ok,
              f"expected {expected}, got {got}")
    total = len(BATTERY)
    correct = tp + tn
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    print(f"  accuracy={correct}/{total}  precision={precision:.2f}  recall={recall:.2f}")


# ---------------------------------------------------------------------------
# 2. Hook subprocess (real CLI over stdin)
# ---------------------------------------------------------------------------


def _hook(host: str, event: str, payload: dict, env_extra: dict | None = None) -> dict:
    import os
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [PY, "-m", "puppetmaster", "invocation-gate", "--host", host, "--event", event],
        input=json.dumps(payload), capture_output=True, text=True, cwd=str(REPO), env=env,
    )
    return json.loads(proc.stdout or "{}")


def run_hook_subprocess() -> None:
    section("2. Hook subprocess (the exact path a host hook runs)")

    out = _hook("cursor", "beforeSubmitPrompt", {"prompt": "audit the repo for security holes"})
    check("cursor prompt-submit injects directive", "additionalContext" in out and "Puppetmaster" in out.get("additionalContext", ""))

    out = _hook("claude", "UserPromptSubmit", {"prompt": "refactor across all modules"})
    ctx = out.get("hookSpecificOutput", {}).get("additionalContext", "")
    check("claude prompt-submit injects directive", "Puppetmaster" in ctx)

    out = _hook("cursor", "beforeShellExecution", {"command": "rg -r 'TODO' ./src"})
    check("cursor deny-redirects broad shell search", out.get("permission") == "deny")

    out = _hook("claude", "PreToolUse", {"tool_name": "Grep"})
    check("claude deny-redirects native Grep", out.get("hookSpecificOutput", {}).get("permissionDecision") == "deny")

    out = _hook("cursor", "beforeReadFile", {"tool_name": "mcp__puppetmaster__codegraph_search"})
    check("puppetmaster tools never denied", out.get("permission") == "allow")

    out = _hook("cursor", "beforeSubmitPrompt", {"prompt": "fix a typo"})
    check("trivial prompt injects nothing", "additionalContext" not in out)


# ---------------------------------------------------------------------------
# 3. Live proxy
# ---------------------------------------------------------------------------


def _post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode())


def run_live_proxy() -> None:
    section("3. Live OpenAI-compatible proxy (advise mode, real HTTP)")
    from puppetmaster.provider_proxy import make_handler

    server = HTTPServer(("127.0.0.1", 0), make_handler(mode="advise"))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        resp = _post(url, {"messages": [{"role": "user", "content": "audit the entire repo for security issues"}]})
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        check("proxy returns synthetic delegate reply", "Puppetmaster" in content,
              json.dumps(resp)[:120])

        resp = _post(url, {"messages": [{"role": "user", "content": "fix a typo"}]})
        check("proxy passes through trivial prompt",
              resp.get("_puppetmaster", {}).get("delegated") is False)
    finally:
        server.shutdown()


def run_proxy_guardrail() -> None:
    section("3b. Proxy upstream allowlist guardrail")
    from puppetmaster.provider_proxy import is_upstream_allowed

    check("allow api.openai.com (https)", is_upstream_allowed("https://api.openai.com"))
    check("allow loopback", is_upstream_allowed("http://127.0.0.1:8080"))
    check("refuse arbitrary host", not is_upstream_allowed("https://evil.example.com"))
    check("refuse http to openai", not is_upstream_allowed("http://api.openai.com"))
    # inject mode must refuse to start with a bad upstream
    from puppetmaster.provider_proxy import serve_proxy
    try:
        serve_proxy(mode="inject", upstream_base_url="https://evil.example.com", port=0)
        check("inject refuses disallowed upstream", False, "did not raise")
    except ValueError:
        check("inject refuses disallowed upstream", True)


# ---------------------------------------------------------------------------
# 4. Installer round-trip
# ---------------------------------------------------------------------------


def run_installer_roundtrip() -> None:
    section("4. Installer round-trip (write + validate + execute installed command)")
    from puppetmaster.hook_installers import install_hooks

    with TemporaryDirectory() as tmp:
        cwd = Path(tmp)
        result = install_hooks(cwd=cwd, python=PY)
        check("install reports installed", result.overall_status == "installed", result.overall_status)

        cursor = json.loads((cwd / ".cursor" / "hooks.json").read_text())
        check("cursor hooks.json valid + has events",
              {"beforeSubmitPrompt", "beforeShellExecution", "beforeReadFile"} <= set(cursor["hooks"]))

        claude = json.loads((cwd / ".claude" / "settings.json").read_text())
        check("claude settings.json valid + has events",
              {"UserPromptSubmit", "PreToolUse"} <= set(claude["hooks"]))

        # Execute the actual command string the installer wrote.
        cmd = cursor["hooks"]["beforeSubmitPrompt"][0]["command"]
        proc = subprocess.run(cmd.split(), input=json.dumps({"prompt": "audit the whole repo"}),
                              capture_output=True, text=True, cwd=str(REPO))
        ok = "Puppetmaster" in proc.stdout
        check("installed cursor command runs + injects", ok, proc.stderr[:120])

        # Idempotency.
        again = install_hooks(cwd=cwd, python=PY)
        check("re-install is unchanged (idempotent)", again.overall_status == "unchanged", again.overall_status)

        # Global scope lands under ~ (fake home), never under the workspace.
        with TemporaryDirectory() as home_tmp:
            home = Path(home_tmp)
            g = install_hooks(cwd=cwd, scope="global", home=home, python=PY)
            check("global install reports installed", g.overall_status == "installed", g.overall_status)
            check("global writes ~/.cursor + ~/.claude",
                  (home / ".cursor" / "hooks.json").exists() and (home / ".claude" / "settings.json").exists())
            g2 = install_hooks(cwd=cwd, scope="global", home=home, python=PY)
            check("global re-install idempotent", g2.overall_status == "unchanged", g2.overall_status)


# ---------------------------------------------------------------------------
# 5. Kill switch
# ---------------------------------------------------------------------------


def run_kill_switch() -> None:
    section("5. Kill switch neuters every layer")
    env = {"PUPPETMASTER_AUTO_INVOKE_DISABLED": "1"}

    from puppetmaster.invocation_gate import should_delegate
    check("gate disabled", not should_delegate("audit everything", env=env).should_delegate)

    out = _hook("claude", "PreToolUse", {"tool_name": "Grep"}, env_extra=env)
    # When disabled the hook allows; claude allow shape has no deny decision.
    denied = out.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
    check("hook stops denying Grep when disabled", not denied)

    out = _hook("cursor", "beforeSubmitPrompt", {"prompt": "audit the repo"}, env_extra=env)
    check("hook stops injecting when disabled", "additionalContext" not in out)


def main() -> int:
    print("Puppetmaster auto-invocation — end-to-end stress / mock test")
    run_gate_battery()
    run_hook_subprocess()
    run_live_proxy()
    run_proxy_guardrail()
    run_installer_roundtrip()
    run_kill_switch()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("RESULT: ALL GREEN — every enforcement layer validated end-to-end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
